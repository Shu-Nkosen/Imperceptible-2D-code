from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYZE_CODE = ROOT / "analyze_code"
for path in (ROOT, ANALYZE_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decode_qr_from_all_frames import (  # noqa: E402
    compare_with_gt,
    decode_qr_with_kernel_candidates_from_array,
)
from run_pipeline import (  # noqa: E402
    ANALYSIS_FRAME_COUNT,
    SyncConfig,
    condition_folder_name,
    detect_black_to_red_sync,
    ensure_dir,
    extract_gt_qr_mask,
    format_common_meta,
    read_manifest,
    resolve_conditions,
    resolve_gt_qr_timeline,
    save_block_frames,
)
from naming import parse_video_name  # noqa: E402

RESULTS_FIELDS = (
    "folder", "decode_note", "decode_variant", "diff_mode", "window_n",
    "stat_kind", "fft_target_hz", "diff_threshold", "pixel_acc_all",
    "pixel_acc_ok", "pixel_acc_best", "best_pair_frame_1",
    "best_pair_frame_2", "video", "display_rate", "exposure", "fluorescent",
    "camera_fps", "rate_hz", "exposure_denom", "fluorescent_flag", "cond",
    "image", "channel", "token", "intensity", "decode_decoded_text",
    "decode_method", "decode_frame_1", "decode_frame_2", "note",
    "analysis_frames", "extract_sec",
)
PAIR_FIELDS = (
    "folder", "cond", "frame_1", "frame_2", "success", "accuracy",
    "recall", "precision", "diff_threshold",
)
STAT_FIELDS = (
    "folder", "cond", "image", "channel", "token", "intensity",
    "delta_l_black_mean", "delta_l_white_mean", "delta_a_black_mean",
    "delta_a_white_mean", "delta_b_black_mean", "delta_b_white_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze complementary frame pairs in CIE Lab space")
    parser.add_argument("--video", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--mid-search", action="store_true")
    parser.add_argument("--module-count", type=int, default=0,
                         help="QR modules per side; 0 infers it from the GT mask")
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def selected_adjacent_pairs(frame_paths: list[Path], each_end: int = 20) -> list[tuple[Path, Path]]:
    ordered = sorted(frame_paths, key=lambda path: int(path.stem.split("_")[-1]))
    pairs = list(zip(ordered, ordered[1:]))
    if len(pairs) <= each_end * 2:
        return pairs
    return pairs[:each_end] + pairs[-each_end:]


def load_lab(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read frame: {path}")
    image_f32 = image.astype(np.float32) / 255.0
    return cv2.cvtColor(image_f32, cv2.COLOR_BGR2LAB)


def infer_module_count(gt_mask: np.ndarray, module_count: int) -> int:
    if module_count > 0:
        return module_count
    h, w = gt_mask.shape
    scores: list[tuple[float, int]] = []
    for count in range(21, 178, 4):
        values = []
        for row in range(count):
            y0, y1 = row * h // count, (row + 1) * h // count
            for col in range(count):
                x0, x1 = col * w // count, (col + 1) * w // count
                values.append(float(gt_mask[y0:y1, x0:x1].mean()))
        values_array = np.asarray(values)
        scores.append((float(np.mean(np.minimum(values_array, 1.0 - values_array))), count))
    return min(scores)[1]


def module_values(image: np.ndarray, module_count: int) -> np.ndarray:
    h, w = image.shape[:2]
    values = np.empty((module_count, module_count), dtype=np.float32)
    for row in range(module_count):
        y0, y1 = row * h // module_count, (row + 1) * h // module_count
        for col in range(module_count):
            x0, x1 = col * w // module_count, (col + 1) * w // module_count
            values[row, col] = float(image[y0:y1, x0:x1].mean())
    return values


def make_decode_image(
    delta: np.ndarray,
    black_when_positive: bool,
    full_shape: tuple[int, int],
    roi_x: int,
    roi_y: int,
) -> np.ndarray:
    signed = np.where(delta >= 0.0, 0, 255).astype(np.uint8)
    if not black_when_positive:
        signed = 255 - signed
    full = np.full(full_shape, 255, dtype=np.uint8)
    full[roi_y:roi_y + signed.shape[0], roi_x:roi_x + signed.shape[1]] = signed
    return cv2.cvtColor(full, cv2.COLOR_GRAY2RGB)


def qr_metrics(delta: np.ndarray, gt_mask: np.ndarray, module_count: int) -> tuple[float, bool, np.ndarray]:
    gt_values = module_values(gt_mask.astype(np.float32), module_count) > 0.5
    delta_values = module_values(delta, module_count)
    positive_black = np.mean((delta_values >= 0) == gt_values)
    negative_black = np.mean((delta_values < 0) == gt_values)
    black_when_positive = bool(positive_black >= negative_black)
    predicted = (delta_values >= 0) if black_when_positive else (delta_values < 0)
    return float(max(positive_black, negative_black)), black_when_positive, predicted


def main() -> int:
    ns = parse_args()
    video_path = Path(ns.video).resolve()
    manifest = read_manifest(Path(ns.manifest).resolve())
    meta = parse_video_name(video_path)
    out_base = Path(ns.out_dir).resolve() if ns.out_dir else Path(__file__).resolve().parent / "out"
    out_root = out_base / (meta.stem if meta else video_path.stem)
    ensure_dir(out_root)

    sync_frame, fps = detect_black_to_red_sync(video_path, SyncConfig())
    conditions = resolve_conditions(manifest)
    slate_sec = float(manifest.get("slate_sec", 0.5))
    padding_sec = float(manifest.get("padding_sec", 5.0))
    cond_starts, slots, _ = resolve_gt_qr_timeline(
        manifest, slate_sec, padding_sec, 6.0, len(conditions)
    )
    gt_path = out_root / "frame_QR.png"
    if not gt_path.exists():
        extract_gt_qr_mask(video_path, sync_frame, fps, slots, gt_path)
    gt_image = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt_image is None:
        raise FileNotFoundError(f"cannot read GT mask: {gt_path}")
    roi_size = int(gt_image.shape[0] * 0.7)
    roi_y = (gt_image.shape[0] - roi_size) // 2
    roi_x = (gt_image.shape[1] - roi_size) // 2
    gt_roi = (gt_image[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size] < 128).astype(np.float32)
    module_count = infer_module_count(gt_roi, ns.module_count)

    common = format_common_meta(video_path.name, meta, fps)
    results: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    existing_results_path = ANALYZE_CODE / "out_rgblab" / video_path.stem / "results.csv"
    existing_pair_path = ANALYZE_CODE / "out_rgblab" / video_path.stem / "pair_accuracy.csv"
    existing_rows: dict[str, dict[str, str]] = {}
    if existing_results_path.exists():
        with existing_results_path.open("r", encoding="utf-8-sig", newline="") as stream:
            existing_rows = {row["folder"]: row for row in csv.DictReader(stream)}
    existing_pair_counts: dict[str, int] = {}
    if existing_pair_path.exists():
        with existing_pair_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("success") in ("1", "True", "true"):
                    folder = row.get("folder", "")
                    existing_pair_counts[folder] = existing_pair_counts.get(folder, 0) + 1

    for index, condition in enumerate(conditions):
        folder = condition_folder_name(condition)
        folder_path = out_root / folder
        frame_count = len(list(folder_path.glob("frame_*.png")))
        if frame_count < ANALYSIS_FRAME_COUNT:
            start = sync_frame + int(round(cond_starts[index] * fps)) + int(round(2.0 * fps))
            end = sync_frame + int(round(cond_starts[index] * fps)) + int(round(4.0 * fps))
            frame_count = save_block_frames(
                video_path, start, end, folder_path, ANALYSIS_FRAME_COUNT
            )
        frame_paths = sorted(folder_path.glob("frame_*.png"))
        pairs = selected_adjacent_pairs(frame_paths)
        all_channel_deltas = [[], [], []]
        condition_pairs: list[dict[str, Any]] = []
        selected_component = {"LabL": 0, "LabA": 1, "LabB": 2}.get(
            str(condition.get("channel"))
        )
        decode_rows: list[tuple[str, str, str, str, float]] = []
        for frame_1, frame_2 in pairs:
            lab_1, lab_2 = load_lab(frame_1), load_lab(frame_2)
            delta = lab_2 - lab_1
            cropped = delta[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
            for component in range(3):
                all_channel_deltas[component].append(cropped[:, :, component])
            component_candidates = (
                [selected_component] if selected_component is not None else [0, 1, 2]
            )
            component_results = [
                (component, qr_metrics(cropped[:, :, component], gt_roi, module_count))
                for component in component_candidates
            ]
            component, (module_accuracy, black_when_positive, _) = max(
                component_results, key=lambda item: item[1][0]
            )
            component_delta = cropped[:, :, component]
            decode_image = make_decode_image(
                component_delta, black_when_positive, gt_image.shape, roi_x, roi_y
            )
            text, method, _ = decode_qr_with_kernel_candidates_from_array(
                decode_image, [3, 5, 7] if ns.mid_search else [5], 1,
                "mid" if ns.mid_search else "fast",
            )
            binary_gray = cv2.cvtColor(decode_image, cv2.COLOR_RGB2GRAY)
            recall, precision, pixel_accuracy_text, _, _ = compare_with_gt(
                binary_gray, str(gt_path)
            )
            pixel_accuracy = float(pixel_accuracy_text) if pixel_accuracy_text else 0.0
            success = bool(text)
            row = {
                "folder": folder, "cond": condition.get("idx", index),
                "frame_1": frame_1.name, "frame_2": frame_2.name,
                "success": int(success), "accuracy": f"{pixel_accuracy:.6f}",
                "recall": recall, "precision": precision, "diff_threshold": "sign",
            }
            pair_rows.append(row)
            condition_pairs.append(row)
            decode_rows.append(
                (text or "", method, frame_1.name, frame_2.name, module_accuracy)
            )

        if not condition_pairs:
            raise RuntimeError(f"{folder}: no adjacent frame pairs")
        best_pair = max(condition_pairs, key=lambda row: float(row["accuracy"]))
        successful = [row for row in condition_pairs if row["success"]]
        best_decode = next((row for row in decode_rows if row[0]), ("", "", "", "", 0.0))
        all_acc = [float(row["accuracy"]) for row in condition_pairs]
        ok_acc = [float(row["accuracy"]) for row in successful]
        result = {
            "folder": folder,
            "decode_note": "" if successful else "QR未検出",
            "decode_variant": best_decode[1],
            "diff_mode": "lab_pair", "diff_threshold": "sign",
            "pixel_acc_all": f"{np.mean(all_acc):.6f}",
            "pixel_acc_ok": f"{np.mean(ok_acc):.6f}" if ok_acc else "",
            "pixel_acc_best": best_pair["accuracy"],
            "best_pair_frame_1": best_pair["frame_1"],
            "best_pair_frame_2": best_pair["frame_2"],
            **common, "cond": condition.get("idx", index),
            "image": condition.get("image", ""), "channel": condition.get("channel", ""),
            "token": condition.get("token", ""), "intensity": condition.get("intensity", ""),
            "decode_decoded_text": best_decode[0], "decode_method": best_decode[1],
            "decode_frame_1": best_decode[2], "decode_frame_2": best_decode[3],
            "note": "", "analysis_frames": frame_count,
            "extract_sec": f"{frame_count / fps:.6f}" if fps else "",
        }
        results.append(result)
        black_mask = gt_roi > 0.5
        stats = {"folder": folder, "cond": condition.get("idx", index),
                 "image": condition.get("image", ""), "channel": condition.get("channel", ""),
                 "token": condition.get("token", ""), "intensity": condition.get("intensity", "")}
        for name, component_deltas in zip(("l", "a", "b"), all_channel_deltas):
            merged = np.concatenate([array.ravel() for array in component_deltas])
            masks = np.tile(black_mask.ravel(), len(component_deltas))
            stats[f"delta_{name}_black_mean"] = f"{np.mean(merged[masks]):.6f}"
            stats[f"delta_{name}_white_mean"] = f"{np.mean(merged[~masks]):.6f}"
        stat_rows.append(stats)
        old = existing_rows.get(folder, {})
        compare_rows.append({
            "folder": folder, "cond": condition.get("idx", index),
            "existing_decode_success": int(bool(old.get("decode_decoded_text"))),
            "lab_decode_success": int(bool(successful)),
            "existing_successful_pairs": existing_pair_counts.get(folder, ""),
            "lab_successful_pairs": len(successful),
            "existing_pixel_acc_all": old.get("pixel_acc_all", ""),
            "lab_pixel_acc_all": result["pixel_acc_all"],
            "existing_pixel_acc_ok": old.get("pixel_acc_ok", ""),
            "lab_pixel_acc_ok": result["pixel_acc_ok"],
            "existing_pixel_acc_best": old.get("pixel_acc_best", ""),
            "lab_pixel_acc_best": result["pixel_acc_best"],
        })

    write_csv(out_root / "results.csv", results, RESULTS_FIELDS)
    write_csv(out_root / "pair_accuracy.csv", pair_rows, PAIR_FIELDS)
    write_csv(out_root / "lab_stats.csv", stat_rows, STAT_FIELDS)
    write_csv(out_root / "compare.csv", compare_rows, tuple(compare_rows[0].keys()))
    print(f"[OK] {len(results)} conditions, module_count={module_count}: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
