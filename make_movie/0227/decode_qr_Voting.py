from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

INPUT_CSV = Path("eval_summary_accuracy.csv")
OUTPUT_CSV = Path("qr_decode_best_frames.csv")
DIFF_SUBDIR = "rgb_max_diff_maps"
SCALE_TAG = "rgbmax_scale1.00"
ANALYSIS_DIR = Path("median_eval_analysis")
DEFAULT_MEDIAN_KERNEL = 5
DEFAULT_MEDIAN_ITERATIONS = 1
DEFAULT_TEMPORAL_WINDOW = 8
DEFAULT_TEMPORAL_MAX_GAP = 1
DEFAULT_TEMPORAL_MAX_PAIRS = 31
DEFAULT_TEMPORAL_MODE = "both"
DEFAULT_VOTE_BINARY_THRESHOLD = 200
DEFAULT_VOTE_RATIO = 0.6


def parse_kernel_list(text: str) -> List[int]:
    kernels: List[int] = []
    for chunk in text.split(","):
        value = chunk.strip()
        if not value:
            continue
        kernel = abs(int(value))
        if kernel == 0:
            continue
        if kernel % 2 == 0:
            kernel += 1
        kernels.append(kernel)
    unique = []
    seen = set()
    for kernel in kernels:
        if kernel not in seen:
            seen.add(kernel)
            unique.append(kernel)
    return unique


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def parse_diff_pair_indices(path: Path) -> Optional[Tuple[int, int]]:
    match = re.match(r"(\d+)-(\d+)-", path.stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def extract_frame_index(frame_name: str) -> Optional[int]:
    match = re.search(r"(\d+)", frame_name)
    if not match:
        return None
    return int(match.group(1))


def resolve_diff_image_path(folder_dir: Path, frame1: str, frame2: str) -> Tuple[Optional[Path], str]:
    idx1 = extract_frame_index(frame1)
    idx2 = extract_frame_index(frame2)
    if idx1 is None or idx2 is None:
        return None, "best_frameの番号抽出に失敗"

    left, right = sorted((idx1, idx2))
    diff_dir = folder_dir / DIFF_SUBDIR
    if not diff_dir.exists():
        return None, f"差分ディレクトリなし: {DIFF_SUBDIR}"

    expected = diff_dir / f"{left:05d}-{right:05d}-FRAME_{SCALE_TAG}.png"
    if expected.exists():
        return expected, ""

    candidates = sorted(diff_dir.glob(f"{left:05d}-{right:05d}-*_rgbmax_scale*.png"))
    if not candidates:
        return None, "対応する差分画像が見つからない"

    return candidates[0], ""


def collect_temporal_diff_paths(
    folder_dir: Path,
    anchor_frame1: str,
    anchor_frame2: str,
    anchor_diff_path: Path,
    temporal_window: int,
    temporal_max_gap: int,
    temporal_max_pairs: int,
) -> List[Path]:
    idx1 = extract_frame_index(anchor_frame1)
    idx2 = extract_frame_index(anchor_frame2)
    if idx1 is None or idx2 is None:
        return [anchor_diff_path]

    diff_dir = folder_dir / DIFF_SUBDIR
    if not diff_dir.exists():
        return [anchor_diff_path]

    left_anchor, right_anchor = sorted((idx1, idx2))
    center = (left_anchor + right_anchor) / 2.0
    left_min = left_anchor - max(0, temporal_window)
    right_max = right_anchor + max(0, temporal_window)
    max_gap = max(1, temporal_max_gap)

    candidates: List[Tuple[float, Path]] = []
    for path in sorted(diff_dir.glob("*.png")):
        pair = parse_diff_pair_indices(path)
        if pair is None:
            continue
        left, right = pair
        if right - left > max_gap:
            continue
        if left < left_min or right > right_max:
            continue

        pair_center = (left + right) / 2.0
        score = abs(pair_center - center)
        candidates.append((score, path))

    if not candidates:
        return [anchor_diff_path]

    candidates.sort(key=lambda item: (item[0], item[1].name))
    selected = [path for _, path in candidates[: max(1, temporal_max_pairs)]]

    if anchor_diff_path not in selected:
        selected.append(anchor_diff_path)
    return sorted(set(selected), key=lambda p: p.name)


def load_gray_diff_images(paths: List[Path]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
    return images


def build_vote_stable_map(images: List[np.ndarray], binary_threshold: int, vote_ratio: float) -> Optional[np.ndarray]:
    if not images:
        return None
    stack = np.stack(images, axis=0)
    threshold = int(np.clip(binary_threshold, 0, 255))
    ratio = float(np.clip(vote_ratio, 0.0, 1.0))

    active = stack <= threshold
    votes = active.mean(axis=0)
    stable = votes >= ratio
    return np.where(stable, 0, 255).astype(np.uint8)


def build_median_stable_map(images: List[np.ndarray]) -> Optional[np.ndarray]:
    if not images:
        return None
    stack = np.stack(images, axis=0)
    median = np.median(stack, axis=0)
    return np.clip(median, 0, 255).astype(np.uint8)


def apply_median_filter(gray: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    iters = max(0, int(iterations))

    filtered = gray.copy()
    if size < 3 or iters == 0:
        return filtered

    for _ in range(iters):
        filtered = cv2.medianBlur(filtered, size)
    return filtered


def build_variants(gray: np.ndarray, median_gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []

    variants.append(("gray", gray))
    variants.append(("median_gray", median_gray))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    _, median_otsu = cv2.threshold(median_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("median_otsu", median_otsu))

    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(("otsu_inv", otsu_inv))

    _, median_otsu_inv = cv2.threshold(median_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(("median_otsu_inv", median_otsu_inv))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    variants.append(("adaptive", adaptive))

    median_adaptive = cv2.adaptiveThreshold(
        median_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    variants.append(("median_adaptive", median_adaptive))

    kernel = np.ones((3, 3), np.uint8)
    close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    variants.append(("otsu_close", close))

    median_close = cv2.morphologyEx(median_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    variants.append(("median_otsu_close", median_close))

    return variants


def try_decode(detector: cv2.QRCodeDetector, image: np.ndarray) -> Optional[str]:
    text, points, _ = detector.detectAndDecode(image)
    if points is not None and text:
        return text

    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    if retval and decoded_info:
        for value in decoded_info:
            if value:
                return value

    return None


def decode_qr_from_gray(
    gray: np.ndarray,
    median_kernel: int,
    median_iterations: int,
    method_prefix: str = "",
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    median_gray = apply_median_filter(gray, median_kernel, median_iterations)
    detector = cv2.QRCodeDetector()
    last_target: Optional[np.ndarray] = None

    prefix = f"{method_prefix}_" if method_prefix else ""

    for variant_name, variant_img in build_variants(gray, median_gray):
        for scale in (1.0, 2.0, 3.0):
            if scale == 1.0:
                target = variant_img
            else:
                target = cv2.resize(
                    variant_img,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            last_target = target

            text = try_decode(detector, target)
            if text:
                return (
                    text,
                    f"{prefix}{variant_name}_x{scale:.1f}_k{median_kernel}_i{median_iterations}",
                    target,
                )

    return None, f"{prefix}decode失敗", last_target


def decode_qr_from_diff(
    diff_path: Path,
    median_kernel: int,
    median_iterations: int,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    image = cv2.imread(str(diff_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, "画像読み込み失敗", None
    return decode_qr_from_gray(image, median_kernel, median_iterations)


def decode_qr_with_kernel_candidates(
    diff_path: Path,
    kernel_candidates: List[int],
    median_iterations: int,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    last_target: Optional[np.ndarray] = None
    for kernel in kernel_candidates:
        text, method, used_img = decode_qr_from_diff(diff_path, kernel, median_iterations)
        if used_img is not None:
            last_target = used_img
        if text:
            return text, method, used_img
    return None, "decode失敗", last_target


def decode_qr_temporal_aggregate(
    temporal_diff_paths: List[Path],
    kernel_candidates: List[int],
    median_iterations: int,
    temporal_mode: str,
    vote_binary_threshold: int,
    vote_ratio: float,
) -> Tuple[Optional[str], str, Optional[np.ndarray], int]:
    gray_images = load_gray_diff_images(temporal_diff_paths)
    frame_count = len(gray_images)
    if frame_count == 0:
        return None, "temporal_no_images", None, 0

    aggregate_candidates: List[Tuple[str, np.ndarray]] = []

    if temporal_mode in ("vote", "both"):
        vote_map = build_vote_stable_map(gray_images, vote_binary_threshold, vote_ratio)
        if vote_map is not None:
            aggregate_candidates.append(("temporal_vote", vote_map))

    if temporal_mode in ("median", "both"):
        median_map = build_median_stable_map(gray_images)
        if median_map is not None:
            aggregate_candidates.append(("temporal_median", median_map))

    last_target: Optional[np.ndarray] = None
    for mode_name, aggregate_gray in aggregate_candidates:
        for kernel in kernel_candidates:
            text, method, used_img = decode_qr_from_gray(
                aggregate_gray,
                kernel,
                median_iterations,
                method_prefix=f"{mode_name}_n{frame_count}",
            )
            if used_img is not None:
                last_target = used_img
            if text:
                return text, method, used_img, frame_count

    return None, "temporal_decode失敗", last_target, frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description="best_frame差分画像からQR文字列を復元する（案C対応）")
    parser.add_argument("--folder", type=str, default="", help="対象フォルダ名を1つに限定（例: ex_B6）")
    parser.add_argument("--limit", type=int, default=0, help="先頭から処理する行数（0で全件）")
    parser.add_argument(
        "--median-kernel",
        type=int,
        default=DEFAULT_MEDIAN_KERNEL,
        help="メディアンフィルタのカーネルサイズ（偶数指定時は+1）",
    )
    parser.add_argument(
        "--median-kernels",
        type=str,
        default="",
        help="カーネル総当たり（例: 3,5,7）。指定時は --median-kernel より優先",
    )
    parser.add_argument(
        "--median-iterations",
        type=int,
        default=DEFAULT_MEDIAN_ITERATIONS,
        help="メディアンフィルタ反復回数（0で無効）",
    )
    parser.add_argument(
        "--temporal-mode",
        type=str,
        choices=["none", "vote", "median", "both"],
        default=DEFAULT_TEMPORAL_MODE,
        help="時系列統合の方式。none で単一ペアのみ",
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=DEFAULT_TEMPORAL_WINDOW,
        help="best_frame前後で統合対象にするフレーム幅",
    )
    parser.add_argument(
        "--temporal-max-gap",
        type=int,
        default=DEFAULT_TEMPORAL_MAX_GAP,
        help="統合対象に含める差分ペアの最大フレーム差",
    )
    parser.add_argument(
        "--temporal-max-pairs",
        type=int,
        default=DEFAULT_TEMPORAL_MAX_PAIRS,
        help="統合に使う差分マップの最大枚数",
    )
    parser.add_argument(
        "--vote-binary-threshold",
        type=int,
        default=DEFAULT_VOTE_BINARY_THRESHOLD,
        help="投票統合で黒画素判定に使う閾値(0-255)",
    )
    parser.add_argument(
        "--vote-ratio",
        type=float,
        default=DEFAULT_VOTE_RATIO,
        help="投票統合で安定画素とみなす比率(0.0-1.0)",
    )
    args = parser.parse_args()
    median_kernel = max(1, args.median_kernel)
    if median_kernel % 2 == 0:
        median_kernel += 1
    median_iterations = max(0, args.median_iterations)
    kernel_candidates = parse_kernel_list(args.median_kernels) if args.median_kernels else [median_kernel]
    if not kernel_candidates:
        kernel_candidates = [median_kernel]

    print(
        f"[INFO] folder filter: {args.folder or '(all)'} / limit: {args.limit} "
        f"/ median kernels: {kernel_candidates}, iter={median_iterations} "
        f"/ temporal: {args.temporal_mode}"
    )

    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / INPUT_CSV
    output_csv = base_dir / OUTPUT_CSV
    analysis_dir = base_dir / ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_csv}")

    rows_out = []
    total = 0
    success = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.folder and row.get("folder", "") != args.folder:
                continue
            if args.limit > 0 and total >= args.limit:
                break

            total += 1
            folder = row.get("folder", "")
            frame1 = row.get("best_frame_1", "")
            frame2 = row.get("best_frame_2", "")

            folder_dir = base_dir / folder
            if not folder or not folder_dir.exists():
                rows_out.append(
                    {
                        "folder": folder,
                        "best_frame_1": frame1,
                        "best_frame_2": frame2,
                        "diff_image": "",
                        "decoded_text": "",
                        "success": 0,
                        "method": "",
                        "note": "フォルダが存在しない",
                    }
                )
                continue

            diff_path, note = resolve_diff_image_path(folder_dir, frame1, frame2)
            if diff_path is None:
                rows_out.append(
                    {
                        "folder": folder,
                        "best_frame_1": frame1,
                        "best_frame_2": frame2,
                        "diff_image": "",
                        "temporal_count": 0,
                        "decoded_text": "",
                        "success": 0,
                        "method": "",
                        "note": note,
                    }
                )
                print(f"[NG] {folder}: {note}")
                continue

            used_temporal = 0
            decoded_text: Optional[str]
            method: str
            used_img: Optional[np.ndarray]

            if args.temporal_mode != "none":
                temporal_paths = collect_temporal_diff_paths(
                    folder_dir,
                    frame1,
                    frame2,
                    diff_path,
                    temporal_window=max(0, args.temporal_window),
                    temporal_max_gap=max(1, args.temporal_max_gap),
                    temporal_max_pairs=max(1, args.temporal_max_pairs),
                )
                decoded_text, method, used_img, used_temporal = decode_qr_temporal_aggregate(
                    temporal_paths,
                    kernel_candidates,
                    median_iterations,
                    temporal_mode=args.temporal_mode,
                    vote_binary_threshold=args.vote_binary_threshold,
                    vote_ratio=args.vote_ratio,
                )
                if decoded_text is None:
                    single_text, single_method, single_img = decode_qr_with_kernel_candidates(
                        diff_path,
                        kernel_candidates,
                        median_iterations,
                    )
                    decoded_text = single_text
                    method = f"{method}|fallback_{single_method}"
                    used_img = single_img if single_img is not None else used_img
            else:
                decoded_text, method, used_img = decode_qr_with_kernel_candidates(
                    diff_path,
                    kernel_candidates,
                    median_iterations,
                )
            ok = int(decoded_text is not None)
            success += ok

            pair_name = f"{Path(frame1).stem}-{Path(frame2).stem}"
            folder_analysis_dir = analysis_dir / folder
            folder_analysis_dir.mkdir(parents=True, exist_ok=True)
            method_tag = sanitize_filename(method if method else "unknown")
            result_tag = "ok" if ok else "ng"
            analysis_image_name = f"{pair_name}_{result_tag}_{method_tag}.png"
            analysis_image_path = folder_analysis_dir / analysis_image_name
            if used_img is not None:
                cv2.imwrite(str(analysis_image_path), used_img)
                analysis_image_rel = str(analysis_image_path.relative_to(base_dir))
            else:
                analysis_image_rel = ""

            rows_out.append(
                {
                    "folder": folder,
                    "best_frame_1": frame1,
                    "best_frame_2": frame2,
                    "diff_image": str(diff_path.relative_to(base_dir)),
                    "temporal_count": used_temporal,
                    "analysis_image": analysis_image_rel,
                    "decoded_text": decoded_text or "",
                    "success": ok,
                    "method": method,
                    "note": "" if ok else "QR未検出",
                }
            )

            if ok:
                print(f"[OK] {folder}: {decoded_text}")
            else:
                print(f"[NG] {folder}: QR未検出 ({diff_path.name})")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "folder",
            "best_frame_1",
            "best_frame_2",
            "diff_image",
            "temporal_count",
            "analysis_image",
            "decoded_text",
            "success",
            "method",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print("\n==== 完了 ====")
    print(f"総件数: {total}")
    print(f"成功件数: {success}")
    print(f"成功率: {success / total:.2%}" if total else "成功率: 0.00%")
    print(f"出力CSV: {output_csv}")


if __name__ == "__main__":
    main()
