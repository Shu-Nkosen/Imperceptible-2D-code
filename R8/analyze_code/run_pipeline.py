from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from naming import VideoNameMeta, parse_video_name


@dataclass(frozen=True)
class SyncConfig:
    min_black_frames: int = 5
    min_red_frames: int = 3
    black_v_max: float = 0.3  # HSV V threshold
    red_r_min: float = 0.5    # normalized mean R threshold
    red_g_max: float = 0.3
    red_b_max: float = 0.3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R8: sync(slate) detect -> split 60 conditions -> run analysis -> results.csv")
    p.add_argument("--video", type=str, required=True, help="input video path (e.g. r180_e250_f1.mp4)")
    p.add_argument("--out-dir", type=str, default="", help="output root dir (default: R8/analyze_code/out)")
    p.add_argument("--conditions", type=int, default=60, help="number of blocks/conditions in a video")
    p.add_argument("--block-sec", type=float, default=6.0, help="block duration seconds")
    p.add_argument("--use-start-sec", type=float, default=2.0, help="within-block start sec to extract")
    p.add_argument("--use-end-sec", type=float, default=4.0, help="within-block end sec to extract (exclusive)")
    p.add_argument("--slate-sec", type=float, default=None, help="slate black and slate red duration sec (default: manifest or 0.5)")
    p.add_argument("--padding-sec", type=float, default=None, help="black padding before/after sync slate sec (default: manifest or 5.0)")
    p.add_argument(
        "--max-frames",
        type=int,
        choices=[1, 120],
        default=120,
        help="解析後に残す frame_*.png の枚数（1 or 120）。差分計算は常に120フレーム分。既定: 120",
    )
    p.add_argument("--manifest", type=str, default="", help="optional presenter manifest.json for metadata join")
    p.add_argument("--workers", type=int, default=1, help="passed to decode_qr_from_all_frames.py")
    p.add_argument("--full-search", action="store_true", help="passed to decode_qr_from_all_frames.py")
    return p.parse_args()


def robust_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        return float(fps)
    return 60.0


def frame_means_bgr(frame: np.ndarray) -> Tuple[float, float, float]:
    b = float(frame[:, :, 0].mean()) / 255.0
    g = float(frame[:, :, 1].mean()) / 255.0
    r = float(frame[:, :, 2].mean()) / 255.0
    return b, g, r


def is_black(frame: np.ndarray, cfg: SyncConfig) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = float(hsv[:, :, 2].mean()) / 255.0
    return v <= cfg.black_v_max


def is_red(frame: np.ndarray, cfg: SyncConfig) -> bool:
    b, g, r = frame_means_bgr(frame)
    return (r >= cfg.red_r_min) and (g <= cfg.red_g_max) and (b <= cfg.red_b_max)


def detect_black_to_red_sync(video_path: Path, cfg: SyncConfig) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    fps = robust_fps(cap)

    state = "seek_black"
    black_run = 0
    red_run = 0
    sync_frame = -1
    idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        if state == "seek_black":
            if is_black(frame, cfg):
                black_run += 1
                if black_run >= cfg.min_black_frames:
                    state = "seek_red"
                    red_run = 0
            else:
                black_run = 0
            continue

        if state == "seek_red":
            if is_red(frame, cfg):
                red_run += 1
                if red_run >= cfg.min_red_frames:
                    sync_frame = idx - (cfg.min_red_frames - 1)
                    break
            else:
                red_run = 0

    cap.release()
    if sync_frame < 0:
        raise RuntimeError("SYNC detection failed (black->red not found). Try adjusting thresholds.")
    return sync_frame, fps


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


ANALYSIS_FRAME_COUNT = 120  # 切り出し枚数（連続フレーム。60fpsならちょうど2秒分）


def select_frame_indices(start_frame: int, end_frame_exclusive: int, max_frames: int) -> List[int]:
    """区間 [start, end) の先頭から連続で最大 max_frames 枚を取る。

    均等間引きだと（例: 2〜5秒から120枚）時間幅が3秒のまま残るため使わない。
    """
    total = max(0, end_frame_exclusive - start_frame)
    if total <= 0:
        return []
    count = min(int(max_frames), total)
    return list(range(start_frame, start_frame + count))


def save_block_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame_exclusive: int,
    out_dir: Path,
    analysis_frames: int = ANALYSIS_FRAME_COUNT,
    resize_to: Optional[Tuple[int, int]] = (1920, 1080),
) -> int:
    """差分計算用に analysis_frames 枚を区間先頭から連続で書き出す。"""
    ensure_dir(out_dir)
    indices = select_frame_indices(start_frame, end_frame_exclusive, analysis_frames)

    saved = 0
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok:
            break
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"frame_{saved:05d}.png"
        cv2.imwrite(str(out_path), frame)
        saved += 1
    return saved


def prune_saved_frames(out_dir: Path, keep_frames: int) -> int:
    """解析後に残す frame_*.png を keep_frames 枚に間引く。"""
    paths = sorted(out_dir.glob("frame_?????.png"))
    if not paths:
        return 0
    keep = min(int(keep_frames), len(paths))
    if keep >= len(paths):
        return len(paths)

    if keep == 1:
        keep_set = {paths[len(paths) // 2]}
    else:
        idxs = [
            int(round(i * (len(paths) - 1) / (keep - 1)))
            for i in range(keep)
        ]
        keep_set = {paths[i] for i in idxs}

    for path in paths:
        if path not in keep_set:
            path.unlink(missing_ok=True)

    # 残したファイルを frame_00000... にリネームし直す
    kept = sorted(keep_set, key=lambda p: p.name)
    tmp_paths: List[Path] = []
    for i, path in enumerate(kept):
        tmp = out_dir / f"_keep_{i:05d}.png"
        path.rename(tmp)
        tmp_paths.append(tmp)
    for i, tmp in enumerate(tmp_paths):
        tmp.rename(out_dir / f"frame_{i:05d}.png")
    return len(tmp_paths)


def pick_decode_row_for_folder(rows: List[Dict[str, str]], folder: str) -> Dict[str, str]:
    matches = [r for r in rows if r.get("folder") == folder]
    if not matches:
        return {}
    # 成功行があれば優先、なければ末尾
    for row in matches:
        if str(row.get("success", "")).strip() in ("1", "True", "true"):
            return row
    return matches[-1]


def read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_py(script: Path, cwd: Path, args: List[str]) -> None:
    cmd = ["python", str(script), *args]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_decode_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_results_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    fieldnames: List[str] = []
    for row in rows_list:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_list:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# present_session.c と同じ並び: channel → image → intensity
BASE_IMAGES = ("rice", "nagaoka_fireworks", "hocho", "ex")
CHANNELS = ("R", "G", "B", "min", "max")
TOKENS = ("R", "G", "B", "I", "X")
INTENSITIES = (4, 8, 12)


def build_default_conditions() -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    idx = 0
    for channel, token in zip(CHANNELS, TOKENS):
        for image in BASE_IMAGES:
            for intensity in INTENSITIES:
                conditions.append(
                    {
                        "idx": idx,
                        "image": image,
                        "channel": channel,
                        "token": token,
                        "intensity": intensity,
                    }
                )
                idx += 1
    return conditions


def condition_folder_name(cond: Dict[str, Any]) -> str:
    image = str(cond.get("image", "unknown"))
    token = str(cond.get("token", "X"))
    intensity = int(cond.get("intensity", 0))
    return f"{image}_{token}_{intensity}"


def resolve_conditions(manifest: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    items = manifest.get("conditions") if isinstance(manifest, dict) else None
    if isinstance(items, list) and items:
        by_idx: Dict[int, Dict[str, Any]] = {}
        for item in items:
            try:
                idx = int(item.get("idx"))
            except Exception:
                continue
            by_idx[idx] = dict(item)
        resolved: List[Dict[str, Any]] = []
        for i in range(n):
            if i in by_idx:
                resolved.append(by_idx[i])
            else:
                raise SystemExit(f"manifest conditions に idx={i} がありません")
        return resolved

    defaults = build_default_conditions()
    if n != len(defaults):
        raise SystemExit(
            f"conditions={n} ですが default は {len(defaults)} 件です。"
            " --manifest を渡すか --conditions を合わせてください。"
        )
    return defaults[:n]


def main() -> None:
    ns = parse_args()

    video_path = Path(ns.video).resolve()
    meta: Optional[VideoNameMeta] = parse_video_name(video_path)
    if meta is None:
        print(f"[WARN] video name does not match r{{rate}}_e{{exp}}_f{{0|1}}.mp4: {video_path.name}")

    base_out = Path(ns.out_dir) if ns.out_dir else (Path(__file__).resolve().parent / "out")
    if not base_out.is_absolute():
        base_out = (Path(__file__).resolve().parent / base_out).resolve()

    out_root = base_out / (meta.stem if meta else video_path.stem)
    ensure_dir(out_root)

    manifest = read_manifest(Path(ns.manifest).resolve()) if ns.manifest else {}

    slate_sec = float(ns.slate_sec) if ns.slate_sec is not None else float(manifest.get("slate_sec", 0.5))
    padding_sec = float(ns.padding_sec) if ns.padding_sec is not None else float(manifest.get("padding_sec", 5.0))
    block_sec = float(ns.block_sec)

    cfg = SyncConfig()
    sync_frame, fps = detect_black_to_red_sync(video_path, cfg)
    print(f"[OK] SYNC at frame={sync_frame}, fps={fps:.4f}")

    slate_frames = int(round(slate_sec * fps))
    padding_frames = int(round(padding_sec * fps))
    block_frames = int(round(block_sec * fps))
    use_start = int(round(float(ns.use_start_sec) * fps))
    use_end = int(round(float(ns.use_end_sec) * fps))
    if use_end <= use_start:
        raise SystemExit("use-end-sec must be > use-start-sec")
    use_duration_sec = float(ns.use_end_sec) - float(ns.use_start_sec)
    print(
        f"[INFO] use_window=[{ns.use_start_sec:.3f}, {ns.use_end_sec:.3f})s "
        f"duration={use_duration_sec:.3f}s / fps={fps:.4f} -> "
        f"window_frames={use_end - use_start}, analysis_frames={ANALYSIS_FRAME_COUNT} (consecutive)"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(str(video_path))

    # block 0 starts after red slate + post-padding (sync_frame = red onset)
    block0 = sync_frame + slate_frames + padding_frames
    print(
        f"[INFO] block0={block0} (sync + slate {slate_frames} + padding {padding_frames}), "
        f"slate_sec={slate_sec}, padding_sec={padding_sec}, block_sec={block_sec}"
    )

    conditions = resolve_conditions(manifest, int(ns.conditions))
    script_dir = Path(__file__).resolve().parent
    diff_script = script_dir / "cal-from-2frame-RGB-oute.py"
    decode_script = script_dir / "decode_qr_from_all_frames.py"
    results_csv = out_root / "results.csv"
    results: List[Dict[str, Any]] = []

    print(
        f"[INFO] analysis_frames={ANALYSIS_FRAME_COUNT} / keep_frames={ns.max_frames} "
        "/ results overwritten after each condition"
    )

    for i, cond in enumerate(conditions):
        block_start = block0 + i * block_frames
        start = block_start + use_start
        end = block_start + use_end
        folder_name = condition_folder_name(cond)
        cond_dir = out_root / folder_name
        cond_dir_abs = str(cond_dir.resolve())
        out_root_abs = str(out_root.resolve())

        analyzed = save_block_frames(
            cap,
            start,
            end,
            cond_dir,
            analysis_frames=ANALYSIS_FRAME_COUNT,
        )
        print(f"[INFO] ({i+1}/{len(conditions)}) extract {folder_name}: analysis_frames={analyzed}")

        decode_note = ""
        try:
            run_py(diff_script, cwd=out_root, args=["--base-dir", cond_dir_abs])
        except subprocess.CalledProcessError as exc:
            decode_note = f"diff失敗: exit={exc.returncode}"
            print(f"[WARN] {folder_name}: {decode_note}")

        if not decode_note:
            decode_args = [
                "--base-dir",
                out_root_abs,
                "--folder",
                folder_name,
                "--workers",
                str(int(ns.workers)),
                "--no-save-analysis",
            ]
            if ns.full_search:
                decode_args.append("--full-search")
            try:
                run_py(decode_script, cwd=out_root, args=decode_args)
            except subprocess.CalledProcessError as exc:
                decode_note = f"decode失敗: exit={exc.returncode}"
                print(f"[WARN] {folder_name}: {decode_note}")

        kept = prune_saved_frames(cond_dir, keep_frames=int(ns.max_frames))
        info = {
            "cond": i,
            "folder": folder_name,
            "image": cond.get("image", ""),
            "channel": cond.get("channel", ""),
            "token": cond.get("token", ""),
            "intensity": cond.get("intensity", ""),
            "start_frame": start,
            "end_frame": end,
            "analysis_frames": analyzed,
            "saved_frames": kept,
            "max_frames": int(ns.max_frames),
        }
        print(f"[INFO] ({i+1}/{len(conditions)}) keep frames={kept}")

        decode_csv = out_root / "qr_decode_all_frames.csv"
        decoded_rows = load_decode_csv(decode_csv)
        dec = pick_decode_row_for_folder(decoded_rows, folder_name)

        row: Dict[str, Any] = {
            "video": video_path.name,
            "rate_hz": meta.rate_hz if meta else "",
            "exp": meta.exp if meta else "",
            "fluoro": meta.fluoro if meta else "",
            "fps": f"{fps:.6f}",
            **info,
            "note": decode_note,
        }
        if dec:
            for k, v in dec.items():
                row[f"decode_{k}"] = v

        results.append(row)
        write_results_csv(results_csv, results)
        print(f"[OK] ({i+1}/{len(conditions)}) wrote {results_csv.name} ({len(results)} rows)")

    cap.release()
    print(f"[OK] results: {results_csv}")


if __name__ == "__main__":
    main()

