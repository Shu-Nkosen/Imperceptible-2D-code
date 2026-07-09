from __future__ import annotations

import argparse
import csv
import json
import math
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
    black_v_max: float = 0.12  # HSV V threshold
    red_r_min: float = 0.75    # normalized mean R threshold
    red_g_max: float = 0.25
    red_b_max: float = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R8: sync(slate) detect -> split 60 conditions -> run analysis -> results.csv")
    p.add_argument("--video", type=str, required=True, help="input video path (e.g. r180_e250_f1.mp4)")
    p.add_argument("--out-dir", type=str, default="", help="output root dir (default: R8/analyze_code/out)")
    p.add_argument("--conditions", type=int, default=60, help="number of blocks/conditions in a video")
    p.add_argument("--block-sec", type=float, default=6.0, help="block duration seconds")
    p.add_argument("--use-start-sec", type=float, default=2.0, help="within-block start sec to extract")
    p.add_argument("--use-end-sec", type=float, default=5.0, help="within-block end sec to extract (exclusive)")
    p.add_argument("--slate-sec", type=float, default=None, help="slate black and slate red duration sec (default: manifest or 0.5)")
    p.add_argument("--padding-sec", type=float, default=None, help="black padding before/after sync slate sec (default: manifest or 5.0)")
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


def save_block_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame_exclusive: int,
    out_dir: Path,
    resize_to: Optional[Tuple[int, int]] = (1920, 1080),
) -> int:
    ensure_dir(out_dir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    saved = 0
    for i in range(start_frame, end_frame_exclusive):
        ok, frame = cap.read()
        if not ok:
            break
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"frame_{saved:05d}.png"
        cv2.imwrite(str(out_path), frame)
        saved += 1
    return saved


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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(str(video_path))

    # block 0 starts after red slate + post-padding (sync_frame = red onset)
    block0 = sync_frame + slate_frames + padding_frames
    print(
        f"[INFO] block0={block0} (sync + slate {slate_frames} + padding {padding_frames}), "
        f"slate_sec={slate_sec}, padding_sec={padding_sec}, block_sec={block_sec}"
    )

    saved_blocks: List[Dict[str, Any]] = []
    for i in range(int(ns.conditions)):
        block_start = block0 + i * block_frames
        start = block_start + use_start
        end = block_start + use_end
        cond_dir = out_root / f"cond_{i:03d}"
        saved = save_block_frames(cap, start, end, cond_dir)
        saved_blocks.append(
            {
                "cond": i,
                "folder": cond_dir.name,
                "start_frame": start,
                "end_frame": end,
                "saved_frames": saved,
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == int(ns.conditions):
            print(f"[INFO] extracted {i+1}/{int(ns.conditions)}")

    cap.release()

    # run diff generation once at out_root (processes subdirs)
    diff_script = Path(__file__).resolve().parent / "cal-from-2frame-RGB-oute.py"
    run_py(diff_script, cwd=out_root, args=[])

    # run decode once at out_root (scans for diff subdirs)
    decode_script = Path(__file__).resolve().parent / "decode_qr_from_all_frames.py"
    decode_args = ["--workers", str(int(ns.workers))]
    if ns.full_search:
        decode_args.append("--full-search")
    run_py(decode_script, cwd=out_root, args=decode_args)

    decode_csv = out_root / "qr_decode_all_frames.csv"
    decoded_rows = load_decode_csv(decode_csv)
    decoded_by_folder = {r.get("folder", ""): r for r in decoded_rows if r.get("folder")}

    manifest_conds: Dict[int, Dict[str, Any]] = {}
    for item in manifest.get("conditions", []) if isinstance(manifest, dict) else []:
        try:
            idx = int(item.get("idx"))
        except Exception:
            continue
        manifest_conds[idx] = dict(item)

    results: List[Dict[str, Any]] = []
    for info in saved_blocks:
        i = int(info["cond"])
        folder = str(info["folder"])
        dec = decoded_by_folder.get(folder, {})
        m = manifest_conds.get(i, {})
        row: Dict[str, Any] = {}
        row.update(
            {
                "video": video_path.name,
                "rate_hz": meta.rate_hz if meta else "",
                "exp": meta.exp if meta else "",
                "fluoro": meta.fluoro if meta else "",
                "fps": f"{fps:.6f}",
                **info,
            }
        )
        if m:
            row.update(
                {
                    "image": m.get("image", ""),
                    "channel": m.get("channel", ""),
                    "token": m.get("token", ""),
                    "intensity": m.get("intensity", ""),
                }
            )
        if dec:
            for k, v in dec.items():
                row[f"decode_{k}"] = v
        results.append(row)

    results_csv = out_root / "results.csv"
    write_results_csv(results_csv, results)
    print(f"[OK] results: {results_csv}")


if __name__ == "__main__":
    main()

