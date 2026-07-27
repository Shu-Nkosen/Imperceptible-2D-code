"""統合 hully パイプライン: 条件ごとに d(t) を1回生成し、5 pass を in-process で実行。"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from decode_qr_from_all_frames import (
    DEFAULT_MEDIAN_ITERATIONS,
    MID_MEDIAN_KERNELS,
    process_array_task,
    resolve_search_mode,
)
from hully_diff import (
    MapSpec,
    SCALE_TAG,
    build_pair_diff_stack,
    generate_accum_maps,
    generate_fourier_maps,
    generate_lockin_maps,
    generate_pair_maps,
    generate_stat_maps,
    save_map_png,
)
from naming import VideoNameMeta, parse_video_name
from run_pipeline import (
    ANALYSIS_FRAME_COUNT,
    DIFF_THRESHOLDS_ACCUM,
    DIFF_THRESHOLDS_FOURIER,
    DIFF_THRESHOLDS_PAIR,
    DIFF_THRESHOLDS_STAT_STD,
    DIFF_THRESHOLDS_STAT_VAR,
    PAIR_OUTPUT_EACH_END,
    RESULTS_FIELDNAMES,
    WINDOW_NS_ACCUM,
    SyncConfig,
    best_accuracy_for_folder,
    condition_folder_name,
    count_consecutive_analysis_frames,
    detect_black_to_red_sync,
    ensure_dir,
    extract_gt_qr_mask,
    format_common_meta,
    has_reusable_analysis_frames,
    parse_decode_variant,
    parse_float_list,
    parse_int_list,
    pick_decode_row_for_folder,
    pixel_accuracy_for_folder,
    read_manifest,
    resolve_conditions,
    resolve_gt_qr_timeline,
    save_block_frames,
    select_first_last_pair_rows,
    set_quiet,
    write_csv,
    write_pair_accuracy_csv,
    write_results_csv,
)
from time_fft import resolve_target_freqs, resolve_target_freqs_hard

# (pass_label, diff_mode, stat_kind, repr_mode)
# lockin = d(t) 複素ロックイン / fourier = d(t) 帯付き FFT（別CSV）
# repr_mode: binary=固定th二値化 / gray=数値スコア正規化グレースケール（hard の *_num）
ALL_PASSES: Tuple[Tuple[str, str, str, str], ...] = (
    ("pair", "pair", "", "binary"),
    ("accum", "accum", "", "binary"),
    ("stat_std", "stat", "std", "binary"),
    ("stat_var", "stat", "var", "binary"),
    ("lockin", "lockin", "", "binary"),
    ("fourier", "fourier", "", "binary"),
)

# --hard-sweeps 時のみ追加: pair以外の数値スコア→grayデコード経路
HARD_NUM_PASSES: Tuple[Tuple[str, str, str, str], ...] = (
    ("accum_num", "accum", "", "gray"),
    ("stat_std_num", "stat", "std", "gray"),
    ("stat_var_num", "stat", "var", "gray"),
    ("lockin_num", "lockin", "", "gray"),
    ("fourier_num", "fourier", "", "gray"),
)

# --hard-sweeps 時の拡大スイープ（通常既定より密）
HARD_DIFF_THRESHOLDS_PAIR: Tuple[int, ...] = (2, 4, 6, 8, 10)
HARD_DIFF_THRESHOLDS_ACCUM: Tuple[int, ...] = (8, 12, 16, 20, 24, 32)
HARD_WINDOW_NS_ACCUM: Tuple[int, ...] = (3, 5)
HARD_DIFF_THRESHOLDS_STAT_STD: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16)
HARD_DIFF_THRESHOLDS_STAT_VAR: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)
HARD_DIFF_THRESHOLDS_FREQ: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16)
HARD_PAIR_EACH_END = 0  # 全隣接ペア
HARD_FOURIER_BAND_RADIUS = 2
HARD_PHASE_STEPS: Tuple[int, ...] = (4, 8, 16)

# --mid-sweeps: hard 相当の手法＋*_num、ただし pair/閾値/phase を軽く（≈hully の数倍）
MID_DIFF_THRESHOLDS_PAIR: Tuple[int, ...] = (2, 4, 6, 8, 10)
MID_DIFF_THRESHOLDS_ACCUM: Tuple[int, ...] = (12, 16, 24, 32)
MID_WINDOW_NS_ACCUM: Tuple[int, ...] = (3, 5)
MID_DIFF_THRESHOLDS_STAT_STD: Tuple[int, ...] = (4, 8, 12)
MID_DIFF_THRESHOLDS_STAT_VAR: Tuple[int, ...] = (1, 2, 4)
MID_DIFF_THRESHOLDS_FREQ: Tuple[int, ...] = (4, 8, 12)
MID_PAIR_EACH_END = 3
MID_FOURIER_BAND_RADIUS = 2
MID_PHASE_STEPS: Tuple[int, ...] = (8,)


@dataclass(frozen=True)
class SweepKey:
    diff_mode: str
    stat_kind: str
    window_n: Optional[int]
    fft_target_hz: Optional[float]
    diff_threshold: int
    phase_steps: Optional[int] = None
    repr_mode: str = "binary"


@dataclass
class Top3Candidate:
    sweep: SweepKey
    spec: MapSpec
    pass_label: str
    success: bool
    accuracy: float
    frame_1: str
    frame_2: str


def resolve_sweep_profile(ns: argparse.Namespace) -> str:
    """hard > mid > normal。両方指定時は hard 優先。"""
    if bool(getattr(ns, "hard_sweeps", False)):
        return "hard"
    if bool(getattr(ns, "mid_sweeps", False)):
        return "mid"
    return "normal"


def resolve_pass_list(ns: argparse.Namespace) -> Tuple[Tuple[str, str, str, str], ...]:
    if resolve_sweep_profile(ns) in ("hard", "mid"):
        return ALL_PASSES + HARD_NUM_PASSES
    return ALL_PASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "R8 hully: sync -> 60 conditions -> d(t) once -> passes in-process "
            "(6, or 11 with --hard-sweeps / --mid-sweeps)"
        )
    )
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--conditions", type=int, default=60)
    p.add_argument("--block-sec", type=float, default=6.0)
    p.add_argument("--use-start-sec", type=float, default=2.0)
    p.add_argument("--use-end-sec", type=float, default=4.0)
    p.add_argument("--slate-sec", type=float, default=None)
    p.add_argument("--padding-sec", type=float, default=None)
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--reuse-frames",
        dest="reuse_frames",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--force-extract",
        dest="reuse_frames",
        action="store_false",
    )
    p.add_argument(
        "--keep-frames",
        type=int,
        default=0,
        help="d(t) 生成後に残す frame PNG 枚数（0=削除、120=全部残す）",
    )
    p.add_argument(
        "--save-diff-maps",
        type=str,
        choices=["top3", "top5", "all", "none"],
        default="top3",
        help="差分 PNG 保存方針（既定: top3。hard/mid は top5）",
    )
    p.add_argument("--mid-search", action="store_true")
    p.add_argument("--full-search", action="store_true")
    p.add_argument("--window-ns", type=str, default="")
    p.add_argument("--diff-thresholds", type=str, default="")
    p.add_argument("--target-freqs", type=str, default="")
    p.add_argument(
        "--pair-each-end",
        type=int,
        default=PAIR_OUTPUT_EACH_END,
        help=f"pair 時の先頭/末尾デコードペア数（既定: {PAIR_OUTPUT_EACH_END}。0=全ペア）",
    )
    p.add_argument(
        "--hard-sweeps",
        action="store_true",
        help=(
            "閾値・窓・全ペア・band_radius・周波数候補・lockin phase_steps を拡大し、"
            "pair以外に数値スコア→grayデコード経路(*_num)も追加"
            "（all_analyze_hard 用。デコード mid/full は orchestrator 側）"
        ),
    )
    p.add_argument(
        "--mid-sweeps",
        action="store_true",
        help=(
            "hard 相当の *_num と周波数候補を残しつつ、pair は先頭/末尾3・"
            "閾値は間引き・phase_steps=8 のみ（all_analyze_mid 用）"
        ),
    )
    p.add_argument(
        "--fourier-band-radius",
        type=int,
        default=1,
        help="fourier(帯付きFFT) の target 前後ビン数（既定: 1）",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def log_info(msg: str) -> None:
    from run_pipeline import log_info as _log

    _log(msg)


def log_warn(msg: str) -> None:
    from run_pipeline import log_warn as _log

    _log(msg)


def delete_frame_pngs(cond_dir: Path) -> int:
    removed = 0
    for path in cond_dir.glob("frame_?????.png"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def load_rgb_frames(cond_dir: Path, count: int) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for i in range(count):
        path = cond_dir / f"frame_{i:05d}.png"
        img = cv2.imread(str(path))
        if img is None:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(rgb)
    return frames


def resolve_kernel_candidates(search_mode: str) -> List[int]:
    # mid / full ともメディアンカーネル 3/5/7（fast のみ既定5）
    if search_mode in ("mid", "full"):
        return list(MID_MEDIAN_KERNELS)
    return [5]


def build_sweeps(
    diff_mode: str,
    stat_kind: str,
    ns: argparse.Namespace,
    meta: Optional[VideoNameMeta],
    fps: float,
    repr_mode: str = "binary",
) -> Tuple[List[SweepKey], Tuple[float, ...]]:
    profile = resolve_sweep_profile(ns)
    gray = repr_mode == "gray"
    target_freqs: Tuple[float, ...] = ()
    if diff_mode == "accum":
        if profile == "hard":
            default_ns = HARD_WINDOW_NS_ACCUM
        elif profile == "mid":
            default_ns = MID_WINDOW_NS_ACCUM
        else:
            default_ns = WINDOW_NS_ACCUM
        window_ns = parse_int_list(ns.window_ns, default_ns)
        if gray:
            thresholds: Tuple[int, ...] = (0,)
        else:
            if profile == "hard":
                default_th = HARD_DIFF_THRESHOLDS_ACCUM
            elif profile == "mid":
                default_th = MID_DIFF_THRESHOLDS_ACCUM
            else:
                default_th = DIFF_THRESHOLDS_ACCUM
            thresholds = parse_int_list(ns.diff_thresholds, default_th)
        sweeps = [
            SweepKey(diff_mode, "", n, None, th, repr_mode=repr_mode)
            for n in window_ns
            for th in thresholds
        ]
    elif diff_mode == "stat":
        if gray:
            thresholds = (0,)
        elif profile == "hard":
            default_th = (
                HARD_DIFF_THRESHOLDS_STAT_VAR
                if stat_kind == "var"
                else HARD_DIFF_THRESHOLDS_STAT_STD
            )
            thresholds = parse_int_list(ns.diff_thresholds, default_th)
        elif profile == "mid":
            default_th = (
                MID_DIFF_THRESHOLDS_STAT_VAR
                if stat_kind == "var"
                else MID_DIFF_THRESHOLDS_STAT_STD
            )
            thresholds = parse_int_list(ns.diff_thresholds, default_th)
        else:
            default_th = DIFF_THRESHOLDS_STAT_VAR if stat_kind == "var" else DIFF_THRESHOLDS_STAT_STD
            thresholds = parse_int_list(ns.diff_thresholds, default_th)
        sweeps = [
            SweepKey(diff_mode, stat_kind, None, None, th, repr_mode=repr_mode)
            for th in thresholds
        ]
    elif diff_mode in ("fourier", "lockin"):
        if gray:
            thresholds = (0,)
        else:
            if profile == "hard":
                default_th = HARD_DIFF_THRESHOLDS_FREQ
            elif profile == "mid":
                default_th = MID_DIFF_THRESHOLDS_FREQ
            else:
                default_th = DIFF_THRESHOLDS_FOURIER
            thresholds = parse_int_list(ns.diff_thresholds, default_th)
        if ns.target_freqs.strip():
            target_freqs = parse_float_list(ns.target_freqs, ())
        elif meta is not None:
            resolver = (
                resolve_target_freqs_hard
                if profile in ("hard", "mid")
                else resolve_target_freqs
            )
            target_freqs = tuple(resolver(meta.rate_hz, fps))
        else:
            raise SystemExit(
                f"{diff_mode} には --target-freqs か rate_hz 付き動画名が必要です"
            )
        if diff_mode == "lockin":
            if profile == "hard":
                phase_list: Tuple[Optional[int], ...] = HARD_PHASE_STEPS
            elif profile == "mid":
                phase_list = MID_PHASE_STEPS
            else:
                phase_list = (8,)
            sweeps = [
                SweepKey(diff_mode, "", None, freq, th, phase_steps=ps, repr_mode=repr_mode)
                for freq in target_freqs
                for th in thresholds
                for ps in phase_list
            ]
        else:
            sweeps = [
                SweepKey(diff_mode, "", None, freq, th, repr_mode=repr_mode)
                for freq in target_freqs
                for th in thresholds
            ]
    else:
        if profile == "hard":
            default_th = HARD_DIFF_THRESHOLDS_PAIR
        elif profile == "mid":
            default_th = MID_DIFF_THRESHOLDS_PAIR
        else:
            default_th = DIFF_THRESHOLDS_PAIR
        thresholds = parse_int_list(ns.diff_thresholds, default_th)
        sweeps = [
            SweepKey(diff_mode, "", None, None, th, repr_mode=repr_mode)
            for th in thresholds
        ]
    return sweeps, target_freqs


def resolve_pair_each_end(ns: argparse.Namespace) -> int:
    profile = resolve_sweep_profile(ns)
    if int(ns.pair_each_end) == PAIR_OUTPUT_EACH_END:
        if profile == "hard":
            return HARD_PAIR_EACH_END
        if profile == "mid":
            return MID_PAIR_EACH_END
    return max(0, int(ns.pair_each_end))


def resolve_fourier_band_radius(ns: argparse.Namespace) -> int:
    profile = resolve_sweep_profile(ns)
    if int(ns.fourier_band_radius) == 1:
        if profile == "hard":
            return HARD_FOURIER_BAND_RADIUS
        if profile == "mid":
            return MID_FOURIER_BAND_RADIUS
    return max(0, int(ns.fourier_band_radius))


def generate_maps_for_sweep(
    d_stack: np.ndarray,
    sweep: SweepKey,
    fps: float,
    pair_each_end: int,
    fourier_band_radius: int = 1,
) -> List[MapSpec]:
    repr_mode = sweep.repr_mode if sweep.repr_mode in ("binary", "gray") else "binary"
    if sweep.diff_mode == "pair":
        return generate_pair_maps(d_stack, sweep.diff_threshold, pair_each_end)
    if sweep.diff_mode == "accum":
        assert sweep.window_n is not None
        return generate_accum_maps(
            d_stack,
            sweep.window_n,
            sweep.diff_threshold,
            pair_each_end,
            repr_mode=repr_mode,
        )
    if sweep.diff_mode == "stat":
        return generate_stat_maps(
            d_stack, sweep.stat_kind, sweep.diff_threshold, repr_mode=repr_mode
        )
    if sweep.diff_mode == "lockin":
        assert sweep.fft_target_hz is not None
        return generate_lockin_maps(
            d_stack,
            fps,
            sweep.fft_target_hz,
            sweep.diff_threshold,
            phase_steps=int(sweep.phase_steps) if sweep.phase_steps is not None else 8,
            repr_mode=repr_mode,
        )
    if sweep.diff_mode == "fourier":
        assert sweep.fft_target_hz is not None
        return generate_fourier_maps(
            d_stack,
            fps,
            sweep.fft_target_hz,
            sweep.diff_threshold,
            band_radius=fourier_band_radius,
            repr_mode=repr_mode,
        )
    return []


def relative_diff_path(folder_name: str, spec: MapSpec) -> str:
    return f"{folder_name}/{spec.diff_subdir}/{spec.pair_name}_{SCALE_TAG}.png"


def run_decode_batch(
    tasks: List[Dict[str, object]],
    workers: int,
) -> List[Dict[str, object]]:
    if not tasks:
        return []
    cpu_count = os.cpu_count() or 1
    pool_workers = min(max(1, workers), cpu_count, len(tasks))
    rows: List[Dict[str, object]] = []
    if pool_workers <= 1:
        for task in tasks:
            rows.append(process_array_task(task))
        return rows

    with ProcessPoolExecutor(max_workers=pool_workers) as executor:
        futures = [executor.submit(process_array_task, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def decode_row_from_result(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "folder": result["folder"],
        "frame_1": result["frame_1"],
        "frame_2": result["frame_2"],
        "diff_image": result.get("diff_image") or "",
        "analysis_image": "",
        "decoded_text": result.get("decoded_text") or "",
        "success": int(result.get("success") or 0),
        "method": result.get("method") or "",
        "note": result.get("note") or "",
        "recall": result.get("recall") or "",
        "precision": result.get("precision") or "",
        "accuracy": result.get("accuracy") or "",
        "noise": result.get("noise") or "",
        "gt_note": result.get("gt_note") or "",
        "task_key": result.get("task_key") or "",
    }


def _parse_acc(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return float("-inf")
    try:
        return float(text)
    except ValueError:
        return float("-inf")


def pick_top_n_candidates(
    candidates: List[Top3Candidate], n: int = 3
) -> List[Top3Candidate]:
    if not candidates or n <= 0:
        return []

    def sweep_id(c: Top3Candidate) -> Tuple[Any, ...]:
        s = c.sweep
        return (
            s.diff_mode,
            s.stat_kind,
            s.window_n,
            s.fft_target_hz,
            s.diff_threshold,
            s.phase_steps,
            s.repr_mode,
            c.frame_1,
            c.frame_2,
        )

    selected: List[Top3Candidate] = []
    used_ids: set[Tuple[Any, ...]] = set()

    successes = [c for c in candidates if c.success]
    if successes:
        best = max(successes, key=lambda c: c.accuracy)
        selected.append(best)
        used_ids.add(sweep_id(best))

    remaining = [c for c in candidates if sweep_id(c) not in used_ids]
    if remaining and len(selected) < n:
        best_acc = max(remaining, key=lambda c: c.accuracy)
        if best_acc.accuracy != float("-inf"):
            selected.append(best_acc)
            used_ids.add(sweep_id(best_acc))

    remaining = [c for c in candidates if sweep_id(c) not in used_ids]
    if remaining and len(selected) < n:
        remaining.sort(
            key=lambda c: (
                c.accuracy if c.accuracy != float("-inf") else -1.0,
                c.success,
            ),
            reverse=True,
        )
        for cand in remaining:
            if sweep_id(cand) in used_ids:
                continue
            if len(selected) >= n:
                break
            selected.append(cand)
            used_ids.add(sweep_id(cand))

    return selected[:n]


def pick_top3_candidates(candidates: List[Top3Candidate]) -> List[Top3Candidate]:
    return pick_top_n_candidates(candidates, n=3)


def save_top_maps(
    cond_dir: Path,
    folder_name: str,
    picks: List[Top3Candidate],
    out_root: Path,
    saved_paths: Dict[str, str],
) -> None:
    _ = out_root
    for pick in picks:
        rel = relative_diff_path(folder_name, pick.spec)
        out_path = cond_dir / pick.spec.diff_subdir / f"{pick.spec.pair_name}_{SCALE_TAG}.png"
        save_map_png(pick.spec.rgb, out_path)
        saved_paths[rel] = str(out_path.resolve())


def save_top3_maps(
    cond_dir: Path,
    folder_name: str,
    picks: List[Top3Candidate],
    out_root: Path,
    saved_paths: Dict[str, str],
) -> None:
    save_top_maps(cond_dir, folder_name, picks, out_root, saved_paths)


def save_all_maps(
    cond_dir: Path,
    folder_name: str,
    specs: List[MapSpec],
    saved_paths: Dict[str, str],
) -> None:
    for spec in specs:
        rel = relative_diff_path(folder_name, spec)
        out_path = cond_dir / spec.diff_subdir / f"{spec.pair_name}_{SCALE_TAG}.png"
        save_map_png(spec.rgb, out_path)
        saved_paths[rel] = str(out_path.resolve())


def tag_decode_rows(
    rows: List[Dict[str, object]],
    sweep: SweepKey,
    saved_paths: Dict[str, str],
    folder_name: str,
    specs_by_key: Dict[str, MapSpec],
) -> List[Dict[str, str]]:
    tagged: List[Dict[str, str]] = []
    for row in rows:
        out = {k: str(v) for k, v in row.items()}
        out["diff_mode"] = sweep.diff_mode
        out["window_n"] = "" if sweep.window_n is None else str(sweep.window_n)
        out["stat_kind"] = sweep.stat_kind if sweep.diff_mode == "stat" else ""
        out["fft_target_hz"] = "" if sweep.fft_target_hz is None else f"{sweep.fft_target_hz:.6f}"
        out["diff_threshold"] = str(sweep.diff_threshold)
        task_key = str(row.get("task_key") or "")
        spec = specs_by_key.get(task_key)
        if spec is not None:
            rel = relative_diff_path(folder_name, spec)
            if rel in saved_paths:
                out["diff_image"] = rel
        tagged.append(out)
    return tagged


def pass_output_paths(out_root: Path, pass_label: str) -> Tuple[Path, Path, Path]:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pass_label)
    results_csv = out_root / f"results_{safe}.csv"
    decode_csv = out_root / f"qr_decode_all_frames_{safe}.csv"
    pair_accuracy_csv = out_root / "pair_accuracy.csv"
    return results_csv, decode_csv, pair_accuracy_csv


def main() -> None:
    ns = parse_args()
    set_quiet(bool(ns.quiet))
    t0 = time.perf_counter()

    video_path = Path(ns.video).resolve()
    meta = parse_video_name(video_path)
    if meta is None:
        log_warn(f"[WARN] video name does not match pattern: {video_path.name}")

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
    log_info(f"[OK] SYNC at frame={sync_frame}, fps={fps:.4f}")

    use_start = int(round(float(ns.use_start_sec) * fps))
    use_end = int(round(float(ns.use_end_sec) * fps))
    if use_end <= use_start:
        raise SystemExit("use-end-sec must be > use-start-sec")

    conditions = resolve_conditions(manifest, int(ns.conditions))
    cond_starts_sec, gt_slots, qr_sec = resolve_gt_qr_timeline(
        manifest, slate_sec, padding_sec, block_sec, len(conditions)
    )

    gt_path = out_root / "frame_QR.png"
    if ns.reuse_frames and gt_path.exists():
        log_info(f"[INFO] reuse GT QR mask: {gt_path.name}")
    else:
        extract_gt_qr_mask(video_path, sync_frame, fps, gt_slots, gt_path)

    search_mode = resolve_search_mode(ns.mid_search, ns.full_search)
    kernel_candidates = resolve_kernel_candidates(search_mode)
    gt_path_str = str(gt_path.resolve())

    pair_each_end = resolve_pair_each_end(ns)
    fourier_band_radius = resolve_fourier_band_radius(ns)
    profile = resolve_sweep_profile(ns)
    if profile in ("hard", "mid"):
        log_info(
            f"[INFO] {profile}-sweeps on: pair_each_end={pair_each_end} "
            f"fourier_band_radius={fourier_band_radius} "
            f"passes={len(resolve_pass_list(ns))} (incl. *_num gray)"
        )

    passes = resolve_pass_list(ns)
    pass_state: Dict[str, Dict[str, Any]] = {}
    for pass_label, diff_mode, stat_kind, repr_mode in passes:
        results_csv, decode_csv, pair_accuracy_csv = pass_output_paths(out_root, pass_label)
        sweeps, target_freqs = build_sweeps(
            diff_mode, stat_kind, ns, meta, fps, repr_mode=repr_mode
        )
        pass_state[pass_label] = {
            "pass_label": pass_label,
            "diff_mode": diff_mode,
            "stat_kind": stat_kind,
            "repr_mode": repr_mode,
            "sweeps": sweeps,
            "target_freqs": target_freqs,
            "results_csv": results_csv,
            "decode_csv": decode_csv,
            "pair_accuracy_csv": pair_accuracy_csv,
            "results": [],
            "all_decode_rows": [],
            "all_pair_accuracy_rows": [],
        }
        log_info(
            f"[INFO] pass={pass_label} diff_mode={diff_mode} "
            f"stat_kind={stat_kind or '-'} repr={repr_mode} sweeps={len(sweeps)}"
            + (
                f" band_radius={fourier_band_radius}"
                if diff_mode == "fourier"
                else ""
            )
        )

    reused_conditions = 0
    extracted_conditions = 0

    for i, cond in enumerate(conditions):
        block_start = sync_frame + int(round(cond_starts_sec[i] * fps))
        start = block_start + use_start
        end = block_start + use_end
        folder_name = condition_folder_name(cond)
        cond_dir = out_root / folder_name
        ensure_dir(cond_dir)

        if ns.reuse_frames and has_reusable_analysis_frames(cond_dir, ANALYSIS_FRAME_COUNT):
            analyzed = count_consecutive_analysis_frames(cond_dir)
            extract_sec = 0.0
            reused_conditions += 1
            log_info(f"[INFO] ({i+1}/{len(conditions)}) reuse {folder_name}: frames={analyzed}")
        else:
            analyzed = save_block_frames(video_path, start, end, cond_dir, analysis_frames=ANALYSIS_FRAME_COUNT)
            extract_sec = (analyzed / fps) if fps > 0 else 0.0
            extracted_conditions += 1
            log_info(f"[INFO] ({i+1}/{len(conditions)}) extract {folder_name}: frames={analyzed}")

        rgb_list = load_rgb_frames(cond_dir, analyzed)
        d_stack = build_pair_diff_stack(rgb_list)

        if int(ns.keep_frames) <= 0:
            removed = delete_frame_pngs(cond_dir)
            log_info(f"[INFO] ({i+1}/{len(conditions)}) deleted {removed} frame PNGs (keep-frames=0)")

        top3_pool: List[Top3Candidate] = []
        all_specs_for_save: List[MapSpec] = []

        for pass_label, diff_mode, stat_kind, repr_mode in passes:
            state = pass_state[pass_label]
            sweeps: List[SweepKey] = state["sweeps"]
            folder_decode_rows: List[Dict[str, str]] = []
            sweep_candidates: List[
                Tuple[
                    Optional[int],
                    Optional[float],
                    int,
                    Dict[str, str],
                    List[Dict[str, str]],
                    float,
                    bool,
                    int,
                    int,
                ]
            ] = []
            target_freqs: Tuple[float, ...] = state["target_freqs"]
            freq_rank_map = {freq: idx for idx, freq in enumerate(target_freqs)}

            pass_tasks: List[Dict[str, object]] = []
            task_specs: Dict[str, MapSpec] = {}
            task_sweeps: Dict[str, SweepKey] = {}

            for sweep in sweeps:
                specs = generate_maps_for_sweep(
                    d_stack,
                    sweep,
                    fps,
                    pair_each_end if diff_mode == "pair" else 0,
                    fourier_band_radius=fourier_band_radius,
                )
                if ns.save_diff_maps == "all":
                    all_specs_for_save.extend(specs)

                for spec_idx, spec in enumerate(specs):
                    task_key = (
                        f"{pass_label}:{sweep.diff_mode}:{sweep.stat_kind}:"
                        f"{sweep.window_n}:{sweep.fft_target_hz}:{sweep.diff_threshold}:"
                        f"{sweep.phase_steps}:{sweep.repr_mode}:{spec_idx}"
                    )
                    pass_tasks.append(
                        {
                            "rgb": spec.rgb,
                            "folder": folder_name,
                            "frame_1": spec.frame_1,
                            "frame_2": spec.frame_2,
                            "diff_image": "",
                            "kernel_candidates": kernel_candidates,
                            "median_iterations": DEFAULT_MEDIAN_ITERATIONS,
                            "search_mode": search_mode,
                            "gt_path": gt_path_str,
                            "task_key": task_key,
                            "diff_stem": spec.pair_name,
                        }
                    )
                    task_specs[task_key] = spec
                    task_sweeps[task_key] = sweep

            decode_results = run_decode_batch(pass_tasks, int(ns.workers))
            decode_results = [decode_row_from_result(r) for r in decode_results]

            results_by_sweep: Dict[Tuple[Any, ...], List[Dict[str, object]]] = {}
            for result in decode_results:
                task_key = str(result.get("task_key") or "")
                sweep = task_sweeps.get(task_key)
                if sweep is None:
                    continue
                key = (
                    sweep.window_n,
                    sweep.fft_target_hz,
                    sweep.diff_threshold,
                    sweep.phase_steps,
                )
                results_by_sweep.setdefault(key, []).append(result)

                acc = _parse_acc(result.get("accuracy"))
                success = bool(result.get("success"))
                spec = task_specs[task_key]
                top3_pool.append(
                    Top3Candidate(
                        sweep=sweep,
                        spec=spec,
                        pass_label=pass_label,
                        success=success,
                        accuracy=acc,
                        frame_1=str(result.get("frame_1") or spec.frame_1),
                        frame_2=str(result.get("frame_2") or spec.frame_2),
                    )
                )

            saved_paths: Dict[str, str] = {}
            for sweep in sweeps:
                key = (
                    sweep.window_n,
                    sweep.fft_target_hz,
                    sweep.diff_threshold,
                    sweep.phase_steps,
                )
                rows = results_by_sweep.get(key, [])
                specs_for_sweep = [
                    spec
                    for tk, spec in task_specs.items()
                    if task_sweeps.get(tk) == sweep
                ]
                specs_by_key = {
                    tk: spec for tk, spec in task_specs.items() if task_sweeps.get(tk) == sweep
                }
                tagged = tag_decode_rows(rows, sweep, saved_paths, folder_name, specs_by_key)
                folder_decode_rows.extend(tagged)

                th_rows = tagged
                dec_th = pick_decode_row_for_folder(th_rows, folder_name)
                ok = str(dec_th.get("success", "")).strip() in ("1", "True", "true")
                acc_all_str, _ = pixel_accuracy_for_folder(th_rows, folder_name)
                acc_all_val = float(acc_all_str) if acc_all_str else float("-inf")
                freq_rank = freq_rank_map.get(sweep.fft_target_hz, 0) if sweep.fft_target_hz is not None else 0
                phase_rank = (
                    {4: 0, 8: 1, 16: 2}.get(int(sweep.phase_steps), 9)
                    if sweep.phase_steps is not None
                    else 0
                )
                sweep_candidates.append(
                    (
                        sweep.window_n,
                        sweep.fft_target_hz,
                        sweep.diff_threshold,
                        dec_th,
                        th_rows,
                        acc_all_val,
                        ok,
                        freq_rank,
                        phase_rank,
                    )
                )

            if sweep_candidates:
                def _sort_key(item: Tuple[Any, ...]) -> Tuple[int, float, int, int, int, int, int]:
                    win_n, _fft, th, _dec, _rows, acc, ok, freq_rank, phase_rank = item
                    has_acc = acc != float("-inf")
                    return (
                        0 if has_acc else 1,
                        -acc if has_acc else 0.0,
                        0 if ok else 1,
                        th,
                        win_n if win_n is not None else 0,
                        freq_rank,
                        phase_rank,
                    )

                sweep_candidates.sort(key=_sort_key)
                best_window_n, best_fft_freq, best_th, best_dec, best_rows, _, best_ok, _, _ = sweep_candidates[0]
            else:
                best_window_n = best_fft_freq = best_th = None
                best_dec = {}
                best_rows = []
                best_ok = False

            decode_note = ""
            if analyzed <= 0:
                decode_note = "extract失敗: 0 frames"

            state["all_decode_rows"] = [
                r for r in state["all_decode_rows"] if r.get("folder") != folder_name
            ]
            state["all_decode_rows"].extend(folder_decode_rows)
            write_csv(state["decode_csv"], state["all_decode_rows"])

            dec = best_dec
            pixel_acc_all, pixel_acc_ok = pixel_accuracy_for_folder(best_rows, folder_name)
            pixel_acc_best, best_pair_f1, best_pair_f2 = best_accuracy_for_folder(best_rows, folder_name)
            method = dec.get("method", "") if dec else ""
            decode_variant = parse_decode_variant(method) if method else ""

            if decode_note:
                decode_note_out = decode_note
            elif not dec:
                decode_note_out = "デコード結果なし"
            elif str(dec.get("success", "")).strip() in ("1", "True", "true"):
                decode_note_out = ""
            else:
                decode_note_out = str(dec.get("note") or "QR未検出")
                decode_variant = ""

            row: Dict[str, Any] = {
                "folder": folder_name,
                "decode_note": decode_note_out,
                "decode_variant": decode_variant if not decode_note_out else "",
                "diff_mode": diff_mode,
                "window_n": "" if best_window_n is None else str(best_window_n),
                "stat_kind": stat_kind if diff_mode == "stat" else "",
                "fft_target_hz": "" if best_fft_freq is None else f"{best_fft_freq:.6f}",
                "diff_threshold": "" if best_th is None else str(best_th),
                "pixel_acc_all": pixel_acc_all,
                "pixel_acc_ok": pixel_acc_ok,
                "pixel_acc_best": pixel_acc_best,
                "best_pair_frame_1": best_pair_f1,
                "best_pair_frame_2": best_pair_f2,
                "cond": i,
                "image": cond.get("image", ""),
                "channel": cond.get("channel", ""),
                "token": cond.get("token", ""),
                "intensity": cond.get("intensity", ""),
                "decode_decoded_text": dec.get("decoded_text", "") if dec else "",
                "decode_method": method if not decode_note_out else "",
                "decode_frame_1": dec.get("frame_1", "") if dec else "",
                "decode_frame_2": dec.get("frame_2", "") if dec else "",
                "note": decode_note,
                "analysis_frames": analyzed,
                "extract_sec": f"{extract_sec:.6f}",
            }
            if i == 0:
                row.update(format_common_meta(video_path.name, meta, fps))
            else:
                row.update(
                    {
                        "video": "",
                        "display_rate": "",
                        "exposure": "",
                        "fluorescent": "",
                        "camera_fps": "",
                    }
                )
            state["results"].append(row)
            write_results_csv(state["results_csv"], state["results"])

            if diff_mode == "pair" and best_rows:
                state["all_pair_accuracy_rows"] = [
                    r for r in state["all_pair_accuracy_rows"] if r.get("folder") != folder_name
                ]
                for pr in select_first_last_pair_rows(best_rows):
                    state["all_pair_accuracy_rows"].append(
                        {
                            "folder": folder_name,
                            "cond": i,
                            "frame_1": pr.get("frame_1", ""),
                            "frame_2": pr.get("frame_2", ""),
                            "success": pr.get("success", ""),
                            "accuracy": pr.get("accuracy", ""),
                            "recall": pr.get("recall", ""),
                            "precision": pr.get("precision", ""),
                            "diff_threshold": pr.get("diff_threshold", ""),
                        }
                    )
                write_pair_accuracy_csv(state["pair_accuracy_csv"], state["all_pair_accuracy_rows"])

        saved_paths_final: Dict[str, str] = {}
        picks: List[Top3Candidate] = []
        top_n = {"top3": 3, "top5": 5}.get(str(ns.save_diff_maps), 0)
        if top_n > 0:
            picks = pick_top_n_candidates(top3_pool, n=top_n)
            save_top_maps(cond_dir, folder_name, picks, out_root, saved_paths_final)
            log_info(
                f"[INFO] ({i+1}/{len(conditions)}) saved top{top_n}={len(picks)} diff PNGs"
            )
        elif ns.save_diff_maps == "all":
            save_all_maps(cond_dir, folder_name, all_specs_for_save, saved_paths_final)
            log_info(
                f"[INFO] ({i+1}/{len(conditions)}) saved all diff PNGs={len(all_specs_for_save)}"
            )

        if saved_paths_final:
            saved_rels = set(saved_paths_final.keys())
            for pass_label in pass_state:
                state = pass_state[pass_label]
                rows = state["all_decode_rows"]
                for row in rows:
                    if row.get("folder") != folder_name:
                        continue
                    rel_hint = str(row.get("diff_image") or "")
                    if rel_hint and rel_hint not in saved_rels:
                        row["diff_image"] = ""
                for pick in picks if top_n > 0 else []:
                    rel = relative_diff_path(folder_name, pick.spec)
                    for row in rows:
                        if row.get("folder") != folder_name:
                            continue
                        if (
                            row.get("frame_1") == pick.frame_1
                            and row.get("frame_2") == pick.frame_2
                            and str(row.get("diff_threshold")) == str(pick.sweep.diff_threshold)
                            and row.get("diff_mode") == pick.sweep.diff_mode
                        ):
                            row["diff_image"] = rel
                if ns.save_diff_maps == "all":
                    for row in rows:
                        if row.get("folder") != folder_name:
                            continue
                        f1 = row.get("frame_1", "")
                        f2 = row.get("frame_2", "")
                        for rel in saved_rels:
                            if rel.startswith(f"{folder_name}/") and f1 in rel and f2 in rel:
                                row["diff_image"] = rel
                                break
                write_csv(state["decode_csv"], rows)

    elapsed = time.perf_counter() - t0
    log_info(
        f"[OK] hully finished in {elapsed:.1f}s "
        f"(extracted={extracted_conditions}, reused={reused_conditions})"
    )
    timing_path = out_root / "hully_timing.txt"
    timing_path.write_text(f"elapsed_sec={elapsed:.3f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
