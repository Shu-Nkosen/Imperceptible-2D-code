"""既存 out_mid_fast_* に、現行 mid_fast 定義で足りないスイープだけ追記する。

既定対象: fourier / lockin（binary）。旧 th=4/8/12 行は残し、p30/p50/p70/otsu/adapt 等を追加。
フレームは mid_fast と同様 keep-frames=0 のため、不足がある条件は動画から再抽出する。
"""
from __future__ import annotations

import argparse
import csv
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from decode_qr_from_all_frames import (
    DEFAULT_MEDIAN_ITERATIONS,
    resolve_search_mode,
)
from hully_diff import build_pair_diff_stack
from naming import VideoNameMeta, parse_video_name
from run_pipeline import (
    ANALYSIS_FRAME_COUNT,
    SyncConfig,
    best_accuracy_for_folder,
    condition_folder_name,
    detect_black_to_red_sync,
    ensure_dir,
    extract_gt_qr_mask,
    format_common_meta,
    parse_decode_variant,
    pick_decode_row_for_folder,
    pixel_accuracy_for_folder,
    read_manifest,
    resolve_conditions,
    resolve_gt_qr_timeline,
    save_block_frames,
    set_quiet,
    write_csv,
    write_results_csv,
    write_results_sweep_csv,
)
from run_pipeline_hully import (
    ALL_PASSES,
    HARD_NUM_PASSES,
    SweepKey,
    build_sweeps,
    decode_row_from_result,
    delete_frame_pngs,
    generate_maps_for_sweep,
    load_rgb_frames,
    log_info,
    log_warn,
    make_decode_executor,
    pass_output_paths,
    resolve_fourier_band_radius,
    resolve_kernel_candidates,
    resolve_pair_each_end,
    run_decode_batch,
    tag_decode_rows,
)

FREQ_MATCH_TOL_HZ = 0.05
DEFAULT_PASSES = ("fourier", "lockin")

PASS_LOOKUP: Dict[str, Tuple[str, str, str]] = {
    label: (diff_mode, stat_kind, repr_mode)
    for label, diff_mode, stat_kind, repr_mode in (ALL_PASSES + HARD_NUM_PASSES)
}


@dataclass(frozen=True)
class SweepId:
    folder: str
    fft_target_hz: Optional[float]
    diff_threshold: int
    phase_steps: Optional[int]
    window_n: Optional[int]


@dataclass
class VideoStats:
    stem: str
    status: str
    missing_sweeps: int
    conditions_touched: int
    elapsed_sec: float
    note: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "既存 out_mid_fast_* に対し、現行 mid_fast スイープ定義で足りない行だけ追記する。"
            "既定は fourier/lockin binary（新 th: p30/p50/p70/otsu/adapt）。"
        )
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="追記先の既存 mid_fast 出力ルート（例: R8/analyze_code/out_mid_fast_0805）",
    )
    p.add_argument("--movie-dir", type=str, default="")
    p.add_argument("--manifest-dir", type=str, default="")
    p.add_argument(
        "--passes",
        type=str,
        default=",".join(DEFAULT_PASSES),
        help="対象 pass（カンマ区切り。既定: fourier,lockin）",
    )
    p.add_argument(
        "--stem",
        type=str,
        default="",
        help="1本だけ処理（例: r60_e250_f1）。省略時は out-dir 内の全 stem",
    )
    p.add_argument("--dry-run", action="store_true", help="不足件数だけ表示して終了")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--decode-pool",
        type=str,
        choices=["thread", "process"],
        default="thread",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--conditions", type=int, default=60)
    p.add_argument("--block-sec", type=float, default=6.0)
    p.add_argument("--use-start-sec", type=float, default=2.0)
    p.add_argument("--use-end-sec", type=float, default=4.0)
    return p.parse_args()


def default_paths() -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    r8_dir = script_dir.parent
    return r8_dir / "movie", r8_dir / "make_movie" / "manifests"


def mid_fast_namespace(ns: argparse.Namespace) -> Namespace:
    """build_sweeps / resolve_* が mid_fast を選ぶための Namespace。"""
    return Namespace(
        mid_fast_sweeps=True,
        mid_sweeps=False,
        hard_sweeps=False,
        window_ns="",
        diff_thresholds="",
        target_freqs="",
        pair_each_end=20,  # resolve_pair_each_end が mid_fast 既定に置換
        fourier_band_radius=1,  # resolve_fourier_band_radius が mid_fast 既定に置換
        mid_search=True,
        full_search=False,
        workers=int(ns.workers),
        decode_pool=str(ns.decode_pool),
        quiet=bool(ns.quiet),
    )


def resolve_manifest(video_path: Path, manifest_dir: Path) -> Optional[Path]:
    meta = parse_video_name(video_path)
    if meta is None:
        return None
    path = manifest_dir / f"r{meta.rate_hz}_e{meta.exp}.json"
    return path if path.is_file() else None


def discover_stems(out_dir: Path, stem_filter: str) -> List[str]:
    if stem_filter.strip():
        return [stem_filter.strip()]
    stems: List[str] = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_dir():
            continue
        meta = parse_video_name(path.name + ".mp4")
        if meta is None:
            continue
        stems.append(path.name)
    return stems


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in csv.DictReader(f)]


def parse_optional_float(text: str) -> Optional[float]:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_optional_int(text: str) -> Optional[int]:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def freqs_close(a: Optional[float], b: Optional[float], tol: float = FREQ_MATCH_TOL_HZ) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def sweep_id_from_key(sweep: SweepKey, folder: str) -> SweepId:
    return SweepId(
        folder=folder,
        fft_target_hz=sweep.fft_target_hz,
        diff_threshold=int(sweep.diff_threshold),
        phase_steps=sweep.phase_steps,
        window_n=sweep.window_n,
    )


def sweep_id_from_row(row: Dict[str, str]) -> SweepId:
    return SweepId(
        folder=str(row.get("folder") or ""),
        fft_target_hz=parse_optional_float(str(row.get("fft_target_hz") or "")),
        diff_threshold=int(parse_optional_int(str(row.get("diff_threshold") or "0")) or 0),
        phase_steps=parse_optional_int(str(row.get("phase_steps") or "")),
        window_n=parse_optional_int(str(row.get("window_n") or "")),
    )


def sweep_ids_match(a: SweepId, b: SweepId) -> bool:
    return (
        a.folder == b.folder
        and a.diff_threshold == b.diff_threshold
        and a.phase_steps == b.phase_steps
        and a.window_n == b.window_n
        and freqs_close(a.fft_target_hz, b.fft_target_hz)
    )


def find_matching_existing(
    existing_ids: Sequence[SweepId], target: SweepId
) -> bool:
    return any(sweep_ids_match(existing, target) for existing in existing_ids)


def parse_acc(text: str) -> float:
    s = str(text or "").strip()
    if not s:
        return float("-inf")
    try:
        return float(s)
    except ValueError:
        return float("-inf")


def adoption_sort_key(
    row: Dict[str, str],
    freq_rank_map: Dict[float, int],
) -> Tuple[int, float, int, int, int, int, int]:
    acc = parse_acc(str(row.get("pixel_acc_all") or ""))
    ok = str(row.get("decode_success") or "").strip() in ("1", "True", "true")
    th = int(parse_optional_int(str(row.get("diff_threshold") or "0")) or 0)
    win_n = parse_optional_int(str(row.get("window_n") or "")) or 0
    fft = parse_optional_float(str(row.get("fft_target_hz") or ""))
    freq_rank = 0
    if fft is not None:
        # nearest expected freq rank
        best = None
        for freq, rank in freq_rank_map.items():
            if freqs_close(fft, freq):
                best = rank
                break
        freq_rank = 0 if best is None else best
    phase = parse_optional_int(str(row.get("phase_steps") or ""))
    phase_rank = {4: 0, 8: 1, 16: 2}.get(int(phase), 9) if phase is not None else 0
    has_acc = acc != float("-inf")
    return (
        0 if has_acc else 1,
        -acc if has_acc else 0.0,
        0 if ok else 1,
        th,
        win_n,
        freq_rank,
        phase_rank,
    )


def recompute_adopted(
    rows: List[Dict[str, str]],
    target_freqs: Sequence[float],
) -> List[Dict[str, str]]:
    freq_rank_map = {float(f): i for i, f in enumerate(target_freqs)}
    by_folder: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        folder = str(row.get("folder") or "")
        by_folder.setdefault(folder, []).append(idx)

    out = [dict(r) for r in rows]
    for folder, idxs in by_folder.items():
        best_i = min(idxs, key=lambda i: adoption_sort_key(out[i], freq_rank_map))
        for i in idxs:
            out[i]["adopted"] = "1" if i == best_i else "0"
    return out


def results_row_from_adopted_sweep(
    sweep_row: Dict[str, str],
    *,
    diff_mode: str,
    stat_kind: str,
) -> Dict[str, Any]:
    ok = str(sweep_row.get("decode_success") or "").strip() in ("1", "True", "true")
    note = str(sweep_row.get("decode_note") or "")
    row: Dict[str, Any] = {
        "folder": sweep_row.get("folder", ""),
        "decode_note": "" if ok else note,
        "decode_variant": sweep_row.get("decode_variant", "") if ok else "",
        "diff_mode": diff_mode,
        "window_n": sweep_row.get("window_n", ""),
        "stat_kind": stat_kind if diff_mode == "stat" else "",
        "fft_target_hz": sweep_row.get("fft_target_hz", ""),
        "diff_threshold": sweep_row.get("diff_threshold", ""),
        "pixel_acc_all": sweep_row.get("pixel_acc_all", ""),
        "pixel_acc_ok": sweep_row.get("pixel_acc_ok", ""),
        "pixel_acc_best": sweep_row.get("pixel_acc_best", ""),
        "best_pair_frame_1": sweep_row.get("best_pair_frame_1", ""),
        "best_pair_frame_2": sweep_row.get("best_pair_frame_2", ""),
        "cond": sweep_row.get("cond", ""),
        "image": sweep_row.get("image", ""),
        "channel": sweep_row.get("channel", ""),
        "token": sweep_row.get("token", ""),
        "intensity": sweep_row.get("intensity", ""),
        "decode_decoded_text": sweep_row.get("decode_decoded_text", "") if ok else "",
        "decode_method": sweep_row.get("decode_method", "") if ok else "",
        "decode_frame_1": sweep_row.get("decode_frame_1", ""),
        "decode_frame_2": sweep_row.get("decode_frame_2", ""),
        "note": sweep_row.get("note", ""),
        "analysis_frames": sweep_row.get("analysis_frames", ""),
        "extract_sec": sweep_row.get("extract_sec", ""),
    }
    for key in (
        "video",
        "display_rate",
        "exposure",
        "fluorescent",
        "camera_fps",
        "rate_hz",
        "exposure_denom",
        "fluorescent_flag",
    ):
        row[key] = sweep_row.get(key, "")
    return row


def decode_fieldnames(existing: Sequence[Dict[str, str]], new_rows: Sequence[Dict[str, str]]) -> List[str]:
    keys: List[str] = []
    seen: Set[str] = set()
    for row in list(existing[:1]) + list(new_rows[:1]):
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    preferred = [
        "folder",
        "frame_1",
        "frame_2",
        "diff_image",
        "analysis_image",
        "decoded_text",
        "success",
        "method",
        "note",
        "recall",
        "precision",
        "accuracy",
        "noise",
        "gt_note",
        "diff_mode",
        "window_n",
        "stat_kind",
        "fft_target_hz",
        "diff_threshold",
        "task_key",
    ]
    ordered = [k for k in preferred if k in seen]
    ordered.extend([k for k in keys if k not in ordered])
    return ordered


def list_missing_sweeps(
    expected: Sequence[SweepKey],
    existing_rows: Sequence[Dict[str, str]],
    folders: Sequence[str],
) -> Dict[str, List[SweepKey]]:
    existing_ids = [sweep_id_from_row(r) for r in existing_rows]
    missing: Dict[str, List[SweepKey]] = {}
    for folder in folders:
        folder_existing = [e for e in existing_ids if e.folder == folder]
        need: List[SweepKey] = []
        for sweep in expected:
            sid = sweep_id_from_key(sweep, folder)
            if not find_matching_existing(folder_existing, sid):
                need.append(sweep)
        if need:
            missing[folder] = need
    return missing


def run_missing_for_folder(
    *,
    d_stack,
    missing_sweeps: Sequence[SweepKey],
    pass_label: str,
    diff_mode: str,
    folder_name: str,
    fps: float,
    pair_each_end: int,
    fourier_band_radius: int,
    kernel_candidates: List[int],
    search_mode: str,
    gt_path_str: str,
    workers: int,
    decode_executor,
    decode_pool: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """不足スイープを実行し、(sweep_result_rows_without_meta_fill, decode_rows) を返す。

    sweep rows は folder/pass/metrics まで埋め、video meta と adopted は呼び出し側で付与する。
    """
    pass_tasks: List[Dict[str, object]] = []
    task_specs = {}
    task_sweeps = {}

    for sweep in missing_sweeps:
        specs = generate_maps_for_sweep(
            d_stack,
            sweep,
            fps,
            pair_each_end if diff_mode == "pair" else 0,
            fourier_band_radius=fourier_band_radius,
        )
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

    decode_results = run_decode_batch(
        pass_tasks,
        workers,
        executor=decode_executor,
        pool_kind=decode_pool,
    )
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

    saved_paths: Dict[str, str] = {}
    sweep_rows: List[Dict[str, str]] = []
    decode_rows: List[Dict[str, str]] = []

    for sweep in missing_sweeps:
        key = (
            sweep.window_n,
            sweep.fft_target_hz,
            sweep.diff_threshold,
            sweep.phase_steps,
        )
        rows = results_by_sweep.get(key, [])
        specs_by_key = {
            tk: spec for tk, spec in task_specs.items() if task_sweeps.get(tk) == sweep
        }
        tagged = tag_decode_rows(rows, sweep, saved_paths, folder_name, specs_by_key)
        decode_rows.extend(tagged)

        dec_th = pick_decode_row_for_folder(tagged, folder_name)
        ok = str(dec_th.get("success", "")).strip() in ("1", "True", "true")
        pixel_acc_all_s, pixel_acc_ok_s = pixel_accuracy_for_folder(tagged, folder_name)
        pixel_acc_best_s, best_f1_s, best_f2_s = best_accuracy_for_folder(tagged, folder_name)
        method_s = str(dec_th.get("method", "") or "")
        variant_s = parse_decode_variant(method_s) if method_s else ""
        if ok:
            note_s = ""
        else:
            note_s = str(dec_th.get("note") or "QR未検出") if dec_th else "デコード結果なし"
            variant_s = ""

        sweep_rows.append(
            {
                "folder": folder_name,
                "pass": pass_label,
                "adopted": "0",
                "decode_success": "1" if ok else "0",
                "decode_note": note_s,
                "decode_variant": variant_s if ok else "",
                "diff_mode": sweep.diff_mode,
                "repr_mode": sweep.repr_mode,
                "window_n": "" if sweep.window_n is None else str(sweep.window_n),
                "stat_kind": sweep.stat_kind if sweep.diff_mode == "stat" else "",
                "fft_target_hz": (
                    "" if sweep.fft_target_hz is None else f"{sweep.fft_target_hz:.6f}"
                ),
                "diff_threshold": str(sweep.diff_threshold),
                "phase_steps": (
                    "" if sweep.phase_steps is None else str(sweep.phase_steps)
                ),
                "n_maps": str(len(tagged)),
                "pixel_acc_all": pixel_acc_all_s,
                "pixel_acc_ok": pixel_acc_ok_s,
                "pixel_acc_best": pixel_acc_best_s,
                "best_pair_frame_1": best_f1_s,
                "best_pair_frame_2": best_f2_s,
                "decode_decoded_text": (
                    str(dec_th.get("decoded_text", "") or "") if ok else ""
                ),
                "decode_method": method_s if ok else "",
                "decode_frame_1": str(dec_th.get("frame_1", "") or ""),
                "decode_frame_2": str(dec_th.get("frame_2", "") or ""),
                "note": note_s,
            }
        )

    return sweep_rows, decode_rows


def infer_fps_from_rows(rows: Sequence[Dict[str, str]], default: float = 60.0) -> float:
    for row in rows:
        text = str(row.get("camera_fps") or "")
        # e.g. "60.009 fps"
        token = text.replace("fps", "").strip().split()[0] if text else ""
        try:
            val = float(token)
            if val > 1.0:
                return val
        except ValueError:
            continue
    return default


def process_video(
    *,
    stem: str,
    out_dir: Path,
    movie_dir: Path,
    manifest_dir: Path,
    pass_labels: Sequence[str],
    dry_run: bool,
    ns: argparse.Namespace,
) -> VideoStats:
    t0 = time.perf_counter()
    out_root = out_dir / stem
    if not out_root.is_dir():
        return VideoStats(stem, "skip", 0, 0, 0.0, note="out stem missing")

    video_path = movie_dir / f"{stem}.mp4"
    meta = parse_video_name(f"{stem}.mp4")
    hully_ns = mid_fast_namespace(ns)

    # 既存 CSV から folder 一覧・fps を拾う（dry-run / 動画無しでも不足検出できるように）
    sample_rows: List[Dict[str, str]] = []
    for pass_label in pass_labels:
        if pass_label not in PASS_LOOKUP:
            continue
        _, sweep_csv, _, _ = pass_output_paths(out_root, pass_label)
        sample_rows = read_csv_dicts(sweep_csv)
        if sample_rows:
            break
    folders_from_csv = sorted({str(r.get("folder") or "") for r in sample_rows if r.get("folder")})
    fps_guess = infer_fps_from_rows(sample_rows)

    video_ok = video_path.is_file()
    manifest_path = resolve_manifest(video_path, manifest_dir) if video_ok else None
    if not video_ok and not dry_run:
        return VideoStats(stem, "error", 0, 0, 0.0, note=f"movie missing: {video_path}")
    if video_ok and manifest_path is None and not dry_run:
        return VideoStats(stem, "error", 0, 0, 0.0, note="manifest missing")

    if video_ok and manifest_path is not None:
        manifest = read_manifest(manifest_path)
        slate_sec = float(manifest.get("slate_sec", 0.5))
        padding_sec = float(manifest.get("padding_sec", 5.0))
        block_sec = float(ns.block_sec)
        cfg = SyncConfig()
        sync_frame, fps = detect_black_to_red_sync(video_path, cfg)
        use_start = int(round(float(ns.use_start_sec) * fps))
        use_end = int(round(float(ns.use_end_sec) * fps))
        conditions = resolve_conditions(manifest, int(ns.conditions))
        folders = [condition_folder_name(c) for c in conditions]
        cond_by_folder = {condition_folder_name(c): (i, c) for i, c in enumerate(conditions)}
        cond_starts_sec, gt_slots, _qr_sec = resolve_gt_qr_timeline(
            manifest, slate_sec, padding_sec, block_sec, len(conditions)
        )
    else:
        # dry-run で動画が無い場合: CSV の folder と推定 fps で不足検出のみ
        fps = fps_guess
        sync_frame = 0
        use_start = use_end = 0
        conditions = []
        folders = folders_from_csv or [
            condition_folder_name(c)
            for c in resolve_conditions({}, int(ns.conditions))
        ]
        cond_by_folder = {}
        cond_starts_sec = []
        gt_slots = []
        if not folders:
            return VideoStats(stem, "error", 0, 0, 0.0, note="no folders in CSV and no movie")

    pair_each_end = resolve_pair_each_end(hully_ns)
    fourier_band_radius = resolve_fourier_band_radius(hully_ns)
    search_mode = resolve_search_mode(True, False)
    kernel_candidates = resolve_kernel_candidates(search_mode)

    total_missing = 0
    folders_needed: Set[str] = set()
    per_pass_missing: Dict[str, Dict[str, List[SweepKey]]] = {}
    per_pass_expected: Dict[str, Tuple[List[SweepKey], Tuple[float, ...]]] = {}
    per_pass_existing: Dict[str, List[Dict[str, str]]] = {}

    for pass_label in pass_labels:
        if pass_label not in PASS_LOOKUP:
            log_warn(f"[WARN] unknown pass skipped: {pass_label}")
            continue
        diff_mode, stat_kind, repr_mode = PASS_LOOKUP[pass_label]
        expected, target_freqs = build_sweeps(
            diff_mode, stat_kind, hully_ns, meta, fps, repr_mode=repr_mode
        )
        _, sweep_csv, _, _ = pass_output_paths(out_root, pass_label)
        existing = read_csv_dicts(sweep_csv)
        missing = list_missing_sweeps(expected, existing, folders)
        per_pass_expected[pass_label] = (expected, target_freqs)
        per_pass_existing[pass_label] = existing
        per_pass_missing[pass_label] = missing
        n_miss = sum(len(v) for v in missing.values())
        total_missing += n_miss
        folders_needed.update(missing.keys())
        log_info(
            f"[INFO] {stem} pass={pass_label} expected_sweeps/cond={len(expected)} "
            f"missing={n_miss} folders={len(missing)}"
        )

    if dry_run or total_missing == 0:
        status = "dry_run" if dry_run else "ok"
        note = "no missing" if total_missing == 0 and not dry_run else ""
        if dry_run and not video_ok:
            note = (note + " " if note else "") + "movie missing (count-only)"
        return VideoStats(
            stem,
            status,
            total_missing,
            len(folders_needed),
            time.perf_counter() - t0,
            note=note.strip(),
        )

    if not video_ok or manifest_path is None:
        return VideoStats(
            stem,
            "error",
            total_missing,
            len(folders_needed),
            time.perf_counter() - t0,
            note="movie/manifest required to fill missing",
        )

    gt_path = out_root / "frame_QR.png"
    if not gt_path.exists():
        extract_gt_qr_mask(video_path, sync_frame, fps, gt_slots, gt_path)
    gt_path_str = str(gt_path.resolve())
    common_meta = format_common_meta(video_path.name, meta, fps)

    decode_executor = make_decode_executor(str(ns.decode_pool), int(ns.workers))
    try:
        new_rows_by_pass: Dict[str, List[Dict[str, str]]] = {p: [] for p in per_pass_missing}
        new_decode_by_pass: Dict[str, List[Dict[str, str]]] = {p: [] for p in per_pass_missing}

        for folder in sorted(folders_needed):
            if folder not in cond_by_folder:
                log_warn(f"[WARN] folder not in conditions, skip: {folder}")
                continue
            cond_i, cond = cond_by_folder[folder]
            block_start = sync_frame + int(round(cond_starts_sec[cond_i] * fps))
            start = block_start + use_start
            end = block_start + use_end
            cond_dir = out_root / folder
            ensure_dir(cond_dir)

            analyzed = save_block_frames(
                video_path, start, end, cond_dir, analysis_frames=ANALYSIS_FRAME_COUNT
            )
            extract_sec = (analyzed / fps) if fps > 0 else 0.0
            log_info(f"[INFO] {stem} extract {folder}: frames={analyzed}")
            rgb_list = load_rgb_frames(cond_dir, analyzed)
            d_stack = build_pair_diff_stack(rgb_list)
            delete_frame_pngs(cond_dir)

            for pass_label, missing_map in per_pass_missing.items():
                need = missing_map.get(folder)
                if not need:
                    continue
                diff_mode, stat_kind, repr_mode = PASS_LOOKUP[pass_label]
                sweep_rows, decode_rows = run_missing_for_folder(
                    d_stack=d_stack,
                    missing_sweeps=need,
                    pass_label=pass_label,
                    diff_mode=diff_mode,
                    folder_name=folder,
                    fps=fps,
                    pair_each_end=pair_each_end,
                    fourier_band_radius=fourier_band_radius,
                    kernel_candidates=kernel_candidates,
                    search_mode=search_mode,
                    gt_path_str=gt_path_str,
                    workers=int(ns.workers),
                    decode_executor=decode_executor,
                    decode_pool=str(ns.decode_pool),
                )
                for row in sweep_rows:
                    row.update(
                        {
                            "cond": str(cond_i),
                            "image": cond.get("image", ""),
                            "channel": cond.get("channel", ""),
                            "token": cond.get("token", ""),
                            "intensity": cond.get("intensity", ""),
                            "analysis_frames": str(analyzed),
                            "extract_sec": f"{extract_sec:.6f}",
                        }
                    )
                    row.update(common_meta)
                new_rows_by_pass[pass_label].extend(sweep_rows)
                new_decode_by_pass[pass_label].extend(decode_rows)
    finally:
        decode_executor.shutdown(wait=True)

    for pass_label in per_pass_missing:
        diff_mode, stat_kind, repr_mode = PASS_LOOKUP[pass_label]
        _expected, target_freqs = per_pass_expected[pass_label]
        results_csv, sweep_csv, decode_csv, _ = pass_output_paths(out_root, pass_label)
        existing = per_pass_existing[pass_label]
        merged = existing + new_rows_by_pass.get(pass_label, [])
        merged = recompute_adopted(merged, target_freqs)
        write_results_sweep_csv(sweep_csv, merged)

        old_results = read_csv_dicts(results_csv)
        by_folder_old = {r.get("folder", ""): r for r in old_results}
        adopted_rows = [r for r in merged if str(r.get("adopted")) == "1"]
        for ar in adopted_rows:
            folder = str(ar.get("folder") or "")
            by_folder_old[folder] = {
                k: str(v)
                for k, v in results_row_from_adopted_sweep(
                    ar, diff_mode=diff_mode, stat_kind=stat_kind
                ).items()
            }
        ordered_results: List[Dict[str, str]] = []
        for folder in folders:
            if folder in by_folder_old:
                ordered_results.append(by_folder_old[folder])
        for folder, row in by_folder_old.items():
            if folder not in folders:
                ordered_results.append(row)
        write_results_csv(results_csv, ordered_results)

        new_decode = new_decode_by_pass.get(pass_label, [])
        if new_decode:
            old_decode = read_csv_dicts(decode_csv)
            fields = decode_fieldnames(old_decode, new_decode)
            write_csv(decode_csv, old_decode + new_decode, fieldnames=fields)

    return VideoStats(
        stem,
        "ok",
        total_missing,
        len(folders_needed),
        time.perf_counter() - t0,
        note=f"appended passes={','.join(per_pass_missing.keys())}",
    )


def write_summary(path: Path, stats: Sequence[VideoStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "stem",
        "status",
        "missing_sweeps",
        "conditions_touched",
        "elapsed_sec",
        "note",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for s in stats:
            writer.writerow(
                {
                    "stem": s.stem,
                    "status": s.status,
                    "missing_sweeps": s.missing_sweeps,
                    "conditions_touched": s.conditions_touched,
                    "elapsed_sec": f"{s.elapsed_sec:.1f}",
                    "note": s.note,
                }
            )


def main() -> int:
    ns = parse_args()
    set_quiet(bool(ns.quiet))
    default_movie, default_manifest = default_paths()
    out_dir = Path(ns.out_dir).resolve()
    movie_dir = Path(ns.movie_dir).resolve() if ns.movie_dir else default_movie.resolve()
    manifest_dir = (
        Path(ns.manifest_dir).resolve() if ns.manifest_dir else default_manifest.resolve()
    )
    if not out_dir.is_dir():
        raise SystemExit(f"out-dir がありません: {out_dir}")

    pass_labels = [p.strip() for p in str(ns.passes).split(",") if p.strip()]
    if not pass_labels:
        raise SystemExit("--passes が空です")

    stems = discover_stems(out_dir, ns.stem)
    if not stems:
        raise SystemExit(f"処理対象 stem がありません: {out_dir}")

    log_info(
        f"[INFO] re_analyze_mid_fast out={out_dir} movie={movie_dir} "
        f"passes={pass_labels} stems={len(stems)} dry_run={ns.dry_run}"
    )

    stats: List[VideoStats] = []
    for stem in stems:
        log_info(f"[INFO] === {stem} ===")
        st = process_video(
            stem=stem,
            out_dir=out_dir,
            movie_dir=movie_dir,
            manifest_dir=manifest_dir,
            pass_labels=pass_labels,
            dry_run=bool(ns.dry_run),
            ns=ns,
        )
        stats.append(st)
        log_info(
            f"[INFO] {stem}: status={st.status} missing={st.missing_sweeps} "
            f"folders={st.conditions_touched} elapsed={st.elapsed_sec:.1f}s {st.note}"
        )

    summary_path = out_dir / "re_analyze_mid_fast_summary.csv"
    write_summary(summary_path, stats)
    total_missing = sum(s.missing_sweeps for s in stats)
    log_info(f"[OK] summary={summary_path} total_missing_sweeps={total_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
