"""日常用 mid_fast 一括解析: 軽いスイープを out_mid_fast_MMDD に残す。"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from naming import VideoNameMeta, parse_video_name
from git_auto_sync import auto_commit_and_push

ALLOWED_RATES = {45, 60, 90, 120, 180}
ALLOWED_EXPS = {250, 125, 60}
ALLOWED_FLUORO = {0, 1}

# run_pipeline_hully の mid_fast 時 pass 一覧（binary 6 + gray *_num 4）
HULLY_PASSES: Tuple[str, ...] = (
    "pair",
    "accum",
    "stat_std",
    "stat_var",
    "lockin",
    "fourier",
    "accum_num",
    "stat_std_num",
    "lockin_num",
    "fourier_num",
)

SUMMARY_FIELDS = ("video", "status", "elapsed_sec", "note", "phone_copied_total")
TIMING_FIELDS = ("finished_at", "video", "status", "elapsed_sec", "note", "phone_copied_total")
PASS_INDEX_FIELDS = (
    "pass",
    "status",
    "elapsed_sec",
    "log_path",
    "results_csv",
    "decode_csv",
    "phone_copied",
    "note",
)

PHONE_TOP_PER_FOLDER = 1
PHONE_RANDOM_PER_FOLDER = 0


@dataclass(frozen=True)
class VideoRunResult:
    video: Path
    status: str
    elapsed_sec: float
    note: str
    phone_copied_total: int = 0


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(
        description=(
            "R8/movie 内の動画を run_pipeline_hully で日常用 mid_fast 解析する（out_mid_fast_MMDD）。"
            "--mid-fast-sweeps（pair±10・間引き th・lockin phase=8・*_num）+ "
            "--mid-search（gray+median_otsu, kernel 3/5/7, scale=1）。"
            "生フレームは削除、差分 PNG は手法ごと代表1枚（per_pass）。"
            "phone_try は pass ごと top1 のみ。"
            "未指定時の出力先は実行日の out_mid_fast_MMDD（例: out_mid_fast_0805）。"
        )
    )
    p.add_argument("--movie-dir", type=str, default="")
    p.add_argument("--manifest-dir", type=str, default="")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="run_pipeline_hully の --out-dir（未指定時: R8/analyze_code/out_mid_fast_MMDD、MMDD=実行日）",
    )
    p.add_argument(
        "--full-search",
        action="store_true",
        help="既定の mid-search の代わりに full-search（全バリアント×拡大1/2/3）を使う",
    )
    p.add_argument(
        "--mid-search",
        action="store_true",
        help="互換用（既定が mid のため何もしない）",
    )
    p.add_argument(
        "--no-mid-search",
        action="store_true",
        help="互換用（既定が mid のため何もしない）",
    )
    p.add_argument(
        "--no-auto-git",
        action="store_true",
        help="終了時・動画ごとの自動 git commit/push を無効化",
    )
    p.add_argument(
        "pipeline_extra",
        nargs=argparse.REMAINDER,
        help="run_pipeline_hully.py に渡す追加引数（--keep-frames 60 等）",
    )
    ns = p.parse_args()
    extra = list(ns.pipeline_extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return ns, extra


def default_out_dir_name(when: Optional[datetime] = None) -> str:
    """実行日の out_mid_fast_MMDD（例: out_mid_fast_0805）。"""
    stamp = when or datetime.now()
    return f"out_mid_fast_{stamp.strftime('%m%d')}"


def default_paths() -> Tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    r8_dir = script_dir.parent
    return (
        r8_dir / "movie",
        r8_dir / "make_movie" / "manifests",
        script_dir / default_out_dir_name(),
    )


def is_valid_video_meta(meta: VideoNameMeta) -> bool:
    return (
        meta.rate_hz in ALLOWED_RATES
        and meta.exp in ALLOWED_EXPS
        and meta.fluoro in ALLOWED_FLUORO
    )


def discover_videos(movie_dir: Path) -> Tuple[List[Path], List[Path]]:
    if not movie_dir.is_dir():
        raise FileNotFoundError(f"movie-dir が存在しません: {movie_dir}")

    accepted: List[Path] = []
    rejected: List[Path] = []
    for path in sorted(movie_dir.glob("*.mp4")):
        meta = parse_video_name(path)
        if meta is None or not is_valid_video_meta(meta):
            rejected.append(path)
            continue
        accepted.append(path)

    accepted.sort(
        key=lambda p: (
            parse_video_name(p).rate_hz,  # type: ignore[union-attr]
            parse_video_name(p).exp,  # type: ignore[union-attr]
            parse_video_name(p).fluoro,  # type: ignore[union-attr]
            p.name,
        )
    )
    return accepted, rejected


def resolve_manifest(video_path: Path, manifest_dir: Path) -> Optional[Path]:
    meta = parse_video_name(video_path)
    if meta is None:
        return None
    manifest_path = manifest_dir / f"r{meta.rate_hz}_e{meta.exp}.json"
    return manifest_path if manifest_path.exists() else None


def resolve_out_root(video_path: Path, out_dir: str, default_out: Path) -> Path:
    base = Path(out_dir).resolve() if out_dir.strip() else default_out.resolve()
    meta = parse_video_name(video_path)
    stem = meta.stem if meta is not None else video_path.stem
    return base / stem


def has_cli_flag(args: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in args)


def build_mid_fast_args(
    video_path: Path,
    manifest_path: Path,
    out_dir: str,
    full_search: bool,
    shared_extra: Sequence[str],
) -> List[str]:
    """mid_fast 既定: 軽いスイープ + mid-search。差分 PNG は pass ごと代表1枚。"""
    extra = list(shared_extra)
    if not has_cli_flag(extra, "--workers"):
        extra = ["--workers", "16", *extra]
    if not has_cli_flag(extra, "--keep-frames"):
        extra = ["--keep-frames", "0", *extra]
    if not has_cli_flag(extra, "--save-diff-maps"):
        extra = ["--save-diff-maps", "per_pass", *extra]
    if (
        "--mid-fast-sweeps" not in extra
        and "--mid-sweeps" not in extra
        and "--hard-sweeps" not in extra
    ):
        extra = ["--mid-fast-sweeps", *extra]
    if not has_cli_flag(extra, "--mid-search") and not has_cli_flag(extra, "--full-search"):
        if full_search:
            extra.append("--full-search")
        else:
            extra.append("--mid-search")

    args = [
        "--video",
        str(video_path.resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        *extra,
    ]
    if out_dir.strip():
        args.extend(["--out-dir", out_dir.strip()])
    return args


def pass_safe_label(pass_label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pass_label)


def write_video_log(out_root: Path, captured: str) -> Path:
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pass_hully_mid_fast.log"
    log_path.write_text(captured or "", encoding="utf-8")
    return log_path


def _parse_accuracy(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def collect_phone_samples(
    out_root: Path,
    pass_label: str,
    *,
    top_n: int = PHONE_TOP_PER_FOLDER,
    random_n: int = PHONE_RANDOM_PER_FOLDER,
    rng: Optional[random.Random] = None,
) -> int:
    """条件ごとに pass 代表差分 PNG を phone_try/{pass}/ にコピー（既定 top1）。"""
    safe = pass_safe_label(pass_label)
    decode_csv = out_root / f"qr_decode_all_frames_{safe}.csv"
    if not decode_csv.exists():
        return 0

    with decode_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    by_folder: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        folder = str(row.get("folder") or "").strip()
        diff_rel = str(row.get("diff_image") or "").strip()
        if not folder or not diff_rel:
            continue
        src = out_root / diff_rel
        if not src.is_file():
            continue
        by_folder.setdefault(folder, []).append(row)

    if not by_folder:
        return 0

    dest_dir = out_root / "phone_try" / safe
    dest_dir.mkdir(parents=True, exist_ok=True)
    rng = rng or random.Random()
    copied = 0
    seen_dest: set[str] = set()

    for folder, folder_rows in by_folder.items():
        scored: List[Tuple[float, Path]] = []
        for row in folder_rows:
            diff_rel = str(row.get("diff_image") or "").strip()
            src = (out_root / diff_rel).resolve()
            if not src.is_file():
                continue
            acc = _parse_accuracy(row.get("accuracy"))
            scored.append((acc if acc is not None else float("-inf"), src))

        unique: Dict[str, Tuple[float, Path]] = {}
        for acc, src in scored:
            key = str(src)
            prev = unique.get(key)
            if prev is None or acc > prev[0]:
                unique[key] = (acc, src)
        items = list(unique.values())
        if not items:
            continue

        items.sort(key=lambda x: x[0], reverse=True)
        selected: List[Path] = []
        for _, src in items[: max(0, top_n)]:
            selected.append(src)

        remaining = [src for _, src in items[max(0, top_n) :]]
        if remaining and random_n > 0:
            k = min(random_n, len(remaining))
            selected.extend(rng.sample(remaining, k))

        for src in selected:
            dest_name = f"{folder}__{src.name}"
            if dest_name in seen_dest:
                continue
            seen_dest.add(dest_name)
            dest = dest_dir / dest_name
            shutil.copy2(src, dest)
            copied += 1

    return copied


def append_pass_index_rows(
    out_root: Path,
    *,
    status: str,
    elapsed_sec: float,
    log_path: Path,
    phone_by_pass: Dict[str, int],
    note: str = "",
) -> None:
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    index_path = logs_dir / "pass_index.csv"
    try:
        log_rel = str(log_path.relative_to(out_root))
    except ValueError:
        log_rel = str(log_path)

    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PASS_INDEX_FIELDS))
        writer.writeheader()
        for pass_label in HULLY_PASSES:
            safe = pass_safe_label(pass_label)
            writer.writerow(
                {
                    "pass": pass_label,
                    "status": status,
                    "elapsed_sec": f"{elapsed_sec:.1f}",
                    "log_path": log_rel,
                    "results_csv": f"results_{safe}.csv",
                    "decode_csv": f"qr_decode_all_frames_{safe}.csv",
                    "phone_copied": str(phone_by_pass.get(pass_label, 0)),
                    "note": note,
                }
            )


def _tail_lines(text: str, n: int = 20) -> List[str]:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-n:] if lines else []


def run_one_video(
    pipeline_script: Path,
    cwd: Path,
    video_path: Path,
    manifest_path: Path,
    out_dir: str,
    default_out: Path,
    full_search: bool,
    shared_extra: Sequence[str],
) -> Tuple[VideoRunResult, str]:
    args = build_mid_fast_args(video_path, manifest_path, out_dir, full_search, shared_extra)
    cmd = [sys.executable, str(pipeline_script), *args]
    out_root = resolve_out_root(video_path, out_dir, default_out)
    captured = ""
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - t0
        captured = "\n".join(
            part for part in (completed.stdout or "", completed.stderr or "") if part
        )
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = write_video_log(out_root, captured)

        phone_by_pass: Dict[str, int] = {}
        phone_total = 0
        if completed.returncode == 0:
            for pass_label in HULLY_PASSES:
                n = collect_phone_samples(out_root, pass_label)
                phone_by_pass[pass_label] = n
                phone_total += n

        status = "OK" if completed.returncode == 0 else "FAIL"
        note = "" if completed.returncode == 0 else f"run_pipeline_hully exit={completed.returncode}"
        append_pass_index_rows(
            out_root,
            status=status,
            elapsed_sec=elapsed,
            log_path=log_path,
            phone_by_pass=phone_by_pass,
            note=note,
        )
        return (
            VideoRunResult(video_path, status, elapsed, note, phone_total),
            captured,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = write_video_log(out_root, captured or str(exc))
        append_pass_index_rows(
            out_root,
            status="FAIL",
            elapsed_sec=elapsed,
            log_path=log_path,
            phone_by_pass={},
            note=str(exc),
        )
        return VideoRunResult(video_path, "FAIL", elapsed, str(exc), 0), captured


def append_timing_row(
    path: Path,
    *,
    video: str,
    status: str,
    elapsed_sec: float,
    note: str = "",
    phone_copied_total: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TIMING_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video": video,
                "status": status,
                "elapsed_sec": f"{elapsed_sec:.1f}",
                "note": note,
                "phone_copied_total": str(phone_copied_total),
            }
        )


def write_summary(path: Path, results: List[VideoRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "video": result.video.name,
                    "status": result.status,
                    "elapsed_sec": f"{result.elapsed_sec:.1f}",
                    "note": result.note,
                    "phone_copied_total": str(result.phone_copied_total),
                }
            )


def main() -> int:
    ns, pipeline_extra = parse_args()
    default_movie, default_manifest, default_out = default_paths()

    movie_dir = Path(ns.movie_dir).resolve() if ns.movie_dir else default_movie.resolve()
    manifest_dir = (
        Path(ns.manifest_dir).resolve() if ns.manifest_dir else default_manifest.resolve()
    )
    out_dir_arg = ns.out_dir.strip() if ns.out_dir.strip() else str(default_out.resolve())
    default_out = Path(out_dir_arg).resolve()
    summary_path = default_out / "all_analyze_mid_fast_summary.csv"
    timing_path = default_out / "all_analyze_mid_fast_timing.csv"
    pipeline_script = Path(__file__).resolve().parent / "run_pipeline_hully.py"
    cwd = Path(__file__).resolve().parent

    videos, rejected = discover_videos(movie_dir)
    print(
        f"[INFO] mid_fast: videos={len(videos)} rejected={len(rejected)} "
        f"pipeline=hully+mid-fast-sweeps+mid-search+per_pass "
        f"passes={list(HULLY_PASSES)} out={default_out}"
    )
    if rejected:
        print("[WARN] skipped (invalid name):")
        for path in rejected:
            print(f"  - {path.name}")

    if not videos:
        print("[ERROR] 解析対象の動画がありません")
        return 1

    results: List[VideoRunResult] = []
    for idx, video_path in enumerate(videos, start=1):
        manifest_path = resolve_manifest(video_path, manifest_dir)
        if manifest_path is None:
            meta = parse_video_name(video_path)
            expected = (
                f"r{meta.rate_hz}_e{meta.exp}.json"
                if meta is not None
                else "r{rate}_e{exp}.json"
            )
            note = f"manifest not found: {manifest_dir / expected}"
            fail = VideoRunResult(video_path, "FAIL", 0.0, note, 0)
            results.append(fail)
            append_timing_row(
                timing_path,
                video=video_path.name,
                status="FAIL",
                elapsed_sec=0.0,
                note=note,
            )
            print(f"[{idx}/{len(videos)}] {video_path.name}: FAIL ({note})")
            continue

        result, captured = run_one_video(
            pipeline_script,
            cwd,
            video_path,
            manifest_path,
            out_dir_arg,
            default_out,
            ns.full_search,
            pipeline_extra,
        )
        results.append(result)
        append_timing_row(
            timing_path,
            video=video_path.name,
            status=result.status,
            elapsed_sec=result.elapsed_sec,
            note=result.note,
            phone_copied_total=result.phone_copied_total,
        )
        if result.status == "OK":
            print(
                f"[{idx}/{len(videos)}] {video_path.name}: OK "
                f"({result.elapsed_sec:.0f}s, phone_png={result.phone_copied_total})"
            )
        else:
            note = result.note.strip() or "FAIL"
            print(
                f"[{idx}/{len(videos)}] {video_path.name}: FAIL "
                f"({result.elapsed_sec:.0f}s) {note}"
            )
            for line in _tail_lines(captured, 20):
                print(f"         {line}")

        if not ns.no_auto_git:
            auto_commit_and_push(source="all_analyze_mid_fast", detail=video_path.name)

    write_summary(summary_path, results)
    ok_count = sum(1 for r in results if r.status == "OK")
    fail_count = len(results) - ok_count
    print(
        f"[INFO] finished: ok={ok_count} fail={fail_count} "
        f"summary={summary_path.resolve()} timing={timing_path.resolve()}"
    )
    if not ns.no_auto_git:
        auto_commit_and_push(source="all_analyze_mid_fast", detail="finished")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
