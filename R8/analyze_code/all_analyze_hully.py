"""全動画に hully 統合パイプラインを1回ずつ実行する orchestrator。"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from naming import VideoNameMeta, parse_video_name

ALLOWED_RATES = {45, 60, 90, 120, 180}
ALLOWED_EXPS = {250, 125, 60}
ALLOWED_FLUORO = {0, 1}

SUMMARY_FIELDS = ("video", "status", "elapsed_sec", "note")
TIMING_FIELDS = ("finished_at", "video", "status", "elapsed_sec", "note", "pass_breakdown")


@dataclass(frozen=True)
class VideoRunResult:
    video: Path
    status: str
    elapsed_sec: float
    note: str


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(
        description="R8/movie 内の動画を run_pipeline_hully.py で1本ずつ解析する"
    )
    p.add_argument("--movie-dir", type=str, default="")
    p.add_argument("--manifest-dir", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--mid-search", action="store_true")
    p.add_argument(
        "pipeline_extra",
        nargs=argparse.REMAINDER,
        help="run_pipeline_hully に渡す追加引数（--save-diff-maps none 等）",
    )
    ns = p.parse_args()
    extra = list(ns.pipeline_extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return ns, extra


def default_paths() -> Tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    r8_dir = script_dir.parent
    return r8_dir / "movie", r8_dir / "make_movie" / "manifests", script_dir / "out"


def is_valid_video_meta(meta: VideoNameMeta) -> bool:
    return meta.rate_hz in ALLOWED_RATES and meta.exp in ALLOWED_EXPS and meta.fluoro in ALLOWED_FLUORO


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


def has_cli_flag(args: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in args)


def build_hully_args(
    video_path: Path,
    manifest_path: Path,
    out_dir: str,
    mid_search: bool,
    shared_extra: Sequence[str],
) -> List[str]:
    extra = list(shared_extra)
    if not has_cli_flag(extra, "--workers"):
        extra = ["--workers", "16", *extra]
    if not has_cli_flag(extra, "--keep-frames"):
        extra = ["--keep-frames", "0", *extra]
    if not has_cli_flag(extra, "--save-diff-maps"):
        extra = ["--save-diff-maps", "top3", *extra]
    if mid_search and "--mid-search" not in extra and "--full-search" not in extra:
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


def run_one_video(
    pipeline_script: Path,
    cwd: Path,
    video_path: Path,
    manifest_path: Path,
    out_dir: str,
    mid_search: bool,
    shared_extra: Sequence[str],
) -> VideoRunResult:
    args = build_hully_args(video_path, manifest_path, out_dir, mid_search, shared_extra)
    cmd = [sys.executable, str(pipeline_script), *args]
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
        if completed.returncode == 0:
            return VideoRunResult(video_path, "OK", elapsed, "")
        tail = "\n".join(
            ln for ln in (completed.stdout or completed.stderr or "").splitlines() if ln.strip()
        )[-500:]
        note = f"exit={completed.returncode}"
        if tail:
            note = f"{note}: {tail[-200:]}"
        return VideoRunResult(video_path, "FAIL", elapsed, note)
    except Exception as exc:
        return VideoRunResult(video_path, "FAIL", time.perf_counter() - t0, str(exc))


def append_timing_row(path: Path, *, video: str, status: str, elapsed_sec: float, note: str = "") -> None:
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
                "pass_breakdown": "hully_single_pass",
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
                }
            )


def main() -> int:
    ns, pipeline_extra = parse_args()
    default_movie, default_manifest, default_out = default_paths()
    movie_dir = Path(ns.movie_dir).resolve() if ns.movie_dir else default_movie.resolve()
    manifest_dir = Path(ns.manifest_dir).resolve() if ns.manifest_dir else default_manifest.resolve()
    summary_path = default_out / "all_analyze_hully_summary.csv"
    timing_path = default_out / "all_analyze_hully_timing.csv"
    pipeline_script = Path(__file__).resolve().parent / "run_pipeline_hully.py"
    cwd = Path(__file__).resolve().parent

    videos, rejected = discover_videos(movie_dir)
    print(f"[INFO] videos={len(videos)} rejected={len(rejected)} pipeline=hully")
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
            expected = f"r{meta.rate_hz}_e{meta.exp}.json" if meta else "r{rate}_e{exp}.json"
            note = f"manifest not found: {manifest_dir / expected}"
            fail = VideoRunResult(video_path, "FAIL", 0.0, note)
            results.append(fail)
            append_timing_row(timing_path, video=video_path.name, status="FAIL", elapsed_sec=0.0, note=note)
            print(f"[{idx}/{len(videos)}] {video_path.name}: FAIL ({note})")
            continue

        result = run_one_video(
            pipeline_script,
            cwd,
            video_path,
            manifest_path,
            ns.out_dir,
            ns.mid_search,
            pipeline_extra,
        )
        results.append(result)
        append_timing_row(
            timing_path,
            video=video_path.name,
            status=result.status,
            elapsed_sec=result.elapsed_sec,
            note=result.note,
        )
        if result.status == "OK":
            print(
                f"[{idx}/{len(videos)}] {video_path.name}: OK "
                f"({result.elapsed_sec:.0f}s)"
            )
        else:
            note = result.note.strip() or "unknown error"
            print(
                f"[{idx}/{len(videos)}] {video_path.name}: FAIL "
                f"({result.elapsed_sec:.0f}s) {note}"
            )

    write_summary(summary_path, results)
    ok_count = sum(1 for r in results if r.status == "OK")
    fail_count = len(results) - ok_count
    print(
        f"[INFO] finished: ok={ok_count} fail={fail_count} "
        f"summary={summary_path.resolve()} timing={timing_path.resolve()}"
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
