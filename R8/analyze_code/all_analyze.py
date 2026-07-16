from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from naming import VideoNameMeta, parse_video_name

ALLOWED_RATES = {45, 60, 90, 120, 180}
ALLOWED_EXPS = {250, 125, 60}
ALLOWED_FLUORO = {0, 1}

SUMMARY_FIELDS = ("video", "manifest", "status", "exit_code", "note")


@dataclass(frozen=True)
class RunResult:
    video: Path
    manifest: str
    status: str
    exit_code: int
    note: str


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(
        description="R8/movie 内の命名規則に合う動画を順番に run_pipeline.py で解析する"
    )
    p.add_argument(
        "--movie-dir",
        type=str,
        default="",
        help="動画ディレクトリ（未指定時: R8/movie）",
    )
    p.add_argument(
        "--manifest-dir",
        type=str,
        default="",
        help="manifest ディレクトリ（未指定時: R8/make_movie/manifests）",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="run_pipeline の --out-dir（未指定時は run_pipeline 既定）",
    )
    p.add_argument(
        "--no-mid-search",
        action="store_true",
        help="既定の --mid-search を付けない",
    )
    p.add_argument(
        "pipeline_extra",
        nargs=argparse.REMAINDER,
        help="run_pipeline.py にそのまま渡す追加引数（例: --diff-mode accum）",
    )
    ns = p.parse_args()
    extra = list(ns.pipeline_extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return ns, extra


def default_paths() -> Tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    r8_dir = script_dir.parent
    movie_dir = r8_dir / "movie"
    manifest_dir = r8_dir / "make_movie" / "manifests"
    out_dir = script_dir / "out"
    return movie_dir, manifest_dir, out_dir


def is_valid_video_meta(meta: VideoNameMeta) -> bool:
    return (
        meta.rate_hz in ALLOWED_RATES
        and meta.exp in ALLOWED_EXPS
        and meta.fluoro in ALLOWED_FLUORO
    )


def discover_videos(movie_dir: Path) -> Tuple[List[Path], List[Path]]:
    """(対象動画, 命名不正で除外した動画)"""
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


def build_pipeline_args(
    video_path: Path,
    manifest_path: Path,
    out_dir: str,
    no_mid_search: bool,
    pipeline_extra: Sequence[str],
) -> List[str]:
    extra = list(pipeline_extra)
    if not any(arg == "--max-frames" or arg.startswith("--max-frames=") for arg in extra):
        extra = ["--max-frames", "120", *extra]
    if (
        not no_mid_search
        and "--mid-search" not in extra
        and "--full-search" not in extra
    ):
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
    no_mid_search: bool,
    pipeline_extra: Sequence[str],
) -> RunResult:
    args = build_pipeline_args(
        video_path, manifest_path, out_dir, no_mid_search, pipeline_extra
    )
    cmd = [sys.executable, str(pipeline_script), *args]
    print(f"[INFO] run: {' '.join(cmd)}")
    try:
        completed = subprocess.run(cmd, cwd=str(cwd), check=False)
        exit_code = int(completed.returncode)
        if exit_code == 0:
            return RunResult(
                video=video_path,
                manifest=str(manifest_path),
                status="OK",
                exit_code=exit_code,
                note="",
            )
        return RunResult(
            video=video_path,
            manifest=str(manifest_path),
            status="FAIL",
            exit_code=exit_code,
            note=f"run_pipeline exit={exit_code}",
        )
    except Exception as exc:
        return RunResult(
            video=video_path,
            manifest=str(manifest_path),
            status="FAIL",
            exit_code=-1,
            note=str(exc),
        )


def write_summary(path: Path, results: List[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "video": result.video.name,
                    "manifest": result.manifest,
                    "status": result.status,
                    "exit_code": str(result.exit_code),
                    "note": result.note,
                }
            )


def main() -> int:
    ns, pipeline_extra = parse_args()
    default_movie, default_manifest, default_out = default_paths()

    movie_dir = Path(ns.movie_dir).resolve() if ns.movie_dir else default_movie.resolve()
    manifest_dir = (
        Path(ns.manifest_dir).resolve() if ns.manifest_dir else default_manifest.resolve()
    )
    summary_path = default_out / "all_analyze_summary.csv"
    pipeline_script = Path(__file__).resolve().parent / "run_pipeline.py"
    cwd = Path(__file__).resolve().parent

    videos, rejected = discover_videos(movie_dir)
    print(f"[INFO] movie_dir={movie_dir}")
    print(f"[INFO] manifest_dir={manifest_dir}")
    print(f"[INFO] target_videos={len(videos)} rejected_by_name={len(rejected)}")
    if rejected:
        print("[WARN] skipped (invalid name):")
        for path in rejected:
            print(f"  - {path.name}")

    if not videos:
        print("[ERROR] 解析対象の動画がありません")
        return 1

    results: List[RunResult] = []
    for i, video_path in enumerate(videos, 1):
        print(f"[INFO] ({i}/{len(videos)}) {video_path.name}")
        manifest_path = resolve_manifest(video_path, manifest_dir)
        if manifest_path is None:
            meta = parse_video_name(video_path)
            expected = (
                f"r{meta.rate_hz}_e{meta.exp}.json"
                if meta is not None
                else "r{rate}_e{exp}.json"
            )
            note = f"manifest not found: {manifest_dir / expected}"
            print(f"[WARN] {note}")
            results.append(
                RunResult(
                    video=video_path,
                    manifest="",
                    status="FAIL",
                    exit_code=-1,
                    note=note,
                )
            )
            continue

        result = run_one_video(
            pipeline_script,
            cwd,
            video_path,
            manifest_path,
            ns.out_dir,
            ns.no_mid_search,
            pipeline_extra,
        )
        results.append(result)
        if result.status == "OK":
            print(f"[OK] {video_path.name}")
        else:
            print(f"[WARN] {video_path.name}: {result.note}")

    write_summary(summary_path, results)
    ok_count = sum(1 for r in results if r.status == "OK")
    fail_count = len(results) - ok_count
    print(
        f"[INFO] finished: ok={ok_count} fail={fail_count} "
        f"summary={summary_path.resolve()}"
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
