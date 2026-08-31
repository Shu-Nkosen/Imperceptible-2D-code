from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from naming import VideoNameMeta, parse_video_name
from git_auto_sync import auto_commit_and_push

ALLOWED_RATES = {45, 60, 90, 120, 180}
ALLOWED_EXPS = {250, 125, 60}
ALLOWED_FLUORO = {0, 1}

# 既定では全手法をこの順で実行（動画ごと → 切り出し再利用が効く）
ALL_ANALYSIS_PASSES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("pair", ("--diff-mode", "pair")),
    ("accum", ("--diff-mode", "accum")),
    ("stat_std", ("--diff-mode", "stat", "--stat-kind", "std")),
    ("stat_var", ("--diff-mode", "stat", "--stat-kind", "var")),
    ("fourier", ("--diff-mode", "fourier")),
)

SUMMARY_FIELDS = ("video", "pass", "manifest", "status", "exit_code", "note")
TIMING_FIELDS = ("finished_at", "video", "pass", "status", "elapsed_sec", "note")


@dataclass(frozen=True)
class RunResult:
    video: Path
    pass_label: str
    manifest: str
    status: str
    exit_code: int
    note: str


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(
        description=(
            "R8/movie 内の命名規則に合う動画を順番に解析する。"
            "既定では pair / accum / stat(std) / stat(var) / fourier を全部実行する。"
        )
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
        "--mid-search",
        action="store_true",
        help="run_pipeline に --mid-search を付ける（既定は fast: gray×kernel5×最精度デコード1回）",
    )
    p.add_argument(
        "--no-mid-search",
        action="store_true",
        help="互換用（既定が fast のため何もしない）",
    )
    p.add_argument(
        "--no-auto-git",
        action="store_true",
        help="終了時・動画ごとの自動 git commit/push を無効化",
    )
    p.add_argument(
        "pipeline_extra",
        nargs=argparse.REMAINDER,
        help=(
            "run_pipeline.py にそのまま渡す追加引数。"
            "例: --max-frames 2 / --diff-mode accum（指定時はその手法のみ）"
        ),
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


def resolve_out_root(video_path: Path, out_dir: str, default_out: Path) -> Path:
    """run_pipeline.py と同じ出力ルート: <out-dir>/<video_stem>/"""
    base = Path(out_dir).resolve() if out_dir.strip() else default_out.resolve()
    meta = parse_video_name(video_path)
    stem = meta.stem if meta is not None else video_path.stem
    return base / stem


def has_cli_flag(args: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in args)


def strip_mode_args(args: Sequence[str]) -> List[str]:
    """--diff-mode / --stat-kind とその値を除去する。"""
    out: List[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--diff-mode", "--stat-kind"):
            skip_next = True
            continue
        if arg.startswith("--diff-mode=") or arg.startswith("--stat-kind="):
            continue
        out.append(arg)
    return out


def resolve_analysis_passes(
    pipeline_extra: Sequence[str],
) -> Tuple[List[Tuple[str, Tuple[str, ...]]], List[str]]:
    """戻り値: (passes, shared_extra)。

    --diff-mode が明示されていればその手法だけ。未指定なら全手法。
    """
    shared = list(pipeline_extra)
    if has_cli_flag(shared, "--diff-mode"):
        # ユーザー指定の1手法のみ（shared に diff-mode を残す）
        label = "custom"
        for i, arg in enumerate(shared):
            if arg == "--diff-mode" and i + 1 < len(shared):
                mode = shared[i + 1]
                kind = ""
                for j, a in enumerate(shared):
                    if a == "--stat-kind" and j + 1 < len(shared):
                        kind = shared[j + 1]
                        break
                    if a.startswith("--stat-kind="):
                        kind = a.split("=", 1)[1]
                        break
                label = f"{mode}_{kind}" if mode == "stat" and kind else mode
                break
            if arg.startswith("--diff-mode="):
                mode = arg.split("=", 1)[1]
                label = mode
                break
        return [(label, ())], shared

    shared = strip_mode_args(shared)
    return list(ALL_ANALYSIS_PASSES), shared


def build_pipeline_args(
    video_path: Path,
    manifest_path: Path,
    out_dir: str,
    mid_search: bool,
    shared_extra: Sequence[str],
    pass_args: Sequence[str],
) -> List[str]:
    extra = list(shared_extra)
    if not has_cli_flag(extra, "--max-frames"):
        extra = ["--max-frames", "120", *extra]
    if not has_cli_flag(extra, "--workers"):
        extra = ["--workers", "16", *extra]
    if not has_cli_flag(extra, "--quiet"):
        extra = ["--quiet", *extra]
    if (
        mid_search
        and "--mid-search" not in extra
        and "--full-search" not in extra
    ):
        extra.append("--mid-search")

    args = [
        "--video",
        str(video_path.resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        *pass_args,
        *extra,
    ]
    if out_dir.strip():
        args.extend(["--out-dir", out_dir.strip()])
    return args


def archive_pass_outputs(out_root: Path, pass_label: str) -> None:
    """results.csv / qr_decode_all_frames.csv / pair_accuracy.csv を手法別ファイル名で残す。"""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pass_label)
    for name in ("results.csv", "qr_decode_all_frames.csv", "pair_accuracy.csv"):
        src = out_root / name
        if not src.exists():
            continue
        dst = out_root / f"{src.stem}_{safe}{src.suffix}"
        shutil.copy2(src, dst)


def _tail_lines(text: str, n: int = 20) -> List[str]:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-n:] if lines else []


def run_one_pass(
    pipeline_script: Path,
    cwd: Path,
    video_path: Path,
    manifest_path: Path,
    out_dir: str,
    default_out: Path,
    mid_search: bool,
    shared_extra: Sequence[str],
    pass_label: str,
    pass_args: Sequence[str],
) -> Tuple[RunResult, str, float]:
    """戻り値: (RunResult, captured_output, elapsed_sec)。"""
    args = build_pipeline_args(
        video_path, manifest_path, out_dir, mid_search, shared_extra, pass_args
    )
    cmd = [sys.executable, str(pipeline_script), *args]
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
        exit_code = int(completed.returncode)
        captured = "\n".join(
            part for part in (completed.stdout or "", completed.stderr or "") if part
        )
        out_root = resolve_out_root(video_path, out_dir, default_out)
        archive_pass_outputs(out_root, pass_label)
        elapsed = time.perf_counter() - t0
        if exit_code == 0:
            return (
                RunResult(
                    video=video_path,
                    pass_label=pass_label,
                    manifest=str(manifest_path),
                    status="OK",
                    exit_code=exit_code,
                    note="",
                ),
                captured,
                elapsed,
            )
        return (
            RunResult(
                video=video_path,
                pass_label=pass_label,
                manifest=str(manifest_path),
                status="FAIL",
                exit_code=exit_code,
                note=f"run_pipeline exit={exit_code}",
            ),
            captured,
            elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return (
            RunResult(
                video=video_path,
                pass_label=pass_label,
                manifest=str(manifest_path),
                status="FAIL",
                exit_code=-1,
                note=str(exc),
            ),
            captured,
            elapsed,
        )


def print_job_lines(
    job_i: int,
    total_jobs: int,
    video_name: str,
    pass_label: str,
    result: RunResult,
    captured: str = "",
    elapsed_sec: float = 0.0,
) -> None:
    prefix = f"[{job_i}/{total_jobs}]"
    elapsed_txt = f"{elapsed_sec:.0f}s"
    print(f"{prefix} target: {video_name}")
    print(f"{prefix} pass:   {pass_label}")
    if result.status == "OK":
        print(f"{prefix} result: OK ({elapsed_txt})")
        return
    note = result.note.strip() or "FAIL"
    print(f"{prefix} result: FAIL ({note}, {elapsed_txt})")
    for line in _tail_lines(captured, 20):
        print(f"         {line}")


def append_timing_row(
    path: Path,
    *,
    video: str,
    pass_label: str,
    status: str,
    elapsed_sec: float,
    note: str = "",
) -> None:
    """out 直下の timing CSV に1行追記（既存データは消さない）。"""
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
                "pass": pass_label,
                "status": status,
                "elapsed_sec": f"{elapsed_sec:.1f}",
                "note": note,
            }
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
                    "pass": result.pass_label,
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
    timing_path = default_out / "all_analyze_timing.csv"
    pipeline_script = Path(__file__).resolve().parent / "run_pipeline.py"
    cwd = Path(__file__).resolve().parent

    passes, shared_extra = resolve_analysis_passes(pipeline_extra)
    videos, rejected = discover_videos(movie_dir)
    print(
        f"[INFO] videos={len(videos)} rejected={len(rejected)} "
        f"passes={[label for label, _ in passes]}"
    )
    if rejected:
        print("[WARN] skipped (invalid name):")
        for path in rejected:
            print(f"  - {path.name}")

    if not videos:
        print("[ERROR] 解析対象の動画がありません")
        return 1

    results: List[RunResult] = []
    total_jobs = len(videos) * len(passes)
    job_i = 0
    for video_path in videos:
        manifest_path = resolve_manifest(video_path, manifest_dir)
        if manifest_path is None:
            meta = parse_video_name(video_path)
            expected = (
                f"r{meta.rate_hz}_e{meta.exp}.json"
                if meta is not None
                else "r{rate}_e{exp}.json"
            )
            note = f"manifest not found: {manifest_dir / expected}"
            for pass_label, _ in passes:
                job_i += 1
                fail = RunResult(
                    video=video_path,
                    pass_label=pass_label,
                    manifest="",
                    status="FAIL",
                    exit_code=-1,
                    note=note,
                )
                results.append(fail)
                append_timing_row(
                    timing_path,
                    video=video_path.name,
                    pass_label=pass_label,
                    status="FAIL",
                    elapsed_sec=0.0,
                    note=note,
                )
                print_job_lines(job_i, total_jobs, video_path.name, pass_label, fail)
            continue

        for pass_label, pass_args in passes:
            job_i += 1
            result, captured, elapsed = run_one_pass(
                pipeline_script,
                cwd,
                video_path,
                manifest_path,
                ns.out_dir,
                default_out,
                ns.mid_search,
                shared_extra,
                pass_label,
                pass_args,
            )
            results.append(result)
            append_timing_row(
                timing_path,
                video=video_path.name,
                pass_label=pass_label,
                status=result.status,
                elapsed_sec=elapsed,
                note=result.note,
            )
            print_job_lines(
                job_i,
                total_jobs,
                video_path.name,
                pass_label,
                result,
                captured,
                elapsed_sec=elapsed,
            )

        if not ns.no_auto_git:
            auto_commit_and_push(source="all_analyze", detail=video_path.name)

    write_summary(summary_path, results)
    ok_count = sum(1 for r in results if r.status == "OK")
    fail_count = len(results) - ok_count
    print(
        f"[INFO] finished: ok={ok_count} fail={fail_count} "
        f"summary={summary_path.resolve()} timing={timing_path.resolve()}"
    )
    if not ns.no_auto_git:
        auto_commit_and_push(source="all_analyze", detail="finished")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
