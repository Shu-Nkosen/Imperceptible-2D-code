"""既存 out_mid_fast_* に accum/accum_num の全長窓（window_n=120）だけ追記する。

mid_fast の短窓 (n=5) は触らず、Fourier と同じ約2秒分の |d| 合算スイープのみ追加する。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from run_pipeline import ANALYSIS_FRAME_COUNT, set_quiet
from run_pipeline_hully import SweepKey
from re_analyze_mid_fast import (
    VideoStats,
    default_paths,
    discover_stems,
    log_info,
    process_video,
    write_summary,
)

DEFAULT_PASSES = ("accum", "accum_num")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "既存 out_mid_fast_* に accum/accum_num の全長窓（window_n="
            f"{ANALYSIS_FRAME_COUNT}）だけ追記する。"
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
        help="対象 pass（カンマ区切り。既定: accum,accum_num）",
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


def accum_full_window_filter(sweep: SweepKey) -> bool:
    return sweep.diff_mode == "accum" and sweep.window_n == ANALYSIS_FRAME_COUNT


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
    for label in pass_labels:
        if label not in DEFAULT_PASSES:
            log_info(f"[WARN] pass {label!r} は accum 系以外（accum,accum_num 推奨）")

    stems = discover_stems(out_dir, ns.stem)
    if not stems:
        raise SystemExit(f"処理対象 stem がありません: {out_dir}")

    log_info(
        f"[INFO] re_analyze_mid_fast_accum_full out={out_dir} movie={movie_dir} "
        f"passes={pass_labels} window_n={ANALYSIS_FRAME_COUNT} stems={len(stems)} "
        f"dry_run={ns.dry_run}"
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
            sweep_filter=accum_full_window_filter,
        )
        stats.append(st)
        log_info(
            f"[INFO] {stem}: status={st.status} missing={st.missing_sweeps} "
            f"folders={st.conditions_touched} elapsed={st.elapsed_sec:.1f}s {st.note}"
        )

    summary_path = out_dir / "re_analyze_mid_fast_accum_full_summary.csv"
    write_summary(summary_path, stats)
    total_missing = sum(s.missing_sweeps for s in stats)
    log_info(f"[OK] summary={summary_path} total_missing_sweeps={total_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
