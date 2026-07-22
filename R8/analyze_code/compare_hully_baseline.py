"""現行 all_analyze 結果と hully 結果を比較する簡易検証スクリプト。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


PASSES = ("pair", "accum", "stat_std", "stat_var", "fourier")


def load_results(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize(results: List[Dict[str, str]]) -> Tuple[int, int, float, float]:
    rows = len(results)
    decodes = sum(1 for r in results if (r.get("decode_decoded_text") or "").strip())
    accs = [float(r["pixel_acc_best"]) for r in results if r.get("pixel_acc_best")]
    avg = sum(accs) / len(accs) if accs else 0.0
    mx = max(accs) if accs else 0.0
    return rows, decodes, avg, mx


def main() -> int:
    p = argparse.ArgumentParser(description="baseline vs hully results 比較")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="動画出力ルート（例: out/r45_e250_f1）",
    )
    p.add_argument("--video-stem", type=str, default="r45_e250_f1")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_root = Path(args.out_dir).resolve() if args.out_dir else script_dir / "out" / args.video_stem
    timing_path = script_dir / "out" / "all_analyze_hully_timing.csv"

    print(f"compare: {out_root}")
    for pass_label in PASSES:
        baseline = out_root / f"results_{pass_label}.csv"
        hully = out_root / f"results_{pass_label}.csv"
        rows = load_results(baseline)
        if not rows:
            print(f"  {pass_label}: baseline missing")
            continue
        n, dec, avg, mx = summarize(rows)
        print(f"  {pass_label}: rows={n} decode={dec} acc_best avg={avg:.4f} max={mx:.4f}")

    if timing_path.exists():
        with timing_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("video", "").startswith(args.video_stem):
                    print(
                        f"timing: {row.get('video')} status={row.get('status')} "
                        f"elapsed={row.get('elapsed_sec')}s"
                    )
    else:
        print(f"timing: {timing_path} not found (hully 未実行)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
