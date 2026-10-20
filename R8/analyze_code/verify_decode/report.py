"""両実験の要約を stdout / 簡易 CSV にまとめる。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="out_verify_decode の要約表示")
    p.add_argument("--root", type=str, default="out_verify_decode")
    ns = p.parse_args()
    root = Path(ns.root)

    print("=== matched_channel / by_channel ===")
    for row in _read(root / "matched_channel" / "by_channel.csv"):
        print(
            f"  {row.get('diff_mode'):8s} {row.get('method'):8s} "
            f"token={row.get('token')}  "
            f"{row.get('any_success_pct')}% ({row.get('success')}/{row.get('n')})"
        )

    print("=== fourier_score / by_score_mode ===")
    for row in _read(root / "fourier_score" / "by_score_mode.csv"):
        print(
            f"  mode={row.get('fourier_score_mode'):6s}  "
            f"{row.get('any_success_pct')}% ({row.get('success')}/{row.get('n')})"
        )

    print("=== fourier_score / by_image_channel (hocho×B 注目) ===")
    for row in _read(root / "fourier_score" / "by_image_channel.csv"):
        if row.get("image") == "hocho" and row.get("token") == "B":
            print(
                f"  {row.get('fourier_score_mode'):6s} hocho B  "
                f"{row.get('any_success_pct')}% ({row.get('success')}/{row.get('n')})"
            )
    print(f"[OK] root={root.resolve()}")


if __name__ == "__main__":
    main()
