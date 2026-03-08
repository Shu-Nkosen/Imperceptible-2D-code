from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

INPUT_CSV = Path("eval_summary_accuracy.csv")
OUTPUT_CSV = Path("max_accuracy_grid_avg4.csv")
DEFAULT_PREFIXES = ["ex", "nagaoka", "hocho", "rice"]
LEVELS = [2, 4, 6, 8, 10]
CHANNEL_ORDER = ["R", "G", "B", "I", "X"]
CHANNEL_LABELS = {
    "R": "R",
    "G": "G",
    "B": "B",
    "I": "min(I)",
    "X": "max(X)",
}


def parse_folder_name(folder: str) -> Tuple[str, str, int] | None:
    # 例: ex_B2, nagaoka_X10
    match = re.match(r"^([^_]+)_([BGRIX])(\d+)$", folder)
    if not match:
        return None
    prefix = match.group(1)
    channel = match.group(2)
    level = int(match.group(3))
    return prefix, channel, level


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "eval_summary_accuracy.csv から max_accuracy を集計し、"
            "行=2,4,6,8,10 / 列=R,G,B,min(I),max(X) のCSVを作る"
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="入力CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="出力CSV")
    parser.add_argument(
        "--prefixes",
        type=str,
        default=",".join(DEFAULT_PREFIXES),
        help="平均対象の系列名(カンマ区切り)。既定: ex,nagaoka,hocho,rice",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="4系列そろっていなくても、存在する系列だけで平均する",
    )
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    if not prefixes:
        raise ValueError("--prefixes が空です")

    base_dir = Path(__file__).resolve().parent
    input_path = args.input if args.input.is_absolute() else base_dir / args.input
    output_path = args.output if args.output.is_absolute() else base_dir / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_path}")

    # key: (level, channel) -> {prefix: max_accuracy}
    buckets: Dict[Tuple[int, str], Dict[str, float]] = {}

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = (row.get("folder") or "").strip()
            parsed = parse_folder_name(folder)
            if parsed is None:
                continue

            prefix, channel, level = parsed
            if prefix not in prefixes:
                continue
            if level not in LEVELS or channel not in CHANNEL_ORDER:
                continue

            max_acc_raw = (row.get("max_accuracy") or "").strip()
            if not max_acc_raw:
                continue

            try:
                max_acc = float(max_acc_raw)
            except ValueError:
                continue

            key = (level, channel)
            if key not in buckets:
                buckets[key] = {}
            buckets[key][prefix] = max_acc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["level"] + [CHANNEL_LABELS[ch] for ch in CHANNEL_ORDER]
    rows: List[Dict[str, str]] = []
    missing_count = 0

    for level in LEVELS:
        row: Dict[str, str] = {"level": str(level)}
        for channel in CHANNEL_ORDER:
            label = CHANNEL_LABELS[channel]
            values_by_prefix = buckets.get((level, channel), {})

            values = [values_by_prefix[p] for p in prefixes if p in values_by_prefix]
            if not values:
                row[label] = ""
                missing_count += 1
                continue

            if (not args.allow_partial) and len(values) != len(prefixes):
                row[label] = ""
                missing_count += 1
                continue

            row[label] = f"{mean(values):.6f}"
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("==== 完了 ====")
    print(f"入力CSV: {input_path}")
    print(f"出力CSV: {output_path}")
    print(f"平均対象系列: {prefixes}")
    print("行: 2,4,6,8,10 / 列: R,G,B,min(I),max(X)")
    if missing_count > 0:
        mode = "部分平均を許可" if args.allow_partial else "4系列そろったセルのみ出力"
        print(f"欠損セル数: {missing_count} ({mode})")


if __name__ == "__main__":
    main()
