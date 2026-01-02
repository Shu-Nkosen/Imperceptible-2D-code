import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from skimage.metrics import structural_similarity as compare_ssim

DERIVED_PATTERN = re.compile(r"^(?P<base>.+?)_(?P<index>\d+)_(?P<mode>normal|inv)(?P<channel>[BGIRX])\.png$")
PER_FILE_FIELDS = [
    "base_image",
    "variant_file",
    "index",
    "mode",
    "channel",
    "psnr",
    "ssim",
]
CODE_FIELDS = [
    "index",
    "channel",
    "mode",
    "sample_count",
    "avg_psnr",
    "avg_ssim",
]
FULL_HD_SIZE = (1920, 1080)


def compute_metrics(ref_path: Path, target_path: Path) -> Tuple[float, float]:
    ref = cv2.imread(str(ref_path))
    tgt = cv2.imread(str(target_path))
    if ref is None:
        raise FileNotFoundError(f"参照画像が読み込めません: {ref_path}")
    if tgt is None:
        raise FileNotFoundError(f"比較画像が読み込めません: {target_path}")
    ref = cv2.resize(ref, FULL_HD_SIZE, interpolation=cv2.INTER_AREA)
    tgt = cv2.resize(tgt, FULL_HD_SIZE, interpolation=cv2.INTER_AREA)
    # if ref.shape != tgt.shape:
    #     raise ValueError(f"画像サイズが一致しません: {ref.shape} vs {tgt.shape}")

    psnr = cv2.PSNR(ref, tgt)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    ssim, _ = compare_ssim(gray_ref, gray_tgt, full=True)
    return psnr, float(ssim)


def parse_variant_info(file_path: Path) -> Optional[Dict[str, object]]:
    match = DERIVED_PATTERN.match(file_path.name)
    if not match:
        return None
    info = match.groupdict()
    base_name = info["base"]
    return {
        "path": file_path,
        "base_name": base_name,
        "base_path": file_path.with_name(f"{base_name}.png"),
        "index": int(info["index"]),
        "mode": info["mode"],
        "channel": info["channel"],
    }


def collect_variants(base_dir: Path, patterns: Sequence[str]) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    for pattern in patterns:
        for file_path in sorted(base_dir.glob(pattern)):
            info = parse_variant_info(file_path)
            if info:
                variants.append(info)
    return variants


def evaluate_variants(variants: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for variant in variants:
        base_path = variant["base_path"]
        if not base_path.exists():
            print(f"[SKIP] {variant['path'].name}: {base_path.name} が見つかりません")
            continue
        try:
            psnr, ssim = compute_metrics(base_path, variant["path"])
        except Exception as exc:
            print(f"[SKIP] {variant['path'].name}: {exc}")
            continue
        records.append({
            "base_image": base_path.name,
            "variant_file": variant["path"].name,
            "index": variant["index"],
            "mode": variant["mode"],
            "channel": variant["channel"],
            "psnr": psnr,
            "ssim": ssim,
        })
    return records


def aggregate_by_code(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, str, str], Dict[str, object]] = {}
    for record in records:
        key = (record["index"], record["channel"], record["mode"])
        bucket = grouped.setdefault(key, {
            "index": record["index"],
            "channel": record["channel"],
            "mode": record["mode"],
            "sample_count": 0,
            "psnr_sum": 0.0,
            "ssim_sum": 0.0,
        })
        bucket["sample_count"] += 1
        bucket["psnr_sum"] += record["psnr"]
        bucket["ssim_sum"] += record["ssim"]

    summaries: List[Dict[str, object]] = []
    for bucket in grouped.values():
        count = bucket["sample_count"]
        summaries.append({
            "index": bucket["index"],
            "channel": bucket["channel"],
            "mode": bucket["mode"],
            "sample_count": count,
            "avg_psnr": bucket["psnr_sum"] / count,
            "avg_ssim": bucket["ssim_sum"] / count,
        })
    return sorted(summaries, key=lambda item: (item["index"], item["channel"], item["mode"]))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inv/normal画像のPSNR・SSIM集計")
    parser.add_argument("--dir", default=".", help="対象ディレクトリ (default: script location)")
    parser.add_argument("--per-file-csv", default="all_image_quality_by_variant.csv", help="変換画像ごとのCSV出力先")
    parser.add_argument("--code-csv", default="all_image_quality_by_code.csv", help="番号×記号集計のCSV出力先")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*_normal?.png", "*_inv?.png"],
        help="探索するファイルパターン (glob)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.dir).resolve()
    variants = collect_variants(base_dir, args.patterns)
    if not variants:
        print("解析対象ファイルが見つかりませんでした")
        return

    per_file_records = evaluate_variants(variants)
    if not per_file_records:
        print("有効な結果がありませんでした")
        return

    per_file_records.sort(key=lambda item: (item["base_image"], item["index"], item["mode"], item["channel"]))
    per_file_csv = base_dir / args.per_file_csv
    write_csv(per_file_csv, PER_FILE_FIELDS, per_file_records)
    print(f"変換画像ごとの結果を {per_file_csv} に保存しました")

    code_rows = aggregate_by_code(per_file_records)
    code_csv = base_dir / args.code_csv
    write_csv(code_csv, CODE_FIELDS, code_rows)
    print(f"番号×記号の平均結果を {code_csv} に保存しました")


if __name__ == "__main__":
    main()