from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

INPUT_CSV = Path("eval_summary_accuracy.csv")
OUTPUT_CSV = Path("qr_decode_all_voting.csv")
DIFF_SUBDIR = "rgb_max_diff_maps"
ANALYSIS_DIR = Path("median_eval_analysis")
TARGET_DIFF_COUNT = 200
DEFAULT_BLACK_RATIO = 0.30
DEFAULT_MEDIAN_KERNELS = "1,3,5,7"
DEFAULT_MEDIAN_ITERATIONS = 1


def parse_kernel_list(text: str) -> List[int]:
    kernels: List[int] = []
    for chunk in text.split(","):
        value = chunk.strip()
        if not value:
            continue
        kernel = abs(int(value))
        if kernel == 0:
            continue
        if kernel % 2 == 0:
            kernel += 1
        kernels.append(kernel)

    unique: List[int] = []
    seen = set()
    for kernel in kernels:
        if kernel not in seen:
            seen.add(kernel)
            unique.append(kernel)
    return unique


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def parse_diff_pair_indices(path: Path) -> Optional[Tuple[int, int]]:
    match = re.match(r"(\d+)-(\d+)-", path.stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def get_target_folders(base_dir: Path, input_csv: Path) -> List[Path]:
    folders: List[Path] = []

    if input_csv.exists():
        with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder_name = (row.get("folder") or "").strip()
                if not folder_name:
                    continue
                folder_path = base_dir / folder_name
                if folder_path.exists() and folder_path.is_dir():
                    folders.append(folder_path)

    if folders:
        unique: List[Path] = []
        seen = set()
        for path in folders:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    return sorted(
        [
            path
            for path in base_dir.iterdir()
            if path.is_dir() and (path / DIFF_SUBDIR).exists()
        ],
        key=lambda p: p.name,
    )


def collect_diff_paths(folder_dir: Path, target_count: int) -> Tuple[List[Path], str]:
    diff_dir = folder_dir / DIFF_SUBDIR
    if not diff_dir.exists():
        return [], f"差分ディレクトリなし: {DIFF_SUBDIR}"

    paths = sorted(diff_dir.glob("*_rgbmax_scale*.png"))
    if not paths:
        return [], "差分画像が見つからない"

    def sort_key(path: Path):
        pair = parse_diff_pair_indices(path)
        if pair is None:
            return (1, 10**9, 10**9, path.name)
        left, right = pair
        return (0, left, right, path.name)

    paths = sorted(paths, key=sort_key)
    if target_count > 0 and len(paths) > target_count:
        return paths[:target_count], f"{len(paths)}枚中先頭{target_count}枚を使用"

    return paths, ""


def load_gray_images(paths: List[Path]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def build_vote_map_by_mean_black_ratio(
    gray_images: List[np.ndarray],
    black_ratio: float,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    if not gray_images:
        return None, None, None

    stack = np.stack(gray_images, axis=0).astype(np.float32)
    darkness = 1.0 - stack / 255.0
    mean_darkness = np.mean(darkness, axis=0)

    ratio = float(np.clip(black_ratio, 0.001, 0.999))
    quantile = 1.0 - ratio
    threshold = float(np.quantile(mean_darkness, quantile))

    black_mask = mean_darkness >= threshold
    vote_map = np.where(black_mask, 0, 255).astype(np.uint8)
    actual_black_ratio = float(np.mean(black_mask))
    return vote_map, threshold, actual_black_ratio


def apply_median_filter(gray: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    iters = max(0, int(iterations))

    filtered = gray.copy()
    if size < 3 or iters == 0:
        return filtered

    for _ in range(iters):
        filtered = cv2.medianBlur(filtered, size)
    return filtered


def build_variants(gray: np.ndarray, median_gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []

    variants.append(("gray", gray))
    variants.append(("median_gray", median_gray))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    _, median_otsu = cv2.threshold(median_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("median_otsu", median_otsu))

    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(("otsu_inv", otsu_inv))

    _, median_otsu_inv = cv2.threshold(median_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(("median_otsu_inv", median_otsu_inv))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    variants.append(("adaptive", adaptive))

    median_adaptive = cv2.adaptiveThreshold(
        median_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    variants.append(("median_adaptive", median_adaptive))

    kernel = np.ones((3, 3), np.uint8)
    close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    variants.append(("otsu_close", close))

    median_close = cv2.morphologyEx(median_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    variants.append(("median_otsu_close", median_close))

    return variants


def try_decode(detector: cv2.QRCodeDetector, image: np.ndarray) -> Optional[str]:
    text, points, _ = detector.detectAndDecode(image)
    if points is not None and text:
        return text

    retval, decoded_info, _, _ = detector.detectAndDecodeMulti(image)
    if retval and decoded_info:
        for value in decoded_info:
            if value:
                return value

    return None


def decode_qr_from_gray(
    gray: np.ndarray,
    kernel_size: int,
    iterations: int,
    prefix: str,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    median_gray = apply_median_filter(gray, kernel_size, iterations)
    detector = cv2.QRCodeDetector()
    last_target: Optional[np.ndarray] = None

    for variant_name, variant_img in build_variants(gray, median_gray):
        for scale in (1.0, 2.0, 3.0):
            if scale == 1.0:
                target = variant_img
            else:
                target = cv2.resize(
                    variant_img,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            last_target = target

            text = try_decode(detector, target)
            if text:
                method = f"{prefix}_{variant_name}_x{scale:.1f}_k{kernel_size}_i{iterations}"
                return text, method, target

    return None, f"{prefix}_decode失敗", last_target


def decode_with_kernel_candidates(
    gray: np.ndarray,
    kernel_candidates: List[int],
    iterations: int,
    prefix: str,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    last_target: Optional[np.ndarray] = None
    for kernel in kernel_candidates:
        text, method, used_img = decode_qr_from_gray(gray, kernel, iterations, prefix)
        if used_img is not None:
            last_target = used_img
        if text:
            return text, method, used_img
    return None, f"{prefix}_decode失敗", last_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="best_frame非依存: 各フォルダの差分200枚を平均投票統合してQR復元"
    )
    parser.add_argument("--folder", type=str, default="", help="対象フォルダ名を1つに限定")
    parser.add_argument("--limit", type=int, default=0, help="先頭から処理するフォルダ数（0で全件）")
    parser.add_argument(
        "--target-diff-count",
        type=int,
        default=TARGET_DIFF_COUNT,
        help="投票に使う差分枚数（0で全件）。既定は200",
    )
    parser.add_argument(
        "--black-ratio",
        type=float,
        default=DEFAULT_BLACK_RATIO,
        help="平均暗度画像で黒にする画素割合（0.0-1.0）",
    )
    parser.add_argument(
        "--median-kernels",
        type=str,
        default=DEFAULT_MEDIAN_KERNELS,
        help="QR復元時メディアンカーネル候補（例: 3,5,7）",
    )
    parser.add_argument(
        "--median-iterations",
        type=int,
        default=DEFAULT_MEDIAN_ITERATIONS,
        help="メディアン反復回数",
    )
    args = parser.parse_args()

    black_ratio = float(np.clip(args.black_ratio, 0.001, 0.999))
    kernel_candidates = parse_kernel_list(args.median_kernels)
    if not kernel_candidates:
        kernel_candidates = parse_kernel_list(DEFAULT_MEDIAN_KERNELS)
    median_iterations = max(0, int(args.median_iterations))

    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / INPUT_CSV
    output_csv = base_dir / OUTPUT_CSV
    analysis_dir = base_dir / ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)

    target_folders = get_target_folders(base_dir, input_csv)
    if args.folder:
        target_folders = [p for p in target_folders if p.name == args.folder]
    if args.limit > 0:
        target_folders = target_folders[: args.limit]

    if not target_folders:
        raise FileNotFoundError("対象フォルダが見つかりません。--folder または入力CSVを確認してください。")

    print(
        f"[INFO] folders={len(target_folders)} / target_diff_count={args.target_diff_count} "
        f"/ black_ratio={black_ratio:.3f} / kernels={kernel_candidates}"
    )

    rows_out = []
    success = 0

    for folder_dir in target_folders:
        diff_paths, note = collect_diff_paths(folder_dir, max(0, args.target_diff_count))
        if not diff_paths:
            rows_out.append(
                {
                    "folder": folder_dir.name,
                    "diff_count": 0,
                    "used_diff_count": 0,
                    "vote_threshold": "",
                    "actual_black_ratio": "",
                    "analysis_image": "",
                    "decoded_text": "",
                    "success": 0,
                    "method": "",
                    "note": note or "差分画像なし",
                }
            )
            print(f"[NG] {folder_dir.name}: {note or '差分画像なし'}")
            continue

        gray_images = load_gray_images(diff_paths)
        if not gray_images:
            rows_out.append(
                {
                    "folder": folder_dir.name,
                    "diff_count": len(diff_paths),
                    "used_diff_count": 0,
                    "vote_threshold": "",
                    "actual_black_ratio": "",
                    "analysis_image": "",
                    "decoded_text": "",
                    "success": 0,
                    "method": "",
                    "note": "差分画像の読み込み失敗",
                }
            )
            print(f"[NG] {folder_dir.name}: 差分画像の読み込み失敗")
            continue

        vote_map, vote_threshold, actual_black_ratio = build_vote_map_by_mean_black_ratio(gray_images, black_ratio)
        if vote_map is None:
            rows_out.append(
                {
                    "folder": folder_dir.name,
                    "diff_count": len(diff_paths),
                    "used_diff_count": 0,
                    "vote_threshold": "",
                    "actual_black_ratio": "",
                    "analysis_image": "",
                    "decoded_text": "",
                    "success": 0,
                    "method": "",
                    "note": "投票画像生成失敗",
                }
            )
            print(f"[NG] {folder_dir.name}: 投票画像生成失敗")
            continue

        prefix = f"voteMean_n{len(gray_images)}_br{black_ratio:.3f}"
        decoded_text, method, used_img = decode_with_kernel_candidates(
            vote_map,
            kernel_candidates,
            median_iterations,
            prefix,
        )

        ok = int(decoded_text is not None)
        success += ok

        folder_analysis_dir = analysis_dir / folder_dir.name
        folder_analysis_dir.mkdir(parents=True, exist_ok=True)
        method_tag = sanitize_filename(method)
        result_tag = "ok" if ok else "ng"
        analysis_path = folder_analysis_dir / f"allVoting_{result_tag}_{method_tag}.png"
        if used_img is not None:
            cv2.imwrite(str(analysis_path), used_img)
            analysis_rel = str(analysis_path.relative_to(base_dir))
        else:
            analysis_rel = ""

        rows_out.append(
            {
                "folder": folder_dir.name,
                "diff_count": len(list((folder_dir / DIFF_SUBDIR).glob("*_rgbmax_scale*.png"))),
                "used_diff_count": len(gray_images),
                "vote_threshold": f"{vote_threshold:.6f}" if vote_threshold is not None else "",
                "actual_black_ratio": f"{actual_black_ratio:.6f}" if actual_black_ratio is not None else "",
                "analysis_image": analysis_rel,
                "decoded_text": decoded_text or "",
                "success": ok,
                "method": method,
                "note": "" if ok else (note + " / QR未検出" if note else "QR未検出"),
            }
        )

        if ok:
            print(f"[OK] {folder_dir.name}: {decoded_text}")
        else:
            print(f"[NG] {folder_dir.name}: QR未検出")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "folder",
            "diff_count",
            "used_diff_count",
            "vote_threshold",
            "actual_black_ratio",
            "analysis_image",
            "decoded_text",
            "success",
            "method",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print("\n==== 完了 ====")
    print(f"総フォルダ数: {len(rows_out)}")
    print(f"成功件数: {success}")
    print(f"成功率: {success / len(rows_out):.2%}" if rows_out else "成功率: 0.00%")
    print(f"出力CSV: {output_csv}")


if __name__ == "__main__":
    main()
