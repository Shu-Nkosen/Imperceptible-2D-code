from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

INPUT_CSV = Path("eval_summary_accuracy.csv")
OUTPUT_CSV = Path("qr_decode_best_frames.csv")
DIFF_SUBDIR = "rgb_max_diff_maps"
SCALE_TAG = "rgbmax_scale1.00"
ANALYSIS_DIR = Path("median_eval_analysis")
DEFAULT_MEDIAN_KERNEL = 5
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
    unique = []
    seen = set()
    for kernel in kernels:
        if kernel not in seen:
            seen.add(kernel)
            unique.append(kernel)
    return unique


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def extract_frame_index(frame_name: str) -> Optional[int]:
    match = re.search(r"(\d+)", frame_name)
    if not match:
        return None
    return int(match.group(1))


def resolve_diff_image_path(folder_dir: Path, frame1: str, frame2: str) -> Tuple[Optional[Path], str]:
    idx1 = extract_frame_index(frame1)
    idx2 = extract_frame_index(frame2)
    if idx1 is None or idx2 is None:
        return None, "best_frameの番号抽出に失敗"

    left, right = sorted((idx1, idx2))
    diff_dir = folder_dir / DIFF_SUBDIR
    if not diff_dir.exists():
        return None, f"差分ディレクトリなし: {DIFF_SUBDIR}"

    expected = diff_dir / f"{left:05d}-{right:05d}-FRAME_{SCALE_TAG}.png"
    if expected.exists():
        return expected, ""

    candidates = sorted(diff_dir.glob(f"{left:05d}-{right:05d}-*_rgbmax_scale*.png"))
    if not candidates:
        return None, "対応する差分画像が見つからない"

    return candidates[0], ""


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

    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    if retval and decoded_info:
        for value in decoded_info:
            if value:
                return value

    return None


def decode_qr_from_diff(
    diff_path: Path,
    median_kernel: int,
    median_iterations: int,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    image = cv2.imread(str(diff_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, "画像読み込み失敗", None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_gray = apply_median_filter(gray, median_kernel, median_iterations)
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
                return text, f"{variant_name}_x{scale:.1f}_k{median_kernel}_i{median_iterations}", target

    return None, "decode失敗", last_target


def decode_qr_with_kernel_candidates(
    diff_path: Path,
    kernel_candidates: List[int],
    median_iterations: int,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    last_target: Optional[np.ndarray] = None
    for kernel in kernel_candidates:
        text, method, used_img = decode_qr_from_diff(diff_path, kernel, median_iterations)
        if used_img is not None:
            last_target = used_img
        if text:
            return text, method, used_img
    return None, "decode失敗", last_target


def main() -> None:
    parser = argparse.ArgumentParser(description="best_frame差分画像からQR文字列を復元する（案A）")
    parser.add_argument("--folder", type=str, default="", help="対象フォルダ名を1つに限定（例: ex_B6）")
    parser.add_argument("--limit", type=int, default=0, help="先頭から処理する行数（0で全件）")
    parser.add_argument(
        "--median-kernel",
        type=int,
        default=DEFAULT_MEDIAN_KERNEL,
        help="メディアンフィルタのカーネルサイズ（偶数指定時は+1）",
    )
    parser.add_argument(
        "--median-kernels",
        type=str,
        default="",
        help="カーネル総当たり（例: 3,5,7）。指定時は --median-kernel より優先",
    )
    parser.add_argument(
        "--median-iterations",
        type=int,
        default=DEFAULT_MEDIAN_ITERATIONS,
        help="メディアンフィルタ反復回数（0で無効）",
    )
    args = parser.parse_args()
    median_kernel = max(1, args.median_kernel)
    if median_kernel % 2 == 0:
        median_kernel += 1
    median_iterations = max(0, args.median_iterations)
    kernel_candidates = parse_kernel_list(args.median_kernels) if args.median_kernels else [median_kernel]
    if not kernel_candidates:
        kernel_candidates = [median_kernel]

    print(
        f"[INFO] folder filter: {args.folder or '(all)'} / limit: {args.limit} "
        f"/ median kernels: {kernel_candidates}, iter={median_iterations}"
    )

    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / INPUT_CSV
    output_csv = base_dir / OUTPUT_CSV
    analysis_dir = base_dir / ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_csv}")

    rows_out = []
    total = 0
    success = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.folder and row.get("folder", "") != args.folder:
                continue
            if args.limit > 0 and total >= args.limit:
                break

            total += 1
            folder = row.get("folder", "")
            frame1 = row.get("best_frame_1", "")
            frame2 = row.get("best_frame_2", "")

            folder_dir = base_dir / folder
            if not folder or not folder_dir.exists():
                rows_out.append(
                    {
                        "folder": folder,
                        "best_frame_1": frame1,
                        "best_frame_2": frame2,
                        "diff_image": "",
                        "decoded_text": "",
                        "success": 0,
                        "method": "",
                        "note": "フォルダが存在しない",
                    }
                )
                continue

            diff_path, note = resolve_diff_image_path(folder_dir, frame1, frame2)
            if diff_path is None:
                rows_out.append(
                    {
                        "folder": folder,
                        "best_frame_1": frame1,
                        "best_frame_2": frame2,
                        "diff_image": "",
                        "decoded_text": "",
                        "success": 0,
                        "method": "",
                        "note": note,
                    }
                )
                print(f"[NG] {folder}: {note}")
                continue

            decoded_text, method, used_img = decode_qr_with_kernel_candidates(
                diff_path,
                kernel_candidates,
                median_iterations,
            )
            ok = int(decoded_text is not None)
            success += ok

            pair_name = f"{Path(frame1).stem}-{Path(frame2).stem}"
            folder_analysis_dir = analysis_dir / folder
            folder_analysis_dir.mkdir(parents=True, exist_ok=True)
            method_tag = sanitize_filename(method if method else "unknown")
            result_tag = "ok" if ok else "ng"
            analysis_image_name = f"{pair_name}_{result_tag}_{method_tag}.png"
            analysis_image_path = folder_analysis_dir / analysis_image_name
            if used_img is not None:
                cv2.imwrite(str(analysis_image_path), used_img)
                analysis_image_rel = str(analysis_image_path.relative_to(base_dir))
            else:
                analysis_image_rel = ""

            rows_out.append(
                {
                    "folder": folder,
                    "best_frame_1": frame1,
                    "best_frame_2": frame2,
                    "diff_image": str(diff_path.relative_to(base_dir)),
                    "analysis_image": analysis_image_rel,
                    "decoded_text": decoded_text or "",
                    "success": ok,
                    "method": method,
                    "note": "" if ok else "QR未検出",
                }
            )

            if ok:
                print(f"[OK] {folder}: {decoded_text}")
            else:
                print(f"[NG] {folder}: QR未検出 ({diff_path.name})")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "folder",
            "best_frame_1",
            "best_frame_2",
            "diff_image",
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
    print(f"総件数: {total}")
    print(f"成功件数: {success}")
    print(f"成功率: {success / total:.2%}" if total else "成功率: 0.00%")
    print(f"出力CSV: {output_csv}")


if __name__ == "__main__":
    main()
