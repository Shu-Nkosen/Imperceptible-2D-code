from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

INPUT_CSV = Path("eval_summary_accuracy.csv")
OUTPUT_CSV = Path("qr_decode_all_frames.csv")
DIFF_SUBDIR = "rgb_max_diff_maps"
SCALE_TAG = "rgbmax_scale1.00"
ANALYSIS_DIR = Path("median_eval_analysis_all_frames")
GT_IMAGE_PATH = "frame_QR.png"
CACHE_GT_SEARCH_PATH = "frame_QR"
GT_BLACK_THRESHOLD = 0.2
ROI_HEIGHT_RATIO = 0.7
DEFAULT_MEDIAN_KERNEL = 5
DEFAULT_MEDIAN_ITERATIONS = 1
_GT_CACHE: Dict[str, Tuple[Optional[np.ndarray], Optional[Tuple[slice, slice, Tuple[int, int, int, int]]]]] = {}
FAST_VARIANT_ORDER = (
    "gray",
)
MID_VARIANT_ORDER = (
    "gray",
    "median_otsu",
)
FULL_VARIANT_ORDER = (
    "gray",
    "median_gray",
    "otsu",
    "median_otsu",
    "otsu_close",
    "median_otsu_close",
    "otsu_inv",
    "median_otsu_inv",
    "adaptive",
    "median_adaptive",
)
FAST_SCALES = (1.0,)
FULL_SCALES = (1.0, 2.0, 3.0)
MID_MEDIAN_KERNELS = (3, 5, 7)

# search mode: fast | mid | full
# fast: gray × kernel=5 × 最精度デコード1回（detectAndDecodeMulti）
# mid:  gray+median_otsu × kernels 3/5/7 × cascade+Multi
# full: all variants + scales + cascade+Multi


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
    """差分ファイル名の先頭・末尾フレーム番号を返す（pair は隣接、accum は窓端点）。"""
    match = re.match(r"(\d+)-(\d+)-", path.stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def extract_frame_index(frame_name: str) -> Optional[int]:
    match = re.search(r"(\d+)", frame_name)
    if not match:
        return None
    return int(match.group(1))


def resolve_diff_image_path(folder_dir: Path, frame1: str, frame2: str) -> Tuple[Optional[Path], str]:
    idx1 = extract_frame_index(frame1)
    idx2 = extract_frame_index(frame2)
    if idx1 is None or idx2 is None:
        return None, "frame番号抽出に失敗"

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


def resolve_gt_path(target_dir: Path) -> Path:
    candidates = [
        target_dir / GT_IMAGE_PATH,
        target_dir.parent / GT_IMAGE_PATH,
        target_dir.parent / CACHE_GT_SEARCH_PATH / GT_IMAGE_PATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{GT_IMAGE_PATH} が見つかりません")


def get_roi_slice(
    img_shape: Tuple[int, int],
    height_ratio: float = ROI_HEIGHT_RATIO,
) -> Tuple[slice, slice, Tuple[int, int, int, int]]:
    h, w = img_shape
    roi_h = int(h * height_ratio)
    roi_w = roi_h
    start_y = max(0, (h - roi_h) // 2)
    start_x = max(0, (w - roi_w) // 2)
    return slice(start_y, start_y + roi_h), slice(start_x, start_x + roi_w), (start_x, start_y, roi_w, roi_h)


def load_folder_gt(gt_path: Path) -> Tuple[Optional[np.ndarray], Optional[Tuple[slice, slice, Tuple[int, int, int, int]]]]:
    cache_key = str(gt_path.resolve())
    if cache_key in _GT_CACHE:
        return _GT_CACHE[cache_key]

    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        _GT_CACHE[cache_key] = (None, None)
        return None, None

    gt_mask = (gt_img.astype(np.float32) / 255.0) <= GT_BLACK_THRESHOLD
    roi_slices = get_roi_slice(gt_mask.shape)
    _GT_CACHE[cache_key] = (gt_mask, roi_slices)
    return gt_mask, roi_slices


def calculate_metrics_roi(
    detected_mask: np.ndarray,
    gt_mask: np.ndarray,
    roi_slices: Tuple[slice, slice, Tuple[int, int, int, int]],
) -> Tuple[float, float, float]:
    slice_y, slice_x, _ = roi_slices
    roi_detected = detected_mask[slice_y, slice_x]
    roi_gt = gt_mask[slice_y, slice_x]

    tp = int(np.count_nonzero(roi_detected & roi_gt))
    fp = int(np.count_nonzero(roi_detected & ~roi_gt))
    fn = int(np.count_nonzero(~roi_detected & roi_gt))
    tn = int(np.count_nonzero(~roi_detected & ~roi_gt))
    total_pixels = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
    return precision, recall, accuracy


def calculate_fpr_full(detected_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    bg_mask = ~gt_mask
    fp_full = int(np.count_nonzero(detected_mask & bg_mask))
    tn_plus_fp = int(np.count_nonzero(bg_mask))
    return fp_full / tn_plus_fp if tn_plus_fp > 0 else 0.0


def align_image_to_gt(gray: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    if gray.shape[:2] == gt_shape:
        return gray
    return cv2.resize(gray, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_NEAREST)


def image_to_detected_mask(
    gray: np.ndarray,
    gt_mask: np.ndarray,
    roi_slices: Tuple[slice, slice, Tuple[int, int, int, int]],
) -> np.ndarray:
    aligned = align_image_to_gt(gray, gt_mask.shape)
    dark_mask = aligned < 128
    light_mask = ~dark_mask
    _, _, acc_dark = calculate_metrics_roi(dark_mask, gt_mask, roi_slices)
    _, _, acc_light = calculate_metrics_roi(light_mask, gt_mask, roi_slices)
    return dark_mask if acc_dark >= acc_light else light_mask


def compare_with_gt(
    used_img: Optional[np.ndarray],
    gt_path_str: str,
) -> Tuple[str, str, str, str, str]:
    if not gt_path_str:
        return "", "", "", "", "正解画像なし"

    gt_mask, roi_slices = load_folder_gt(Path(gt_path_str))
    if gt_mask is None or roi_slices is None:
        return "", "", "", "", "正解画像読み込み失敗"
    if used_img is None:
        return "", "", "", "", "比較対象画像なし"

    detected_mask = image_to_detected_mask(used_img, gt_mask, roi_slices)
    precision, recall, accuracy = calculate_metrics_roi(detected_mask, gt_mask, roi_slices)
    noise = calculate_fpr_full(detected_mask, gt_mask)
    return (
        f"{recall:.6f}",
        f"{precision:.6f}",
        f"{accuracy:.6f}",
        f"{noise:.6f}",
        "",
    )


def empty_gt_metrics() -> Dict[str, str]:
    return {
        "recall": "",
        "precision": "",
        "accuracy": "",
        "noise": "",
        "gt_note": "",
    }


def collect_all_diff_paths(folder_dir: Path, pair_each_end: int = 0) -> Tuple[List[Path], str]:
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
    total = len(paths)
    if pair_each_end > 0 and total > pair_each_end * 2:
        selected = paths[:pair_each_end] + paths[-pair_each_end:]
        seen: set[str] = set()
        limited: List[Path] = []
        for path in selected:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            limited.append(path)
        paths = limited
        print(
            f"[INFO] {folder_dir.name}: pair limit first/last {pair_each_end} each "
            f"-> {len(paths)}/{total} diffs"
        )
    return paths, ""


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


class VariantCache:
    def __init__(self, gray: np.ndarray, median_gray: np.ndarray) -> None:
        self.gray = gray
        self.median_gray = median_gray
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        if name == "gray":
            image = self.gray
        elif name == "median_gray":
            image = self.median_gray
        elif name == "otsu":
            _, image = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif name == "median_otsu":
            _, image = cv2.threshold(self.median_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif name == "otsu_inv":
            _, image = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif name == "median_otsu_inv":
            _, image = cv2.threshold(self.median_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif name == "adaptive":
            image = cv2.adaptiveThreshold(
                self.gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                2,
            )
        elif name == "median_adaptive":
            image = cv2.adaptiveThreshold(
                self.median_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                2,
            )
        elif name == "otsu_close":
            otsu = self.get("otsu")
            kernel = np.ones((3, 3), np.uint8)
            image = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        elif name == "median_otsu_close":
            median_otsu = self.get("median_otsu")
            kernel = np.ones((3, 3), np.uint8)
            image = cv2.morphologyEx(median_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            raise KeyError(f"unknown variant: {name}")

        self._cache[name] = image
        return image


def iter_variant_targets(
    gray: np.ndarray,
    median_gray: np.ndarray,
    variant_order: Tuple[str, ...],
    scales: Tuple[float, ...],
) -> Iterator[Tuple[str, np.ndarray]]:
    cache = VariantCache(gray, median_gray)
    for variant_name in variant_order:
        variant_img = cache.get(variant_name)
        for scale in scales:
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
            yield f"{variant_name}_x{scale:.1f}", target


def trim_white_borders(gray: np.ndarray, white_min: int = 250) -> np.ndarray:
    """matplotlib余白などが残っている差分画像の白縁を落とす。"""
    if gray.ndim != 2 or gray.size == 0:
        return gray
    content = gray < white_min
    if not np.any(content):
        return gray
    rows = np.any(content, axis=1)
    cols = np.any(content, axis=0)
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    # わずかに余白を残す
    pad = 8
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(gray.shape[0], y1 + pad)
    x1 = min(gray.shape[1], x1 + pad)
    if y1 - y0 < 16 or x1 - x0 < 16:
        return gray
    return gray[y0:y1, x0:x1]


def try_decode_zxing(image: np.ndarray) -> Optional[str]:
    """ZXing (C++) — 欠損・低コントラストQRに強く、本パイプラインの主デコーダ。"""
    try:
        import zxingcpp
    except ImportError:
        return None

    if image is None or image.size == 0:
        return None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    formats = zxingcpp.BarcodeFormat.QRCode
    attempts = (
        {"formats": formats, "try_rotate": True, "try_downscale": True, "try_invert": True},
        {
            "formats": formats,
            "try_rotate": True,
            "try_downscale": True,
            "try_invert": True,
            "binarizer": zxingcpp.Binarizer.FixedThreshold,
        },
        {
            "formats": formats,
            "try_rotate": False,
            "try_downscale": False,
            "try_invert": True,
            "is_pure": True,
        },
    )
    for kwargs in attempts:
        try:
            results = zxingcpp.read_barcodes(gray, **kwargs)
        except Exception:
            continue
        for barcode in results:
            text = getattr(barcode, "text", None) or ""
            if text:
                return text
    return None


def try_decode_multi_once(detector: cv2.QRCodeDetector, image: np.ndarray) -> Optional[str]:
    """OpenCV フォールバック（detectAndDecodeMulti）。"""
    try:
        retval, decoded_info, _, _ = detector.detectAndDecodeMulti(image)
        if retval and decoded_info:
            for value in decoded_info:
                if value:
                    return value
    except cv2.error:
        pass
    return None


def try_decode(detector: cv2.QRCodeDetector, image: np.ndarray, allow_multi: bool) -> Optional[str]:
    """OpenCV QR 読取。スマホより弱いので detectAndDecode も常に試す。"""
    candidates = [image]
    trimmed = trim_white_borders(image)
    if trimmed is not image and trimmed.shape != image.shape:
        candidates.append(trimmed)

    for target in candidates:
        try:
            text, points, _ = detector.detectAndDecode(target)
            if text:
                return text
        except cv2.error:
            pass

        try:
            ok, points = detector.detect(target)
            if ok and points is not None:
                text, _ = detector.decode(target, points)
                if text:
                    return text
        except cv2.error:
            pass

        if allow_multi:
            text = try_decode_multi_once(detector, target)
            if text:
                return text

    return None


def resolve_search_mode(mid_search: bool, full_search: bool) -> str:
    if full_search and mid_search:
        print("[WARN] --full-search と --mid-search が両方指定されています。full-search を優先します。")
    if full_search:
        return "full"
    if mid_search:
        return "mid"
    return "fast"


def search_mode_params(mode: str) -> Tuple[Tuple[str, ...], Tuple[float, ...], str]:
    """(variants, scales, decode_strategy)。strategy: best_once | cascade."""
    if mode == "full":
        return FULL_VARIANT_ORDER, FULL_SCALES, "cascade"
    if mode == "mid":
        return MID_VARIANT_ORDER, FAST_SCALES, "cascade"
    return FAST_VARIANT_ORDER, FAST_SCALES, "best_once"


def decode_qr_from_rgb_array(
    rgb: np.ndarray,
    median_kernel: int,
    median_iterations: int,
    search_mode: str,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    if rgb is None or rgb.size == 0:
        return None, "画像データなし", None
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif rgb.ndim == 2:
        gray = rgb
    else:
        return None, "画像形式不正", None

    median_gray = apply_median_filter(gray, median_kernel, median_iterations)
    detector = cv2.QRCodeDetector()
    last_target: Optional[np.ndarray] = None

    variant_order, scales, decode_strategy = search_mode_params(search_mode)

    for variant_tag, target in iter_variant_targets(gray, median_gray, variant_order, scales):
        last_target = target
        zx_text = try_decode_zxing(target)
        if zx_text:
            return (
                zx_text,
                f"zxing_{variant_tag}_k{median_kernel}_i{median_iterations}",
                target,
            )
        if decode_strategy == "best_once":
            text = try_decode_multi_once(detector, target)
        else:
            text = try_decode(detector, target, allow_multi=True)
        if text:
            return (
                text,
                f"opencv_{variant_tag}_k{median_kernel}_i{median_iterations}",
                target,
            )

    return None, "decode失敗", last_target


def decode_qr_from_diff(
    diff_path: Path,
    median_kernel: int,
    median_iterations: int,
    search_mode: str,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    gray = cv2.imread(str(diff_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, "画像読み込み失敗", None
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return decode_qr_from_rgb_array(rgb, median_kernel, median_iterations, search_mode)


def decode_qr_with_kernel_candidates_from_array(
    rgb: np.ndarray,
    kernel_candidates: List[int],
    median_iterations: int,
    search_mode: str,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    last_target: Optional[np.ndarray] = None
    for kernel in kernel_candidates:
        text, method, used_img = decode_qr_from_rgb_array(
            rgb,
            kernel,
            median_iterations,
            search_mode,
        )
        if used_img is not None:
            last_target = used_img
        if text:
            return text, method, used_img
    return None, "decode失敗", last_target


def decode_qr_with_kernel_candidates(
    diff_path: Path,
    kernel_candidates: List[int],
    median_iterations: int,
    search_mode: str,
) -> Tuple[Optional[str], str, Optional[np.ndarray]]:
    last_target: Optional[np.ndarray] = None
    for kernel in kernel_candidates:
        text, method, used_img = decode_qr_from_diff(
            diff_path,
            kernel,
            median_iterations,
            search_mode,
        )
        if used_img is not None:
            last_target = used_img
        if text:
            return text, method, used_img
    return None, "decode失敗", last_target


def read_target_folders(base_dir: Path, input_csv: Path) -> List[Path]:
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
        [path for path in base_dir.iterdir() if path.is_dir() and (path / DIFF_SUBDIR).exists()],
        key=lambda p: p.name,
    )


def process_array_task(task: Dict[str, object]) -> Dict[str, object]:
    rgb = task["rgb"]
    if not isinstance(rgb, np.ndarray):
        rgb = np.asarray(rgb)
    kernel_candidates = [int(value) for value in task["kernel_candidates"]]
    median_iterations = int(task["median_iterations"])
    search_mode = str(task.get("search_mode") or "fast")
    gt_path_str = str(task.get("gt_path") or "")

    decoded_text, method, used_img = decode_qr_with_kernel_candidates_from_array(
        rgb,
        kernel_candidates,
        median_iterations,
        search_mode,
    )
    ok = decoded_text is not None

    used_img_bytes = None
    if ok and used_img is not None:
        ok_encode, encoded = cv2.imencode(".png", used_img)
        if ok_encode:
            used_img_bytes = encoded.tobytes()

    if used_img is not None:
        img_for_gt = used_img
    elif rgb.ndim == 3:
        img_for_gt = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        img_for_gt = rgb
    recall, precision, accuracy, noise, gt_note = compare_with_gt(img_for_gt, gt_path_str)

    diff_stem = str(task.get("diff_stem") or "in_memory")
    return {
        "folder": str(task["folder"]),
        "frame_1": str(task["frame_1"]),
        "frame_2": str(task["frame_2"]),
        "diff_image": str(task.get("diff_image") or ""),
        "decoded_text": decoded_text or "",
        "success": ok,
        "method": method,
        "note": "" if ok else "QR未検出",
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "noise": noise,
        "gt_note": gt_note,
        "used_img_bytes": used_img_bytes,
        "method_tag": sanitize_filename(method if method else "unknown"),
        "diff_stem": diff_stem,
        "task_key": str(task.get("task_key") or ""),
    }


def process_diff_task(task: Dict[str, object]) -> Dict[str, object]:
    diff_path = Path(str(task["diff_path"]))
    kernel_candidates = [int(value) for value in task["kernel_candidates"]]
    median_iterations = int(task["median_iterations"])
    search_mode = str(task.get("search_mode") or "fast")
    gt_path_str = str(task.get("gt_path") or "")

    gray = cv2.imread(str(diff_path), cv2.IMREAD_GRAYSCALE)

    decoded_text, method, used_img = decode_qr_with_kernel_candidates(
        diff_path,
        kernel_candidates,
        median_iterations,
        search_mode,
    )
    ok = decoded_text is not None

    used_img_bytes = None
    if ok and used_img is not None:
        ok_encode, encoded = cv2.imencode(".png", used_img)
        if ok_encode:
            used_img_bytes = encoded.tobytes()

    img_for_gt = used_img if used_img is not None else gray
    recall, precision, accuracy, noise, gt_note = compare_with_gt(img_for_gt, gt_path_str)

    return {
        "folder": str(task["folder"]),
        "frame_1": str(task["frame_1"]),
        "frame_2": str(task["frame_2"]),
        "diff_image": str(task["diff_image"]),
        "decoded_text": decoded_text or "",
        "success": ok,
        "method": method,
        "note": "" if ok else "QR未検出",
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "noise": noise,
        "gt_note": gt_note,
        "used_img_bytes": used_img_bytes,
        "method_tag": sanitize_filename(method if method else "unknown"),
        "diff_stem": diff_path.stem,
    }


def build_tasks(
    target_folders: List[Path],
    base_dir: Path,
    kernel_candidates: List[int],
    median_iterations: int,
    search_mode: str,
    pair_each_end: int = 0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    tasks: List[Dict[str, object]] = []
    pre_rows: List[Dict[str, object]] = []

    for folder_dir in target_folders:
        diff_paths, note = collect_all_diff_paths(folder_dir, pair_each_end=pair_each_end)
        gt_path_str = ""
        gt_note = ""
        try:
            gt_path_str = str(resolve_gt_path(folder_dir).resolve())
        except FileNotFoundError as exc:
            gt_note = str(exc)

        if not diff_paths:
            pre_rows.append(
                {
                    "folder": folder_dir.name,
                    "frame_1": "",
                    "frame_2": "",
                    "diff_image": "",
                    "analysis_image": "",
                    "decoded_text": "",
                    "success": 0,
                    "method": "",
                    "note": note,
                    **empty_gt_metrics(),
                    "gt_note": gt_note,
                }
            )
            print(f"[NG] {folder_dir.name}: {note}")
            continue

        for diff_path in diff_paths:
            pair = parse_diff_pair_indices(diff_path)
            if pair is None:
                pre_rows.append(
                    {
                        "folder": folder_dir.name,
                        "frame_1": "",
                        "frame_2": "",
                        "diff_image": str(diff_path.relative_to(base_dir)),
                        "analysis_image": "",
                        "decoded_text": "",
                        "success": 0,
                        "method": "",
                        "note": "差分画像名の解析に失敗",
                        **empty_gt_metrics(),
                        "gt_note": gt_note,
                    }
                )
                continue

            frame_left, frame_right = pair
            tasks.append(
                {
                    "diff_path": str(diff_path.resolve()),
                    "folder": folder_dir.name,
                    "frame_1": f"frame_{frame_left:05d}.png",
                    "frame_2": f"frame_{frame_right:05d}.png",
                    "diff_image": str(diff_path.relative_to(base_dir)),
                    "kernel_candidates": kernel_candidates,
                    "median_iterations": median_iterations,
                    "search_mode": search_mode,
                    "gt_path": gt_path_str,
                }
            )

    return tasks, pre_rows


def finalize_result_row(
    result: Dict[str, object],
    base_dir: Path,
    analysis_dir: Path,
    save_analysis: bool,
) -> Dict[str, object]:
    analysis_image_rel = ""
    if save_analysis and result["success"] and result.get("used_img_bytes"):
        folder_analysis_dir = analysis_dir / str(result["folder"])
        folder_analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_image_name = (
            f"{result['diff_stem']}_ok_{result['method_tag']}.png"
        )
        analysis_image_path = folder_analysis_dir / analysis_image_name
        analysis_image_path.write_bytes(result["used_img_bytes"])
        analysis_image_rel = str(analysis_image_path.relative_to(base_dir))

    return {
        "folder": result["folder"],
        "frame_1": result["frame_1"],
        "frame_2": result["frame_2"],
        "diff_image": result["diff_image"],
        "analysis_image": analysis_image_rel,
        "decoded_text": result["decoded_text"],
        "success": int(result["success"]),
        "method": result["method"],
        "note": result["note"],
        "recall": result.get("recall", ""),
        "precision": result.get("precision", ""),
        "accuracy": result.get("accuracy", ""),
        "noise": result.get("noise", ""),
        "gt_note": result.get("gt_note", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="全フレーム差分画像からQR文字列を復元する")
    parser.add_argument("--folder", type=str, default="", help="対象フォルダ名を1つに限定（例: ex_B6）")
    parser.add_argument("--limit", type=int, default=0, help="先頭から処理するフォルダ数（0で全件）")
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
    parser.add_argument(
        "--mid-search",
        action="store_true",
        help="gray+median_otsu × cascade+Multi × median kernels 3/5/7（拡大なし）",
    )
    parser.add_argument(
        "--full-search",
        action="store_true",
        help="全バリアント×全スケール(1/2/3)×cascade+Multiで徹底探索（遅い）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="並列ワーカー数（既定: 16。実コア数を超えないよう自動上限。1で逐次処理）",
    )
    parser.add_argument(
        "--no-save-analysis",
        action="store_true",
        help="成功時の解析画像を保存しない",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="対象ルートの絶対パス（未指定時は本スクリプト配置ディレクトリ）",
    )
    parser.add_argument(
        "--diff-subdir",
        type=str,
        default="",
        help="差分サブディレクトリ名（未指定時: rgb_max_diff_maps）",
    )
    parser.add_argument(
        "--pair-each-end",
        type=int,
        default=0,
        help="pair 時: 先頭/末尾それぞれ N ペアだけデコード（0=全ペア）",
    )
    args = parser.parse_args()

    global DIFF_SUBDIR
    if args.diff_subdir.strip():
        DIFF_SUBDIR = args.diff_subdir.strip()

    median_kernel = max(1, args.median_kernel)
    if median_kernel % 2 == 0:
        median_kernel += 1
    median_iterations = max(0, args.median_iterations)
    search_mode = resolve_search_mode(args.mid_search, args.full_search)
    if args.median_kernels:
        kernel_candidates = parse_kernel_list(args.median_kernels)
    elif search_mode == "mid":
        kernel_candidates = list(MID_MEDIAN_KERNELS)
    else:
        kernel_candidates = [median_kernel]
    if not kernel_candidates:
        kernel_candidates = [median_kernel]
    requested_workers = max(1, args.workers)
    cpu_count = os.cpu_count() or 1
    workers = min(requested_workers, cpu_count)
    if workers < requested_workers:
        print(
            f"[INFO] workers capped: requested={requested_workers} -> {workers} "
            f"(cpu_count={cpu_count})"
        )
    save_analysis = not args.no_save_analysis

    try:
        import zxingcpp  # noqa: F401

        decode_backend = "zxing-cpp(+opencv fallback)"
    except ImportError:
        decode_backend = "opencv-only (pip install zxing-cpp 推奨)"

    print(
        f"[INFO] mode: {search_mode} / folder filter: {args.folder or '(all)'} / limit: {args.limit} "
        f"/ median kernels: {kernel_candidates}, iter={median_iterations}, workers={workers} "
        f"/ diff_subdir={DIFF_SUBDIR} / decode={decode_backend}"
    )

    if args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = Path(__file__).resolve().parent
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base-dir が存在しません: {base_dir}")
    print(f"[INFO] base_dir: {base_dir}")

    input_csv = base_dir / INPUT_CSV
    output_csv = base_dir / OUTPUT_CSV
    analysis_dir = base_dir / ANALYSIS_DIR
    if save_analysis:
        analysis_dir.mkdir(parents=True, exist_ok=True)

    target_folders = read_target_folders(base_dir, input_csv)
    if args.folder:
        target_folders = [path for path in target_folders if path.name == args.folder]
    if args.limit > 0:
        target_folders = target_folders[: args.limit]

    if not target_folders:
        raise FileNotFoundError("対象フォルダが見つかりません。--folder または入力CSVを確認してください。")

    tasks, rows_out = build_tasks(
        target_folders,
        base_dir,
        kernel_candidates,
        median_iterations,
        search_mode,
        pair_each_end=max(0, int(args.pair_each_end)),
    )

    total = len(tasks)
    success = 0

    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            result = process_diff_task(task)
            row = finalize_result_row(result, base_dir, analysis_dir, save_analysis)
            rows_out.append(row)
            if row["success"]:
                success += 1
                print(
                    f"[OK] {row['folder']}: {row['frame_1']} / {row['frame_2']} -> {row['decoded_text']}"
                )
            elif index % 50 == 0 or index == total:
                print(f"[INFO] progress: {index}/{total}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_diff_task, task) for task in tasks]
            done = 0
            for future in as_completed(futures):
                result = future.result()
                row = finalize_result_row(result, base_dir, analysis_dir, save_analysis)
                rows_out.append(row)
                done += 1
                if row["success"]:
                    success += 1
                    print(
                        f"[OK] {row['folder']}: {row['frame_1']} / {row['frame_2']} -> {row['decoded_text']}"
                    )
                elif done % 50 == 0 or done == total:
                    print(f"[INFO] progress: {done}/{total}")

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "folder",
            "frame_1",
            "frame_2",
            "diff_image",
            "analysis_image",
            "decoded_text",
            "success",
            "method",
            "note",
            "recall",
            "precision",
            "accuracy",
            "noise",
            "gt_note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n==== 完了 ====")
    print(f"総件数: {total}")
    print(f"成功件数: {success}")
    print(f"成功率: {success / total:.2%}" if total else "成功率: 0.00%")
    print(f"出力CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
