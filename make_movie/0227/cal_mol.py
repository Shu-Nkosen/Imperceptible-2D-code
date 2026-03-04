import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Sequence, Tuple
from PIL import Image
from types import SimpleNamespace


# =====================================================
#  ユーザー設定 (Morphology + Accuracy 評価)
# =====================================================
THRESHOLD = 2 / 255
QUALITY_SCALE = 1
RESIZE_METHOD = "nearest"
GT_IMAGE_PATH = "frame_QR.png"
GT_BLACK_THRESHOLD = 0.2
ROI_HEIGHT_RATIO = 0.7
OUTPUT_DIR = "mol_eval_analysis"
# [] にすると単一サイズのみ。値を入れるとここで列挙したカーネルを順番に試す。
KERNEL_SIZE_PRESETS: List[int] = [1, 2, 3, 4]
DEFAULT_OPERATIONS = "open,close"
DEFAULT_KERNEL_SIZE = 5
DEFAULT_ITERATIONS = 1

DEFAULT_CONFIG = {
    "threshold": THRESHOLD,
    "quality_scale": QUALITY_SCALE,
    "resize_method": RESIZE_METHOD,
    "gt_black_threshold": GT_BLACK_THRESHOLD,
    "roi_height_ratio": ROI_HEIGHT_RATIO,
    "operations": DEFAULT_OPERATIONS,
    "kernel_size": DEFAULT_KERNEL_SIZE,
    "kernel_sizes": "",
    "iterations": DEFAULT_ITERATIONS,
    "output_dir": OUTPUT_DIR,
    "output_name_template": "",
    "gt": GT_IMAGE_PATH,
}

# ここに評価したい画像ペアを追加 (script ディレクトリ基準のパスを推奨)
EVALUATION_TASKS: List[dict] = [
    {
        "image1": "rice_G4/frame_00004.png",
        "image2": "rice_G4/frame_00005.png",
    },
]


def load_image_as_rgb(image_path: Path, scale: float, method: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    if scale != 1.0:
        width, height = int(img.width * scale), int(img.height * scale)
        resample = getattr(Image, method.upper(), Image.NEAREST)
        img = img.resize((width, height), resample)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_image_as_grayscale(image_path: Path, scale: float, method: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    if scale != 1.0:
        width, height = int(img.width * scale), int(img.height * scale)
        resample = getattr(Image, method.upper(), Image.NEAREST)
        img = img.resize((width, height), resample)
    return np.asarray(img, dtype=np.float32) / 255.0


def get_roi_slice(img_shape: Tuple[int, int], height_ratio: float):
    h, w = img_shape
    roi_h = int(h * height_ratio)
    roi_w = roi_h
    start_y = max(0, (h - roi_h) // 2)
    start_x = max(0, (w - roi_w) // 2)
    return slice(start_y, start_y + roi_h), slice(start_x, start_x + roi_w), (start_x, start_y, roi_w, roi_h)


def calculate_metrics_roi(detected_mask: np.ndarray, gt_mask: np.ndarray, roi_slices):
    slice_y, slice_x, _ = roi_slices
    roi_detected = detected_mask[slice_y, slice_x]
    roi_gt = gt_mask[slice_y, slice_x]

    tp = np.count_nonzero(roi_detected & roi_gt)
    fp = np.count_nonzero(roi_detected & ~roi_gt)
    fn = np.count_nonzero(~roi_detected & roi_gt)
    tn = np.count_nonzero(~roi_detected & ~roi_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return precision, recall, accuracy


def calculate_fpr_full(detected_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    bg_mask = ~gt_mask
    fp_full = np.count_nonzero(detected_mask & bg_mask)
    tn_plus_fp = np.count_nonzero(bg_mask)
    return fp_full / tn_plus_fp if tn_plus_fp else 0.0


def save_evaluated_map(detected: np.ndarray, roi_rect, scores, output_path: Path, quality_scale: float):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    combined = np.ones((*detected.shape, 3))
    combined[detected] = [0, 0, 0]

    ax.imshow(combined)

    precision, recall, accuracy, fpr = scores
    title_text = (
        f"Morphology Analysis (Scale: {quality_scale})\n"
        f"ROI Recall: {recall:.2%} | Precision: {precision:.2%} | Accuracy: {accuracy:.2%}\n"
        f"Full Image Noise Rate (FPR): {fpr:.2%}"
    )
    ax.set_title(title_text)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_operations(sequence: str) -> List[str]:
    if not sequence:
        return []
    return [op.strip().lower() for op in sequence.split(',') if op.strip()]


def parse_kernel_sizes(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = abs(int(chunk))
        if value == 0:
            continue
        if value % 2 == 0:
            value += 1
        values.append(value)
    return list(dict.fromkeys(values))  # preserve order, deduplicate


def resolve_kernel_size_list(config) -> List[int]:
    if getattr(config, "kernel_sizes", ""):
        return parse_kernel_sizes(config.kernel_sizes)
    if KERNEL_SIZE_PRESETS:
        return KERNEL_SIZE_PRESETS
    return [config.kernel_size]


def build_task_configs() -> List[SimpleNamespace]:
    configs: List[SimpleNamespace] = []
    base_dir = Path(__file__).parent
    for idx, overrides in enumerate(EVALUATION_TASKS, start=1):
        if "image1" not in overrides or "image2" not in overrides:
            print(f"[SKIP] タスク{idx}: image1 / image2 が未設定です")
            continue
        data = DEFAULT_CONFIG.copy()
        data.update(overrides)

        def resolve_path(value):
            path = Path(value)
            return path if path.is_absolute() else base_dir / path

        data["image1"] = resolve_path(data["image1"])
        data["image2"] = resolve_path(data["image2"])
        data["gt"] = resolve_path(data.get("gt", GT_IMAGE_PATH))
        output_dir = Path(data.get("output_dir", OUTPUT_DIR))
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        data["output_dir"] = output_dir

        configs.append(SimpleNamespace(**data))
    return configs


def compute_difference_mask(img1: np.ndarray, img2: np.ndarray, threshold: float) -> np.ndarray:
    diff = np.abs(img2 - img1)
    max_diff = diff.max(axis=2)
    return max_diff > threshold


def apply_morphology(mask: np.ndarray, operations: Sequence[str], kernel_size: int, iterations: int) -> np.ndarray:
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    work = mask.astype(np.uint8) * 255
    for op in operations:
        if op == "open":
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif op == "close":
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif op == "dilate":
            work = cv2.dilate(work, kernel, iterations=iterations)
        elif op == "erode":
            work = cv2.erode(work, kernel, iterations=iterations)
        else:
            print(f"[WARN] 未対応モルフォロジー演算: {op} をスキップします")
    return work > 0


def build_filename(config, pair_name: str, kernel_size: int, precision: float, recall: float, accuracy: float, fpr: float) -> str:
    metrics = {
        "pair": pair_name,
        "kernel": kernel_size,
        "recall": int(recall * 100),
        "precision": int(precision * 100),
        "accuracy": int(accuracy * 100),
        "noise": int(fpr * 100),
    }
    if getattr(config, "output_name_template", ""):
        try:
            filename = config.output_name_template.format(**metrics)
        except KeyError as exc:
            raise ValueError(f"テンプレート内の未知キー: {exc}") from exc
        if not filename.lower().endswith(".png"):
            filename += ".png"
        return filename
    return (
        f"{pair_name}_K{kernel_size}_R{metrics['recall']}_P{metrics['precision']}_"
        f"A{metrics['accuracy']}_N{metrics['noise']}.png"
    )


def evaluate_pair(config, kernel_size: int) -> None:
    image1: Path = config.image1
    image2: Path = config.image2
    gt_path: Path = config.gt

    img1 = load_image_as_rgb(image1, config.quality_scale, config.resize_method)
    img2 = load_image_as_rgb(image2, config.quality_scale, config.resize_method)
    if img1.shape != img2.shape:
        raise ValueError("入力画像のサイズが一致しません")

    gt_img = load_image_as_grayscale(gt_path, config.quality_scale, config.resize_method)
    if gt_img.shape != img1.shape[:2]:
        raise ValueError("GT画像のサイズが入力画像と一致しません")
    gt_mask = gt_img <= config.gt_black_threshold

    roi_slices = get_roi_slice(gt_mask.shape, config.roi_height_ratio)

    diff_mask = compute_difference_mask(img1, img2, config.threshold)
    morph_mask = apply_morphology(diff_mask, build_operations(config.operations), kernel_size, config.iterations)

    precision, recall, accuracy = calculate_metrics_roi(morph_mask, gt_mask, roi_slices)
    fpr = calculate_fpr_full(morph_mask, gt_mask)

    pair_name = f"{image1.stem}-{image2.stem}"
    output_dir = Path(config.output_dir) / f"K{kernel_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = build_filename(config, pair_name, kernel_size, precision, recall, accuracy, fpr)
    output_path = output_dir / filename

    save_evaluated_map(
        detected=morph_mask,
        roi_rect=roi_slices[2],
        scores=(precision, recall, accuracy, fpr),
        output_path=output_path,
        quality_scale=config.quality_scale,
    )

    print("================ Morphology Evaluation ================")
    print(f"Image 1 : {image1}")
    print(f"Image 2 : {image2}")
    print(f"GT Path : {gt_path}")
    print(f"Operations : {config.operations or 'none'} | Kernel : {kernel_size} | Iterations : {config.iterations}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Noise (FPR) : {fpr:.4f}")
    print(f"Output Image : {output_path}")


def main():
    configs = build_task_configs()
    if not configs:
        print("EVALUATION_TASKS に有効な設定がありません。ファイル内のリストを編集してください。")
        return

    for config in configs:
        missing = [path for path in (config.image1, config.image2) if not path.exists()]
        if missing:
            print(f"[SKIP] 入力画像が見つかりません: {', '.join(str(p) for p in missing)}")
            continue
        if not config.gt.exists():
            print(f"[SKIP] GT画像が見つかりません: {config.gt}")
            continue

        for kernel_size in resolve_kernel_size_list(config):
            evaluate_pair(config, kernel_size)


if __name__ == "__main__":
    main()
