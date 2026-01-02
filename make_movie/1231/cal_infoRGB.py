import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

# ======= ユーザー設定 (RGB版 + 全体ノイズ評価) ==========
IMAGE_PATTERN = "frame_?????.png"
GT_IMAGE_PATH = "frames_QR.png"

# 変化検出パラメータ
THRESHOLD = 2/255        # 変化判定のしきい値
QUALITY_SCALE = 1        
RESIZE_METHOD = "nearest"

# 正解データ作成用 (輝度20%以下を黒とする)
GT_BLACK_THRESHOLD = 0.2

# ROI設定 (Recall, Precision用)
ROI_HEIGHT_RATIO = 0.7   
ROI_SHAPE = "square"

OUTPUT_DIR = "frame_eval_analysis_rgb_v2"
MAX_OFFSET = 1
CACHE_GT_SEARCH_PATH = "frames_QR"
SUMMARY_CSV = "eval_summary.csv"
DEFAULT_PREFIXES = ["ex", "nagaoka", "hocho", "rice"]
DEFAULT_TARGETS = ["B", "G", "I", "R", "X"]
DEFAULT_INDEXES = range(1, 5)
CSV_FIELDS = [
    "folder",
    "pair_count",
    "avg_recall",
    "max_recall",
    "min_recall",
    "avg_precision",
    "max_precision",
    "min_precision",
    "avg_noise",
    "max_noise",
    "min_noise",
    "best_frame_1",
    "best_frame_2",
    "best_pair_recall",
    "best_pair_noise",
    "best_pair_score",
]
# ======================================================

def parse_frame_info(stem: str):
    match_frame = re.match(r"frame_(\d+)", stem)
    if match_frame: return match_frame.group(1), "FRAME"
    match = re.match(r"DSC_(\d+)_BURST(\d+)", stem)
    if match: return match.group(1), f"DSC_BURST{match.group(2)}"
    return stem, "UNKNOWN"

def build_pair_folder_name(path1: Path, path2: Path):
    idx1, ts1 = parse_frame_info(path1.stem)
    idx2, ts2 = parse_frame_info(path2.stem)
    timestamp = ts1 if ts1 == ts2 else f"{ts1}-{ts2}"
    return f"{idx1}-{idx2}-{timestamp}"

def extract_frame_index(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None

def load_image_as_rgb(image_path, scale=1.0, method="bilinear"):
    img = Image.open(image_path).convert("RGB")
    if scale != 1.0:
        width, height = int(img.width * scale), int(img.height * scale)
        resample = getattr(Image, method.upper(), Image.BILINEAR)
        img = img.resize((width, height), resample)
    return np.array(img, dtype=np.float32) / 255.0

def load_image_as_grayscale_for_gt(image_path, scale=1.0, method="bilinear"):
    img = Image.open(image_path).convert('L')
    if scale != 1.0:
        width, height = int(img.width * scale), int(img.height * scale)
        resample = getattr(Image, method.upper(), Image.BILINEAR)
        img = img.resize((width, height), resample)
    return np.array(img, dtype=np.float32) / 255.0

def compute_max_channel_difference(img1, img2):
    diff_all = img2 - img1
    abs_diff = np.abs(diff_all)
    max_indices = np.argmax(abs_diff, axis=2)
    selected = np.take_along_axis(diff_all, max_indices[..., None], axis=2).squeeze(-1)
    other_mean = (diff_all.sum(axis=2) - selected) / 2.0
    return selected - other_mean

def classify_rgb_change(img1, img2, threshold):
    diff_map = compute_max_channel_difference(img1, img2)
    increased = diff_map > threshold
    decreased = diff_map < -threshold
    return increased, decreased

def get_roi_slice(img_shape, height_ratio=0.7):
    h, w = img_shape
    roi_h = int(h * height_ratio)
    roi_w = roi_h
    start_y = max(0, (h - roi_h) // 2)
    start_x = max(0, (w - roi_w) // 2)
    return slice(start_y, start_y + roi_h), slice(start_x, start_x + roi_w), (start_x, start_y, roi_w, roi_h)

def calculate_metrics_roi(detected_mask, gt_mask, roi_slices):
    """ROI内でのPrecisionとRecallを計算"""
    slice_y, slice_x, _ = roi_slices
    roi_detected = detected_mask[slice_y, slice_x]
    roi_gt = gt_mask[slice_y, slice_x]
    
    tp = np.count_nonzero(roi_detected & roi_gt)
    fp = np.count_nonzero(roi_detected & ~roi_gt)
    fn = np.count_nonzero(~roi_detected & roi_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall (再現率): 正解データ(TP+FN)のうち、どれだけ検出(TP)できたか
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall

def calculate_fpr_full(detected_mask, gt_mask):
    """画像全体での誤検出率 (FPR) を計算"""
    # 背景領域（正解が黒じゃない場所すべて）
    bg_mask = ~gt_mask
    
    # 背景なのに検出してしまった画素 (False Positive)
    fp_full = np.count_nonzero(detected_mask & bg_mask)
    
    # 背景の総画素数 (True Negative + False Positive)
    tn_plus_fp = np.count_nonzero(bg_mask)
    
    # FPR: 背景全体のうち、何％を誤って黒と判定したか
    fpr = fp_full / tn_plus_fp if tn_plus_fp > 0 else 0.0
    return fpr

def save_evaluated_map(increased, decreased, roi_rect, scores, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    combined = np.ones((*increased.shape, 3))
    detected = increased | decreased
    combined[detected] = [0, 0, 0]
    
    ax.imshow(combined)
    
    rx, ry, rw, rh = roi_rect
    rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    precision, recall, fpr = scores
    title_text = (f'RGB Analysis (Scale: {QUALITY_SCALE})\n'
                  f'ROI Recall: {recall:.2%} | ROI Precision: {precision:.2%}\n'
                  f'Full Image Noise Rate (FPR): {fpr:.2%}')
    
    ax.set_title(title_text)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

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

def build_candidate_pairs(image_paths: Sequence[Path]) -> List[Tuple[Path, Path]]:
    indexed_paths = sorted(
        image_paths,
        key=lambda p: (
            (idx := extract_frame_index(p)) is None,
            idx if idx is not None else float('inf'),
            p.stem,
        ),
    )
    candidate_pairs: List[Tuple[Path, Path]] = []
    for i, path1 in enumerate(indexed_paths):
        idx1 = extract_frame_index(path1)
        if idx1 is None:
            continue
        for j in range(i + 1, len(indexed_paths)):
            path2 = indexed_paths[j]
            idx2 = extract_frame_index(path2)
            if idx2 is None:
                continue
            if idx2 - idx1 > MAX_OFFSET:
                break
            candidate_pairs.append((path1, path2))
    return candidate_pairs

def summarize_pair_metrics(folder_name: str, pair_records: List[Dict[str, float]]) -> Dict[str, float]:
    recalls = np.array([record["recall"] for record in pair_records])
    precisions = np.array([record["precision"] for record in pair_records])
    noises = np.array([record["noise"] for record in pair_records])
    best_pair = max(pair_records, key=lambda record: record["recall"] - record["noise"])
    return {
        "folder": folder_name,
        "pair_count": len(pair_records),
        "avg_recall": float(recalls.mean()),
        "max_recall": float(recalls.max()),
        "min_recall": float(recalls.min()),
        "avg_precision": float(precisions.mean()),
        "max_precision": float(precisions.max()),
        "min_precision": float(precisions.min()),
        "avg_noise": float(noises.mean()),
        "max_noise": float(noises.max()),
        "min_noise": float(noises.min()),
        "best_frame_1": best_pair["frame1"],
        "best_frame_2": best_pair["frame2"],
        "best_pair_recall": best_pair["recall"],
        "best_pair_noise": best_pair["noise"],
        "best_pair_score": best_pair["recall"] - best_pair["noise"],
    }

def evaluate_directory(target_dir: Path) -> Optional[Dict[str, float]]:
    image_paths = sorted(target_dir.glob(IMAGE_PATTERN))
    if not image_paths:
        print(f"[SKIP] {target_dir.name}: {IMAGE_PATTERN} が見つかりません")
        return None

    try:
        gt_path = resolve_gt_path(target_dir)
    except FileNotFoundError as exc:
        print(f"[SKIP] {target_dir.name}: {exc}")
        return None

    print(f"GT読み込み: {gt_path.relative_to(target_dir.parent)}")
    gt_img = load_image_as_grayscale_for_gt(str(gt_path), scale=QUALITY_SCALE, method=RESIZE_METHOD)
    gt_mask = gt_img <= GT_BLACK_THRESHOLD
    roi_slices = get_roi_slice(gt_mask.shape, ROI_HEIGHT_RATIO)

    candidate_pairs = build_candidate_pairs(image_paths)
    if not candidate_pairs:
        print(f"[SKIP] {target_dir.name}: 比較可能なペアがありません")
        return None

    output_dir = target_dir / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    cache: Dict[Path, np.ndarray] = {}

    def get_image(path: Path) -> np.ndarray:
        if path not in cache:
            cache[path] = load_image_as_rgb(str(path), scale=QUALITY_SCALE, method=RESIZE_METHOD)
        return cache[path]

    pair_records: List[Dict[str, float]] = []
    print(f"{target_dir.name}: {len(candidate_pairs)}ペアを処理開始...")
    for idx, (path1, path2) in enumerate(candidate_pairs, start=1):
        img1 = get_image(path1)
        img2 = get_image(path2)
        if img1.shape[:2] != gt_mask.shape:
            continue

        increased, decreased = classify_rgb_change(img1, img2, THRESHOLD)
        detected_mask = increased | decreased
        precision, recall = calculate_metrics_roi(detected_mask, gt_mask, roi_slices)
        fpr = calculate_fpr_full(detected_mask, gt_mask)

        pair_name = build_pair_folder_name(path1, path2)
        filename = f"{pair_name}_R{int(recall*100)}_P{int(precision*100)}_N{int(fpr*100)}.png"
        output_path = output_dir / filename
        save_evaluated_map(increased, decreased, roi_slices[2], (precision, recall, fpr), output_path)

        pair_records.append({
            "frame1": path1.name,
            "frame2": path2.name,
            "recall": recall,
            "precision": precision,
            "noise": fpr,
        })
        print(f"[{idx}/{len(candidate_pairs)}] {pair_name} -> Recall: {recall:.1%}, Precision: {precision:.1%}, Noise(FPR): {fpr:.2%}")

    summary = summarize_pair_metrics(target_dir.name, pair_records)
    print(f"完了: {target_dir.name} -> ペア数 {summary['pair_count']}")
    return summary

def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value if value is not None else ""

def write_summary_csv(records: List[Dict[str, float]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: format_value(record.get(field)) for field in CSV_FIELDS})

def parse_args():
    parser = argparse.ArgumentParser(description="RGB差分評価とCSV集計ツール")
    parser.add_argument("--dirs", nargs="*", help="解析対象ディレクトリ（指定が無い場合は既定の {prefix}_{target}{idx} 全てを探索）")
    parser.add_argument("--csv", default=SUMMARY_CSV, help="CSV出力ファイル名")
    return parser.parse_args()

def collect_target_directories(base_dir: Path, dir_names: Optional[Sequence[str]]) -> List[Path]:
    if dir_names:
        dirs = []
        for name in dir_names:
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = base_dir / name
            if candidate.is_dir():
                dirs.append(candidate)
            else:
                print(f"[SKIP] {candidate}: ディレクトリが存在しません")
        return dirs

    dirs: List[Path] = []
    for prefix in DEFAULT_PREFIXES:
        for target in DEFAULT_TARGETS:
            for idx in DEFAULT_INDEXES:
                candidate = base_dir / f"{prefix}_{target}{idx}"
                if candidate.is_dir():
                    dirs.append(candidate)
    return dirs

def main():
    args = parse_args()
    base_dir = Path(__file__).parent
    target_dirs = collect_target_directories(base_dir, args.dirs)
    if not target_dirs:
        print("解析対象ディレクトリが見つかりませんでした。")
        return

    summaries: List[Dict[str, float]] = []
    for directory in target_dirs:
        result = evaluate_directory(directory)
        if result:
            summaries.append(result)

    if summaries:
        csv_path = base_dir / args.csv
        write_summary_csv(summaries, csv_path)
        print(f"CSV出力: {csv_path}")
    else:
        print("有効な解析結果が得られませんでした。")

if __name__ == "__main__":
    main()