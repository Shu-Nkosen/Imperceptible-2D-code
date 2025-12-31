from itertools import combinations
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import re

# ======= ユーザー設定 (輝度版 + 全体ノイズ評価) ==========
IMAGE_PATTERN = "frame_?????.png"
GT_IMAGE_PATH = "frames_QR.png"

# 変化検出パラメータ (輝度用)
THRESHOLD = 1/255        # 変化判定のしきい値 (1/255 = 約0.4%)
QUALITY_SCALE = 1        
RESIZE_METHOD = "nearest"

# 正解データ作成用 (輝度20%以下を黒とする)
GT_BLACK_THRESHOLD = 0.2

# ROI設定 (Recall, Precision用)
ROI_HEIGHT_RATIO = 0.7   
ROI_SHAPE = "square"

OUTPUT_DIR = "frame_eval_analysis_gray_v2"
MAX_OFFSET = 1
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

def load_image_as_grayscale(image_path, scale=1.0, method="bilinear"):
    """画像をグレースケールで読み込み、0-1の範囲に正規化"""
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    
    if scale != 1.0:
        width, height = int(img.width * scale), int(img.height * scale)
        resample = getattr(Image, method.upper(), Image.BILINEAR)
        img = img.resize((width, height), resample)
    
    return np.array(img, dtype=np.float32) / 255.0

def classify_luminance_change(img1, img2, threshold):
    """輝度差分に基づいて変化を分類"""
    diff = img2 - img1
    increased = diff > threshold
    decreased = diff < -threshold
    unchanged = np.abs(diff) <= threshold
    return increased, decreased, unchanged

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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall

def calculate_fpr_full(detected_mask, gt_mask):
    """画像全体での誤検出率 (FPR / Noise Rate) を計算"""
    # 背景領域（正解が黒じゃない場所すべて）
    bg_mask = ~gt_mask
    
    # 背景なのに検出してしまった画素 (False Positive)
    fp_full = np.count_nonzero(detected_mask & bg_mask)
    
    # 背景の総画素数
    tn_plus_fp = np.count_nonzero(bg_mask)
    
    # FPR: 背景全体のうち、何％を誤って黒と判定したか
    fpr = fp_full / tn_plus_fp if tn_plus_fp > 0 else 0.0
    return fpr

def save_evaluated_map(increased, decreased, unchanged, roi_rect, scores, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 可視化: 変化があった場所を黒、なしを白
    combined = np.ones((*increased.shape, 3))
    detected = increased | decreased
    combined[detected] = [0, 0, 0] # Black
    
    ax.imshow(combined)
    
    rx, ry, rw, rh = roi_rect
    rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    precision, recall, fpr = scores
    title_text = (f'Luminance Analysis (Scale: {QUALITY_SCALE})\n'
                  f'ROI Recall: {recall:.2%} | ROI Precision: {precision:.2%}\n'
                  f'Full Image Noise Rate (FPR): {fpr:.2%}')
    
    ax.set_title(title_text)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    image_paths = sorted(Path.cwd().glob(IMAGE_PATTERN))
    if not image_paths: raise FileNotFoundError(f"{IMAGE_PATTERN} が見つかりません")
    
    gt_path = Path.cwd() / GT_IMAGE_PATH
    if not gt_path.exists(): raise FileNotFoundError(f"{GT_IMAGE_PATH} が見つかりません")
    
    print(f"GT読み込み(Gray): {GT_IMAGE_PATH}")
    gt_img = load_image_as_grayscale(str(gt_path), scale=QUALITY_SCALE, method=RESIZE_METHOD)
    gt_mask = gt_img <= GT_BLACK_THRESHOLD
    
    roi_slices = get_roi_slice(gt_mask.shape, ROI_HEIGHT_RATIO)
    print(f"ROI設定: 中心から縦{ROI_HEIGHT_RATIO*100}%の正方形領域")

    indexed_paths = sorted(
        image_paths,
        key=lambda p: (
            (idx := extract_frame_index(p)) is None,
            idx if idx is not None else float('inf'),
            p.stem,
        ),
    )
    candidate_pairs = []
    for i, path1 in enumerate(indexed_paths):
        idx1 = extract_frame_index(path1)
        if idx1 is None: continue
        for j in range(i+1, len(indexed_paths)):
            path2 = indexed_paths[j]
            idx2 = extract_frame_index(path2)
            if idx2 is None or idx2 - idx1 > MAX_OFFSET: break
            candidate_pairs.append((path1, path2))

    output_dir = Path.cwd() / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    cache = {}
    def get_image(path):
        if path not in cache: cache[path] = load_image_as_grayscale(str(path), scale=QUALITY_SCALE, method=RESIZE_METHOD)
        return cache[path]

    print(f"{len(candidate_pairs)}ペアを輝度モードで処理開始...")
    for i, (path1, path2) in enumerate(candidate_pairs):
        img1 = get_image(path1)
        img2 = get_image(path2)
        if img1.shape != gt_mask.shape: continue

        # 輝度変化検出
        increased, decreased, unchanged = classify_luminance_change(img1, img2, THRESHOLD)
        detected_mask = increased | decreased
        
        # 指標計算
        precision, recall = calculate_metrics_roi(detected_mask, gt_mask, roi_slices)
        fpr = calculate_fpr_full(detected_mask, gt_mask)

        pair_name = build_pair_folder_name(path1, path2)
        filename = f"{pair_name}_Gray_R{int(recall*100)}_P{int(precision*100)}_N{int(fpr*100)}.png"
        output_path = output_dir / filename
        
        save_evaluated_map(increased, decreased, unchanged, roi_slices[2], (precision, recall, fpr), output_path)
        print(f"[{i+1}/{len(candidate_pairs)}] {pair_name} -> Recall: {recall:.1%}, Precision: {precision:.1%}, Noise(FPR): {fpr:.2%}")

    print(f"\n完了: {output_dir}")

if __name__ == "__main__":
    main()