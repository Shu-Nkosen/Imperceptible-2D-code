from itertools import combinations
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ======= ユーザー設定 ==========
# 比較する2つの画像のパス
IMAGE1_PATH = "DSC_0003_BURST20251111151450102.JPG"  # 1フレーム目
IMAGE2_PATH = "DSC_0004_BURST20251111151450102.JPG"  # 2フレーム目

IMAGE_PATTERN = "DSC_????_BURST20251111151450102.JPG"  # 対象ファイルパターン
THRESHOLD = 4/255
QUALITY_SCALE = 1
RESIZE_METHOD = "nearest"
OUTPUT_DIR = "frame_diff_analysis"
# ==============================

def load_image_as_grayscale(image_path, scale=1.0, method="bilinear"):
    """画像をグレースケールで読み込み、0-1の範囲に正規化"""
    img = Image.open(image_path)
    
    # グレースケールに変換
    if img.mode == 'RGB' or img.mode == 'RGBA':
        img_gray = img.convert('L')
    else:
        img_gray = img
    
    # 解像度を変更
    if scale != 1.0:
        new_width = int(img_gray.width * scale)
        new_height = int(img_gray.height * scale)
        
        # リサイズ方法を選択
        if method == "nearest":
            resample = Image.NEAREST
        elif method == "bilinear":
            resample = Image.BILINEAR
        elif method == "bicubic":
            resample = Image.BICUBIC
        else:
            resample = Image.BILINEAR
        
        img_gray = img_gray.resize((new_width, new_height), resample)
        print(f"  解像度変更: {img.width}x{img.height} -> {new_width}x{new_height}")
    
    # numpy配列に変換し、0-1の範囲に正規化
    img_array = np.array(img_gray, dtype=np.float32) / 255.0
    
    return img_array

def classify_luminance_change(img1, img2, threshold):
    """2つの画像間の輝度変化を分類"""
    # 輝度差を計算（2フレーム目 - 1フレーム目）
    diff = img2 - img1
    
    # 各領域を分類
    increased = diff > threshold
    decreased = diff < -threshold
    unchanged = np.abs(diff) <= threshold
    
    return increased, decreased, unchanged, diff

def save_combined_map(increased, decreased, unchanged, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    combined = np.zeros((*increased.shape, 3))
    combined[increased] = [1, 0, 0]
    combined[decreased] = [0, 0, 1]
    combined[unchanged] = [0.5, 0.5, 0.5]

    ax.imshow(combined)
    ax.set_title(f'Combined Classification Map (Scale: {QUALITY_SCALE})\nRed: Increased | Blue: Decreased | Gray: Unchanged')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    

def main():
    image_paths = sorted(Path.cwd().glob(IMAGE_PATTERN))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"パターン {IMAGE_PATTERN} に一致する画像が見つかりません。")
    
    print(f"{len(image_paths)}枚の画像を検出しました。全組合せ {len(image_paths)*(len(image_paths)-1)//2} 通りを処理します。")

    output_dir = Path.cwd() / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    cache = {}

    def get_image(path):
        if path not in cache:
            cache[path] = load_image_as_grayscale(str(path), scale=QUALITY_SCALE, method=RESIZE_METHOD)
        return cache[path]

    pair_count = 0
    for path1, path2 in combinations(image_paths, 2):
        img1 = get_image(path1)
        img2 = get_image(path2)

        if img1.shape != img2.shape:
            raise ValueError(f"画像サイズが一致しません: {path1.name} {img1.shape} vs {path2.name} {img2.shape}")

        increased, decreased, unchanged, _ = classify_luminance_change(img1, img2, THRESHOLD)

        output_path = output_dir / f"{path1.stem}__{path2.stem}_combined_scale{QUALITY_SCALE}.png"
        save_combined_map(increased, decreased, unchanged, output_path)

        pair_count += 1
        if pair_count % 20 == 0:
            print(f"{pair_count} / {(len(image_paths)*(len(image_paths)-1))//2} 組を出力済み。")

    print(f"\n処理が完了しました。出力先: {output_dir}")


if __name__ == "__main__":
    main()