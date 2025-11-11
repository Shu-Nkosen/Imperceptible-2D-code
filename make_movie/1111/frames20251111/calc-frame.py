import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# ======= ユーザー設定 ==========
# 比較する2つの画像のパス
IMAGE1_PATH = "frame_00030.png"  # 1フレーム目
IMAGE2_PATH = "frame_00029.png"  # 2フレーム目

# 閾値設定（0-1の範囲で指定）
THRESHOLD = 4/255  # 1/255以上の変化を検出

# 画質設定（0.0〜1.0の範囲）
QUALITY_SCALE = 1  # 解像度を縮小

# リサイズ方法の選択
RESIZE_METHOD = "nearest"

# 出力ディレクトリ
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

def save_results(img1, img2, increased, decreased, unchanged, diff, output_dir):
    """3つの結果画像のみを保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Combined Classification Map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    combined = np.zeros((*increased.shape, 3))
    combined[increased] = [1, 0, 0]  # 赤：増加
    combined[decreased] = [0, 0, 1]  # 青：減少
    combined[unchanged] = [0.5, 0.5, 0.5]  # グレー：変化なし
    
    ax.imshow(combined)
    ax.set_title(f'Combined Classification Map (Scale: {QUALITY_SCALE})\nRed: Increased | Blue: Decreased | Gray: Unchanged')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'combined_map_scale{QUALITY_SCALE}.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Combined mapを保存")
    plt.close()
    
    # 2. Diff Abs Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    diff_abs = np.abs(diff)
    im = ax.imshow(diff_abs, cmap='hot', vmin=0, vmax=0.05)
    ax.set_title(f'Absolute Luminance Difference (Scale: {QUALITY_SCALE})\nHot spots may indicate embedded patterns')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'diff_abs_heatmap_scale{QUALITY_SCALE}.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Diff abs heatmapを保存")
    plt.close()
    
    # 3. Luminance Analysis (3x2グリッド)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Frame Luminance Difference Analysis (Scale: {QUALITY_SCALE})', fontsize=16)
    
    # Frame 1
    axes[0, 0].imshow(img1, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Frame 1')
    axes[0, 0].axis('off')
    
    # Frame 2
    axes[0, 1].imshow(img2, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Frame 2')
    axes[0, 1].axis('off')
    
    # Difference
    im_diff = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('Luminance Difference\n(Frame2 - Frame1)')
    axes[0, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Increased
    increased_vis = np.zeros((*increased.shape, 3))
    increased_vis[increased] = [1, 0, 0]
    axes[1, 0].imshow(increased_vis)
    axes[1, 0].set_title(f'Increased (>{THRESHOLD:.4f})\nPixels: {np.sum(increased):,}')
    axes[1, 0].axis('off')
    
    # Decreased
    decreased_vis = np.zeros((*decreased.shape, 3))
    decreased_vis[decreased] = [0, 0, 1]
    axes[1, 1].imshow(decreased_vis)
    axes[1, 1].set_title(f'Decreased (<-{THRESHOLD:.4f})\nPixels: {np.sum(decreased):,}')
    axes[1, 1].axis('off')
    
    # Unchanged
    unchanged_vis = np.zeros((*unchanged.shape, 3))
    unchanged_vis[unchanged] = [0.5, 0.5, 0.5]
    axes[1, 2].imshow(unchanged_vis)
    axes[1, 2].set_title(f'Unchanged (±{THRESHOLD:.4f})\nPixels: {np.sum(unchanged):,}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'luminance_analysis_scale{QUALITY_SCALE}.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Luminance analysisを保存")
    plt.close()

def main():
    # カレントディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 画像パスを構築
    img1_path = os.path.join(current_dir, IMAGE1_PATH)
    img2_path = os.path.join(current_dir, IMAGE2_PATH)
    
    print(f"画像を読み込んでいます...")
    
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"画像が見つかりません: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"画像が見つかりません: {img2_path}")
    
    # 画像読み込み
    img1 = load_image_as_grayscale(img1_path, scale=QUALITY_SCALE, method=RESIZE_METHOD)
    img2 = load_image_as_grayscale(img2_path, scale=QUALITY_SCALE, method=RESIZE_METHOD)
    
    if img1.shape != img2.shape:
        raise ValueError(f"画像のサイズが一致しません: {img1.shape} vs {img2.shape}")
    
    print("✓ 画像読み込み完了\n")
    
    # 輝度変化を分類
    print("輝度変化を分析しています...")
    increased, decreased, unchanged, diff = classify_luminance_change(img1, img2, THRESHOLD)
    print("✓ 分析完了\n")
    
    # 結果を保存
    output_dir = os.path.join(current_dir, OUTPUT_DIR)
    print("結果を保存しています...")
    save_results(img1, img2, increased, decreased, unchanged, diff, output_dir)
    
    print("\n処理が完了しました！")
    print(f"出力先: {output_dir}")

if __name__ == "__main__":
    main()