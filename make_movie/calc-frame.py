import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# ======= ユーザー設定 ==========
# 比較する2つの画像のパス
IMAGE1_PATH = "ex.png"  # 1フレーム目
IMAGE2_PATH = "ex_b2_d2_normal.png"  # 2フレーム目

# 閾値設定（0-1の範囲で指定）
THRESHOLD = 1/255  # 1/255以上の変化を検出

# 出力ディレクトリ
OUTPUT_DIR = "frame_diff_analysis"
# ==============================

def load_image_as_grayscale(image_path):
    """画像をグレースケールで読み込み、0-1の範囲に正規化"""
    img = Image.open(image_path)
    
    # グレースケールに変換
    if img.mode == 'RGB' or img.mode == 'RGBA':
        img_gray = img.convert('L')
    else:
        img_gray = img
    
    # numpy配列に変換し、0-1の範囲に正規化
    img_array = np.array(img_gray, dtype=np.float32) / 255.0
    
    return img_array

def classify_luminance_change(img1, img2, threshold):
    """
    2つの画像間の輝度変化を分類
    
    Returns:
        increased: 輝度が閾値以上増加した領域（True/False）
        decreased: 輝度が閾値以上減少した領域（True/False）
        unchanged: 変化が閾値以内の領域（True/False）
        diff: 輝度差（img2 - img1）
    """
    # 輝度差を計算（2フレーム目 - 1フレーム目）
    diff = img2 - img1
    
    # 各領域を分類
    increased = diff > threshold  # 閾値以上増加
    decreased = diff < -threshold  # 閾値以上減少
    unchanged = np.abs(diff) <= threshold  # 閾値以内の変化
    
    return increased, decreased, unchanged, diff

def visualize_results(img1, img2, increased, decreased, unchanged, diff, output_dir):
    """結果を可視化して保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 図のサイズを設定
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Frame Luminance Difference Analysis', fontsize=16)
    
    # 1. 元画像1
    axes[0, 0].imshow(img1, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Frame 1')
    axes[0, 0].axis('off')
    
    # 2. 元画像2
    axes[0, 1].imshow(img2, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Frame 2')
    axes[0, 1].axis('off')
    
    # 3. 輝度差（カラーマップで表示）
    im_diff = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('Luminance Difference\n(Frame2 - Frame1)')
    axes[0, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. 増加領域（赤）
    increased_vis = np.zeros((*increased.shape, 3))
    increased_vis[increased] = [1, 0, 0]  # 赤
    axes[1, 0].imshow(increased_vis)
    axes[1, 0].set_title(f'Increased (>{THRESHOLD:.4f})\nPixels: {np.sum(increased):,}')
    axes[1, 0].axis('off')
    
    # 5. 減少領域（青）
    decreased_vis = np.zeros((*decreased.shape, 3))
    decreased_vis[decreased] = [0, 0, 1]  # 青
    axes[1, 1].imshow(decreased_vis)
    axes[1, 1].set_title(f'Decreased (<-{THRESHOLD:.4f})\nPixels: {np.sum(decreased):,}')
    axes[1, 1].axis('off')
    
    # 6. 変化なし領域（緑）
    unchanged_vis = np.zeros((*unchanged.shape, 3))
    unchanged_vis[unchanged] = [0, 1, 0]  # 緑
    axes[1, 2].imshow(unchanged_vis)
    axes[1, 2].set_title(f'Unchanged (±{THRESHOLD:.4f})\nPixels: {np.sum(unchanged):,}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'luminance_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"可視化結果を保存: {os.path.join(output_dir, 'luminance_analysis.png')}")
    plt.close()
    
    # 統合マップ（全ての領域を1つの画像に）
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    combined = np.zeros((*increased.shape, 3))
    combined[increased] = [1, 0, 0]  # 赤：増加
    combined[decreased] = [0, 0, 1]  # 青：減少
    combined[unchanged] = [0.5, 0.5, 0.5]  # グレー：変化なし
    
    ax.imshow(combined)
    ax.set_title('Combined Classification Map\nRed: Increased | Blue: Decreased | Gray: Unchanged')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_map.png'), dpi=150, bbox_inches='tight')
    print(f"統合マップを保存: {os.path.join(output_dir, 'combined_map.png')}")
    plt.close()

def save_masks(increased, decreased, unchanged, output_dir):
    """各領域のマスクを画像として保存"""
    # 増加領域
    increased_img = Image.fromarray((increased * 255).astype(np.uint8))
    increased_img.save(os.path.join(output_dir, 'mask_increased.png'))
    
    # 減少領域
    decreased_img = Image.fromarray((decreased * 255).astype(np.uint8))
    decreased_img.save(os.path.join(output_dir, 'mask_decreased.png'))
    
    # 変化なし領域
    unchanged_img = Image.fromarray((unchanged * 255).astype(np.uint8))
    unchanged_img.save(os.path.join(output_dir, 'mask_unchanged.png'))
    
    print(f"マスク画像を保存: {output_dir}")

def print_statistics(img1, img2, increased, decreased, unchanged, diff):
    """統計情報を出力"""
    total_pixels = img1.size
    
    print("\n" + "="*60)
    print("輝度変化分析結果")
    print("="*60)
    print(f"画像サイズ: {img1.shape[1]} x {img1.shape[0]}")
    print(f"総ピクセル数: {total_pixels:,}")
    print(f"閾値: ±{THRESHOLD:.6f} ({THRESHOLD * 255:.2f}/255)")
    print("-"*60)
    
    # 増加領域
    increased_count = np.sum(increased)
    increased_ratio = increased_count / total_pixels * 100
    print(f"輝度増加領域: {increased_count:,} pixels ({increased_ratio:.2f}%)")
    if increased_count > 0:
        print(f"  最大増加量: {np.max(diff[increased]):.6f} ({np.max(diff[increased]) * 255:.2f}/255)")
        print(f"  平均増加量: {np.mean(diff[increased]):.6f} ({np.mean(diff[increased]) * 255:.2f}/255)")
    
    # 減少領域
    decreased_count = np.sum(decreased)
    decreased_ratio = decreased_count / total_pixels * 100
    print(f"輝度減少領域: {decreased_count:,} pixels ({decreased_ratio:.2f}%)")
    if decreased_count > 0:
        print(f"  最大減少量: {np.min(diff[decreased]):.6f} ({np.min(diff[decreased]) * 255:.2f}/255)")
        print(f"  平均減少量: {np.mean(diff[decreased]):.6f} ({np.mean(diff[decreased]) * 255:.2f}/255)")
    
    # 変化なし領域
    unchanged_count = np.sum(unchanged)
    unchanged_ratio = unchanged_count / total_pixels * 100
    print(f"変化なし領域: {unchanged_count:,} pixels ({unchanged_ratio:.2f}%)")
    
    # 全体の統計
    print("-"*60)
    print(f"輝度差の範囲: [{np.min(diff):.6f}, {np.max(diff):.6f}]")
    print(f"輝度差の平均: {np.mean(diff):.6f} ({np.mean(diff) * 255:.2f}/255)")
    print(f"輝度差の標準偏差: {np.std(diff):.6f} ({np.std(diff) * 255:.2f}/255)")
    print("="*60 + "\n")

def main():
    # カレントディレクトリを基準にパスを解決
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 画像を読み込み
    print(f"画像を読み込んでいます...")
    print(f"  Frame 1: {IMAGE1_PATH}")
    print(f"  Frame 2: {IMAGE2_PATH}")
    
    # パスが相対パスの場合、1つ上のディレクトリを探す
    img1_path = IMAGE1_PATH if os.path.isabs(IMAGE1_PATH) else os.path.join(os.path.dirname(current_dir), "1022", IMAGE1_PATH)
    img2_path = IMAGE2_PATH if os.path.isabs(IMAGE2_PATH) else os.path.join(os.path.dirname(current_dir), "1022", IMAGE2_PATH)
    
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"画像が見つかりません: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"画像が見つかりません: {img2_path}")
    
    img1 = load_image_as_grayscale(img1_path)
    img2 = load_image_as_grayscale(img2_path)
    
    # サイズチェック
    if img1.shape != img2.shape:
        raise ValueError(f"画像のサイズが一致しません: {img1.shape} vs {img2.shape}")
    
    print("画像読み込み完了\n")
    
    # 輝度変化を分類
    print("輝度変化を分析しています...")
    increased, decreased, unchanged, diff = classify_luminance_change(img1, img2, THRESHOLD)
    
    # 統計情報を出力
    print_statistics(img1, img2, increased, decreased, unchanged, diff)
    
    # 結果を可視化
    output_dir = os.path.join(current_dir, OUTPUT_DIR)
    print("結果を可視化しています...")
    visualize_results(img1, img2, increased, decreased, unchanged, diff, output_dir)
    
    # マスクを保存
    save_masks(increased, decreased, unchanged, output_dir)
    
    print("\n処理が完了しました！")

if __name__ == "__main__":
    main()