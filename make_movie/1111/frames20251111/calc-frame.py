import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# ======= ユーザー設定 ==========
THRESHOLD = 4 / 255  # 1/255以上の変化を検出
QUALITY_SCALE = 1  # 解像度を縮小
RESIZE_METHOD = "nearest"
OUTPUT_DIR = "frame_diff_analysis"
MAX_OFFSET = 5  # 前後に比較するフレーム数
# ==============================

_image_cache = {}


def load_image_as_grayscale(image_path, scale=1.0, method="bilinear"):
    """画像をグレースケールで読み込み、0-1の範囲に正規化"""
    img = Image.open(image_path)

    # グレースケールに変換
    if img.mode == "RGB" or img.mode == "RGBA":
        img_gray = img.convert("L")
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


def get_image(image_path):
    """キャッシュを用いて画像を取得"""
    if image_path not in _image_cache:
        _image_cache[image_path] = load_image_as_grayscale(
            image_path, scale=QUALITY_SCALE, method=RESIZE_METHOD
        )
    return _image_cache[image_path]


def classify_luminance_change(img1, img2, threshold):
    """2つの画像間の輝度変化を分類"""
    diff = img2 - img1

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
    ax.set_title(
        f"Combined Classification Map (Scale: {QUALITY_SCALE})\nRed: Increased | Blue: Decreased | Gray: Unchanged"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"combined_map_scale{QUALITY_SCALE}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Diff Abs Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    diff_abs = np.abs(diff)
    im = ax.imshow(diff_abs, cmap="hot", vmin=0, vmax=0.05)
    ax.set_title(
        f"Absolute Luminance Difference (Scale: {QUALITY_SCALE})\nHot spots may indicate embedded patterns"
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"diff_abs_heatmap_scale{QUALITY_SCALE}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Luminance Analysis (3x2グリッド)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Frame Luminance Difference Analysis (Scale: {QUALITY_SCALE})", fontsize=16
    )

    axes[0, 0].imshow(img1, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Frame 1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Frame 2")
    axes[0, 1].axis("off")

    im_diff = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title("Luminance Difference\n(Frame2 - Frame1)")
    axes[0, 2].axis("off")
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    increased_vis = np.zeros((*increased.shape, 3))
    increased_vis[increased] = [1, 0, 0]
    axes[1, 0].imshow(increased_vis)
    axes[1, 0].set_title(f"Increased (>{THRESHOLD:.4f})\nPixels: {np.sum(increased):,}")
    axes[1, 0].axis("off")

    decreased_vis = np.zeros((*decreased.shape, 3))
    decreased_vis[decreased] = [0, 0, 1]
    axes[1, 1].imshow(decreased_vis)
    axes[1, 1].set_title(
        f"Decreased (<-{THRESHOLD:.4f})\nPixels: {np.sum(decreased):,}"
    )
    axes[1, 1].axis("off")

    unchanged_vis = np.zeros((*unchanged.shape, 3))
    unchanged_vis[unchanged] = [0.5, 0.5, 0.5]
    axes[1, 2].imshow(unchanged_vis)
    axes[1, 2].set_title(
        f"Unchanged (±{THRESHOLD:.4f})\nPixels: {np.sum(unchanged):,}"
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"luminance_analysis_scale{QUALITY_SCALE}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def collect_frame_files(base_dir):
    """frame_#####.png を収集して辞書で返却"""
    frame_paths = {}
    for filename in os.listdir(base_dir):
        if filename.lower().startswith("frame_") and filename.lower().endswith(".png"):
            stem, _ = os.path.splitext(filename)
            parts = stem.split("_")
            if not parts:
                continue
            index_str = parts[-1]
            if not index_str.isdigit():
                continue
            idx = int(index_str)
            frame_paths[idx] = os.path.join(base_dir, filename)
    return frame_paths


def process_frame_pair(frame1_idx, frame2_idx, frame_paths, anchor_dir, label):
    """フレームペアを解析し結果を保存"""
    os.makedirs(anchor_dir, exist_ok=True)
    pair_output_dir = os.path.join(anchor_dir, label)

    img1 = get_image(frame_paths[frame1_idx])
    img2 = get_image(frame_paths[frame2_idx])

    if img1.shape != img2.shape:
        raise ValueError(
            f"画像のサイズが一致しません: frame_{frame1_idx:05d} vs frame_{frame2_idx:05d}"
        )

    increased, decreased, unchanged, diff = classify_luminance_change(
        img1, img2, THRESHOLD
    )
    save_results(img1, img2, increased, decreased, unchanged, diff, pair_output_dir)
    print(f"    {label}: frame_{frame1_idx:05d} vs frame_{frame2_idx:05d}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frame_paths = collect_frame_files(current_dir)

    if not frame_paths:
        raise FileNotFoundError("frame_#####.png が見つかりません。")

    sorted_indices = sorted(frame_paths.keys())
    print(f"{len(sorted_indices)}枚のフレームを検出しました。")

    for idx in sorted_indices:
        anchor_dir = os.path.join(current_dir, OUTPUT_DIR, f"frame_{idx:05d}")
        pair_count = 0

        for offset in range(1, MAX_OFFSET + 1):
            prev_idx = idx - offset
            if prev_idx in frame_paths:
                process_frame_pair(prev_idx, idx, frame_paths, anchor_dir, f"prev_{offset:02d}")
                pair_count += 1

            next_idx = idx + offset
            if next_idx in frame_paths:
                process_frame_pair(idx, next_idx, frame_paths, anchor_dir, f"next_{offset:02d}")
                pair_count += 1

        if pair_count > 0:
            print(f"frame_{idx:05d}: {pair_count}ペアを保存")

    print("\n処理が完了しました。")
    print(f"出力先: {os.path.join(current_dir, OUTPUT_DIR)}")


if __name__ == "__main__":
    main()