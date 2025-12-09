import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import Tuple, Union

# ======= ユーザー設定 ==========
IMAGE1_PATH = Path("frame_00069.png")
IMAGE2_PATH = Path("frame_00070.png")

THRESHOLD = 2 / 255
QUALITY_SCALE = 1.0
RESIZE_METHOD = "nearest"  # "nearest" | "bilinear" | "bicubic"

COLOR_INCREASED: Union[str, Tuple[float, float, float]] = "#ffffff"
COLOR_DECREASED: Union[str, Tuple[float, float, float]] = "#000000"
COLOR_UNCHANGED: Union[str, Tuple[float, float, float]] = "#ffffff"

OUTPUT_PATH = Path("1frame_diff_HSV.png")
# ==============================


def parse_color(color: Union[str, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if isinstance(color, tuple) and len(color) == 3:
        return tuple(float(c) for c in color)
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError("カラーコードは #RRGGBB 形式で指定してください。")
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return (r, g, b)


def load_image_v_channel(image_path: Path, scale=1.0, method="bilinear"):
    img = Image.open(image_path)
    if img.mode in ("RGB", "RGBA"):
        img_rgb = img.convert("RGB")
    else:
        img_rgb = img.convert("RGB")

    if scale != 1.0:
        width = int(img_rgb.width * scale)
        height = int(img_rgb.height * scale)
        resample = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
        }.get(method, Image.BILINEAR)
        img_rgb = img_rgb.resize((width, height), resample)

    rgb = np.array(img_rgb, dtype=np.float32) / 255.0  # shape (H,W,3)
    v_channel = np.max(rgb, axis=2)  # HSVのV成分
    return v_channel

def classify_luminance_change(img1: np.ndarray, img2: np.ndarray, threshold: float):
    diff = img2 - img1
    increased = diff > threshold
    decreased = diff < -threshold
    unchanged = np.abs(diff) <= threshold
    return increased, decreased, unchanged, diff


def save_combined_map(increased, decreased, unchanged, colors, output_path: Path):
    combined = np.zeros((*increased.shape, 3), dtype=np.float32)
    combined[increased] = colors["increased"]
    combined[decreased] = colors["decreased"]
    combined[unchanged] = colors["unchanged"]

    plt.figure(figsize=(10, 8))
    plt.imshow(combined)
    plt.title("Combined Classification Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"結果を保存しました: {output_path}")


def main():
    if not IMAGE1_PATH.exists() or not IMAGE2_PATH.exists():
        raise FileNotFoundError("指定した画像ファイルが見つかりません。")

    img1 = load_image_v_channel(IMAGE1_PATH, QUALITY_SCALE, RESIZE_METHOD)
    img2 = load_image_v_channel(IMAGE2_PATH, QUALITY_SCALE, RESIZE_METHOD)

    if img1.shape != img2.shape:
        raise ValueError(f"画像サイズが一致しません: {IMAGE1_PATH.name} {img1.shape} vs {IMAGE2_PATH.name} {img2.shape}")

    increased, decreased, unchanged, _ = classify_luminance_change(img1, img2, THRESHOLD)

    colors = {
        "increased": parse_color(COLOR_INCREASED),
        "decreased": parse_color(COLOR_DECREASED),
        "unchanged": parse_color(COLOR_UNCHANGED),
    }

    save_combined_map(increased, decreased, unchanged, colors, OUTPUT_PATH)


if __name__ == "__main__":
    main()