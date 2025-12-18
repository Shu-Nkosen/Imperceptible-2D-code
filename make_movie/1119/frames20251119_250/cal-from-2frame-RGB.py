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

OUTPUT_DIR = Path("rgb_max_diff_maps")
OUTPUT_FILENAME_TEMPLATE = "{name1}-{name2}_rgbmax_scale{scale}.png"
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

def load_image_as_rgb(image_path: Path, scale=1.0, method="bilinear"):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if scale != 1.0:
        width = int(img.width * scale)
        height = int(img.height * scale)
        resample = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
        }.get(method, Image.BILINEAR)
        img = img.resize((width, height), resample)

    array = np.array(img, dtype=np.float32) / 255.0
    return array

def compute_max_channel_difference(img1: np.ndarray, img2: np.ndarray):
    diff_all = img2 - img1
    abs_diff = np.abs(diff_all)
    max_indices = np.argmax(abs_diff, axis=2)
    diff_map = np.take_along_axis(diff_all, max_indices[..., None], axis=2).squeeze(-1)
    return diff_map

def classify_luminance_change(img1: np.ndarray, img2: np.ndarray, threshold: float):
    diff_map = compute_max_channel_difference(img1, img2)
    increased = diff_map > threshold
    decreased = diff_map < -threshold
    unchanged = ~ (increased | decreased)
    return increased, decreased, unchanged, diff_map

def save_combined_map(increased, decreased, unchanged, colors, output_path: Path):
    combined = np.zeros((*increased.shape, 3), dtype=np.float32)
    combined[increased] = colors["increased"]
    combined[decreased] = colors["decreased"]
    combined[unchanged] = colors["unchanged"]

    plt.figure(figsize=(10, 8))
    plt.imshow(combined)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {output_path.name}")

def main():

    img1 = load_image_as_rgb(IMAGE1_PATH, QUALITY_SCALE, RESIZE_METHOD)
    img2 = load_image_as_rgb(IMAGE2_PATH, QUALITY_SCALE, RESIZE_METHOD)

    if img1.shape != img2.shape:
        raise ValueError(f"画像サイズが一致しません: {IMAGE1_PATH.name} {img1.shape} vs {IMAGE2_PATH.name} {img2.shape}")

    colors = {
        "increased": parse_color(COLOR_INCREASED),
        "decreased": parse_color(COLOR_DECREASED),
        "unchanged": parse_color(COLOR_UNCHANGED),
    }

    increased, decreased, unchanged, _ = classify_luminance_change(img1, img2, THRESHOLD)

    filename = OUTPUT_FILENAME_TEMPLATE.format(
        name1=IMAGE1_PATH.stem,
        name2=IMAGE2_PATH.stem,
        scale=QUALITY_SCALE,
    )
    output_path = Path(filename)
    save_combined_map(increased, decreased, unchanged, colors, output_path)

    print(f"出力先: {output_path.resolve()}")


if __name__ == "__main__":
    main()