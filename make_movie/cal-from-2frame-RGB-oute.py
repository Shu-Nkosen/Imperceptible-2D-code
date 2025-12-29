# ...existing code...
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import Tuple, Union, List

# ======= ユーザー設定 ==========
IMAGE_PATTERN = "frame_?????.png"
THRESHOLD = 2 / 255
QUALITY_SCALE = 1.0
RESIZE_METHOD = "nearest"  # "nearest" | "bilinear" | "bicubic"

COLOR_INCREASED: Union[str, Tuple[float, float, float]] = "#ffffff"
COLOR_DECREASED: Union[str, Tuple[float, float, float]] = "#000000"
COLOR_UNCHANGED: Union[str, Tuple[float, float, float]] = "#ffffff"

OUTPUT_DIR = Path("rgb_max_diff_maps")
MAX_OFFSET = 5
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

    selected = np.take_along_axis(diff_all, max_indices[..., None], axis=2).squeeze(-1)
    sum_all = diff_all.sum(axis=2)
    other_mean = (sum_all - selected) / 2.0  # 選択チャネル以外の平均
    diff_map = selected - other_mean         # ノイズ除去後の差分

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
    print(f"  保存: {output_path.name}")

def parse_frame_info(stem: str):
    match_frame = re.match(r"frame_(\d+)", stem)
    if match_frame:
        return match_frame.group(1), "FRAME"
    match = re.match(r"DSC_(\d+)_BURST(\d+)", stem)
    if match:
        return match.group(1), f"DSC_BURST{match.group(2)}"
    return stem, "UNKNOWN"

def build_pair_folder_name(path1: Path, path2: Path):
    idx1, ts1 = parse_frame_info(path1.stem)
    idx2, ts2 = parse_frame_info(path2.stem)
    timestamp = ts1 if ts1 == ts2 else f"{ts1}-{ts2}"
    return f"{idx1}-{idx2}-{timestamp}"

def extract_frame_index(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None

def collect_candidate_pairs(image_paths: List[Path], max_offset: int):
    indexed_paths = sorted(
        image_paths,
        key=lambda p: (
            (idx := extract_frame_index(p)) is None,
            idx if idx is not None else p.stem,
        ),
    )
    candidate_pairs = []
    for i, path1 in enumerate(indexed_paths):
        idx1 = extract_frame_index(path1)
        if idx1 is None:
            continue
        for j in range(i + 1, len(indexed_paths)):
            path2 = indexed_paths[j]
            idx2 = extract_frame_index(path2)
            if idx2 is None:
                continue
            if idx2 - idx1 > max_offset:
                break
            candidate_pairs.append((path1, path2))
    return candidate_pairs

def main():
    image_paths = sorted(Path.cwd().glob(IMAGE_PATTERN))
    if not image_paths:
        raise FileNotFoundError(f"パターン {IMAGE_PATTERN} に一致する画像が見つかりません。")

    pairs = collect_candidate_pairs(image_paths, MAX_OFFSET)
    if not pairs:
        print("処理対象のペアがありません。")
        return

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    colors = {
        "increased": parse_color(COLOR_INCREASED),
        "decreased": parse_color(COLOR_DECREASED),
        "unchanged": parse_color(COLOR_UNCHANGED),
    }

    cache = {}

    def get_image(path: Path):
        if path not in cache:
            cache[path] = load_image_as_rgb(path, QUALITY_SCALE, RESIZE_METHOD)
        return cache[path]

    print(f"{len(image_paths)}枚を検出。対象ペア {len(pairs)} 組を処理します。")
    for idx, (path1, path2) in enumerate(pairs, 1):
        img1 = get_image(path1)
        img2 = get_image(path2)

        if img1.shape != img2.shape:
            raise ValueError(f"画像サイズが一致しません: {path1.name} {img1.shape} vs {path2.name} {img2.shape}")

        increased, decreased, unchanged, _ = classify_luminance_change(img1, img2, THRESHOLD)

        pair_name = build_pair_folder_name(path1, path2)
        output_path = OUTPUT_DIR / f"{pair_name}_rgbmax_scale{QUALITY_SCALE:.2f}.png"
        save_combined_map(increased, decreased, unchanged, colors, output_path)

        if idx % 20 == 0 or idx == len(pairs):
            print(f"  {idx} / {len(pairs)} 組完了")

    print(f"\n処理が完了しました。出力先: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
# ...existing code...