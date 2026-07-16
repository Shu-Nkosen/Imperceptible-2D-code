import argparse
import re
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Union, List

# ======= ユーザー設定 ==========
IMAGE_PATTERN = "frame_?????.png"
THRESHOLD = 4 / 255
QUALITY_SCALE = 1.0
RESIZE_METHOD = "nearest"  # "nearest" | "bilinear" | "bicubic"

COLOR_INCREASED: Union[str, Tuple[float, float, float]] = "#000000"
COLOR_DECREASED: Union[str, Tuple[float, float, float]] = "#000000"
COLOR_UNCHANGED: Union[str, Tuple[float, float, float]] = "#ffffff"

OUTPUT_DIR = Path("rgb_max_diff_maps")
MAX_OFFSET = 1
# ==============================


def resolve_output_dir(base_dir: Path, output_subdir: str = "") -> Path:
    sub = Path(output_subdir) if output_subdir else OUTPUT_DIR
    if sub.is_absolute():
        return sub
    return base_dir / sub

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

def color_to_uint8_rgb(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (
        int(round(float(color[0]) * 255.0)),
        int(round(float(color[1]) * 255.0)),
        int(round(float(color[2]) * 255.0)),
    )

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


def compute_max_channel_signal(img: np.ndarray) -> np.ndarray:
    """1フレームから max-channel 強調のスカラー場を作る。"""
    mean_all = img.mean(axis=2, keepdims=True)
    abs_dev = np.abs(img - mean_all)
    max_indices = np.argmax(abs_dev, axis=2)
    selected = np.take_along_axis(img, max_indices[..., None], axis=2).squeeze(-1)
    sum_all = img.sum(axis=2)
    other_mean = (sum_all - selected) / 2.0
    return selected - other_mean

def classify_luminance_change(img1: np.ndarray, img2: np.ndarray, threshold: float):
    diff_map = compute_max_channel_difference(img1, img2)
    increased = diff_map > threshold
    decreased = diff_map < -threshold
    unchanged = ~ (increased | decreased)
    return increased, decreased, unchanged, diff_map


def classify_accumulated_abs_change(acc_map: np.ndarray, threshold: float):
    """窓合算 |diff| を二値化。変化は両方向とも黒（increased にまとめる）。"""
    changed = acc_map > threshold
    unchanged = ~changed
    increased = changed
    decreased = np.zeros_like(changed, dtype=bool)
    return increased, decreased, unchanged


def classify_stat_change(stat_map: np.ndarray, threshold: float):
    changed = stat_map > threshold
    unchanged = ~changed
    increased = changed
    decreased = np.zeros_like(changed, dtype=bool)
    return increased, decreased, unchanged


def save_combined_map(increased, decreased, unchanged, colors, output_path: Path):
    """差分マップをピクセル等倍のPNGで保存する（matplotlib余白なし）。

    以前の plt.savefig は余白・縮小が入るため、目視/スマホでは読めても
    OpenCV の QRCodeDetector が失敗しやすかった。
    """
    rgb = np.empty((*increased.shape, 3), dtype=np.uint8)
    rgb[unchanged] = color_to_uint8_rgb(colors["unchanged"])
    rgb[increased] = color_to_uint8_rgb(colors["increased"])
    rgb[decreased] = color_to_uint8_rgb(colors["decreased"])
    Image.fromarray(rgb, mode="RGB").save(output_path)
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

def sort_indexed_paths(image_paths: List[Path]) -> List[Path]:
    return sorted(
        image_paths,
        key=lambda p: (
            (idx := extract_frame_index(p)) is None,
            idx if idx is not None else p.stem,
        ),
    )


def collect_candidate_pairs(image_paths: List[Path], max_offset: int):
    indexed_paths = sort_indexed_paths(image_paths)
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


def collect_nonoverlap_windows(image_paths: List[Path], window_n: int) -> List[List[Path]]:
    """長さ window_n の非重複窓。余りフレームは捨てる。"""
    if window_n < 2:
        raise ValueError(f"window_n は 2 以上である必要があります: {window_n}")
    indexed_paths = sort_indexed_paths(image_paths)
    windows: List[List[Path]] = []
    for start in range(0, len(indexed_paths) - window_n + 1, window_n):
        windows.append(indexed_paths[start : start + window_n])
    return windows


def process_directory(target_dir: Path, colors, threshold: float, output_subdir: str = "") -> bool:
    image_paths = sorted(target_dir.glob(IMAGE_PATTERN))
    if not image_paths:
        return False

    pairs = collect_candidate_pairs(image_paths, MAX_OFFSET)
    if not pairs:
        print(f"[{target_dir.name}] 処理対象のペアがありません。")
        return True

    output_dir = resolve_output_dir(target_dir, output_subdir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cache = {}

    def get_image(path: Path):
        if path not in cache:
            cache[path] = load_image_as_rgb(path, QUALITY_SCALE, RESIZE_METHOD)
        return cache[path]

    th255 = threshold * 255.0
    print(
        f"\n[{target_dir.name}] {len(image_paths)}枚を検出。対象ペア {len(pairs)} 組を処理します。"
        f" threshold={th255:.0f}/255 ({threshold:.6f})"
    )
    for idx, (path1, path2) in enumerate(pairs, 1):
        img1 = get_image(path1)
        img2 = get_image(path2)

        if img1.shape != img2.shape:
            raise ValueError(f"画像サイズが一致しません: {path1.name} {img1.shape} vs {path2.name} {img2.shape}")

        increased, decreased, unchanged, _ = classify_luminance_change(img1, img2, threshold)

        pair_name = build_pair_folder_name(path1, path2)
        output_path = output_dir / f"{pair_name}_rgbmax_scale{QUALITY_SCALE:.2f}.png"
        save_combined_map(increased, decreased, unchanged, colors, output_path)

        if idx % 20 == 0 or idx == len(pairs):
            print(f"  {idx} / {len(pairs)} 組完了")

    print(f"[{target_dir.name}] 完了。出力先: {output_dir.resolve()}")
    return True


def process_directory_accum(
    target_dir: Path,
    colors,
    threshold: float,
    window_n: int,
    output_subdir: str = "",
) -> bool:
    """非重複窓で |max-channel差分| を合算して二値化マップを出す。"""
    image_paths = sorted(target_dir.glob(IMAGE_PATTERN))
    if not image_paths:
        return False

    windows = collect_nonoverlap_windows(image_paths, window_n)
    if not windows:
        print(
            f"[{target_dir.name}] 処理対象の窓がありません "
            f"(frames={len(image_paths)}, window_n={window_n})。"
        )
        return True

    output_dir = resolve_output_dir(target_dir, output_subdir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cache = {}

    def get_image(path: Path):
        if path not in cache:
            cache[path] = load_image_as_rgb(path, QUALITY_SCALE, RESIZE_METHOD)
        return cache[path]

    th255 = threshold * 255.0
    print(
        f"\n[{target_dir.name}] {len(image_paths)}枚を検出。非重複窓 {len(windows)} 個を処理します。"
        f" window_n={window_n} threshold={th255:.0f}/255 ({threshold:.6f})"
    )
    for idx, window in enumerate(windows, 1):
        acc = None
        for i in range(len(window) - 1):
            img1 = get_image(window[i])
            img2 = get_image(window[i + 1])
            if img1.shape != img2.shape:
                raise ValueError(
                    f"画像サイズが一致しません: {window[i].name} {img1.shape} "
                    f"vs {window[i + 1].name} {img2.shape}"
                )
            abs_diff = np.abs(compute_max_channel_difference(img1, img2))
            acc = abs_diff if acc is None else (acc + abs_diff)

        assert acc is not None
        increased, decreased, unchanged = classify_accumulated_abs_change(acc, threshold)
        pair_name = build_pair_folder_name(window[0], window[-1])
        output_path = output_dir / f"{pair_name}_rgbmax_scale{QUALITY_SCALE:.2f}.png"
        save_combined_map(increased, decreased, unchanged, colors, output_path)

        if idx % 20 == 0 or idx == len(windows):
            print(f"  {idx} / {len(windows)} 窓完了")

    print(f"[{target_dir.name}] 完了。出力先: {output_dir.resolve()}")
    return True


def process_directory_stat(
    target_dir: Path,
    colors,
    threshold: float,
    stat_kind: str,
    output_subdir: str = "",
) -> bool:
    """120フレーム時系列の各ピクセル統計（std/var）を二値化して1枚出力。"""
    image_paths = sort_indexed_paths(list(target_dir.glob(IMAGE_PATTERN)))
    if not image_paths:
        return False

    cache = {}

    def get_image(path: Path):
        if path not in cache:
            cache[path] = load_image_as_rgb(path, QUALITY_SCALE, RESIZE_METHOD)
        return cache[path]

    series: List[np.ndarray] = []
    base_shape = None
    for path in image_paths:
        img = get_image(path)
        if base_shape is None:
            base_shape = img.shape
        elif img.shape != base_shape:
            raise ValueError(f"画像サイズが一致しません: {path.name} {img.shape} vs {base_shape}")
        series.append(compute_max_channel_signal(img))

    if not series:
        return True

    stack = np.stack(series, axis=0)
    if stat_kind == "var":
        stat_map = np.var(stack, axis=0)
    else:
        stat_map = np.std(stack, axis=0)

    increased, decreased, unchanged = classify_stat_change(stat_map, threshold)
    output_dir = resolve_output_dir(target_dir, output_subdir)
    output_dir.mkdir(exist_ok=True, parents=True)

    first = image_paths[0]
    last = image_paths[-1]
    pair_name = build_pair_folder_name(first, last)
    output_path = output_dir / f"{pair_name}_rgbmax_scale{QUALITY_SCALE:.2f}.png"
    save_combined_map(increased, decreased, unchanged, colors, output_path)

    th255 = threshold * 255.0
    print(
        f"\n[{target_dir.name}] stat={stat_kind} frames={len(image_paths)} "
        f"threshold={th255:.0f}/255 ({threshold:.6f}) -> {output_path.name}"
    )
    print(f"[{target_dir.name}] 完了。出力先: {output_dir.resolve()}")
    return True


def process_directory_fourier(
    target_dir: Path,
    colors,
    threshold: float,
    fps: float,
    target_freq: float,
    band_radius: int,
    output_subdir: str = "",
) -> bool:
    """120フレーム時系列の時間軸FFTスコアマップを二値化して1枚出力。"""
    from time_fft import build_score_map, normalize_score_map

    image_paths = sort_indexed_paths(list(target_dir.glob(IMAGE_PATTERN)))
    if not image_paths:
        return False

    cache = {}

    def get_image(path: Path):
        if path not in cache:
            cache[path] = load_image_as_rgb(path, QUALITY_SCALE, RESIZE_METHOD)
        return cache[path]

    series: List[np.ndarray] = []
    base_shape = None
    for path in image_paths:
        img = get_image(path)
        if base_shape is None:
            base_shape = img.shape
        elif img.shape != base_shape:
            raise ValueError(f"画像サイズが一致しません: {path.name} {img.shape} vs {base_shape}")
        series.append(compute_max_channel_signal(img))

    if not series:
        return True

    stack = np.stack(series, axis=0)
    score_map = build_score_map(stack, fps, target_freq, band_radius=band_radius)
    norm_map = normalize_score_map(score_map)

    increased, decreased, unchanged = classify_stat_change(norm_map, threshold)
    output_dir = resolve_output_dir(target_dir, output_subdir)
    output_dir.mkdir(exist_ok=True, parents=True)

    first = image_paths[0]
    last = image_paths[-1]
    pair_name = build_pair_folder_name(first, last)
    output_path = output_dir / f"{pair_name}_rgbmax_scale{QUALITY_SCALE:.2f}.png"
    save_combined_map(increased, decreased, unchanged, colors, output_path)

    th255 = threshold * 255.0
    print(
        f"\n[{target_dir.name}] fourier target={target_freq:.3f}Hz fps={fps:.3f} "
        f"frames={len(image_paths)} threshold={th255:.0f}/255 ({threshold:.6f}) "
        f"-> {output_path.name}"
    )
    print(f"[{target_dir.name}] 完了。出力先: {output_dir.resolve()}")
    return True


def main():
    parser = argparse.ArgumentParser(description="frame_*.png の差分マップを生成する（pair / accum / stat / fourier）")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="対象ルートの絶対パス（未指定時はカレントディレクトリ）",
    )
    parser.add_argument(
        "--diff-mode",
        type=str,
        choices=["pair", "accum", "stat", "fourier"],
        default="pair",
        help="差分モード: pair=隣接ペア / accum=非重複窓|差分|合算 / stat=時間方向統計 / fourier=時間軸FFT",
    )
    parser.add_argument(
        "--window-n",
        type=int,
        default=4,
        help="accum 時の窓長（フレーム数）。既定: 4",
    )
    parser.add_argument(
        "--stat-kind",
        type=str,
        choices=["std", "var"],
        default="std",
        help="stat 時の統計量: std（既定） or var",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="差分二値化閾値（0-255 dual。例: 4 → 4/255）。既定: 4",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="差分の出力サブディレクトリ名（未指定時: rgb_max_diff_maps）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="fourier 時のカメラ fps（既定: 60）",
    )
    parser.add_argument(
        "--target-freq",
        type=float,
        default=30.0,
        help="fourier 時のターゲット周波数 Hz（既定: 30）",
    )
    parser.add_argument(
        "--band-radius",
        type=int,
        default=1,
        help="fourier 時の target_idx 前後ビン数（既定: 1）",
    )
    args = parser.parse_args()

    colors = {
        "increased": parse_color(COLOR_INCREASED),
        "decreased": parse_color(COLOR_DECREASED),
        "unchanged": parse_color(COLOR_UNCHANGED),
    }

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base-dir が存在しません: {base_dir}")
    print(f"[INFO] base_dir: {base_dir}")
    print(f"[INFO] diff_mode: {args.diff_mode}")

    threshold = float(args.threshold) / 255.0
    output_subdir = args.output_subdir.strip() or str(OUTPUT_DIR)
    window_n = int(args.window_n)
    if args.diff_mode == "accum" and window_n < 2:
        raise SystemExit("--window-n は 2 以上である必要があります")

    target_dirs = [base_dir] + sorted(path for path in base_dir.iterdir() if path.is_dir())

    processed_count = 0
    for target_dir in target_dirs:
        try:
            if args.diff_mode == "accum":
                ok = process_directory_accum(
                    target_dir,
                    colors,
                    threshold=threshold,
                    window_n=window_n,
                    output_subdir=output_subdir,
                )
            elif args.diff_mode == "stat":
                ok = process_directory_stat(
                    target_dir,
                    colors,
                    threshold=threshold,
                    stat_kind=str(args.stat_kind),
                    output_subdir=output_subdir,
                )
            elif args.diff_mode == "fourier":
                ok = process_directory_fourier(
                    target_dir,
                    colors,
                    threshold=threshold,
                    fps=float(args.fps),
                    target_freq=float(args.target_freq),
                    band_radius=int(args.band_radius),
                    output_subdir=output_subdir,
                )
            else:
                ok = process_directory(
                    target_dir, colors, threshold=threshold, output_subdir=output_subdir
                )
            if ok:
                processed_count += 1
        except Exception as exc:
            print(f"[WARN] {target_dir} の処理中にエラー: {exc}")

    if processed_count == 0:
        raise FileNotFoundError(
            f"{base_dir} およびサブディレクトリに {IMAGE_PATTERN} が見つかりません。"
        )

    print(f"\n処理終了: {processed_count} フォルダを処理しました。")

if __name__ == "__main__":
    main()
