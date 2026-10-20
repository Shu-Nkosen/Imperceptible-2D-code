"""pair-diff 系列 d(t) から各種差分マップを生成（hully パイプライン用）。

TODO(hully): pair以外は数値スコア→grayデコードの方が良ければ、
hard の gray 経路を hully 既定にも適用する（現状 hully は binary+th のみ）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from gpu_ops import max_channel_difference, temporal_std_var


@dataclass(frozen=True)
class MapSpec:
    """in-memory 差分マップ + メタデータ。"""

    rgb: np.ndarray  # (H,W,3) uint8
    diff_mode: str
    diff_threshold: int
    window_n: Optional[int] = None
    stat_kind: str = ""
    fft_target_hz: Optional[float] = None
    frame_1: str = ""
    frame_2: str = ""
    diff_subdir: str = ""
    pair_name: str = ""


def build_pair_diff_stack(rgb_list: List[np.ndarray]) -> np.ndarray:
    """120 RGB float → (T-1,H,W) 隣接 max-channel 差分。"""
    if len(rgb_list) < 2:
        h = rgb_list[0].shape[0] if rgb_list else 0
        w = rgb_list[0].shape[1] if rgb_list else 0
        return np.zeros((0, h, w), dtype=np.float32)
    diffs = [max_channel_difference(rgb_list[t], rgb_list[t + 1]) for t in range(len(rgb_list) - 1)]
    return np.stack(diffs, axis=0).astype(np.float32)


SCALE_TAG = "rgbmax_scale1.00"


def _diff_to_rgb(increased, decreased, unchanged) -> np.ndarray:
    """cal-from-2frame 既定色: increased/decreased=黒, unchanged=白。"""
    rgb = np.empty((*increased.shape, 3), dtype=np.uint8)
    rgb[unchanged] = (255, 255, 255)
    rgb[increased] = (0, 0, 0)
    rgb[decreased] = (0, 0, 0)
    return rgb


def score_to_gray_rgb(score_map: np.ndarray) -> np.ndarray:
    """連続スコアを 0..1 正規化し、高スコア=黒のグレースケール RGB にする。"""
    from time_fft import normalize_score_map

    norm = normalize_score_map(np.asarray(score_map, dtype=np.float32))
    gray = ((1.0 - norm) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def save_map_png(rgb: np.ndarray, output_path: Path) -> None:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def classify_pair_diff(diff_map: np.ndarray, threshold: float):
    increased = diff_map > threshold
    decreased = diff_map < -threshold
    unchanged = ~(increased | decreased)
    return increased, decreased, unchanged


def classify_scalar_map(stat_map: np.ndarray, threshold: float):
    changed = stat_map > threshold
    unchanged = ~changed
    return changed, np.zeros_like(changed, dtype=bool), unchanged


# lockin / fourier の正規化スコア二値化コード（diff_threshold に格納）
# 20未満: 旧スケール th/255（mid/hard 互換）
# 20〜100: 正規化[0,1]上のパーセント（30 → 0.30）
# 901: Otsu / 902: adaptiveThreshold
NORM_BIN_OTSU = 901
NORM_BIN_ADAPTIVE = 902


def format_norm_bin_label(threshold: int) -> str:
    """差分サブディレクトリ用ラベル。"""
    if int(threshold) == NORM_BIN_OTSU:
        return "otsu"
    if int(threshold) == NORM_BIN_ADAPTIVE:
        return "adapt"
    if int(threshold) >= 20:
        return f"p{int(threshold)}"
    return f"th{int(threshold)}"


def binarize_normalized_score_map(score_map: np.ndarray, threshold: int) -> np.ndarray:
    """正規化スコアマップ → 黒白 RGB。

    - threshold < 20: 旧仕様（norm 後に th/255）。壊れたスケールだが mid/hard 互換のため残す
    - 20..100: 正規化[0,1]上の固定閾値（percent/100）
    - 901: Otsu（大域・自動）
    - 902: adaptiveThreshold（局所）
    """
    from time_fft import normalize_score_map

    import cv2

    norm = normalize_score_map(np.asarray(score_map, dtype=np.float32))
    code = int(threshold)

    if code == NORM_BIN_OTSU:
        # 高スコアを明るくして Otsu。明るい側をモジュール（出力黒）にする
        gray_u8 = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
        _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        changed = bw == 255
    elif code == NORM_BIN_ADAPTIVE:
        gray_u8 = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
        bw = cv2.adaptiveThreshold(
            gray_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        changed = bw == 255
    else:
        if code >= 20:
            th = float(code) / 100.0
        else:
            th = float(code) / 255.0
        changed, _, _ = classify_scalar_map(norm, th)

    unchanged = ~changed
    return _diff_to_rgb(changed, np.zeros_like(changed, dtype=bool), unchanged)


def pair_index_pairs(n_diffs: int, n_each: int) -> List[Tuple[int, int]]:
    """隣接 diff インデックス t → フレーム (t, t+1)。先頭/末尾 n_each。"""
    if n_diffs <= 0:
        return []
    all_pairs = [(t, t + 1) for t in range(n_diffs)]
    if n_each <= 0 or len(all_pairs) <= n_each * 2:
        return all_pairs
    selected = all_pairs[:n_each] + all_pairs[-n_each:]
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for item in selected:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def accum_maps_from_diff(
    d_stack: np.ndarray, window_n: int
) -> List[Tuple[np.ndarray, int, int]]:
    """非重複窓ごとに |d| を合算 → [(acc_map, start_frame, end_frame), ...]。

    baseline ``process_directory_accum`` と同じ: 長さ window_n のフレーム窓ごとに
    隣接差分 (window_n-1) 本の絶対値を合算し、窓ごとに1マップを返す。
    """
    if window_n < 2:
        raise ValueError(f"window_n は 2 以上である必要があります: {window_n}")
    t_len = d_stack.shape[0]
    n_frames = t_len + 1
    if n_frames < window_n or t_len < 1:
        return []
    out: List[Tuple[np.ndarray, int, int]] = []
    for start in range(0, n_frames - window_n + 1, window_n):
        # フレーム [start, start+window_n) → 差分 d[start : start+window_n-1]
        chunk = d_stack[start : start + window_n - 1]
        if chunk.shape[0] == 0:
            continue
        acc = np.abs(chunk).sum(axis=0).astype(np.float32)
        end_frame = start + window_n - 1
        out.append((acc, start, end_frame))
    return out


def stat_map_from_diff(d_stack: np.ndarray, kind: str) -> np.ndarray:
    return temporal_std_var(d_stack, kind)


def lockin_map_from_diff(
    d_stack: np.ndarray,
    fps: float,
    target_freq: float,
    phase_steps: int = 8,
) -> np.ndarray:
    from time_fft import lockin_score_map

    return lockin_score_map(d_stack, fps=fps, target_freq=target_freq, phase_steps=phase_steps)


def frame_name(idx: int) -> str:
    return f"frame_{idx:05d}.png"


def build_pair_name(idx1: int, idx2: int) -> str:
    return f"{idx1:05d}-{idx2:05d}-FRAME"


def generate_pair_maps(
    d_stack: np.ndarray,
    threshold: int,
    pair_each_end: int,
) -> List[MapSpec]:
    th = float(threshold) / 255.0
    subdir = f"rgb_max_diff_maps_th{threshold}"
    specs: List[MapSpec] = []
    n_diffs = d_stack.shape[0]
    for t1, t2 in pair_index_pairs(n_diffs, pair_each_end):
        if t1 >= n_diffs:
            continue
        inc, dec, unch = classify_pair_diff(d_stack[t1], th)
        rgb = _diff_to_rgb(inc, dec, unch)
        pname = build_pair_name(t1, t2)
        specs.append(
            MapSpec(
                rgb=rgb,
                diff_mode="pair",
                diff_threshold=threshold,
                frame_1=frame_name(t1),
                frame_2=frame_name(t2),
                diff_subdir=subdir,
                pair_name=pname,
            )
        )
    return specs


def generate_accum_maps(
    d_stack: np.ndarray,
    window_n: int,
    threshold: int,
    pair_each_end: int = 0,
    repr_mode: str = "binary",
) -> List[MapSpec]:
    """窓ごと1枚の MapSpec（pair_each_end は無視。baseline 同様全窓）。"""
    _ = pair_each_end
    mode = "gray" if repr_mode == "gray" else "binary"
    if mode == "gray":
        subdir = f"rgb_max_accum_n{window_n}_gray"
    else:
        subdir = f"rgb_max_accum_n{window_n}_th{threshold}"
    specs: List[MapSpec] = []
    for acc, t1, t2 in accum_maps_from_diff(d_stack, window_n):
        if mode == "gray":
            rgb = score_to_gray_rgb(acc)
        else:
            th = float(threshold) / 255.0
            inc, dec, unch = classify_scalar_map(acc, th)
            rgb = _diff_to_rgb(inc, dec, unch)
        specs.append(
            MapSpec(
                rgb=rgb,
                diff_mode="accum",
                diff_threshold=threshold,
                window_n=window_n,
                frame_1=frame_name(t1),
                frame_2=frame_name(t2),
                diff_subdir=subdir,
                pair_name=build_pair_name(t1, t2),
            )
        )
    return specs


def generate_stat_maps(
    d_stack: np.ndarray,
    stat_kind: str,
    threshold: int,
    repr_mode: str = "binary",
) -> List[MapSpec]:
    mode = "gray" if repr_mode == "gray" else "binary"
    if mode == "gray":
        subdir = f"rgb_max_stat_{stat_kind}_gray"
    else:
        subdir = f"rgb_max_stat_{stat_kind}_th{threshold}"
    stat_map = stat_map_from_diff(d_stack, stat_kind)
    if mode == "gray":
        rgb = score_to_gray_rgb(stat_map)
    else:
        th = float(threshold) / 255.0
        inc, dec, unch = classify_scalar_map(stat_map, th)
        rgb = _diff_to_rgb(inc, dec, unch)
    n = d_stack.shape[0]
    return [
        MapSpec(
            rgb=rgb,
            diff_mode="stat",
            diff_threshold=threshold,
            stat_kind=stat_kind,
            frame_1=frame_name(0),
            frame_2=frame_name(n),
            diff_subdir=subdir,
            pair_name=build_pair_name(0, n),
        )
    ]


def generate_lockin_maps(
    d_stack: np.ndarray,
    fps: float,
    target_freq: float,
    threshold: int,
    phase_steps: int = 8,
    repr_mode: str = "binary",
) -> List[MapSpec]:
    """d(t) への複素ロックイン振幅マップ。"""
    from time_fft import format_freq_label

    mode = "gray" if repr_mode == "gray" else "binary"
    freq_label = format_freq_label(target_freq)
    steps = max(1, int(phase_steps))
    if mode == "gray":
        subdir = f"rgb_max_lockin_{freq_label}_ps{steps}_gray"
    else:
        subdir = f"rgb_max_lockin_{freq_label}_ps{steps}_{format_norm_bin_label(threshold)}"
    score = lockin_map_from_diff(d_stack, fps, target_freq, phase_steps=steps)
    if mode == "gray":
        rgb = score_to_gray_rgb(score)
    else:
        rgb = binarize_normalized_score_map(score, threshold)
    n = d_stack.shape[0]
    return [
        MapSpec(
            rgb=rgb,
            diff_mode="lockin",
            diff_threshold=threshold,
            fft_target_hz=target_freq,
            frame_1=frame_name(0),
            frame_2=frame_name(n),
            diff_subdir=subdir,
            pair_name=build_pair_name(0, n),
        )
    ]


def generate_fourier_maps(
    d_stack: np.ndarray,
    fps: float,
    target_freq: float,
    threshold: int,
    band_radius: int = 1,
    repr_mode: str = "binary",
) -> List[MapSpec]:
    """d(t) への帯付き FFT スコアマップ（target 近傍 / noise floor）。"""
    from time_fft import build_score_map, format_freq_label

    mode = "gray" if repr_mode == "gray" else "binary"
    freq_label = format_freq_label(target_freq)
    if mode == "gray":
        subdir = f"rgb_max_fourier_{freq_label}_gray"
    else:
        subdir = f"rgb_max_fourier_{freq_label}_{format_norm_bin_label(threshold)}"
    score = build_score_map(
        d_stack,
        fps=fps,
        target_freq=target_freq,
        band_radius=band_radius,
        use_window=True,
    )
    if mode == "gray":
        rgb = score_to_gray_rgb(score)
    else:
        rgb = binarize_normalized_score_map(score, threshold)
    n = d_stack.shape[0]
    return [
        MapSpec(
            rgb=rgb,
            diff_mode="fourier",
            diff_threshold=threshold,
            fft_target_hz=target_freq,
            frame_1=frame_name(0),
            frame_2=frame_name(n),
            diff_subdir=subdir,
            pair_name=build_pair_name(0, n),
        )
    ]
