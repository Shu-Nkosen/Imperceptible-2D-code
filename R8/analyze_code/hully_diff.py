"""pair-diff 系列 d(t) から各種差分マップを生成（hully パイプライン用）。"""
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


def accum_map_from_diff(d_stack: np.ndarray, window_n: int) -> np.ndarray:
    """非重複窓で |d| を合算 → (H,W)。"""
    t_len = d_stack.shape[0]
    if t_len < window_n:
        return np.zeros(d_stack.shape[1:], dtype=np.float32)
    acc = None
    for start in range(0, t_len - window_n + 1, window_n):
        chunk = d_stack[start : start + window_n]
        part = np.abs(chunk).sum(axis=0)
        acc = part if acc is None else acc + part
    assert acc is not None
    return acc.astype(np.float32)


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
    pair_each_end: int,
) -> List[MapSpec]:
    th = float(threshold) / 255.0
    subdir = f"rgb_max_accum_n{window_n}_th{threshold}"
    acc = accum_map_from_diff(d_stack, window_n)
    inc, dec, unch = classify_scalar_map(acc, th)
    rgb = _diff_to_rgb(inc, dec, unch)
    n_diffs = d_stack.shape[0]
    if n_diffs <= 0:
        return []
    pairs = pair_index_pairs(n_diffs, pair_each_end)
    t1, t2 = pairs[0] if pairs else (0, min(1, n_diffs))
    end_t2 = pairs[-1][1] if pairs else t2
    specs = [
        MapSpec(
            rgb=rgb,
            diff_mode="accum",
            diff_threshold=threshold,
            window_n=window_n,
            frame_1=frame_name(t1),
            frame_2=frame_name(end_t2),
            diff_subdir=subdir,
            pair_name=build_pair_name(t1, end_t2),
        )
    ]
    return specs


def generate_stat_maps(
    d_stack: np.ndarray,
    stat_kind: str,
    threshold: int,
) -> List[MapSpec]:
    th = float(threshold) / 255.0
    subdir = f"rgb_max_stat_{stat_kind}_th{threshold}"
    stat_map = stat_map_from_diff(d_stack, stat_kind)
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


def generate_fourier_maps(
    d_stack: np.ndarray,
    fps: float,
    target_freq: float,
    threshold: int,
    phase_steps: int = 8,
) -> List[MapSpec]:
    from time_fft import format_freq_label, normalize_score_map

    th = float(threshold) / 255.0
    freq_label = format_freq_label(target_freq)
    subdir = f"rgb_max_fourier_{freq_label}_th{threshold}"
    score = lockin_map_from_diff(d_stack, fps, target_freq, phase_steps=phase_steps)
    norm = normalize_score_map(score)
    inc, dec, unch = classify_scalar_map(norm, th)
    rgb = _diff_to_rgb(inc, dec, unch)
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
