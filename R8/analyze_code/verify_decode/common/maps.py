"""差分スタック → 復号用地図（検証用・薄い実装）。"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# 本体 hully_diff の二値化だけ再利用（書込・改変なし）
from hully_diff import (
    NORM_BIN_ADAPTIVE,
    NORM_BIN_OTSU,
    binarize_normalized_score_map,
    classify_pair_diff,
    classify_scalar_map,
)
from time_fft import lockin_score_map, resolve_target_freqs_hard

from verify_decode.common.fourier_score import fourier_score_map


def _diff_to_rgb(increased, decreased, unchanged) -> np.ndarray:
    rgb = np.empty((*increased.shape, 3), dtype=np.uint8)
    rgb[unchanged] = (255, 255, 255)
    rgb[increased] = (0, 0, 0)
    rgb[decreased] = (0, 0, 0)
    return rgb


def pair_maps(d_stack: np.ndarray, thresholds: Tuple[int, ...] = (4, 8, 12), each_end: int = 10) -> List[Tuple[str, np.ndarray, dict]]:
    n = d_stack.shape[0]
    indices = list(range(min(each_end, n))) + list(range(max(0, n - each_end), n))
    indices = sorted(set(indices))
    out: List[Tuple[str, np.ndarray, dict]] = []
    for th in thresholds:
        scale = float(th) / 255.0
        for t in indices:
            inc, dec, unch = classify_pair_diff(d_stack[t], scale)
            rgb = _diff_to_rgb(inc, dec, unch)
            meta = {"method": "pair", "diff_threshold": th, "pair_index": t}
            out.append((f"pair_th{th}_i{t}", rgb, meta))
    return out


def lockin_maps(
    d_stack: np.ndarray,
    fps: float,
    target_freqs: List[float],
    thresholds: Tuple[int, ...] = (30, 50, 70, NORM_BIN_OTSU, NORM_BIN_ADAPTIVE),
    phase_steps: int = 8,
) -> List[Tuple[str, np.ndarray, dict]]:
    out: List[Tuple[str, np.ndarray, dict]] = []
    for freq in target_freqs:
        score = lockin_score_map(d_stack, fps=fps, target_freq=freq, phase_steps=phase_steps)
        for th in thresholds:
            rgb = binarize_normalized_score_map(score, th)
            meta = {
                "method": "lockin",
                "fft_target_hz": float(freq),
                "diff_threshold": int(th),
                "phase_steps": phase_steps,
            }
            out.append((f"lockin_f{freq:.3f}_th{th}", rgb, meta))
    return out


def fourier_maps(
    d_stack: np.ndarray,
    fps: float,
    target_freqs: List[float],
    score_mode: str = "ratio",
    band_radius: int = 2,
    thresholds: Tuple[int, ...] = (30, 50, 70, NORM_BIN_OTSU, NORM_BIN_ADAPTIVE),
) -> List[Tuple[str, np.ndarray, dict]]:
    out: List[Tuple[str, np.ndarray, dict]] = []
    for freq in target_freqs:
        score = fourier_score_map(
            d_stack,
            fps=fps,
            target_freq=freq,
            band_radius=band_radius,
            use_window=True,
            score_mode=score_mode,
        )
        for th in thresholds:
            rgb = binarize_normalized_score_map(score, th)
            meta = {
                "method": "fourier",
                "fourier_score_mode": score_mode,
                "fft_target_hz": float(freq),
                "diff_threshold": int(th),
                "band_radius": band_radius,
            }
            out.append((f"fourier_{score_mode}_f{freq:.3f}_th{th}", rgb, meta))
    return out


def target_freqs_for_rate(rate_hz: int, camera_fps: float) -> List[float]:
    return list(resolve_target_freqs_hard(rate_hz, camera_fps))
