"""検証用時間FFTスコア: ratio（現行）と amp（割りなし）。"""
from __future__ import annotations

import numpy as np


def _prepare(frames_thw: np.ndarray, use_window: bool) -> np.ndarray:
    prepared = frames_thw.astype(np.float32) - frames_thw.mean(axis=0, keepdims=True)
    if use_window and prepared.shape[0] >= 2:
        window = np.hanning(prepared.shape[0]).astype(np.float32)[:, None, None]
        prepared = prepared * window
    return prepared


def fourier_score_map(
    frames_thw: np.ndarray,
    fps: float,
    target_freq: float,
    band_radius: int = 2,
    use_window: bool = True,
    score_mode: str = "ratio",
) -> np.ndarray:
    """score_mode: ratio=target/noise、amp=target振幅のみ。"""
    if frames_thw.ndim != 3:
        raise ValueError(f"frames must be (T,H,W), got {frames_thw.shape}")
    mode = score_mode.lower().strip()
    if mode not in ("ratio", "amp"):
        raise ValueError(f"score_mode must be ratio|amp, got {score_mode}")

    prepared = _prepare(frames_thw, use_window=use_window)
    spectrum = np.fft.fft(prepared, axis=0)
    freqs = np.fft.fftfreq(prepared.shape[0], d=1.0 / float(fps))
    positive = freqs >= 0
    freqs_p = freqs[positive]
    amp_p = np.abs(spectrum[positive])
    target_idx = int(np.argmin(np.abs(freqs_p - float(target_freq))))
    start = max(0, target_idx - int(band_radius))
    stop = min(len(freqs_p), target_idx + int(band_radius) + 1)
    target_amplitude = amp_p[start:stop].max(axis=0)

    if mode == "amp":
        return target_amplitude.astype(np.float32)

    if amp_p.shape[0] > 1:
        noise_floor = amp_p[1:].mean(axis=0)
    else:
        noise_floor = amp_p[0]
    return (target_amplitude / (noise_floor + 1e-12)).astype(np.float32)
