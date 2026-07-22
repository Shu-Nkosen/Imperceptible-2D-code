from __future__ import annotations

from typing import List

import numpy as np

try:
    from scipy import fft
except ImportError as exc:
    raise ImportError(
        "fourier 解析には scipy が必要です: pip install scipy"
    ) from exc


def alias_to_nyquist(f_hz: float, sample_fps: float) -> float:
    """信号周波数をカメラ fps で [0, fps/2] に折り畳む。"""
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive: {sample_fps}")
    f = float(f_hz) % float(sample_fps)
    if f > sample_fps / 2:
        f = sample_fps - f
    return f


def resolve_target_freqs(rate_hz: int, camera_fps: float) -> List[float]:
    """表示レートから第一候補とその半分の2周波数を返す。"""
    primary = alias_to_nyquist(rate_hz / 2.0, camera_fps)
    if primary < 2.0:
        primary = alias_to_nyquist(rate_hz / 4.0, camera_fps)
    secondary = primary / 2.0
    freqs = [primary]
    if abs(secondary - primary) > 0.5:
        freqs.append(secondary)
    return freqs


def format_freq_label(freq_hz: float) -> str:
    """差分サブディレクトリ用ラベル（例: 30→f30, 22.5→f225）。"""
    return f"f{int(round(float(freq_hz) * 10))}"


def detrend_frames(frames: np.ndarray) -> np.ndarray:
    return frames - frames.mean(axis=0, keepdims=True)


def apply_window(frames: np.ndarray, enabled: bool = True) -> np.ndarray:
    if not enabled or frames.shape[0] < 2:
        return frames
    window = np.hanning(frames.shape[0]).astype(np.float32)[:, None, None]
    return frames * window


def compute_frequency_spectrum(
    frames: np.ndarray, fps: float, use_window: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    prepared = detrend_frames(frames)
    prepared = apply_window(prepared, enabled=use_window)
    spectrum = fft.fft(prepared, axis=0, workers=-1)
    freqs = fft.fftfreq(prepared.shape[0], d=1.0 / fps)
    positive = freqs >= 0
    return freqs[positive], np.abs(spectrum[positive])


def resolve_target_band(
    freqs: np.ndarray, target_freq: float, band_radius: int = 1
) -> tuple[int, slice]:
    target_idx = int(np.argmin(np.abs(freqs - target_freq)))
    start = max(0, target_idx - band_radius)
    stop = min(len(freqs), target_idx + band_radius + 1)
    return target_idx, slice(start, stop)


def build_score_map(
    frames: np.ndarray,
    fps: float,
    target_freq: float,
    band_radius: int = 1,
    use_window: bool = True,
) -> np.ndarray:
    """(T,H,W) → (H,W) の FFT スコアマップ（target_amp / noise_floor）。GPU 優先。"""
    from gpu_ops import frequency_score_map

    return frequency_score_map(
        frames,
        fps=fps,
        target_freq=target_freq,
        band_radius=band_radius,
        use_window=use_window,
    )


def lockin_score_map(
    d_stack: np.ndarray,
    fps: float,
    target_freq: float,
    phase_steps: int = 8,
) -> np.ndarray:
    """pair-diff 系列 d(t) に対する複素ロックイン振幅 → (H,W)。"""
    if d_stack.ndim != 3:
        raise ValueError(f"d_stack must be (T,H,W), got {d_stack.shape}")
    t_len = d_stack.shape[0]
    if t_len < 2 or fps <= 0:
        return np.zeros(d_stack.shape[1:], dtype=np.float32)

    steps = max(1, int(phase_steps))
    t = np.arange(t_len, dtype=np.float32) / float(fps)
    omega = 2.0 * np.pi * float(target_freq)
    best = np.zeros(d_stack.shape[1:], dtype=np.float32)

    for k in range(steps):
        phi = 2.0 * np.pi * k / steps
        ref_cos = np.cos(omega * t + phi).astype(np.float32)[:, None, None]
        ref_sin = np.sin(omega * t + phi).astype(np.float32)[:, None, None]
        in_phase = (d_stack * ref_cos).sum(axis=0)
        quadrature = (d_stack * ref_sin).sum(axis=0)
        amp = np.sqrt(in_phase * in_phase + quadrature * quadrature)
        best = np.maximum(best, amp)

    return best.astype(np.float32)


def normalize_score_map(score_map: np.ndarray) -> np.ndarray:
    min_value = float(np.min(score_map))
    max_value = float(np.max(score_map))
    if max_value <= min_value:
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - min_value) / (max_value - min_value)).astype(np.float32)
