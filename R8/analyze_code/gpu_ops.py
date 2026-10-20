"""差分・統計・FFT 用の GPU/CPU 共通演算。

torch+CUDA があれば GPU、なければ NumPy。
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

_TORCH = None
_DEVICE = None
_INIT_DONE = False


def _ensure_torch() -> None:
    global _TORCH, _DEVICE, _INIT_DONE
    if _INIT_DONE:
        return
    _INIT_DONE = True
    try:
        import torch as _torch

        _TORCH = _torch
        if _torch.cuda.is_available():
            _DEVICE = _torch.device("cuda")
        else:
            _DEVICE = _torch.device("cpu")
    except Exception:
        _TORCH = None
        _DEVICE = None


def gpu_available() -> bool:
    _ensure_torch()
    return _TORCH is not None and _DEVICE is not None and _DEVICE.type == "cuda"


def backend_name() -> str:
    _ensure_torch()
    if _TORCH is None:
        return "numpy"
    if _DEVICE is not None and _DEVICE.type == "cuda":
        try:
            return f"torch-cuda:{_TORCH.cuda.get_device_name(0)}"
        except Exception:
            return "torch-cuda"
    return "torch-cpu"


def as_numpy(x: Any) -> np.ndarray:
    _ensure_torch()
    if _TORCH is not None and isinstance(x, _TORCH.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_device(arr: np.ndarray, dtype: Optional[Any] = None) -> Any:
    """NumPy → torch tensor（GPU優先）。torch 無しなら NumPy のまま。"""
    _ensure_torch()
    if _TORCH is None:
        out = np.asarray(arr)
        return out.astype(np.float32, copy=False) if dtype is None else out.astype(dtype, copy=False)
    t = _TORCH.as_tensor(arr, device=_DEVICE)
    if dtype is None:
        return t.to(dtype=_TORCH.float32)
    return t.to(dtype=dtype)


def max_channel_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """(H,W,3) float → (H,W) max-channel 差分。"""
    _ensure_torch()
    if _TORCH is None:
        diff_all = img2 - img1
        abs_diff = np.abs(diff_all)
        max_indices = np.argmax(abs_diff, axis=2)
        selected = np.take_along_axis(diff_all, max_indices[..., None], axis=2).squeeze(-1)
        other_mean = (diff_all.sum(axis=2) - selected) / 2.0
        return selected - other_mean

    a = to_device(img1)
    b = to_device(img2)
    diff_all = b - a
    abs_diff = diff_all.abs()
    max_indices = abs_diff.argmax(dim=2)
    selected = diff_all.gather(2, max_indices.unsqueeze(-1)).squeeze(-1)
    other_mean = (diff_all.sum(dim=2) - selected) / 2.0
    return as_numpy(selected - other_mean)


def max_channel_signal(img: np.ndarray) -> np.ndarray:
    """(H,W,3) float → (H,W) max-channel 強調。"""
    _ensure_torch()
    if _TORCH is None:
        mean_all = img.mean(axis=2, keepdims=True)
        abs_dev = np.abs(img - mean_all)
        max_indices = np.argmax(abs_dev, axis=2)
        selected = np.take_along_axis(img, max_indices[..., None], axis=2).squeeze(-1)
        other_mean = (img.sum(axis=2) - selected) / 2.0
        return selected - other_mean

    t = to_device(img)
    mean_all = t.mean(dim=2, keepdim=True)
    abs_dev = (t - mean_all).abs()
    max_indices = abs_dev.argmax(dim=2)
    selected = t.gather(2, max_indices.unsqueeze(-1)).squeeze(-1)
    other_mean = (t.sum(dim=2) - selected) / 2.0
    return as_numpy(selected - other_mean)


def stack_max_channel_signals(images_rgb: list[np.ndarray]) -> np.ndarray:
    """複数 RGB → (T,H,W) max-channel 信号。GPU なら一括。"""
    if not images_rgb:
        return np.zeros((0, 0, 0), dtype=np.float32)
    _ensure_torch()
    if _TORCH is None or len(images_rgb) < 2:
        return np.stack([max_channel_signal(img) for img in images_rgb], axis=0).astype(np.float32)

    batch = to_device(np.stack(images_rgb, axis=0))  # (T,H,W,3)
    mean_all = batch.mean(dim=3, keepdim=True)
    abs_dev = (batch - mean_all).abs()
    max_indices = abs_dev.argmax(dim=3)
    selected = batch.gather(3, max_indices.unsqueeze(-1)).squeeze(-1)
    other_mean = (batch.sum(dim=3) - selected) / 2.0
    return as_numpy(selected - other_mean).astype(np.float32)


def temporal_std_var(stack_thw: np.ndarray, kind: str) -> np.ndarray:
    """(T,H,W) → (H,W) std or var。"""
    _ensure_torch()
    kind = (kind or "std").lower()
    if _TORCH is None:
        if kind == "var":
            return np.var(stack_thw, axis=0).astype(np.float32)
        return np.std(stack_thw, axis=0).astype(np.float32)

    t = to_device(stack_thw)
    if kind == "var":
        out = t.var(dim=0, unbiased=False)
    else:
        out = t.std(dim=0, unbiased=False)
    return as_numpy(out).astype(np.float32)


def frequency_score_map(
    frames_thw: np.ndarray,
    fps: float,
    target_freq: float,
    band_radius: int = 1,
    use_window: bool = True,
) -> np.ndarray:
    """(T,H,W) → (H,W) target/noise スコア。torch.fft 優先。"""
    _ensure_torch()
    if frames_thw.ndim != 3:
        raise ValueError(f"frames must be (T,H,W), got {frames_thw.shape}")

    if _TORCH is not None:
        t = to_device(frames_thw)
        t = t - t.mean(dim=0, keepdim=True)
        if use_window and t.shape[0] >= 2:
            window = _TORCH.hann_window(t.shape[0], device=t.device, dtype=t.dtype)
            t = t * window.view(-1, 1, 1)
        spectrum = _TORCH.fft.fft(t, dim=0)
        amp = spectrum.abs()
        n = t.shape[0]
        freqs = _TORCH.fft.fftfreq(n, d=1.0 / float(fps), device=t.device)
        positive = freqs >= 0
        freqs_p = freqs[positive]
        amp_p = amp[positive]
        target_idx = int(_TORCH.argmin(_TORCH.abs(freqs_p - float(target_freq))).item())
        start = max(0, target_idx - int(band_radius))
        stop = min(int(freqs_p.shape[0]), target_idx + int(band_radius) + 1)
        target_amplitude = amp_p[start:stop].max(dim=0).values
        if amp_p.shape[0] > 1:
            noise_floor = amp_p[1:].mean(dim=0)
        else:
            noise_floor = amp_p[0]
        score = target_amplitude / (noise_floor + 1e-12)
        return as_numpy(score).astype(np.float32)

    # NumPy + scipy fallback
    try:
        from scipy import fft as sp_fft
    except ImportError as exc:
        raise ImportError("fourier 解析には scipy が必要です: pip install scipy") from exc

    prepared = frames_thw - frames_thw.mean(axis=0, keepdims=True)
    if use_window and prepared.shape[0] >= 2:
        window = np.hanning(prepared.shape[0]).astype(np.float32)[:, None, None]
        prepared = prepared * window
    spectrum = sp_fft.fft(prepared, axis=0, workers=-1)
    freqs = sp_fft.fftfreq(prepared.shape[0], d=1.0 / float(fps))
    positive = freqs >= 0
    freqs_p = freqs[positive]
    amp_p = np.abs(spectrum[positive])
    target_idx = int(np.argmin(np.abs(freqs_p - float(target_freq))))
    start = max(0, target_idx - int(band_radius))
    stop = min(len(freqs_p), target_idx + int(band_radius) + 1)
    target_amplitude = amp_p[start:stop].max(axis=0)
    if amp_p.shape[0] > 1:
        noise_floor = amp_p[1:].mean(axis=0)
    else:
        noise_floor = amp_p[0]
    return (target_amplitude / (noise_floor + 1e-12)).astype(np.float32)
