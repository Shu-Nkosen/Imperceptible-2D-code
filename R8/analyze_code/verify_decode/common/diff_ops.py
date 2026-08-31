"""検証用差分（本体 gpu_ops / hully_diff は変更しない）。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


def _selected_minus_other_mean(diff_all: np.ndarray, selected: np.ndarray) -> np.ndarray:
    other_mean = (diff_all.sum(axis=2) - selected) / 2.0
    return (selected - other_mean).astype(np.float32)


def max_channel_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """現行と同式: 画素ごと |Δ|最大チャネル − 他2平均。"""
    diff_all = img2.astype(np.float32) - img1.astype(np.float32)
    abs_diff = np.abs(diff_all)
    max_indices = np.argmax(abs_diff, axis=2)
    selected = np.take_along_axis(diff_all, max_indices[..., None], axis=2).squeeze(-1)
    return _selected_minus_other_mean(diff_all, selected)


def fixed_channel_difference(img1: np.ndarray, img2: np.ndarray, channel: str) -> np.ndarray:
    """埋め込みチャネル固定: 指定チャネル − 他2平均。"""
    ch = channel.upper()
    if ch not in CHANNEL_INDEX:
        raise ValueError(f"channel must be R/G/B, got {channel}")
    idx = CHANNEL_INDEX[ch]
    diff_all = img2.astype(np.float32) - img1.astype(np.float32)
    selected = diff_all[:, :, idx]
    return _selected_minus_other_mean(diff_all, selected)


def build_pair_diff_stack(
    rgb_list: List[np.ndarray],
    channel: Optional[str] = None,
) -> np.ndarray:
    """channel=None → max、'R'|'G'|'B' → 固定平面。"""
    if len(rgb_list) < 2:
        h = rgb_list[0].shape[0] if rgb_list else 0
        w = rgb_list[0].shape[1] if rgb_list else 0
        return np.zeros((0, h, w), dtype=np.float32)
    diffs = []
    for t in range(len(rgb_list) - 1):
        a, b = rgb_list[t], rgb_list[t + 1]
        if channel is None:
            diffs.append(max_channel_difference(a, b))
        else:
            diffs.append(fixed_channel_difference(a, b, channel))
    return np.stack(diffs, axis=0).astype(np.float32)
