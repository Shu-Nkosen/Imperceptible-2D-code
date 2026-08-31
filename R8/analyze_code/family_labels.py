# -*- coding: utf-8 -*-
"""発表・要約図用ラベル（内部キーは英語のまま）。"""
from __future__ import annotations

FAMILY_JP = {
    "pair": "2フレーム差分",
    "accum": "5フレーム積算",
    "lockin": "同期検波",
    "fourier": "時間FFT",
    "stat_std": "時間標準偏差",
    "stat_var": "時間分散",
}

# 指標・凡例・軸
METRIC_ANY_SUCCESS = "復号成功率"
METRIC_ANY_SUCCESS_PCT = "復号成功率（%）"
LABEL_BINARY = "二値化"
LABEL_GRAY = "濃淡"
LABEL_INTENSITY = "強度"
LABEL_RATE = "表示レート"
LABEL_EXPOSURE = "露光"
LABEL_FLUORO_OFF = "蛍光灯なし"
LABEL_FLUORO_ON = "蛍光灯あり"
LABEL_FLUORO_COMPARE = "蛍光灯の有無別 復号成功率"
LABEL_FAMILY = "手法"

TALK_FAMILIES = ["pair", "accum", "lockin", "fourier"]

# 発表の accum は window_n=5（binary+gray）のみ
ACCUM_TALK_WINDOW_N = 5

CHANNEL_JP = {
    "R": "R",
    "G": "G",
    "B": "B",
    "I": "最暗",
    "X": "最明",
    "min": "最暗",
    "max": "最明",
}


def jp_family(name: str) -> str:
    return FAMILY_JP.get(name, name)


def jp_channel(name: str) -> str:
    return CHANNEL_JP.get(name, name)


def jp_channels(names: list[str]) -> list[str]:
    return [jp_channel(n) for n in names]


def jp_intensity(val: str | int) -> str:
    return f"強度 {val}"


def jp_rate_legend(rate: int) -> str:
    return f"{rate} Hz"


def accum_sweep_keep(row: dict) -> bool:
    """accum / accum_num のスイープ行を発表用に残すか。"""
    try:
        return int(float(row.get("window_n") or 0)) == ACCUM_TALK_WINDOW_N
    except (TypeError, ValueError):
        return False
