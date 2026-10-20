# -*- coding: utf-8 -*-
"""復号条件数グラフ向けのラベル。図タイトルは見出しのみ。"""
from __future__ import annotations

METRIC_YLABEL = "復号できた条件数"

# 実験条件スライドと同じ順：画像 × チャネル × 強度 × 切替Hz × 露光 × 蛍光
IMG = 4
CH = 5
INTEN = 3
HZ = 5
HZ4590 = 3  # 本編の 45・60・90
EXP = 3
TALK_EXP = 125  # 露光比較以外の本編は 1/125
FL = 2

_FACTORS = (
    ("img", "画像"),
    ("ch", "チャネル"),
    ("inten", "強度"),
    ("hz", "切替Hz"),
    ("exp", "露光"),
    ("fl", "蛍光"),
)
_KNOWN = {k for k, _ in _FACTORS}


def denom(**parts: int) -> str:
    """例: 画像4×チャネル5×強度3×切替Hz5×蛍光2（600条件）"""
    unknown = set(parts) - _KNOWN
    if unknown:
        raise TypeError(f"unknown factor keys: {sorted(unknown)}")
    bits: list[str] = []
    n = 1
    for key, label in _FACTORS:
        if key not in parts:
            continue
        v = int(parts[key])
        bits.append(f"{label}{v}")
        n *= v
    if not bits:
        raise TypeError("denom() needs at least one factor")
    return f"{'×'.join(bits)}（{n}条件）"


def scope_n(**parts: int) -> int:
    unknown = set(parts) - _KNOWN
    if unknown:
        raise TypeError(f"unknown factor keys: {sorted(unknown)}")
    n = 1
    used = False
    for key, _label in _FACTORS:
        if key not in parts:
            continue
        n *= int(parts[key])
        used = True
    if not used:
        raise TypeError("scope_n() needs at least one factor")
    return n


def title_scope(main: str, **parts: int) -> str:
    """図タイトルは主語のみ。parts は互換のため残す。"""
    return main


def title_scope_text(main: str, scope: str) -> str:
    """図タイトルは主語のみ。scope は互換のため残す。"""
    return main


def heatmap_cell_text(ok: float | int) -> str:
    return str(int(ok))
