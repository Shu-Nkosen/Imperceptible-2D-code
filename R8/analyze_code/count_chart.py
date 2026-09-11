# -*- coding: utf-8 -*-
"""復号条件数グラフ向けのラベル（図タイトルに N を載せる）。"""
from __future__ import annotations

METRIC_YLABEL = "復号できた条件数"


def title_scope(main: str, n: int) -> str:
    """例: 露光別 + （各600条件中）"""
    return f"{main}\n（各{n}条件中）"


def title_scope_text(main: str, scope: str) -> str:
    """例: 強度8の穴 + （45 Hz・各6条件中）"""
    return f"{main}\n（{scope}）"


def heatmap_cell_text(ok: float | int) -> str:
    return str(int(ok))
