"""デコードなしの数値自己検査（数秒で終わる）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANALYZE = Path(__file__).resolve().parents[1]
if str(_ANALYZE) not in sys.path:
    sys.path.insert(0, str(_ANALYZE))

from verify_decode.common.diff_ops import build_pair_diff_stack, fixed_channel_difference, max_channel_difference
from verify_decode.common.fourier_score import fourier_score_map


def main() -> None:
    rng = np.random.default_rng(0)
    h, w = 32, 32
    a = rng.random((h, w, 3), dtype=np.float32) * 0.2 + 0.4
    b = a.copy()
    # B にだけ大きな変化、R に小さいノイズ
    b[:, :, 2] += 0.05
    b[:, :, 0] += rng.normal(0, 0.001, size=(h, w)).astype(np.float32)

    d_max = max_channel_difference(a, b)
    d_b = fixed_channel_difference(a, b, "B")
    assert d_max.shape == d_b.shape == (h, w)
    # 多くの画素で B 固定と max は近いが、完全一致しないノイズ条件もあり得る
    print(f"[OK] max mean={d_max.mean():.5f} matchedB mean={d_b.mean():.5f}")

    t = 64
    stack = np.zeros((t, h, w), dtype=np.float32)
    tt = np.arange(t, dtype=np.float32) / 60.0
    stack += (0.02 * np.sin(2 * np.pi * 15.0 * tt))[:, None, None]
    stack += rng.normal(0, 0.01, size=stack.shape).astype(np.float32)

    r = fourier_score_map(stack, 60.0, 15.0, score_mode="ratio")
    am = fourier_score_map(stack, 60.0, 15.0, score_mode="amp")
    assert r.shape == am.shape == (h, w)
    assert not np.allclose(r, am), "ratio と amp は異なるスコアであるべき"
    print(f"[OK] ratio mean={r.mean():.5f} amp mean={am.mean():.5f}")

    rgb_list = [a, b, a, b]
    s_max = build_pair_diff_stack(rgb_list, channel=None)
    s_b = build_pair_diff_stack(rgb_list, channel="B")
    assert s_max.shape[0] == 3 and s_b.shape[0] == 3
    print("[OK] selfcheck passed")


if __name__ == "__main__":
    main()
