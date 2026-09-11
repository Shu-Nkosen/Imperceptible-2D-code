"""地図スイープ → QR decode（検証用）。"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from decode_qr_from_all_frames import decode_qr_with_kernel_candidates_from_array
from decode_qr_from_all_frames import MID_MEDIAN_KERNELS, DEFAULT_MEDIAN_ITERATIONS


def try_decode_rgb(rgb: np.ndarray, search_mode: str = "mid") -> Tuple[bool, str]:
    text, method, _ = decode_qr_with_kernel_candidates_from_array(
        rgb,
        kernel_candidates=list(MID_MEDIAN_KERNELS),
        median_iterations=DEFAULT_MEDIAN_ITERATIONS,
        search_mode=search_mode,
    )
    ok = bool(text)
    return ok, (method if ok else "decode失敗")


def sweep_decode(
    maps: Sequence[Tuple[str, np.ndarray, dict]],
    *,
    search_mode: str = "mid",
    stop_on_success: bool = True,
) -> Dict[str, object]:
    """maps を順に decode。any-success と最初の成功メタを返す。"""
    best: Dict[str, object] = {
        "decode_success": 0,
        "decode_method": "",
        "map_id": "",
        "n_tried": 0,
    }
    for map_id, rgb, meta in maps:
        best["n_tried"] = int(best["n_tried"]) + 1
        ok, method = try_decode_rgb(rgb, search_mode=search_mode)
        if ok:
            best.update(
                {
                    "decode_success": 1,
                    "decode_method": method,
                    "map_id": map_id,
                    **{k: v for k, v in meta.items()},
                }
            )
            if stop_on_success:
                return best
    return best


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_any_success(
    rows: Iterable[dict],
    group_keys: Sequence[str],
    success_key: str = "decode_success",
) -> List[dict]:
    buckets: Dict[tuple, List[int]] = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in group_keys)
        buckets.setdefault(key, []).append(int(r.get(success_key, 0) or 0))
    out = []
    for key, vals in sorted(buckets.items()):
        n = len(vals)
        s = sum(vals)
        row = {k: v for k, v in zip(group_keys, key)}
        row.update(
            {
                "n": n,
                "success": s,
                "any_success_pct": round(100.0 * s / n, 4) if n else 0.0,
            }
        )
        out.append(row)
    return out
