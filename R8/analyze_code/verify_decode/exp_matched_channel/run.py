"""実験A: max-channel vs 埋め込みチャネル固定差分。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# analyze_code を cwd 前提で import
_ANALYZE = Path(__file__).resolve().parents[2]
if str(_ANALYZE) not in sys.path:
    sys.path.insert(0, str(_ANALYZE))

from verify_decode.common.decode_sweep import aggregate_any_success, sweep_decode, write_csv
from verify_decode.common.diff_ops import build_pair_diff_stack
from verify_decode.common.io_frames import count_frame_dirs, discover_conditions, load_rgb_frames
from verify_decode.common.maps import fourier_maps, lockin_maps, pair_maps, target_freqs_for_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="検証A: max vs matched（R/G/B）。結果は --out-root のみへ。"
    )
    p.add_argument("--in-root", type=str, default="out_mid_fast_0805")
    p.add_argument("--out-root", type=str, default="out_verify_decode/matched_channel")
    p.add_argument("--rates", type=str, default="45,60,90")
    p.add_argument("--intensities", type=str, default="12")
    p.add_argument("--tokens", type=str, default="R,G,B")
    p.add_argument("--methods", type=str, default="pair,lockin,fourier")
    p.add_argument("--limit-stems", type=int, default=0)
    p.add_argument("--limit-folders", type=int, default=0)
    p.add_argument("--min-frames", type=int, default=120)
    p.add_argument("--search-mode", type=str, default="mid")
    return p.parse_args()


def _ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def run_one(
    cond,
    *,
    methods: List[str],
    search_mode: str,
) -> List[dict]:
    rgb = load_rgb_frames(cond.frame_dir)
    freqs = target_freqs_for_rate(cond.rate_hz, cond.camera_fps)
    modes = [("max", None)]
    if cond.token in ("R", "G", "B"):
        modes.append(("matched", cond.token))

    rows: List[dict] = []
    for mode_name, ch in modes:
        d_stack = build_pair_diff_stack(rgb, channel=ch)
        for method in methods:
            if method == "pair":
                maps = pair_maps(d_stack)
            elif method == "lockin":
                maps = lockin_maps(d_stack, cond.camera_fps, freqs)
            elif method == "fourier":
                maps = fourier_maps(d_stack, cond.camera_fps, freqs, score_mode="ratio")
            else:
                raise SystemExit(f"unknown method: {method}")
            result = sweep_decode(maps, search_mode=search_mode, stop_on_success=True)
            rows.append(
                {
                    "stem": cond.stem,
                    "folder": cond.folder,
                    "rate_hz": cond.rate_hz,
                    "image": cond.image,
                    "token": cond.token,
                    "channel": cond.channel,
                    "intensity": cond.intensity,
                    "diff_mode": mode_name,
                    "method": method,
                    "camera_fps": cond.camera_fps,
                    "decode_success": result.get("decode_success", 0),
                    "decode_method": result.get("decode_method", ""),
                    "map_id": result.get("map_id", ""),
                    "fft_target_hz": result.get("fft_target_hz", ""),
                    "diff_threshold": result.get("diff_threshold", ""),
                    "n_tried": result.get("n_tried", 0),
                }
            )
    return rows


def main() -> None:
    ns = parse_args()
    in_root = Path(ns.in_root)
    out_root = Path(ns.out_root)
    if not in_root.is_dir():
        raise SystemExit(f"in-root がありません: {in_root}")

    n_stems, n_folders = count_frame_dirs(in_root)
    print(f"[INFO] in-root={in_root} stems_with_meta={n_stems} folders_with_frames={n_folders}")
    if n_folders == 0:
        print(
            "[WARN] frame_*.png が見つかりません。"
            " mid_fast は keep-frames=0 で削除済みのことが多いです。"
            " verify_decode.make_fixture で合成入力を作るか、"
            " フレーム付きディレクトリを --in-root に指定してください。"
        )

    conds = discover_conditions(
        in_root,
        rates=_ints(ns.rates),
        intensities=_ints(ns.intensities),
        tokens=_strs(ns.tokens),
        limit_stems=ns.limit_stems or None,
        limit_folders=ns.limit_folders or None,
        min_frames=int(ns.min_frames),
    )
    print(f"[INFO] conditions={len(conds)} methods={ns.methods}")
    if not conds:
        write_csv(out_root / "summary.csv", [])
        write_csv(out_root / "by_channel.csv", [])
        raise SystemExit(2)

    methods = _strs(ns.methods)
    all_rows: List[dict] = []
    for i, cond in enumerate(conds, 1):
        print(f"[{i}/{len(conds)}] {cond.stem}/{cond.folder}")
        all_rows.extend(run_one(cond, methods=methods, search_mode=ns.search_mode))

    write_csv(out_root / "runs.csv", all_rows)
    # 条件×mode×method の any-success（条件単位は既に1行）
    by_ch = aggregate_any_success(
        all_rows,
        ["diff_mode", "method", "token"],
    )
    write_csv(out_root / "by_channel.csv", by_ch)
    by_img = aggregate_any_success(
        all_rows,
        ["diff_mode", "method", "image", "token"],
    )
    write_csv(out_root / "by_image_channel.csv", by_img)
    write_csv(out_root / "summary.csv", all_rows)
    print(f"[OK] wrote {out_root.resolve()}")


if __name__ == "__main__":
    main()
