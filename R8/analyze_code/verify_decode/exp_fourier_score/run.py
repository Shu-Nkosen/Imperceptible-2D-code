"""実験B: 時間FFT score=ratio vs amp（割りなし）。差分は max-channel。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_ANALYZE = Path(__file__).resolve().parents[2]
if str(_ANALYZE) not in sys.path:
    sys.path.insert(0, str(_ANALYZE))

from verify_decode.common.decode_sweep import aggregate_any_success, sweep_decode, write_csv
from verify_decode.common.diff_ops import build_pair_diff_stack
from verify_decode.common.io_frames import count_frame_dirs, discover_conditions, load_rgb_frames
from verify_decode.common.maps import fourier_maps, lockin_maps, target_freqs_for_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="検証B: fourier ratio vs amp。結果は --out-root のみへ。"
    )
    p.add_argument("--in-root", type=str, default="out_mid_fast_0805")
    p.add_argument("--out-root", type=str, default="out_verify_decode/fourier_score")
    p.add_argument("--rates", type=str, default="45,60,90")
    p.add_argument("--intensities", type=str, default="12")
    p.add_argument("--tokens", type=str, default="R,G,B")
    p.add_argument("--limit-stems", type=int, default=0)
    p.add_argument("--limit-folders", type=int, default=0)
    p.add_argument("--min-frames", type=int, default=120)
    p.add_argument("--search-mode", type=str, default="mid")
    p.add_argument(
        "--with-lockin",
        action="store_true",
        help="対照として同じ d(t) で lockin も1回走らせる",
    )
    return p.parse_args()


def _ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def run_one(cond, *, search_mode: str, with_lockin: bool) -> List[dict]:
    rgb = load_rgb_frames(cond.frame_dir)
    d_stack = build_pair_diff_stack(rgb, channel=None)  # max
    freqs = target_freqs_for_rate(cond.rate_hz, cond.camera_fps)
    rows: List[dict] = []

    for score_mode in ("ratio", "amp"):
        maps = fourier_maps(
            d_stack,
            cond.camera_fps,
            freqs,
            score_mode=score_mode,
        )
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
                "diff_mode": "max",
                "method": "fourier",
                "fourier_score_mode": score_mode,
                "camera_fps": cond.camera_fps,
                "decode_success": result.get("decode_success", 0),
                "decode_method": result.get("decode_method", ""),
                "map_id": result.get("map_id", ""),
                "fft_target_hz": result.get("fft_target_hz", ""),
                "diff_threshold": result.get("diff_threshold", ""),
                "n_tried": result.get("n_tried", 0),
            }
        )

    if with_lockin:
        maps = lockin_maps(d_stack, cond.camera_fps, freqs)
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
                "diff_mode": "max",
                "method": "lockin",
                "fourier_score_mode": "",
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
            " verify_decode.make_fixture か、フレーム付き --in-root を使ってください。"
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
    print(f"[INFO] conditions={len(conds)} with_lockin={ns.with_lockin}")
    if not conds:
        write_csv(out_root / "summary.csv", [])
        write_csv(out_root / "by_image_channel.csv", [])
        raise SystemExit(2)

    all_rows: List[dict] = []
    for i, cond in enumerate(conds, 1):
        print(f"[{i}/{len(conds)}] {cond.stem}/{cond.folder}")
        all_rows.extend(
            run_one(cond, search_mode=ns.search_mode, with_lockin=bool(ns.with_lockin))
        )

    write_csv(out_root / "runs.csv", all_rows)
    write_csv(out_root / "summary.csv", all_rows)
    fourier_rows = [r for r in all_rows if r.get("method") == "fourier"]
    write_csv(
        out_root / "by_image_channel.csv",
        aggregate_any_success(
            fourier_rows,
            ["fourier_score_mode", "image", "token"],
        ),
    )
    write_csv(
        out_root / "by_channel.csv",
        aggregate_any_success(
            fourier_rows,
            ["fourier_score_mode", "token"],
        ),
    )
    write_csv(
        out_root / "by_score_mode.csv",
        aggregate_any_success(fourier_rows, ["fourier_score_mode"]),
    )
    print(f"[OK] wrote {out_root.resolve()}")


if __name__ == "__main__":
    main()
