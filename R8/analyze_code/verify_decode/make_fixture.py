"""合成点滅フレーム（スモーク用）。本体データは使わない。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_qr_like_mask(h: int, w: int, modules: int = 21) -> np.ndarray:
    """簡易チェッカー + 角ファインダ風マスク (H,W) bool。"""
    yy, xx = np.mgrid[0:h, 0:w]
    cell = max(4, min(h, w) // modules)
    checker = ((xx // cell) + (yy // cell)) % 2 == 0
    mask = checker.copy()
    # finder-ish blocks
    for y0, x0 in ((0, 0), (0, w - 7 * cell), (h - 7 * cell, 0)):
        mask[y0 : y0 + 7 * cell, x0 : x0 + 7 * cell] = True
        mask[y0 + cell : y0 + 6 * cell, x0 + cell : x0 + 6 * cell] = False
        mask[y0 + 2 * cell : y0 + 5 * cell, x0 + 2 * cell : x0 + 5 * cell] = True
    return mask


def write_blink_condition(
    out_dir: Path,
    *,
    channel: str,
    n_frames: int = 120,
    fps: float = 60.0,
    blink_hz: float = 22.5,
    h: int = 160,
    w: int = 160,
    amp: float = 0.08,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ch_i = {"R": 0, "G": 1, "B": 2}[channel.upper()]
    mask = make_qr_like_mask(h, w)
    base = np.full((h, w, 3), 0.55, dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float32) / float(fps)
    phase = (np.sin(2.0 * np.pi * blink_hz * t) >= 0).astype(np.float32)
    for i in range(n_frames):
        frame = base.copy()
        delta = amp if phase[i] > 0.5 else -amp
        frame[:, :, ch_i] = np.clip(frame[:, :, ch_i] + mask * delta, 0.0, 1.0)
        bgr = cv2.cvtColor((frame * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"frame_{i:05d}.png"), bgr)


def main() -> None:
    p = argparse.ArgumentParser(description="検証用合成フレームを out 配下に作る")
    p.add_argument(
        "--out-root",
        type=str,
        default="out_verify_decode/_fixture",
        help="stem/folder 構造のルート",
    )
    p.add_argument("--n-frames", type=int, default=120)
    ns = p.parse_args()
    root = Path(ns.out_root)
    # 45Hz表示 → 点滅想定 22.5Hz（カメラ60fps）
    specs = [
        ("r45_e125_f0", "ex_R_12", "R", 22.5),
        ("r45_e125_f0", "ex_G_12", "G", 22.5),
        ("r45_e125_f0", "ex_B_12", "B", 22.5),
        ("r45_e125_f0", "hocho_B_12", "B", 22.5),
    ]
    for stem, folder, ch, hz in specs:
        write_blink_condition(
            root / stem / folder,
            channel=ch,
            n_frames=int(ns.n_frames),
            blink_hz=hz,
        )
    # camera_fps を CSV から読めるようダミー sweeps を置く
    sweeps = root / "r45_e125_f0" / "results_sweeps_lockin.csv"
    sweeps.write_text(
        "folder,camera_fps,fft_target_hz\nex_R_12,60.0 fps,22.5\n",
        encoding="utf-8",
    )
    print(f"[OK] fixture written under {root.resolve()}")


if __name__ == "__main__":
    main()
