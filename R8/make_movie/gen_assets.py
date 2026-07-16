from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np


Channel = Literal["R", "G", "B", "max", "min"]


@dataclass(frozen=True)
class Args:
    images: list[str]
    intensities: list[int]
    channels: list[Channel]
    clip_margin: int
    out_dir: Path
    qr_path: Path
    width: int
    height: int
    slate_sec: float
    fps_hint: int


def parse_list(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def parse_int_list(values: str) -> list[int]:
    out: list[int] = []
    for chunk in values.split(","):
        t = chunk.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="R8 make_movie assets generator (normal/inv per channel/intensity)")
    p.add_argument("--images", type=str, default="rice,nagaoka_fireworks,hocho,ex", help="base image names (no extension), comma-separated")
    p.add_argument("--intensities", type=str, default="4,8,12", help="intensities (e.g. 4,8,12)")
    p.add_argument("--channels", type=str, default="R,G,B,max,min", help="channels: R,G,B,max,min")
    p.add_argument("--clip-margin", type=int, default=4, help="pre-clip margin applied to all pixels/channels")
    p.add_argument("--out-dir", type=str, default="", help="output directory (default: same folder as inputs)")
    p.add_argument("--qr", type=str, default="HP_QR.png", help="QR mask image path (relative to this script)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--slate-sec", type=float, default=0.5, help="slate duration (used by presenter; here only for metadata)")
    p.add_argument("--fps-hint", type=int, default=60, help="fps hint for slate metadata only")
    ns = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    qr_path = Path(ns.qr)
    if not qr_path.is_absolute():
        qr_path = script_dir / qr_path

    out_dir = Path(ns.out_dir) if ns.out_dir else script_dir
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()

    channels = [c.strip() for c in ns.channels.split(",") if c.strip()]
    parsed_channels: list[Channel] = []
    for c in channels:
        cu = c.upper()
        if cu in ("R", "G", "B"):
            parsed_channels.append(cu)  # type: ignore[arg-type]
        elif c.lower() in ("max", "min"):
            parsed_channels.append(c.lower())  # type: ignore[arg-type]
        else:
            raise SystemExit(f"unknown channel: {c} (allowed: R,G,B,max,min)")

    return Args(
        images=parse_list(ns.images),
        intensities=parse_int_list(ns.intensities),
        channels=parsed_channels,
        clip_margin=int(ns.clip_margin),
        out_dir=out_dir,
        qr_path=qr_path,
        width=int(ns.width),
        height=int(ns.height),
        slate_sec=float(ns.slate_sec),
        fps_hint=int(ns.fps_hint),
    )


def load_base_image(path: Path, width: int, height: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def load_qr_mask(path: Path, square_size: int) -> np.ndarray:
    qr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if qr is None:
        raise FileNotFoundError(str(path))
    qr = cv2.resize(qr, (square_size, square_size), interpolation=cv2.INTER_NEAREST)
    _, qr_bin = cv2.threshold(qr, 127, 255, cv2.THRESH_BINARY)
    return (qr_bin == 0)


def center_square_params(width: int, height: int) -> tuple[int, int, int]:
    square = height
    x0 = (width - square) // 2
    return x0, 0, square


def preclip(img: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return img
    return np.clip(img, float(margin), 255.0 - float(margin))


def apply_channel_delta(
    img: np.ndarray,
    black_mask: np.ndarray,
    x0: int,
    square: int,
    channel: Channel,
    delta: float,
) -> np.ndarray:
    out = img.copy()
    roi = out[:, x0 : x0 + square, :]

    if channel in ("R", "G", "B"):
        ch_idx = {"B": 0, "G": 1, "R": 2}[channel]
        plane = roi[:, :, ch_idx]
        plane[black_mask] -= delta
        roi[:, :, ch_idx] = np.clip(plane, 0, 255)
        out[:, x0 : x0 + square, :] = roi
        return out

    # max/min mode: choose per-pixel channel index from current roi (preclipped)
    if channel == "max":
        idx = np.argmax(roi, axis=2)
    elif channel == "min":
        idx = np.argmin(roi, axis=2)
    else:
        raise ValueError(channel)

    for c in range(3):
        mask = black_mask & (idx == c)
        if np.any(mask):
            roi[:, :, c][mask] -= delta
    roi = np.clip(roi, 0, 255)
    out[:, x0 : x0 + square, :] = roi
    return out


def token_for_channel(channel: Channel) -> str:
    if channel in ("R", "G", "B"):
        return channel
    if channel == "max":
        return "X"
    if channel == "min":
        return "I"
    raise ValueError(channel)


def iter_base_paths(script_dir: Path, names: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for name in names:
        path = script_dir / f"{name}.png"
        yield name, path


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x0, _, square = center_square_params(args.width, args.height)
    black_mask = load_qr_mask(args.qr_path, square_size=square)

    generated = 0
    for base_name, base_path in iter_base_paths(script_dir, args.images):
        if not base_path.exists():
            print(f"[SKIP] not found: {base_path.name}")
            continue

        base = load_base_image(base_path, width=args.width, height=args.height)
        base = preclip(base, args.clip_margin)

        for intensity in args.intensities:
            for channel in args.channels:
                token = token_for_channel(channel)

                # normal: darken (delta positive)
                normal = apply_channel_delta(base, black_mask, x0, square, channel, delta=float(intensity))
                normal_name = f"{base_name}_{intensity}_normal{token}.png"
                cv2.imwrite(str(args.out_dir / normal_name), normal.astype(np.uint8))
                generated += 1

                # inv: brighten (delta negative) => subtract(-intensity)
                inv = apply_channel_delta(base, black_mask, x0, square, channel, delta=float(-intensity))
                inv_name = f"{base_name}_{intensity}_inv{token}.png"
                cv2.imwrite(str(args.out_dir / inv_name), inv.astype(np.uint8))
                generated += 1

    # slate assets (optional but useful to keep as files)
    slate_black = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    slate_red = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    slate_red[:, :, 2] = 255
    cv2.imwrite(str(args.out_dir / "slate_black.png"), slate_black)
    cv2.imwrite(str(args.out_dir / "slate_red.png"), slate_red)

    # GT QR display: same center-square placement as embedded conditions (white bg).
    # present_session prefers this over stretching raw HP_QR.png to full screen.
    qr_gray = cv2.imread(str(args.qr_path), cv2.IMREAD_GRAYSCALE)
    if qr_gray is None:
        print(f"[WARN] could not write gt_qr_display.png (missing {args.qr_path.name})")
    else:
        qr_sq = cv2.resize(qr_gray, (square, square), interpolation=cv2.INTER_NEAREST)
        _, qr_bin = cv2.threshold(qr_sq, 127, 255, cv2.THRESH_BINARY)
        gt_display = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
        gt_display[:, x0 : x0 + square, 0] = qr_bin
        gt_display[:, x0 : x0 + square, 1] = qr_bin
        gt_display[:, x0 : x0 + square, 2] = qr_bin
        gt_path = args.out_dir / "gt_qr_display.png"
        cv2.imwrite(str(gt_path), gt_display)
        print(f"[OK] wrote {gt_path.name} (center square={square}, x0={x0})")

    print(f"[OK] generated={generated} (+ slates) out_dir={args.out_dir}")


if __name__ == "__main__":
    main()

