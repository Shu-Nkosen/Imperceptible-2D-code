"""中央の正方形領域を周期的に点滅させる表示／動画生成スクリプト

使い方例:
  python R8/generate_blink_display.py --freq 20 --fps 60 --duration 5 --outfile R8/blink20.mp4
  python R8/generate_blink_display.py --freq 20 --fps 60 --duration 10 --window

"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="中央正方形を周期的に点滅させる表示／動画生成ツール")
    p.add_argument("--freq", type=float, default=20.0, help="点滅周波数(Hz)。例:20")
    p.add_argument("--fps", type=float, default=60.0, help="出力フレームレート(fps)。推奨: >= 2*freq")
    p.add_argument("--duration", type=float, default=5.0, help="秒単位の長さ")
    p.add_argument("--width", type=int, default=640, help="出力幅ピクセル")
    p.add_argument("--height", type=int, default=480, help="出力高さピクセル")
    p.add_argument("--size-ratio", type=float, default=0.33, help="画面幅に対する正方形サイズ比 (0-1)")
    p.add_argument("--outfile", type=str, default="", help="保存する動画ファイルパス(mp4)。指定しないと保存しない")
    p.add_argument("--window", action="store_true", help="ウィンドウで表示する")
    p.add_argument("--gray", action="store_true", help="点滅をグレースケールで行う(白黒)。指定しない場合は白で点滅)")
    return p.parse_args()


def make_frame(t: float, width: int, height: int, freq: float, size_ratio: float, gray: bool) -> np.ndarray:
    # 背景は中間輝度
    bg = 64
    frame = np.full((height, width, 3), bg, dtype=np.uint8)

    # 正方形サイズ
    sq_w = int(round(width * size_ratio))
    sq_h = sq_w
    cx = width // 2
    cy = height // 2
    x0 = cx - sq_w // 2
    y0 = cy - sq_h // 2
    x1 = x0 + sq_w
    y1 = y0 + sq_h

    # 点滅信号 (矩形波)
    val = 1.0 if np.sin(2.0 * np.pi * freq * t) > 0 else 0.0

    if gray:
        color = int(round(255 * val))
        frame[y0:y1, x0:x1, :] = color
    else:
        # 白で点滅
        if val > 0:
            frame[y0:y1, x0:x1, :] = (55, 55, 55)

    return frame


def main():
    args = parse_args()
    if args.fps <= 0 or args.duration <= 0:
        print("fps と duration は正の数を指定してください", file=sys.stderr)
        raise SystemExit(1)

    total_frames = int(round(args.fps * args.duration))
    out_writer = None
    if args.outfile:
        out_path = Path(args.outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
        if not out_writer.isOpened():
            print(f"動画ファイルを作成できません: {out_path}", file=sys.stderr)
            out_writer = None

    window_name = "BlinkDisplay"
    if args.window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"生成: {total_frames} フレーム | fps={args.fps} | freq={args.freq}Hz | size_ratio={args.size_ratio}")

    for i in range(total_frames):
        t = i / args.fps
        frame = make_frame(t, args.width, args.height, args.freq, args.size_ratio, args.gray)

        if out_writer is not None:
            out_writer.write(frame)

        if args.window:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(int(round(1000.0 / args.fps))) & 0xFF
            if key == 27:  # ESC
                break

    if out_writer is not None:
        out_writer.release()
        print(f"保存しました: {out_path.resolve()}")

    if args.window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()