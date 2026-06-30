from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft


matplotlib.use("Agg")


@dataclass
class DetectionResult:
    fps: float
    total_frames: int
    used_frames: int
    target_freq: float
    target_idx: int
    freq_at_target: float
    score_map: np.ndarray
    amplitude_map: np.ndarray
    mask: np.ndarray


def load_video_frames(video_path: Path, max_seconds: Optional[float] = None, fps_override: Optional[float] = None) -> Tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps_override is not None and fps_override > 0:
        fps = fps_override
    if fps <= 0:
        raise ValueError("FPS を取得できませんでした。--fps で指定してください。")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames if total_frames > 0 else None
    if max_seconds is not None and max_seconds > 0:
        limit = max(1, int(round(max_seconds * fps)))
        max_frames = limit if max_frames is None else min(max_frames, limit)

    frames = []
    count = 0
    while True:
        if max_frames is not None and count >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        frames.append(gray.astype(np.float32) / 255.0)
        count += 1

    capture.release()

    if not frames:
        raise ValueError("動画からフレームを読み込めませんでした。")

    return np.stack(frames, axis=0), fps, total_frames


def build_demo_frames(fps: float, seconds: float, height: int, width: int, target_freq: float) -> np.ndarray:
    frame_count = max(2, int(round(fps * seconds)))
    time = np.arange(frame_count, dtype=np.float32) / fps
    background = np.full((frame_count, height, width), 0.25, dtype=np.float32)
    noise = 0.03 * np.random.default_rng(1).standard_normal((frame_count, height, width)).astype(np.float32)

    blink = (np.sin(2.0 * np.pi * target_freq * time) > 0).astype(np.float32)
    blink = blink * 0.7

    region = np.zeros((height, width), dtype=np.float32)
    y0 = height // 4
    y1 = y0 + height // 3
    x0 = width // 4
    x1 = x0 + width // 3
    region[y0:y1, x0:x1] = 1.0

    frames = background + noise + blink[:, None, None] * region[None, :, :]
    return np.clip(frames, 0.0, 1.0)


def detrend_frames(frames: np.ndarray) -> np.ndarray:
    return frames - frames.mean(axis=0, keepdims=True)


def apply_window(frames: np.ndarray, enabled: bool = True) -> np.ndarray:
    if not enabled or frames.shape[0] < 2:
        return frames
    window = np.hanning(frames.shape[0]).astype(np.float32)[:, None, None]
    return frames * window


def compute_frequency_spectrum(frames: np.ndarray, fps: float, use_window: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    prepared = detrend_frames(frames)
    prepared = apply_window(prepared, enabled=use_window)

    spectrum = fft.fft(prepared, axis=0, workers=-1)
    freqs = fft.fftfreq(prepared.shape[0], d=1.0 / fps)
    positive = freqs >= 0
    return freqs[positive], np.abs(spectrum[positive])


def resolve_target_band(freqs: np.ndarray, target_freq: float, band_radius: int = 1) -> Tuple[int, slice]:
    target_idx = int(np.argmin(np.abs(freqs - target_freq)))
    start = max(0, target_idx - band_radius)
    stop = min(len(freqs), target_idx + band_radius + 1)
    return target_idx, slice(start, stop)


def build_detection_result(frames: np.ndarray, fps: float, target_freq: float, band_radius: int, threshold: float, use_window: bool = True) -> DetectionResult:
    freqs, amplitude = compute_frequency_spectrum(frames, fps, use_window=use_window)
    target_idx, band = resolve_target_band(freqs, target_freq, band_radius=band_radius)

    target_amplitude = amplitude[band].max(axis=0)
    noise_floor = amplitude[1:].mean(axis=0) if amplitude.shape[0] > 1 else amplitude[0]
    score_map = target_amplitude / (noise_floor + 1e-12)
    mask = score_map >= threshold

    return DetectionResult(
        fps=fps,
        total_frames=frames.shape[0],
        used_frames=frames.shape[0],
        target_freq=target_freq,
        target_idx=target_idx,
        freq_at_target=float(freqs[target_idx]),
        score_map=score_map,
        amplitude_map=target_amplitude,
        mask=mask,
    )


def normalize_map(values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def save_heatmap(values: np.ndarray, output_path: Path, title: str, cmap: str = "hot") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    image = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_path), image)


def save_summary_csv(output_path: Path, result: DetectionResult, threshold: float, band_radius: int) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fps",
                "total_frames",
                "used_frames",
                "target_freq",
                "target_idx",
                "freq_at_target",
                "band_radius",
                "threshold",
                "mask_pixels",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "fps": f"{result.fps:.6f}",
                "total_frames": result.total_frames,
                "used_frames": result.used_frames,
                "target_freq": f"{result.target_freq:.6f}",
                "target_idx": result.target_idx,
                "freq_at_target": f"{result.freq_at_target:.6f}",
                "band_radius": band_radius,
                "threshold": f"{threshold:.6f}",
                "mask_pixels": int(result.mask.sum()),
            }
        )


def run_pipeline(frames: np.ndarray, fps: float, target_freq: float, threshold: float, band_radius: int, use_window: bool = True) -> DetectionResult:
    return build_detection_result(frames, fps, target_freq, band_radius, threshold, use_window=use_window)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="時間軸FFTで特定周波数の点滅領域を検出する。")
    parser.add_argument("video", nargs="?", help="入力動画ファイルのパス")
    parser.add_argument("--output-dir", default="R8_output", help="出力先ディレクトリ")
    parser.add_argument("--target-freq", type=float, default=20.0, help="検出したい周波数(Hz)")
    parser.add_argument("--threshold", type=float, default=12.0, help="マスク化のしきい値")
    parser.add_argument("--band-radius", type=int, default=1, help="target_idx 前後に含めるセル数")
    parser.add_argument("--clip-seconds", type=float, default=5.0, help="先頭から切り出す秒数")
    parser.add_argument("--fps", type=float, default=None, help="FPSが取れない場合の上書き値")
    parser.add_argument("--no-window", action="store_true", help="窓関数を使わない")
    parser.add_argument("--demo", action="store_true", help="合成データで動作確認する")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        fps = 60.0 if args.fps is None else float(args.fps)
        frames = build_demo_frames(fps=fps, seconds=args.clip_seconds, height=160, width=160, target_freq=args.target_freq)
        video_name = "demo"
        detected_fps = fps
        total_frames = frames.shape[0]
    else:
        if not args.video:
            raise SystemExit("動画ファイルを指定してください。--demo で合成データも試せます。")
        video_path = Path(args.video)
        frames, detected_fps, total_frames = load_video_frames(video_path, max_seconds=args.clip_seconds, fps_override=args.fps)
        video_name = video_path.stem

    result = run_pipeline(
        frames=frames,
        fps=detected_fps,
        target_freq=args.target_freq,
        threshold=args.threshold,
        band_radius=args.band_radius,
        use_window=not args.no_window,
    )

    stem = f"{video_name}_fft_{int(round(args.target_freq * 100)):04d}mHz"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    mask_path = output_dir / f"{stem}_mask.png"
    summary_path = output_dir / f"{stem}_summary.csv"

    save_heatmap(
        normalize_map(result.score_map),
        heatmap_path,
        title=(
            f"Time-axis FFT score map | fps={result.fps:.2f} | target={result.target_freq:.2f} Hz | "
            f"freq_bin={result.freq_at_target:.2f} Hz"
        ),
        cmap="hot",
    )
    save_mask(result.mask, mask_path)
    save_summary_csv(summary_path, result, threshold=args.threshold, band_radius=args.band_radius)

    print(f"output_dir: {output_dir.resolve()}")
    print(f"fps: {result.fps:.3f}")
    print(f"frames: {result.used_frames}/{total_frames}")
    print(f"target_idx: {result.target_idx}")
    print(f"freq_at_target: {result.freq_at_target:.3f} Hz")
    print(f"heatmap: {heatmap_path.name}")
    print(f"mask: {mask_path.name}")
    print(f"summary: {summary_path.name}")


if __name__ == "__main__":
    main()