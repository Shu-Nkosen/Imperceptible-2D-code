"""QR黒領域の Sobel / Laplacian を同じ画素で出す（発表スライドには載せない診断図）。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_image_channel import (  # noqa: E402
    IMAGE_NAMES,
    OUT_DIR,
    bgr_planes,
    load_prepared,
)

IMAGE_JP = {
    "ex": "実験",
    "rice": "自然",
    "hocho": "工事",
    "nagaoka_fireworks": "花火",
}
CHANNELS = ("R", "G", "B", "gray")
CH_COLOR = {"R": "#E31A1C", "G": "#1B8A3A", "B": "#1F5AA6", "gray": "#555555"}


def _metrics(plane: np.ndarray, mask: np.ndarray) -> dict:
    p = plane.astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(gx * gx + gy * gy)[mask]
    lap = cv2.Laplacian(p, cv2.CV_32F, ksize=3)[mask]
    return {
        "sobel_mean": float(sobel.mean()),
        "sobel_std": float(sobel.std()),
        "sobel_p95": float(np.percentile(sobel, 95)),
        "lap_abs_mean": float(np.abs(lap).mean()),
        "lap_var": float(lap.var()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name in IMAGE_NAMES:
        base, black_mask, x0, square = load_prepared(name)
        roi = base[:, x0 : x0 + square, :]
        planes = bgr_planes(roi)
        gray = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        for ch in CHANNELS:
            plane = gray if ch == "gray" else planes[ch]
            m = _metrics(plane, black_mask)
            rows.append({"image": name, "image_jp": IMAGE_JP[name], "channel": ch, **m})

    csv_path = OUT_DIR / "sobel_laplacian.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    labels = [IMAGE_JP[n] for n in IMAGE_NAMES]
    x = np.arange(len(IMAGE_NAMES))
    width = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    specs = (
        ("sobel_mean", "Sobel 平均 |∇|", axes[0]),
        ("lap_abs_mean", "Laplacian 平均 |L|", axes[1]),
        ("lap_var", "Laplacian 分散", axes[2]),
    )
    for key, title, ax in specs:
        for i, ch in enumerate(CHANNELS):
            vals = [next(r[key] for r in rows if r["image"] == img and r["channel"] == ch) for img in IMAGE_NAMES]
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width,
                label=ch,
                color=CH_COLOR[ch],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=4, loc="upper left", fontsize=8)
    axes[0].set_ylabel("QR黒画素")
    fig.suptitle("QR黒領域の空間勾配（埋め込みと同じマスク・preclip）", fontsize=12)
    fig.tight_layout()
    fig_path = OUT_DIR / "bar_sobel_laplacian.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(csv_path)
    print(fig_path)
    for r in rows:
        print(
            f"{r['image_jp']:4} {r['channel']:4}  "
            f"Sobel {r['sobel_mean']:6.1f}  "
            f"|L| {r['lap_abs_mean']:6.1f}  "
            f"var(L) {r['lap_var']:8.1f}"
        )


if __name__ == "__main__":
    main()
