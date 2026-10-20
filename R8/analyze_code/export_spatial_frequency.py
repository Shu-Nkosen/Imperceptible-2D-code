# -*- coding: utf-8 -*-
"""元画像4枚の空間周波数（発表Q&A用）。QR配置の中央正方形・輝度。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_image_channel import IMAGE_NAMES, OUT_DIR, load_prepared
from family_labels import IMAGE_LABEL

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# QR 1セル = 32 px → 基本空間周波数
F_QR = 1.0 / 32.0
BANDS = (
    ("low_<1/64", 0.0, 1.0 / 64.0),
    ("mid_1/64-1/16", 1.0 / 64.0, 1.0 / 16.0),
    ("high_>1/16", 1.0 / 16.0, 0.5),
    ("above_qr_>1/32", F_QR, 0.5),
)


def luminance(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    return 0.114 * b + 0.587 * g + 0.299 * r


def radial_psd(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = gray.shape[0]
    w = np.hanning(n)
    win = np.outer(w, w)
    x = (gray - gray.mean()) * win
    spec = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(spec) ** 2
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(yy - cy, xx - cx)
    freq = rr / n  # cycles/pixel
    r_int = np.floor(rr).astype(int)
    r_max = int(r_int.max())
    radial = np.zeros(r_max + 1, dtype=np.float64)
    count = np.zeros(r_max + 1, dtype=np.float64)
    np.add.at(radial, r_int.ravel(), power.ravel())
    np.add.at(count, r_int.ravel(), 1.0)
    count = np.maximum(count, 1.0)
    radial /= count
    f = np.arange(r_max + 1) / n
    keep = f <= 0.5
    return f[keep], radial[keep]


def band_energy(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    m = (f > lo) & (f <= hi) if lo > 0 else (f > 0) & (f <= hi)
    return float(psd[m].sum()) if np.any(m) else 0.0


def centroid(f: np.ndarray, psd: np.ndarray, fmin: float = 0.0) -> float:
    m = f > max(fmin, 0.0)
    w = psd[m]
    ff = f[m]
    s = w.sum()
    return float((ff * w).sum() / s) if s > 0 else 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    spectra = {}
    for name in IMAGE_NAMES:
        base, _mask, x0, square = load_prepared(name)
        roi = base[:, x0 : x0 + square, :]
        gray = luminance(roi)
        f, psd = radial_psd(gray)
        spectra[name] = (f, psd)
        total = float(psd[f > 0].sum())
        c = centroid(f, psd)
        c_hf = centroid(f, psd, fmin=F_QR)
        rec = {
            "image": name,
            "label": IMAGE_LABEL[name],
            "size": square,
            "centroid_cyc_per_px": c,
            "centroid_period_px": (1.0 / c) if c > 0 else float("nan"),
            "centroid_above_qr_cyc_per_px": c_hf,
        }
        for key, lo, hi in BANDS:
            e = band_energy(f, psd, lo, hi)
            rec[f"frac_{key}"] = e / total if total else 0.0
        rows.append(rec)

    path = OUT_DIR / "spatial_frequency.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = {
        "ex": "#2f6fed",
        "rice": "#2e7d32",
        "hocho": "#c62828",
        "nagaoka_fireworks": "#212121",
    }
    for name in IMAGE_NAMES:
        f, psd = spectra[name]
        ax.plot(f[1:], psd[1:], color=colors[name], lw=1.8, label=IMAGE_LABEL[name])
    ax.axvline(F_QR, color="#888888", ls="--", lw=1.0, label="QRセル 1/32 px")
    ax.set_yscale("log")
    ax.set_xlim(0.002, 0.25)
    ax.set_xlabel("空間周波数 [cycles/pixel]")
    ax.set_ylabel("放射平均パワー（相対・対数）")
    ax.set_title("元画像の空間周波数（中央1080×1080・輝度）")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_frequency_psd.png", dpi=160)
    plt.close(fig)

    print("wrote", path)
    for r in rows:
        print(
            f"{r['label']:12s}  centroid={r['centroid_cyc_per_px']:.4f} cyc/px "
            f"({r['centroid_period_px']:.1f} px)  "
            f"low={r['frac_low_<1/64']:.3f}  mid={r['frac_mid_1/64-1/16']:.3f}  "
            f"high={r['frac_high_>1/16']:.3f}  >QR={r['frac_above_qr_>1/32']:.3f}"
        )


if __name__ == "__main__":
    main()
