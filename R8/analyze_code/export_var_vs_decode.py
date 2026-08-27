"""QR黒の RGB 分散（画素 / Laplacian）と埋め込みチャネルの復号成功率を突き合わせる。"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
STUDY = HERE / "out_image_channel_study"
KEEP = {45, 60, 90}
FAMS = ("pair", "accum", "lockin", "fourier")
IMAGE_JP = {"ex": "実験", "rice": "自然", "hocho": "工事", "nagaoka_fireworks": "花火"}
FAM_JP = {"pair": "差分", "accum": "積算", "lockin": "同期検波", "fourier": "FFT"}
CH_COLOR = {"R": "#E31A1C", "G": "#1B8A3A", "B": "#1F5AA6"}
MARK = {"ex": "o", "rice": "s", "hocho": "^", "nagaoka_fireworks": "D"}


def _load_ix():
    spec = importlib.util.spec_from_file_location("ix", HERE / "export_intensity_cross_figs.py")
    ix = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ix)
    return ix


def spearman(xs: list[float], ys: list[float]) -> float:
    a = np.array(xs, dtype=float)
    b = np.array(ys, dtype=float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    pixel: dict[tuple[str, str], dict] = {}
    with (STUDY / "color_stats.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["region"] != "qr_black" or row["channel"] not in ("R", "G", "B"):
                continue
            std = float(row["std"])
            pixel[(row["image"], row["channel"])] = {
                "mean": float(row["mean"]),
                "std": std,
                "var": std * std,
            }
    lap: dict[tuple[str, str], dict] = {}
    with (STUDY / "sobel_laplacian.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["channel"] not in ("R", "G", "B"):
                continue
            lap[(row["image"], row["channel"])] = {
                "sobel": float(row["sobel_mean"]),
                "lap_abs": float(row["lap_abs_mean"]),
                "lap_var": float(row["lap_var"]),
            }

    ix = _load_ix()
    family_rows = ix.load_family_rows()
    keys = [(img, ch) for img in IMAGE_JP for ch in ("R", "G", "B")]
    success: dict[tuple[str, str], dict[str, tuple[int, int]]] = {k: {} for k in keys}
    for fam in FAMS:
        for r in family_rows[fam]:
            if r["rate"] not in KEEP or r["channel"] not in ("R", "G", "B"):
                continue
            k = (r["image"], r["channel"])
            n_ok, n = success[k].get(fam, (0, 0))
            success[k][fam] = (n_ok + int(r["any_ok"]), n + 1)

    out_rows = []
    for img, ch in keys:
        row = {
            "image": img,
            "image_jp": IMAGE_JP[img],
            "channel": ch,
            "mean": pixel[(img, ch)]["mean"],
            "pixel_var": pixel[(img, ch)]["var"],
            "pixel_std": pixel[(img, ch)]["std"],
            "sobel": lap[(img, ch)]["sobel"],
            "lap_var": lap[(img, ch)]["lap_var"],
        }
        for fam in FAMS:
            n_ok, n = success[(img, ch)][fam]
            row[f"{fam}_ok"] = n_ok
            row[f"{fam}_n"] = n
            row[f"{fam}_pct"] = 100.0 * n_ok / n
        out_rows.append(row)

    csv_path = STUDY / "var_vs_decode.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("Spearman（12点 = 4画像×RGB）分散 vs 成功率")
    for fam in FAMS:
        ys = [r[f"{fam}_pct"] for r in out_rows]
        rp = spearman([r["pixel_var"] for r in out_rows], ys)
        rl = spearman([r["lap_var"] for r in out_rows], ys)
        print(f"  {FAM_JP[fam]:6}  画素分散 ρ={rp:+.2f}   Laplacian分散 ρ={rl:+.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    for ax, xkey, xlabel in (
        (axes[0], "pixel_var", "QR黒 画素分散 (std²)"),
        (axes[1], "lap_var", "QR黒 Laplacian 分散"),
    ):
        for r in out_rows:
            ax.scatter(
                r[xkey],
                r["lockin_pct"],
                s=90,
                c=CH_COLOR[r["channel"]],
                marker=MARK[r["image"]],
                edgecolors="0.2",
                linewidths=0.6,
                zorder=3,
            )
            if r["image"] == "rice" and r["channel"] == "B":
                ax.annotate(
                    "自然B",
                    (r[xkey], r["lockin_pct"]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=10,
                    color=CH_COLOR["B"],
                )
            if r["image"] == "hocho" and r["channel"] == "B":
                ax.annotate(
                    "工事B",
                    (r[xkey], r["lockin_pct"]),
                    textcoords="offset points",
                    xytext=(8, -12),
                    fontsize=10,
                    color=CH_COLOR["B"],
                )
        ax.set_xlabel(xlabel)
        ax.set_title("同期検波 成功率（45/60/90、各 n=54）")
    axes[0].set_ylabel("同期検波 成功率（%）")
    from matplotlib.lines import Line2D

    ch_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CH_COLOR[c], markersize=8, label=c) for c in "RGB"]
    im_h = [
        Line2D([0], [0], marker=MARK[k], color="0.3", linestyle="None", markersize=8, label=IMAGE_JP[k])
        for k in IMAGE_JP
    ]
    axes[1].legend(handles=ch_h + im_h, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("分散と埋め込みチャネルの復号 — 自然Bは画素分散が大きくても読めない", fontsize=12)
    fig.tight_layout()
    fig_path = STUDY / "scatter_var_vs_lockin.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    imgs = list(IMAGE_JP)
    x = np.arange(len(imgs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, ch in enumerate("RGB"):
        vals = [pixel[(img, ch)]["var"] for img in imgs]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=ch,
            color=CH_COLOR[ch],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([IMAGE_JP[k] for k in imgs])
    ax.set_ylabel("画素分散 (std²)")
    ax.set_title("QR黒領域の R/G/B 画素分散")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    bar_path = STUDY / "bar_rgb_pixel_var.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    print("\n画像×チャネル（画素分散 / Lap分散 / 同期検波%）")
    print(f"{'':6} {'画素var':>8} {'Lap var':>8} {'差分':>6} {'積算':>6} {'同期':>6} {'FFT':>6}")
    for r in out_rows:
        print(
            f"{r['image_jp']}{r['channel']:2}  {r['pixel_var']:8.0f} {r['lap_var']:8.0f}  "
            f"{r['pair_pct']:5.1f} {r['accum_pct']:5.1f} {r['lockin_pct']:5.1f} {r['fourier_pct']:5.1f}"
        )
    print(csv_path)
    print(fig_path)
    print(bar_path)


if __name__ == "__main__":
    main()
