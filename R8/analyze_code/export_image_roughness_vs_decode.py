"""画像単位の QR黒 模様の荒さ vs 復号成功率（発表スライドには載せない診断）。"""
from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
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
# 表の並び: 実験・自然・工事・花火
IMAGE_ORDER = ("ex", "rice", "hocho", "nagaoka_fireworks")
IMAGE_JP = {"ex": "実験", "rice": "自然", "hocho": "工事", "nagaoka_fireworks": "花火"}
FAM_JP = {"pair": "差分", "accum": "積算", "lockin": "同期検波", "fourier": "FFT"}
FAM_COLOR = {
    "pair": "#457b9d",
    "accum": "#2a9d8f",
    "lockin": "#e9c46a",
    "fourier": "#e76f51",
}
SOBEL_COLOR = "#555555"


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


def _cond_key(r: dict) -> tuple:
    return (r["stem"], r["folder"])


def main() -> None:
    gray: dict[str, dict] = {}
    with (STUDY / "sobel_laplacian.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["channel"] != "gray":
                continue
            gray[row["image"]] = {
                "sobel": float(row["sobel_mean"]),
                "lap_abs": float(row["lap_abs_mean"]),
                "lap_var": float(row["lap_var"]),
            }

    rgb_vars: dict[str, list[float]] = defaultdict(list)
    with (STUDY / "color_stats.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["region"] != "qr_black" or row["channel"] not in ("R", "G", "B"):
                continue
            std = float(row["std"])
            rgb_vars[row["image"]].append(std * std)
    rgb_var_mean = {img: float(np.mean(vs)) for img, vs in rgb_vars.items()}

    ix = _load_ix()
    family_rows = ix.load_family_rows()

    per_fam: dict[str, dict[str, list[int]]] = {img: {fam: [] for fam in FAMS} for img in IMAGE_ORDER}
    any4: dict[str, dict[tuple, int]] = {img: {} for img in IMAGE_ORDER}
    for fam in FAMS:
        for r in family_rows[fam]:
            if r["rate"] not in KEEP or r["image"] not in IMAGE_ORDER:
                continue
            per_fam[r["image"]][fam].append(int(r["any_ok"]))
            k = _cond_key(r)
            any4[r["image"]][k] = max(any4[r["image"]].get(k, 0), int(r["any_ok"]))

    out_rows = []
    for img in IMAGE_ORDER:
        row = {
            "image": img,
            "image_jp": IMAGE_JP[img],
            "sobel_gray": gray[img]["sobel"],
            "lap_abs_gray": gray[img]["lap_abs"],
            "rgb_pixel_var_mean": rgb_var_mean[img],
        }
        for fam in FAMS:
            vals = per_fam[img][fam]
            n_ok = sum(vals)
            n = len(vals)
            row[f"{fam}_ok"] = n_ok
            row[f"{fam}_n"] = n
            row[f"{fam}_pct"] = 100.0 * n_ok / n if n else float("nan")
        avals = list(any4[img].values())
        row["any4_ok"] = sum(avals)
        row["any4_n"] = len(avals)
        row["any4_pct"] = 100.0 * sum(avals) / len(avals) if avals else float("nan")
        out_rows.append(row)

    csv_path = STUDY / "image_roughness_vs_decode.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("画像単位（45/60/90, talk any_ok, accum window_n=5）")
    print(
        f"{'画像':4} {'Sobel灰':>8} {'画素var':>9} {'|L|灰':>7}  "
        f"{'差分':>6} {'積算':>6} {'同期検波':>8} {'FFT':>6} {'4手法どれか':>10}   n"
    )
    for r in out_rows:
        print(
            f"{r['image_jp']:4} {r['sobel_gray']:8.2f} {r['rgb_pixel_var_mean']:9.0f} "
            f"{r['lap_abs_gray']:7.2f}  "
            f"{r['pair_pct']:5.1f}% {r['accum_pct']:5.1f}% {r['lockin_pct']:7.1f}% "
            f"{r['fourier_pct']:5.1f}% {r['any4_pct']:9.1f}%  "
            f"{r['pair_n']}/{r['any4_n']}"
        )

    print("\nSpearman（4画像、荒いほど読めないなら負）")
    xs_s = [r["sobel_gray"] for r in out_rows]
    xs_v = [r["rgb_pixel_var_mean"] for r in out_rows]
    for fam in FAMS:
        ys = [r[f"{fam}_pct"] for r in out_rows]
        print(
            f"  {FAM_JP[fam]:6}  Sobel灰 ρ={spearman(xs_s, ys):+.2f}   "
            f"画素var平均 ρ={spearman(xs_v, ys):+.2f}"
        )
    ys = [r["any4_pct"] for r in out_rows]
    print(
        f"  4手法どれか  Sobel灰 ρ={spearman(xs_s, ys):+.2f}   "
        f"画素var平均 ρ={spearman(xs_v, ys):+.2f}"
    )

    labels = [IMAGE_JP[k] for k in IMAGE_ORDER]
    x = np.arange(len(IMAGE_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    axes[0].bar(x, xs_s, color=SOBEL_COLOR, edgecolor="white", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Sobel 平均 |∇|（灰）")
    axes[0].set_title("QR黒 模様の荒さ")
    axes[0].set_axisbelow(True)
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(xs_s):
        axes[0].text(i, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    width = 0.18
    for i, fam in enumerate(FAMS):
        vals = [r[f"{fam}_pct"] for r in out_rows]
        axes[1].bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=FAM_JP[fam],
            color=FAM_COLOR[fam],
            edgecolor="white",
            linewidth=0.4,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("復号成功率（%）")
    axes[1].set_title("45/60/90 手法別（各 n=270）")
    axes[1].set_axisbelow(True)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2, loc="upper right")

    fig.suptitle("画像単位の荒さ vs 復号 — 工事は花火より少し滑らかでも同期検波は高い", fontsize=12)
    fig.tight_layout()
    fig_path = STUDY / "bar_image_roughness_vs_decode.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(csv_path)
    print(fig_path)


if __name__ == "__main__":
    main()
