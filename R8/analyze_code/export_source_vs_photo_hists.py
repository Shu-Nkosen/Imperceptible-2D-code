"""元画像 PNG と撮影 JPG の RGB ヒストグラム比較を summary_slides に出す。"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.axisbelow"] = True

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out_mid_fast_0805" / "summary_slides"
SRC_DIR = ROOT.parent / "make_movie"
PIC_DIR = OUT / "pic"

IMAGE_ORDER = ["ex", "hocho", "nagaoka_fireworks", "rice"]
IMAGE_JP = {
    "ex": "実験",
    "hocho": "工事",
    "nagaoka_fireworks": "花火",
    "rice": "自然",
}
IMAGE_THEME = {
    "ex": "#2f6fed",
    "hocho": "#c62828",
    "nagaoka_fireworks": "#212121",
    "rice": "#2e7d32",
}
CH_COLORS = {"R": "#d62828", "G": "#2a9d8f", "B": "#1d3557"}


def jp_image(name: str) -> str:
    return IMAGE_JP.get(name, name)


def find_photo(name: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".JPEG", ".jpeg", ".png", ".PNG"):
        p = PIC_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def load_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def channel_stats(rgb: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, ch in enumerate("RGB"):
        vals = rgb[:, :, i].astype(np.float64).ravel()
        out[f"{ch}_mean"] = float(vals.mean())
        out[f"{ch}_std"] = float(vals.std())
        out[f"{ch}_p10"] = float(np.percentile(vals, 10))
        out[f"{ch}_p50"] = float(np.percentile(vals, 50))
        out[f"{ch}_p90"] = float(np.percentile(vals, 90))
    # rough chroma / white balance proxies
    means = np.array([out["R_mean"], out["G_mean"], out["B_mean"]])
    out["luma_mean"] = float(0.299 * means[0] + 0.587 * means[1] + 0.114 * means[2])
    out["rg_diff"] = float(means[0] - means[1])
    out["gb_diff"] = float(means[1] - means[2])
    out["rb_diff"] = float(means[0] - means[2])
    return out


def hist_density_ymax(arrays: list[np.ndarray], bins: int = 64) -> float:
    """density=True ヒストの共通縦軸上限を求める。"""
    peak = 0.0
    for arr in arrays:
        counts, _ = np.histogram(arr.ravel(), bins=bins, range=(0, 255), density=True)
        if counts.size:
            peak = max(peak, float(counts.max()))
    return peak * 1.08 if peak > 0 else 0.05


def plot_hist_on_ax(ax, rgb: np.ndarray, alpha: float = 0.45, ls: str = "-") -> None:
    for i, ch in enumerate("RGB"):
        ax.hist(
            rgb[:, :, i].ravel(),
            bins=64,
            range=(0, 255),
            histtype="stepfilled",
            alpha=alpha,
            color=CH_COLORS[ch],
            label=ch,
            density=True,
            linestyle=ls,
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not PIC_DIR.is_dir():
        raise SystemExit(f"pic dir not found: {PIC_DIR}")

    pairs: list[tuple[str, np.ndarray, np.ndarray, Path, Path]] = []
    rows: list[dict] = []
    for name in IMAGE_ORDER:
        src_path = SRC_DIR / f"{name}.png"
        pic_path = find_photo(name)
        if not src_path.exists() or pic_path is None:
            print(f"[WARN] skip {name}: src={src_path.exists()} pic={pic_path}")
            continue
        src = load_rgb(src_path)
        pic = load_rgb(pic_path)
        if src is None or pic is None:
            print(f"[WARN] read fail {name}")
            continue
        pairs.append((name, src, pic, src_path, pic_path))
        ss = channel_stats(src)
        ps = channel_stats(pic)
        row = {
            "image": name,
            "image_ja": jp_image(name),
            "src_path": str(src_path),
            "pic_path": str(pic_path),
            "src_h": src.shape[0],
            "src_w": src.shape[1],
            "pic_h": pic.shape[0],
            "pic_w": pic.shape[1],
        }
        for k, v in ss.items():
            row[f"src_{k}"] = f"{v:.4f}"
        for k, v in ps.items():
            row[f"pic_{k}"] = f"{v:.4f}"
        for ch in "RGB":
            row[f"delta_{ch}_mean"] = f"{ps[f'{ch}_mean'] - ss[f'{ch}_mean']:.4f}"
        row["delta_luma_mean"] = f"{ps['luma_mean'] - ss['luma_mean']:.4f}"
        rows.append(row)

    if not pairs:
        raise SystemExit("no image pairs loaded")

    csv_path = OUT / "source_vs_photo_color.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # --- photo-only histograms (08相当) ---
    pic_ymax = hist_density_ymax([pic for _n, _s, pic, *_ in pairs])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (name, _src, pic, *_rest) in zip(axes.ravel(), pairs):
        plot_hist_on_ax(ax, pic)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, pic_ymax)
        ax.set_xlabel("輝度")
        ax.set_ylabel("密度")
        ax.set_title(jp_image(name), color=IMAGE_THEME[name], fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(IMAGE_THEME[name])
            spine.set_linewidth(2.0)
    fig.suptitle("撮影画像のRGBヒストグラム（全体画素 / pic/*.JPG）", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "16_photo_rgb_histograms.png")
    plt.close(fig)

    # --- overlay comparison per channel ---
    cmp_arrays: list[np.ndarray] = []
    for _name, src, pic, *_rest in pairs:
        for ci in range(3):
            cmp_arrays.append(src[:, :, ci])
            cmp_arrays.append(pic[:, :, ci])
    cmp_ymax = hist_density_ymax(cmp_arrays)

    fig, axes = plt.subplots(4, 3, figsize=(12.5, 12.5), sharex=True, sharey=True)
    for ri, (name, src, pic, *_rest) in enumerate(pairs):
        for ci, ch in enumerate("RGB"):
            ax = axes[ri, ci]
            ax.hist(
                src[:, :, ci].ravel(),
                bins=64,
                range=(0, 255),
                histtype="stepfilled",
                alpha=0.35,
                color="#457b9d",
                label="元画像",
                density=True,
            )
            ax.hist(
                pic[:, :, ci].ravel(),
                bins=64,
                range=(0, 255),
                histtype="stepfilled",
                alpha=0.40,
                color=CH_COLORS[ch],
                label="撮影",
                density=True,
            )
            ax.set_xlim(0, 255)
            ax.set_ylim(0, cmp_ymax)
            if ri == 0:
                ax.set_title(ch, color=CH_COLORS[ch], fontweight="bold", fontsize=12)
            if ci == 0:
                ax.set_ylabel(jp_image(name), color=IMAGE_THEME[name], fontweight="bold")
            if ri == len(pairs) - 1:
                ax.set_xlabel("輝度")
            if ri == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "元画像 vs 撮影画像のRGBヒストグラム比較（密度正規化・縦軸共通）",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "17_source_vs_photo_histograms.png")
    plt.close(fig)

    # --- mean RGB bar comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(pairs))
    w = 0.12
    ax = axes[0]
    for i, ch in enumerate("RGB"):
        src_means = [channel_stats(src)[f"{ch}_mean"] for _, src, _pic, *_ in pairs]
        pic_means = [channel_stats(pic)[f"{ch}_mean"] for _, _src, pic, *_ in pairs]
        ax.bar(
            x + (i - 1) * 2 * w - w / 2,
            src_means,
            w,
            color=CH_COLORS[ch],
            alpha=0.45,
            label=f"元{ch}",
            edgecolor=CH_COLORS[ch],
        )
        ax.bar(
            x + (i - 1) * 2 * w + w / 2,
            pic_means,
            w,
            color=CH_COLORS[ch],
            alpha=0.95,
            label=f"撮{ch}",
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([jp_image(n) for n, *_ in pairs])
    for tick, (name, *_rest) in zip(ax.get_xticklabels(), pairs):
        tick.set_color(IMAGE_THEME[name])
        tick.set_fontweight("bold")
    ax.set_ylabel("平均輝度 (0–255)")
    ax.set_ylim(0, 255)
    ax.set_title("チャネル平均（淡=元 / 濃=撮影）")
    ax.legend(frameon=False, ncol=3, fontsize=8)

    ax = axes[1]
    deltas = []
    labels = []
    colors = []
    for name, src, pic, *_rest in pairs:
        ss = channel_stats(src)
        ps = channel_stats(pic)
        for ch in "RGB":
            deltas.append(ps[f"{ch}_mean"] - ss[f"{ch}_mean"])
            labels.append(f"{jp_image(name)}\n{ch}")
            colors.append(CH_COLORS[ch])
    ypos = np.arange(len(deltas))
    ax.barh(ypos, deltas, color=colors, edgecolor="white")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("撮影 − 元画像（平均輝度）")
    ax.set_title("チャネル平均の差（正=撮影の方が明るい）")
    dmax = max(abs(v) for v in deltas) if deltas else 1.0
    ax.set_xlim(-dmax * 1.15, dmax * 1.15)
    fig.suptitle("元画像と撮影画像の色の違い（平均）", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "18_source_vs_photo_mean_delta.png")
    plt.close(fig)

    print(f"[OK] wrote under {OUT}")
    for p in sorted(OUT.glob("1[678]_*.png")) + [csv_path]:
        print(f"  {p.name} ({p.stat().st_size} bytes)")
    print("\n--- mean deltas (photo - source) ---")
    for row in rows:
        print(
            f"{row['image_ja']}: "
            f"dR={float(row['delta_R_mean']):+.1f} "
            f"dG={float(row['delta_G_mean']):+.1f} "
            f"dB={float(row['delta_B_mean']):+.1f} "
            f"dLuma={float(row['delta_luma_mean']):+.1f}"
        )


if __name__ == "__main__":
    main()
