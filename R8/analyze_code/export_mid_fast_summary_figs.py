"""Export slide-ready PNGs + metrics JSON/CSV from out_mid_fast_0805."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent / "out_mid_fast_0805"
OUT = ROOT / "summary_slides"

from family_labels import (
    FAMILY_JP,
    LABEL_BINARY,
    LABEL_FLUORO_OFF,
    LABEL_FLUORO_ON,
    LABEL_GRAY,
    LABEL_INTENSITY,
    METRIC_ANY_SUCCESS,
    METRIC_ANY_SUCCESS_PCT,
    TALK_FAMILIES,
    accum_sweep_keep,
    jp_channels,
    jp_family,
)

PASS_TO_FAMILY = {
    "pair": "pair",
    "accum": "accum",
    "accum_num": "accum",
    "stat_std": "stat_std",
    "stat_std_num": "stat_std",
    "stat_var": "stat_var",
    "lockin": "lockin",
    "lockin_num": "lockin",
    "fourier": "fourier",
    "fourier_num": "fourier",
}
FAMILIES = ["pair", "accum", "stat_std", "stat_var", "lockin", "fourier"]
C = {
    "pair": "#457b9d",
    "accum": "#2a9d8f",
    "stat_std": "#6c757d",
    "stat_var": "#adb5bd",
    "lockin": "#e9c46a",
    "fourier": "#e76f51",
}
# スライド用画像ラベル（英キー → 日本語名・テーマ色）
IMAGE_ORDER = ["ex", "hocho", "nagaoka_fireworks", "rice"]
IMAGE_JP = {
    "ex": "実験",
    "hocho": "工事",
    "nagaoka_fireworks": "花火",
    "rice": "自然",
}
IMAGE_THEME = {
    "ex": "#2f6fed",  # 青
    "hocho": "#c62828",  # 赤
    "nagaoka_fireworks": "#212121",  # 黒
    "rice": "#2e7d32",  # 緑
}
STUDY_DIR = Path(__file__).resolve().parent / "out_image_channel_study"
ASSET_DIR = Path(__file__).resolve().parent.parent / "make_movie"


def jp_image(name: str) -> str:
    return IMAGE_JP.get(name, name)


def parse_stem(stem: str) -> tuple[int, int, int]:
    parts = stem.split("_")
    return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])


def to_float(x: str | None) -> float | None:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def rate_of(rows: list[dict], key: str = "any_ok") -> float:
    return 100.0 * sum(r[key] for r in rows) / len(rows)


def heatmap(ax, matrix, title, images, channels, vmax):
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(jp_channels(channels))
    ax.set_yticks(range(len(images)))
    labels = [jp_image(name) for name in images]
    ax.set_yticklabels(labels)
    for tick, name in zip(ax.get_yticklabels(), images):
        tick.set_color(IMAGE_THEME.get(name, "#000000"))
        tick.set_fontweight("bold")
    for i in range(len(images)):
        for j in range(len(channels)):
            v = matrix[i][j]
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                color="white" if v > vmax * 0.55 else "black",
                fontsize=10,
            )
    ax.set_title(title)
    return im


def load_color_stats() -> dict[str, dict[str, float]]:
    """qr_black 領域の R/G/B mean と max占有率."""
    path = STUDY_DIR / "color_stats.csv"
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["region"] != "qr_black":
                continue
            img = row["image"]
            ch = row["channel"]
            mean = row.get("mean") or ""
            if ch in ("R", "G", "B") and mean:
                out[img][ch] = float(mean)
            elif ch.startswith("max_is_") and mean:
                out[img][ch] = float(mean)
    return out


def load_texture_gray() -> dict[str, float]:
    path = STUDY_DIR / "texture.csv"
    out: dict[str, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["channel"] == "gray":
                out[row["image"]] = float(row["grad_mean"])
    return out


def plot_color_distribution(out_dir: Path) -> None:
    """画像ごとのテクスチャ（07）と RGB ヒストグラム（08）."""
    import cv2

    stats = load_color_stats()
    texture = load_texture_gray()
    images = [name for name in IMAGE_ORDER if name in stats]

    # --- 07: テクスチャのみ（大きい単一棒グラフ）---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fs_t, fs_l, fs_tick, fs_val = 18, 15, 14, 16
    x = np.arange(len(images))
    grads = [texture.get(img, 0.0) for img in images]
    ax.bar(
        x,
        grads,
        color=[IMAGE_THEME[n] for n in images],
        edgecolor="white",
        width=0.62,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([jp_image(n) for n in images], fontsize=fs_tick)
    for tick, name in zip(ax.get_xticklabels(), images):
        tick.set_color(IMAGE_THEME[name])
        tick.set_fontweight("bold")
    ax.set_ylabel("テクスチャ", fontsize=fs_l)
    ax.set_title("テクスチャ（模様の細かさ／QR黒領域の明暗変化の平均）", fontsize=fs_t, pad=10)
    for i, v in enumerate(grads):
        ax.text(i, v + 1.8, f"{v:.1f}", ha="center", fontsize=fs_val, fontweight="bold")
    ax.tick_params(labelsize=fs_tick)
    ax.set_ylim(0, max(grads) * 1.22)
    fig.tight_layout()
    fig.savefig(out_dir / "07_image_color_strength.png")
    plt.close(fig)

    # --- 08: per-image RGB histograms from source PNGs ---
    loaded: list[tuple[str, np.ndarray]] = []
    for name in images:
        path = ASSET_DIR / f"{name}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        loaded.append((name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    hist_peak = 0.0
    for _name, rgb in loaded:
        for ch in range(3):
            counts, _ = np.histogram(
                rgb[:, :, ch].ravel(), bins=64, range=(0, 255), density=True
            )
            if counts.size:
                hist_peak = max(hist_peak, float(counts.max()))
    hist_ymax = hist_peak * 1.08 if hist_peak > 0 else 0.05

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, (name, rgb) in zip(axes_flat, loaded):
        for ch, color, label in (
            (0, "#d62828", "R"),
            (1, "#2a9d8f", "G"),
            (2, "#1d3557", "B"),
        ):
            ax.hist(
                rgb[:, :, ch].ravel(),
                bins=64,
                range=(0, 255),
                histtype="stepfilled",
                alpha=0.45,
                color=color,
                label=label,
                density=True,
            )
        ax.set_xlim(0, 255)
        ax.set_ylim(0, hist_ymax)
        ax.set_xlabel("輝度")
        ax.set_ylabel("密度")
        ax.set_title(jp_image(name), color=IMAGE_THEME[name], fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(IMAGE_THEME[name])
            spine.set_linewidth(2.0)
    for ax in axes_flat[len(loaded) :]:
        ax.axis("off")
    fig.suptitle("元画像のRGBヒストグラム（全体画素・縦軸共通）", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "08_image_rgb_histograms.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_by_pass: dict[str, list[dict]] = defaultdict(list)
    stems = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("r"))

    for stem_dir in stems:
        stem = stem_dir.name
        rate, exp, fluoro = parse_stem(stem)
        for pass_name in PASS_TO_FAMILY:
            path = stem_dir / f"results_sweeps_{pass_name}.csv"
            by_folder: dict[str, list[dict]] = defaultdict(list)
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if pass_name in ("accum", "accum_num") and not accum_sweep_keep(row):
                        continue
                    by_folder[row["folder"]].append(row)
            for folder, sweeps in by_folder.items():
                if not sweeps:
                    continue
                any_ok = 0
                adopted_ok = 0
                adopted_acc = None
                for s in sweeps:
                    ok = s.get("decode_success", "") in ("1", "True", "true")
                    if ok:
                        any_ok = 1
                    if s.get("adopted", "") in ("1", "True", "true"):
                        adopted_ok = 1 if ok else 0
                        adopted_acc = to_float(s.get("pixel_acc_all"))
                toks = folder.split("_")
                intensity, channel, image = toks[-1], toks[-2], "_".join(toks[:-2])
                rows_by_pass[pass_name].append(
                    {
                        "stem": stem,
                        "folder": folder,
                        "image": image,
                        "channel": channel,
                        "intensity": intensity,
                        "rate": rate,
                        "exp": exp,
                        "fluoro": fluoro,
                        "any_ok": any_ok,
                        "adopted_ok": adopted_ok,
                        "adopted_acc": adopted_acc,
                    }
                )

    folders_by_stem: dict[str, set[str]] = defaultdict(set)
    for rows in rows_by_pass.values():
        for r in rows:
            folders_by_stem[r["stem"]].add(r["folder"])

    family_rows: dict[str, list[dict]] = {fam: [] for fam in FAMILIES}
    for stem_dir in stems:
        stem = stem_dir.name
        rate, exp, fluoro = parse_stem(stem)
        for folder in folders_by_stem[stem]:
            toks = folder.split("_")
            intensity, channel, image = toks[-1], toks[-2], "_".join(toks[:-2])
            for fam in FAMILIES:
                passes = [p for p, f in PASS_TO_FAMILY.items() if f == fam]
                any_ok = adopted_ok = 0
                best_acc = None
                for pn in passes:
                    for r in rows_by_pass[pn]:
                        if r["stem"] == stem and r["folder"] == folder:
                            any_ok = max(any_ok, r["any_ok"])
                            adopted_ok = max(adopted_ok, r["adopted_ok"])
                            if r["adopted_acc"] is not None:
                                best_acc = (
                                    r["adopted_acc"]
                                    if best_acc is None
                                    else max(best_acc, r["adopted_acc"])
                                )
                family_rows[fam].append(
                    {
                        "stem": stem,
                        "folder": folder,
                        "image": image,
                        "channel": channel,
                        "intensity": intensity,
                        "rate": rate,
                        "exp": exp,
                        "fluoro": fluoro,
                        "any_ok": any_ok,
                        "adopted_ok": adopted_ok,
                        "acc": best_acc,
                    }
                )

    fam_any = {fam: rate_of(family_rows[fam]) for fam in FAMILIES}
    fam_ad = {fam: rate_of(family_rows[fam], "adopted_ok") for fam in FAMILIES}
    fam_acc = {
        fam: float(
            np.mean([r["acc"] for r in family_rows[fam] if r["acc"] is not None])
        )
        for fam in FAMILIES
    }
    pass_any = {pn: rate_of(rows) for pn, rows in rows_by_pass.items()}
    pass_ad = {pn: rate_of(rows, "adopted_ok") for pn, rows in rows_by_pass.items()}

    rates = sorted({r["rate"] for r in family_rows["pair"]})
    exps = sorted({r["exp"] for r in family_rows["pair"]})
    images = [name for name in IMAGE_ORDER if name in {r["image"] for r in family_rows["pair"]}]
    channels = ["R", "G", "B", "I", "X"]

    payload = {
        "source": "out_mid_fast_0805",
        "note": (
            "復号成功率＝条件のうちスイープのいずれかで decode_success=1 の割合。"
            "手法は二値化と濃淡の OR（積算は窓5のみ）。"
            "re_analyze_mid_fast により lockin/fourier スイープを追記済み。"
        ),
        "n_stems": len(stems),
        "n_conditions": 1800,
        "family_any_success_pct": fam_any,
        "family_adopted_decode_pct": fam_ad,
        "family_mean_adopted_pixel_acc": fam_acc,
        "pass_any_success_pct": pass_any,
        "pass_adopted_decode_pct": pass_ad,
        "family_any_by_rate": {
            fam: {
                str(rate): rate_of([r for r in family_rows[fam] if r["rate"] == rate])
                for rate in rates
            }
            for fam in FAMILIES
        },
        "family_any_by_exp": {
            fam: {
                str(exp): rate_of([r for r in family_rows[fam] if r["exp"] == exp])
                for exp in exps
            }
            for fam in FAMILIES
        },
        "family_any_by_fluoro": {
            fam: {
                f"f{fl}": rate_of([r for r in family_rows[fam] if r["fluoro"] == fl])
                for fl in (0, 1)
            }
            for fam in FAMILIES
        },
        "image_channel_any_pair": {},
        "image_channel_any_accum": {},
    }
    for fam, key in (("pair", "image_channel_any_pair"), ("accum", "image_channel_any_accum")):
        for img in images:
            payload[key][img] = {
                ch: rate_of(
                    [
                        r
                        for r in family_rows[fam]
                        if r["image"] == img and r["channel"] == ch
                    ]
                )
                for ch in channels
            }

    (OUT / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (OUT / "family_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["family", "any_success_pct", "adopted_decode_pct", "mean_adopted_pixel_acc"]
        )
        for fam in FAMILIES:
            w.writerow(
                [fam, f"{fam_any[fam]:.4f}", f"{fam_ad[fam]:.4f}", f"{fam_acc[fam]:.6f}"]
            )

    with (OUT / "pass_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pass", "family", "any_success_pct", "adopted_decode_pct"])
        for pn in PASS_TO_FAMILY:
            w.writerow(
                [pn, PASS_TO_FAMILY[pn], f"{pass_any[pn]:.4f}", f"{pass_ad[pn]:.4f}"]
            )

    with (OUT / "family_by_rate.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "rate_hz", "any_success_pct"])
        for fam in FAMILIES:
            for rate in rates:
                w.writerow(
                    [
                        fam,
                        rate,
                        f"{payload['family_any_by_rate'][fam][str(rate)]:.4f}",
                    ]
                )

    with (OUT / "family_by_exp.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "exp", "any_success_pct"])
        for fam in FAMILIES:
            for exp in exps:
                w.writerow(
                    [fam, exp, f"{payload['family_any_by_exp'][fam][str(exp)]:.4f}"]
                )

    for name, key in (
        ("image_channel_pair.csv", "image_channel_any_pair"),
        ("image_channel_accum.csv", "image_channel_any_accum"),
    ):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "channel", "any_success_pct"])
            for img in images:
                for ch in channels:
                    w.writerow([img, ch, f"{payload[key][img][ch]:.4f}"])

    # figures
    talk_any = {fam: fam_any[fam] for fam in TALK_FAMILIES}
    order = sorted(TALK_FAMILIES, key=lambda fam: talk_any[fam], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ys = np.arange(len(order))
    vals = [fam_any[f] for f in order]
    bars = ax.barh(ys, vals, color=[C[f] for f in order], edgecolor="white", height=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([jp_family(f) for f in order], fontsize=12)
    ax.set_xlabel(METRIC_ANY_SUCCESS_PCT, fontsize=12)
    ax.set_title(
        "手法別 QR 復号成功率\n（条件数 n=1800、5フレーム積算は窓5のみ）",
        fontsize=13,
    )
    ax.set_xlim(0, max(vals) * 1.25)
    for b, v in zip(bars, vals):
        ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=11)
    ax.invert_yaxis()
    fig.savefig(OUT / "01_family_any_success.png")
    plt.close(fig)

    pairs = [
        ("accum", "accum", "accum_num"),
        ("stat_std", "stat_std", "stat_std_num"),
        ("lockin", "lockin", "lockin_num"),
        ("fourier", "fourier", "fourier_num"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pairs))
    w = 0.36
    bin_vals = [pass_any[b] for _, b, _ in pairs]
    num_vals = [pass_any[n] for _, _, n in pairs]
    ax.bar(x - w / 2, bin_vals, w, label=LABEL_BINARY, color="#1f4e79")
    ax.bar(x + w / 2, num_vals, w, label=LABEL_GRAY, color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels([jp_family(p[0]) for p in pairs], fontsize=11)
    ax.set_ylabel(METRIC_ANY_SUCCESS_PCT)
    ax.set_title(
        f"{LABEL_BINARY}と{LABEL_GRAY}の復号成功率の比較\n"
        "（5フレーム積算は窓5のみ。時間FFTの二値化は旧 th/255 でほぼ無効）"
    )
    ax.legend(frameon=False)
    for i, (bv, nv) in enumerate(zip(bin_vals, num_vals)):
        ax.text(i - w / 2, bv + 0.2, f"{bv:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, nv + 0.2, f"{nv:.1f}", ha="center", fontsize=9)
    fig.savefig(OUT / "02_binary_vs_num.png")
    plt.close(fig)

    key_fams = ["accum", "lockin", "fourier", "pair"]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(rates))
    w = 0.2
    for i, fam in enumerate(key_fams):
        vals = [payload["family_any_by_rate"][fam][str(r)] for r in rates]
        ax.bar(x + (i - 1.5) * w, vals, w, label=jp_family(fam), color=C[fam])
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rates], fontsize=14)
    ax.set_xlabel("表示の切替速さ [Hz]", fontsize=14)
    ax.set_ylabel(METRIC_ANY_SUCCESS_PCT, fontsize=14)
    ax.set_title("表示速さ別 復号成功率", fontsize=16)
    ax.legend(frameon=False, ncol=2, fontsize=12)
    ax.tick_params(labelsize=13)
    fig.savefig(OUT / "03_by_rate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(exps))
    all_vals: list[float] = []
    for i, fam in enumerate(key_fams):
        vals = [payload["family_any_by_exp"][fam][str(e)] for e in exps]
        all_vals.extend(vals)
        ax.bar(x + (i - 1.5) * w, vals, w, label=jp_family(fam), color=C[fam])
    ax.set_xticks(x)
    ax.set_xticklabels([f"1/{e}" for e in exps], fontsize=14)
    ax.set_xlabel("露光", fontsize=14)
    ax.set_ylabel(METRIC_ANY_SUCCESS_PCT, fontsize=14)
    ax.set_title("露光別 復号成功率", fontsize=16)
    finite = [v for v in all_vals if np.isfinite(v)]
    ymax = max(finite) if finite else 1.0
    ax.set_ylim(0, max(ymax * 1.45, ymax + 8.0))
    ax.legend(frameon=False, ncol=2, fontsize=12, loc="upper right")
    ax.tick_params(labelsize=13)
    fig.savefig(OUT / "04_by_exposure.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TALK_FAMILIES))
    w = 0.36
    f0 = [payload["family_any_by_fluoro"][f]["f0"] for f in TALK_FAMILIES]
    f1 = [payload["family_any_by_fluoro"][f]["f1"] for f in TALK_FAMILIES]
    ax.bar(x - w / 2, f0, w, label=LABEL_FLUORO_OFF, color="#8d99ae")
    ax.bar(x + w / 2, f1, w, label=LABEL_FLUORO_ON, color="#ef233c")
    ax.set_xticks(x)
    ax.set_xticklabels([jp_family(f) for f in TALK_FAMILIES], rotation=15, fontsize=13)
    ax.set_ylabel(METRIC_ANY_SUCCESS_PCT, fontsize=14)
    ax.set_title("蛍光灯の有無別 復号成功率", fontsize=16)
    fluoro_max = max(max(f0), max(f1)) if f0 and f1 else 1.0
    ax.set_ylim(0, max(fluoro_max * 1.35, fluoro_max + 4.0))
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    ax.tick_params(labelsize=13)
    fig.savefig(OUT / "05_by_fluoro.png")
    plt.close(fig)

    mat_pair = [
        [payload["image_channel_any_pair"][img][ch] for ch in channels] for img in images
    ]
    mat_acc = [
        [payload["image_channel_any_accum"][img][ch] for ch in channels]
        for img in images
    ]
    vmax = max(max(max(r) for r in mat_pair), max(max(r) for r in mat_acc))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    heatmap(axes[0], mat_pair, f"{jp_family('pair')} の復号成功率（%）", images, channels, vmax)
    im1 = heatmap(
        axes[1], mat_acc, f"{jp_family('accum')} の復号成功率（%）", images, channels, vmax
    )
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85, label=METRIC_ANY_SUCCESS_PCT)
    fig.suptitle("画像×チャネル別 復号成功率", fontsize=13, y=1.02)
    fig.savefig(OUT / "06_image_channel_heatmap.png")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
    pct_max = max(
        max(fam_any.values()),
        max(
            payload["family_any_by_rate"][fam][str(r)]
            for fam in key_fams
            for r in rates
        ),
        max(
            payload["family_any_by_exp"][fam][str(e)]
            for fam in key_fams
            for e in exps
        ),
    )
    pct_ylim = (0, max(10.0, pct_max * 1.15))

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(
        range(len(order)),
        [fam_any[f] for f in order],
        color=[C[f] for f in order],
    )
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([jp_family(f) for f in order])
    ax1.invert_yaxis()
    ax1.set_xlabel(METRIC_ANY_SUCCESS_PCT)
    ax1.set_xlim(pct_ylim)
    ax1.set_title("手法別 復号成功率")

    ax2 = fig.add_subplot(gs[0, 1])
    for fam in key_fams:
        ax2.plot(
            rates,
            [payload["family_any_by_rate"][fam][str(r)] for r in rates],
            marker="o",
            label=jp_family(fam),
            color=C[fam],
        )
    ax2.set_xlabel("レート [Hz]")
    ax2.set_ylabel(METRIC_ANY_SUCCESS_PCT)
    ax2.set_ylim(pct_ylim)
    ax2.set_title("レート別")
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(exps))
    w = 0.2
    for i, fam in enumerate(key_fams):
        ax3.bar(
            x + (i - 1.5) * w,
            [payload["family_any_by_exp"][fam][str(e)] for e in exps],
            w,
            label=jp_family(fam),
            color=C[fam],
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"1/{e}" for e in exps])
    ax3.set_ylim(pct_ylim)
    ax3.set_ylabel(METRIC_ANY_SUCCESS_PCT)
    ax3.set_title("露光別")
    ax3.legend(frameon=False, fontsize=8, ncol=2)

    ax4 = fig.add_subplot(gs[1, 1])
    im = heatmap(ax4, mat_acc, f"{jp_family('accum')}：画像×チャネル（%）", images, channels, vmax)
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    fig.suptitle(
        "実験概要（復号成功率）／n=1800／5フレーム積算は窓5",
        fontsize=14,
    )
    fig.savefig(OUT / "00_overview.png")
    plt.close(fig)

    plot_color_distribution(OUT)

    # also write Japanese name map into metrics
    payload["image_labels_ja"] = IMAGE_JP
    payload["image_theme_colors"] = IMAGE_THEME
    payload["family_labels_ja"] = FAMILY_JP
    payload["metric_labels_ja"] = {
        "any_success": METRIC_ANY_SUCCESS,
        "any_success_pct": METRIC_ANY_SUCCESS_PCT,
        "binary": LABEL_BINARY,
        "gray": LABEL_GRAY,
        "intensity": LABEL_INTENSITY,
    }
    payload["accum_window_n"] = 5
    # pair vs accum(w5) exclusive successes
    pair_by = {(r["stem"], r["folder"]): r["any_ok"] for r in family_rows["pair"]}
    accum_by = {(r["stem"], r["folder"]): r["any_ok"] for r in family_rows["accum"]}
    only_accum = only_pair = both = 0
    for k in pair_by:
        p_ok = pair_by[k]
        a_ok = accum_by.get(k, 0)
        if a_ok and not p_ok:
            only_accum += 1
        elif p_ok and not a_ok:
            only_pair += 1
        elif p_ok and a_ok:
            both += 1
    payload["pair_vs_accum_w5"] = {
        "only_accum": only_accum,
        "only_pair": only_pair,
        "both": both,
        "pair_pct": fam_any["pair"],
        "accum_pct": fam_any["accum"],
    }
    (OUT / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = """# mid_fast_0805 スライド用図

データ元: `R8/analyze_code/out_mid_fast_0805`（30条件 × 60組み合わせ = 1800）。

## 手法ラベル（日本語）
- pair → **2フレーム差分**
- accum → **5フレーム積算**（窓5フレームの二値化∨濃淡のみ。全長120は含めない）
- lockin → **同期検波**
- fourier → **時間FFT**

## 画像ラベル（日本語）
- ex → **実験**（青）
- hocho → **工事**（赤）
- nagaoka_fireworks → **花火**（黒）
- rice → **自然**（緑）

## 指標
- **復号成功率**: 条件のうち、スイープのどれかでデコード成功した割合
- **手法**: 二値化と濃淡の OR（積算は窓5のみ）
- **採用スイープ**: 画素一致率が最大のスイープでの成功

## 注意
`re_analyze_mid_fast` により同期検波・時間FFTの不足スイープを追記済み。
時間FFTの二値化に旧 th/255 行が残る場合あり（手法は二値化+濃淡の OR）。

## ファイル
- `00_overview.png` … 一枚もの
- `01_family_any_success.png`
- `02_binary_vs_num.png`
- `03_by_rate.png`
- `04_by_exposure.png`
- `05_by_fluoro.png`
- `06_image_channel_heatmap.png`
- `07_image_color_strength.png` … テクスチャ（模様の細かさ／QR黒領域の明暗変化の平均）
- `08_image_rgb_histograms.png` … 元画像 RGB ヒストグラム
- `metrics.json` / `*.csv`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
