"""intensity(4/8/12) × 画像・チャネル・露光・表示レート の関係図を summary_slides に出す。"""
from __future__ import annotations

import csv
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
CHANNELS = ["R", "G", "B", "I", "X"]
INTENSITIES = ["4", "8", "12"]
FREQ_RATES = [45, 60, 90]
FREQ_FAMS = ["lockin", "fourier"]
FREQ_PASSES = {
    "lockin": ["lockin", "lockin_num"],
    "fourier": ["fourier", "fourier_num"],
}
FREQ_NOMINALS = [7.5, 10.0, 11.25, 15.0, 22.5, 26.25, 30.0]


def jp_image(name: str) -> str:
    return IMAGE_JP.get(name, name)


def bin_target_hz(hz: float, tol: float = 0.6) -> float:
    for nom in FREQ_NOMINALS:
        if abs(hz - nom) <= tol:
            return nom
    return round(hz * 2.0) / 2.0


def format_hz_label(hz: float) -> str:
    if abs(hz - round(hz)) < 1e-6:
        return f"{int(round(hz))}"
    if abs(hz * 4 - round(hz * 4)) < 1e-6:
        return f"{hz:.2f}".rstrip("0").rstrip(".")
    return f"{hz:.1f}"


def load_freq_rate_rows() -> list[dict]:
    """表示レート×解析周波数ごとの lockin/fourier any-success。"""
    stems = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("r"))
    # (rate, fam, freq, stem, folder) -> any_ok
    ok: dict[tuple, int] = defaultdict(int)
    seen: set[tuple] = set()
    for stem_dir in stems:
        stem = stem_dir.name
        rate, _exp, _fluoro = parse_stem(stem)
        if rate not in FREQ_RATES:
            continue
        for fam in FREQ_FAMS:
            for pass_name in FREQ_PASSES[fam]:
                path = stem_dir / f"results_sweeps_{pass_name}.csv"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            hz = bin_target_hz(float(row["fft_target_hz"]))
                        except (TypeError, ValueError, KeyError):
                            continue
                        key = (rate, fam, hz, stem, row["folder"])
                        seen.add(key)
                        if row.get("decode_success", "") in ("1", "True", "true"):
                            ok[key] = 1

    # aggregate
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for key in seen:
        rate, fam, hz, _stem, _folder = key
        buckets[(rate, fam, hz)].append(ok[key])

    rows: list[dict] = []
    for (rate, fam, hz), vals in sorted(buckets.items()):
        rows.append(
            {
                "rate_hz": rate,
                "family": fam,
                "target_hz": hz,
                "n": len(vals),
                "any_success_pct": rate_of(
                    [{"any_ok": v} for v in vals], key="any_ok"
                ),
            }
        )
    return rows


def parse_stem(stem: str) -> tuple[int, int, int]:
    parts = stem.split("_")
    return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])


def rate_of(rows: list[dict], key: str = "any_ok") -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(r[key] for r in rows) / len(rows)


def load_family_rows() -> dict[str, list[dict]]:
    stems = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("r"))
    rows_by_pass: dict[str, list[dict]] = defaultdict(list)

    for stem_dir in stems:
        stem = stem_dir.name
        rate, exp, fluoro = parse_stem(stem)
        for pass_name in PASS_TO_FAMILY:
            path = stem_dir / f"results_sweeps_{pass_name}.csv"
            if not path.exists():
                continue
            by_folder: dict[str, list[dict]] = defaultdict(list)
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    by_folder[row["folder"]].append(row)
            for folder, sweeps in by_folder.items():
                any_ok = 0
                adopted_ok = 0
                for s in sweeps:
                    ok = s.get("decode_success", "") in ("1", "True", "true")
                    if ok:
                        any_ok = 1
                    if s.get("adopted", "") in ("1", "True", "true"):
                        adopted_ok = 1 if ok else 0
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
                for pn in passes:
                    for r in rows_by_pass[pn]:
                        if r["stem"] == stem and r["folder"] == folder:
                            any_ok = max(any_ok, r["any_ok"])
                            adopted_ok = max(adopted_ok, r["adopted_ok"])
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
                    }
                )
    return family_rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    family_rows = load_family_rows()

    # --- CSV: intensity × factor breakdown ---
    csv_rows: list[dict] = []
    for fam in FAMILIES:
        rows = family_rows[fam]
        for inten in INTENSITIES:
            sub = [r for r in rows if r["intensity"] == inten]
            csv_rows.append(
                {
                    "family": fam,
                    "factor": "all",
                    "level": "-",
                    "intensity": inten,
                    "n": len(sub),
                    "any_success_pct": f"{rate_of(sub):.2f}",
                }
            )
            for factor, key in (
                ("image", "image"),
                ("channel", "channel"),
                ("rate", "rate"),
                ("exp", "exp"),
            ):
                levels = sorted({r[key] for r in sub}, key=lambda x: (str(type(x)), x))
                for level in levels:
                    ss = [r for r in sub if r[key] == level]
                    csv_rows.append(
                        {
                            "family": fam,
                            "factor": factor,
                            "level": str(level),
                            "intensity": inten,
                            "n": len(ss),
                            "any_success_pct": f"{rate_of(ss):.2f}",
                        }
                    )
    write_csv(
        OUT / "intensity_cross.csv",
        csv_rows,
        ["family", "factor", "level", "intensity", "n", "any_success_pct"],
    )

    # --- 09: overall intensity × family ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(INTENSITIES))
    width = 0.12
    for i, fam in enumerate(FAMILIES):
        ys = [
            rate_of([r for r in family_rows[fam] if r["intensity"] == inten])
            for inten in INTENSITIES
        ]
        ax.bar(x + (i - 2.5) * width, ys, width, label=fam, color=C[fam])
    ax.set_xticks(x)
    ax.set_xticklabels([f"i={t}" for t in INTENSITIES])
    ax.set_ylabel("any-success (%)")
    ax.set_title("埋め込み強度 × 手法ファミリー（全体）")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(OUT / "09_intensity_by_family.png")
    plt.close(fig)

    # --- 10: intensity × display rate ---
    rates = sorted({r["rate"] for r in family_rows["pair"]})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        for rate in rates:
            ys = [
                rate_of(
                    [
                        r
                        for r in family_rows[fam]
                        if r["intensity"] == inten and r["rate"] == rate
                    ]
                )
                for inten in INTENSITIES
            ]
            ax.plot(INTENSITIES, ys, marker="o", label=f"r{rate}")
        ax.set_title(fam)
        ax.set_xlabel("intensity")
        ax.set_ylabel("any-success (%)")
        ax.set_ylim(0, 100)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("埋め込み強度 × 表示レート（周波数）", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "10_intensity_by_rate.png")
    plt.close(fig)

    # --- 11: intensity × exposure ---
    exps = sorted({r["exp"] for r in family_rows["pair"]})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        for exp in exps:
            ys = [
                rate_of(
                    [
                        r
                        for r in family_rows[fam]
                        if r["intensity"] == inten and r["exp"] == exp
                    ]
                )
                for inten in INTENSITIES
            ]
            ax.plot(INTENSITIES, ys, marker="o", label=f"1/{exp}")
        ax.set_title(fam)
        ax.set_xlabel("intensity")
        ax.set_ylabel("any-success (%)")
        ax.set_ylim(0, 100)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("埋め込み強度 × 露光時間", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "11_intensity_by_exposure.png")
    plt.close(fig)

    # --- 12: intensity × channel ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        for ch in CHANNELS:
            ys = [
                rate_of(
                    [
                        r
                        for r in family_rows[fam]
                        if r["intensity"] == inten and r["channel"] == ch
                    ]
                )
                for inten in INTENSITIES
            ]
            ax.plot(INTENSITIES, ys, marker="o", label=ch)
        ax.set_title(fam)
        ax.set_xlabel("intensity")
        ax.set_ylabel("any-success (%)")
        ax.set_ylim(0, 100)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("埋め込み強度 × チャネル", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "12_intensity_by_channel.png")
    plt.close(fig)

    # --- 13: intensity × family, 2×2 by image ---
    images = [name for name in IMAGE_ORDER if name in {r["image"] for r in family_rows["pair"]}]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    x = np.arange(len(INTENSITIES))
    width = 0.12
    for ax, img in zip(axes.ravel(), images):
        for i, fam in enumerate(FAMILIES):
            ys = [
                rate_of(
                    [
                        r
                        for r in family_rows[fam]
                        if r["intensity"] == inten and r["image"] == img
                    ]
                )
                for inten in INTENSITIES
            ]
            ax.bar(x + (i - 2.5) * width, ys, width, label=fam, color=C[fam])
        ax.set_xticks(x)
        ax.set_xticklabels([f"i={t}" for t in INTENSITIES])
        ax.set_ylabel("any-success (%)")
        ax.set_ylim(0, 100)
        ax.set_title(jp_image(img), color=IMAGE_THEME[img], fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(IMAGE_THEME[img])
            spine.set_linewidth(2.0)
    axes[0, 0].legend(ncol=3, fontsize=7)
    fig.suptitle("埋め込み強度 × 手法（画像別）", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "13_intensity_by_family.png")
    plt.close(fig)

    # --- 14: heatmap image×channel at each intensity (4 families × 3 intensities) ---
    fams_14 = ["pair", "lockin", "accum", "fourier"]
    all_mats: dict[str, list[np.ndarray]] = {}
    for fam in fams_14:
        mats = []
        for inten in INTENSITIES:
            mat = []
            for img in images:
                row = []
                for ch in CHANNELS:
                    ss = [
                        r
                        for r in family_rows[fam]
                        if r["intensity"] == inten
                        and r["image"] == img
                        and r["channel"] == ch
                    ]
                    row.append(rate_of(ss))
                mat.append(row)
            mats.append(np.array(mat, dtype=float))
        all_mats[fam] = mats
    vmax = max(float(np.nanmax(m)) for mats in all_mats.values() for m in mats)
    vmax = max(10.0, vmax)

    fig, axes = plt.subplots(
        len(fams_14),
        len(INTENSITIES),
        figsize=(11, 11.5),
        constrained_layout=True,
    )
    last_im = None
    for ri, fam in enumerate(fams_14):
        for ci, (inten, mat) in enumerate(zip(INTENSITIES, all_mats[fam])):
            ax = axes[ri, ci]
            last_im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(CHANNELS)))
            ax.set_xticklabels(CHANNELS, fontsize=8)
            if ci == 0:
                ax.set_yticks(range(len(images)))
                ax.set_yticklabels([jp_image(n) for n in images], fontsize=8)
                for tick, name in zip(ax.get_yticklabels(), images):
                    tick.set_color(IMAGE_THEME.get(name, "#000"))
                    tick.set_fontweight("bold")
                ax.set_ylabel(fam, fontsize=11, fontweight="bold", color=C[fam])
            else:
                ax.set_yticks(range(len(images)))
                ax.set_yticklabels([])
            for i in range(len(images)):
                for j in range(len(CHANNELS)):
                    v = mat[i, j]
                    ax.text(
                        j,
                        i,
                        f"{v:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if v < vmax * 0.65 else "white",
                    )
            if ri == 0:
                ax.set_title(f"i={inten}", fontsize=11)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.55, label="any-success (%)")
    fig.suptitle("画像×チャネル（intensity別 any-success %）", fontsize=13)
    fig.savefig(OUT / "14_intensity_heatmap.png")
    plt.close(fig)
    for old in OUT.glob("14_intensity_heatmap_*.png"):
        old.unlink(missing_ok=True)

    # --- 15: lockin/fourier × target frequency at display rates 45/60/90 ---
    freq_rows = load_freq_rate_rows()
    write_csv(
        OUT / "freq_by_rate_lockin_fourier.csv",
        [
            {
                "rate_hz": r["rate_hz"],
                "family": r["family"],
                "target_hz": r["target_hz"],
                "n": r["n"],
                "any_success_pct": f"{r['any_success_pct']:.4f}",
            }
            for r in freq_rows
        ],
        ["rate_hz", "family", "target_hz", "n", "any_success_pct"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    width = 0.36
    freq_ymax = 0.0
    for rate in FREQ_RATES:
        for r in freq_rows:
            if r["rate_hz"] == rate:
                freq_ymax = max(freq_ymax, float(r["any_success_pct"]))
    freq_ylim = (0, max(10.0, freq_ymax * 1.2))
    for ax, rate in zip(axes, FREQ_RATES):
        sub = [r for r in freq_rows if r["rate_hz"] == rate]
        freqs = sorted({r["target_hz"] for r in sub})
        x = np.arange(len(freqs))
        for i, fam in enumerate(FREQ_FAMS):
            ys = []
            for hz in freqs:
                hit = [r for r in sub if r["family"] == fam and r["target_hz"] == hz]
                ys.append(hit[0]["any_success_pct"] if hit else float("nan"))
            bars = ax.bar(
                x + (i - 0.5) * width,
                ys,
                width,
                label=fam,
                color=C[fam],
                edgecolor="white",
            )
            for b, v in zip(bars, ys):
                if v == v:  # not nan
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        v + freq_ylim[1] * 0.015,
                        f"{v:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{format_hz_label(h)} Hz" for h in freqs], fontsize=9)
        ax.set_title(f"表示 {rate} Hz", fontsize=12)
        ax.set_xlabel("解析ターゲット周波数")
        ax.set_ylim(freq_ylim)
        ax.axhline(0, color="#888", lw=0.5)
    axes[0].set_ylabel("any-success（%）")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "表示レート 45/60/90 × lockin・fourier の周波数別復元率\n"
        "（binary∨num、条件 any-success）",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "15_freq_by_rate_lockin_fourier.png")
    plt.close(fig)

    # README append note
    readme = OUT / "README.md"
    extra = """
## intensity クロス（追加）
- `09_intensity_by_family.png` … 強度×手法
- `10_intensity_by_rate.png` … 強度×表示レート
- `11_intensity_by_exposure.png` … 強度×露光
- `12_intensity_by_channel.png` … 強度×チャネル
- `13_intensity_by_family.png` … 画像別 2×2（強度×手法）
- `14_intensity_heatmap.png` … 4手法×3強度の画像×チャネル一枚まとめ
- `15_freq_by_rate_lockin_fourier.png` … 表示45/60/90の周波数別 lockin/fourier
- `intensity_cross.csv` / `freq_by_rate_lockin_fourier.csv` … 数値表
"""
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "09_intensity_by_family" not in text:
            readme.write_text(text.rstrip() + "\n" + extra, encoding="utf-8")
        else:
            text = text.replace(
                "- `14_intensity_heatmap_{pair,lockin,accum,fourier}.png` … 強度別 画像×チャネル（カラーバーなし）",
                "- `14_intensity_heatmap.png` … 4手法×3強度の画像×チャネル一枚まとめ",
            )
            text = text.replace(
                "- `14_intensity_heatmap_{pair,lockin,accum}.png` … 強度別 画像×チャネル",
                "- `14_intensity_heatmap.png` … 4手法×3強度の画像×チャネル一枚まとめ",
            )
            if "15_freq_by_rate_lockin_fourier" not in text:
                text = text.rstrip() + (
                    "\n- `15_freq_by_rate_lockin_fourier.png` … 表示45/60/90の周波数別 lockin/fourier"
                    "\n- `freq_by_rate_lockin_fourier.csv` … 周波数別数値表\n"
                )
            readme.write_text(text, encoding="utf-8")
    print(f"[OK] wrote intensity cross figs under {OUT}")


if __name__ == "__main__":
    main()
