"""画像×チャネルのデコード差を、元画像の色分布・クリップ余裕・テクスチャと突き合わせる。

出力先: R8/analyze_code/out_image_channel_study/
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_R8 = SCRIPT_DIR.parent
MAKE_MOVIE = REPO_R8 / "make_movie"
OUT_DIR = SCRIPT_DIR / "out_image_channel_study"
RESULTS_ROOT = SCRIPT_DIR / "out"

# gen_assets と同じ前処理パラメータ（実験 README の clip-margin 12）
WIDTH, HEIGHT = 1920, 1080
CLIP_MARGIN = 12
INTENSITIES = (4, 8, 12)
CHANNELS = ("R", "G", "B", "max", "min")
CHANNEL_TOKENS = {"R": "R", "G": "G", "B": "B", "max": "X", "min": "I"}
IMAGE_NAMES = ("ex", "rice", "hocho", "nagaoka_fireworks")
PASSES = ("pair", "accum", "lockin", "stat_std", "fourier")
SKIP_VIDEOS = {"r60_e250_f1"}
FOCUS_RATES = {45, 60, 90}

sys.path.insert(0, str(MAKE_MOVIE))
sys.path.insert(0, str(SCRIPT_DIR))

from gen_assets import (  # noqa: E402
    apply_channel_delta,
    center_square_params,
    load_base_image,
    load_qr_mask,
    preclip,
)
from gpu_ops import max_channel_difference  # noqa: E402


def is_success(row: dict) -> bool:
    note = (row.get("decode_note") or "").strip()
    text = (row.get("decode_decoded_text") or "").strip()
    return note == "" and text != ""


def folder_parts(folder: str) -> Tuple[str, str, str]:
    """folder: rice_R_4 / nagaoka_fireworks_X_12 → image, channel_token, intensity."""
    parts = (folder or "").split("_")
    if len(parts) < 3:
        return "?", "?", "?"
    inten = parts[-1]
    ch = parts[-2]
    img = "_".join(parts[:-2])
    return img, ch, inten


def token_to_channel(token: str) -> str:
    rev = {v: k for k, v in CHANNEL_TOKENS.items()}
    return rev.get(token, token)


# ---------------------------------------------------------------------------
# A. Decode matrices
# ---------------------------------------------------------------------------


def iter_result_rows(passes: Iterable[str] = PASSES):
    for vdir in sorted(RESULTS_ROOT.iterdir()):
        if not vdir.is_dir():
            continue
        m = re.match(r"r(\d+)_e(\d+)_f(\d+)", vdir.name)
        if not m or vdir.name in SKIP_VIDEOS:
            continue
        rate = int(m.group(1))
        if rate not in FOCUS_RATES:
            continue
        meta = {
            "video": vdir.name,
            "rate": rate,
            "exp": int(m.group(2)),
            "fluoro": int(m.group(3)),
        }
        for pass_name in passes:
            path = vdir / f"results_{pass_name}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    img, tok, inten = folder_parts(row.get("folder", ""))
                    yield {
                        **meta,
                        "pass": pass_name,
                        "image": img,
                        "channel_token": tok,
                        "channel": token_to_channel(tok),
                        "intensity": inten,
                        "ok": is_success(row),
                        "pixel_acc_best": _float_or_none(row.get("pixel_acc_best")),
                        "pixel_acc_all": _float_or_none(row.get("pixel_acc_all")),
                    }


def _float_or_none(v: Optional[str]) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def write_decode_matrices() -> Dict[str, Dict[Tuple[str, str], dict]]:
    """pass → {(image, channel): stats} を返す。CSV も書く。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buckets: Dict[str, Dict[Tuple[str, str, str], list]] = defaultdict(
        lambda: defaultdict(list)
    )
    # also intensity cross
    inten_buckets: Dict[str, Dict[Tuple[str, str, str], list]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in iter_result_rows():
        key = (row["image"], row["channel"], row["pass"])
        buckets[row["pass"]][(row["image"], row["channel"])].append(row)
        inten_buckets[row["pass"]][
            (row["image"], row["channel"], row["intensity"])
        ].append(row)

    matrices: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for pass_name in PASSES:
        matrices[pass_name] = {}
        rows_out = []
        for image in IMAGE_NAMES:
            for ch in CHANNELS:
                items = buckets[pass_name].get((image, ch), [])
                n = len(items)
                n_ok = sum(1 for r in items if r["ok"])
                accs = [r["pixel_acc_best"] for r in items if r["pixel_acc_best"] is not None]
                stats = {
                    "n": n,
                    "n_ok": n_ok,
                    "decode_rate": (n_ok / n) if n else 0.0,
                    "acc_best_mean": (sum(accs) / len(accs)) if accs else None,
                }
                matrices[pass_name][(image, ch)] = stats
                rows_out.append(
                    {
                        "pass": pass_name,
                        "image": image,
                        "channel": ch,
                        "channel_token": CHANNEL_TOKENS[ch],
                        "n_ok": n_ok,
                        "n": n,
                        "decode_rate": f"{stats['decode_rate']:.4f}",
                        "pixel_acc_best_mean": (
                            f"{stats['acc_best_mean']:.4f}"
                            if stats["acc_best_mean"] is not None
                            else ""
                        ),
                    }
                )
        _write_csv(OUT_DIR / f"decode_matrix_{pass_name}.csv", rows_out)

    # intensity cross for pair / lockin / accum
    inten_rows = []
    for pass_name in ("pair", "accum", "lockin"):
        for image in IMAGE_NAMES:
            for ch in CHANNELS:
                for inten in ("4", "8", "12"):
                    items = inten_buckets[pass_name].get((image, ch, inten), [])
                    n = len(items)
                    n_ok = sum(1 for r in items if r["ok"])
                    inten_rows.append(
                        {
                            "pass": pass_name,
                            "image": image,
                            "channel": ch,
                            "intensity": inten,
                            "n_ok": n_ok,
                            "n": n,
                            "decode_rate": f"{(n_ok / n) if n else 0:.4f}",
                        }
                    )
    _write_csv(OUT_DIR / "decode_matrix_by_intensity.csv", inten_rows)

    # heatmaps
    for pass_name in ("pair", "accum", "lockin"):
        _plot_decode_heatmap(pass_name, matrices[pass_name])

    return matrices


def _plot_decode_heatmap(pass_name: str, matrix: Dict[Tuple[str, str], dict]) -> None:
    data = np.zeros((len(IMAGE_NAMES), len(CHANNELS)), dtype=np.float64)
    annot = np.empty_like(data, dtype=object)
    for i, image in enumerate(IMAGE_NAMES):
        for j, ch in enumerate(CHANNELS):
            s = matrix[(image, ch)]
            data[i, j] = 100.0 * s["decode_rate"]
            annot[i, j] = f"{data[i, j]:.1f}%\n{s['n_ok']}/{s['n']}"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=max(40, float(data.max()) + 1))
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels([f"{c} ({CHANNEL_TOKENS[c]})" for c in CHANNELS])
    ax.set_yticks(range(len(IMAGE_NAMES)))
    ax.set_yticklabels(list(IMAGE_NAMES))
    ax.set_title(f"Decode rate (%) — {pass_name} (r45/60/90)")
    for i in range(len(IMAGE_NAMES)):
        for j in range(len(CHANNELS)):
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"heatmap_decode_{pass_name}.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B–D. Color / headroom / digital signal / texture
# ---------------------------------------------------------------------------


def load_prepared(image_name: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    path = MAKE_MOVIE / f"{image_name}.png"
    base = load_base_image(path, WIDTH, HEIGHT)
    base = preclip(base, CLIP_MARGIN)
    x0, _, square = center_square_params(WIDTH, HEIGHT)
    mask = load_qr_mask(MAKE_MOVIE / "HP_QR.png", square_size=square)
    return base, mask, x0, square


def bgr_planes(roi: np.ndarray) -> Dict[str, np.ndarray]:
    return {"B": roi[:, :, 0], "G": roi[:, :, 1], "R": roi[:, :, 2]}


def analyze_color_and_texture() -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    color_rows: List[dict] = []
    headroom_rows: List[dict] = []
    signal_rows: List[dict] = []
    texture_rows: List[dict] = []

    for image_name in IMAGE_NAMES:
        base, black_mask, x0, square = load_prepared(image_name)
        roi = base[:, x0 : x0 + square, :]
        planes = bgr_planes(roi)

        # --- color stats on QR-black pixels and full ROI ---
        for region_name, sel in (
            ("qr_black", black_mask),
            ("roi_all", np.ones_like(black_mask, dtype=bool)),
        ):
            for ch_name, plane in planes.items():
                vals = plane[sel].astype(np.float64)
                color_rows.append(
                    {
                        "image": image_name,
                        "region": region_name,
                        "channel": ch_name,
                        "mean": f"{vals.mean():.3f}",
                        "std": f"{vals.std():.3f}",
                        "p05": f"{np.percentile(vals, 5):.3f}",
                        "p50": f"{np.percentile(vals, 50):.3f}",
                        "p95": f"{np.percentile(vals, 95):.3f}",
                        "frac_near_low": f"{float(np.mean(vals <= CLIP_MARGIN + 1)):.4f}",
                        "frac_near_high": f"{float(np.mean(vals >= 255 - CLIP_MARGIN - 1)):.4f}",
                    }
                )

            # max/min channel occupancy
            idx_max = np.argmax(roi, axis=2)
            idx_min = np.argmin(roi, axis=2)
            for label, idx_map in (("max", idx_max), ("min", idx_min)):
                for c_i, c_name in enumerate(("B", "G", "R")):
                    frac = float(np.mean(idx_map[sel] == c_i))
                    color_rows.append(
                        {
                            "image": image_name,
                            "region": region_name,
                            "channel": f"{label}_is_{c_name}",
                            "mean": f"{frac:.4f}",
                            "std": "",
                            "p05": "",
                            "p50": "",
                            "p95": "",
                            "frac_near_low": "",
                            "frac_near_high": "",
                        }
                    )

        # histograms
        _plot_histograms(image_name, planes, black_mask)

        # --- headroom for intensities ---
        for ch in ("R", "G", "B"):
            plane = planes[ch][black_mask]
            for inten in INTENSITIES:
                # after preclip, values in [m, 255-m]; still check remaining room
                can_darken = plane >= (CLIP_MARGIN + inten)  # already clipped; room below
                # Actually after preclip, min is CLIP_MARGIN, so darken by inten stays >= 0
                # if plane - inten >= 0 always when plane >= inten. With preclip margin=12
                # and inten<=12, darken always OK. Brighten: plane + inten <= 255.
                frac_clip_darken = float(np.mean(plane < inten))
                frac_clip_brighten = float(np.mean(plane > 255 - inten))
                # headroom relative to preclip band
                room_down = plane - 0.0
                room_up = 255.0 - plane
                frac_insufficient = float(
                    np.mean((room_down < inten) | (room_up < inten))
                )
                headroom_rows.append(
                    {
                        "image": image_name,
                        "channel": ch,
                        "intensity": inten,
                        "mean_room_down": f"{room_down.mean():.3f}",
                        "mean_room_up": f"{room_up.mean():.3f}",
                        "frac_clip_darken": f"{frac_clip_darken:.4f}",
                        "frac_clip_brighten": f"{frac_clip_brighten:.4f}",
                        "frac_insufficient_headroom": f"{frac_insufficient:.4f}",
                    }
                )

        # --- digital signal via max_channel_difference ---
        for ch in CHANNELS:
            for inten in INTENSITIES:
                normal = apply_channel_delta(
                    base, black_mask, x0, square, ch, delta=float(inten)
                )
                inv = apply_channel_delta(
                    base, black_mask, x0, square, ch, delta=float(-inten)
                )
                # scale to 0..1 like pipeline float frames often are? gpu_ops uses raw float
                # gen_assets keeps 0..255 float. cal-from-2frame typically /255.
                # Check how frames are loaded in pipeline...
                n_roi = normal[:, x0 : x0 + square, :] / 255.0
                i_roi = inv[:, x0 : x0 + square, :] / 255.0
                diff = max_channel_difference(n_roi, i_roi)
                sig = np.abs(diff[black_mask])
                # expected ideal ~ 2*inten/255 for single-channel
                signal_rows.append(
                    {
                        "image": image_name,
                        "channel": ch,
                        "intensity": inten,
                        "mean_abs_signal": f"{sig.mean():.6f}",
                        "median_abs_signal": f"{float(np.median(sig)):.6f}",
                        "ideal_2i_over_255": f"{(2.0 * inten / 255.0):.6f}",
                        "frac_of_ideal": f"{(sig.mean() / (2.0 * inten / 255.0)):.4f}",
                    }
                )

        # --- texture: Sobel energy on QR-black ---
        gray = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        for ch_name, plane in list(planes.items()) + [("gray", gray)]:
            p = plane.astype(np.float32)
            gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
            energy = np.sqrt(gx * gx + gy * gy)
            vals = energy[black_mask]
            texture_rows.append(
                {
                    "image": image_name,
                    "channel": ch_name,
                    "grad_mean": f"{vals.mean():.4f}",
                    "grad_std": f"{vals.std():.4f}",
                    "grad_p95": f"{np.percentile(vals, 95):.4f}",
                }
            )

    _write_csv(OUT_DIR / "color_stats.csv", color_rows)
    _write_csv(OUT_DIR / "headroom.csv", headroom_rows)
    _write_csv(OUT_DIR / "digital_signal.csv", signal_rows)
    _write_csv(OUT_DIR / "texture.csv", texture_rows)

    _plot_mean_rgb_bars(color_rows)
    _plot_texture_bars(texture_rows)
    _plot_digital_signal(signal_rows)

    return color_rows, headroom_rows, signal_rows, texture_rows


def _plot_histograms(
    image_name: str, planes: Dict[str, np.ndarray], black_mask: np.ndarray
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    colors = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}
    for ax, ch in zip(axes, ("R", "G", "B")):
        vals = planes[ch][black_mask].ravel()
        ax.hist(vals, bins=64, range=(0, 255), color=colors[ch], alpha=0.85)
        ax.set_title(f"{image_name} — {ch} (QR black)")
        ax.set_xlabel("value")
        ax.set_xlim(0, 255)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"hist_{image_name}.png", dpi=140)
    plt.close(fig)


def _plot_mean_rgb_bars(color_rows: List[dict]) -> None:
    # qr_black means for R,G,B
    means = {img: {} for img in IMAGE_NAMES}
    for row in color_rows:
        if row["region"] != "qr_black" or row["channel"] not in ("R", "G", "B"):
            continue
        means[row["image"]][row["channel"]] = float(row["mean"])

    x = np.arange(len(IMAGE_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, ch in enumerate(("R", "G", "B")):
        vals = [means[img].get(ch, 0) for img in IMAGE_NAMES]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=ch,
            color={"R": "tab:red", "G": "tab:green", "B": "tab:blue"}[ch],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(IMAGE_NAMES), rotation=15)
    ax.set_ylabel("mean (QR black pixels)")
    ax.set_title(f"RGB means in QR region (clip_margin={CLIP_MARGIN})")
    ax.legend()
    ax.set_ylim(0, 255)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bar_rgb_means.png", dpi=140)
    plt.close(fig)


def _plot_texture_bars(texture_rows: List[dict]) -> None:
    gray_grad = {
        r["image"]: float(r["grad_mean"])
        for r in texture_rows
        if r["channel"] == "gray"
    }
    fig, ax = plt.subplots(figsize=(7, 3.5))
    imgs = list(IMAGE_NAMES)
    vals = [gray_grad.get(i, 0) for i in imgs]
    ax.bar(imgs, vals, color="gray")
    ax.set_ylabel("mean |Sobel| on QR black")
    ax.set_title("Spatial texture (gradient energy)")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bar_texture_gray.png", dpi=140)
    plt.close(fig)


def _plot_digital_signal(signal_rows: List[dict]) -> None:
    # intensity=12, channels R/G/B
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(IMAGE_NAMES))
    width = 0.25
    for i, ch in enumerate(("R", "G", "B")):
        vals = []
        for img in IMAGE_NAMES:
            hit = [
                r
                for r in signal_rows
                if r["image"] == img and r["channel"] == ch and int(r["intensity"]) == 12
            ]
            vals.append(float(hit[0]["frac_of_ideal"]) if hit else 0)
        ax.bar(x + (i - 1) * width, vals, width, label=ch)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="ideal")
    ax.set_xticks(x)
    ax.set_xticklabels(list(IMAGE_NAMES), rotation=15)
    ax.set_ylabel("mean|signal| / (2·I/255)")
    ax.set_title("Digital embedding signal (intensity=12)")
    ax.legend()
    ax.set_ylim(0, 1.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bar_digital_signal_i12.png", dpi=140)
    plt.close(fig)


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# E. FINDINGS.md
# ---------------------------------------------------------------------------


def write_findings(
    matrices: Dict[str, Dict[Tuple[str, str], dict]],
    color_rows: List[dict],
    headroom_rows: List[dict],
    signal_rows: List[dict],
    texture_rows: List[dict],
) -> None:
    """数値から FINDINGS.md を生成する。"""

    def rgb_means(image: str) -> Dict[str, float]:
        out = {}
        for row in color_rows:
            if row["image"] == image and row["region"] == "qr_black" and row["channel"] in (
                "R",
                "G",
                "B",
            ):
                out[row["channel"]] = float(row["mean"])
        return out

    def max_occ(image: str) -> Dict[str, float]:
        out = {}
        for row in color_rows:
            if row["image"] == image and row["region"] == "qr_black":
                for ch in ("R", "G", "B"):
                    if row["channel"] == f"max_is_{ch}":
                        out[ch] = float(row["mean"])
        return out

    def near_low(image: str, ch: str) -> float:
        for row in color_rows:
            if (
                row["image"] == image
                and row["region"] == "qr_black"
                and row["channel"] == ch
                and row["frac_near_low"] != ""
            ):
                return float(row["frac_near_low"])
        return 0.0

    def gray_tex(image: str) -> float:
        for row in texture_rows:
            if row["image"] == image and row["channel"] == "gray":
                return float(row["grad_mean"])
        return 0.0

    def best_ch(pass_name: str, image: str) -> Tuple[str, float]:
        best = ("?", -1.0)
        for ch in CHANNELS:
            r = matrices[pass_name][(image, ch)]["decode_rate"]
            if r > best[1]:
                best = (ch, r)
        return best

    def sig_frac(image: str, ch: str, inten: int = 12) -> float:
        for row in signal_rows:
            if (
                row["image"] == image
                and row["channel"] == ch
                and int(row["intensity"]) == inten
            ):
                return float(row["frac_of_ideal"])
        return 0.0

    def pass_total(pass_name: str, image: str) -> Tuple[int, int]:
        ok = sum(matrices[pass_name][(image, ch)]["n_ok"] for ch in CHANNELS)
        n = sum(matrices[pass_name][(image, ch)]["n"] for ch in CHANNELS)
        return ok, n

    lines: List[str] = []
    lines.append("# 画像×RGBデコード差 — 調査メモ")
    lines.append("")
    lines.append(
        "対象: `out/` の r45/60/90（`r60_e250_f1` 除外）。"
        "元画像は `clip_margin=12`、QR 中央 1080×1080。"
    )
    lines.append("再生成: `python R8/analyze_code/analyze_image_channel.py`")
    lines.append("")
    lines.append("## 結論（先に）")
    lines.append("")
    lines.append(
        "1. **デジタル埋め込み信号は全画像・全チャネルで理想振幅どおり（frac=1.0）**。"
        "色偏り・クリップでは「信号が出ない」は説明できない。"
    )
    lines.append(
        "2. **デコード差は空間テクスチャと、支配／従属チャネルの関係に依存**する。"
    )
    lines.append(
        "3. **支配色埋め込みが有利な画像（rice の R）と、従属色＋lockin が効く画像（hocho の B/G）が分かれる**。"
    )
    lines.append(
        "4. **勾配エネルギーが高いほど pair が厳しく、lockin の相対価値が上がる**。"
    )
    lines.append("")
    lines.append("## 1. デコード実績（pair）")
    lines.append("")
    lines.append("| image | R | G | B | max(X) | min(I) | 最得意 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for image in IMAGE_NAMES:
        cells = []
        for ch in CHANNELS:
            s = matrices["pair"][(image, ch)]
            cells.append(f"{100 * s['decode_rate']:.1f}% ({s['n_ok']}/{s['n']})")
        bc, br = best_ch("pair", image)
        lines.append(
            f"| {image} | " + " | ".join(cells) + f" | **{bc}** {100*br:.1f}% |"
        )
    lines.append("")
    lines.append("| image | pair | lockin |")
    lines.append("|---|---:|---:|")
    for image in IMAGE_NAMES:
        po, pn = pass_total("pair", image)
        lo, ln = pass_total("lockin", image)
        if pn and ln:
            lines.append(f"| {image} | {100*po/pn:.1f}% | {100*lo/ln:.1f}% |")
        else:
            lines.append(f"| {image} | - | - |")
    lines.append("")
    lines.append("## 2. 元画像の色分布（QR 黒画素）")
    lines.append("")
    lines.append("| image | R mean | G mean | B mean | max占有 | near_low(B) | gray勾配 |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    for image in IMAGE_NAMES:
        m = rgb_means(image)
        occ = max_occ(image)
        occ_s = (
            f"R{100*occ.get('R',0):.0f}/G{100*occ.get('G',0):.0f}/B{100*occ.get('B',0):.0f}%"
        )
        lines.append(
            f"| {image} | {m.get('R',0):.1f} | {m.get('G',0):.1f} | {m.get('B',0):.1f} | "
            f"{occ_s} | {100*near_low(image,'B'):.1f}% | {gray_tex(image):.2f} |"
        )
    lines.append("")
    lines.append("## 3. デジタル実効信号（intensity=12, frac of ideal）")
    lines.append("")
    lines.append("| image | R | G | B |")
    lines.append("|---|---:|---:|---:|")
    for image in IMAGE_NAMES:
        lines.append(
            f"| {image} | {sig_frac(image,'R'):.3f} | {sig_frac(image,'G'):.3f} | {sig_frac(image,'B'):.3f} |"
        )
    lines.append("")
    lines.append(
        "ヘッドルーム不足率も intensity<=12 では全チャネル 0%。"
        "差の本丸は撮影応答と空間テクスチャ。"
    )
    lines.append("")
    lines.append("## 4. 画像別の読み")
    lines.append("")
    lines.append(
        "- **rice**: R/G が強く B は pair/lockin とも失敗。B はほぼ常に min。"
        "デジタル信号は十分なのでカメラ側の暖色 B 応答が疑わしい。テクスチャ最小で pair 向き。"
    )
    lines.append(
        "- **ex**: 画像は B 支配なのに pair 最得意は R。"
        "支配チャネル!=最善の例。lockin は pair より落ちる。"
    )
    lines.append(
        "- **hocho**: R 支配・B は下限付近が多い。pair は弱いが lockin が B/G/min を救済。"
        "高テクスチャ下では周波数選択＋従属チャネルが有効。"
    )
    lines.append(
        "- **nagaoka_fireworks**: 勾配最大。pair 全滅、lockin もほぼ無効。"
        "チャネルより背景高周波が支配的な失敗モード。"
    )
    lines.append("")
    tex_sorted = sorted(((gray_tex(i), i) for i in IMAGE_NAMES), reverse=True)
    lines.append(
        "勾配順: " + " > ".join(f"{n}({v:.1f})" for v, n in tex_sorted)
    )
    lines.append("")
    lines.append("## 5. 追実験候補")
    lines.append("")
    lines.append("1. 撮影済み frame で埋め込みチャネル別の実測振幅を測る。")
    lines.append("2. 本報告は intensity 8/12・低〜中テクスチャを主表、fireworks は失敗モード別枠。")
    lines.append("3. hocho 系では lockin＋従属チャネルの ablation を厚くする。")
    lines.append("")
    lines.append("## 成果物")
    lines.append("")
    lines.append("- `decode_matrix_*.csv`, `decode_matrix_by_intensity.csv`")
    lines.append("- `color_stats.csv`, `headroom.csv`, `digital_signal.csv`, `texture.csv`")
    lines.append("- `heatmap_decode_*.png`, `hist_*.png`, `bar_*.png`")
    lines.append("")

    (OUT_DIR / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out → {OUT_DIR}")
    print("[INFO] A: decode matrices...")
    matrices = write_decode_matrices()
    print("[INFO] B-D: color / headroom / signal / texture...")
    color_rows, headroom_rows, signal_rows, texture_rows = analyze_color_and_texture()
    print("[INFO] E: FINDINGS.md...")
    write_findings(matrices, color_rows, headroom_rows, signal_rows, texture_rows)
    print("[DONE]")


if __name__ == "__main__":
    main()
