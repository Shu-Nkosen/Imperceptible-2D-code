from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from naming import VideoNameMeta, parse_video_name


@dataclass(frozen=True)
class SyncConfig:
    min_black_frames: int = 5
    min_red_frames: int = 3
    black_v_max: float = 0.4  # HSV V threshold
    red_r_min: float = 0.6    # normalized mean R threshold
    red_g_max: float = 0.3
    red_b_max: float = 0.3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R8: sync(slate) detect -> split 60 conditions -> run analysis -> results.csv")
    p.add_argument("--video", type=str, required=True, help="input video path (e.g. r180_e250_f1.mp4)")
    p.add_argument("--out-dir", type=str, default="", help="output root dir (default: R8/analyze_code/out)")
    p.add_argument("--conditions", type=int, default=60, help="number of blocks/conditions in a video")
    p.add_argument("--block-sec", type=float, default=6.0, help="block duration seconds")
    p.add_argument("--use-start-sec", type=float, default=2.0, help="within-block start sec to extract")
    p.add_argument("--use-end-sec", type=float, default=4.0, help="within-block end sec to extract (exclusive)")
    p.add_argument("--slate-sec", type=float, default=None, help="slate black and slate red duration sec (default: manifest or 0.5)")
    p.add_argument("--padding-sec", type=float, default=None, help="black padding before/after sync slate sec (default: manifest or 5.0)")
    p.add_argument(
        "--max-frames",
        type=int,
        choices=[1, 120],
        default=120,
        help="解析後に残す frame_*.png の枚数（1 or 120）。差分計算は常に120フレーム分。既定: 120",
    )
    p.add_argument("--manifest", type=str, default="", help="optional presenter manifest.json for metadata join")
    p.add_argument("--workers", type=int, default=1, help="passed to decode_qr_from_all_frames.py")
    p.add_argument(
        "--mid-search",
        action="store_true",
        help="passed to decode_qr_from_all_frames.py (gray+median_otsu+Multi+kernels 3/5/7)",
    )
    p.add_argument(
        "--full-search",
        action="store_true",
        help="passed to decode_qr_from_all_frames.py (全バリアント+拡大+Multi)",
    )
    p.add_argument(
        "--diff-mode",
        type=str,
        choices=["pair", "accum"],
        default="pair",
        help="差分モード: pair=隣接ペア（既定） / accum=非重複窓の|差分|合算",
    )
    p.add_argument(
        "--window-ns",
        type=str,
        default="",
        help="accum 時の窓長スイープ（カンマ区切り）。未指定時: 3,4,5",
    )
    p.add_argument(
        "--diff-thresholds",
        type=str,
        default="",
        help="差分閾値スイープ（カンマ区切り）。未指定時: pair=4,8,12 / accum=12,16,24,32",
    )
    return p.parse_args()


def parse_int_list(raw: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    text = (raw or "").strip()
    if not text:
        return default
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise SystemExit(f"空の整数リストです: {raw!r}")
    return tuple(values)

def robust_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        return float(fps)
    return 60.0


def frame_means_bgr(frame: np.ndarray) -> Tuple[float, float, float]:
    b = float(frame[:, :, 0].mean()) / 255.0
    g = float(frame[:, :, 1].mean()) / 255.0
    r = float(frame[:, :, 2].mean()) / 255.0
    return b, g, r


def is_black(frame: np.ndarray, cfg: SyncConfig) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = float(hsv[:, :, 2].mean()) / 255.0
    return v <= cfg.black_v_max


def is_red(frame: np.ndarray, cfg: SyncConfig) -> bool:
    b, g, r = frame_means_bgr(frame)
    return (r >= cfg.red_r_min) and (g <= cfg.red_g_max) and (b <= cfg.red_b_max)


def detect_black_to_red_sync(video_path: Path, cfg: SyncConfig) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    fps = robust_fps(cap)

    state = "seek_black"
    black_run = 0
    red_run = 0
    sync_frame = -1
    idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        if state == "seek_black":
            if is_black(frame, cfg):
                black_run += 1
                if black_run >= cfg.min_black_frames:
                    state = "seek_red"
                    red_run = 0
            else:
                black_run = 0
            continue

        if state == "seek_red":
            if is_red(frame, cfg):
                red_run += 1
                if red_run >= cfg.min_red_frames:
                    sync_frame = idx - (cfg.min_red_frames - 1)
                    break
            else:
                red_run = 0

    cap.release()
    if sync_frame < 0:
        raise RuntimeError("SYNC detection failed (black->red not found). Try adjusting thresholds.")
    return sync_frame, fps


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


ANALYSIS_FRAME_COUNT = 120  # 切り出し枚数（連続フレーム。60fpsならちょうど2秒分）


def select_frame_indices(start_frame: int, end_frame_exclusive: int, max_frames: int) -> List[int]:
    """区間 [start, end) の先頭から連続で最大 max_frames 枚を取る。

    均等間引きだと（例: 2〜5秒から120枚）時間幅が3秒のまま残るため使わない。
    """
    total = max(0, end_frame_exclusive - start_frame)
    if total <= 0:
        return []
    count = min(int(max_frames), total)
    return list(range(start_frame, start_frame + count))


def clear_frame_pngs(out_dir: Path) -> None:
    """前回実行の frame_*.png を消す（シーク失敗時に古い3秒間引きが残るのを防ぐ）。"""
    if not out_dir.exists():
        return
    for path in out_dir.glob("frame_?????.png"):
        path.unlink(missing_ok=True)


def open_capture_at(video_path: Path, frame_idx: int) -> cv2.VideoCapture:
    """条件ごとに VideoCapture を開き直し、指定フレーム直前まで進める。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(str(video_path))

    target = max(0, int(frame_idx))
    if target > 0:
        # 直接シーク（速いが不正確なことがある）
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # 後ろに飛びすぎた／遠い場合は先頭から進め直す
        if pos > target or abs(pos - target) > 2:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(str(video_path))
            pos = 0
        # 足りない分は grab で進める（正確）
        while pos < target:
            if not cap.grab():
                break
            pos += 1
    return cap


def save_block_frames(
    video_path: Path,
    start_frame: int,
    end_frame_exclusive: int,
    out_dir: Path,
    analysis_frames: int = ANALYSIS_FRAME_COUNT,
    resize_to: Optional[Tuple[int, int]] = (1920, 1080),
) -> int:
    """差分計算用に analysis_frames 枚を区間先頭から連続で書き出す。

    フレームごとに CAP_PROP_POS_FRAMES すると MP4 で後半条件だけシークが壊れ、
    0枚書き込みのまま旧フレームが残ることがある。条件ごとに開き直し、連続 read する。
    """
    ensure_dir(out_dir)
    clear_frame_pngs(out_dir)
    indices = select_frame_indices(start_frame, end_frame_exclusive, analysis_frames)
    if not indices:
        return 0

    cap = open_capture_at(video_path, indices[0])
    saved = 0
    try:
        for _ in indices:
            ok, frame = cap.read()
            if not ok:
                break
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"frame_{saved:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
    finally:
        cap.release()
    return saved


def prune_saved_frames(out_dir: Path, keep_frames: int) -> int:
    """解析後に残す frame_*.png を keep_frames 枚に間引く。"""
    paths = sorted(out_dir.glob("frame_?????.png"))
    if not paths:
        return 0
    keep = min(int(keep_frames), len(paths))
    if keep >= len(paths):
        return len(paths)

    if keep == 1:
        keep_set = {paths[len(paths) // 2]}
    else:
        idxs = [
            int(round(i * (len(paths) - 1) / (keep - 1)))
            for i in range(keep)
        ]
        keep_set = {paths[i] for i in idxs}

    for path in paths:
        if path not in keep_set:
            path.unlink(missing_ok=True)

    # 残したファイルを frame_00000... にリネームし直す
    kept = sorted(keep_set, key=lambda p: p.name)
    tmp_paths: List[Path] = []
    for i, path in enumerate(kept):
        tmp = out_dir / f"_keep_{i:05d}.png"
        path.rename(tmp)
        tmp_paths.append(tmp)
    for i, tmp in enumerate(tmp_paths):
        tmp.rename(out_dir / f"frame_{i:05d}.png")
    return len(tmp_paths)


def pick_decode_row_for_folder(rows: List[Dict[str, str]], folder: str) -> Dict[str, str]:
    matches = [r for r in rows if r.get("folder") == folder]
    if not matches:
        return {}
    # 成功行があれば優先、なければ末尾
    for row in matches:
        if str(row.get("success", "")).strip() in ("1", "True", "true"):
            return row
    return matches[-1]


def parse_decode_variant(method: str) -> str:
    """method タグ (例: median_otsu_x1.0_k5_i1) からバリアント名だけ取り出す。"""
    text = str(method or "").strip()
    if not text:
        return ""
    match = re.match(r"^(.+)_x[\d.]+_k\d+_i\d+$", text)
    if match:
        return match.group(1)
    # scale 無しの古い形式など
    match = re.match(r"^(.+)_k\d+_i\d+$", text)
    if match:
        return match.group(1)
    return text


def _parse_accuracy(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pixel_accuracy_for_folder(rows: List[Dict[str, str]], folder: str) -> Tuple[str, str]:
    """folder 単位の accuracy 平均。GT が無ければ両方空。"""
    matches = [r for r in rows if r.get("folder") == folder]
    all_vals: List[float] = []
    ok_vals: List[float] = []
    for row in matches:
        acc = _parse_accuracy(row.get("accuracy"))
        if acc is None:
            continue
        all_vals.append(acc)
        if str(row.get("success", "")).strip() in ("1", "True", "true"):
            ok_vals.append(acc)
    pixel_acc_all = f"{(sum(all_vals) / len(all_vals)):.6f}" if all_vals else ""
    pixel_acc_ok = f"{(sum(ok_vals) / len(ok_vals)):.6f}" if ok_vals else ""
    return pixel_acc_all, pixel_acc_ok


def format_common_meta(
    video_name: str,
    meta: Optional[VideoNameMeta],
    fps: float,
) -> Dict[str, str]:
    """先頭行だけに書く共通メタ（単位・言葉付き）。"""
    if meta is not None:
        display_rate = f"{meta.rate_hz} Hz"
        exposure = f"1/{meta.exp}"
        fluorescent = "蛍光灯あり" if meta.fluoro == 1 else "蛍光灯なし"
    else:
        display_rate = ""
        exposure = ""
        fluorescent = ""
    return {
        "video": video_name,
        "display_rate": display_rate,
        "exposure": exposure,
        "fluorescent": fluorescent,
        "camera_fps": f"{fps:.3f} fps",
    }


# results.csv の固定列順（左: 見たい要約 → 共通メタ → 条件 → デコード詳細）
RESULTS_FIELDNAMES: Tuple[str, ...] = (
    "folder",
    "decode_note",
    "decode_variant",
    "diff_mode",
    "window_n",
    "diff_threshold",
    "pixel_acc_all",
    "pixel_acc_ok",
    "video",
    "display_rate",
    "exposure",
    "fluorescent",
    "camera_fps",
    "cond",
    "image",
    "channel",
    "token",
    "intensity",
    "decode_decoded_text",
    "decode_method",
    "decode_frame_1",
    "decode_frame_2",
    "note",
    "analysis_frames",
    "extract_sec",
)

# 差分二値化閾値の既定スイープ（0-255 dual）
DIFF_THRESHOLDS_PAIR: Tuple[int, ...] = (4, 8, 12)
DIFF_THRESHOLDS_ACCUM: Tuple[int, ...] = (12, 16, 24, 32)
WINDOW_NS_ACCUM: Tuple[int, ...] = (3, 4, 5)
# GT QR 挿入位置（present_session.c と同じ: before cond 0 / 30 / after last）
DEFAULT_GT_QR_INSERT_BEFORE: Tuple[int, ...] = (0, 30, 60)
DEFAULT_GT_QR_SEC = 3.0


def build_content_timeline(
    slate_sec: float,
    padding_sec: float,
    block_sec: float,
    n_conditions: int,
    qr_sec: float,
    qr_insert_before: Iterable[int],
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """sync(赤 onset) からの秒オフセットで、各条件開始と GT QR スロットを返す。"""
    inserts = sorted({int(x) for x in qr_insert_before})
    t = float(slate_sec) + float(padding_sec)
    cond_starts: List[float] = []
    slots: List[Dict[str, Any]] = []
    insert_set = set(inserts)

    for i in range(n_conditions + 1):
        if i in insert_set and qr_sec > 0:
            slots.append(
                {
                    "name": {0: "start", 30: "mid", 60: "end"}.get(i, f"before_{i}"),
                    "insert_before_cond": i,
                    "start_sec_from_sync": t,
                    "duration_sec": float(qr_sec),
                }
            )
            t += float(qr_sec)
        if i < n_conditions:
            cond_starts.append(t)
            t += float(block_sec)
    return cond_starts, slots


def resolve_gt_qr_timeline(
    manifest: Dict[str, Any],
    slate_sec: float,
    padding_sec: float,
    block_sec: float,
    n_conditions: int,
) -> Tuple[List[float], List[Dict[str, Any]], float]:
    """manifest に GT QR 定義があれば差し込みタイムライン、無ければ従来（QRなし）。"""
    slots_m = manifest.get("gt_qr_slots") if isinstance(manifest, dict) else None
    has_qr = isinstance(slots_m, list) and len(slots_m) > 0
    if not has_qr and isinstance(manifest, dict) and "gt_qr_sec" in manifest:
        has_qr = float(manifest.get("gt_qr_sec", 0) or 0) > 0

    if not has_qr:
        # 旧動画互換: QRなし、条件は post-padding 直後から連続
        cond_starts, _ = build_content_timeline(
            slate_sec, padding_sec, block_sec, n_conditions, 0.0, ()
        )
        return cond_starts, [], 0.0

    qr_sec = float(manifest.get("gt_qr_sec", DEFAULT_GT_QR_SEC))
    inserts: List[int] = []
    if isinstance(slots_m, list):
        for item in slots_m:
            try:
                inserts.append(int(item.get("insert_before_cond")))
            except Exception:
                continue
    if not inserts:
        inserts = list(DEFAULT_GT_QR_INSERT_BEFORE)

    cond_starts, slots = build_content_timeline(
        slate_sec, padding_sec, block_sec, n_conditions, qr_sec, inserts
    )
    by_before = {
        int(s.get("insert_before_cond", -1)): s
        for s in (slots_m or [])
        if isinstance(s, dict)
    }
    for s in slots:
        m = by_before.get(int(s["insert_before_cond"]))
        if m and m.get("name"):
            s["name"] = str(m["name"])
    return cond_starts, slots, qr_sec


def frame_to_gt_mask(frame: np.ndarray, out_size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """撮影した GT QR 表示を、decode 用の白背景・黒モジュール画像にする。"""
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 背景を白・モジュールを黒に揃える
    if float(np.mean(binary)) < 127.0:
        binary = 255 - binary
    return binary


def extract_gt_qr_mask(
    video_path: Path,
    sync_frame: int,
    fps: float,
    slots: List[Dict[str, Any]],
    out_path: Path,
    samples_per_slot: int = 5,
) -> bool:
    """manifest の GT QR スロット中央付近からマスクを合成して frame_QR.png を書く。"""
    if not slots or fps <= 0:
        return False

    masks: List[np.ndarray] = []
    for slot in slots:
        start_sec = float(slot.get("start_sec_from_sync", 0.0))
        dur_sec = float(slot.get("duration_sec", DEFAULT_GT_QR_SEC))
        if dur_sec <= 0:
            continue
        slot_masks = 0
        mid0 = start_sec + dur_sec * 0.25
        mid1 = start_sec + dur_sec * 0.75
        for k in range(samples_per_slot):
            frac = k / max(1, samples_per_slot - 1)
            t = mid0 + (mid1 - mid0) * frac
            frame_idx = sync_frame + int(round(t * fps))
            cap = open_capture_at(video_path, frame_idx)
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok or frame is None:
                continue
            try:
                masks.append(frame_to_gt_mask(frame))
                slot_masks += 1
            except Exception:
                continue
        print(
            f"[INFO] GT QR slot={slot.get('name')} "
            f"start_sec_from_sync={start_sec:.3f} dur={dur_sec:.3f} samples_ok={slot_masks}"
        )

    if not masks:
        print("[WARN] GT QR: no frames extracted; pixel_acc will stay empty")
        return False

    stacked = np.stack(masks, axis=0)
    # 画素ごとに黒(=0)多数決
    black_votes = np.sum(stacked < 128, axis=0)
    merged = np.where(black_votes >= (len(masks) / 2.0), 0, 255).astype(np.uint8)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), merged)
    print(f"[OK] wrote GT mask: {out_path} (from {len(masks)} samples)")
    return True


def read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_py(script: Path, cwd: Path, args: List[str]) -> None:
    cmd = ["python", str(script), *args]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_decode_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    rows_list = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows_list:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_list:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_results_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """条件サマリ results.csv（固定列順）。"""
    write_csv(path, rows, fieldnames=list(RESULTS_FIELDNAMES))


# present_session.c と同じ並び: channel → image → intensity
BASE_IMAGES = ("rice", "nagaoka_fireworks", "hocho", "ex")
CHANNELS = ("R", "G", "B", "min", "max")
TOKENS = ("R", "G", "B", "I", "X")
INTENSITIES = (4, 8, 12)


def build_default_conditions() -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    idx = 0
    for channel, token in zip(CHANNELS, TOKENS):
        for image in BASE_IMAGES:
            for intensity in INTENSITIES:
                conditions.append(
                    {
                        "idx": idx,
                        "image": image,
                        "channel": channel,
                        "token": token,
                        "intensity": intensity,
                    }
                )
                idx += 1
    return conditions


def condition_folder_name(cond: Dict[str, Any]) -> str:
    image = str(cond.get("image", "unknown"))
    token = str(cond.get("token", "X"))
    intensity = int(cond.get("intensity", 0))
    return f"{image}_{token}_{intensity}"


def resolve_conditions(manifest: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    items = manifest.get("conditions") if isinstance(manifest, dict) else None
    if isinstance(items, list) and items:
        by_idx: Dict[int, Dict[str, Any]] = {}
        for item in items:
            try:
                idx = int(item.get("idx"))
            except Exception:
                continue
            by_idx[idx] = dict(item)
        resolved: List[Dict[str, Any]] = []
        for i in range(n):
            if i in by_idx:
                resolved.append(by_idx[i])
            else:
                raise SystemExit(f"manifest conditions に idx={i} がありません")
        return resolved

    defaults = build_default_conditions()
    if n != len(defaults):
        raise SystemExit(
            f"conditions={n} ですが default は {len(defaults)} 件です。"
            " --manifest を渡すか --conditions を合わせてください。"
        )
    return defaults[:n]


def main() -> None:
    ns = parse_args()

    video_path = Path(ns.video).resolve()
    meta: Optional[VideoNameMeta] = parse_video_name(video_path)
    if meta is None:
        print(f"[WARN] video name does not match r{{rate}}_e{{exp}}_f{{0|1}}.mp4: {video_path.name}")

    base_out = Path(ns.out_dir) if ns.out_dir else (Path(__file__).resolve().parent / "out")
    if not base_out.is_absolute():
        base_out = (Path(__file__).resolve().parent / base_out).resolve()

    out_root = base_out / (meta.stem if meta else video_path.stem)
    ensure_dir(out_root)

    manifest = read_manifest(Path(ns.manifest).resolve()) if ns.manifest else {}

    slate_sec = float(ns.slate_sec) if ns.slate_sec is not None else float(manifest.get("slate_sec", 0.5))
    padding_sec = float(ns.padding_sec) if ns.padding_sec is not None else float(manifest.get("padding_sec", 5.0))
    block_sec = float(ns.block_sec)

    cfg = SyncConfig()
    sync_frame, fps = detect_black_to_red_sync(video_path, cfg)
    print(f"[OK] SYNC at frame={sync_frame}, fps={fps:.4f}")

    use_start = int(round(float(ns.use_start_sec) * fps))
    use_end = int(round(float(ns.use_end_sec) * fps))
    if use_end <= use_start:
        raise SystemExit("use-end-sec must be > use-start-sec")
    use_duration_sec = float(ns.use_end_sec) - float(ns.use_start_sec)
    print(
        f"[INFO] use_window=[{ns.use_start_sec:.3f}, {ns.use_end_sec:.3f})s "
        f"duration={use_duration_sec:.3f}s / fps={fps:.4f} -> "
        f"window_frames={use_end - use_start}, analysis_frames={ANALYSIS_FRAME_COUNT} (consecutive)"
    )

    # sync_frame = red onset. Content timeline includes GT QR slots interleaved with conditions.
    conditions = resolve_conditions(manifest, int(ns.conditions))
    cond_starts_sec, gt_slots, qr_sec = resolve_gt_qr_timeline(
        manifest, slate_sec, padding_sec, block_sec, len(conditions)
    )
    print(
        f"[INFO] timeline from sync: slate+padding={slate_sec + padding_sec:.3f}s, "
        f"gt_qr_sec={qr_sec:.3f}, slots={len(gt_slots)}, conditions={len(conditions)}"
    )
    for s in gt_slots:
        print(
            f"[INFO]   QR {s.get('name')}: start_sec_from_sync={s['start_sec_from_sync']:.3f} "
            f"(before cond {s.get('insert_before_cond')})"
        )

    gt_path = out_root / "frame_QR.png"
    extract_gt_qr_mask(video_path, sync_frame, fps, gt_slots, gt_path)

    print(
        f"[INFO] block0(cond0) ≈ sync + {cond_starts_sec[0]:.3f}s "
        f"(frame {sync_frame + int(round(cond_starts_sec[0] * fps))}), "
        f"slate_sec={slate_sec}, padding_sec={padding_sec}, block_sec={block_sec}"
    )

    script_dir = Path(__file__).resolve().parent
    diff_script = script_dir / "cal-from-2frame-RGB-oute.py"
    decode_script = script_dir / "decode_qr_from_all_frames.py"
    results_csv = out_root / "results.csv"
    decode_csv = out_root / "qr_decode_all_frames.csv"
    results: List[Dict[str, Any]] = []
    all_decode_rows: List[Dict[str, str]] = []

    diff_mode = str(ns.diff_mode)
    if diff_mode == "accum":
        window_ns = parse_int_list(ns.window_ns, WINDOW_NS_ACCUM)
        if any(n < 2 for n in window_ns):
            raise SystemExit("--window-ns の各値は 2 以上である必要があります")
        diff_thresholds = parse_int_list(ns.diff_thresholds, DIFF_THRESHOLDS_ACCUM)
        sweep: List[Tuple[Optional[int], int]] = [(n, th) for n in window_ns for th in diff_thresholds]
    else:
        window_ns = ()
        diff_thresholds = parse_int_list(ns.diff_thresholds, DIFF_THRESHOLDS_PAIR)
        sweep = [(None, th) for th in diff_thresholds]

    print(
        f"[INFO] analysis_frames={ANALYSIS_FRAME_COUNT} / keep_frames={ns.max_frames} "
        f"/ diff_mode={diff_mode} "
        f"/ window_ns={list(window_ns) if window_ns else '-'} "
        f"/ diff_thresholds={list(diff_thresholds)} "
        "/ results+decode CSV overwritten after each condition (accumulated)"
    )

    for i, cond in enumerate(conditions):
        block_start = sync_frame + int(round(cond_starts_sec[i] * fps))
        start = block_start + use_start
        end = block_start + use_end
        folder_name = condition_folder_name(cond)
        cond_dir = out_root / folder_name
        cond_dir_abs = str(cond_dir.resolve())
        out_root_abs = str(out_root.resolve())

        analyzed = save_block_frames(
            video_path,
            start,
            end,
            cond_dir,
            analysis_frames=ANALYSIS_FRAME_COUNT,
        )
        extract_sec = (analyzed / fps) if fps > 0 else 0.0
        print(
            f"[INFO] ({i+1}/{len(conditions)}) extract {folder_name}: "
            f"frames={analyzed} span≈{extract_sec:.3f}s "
            f"(video [{start}, {start + analyzed}))"
        )
        if analyzed < ANALYSIS_FRAME_COUNT:
            print(
                f"[WARN] {folder_name}: expected {ANALYSIS_FRAME_COUNT} frames, got {analyzed}. "
                "切り出しが不完全です（シーク失敗の可能性）。"
            )

        decode_note = ""
        best_dec: Dict[str, str] = {}
        best_th: Optional[int] = None
        best_window_n: Optional[int] = None
        best_rows: List[Dict[str, str]] = []
        folder_decode_rows: List[Dict[str, str]] = []
        # (window_n, th, decode_row, rows, pixel_acc_ok)
        success_candidates: List[
            Tuple[Optional[int], int, Dict[str, str], List[Dict[str, str]], float]
        ] = []

        if analyzed <= 0:
            decode_note = "extract失敗: 0 frames"
            print(f"[WARN] {folder_name}: {decode_note}")
        else:
            for win_n, th in sweep:
                if diff_mode == "accum":
                    assert win_n is not None
                    diff_subdir = f"rgb_max_accum_n{win_n}_th{th}"
                    print(
                        f"[INFO] ({i+1}/{len(conditions)}) {folder_name}: "
                        f"accum window_n={win_n} threshold={th}"
                    )
                    diff_args = [
                        "--base-dir",
                        cond_dir_abs,
                        "--diff-mode",
                        "accum",
                        "--window-n",
                        str(win_n),
                        "--threshold",
                        str(th),
                        "--output-subdir",
                        diff_subdir,
                    ]
                else:
                    diff_subdir = f"rgb_max_diff_maps_th{th}"
                    print(f"[INFO] ({i+1}/{len(conditions)}) {folder_name}: threshold={th}")
                    diff_args = [
                        "--base-dir",
                        cond_dir_abs,
                        "--diff-mode",
                        "pair",
                        "--threshold",
                        str(th),
                        "--output-subdir",
                        diff_subdir,
                    ]

                try:
                    run_py(diff_script, cwd=out_root, args=diff_args)
                except subprocess.CalledProcessError as exc:
                    label = f"n={win_n} th={th}" if win_n is not None else f"th={th}"
                    print(f"[WARN] {folder_name}: diff失敗 {label} exit={exc.returncode}")
                    continue

                decode_args = [
                    "--base-dir",
                    out_root_abs,
                    "--folder",
                    folder_name,
                    "--diff-subdir",
                    diff_subdir,
                    "--workers",
                    str(int(ns.workers)),
                    "--no-save-analysis",
                ]
                if ns.full_search:
                    decode_args.append("--full-search")
                elif ns.mid_search:
                    decode_args.append("--mid-search")
                try:
                    run_py(decode_script, cwd=out_root, args=decode_args)
                except subprocess.CalledProcessError as exc:
                    label = f"n={win_n} th={th}" if win_n is not None else f"th={th}"
                    print(f"[WARN] {folder_name}: decode失敗 {label} exit={exc.returncode}")
                    continue

                th_rows = [r for r in load_decode_csv(decode_csv) if r.get("folder") == folder_name]
                tagged_rows: List[Dict[str, str]] = []
                for r in th_rows:
                    row_th = dict(r)
                    row_th["diff_mode"] = diff_mode
                    row_th["window_n"] = "" if win_n is None else str(win_n)
                    row_th["diff_threshold"] = str(th)
                    tagged_rows.append(row_th)
                folder_decode_rows.extend(tagged_rows)

                dec_th = pick_decode_row_for_folder(th_rows, folder_name)
                ok = str(dec_th.get("success", "")).strip() in ("1", "True", "true")
                if ok:
                    _, acc_ok = pixel_accuracy_for_folder(th_rows, folder_name)
                    acc_val = float(acc_ok) if acc_ok else -1.0
                    success_candidates.append((win_n, th, dec_th, th_rows, acc_val))
                    n_label = f"n={win_n} " if win_n is not None else ""
                    print(
                        f"[OK] {folder_name}: success at {n_label}threshold={th} "
                        f"(pixel_acc_ok={acc_ok or 'n/a'})"
                    )

            if success_candidates:
                # 成功のうち pixel_acc_ok 最大 → 小さい th → 小さい n
                success_candidates.sort(
                    key=lambda x: (-x[4], x[1], x[0] if x[0] is not None else 0)
                )
                best_window_n, best_th, best_dec, best_rows, _ = success_candidates[0]
                n_label = f"n={best_window_n} " if best_window_n is not None else ""
                print(f"[OK] {folder_name}: selected {n_label}threshold={best_th}")
            elif folder_decode_rows:
                last = folder_decode_rows[-1]
                best_th = int(last.get("diff_threshold") or diff_thresholds[-1])
                win_raw = str(last.get("window_n") or "").strip()
                best_window_n = int(win_raw) if win_raw else None
                best_rows = [
                    r
                    for r in folder_decode_rows
                    if r.get("diff_threshold") == str(best_th)
                    and r.get("window_n", "") == ("" if best_window_n is None else str(best_window_n))
                ]
                best_dec = pick_decode_row_for_folder(
                    [
                        {
                            k: v
                            for k, v in r.items()
                            if k not in ("diff_threshold", "diff_mode", "window_n")
                        }
                        for r in best_rows
                    ],
                    folder_name,
                )
            else:
                decode_note = "diff/decode失敗: 全threshold"
                print(f"[WARN] {folder_name}: {decode_note}")

            all_decode_rows = [r for r in all_decode_rows if r.get("folder") != folder_name]
            all_decode_rows.extend(folder_decode_rows)
            write_csv(decode_csv, all_decode_rows)

        kept = prune_saved_frames(cond_dir, keep_frames=int(ns.max_frames))
        print(f"[INFO] ({i+1}/{len(conditions)}) keep frames={kept}")

        dec = best_dec
        pixel_acc_all, pixel_acc_ok = pixel_accuracy_for_folder(best_rows, folder_name)
        method = dec.get("method", "") if dec else ""
        decode_variant = parse_decode_variant(method) if method else ""

        if decode_note:
            decode_note_out = decode_note
        elif not dec:
            decode_note_out = "デコード結果なし"
        elif str(dec.get("success", "")).strip() in ("1", "True", "true"):
            decode_note_out = ""
        else:
            decode_note_out = str(dec.get("note") or "QR未検出")
            decode_variant = ""

        row: Dict[str, Any] = {
            "folder": folder_name,
            "decode_note": decode_note_out,
            "decode_variant": decode_variant if not decode_note_out else "",
            "diff_mode": diff_mode,
            "window_n": "" if best_window_n is None else str(best_window_n),
            "diff_threshold": "" if best_th is None else str(best_th),
            "pixel_acc_all": pixel_acc_all,
            "pixel_acc_ok": pixel_acc_ok,
            "cond": i,
            "image": cond.get("image", ""),
            "channel": cond.get("channel", ""),
            "token": cond.get("token", ""),
            "intensity": cond.get("intensity", ""),
            "decode_decoded_text": dec.get("decoded_text", "") if dec else "",
            "decode_method": method if not decode_note_out else "",
            "decode_frame_1": dec.get("frame_1", "") if dec else "",
            "decode_frame_2": dec.get("frame_2", "") if dec else "",
            "note": decode_note,
            "analysis_frames": analyzed,
            "extract_sec": f"{extract_sec:.6f}",
        }
        # 共通メタは先頭行だけ（単位・言葉付き）
        if i == 0:
            row.update(format_common_meta(video_path.name, meta, fps))
        else:
            row.update(
                {
                    "video": "",
                    "display_rate": "",
                    "exposure": "",
                    "fluorescent": "",
                    "camera_fps": "",
                }
            )

        results.append(row)
        write_results_csv(results_csv, results)
        print(
            f"[OK] ({i+1}/{len(conditions)}) wrote {results_csv.name} ({len(results)} rows), "
            f"{decode_csv.name} ({len(all_decode_rows)} decode rows)"
        )

    print(f"[OK] results: {results_csv}")
    print(f"[OK] decode: {decode_csv}")


if __name__ == "__main__":
    main()

