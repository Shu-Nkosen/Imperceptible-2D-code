#!/usr/bin/env python3
"""Visibility (flicker) rating experiment at 60 Hz display.

Uses existing normal/inv embedded PNGs from gen_assets.py.
Practice: 1 trial. Main: 4 images x RGB x intensities 4/8/12/16 = 48 trials.

Usage:
  python visibility_experiment.py
  python visibility_experiment.py --name 山田
  python visibility_experiment.py --id P01
  python gen_assets.py --images rice,nagaoka_fireworks,hocho,ex \\
      --intensities 4,8,12,16 --channels R,G,B --clip-margin 16
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import glfw
import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_LINEAR,
    GL_NEAREST,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGBA,
    GL_RGB,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
)
from PIL import Image, ImageDraw, ImageFont

# #region agent log
_DEBUG_LOG = Path(__file__).resolve().parents[2] / "debug-d29637.log"
_DEBUG_LOG_LOCAL = Path(__file__).resolve().parent / "debug-d29637.log"
_DEBUG_SESSION = "d29637"


def _dbg(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION,
            "runId": "post-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        for path in (_DEBUG_LOG, _DEBUG_LOG_LOCAL):
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass
    except Exception:
        pass


# #endregion

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "visibility_results"

IMAGES = ["ex", "rice", "hocho", "nagaoka_fireworks"]
CHANNELS = ["R", "G", "B"]
INTENSITIES = [4, 8, 12, 16]

PRACTICE_IMAGE = "rice"
PRACTICE_CHANNEL = "G"
PRACTICE_INTENSITY = 8

ISI_SEC = 1.0
STIM_SEC = 3.0
TARGET_HZ = 60

ISI_GRAY = (128, 128, 128)
BG_DARK = (32, 32, 32)

RATING_LABELS = {
    1: "全く分からない",
    2: "よく見ると、わずかにちらつく",
    3: "ちらつきが分かる",
    4: "QRコードのようなものが見える",
}

CSV_FIELDS = [
    "participant_id",
    "participant_name",
    "trial_index",
    "is_practice",
    "image",
    "channel",
    "intensity",
    "rating",
    "rt_ms",
    "t_isi",
    "t_stim",
    "t_response",
    "aborted",
]

# ---------------------------------------------------------------------------
# Windows timer
# ---------------------------------------------------------------------------

_winmm = None
try:
    _winmm = ctypes.WinDLL("winmm")
    _winmm.timeBeginPeriod(1)
except Exception:
    _winmm = None


def _time_end() -> None:
    if _winmm is not None:
        try:
            _winmm.timeEndPeriod(1)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    trial_index: int  # 0 for practice, 1..48 for main
    is_practice: bool
    image: str
    channel: str
    intensity: int

    @property
    def normal_name(self) -> str:
        return f"{self.image}_{self.intensity}_normal{self.channel}.png"

    @property
    def inv_name(self) -> str:
        return f"{self.image}_{self.intensity}_inv{self.channel}.png"


def build_main_trials(seed: int) -> list[Trial]:
    trials: list[Trial] = []
    idx = 1
    for image in IMAGES:
        for ch in CHANNELS:
            for intensity in INTENSITIES:
                trials.append(
                    Trial(
                        trial_index=idx,
                        is_practice=False,
                        image=image,
                        channel=ch,
                        intensity=intensity,
                    )
                )
                idx += 1
    rng = random.Random(seed)
    rng.shuffle(trials)
    # re-number after shuffle so trial_index reflects presentation order
    return [
        Trial(
            trial_index=i + 1,
            is_practice=False,
            image=t.image,
            channel=t.channel,
            intensity=t.intensity,
        )
        for i, t in enumerate(trials)
    ]


def practice_trial() -> Trial:
    return Trial(
        trial_index=0,
        is_practice=True,
        image=PRACTICE_IMAGE,
        channel=PRACTICE_CHANNEL,
        intensity=PRACTICE_INTENSITY,
    )


def required_asset_names() -> list[str]:
    names: set[str] = set()
    for image in IMAGES:
        for ch in CHANNELS:
            for intensity in INTENSITIES:
                names.add(f"{image}_{intensity}_normal{ch}.png")
                names.add(f"{image}_{intensity}_inv{ch}.png")
    names.add(f"{PRACTICE_IMAGE}_{PRACTICE_INTENSITY}_normal{PRACTICE_CHANNEL}.png")
    names.add(f"{PRACTICE_IMAGE}_{PRACTICE_INTENSITY}_inv{PRACTICE_CHANNEL}.png")
    return sorted(names)


def check_assets(asset_dir: Path) -> None:
    missing = [n for n in required_asset_names() if not (asset_dir / n).exists()]
    if missing:
        preview = "\n  ".join(missing[:12])
        more = f"\n  ... and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise SystemExit(
            "Missing embedded assets. Generate with:\n"
            "  python gen_assets.py --images rice,nagaoka_fireworks,hocho,ex "
            "--intensities 4,8,12,16 --channels R,G,B --clip-margin 16\n"
            f"Missing ({len(missing)}):\n  {preview}{more}"
        )


# ---------------------------------------------------------------------------
# Session / CSV persistence
# ---------------------------------------------------------------------------


def session_dir(participant_id: str) -> Path:
    safe = "".join(c for c in participant_id if c.isalnum() or c in ("-", "_"))
    if not safe:
        raise SystemExit("participant id must contain alphanumeric characters")
    return RESULTS_ROOT / safe


PARTICIPANTS_PATH = RESULTS_ROOT / "participants.json"


def _norm_name(name: str) -> str:
    return " ".join(name.strip().split())


def load_roster() -> list[dict]:
    people: list[dict] = []
    if PARTICIPANTS_PATH.exists():
        try:
            raw = json.loads(PARTICIPANTS_PATH.read_text(encoding="utf-8"))
            people = list(raw.get("participants", raw if isinstance(raw, list) else []))
        except (OSError, json.JSONDecodeError):
            people = []
    known = {str(p.get("id", "")) for p in people}
    if RESULTS_ROOT.exists():
        for d in sorted(RESULTS_ROOT.iterdir()):
            if not d.is_dir():
                continue
            if d.name in known:
                continue
            people.append(
                {
                    "id": d.name,
                    "name": "",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            known.add(d.name)
    return people


def save_roster(people: list[dict]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"participants": people}
    tmp = PARTICIPANTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PARTICIPANTS_PATH)


def next_participant_id(people: list[dict]) -> str:
    n = 0
    for p in people:
        m = re.fullmatch(r"P(\d+)", str(p.get("id", "")), flags=re.IGNORECASE)
        if m:
            n = max(n, int(m.group(1)))
    return f"P{n + 1:02d}"


def print_roster(people: list[dict]) -> None:
    print("登録済みの観察者:")
    if not people:
        print("  （まだいません）")
        return
    for p in people:
        label = p.get("name") or "（名前未登録）"
        print(f"  {p.get('id', '?')}  {label}")


def resolve_participant(name: str, id_hint: str) -> tuple[str, str]:
    """Return (id, name). Registers a new person when needed."""
    people = load_roster()
    name = _norm_name(name)
    id_hint = id_hint.strip()

    by_id = {str(p.get("id", "")).lower(): p for p in people if p.get("id")}
    by_name = {
        _norm_name(str(p.get("name", ""))).lower(): p
        for p in people
        if _norm_name(str(p.get("name", "")))
    }

    if name and name.lower() in by_name:
        found = by_name[name.lower()]
        found_id = str(found["id"])
        if id_hint and found_id.lower() != id_hint.lower():
            raise SystemExit(f"名前「{name}」は既に {found_id} です（指定ID: {id_hint}）")
        return found_id, str(found.get("name") or name)

    if id_hint and id_hint.lower() in by_id:
        found = by_id[id_hint.lower()]
        found_id = str(found["id"])
        existing_name = _norm_name(str(found.get("name", "")))
        if name and existing_name and existing_name.lower() != name.lower():
            raise SystemExit(
                f"ID {found_id} は既に「{existing_name}」です（指定名: {name}）"
            )
        if name and not existing_name:
            found["name"] = name
            save_roster(people)
            return found_id, name
        return found_id, existing_name or name

    if not name:
        raise SystemExit("観察者の名前が必要です")

    new_id = id_hint if id_hint else next_participant_id(people)
    if new_id.lower() in by_id:
        raise SystemExit(f"ID {new_id} は既に使われています")
    people.append(
        {
            "id": new_id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_roster(people)
    print(f"[INFO] 新規登録: {name} ({new_id})")
    return new_id, name


def prompt_identity(args: argparse.Namespace) -> tuple[str, str]:
    people = load_roster()
    name = _norm_name(getattr(args, "name", "") or "")
    pid = (getattr(args, "id", "") or "").strip()

    if not name and not pid:
        print_roster(people)
        try:
            raw = input("観察者の名前を入力してください（新規はそのまま登録。ID でも可）: ").strip()
        except EOFError:
            raw = ""
        if re.fullmatch(r"P\d+", raw, flags=re.IGNORECASE):
            pid = raw
        else:
            name = raw
    elif not name and pid:
        found = next(
            (p for p in people if str(p.get("id", "")).lower() == pid.lower()),
            None,
        )
        if found and _norm_name(str(found.get("name", ""))):
            name = _norm_name(str(found["name"]))
        else:
            print_roster(people)
            try:
                name = input(f"ID {pid} の名前を入力してください: ").strip()
            except EOFError:
                name = ""

    return resolve_participant(name, pid)


def append_rating(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        f.flush()


def save_session(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def create_session(
    participant_id: str,
    participant_name: str,
    monitor_hz: float,
) -> tuple[dict, list[Trial], Path, Path]:
    """Always start a new run. Previous runs stay in timestamped folders."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = session_dir(participant_id) / stamp
    out.mkdir(parents=True, exist_ok=True)
    session_path = out / "session.json"
    csv_path = out / "ratings.csv"

    seed = random.randrange(0, 2**31 - 1)
    trials = build_main_trials(seed)
    data = {
        "participant_id": participant_id,
        "participant_name": participant_name,
        "seed": seed,
        "monitor_hz": monitor_hz,
        "target_hz": TARGET_HZ,
        "stim_sec": STIM_SEC,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "practice_done": False,
        "aborted": False,
        "completed": False,
        "trial_order": [
            {
                "trial_index": t.trial_index,
                "image": t.image,
                "channel": t.channel,
                "intensity": t.intensity,
            }
            for t in trials
        ],
    }
    save_session(session_path, data)
    print(f"[INFO] new run -> {out}")
    return data, trials, session_path, csv_path


# ---------------------------------------------------------------------------
# OpenGL helpers
# ---------------------------------------------------------------------------


def load_texture_from_path(path: Path, size: tuple[int, int] | None = None) -> int:
    """Load PNG as GL texture. Resize only when size is given and differs from image."""
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if size is not None and (img.width, img.height) != size:
        img = img.resize(size, Image.NEAREST)
    data = np.array(img, dtype=np.uint8)
    # #region agent log
    _dbg(
        "E",
        "visibility_experiment.py:load_texture_from_path",
        "loaded png",
        {
            "path": str(path.name),
            "shape": list(data.shape),
            "mean": float(data.mean()),
            "sum": int(data.sum()),
            "resized_to": list(size) if size is not None else None,
        },
    )
    # #endregion
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    if img.mode == "RGBA":
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            img.width,
            img.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            data,
        )
    else:
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            img.width,
            img.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            data,
        )
    return tex


def load_texture_from_rgba(arr: np.ndarray) -> int:
    """arr: HxWx4 uint8. HUD only (LINEAR is fine for text)."""
    h, w = arr.shape[:2]
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr
    )
    return tex


def solid_texture(rgb: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> int:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = rgb
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        size[0],
        size[1],
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        arr,
    )
    return tex


def draw_fullscreen(tex: int) -> None:
    glClear(GL_COLOR_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(-1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, -1)
    glTexCoord2f(1, 0)
    glVertex2f(1, 1)
    glTexCoord2f(0, 0)
    glVertex2f(-1, 1)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_overlay(tex: int) -> None:
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(-1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, -1)
    glTexCoord2f(1, 0)
    glVertex2f(1, 1)
    glTexCoord2f(0, 0)
    glVertex2f(-1, 1)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)


def setup_ortho() -> None:
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def frames_for_seconds(sec: float, refresh_hz: float) -> int:
    """Match present_session calc_frames with interval=1."""
    if sec <= 0:
        return 0
    n = int(sec * float(refresh_hz) + 0.5)
    return max(1, n)


# ---------------------------------------------------------------------------
# HUD (PIL overlays, rebuilt only when state changes)
# ---------------------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_hud(
    width: int,
    height: int,
    *,
    progress: str,
    show_abort: bool,
    title: str = "",
    body_lines: Optional[list[str]] = None,
    footer: str = "",
    params: str = "",
) -> np.ndarray:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_sm = _font(22)
    font_md = _font(28)
    font_lg = _font(36)

    def _draw_abort() -> None:
        if not show_abort:
            return
        bx0, by0, bx1, by1 = width - 140, 16, width - 24, 56
        draw.rectangle((bx0, by0, bx1, by1), fill=(60, 60, 60, 220), outline=(200, 200, 200, 220))
        draw.text((bx0 + 28, by0 + 8), "中断", fill=(255, 255, 255, 240), font=font_md)

    def _draw_top_left() -> None:
        if not progress and not params:
            return
        box_w = 520 if params else 220
        box_h = 72 if params else 48
        draw.rectangle((12, 12, 12 + box_w, 12 + box_h), fill=(0, 0, 0, 160))
        if progress:
            draw.text((20, 16), progress, fill=(255, 255, 255, 230), font=font_sm)
        if params:
            draw.text((20, 42), params, fill=(200, 200, 200, 220), font=font_sm)

    _draw_abort()

    # title / body (start / rating / end screens)
    if title:
        draw.rectangle((0, 0, width, height), fill=(20, 20, 20, 230))
        draw.text((width // 2 - 200, height // 5), title, fill=(255, 255, 255, 255), font=font_lg)
        y = height // 5 + 70
        for line in body_lines or []:
            draw.text((width // 2 - 380, y), line, fill=(230, 230, 230, 255), font=font_md)
            y += 40
        if footer:
            draw.text(
                (width // 2 - 200, height * 4 // 5),
                footer,
                fill=(200, 220, 255, 255),
                font=font_md,
            )
        _draw_abort()

    # last: dark panel must not cover progress / params
    _draw_top_left()

    return np.array(img, dtype=np.uint8)


# Abort button hit box in pixel coords (matches make_hud)
def abort_hit_box(width: int, height: int) -> tuple[int, int, int, int]:
    return width - 140, 16, width - 24, 56


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class ExperimentApp:
    def __init__(
        self,
        participant_id: str,
        participant_name: str,
        asset_dir: Path,
        window_mode: bool = False,
    ) -> None:
        self.participant_id = participant_id
        self.participant_name = participant_name
        self.asset_dir = asset_dir
        self.window_mode = window_mode

        self.window = None
        self.width = 1920
        self.height = 1080
        self.monitor_hz = float(TARGET_HZ)
        self.swap_interval_n = 1
        self.present_hz = float(TARGET_HZ)
        self.hold_frames = 1
        self.flicker_hz = 30.0

        self.tex_cache: dict[str, int] = {}
        self.tex_gray = 0
        self.tex_dark = 0
        self.hud_tex: Optional[int] = None
        self.hud_key: Optional[tuple] = None

        self.abort_requested = False
        self.enter_pressed = False
        self.rating_key: Optional[int] = None
        self.mouse_clicked = False
        self.mouse_xy = (0.0, 0.0)

        self.session: dict = {}
        self.session_path: Path = Path()
        self.csv_path: Path = Path()
        self.main_trials: list[Trial] = []
        self._fps_logged = False

    # ---- GLFW callbacks ----

    def _on_key(self, window, key, scancode, action, mods) -> None:  # noqa: ARG002
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            self.abort_requested = True
            return
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self.enter_pressed = True
            return
        mapping = {
            glfw.KEY_1: 1,
            glfw.KEY_2: 2,
            glfw.KEY_3: 3,
            glfw.KEY_4: 4,
            glfw.KEY_KP_1: 1,
            glfw.KEY_KP_2: 2,
            glfw.KEY_KP_3: 3,
            glfw.KEY_KP_4: 4,
        }
        if key in mapping:
            self.rating_key = mapping[key]

    def _on_mouse(self, window, button, action, mods) -> None:  # noqa: ARG002
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.mouse_clicked = True
            self.mouse_xy = glfw.get_cursor_pos(window)

    def _check_abort_click(self) -> bool:
        if not self.mouse_clicked:
            return False
        self.mouse_clicked = False
        x, y = self.mouse_xy
        # GLFW cursor is top-left origin in window coords
        ax0, ay0, ax1, ay1 = abort_hit_box(self.width, self.height)
        return ax0 <= x <= ax1 and ay0 <= y <= ay1

    # ---- textures ----

    def _ensure_pair(self, trial: Trial) -> tuple[int, int]:
        n_path = self.asset_dir / trial.normal_name
        i_path = self.asset_dir / trial.inv_name
        # Resize only if asset resolution differs from the fullscreen window
        size = (self.width, self.height)
        if trial.normal_name not in self.tex_cache:
            self.tex_cache[trial.normal_name] = load_texture_from_path(n_path, size)
        if trial.inv_name not in self.tex_cache:
            self.tex_cache[trial.inv_name] = load_texture_from_path(i_path, size)
        # #region agent log
        try:
            na = np.array(Image.open(n_path).convert("RGB"), dtype=np.int16)
            ia = np.array(Image.open(i_path).convert("RGB"), dtype=np.int16)
            d = np.abs(na - ia)
            _dbg(
                "E",
                "visibility_experiment.py:_ensure_pair",
                "png pixel diff",
                {
                    "normal": trial.normal_name,
                    "inv": trial.inv_name,
                    "max_abs": int(d.max()),
                    "mean_abs": float(d.mean()),
                    "normal_tex": int(self.tex_cache[trial.normal_name]),
                    "inv_tex": int(self.tex_cache[trial.inv_name]),
                },
            )
        except Exception as e:
            _dbg(
                "E",
                "visibility_experiment.py:_ensure_pair",
                "png diff failed",
                {"error": str(e)},
            )
        # #endregion
        return self.tex_cache[trial.normal_name], self.tex_cache[trial.inv_name]

    def _set_hud(self, **kwargs) -> None:
        key = tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))
        # body_lines is a list — make hashable
        key2 = []
        for k, v in key:
            if isinstance(v, list):
                key2.append((k, tuple(v)))
            else:
                key2.append((k, v))
        key_t = tuple(key2)
        if key_t == self.hud_key and self.hud_tex is not None:
            return
        arr = make_hud(self.width, self.height, **kwargs)
        if self.hud_tex is not None:
            glDeleteTextures([self.hud_tex])
        self.hud_tex = load_texture_from_rgba(arr)
        self.hud_key = key_t

    def _clear_hud(self) -> None:
        if self.hud_tex is not None:
            glDeleteTextures([self.hud_tex])
            self.hud_tex = None
            self.hud_key = None

    def _frame(self, base_tex: int, *, with_hud: bool = True) -> None:
        """UI frames (start / rating / end). May blend HUD."""
        draw_fullscreen(base_tex)
        if with_hud and self.hud_tex is not None:
            draw_overlay(self.hud_tex)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        if self._check_abort_click():
            self.abort_requested = True

    def _present_texture(self, tex: int) -> None:
        """present_session-style: one texture, no HUD, one swap = one display frame."""
        draw_fullscreen(tex)
        # #region agent log
        if getattr(self, "_dbg_bind_left", 0) > 0:
            self._dbg_bind_left -= 1
            from OpenGL.GL import glGetError

            err = int(glGetError())
            _dbg(
                "D",
                "visibility_experiment.py:_present_texture",
                "after draw",
                {"tex": int(tex), "glError": err},
            )
        # #endregion
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _present_frames(
        self,
        tex_or_pair: int | tuple[int, int],
        n_frames: int,
        *,
        measure_fps: bool = False,
        hold: int = 1,
    ) -> bool:
        """Show tex for n_frames. Pair: alternate with present_session repeat=(hold).

        flip = (f // hold) & 1 so 180 Hz + hold=3 => 30 Hz flicker.
        """
        hold = max(1, int(hold))
        measure_n = min(60, n_frames) if measure_fps and not self._fps_logged else 0
        t0 = time.perf_counter() if measure_n else time.perf_counter()
        # #region agent log
        first_choices: list[dict] = []
        is_pair = isinstance(tex_or_pair, tuple)
        if is_pair:
            n_id, i_id = int(tex_or_pair[0]), int(tex_or_pair[1])
            _dbg(
                "A",
                "visibility_experiment.py:_present_frames",
                "pair ids",
                {
                    "normal_tex": n_id,
                    "inv_tex": i_id,
                    "same_id": n_id == i_id,
                    "n_frames": n_frames,
                    "hold": hold,
                    "flicker_hz": (self.monitor_hz / (2.0 * hold)) if hold else None,
                    "type_name": type(tex_or_pair).__name__,
                },
            )
        else:
            _dbg(
                "C",
                "visibility_experiment.py:_present_frames",
                "not a pair",
                {"tex": int(tex_or_pair), "n_frames": n_frames, "hold": hold},
            )
        self._dbg_bind_left = 4
        # #endregion
        for f in range(n_frames):
            if glfw.window_should_close(self.window) or self.abort_requested:
                return False
            if isinstance(tex_or_pair, tuple):
                normal_tex, inv_tex = tex_or_pair
                tex = inv_tex if ((f // hold) & 1) else normal_tex
            else:
                tex = tex_or_pair
            # #region agent log
            if f < 12:
                use_inv = is_pair and ((f // hold) & 1)
                first_choices.append(
                    {
                        "f": f,
                        "tex": int(tex),
                        "branch": (
                            "inv" if use_inv else ("normal" if is_pair else "single")
                        ),
                    }
                )
            # #endregion
            self._present_texture(tex)
            if measure_n and f + 1 == measure_n:
                elapsed = time.perf_counter() - t0
                if elapsed > 0:
                    hz = measure_n / elapsed
                    print(f"[INFO] present ~{hz:.1f} Hz (target {TARGET_HZ})")
                    # #region agent log
                    _dbg(
                        "B",
                        "visibility_experiment.py:_present_frames",
                        "measured present hz",
                        {
                            "hz": hz,
                            "elapsed_s": elapsed,
                            "measure_n": measure_n,
                            "monitor_hz": self.monitor_hz,
                            "swap_interval": 1,
                            "hold_frames": self.hold_frames,
                            "present_hz": self.monitor_hz,
                            "flicker_hz": self.flicker_hz,
                        },
                    )
                    # #endregion
                    if abs(hz - TARGET_HZ) > 3.0:
                        print(
                            f"[WARN] Measured present rate far from {TARGET_HZ} Hz. "
                            "Check OS refresh / exclusive fullscreen / vsync."
                        )
                    self._fps_logged = True
        # #region agent log
        _dbg(
            "C",
            "visibility_experiment.py:_present_frames",
            "stim finished",
            {
                "first_choices": first_choices,
                "total_frames": n_frames,
                "wall_s": time.perf_counter() - t0,
                "unique_tex_in_first8": sorted({c["tex"] for c in first_choices}),
            },
        )
        # #endregion
        return True

    # ---- lifecycle ----

    def init_gl(self) -> None:
        if not glfw.init():
            raise SystemExit("GLFW initialization failed")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.width = int(mode.size.width)
        self.height = int(mode.size.height)
        self.monitor_hz = float(mode.refresh_rate) if mode.refresh_rate else float(TARGET_HZ)
        # Always vsync 1:1 with the panel. 30 Hz flicker via hold (present_session --repeat),
        # because glfw.swap_interval(n>1) is often ignored on Windows.
        self.swap_interval_n = 1
        self.hold_frames = max(1, int(round(self.monitor_hz / TARGET_HZ)))
        self.present_hz = self.monitor_hz
        self.flicker_hz = self.monitor_hz / (2.0 * self.hold_frames)
        print(f"[INFO] monitor {self.width}x{self.height} @ {self.monitor_hz:.1f} Hz")
        print(
            f"[INFO] swap_interval=1 hold={self.hold_frames} => "
            f"flicker ~{self.flicker_hz:.1f} Hz (target 30 Hz)"
        )
        if abs(self.monitor_hz - TARGET_HZ) > 1.5:
            print(
                f"[WARN] OS refresh is {self.monitor_hz:.0f} Hz, not {TARGET_HZ}. "
                f"Holding each image {self.hold_frames} frames (present_session --repeat) "
                "so flicker stays ~30 Hz."
            )

        # Match present_session.c: video-mode hints before exclusive fullscreen
        glfw.window_hint(glfw.RED_BITS, int(mode.bits.red) if mode else 8)
        glfw.window_hint(glfw.GREEN_BITS, int(mode.bits.green) if mode else 8)
        glfw.window_hint(glfw.BLUE_BITS, int(mode.bits.blue) if mode else 8)
        glfw.window_hint(glfw.REFRESH_RATE, int(self.monitor_hz))
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        if self.window_mode:
            self.window = glfw.create_window(
                min(1280, self.width),
                min(720, self.height),
                "Visibility Experiment",
                None,
                None,
            )
            self.width = min(1280, self.width)
            self.height = min(720, self.height)
        else:
            self.window = glfw.create_window(
                self.width, self.height, "Visibility Experiment", monitor, None
            )

        if not self.window:
            glfw.terminate()
            raise SystemExit("Window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse)
        if not self.window_mode:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        glClearColor(0, 0, 0, 1)
        setup_ortho()

        self.tex_gray = solid_texture(ISI_GRAY)
        self.tex_dark = solid_texture(BG_DARK)
        print(
            f"[INFO] present frames: ISI={frames_for_seconds(ISI_SEC, self.monitor_hz)} "
            f"stim={frames_for_seconds(STIM_SEC, self.monitor_hz)} "
            f"(@ {self.monitor_hz:.1f} Hz vsync, hold={self.hold_frames})"
        )
        # #region agent log
        print(f"[DEBUG] log -> {_DEBUG_LOG}")
        _dbg(
            "B",
            "visibility_experiment.py:init_gl",
            "gl init",
            {
                "width": self.width,
                "height": self.height,
                "monitor_hz": self.monitor_hz,
                "window_mode": self.window_mode,
                "swap_interval": 1,
                "hold_frames": self.hold_frames,
                "flicker_hz": self.flicker_hz,
                "present_hz": self.monitor_hz,
            },
        )
        # #endregion

    def shutdown(self) -> None:
        for tex in list(self.tex_cache.values()):
            try:
                glDeleteTextures([tex])
            except Exception:
                pass
        for tex in (self.tex_gray, self.tex_dark, self.hud_tex):
            if tex:
                try:
                    glDeleteTextures([tex])
                except Exception:
                    pass
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        _time_end()

    def run(self) -> None:
        check_assets(self.asset_dir)
        self.init_gl()
        try:
            self.session, self.main_trials, self.session_path, self.csv_path = (
                create_session(
                    self.participant_id,
                    self.participant_name,
                    self.monitor_hz,
                )
            )
            self.session["monitor_hz"] = self.monitor_hz
            self.session["swap_interval"] = 1
            self.session["hold_frames"] = self.hold_frames
            self.session["flicker_hz"] = self.flicker_hz
            self.session["present_hz"] = self.monitor_hz
            save_session(self.session_path, self.session)

            if not self._show_start():
                self._abort_session()
                return

            ok = self._run_trial(practice_trial(), progress="練習 1/1")
            if not ok:
                self._abort_session()
                return
            self.session["practice_done"] = True
            save_session(self.session_path, self.session)

            if not self._show_main_ready():
                self._abort_session()
                return

            for trial in self.main_trials:
                progress = f"{trial.trial_index}/48"
                ok = self._run_trial(trial, progress=progress)
                if not ok:
                    self._abort_session()
                    return

            self.session["completed"] = True
            self.session["aborted"] = False
            self.session["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_session(self.session_path, self.session)
            self._show_end()
        finally:
            self.shutdown()

    def _abort_session(self) -> None:
        self.session["aborted"] = True
        self.session["aborted_at"] = datetime.now(timezone.utc).isoformat()
        save_session(self.session_path, self.session)
        print(
            f"[INFO] Aborted. This run was saved under {self.csv_path.parent} "
            "(next launch starts a new run from the beginning)."
        )

    # ---- screens ----

    def _show_start(self) -> bool:
        who = f"観察者: {self.participant_name}（{self.participant_id}）"
        body = [
            who,
            "",
            "これから画像が点滅します。",
            "「ちらつき」の強さだけを、1〜4 で答えてください。",
            "",
            "1. 全く分からない",
            "2. よく見ると、わずかにちらつく",
            "3. ちらつきが分かる",
            "4. QRコードのようなものが見える",
            "",
            "最初に練習が 1 回あり、そのあと本番（48試行）に入ります。",
            "各試行は灰色 → 刺激3秒 → 画面が切り替わってから 1〜4 で回答。",
            "視聴距離は実験者の指示どおりに固定してください。",
            "右上「中断」または Esc で中断できます。",
            "中断した場合も、次の起動では最初から取り直します。",
        ]
        self.enter_pressed = False
        self.abort_requested = False
        self._set_hud(
            progress="",
            show_abort=True,
            title="ちらつき評定実験",
            body_lines=body,
            footer="Enter で開始",
        )
        while not glfw.window_should_close(self.window):
            self._frame(self.tex_dark)
            if self.abort_requested:
                return False
            if self.enter_pressed:
                return True
        return False

    def _show_main_ready(self) -> bool:
        title = "本番を始めます"
        body = [
            "練習は終わりです。ここからが本番です（48試行）。",
            "",
            "やり方は練習と同じです。",
            "灰色 → 刺激3秒 → 画面が切り替わってから 1〜4 で回答。",
        ]
        footer = "Enter で本番開始"
        self.enter_pressed = False
        self.abort_requested = False
        self.rating_key = None
        self._set_hud(
            progress="",
            show_abort=True,
            title=title,
            body_lines=body,
            footer=footer,
        )
        while not glfw.window_should_close(self.window):
            self._frame(self.tex_dark)
            if self.abort_requested:
                return False
            if self.enter_pressed:
                return True
        return False

    def _show_end(self) -> None:
        self.enter_pressed = False
        self.abort_requested = False
        self._set_hud(
            progress="完了",
            show_abort=False,
            title="ご協力ありがとうございました",
            body_lines=[
                f"結果は {self.csv_path} に保存しました。",
                "Enter または Esc で終了します。",
            ],
            footer="",
        )
        while not glfw.window_should_close(self.window):
            self._frame(self.tex_dark)
            if self.enter_pressed or self.abort_requested:
                return

    def _run_trial(self, trial: Trial, progress: str) -> bool:
        """Return False if aborted."""
        normal_tex, inv_tex = self._ensure_pair(trial)
        self.rating_key = None
        self.abort_requested = False
        self.mouse_clicked = False

        isi_frames = frames_for_seconds(ISI_SEC, self.monitor_hz)
        stim_frames = frames_for_seconds(STIM_SEC, self.monitor_hz)
        hold = self.hold_frames
        if trial.is_practice:
            # ~8 Hz so the two frames are obviously different (practice only)
            hold = max(self.hold_frames, max(1, int(round(self.monitor_hz / 16.0))))

        # --- ISI gray (frame-counted; progress HUD only, no flicker) ---
        self._set_hud(
            progress=progress,
            show_abort=True,
            title="",
            body_lines=None,
            footer="",
        )
        t_isi = time.perf_counter()
        for _ in range(isi_frames):
            if glfw.window_should_close(self.window) or self.abort_requested:
                return False
            self._frame(self.tex_gray)

        # --- stimulus (frame-counted normal/inv, no HUD — present_session style) ---
        self._clear_hud()
        t_stim = time.perf_counter()
        self.rating_key = None
        if not self._present_frames(
            (normal_tex, inv_tex),
            stim_frames,
            measure_fps=True,
            hold=hold,
        ):
            return False

        # --- rating: stop stimulus, text-only like the start screen ---
        rating_body = [
            "ちらつきの強さはどれでしたか。",
            "1〜4 のキーで答えてください。",
            "",
            "1. 全く分からない",
            "2. よく見ると、わずかにちらつく",
            "3. ちらつきが分かる",
            "4. QRコードのようなものが見える",
        ]
        self._set_hud(
            progress=progress,
            show_abort=True,
            title="ちらつきの強さ",
            body_lines=rating_body,
            footer="1〜4 のキーで回答",
            params=f"{trial.image}  {trial.channel}  {trial.intensity}",
        )
        self.rating_key = None
        t_rate_start = time.perf_counter()
        while True:
            if glfw.window_should_close(self.window) or self.abort_requested:
                return False
            self._frame(self.tex_dark)
            if self.rating_key in (1, 2, 3, 4):
                break

        t_response = time.perf_counter()
        rating = int(self.rating_key)
        rt_ms = int(round((t_response - t_rate_start) * 1000))

        append_rating(
            self.csv_path,
            {
                "participant_id": self.participant_id,
                "participant_name": self.participant_name,
                "trial_index": trial.trial_index,
                "is_practice": int(trial.is_practice),
                "image": trial.image,
                "channel": trial.channel,
                "intensity": trial.intensity,
                "rating": rating,
                "rt_ms": rt_ms,
                "t_isi": f"{t_isi:.6f}",
                "t_stim": f"{t_stim:.6f}",
                "t_response": f"{t_response:.6f}",
                "aborted": 0,
            },
        )
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visibility (flicker) rating experiment @ 60 Hz")
    p.add_argument("--id", type=str, default="", help="participant id (e.g. P01). omitted = auto")
    p.add_argument("--name", type=str, default="", help="observer name (registered on first use)")
    p.add_argument(
        "--asset-dir",
        type=str,
        default=str(SCRIPT_DIR),
        help="directory with *_normalR.png / *_invR.png assets",
    )
    p.add_argument(
        "--window",
        action="store_true",
        help="run in windowed mode (debug; default is fullscreen)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    participant_id, participant_name = prompt_identity(args)
    print(f"[INFO] 観察者: {participant_name} ({participant_id})")

    asset_dir = Path(args.asset_dir)
    if not asset_dir.is_absolute():
        asset_dir = (SCRIPT_DIR / asset_dir).resolve()

    app = ExperimentApp(
        participant_id=participant_id,
        participant_name=participant_name,
        asset_dir=asset_dir,
        window_mode=bool(args.window),
    )
    app.run()


if __name__ == "__main__":
    main()
