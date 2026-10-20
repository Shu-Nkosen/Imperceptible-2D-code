"""フレーム発見・読込。out_mid_fast_0805 へは書かない。"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from naming import VideoNameMeta, parse_video_name

FOLDER_RE = re.compile(
    r"^(?P<image>.+)_(?P<token>R|G|B|I|X)_(?P<intensity>\d+)$",
    re.IGNORECASE,
)
TOKEN_TO_CHANNEL = {"R": "R", "G": "G", "B": "B", "I": "min", "X": "max"}


@dataclass(frozen=True)
class ConditionRef:
    stem: str
    folder: str
    rate_hz: int
    exp: int
    fluoro: int
    image: str
    token: str
    channel: str
    intensity: int
    frame_dir: Path
    camera_fps: float


def parse_stem(stem: str) -> Optional[VideoNameMeta]:
    return parse_video_name(stem)


def parse_folder(folder: str) -> Optional[Tuple[str, str, str, int]]:
    m = FOLDER_RE.match(folder)
    if not m:
        return None
    token = m.group("token").upper()
    return (
        m.group("image"),
        token,
        TOKEN_TO_CHANNEL[token],
        int(m.group("intensity")),
    )


def read_camera_fps_from_sweeps(stem_dir: Path, default: float = 60.0) -> float:
    for name in (
        "results_sweeps_lockin.csv",
        "results_sweeps_fourier.csv",
        "results_sweeps_pair.csv",
    ):
        path = stem_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = (row.get("camera_fps") or "").strip()
                if not raw:
                    continue
                # "60.009 fps" or "60.009"
                num = raw.replace("fps", "").strip().split()[0]
                try:
                    return float(num)
                except ValueError:
                    continue
    return float(default)


def list_frame_paths(frame_dir: Path) -> List[Path]:
    return sorted(frame_dir.glob("frame_*.png"))


def load_rgb_frames(frame_dir: Path, count: Optional[int] = None) -> List[np.ndarray]:
    paths = list_frame_paths(frame_dir)
    if count is not None:
        paths = paths[: int(count)]
    out: List[np.ndarray] = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(rgb)
    return out


def discover_conditions(
    in_root: Path,
    *,
    rates: Sequence[int] = (45, 60, 90),
    intensities: Sequence[int] = (12,),
    tokens: Sequence[str] = ("R", "G", "B"),
    limit_stems: Optional[int] = None,
    limit_folders: Optional[int] = None,
    min_frames: int = 120,
) -> List[ConditionRef]:
    """in_root/{stem}/{folder}/frame_*.png を探す。"""
    rates_set = set(int(r) for r in rates)
    inten_set = set(int(i) for i in intensities)
    token_set = {t.upper() for t in tokens}
    found: List[ConditionRef] = []

    stems = sorted(
        p for p in in_root.iterdir() if p.is_dir() and parse_stem(p.name) is not None
    )
    if limit_stems is not None:
        stems = stems[: int(limit_stems)]

    for stem_dir in stems:
        meta = parse_stem(stem_dir.name)
        assert meta is not None
        if meta.rate_hz not in rates_set:
            continue
        fps = read_camera_fps_from_sweeps(stem_dir)
        folders = sorted(p for p in stem_dir.iterdir() if p.is_dir())
        n_added = 0
        for folder_dir in folders:
            parsed = parse_folder(folder_dir.name)
            if parsed is None:
                continue
            image, token, channel, intensity = parsed
            if token not in token_set or intensity not in inten_set:
                continue
            n_frames = len(list_frame_paths(folder_dir))
            if n_frames < min_frames:
                continue
            found.append(
                ConditionRef(
                    stem=stem_dir.name,
                    folder=folder_dir.name,
                    rate_hz=meta.rate_hz,
                    exp=meta.exp,
                    fluoro=meta.fluoro,
                    image=image,
                    token=token,
                    channel=channel,
                    intensity=intensity,
                    frame_dir=folder_dir,
                    camera_fps=fps,
                )
            )
            n_added += 1
            if limit_folders is not None and n_added >= int(limit_folders):
                break
    return found


def count_frame_dirs(in_root: Path) -> Tuple[int, int]:
    """(stem数, frame_*.png を持つ folder数)。"""
    stems = [p for p in in_root.iterdir() if p.is_dir() and parse_stem(p.name)]
    folders = 0
    for stem in stems:
        for d in stem.iterdir():
            if d.is_dir() and list_frame_paths(d):
                folders += 1
    return len(stems), folders
