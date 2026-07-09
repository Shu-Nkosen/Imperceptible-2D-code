from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VideoNameMeta:
    rate_hz: int
    exp: int
    fluoro: int

    @property
    def stem(self) -> str:
        return f"r{self.rate_hz}_e{self.exp}_f{self.fluoro}"


_VIDEO_RE = re.compile(r"^r(?P<rate>\d+)_e(?P<exp>\d+)_f(?P<fluoro>[01])$", re.IGNORECASE)


def parse_video_name(path: str | Path) -> Optional[VideoNameMeta]:
    p = Path(path)
    m = _VIDEO_RE.match(p.stem)
    if not m:
        return None
    return VideoNameMeta(
        rate_hz=int(m.group("rate")),
        exp=int(m.group("exp")),
        fluoro=int(m.group("fluoro")),
    )

