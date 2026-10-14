"""VibeSpec: the single, interpretable description of "how this document is written".

Every prompt interpreter (rule-based, LLM, learned) must output a VibeSpec, and every
downstream stage (drift, generator, renderer, layout) only reads a VibeSpec.
All style fields are normalised to [0, 1] unless a unit is given.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

PEN_TYPES = ("ballpoint", "gel", "fountain", "pencil", "marker", "brush", "broad_nib")
PAPER_TYPES = ("plain", "ivory", "ruled", "grid", "kraft")


@dataclass
class StyleSpec:
    slant_deg: float = 6.0  # + = lean right
    size_mm: float = 3.0  # x-height on paper
    width_scale: float = 1.0  # horizontal stretch of letters
    letter_spacing: float = 1.0
    word_spacing: float = 1.0
    roundness: float = 0.5  # 0 angular .. 1 round
    connectedness: float = 0.5  # 0 print .. 1 fully cursive
    neatness: float = 0.6  # 0 scrawl .. 1 calligraphic; drives sampling temperature
    speed: float = 0.5  # 0 slow/careful .. 1 hurried
    pressure: float = 0.5
    baseline_wobble: float = 0.3
    consistency: float = 0.6  # 0 drifts a lot over the page .. 1 very steady
    fatigue: float = 0.2  # how much the writing loosens toward the end of the document


@dataclass
class PenSpec:
    type: str = "ballpoint"
    width_mm: float = 0.5
    nib_angle_deg: float = 35.0  # broad_nib / brush only


@dataclass
class InkSpec:
    color: tuple[float, float, float] = (0.10, 0.12, 0.30)  # sRGB 0..1 (dark blue-black)
    opacity: float = 1.0


@dataclass
class PaperSpec:
    type: str = "plain"
    line_spacing_mm: float = 8.0  # ruled / grid pitch


@dataclass
class LayoutSpec:
    page: str = "A4"
    margin_mm: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 20.0)  # top right bottom left
    line_height_mm: float = 9.0  # used for plain paper; ruled paper snaps to its lines


@dataclass
class VibeSpec:
    style: StyleSpec = field(default_factory=StyleSpec)
    pen: PenSpec = field(default_factory=PenSpec)
    ink: InkSpec = field(default_factory=InkSpec)
    paper: PaperSpec = field(default_factory=PaperSpec)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    seed: int | None = None
    prompt: str = ""
    matched: list[str] = field(default_factory=list)  # explainability: which rules fired

    # ------------------------------------------------------------- helpers
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VibeSpec":
        spec = cls()
        spec.update(d)
        return spec

    def update(self, d: dict[str, Any]) -> "VibeSpec":
        """Recursively set known fields from a (possibly partial) dict; unknown keys ignored."""
        for k, v in d.items():
            if not hasattr(self, k):
                continue
            cur = getattr(self, k)
            if dataclasses.is_dataclass(cur) and isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(cur, kk):
                        old = getattr(cur, kk)
                        setattr(cur, kk, tuple(vv) if isinstance(old, tuple) and isinstance(vv, list) else vv)
            else:
                setattr(self, k, v)
        self.clamp()
        return self

    def clamp(self) -> None:
        s = self.style
        for name in ("roundness", "connectedness", "neatness", "speed", "pressure",
                     "baseline_wobble", "consistency", "fatigue"):
            setattr(s, name, float(min(1.0, max(0.0, getattr(s, name)))))
        s.slant_deg = float(min(40.0, max(-25.0, s.slant_deg)))
        s.size_mm = float(min(15.0, max(1.2, s.size_mm)))
        s.width_scale = float(min(1.8, max(0.6, s.width_scale)))
        s.letter_spacing = float(min(2.0, max(0.6, s.letter_spacing)))
        s.word_spacing = float(min(3.0, max(0.5, s.word_spacing)))
        if self.pen.type not in PEN_TYPES:
            self.pen.type = "ballpoint"
        if self.paper.type not in PAPER_TYPES:
            self.paper.type = "plain"
        self.pen.width_mm = float(min(5.0, max(0.1, self.pen.width_mm)))
