"""Page geometry and line placement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..ink import Ink
from ..vibe.spec import VibeSpec

PAGE_SIZES_MM = {"A4": (210.0, 297.0), "A5": (148.0, 210.0), "Letter": (215.9, 279.4), "B5": (176.0, 250.0)}


@dataclass
class Page:
    width_mm: float
    height_mm: float
    lines: list[Ink] = field(default_factory=list)  # in page millimetres


def page_size(spec: VibeSpec) -> tuple[float, float]:
    return PAGE_SIZES_MM.get(spec.layout.page, PAGE_SIZES_MM["A4"])


def rule_positions(spec: VibeSpec) -> np.ndarray:
    """y (mm) of printed rules for ruled/grid paper. Shared by layout and paper renderer."""
    _, h = page_size(spec)
    top = spec.layout.margin_mm[0]
    return np.arange(top, h - spec.layout.margin_mm[2] + 1e-6, spec.paper.line_spacing_mm)


def line_pitch_mm(spec: VibeSpec) -> float:
    need = max(spec.layout.line_height_mm, spec.style.size_mm * 3.6)
    if spec.paper.type in ("ruled", "grid"):
        sp = spec.paper.line_spacing_mm
        return sp * max(1, math.ceil(need / sp - 0.15))
    return need


def baselines_mm(spec: VibeSpec) -> np.ndarray:
    """Ideal baseline positions for one page."""
    _, h = page_size(spec)
    top, _, bottom, _ = spec.layout.margin_mm
    pitch = line_pitch_mm(spec)
    if spec.paper.type in ("ruled", "grid"):
        rules = rule_positions(spec)
        step = max(1, int(round(pitch / spec.paper.line_spacing_mm)))
        # write ON the rule; skip the first rule so ascenders fit
        return rules[step::step]
    first = top + spec.style.size_mm * 2.0
    return np.arange(first, h - bottom + 1e-6, pitch)
