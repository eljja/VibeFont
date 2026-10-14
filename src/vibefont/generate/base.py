"""Stroke generator contract. Everything that produces handwriting implements this."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..document.drift import LineState
from ..ink import Ink
from ..vibe.spec import StyleSpec


@dataclass
class LineRequest:
    text: str
    style: StyleSpec
    state: LineState
    rng: np.random.Generator
    # Previously generated lines of THIS document (normalised units), most recent last.
    # Model generators use them for priming so that line N continues the hand of line N-1.
    context: list[Ink] = field(default_factory=list)


class StrokeGenerator(Protocol):
    name: str

    def generate_line(self, req: LineRequest) -> Ink:
        """Return one line in NORMALISED units: x-height 1, baseline y=0, starts near x=0, y down.

        Must include natural variation itself (word- and point-level). Must NOT apply
        document-level transforms (size in mm, tilt, margins): the pipeline does that.
        Slant: should already approximate ``req.state.slant_deg``; the pipeline only applies
        a small residual correction.
        """
        ...

    def estimate_width(self, text: str, style: StyleSpec) -> float:
        """Rough line width in x-heights, used for line breaking before generation."""
        ...
