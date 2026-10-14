"""Document-level memory: how the writing slowly changes from the first line to the last.

Why: a human writes line 20 *like* line 1 (same hand) but never identically. We model
this with mean-reverting Ornstein-Uhlenbeck processes on a few interpretable quantities,
plus an optional fatigue trend. Lines that are close in the document are correlated;
lines far apart are only tied together by the shared style (the OU mean).

Variation hierarchy (each level is handled by exactly one component):
    document : VibeSpec.style (+ style latent once the model exists)   -> spec / interpreter
    line     : LineState from DocumentDrift (this file)                 -> pipeline
    word     : per-word jitter                                          -> generator
    point    : sampling noise                                           -> generator
    ink      : pen/paper noise                                          -> renderer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vibe.spec import StyleSpec


@dataclass
class LineState:
    index: int
    progress: float  # 0 at first line, -> 1 near the end of the document
    slant_deg: float  # absolute target slant for this line
    size_scale: float  # multiplies style.size_mm
    tilt_deg: float  # baseline rotation of the whole line (+ = rising to the right)
    baseline_offset_mm: float  # vertical deviation from the ideal line position
    margin_offset_mm: float  # left margin deviation
    word_spacing_scale: float
    temperature: float  # sampling temperature handed to the stroke model
    neatness: float  # effective neatness after fatigue


class _OU:
    def __init__(self, mean: float, std: float, theta: float, rng: np.random.Generator):
        self.mean, self.theta = mean, theta
        self.sigma = std * np.sqrt(1.0 - (1.0 - theta) ** 2)  # keeps stationary std == std
        self.rng = rng
        self.x = mean + std * rng.standard_normal()

    def step(self) -> float:
        self.x += self.theta * (self.mean - self.x) + self.sigma * self.rng.standard_normal()
        return self.x


class DocumentDrift:
    def __init__(self, style: StyleSpec, rng: np.random.Generator, expected_lines: int = 25):
        self.style = style
        self.rng = rng
        self.expected_lines = max(expected_lines, 1)
        self.k = 0
        loose = 1.6 - 1.2 * style.consistency  # 0.4 (steady) .. 1.6 (drifty)
        self._slant = _OU(0.0, 2.5 * loose, 0.30, rng)
        self._size = _OU(0.0, 0.05 * loose, 0.30, rng)
        self._tilt = _OU(0.0, 0.7 * loose * (0.5 + style.baseline_wobble), 0.50, rng)
        self._base = _OU(0.0, 0.5 * loose, 0.60, rng)
        self._margin = _OU(0.0, 1.5 * loose, 0.40, rng)
        self._wsp = _OU(0.0, 0.08 * loose, 0.30, rng)

    def next_line(self) -> LineState:
        s = self.style
        prog = min(1.0, self.k / self.expected_lines)
        fat = s.fatigue * prog
        neat = float(np.clip(s.neatness - 0.35 * fat, 0.0, 1.0))
        st = LineState(
            index=self.k,
            progress=prog,
            slant_deg=s.slant_deg + self._slant.step() + 4.0 * fat,
            size_scale=float(np.exp(self._size.step())) * (1.0 + 0.10 * fat),
            tilt_deg=self._tilt.step(),
            baseline_offset_mm=self._base.step(),
            margin_offset_mm=self._margin.step(),
            word_spacing_scale=float(np.exp(self._wsp.step())) * (1.0 + 0.10 * fat),
            temperature=0.55 + 0.6 * (1.0 - neat) + 0.15 * s.speed,
            neatness=neat,
        )
        self.k += 1
        return st
