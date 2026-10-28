"""Organic procedural Hangul handwriting synthesizer.

Decomposes any Hangul syllable into Choseong (initial), Jungseong (medial),
and Jongseong (final), and composes them into natural, curved human handwriting
stroke paths in normalized units (x-height == 1, baseline y == 0, y down).

Key design features:
  1. Proportion balance with Latin: Hangul syllables match English cap/ascender height
     (y in [-1.72, +0.18], width ~1.42 x-heights).
  2. Smooth spline trajectories: every stroke is a multi-point organic curve with
     natural entry/exit hooks, slight upward horizontal tilt (+4 deg), and rounded corners.
  3. Sub-syllable stochasticity: each Jamo receives independent micro-variations in
     position, scale, and slant when an RNG is provided.
  4. Cursive ligatures: consecutive strokes softly link when connectedness is high.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vibe.spec import StyleSpec
from .ligature import JoinInfo, join_strokes, turn_angle_deg, polylines_intersect

CHOSEONG_NAMES = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNGSEONG_NAMES = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JUNGSEONG_VERTICAL = {0, 1, 2, 3, 4, 5, 6, 7, 20}  # ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅣ
JUNGSEONG_HORIZONTAL = {8, 12, 13, 17, 18}         # ㅗ ㅛ ㅜ ㅠ ㅡ
JUNGSEONG_COMPLEX = {9, 10, 11, 14, 15, 16, 19}    # ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ

WRITER_ARCHETYPES = (
    "standard",
    "tall_slender",
    "wide_round",
    "bold_head",
    "upward_flow",
    "compact_loose",
    "gosi",
    "classic_jeongja",
    "rounded_diary",
)


@dataclass
class WriterProfile:
    """Encodes document-level personal handwriting archetype & structural habits."""

    archetype: str = "standard"
    w_scale: float = 1.0          # Overall syllable width ratio (tall_slender < 1 < wide_round)
    h_scale: float = 1.0          # Overall syllable height ratio
    cho_scale: float = 1.0        # Initial consonant (초성) relative size
    jong_scale: float = 1.0       # Final consonant (받침) relative size
    horiz_tilt: float = 0.04      # Personal upward/downward tilt of horizontal strokes
    stem_bend: float = 0.0        # Personal bow of vertical stems (-: concave left, +: convex right)
    loop_roundness: float = 1.0   # Aspect inflation of loops (ㅇ, ㅁ, ㅎ)
    allograph_bias: float = 0.5   # Preference for cursive/simplified vs. multi-stroke allographs


def sample_writer_profile(
    rng: np.random.Generator | None = None,
    style: StyleSpec | None = None,
) -> WriterProfile:
    """Samples a distinct personal handwriting archetype and continuous structural traits."""
    if rng is None:
        return WriterProfile()

    if style is not None and getattr(style, "archetype", "standard") in WRITER_ARCHETYPES and style.archetype != "standard":
        arch = style.archetype
    else:
        arch = str(rng.choice(WRITER_ARCHETYPES))
    conn = style.connectedness if style is not None else 0.5
    rnd = style.roundness if style is not None else 0.5

    if arch == "gosi":
        w_sc = float(rng.uniform(0.92, 1.04))
        h_sc = float(rng.uniform(0.96, 1.06))
        cho_sc = float(rng.uniform(0.94, 1.06))
        jong_sc = float(rng.uniform(0.90, 1.04))
        h_tilt = float(rng.uniform(0.08, 0.16))
        s_bend = float(rng.uniform(0.01, 0.06))
        loop_r = float(rng.uniform(0.82, 0.96))
        allo_bias = 0.95
    elif arch == "classic_jeongja":
        w_sc = float(rng.uniform(0.96, 1.02))
        h_sc = float(rng.uniform(0.98, 1.03))
        cho_sc = float(rng.uniform(0.98, 1.04))
        jong_sc = float(rng.uniform(0.96, 1.04))
        h_tilt = 0.0
        s_bend = 0.0
        loop_r = 1.0
        allo_bias = 0.05
    elif arch == "rounded_diary":
        w_sc = float(rng.uniform(1.12, 1.25))
        h_sc = float(rng.uniform(0.90, 0.98))
        cho_sc = float(rng.uniform(1.10, 1.22))
        jong_sc = float(rng.uniform(1.05, 1.18))
        h_tilt = float(rng.uniform(-0.02, 0.04))
        s_bend = float(rng.uniform(-0.02, 0.04))
        loop_r = float(rng.uniform(1.28, 1.48))
        allo_bias = 0.40
    elif arch == "tall_slender":
        w_sc = float(rng.uniform(0.75, 0.85))
        h_sc = float(rng.uniform(1.04, 1.12))
        cho_sc = float(rng.uniform(0.92, 1.04))
        jong_sc = float(rng.uniform(0.86, 0.96))
        h_tilt = float(rng.uniform(0.03, 0.10))
        s_bend = float(rng.uniform(-0.04, 0.04))
        loop_r = float(rng.uniform(0.82, 0.96))
    elif arch == "wide_round":
        w_sc = float(rng.uniform(1.14, 1.26))
        h_sc = float(rng.uniform(0.93, 0.99))
        cho_sc = float(rng.uniform(1.02, 1.12))
        jong_sc = float(rng.uniform(1.02, 1.14))
        h_tilt = float(rng.uniform(-0.02, 0.05))
        s_bend = float(rng.uniform(0.01, 0.06))
        loop_r = float(rng.uniform(1.10, 1.28))
    elif arch == "bold_head":
        w_sc = float(rng.uniform(0.95, 1.08))
        h_sc = float(rng.uniform(0.96, 1.08))
        cho_sc = float(rng.uniform(1.12, 1.24))
        jong_sc = float(rng.uniform(0.76, 0.88))
        h_tilt = float(rng.uniform(0.05, 0.13))
        s_bend = float(rng.uniform(-0.03, 0.05))
        loop_r = float(rng.uniform(0.95, 1.15))
    elif arch == "upward_flow":
        w_sc = float(rng.uniform(0.92, 1.06))
        h_sc = float(rng.uniform(0.95, 1.08))
        cho_sc = float(rng.uniform(0.96, 1.10))
        jong_sc = float(rng.uniform(0.85, 1.02))
        h_tilt = float(rng.uniform(0.09, 0.17))
        s_bend = float(rng.uniform(0.02, 0.08))
        loop_r = float(rng.uniform(0.90, 1.12))
    elif arch == "compact_loose":
        w_sc = float(rng.uniform(0.88, 1.02))
        h_sc = float(rng.uniform(0.88, 0.98))
        cho_sc = float(rng.uniform(0.84, 0.94))
        jong_sc = float(rng.uniform(0.82, 0.94))
        h_tilt = float(rng.uniform(-0.03, 0.08))
        s_bend = float(rng.uniform(-0.05, 0.05))
        loop_r = float(rng.uniform(0.88, 1.08))
    else:  # standard
        w_sc = float(rng.uniform(0.94, 1.06))
        h_sc = float(rng.uniform(0.95, 1.05))
        cho_sc = float(rng.uniform(0.95, 1.06))
        jong_sc = float(rng.uniform(0.92, 1.05))
        h_tilt = float(rng.uniform(0.01, 0.08))
        s_bend = float(rng.uniform(-0.03, 0.03))
        loop_r = float(rng.uniform(0.94, 1.08))

    # Blend with explicit style roundness / connectedness
    loop_r *= 0.88 + 0.24 * rnd
    allo_bias = float(np.clip(0.5 * conn + float(rng.uniform(0.0, 0.5)), 0.0, 1.0))
    return WriterProfile(
        archetype=arch,
        w_scale=w_sc,
        h_scale=h_sc,
        cho_scale=cho_sc,
        jong_scale=jong_sc,
        horiz_tilt=h_tilt,
        stem_bend=s_bend,
        loop_roundness=loop_r,
        allograph_bias=allo_bias,
    )


def _curve(
    ctrl: list[list[float]] | np.ndarray,
    n: int = 12,
    rng: np.random.Generator | None = None,
    jitter: float = 0.038,
) -> np.ndarray:
    """Evaluates a smooth B-spline / quadratic Bezier chain with optional control-point elasticity."""
    pts = np.asarray(ctrl, dtype=np.float64).copy()
    m = len(pts)
    if rng is not None and jitter > 0.0:
        pts += rng.normal(0.0, jitter, size=pts.shape)

    if m == 2:
        mid = 0.5 * (pts[0] + pts[1])
        d = pts[1] - pts[0]
        bow = 0.035 if rng is None else float(rng.normal(0.035, 0.045))
        perp = np.array([-d[1], d[0]]) * bow
        pts = np.array([pts[0], mid + perp, pts[1]])
        m = 3

    if m == 3:
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t ** 2 * pts[2]

    segments = []
    steps_per_seg = max(4, n // (m - 2))
    for i in range(m - 2):
        p0 = pts[i] if i == 0 else 0.5 * (pts[i] + pts[i + 1])
        p1 = pts[i + 1]
        p2 = pts[i + 2] if i == m - 3 else 0.5 * (pts[i + 1] + pts[i + 2])
        t = np.linspace(0.0, 1.0, steps_per_seg, endpoint=(i == m - 3))[:, None]
        seg = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pts[1] + t ** 2 * p2 if False else (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        segments.append(seg)
    return np.concatenate(segments, axis=0)


def _ellipse_strokes(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    n: int = 18,
    rng: np.random.Generator | None = None,
    profile: WriterProfile | None = None,
) -> np.ndarray:
    """Natural handwriting loop with varied start angle, aspect ratio, and clean closure."""
    start_ang = -0.50 * np.pi
    span_ang = 2.04 * np.pi
    skew = 0.02
    if profile is not None:
        lr = float(np.clip(profile.loop_roundness, 0.70, 1.65))
        rx *= lr
        ry *= min(lr, 1.40)
    if rng is not None:
        start_ang += float(rng.uniform(-0.22 * np.pi, 0.18 * np.pi))
        span_ang = float(rng.uniform(1.95 * np.pi, 2.12 * np.pi))
        rx *= float(np.clip(np.exp(rng.normal(0.0, 0.055)), 0.88, 1.14))
        ry *= float(np.clip(np.exp(rng.normal(0.0, 0.055)), 0.88, 1.14))
        cx += float(np.clip(rng.normal(0.0, 0.020), -0.04, 0.04))
        cy += float(np.clip(rng.normal(0.0, 0.020), -0.04, 0.04))
        skew = float(np.clip(rng.normal(0.02, 0.025), -0.03, 0.05))
    th = np.linspace(start_ang, start_ang + span_ang, n)
    x = cx + rx * np.cos(th) + skew * rx * np.sin(2 * th)
    y = cy + ry * np.sin(th) + 0.5 * skew * ry * np.cos(2 * th)
    return np.stack([x, y], axis=1)


def choseong_strokes(
    idx: int,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
    profile: WriterProfile | None = None,
) -> list[np.ndarray]:
    """Returns list of smooth curved (M, 2) strokes in normalized [0, 1] x [0, 1] box with legible allographs."""
    ht = float(np.clip(profile.horiz_tilt if profile is not None else 0.04, -0.02, 0.09))
    arch = profile.archetype if profile is not None else "standard"
    roll = float(rng.random()) if rng is not None else 0.5
    jit = 0.018 if rng is not None else 0.0

    if idx == 0:  # ㄱ: 2 clean allographs (crisp corner vs gently arched downstroke)
        if arch == "classic_jeongja":
            return [
                _curve([[0.08, 0.14], [0.86, 0.14]], 8, rng, 0.0),
                _curve([[0.86, 0.14], [0.86, 0.90]], 10, rng, 0.0),
            ]
        if arch == "gosi":
            return [_curve([[0.06, 0.16], [0.88, 0.10 - ht], [0.72, 0.92]], 14, rng, jit)]
        if arch == "rounded_diary":
            return [_curve([[0.10, 0.18], [0.50, 0.14], [0.84, 0.20], [0.82, 0.56], [0.72, 0.88]], 15, rng, jit)]
        if roll < 0.50:
            return [_curve([[0.08, 0.16], [0.86, 0.12 - ht], [0.76, 0.90]], 14, rng, jit)]
        return [_curve([[0.06, 0.18], [0.50, 0.12 - 0.5 * ht], [0.88, 0.12 - ht], [0.82, 0.54], [0.68, 0.92]], 15, rng, jit)]
    elif idx == 1:  # ㄲ
        s1 = _curve([[0.04, 0.16], [0.44, 0.11 - ht], [0.42, 0.55], [0.35, 0.92]], 12, rng, jit)
        s2 = _curve([[0.52, 0.16], [0.94, 0.11 - ht], [0.92, 0.55], [0.83, 0.92]], 12, rng, jit)
        return [s1, s2]
    elif idx == 2:  # ㄴ: 2 clean allographs (standard L vs softly scooped bottom)
        if arch == "classic_jeongja":
            return [
                _curve([[0.16, 0.08], [0.16, 0.88]], 10, rng, 0.0),
                _curve([[0.16, 0.88], [0.92, 0.88]], 9, rng, 0.0),
            ]
        if roll < 0.55:
            return [_curve([[0.16, 0.08], [0.13, 0.56], [0.15, 0.88], [0.54, 0.90], [0.92, 0.85 - ht]], 14, rng, jit)]
        return [_curve([[0.18, 0.08], [0.12, 0.78], [0.36, 0.91], [0.92, 0.83 - ht]], 14, rng, jit)]
    elif idx == 3:  # ㄷ: always 2 strokes so it is never confused with ㄹ
        if arch == "classic_jeongja":
            return [
                _curve([[0.12, 0.14], [0.88, 0.14]], 8, rng, 0.0),
                _curve([[0.14, 0.14], [0.14, 0.88], [0.90, 0.88]], 12, rng, 0.0),
            ]
        if arch == "gosi":
            return [_curve([[0.12, 0.14], [0.86, 0.11 - ht], [0.14, 0.48], [0.12, 0.88], [0.90, 0.85 - ht]], 14, rng, jit)]
        return [
            _curve([[0.13, 0.15], [0.52, 0.11 - 0.5 * ht], [0.89, 0.09 - ht]], 10, rng, jit),
            _curve([[0.16, 0.15], [0.13, 0.56], [0.15, 0.89], [0.54, 0.90], [0.92, 0.85 - ht]], 14, rng, jit),
        ]
    elif idx == 4:  # ㄸ
        return [
            _curve([[0.05, 0.16], [0.44, 0.11 - ht]], 8, rng, jit),
            _curve([[0.08, 0.15], [0.06, 0.88], [0.45, 0.86 - ht]], 12, rng, jit),
            _curve([[0.54, 0.16], [0.94, 0.11 - ht]], 8, rng, jit),
            _curve([[0.56, 0.15], [0.54, 0.88], [0.95, 0.86 - ht]], 12, rng, jit),
        ]
    elif idx == 5:  # ㄹ: 3-stroke print vs 2-stroke natural handwriting
        if arch == "classic_jeongja":
            return [
                _curve([[0.10, 0.14], [0.86, 0.14], [0.86, 0.46]], 10, rng, 0.0),
                _curve([[0.14, 0.46], [0.86, 0.46]], 8, rng, 0.0),
                _curve([[0.14, 0.46], [0.14, 0.88], [0.88, 0.88]], 10, rng, 0.0),
            ]
        if arch == "gosi":
            return [_curve([[0.08, 0.14], [0.88, 0.10 - ht], [0.82, 0.46], [0.12, 0.50], [0.14, 0.88], [0.92, 0.84 - ht]], 16, rng, jit)]
        if connectedness < 0.35 or (connectedness < 0.78 and roll < 0.45):
            return [
                _curve([[0.08, 0.15], [0.87, 0.10 - ht], [0.85, 0.46]], 11, rng, jit),
                _curve([[0.14, 0.48], [0.50, 0.46], [0.85, 0.45 - ht]], 9, rng, jit),
                _curve([[0.14, 0.48], [0.12, 0.89], [0.54, 0.90], [0.93, 0.85 - ht]], 12, rng, jit),
            ]
        return [
            _curve([[0.08, 0.15], [0.87, 0.10 - ht], [0.85, 0.46], [0.14, 0.48]], 14, rng, jit),
            _curve([[0.14, 0.48], [0.12, 0.89], [0.54, 0.90], [0.93, 0.85 - ht]], 12, rng, jit),
        ]
    elif idx == 6:  # ㅁ: 3-stroke crisp box vs 2-stroke box (never a circle)
        if arch == "classic_jeongja":
            return [
                _curve([[0.14, 0.14], [0.14, 0.88]], 8, rng, 0.0),
                _curve([[0.14, 0.14], [0.86, 0.14], [0.86, 0.88]], 12, rng, 0.0),
                _curve([[0.14, 0.88], [0.86, 0.88]], 8, rng, 0.0),
            ]
        if arch == "gosi":
            return [_curve([[0.14, 0.18], [0.12, 0.86], [0.86, 0.86], [0.86, 0.16], [0.16, 0.16]], 14, rng, jit)]
        if arch == "rounded_diary":
            return [_ellipse_strokes(0.50, 0.50, 0.38, 0.36, 16, rng=rng, profile=profile)]
        if roll < 0.50:
            return [
                _curve([[0.14, 0.13], [0.13, 0.89]], 9, rng, jit),
                _curve([[0.14, 0.14], [0.86, 0.11 - ht], [0.85, 0.88]], 12, rng, jit),
                _curve([[0.14, 0.89], [0.86, 0.86 - ht]], 9, rng, jit),
            ]
        return [
            _curve([[0.14, 0.13], [0.13, 0.52], [0.15, 0.90]], 10, rng, jit),
            _curve([[0.14, 0.14], [0.86, 0.11 - ht], [0.84, 0.88], [0.15, 0.89]], 15, rng, jit),
        ]
    elif idx == 7:  # ㅂ: always clean 4-stroke or 3-stroke with clear vertical posts
        if arch == "classic_jeongja":
            return [
                _curve([[0.16, 0.08], [0.16, 0.90]], 9, rng, 0.0),
                _curve([[0.84, 0.08], [0.84, 0.90]], 9, rng, 0.0),
                _curve([[0.16, 0.48], [0.84, 0.48]], 7, rng, 0.0),
                _curve([[0.16, 0.89], [0.84, 0.89]], 7, rng, 0.0),
            ]
        return [
            _curve([[0.16, 0.08], [0.14, 0.52], [0.16, 0.90]], 10, rng, jit),
            _curve([[0.84, 0.08], [0.85, 0.52], [0.82, 0.90]], 10, rng, jit),
            _curve([[0.15, 0.50], [0.50, 0.48], [0.84, 0.46 - ht]], 8, rng, jit),
            _curve([[0.16, 0.89], [0.50, 0.87], [0.83, 0.85 - ht]], 8, rng, jit),
        ]
    elif idx == 8:  # ㅃ
        return [
            _curve([[0.06, 0.08], [0.06, 0.89], [0.44, 0.88], [0.44, 0.08]], 14, rng, jit),
            _curve([[0.06, 0.50], [0.44, 0.48]], 8, rng, jit),
            _curve([[0.54, 0.08], [0.54, 0.89], [0.92, 0.88], [0.92, 0.08]], 14, rng, jit),
            _curve([[0.54, 0.50], [0.92, 0.48]], 8, rng, jit),
        ]
    elif idx == 9:  # ㅅ: 2-stroke with varied branch point
        by = 0.45 + (float(rng.uniform(-0.08, 0.08)) if rng is not None else 0.0)
        return [
            _curve([[0.52, 0.08], [0.36, 0.52], [0.08, 0.90]], 10, rng, jit),
            _curve([[0.38, by], [0.66, 0.68], [0.92, 0.88]], 10, rng, jit),
        ]
    elif idx == 10:  # ㅆ
        return [
            _curve([[0.28, 0.08], [0.18, 0.52], [0.04, 0.90]], 9, rng, jit),
            _curve([[0.20, 0.45], [0.35, 0.68], [0.46, 0.88]], 9, rng, jit),
            _curve([[0.76, 0.08], [0.66, 0.52], [0.52, 0.90]], 9, rng, jit),
            _curve([[0.68, 0.45], [0.82, 0.68], [0.95, 0.88]], 9, rng, jit),
        ]
    elif idx == 11:  # ㅇ
        rx = 0.44 if arch == "rounded_diary" else 0.38
        ry = 0.44 if arch == "rounded_diary" else 0.39
        return [_ellipse_strokes(0.50, 0.50, rx, ry, 18, rng=rng, profile=profile)]
    elif idx == 12:  # ㅈ: 2-stroke standard vs 3-stroke print
        if roll > 0.65:
            return [
                _curve([[0.10, 0.16], [0.88, 0.11 - ht]], 9, rng, jit),
                _curve([[0.52, 0.14], [0.08, 0.90]], 9, rng, jit),
                _curve([[0.42, 0.48], [0.92, 0.88]], 9, rng, jit),
            ]
        return [
            _curve([[0.10, 0.16], [0.52, 0.11 - ht], [0.88, 0.11 - ht], [0.48, 0.52], [0.08, 0.90]], 14, rng, jit),
            _curve([[0.42, 0.48], [0.68, 0.70], [0.92, 0.88]], 10, rng, jit),
        ]
    elif idx == 13:  # ㅉ
        return [
            _curve([[0.04, 0.16], [0.44, 0.11 - ht], [0.24, 0.52], [0.04, 0.90]], 12, rng, jit),
            _curve([[0.22, 0.50], [0.46, 0.88]], 8, rng, jit),
            _curve([[0.54, 0.16], [0.94, 0.11 - ht], [0.74, 0.52], [0.54, 0.90]], 12, rng, jit),
            _curve([[0.72, 0.50], [0.96, 0.88]], 8, rng, jit),
        ]
    elif idx == 14:  # ㅊ: vertical top tick vs horizontal top tick + ㅈ
        if roll < 0.50:
            return [
                _curve([[0.50, 0.02], [0.50, 0.18]], 7, rng, jit),
                _curve([[0.08, 0.26], [0.52, 0.21 - ht], [0.88, 0.21 - ht], [0.48, 0.58], [0.08, 0.92]], 14, rng, jit),
                _curve([[0.42, 0.54], [0.68, 0.74], [0.92, 0.90]], 10, rng, jit),
            ]
        return [
            _curve([[0.34, 0.06], [0.52, 0.08], [0.68, 0.10]], 8, rng, jit),
            _curve([[0.08, 0.26], [0.52, 0.21 - ht], [0.88, 0.21 - ht], [0.48, 0.58], [0.08, 0.92]], 14, rng, jit),
            _curve([[0.42, 0.54], [0.68, 0.74], [0.92, 0.90]], 10, rng, jit),
        ]
    elif idx == 15:  # ㅋ
        return [
            _curve([[0.08, 0.16], [0.55, 0.11 - ht], [0.90, 0.11 - ht], [0.85, 0.55], [0.75, 0.92]], 14, rng, jit),
            _curve([[0.12, 0.52], [0.50, 0.49], [0.84, 0.46 - ht]], 9, rng, jit),
        ]
    elif idx == 16:  # ㅌ: always crisp 3-stroke
        return [
            _curve([[0.14, 0.15], [0.52, 0.11], [0.88, 0.09 - ht]], 9, rng, jit),
            _curve([[0.16, 0.50], [0.50, 0.48], [0.82, 0.46 - ht]], 9, rng, jit),
            _curve([[0.16, 0.14], [0.13, 0.56], [0.15, 0.89], [0.55, 0.90], [0.92, 0.85 - ht]], 14, rng, jit),
        ]
    elif idx == 17:  # ㅍ
        return [
            _curve([[0.08, 0.15], [0.50, 0.11], [0.92, 0.09 - ht]], 9, rng, jit),
            _curve([[0.32, 0.14], [0.30, 0.52], [0.29, 0.88]], 9, rng, jit),
            _curve([[0.68, 0.12], [0.66, 0.50], [0.65, 0.86]], 9, rng, jit),
            _curve([[0.06, 0.89], [0.50, 0.87], [0.94, 0.84 - ht]], 9, rng, jit),
        ]
    elif idx == 18:  # ㅎ: 3 clean allographs (2-stroke connected top cap + circle, or 3-stroke vertical/slanted dot + bar + circle)
        if arch == "rounded_diary":
            return [
                _curve([[0.48, 0.04], [0.52, 0.16]], 6, rng, jit),
                _curve([[0.16, 0.22], [0.50, 0.20], [0.84, 0.22]], 8, rng, jit),
                _ellipse_strokes(0.50, 0.62, 0.38, 0.34, 18, rng=rng, profile=profile),
            ]
        if arch == "gosi":
            return [
                _curve([[0.48, 0.04], [0.50, 0.18], [0.10, 0.24], [0.90, 0.18 - ht]], 12, rng, jit),
                _ellipse_strokes(0.50, 0.64, 0.34, 0.28, 16, rng=rng, profile=profile),
            ]
        if roll < 0.36:
            return [
                _curve([[0.48, 0.02], [0.48, 0.19], [0.08, 0.24], [0.92, 0.19 - ht]], 12, rng, jit),
                _ellipse_strokes(0.50, 0.64, 0.34, 0.28, 16, rng=rng, profile=profile),
            ]
        elif roll < 0.68:
            return [
                _curve([[0.50, 0.02], [0.49, 0.19]], 7, rng, jit),
                _curve([[0.08, 0.25], [0.50, 0.21], [0.92, 0.18 - ht]], 9, rng, jit),
                _ellipse_strokes(0.50, 0.64, 0.34, 0.28, 16, rng=rng, profile=profile),
            ]
        return [
            _curve([[0.34, 0.05], [0.50, 0.08], [0.66, 0.10]], 8, rng, jit),
            _curve([[0.08, 0.25], [0.50, 0.21], [0.92, 0.18 - ht]], 9, rng, jit),
            _ellipse_strokes(0.50, 0.64, 0.34, 0.28, 16, rng=rng, profile=profile),
        ]
    return []


def jungseong_strokes(
    idx: int,
    connectedness: float = 0.5,
    rng: np.random.Generator | None = None,
    profile: WriterProfile | None = None,
) -> list[np.ndarray]:
    """Returns list of smooth curved (M, 2) strokes in normalized [0, 1] x [0, 1] box with clean legibility."""
    ht = float(np.clip(profile.horiz_tilt if profile is not None else 0.04, -0.02, 0.08))
    sb = float(np.clip(profile.stem_bend if profile is not None else 0.0, -0.035, 0.035))
    jit = 0.015 if rng is not None else 0.0

    def v_main(x: float) -> np.ndarray:
        bend = sb + (float(rng.normal(0.0, 0.018)) if rng is not None else -0.01)
        return _curve([[x, 0.02], [x + bend, 0.52], [x, 0.98]], 12, rng, jit)

    def h_main(y: float) -> np.ndarray:
        tilt = ht + (float(rng.normal(0.0, 0.018)) if rng is not None else 0.02)
        return _curve([[0.04, y + 0.02], [0.50, y - 0.01], [0.96, y - tilt]], 11, rng, jit)

    if idx == 0:  # ㅏ
        return [v_main(0.42), _curve([[0.42, 0.48], [0.68, 0.46], [0.94, 0.44 - ht]], 8, rng, jit)]
    elif idx == 1:  # ㅐ
        return [v_main(0.32), _curve([[0.32, 0.48], [0.54, 0.46], [0.74, 0.45 - ht]], 8, rng, jit), v_main(0.76)]
    elif idx == 2:  # ㅑ
        return [
            v_main(0.42),
            _curve([[0.42, 0.35], [0.68, 0.33], [0.94, 0.31 - ht]], 8, rng, jit),
            _curve([[0.42, 0.62], [0.68, 0.60], [0.94, 0.58 - ht]], 8, rng, jit),
        ]
    elif idx == 3:  # ㅒ
        return [
            v_main(0.32),
            _curve([[0.32, 0.35], [0.54, 0.33], [0.74, 0.32 - ht]], 8, rng, jit),
            _curve([[0.32, 0.62], [0.54, 0.60], [0.74, 0.59 - ht]], 8, rng, jit),
            v_main(0.76),
        ]
    elif idx == 4:  # ㅓ
        return [_curve([[0.06, 0.50], [0.34, 0.47], [0.60, 0.45 - ht]], 8, rng, jit), v_main(0.62)]
    elif idx == 5:  # ㅔ
        return [_curve([[0.04, 0.50], [0.22, 0.48], [0.40, 0.46 - ht]], 8, rng, jit), v_main(0.42), v_main(0.80)]
    elif idx == 6:  # ㅕ
        return [
            _curve([[0.06, 0.36], [0.34, 0.34], [0.60, 0.32 - ht]], 8, rng, jit),
            _curve([[0.06, 0.62], [0.34, 0.60], [0.60, 0.58 - ht]], 8, rng, jit),
            v_main(0.62),
        ]
    elif idx == 7:  # ㅖ
        return [
            _curve([[0.04, 0.36], [0.22, 0.34], [0.40, 0.33 - ht]], 8, rng, jit),
            _curve([[0.04, 0.62], [0.22, 0.60], [0.40, 0.59 - ht]], 8, rng, jit),
            v_main(0.42),
            v_main(0.80),
        ]
    elif idx == 8:  # ㅗ
        return [_curve([[0.50, 0.14], [0.49, 0.42], [0.50, 0.68]], 8, rng, jit), h_main(0.68)]
    elif idx == 9:  # ㅘ
        return [
            _curve([[0.28, 0.50], [0.27, 0.64], [0.28, 0.78]], 8, rng, jit),
            _curve([[0.02, 0.80], [0.30, 0.76], [0.58, 0.72 - ht]], 9, rng, jit),
            v_main(0.70),
            _curve([[0.70, 0.48], [0.84, 0.46], [0.98, 0.44 - ht]], 8, rng, jit),
        ]
    elif idx == 10:  # ㅙ
        return [
            _curve([[0.24, 0.50], [0.23, 0.64], [0.24, 0.78]], 8, rng, jit),
            _curve([[0.02, 0.80], [0.28, 0.76], [0.52, 0.72 - ht]], 9, rng, jit),
            v_main(0.62),
            _curve([[0.62, 0.48], [0.76, 0.46], [0.88, 0.45 - ht]], 8, rng, jit),
            v_main(0.90),
        ]
    elif idx == 11:  # ㅚ
        return [
            _curve([[0.30, 0.50], [0.29, 0.64], [0.30, 0.78]], 8, rng, jit),
            _curve([[0.02, 0.80], [0.32, 0.76], [0.62, 0.72 - ht]], 9, rng, jit),
            v_main(0.76),
        ]
    elif idx == 12:  # ㅛ
        return [
            _curve([[0.35, 0.14], [0.34, 0.42], [0.35, 0.68]], 8, rng, jit),
            _curve([[0.65, 0.14], [0.64, 0.42], [0.65, 0.68]], 8, rng, jit),
            h_main(0.68),
        ]
    elif idx == 13:  # ㅜ
        return [h_main(0.32), _curve([[0.50, 0.32], [0.49, 0.62], [0.50, 0.92]], 9, rng, jit)]
    elif idx == 14:  # ㅝ
        return [
            _curve([[0.02, 0.58], [0.32, 0.55], [0.60, 0.52 - ht]], 9, rng, jit),
            _curve([[0.30, 0.56], [0.29, 0.74], [0.30, 0.92]], 8, rng, jit),
            _curve([[0.48, 0.68], [0.62, 0.66], [0.76, 0.64 - ht]], 8, rng, jit),
            v_main(0.78),
        ]
    elif idx == 15:  # ㅞ
        return [
            _curve([[0.02, 0.58], [0.28, 0.55], [0.54, 0.52 - ht]], 9, rng, jit),
            _curve([[0.26, 0.56], [0.25, 0.74], [0.26, 0.92]], 8, rng, jit),
            _curve([[0.44, 0.68], [0.55, 0.66], [0.66, 0.64 - ht]], 8, rng, jit),
            v_main(0.68),
            v_main(0.90),
        ]
    elif idx == 16:  # ㅟ
        return [
            _curve([[0.02, 0.58], [0.32, 0.55], [0.62, 0.52 - ht]], 9, rng, jit),
            _curve([[0.32, 0.56], [0.31, 0.74], [0.32, 0.92]], 8, rng, jit),
            v_main(0.78),
        ]
    elif idx == 17:  # ㅠ
        return [
            h_main(0.32),
            _curve([[0.35, 0.32], [0.34, 0.62], [0.35, 0.92]], 8, rng, jit),
            _curve([[0.65, 0.32], [0.64, 0.62], [0.65, 0.92]], 8, rng, jit),
        ]
    elif idx == 18:  # ㅡ
        return [h_main(0.50)]
    elif idx == 19:  # ㅢ
        return [_curve([[0.02, 0.68], [0.34, 0.65], [0.66, 0.62 - ht]], 9, rng, jit), v_main(0.78)]
    elif idx == 20:  # ㅣ
        return [v_main(0.50)]
    return []


JONGSEONG_MAP = [
    None, 0, 1, None, 2, None, None, 3, 5, None,
    None, None, None, None, None, None, 6, 7, None, 9,
    10, 11, 12, 14, 15, 16, 17, 18,
]

COMPOSITE_JONGSEONG = {
    3: (0, 9),    # ㄳ
    5: (2, 12),   # ㄵ
    6: (2, 18),   # ㄶ
    9: (5, 0),    # ㄺ
    10: (5, 6),   # ㄻ
    11: (5, 7),   # ㄼ
    12: (5, 9),   # ㄽ
    13: (5, 16),  # ㄾ
    14: (5, 17),  # ㄿ
    15: (5, 18),  # ㅀ
    18: (7, 9),   # ㅄ
}


def jongseong_strokes(
    idx: int,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
    profile: WriterProfile | None = None,
) -> list[np.ndarray]:
    if idx == 0:
        return []
    base_idx = JONGSEONG_MAP[idx]
    if base_idx is not None:
        return choseong_strokes(base_idx, rng=rng, connectedness=connectedness, profile=profile)
    if idx in COMPOSITE_JONGSEONG:
        left_c, right_c = COMPOSITE_JONGSEONG[idx]
        s_left = _transform_strokes(choseong_strokes(left_c, rng=rng, connectedness=connectedness, profile=profile), 0.0, 0.0, 0.47, 1.0)
        s_right = _transform_strokes(choseong_strokes(right_c, rng=rng, connectedness=connectedness, profile=profile), 0.53, 0.0, 0.47, 1.0)
        return s_left + s_right
    return []


def _transform_strokes(
    strokes: list[np.ndarray],
    x: float,
    y: float,
    w: float,
    h: float,
    rng: np.random.Generator | None = None,
    neatness: float = 0.6,
) -> list[np.ndarray]:
    if rng is not None:
        clumsy = max(0.45, 1.75 - 1.35 * float(np.clip(neatness, 0.0, 1.0)))
        dx = float(np.clip(rng.normal(0.0, 0.018 * w * clumsy), -0.042 * w, 0.042 * w))
        dy = float(np.clip(rng.normal(0.0, 0.018 * h * clumsy), -0.042 * h, 0.042 * h))
        x += dx
        y += dy
        w *= float(np.clip(np.exp(rng.normal(0.0, 0.032 * clumsy)), 0.90, 1.11))
        h *= float(np.clip(np.exp(rng.normal(0.0, 0.032 * clumsy)), 0.90, 1.11))
        rot = float(np.radians(np.clip(rng.normal(0.0, 0.9 * clumsy), -2.2, 2.2)))
        shear_x = float(np.clip(rng.normal(0.0, 0.020 * clumsy), -0.045, 0.045))
    else:
        rot = 0.0
        shear_x = 0.0

    c, s_rot = np.cos(rot), np.sin(rot)
    cx, cy = x + 0.5 * w, y + 0.5 * h

    out = []
    for s in strokes:
        if len(s) == 0:
            continue
        p = s.copy()
        px = x + p[:, 0] * w + shear_x * (p[:, 1] - 0.5) * w
        py = y + p[:, 1] * h
        if rot != 0.0:
            dx, dy = px - cx, py - cy
            px = cx + c * dx - s_rot * dy
            py = cy + s_rot * dx + c * dy
        out.append(np.stack([px, py], axis=1))
    return out


def _soft_connect(strokes: list[np.ndarray], connectedness: float, rng: np.random.Generator | None) -> list[np.ndarray]:
    """Cursive ligature: only merges consecutive strokes whose endpoints are already virtually touching."""
    if connectedness < 0.45 or len(strokes) < 2:
        return strokes
    p_link = min(0.75, 0.65 * (connectedness - 0.35))
    max_dist = 0.14
    out = [strokes[0]]
    for s in strokes[1:]:
        prev_end = out[-1][-1]
        cur_start = s[0]
        dist = float(np.linalg.norm(cur_start - prev_end))
        roll = float(rng.random()) if rng is not None else 0.5
        if dist < max_dist and roll < p_link:
            mid = 0.5 * (prev_end + cur_start) + np.array([0.0, 0.02 * dist])
            bridge = _curve([prev_end.tolist(), mid.tolist(), cur_start.tolist()], n=4)
            out[-1] = np.concatenate([out[-1], bridge[1:-1], s], axis=0)
        else:
            out.append(s)
    return out


def join_jamo_groups(
    groups: list[list[np.ndarray]],
    connectedness: float,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[JoinInfo]]:
    """Connect across distinct jamo groups (choseong, jungseong, jongseong) in cursive styles."""
    cleaned_groups: list[list[np.ndarray]] = []
    for g in groups:
        if not g:
            continue
        c_g = _soft_connect(g, connectedness, rng)
        if c_g:
            cleaned_groups.append([s.copy() for s in c_g])

    if not cleaned_groups:
        return [], []

    p_cross = float(np.clip(1.5 * (connectedness - 0.40), 0.0, 0.88))
    joins: list[JoinInfo] = []

    out_strokes: list[np.ndarray] = [s for s in cleaned_groups[0]]

    for g_next in cleaned_groups[1:]:
        if not g_next or not out_strokes:
            out_strokes.extend(g_next)
            continue

        tail_a = out_strokes[-1]
        head_b = g_next[0]
        gap = float(np.linalg.norm(head_b[0] - tail_a[-1]))
        roll = float(rng.random()) if rng is not None else 0.5

        joined_done = False
        if roll < p_cross and len(tail_a) >= 2 and len(head_b) >= 2:
            if gap <= 0.30:
                joined = join_strokes(tail_a, head_b, rng, zone=0.20, max_gap=0.30)
                if joined is not None:
                    j_idx = len(tail_a) - 1
                    turn = turn_angle_deg(joined, j_idx, k=1)
                    if turn <= 75.0:
                        pt = (float(joined[j_idx, 0]), float(joined[j_idx, 1]))
                        joins.append(JoinInfo(prev_ch="", next_ch="", point=pt, turn_deg=turn))
                        out_strokes[-1] = joined
                        out_strokes.extend(g_next[1:])
                        joined_done = True
            elif 0.30 < gap <= 1.65 and connectedness >= 0.70:
                p0 = tail_a[-1]
                p1 = head_b[0]
                dxy = p1 - p0
                d_len = max(float(np.linalg.norm(dxy)), 1e-4)
                norm = np.array([-dxy[1] / d_len, dxy[0] / d_len])
                offset = 0.04 * d_len if rng is None else float(rng.uniform(0.02, 0.06))
                mid_ctrl = 0.5 * (p0 + p1) + norm * offset
                path = _curve([p0.tolist(), mid_ctrl.tolist(), p1.tolist()], n=7)

                all_current = out_strokes + g_next
                intersects = any(polylines_intersect(path, s, ignore_endpoints=True) for s in all_current)
                if not intersects:
                    joins.append(JoinInfo(prev_ch="", next_ch="", point=(float(mid_ctrl[0]), float(mid_ctrl[1])), turn_deg=0.0, path=path))
                    out_strokes[-1] = np.concatenate([tail_a, path[1:-1], head_b], axis=0)
                    out_strokes.extend(g_next[1:])
                    joined_done = True

        if not joined_done:
            out_strokes.extend(g_next)

    return out_strokes, joins


def syllable_strokes_with_joins(
    char: str,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
    neatness: float = 0.6,
    profile: WriterProfile | None = None,
) -> tuple[list[np.ndarray], list[JoinInfo]]:
    """Given a Hangul syllable, returns natural handwriting strokes and cross-jamo joins."""
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):
        return [], []

    s_idx = code - 0xAC00
    c_idx = s_idx // 588
    j_idx = (s_idx % 588) // 28
    f_idx = s_idx % 28

    has_jong = f_idx > 0
    eff_conn = connectedness if profile is None else float(np.clip(0.6 * connectedness + 0.4 * profile.allograph_bias, 0.0, 1.0))
    if profile is not None and profile.archetype == "classic_jeongja":
        eff_conn = 0.0
    c_st = choseong_strokes(c_idx, rng=rng, connectedness=eff_conn, profile=profile)
    j_st = jungseong_strokes(j_idx, connectedness=eff_conn, rng=rng, profile=profile)
    f_st = jongseong_strokes(f_idx, rng=rng, connectedness=eff_conn, profile=profile)

    w_sc = profile.w_scale if profile is not None else 1.0
    h_sc = profile.h_scale if profile is not None else 1.0
    cho_sc = float(np.clip(profile.cho_scale if profile is not None else 1.0, 0.90, 1.12))
    jong_sc = float(np.clip(profile.jong_scale if profile is not None else 1.0, 0.88, 1.10))

    if rng is not None:
        clumsy_s = max(0.28, 1.85 - 1.68 * float(np.clip(neatness, 0.0, 1.0)))
        w_sc *= float(np.clip(np.exp(rng.normal(0.0, 0.080 * clumsy_s)), 0.85, 1.18))
        h_sc *= float(np.clip(np.exp(rng.normal(0.0, 0.072 * clumsy_s)), 0.86, 1.16))

    if has_jong:
        W = 1.56 * w_sc
        H = 2.02 * h_sc
        Y_TOP = -0.96 * H
    else:
        W = 1.46 * w_sc
        H = 1.84 * h_sc
        Y_TOP = -0.98 * H

    group_c: list[np.ndarray] = []
    group_j: list[np.ndarray] = []
    group_f: list[np.ndarray] = []

    if j_idx in JUNGSEONG_VERTICAL:
        if not has_jong:
            group_c = _transform_strokes(c_st, 0.04 * W, Y_TOP + 0.08 * H, 0.44 * W * cho_sc, 0.80 * H * cho_sc, rng, neatness=neatness)
            group_j = _transform_strokes(j_st, 0.50 * W, Y_TOP + 0.02 * H, 0.46 * W, 0.94 * H, rng, neatness=neatness)
        else:
            group_c = _transform_strokes(c_st, 0.05 * W, Y_TOP + 0.05 * H, 0.43 * W * cho_sc, 0.50 * H * cho_sc, rng, neatness=neatness)
            group_j = _transform_strokes(j_st, 0.50 * W, Y_TOP + 0.02 * H, 0.45 * W, 0.56 * H, rng, neatness=neatness)
            group_f = _transform_strokes(f_st, 0.16 * W, Y_TOP + 0.60 * H, 0.68 * W * jong_sc, 0.37 * H * jong_sc, rng, neatness=neatness)

    elif j_idx in JUNGSEONG_HORIZONTAL:
        if not has_jong:
            group_c = _transform_strokes(c_st, 0.16 * W, Y_TOP + 0.04 * H, 0.66 * W * cho_sc, 0.46 * H * cho_sc, rng, neatness=neatness)
            group_j = _transform_strokes(j_st, 0.05 * W, Y_TOP + 0.52 * H, 0.90 * W, 0.44 * H, rng, neatness=neatness)
        else:
            group_c = _transform_strokes(c_st, 0.18 * W, Y_TOP + 0.03 * H, 0.64 * W * cho_sc, 0.36 * H * cho_sc, rng, neatness=neatness)
            group_j = _transform_strokes(j_st, 0.06 * W, Y_TOP + 0.40 * H, 0.88 * W, 0.24 * H, rng, neatness=neatness)
            group_f = _transform_strokes(f_st, 0.16 * W, Y_TOP + 0.65 * H, 0.68 * W * jong_sc, 0.33 * H * jong_sc, rng, neatness=neatness)

    else:
        if not has_jong:
            group_c = _transform_strokes(c_st, 0.06 * W, Y_TOP + 0.05 * H, 0.44 * W * cho_sc, 0.42 * H * cho_sc, rng, neatness=neatness)
            group_j = _transform_strokes(j_st, 0.04 * W, Y_TOP + 0.02 * H, 0.92 * W, 0.94 * H, rng, neatness=neatness)
        else:
            group_c = _transform_strokes(c_st, 0.06 * W, Y_TOP + 0.03 * H, 0.44 * W * cho_sc, 0.30 * H * cho_sc, rng, neatness=neatness)
            group_j = _transform_strokes(j_st, 0.04 * W, Y_TOP + 0.02 * H, 0.92 * W, 0.58 * H, rng, neatness=neatness)
            group_f = _transform_strokes(f_st, 0.16 * W, Y_TOP + 0.63 * H, 0.68 * W * jong_sc, 0.35 * H * jong_sc, rng, neatness=neatness)

    groups = [group_c, group_j] if not has_jong else [group_c, group_j, group_f]
    return join_jamo_groups(groups, connectedness=eff_conn, rng=rng)


def syllable_strokes(
    char: str,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
    neatness: float = 0.6,
    profile: WriterProfile | None = None,
) -> list[np.ndarray]:
    """Given a Hangul syllable, returns natural handwriting strokes with allograph & writer profile variation."""
    return syllable_strokes_with_joins(char, rng=rng, connectedness=connectedness, neatness=neatness, profile=profile)[0]


def is_hangul(char: str) -> bool:
    if not char:
        return False
    code = ord(char[0])
    return (
        (0xAC00 <= code <= 0xD7A3)
        or (0x3131 <= code <= 0x318E)
        or (0x1100 <= code <= 0x11FF)
    )


def hangul_char_strokes(
    char: str,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
    neatness: float = 0.6,
    profile: WriterProfile | None = None,
) -> list[np.ndarray]:
    code = ord(char)
    if 0xAC00 <= code <= 0xD7A3:
        return syllable_strokes(char, rng=rng, connectedness=connectedness, neatness=neatness, profile=profile)
    if char in CHOSEONG_NAMES:
        idx = CHOSEONG_NAMES.index(char)
        st = _transform_strokes(choseong_strokes(idx, rng=rng, connectedness=connectedness, profile=profile), 0.1, -1.35, 1.15, 1.35, rng, neatness=neatness)
        return _soft_connect(st, connectedness, rng)
    if char in JUNGSEONG_NAMES:
        idx = JUNGSEONG_NAMES.index(char)
        st = _transform_strokes(jungseong_strokes(idx, connectedness=connectedness, rng=rng, profile=profile), 0.1, -1.35, 1.15, 1.35, rng, neatness=neatness)
        return _soft_connect(st, connectedness, rng)
    return []
