"""Pen models: per-point stroke width and ink deposition from pen dynamics.

Input ink is in page millimetres with real timestamps. Only (x, y, t, p) are used;
speed / direction are derived here, so ANY generator works with ANY pen.

Deposition ``dens`` is optical density at full coverage: 1.0 -> exactly the ink colour,
>1 darker (pooling / overlaps), <1 lighter. The rasteriser composes with Beer-Lambert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ink import P, T, Ink
from ..preprocess import pseudo_pressure
from ..vibe.spec import PenSpec, StyleSpec


@dataclass
class PenSample:
    width_mm: np.ndarray  # (N,)
    dens: np.ndarray  # (N,)
    grain: float  # 0..1: how much the paper tooth modulates deposition (pencil ~1)
    feather: float  # 0..1: lateral ink spread into paper fibres (fountain)
    rim_halo: float = 0.0  # 0..1: capillary edge-ring darkening when liquid ink dries
    dry_streak: float = 0.0  # 0..1: transverse bristle striations (brush) or ballpoint tracks
    porous_threshold: float = 0.0  # 0..1: paper valley threshold where dry media (pencil) skips
    dryness: np.ndarray | None = None  # (N,) per-point dry-brush / skip intensity in [0, 1]


def _speed_mm_s(ink: Ink) -> np.ndarray:
    v = np.zeros(len(ink))
    for sl in ink.stroke_slices():
        s = ink.points[sl].astype(np.float64)
        if len(s) < 2:
            continue
        d = np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1)
        dt = np.maximum(np.diff(s[:, T]), 1e-3)
        vv = np.concatenate([[d[0] / dt[0]], d / dt])
        k = min(5, len(vv))
        v[sl] = np.convolve(vv, np.ones(k) / k, mode="same")
    return v


def _direction(ink: Ink) -> np.ndarray:
    """Movement direction (radians, math convention: CCW from +x with y UP)."""
    a = np.zeros(len(ink))
    for sl in ink.stroke_slices():
        s = ink.points[sl].astype(np.float64)
        if len(s) < 2:
            continue
        d = np.gradient(s[:, :2], axis=0)
        a[sl] = np.arctan2(-d[:, 1], d[:, 0])
    return a


def _arc_noise(ink: Ink, rng: np.random.Generator, period_mm: float) -> np.ndarray:
    """1D value noise along arc length, in [0, 1]."""
    out = np.zeros(len(ink))
    for sl in ink.stroke_slices():
        s = ink.points[sl].astype(np.float64)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1))])
        knots = rng.random(int(arc[-1] / period_mm) + 2)
        out[sl] = np.interp(arc / period_mm, np.arange(len(knots)), knots)
    return out


def _stroke_ends(ink: Ink, mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights in [0,1] that are 1 at stroke start / end and fade over ``mm``."""
    start, end = np.zeros(len(ink)), np.zeros(len(ink))
    for sl in ink.stroke_slices():
        s = ink.points[sl].astype(np.float64)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1))])
        start[sl] = np.clip(1 - arc / mm, 0, 1)
        end[sl] = np.clip(1 - (arc[-1] - arc) / mm, 0, 1)
    return start, end


def _curvature_and_downbias(ink: Ink) -> tuple[np.ndarray, np.ndarray]:
    """Returns (curvature in 1/mm, downward movement bias in [-1, 1])."""
    curv = np.zeros(len(ink))
    down = np.zeros(len(ink))
    for sl in ink.stroke_slices():
        s = ink.points[sl].astype(np.float64)
        if len(s) < 2:
            continue
        dxy = np.gradient(s[:, :2], axis=0)
        ds = np.maximum(np.linalg.norm(dxy, axis=1), 1e-4)
        down[sl] = np.clip(dxy[:, 1] / ds, -1.0, 1.0)
        if len(s) >= 3:
            ang = np.arctan2(dxy[:, 1], dxy[:, 0])
            dang = np.abs((np.gradient(ang) + np.pi) % (2 * np.pi) - np.pi)
            curv[sl] = np.clip(dang / ds, 0.0, 20.0)
    return curv, down


def _dwell_pool(ink: Ink, radius_mm: float = 0.25, tau_s: float = 0.25) -> np.ndarray:
    """Per-point ink pooling based on local dwell time (seconds spent within radius_mm)."""
    n = len(ink)
    pool = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pool

    for sl in ink.stroke_slices():
        s = ink.points[sl].astype(np.float64)
        if len(s) < 2:
            continue
        xy = s[:, :2]
        t = s[:, T]
        diffs = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(diffs)])
        dt = np.zeros(len(s))
        if len(t) >= 2:
            dt[0] = 0.5 * max(t[1] - t[0], 0.0)
            dt[-1] = 0.5 * max(t[-1] - t[-2], 0.0)
            if len(t) > 2:
                dt[1:-1] = 0.5 * np.maximum(t[2:] - t[:-2], 0.0)

        left = np.searchsorted(arc, arc - radius_mm, side="left")
        right = np.searchsorted(arc, arc + radius_mm, side="right")
        cum_dt = np.concatenate([[0.0], np.cumsum(dt)])
        local_t = cum_dt[right] - cum_dt[left]

        # Net pause dwell: excess time spent locally above rapid stroke speed (~0.02s)
        excess_t = np.maximum(local_t - 0.02, 0.0)
        pool[sl] = 1.0 - np.exp(-excess_t / tau_s)

    return pool


def pen_samples(ink: Ink, pen: PenSpec, style: StyleSpec, rng: np.random.Generator) -> PenSample:
    n = len(ink)
    p = ink.points[:, P].astype(np.float64) if ink.has_pressure else pseudo_pressure(ink).astype(np.float64)
    p = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)
    v = _speed_mm_s(ink)
    curv, down = _curvature_and_downbias(ink)
    pool = _dwell_pool(ink)
    W = pen.width_mm
    start, end = _stroke_ends(ink, 0.75)
    grain, feather = 0.15, 0.0
    rim_halo, dry_streak, porous_threshold = 0.0, 0.0, 0.0
    dryness = np.zeros(n, dtype=np.float64)

    if pen.type == "ballpoint":
        # Crisp oil-paste ballpoint: thin on upstrokes/flicks, fuller on downstrokes
        taper = np.clip(1.0 - 0.52 * end, 0.30, 1.0)
        w = W * (0.48 + 0.50 * p + 0.12 * np.maximum(down, 0)) * taper
        # Paste ink transfers strongly when pulling down, skims lightly when pushing up
        dens = (0.58 + 0.72 * p) * (0.78 + 0.28 * down)
        skip_amt = np.clip((v - 38.0) / 145.0, 0.0, 0.75)
        nz = _arc_noise(ink, rng, 0.55)
        dryness = np.clip(skip_amt * np.clip((nz - 0.35) * 2.0, 0, 1) + 0.25 * np.maximum(-down, 0), 0.0, 0.78)
        dens *= 1.0 - 0.52 * dryness
        # Occasional dry start before ball catches traction + dark ink blob at sharp turns
        start_gate = np.zeros(n)
        for sl in ink.stroke_slices():
            start_gate[sl] = -0.38 if rng.random() < 0.30 else (0.70 if rng.random() < 0.25 else 0.0)
        corner_blob = np.clip((curv - 2.2) / 5.0, 0.0, 1.0) * (0.5 + 0.5 * nz)
        dens = np.clip(dens * (1.0 + start * start_gate) + 0.90 * corner_blob, 0.24, 2.4)
        dens *= 1.0 + 0.30 * pool
        w *= 1.0 + 0.22 * corner_blob
        grain = 0.36
        dry_streak = 0.42
    elif pen.type == "gel":
        taper = np.clip(1.0 + 0.12 * start - 0.48 * end, 0.32, 1.12)
        w = W * (0.72 + 0.36 * p) * taper
        dens = (0.88 + 0.38 * p) * (1.0 + 0.38 * end + 0.18 * start) * (1.0 + 0.60 * pool)
        rim_halo = 0.22
        grain = 0.12
    elif pen.type == "fountain":
        th = _direction(ink)
        # Chisel + flex nib: thin cross-strokes/upstrokes, rich shaded downstrokes
        nib_mod = 0.62 + 0.52 * np.abs(np.sin(th - np.radians(38.0)))
        flex_mod = 0.52 + 0.78 * np.power(p, 1.15) + 0.18 * np.maximum(down, 0)
        taper = np.clip(1.0 + 0.14 * start - 0.52 * end, 0.32, 1.15)
        w = W * flex_mod * nib_mod * taper * (1.0 + 0.25 * pool)
        # Liquid dye ink shading: translucent in fast mid-strokes, deep dark pooling at slow turns & stroke ends
        v0 = 26.0
        flow = v0 / (v + v0)
        dens = (0.38 + 1.12 * flow) * (0.82 + 0.32 * p)
        dens *= (1.0 + 0.78 * end + 0.28 * start + 0.25 * np.clip(curv / 4.0, 0, 1)) * (1.0 + 0.90 * pool)
        feather = 0.65
        grain = 0.18
        rim_halo = 0.62
    elif pen.type == "pencil":
        # Graphite lead: directional pressure shading + porous paper tooth valleys
        taper = np.clip(1.0 - 0.36 * end, 0.45, 1.05)
        w = W * (0.64 + 0.50 * p + 0.10 * np.maximum(down, 0)) * taper
        dens = (0.38 + 0.92 * np.power(p, 1.12)) * (0.80 + 0.28 * down)
        nz = _arc_noise(ink, rng, 0.9)
        dens *= 0.84 + 0.34 * nz
        grain = 1.0
        porous_threshold = 0.40
        dryness = np.clip(0.58 - 0.45 * p, 0.08, 0.58)
    elif pen.type == "marker":
        chisel = 0.78 + 0.32 * np.abs(np.sin(_direction(ink) - np.radians(25.0)))
        taper = np.clip(1.0 - 0.25 * end, 0.55, 1.05)
        w = W * (0.82 + 0.24 * p) * chisel * taper
        v0 = 45.0
        dens = (0.48 + 0.42 * (v0 / (v + v0))) * (1.0 + 0.42 * end + 0.20 * start) * (1.0 + 0.60 * pool)
        grain = 0.22
        feather = 0.38
        rim_halo = 0.28
    elif pen.type == "brush":
        # Oriental brush / modern brush pen: 기필 (entry press), 송필 (belly), 수필 (needle-sharp exit flick)
        th = _direction(ink)
        dir_belly = 0.76 + 0.36 * np.abs(np.sin(th - np.radians(42.0))) + 0.18 * np.maximum(down, 0)
        taper = np.clip(1.0 + 0.34 * start - 0.84 * end, 0.10, 1.32)
        w = W * (0.18 + 1.02 * np.power(p, 1.22)) * dir_belly * taper
        # Bristle ink depletion & high-speed dry-brush (갈필 / 비백)
        nz = _arc_noise(ink, rng, 1.1)
        speed_dry = np.clip((v - 58.0) / 160.0, 0.0, 0.80)
        tail_dry = np.clip(end * 1.15 - 0.18, 0.0, 0.85) * (1.0 - 0.5 * p)
        dryness = np.clip((0.50 * speed_dry + 0.60 * tail_dry) * (0.45 + 0.75 * nz), 0.0, 0.88)
        dens = (0.88 + 0.62 * p) * (1.0 - 0.38 * dryness) * (1.0 + 0.32 * start)
        grain = 0.62
        feather = 0.20
        dry_streak = 0.76
        rim_halo = 0.22
    elif pen.type == "broad_nib":
        th = _direction(ink)
        chisel = np.abs(np.sin(th - np.radians(pen.nib_angle_deg)))
        taper = np.clip(1.0 + 0.10 * start - 0.22 * end, 0.65, 1.12)
        w = W * (0.11 + 1.05 * chisel) * (0.82 + 0.28 * p) * taper
        v0 = 32.0
        dens = (0.58 + 0.72 * (v0 / (v + v0))) * (1.0 + 0.52 * end + 0.22 * start)
        feather = 0.32
        grain = 0.22
        rim_halo = 0.52
    else:  # fall back to ballpoint-like
        w = np.full(n, W)
        dens = np.ones(n)
    return PenSample(
        np.asarray(w, np.float64),
        np.asarray(dens, np.float64),
        grain,
        feather,
        rim_halo=rim_halo,
        dry_streak=dry_streak,
        porous_threshold=porous_threshold,
        dryness=dryness,
    )
