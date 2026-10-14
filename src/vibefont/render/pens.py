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


def pen_samples(ink: Ink, pen: PenSpec, style: StyleSpec, rng: np.random.Generator) -> PenSample:
    n = len(ink)
    p = ink.points[:, P].astype(np.float64) if ink.has_pressure else pseudo_pressure(ink).astype(np.float64)
    p = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)
    v = _speed_mm_s(ink)
    W = pen.width_mm
    start, end = _stroke_ends(ink, 0.5)
    grain, feather = 0.15, 0.0

    if pen.type == "ballpoint":
        w = W * (0.85 + 0.3 * p)
        dens = 0.70 + 0.40 * p
        skip_amt = np.clip((v - 60.0) / 220.0, 0.0, 0.7)
        nz = _arc_noise(ink, rng, 0.7)
        dens *= 1.0 - skip_amt * np.clip((nz - 0.45) * 2.2, 0, 1)
        gate = np.zeros(n)
        for sl in ink.stroke_slices():  # ink blob at the start of ~1 in 4 strokes
            gate[sl] = float(rng.random() < 0.25)
        dens *= 1.0 + 0.6 * start * gate
    elif pen.type == "gel":
        w = W * (0.95 + 0.1 * p)
        dens = np.full(n, 1.15)
        dens *= 1.0 + 0.3 * start
    elif pen.type == "fountain":
        w = W * (0.70 + 0.55 * p)
        v0 = 35.0
        dens = 0.55 + 0.85 * v0 / (v + v0)  # slow -> pooling
        dens *= 1.0 + 0.45 * end + 0.2 * start
        feather = 0.6
        grain = 0.05
    elif pen.type == "pencil":
        w = W * (0.9 + 0.35 * p)
        dens = 0.25 + 0.60 * p
        grain = 1.0
    elif pen.type == "marker":
        w = W * (0.9 + 0.15 * p)
        dens = np.full(n, 0.6)
        dens *= 1.0 + 0.25 * end
        grain = 0.05
        feather = 0.3
    elif pen.type == "brush":
        w = W * (0.12 + 1.0 * np.power(p, 1.3))
        dens = 1.2 * (1.0 - 0.5 * np.clip((v - 120.0) / 300.0, 0, 1))
        grain = 0.5
    elif pen.type == "broad_nib":
        th = _direction(ink)
        w = W * np.abs(np.sin(th - np.radians(pen.nib_angle_deg))) + 0.12 * W
        dens = np.full(n, 1.1) * (1.0 + 0.3 * end)
        feather = 0.2
    else:  # fall back to ballpoint-like
        w = np.full(n, W)
        dens = np.ones(n)
    return PenSample(np.asarray(w, np.float64), np.asarray(dens, np.float64), grain, feather)
