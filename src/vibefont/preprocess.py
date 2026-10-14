"""Preprocessing of raw online ink into the normalised model space.

Normalised space: x-height == 1, baseline y == 0, line starts at x == 0, y down.

NOTE (from review): resampling is done on a uniform TIME grid, never on arc length,
because pen speed drives the ink shader (pooling, skips) and is part of the style.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ink import EOS, NCOLS, P, T, X, Y, Ink


def _clean_time(t: np.ndarray) -> np.ndarray:
    """Make timestamps strictly increasing (devices often emit duplicates)."""
    t = t.astype(np.float64).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-4
    return t


def resample_time(ink: Ink, hz: float = 100.0) -> Ink:
    """Resample every stroke on a uniform time grid of ``hz`` samples / second.

    Pen-up gaps between strokes are preserved (their duration stays in ``t``).
    Strokes shorter than one sample period keep their first and last point.
    """
    dt = 1.0 / hz
    out = []
    for s in ink.strokes():
        t = _clean_time(s[:, T])
        if t[-1] - t[0] < dt or len(s) < 2:
            idx = [0] if len(s) == 1 else [0, len(s) - 1]
            r = s[idx].astype(np.float64)
        else:
            tn = np.arange(t[0], t[-1] + 1e-9, dt)
            if tn[-1] < t[-1] - 1e-6:
                tn = np.append(tn, t[-1])
            r = np.empty((len(tn), NCOLS))
            r[:, X] = np.interp(tn, t, s[:, X])
            r[:, Y] = np.interp(tn, t, s[:, Y])
            r[:, T] = tn
            r[:, P] = np.interp(tn, t, s[:, P]) if not np.isnan(s[:, P]).all() else np.nan
        r[:, EOS] = 0.0
        r[-1, EOS] = 1.0
        out.append(r)
    pts = np.concatenate(out) if out else np.zeros((0, NCOLS))
    return Ink(pts.astype(np.float32), ink.text, dict(ink.meta))


def _robust_line(x: np.ndarray, y: np.ndarray, iters: int = 3) -> tuple[float, float]:
    """Fit y = a*x + b with iterative outlier rejection."""
    keep = np.ones(len(x), bool)
    a, b = 0.0, float(np.median(y)) if len(y) else 0.0
    for _ in range(iters):
        if keep.sum() < 2:
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r[keep])) + 1e-9
        keep = np.abs(r) < 2.5 * s
    return float(a), float(b)


def _local_extrema(ink: Ink, which: str) -> np.ndarray:
    """Local y maxima ('bottom', since y is down) or minima ('top') of pen-down strokes."""
    pts = []
    for s in ink.strokes():
        y = s[:, Y]
        if len(y) < 3:
            continue
        if which == "bottom":
            m = (y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])
        else:
            m = (y[1:-1] <= y[:-2]) & (y[1:-1] < y[2:])
        idx = np.flatnonzero(m) + 1
        pts.append(s[idx, :2])
    return np.concatenate(pts) if pts else np.zeros((0, 2), np.float32)


@dataclass
class LineGeometry:
    baseline_slope: float  # dy/dx of the baseline (raw units)
    baseline_offset: float
    x_height: float  # raw units
    x_min: float


def estimate_line_geometry(ink: Ink) -> LineGeometry:
    """Estimate baseline and x-height of a single text line.

    Baseline: robust line through stroke bottoms. Corpus (mean) line: robust line through
    stroke tops. x-height = median vertical distance between them. Heuristic but adequate
    for normalisation; descenders/ascenders are rejected as outliers by the robust fit.
    """
    bottoms = _local_extrema(ink, "bottom")
    tops = _local_extrema(ink, "top")
    x0 = float(ink.xy[:, 0].min()) if len(ink) else 0.0
    if len(bottoms) < 2 or len(tops) < 2:
        ymin, ymax = (ink.xy[:, 1].min(), ink.xy[:, 1].max()) if len(ink) else (0, 1)
        return LineGeometry(0.0, float(ymax), float(max(ymax - ymin, 1e-6) * 0.5), x0)
    a, b = _robust_line(bottoms[:, 0], bottoms[:, 1])
    # tops measured relative to the baseline, keep the core (x-height) band
    rel = (a * tops[:, 0] + b) - tops[:, 1]
    rel = rel[rel > 0]
    if len(rel) == 0:
        xh = float(np.ptp(ink.xy[:, 1]) * 0.5)
    else:
        # ascenders inflate the upper tail -> use a lower quantile
        xh = float(np.quantile(rel, 0.4))
    return LineGeometry(a, b, max(xh, 1e-6), x0)


def normalize_line(ink: Ink, geom: LineGeometry | None = None) -> tuple[Ink, LineGeometry]:
    """Remove baseline tilt, move baseline to y=0 / start to x=0, scale x-height to 1.

    Slant is NOT removed: it is part of the writer's style.
    """
    geom = geom or estimate_line_geometry(ink)
    ang = np.degrees(np.arctan(geom.baseline_slope))
    out = ink.rotate(-ang, cx=geom.x_min, cy=geom.baseline_slope * geom.x_min + geom.baseline_offset)
    base_y = geom.baseline_slope * geom.x_min + geom.baseline_offset
    out = out.translate(-geom.x_min, -base_y).scale(1.0 / geom.x_height)
    # time relative to line start
    if len(out):
        out.points[:, T] -= out.points[0, T]
    return out, geom


def pseudo_pressure(ink: Ink, a: float = 1.5, b: float = 0.12) -> np.ndarray:
    """Pressure proxy from speed when the device recorded none: slow -> heavy.

    Expects normalised units (x-height / s). Returns values in (0, 1).
    """
    p = np.full(len(ink), 0.5, np.float32)
    for sl in ink.stroke_slices():
        s = ink.points[sl]
        if len(s) < 2:
            continue
        d = np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1)
        dt = np.maximum(np.diff(s[:, T]), 1e-3)
        v = np.concatenate([[d[0] / dt[0]], d / dt])
        k = min(5, len(v))
        v = np.convolve(v, np.ones(k) / k, mode="same")
        pp = 1.0 / (1.0 + np.exp(-(a - b * v)))
        # pen lands/leaves lightly
        n = len(pp)
        ramp = np.minimum(1.0, np.minimum(np.arange(n) + 1, n - np.arange(n)) / 4.0)
        p[sl] = (0.25 + 0.75 * ramp) * pp
    return p


def ensure_pressure(ink: Ink) -> Ink:
    if ink.has_pressure:
        return ink
    out = ink.copy()
    out.points[:, P] = pseudo_pressure(ink)
    return out
