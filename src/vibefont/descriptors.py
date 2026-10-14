"""Automatically measurable style descriptors.

These numbers are the bridge between "vibe" words and the model:
  * measured on every training line -> conditioning vector (no human labels needed)
  * measured on writers -> prototypes for prompt -> style lookup
  * measured on generated lines -> post-correction toward the requested VibeSpec

All descriptors expect NORMALISED ink (x-height = 1, baseline y = 0, y down).
"""

from __future__ import annotations

import numpy as np

from .ink import T, Ink

DESCRIPTOR_NAMES = (
    "slant_deg",  # dominant lean of down-strokes, + = right
    "width_per_char",  # line width / n_chars (x-heights)
    "speed",  # mean pen-down speed (x-heights / s)
    "connectedness",  # 0 = every letter separate, 1 = one stroke per word
    "roundness",  # 0 = angular, 1 = round
    "baseline_wobble",  # std of stroke bottoms around baseline (x-heights)
    "ascender_ratio",  # 95th pct height above baseline (x-heights)
    "pressure",  # mean pressure (0.5 if unknown)
)


def _down_segments(ink: Ink) -> tuple[np.ndarray, np.ndarray]:
    segs, dts = [], []
    for s in ink.strokes():
        if len(s) >= 2:
            segs.append(np.diff(s[:, :2], axis=0))
            dts.append(np.diff(s[:, T]))
    if not segs:
        return np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(segs).astype(np.float64), np.concatenate(dts).astype(np.float64)


def slant_deg(ink: Ink) -> float:
    d, _ = _down_segments(ink)
    if len(d) == 0:
        return 0.0
    # orient every segment upward (dy < 0 in y-down coords)
    d = np.where(d[:, 1:2] > 0, -d, d)
    ang = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))  # 0 = vertical, + = leaning right
    w = np.linalg.norm(d, axis=1)
    m = np.abs(ang) < 45
    if w[m].sum() <= 0:
        return 0.0
    a, ww = ang[m], w[m]
    order = np.argsort(a)
    cw = np.cumsum(ww[order])
    return float(a[order][np.searchsorted(cw, cw[-1] / 2)])  # weighted median


def roundness(ink: Ink) -> float:
    """1 - fraction of total turning that happens in sharp corners."""
    total = sharp = 0.0
    for s in ink.strokes():
        if len(s) < 3:
            continue
        d = np.diff(s[:, :2].astype(np.float64), axis=0)
        L = np.linalg.norm(d, axis=1)
        keep = L > 1e-4
        d = d[keep]
        if len(d) < 2:
            continue
        a = np.arctan2(d[:, 1], d[:, 0])
        turn = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
        total += turn.sum()
        sharp += turn[turn > np.radians(35)].sum()
    return float(1.0 - sharp / total) if total > 0 else 0.5


def measure(ink: Ink, text: str | None = None) -> dict[str, float]:
    text = ink.text if text is None else text
    n_chars = max(len(text.replace(" ", "")), 1)
    n_words = max(len(text.split()), 1)
    xmin, ymin, xmax, _ = ink.bbox()
    d, dt = _down_segments(ink)
    v = np.linalg.norm(d, axis=1) / np.maximum(dt, 1e-3) if len(d) else np.zeros(1)
    n_strokes = len(ink.stroke_slices())
    conn = 1.0 - (n_strokes - n_words) / max(n_chars - n_words, 1)
    bottoms = []
    for s in ink.strokes():
        if len(s) >= 3:
            y = s[:, 1]
            m = (y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])
            bottoms.append(y[1:-1][m])
    b = np.concatenate(bottoms) if bottoms else np.zeros(1)
    b = b[np.abs(b) < 0.5]  # ignore descenders
    return {
        "slant_deg": slant_deg(ink),
        "width_per_char": float((xmax - xmin) / n_chars),
        "speed": float(np.median(v)),
        "connectedness": float(np.clip(conn, 0.0, 1.0)),
        "roundness": roundness(ink),
        "baseline_wobble": float(b.std()) if len(b) > 1 else 0.0,
        "ascender_ratio": float(np.quantile(-ink.xy[:, 1], 0.95)) if len(ink) else 1.0,
        "pressure": float(np.nanmean(ink.points[:, 3])) if ink.has_pressure else 0.5,
    }


def as_vector(desc: dict[str, float]) -> np.ndarray:
    return np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32)
