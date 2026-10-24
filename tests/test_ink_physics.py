import numpy as np
import pytest

from vibefont.ink import Ink, X, Y, T, P, EOS
from vibefont.render.pens import pen_samples
from vibefont.vibe.spec import PenSpec, StyleSpec


def ink_with_pause(pause_s: float) -> Ink:
    """A 20mm straight horizontal stroke at 25mm/s with pauses at 5mm (0.1s) and 10mm (pause_s)."""
    xs = np.linspace(0.0, 20.0, 201)
    pts = np.zeros((len(xs), 5), dtype=np.float32)
    pts[:, X] = xs
    pts[:, Y] = 0.0

    cur_t = 0.0
    for i in range(len(xs)):
        if i > 0:
            dt = (xs[i] - xs[i - 1]) / 25.0
            cur_t += dt
        if abs(xs[i] - 5.0) < 0.06:
            cur_t += 0.10
        elif abs(xs[i] - 10.0) < 0.06:
            cur_t += pause_s
        pts[i, T] = cur_t
        pts[i, P] = 0.5
        pts[i, EOS] = 0.0
    pts[-1, EOS] = 1.0
    return Ink(pts)


def idx_at(ink: Ink, x_target: float) -> int:
    return int(np.argmin(np.abs(ink.points[:, X] - x_target)))


def test_longer_pause_pools_more_ink_fountain():
    ink = ink_with_pause(0.6)
    d = pen_samples(ink, PenSpec(type="fountain", width_mm=0.5), StyleSpec(), np.random.default_rng(0)).dens
    # Longer pause at 10mm should pool more ink than shorter pause at 5mm
    assert d[idx_at(ink, 10.0)] >= 1.20 * d[idx_at(ink, 5.0)]


def test_dwell_pooling_does_not_leak_into_pencil():
    ink = ink_with_pause(0.6)
    d = pen_samples(ink, PenSpec(type="pencil", width_mm=0.5), StyleSpec(), np.random.default_rng(0)).dens
    assert d[idx_at(ink, 10.0)] <= 1.15 * np.median(d)
