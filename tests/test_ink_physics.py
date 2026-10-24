import numpy as np
import pytest

from vibefont.ink import Ink, X, Y, T, P, EOS
from vibefont.render.pens import pen_samples
from vibefont.render.fluid import feather_anisotropic, fibre_field
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


def gaussian_dot(size: int = 64, sigma: float = 3.0) -> np.ndarray:
    y, x = np.mgrid[-size // 2:size // 2, -size // 2:size // 2]
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)


def second_moments(img: np.ndarray) -> tuple[float, float]:
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]
    total = float(img.sum())
    if total < 1e-9:
        return 0.0, 0.0
    cx = float((x * img).sum() / total)
    cy = float((y * img).sum() / total)
    Ixx = float(((x - cx)**2 * img).sum() / total)
    Iyy = float(((y - cy)**2 * img).sum() / total)
    return Ixx, Iyy


def test_feather_spreads_along_fibres():
    D = gaussian_dot(64, sigma=3.0)
    th = np.zeros_like(D)
    s = np.ones_like(D)
    out = feather_anisotropic(D, th, s, amount=0.8, px_per_mm=11.8)
    Ixx, Iyy = second_moments(out)
    assert Ixx / Iyy >= 1.4


def test_feather_amount_zero_is_identity():
    D = gaussian_dot(64, 3.0)
    assert np.array_equal(feather_anisotropic(D, np.zeros_like(D), np.ones_like(D), 0.0, 11.8), D)


def test_feather_mass_bounded():
    D = gaussian_dot(64, 3.0)
    out = feather_anisotropic(D, np.zeros_like(D), np.ones_like(D), 0.8, 11.8)
    assert D.sum() <= out.sum() <= 1.25 * D.sum()
