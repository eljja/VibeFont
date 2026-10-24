import numpy as np
import pytest

from vibefont.ink import Ink, X, Y, T, P, EOS
from vibefont.render.pens import pen_samples, _ballpoint_goop
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


def line_ink(length_mm: float, speed: float = 60.0, step_mm: float = 0.05) -> Ink:
    xs = np.arange(0.0, length_mm + step_mm / 2, step_mm)
    pts = np.zeros((len(xs), 5), dtype=np.float32)
    pts[:, X] = xs
    pts[:, Y] = 0.0
    pts[:, T] = xs / max(speed, 1e-3)
    pts[:, P] = 0.5
    pts[:, EOS] = 0.0
    pts[-1, EOS] = 1.0
    return Ink(pts)


def zigzag_ink(corners_at_mm: list[float], speed: float = 60.0, step_mm: float = 0.05) -> Ink:
    # Path along x with 90-degree corners at specified arc distances
    total_len = max(corners_at_mm) + 5.0
    n_pts = int(total_len / step_mm) + 1
    pts = np.zeros((n_pts, 5), dtype=np.float32)
    cur_x, cur_y = 0.0, 0.0
    heading = 0.0  # 0: +x, 1: +y
    cur_dist = 0.0
    corner_set = sorted(corners_at_mm)
    c_idx = 0
    for i in range(n_pts):
        pts[i, X] = cur_x
        pts[i, Y] = cur_y
        pts[i, T] = cur_dist / speed
        pts[i, P] = 0.5
        pts[i, EOS] = 0.0
        if c_idx < len(corner_set) and cur_dist >= corner_set[c_idx]:
            heading = 1.0 - heading  # turn 90 deg
            c_idx += 1
        if heading == 0.0:
            cur_x += step_mm
        else:
            cur_y += step_mm
        cur_dist += step_mm
    pts[-1, EOS] = 1.0
    return Ink(pts)


def dominant_period_mm(signal: np.ndarray, ink: Ink) -> float:
    diffs = np.linalg.norm(np.diff(ink.xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    step = float(np.median(diffs))
    # Resample evenly
    n = len(signal)
    fft = np.abs(np.fft.rfft(signal - np.mean(signal)))
    freqs = np.fft.rfftfreq(n, d=step)
    # Ignore DC and very low frequencies (period > 10mm)
    valid = freqs > 0.1
    if not np.any(valid):
        return 1.0
    peak_idx = int(np.argmax(fft[valid]))
    peak_freq = float(freqs[valid][peak_idx])
    return 1.0 / peak_freq if peak_freq > 1e-4 else 1.0


def idx_at_dist(ink: Ink, dist_mm: float) -> int:
    diffs = np.linalg.norm(np.diff(ink.xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    return int(np.argmin(np.abs(arc - dist_mm)))


def test_goop_grows_with_travel_before_release():
    ink = zigzag_ink(corners_at_mm=[5.0, 25.0], speed=60.0)
    blob, _ = _ballpoint_goop(ink, np.random.default_rng(0))
    # Corner at 25mm traveled 20mm since corner at 5mm; should have much larger blob
    i1 = idx_at_dist(ink, 5.0)
    i2 = idx_at_dist(ink, 25.0)
    b1 = float(np.max(blob[max(0, i1 - 3):i1 + 4]))
    b2 = float(np.max(blob[max(0, i2 - 3):i2 + 4]))
    assert b2 > b1 + 0.30


def test_skip_increases_with_speed():
    _, s_fast = _ballpoint_goop(line_ink(60.0, speed=200.0), np.random.default_rng(0))
    _, s_slow = _ballpoint_goop(line_ink(60.0, speed=30.0), np.random.default_rng(0))
    assert s_fast.mean() >= 3.0 * s_slow.mean() + 0.02


def test_skip_has_ball_period():
    ink = line_ink(60.0, speed=200.0)
    _, s = _ballpoint_goop(ink, np.random.default_rng(0))
    period = dominant_period_mm(s, ink)
    expected_period = np.pi * 0.7
    assert abs(period - expected_period) <= 0.40
