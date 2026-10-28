"""Synthetic pen kinematics for geometry that has no timing (placeholder generator, fonts).

Uses the two-thirds power law of human movement: tangential speed v = K * curvature^(-1/3),
i.e. the pen slows down in tight curves and speeds up on straight parts.
"""

from __future__ import annotations

import numpy as np

from ..ink import NCOLS, Ink
from ..preprocess import resample_time


def densify(poly: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline (M, 2) at uniform arc-length ``step``."""
    poly = np.asarray(poly, np.float64)
    if len(poly) < 2:
        return poly
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return poly[:1]
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    sn = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(sn, s, poly[:, 0]), np.interp(sn, s, poly[:, 1])], axis=1)


def power_law_times(poly: np.ndarray, k_speed: float, v_min: float = 1.0, v_max: float = 30.0) -> np.ndarray:
    """Timestamps for a dense polyline. Units: x-heights and seconds."""
    if len(poly) < 2:
        return np.zeros(len(poly))
    d = np.diff(poly, axis=0)
    ds = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    kappa = np.concatenate([[turn[0] if len(turn) else 0.0], turn / np.maximum(ds[1:], 1e-6)])
    k = min(5, len(kappa))
    kappa = np.convolve(kappa, np.ones(k) / k, mode="same")
    v = k_speed * np.power(np.maximum(kappa, 1e-3), -1.0 / 3.0)
    # accelerate from / decelerate to rest at stroke ends
    n = len(v)
    ramp = np.minimum(1.0, (np.minimum(np.arange(n), n - 1 - np.arange(n)) + 1) / 6.0)
    v = np.clip(v * (0.3 + 0.7 * ramp), v_min, v_max)
    return np.concatenate([[0.0], np.cumsum(ds / v)])


def strokes_to_ink(strokes: list[np.ndarray], speed: float, rng: np.random.Generator,
                   hz: float = 100.0, text: str = "") -> Ink:
    """Geometry-only strokes (x-height units) -> timed, time-resampled Ink.

    speed in [0, 1] (VibeSpec.style.speed).
    """
    k = 2.5 + 4.0 * speed
    rows, t = [], 0.0
    for s in strokes:
        dense = densify(s, 0.02)
        if len(dense) == 0:
            continue
        ts = power_law_times(dense, k * float(np.exp(0.1 * rng.standard_normal())))
        r = np.zeros((len(dense), NCOLS))
        r[:, :2] = dense
        r[:, 2] = t + ts
        r[:, 3] = np.nan
        r[-1, 4] = 1.0
        rows.append(r)
        t = r[-1, 2] + (0.18 - 0.1 * speed) * float(np.exp(0.3 * rng.standard_normal()))
    pts = np.concatenate(rows) if rows else np.zeros((0, NCOLS))
    return resample_time(Ink(pts.astype(np.float32), text), hz)


def ballistic_flourish(
    start_pt: np.ndarray | list[float],
    initial_direction: np.ndarray | list[float],
    length_mm: float,
    inertia_curvature: float = 0.05,
    initial_pressure: float = 0.70,
    terminal_pressure: float = 0.05,
    initial_speed: float = 1.0,
    n_pts: int = 20,
    is_entrance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a biomechanically realistic ballistic flourish stroke.

    Models arm/wrist rotational inertia and neuromotor ballistic momentum:
    - Exit swashes: start at start_pt, sweep out along initial_direction, curving naturally
      due to joint pivot inertia, with speed and pressure decaying smoothly to zero.
    - Entrance swashes: touch down lightly and accelerate into start_pt, arriving with
      tangent matching initial_direction.

    Returns:
        (points: np.ndarray of shape (n_pts, 2), pressures: np.ndarray of shape (n_pts,))
    """
    p0 = np.asarray(start_pt, dtype=np.float64)
    v0 = np.asarray(initial_direction, dtype=np.float64)
    norm_v = np.linalg.norm(v0)
    if norm_v < 1e-6:
        v0 = np.array([1.0, 0.0], dtype=np.float64)
    else:
        v0 = v0 / norm_v

    theta_0 = np.arctan2(v0[1], v0[0])
    u = np.linspace(0.0, 1.0, n_pts)
    ds = float(length_mm) / max(1, n_pts - 1)

    if not is_entrance:
        # Exit flourish: travels forward from p0
        # dtheta/ds = inertia_curvature * (1.0 - 0.25 * u)
        delta_theta = inertia_curvature * float(length_mm) * (u - 0.125 * (u ** 2))
        theta = theta_0 + delta_theta

        pts = np.zeros((n_pts, 2), dtype=np.float64)
        pts[0] = p0
        for i in range(1, n_pts):
            pts[i] = pts[i - 1] + ds * np.array([np.cos(theta[i]), np.sin(theta[i])])

        pressures = terminal_pressure + (initial_pressure - terminal_pressure) * ((1.0 - u) ** 1.2)
    else:
        # Entrance flourish: traces backward from p0, arriving at p0 with angle theta_0
        # When moving backward from p0, angle is theta_0 + pi, curving in reverse
        delta_theta = inertia_curvature * float(length_mm) * (u - 0.125 * (u ** 2))
        theta_back = theta_0 + np.pi - delta_theta

        pts_back = np.zeros((n_pts, 2), dtype=np.float64)
        pts_back[0] = p0
        for i in range(1, n_pts):
            pts_back[i] = pts_back[i - 1] + ds * np.array([np.cos(theta_back[i]), np.sin(theta_back[i])])

        # Reverse so path ends at p0
        pts = pts_back[::-1].copy()
        pressures = terminal_pressure + (initial_pressure - terminal_pressure) * (u ** 1.2)

    return pts.astype(np.float32), pressures.astype(np.float32)

