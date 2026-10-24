"""Fluid and anisotropic paper-fibre interaction physics for ink rendering."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def fibre_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    px_per_mm: float,
    mean_angle_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate paper fibre orientation angle field (theta) and fibre density (strength)."""
    H, W = shape
    if mean_angle_deg is None:
        mean_th = float(rng.uniform(-np.pi / 2, np.pi / 2))
    else:
        mean_th = float(np.radians(mean_angle_deg))

    sigma_px = max(0.6 * px_per_mm, 1.0)
    raw_ang = rng.standard_normal((H, W)).astype(np.float32)
    ang_noise = gaussian_filter(raw_ang, sigma_px)
    std_ang = float(ang_noise.std())
    if std_ang > 1e-6:
        ang_noise = (ang_noise / std_ang) * float(np.radians(35.0))
    theta = mean_th + ang_noise

    raw_str = rng.standard_normal((H, W)).astype(np.float32)
    str_noise = gaussian_filter(raw_str, sigma_px)
    std_str = float(str_noise.std())
    if std_str > 1e-6:
        strength = np.clip(0.5 + 0.35 * (str_noise / std_str), 0.0, 1.0)
    else:
        strength = np.full((H, W), 0.5, dtype=np.float32)

    return theta.astype(np.float32), strength.astype(np.float32)


def feather_anisotropic(
    D: np.ndarray,
    theta: np.ndarray,
    strength: np.ndarray,
    amount: float,
    px_per_mm: float,
    iters: int = 3,
) -> np.ndarray:
    """Anisotropic ink diffusion along paper fibre direction (feathering)."""
    if amount <= 1e-5:
        return D

    step_px = max(0.12 * px_per_mm, 1.0)
    H, W = D.shape
    y_coords, x_coords = np.mgrid[:H, :W].astype(np.float32)

    cos_th = np.cos(theta).astype(np.float32)
    sin_th = np.sin(theta).astype(np.float32)

    curr_D = D.copy()
    for it in range(iters):
        mult = float(it + 1)
        dx = mult * step_px * cos_th
        dy = mult * step_px * sin_th

        coords_fwd = np.array([y_coords + dy, x_coords + dx])
        coords_bwd = np.array([y_coords - dy, x_coords - dx])

        d_fwd = map_coordinates(curr_D, coords_fwd, order=1, mode="nearest")
        d_bwd = map_coordinates(curr_D, coords_bwd, order=1, mode="nearest")

        avg = 0.5 * (d_fwd + d_bwd)
        spread = amount * strength * np.maximum(avg - curr_D, 0.0)
        curr_D = curr_D + (1.20 / iters) * spread

    return curr_D
