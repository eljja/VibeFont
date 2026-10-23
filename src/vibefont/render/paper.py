"""Procedural paper: base colour, fibre/tooth texture, printed rules."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..document.layout import page_size, rule_positions
from ..vibe.spec import VibeSpec

PAPER_RGB = {
    "plain": (0.985, 0.983, 0.975),
    "ruled": (0.985, 0.983, 0.975),
    "grid": (0.985, 0.983, 0.975),
    "ivory": (0.975, 0.950, 0.880),
    "kraft": (0.760, 0.630, 0.460),
}


def _noise(h: int, w: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma)
    n -= n.mean()
    return n / (n.std() + 1e-6)


def make_paper(spec: VibeSpec, dpi: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb float32 HxWx3 in 0..1, tooth float32 HxW in 0..1)."""
    w_mm, h_mm = page_size(spec)
    px = dpi / 25.4
    H, W = int(round(h_mm * px)), int(round(w_mm * px))
    base = np.array(PAPER_RGB.get(spec.paper.type, PAPER_RGB["plain"]), np.float32)

    # Multi-scale paper mottle + directional cellulose fibres
    mottle = _noise(H // 8 + 1, W // 8 + 1, 2.8, rng)
    mottle = np.kron(mottle, np.ones((8, 8), np.float32))[:H, :W]
    fine = _noise(H, W, 0.65, rng)
    micro = rng.standard_normal((H, W)).astype(np.float32)
    # Anisotropic laid-paper / cotton fiber weave
    fib_h = gaussian_filter(micro, sigma=(0.35, 2.6))
    fib_h = (fib_h - fib_h.mean()) / (fib_h.std() + 1e-6)
    fib_v = gaussian_filter(micro, sigma=(2.2, 0.35))
    fib_v = (fib_v - fib_v.mean()) / (fib_v.std() + 1e-6)

    # Subtle studio desk lighting (warm center-top, gentle corner vignette)
    yy = np.linspace(-0.8, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(-0.9, 0.9, W, dtype=np.float32)[None, :]
    vignette = 1.012 - 0.024 * (0.55 * xx**2 + 0.45 * yy**2)

    lum = vignette + 0.010 * mottle + 0.013 * fine + 0.007 * (0.65 * fib_h + 0.35 * fib_v)
    if spec.paper.type == "ivory":
        lum += 0.008 * fib_h
    elif spec.paper.type == "kraft":
        lum += 0.036 * _noise(H, W, 1.4, rng) + 0.015 * fib_h
        # Tiny dark wood-pulp flecks characteristic of unbleached kraft paper
        flecks = (micro > 2.85).astype(np.float32) * 0.08
        lum -= gaussian_filter(flecks, 0.45)

    rgb = np.clip(base[None, None, :] * lum[..., None], 0, 1)

    if spec.paper.type in ("ruled", "grid"):
        line_rgb = np.array((0.62, 0.74, 0.88), np.float32)
        thick = max(1, int(round(0.18 * px)))
        ys = rule_positions(spec)
        for y in ys:
            r = int(round(y * px))
            rgb[r:r + thick] = rgb[r:r + thick] * 0.35 + line_rgb * 0.65 * rgb[r:r + thick]
        if spec.paper.type == "grid":
            sp = spec.paper.line_spacing_mm
            for x in np.arange(spec.layout.margin_mm[3], w_mm - spec.layout.margin_mm[1] + 1e-6, sp):
                c = int(round(x * px))
                rgb[:, c:c + thick] = rgb[:, c:c + thick] * 0.35 + line_rgb * 0.65 * rgb[:, c:c + thick]
        else:  # red margin line
            c = int(round((spec.layout.margin_mm[3] - 4.0) * px))
            red = np.array((0.88, 0.45, 0.45), np.float32)
            rgb[:, c:c + thick] = rgb[:, c:c + thick] * 0.35 + red * 0.65 * rgb[:, c:c + thick]

    tooth = np.clip(0.5 + 0.28 * fine + 0.22 * micro + 0.14 * fib_h, 0.0, 1.0)
    return rgb.astype(np.float32), tooth.astype(np.float32)
