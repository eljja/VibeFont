"""CPU rasteriser (numpy only): works on any machine, print quality at 300-600 dpi.

Model: each stroke deposits optical density with a moving round dab (max-composited
within a stroke so a single stroke never darkens itself), strokes add up (crossings get
darker, like real ink), then paper tooth / fibre feathering modulate the density and the
colour is composed with Beer-Lambert: out = paper * ink_colour ** density.

A WebGL2/WebGPU port should follow exactly the same maths (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from ..document.layout import Page
from ..vibe.spec import VibeSpec
from .paper import make_paper
from .pens import pen_samples

_CHUNK = 6000


def _resample_stroke(xy: np.ndarray, r: np.ndarray, d: np.ndarray, step_px: float):
    if len(xy) == 1:
        return xy, r, d
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return xy[:1], r[:1], d[:1]
    h = max(0.3, step_px)
    sn = np.append(np.arange(0.0, s[-1], h), s[-1])
    return (np.stack([np.interp(sn, s, xy[:, 0]), np.interp(sn, s, xy[:, 1])], 1),
            np.interp(sn, s, r), np.interp(sn, s, d))


def stamp_stroke(D: np.ndarray, xy: np.ndarray, r: np.ndarray, dens: np.ndarray) -> None:
    """Add one stroke (pixel coords, radius px, density) into density buffer D in place."""
    step = 0.3 * float(np.median(r)) if len(r) else 0.5
    xy, r, dens = _resample_stroke(xy, r, dens, step)
    H, W = D.shape
    R = int(np.ceil(r.max())) + 1
    x0 = max(int(np.floor(xy[:, 0].min())) - R, 0)
    y0 = max(int(np.floor(xy[:, 1].min())) - R, 0)
    x1 = min(int(np.ceil(xy[:, 0].max())) + R + 1, W)
    y1 = min(int(np.ceil(xy[:, 1].max())) + R + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    B = np.zeros((y1 - y0, x1 - x0), np.float32)
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    ox, oy = ox.ravel(), oy.ravel()
    for i in range(0, len(xy), _CHUNK):
        cx, cy = xy[i:i + _CHUNK, 0:1], xy[i:i + _CHUNK, 1:2]
        rr, dd = r[i:i + _CHUNK, None], dens[i:i + _CHUNK, None]
        ix = np.floor(cx).astype(np.int64) + ox[None]
        iy = np.floor(cy).astype(np.int64) + oy[None]
        dist = np.hypot(ix + 0.5 - cx, iy + 0.5 - cy)
        cov = (np.clip(rr + 0.5 - dist, 0.0, 1.0) * dd).astype(np.float32)
        m = (cov > 0) & (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
        np.maximum.at(B, (iy[m] - y0, ix[m] - x0), cov[m])
    D[y0:y1, x0:x1] += B


def render_page(page: Page, spec: VibeSpec, dpi: int = 300, rng: np.random.Generator | None = None) -> Image.Image:
    rng = rng or np.random.default_rng(spec.seed)
    rgb, tooth = make_paper(spec, dpi, rng)
    D = np.zeros(tooth.shape, np.float32)
    px = dpi / 25.4
    grain, feather = 0.15, 0.0
    for ink in page.lines:
        if len(ink) == 0:
            continue
        ps = pen_samples(ink, spec.pen, spec.style, rng)
        grain, feather = ps.grain, ps.feather
        xy = ink.xy.astype(np.float64) * px
        rad = np.maximum(ps.width_mm * px / 2.0, 0.35)
        for sl in ink.stroke_slices():
            stamp_stroke(D, xy[sl], rad[sl], ps.dens[sl])

    # paper interaction
    if grain > 0:
        t = np.clip(0.3 + 1.1 * gaussian_filter(tooth, 0.6), 0.0, 1.4)
        D *= (1.0 - grain) + grain * t
    if feather > 0:
        spread = gaussian_filter(D, 0.10 * px)
        D = D + feather * 0.6 * np.maximum(spread - D, 0) * (0.4 + tooth)
    D = gaussian_filter(D, 0.35)  # optical softness of real ink edges

    c = np.clip(np.asarray(spec.ink.color, np.float32), 0.02, 1.0)
    out = rgb * np.exp(D[..., None] * spec.ink.opacity * np.log(c)[None, None, :])
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def save_images(images: list[Image.Image], path: str | Path, dpi: int = 300) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        images[0].save(path, save_all=True, append_images=images[1:], resolution=dpi)
        return [path]
    if len(images) == 1:
        images[0].save(path, dpi=(dpi, dpi))
        return [path]
    outs = []
    for i, im in enumerate(images, 1):
        p = path.with_name(f"{path.stem}_p{i}{path.suffix}")
        im.save(p, dpi=(dpi, dpi))
        outs.append(p)
    return outs
