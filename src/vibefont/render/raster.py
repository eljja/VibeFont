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


def _resample_stroke(
    xy: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    dry: np.ndarray,
    step_px: float,
):
    if len(xy) == 1:
        return xy, r, d, dry
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return xy[:1], r[:1], d[:1], dry[:1]
    h = max(0.25, step_px)
    sn = np.append(np.arange(0.0, s[-1], h), s[-1])
    return (
        np.stack([np.interp(sn, s, xy[:, 0]), np.interp(sn, s, xy[:, 1])], 1),
        np.interp(sn, s, r),
        np.interp(sn, s, d),
        np.interp(sn, s, dry),
    )


def stamp_stroke(
    D: np.ndarray,
    xy: np.ndarray,
    r: np.ndarray,
    dens: np.ndarray,
    dryness: np.ndarray | None = None,
    dry_streak: float = 0.0,
) -> None:
    """Add one stroke (pixel coords, radius px, density) into density buffer D in place."""
    if len(xy) == 0:
        return
    dry = np.zeros(len(xy), dtype=np.float64) if dryness is None else np.asarray(dryness, dtype=np.float64)
    step = 0.26 * float(np.median(r)) if len(r) else 0.45
    xy, r, dens, dry = _resample_stroke(xy, r, dens, dry, step)
    H, W = D.shape
    R = int(np.ceil(r.max())) + 1
    x0 = max(int(np.floor(xy[:, 0].min())) - R, 0)
    y0 = max(int(np.floor(xy[:, 1].min())) - R, 0)
    x1 = min(int(np.ceil(xy[:, 0].max())) + R + 1, W)
    y1 = min(int(np.ceil(xy[:, 1].max())) + R + 1, H)
    if x1 <= x0 or y1 <= y0:
        return

    # Unit normal across stroke for transverse bristle / ballpoint track physics
    if len(xy) >= 2:
        dxy = np.gradient(xy, axis=0)
        dn = np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-5)
        t_vec = dxy / dn
        n_vec = np.stack([-t_vec[:, 1], t_vec[:, 0]], axis=1)
    else:
        n_vec = np.array([[0.0, 1.0]])

    B = np.zeros((y1 - y0, x1 - x0), np.float32)
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    ox, oy = ox.ravel(), oy.ravel()
    for i in range(0, len(xy), _CHUNK):
        cx, cy = xy[i:i + _CHUNK, 0:1], xy[i:i + _CHUNK, 1:2]
        rr, dd = r[i:i + _CHUNK, None], dens[i:i + _CHUNK, None]
        ddry = dry[i:i + _CHUNK, None]
        nx, ny = n_vec[i:i + _CHUNK, 0:1], n_vec[i:i + _CHUNK, 1:2]
        ix = np.floor(cx).astype(np.int64) + ox[None]
        iy = np.floor(cy).astype(np.int64) + oy[None]
        dx = ix + 0.5 - cx
        dy = iy + 0.5 - cy
        dist = np.hypot(dx, dy)
        cov = np.clip(rr + 0.5 - dist, 0.0, 1.0)
        if dry_streak > 0.01:
            u_perp = (dx * nx + dy * ny) / np.maximum(rr, 0.45)
            bristle = 0.55 * np.cos(u_perp * 4.5 * np.pi) + 0.45 * np.cos(u_perp * 9.0 * np.pi + 0.8)
            streak_mod = np.clip(1.0 - dry_streak * ddry * (0.68 - 0.68 * bristle), 0.0, 1.18)
            cov = cov * streak_mod
        cov = (cov * dd).astype(np.float32)
        m = (cov > 0) & (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
        np.maximum.at(B, (iy[m] - y0, ix[m] - x0), cov[m])
    D[y0:y1, x0:x1] += B


def _render_seal_stamp(
    rgb: np.ndarray,
    tooth: np.ndarray,
    box_mm: tuple[float, float, float, float],
    px: float,
    rng: np.random.Generator,
) -> None:
    """Renders an authentic hand-carved vermilion stone seal (전각 낙관 인장) onto rgb."""
    x0_mm, y0_mm, x1_mm, y1_mm = box_mm
    H, W, _ = rgb.shape
    x0 = max(0, int(round(x0_mm * px)))
    y0 = max(0, int(round(y0_mm * px)))
    x1 = min(W, int(round(x1_mm * px)))
    y1 = min(H, int(round(y1_mm * px)))
    sh, sw = y1 - y0, x1 - x0
    if sh < 6 or sw < 6:
        return

    yy, xx = np.mgrid[0:sh, 0:sw]
    ny = yy / max(sh - 1, 1)
    nx = xx / max(sw - 1, 1)

    # Outer carved stone border with natural hand-chipped irregularities
    edge_noise = gaussian_filter(rng.standard_normal((sh, sw)).astype(np.float32), 1.1)
    border = (
        (nx > 0.05 + 0.03 * edge_noise)
        & (nx < 0.95 + 0.03 * edge_noise)
        & (ny > 0.05 + 0.03 * edge_noise)
        & (ny < 0.95 + 0.03 * edge_noise)
    )
    inner_void = (nx > 0.17) & (nx < 0.83) & (ny > 0.17) & (ny < 0.83)

    # Traditional seal-script (전서체) geometric strokes inside the frame
    lw = 0.095
    glyph = (
        ((np.abs(nx - 0.50) < lw * 0.65) & (ny > 0.16) & (ny < 0.84))
        | ((np.abs(ny - 0.36) < lw * 0.65) & (nx > 0.22) & (nx < 0.78))
        | ((np.abs(ny - 0.64) < lw * 0.65) & (nx > 0.24) & (nx < 0.76))
        | ((np.abs(nx - 0.30) < lw * 0.60) & (ny > 0.34) & (ny < 0.68))
        | ((np.abs(nx - 0.70) < lw * 0.60) & (ny > 0.34) & (ny < 0.68))
    )
    seal_mask = (border & (~inner_void | glyph)).astype(np.float32)
    # Stone-chip voids + porous cinnabar ink paste (인주) on paper tooth
    chip_void = (edge_noise > 1.35).astype(np.float32)
    seal_mask = np.clip(seal_mask * (1.0 - 0.75 * chip_void), 0.0, 1.0)
    seal_mask = gaussian_filter(seal_mask, 0.45)
    local_tooth = tooth[y0:y1, x0:x1]
    cinnabar_dens = seal_mask * np.clip(0.35 + 1.15 * local_tooth, 0.0, 1.25)
    cinnabar_rgb = np.array([0.76, 0.13, 0.10], dtype=np.float32)
    patch = rgb[y0:y1, x0:x1]
    rgb[y0:y1, x0:x1] = patch * (1.0 - 0.82 * cinnabar_dens[..., None]) + cinnabar_rgb * (0.82 * cinnabar_dens[..., None])


def _render_wash_accents(
    rgb: np.ndarray,
    tooth: np.ndarray,
    accents: list[dict],
    px: float,
) -> None:
    """Renders soft watercolor / sumi ink wash accents into paper fibres."""
    H, W, _ = rgb.shape
    for acc in accents:
        cx = float(acc.get("x_mm", 0.0)) * px
        cy = float(acc.get("y_mm", 0.0)) * px
        rx = max(2.0, float(acc.get("rx_mm", 4.0)) * px)
        ry = max(2.0, float(acc.get("ry_mm", 4.0)) * px)
        col = np.asarray(acc.get("color", (0.88, 0.55, 0.60)), dtype=np.float32)
        alpha = float(acc.get("opacity", 0.14))
        x0 = max(0, int(cx - 2.2 * rx))
        x1 = min(W, int(cx + 2.2 * rx))
        y0 = max(0, int(cy - 2.2 * ry))
        y1 = min(H, int(cy + 2.2 * ry))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        # Watercolor wet-edge ring profile
        w_prof = np.clip(1.0 - d2, 0.0, 1.0) ** 0.65 * (0.75 + 0.45 * np.clip(d2, 0.0, 1.0))
        w_prof *= alpha * (0.65 + 0.55 * tooth[y0:y1, x0:x1])
        rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1.0 - w_prof[..., None]) + col[None, None, :] * w_prof[..., None]


def render_page(page: Page, spec: VibeSpec, dpi: int = 300, rng: np.random.Generator | None = None) -> Image.Image:
    rng = rng or np.random.default_rng(spec.seed)
    rgb, tooth = make_paper(spec, dpi, rng)
    px = dpi / 25.4

    if getattr(page, "wash_accents", None):
        _render_wash_accents(rgb, tooth, page.wash_accents, px)

    D = np.zeros(tooth.shape, np.float32)
    grain, feather, rim_halo, porous_threshold = 0.15, 0.0, 0.0, 0.0
    for ink in page.lines:
        if len(ink) == 0:
            continue
        ps = pen_samples(ink, spec.pen, spec.style, rng)
        grain, feather = ps.grain, ps.feather
        rim_halo, porous_threshold = ps.rim_halo, ps.porous_threshold
        xy = ink.xy.astype(np.float64) * px
        min_rad = 0.18 if spec.pen.type in ("brush", "broad_nib", "fountain", "ballpoint") else 0.30
        rad = np.maximum(ps.width_mm * px / 2.0, min_rad)
        for sl in ink.stroke_slices():
            sl_dry = ps.dryness[sl] if ps.dryness is not None else None
            stamp_stroke(D, xy[sl], rad[sl], ps.dens[sl], dryness=sl_dry, dry_streak=ps.dry_streak)

    # 1. Porous dry-media interaction (pencil graphite catching only on paper tooth peaks)
    if porous_threshold > 0.01:
        # Sharp micro-peaks of paper tooth: valleys remain un-inked white paper
        effective_thresh = porous_threshold * np.clip(1.08 - 0.42 * D, 0.35, 1.15)
        valley_mask = np.clip((tooth - effective_thresh) * 3.6, 0.0, 1.45)
        D *= (1.0 - 0.85 * grain) + 0.85 * grain * valley_mask
    elif grain > 0:
        t = np.clip(0.22 + 1.25 * tooth, 0.0, 1.55)
        D *= (1.0 - grain) + grain * t

    # 2. Liquid ink capillary edge-ring halo (fountain pen / broad_nib / marker sheen rim)
    if rim_halo > 0.01:
        interior = gaussian_filter(D, 0.75)
        rim = np.clip(D - 0.76 * interior, 0.0, 0.95) * (D > 0.10)
        D = D + rim_halo * 0.75 * rim * (0.45 + 0.85 * tooth)

    # 3. Fibre feathering along paper weave
    if feather > 0:
        spread = gaussian_filter(D, 0.11 * px)
        D = D + feather * 0.65 * np.maximum(spread - D, 0) * (0.35 + 0.90 * tooth)

    # Keep pencil grain crisp; softly anti-alias wet ink edges
    blur_sigma = 0.14 if spec.pen.type == "pencil" else 0.28
    D = gaussian_filter(D, blur_sigma)

    # 4. Physical paper indentation micro-relief under pen tip (ballpoint / pencil pressure)
    if spec.pen.type in ("ballpoint", "pencil"):
        gy, gx = np.gradient(gaussian_filter(np.clip(D, 0.0, 1.2), 0.6))
        relief = np.clip(1.0 - 0.032 * (gx + gy), 0.95, 1.04)
        rgb = np.clip(rgb * relief[..., None], 0.0, 1.0)

    c = np.clip(np.asarray(spec.ink.color, np.float32), 0.02, 1.0)
    if spec.pen.type == "pencil" and spec.paper.type == "kraft":
        # Use a softer, darker 4B graphite lead on dark kraft paper for clear contrast
        c = np.minimum(c, np.array([0.11, 0.11, 0.13], dtype=np.float32))
        D *= 1.28
    out = rgb * np.exp(D[..., None] * spec.ink.opacity * np.log(c)[None, None, :])

    if getattr(page, "seal_box_mm", None) is not None:
        _render_seal_stamp(out, tooth, page.seal_box_mm, px, rng)

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
