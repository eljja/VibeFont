"""Vector export: each stroke becomes a filled variable-width outline (no paper texture)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..document.layout import Page
from ..vibe.spec import VibeSpec
from .pens import pen_samples


def _outline(xy: np.ndarray, w: np.ndarray) -> str:
    if len(xy) == 1:
        x, y = xy[0]
        r = w[0] / 2
        return f"M{x - r:.3f},{y:.3f}a{r:.3f},{r:.3f} 0 1,0 {2 * r:.3f},0a{r:.3f},{r:.3f} 0 1,0 {-2 * r:.3f},0Z"
    d = np.gradient(xy, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    left = xy + n * (w[:, None] / 2)
    right = xy - n * (w[:, None] / 2)
    pts = np.concatenate([left, right[::-1]])
    path = "M" + "L".join(f"{x:.3f},{y:.3f}" for x, y in pts) + "Z"
    # round caps
    for (x, y), ww in ((xy[0], w[0]), (xy[-1], w[-1])):
        r = ww / 2
        path += f"M{x - r:.3f},{y:.3f}a{r:.3f},{r:.3f} 0 1,0 {2 * r:.3f},0a{r:.3f},{r:.3f} 0 1,0 {-2 * r:.3f},0Z"
    return path


def page_to_svg(page: Page, spec: VibeSpec, rng: np.random.Generator | None = None) -> str:
    rng = rng or np.random.default_rng(spec.seed)
    r, g, b = (int(255 * c) for c in spec.ink.color)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{page.width_mm}mm" height="{page.height_mm}mm" '
             f'viewBox="0 0 {page.width_mm} {page.height_mm}">',
             f'<rect width="100%" height="100%" fill="white"/><g fill="rgb({r},{g},{b})" fill-rule="nonzero">']
    for ink in page.lines:
        if len(ink) == 0:
            continue
        ps = pen_samples(ink, spec.pen, spec.style, rng)
        for sl in ink.stroke_slices():
            op = float(np.clip(np.mean(ps.dens[sl]), 0.15, 1.0))
            parts.append(f'<path d="{_outline(ink.xy[sl].astype(np.float64), ps.width_mm[sl])}" fill-opacity="{op:.2f}"/>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def save_svg(pages: list[Page], spec: VibeSpec, path: str | Path) -> list[Path]:
    path = Path(path)
    outs = []
    for i, pg in enumerate(pages, 1):
        p = path if len(pages) == 1 else path.with_name(f"{path.stem}_p{i}{path.suffix}")
        p.write_text(page_to_svg(pg, spec), encoding="utf-8")
        outs.append(p)
    return outs
