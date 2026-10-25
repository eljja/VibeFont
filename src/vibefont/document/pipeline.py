"""text + VibeSpec + generator -> pages of placed ink (millimetres).

Line breaking is "estimate, generate, verify": break greedily with the generator's width
estimate, generate the line, and if the real line overflows push the last word to the next
line and regenerate (cheap: lines are short). Long single words are squeezed horizontally.
"""

from __future__ import annotations

import numpy as np

from ..descriptors import slant_deg
from ..generate.base import LineRequest, StrokeGenerator
from ..generate.glyph_guard import MissingGlyphError, validate_text_glyphs
from ..ink import Ink
from ..vibe.spec import VibeSpec
from .drift import DocumentDrift, LineState
from .layout import Page, baselines_mm, line_pitch_mm, page_size

CONTEXT_LINES = 2  # how many previous lines are handed to the generator for priming


def _place(ink: Ink, spec: VibeSpec, st: LineState, x_mm: float, y_mm: float,
           slant_corr_weight: float) -> Ink:
    """Normalised line -> page millimetres."""
    if slant_corr_weight > 0 and len(ink) > 20:
        resid = float(np.clip(st.slant_deg - slant_deg(ink), -10, 10))
        ink = ink.shear_slant(slant_corr_weight * resid)
    xmin = ink.bbox()[0] if len(ink) else 0.0
    ink = ink.translate(-xmin, 0.0)
    size = spec.style.size_mm * st.size_scale
    ink = ink.scale(size)
    tilt = st.tilt_deg * (0.20 if spec.paper.type in ("ruled", "grid") else 1.0)
    ink = ink.rotate(-tilt)  # + tilt = rising to the right (y down -> negative rotation)
    y_off = st.baseline_offset_mm * (0.35 if spec.paper.type in ("ruled", "grid") else 1.0)
    return ink.translate(x_mm, y_mm + y_off)


def write_document(text: str, spec: VibeSpec, gen: StrokeGenerator, rng: np.random.Generator | None = None,
                   max_pages: int = 20) -> list[Page]:
    text = text.replace("\\n", "\n")
    text = validate_text_glyphs(text, allow_hangul=True)
    rng = rng or np.random.default_rng(spec.seed)
    w_mm, h_mm = page_size(spec)
    top, right, bottom, left = spec.layout.margin_mm
    usable = w_mm - left - right
    bases = baselines_mm(spec)
    est_lines = max(1, int(len(text) * spec.style.size_mm * 0.9 / usable) + text.count("\n"))
    drift = DocumentDrift(spec.style, rng, expected_lines=est_lines)
    slant_w = 0.0 if getattr(gen, "name", "") in ("procedural", "model") else 0.7

    pages = [Page(w_mm, h_mm)]
    line_idx = 0
    context: list[Ink] = []

    def x_units(t: str, scale: float) -> float:
        return gen.estimate_width(t, spec.style) * spec.style.size_mm * scale

    for para in text.split("\n"):
        words = para.split()
        if not words:  # empty paragraph -> blank line
            line_idx += 1
            continue
        while words:
            st = drift.next_line()
            avail = usable - max(0.0, st.margin_offset_mm)
            n = 1
            while n < len(words) and x_units(" ".join(words[: n + 1]), st.size_scale) <= avail:
                n += 1
            for _attempt in range(8):
                line_text = " ".join(words[:n])
                req = LineRequest(line_text, spec.style, st, rng, context[-CONTEXT_LINES:])
                ink = gen.generate_line(req)
                width_mm = (ink.bbox()[2] - ink.bbox()[0]) * spec.style.size_mm * st.size_scale
                if width_mm <= avail * 1.02 or n == 1:
                    break
                # Step down proportionally so even long lines converge in 1-2 tries without horizontal squishing
                step_down = max(1, int(n - n * (avail / max(width_mm, 1e-3))))
                n = max(1, n - step_down)
            if width_mm > avail and len(ink):
                ink = ink.scale(avail / width_mm, 1.0)
            if line_idx >= len(bases):
                if len(pages) >= max_pages:
                    return pages
                pages.append(Page(w_mm, h_mm))
                line_idx = 0
            placed_line = _place(ink, spec, st, left + st.margin_offset_mm, bases[line_idx], slant_w)
            if spec.layout.align == "center" and len(placed_line):
                lx0, _, lx1, _ = placed_line.bbox()
                lw = lx1 - lx0
                target_x = 0.5 * (w_mm - lw) + 0.35 * st.margin_offset_mm
                placed_line = placed_line.translate(target_x - lx0, 0.0)
            pages[-1].lines.append(placed_line)
            context.append(ink)
            words = words[n:]
            line_idx += 1

    if spec.layout.v_align == "center":
        for page in pages:
            non_empty = [ln for ln in page.lines if len(ln)]
            if non_empty:
                ymin = min(ln.bbox()[1] for ln in non_empty)
                ymax = max(ln.bbox()[3] for ln in non_empty)
                dy = 0.5 * page.height_mm - 0.5 * (ymin + ymax)
                page.lines = [ln.translate(0.0, dy) for ln in page.lines]

    # Strict post-generation word verification: verify no words were dropped
    placed_words = [w for p in pages for ln in p.lines for w in getattr(ln, "text", "").split()]
    expected_words = text.split()
    if placed_words != expected_words:
        raise MissingGlyphError(
            f"Document word integrity violation: expected {len(expected_words)} words, placed {len(placed_words)}.",
            text=text,
        )

    return pages


__all__ = ["write_document", "line_pitch_mm"]
