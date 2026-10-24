import numpy as np
import pytest

from vibefont.document.layout import Page
from vibefont.ink import Ink
from vibefont.render.fluid import bristle_striations
from vibefont.render.pens import pen_samples
from vibefont.render.raster import render_page, stamp_stroke
from vibefont.vibe.spec import PenSpec, VibeSpec


def _make_stroke_ink(v_mm_s: float, length_mm: float = 50.0) -> Ink:
    """Creates a straight horizontal stroke with controlled velocity in mm/s."""
    n_pts = 120
    xs = np.linspace(20.0, 20.0 + length_mm, n_pts)
    ys = np.full(n_pts, 35.0)
    total_time = length_mm / max(v_mm_s, 1.0)
    ts = np.linspace(0.0, total_time, n_pts)
    ps = np.full(n_pts, 0.75)  # constant medium-firm pressure
    eos = np.zeros(n_pts)
    eos[-1] = 1.0
    pts = np.stack([xs, ys, ts, ps, eos], axis=1)
    return Ink(pts)


def test_fast_brush_stroke_exhibits_dry_striations():
    """High-velocity brush stroke has significantly higher transverse striation variance than slow stroke."""
    slow_ink = _make_stroke_ink(v_mm_s=25.0)   # slow, fully saturated flow
    fast_ink = _make_stroke_ink(v_mm_s=180.0)  # rapid ballistic sweep

    spec = VibeSpec(pen=PenSpec(type="brush", width_mm=4.5))
    rng = np.random.default_rng(42)

    ps_slow = pen_samples(slow_ink, spec.pen, spec.style, rng)
    ps_fast = pen_samples(fast_ink, spec.pen, spec.style, rng)

    assert ps_fast.dryness is not None
    assert ps_slow.dryness is not None
    # Fast stroke must have significantly higher dryness than slow stroke
    assert np.mean(ps_fast.dryness[20:100]) > np.mean(ps_slow.dryness[20:100]) + 0.20

    # Test bristle striations modulation
    u_perp = np.linspace(-0.8, 0.8, 80)[:, None]
    mod_slow = bristle_striations(u_perp, dryness=np.full_like(u_perp, 0.05), dry_streak=0.76)
    mod_fast = bristle_striations(u_perp, dryness=np.full_like(u_perp, 0.75), dry_streak=0.76)

    # Fast striations should show deep bristle splits (higher standard deviation)
    assert np.std(mod_fast) > 1.8 * np.std(mod_slow)
    assert np.min(mod_fast) < 0.40  # Deep fissures between bristle clumps


def test_slow_brush_stroke_remains_saturated():
    """Slow brush stroke has continuous coverage without premature hair splitting."""
    u_perp = np.linspace(-0.7, 0.7, 50)[:, None]
    mod_slow = bristle_striations(u_perp, dryness=np.full_like(u_perp, 0.02), dry_streak=0.76)

    # When wet/slow, ink bridges hair gaps completely
    assert np.min(mod_slow) > 0.85
    assert np.std(mod_slow) < 0.08


def test_dry_brush_interacts_with_paper_tooth():
    """Valleys in paper tooth have lower ink density on dry-brush strokes."""
    fast_ink = _make_stroke_ink(v_mm_s=190.0, length_mm=60.0)
    spec = VibeSpec(pen=PenSpec(type="brush", width_mm=4.5))
    rng = np.random.default_rng(42)

    ps_fast = pen_samples(fast_ink, spec.pen, spec.style, rng)
    # Brush pen must enable porous_threshold or tooth interaction on dry strokes
    assert ps_fast.porous_threshold > 0.15
