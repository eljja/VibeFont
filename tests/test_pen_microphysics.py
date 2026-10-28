"""Tests for physical pen micro-footprints, tine bifurcation, graphite tooth shear, and ballpoint paste ridges."""

import numpy as np
import pytest

from vibefont.ink import EOS, NCOLS, P, T, X, Y, Ink
from vibefont.render.pens import pen_samples
from vibefont.render.raster import stamp_stroke
from vibefont.vibe.spec import PenSpec, StyleSpec


def _make_straight_stroke(n: int = 50, length_mm: float = 20.0, pressure: float = 0.8) -> Ink:
    pts = np.zeros((n, NCOLS), dtype=np.float32)
    pts[:, X] = np.linspace(2.0, 2.0 + length_mm, n)
    pts[:, Y] = 10.0
    pts[:, T] = np.linspace(0.0, 0.5, n)
    pts[:, P] = pressure
    pts[-1, EOS] = 1.0
    return Ink(pts)


def test_fountain_pen_has_tine_split_under_flex_pressure():
    ink = _make_straight_stroke(pressure=0.9)
    pen = PenSpec(type="fountain", width_mm=0.7)
    style = StyleSpec()
    ps = pen_samples(ink, pen, style, np.random.default_rng(42))

    assert hasattr(ps, "tine_split")
    assert ps.tine_split > 0.35, "Flex fountain pen must exhibit physical nib tine splitting"


def test_pencil_has_graphite_shear_on_paper_tooth():
    ink = _make_straight_stroke(pressure=0.6)
    pen = PenSpec(type="pencil", width_mm=0.5)
    style = StyleSpec()
    ps = pen_samples(ink, pen, style, np.random.default_rng(42))

    assert hasattr(ps, "graphite_shear")
    assert ps.graphite_shear > 0.5, "Pencil must exhibit graphite particulate shear on paper tooth"


def test_ballpoint_has_ball_rim_paste_parameter():
    ink = _make_straight_stroke(pressure=0.7)
    pen = PenSpec(type="ballpoint", width_mm=0.5)
    style = StyleSpec()
    ps = pen_samples(ink, pen, style, np.random.default_rng(42))

    assert hasattr(ps, "ball_rim_paste")
    assert ps.ball_rim_paste > 0.3, "Ballpoint must exhibit ball rim paste ridge accumulation"


def test_stamp_stroke_tine_bifurcation_creates_dual_rail_darkening():
    """Under flex pressure with tine_split > 0, transverse cross-section must show dual outer peaks (rail tracks)."""
    # 50 points horizontal line at y=50 in pixel coords
    xy = np.stack([np.linspace(20, 80, 60), np.full(60, 50.0)], axis=1)
    r = np.full(60, 8.0)  # 8 pixel radius
    dens = np.full(60, 1.0)
    D = np.zeros((100, 100), dtype=np.float32)

    res = stamp_stroke(D, xy, r, dens, tine_split=0.65)
    assert res is not None
    y0, x0, B = res

    # Look at vertical cross section through mid x (around x=50, which is col index 30 in B)
    mid_col = B[:, 30]
    # The center of the stroke is at peak index
    # With tine split, the profile should have two peaks (bifurcation) or outer edges should be darker than center
    center_idx = np.argmax(mid_col > 0.1) + int(np.sum(mid_col > 0.1) // 2)
    center_val = mid_col[center_idx]
    upper_rail = mid_col[center_idx - 3]
    lower_rail = mid_col[center_idx + 3]

    assert upper_rail > center_val or lower_rail > center_val, "Tine split must produce dual-rail darker edges than center"


def test_stamp_stroke_graphite_particulate_tooth_adhesion():
    """Graphite on textured tooth must have micro-voids in tooth valleys compared to uniform disk stamping."""
    xy = np.stack([np.linspace(20, 80, 60), np.full(60, 50.0)], axis=1)
    r = np.full(60, 6.0)
    dens = np.full(60, 0.8)

    # Create paper tooth heightmap (peaks and valleys)
    yy, xx = np.mgrid[0:100, 0:100]
    tooth = (np.sin(xx * 0.8) * np.cos(yy * 0.8) * 0.5 + 0.5).astype(np.float32)

    D_smooth = np.zeros((100, 100), dtype=np.float32)
    stamp_stroke(D_smooth, xy, r, dens, graphite_shear=0.0)

    D_graphite = np.zeros((100, 100), dtype=np.float32)
    stamp_stroke(D_graphite, xy, r, dens, graphite_shear=0.8, tooth=tooth)

    # Graphite deposition should have higher standard deviation (particulate texture) than smooth
    mask = D_smooth > 0.2
    assert np.std(D_graphite[mask]) > 1.4 * np.std(D_smooth[mask]), "Graphite shear must create distinct microscopic particulate grain"
