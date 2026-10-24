import numpy as np
import pytest

from vibefont.generate.calligraphy import _strokes_to_calligraphy_ink
from vibefont.ink import P, T, X, Y


def test_corner_dwell_hesitation():
    """Time spent per mm at a sharp corner is at least 1.8x the time spent on a straight segment."""
    # Create an L-shaped stroke with a sharp 90-degree corner: (0, 0) -> (10, 0) -> (10, 10)
    seg1 = np.stack([np.linspace(0.0, 10.0, 30), np.zeros(30)], axis=1)
    seg2 = np.stack([np.full(30, 10.0), np.linspace(0.0, 10.0, 30)], axis=1)
    corner_stroke = np.concatenate([seg1, seg2[1:]], axis=0)

    ink = _strokes_to_calligraphy_ink([(corner_stroke, 1.0, False)], text="corner", has_ko=True)
    pts = ink.points

    # Corner is near (10.0, 0.0)
    dists_to_corner = np.hypot(pts[:, X] - 10.0, pts[:, Y] - 0.0)
    corner_mask = dists_to_corner <= 1.0
    straight_mask = (pts[:, X] <= 5.0) & (pts[:, Y] <= 0.1)

    corner_indices = np.where(corner_mask)[0]
    straight_indices = np.where(straight_mask)[0]

    assert len(corner_indices) >= 2
    assert len(straight_indices) >= 2

    # Measure average speed (mm/s) or inverse time density (s/mm)
    dt_corner = float(pts[corner_indices[-1], T] - pts[corner_indices[0], T])
    dist_corner = float(np.sum(np.linalg.norm(np.diff(pts[corner_indices, :2], axis=0), axis=1)))
    dwell_rate_corner = dt_corner / max(dist_corner, 1e-3)

    dt_straight = float(pts[straight_indices[-1], T] - pts[straight_indices[0], T])
    dist_straight = float(np.sum(np.linalg.norm(np.diff(pts[straight_indices, :2], axis=0), axis=1)))
    dwell_rate_straight = dt_straight / max(dist_straight, 1e-3)

    # Calligrapher dwells at inflection to redirect brush
    assert dwell_rate_corner >= 1.8 * dwell_rate_straight


def test_downstroke_ballistic_acceleration():
    """Downward stroke accelerates significantly compared to entry, reaching brisk ballistic speed."""
    # Vertical downstroke from (10, 0) to (10, 35)
    down_stroke = np.stack([np.full(50, 10.0), np.linspace(0.0, 35.0, 50)], axis=1)
    ink = _strokes_to_calligraphy_ink([(down_stroke, 1.0, False)], text="down", has_ko=True)

    pts = ink.points
    d = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    dt = np.maximum(np.diff(pts[:, T]), 1e-4)
    v = d / dt

    v_peak = float(np.max(v))
    v_entry = float(v[1])

    # Ballistic acceleration reaches brisk speed (> 80 mm/s) and accelerates past entry
    assert v_peak > 80.0
    assert v_peak >= 1.5 * v_entry


def test_tremor_removes_perfect_bezier_collinearity():
    """Consecutive points on an extended stroke deviate slightly from a perfect line by natural micro-wobble."""
    # Pure straight diagonal line of 40mm
    diag_stroke = np.stack([np.linspace(5.0, 35.0, 60), np.linspace(5.0, 35.0, 60)], axis=1)
    ink = _strokes_to_calligraphy_ink([(diag_stroke, 0.9, False)], text="tremor", has_ko=True)

    pts = ink.points
    # Measure perpendicular distance of points to the ideal chord between start and end
    p0 = pts[0, :2]
    p1 = pts[-1, :2]
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)
    n = np.array([-chord[1], chord[0]]) / chord_len

    perp_devs = np.abs(np.dot(pts[:, :2] - p0, n))
    max_dev = float(np.max(perp_devs))

    # Natural physiological tremor must be visible (micro-wobble in [0.005, 0.035] mm),
    # not 0.0 (mechanical CAD spline) and not huge (shaky stroke)
    assert 0.005 <= max_dev <= 0.035
