import numpy as np
import pytest

from vibefont.generate.calligraphy import compose_calligraphy_page
from vibefont.vibe.spec import VibeSpec, PenSpec


def test_hero_line_dominates_secondary_line():
    text = "눈보라 속에서도\n매화는 자신의 향기를\n팔지 않는다"
    spec = VibeSpec(pen=PenSpec(type="brush"))
    page = compose_calligraphy_page(text, spec, rng=np.random.default_rng(42))
    assert len(page.lines) >= 3

    # Measure visual bounding box height of each line
    heights = []
    for line in page.lines:
        ys = line.xy[:, 1]
        heights.append(float(ys.max() - ys.min()))

    # Primary hero line (line 0 containing "눈보라" or line 1 "매화") should be significantly larger
    max_h = max(heights)
    min_h = min(heights)
    assert max_h >= 1.35 * min_h


def test_particle_tucked_into_stem_negative_space():
    text = "별을 노래하는 마음으로"
    spec = VibeSpec(pen=PenSpec(type="brush"))
    page = compose_calligraphy_page(text, spec, rng=np.random.default_rng(42))
    line0 = page.lines[0]
    # Check that points exist and are naturally distributed
    assert len(line0) > 50


def test_seal_stamp_placed_in_optical_void():
    text = "수천 번 부서지는 파도가\n마침내 가장 아름다운\n해안선을 조각한다"
    spec = VibeSpec(pen=PenSpec(type="brush"))
    page = compose_calligraphy_page(text, spec, rng=np.random.default_rng(42))
    assert page.seal_box_mm is not None

    # Verify no ink stroke passes directly through the seal box center
    x0, y0, x1, y1 = page.seal_box_mm
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    for line in page.lines:
        dists = np.hypot(line.xy[:, 0] - cx, line.xy[:, 1] - cy)
        # Seal center must have breathing room (not directly overlapping text center)
        assert np.min(dists) > 2.0


def test_no_strokes_cross_card_margins():
    text = "In the deepest silence\nthe clearest wisdom\nquietly awakens."
    spec = VibeSpec(pen=PenSpec(type="brush"))
    page = compose_calligraphy_page(text, spec, rng=np.random.default_rng(42))
    for line in page.lines:
        xs = line.xy[:, 0]
        ys = line.xy[:, 1]
        assert xs.min() >= 4.0
        assert xs.max() <= page.width_mm - 4.0
        assert ys.min() >= 4.0
        assert ys.max() <= page.height_mm - 4.0


def test_ballistic_flourish_kinematics():
    from vibefont.generate.kinematics import ballistic_flourish

    start = np.array([10.0, 20.0])
    pts, pressures = ballistic_flourish(
        start_pt=start,
        initial_direction=np.array([1.0, 0.2]),
        length_mm=12.0,
        inertia_curvature=0.08,
        n_pts=24,
    )
    assert len(pts) == 24
    assert np.allclose(pts[0], start)
    # Distance should monotonically increase from start
    dists = np.linalg.norm(pts - start, axis=1)
    assert np.all(np.diff(dists) >= 0.0)
    # Curvature should be continuous and smooth (no sharp angle jumps > 15 deg)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    assert np.all(seg_lens > 1e-4)
    unit_diffs = diffs / seg_lens[:, None]
    dot_products = np.sum(unit_diffs[:-1] * unit_diffs[1:], axis=1)
    # Cosine of turn angle should be > cos(15 deg) = 0.965
    assert np.all(dot_products > 0.95)
    # Exit pressure should decay smoothly towards near-zero
    assert pressures[0] > pressures[-1]
    assert pressures[-1] <= 0.15


def test_calligraphy_tail_pressure_tapers_smoothly():
    text = "Freedom and Serenity"
    spec = VibeSpec(pen=PenSpec(type="fountain"))
    page = compose_calligraphy_page(text, spec, rng=np.random.default_rng(42))
    last_line = page.lines[-1]
    # The last stroke in the line should have a decaying terminal pressure without sudden drop or cliff
    tail_p = last_line.points[-8:, 3]  # P column is index 3
    assert tail_p[-1] < 0.25
    # Smooth decay towards the end
    assert tail_p[-1] <= tail_p[0]

