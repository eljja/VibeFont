import numpy as np
import pytest

from vibefont.generate.procedural import _Font
from vibefont.generate.ligature import (
    PlacedGlyph,
    GlyphAnchors,
    JoinInfo,
    find_anchors,
    join_strokes,
    merge_glyph_internal,
    split_delayed,
    turn_angle_deg,
    polylines_intersect,
)


def line(p0: tuple[float, float], p1: tuple[float, float], n: int) -> np.ndarray:
    return np.stack([
        np.linspace(p0[0], p1[0], n),
        np.linspace(p0[1], p1[1], n)
    ], axis=1)


def glyph(font: _Font, ch: str) -> list[np.ndarray]:
    raw = [np.asarray(s, float) for s in font.f.strokes_for_text(ch) if len(s)]
    allc = np.concatenate(raw)
    return font._norm_hershey(raw, float(allc[:, 0].min()))


def point_to_polyline(p: np.ndarray, poly: np.ndarray) -> float:
    # Minimum distance from point p to polyline poly
    dists = []
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-12:
            dists.append(float(np.linalg.norm(p - a)))
        else:
            t = float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
            proj = a + t * ab
            dists.append(float(np.linalg.norm(p - proj)))
    return min(dists) if dists else float(np.linalg.norm(p - poly[0]))


def test_find_anchors_cursive_lowercase():
    f = _Font("cursive")
    for ch in "acehilmnrsuz":
        g = glyph(f, ch)
        a = find_anchors(g)
        assert a.entry_stroke is not None, f"missing entry for {ch}"
        assert a.exit_stroke is not None, f"missing exit for {ch}"


def test_join_strokes_meets_smoothly():
    a = line((0.0, -0.56), (1.0, -0.56), 21)
    b = line((1.2, -0.56), (2.2, -0.56), 21)
    m = join_strokes(a, b, np.random.default_rng(0))
    assert m is not None
    assert len(m) == 41  # 21 + 21 - 1
    # Check max turn angle along the junction
    for i in range(3, 38):
        assert turn_angle_deg(m, i) <= 20.0


def test_join_strokes_respects_max_gap():
    a = line((0.0, -0.56), (1.0, -0.56), 21)
    b = line((1.8, -0.56), (2.8, -0.56), 21)
    assert join_strokes(a, b, np.random.default_rng(0)) is None


def test_join_strokes_leaves_glyph_body_untouched():
    # first 80% of a's arc and last 80% of b's arc must be bit-identical
    a = line((0.0, -0.56), (1.0, -0.56), 21)
    b = line((1.2, -0.56), (2.2, -0.56), 21)
    m = join_strokes(a, b, np.random.default_rng(0), zone=0.20)
    assert m is not None
    # 80% of 21 is ~16.8, first 16 points should be exact
    np.testing.assert_allclose(m[:16], a[:16], atol=1e-7)
    # last 16 points of m should match last 16 points of b
    np.testing.assert_allclose(m[-16:], b[-16:], atol=1e-7)


def test_merge_glyph_internal_m_is_one_stroke_and_shape_preserved():
    g = glyph(_Font("cursive"), "m")
    out = merge_glyph_internal(g)
    assert len(out) == 1
    for s in g:
        for p in s:
            assert point_to_polyline(p, out[0]) <= 0.02


def test_split_delayed_i_and_t():
    f = _Font("cursive")
    for ch in "it":
        g = glyph(f, ch)
        anchors = find_anchors(g)
        body, delayed = split_delayed(g, anchors)
        assert len(delayed) == 1, f"expected 1 delayed stroke for {ch}, got {len(delayed)}"
