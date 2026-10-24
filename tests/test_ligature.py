import numpy as np
import pytest

from vibefont.generate.procedural import _Font
from vibefont.generate.hangul import (
    join_jamo_groups,
    syllable_strokes,
    syllable_strokes_with_joins,
)
from vibefont.generate.ligature import (
    PlacedGlyph,
    GlyphAnchors,
    JoinInfo,
    ConnectResult,
    connect_word,
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


def _strokes(word: str, conn: float, seed: int):
    return _Font("cursive")._latin_word_strokes(word, rng=np.random.default_rng(seed), connectedness=conn, neatness=0.7)[0]


def test_cursive_word_pen_lifts_drop():
    hi = np.mean([len(_strokes("minimum", 0.95, s)) for s in range(5)])
    lo = np.mean([len(_strokes("minimum", 0.30, s)) for s in range(5)])
    assert hi <= 4.5 and hi <= 0.35 * lo


def test_print_style_stroke_count_unchanged():
    assert len(_strokes("minimum", 0.2, 0)) == 17


def test_delayed_strokes_drawn_last():
    st = _strokes("kitten", 0.95, 0)
    def is_del(s):
        arc = float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))
        return s[:, 1].max() < -0.9 and arc < 0.8
    is_del_list = [is_del(s) for s in st]
    assert is_del_list[-3:] == [True, True, True] and not any(is_del_list[:-3])


def test_high_join_after_o_and_baseline_join_after_a():
    f = _Font("cursive")
    for seed in range(5):
        rng = np.random.default_rng(seed)
        r_on = connect_word(f.place_glyphs("on", rng, 0.95, 0.7, None), rng, 0.97)
        r_an = connect_word(f.place_glyphs("an", rng, 0.95, 0.7, None), rng, 0.97)
        if r_on.joins:
            assert r_on.joins[0].point[1] <= -0.85
        if r_an.joins:
            assert -0.75 <= r_an.joins[0].point[1] <= -0.40


def test_join_turn_angles_bounded_over_corpus():
    f = _Font("cursive")
    words = "the quick brown fox jumps over the lazy dog".split()
    turns = []
    for seed in range(5):
        for w in words:
            rng = np.random.default_rng(seed)
            r = connect_word(f.place_glyphs(w, rng, 0.95, 0.7, None), rng, 0.97)
            for j in r.joins:
                turns.append(j.turn_deg)
    assert len(turns) > 0
    assert max(turns) <= 75.0 and np.mean(turns) <= 40.0


def test_edge_words_capitals_punctuation_single_letters():
    for w in ["Hello,", "don't", "A4", "a", "I", ""]:
        st, width = _Font("cursive")._latin_word_strokes(w, rng=np.random.default_rng(1), connectedness=0.95, neatness=0.7)
        assert width > 0
    r = connect_word(_Font("cursive").place_glyphs("Hello,", np.random.default_rng(1), 0.95, 0.7, None), np.random.default_rng(1), 0.97)
    assert all(j.prev_ch.islower() and j.next_ch.islower() for j in r.joins)


def test_join_jamo_groups_never_joins_inside_a_group_beyond_touching():
    g1 = [line((0.0, 0.0), (1.0, 0.0), 11), line((0.0, 0.5), (1.0, 0.5), 11)]   # ㄷ-like: 2 strokes, gap 0.5
    g2 = [line((1.1, 0.5), (1.1, -1.0), 11)]
    out, joins = join_jamo_groups([g1, g2], 0.95, np.random.default_rng(0))
    assert len(out) >= 2 and len(joins) <= 1


def test_hangul_cursive_reduces_pen_lifts():
    txt = "가나다라마바사아자차"
    def mean_strokes(conn):
        return np.mean([len(syllable_strokes(c, rng=np.random.default_rng(i), connectedness=conn, neatness=0.7)) for i in range(10) for c in txt])
    assert mean_strokes(0.9) <= 0.85 * mean_strokes(0.3)


def test_hangul_connectors_never_cross_other_strokes():
    for seed in range(20):
        for ch in "강물이흐르는밤":
            st, joins = syllable_strokes_with_joins(ch, np.random.default_rng(seed), 0.95, 0.7, None)
            for j in joins:
                if j.path is not None:
                    # Connector path must not intersect any other stroke in syllable
                    assert not any(polylines_intersect(j.path, s) for s in st if not np.array_equal(s, j.path))


def test_mixed_script_word_no_join_across_boundary():
    st, _ = _Font("cursive").word("VibeFont한글", rng=np.random.default_rng(0), connectedness=0.95, neatness=0.7)
    # Confirm strokes are separated across Latin/Hangul boundary
    assert len(st) > 1
    # Check that no single stroke stretches across the transition between Latin (x ~ 6) and Hangul
    latin_end_x = 5.5
    hangul_start_x = 6.2
    for s in st:
        min_sx, max_sx = s[:, 0].min(), s[:, 0].max()
        assert not (min_sx < latin_end_x and max_sx > hangul_start_x), "stroke crosses script boundary"
