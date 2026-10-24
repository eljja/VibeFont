import numpy as np
import pytest

from vibefont.generate.procedural import position_factor, _Font
from vibefont.generate.ligature import connect_word, pair_bias


def test_position_factor_values():
    assert position_factor(0, 5) == 0.85 and position_factor(4, 5) == 1.30 and position_factor(2, 5) == 1.0
    assert position_factor(0, 1) == 1.0


def test_word_final_glyph_looser_than_initial():
    f = _Font("cursive")
    first, last = [], []
    for s in range(100):
        g = f.place_glyphs("letters", np.random.default_rng(s), 0.95, 0.5, None)
        first.append(g[0].jitter)
        last.append(g[-1].jitter)
    assert np.mean(last) >= 1.3 * np.mean(first)


def test_connector_habit_consistent_within_pair_distinct_across_pairs():
    f = _Font("cursive")
    pairs = ["an", "er", "im", "us"]
    means, stds = [], []
    for p in pairs:
        ys = []
        for s in range(30):
            rng = np.random.default_rng(s)
            r = connect_word(f.place_glyphs(p, rng, 0.95, 0.9, None), rng, 0.97, writer_key=7)
            if r.joins:
                ys.append(r.joins[0].point[1])
        means.append(np.mean(ys))
        stds.append(np.std(ys))
    assert np.mean(stds) <= 0.5 * np.std(means)
