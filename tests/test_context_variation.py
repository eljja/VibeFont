import numpy as np
import pytest

from vibefont.generate.procedural import position_factor, _Font


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
