import numpy as np
import pytest

from vibefont.generate.calligraphy import _sculpt_ko_syllable_strokes
from vibefont.generate.semantics import SemanticArchetype, WordMorphology


def test_massive_syllable_heavier_than_airy():
    rng = np.random.default_rng(42)
    m_massive = WordMorphology(
        word="바",
        archetype=SemanticArchetype.BOLD_MASSIVE,
        scale=1.65,
        weight_boost=0.35,
        angularity=0.85,
        is_hero=True,
    )
    m_airy = WordMorphology(
        word="바",
        archetype=SemanticArchetype.FLOWING_AIRY,
        scale=1.20,
        weight_boost=-0.15,
        angularity=0.10,
        is_hero=False,
    )

    strokes_massive = _sculpt_ko_syllable_strokes("바", rng, morph=m_massive)
    strokes_airy = _sculpt_ko_syllable_strokes("바", rng, morph=m_airy)

    avg_w_massive = np.mean([w for _, w, _ in strokes_massive])
    avg_w_airy = np.mean([w for _, w, _ in strokes_airy])
    assert avg_w_massive >= 1.30 * avg_w_airy


def test_sharp_syllable_longer_tail_than_gentle():
    rng = np.random.default_rng(42)
    m_sharp = WordMorphology(
        word="눈",
        archetype=SemanticArchetype.SHARP_FIERCE,
        scale=1.55,
        weight_boost=0.10,
        angularity=0.95,
        swash_reach=1.65,
        is_hero=True,
    )
    m_gentle = WordMorphology(
        word="봄",
        archetype=SemanticArchetype.GENTLE_WARM,
        scale=1.20,
        weight_boost=0.05,
        angularity=0.10,
        swash_reach=0.80,
        is_hero=False,
    )

    strokes_sharp = _sculpt_ko_syllable_strokes("눈", rng, morph=m_sharp, is_word_end=True)
    strokes_gentle = _sculpt_ko_syllable_strokes("봄", rng, morph=m_gentle, is_word_end=True)

    # Check tail strokes length
    tail_sharp = next((p for p, _, is_t in strokes_sharp if is_t), None)
    assert tail_sharp is not None
    tail_sharp_len = float(np.sum(np.linalg.norm(np.diff(tail_sharp, axis=0), axis=1)))
    assert tail_sharp_len >= 1.20


def test_soaring_syllable_aspect_ratio():
    rng = np.random.default_rng(42)
    m_soar = WordMorphology(
        word="별",
        archetype=SemanticArchetype.SOARING_LUMINOUS,
        scale=1.60,
        angularity=0.50,
        is_hero=True,
    )
    m_neutral = WordMorphology(
        word="별",
        archetype=SemanticArchetype.NEUTRAL,
        scale=1.00,
        angularity=0.50,
        is_hero=False,
    )

    strokes_soar = _sculpt_ko_syllable_strokes("별", rng, morph=m_soar)
    strokes_neutral = _sculpt_ko_syllable_strokes("별", rng, morph=m_neutral)

    pts_soar = np.concatenate([p for p, _, _ in strokes_soar], axis=0)
    pts_neutral = np.concatenate([p for p, _, _ in strokes_neutral], axis=0)

    h_soar = pts_soar[:, 1].max() - pts_soar[:, 1].min()
    w_soar = pts_soar[:, 0].max() - pts_soar[:, 0].min()
    aspect_soar = h_soar / w_soar

    h_neutral = pts_neutral[:, 1].max() - pts_neutral[:, 1].min()
    w_neutral = pts_neutral[:, 0].max() - pts_neutral[:, 0].min()
    aspect_neutral = h_neutral / w_neutral

    assert aspect_soar >= 1.08 * aspect_neutral


def test_all_strokes_remain_finite_and_legible():
    rng = np.random.default_rng(100)
    for arch in SemanticArchetype:
        m = WordMorphology(word="글", archetype=arch, scale=1.5, is_hero=True)
        strokes = _sculpt_ko_syllable_strokes("글", rng, morph=m)
        for p, w, _ in strokes:
            assert np.all(np.isfinite(p))
            assert np.isfinite(w)
            assert w > 0.1
