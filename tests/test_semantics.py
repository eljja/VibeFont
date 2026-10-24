import pytest

from vibefont.generate.semantics import (
    SemanticArchetype,
    WordMorphology,
    analyze_semantics,
    classify_word_archetype,
)


def test_korean_semantic_archetypes():
    assert classify_word_archetype("눈보라") == SemanticArchetype.SHARP_FIERCE
    assert classify_word_archetype("바위") == SemanticArchetype.BOLD_MASSIVE
    assert classify_word_archetype("봄") == SemanticArchetype.GENTLE_WARM
    assert classify_word_archetype("별") == SemanticArchetype.SOARING_LUMINOUS
    assert classify_word_archetype("파도") == SemanticArchetype.FLOWING_AIRY


def test_english_semantic_archetypes():
    assert classify_word_archetype("storm") == SemanticArchetype.SHARP_FIERCE
    assert classify_word_archetype("oak") == SemanticArchetype.BOLD_MASSIVE
    assert classify_word_archetype("silence") == SemanticArchetype.FLOWING_AIRY
    assert classify_word_archetype("stars") == SemanticArchetype.SOARING_LUMINOUS
    assert classify_word_archetype("love") == SemanticArchetype.GENTLE_WARM


def test_hero_particle_scale_contrast():
    lines = analyze_semantics("눈보라 속에서도\n매화는 자신의 향기를\n팔지 않는다")
    assert len(lines) == 3
    # Check first line: "눈보라" (hero stem) vs "속에서도" (particle/function word)
    morphs_line1 = lines[0]
    hero = next(m for m in morphs_line1 if m.word == "눈보라")
    particle = next(m for m in morphs_line1 if "속에서도" in m.word or m.is_particle)
    assert hero.scale >= 1.35 * particle.scale
    assert hero.archetype == SemanticArchetype.SHARP_FIERCE


def test_morphology_clipping_bounds():
    lines = analyze_semantics("거대한 바위처럼 무거운 침묵이 흐른다")
    all_morphs = [m for line in lines for m in line]
    for m in all_morphs:
        assert 0.60 <= m.scale <= 2.20
        assert -0.30 <= m.weight_boost <= 0.50
        assert 0.0 <= m.angularity <= 1.0
        assert -10.0 <= m.slant_drift_deg <= 10.0
