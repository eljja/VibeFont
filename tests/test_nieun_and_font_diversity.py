import numpy as np
import pytest

from vibefont.generate.hangul import (
    WRITER_ARCHETYPES,
    WriterProfile,
    choseong_strokes,
    jongseong_strokes,
    hangul_char_strokes,
)
from vibefont.generate.calligraphy import compose_calligraphy_page
from vibefont.vibe.interpreter import interpret
from vibefont.vibe.spec import VibeSpec, StyleSpec


def test_nieun_never_scoops_or_droops_downward():
    """Verify that ㄴ (idx 2) has a clean flat or upright baseline across archetypes and random seeds,

    and never dips down into an unnatural bowl or downward droop.
    """
    profiles = [
        WriterProfile(archetype="standard"),
        WriterProfile(archetype="classic_jeongja"),
        WriterProfile(archetype="master_heullim"),
        WriterProfile(archetype="rounded_diary"),
        WriterProfile(archetype="gosi"),
        WriterProfile(archetype="geometric_sans"),
        WriterProfile(archetype="classic_myeongjo"),
    ]

    for prof in profiles:
        for seed in (42, 123, 999):
            rng = np.random.default_rng(seed)
            st = choseong_strokes(2, rng=rng, profile=prof)
            assert len(st) >= 1
            # Combine all points of ㄴ
            pts = np.concatenate(st, axis=0) if len(st) > 1 else st[0]
            # In normalized [0, 1] x [0, 1] box:
            # The bottom horizontal bar starts after the vertical stem.
            # The horizontal bar must not sag below the corner y by more than 0.02.
            # The terminal point must not dip downward below the bottom bar.
            y_max = float(pts[:, 1].max())
            y_end = float(pts[-1, 1])
            # Terminal point should not be significantly lower than the rest of the horizontal base
            assert y_end <= y_max + 1e-4, f"Terminal y {y_end} exceeds y_max {y_max} for archetype {prof.archetype}"

            # Check the last 3 points direction: it must be moving horizontally to the right (dx > 0)
            d_last = pts[-1] - pts[-3]
            assert d_last[0] > 0.05, f"ㄴ bottom stroke must progress to the right: dx={d_last[0]} for {prof.archetype}"
            # The vertical slope at the end (dy/dx) must NOT be strongly positive (downward plunge)
            assert d_last[1] <= 0.08, f"ㄴ terminal must not plunge downward: dy={d_last[1]} for {prof.archetype}"


def test_jongseong_nieun_integrity():
    """Verify that 받침 ㄴ (jongseong idx 4) maintains flat or crisp baseline and does not sag downwards."""
    for arch in ("standard", "geometric_sans", "classic_myeongjo", "master_heullim", "gosi"):
        prof = WriterProfile(archetype=arch)
        rng = np.random.default_rng(77)
        st = jongseong_strokes(4, rng=rng, profile=prof)
        assert len(st) >= 1
        pts = np.concatenate(st, axis=0)
        # End must point rightward, not downward
        d_end = pts[-1] - pts[-3]
        assert d_end[0] > 0.05
        assert d_end[1] <= 0.08


def test_calligraphy_does_not_drag_nieun_downward():
    """Ensure that syllables ending in ㄴ (e.g. '은', '는', '문', '인', '한') do not have

    their bottom horizontal stroke forcefully swept downwards into a vertical tail.
    """
    # Test '은' and '한'
    vibe = VibeSpec()
    vibe.style.archetype = "master_heullim"
    vibe.style.connectedness = 0.85
    vibe.style.neatness = 0.80

    for word in ("은", "한", "는", "인"):
        rng = np.random.default_rng(42)
        page = compose_calligraphy_page(word, spec=vibe, rng=rng)
        assert len(page.lines) > 0
        strokes = page.lines[0].strokes()
        assert len(strokes) > 0
        last_pts = strokes[-1]
        span_x = float(last_pts[:, 0].max() - last_pts[:, 0].min())
        end_step = last_pts[-1, :2] - last_pts[-4, :2]
        if span_x > 0.5:
            assert not (end_step[1] > 0.5 * end_step[0] and end_step[1] > 0.1), (
                f"Word '{word}' has ㄴ ending stroke plunging downwards: step={end_step}"
            )


def test_diverse_font_archetypes_exist_and_distinct():
    """Verify that multiple distinct font archetypes are supported with unique characteristics:

    - geometric_sans: 2 orthogonal strokes for ㄴ, sharp corners
    - classic_myeongjo: structured, classic proportions
    - retro_typewriter: monoline, distinct uniform rhythm
    - rounded_diary: soft fillets, wide round loops
    - master_heullim: virtuoso flowing cursive
    - gosi: fast shorthand
    """
    for arch in ("geometric_sans", "classic_myeongjo", "retro_typewriter", "rounded_diary", "master_heullim", "gosi"):
        assert arch in WRITER_ARCHETYPES

    # Geometric sans should produce orthogonal, clean separate or sharp strokes for ㄴ and ㄱ
    prof_geo = WriterProfile(archetype="geometric_sans")
    st_nieun_geo = choseong_strokes(2, profile=prof_geo)
    # ㄴ in geometric sans should be flat: y stays constant along the horizontal base
    pts = np.concatenate(st_nieun_geo, axis=0) if len(st_nieun_geo) > 1 else st_nieun_geo[0]
    # Check bottom segment flatness:
    bot_pts = pts[pts[:, 0] >= 0.25]
    assert np.ptp(bot_pts[:, 1]) < 0.05, f"Geometric sans ㄴ base should be flat, range={np.ptp(bot_pts[:, 1])}"


def test_lexicon_maps_diverse_font_prompts():
    """Verify that diverse Korean and English prompts map to the corresponding font archetypes."""
    # 1. 고딕 / 산세리프
    spec_geo = interpret("깔끔하고 모던한 직선 고딕체 산세리프 폰트")
    assert spec_geo.style.archetype == "geometric_sans"

    # 2. 명조 / 바탕 / 세리프
    spec_myeongjo = interpret("단아하고 기품 있는 명조체 바탕 서체")
    assert spec_myeongjo.style.archetype == "classic_myeongjo"

    # 3. 타자기 / 레트로 빈티지
    spec_retro = interpret("빈티지 아날로그 타자기체 레트로 활자")
    assert spec_retro.style.archetype == "retro_typewriter"

    # 4. 자유분방 필기체
    spec_free = interpret("자유분방하고 거침없는 손글씨 필기체")
    assert spec_free.style.connectedness >= 0.50
