import numpy as np
import pytest

from vibefont.generate.hangul import WriterProfile, choseong_strokes, hangul_char_strokes
from vibefont.vibe.interpreter import interpret
from vibefont.vibe.spec import VibeSpec


def test_gosi_produces_single_stroke_cursive_allographs():
    profile = WriterProfile(archetype="gosi")
    # In gosi archetype, ㄹ (idx 5) and ㅁ (idx 6) should be rendered as a single continuous cursive stroke
    st_rieul = choseong_strokes(5, profile=profile)
    assert len(st_rieul) == 1, f"Expected single-stroke z-form for gosi ㄹ, got {len(st_rieul)}"
    pts_rieul = st_rieul[0]
    assert len(pts_rieul) >= 10

    st_mieum = choseong_strokes(6, profile=profile)
    assert len(st_mieum) == 1, f"Expected single-stroke loop for gosi ㅁ, got {len(st_mieum)}"


def test_classic_jeongja_produces_orthogonal_separate_strokes():
    profile = WriterProfile(archetype="classic_jeongja")
    # In classic_jeongja archetype:
    # ㄱ (idx 0) has 2 separate strokes with 90-degree corner
    st_giyeok = choseong_strokes(0, profile=profile)
    assert len(st_giyeok) == 2, f"Expected 2 orthogonal strokes for jeongja ㄱ, got {len(st_giyeok)}"

    # ㅁ (idx 6) has 3 distinct strokes (left post, top-right right-angle, bottom closure)
    st_mieum = choseong_strokes(6, profile=profile)
    assert len(st_mieum) == 3, f"Expected 3 strokes for jeongja ㅁ, got {len(st_mieum)}"

    # ㅂ (idx 7) has 4 distinct upright strokes
    st_bieup = choseong_strokes(7, profile=profile)
    assert len(st_bieup) == 4, f"Expected 4 strokes for jeongja ㅂ, got {len(st_bieup)}"


def test_rounded_diary_produces_exaggerated_round_loops():
    profile_diary = WriterProfile(archetype="rounded_diary", loop_roundness=1.35)
    profile_std = WriterProfile(archetype="standard", loop_roundness=1.0)

    # ㅇ (idx 11) loop width and height
    st_ieung_diary = choseong_strokes(11, profile=profile_diary)[0]
    st_ieung_std = choseong_strokes(11, profile=profile_std)[0]

    diary_w = float(st_ieung_diary[:, 0].max() - st_ieung_diary[:, 0].min())
    std_w = float(st_ieung_std[:, 0].max() - st_ieung_std[:, 0].min())
    assert diary_w > std_w * 1.08, f"Diary ㅇ width {diary_w} should be noticeably larger than standard {std_w}"


def test_master_heullim_produces_flowing_calligraphic_allographs():
    profile = WriterProfile(archetype="master_heullim")
    # In master_heullim:
    # ㄹ (idx 5) is a single continuous S-river classical calligraphy stroke
    st_rieul = choseong_strokes(5, profile=profile)
    assert len(st_rieul) == 1, f"Expected 1 continuous S-river stroke for master heullim ㄹ, got {len(st_rieul)}"
    pts_rieul = st_rieul[0]
    assert len(pts_rieul) >= 16

    # ㅁ (idx 6) has 2 refined strokes: left pillar + continuous flowing enclosure
    st_mieum = choseong_strokes(6, profile=profile)
    assert len(st_mieum) == 2, f"Expected 2 refined strokes for master heullim ㅁ, got {len(st_mieum)}"

    # ㅂ (idx 7) has 3 strokes: left pillar, commanding right pillar, and flowing ligature
    st_bieup = choseong_strokes(7, profile=profile)
    assert len(st_bieup) == 3, f"Expected 3 strokes for master heullim ㅂ, got {len(st_bieup)}"

    # ㅅ (idx 9) has sweeping graceful curved arcs for both sweeps
    st_siot = choseong_strokes(9, profile=profile)
    assert len(st_siot) == 2
    assert len(st_siot[0]) >= 12
    assert len(st_siot[1]) >= 12


def test_lexicon_maps_personas_from_prompts():
    spec_gosi = interpret("고시체 수험생 속기 필기")
    assert spec_gosi.style.archetype == "gosi"
    assert spec_gosi.style.connectedness >= 0.70

    spec_jeongja = interpret("정갈한 정자체 교과서처럼 반듯하게")
    assert spec_jeongja.style.archetype == "classic_jeongja"
    assert spec_jeongja.style.roundness <= 0.35

    spec_diary = interpret("둥근 다이어리체 귀엽고 아기자기하게")
    assert spec_diary.style.archetype == "rounded_diary"
    assert spec_diary.style.roundness >= 0.70

    spec_master = interpret("명필처럼 멋진 필기체로 유려한 서예 흘림")
    assert spec_master.style.archetype == "master_heullim"
    assert spec_master.style.neatness >= 0.70
    assert spec_master.style.connectedness >= 0.60

