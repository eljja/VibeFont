import numpy as np
import pytest

from vibefont.document.drift import LineState
from vibefont.generate.base import LineRequest
from vibefont.generate.procedural import ProceduralGenerator
from vibefont.vibe.spec import StyleSpec


def test_procedural_generator_korean():
    gen = ProceduralGenerator()
    style = StyleSpec()
    state = LineState(0, 0, 0, 1, 0, 0, 0, 1, 1, 0.6)
    rng = np.random.default_rng(42)

    # Pure Korean word
    req = LineRequest("동해물과 백두산이", style, state, rng)
    ink = gen.generate_line(req)

    # Must produce valid stroke points for Korean text!
    assert len(ink) > 0, "Korean text generated 0 stroke points"
    assert len(ink.stroke_slices()) >= 6, "Expected multiple strokes for Korean text"

    x0, y0, x1, y1 = ink.bbox()
    assert x1 > x0 > -1e-3
    assert ink.has_pressure


def test_procedural_generator_mixed_korean_english():
    gen = ProceduralGenerator()
    style = StyleSpec()
    state = LineState(0, 0, 0, 1, 0, 0, 0, 1, 1, 0.6)
    rng = np.random.default_rng(123)

    req = LineRequest("VibeFont 한글 123", style, state, rng)
    ink = gen.generate_line(req)

    assert len(ink) > 0
    # Must have strokes for VibeFont, 한글, and 123
    assert len(ink.stroke_slices()) >= 8
    w = gen.estimate_width("VibeFont 한글 123", style)
    assert w > 8.0  # reasonable width in x-height units


def test_procedural_generator_standalone_jamo():
    gen = ProceduralGenerator()
    style = StyleSpec()
    state = LineState(0, 0, 0, 1, 0, 0, 0, 1, 1, 0.6)
    rng = np.random.default_rng(999)

    req = LineRequest("ㅋㅋㅋ ㅎㅎ", style, state, rng)
    ink = gen.generate_line(req)

    assert len(ink) > 0
    assert len(ink.stroke_slices()) >= 5


def test_korean_rasterize_smoke():
    from vibefont.document.pipeline import write_document
    from vibefont.generate.procedural import ProceduralGenerator
    from vibefont.render.raster import render_page
    from vibefont.vibe.spec import VibeSpec

    gen = ProceduralGenerator()
    spec = VibeSpec()
    pages = write_document("손글씨 테스트: 다람쥐 헌 쳇바퀴에 타고파", spec, gen)
    assert len(pages) >= 1
    assert len(pages[0].lines) >= 1

    img = render_page(pages[0], spec, dpi=100)
    assert img.width > 100
    assert img.height > 100


def test_hangul_and_english_cap_height_balance():
    gen = ProceduralGenerator()
    style = StyleSpec()
    state = LineState(0, 0, 0, 1, 0, 0, 0, 1, 1, 0.6)

    ink_en = gen.generate_line(LineRequest("VibeFont", style, state, np.random.default_rng(1)))
    ink_ko = gen.generate_line(LineRequest("한글손글씨", style, state, np.random.default_rng(2)))

    _, ymin_en, _, ymax_en = ink_en.bbox()
    _, ymin_ko, _, ymax_ko = ink_ko.bbox()
    h_en = ymax_en - ymin_en
    h_ko = ymax_ko - ymin_ko

    # Hangul height must be visually balanced with English cap/ascender height (at least 1.45 x-heights)
    assert h_ko >= 1.45, f"Hangul is too small compared to English: h_ko={h_ko:.2f}, h_en={h_en:.2f}"
    assert ymin_ko <= -1.35, f"Hangul top should reach cap-height (~ -1.5), got {ymin_ko:.2f}"
    assert 0.70 <= (h_ko / h_en) <= 1.25


def test_hangul_sub_syllable_organic_variation_and_smooth_curves():
    from vibefont.generate.hangul import hangul_char_strokes

    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(20)
    st1 = hangul_char_strokes("한", rng=rng1, connectedness=0.6)
    st2 = hangul_char_strokes("한", rng=rng2, connectedness=0.6)

    # Every stroke must be a smooth multi-point curve, not a 2-point straight segment
    assert all(len(s) >= 6 for s in st1), f"Expected curved multi-point strokes, got lengths {[len(s) for s in st1]}"

    # Repeated calls with different rng must produce non-identical sub-syllable geometry
    pts1 = np.concatenate(st1, axis=0)
    pts2 = np.concatenate(st2, axis=0)
    if len(pts1) == len(pts2):
        assert not np.allclose(pts1, pts2, atol=1e-4)


def test_escaped_newline_handling_in_pipeline():
    from vibefont.document.pipeline import write_document
    from vibefont.vibe.spec import VibeSpec

    gen = ProceduralGenerator()
    spec = VibeSpec()
    pages = write_document("첫 번째 줄\\n두 번째 줄", spec, gen)
    assert len(pages[0].lines) == 2


def test_complex_jungseong_full_height_and_jongseong_rhythm():
    from vibefont.generate.hangul import syllable_strokes

    # '과' (complex vowel ㅘ): the right vertical stem must reach near the top of the syllable (< -1.45)
    gwa_strokes = syllable_strokes("과", rng=None, connectedness=0.0)
    v_stem = gwa_strokes[-2]
    assert v_stem[:, 1].min() < -1.45, f"Expected full-height right stem in '과', got min_y={v_stem[:, 1].min():.3f}"

    # Syllable without jongseong ('가') should be vertically more compact than syllable with jongseong ('각')
    ga_pts = np.concatenate(syllable_strokes("가", rng=None, connectedness=0.0), axis=0)
    gak_pts = np.concatenate(syllable_strokes("각", rng=None, connectedness=0.0), axis=0)
    ga_h = float(ga_pts[:, 1].max() - ga_pts[:, 1].min())
    gak_h = float(gak_pts[:, 1].max() - gak_pts[:, 1].min())
    assert gak_h > ga_h + 0.12, f"Expected jongseong syllable '각' ({gak_h:.2f}) taller than '가' ({ga_h:.2f})"


def test_hangul_kid_mode_vs_master_cursive_and_calligraphy():
    from vibefont.generate.hangul import hangul_char_strokes

    # Master cursive mode should connect more strokes than clumsy kid mode
    rng_m = np.random.default_rng(7)
    rng_k = np.random.default_rng(7)
    st_master = hangul_char_strokes("글", rng=rng_m, connectedness=0.92, neatness=0.92)
    st_kid = hangul_char_strokes("글", rng=rng_k, connectedness=0.05, neatness=0.10)
    assert len(st_master) < len(st_kid), f"Expected fewer (more connected) strokes in master cursive: {len(st_master)} vs {len(st_kid)}"

    # Clumsy kid mode should exhibit significantly higher bounding-box height variance across samples
    master_heights = []
    kid_heights = []
    for s in range(25):
        pm = np.concatenate(hangul_char_strokes("한", rng=np.random.default_rng(100 + s), connectedness=0.88, neatness=0.92), axis=0)
        pk = np.concatenate(hangul_char_strokes("한", rng=np.random.default_rng(100 + s), connectedness=0.05, neatness=0.10), axis=0)
        master_heights.append(float(pm[:, 1].max() - pm[:, 1].min()))
        kid_heights.append(float(pk[:, 1].max() - pk[:, 1].min()))
    assert np.std(kid_heights) > np.std(master_heights) * 1.8


def test_layout_center_and_vertical_center_alignment():
    from vibefont.document.pipeline import write_document
    from vibefont.vibe.spec import VibeSpec

    gen = ProceduralGenerator()
    spec = VibeSpec()
    spec.layout.page = "Card"
    spec.layout.align = "center"
    spec.layout.v_align = "center"
    pages = write_document("꽃이 피는 봄날", spec, gen, rng=np.random.default_rng(42))
    page = pages[0]
    assert len(page.lines) == 1
    xmin, ymin, xmax, ymax = page.lines[0].bbox()
    line_center_x = 0.5 * (xmin + xmax)
    line_center_y = 0.5 * (ymin + ymax)
    assert abs(line_center_x - 0.5 * page.width_mm) < 12.0
    assert abs(line_center_y - 0.5 * page.height_mm) < 18.0


def test_brush_vs_broad_nib_calligraphy_lexicon():
    from vibefont.vibe.interpreter import interpret

    spec_brush = interpret("아이보리 종이에 먹색 붓으로 힘차고 우아하게 쓴 캘리그라피")
    assert spec_brush.pen.type == "brush"

    spec_brush_en = interpret("ivory paper, charcoal sumi brush, calligraphy, bold graceful slow cursive")
    assert spec_brush_en.pen.type == "brush"

    spec_broad = interpret("아이보리 종이에 버건디 잉크 평촉으로 우아하게 쓴 캘리그라피")
    assert spec_broad.pen.type == "broad_nib"


def test_physical_pen_dynamics_and_contrast():
    from vibefont.document.pipeline import write_document
    from vibefont.render.pens import pen_samples
    from vibefont.vibe.spec import VibeSpec

    gen = ProceduralGenerator()
    spec = VibeSpec()
    spec.pen.type = "brush"
    spec.pen.width_mm = 1.2
    pages = write_document("아름다운 붓글씨", spec, gen, rng=np.random.default_rng(11))
    ink = pages[0].lines[0]
    ps = pen_samples(ink, spec.pen, spec.style, np.random.default_rng(11))
    # Brush must exhibit dramatic dynamic width contrast (thick press vs sharp tapered tip)
    width_ratio = float(ps.width_mm.max() / max(ps.width_mm.min(), 1e-4))
    assert width_ratio >= 5.5, f"Expected brush width contrast >= 5.5x, got {width_ratio:.2f}x"


def test_calligraphy_art_composition():
    from vibefont.generate.calligraphy import compose_calligraphy_page
    from vibefont.vibe.interpreter import interpret

    spec_ko = interpret("아이보리 종이에 먹색 붓으로 힘차고 우아하게 쓴 캘리그라피")
    spec_ko.seed = 701
    page_ko = compose_calligraphy_page(
        "눈보라 속에서도\n매화는 자신의 향기를\n팔지 않는다",
        spec_ko,
        rng=np.random.default_rng(701),
    )
    assert len(page_ko.lines) >= 3
    # Should include decorative calligraphic seal or flourish metadata
    assert page_ko.seal_box_mm is not None

    spec_en = interpret("ivory paper, burgundy broad nib calligraphy pen, elegant masterful cursive")
    spec_en.seed = 802
    page_en = compose_calligraphy_page(
        "Every shadow upon the earth\nis born only because\nthe light endures.",
        spec_en,
        rng=np.random.default_rng(802),
    )
    assert len(page_en.lines) >= 3






