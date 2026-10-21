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

