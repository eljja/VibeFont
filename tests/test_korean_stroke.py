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
