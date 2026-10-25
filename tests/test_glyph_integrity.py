"""Glyph integrity: every input character must be rendered. Silent drops are never acceptable."""

from pathlib import Path

import numpy as np
import pytest

from vibefont.generate.calligraphy import compose_calligraphy_page
from vibefont.vibe.spec import PenSpec, VibeSpec

KO_CALLI_TEXTS = [
    "눈보라 속에서도\n매화는 자신의 향기를\n팔지 않는다",
    "별을 노래하는 마음으로\n모든 죽어가는 것을\n사랑해야지",
    "바람이 그치면\n비로소 호수 위에\n둥근 달이 머문다",
    "가만히 내리는 봄비에\n세상 모든 나뭇가지마다\n연분홍 등불이 켜진다",
    "바람에 흔들릴지언정\n푸른 대나무는\n결코 꺾이지 않는다",
    "따뜻한 찻잔 속에\n오늘 하루의 수고를\n조용히 내려놓는다",
    "천천히 걸어가는 사람만이\n숲이 들려주는 작은 노래를\n온전히 듣는다",
    "가장 어두운 밤의 끝에서\n가장 눈부신 새벽이\n시작된다",
    "수천 번 부서지는 파도가\n마침내 가장 아름다운\n해안선을 조각한다",
    "당신이 웃던 그 순간\n나의 모든 시간은\n따뜻한 봄이 되었다",
]

EN_CALLI_TEXTS = [
    "In the deepest silence\nthe clearest wisdom\nquietly awakens.",
    "Where there is kindness,\nthere is goodness, and\nevery winter turns to spring.",
    "Courage is not the absence\nof fear, but the triumph\nof purpose over hesitation.",
]


def _brush_spec() -> VibeSpec:
    return VibeSpec(pen=PenSpec(type="brush"))


def _glyph_records(page):
    recs = []
    for line in page.lines:
        for g in line.meta.get("glyphs", []):
            recs.append((line, g))
    return recs


def _glyph_xy(line, g) -> np.ndarray:
    sls = line.stroke_slices()
    i0, i1 = g["strokes"]
    return np.concatenate([line.xy[sls[i]] for i in range(i0, i1)], axis=0)


def test_korean_particle_is_rendered_not_dropped():
    """'별을' must be visibly wider than '별': the particle 을 needs its own ink."""

    def extent(text: str) -> float:
        page = compose_calligraphy_page(text, _brush_spec(), rng=np.random.default_rng(3))
        xs = page.lines[0].xy[:, 0]
        return float(xs.max() - xs.min())

    assert extent("별을") > 1.3 * extent("별")


@pytest.mark.parametrize("text", KO_CALLI_TEXTS)
def test_every_korean_syllable_has_its_own_ink(text):
    page = compose_calligraphy_page(text, _brush_spec(), rng=np.random.default_rng(11))
    recs = _glyph_records(page)
    expected = [c for c in text if not c.isspace()]
    assert [g["ch"] for _, g in recs] == expected

    boxes = []
    for line, g in recs:
        xy = _glyph_xy(line, g)
        assert len(xy) >= 2, f"glyph {g['ch']!r} has no ink"
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        assert (x1 - x0) > 0.3 and (y1 - y0) > 0.3, f"glyph {g['ch']!r} is degenerate"
        boxes.append((line, x0, y0, x1, y1, g["ch"]))

    # Within a line, glyph centres must strictly advance left -> right (no glyph stacked on another)
    for a, b in zip(boxes, boxes[1:]):
        if a[0] is b[0]:
            assert 0.5 * (b[1] + b[3]) > 0.5 * (a[1] + a[3]) + 0.5, f"{a[5]!r} and {b[5]!r} are stacked"


@pytest.mark.parametrize("text", EN_CALLI_TEXTS)
def test_every_english_word_has_its_own_ink(text):
    page = compose_calligraphy_page(text, VibeSpec(pen=PenSpec(type="broad_nib")), rng=np.random.default_rng(5))
    recs = _glyph_records(page)
    assert [g["ch"] for _, g in recs] == text.split()
    for line, g in recs:
        assert len(_glyph_xy(line, g)) >= 2


def test_smart_punctuation_is_not_dropped():
    from vibefont.generate.procedural import ProceduralGenerator

    font = ProceduralGenerator().fonts["cursive"]
    plain, _ = font.word("It's \"fine\"...-")
    smart, _ = font.word("It’s “fine”…—")
    assert len(smart) == len(plain)


def test_unsupported_glyph_raises_instead_of_silent_drop():
    from vibefont.generate.glyph_guard import MissingGlyphError
    from vibefont.generate.procedural import ProceduralGenerator

    font = ProceduralGenerator().fonts["cursive"]
    with pytest.raises(MissingGlyphError):
        font.word("a☃b", rng=np.random.default_rng(0))
    with pytest.raises(MissingGlyphError):
        compose_calligraphy_page("눈 ☃ 꽃", _brush_spec(), rng=np.random.default_rng(0))


CKPT = Path("checkpoints/bilingual_5k_core")


@pytest.mark.skipif(not (CKPT / "model.pt").exists(), reason="trained checkpoint not available")
def test_model_generator_never_truncates_trailing_characters(monkeypatch):
    from vibefont.document.drift import LineState
    from vibefont.generate.base import LineRequest
    from vibefont.generate.model import ModelGenerator
    from vibefont.ink import Ink

    gen = ModelGenerator(CKPT, device="cpu")
    real_decode = gen.tokenizer.decode

    def short_decode(*a, **k):
        ink = real_decode(*a, **k)
        return Ink(ink.points[: len(ink) // 2].copy(), text=ink.text)

    monkeypatch.setattr(gen.tokenizer, "decode", short_decode)
    spec = VibeSpec()
    st = LineState(index=0, progress=0.0, slant_deg=6.0, size_scale=1.0, tilt_deg=0.0,
                   baseline_offset_mm=0.0, margin_offset_mm=0.0, word_spacing_scale=1.0,
                   temperature=0.9, neatness=0.8)
    text = "끝까지 한 글자도 빠짐없이"
    base = gen._scaffold.generate_line(LineRequest(text, spec.style, st, np.random.default_rng(1)))
    out = gen.generate_line(LineRequest(text, spec.style, st, np.random.default_rng(1)))
    assert len(out) == len(base)
    assert out.xy[:, 0].max() >= base.xy[:, 0].max() - 0.3


def test_interpreter_preserves_fountain_pen_in_calligraphy():
    from vibefont.vibe.interpreter import interpret

    spec = interpret("아이보리 엽서에 블루블랙 만년필로 정성스럽게 쓴 캘리그라피")
    assert spec.pen.type == "fountain"
    assert spec.paper.type == "ivory"


def test_fountain_pen_calligraphy_renders_with_rich_flex_and_shading():
    from vibefont.render.pens import pen_samples
    from vibefont.render.raster import render_page
    from vibefont.vibe.spec import InkSpec, PaperSpec, StyleSpec

    spec = VibeSpec(
        pen=PenSpec(type="fountain", width_mm=0.65),
        ink=InkSpec(color=[0.08, 0.12, 0.30]),
        paper=PaperSpec(type="ivory"),
        style=StyleSpec(size_mm=4.5),
    )
    text = "별을 노래하는 마음으로\n모든 죽어가는 것을\n사랑해야지"
    page = compose_calligraphy_page(text, spec, rng=np.random.default_rng(702))
    assert len(page.lines) == 3

    # Check 100% glyph integrity
    recs = _glyph_records(page)
    expected = [c for c in text if not c.isspace()]
    assert [g["ch"] for _, g in recs] == expected

    # Verify pen dynamics: flex nib produces > 3x width modulation and liquid shading
    ps = pen_samples(page.lines[0], spec.pen, spec.style, np.random.default_rng(702))
    assert ps.width_mm.max() > 2.5 * ps.width_mm.min()
    assert ps.dens.max() > 1.8 * ps.dens.min()
    assert ps.feather > 0.4
    assert ps.rim_halo > 0.4

    # Render image and confirm non-empty
    img = render_page(page, spec, dpi=150, rng=np.random.default_rng(702))
    assert img.width > 500 and img.height > 500


def test_write_document_validates_text_and_prevents_dropped_words():
    from vibefont.document.pipeline import write_document
    from vibefont.generate.glyph_guard import MissingGlyphError
    from vibefont.generate.procedural import ProceduralGenerator

    gen = ProceduralGenerator()
    spec = VibeSpec()

    # Unsupported glyph raises immediately
    with pytest.raises(MissingGlyphError):
        write_document("안녕하세요 ☃ 반갑습니다", spec, gen)

    # Valid text renders all words
    pages = write_document("달 밝은 밤에 별을 노래하다", spec, gen)
    placed_words = [w for p in pages for ln in p.lines for w in getattr(ln, "text", "").split()]
    assert placed_words == ["달", "밝은", "밤에", "별을", "노래하다"]
