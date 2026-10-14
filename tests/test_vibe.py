import pytest
from vibefont.vibe.interpreter import interpret


def test_interpret_korean_and_english():
    spec1 = interpret("급하게 쓴 만년필 메모, 파란 잉크")
    assert spec1.pen.type == "fountain"
    assert spec1.style.speed > 0.5
    assert spec1.ink.color == (0.08, 0.17, 0.55)  # blue

    spec2 = interpret("very neat and clean, black ink, slow")
    assert spec2.style.neatness > 0.6
    assert spec2.style.speed < 0.5
    assert spec2.ink.color == (0.06, 0.06, 0.07)  # black

    spec3 = interpret("연필로 쓴 아이 글씨, 줄노트")
    assert spec3.pen.type == "pencil"
    assert spec3.paper.type == "ruled"


def test_interpret_intensity_and_seed():
    spec = interpret("엄청 빠르게 날려 쓴 글씨, seed: 42")
    assert spec.seed == 42
    assert spec.style.speed > 0.8
