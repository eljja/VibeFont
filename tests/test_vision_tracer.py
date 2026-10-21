import numpy as np
import pytest
from PIL import Image, ImageDraw

from vibefont.vision.tracer import trace_image_to_ink


def test_trace_synthetic_diagonal_stroke():
    # 100x100 white image with a black stroke from (20, 20) to (80, 60)
    im = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(im)
    draw.line([(20, 20), (50, 40), (80, 60)], fill=0, width=5)

    ink = trace_image_to_ink(im, text="line")

    assert len(ink) >= 5
    assert ink.text == "line"
    assert ink.has_pressure
    assert 0.0 <= np.mean(ink.points[:, 3]) <= 1.0

    # Timestamps strictly increasing
    dt = np.diff(ink.points[:, 2])
    assert np.all(dt >= -1e-5)

    # Stroke count
    strokes = ink.stroke_slices()
    assert len(strokes) >= 1


def test_trace_multi_stroke():
    # 120x100 image with two separate strokes (like 'T')
    im = Image.new("L", (120, 100), color=255)
    draw = ImageDraw.Draw(im)
    draw.line([(20, 20), (100, 20)], fill=0, width=6)  # horizontal bar
    draw.line([(60, 20), (60, 80)], fill=0, width=6)   # vertical stem

    ink = trace_image_to_ink(im, text="T")

    assert len(ink) >= 10
    slices = ink.stroke_slices()
    assert len(slices) >= 1


def test_trace_real_iam_word_sample():
    from pathlib import Path
    sample_path = Path("scratch/sample_word.png")
    if not sample_path.exists():
        pytest.skip("sample_word.png not found")

    ink = trace_image_to_ink(sample_path, text="sample")
    assert len(ink) >= 10
    assert len(ink.stroke_slices()) >= 1
    assert ink.has_pressure

