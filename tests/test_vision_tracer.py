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
    xmin, ymin, xmax, ymax = ink.bbox()
    assert abs(xmin) <= 0.5
    assert -3.0 <= ymin < 0.0
    assert -0.5 <= ymax <= 1.5
    assert 0.5 <= (ymax - ymin) <= 2.8
    assert 0.3 <= (xmax - xmin) <= 20.0


def test_trace_never_explodes_on_short_word_or_hangul():
    # Simulate a word like "MR." with a dot that previously caused estimate_line_geometry to explode to 1415
    im = Image.new("L", (160, 80), color=255)
    draw = ImageDraw.Draw(im)
    draw.line([(15, 65), (25, 20), (40, 50), (55, 20), (65, 65)], fill=0, width=4)  # M
    draw.line([(80, 65), (80, 20), (105, 25), (80, 42), (110, 65)], fill=0, width=4)  # R
    draw.ellipse([(125, 58), (133, 66)], fill=0)  # period '.'

    ink = trace_image_to_ink(im, text="MR.")
    xmin, ymin, xmax, ymax = ink.bbox()
    assert abs(xmin) <= 0.5, f"Expected xmin near 0, got {xmin}"
    assert -3.0 <= ymin < 0.0, f"Expected ymin in [-3, 0), got {ymin}"
    assert -0.5 <= ymax <= 1.5, f"Expected ymax in [-0.5, 1.5], got {ymax}"
    assert 0.8 <= (ymax - ymin) <= 2.6, f"Expected normalized height in [0.8, 2.6], got {ymax - ymin}"
    assert 1.0 <= (xmax - xmin) <= 10.0, f"Expected reasonable width, got {xmax - xmin}"


