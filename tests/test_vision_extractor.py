import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from vibefont.vibe.spec import StyleSpec
from vibefont.vision.extractor import extract_style_from_image


def test_extract_style_synthetic_slanted_image():
    # 200x80 white image with clean slanted strokes (~15 degrees right lean)
    im = Image.new("L", (200, 80), color=255)
    draw = ImageDraw.Draw(im)

    # Draw several vertical lines slanted by ~15 deg (dx=15 over dy=50)
    for x in range(30, 180, 25):
        draw.line([(x + 15, 15), (x, 65)], fill=0, width=4)

    style = extract_style_from_image(im)

    assert isinstance(style, StyleSpec)
    # Measured slant should be approximately between 10 and 20 degrees
    assert 8.0 <= style.slant_deg <= 25.0
    assert 0.0 <= style.pressure <= 1.0
    assert 0.0 <= style.roundness <= 1.0


def test_extract_style_real_image_smoke():
    from pathlib import Path
    sample_path = Path("scratch/sample_word.png")
    if not sample_path.exists():
        pytest.skip("sample_word.png not found")

    style = extract_style_from_image(sample_path)
    assert isinstance(style, StyleSpec)
    assert -20.0 <= style.slant_deg <= 40.0
    assert 0.1 <= style.pressure <= 1.0


def test_extract_style_korean_ocr_sample():
    import zipfile, os
    korean_dirs = [p for p in Path(".").iterdir() if p.is_dir() and "OCR" in p.name]
    if not korean_dirs:
        pytest.skip("Korean OCR directory not found")

    val_img = korean_dirs[0] / "Validation" / "[원천]validation_필기체.zip"
    if not val_img.exists():
        pytest.skip("Validation zip not found")

    with zipfile.ZipFile(val_img) as z:
        jpgs = [n for n in z.namelist() if n.endswith(".jpg")]
        if not jpgs:
            pytest.skip("No jpgs in zip")
        img_bytes = z.read(jpgs[0])

    style = extract_style_from_image(img_bytes)
    assert isinstance(style, StyleSpec)
    assert -25.0 <= style.slant_deg <= 35.0
    assert 0.1 <= style.pressure <= 1.0

