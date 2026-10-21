import os
from pathlib import Path
import pytest

from vibefont.data.korean_ocr import ingest_korean_ocr
from vibefont.dataset import LineStore


def test_korean_ocr_ingest_smoke(tmp_path: Path):
    # Locate Korean OCR dataset directory
    korean_dirs = [p for p in Path(".").iterdir() if p.is_dir() and "OCR" in p.name]
    if not korean_dirs:
        pytest.skip("Korean OCR dataset directory not found")

    korean_dir = korean_dirs[0]
    val_lbl = korean_dir / "Validation" / "[라벨]validation_필기체.zip"
    val_img = korean_dir / "Validation" / "[원천]validation_필기체.zip"

    if not val_lbl.exists() or not val_img.exists():
        pytest.skip("Validation zip files not found")

    out_dir = tmp_path / "korean_ocr_sample"
    # Ingest 5 Korean characters/words
    n = ingest_korean_ocr(val_lbl, val_img, out_dir, limit=5)

    assert n == 5
    store = LineStore.load(out_dir)
    assert len(store) == 5

    ink0 = store.ink(0)
    assert len(ink0) > 0
    assert ink0.has_pressure
    assert len(store.rows[0]["text"]) > 0
    assert store.rows[0]["source"] == "korean_ocr"
