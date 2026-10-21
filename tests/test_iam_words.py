from pathlib import Path
import pytest

from vibefont.data.iam_words import parse_words_txt, ingest_iam_words
from vibefont.dataset import LineStore


def test_parse_words_txt(tmp_path: Path):
    txt_content = """# Header comment
#
a01-000u-00-00 ok 154 408 768 27 51 AT A
a01-000u-00-01 ok 154 507 766 213 48 NN MOVE
a01-000u-00-02 err 154 796 764 70 50 TO to
"""
    txt_file = tmp_path / "test_words.txt"
    txt_file.write_text(txt_content, encoding="utf-8")

    meta = parse_words_txt(txt_file)
    assert len(meta) == 2  # Only 'ok' lines
    assert "a01-000u-00-00" in meta
    assert meta["a01-000u-00-00"]["text"] == "A"
    assert meta["a01-000u-00-01"]["text"] == "MOVE"


def test_iam_words_ingest_smoke(tmp_path: Path):
    zip_path = Path("words.tgz.zip")
    txt_path = Path("data/raw/iam_words/words.txt")
    if not zip_path.exists() or not txt_path.exists():
        pytest.skip("words.tgz.zip or words.txt not found")

    out_dir = tmp_path / "iam_words_sample"
    # Ingest small batch of 5 words
    n = ingest_iam_words(zip_path, txt_path, out_dir, limit=5)

    assert n == 5
    store = LineStore.load(out_dir)
    assert len(store) == 5

    ink0 = store.ink(0)
    assert len(ink0) > 0
    assert ink0.has_pressure
    assert len(store.rows[0]["text"]) > 0
    assert store.rows[0]["source"] == "iam_words"
