import numpy as np
import pytest
from pathlib import Path
from vibefont.dataset import DatasetWriter
from vibefont.ink import Ink
from vibefont.data.validator import validate_dataset, ValidationReport


def test_validator_on_healthy_dataset(tmp_path: Path):
    ds_dir = tmp_path / "healthy"
    writer = DatasetWriter(ds_dir)

    # 1st line: English
    pts1 = np.array([
        [0.0, 0.0, 0.0, 0.5, 0.0],
        [0.1, 0.1, 10.0, 0.6, 0.0],
        [0.2, 0.2, 20.0, 0.4, 1.0],
    ], dtype=np.float32)
    ink1 = Ink(pts1, "Hello")
    writer.add(ink1, "w1", "doc1", 0, "test")

    # 2nd line: Korean Hangul
    pts2 = np.array([
        [0.0, 0.0, 0.0, 0.5, 0.0],
        [0.05, 0.1, 10.0, 0.5, 1.0],
        [0.1, 0.0, 15.0, 0.7, 1.0],
    ], dtype=np.float32)
    ink2 = Ink(pts2, "한글 손글씨")
    writer.add(ink2, "w1", "doc1", 1, "test")

    writer.close()

    report = validate_dataset(ds_dir)
    assert isinstance(report, ValidationReport)
    assert report.is_healthy is True
    assert len(report.errors) == 0
    assert report.total_lines == 2
    assert report.total_points == 6
    assert report.vocab_stats["latin"] == 5
    assert report.vocab_stats["hangul"] == 5  # "한글 손글씨" has 5 hangul chars
    assert report.vocab_stats["unique_chars"] > 0


def test_validator_detects_nan_and_bad_offsets(tmp_path: Path):
    ds_dir = tmp_path / "corrupt"
    ds_dir.mkdir(parents=True)

    # Corrupt points with NaN
    pts = np.array([
        [0.0, np.nan, 0.0, 0.5, 0.0],
        [0.1, 0.1, 10.0, 0.5, 1.0]
    ], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int64)
    np.savez_compressed(ds_dir / "lines.npz", points=pts, offsets=offsets)

    # index with mismatched length or bad data
    (ds_dir / "index.jsonl").write_text('{"i": 0, "text": "bad"}\n{"i": 1, "text": "extra"}\n', encoding="utf-8")

    report = validate_dataset(ds_dir)
    assert report.is_healthy is False
    assert len(report.errors) >= 2  # NaN detected and row count mismatch


def test_validator_detects_non_monotonic_time(tmp_path: Path):
    ds_dir = tmp_path / "bad_time"
    ds_dir.mkdir(parents=True)

    pts = np.array([
        [0.0, 0.0, 100.0, 0.5, 0.0],
        [0.1, 0.1, 50.0, 0.5, 1.0],  # time goes backwards
    ], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int64)
    np.savez_compressed(ds_dir / "lines.npz", points=pts, offsets=offsets)
    (ds_dir / "index.jsonl").write_text('{"i": 0, "text": "time_test"}\n', encoding="utf-8")

    report = validate_dataset(ds_dir)
    assert report.is_healthy is False
    assert any("time" in err.lower() or "timestamp" in err.lower() or "monotonic" in err.lower() for err in report.errors)
