import json
from pathlib import Path
import numpy as np
import pytest

from vibefont.data.collector import ingest_collector_json
from vibefont.dataset import LineStore


@pytest.fixture
def sample_collector_json(tmp_path: Path) -> Path:
    data = {
        "format": "vibefont-collector/1",
        "writer": "tester01",
        "exported_at": "2026-10-14T12:00:00Z",
        "items": [
            {
                "text": "다람쥐 헌 쳇바퀴에 타고파",
                "prompt_idx": 0,
                "writer": "tester01",
                "strokes": [
                    {
                        "x": [100.0, 105.0, 110.0, 115.0],
                        "y": [150.0, 152.0, 155.0, 160.0],
                        "t": [0.0, 0.05, 0.10, 0.15],
                        "p": [0.35, 0.50, 0.70, 0.40],
                    },
                    {
                        "x": [130.0, 135.0, 140.0],
                        "y": [140.0, 145.0, 150.0],
                        "t": [0.30, 0.35, 0.40],
                        "p": [0.60, 0.65, 0.55],
                    },
                ],
            },
            {
                "text": "The quick brown fox",
                "prompt_idx": 1,
                "writer": "tester01",
                "strokes": [
                    {
                        "x": [50.0, 60.0, 70.0, 80.0],
                        "y": [100.0, 105.0, 102.0, 100.0],
                        "t": [0.0, 0.04, 0.08, 0.12],
                        "p": [0.45, 0.55, 0.60, 0.30],
                    }
                ],
            },
        ],
    }
    json_path = tmp_path / "collector_export.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return json_path


def test_collector_ingest_pipeline(sample_collector_json: Path, tmp_path: Path):
    out_dir = tmp_path / "processed_collector"
    n_lines = ingest_collector_json(sample_collector_json, out_dir)

    assert n_lines == 2

    store = LineStore.load(out_dir)
    assert len(store) == 2

    # Verify line 0 (Korean)
    ink0 = store.ink(0)
    row0 = store.rows[0]
    assert row0["text"] == "다람쥐 헌 쳇바퀴에 타고파"
    assert row0["writer"] == "tester01"
    assert row0["source"] == "collector"
    assert row0["line_no"] == 0
    assert len(ink0) > 0
    assert ink0.has_pressure
    assert 0.0 <= np.median(ink0.points[:, 3]) <= 1.0  # valid pressure

    # Verify line 1 (English)
    ink1 = store.ink(1)
    row1 = store.rows[1]
    assert row1["text"] == "The quick brown fox"
    assert row1["line_no"] == 1

    # Verify document context linking
    prev_indices = store.previous_line()
    assert prev_indices[0] == -1
    assert prev_indices[1] == 0  # line 1 is preceded by line 0
