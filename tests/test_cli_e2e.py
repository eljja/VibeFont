import json
import os
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def run_cli():
    def _run(*args):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        cmd = [sys.executable, "-m", "vibefont", *args]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        return res
    return _run


def test_cli_write_png(tmp_path: Path, run_cli):
    out_png = tmp_path / "test_out.png"
    res = run_cli(
        "write",
        "Hello VibeFont 한글 테스트",
        "-p", "만년필로 단정하게",
        "-o", str(out_png),
        "--dpi", "100",
        "--generator", "procedural"
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert out_png.exists()
    assert out_png.stat().st_size > 1000
    # Check PNG magic bytes
    with open(out_png, "rb") as f:
        header = f.read(8)
        assert header == b"\x89PNG\r\n\x1a\n"


def test_cli_write_svg(tmp_path: Path, run_cli):
    out_svg = tmp_path / "test_out.svg"
    res = run_cli(
        "write",
        "Vector ink line",
        "-p", "볼펜",
        "-o", str(out_svg),
        "--generator", "procedural"
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert out_svg.exists()
    content = out_svg.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "<path" in content


def test_cli_interpret(run_cli):
    res = run_cli("interpret", "급하게 휘갈겨 쓴 4B 연필 메모")
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    data = json.loads(res.stdout)
    assert "pen" in data
    assert data["pen"]["type"] == "pencil"
    assert data["style"]["speed"] > 0.6  # "급하게" triggers high speed


def test_cli_validate_data_command(run_cli):
    # Test on existing processed dataset
    res = run_cli("validate-data", "data/processed/bilingual_5k")
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Dataset Health: [HEALTHY]" in res.stdout
    assert "Hangul=" in res.stdout


def test_cli_write_with_save_spec(tmp_path: Path, run_cli):
    out_png = tmp_path / "doc.png"
    out_spec = tmp_path / "spec.json"
    res = run_cli(
        "write",
        "Coherence test",
        "-o", str(out_png),
        "--save-spec", str(out_spec),
        "--generator", "procedural",
        "--dpi", "72"
    )
    assert res.returncode == 0
    assert out_png.exists()
    assert out_spec.exists()
    spec_data = json.loads(out_spec.read_text(encoding="utf-8"))
    assert "style" in spec_data


def test_ruled_paper_does_not_skip_lines():
    from vibefont.document.layout import line_pitch_mm
    from vibefont.vibe.interpreter import interpret

    spec = interpret("ruled notebook, blue ballpoint pen, hurried fast slanted lecture notes")
    assert spec.paper.type == "ruled"
    assert line_pitch_mm(spec) == pytest.approx(spec.paper.line_spacing_mm)


def test_grid_paper_print_letters_does_not_trigger_formal_letter():
    from vibefont.vibe.interpreter import interpret

    spec = interpret("grid paper, gray pencil, small neat upright print letters, careful")
    assert spec.paper.type == "grid"
    assert "formal_letter" not in spec.matched


def test_print_mode_kerning_separates_narrow_letters():
    import numpy as np
    from vibefont.generate.procedural import _Font

    f = _Font("futural")
    rng = np.random.default_rng(42)
    strokes, _ = f.word("rn", rng=rng, connectedness=0.1)
    assert len(strokes) >= 4
    r_max_x = max(float(s[:, 0].max()) for s in strokes[:2])
    n_min_x = min(float(s[:, 0].min()) for s in strokes[2:])
    assert n_min_x - r_max_x >= 0.16


