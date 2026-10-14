import numpy as np
import pytest
import torch
from pathlib import Path

from vibefont.export_onnx import export_core_model_onnx, verify_onnx_models
from vibefont.models.core import CoreConfig, CoreModel, PRESETS


def test_export_and_verify_onnx_cpu(tmp_path: Path):
    cfg = CoreConfig(vocab_size=150, n_dx=128, n_dy=128, n_dt=32, **PRESETS["xs"])
    model = CoreModel(cfg)
    model.eval()

    out_dir = tmp_path / "onnx_export"
    files = export_core_model_onnx(model, out_dir)

    assert "text_encoder" in files
    assert "style_encoder" in files
    assert "heads" in files

    for name, path in files.items():
        assert path.exists(), f"ONNX file {path} was not created"
        assert path.stat().st_size > 1000

    # Verify onnxruntime outputs match PyTorch outputs on CPU
    diffs = verify_onnx_models(model, out_dir)
    for component, max_diff in diffs.items():
        assert max_diff < 1e-4, f"{component} ONNX diff {max_diff:.6e} exceeds tolerance 1e-4"
