"""ONNX export and CPU validation for VibeFont models.

Exports the core neural modules to ONNX format for low-spec CPU, mobile,
and in-browser (ONNX Runtime Web / WebGPU) inference without Python dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from .models.core import CoreConfig, CoreModel


class TextEncoderONNX(nn.Module):
    """Encodes text token IDs into contextual embeddings."""

    def __init__(self, m: CoreModel):
        super().__init__()
        self.tok = m.tok
        self.tpos = m.tpos
        self.enc = m.enc
        self.norm = m.enc_norm

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(text.size(1), device=text.device).clamp(max=511)
        h = self.tok(text) + self.tpos(pos)[None]
        return self.norm(self.enc(h))


class StyleEncoderONNX(nn.Module):
    """Maps continuous style descriptors + writer ID to style prompt tokens."""

    def __init__(self, m: CoreModel):
        super().__init__()
        self.writer = m.writer
        self.style_mlp = m.style_mlp
        self.n_tokens = m.c.n_style_tokens
        self.d_model = m.c.d_model

    def forward(self, desc: torch.Tensor, writer_id: torch.Tensor) -> torch.Tensor:
        z = self.writer(writer_id)
        inp = torch.cat([z, desc], dim=-1)
        return self.style_mlp(inp).view(desc.size(0), self.n_tokens, self.d_model)


class HeadsONNX(nn.Module):
    """Computes autoregressively factorized logits for pen, dx, dy, dt."""

    def __init__(self, m: CoreModel):
        super().__init__()
        self.h_pen = m.h_pen
        self.h_dx = m.h_dx
        self.c_dx = m.c_dx
        self.c_dy = m.c_dy
        self.h_dy = m.h_dy
        self.h_dt = m.h_dt

    def forward(self, h: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor):
        l_pen = self.h_pen(h)
        l_dx = self.h_dx(h)
        l_dy = self.h_dy(h + self.c_dx(dx))
        l_dt = self.h_dt(h + self.c_dx(dx) + self.c_dy(dy))
        return l_pen, l_dx, l_dy, l_dt


def export_core_model_onnx(model: CoreModel, out_dir: str | Path, opset_version: int = 17) -> dict[str, Path]:
    """Exports CoreModel components into standalone ONNX files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()

    files = {}

    # 1. Text Encoder
    seq_len = 64
    te = TextEncoderONNX(model)
    te.eval()
    dummy_text = torch.zeros((1, seq_len), dtype=torch.long)
    te_path = out / "text_encoder.onnx"
    torch.onnx.export(
        te,
        dummy_text,
        str(te_path),
        input_names=["text"],
        output_names=["mem"],
        dynamic_axes={"text": {0: "batch"}, "mem": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    files["text_encoder"] = te_path

    # 2. Style Encoder
    se = StyleEncoderONNX(model)
    se.eval()
    dummy_desc = torch.randn(1, model.c.desc_dim, dtype=torch.float32)
    dummy_writer = torch.tensor([0], dtype=torch.long)
    se_path = out / "style_encoder.onnx"
    torch.onnx.export(
        se,
        (dummy_desc, dummy_writer),
        str(se_path),
        input_names=["desc", "writer_id"],
        output_names=["style_tokens"],
        dynamic_axes={"desc": {0: "batch"}, "writer_id": {0: "batch"}, "style_tokens": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    files["style_encoder"] = se_path

    # 3. Heads
    hm = HeadsONNX(model)
    hm.eval()
    dummy_h = torch.randn(1, model.c.d_model, dtype=torch.float32)
    dummy_dx = torch.tensor([0], dtype=torch.long)
    dummy_dy = torch.tensor([0], dtype=torch.long)
    hm_path = out / "heads.onnx"
    torch.onnx.export(
        hm,
        (dummy_h, dummy_dx, dummy_dy),
        str(hm_path),
        input_names=["h", "dx", "dy"],
        output_names=["logits_pen", "logits_dx", "logits_dy", "logits_dt"],
        dynamic_axes={
            "h": {0: "batch"},
            "dx": {0: "batch"},
            "dy": {0: "batch"},
            "logits_pen": {0: "batch"},
            "logits_dx": {0: "batch"},
            "logits_dy": {0: "batch"},
            "logits_dt": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    files["heads"] = hm_path

    return files


def verify_onnx_models(model: CoreModel, onnx_dir: str | Path) -> dict[str, float]:
    """Runs verification checks comparing ONNX Runtime outputs with PyTorch outputs on CPU."""
    out = Path(onnx_dir)
    model.eval()
    diffs = {}

    # 1. Text Encoder verification
    te_session = ort.InferenceSession(str(out / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
    dummy_text = torch.zeros((1, 64), dtype=torch.long)
    dummy_text[0, :5] = torch.tensor([10, 20, 30, 40, 50])
    te = TextEncoderONNX(model).eval()
    with torch.no_grad():
        py_mem = te(dummy_text).numpy()
    ort_mem = te_session.run(["mem"], {"text": dummy_text.numpy()})[0]
    diffs["text_encoder"] = float(np.max(np.abs(py_mem - ort_mem)))

    # 2. Style Encoder verification
    se_session = ort.InferenceSession(str(out / "style_encoder.onnx"), providers=["CPUExecutionProvider"])
    dummy_desc = torch.randn(1, model.c.desc_dim, dtype=torch.float32)
    dummy_writer = torch.tensor([0], dtype=torch.long)
    se = StyleEncoderONNX(model).eval()
    with torch.no_grad():
        py_st = se(dummy_desc, dummy_writer).numpy()
    ort_st = se_session.run(
        ["style_tokens"],
        {"desc": dummy_desc.numpy(), "writer_id": dummy_writer.numpy()},
    )[0]
    diffs["style_encoder"] = float(np.max(np.abs(py_st - ort_st)))

    # 3. Heads verification
    hm_session = ort.InferenceSession(str(out / "heads.onnx"), providers=["CPUExecutionProvider"])
    dummy_h = torch.randn(1, model.c.d_model, dtype=torch.float32)
    dummy_dx = torch.tensor([min(15, model.c.n_dx - 1)], dtype=torch.long)
    dummy_dy = torch.tensor([min(20, model.c.n_dy - 1)], dtype=torch.long)
    hm = HeadsONNX(model).eval()
    with torch.no_grad():
        py_outs = [x.numpy() for x in hm(dummy_h, dummy_dx, dummy_dy)]
    ort_outs = hm_session.run(
        None,
        {"h": dummy_h.numpy(), "dx": dummy_dx.numpy(), "dy": dummy_dy.numpy()},
    )
    h_diffs = [np.max(np.abs(p - o)) for p, o in zip(py_outs, ort_outs)]
    diffs["heads"] = float(max(h_diffs))

    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VibeFont PyTorch model to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/test_run/model.pt", help="Path to model checkpoint")
    parser.add_argument("--output", default="checkpoints/onnx", help="Output directory for ONNX models")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = CoreConfig(**config_dict)
    model = CoreModel(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded CoreModel ({cfg.d_model}d, {cfg.n_enc} enc, {cfg.n_dec} dec)")
    files = export_core_model_onnx(model, args.output, opset_version=args.opset)
    print(f"Exported {len(files)} ONNX models to {args.output}")

    diffs = verify_onnx_models(model, args.output)
    print("Verification on CPU (max absolute difference vs PyTorch):")
    for name, diff in diffs.items():
        print(f"  - {name}: {diff:.6e}")


if __name__ == "__main__":
    main()
