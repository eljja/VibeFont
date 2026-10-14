import numpy as np
import pytest
import torch

from vibefont.document.drift import DocumentDrift, LineState
from vibefont.document.pipeline import write_document
from vibefont.generate.base import LineRequest
from vibefont.generate.model import ModelGenerator
from vibefont.models.core import build, count_params
from vibefont.render.raster import render_page
from vibefont.text_codec import VOCAB_SIZE, encode as encode_text
from vibefont.tokenizer import StrokeTokenizer
from vibefont.vibe.spec import StyleSpec, VibeSpec


def test_core_model_build_and_forward():
    tok = StrokeTokenizer()
    model = build(
        "xs",
        vocab_size=VOCAB_SIZE,
        n_dx=tok.sizes["dx"],
        n_dy=tok.sizes["dy"],
        n_dt=tok.sizes["dt"],
        n_writers=10,
    )
    assert 3_500_000 < count_params(model) < 5_000_000

    B, L, N, C = 2, 8, 30, 20
    tgt = torch.zeros((B, N, 4), dtype=torch.long)
    tgt[..., 0] = torch.randint(0, 255, (B, N))
    tgt[..., 1] = torch.randint(0, 255, (B, N))
    tgt[..., 2] = torch.randint(0, 32, (B, N))
    tgt[..., 3] = torch.randint(0, 3, (B, N))

    ctx = torch.zeros((B, C, 4), dtype=torch.long)
    ctx[..., 0] = torch.randint(0, 255, (B, C))
    ctx[..., 1] = torch.randint(0, 255, (B, C))
    ctx[..., 2] = torch.randint(0, 32, (B, C))
    ctx[..., 3] = torch.randint(0, 3, (B, C))

    batch = {
        "text": torch.randint(0, VOCAB_SIZE, (B, L)),
        "text_mask": torch.ones((B, L), dtype=torch.bool),
        "tgt": tgt,
        "tgt_mask": torch.ones((B, N), dtype=torch.bool),
        "ctx": ctx,
        "ctx_mask": torch.ones((B, C), dtype=torch.bool),
        "writer": torch.randint(0, 10, (B,)),
        "desc": torch.randn((B, 8)),
        "z_drop": torch.zeros((B,), dtype=torch.bool),
    }

    losses = model(batch)
    assert "loss" in losses
    assert losses["loss"].item() > 0.0


def test_core_model_generate():
    tok = StrokeTokenizer()
    model = build(
        "xs",
        vocab_size=VOCAB_SIZE,
        n_dx=tok.sizes["dx"],
        n_dy=tok.sizes["dy"],
        n_dt=tok.sizes["dt"],
        n_writers=5,
    )
    text = torch.tensor([encode_text("test")], dtype=torch.long)
    z = torch.zeros((1, 64))
    desc = torch.zeros((1, 8))

    out = model.generate(text, z, desc, max_points=15)
    assert out.dim() == 2
    assert out.shape[1] == 4
    assert len(out) <= 15


def test_model_generator_pipeline(tmp_path):
    # Use the test_run checkpoint if available
    ckpt_dir = "checkpoints/test_run"
    import os
    if not os.path.exists(os.path.join(ckpt_dir, "model.pt")):
        pytest.skip("No trained checkpoint found")

    gen = ModelGenerator(ckpt_dir, device="cpu")
    spec = VibeSpec()
    spec.layout.page = "A5"
    spec.seed = 42

    rng = np.random.default_rng(spec.seed)
    pages = write_document("hi", spec, gen, rng)
    assert len(pages) >= 1
    assert len(pages[0].lines) >= 1

    img = render_page(pages[0], spec, dpi=72)
    assert img.width > 0
    assert img.height > 0
