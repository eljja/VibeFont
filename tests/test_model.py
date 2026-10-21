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


def test_stroke_dataset_provides_monotonic_char_idx_and_adv(tmp_path):
    from vibefont.data.loader import StrokeDataset, collate_stroke_batch
    from vibefont.dataset import DatasetWriter, LineStore
    from vibefont.ink import EOS, Ink, P, T, X, Y

    pts = np.zeros((20, 5), dtype=np.float32)
    pts[:, X] = np.linspace(0.0, 3.0, 20)
    pts[:, Y] = np.sin(np.linspace(0.0, 3.0, 20)) * 0.5
    pts[:, T] = np.linspace(0.0, 0.4, 20)
    pts[:, P] = 0.7
    pts[-1, EOS] = 1.0
    ink = Ink(pts, text="abc")

    w = DatasetWriter(tmp_path / "store")
    w.add(ink, writer="w0", doc="d0", line_no=0, source="test")
    w.close()

    store = LineStore.load(tmp_path / "store")
    ds = StrokeDataset(store, StrokeTokenizer())
    item = ds[0]
    assert "char_idx" in item
    assert "adv" in item
    assert len(item["char_idx"]) == len(item["tgt"])
    assert len(item["adv"]) == len(item["tgt"])
    # Monotonic non-decreasing char_idx starting at 1
    assert item["char_idx"][0] == 1
    assert np.all(np.diff(item["char_idx"]) >= 0)
    assert item["adv"][-1] == 1

    batch = collate_stroke_batch([item])
    assert "char_idx" in batch
    assert "adv" in batch

    tok = StrokeTokenizer()
    model = build(
        "xs",
        vocab_size=VOCAB_SIZE,
        n_dx=tok.sizes["dx"],
        n_dy=tok.sizes["dy"],
        n_dt=tok.sizes["dt"],
        n_writers=2,
    )
    losses = model(batch)
    assert "adv" in losses
    assert losses["adv"].item() > 0.0


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

    out = model.generate(text, z, desc, max_points=200)
    assert out.dim() == 2
    assert out.shape[1] == 4
    # Must never terminate prematurely after 1-2 points on a 4-char word
    assert len(out) >= 16
    assert len(out) <= 200
    assert int(out[-1, 3]) == 2


def test_model_generator_pipeline(tmp_path):
    ckpt_dir = "checkpoints/bilingual_5k_core"
    import os
    if not os.path.exists(os.path.join(ckpt_dir, "model.pt")):
        ckpt_dir = "checkpoints/test_run"
    if not os.path.exists(os.path.join(ckpt_dir, "model.pt")):
        pytest.skip("No trained checkpoint found")

    gen = ModelGenerator(ckpt_dir, device="cpu")
    spec = VibeSpec()
    spec.layout.page = "A5"
    spec.seed = 42

    text = "Hello VibeFont! 자연스러운 한글"
    rng1 = np.random.default_rng(42)
    pages1 = write_document(text, spec, gen, rng1)
    assert len(pages1) >= 1
    assert len(pages1[0].lines) >= 1

    line_ink1 = pages1[0].lines[0]
    xmin, ymin, xmax, ymax = line_ink1.bbox()
    # Must span a realistic line width and height (not collapsed to a tiny dot or squiggle)
    assert (xmax - xmin) > 8.0
    assert 0.8 <= (ymax - ymin) <= 12.0

    # Non-determinism: different seed must produce distinct stroke coordinates
    rng2 = np.random.default_rng(99)
    pages2 = write_document(text, spec, gen, rng2)
    line_ink2 = pages2[0].lines[0]
    assert len(line_ink1) != len(line_ink2) or not np.allclose(line_ink1.points[:, :2], line_ink2.points[:, :2])

    img = render_page(pages1[0], spec, dpi=72)
    assert img.width > 0
    assert img.height > 0
