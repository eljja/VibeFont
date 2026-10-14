"""ModelGenerator: bridges the trained CoreModel into the StrokeGenerator protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..descriptors import DESCRIPTOR_NAMES
from ..document.drift import LineState
from ..generate.base import LineRequest, StrokeGenerator
from ..ink import P, Ink
from ..models.core import CoreConfig, CoreModel
from ..preprocess import ensure_pressure
from ..text_codec import encode as encode_text
from ..tokenizer import StrokeTokenizer, TokenSeq
from ..vibe.spec import StyleSpec


def style_to_descriptor_vec(style: StyleSpec, state: LineState) -> np.ndarray:
    """Maps VibeSpec.style + LineState to the 8-dimensional model descriptor vector."""
    # Order matches DESCRIPTOR_NAMES:
    # ("slant_deg", "width_per_char", "speed", "connectedness", "roundness",
    #  "baseline_wobble", "ascender_ratio", "pressure")
    return np.array(
        [
            state.slant_deg,
            0.6 * style.width_scale,
            style.speed * 10.0,  # x-height / s scale
            style.connectedness,
            style.roundness,
            style.baseline_wobble * 0.15,
            1.0,
            style.pressure,
        ],
        dtype=np.float32,
    )


class ModelGenerator:
    name = "model"

    def __init__(self, model_dir_or_file: str | Path, device: str | None = None):
        path = Path(model_dir_or_file)
        if path.is_file():
            checkpoint_path = path
            model_dir = path.parent
        else:
            checkpoint_path = path / "model.pt"
            model_dir = path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer
        tok_path = model_dir / "tokenizer.json"
        if tok_path.exists():
            self.tokenizer = StrokeTokenizer.load(tok_path)
        else:
            self.tokenizer = StrokeTokenizer()

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "config" in ckpt:
            cfg = CoreConfig(**ckpt["config"])
        elif (model_dir / "config.json").exists():
            cfg = CoreConfig(**json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
        else:
            raise ValueError(f"Could not find model configuration in {checkpoint_path}")

        self.model = CoreModel(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        self.model.eval()

    def estimate_width(self, text: str, style: StyleSpec) -> float:
        """Rough estimate in x-heights for layout planning."""
        n_chars = max(len(text), 1)
        return float(n_chars * 0.65 * style.width_scale * style.letter_spacing)

    @torch.no_grad()
    def generate_line(self, req: LineRequest) -> Ink:
        text_ids = encode_text(req.text)
        t_text = torch.tensor([text_ids], dtype=torch.long, device=self.device)

        desc_vec = style_to_descriptor_vec(req.style, req.state)
        t_desc = torch.from_numpy(desc_vec[None]).float().to(self.device)

        # Style latent vector z: zero for generic, or random perturb based on neatness
        z_dim = self.model.c.style_dim
        z = torch.zeros((1, z_dim), device=self.device)

        # Context from previous line if available
        ctx_tensor = None
        if req.context and len(req.context[-1]) > 0:
            prev_ink = req.context[-1]
            prev_toks = self.tokenizer.encode(prev_ink).stack()
            max_c = self.model.c.ctx_len
            if len(prev_toks) > max_c:
                prev_toks = prev_toks[-max_c:]
            ctx_tensor = torch.from_numpy(prev_toks[None]).long().to(self.device)

        # Sampling with seed from req.rng
        seed = int(req.rng.integers(0, 2**31 - 1))
        torch_gen = torch.Generator(device=self.device).manual_seed(seed)

        # Temperature & sampling bias
        bias = 0.5 * req.state.neatness
        tokens = self.model.generate(
            text=t_text,
            z=z,
            desc=t_desc,
            ctx=ctx_tensor,
            temperature=req.state.temperature,
            bias=bias,
            max_points=2500,
            generator=torch_gen,
        )

        tokens_np = tokens.cpu().numpy()
        token_seq = TokenSeq.from_stack(tokens_np)
        ink = self.tokenizer.decode(token_seq, text=req.text)

        # Fill pressure & meta
        ink = ensure_pressure(ink)
        ink.meta["generator"] = self.name
        return ink
