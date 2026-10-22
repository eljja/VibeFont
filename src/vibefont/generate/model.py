"""ModelGenerator: bridges the trained CoreModel into the StrokeGenerator protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..descriptors import DESCRIPTOR_NAMES
from ..document.drift import LineState
from ..generate.base import LineRequest, StrokeGenerator
from ..generate.procedural import ProceduralGenerator
from ..ink import EOS, Ink, P, T, X, Y
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
        self._scaffold = ProceduralGenerator()

    def estimate_width(self, text: str, style: StyleSpec) -> float:
        """Estimate line width in x-heights for layout planning."""
        return self._scaffold.estimate_width(text, style)

    @torch.no_grad()
    def generate_line(self, req: LineRequest) -> Ink:
        if not req.text.strip():
            return Ink(np.zeros((0, 5), dtype=np.float32), text=req.text, meta={"generator": self.name})

        # 1. Generate structural online stroke scaffold (handles all 11,172 Hangul syllables + Latin)
        base_ink = self._scaffold.generate_line(req)
        if len(base_ink) < 2:
            base_ink.meta["generator"] = self.name
            return base_ink

        base_stack = self.tokenizer.encode(base_ink).stack()

        text_ids = encode_text(req.text)
        t_text = torch.tensor([text_ids], dtype=torch.long, device=self.device)

        desc_vec = style_to_descriptor_vec(req.style, req.state)
        t_desc = torch.from_numpy(desc_vec[None]).float().to(self.device)

        # Style latent vector z: stochastic writer variation scaled by (1 - neatness)
        z_dim = self.model.c.style_dim
        z_noise = req.rng.normal(0.0, 0.25 * (1.15 - req.state.neatness), size=(1, z_dim)).astype(np.float32)
        z = torch.from_numpy(z_noise).to(self.device)

        # Context from previous line if available
        ctx_tensor = None
        max_c = self.model.c.ctx_len
        if req.context and len(req.context[-1]) > 0:
            prev_ink = req.context[-1]
            prev_toks = self.tokenizer.encode(prev_ink).stack()
            if len(prev_toks) > max_c:
                prev_toks = prev_toks[-max_c:]
            ctx_tensor = torch.from_numpy(prev_toks[None]).long().to(self.device)

        seed = int(req.rng.integers(0, 2**31 - 1))
        torch_gen = torch.Generator(device=self.device).manual_seed(seed)

        sigma_bins = 1.2 + 1.0 * (1.0 - req.state.neatness)
        chunk_size = 1500
        mod_chunks: list[np.ndarray] = []
        for start in range(0, len(base_stack), chunk_size):
            chunk = base_stack[start : start + chunk_size]
            t_chunk = torch.from_numpy(chunk[None]).long().to(self.device)
            mod_chunk = self.model.modulate(
                text=t_text,
                z=z,
                desc=t_desc,
                base_toks=t_chunk,
                ctx=ctx_tensor,
                temperature=req.state.temperature,
                sigma_bins=sigma_bins,
                generator=torch_gen,
            )
            mod_chunk_np = mod_chunk.cpu().numpy()
            mod_chunks.append(mod_chunk_np)
            tail = mod_chunk_np[-max_c:]
            ctx_tensor = torch.from_numpy(tail[None]).long().to(self.device)

        mod_np = np.concatenate(mod_chunks, axis=0)
        mod_ink = self.tokenizer.decode(TokenSeq.from_stack(mod_np), text=req.text)

        # Blend neural stroke deltas with structural stroke anchors so cumulative drift is zero
        n_pts = min(len(base_ink), len(mod_ink))
        pts = base_ink.points[:n_pts].copy()
        mod_pts = mod_ink.points[:n_pts]

        # Transfer neural velocity/timing profile
        pts[:, T] = mod_pts[:, T]

        ink_tmp = Ink(pts, text=req.text)
        for sl in ink_tmp.stroke_slices():
            m_xy = mod_pts[sl, :2]
            b_xy = pts[sl, :2]
            if len(b_xy) < 2:
                continue
            # Brownian-bridge endpoint-pinned neural residual: preserves closed loops (ㅇ, o, a, e)
            # while injecting the trained CoreModel's stroke curvature and rhythm in the interior
            rel_mod = m_xy - m_xy[0:1]
            rel_base = b_xy - b_xy[0:1]
            t_lin = np.linspace(0.0, 1.0, len(rel_mod), dtype=np.float32)[:, None]
            drift_corr = t_lin * (rel_mod[-1:] - rel_base[-1:])
            rel_mod_pinned = rel_mod - 0.80 * drift_corr
            blended_rel = 0.60 * rel_base + 0.40 * rel_mod_pinned
            if len(blended_rel) >= 3:
                blended_rel[1:-1] = (
                    0.2 * blended_rel[:-2] + 0.6 * blended_rel[1:-1] + 0.2 * blended_rel[2:]
                )
            pts[sl, :2] = b_xy[0:1] + blended_rel

        ink = ensure_pressure(Ink(pts, text=req.text))
        ink.meta["generator"] = self.name
        return ink
