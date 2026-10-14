"""Track A core stroke model (PyTorch reference implementation).

Sequence layout fed to the decoder (one position per token):

    [style_1..style_S] [ctx_1..ctx_C] [BOS] [tgt_1 .. tgt_{N-1}]
                                        |      |          |
                                     predicts tgt_1 ... tgt_N

  * style tokens: from (style latent z, descriptor vector) -> "who writes, in which vibe"
  * ctx tokens:   tail of the PREVIOUS line of the same document -> long-text coherence
                  (left-padded to a fixed length, padded keys are masked)
  * tgt tokens:   the line being written; teacher forcing during training

Each point token = E_dx + E_dy + E_dt + E_pen + E_segment (+ pointer char embedding when
CTC alignments are available). Output heads are autoregressively factorised:
    P(pen) P(adv) P(dx) P(dy | dx) P(dt | dx, dy)

Self-attention uses RoPE and an explicit KV cache so the step function can be exported
to ONNX for CPU / iGPU / WebGPU inference.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoreConfig:
    vocab_size: int
    n_dx: int
    n_dy: int
    n_dt: int
    n_pen: int = 3
    d_model: int = 256
    n_enc: int = 3
    n_dec: int = 6
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    style_dim: int = 64
    n_style_tokens: int = 4
    desc_dim: int = 8
    n_writers: int = 1
    ctx_len: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "xs": dict(d_model=192, n_enc=2, n_dec=4, n_heads=4),  # ~3M   smoke tests
    "s": dict(d_model=256, n_enc=3, n_dec=6, n_heads=4),  # ~9M   first real model
    "m": dict(d_model=384, n_enc=4, n_dec=8, n_heads=6),  # ~26M  upper bound for low-spec target
}


# ----------------------------------------------------------------------------- utils
def rope(positions: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
    f = positions.float()[:, None] * inv[None]
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class FFN(nn.Module):
    def __init__(self, d: int, mult: int, p: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(p), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class SelfAttn(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.h, self.p = h, p
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, cos, sin, mask=None, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None and cache[0] is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        drop = self.p if self.training else 0.0
        if mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        elif T > 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        return self.o(y.transpose(1, 2).reshape(B, T, D)), new_cache


class CrossAttn(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.h, self.p = h, p
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def memory_kv(self, mem):
        B, L, D = mem.shape
        k, v = self.kv(mem).view(B, L, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, kv, mem_mask=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=mem_mask,
                                           dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class DecBlock(nn.Module):
    def __init__(self, c: CoreConfig):
        super().__init__()
        d = c.d_model
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = SelfAttn(d, c.n_heads, c.dropout)
        self.ca = CrossAttn(d, c.n_heads, c.dropout)
        self.ff = FFN(d, c.ff_mult, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, kv, mem_mask, mask=None, cache=None):
        a, cache = self.sa(self.n1(x), cos, sin, mask, cache)
        x = x + self.drop(a)
        x = x + self.drop(self.ca(self.n2(x), kv, mem_mask))
        x = x + self.drop(self.ff(self.n3(x)))
        return x, cache


# ----------------------------------------------------------------------------- model
class CoreModel(nn.Module):
    SEG_CTX, SEG_TGT = 0, 1

    def __init__(self, c: CoreConfig):
        super().__init__()
        self.c = c
        d = c.d_model
        # text encoder
        self.tok = nn.Embedding(c.vocab_size, d)
        self.tpos = nn.Embedding(512, d)
        layer = nn.TransformerEncoderLayer(d, c.n_heads, d * c.ff_mult, c.dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, c.n_enc, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        # style
        self.writer = nn.Embedding(c.n_writers, c.style_dim)
        self.style_mlp = nn.Sequential(nn.Linear(c.style_dim + c.desc_dim, d), nn.GELU(),
                                       nn.Linear(d, c.n_style_tokens * d))
        # point embeddings
        self.e_dx, self.e_dy = nn.Embedding(c.n_dx, d), nn.Embedding(c.n_dy, d)
        self.e_dt, self.e_pen = nn.Embedding(c.n_dt, d), nn.Embedding(c.n_pen, d)
        self.e_seg = nn.Embedding(2, d)
        self.bos = nn.Parameter(torch.zeros(d))
        self.ptr_cur, self.ptr_next = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList(DecBlock(c) for _ in range(c.n_dec))
        self.norm = nn.LayerNorm(d)
        # heads
        self.h_pen, self.h_adv, self.h_dx = nn.Linear(d, c.n_pen), nn.Linear(d, 2), nn.Linear(d, c.n_dx)
        self.c_dx, self.c_dy = nn.Embedding(c.n_dx, d), nn.Embedding(c.n_dy, d)
        self.h_dy = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c.n_dy))
        self.h_dt = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c.n_dt))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------ pieces
    def encode_text(self, text, text_mask):
        """text (B, L) ids, text_mask (B, L) True = real token."""
        pos = torch.arange(text.shape[1], device=text.device).clamp(max=511)
        h = self.tok(text) + self.tpos(pos)[None]
        h = self.enc(h, src_key_padding_mask=~text_mask)
        return self.enc_norm(h)

    def style_tokens(self, z, desc):
        B = z.shape[0]
        return self.style_mlp(torch.cat([z, desc], -1)).view(B, self.c.n_style_tokens, self.c.d_model)

    def embed_points(self, toks, seg):
        """toks (B, T, 4) long: dx, dy, dt, pen."""
        return (self.e_dx(toks[..., 0]) + self.e_dy(toks[..., 1]) + self.e_dt(toks[..., 2])
                + self.e_pen(toks[..., 3]) + self.e_seg.weight[seg])

    def pointer_cond(self, mem, char_idx):
        L = mem.shape[1]
        cur = torch.gather(mem, 1, char_idx.clamp(0, L - 1)[..., None].expand(-1, -1, mem.shape[2]))
        nxt = torch.gather(mem, 1, (char_idx + 1).clamp(0, L - 1)[..., None].expand(-1, -1, mem.shape[2]))
        return self.ptr_cur(cur) + self.ptr_next(nxt)

    def heads(self, h, dx=None, dy=None):
        out = {"pen": self.h_pen(h), "adv": self.h_adv(h), "dx": self.h_dx(h)}
        if dx is not None:
            out["dy"] = self.h_dy(h + self.c_dx(dx))
            if dy is not None:
                out["dt"] = self.h_dt(h + self.c_dx(dx) + self.c_dy(dy))
        return out

    # ------------------------------------------------------------ training
    def forward(self, batch: dict) -> dict:
        """batch keys:
        text (B,L) text_mask (B,L) writer (B,) desc (B,Dd) z_drop (B,) bool
        ctx (B,C,4) ctx_mask (B,C)   tgt (B,N,4) tgt_mask (B,N)
        char_idx (B,N) optional, adv (B,N) optional
        """
        c = self.c
        mem = self.encode_text(batch["text"], batch["text_mask"])
        mem_mask = batch["text_mask"][:, None, None, :]
        z = self.writer(batch["writer"])
        if "z_drop" in batch:
            z = z.masked_fill(batch["z_drop"][:, None], 0.0)
        st = self.style_tokens(z, batch["desc"])
        B, N = batch["tgt"].shape[:2]
        C = batch["ctx"].shape[1]
        ctx_e = self.embed_points(batch["ctx"], self.SEG_CTX)
        tgt_in = self.embed_points(batch["tgt"][:, :-1], self.SEG_TGT)
        tgt_in = torch.cat([self.bos.expand(B, 1, -1), tgt_in], 1)
        if batch.get("char_idx") is not None:
            tgt_in = tgt_in + self.pointer_cond(mem, batch["char_idx"])
        x = torch.cat([st, ctx_e, tgt_in], 1)
        S = st.shape[1]
        T = x.shape[1]
        valid = torch.cat([torch.ones(B, S, dtype=torch.bool, device=x.device), batch["ctx_mask"],
                           torch.ones(B, N, dtype=torch.bool, device=x.device)], 1)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        cos, sin = rope(torch.arange(T, device=x.device), c.d_model // c.n_heads)
        for blk in self.blocks:
            x, _ = blk(x, cos, sin, [blk.ca.memory_kv(mem)][0], mem_mask, mask)
        h = self.norm(x[:, S + C:])
        tgt = batch["tgt"]
        o = self.heads(h, tgt[..., 0], tgt[..., 1])
        m = batch["tgt_mask"].float()
        down = m * (tgt[..., 3] == 0).float()  # dx/dy of pen-down AND pen-up points are both real

        def ce(logits, target, w):
            l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="none")
            return (l * w.reshape(-1)).sum() / w.sum().clamp(min=1)

        losses = {"pen": ce(o["pen"], tgt[..., 3], m), "dx": ce(o["dx"], tgt[..., 0], m),
                  "dy": ce(o["dy"], tgt[..., 1], m), "dt": ce(o["dt"], tgt[..., 2], m)}
        if batch.get("adv") is not None:
            losses["adv"] = ce(o["adv"], batch["adv"], m)
        losses["loss"] = sum(losses.values())
        del down
        return losses

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, text, z, desc, ctx=None, temperature: float = 1.0, max_points: int = 2500,
                 bias: float = 0.0, generator: torch.Generator | None = None) -> torch.Tensor:
        """Batch size 1. text (1,L), z (1,Ds), desc (1,Dd), ctx (1,C,4) or None.

        temperature scales every head; ``bias`` (Graves' sampling bias) sharpens further.
        Returns (N, 4) long tokens.
        """
        self.eval()
        c, dev = self.c, text.device
        mem = self.encode_text(text, torch.ones_like(text, dtype=torch.bool))
        kvs = [blk.ca.memory_kv(mem) for blk in self.blocks]
        parts = [self.style_tokens(z, desc)]
        if ctx is not None and ctx.shape[1] > 0:
            parts.append(self.embed_points(ctx, self.SEG_CTX))
        parts.append(self.bos.view(1, 1, -1))
        x = torch.cat(parts, 1)
        caches: list = [(None, None)] * len(self.blocks)
        pos = 0
        out = []
        tau = max(temperature, 1e-3) / (1.0 + bias)

        def sample(logits):
            p = F.softmax(logits.float() / tau, -1)
            return torch.multinomial(p, 1, generator=generator)[:, 0]

        for _ in range(max_points):
            T = x.shape[1]
            cos, sin = rope(torch.arange(pos, pos + T, device=dev), c.d_model // c.n_heads)
            h = x
            for i, blk in enumerate(self.blocks):
                h, caches[i] = blk(h, cos, sin, kvs[i], None, None, caches[i])
            pos += T
            h = self.norm(h[:, -1])
            o = self.heads(h)
            pen, dx = sample(o["pen"]), sample(o["dx"])
            dy = sample(self.h_dy(h + self.c_dx(dx)))
            dt = sample(self.h_dt(h + self.c_dx(dx) + self.c_dy(dy)))
            tok = torch.stack([dx, dy, dt, pen], -1)  # (1, 4)
            out.append(tok[0])
            if int(pen) == 2:
                break
            x = self.embed_points(tok[:, None], self.SEG_TGT)
        return torch.stack(out) if out else torch.zeros(0, 4, dtype=torch.long, device=dev)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build(preset: str, **kw) -> CoreModel:
    cfg = CoreConfig(**{**PRESETS[preset], **kw})
    return CoreModel(cfg)


def math_check() -> None:  # pragma: no cover - quick manual sanity
    print(math.pi)
