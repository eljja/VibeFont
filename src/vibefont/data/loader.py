"""PyTorch Dataset & DataLoader utilities for training the stroke model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..dataset import LineStore
from ..descriptors import as_vector
from ..text_codec import encode as encode_text
from ..tokenizer import StrokeTokenizer, TokenSeq


class StrokeDataset(Dataset):
    """Wraps a LineStore and StrokeTokenizer into a PyTorch Dataset."""

    def __init__(
        self,
        store: LineStore,
        tokenizer: StrokeTokenizer,
        writer_map: dict[str, int] | None = None,
        max_seq_len: int = 2000,
        max_ctx_len: int = 128,
    ):
        self.store = store
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_ctx_len = max_ctx_len
        self.prev_indices = store.previous_line()

        # Build or use existing writer map
        if writer_map is None:
            writers = sorted({r["writer"] for r in store.rows})
            self.writer_map = {w: i for i, w in enumerate(writers)}
        else:
            self.writer_map = writer_map

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.store.rows[idx]
        ink = self.store.ink(idx)
        toks = self.tokenizer.encode(ink)
        tgt_stack = toks.stack()

        # Truncate if excessively long
        if len(tgt_stack) > self.max_seq_len:
            tgt_stack = tgt_stack[: self.max_seq_len]

        # Context: tail of the previous line in the same document
        prev_idx = self.prev_indices[idx]
        if prev_idx >= 0:
            prev_ink = self.store.ink(prev_idx)
            prev_toks = self.tokenizer.encode(prev_ink).stack()
            ctx_stack = prev_toks[-self.max_ctx_len :]
        else:
            ctx_stack = np.zeros((0, 4), dtype=np.int64)

        text_ids = encode_text(row["text"])
        desc_vec = as_vector(row["desc"])
        writer_id = self.writer_map.get(row["writer"], 0)

        N = len(tgt_stack)
        M = max(1, len(text_ids) - 2)
        if N > 0:
            t_idx = np.arange(N, dtype=np.int64)
            char_idx = np.clip(1 + (t_idx * M) // N, 1, M)
            adv = np.zeros(N, dtype=np.int64)
            if N > 1:
                adv[:-1] = (char_idx[1:] > char_idx[:-1]).astype(np.int64)
            adv[-1] = 1
        else:
            char_idx = np.zeros(0, dtype=np.int64)
            adv = np.zeros(0, dtype=np.int64)

        return {
            "text": text_ids,
            "tgt": tgt_stack,
            "char_idx": char_idx,
            "adv": adv,
            "ctx": ctx_stack,
            "writer": writer_id,
            "desc": desc_vec,
        }


def collate_stroke_batch(batch: list[dict[str, Any]], z_drop_prob: float = 0.1) -> dict[str, torch.Tensor]:
    """Collates variable-length items into padded PyTorch tensors."""
    B = len(batch)

    # 1. Text padding
    text_lens = [len(item["text"]) for item in batch]
    max_text_len = max(text_lens)
    text_tensor = torch.zeros((B, max_text_len), dtype=torch.long)
    text_mask = torch.zeros((B, max_text_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        L = len(item["text"])
        text_tensor[i, :L] = torch.tensor(item["text"], dtype=torch.long)
        text_mask[i, :L] = True

    # 2. Target stroke points & pointer alignment padding
    tgt_lens = [len(item["tgt"]) for item in batch]
    max_tgt_len = max(max(tgt_lens), 1)
    tgt_tensor = torch.zeros((B, max_tgt_len, 4), dtype=torch.long)
    tgt_mask = torch.zeros((B, max_tgt_len), dtype=torch.bool)
    char_idx_tensor = torch.ones((B, max_tgt_len), dtype=torch.long)
    adv_tensor = torch.zeros((B, max_tgt_len), dtype=torch.long)
    for i, item in enumerate(batch):
        N = len(item["tgt"])
        if N > 0:
            tgt_tensor[i, :N] = torch.from_numpy(item["tgt"])
            tgt_mask[i, :N] = True
            if "char_idx" in item:
                char_idx_tensor[i, :N] = torch.from_numpy(item["char_idx"])
            if "adv" in item:
                adv_tensor[i, :N] = torch.from_numpy(item["adv"])

    # 3. Context points padding (left-padded so recent points align near the boundary)
    ctx_lens = [len(item["ctx"]) for item in batch]
    max_ctx_len = max(max(ctx_lens), 1)
    ctx_tensor = torch.zeros((B, max_ctx_len, 4), dtype=torch.long)
    ctx_mask = torch.zeros((B, max_ctx_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        C = len(item["ctx"])
        if C > 0:
            ctx_tensor[i, max_ctx_len - C :] = torch.from_numpy(item["ctx"])
            ctx_mask[i, max_ctx_len - C :] = True

    # 4. Meta & Style
    writer_tensor = torch.tensor([item["writer"] for item in batch], dtype=torch.long)
    desc_tensor = torch.from_numpy(np.stack([item["desc"] for item in batch])).float()

    # Writer dropout (CFG / unconditioned writing)
    if z_drop_prob > 0.0:
        z_drop = torch.rand(B) < z_drop_prob
    else:
        z_drop = torch.zeros(B, dtype=torch.bool)

    return {
        "text": text_tensor,
        "text_mask": text_mask,
        "tgt": tgt_tensor,
        "tgt_mask": tgt_mask,
        "char_idx": char_idx_tensor,
        "adv": adv_tensor,
        "ctx": ctx_tensor,
        "ctx_mask": ctx_mask,
        "writer": writer_tensor,
        "desc": desc_tensor,
        "z_drop": z_drop,
    }
