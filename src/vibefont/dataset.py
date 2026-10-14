"""On-disk dataset format shared by every source (IAM-OnDB, collector app, synthetic).

A dataset is a directory with:
    lines.npz    points (sum_N, 5) float32 concatenated NORMALISED lines, offsets (L+1,) int64
    index.jsonl  one JSON per line: {"i", "text", "writer", "doc", "line_no", "source", "desc": {...}}
                 ``doc`` + ``line_no`` let training pair a line with the previous line of the
                 same document (priming / long-text coherence).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .descriptors import measure
from .ink import Ink


class DatasetWriter:
    def __init__(self, out_dir: str | Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.points: list[np.ndarray] = []
        self.rows: list[dict] = []

    def add(self, ink: Ink, writer: str, doc: str, line_no: int, source: str) -> None:
        if len(ink) < 2:
            return
        row = {"i": len(self.rows), "text": ink.text, "writer": writer, "doc": doc,
               "line_no": line_no, "source": source, "n": len(ink),
               "desc": {k: round(v, 4) for k, v in measure(ink).items()}}
        self.points.append(ink.points.astype(np.float32))
        self.rows.append(row)

    def close(self) -> None:
        offsets = np.zeros(len(self.points) + 1, np.int64)
        offsets[1:] = np.cumsum([len(p) for p in self.points])
        pts = np.concatenate(self.points) if self.points else np.zeros((0, 5), np.float32)
        np.savez_compressed(self.out / "lines.npz", points=pts, offsets=offsets)
        with open(self.out / "index.jsonl", "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass
class LineStore:
    points: np.ndarray
    offsets: np.ndarray
    rows: list[dict]

    @classmethod
    def load(cls, d: str | Path) -> "LineStore":
        d = Path(d)
        z = np.load(d / "lines.npz")
        rows = [json.loads(x) for x in (d / "index.jsonl").read_text(encoding="utf-8").splitlines() if x]
        return cls(z["points"], z["offsets"], rows)

    def __len__(self) -> int:
        return len(self.rows)

    def ink(self, i: int) -> Ink:
        a, b = self.offsets[i], self.offsets[i + 1]
        return Ink(self.points[a:b], self.rows[i]["text"], {"writer": self.rows[i]["writer"]})

    def previous_line(self) -> list[int]:
        """For each line, index of the previous line of the same document, or -1."""
        key = {(r["doc"], r["line_no"]): r["i"] for r in self.rows}
        return [key.get((r["doc"], r["line_no"] - 1), -1) for r in self.rows]
