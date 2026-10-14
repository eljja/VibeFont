"""Stroke tokenizer: 1 point == 1 sequence position, 4 discrete channels.

Per point i the model sees / predicts:
    dx_i, dy_i : quantised displacement from the RECONSTRUCTED previous point
                 (error-feedback quantisation -> no cumulative drift)
    dt_i       : quantised time since previous point (log-spaced; long values = pen-up pauses)
    pen_i      : 0 = pen stays down after this point
                 1 = pen lifts after this point (end of stroke)
                 2 = end of line (last point)

Bins are non-uniform (mu-law by default, or fitted quantiles) because displacement
distributions are sharply peaked at zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ink import EOS, NCOLS, P, T, X, Y, Ink

PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2


@dataclass
class Bins:
    centers: np.ndarray  # sorted, float64

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float64)
        self._mid = (self.centers[1:] + self.centers[:-1]) / 2.0

    @property
    def n(self) -> int:
        return len(self.centers)

    def quantize(self, v: np.ndarray | float) -> np.ndarray:
        return np.searchsorted(self._mid, np.asarray(v, dtype=np.float64)).astype(np.int64)

    def dequantize(self, idx: np.ndarray | int) -> np.ndarray:
        return self.centers[np.asarray(idx)]

    @classmethod
    def mulaw(cls, n: int, max_abs: float, mu: float = 64.0) -> "Bins":
        """Symmetric mu-law bins with an exact zero bin (n is forced odd)."""
        n = n if n % 2 == 1 else n + 1
        u = np.linspace(-1.0, 1.0, n)
        c = np.sign(u) * ((1 + mu) ** np.abs(u) - 1) / mu * max_abs
        c[n // 2] = 0.0
        return cls(c)

    @classmethod
    def log_time(cls, n: int, dt_min: float = 1e-3, dt_max: float = 2.0) -> "Bins":
        return cls(np.concatenate([[0.0], np.geomspace(dt_min, dt_max, n - 1)]))

    @classmethod
    def fit_quantile(cls, values: np.ndarray, n: int) -> "Bins":
        q = np.quantile(np.asarray(values, np.float64), np.linspace(0, 1, n))
        return cls(np.unique(q))

    def to_list(self) -> list[float]:
        return self.centers.tolist()


@dataclass
class TokenSeq:
    dx: np.ndarray
    dy: np.ndarray
    dt: np.ndarray
    pen: np.ndarray

    def __len__(self) -> int:
        return len(self.dx)

    def stack(self) -> np.ndarray:
        """(N, 4) int64 array: dx, dy, dt, pen."""
        return np.stack([self.dx, self.dy, self.dt, self.pen], axis=1)

    @classmethod
    def from_stack(cls, a: np.ndarray) -> "TokenSeq":
        a = np.asarray(a, dtype=np.int64)
        return cls(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


class StrokeTokenizer:
    """Defaults assume normalised ink (x-height = 1) resampled at ~100 Hz."""

    def __init__(self, bx: Bins | None = None, by: Bins | None = None, bt: Bins | None = None):
        # Large |dx| only occurs at pen-up jumps between strokes/words.
        self.bx = bx or Bins.mulaw(255, max_abs=8.0)
        self.by = by or Bins.mulaw(255, max_abs=6.0)
        self.bt = bt or Bins.log_time(32)

    @property
    def sizes(self) -> dict[str, int]:
        return {"dx": self.bx.n, "dy": self.by.n, "dt": self.bt.n, "pen": 3}

    def encode(self, ink: Ink) -> TokenSeq:
        pts = ink.points.astype(np.float64)
        n = len(pts)
        dx = np.zeros(n, np.int64)
        dy = np.zeros(n, np.int64)
        rx = ry = 0.0  # reconstructed position, line origin at (0, 0)
        for i in range(n):
            dx[i] = self.bx.quantize(pts[i, X] - rx)
            rx += self.bx.centers[dx[i]]
            dy[i] = self.by.quantize(pts[i, Y] - ry)
            ry += self.by.centers[dy[i]]
        t = pts[:, T]
        dts = np.diff(t, prepend=t[0] if n else 0.0)
        dt = self.bt.quantize(np.clip(dts, 0.0, None))
        pen = np.where(pts[:, EOS] > 0.5, PEN_UP, PEN_DOWN).astype(np.int64)
        if n:
            pen[-1] = PEN_END
        return TokenSeq(dx, dy, dt, pen)

    def decode(self, toks: TokenSeq, text: str = "") -> Ink:
        n = len(toks)
        pts = np.zeros((n, NCOLS), np.float64)
        pts[:, X] = np.cumsum(self.bx.dequantize(toks.dx))
        pts[:, Y] = np.cumsum(self.by.dequantize(toks.dy))
        pts[:, T] = np.cumsum(self.bt.dequantize(toks.dt))
        pts[:, P] = np.nan
        pts[:, EOS] = (toks.pen != PEN_DOWN).astype(np.float64)
        return Ink(pts.astype(np.float32), text)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"bx": self.bx.to_list(), "by": self.by.to_list(), "bt": self.bt.to_list()}

    @classmethod
    def from_dict(cls, d: dict) -> "StrokeTokenizer":
        return cls(Bins(d["bx"]), Bins(d["by"]), Bins(d["bt"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StrokeTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def roundtrip_error(tok: StrokeTokenizer, ink: Ink) -> dict[str, float]:
    """Point-wise reconstruction error (same units as the ink)."""
    rec = tok.decode(tok.encode(ink))
    e = np.linalg.norm(rec.xy - ink.xy, axis=1)
    return {"mean": float(e.mean()) if len(e) else 0.0, "max": float(e.max()) if len(e) else 0.0}
