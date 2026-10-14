"""Core online-handwriting representation shared by every module.

Conventions (IMPORTANT - every module relies on these):
  * ``points`` is float32 (N, 5): columns ``x, y, t, p, eos``.
  * y grows DOWNWARD (image convention, same as IAM-OnDB and browser Pointer Events).
  * ``t`` in seconds, non-decreasing.
  * ``p`` pressure in [0, 1]; NaN when the device did not record pressure.
  * ``eos`` = 1.0 when the pen is lifted AFTER this point (last point of a stroke).
  * Units depend on the stage:
      - raw:        device units (IAM-OnDB whiteboard units, CSS px, ...)
      - normalised: x-height == 1.0, baseline at y == 0, line starts near x == 0
                    (this is what generators produce and models consume)
      - page:       millimetres on the page (after layout)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

X, Y, T, P, EOS = range(5)
NCOLS = 5


@dataclass
class Ink:
    points: np.ndarray
    text: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != NCOLS:
            raise ValueError(f"points must be (N, {NCOLS}), got {pts.shape}")
        self.points = pts

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.points)

    @property
    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    @property
    def has_pressure(self) -> bool:
        return len(self) > 0 and not np.isnan(self.points[:, P]).all()

    def copy(self) -> "Ink":
        return Ink(self.points.copy(), self.text, dict(self.meta))

    def stroke_slices(self) -> list[slice]:
        """Slices of ``points`` for each pen-down stroke."""
        if len(self) == 0:
            return []
        ends = np.flatnonzero(self.points[:, EOS] > 0.5)
        if len(ends) == 0 or ends[-1] != len(self) - 1:
            ends = np.append(ends, len(self) - 1)
        out, start = [], 0
        for e in ends:
            out.append(slice(start, int(e) + 1))
            start = int(e) + 1
        return out

    def strokes(self) -> list[np.ndarray]:
        return [self.points[s] for s in self.stroke_slices()]

    @classmethod
    def from_strokes(cls, strokes: Iterable[np.ndarray], text: str = "", meta: dict | None = None) -> "Ink":
        """Build from a list of (M, 2..4) arrays: x, y[, t[, p]]. Missing t is filled at 100 Hz."""
        rows, t0 = [], 0.0
        for s in strokes:
            s = np.asarray(s, dtype=np.float64)
            if len(s) == 0:
                continue
            m = len(s)
            out = np.full((m, NCOLS), np.nan, dtype=np.float64)
            out[:, :2] = s[:, :2]
            if s.shape[1] >= 3:
                out[:, T] = s[:, 2]
            else:
                out[:, T] = t0 + np.arange(m) / 100.0
                t0 = out[-1, T] + 0.1
            if s.shape[1] >= 4:
                out[:, P] = s[:, 3]
            out[:, EOS] = 0.0
            out[-1, EOS] = 1.0
            rows.append(out)
        pts = np.concatenate(rows) if rows else np.zeros((0, NCOLS))
        return cls(pts.astype(np.float32), text, meta or {})

    @staticmethod
    def concat(inks: list["Ink"], text_sep: str = "") -> "Ink":
        inks = [i for i in inks if len(i)]
        if not inks:
            return Ink(np.zeros((0, NCOLS), np.float32))
        pts = np.concatenate([i.points for i in inks])
        return Ink(pts, text_sep.join(i.text for i in inks))

    # -------------------------------------------------------------- geometry
    def bbox(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)."""
        if len(self) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mn = self.xy.min(axis=0)
        mx = self.xy.max(axis=0)
        return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))

    def affine(self, m: np.ndarray) -> "Ink":
        """Apply a 2x3 affine matrix to x, y. Returns a new Ink."""
        m = np.asarray(m, dtype=np.float64)
        out = self.copy()
        xy = self.xy.astype(np.float64)
        out.points[:, :2] = (xy @ m[:, :2].T + m[:, 2]).astype(np.float32)
        return out

    def translate(self, dx: float, dy: float) -> "Ink":
        return self.affine(np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64))

    def scale(self, sx: float, sy: float | None = None) -> "Ink":
        sy = sx if sy is None else sy
        return self.affine(np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float64))

    def shear_slant(self, deg: float) -> "Ink":
        """Lean the writing by ``deg`` degrees to the right (positive) about the baseline y=0."""
        k = np.tan(np.radians(deg))
        # y down: points above baseline have y<0 and must move right -> x' = x - k*y
        return self.affine(np.array([[1, -k, 0], [0, 1, 0]], dtype=np.float64))

    def rotate(self, deg: float, cx: float = 0.0, cy: float = 0.0) -> "Ink":
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        r = np.array([[c, -s], [s, c]])
        t = np.array([cx, cy]) - r @ np.array([cx, cy])
        return self.affine(np.hstack([r, t[:, None]]))

    # -------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        strokes = []
        for s in self.strokes():
            d = {"x": s[:, X].round(4).tolist(), "y": s[:, Y].round(4).tolist(), "t": s[:, T].round(5).tolist()}
            if not np.isnan(s[:, P]).all():
                d["p"] = np.nan_to_num(s[:, P], nan=0.5).round(4).tolist()
            strokes.append(d)
        return {"format": "vibefont-ink/1", "text": self.text, "meta": self.meta, "strokes": strokes}

    @classmethod
    def from_dict(cls, d: dict) -> "Ink":
        strokes = []
        for s in d["strokes"]:
            cols = [s["x"], s["y"], s.get("t") or list(np.arange(len(s["x"])) / 100.0)]
            if "p" in s:
                cols.append(s["p"])
            strokes.append(np.stack([np.asarray(c, dtype=np.float64) for c in cols], axis=1))
        return cls.from_strokes(strokes, d.get("text", ""), d.get("meta", {}))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".json":
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        else:
            np.savez_compressed(path, points=self.points, text=self.text, meta=json.dumps(self.meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "Ink":
        path = Path(path)
        if path.suffix == ".json":
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        z = np.load(path, allow_pickle=False)
        return cls(z["points"], str(z["text"]), json.loads(str(z["meta"])))
