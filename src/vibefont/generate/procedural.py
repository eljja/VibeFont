"""Placeholder generator (NO machine learning) built on Hershey single-stroke fonts.

Purpose: exercise the full product pipeline (prompt -> layout -> drift -> renderer -> file)
before the stroke model is trained, and serve as a sanity baseline. It fakes natural
variation with smooth random deformation fields, per-word jitter and power-law timing.
It will be replaced by ``generate.model.ModelGenerator``; keep the interface identical.

Limitations: Latin only (Hershey has no Hangul), letter shapes are fixed templates.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..descriptors import slant_deg
from ..ink import P, Ink
from ..preprocess import pseudo_pressure
from ..vibe.spec import StyleSpec
from .base import LineRequest
from .hangul import hangul_char_strokes, is_hangul
from .kinematics import densify, strokes_to_ink


class _Font:
    def __init__(self, name: str):
        from HersheyFonts import HersheyFonts  # optional dependency: pip install Hershey-Fonts

        self.f = HersheyFonts()
        self.f.load_default_font(name)
        xs = np.concatenate([np.asarray(s, float) for s in self.f.strokes_for_text("x")])
        self.base = xs[:, 1].max()
        self.xh = xs[:, 1].max() - xs[:, 1].min()
        self.slant = slant_deg(Ink.from_strokes(self.word("lldhkbtf")[0]))
        self._cache: dict[str, float] = {}

    def word(self, word: str) -> tuple[list[np.ndarray], float]:
        if not any(is_hangul(c) for c in word):
            strokes = [np.asarray(s, np.float64) for s in self.f.strokes_for_text(word)]
            strokes = [s for s in strokes if len(s)]
            if not strokes:
                return [], 0.6 * len(word)
            allp = np.concatenate(strokes)
            x0 = allp[:, 0].min()
            out = [np.stack([(s[:, 0] - x0) / self.xh, (s[:, 1] - self.base) / self.xh], axis=1) for s in strokes]
            width = (allp[:, 0].max() - x0) / self.xh
            return out, float(width)

        # Mixed or pure Hangul
        runs: list[tuple[bool, str]] = []
        cur_h: bool | None = None
        cur_text: list[str] = []
        for ch in word:
            h = is_hangul(ch)
            if cur_h is None or h == cur_h:
                cur_h = h
                cur_text.append(ch)
            else:
                runs.append((cur_h, "".join(cur_text)))
                cur_h = h
                cur_text = [ch]
        if cur_text and cur_h is not None:
            runs.append((cur_h, "".join(cur_text)))

        all_strokes: list[np.ndarray] = []
        cursor_x = 0.0
        gap = 0.05

        for is_h, text in runs:
            if is_h:
                for ch in text:
                    ch_strokes = hangul_char_strokes(ch)
                    if ch_strokes:
                        for s in ch_strokes:
                            sp = s.copy()
                            sp[:, 0] += cursor_x
                            all_strokes.append(sp)
                        cursor_x += 1.0 + gap
                    else:
                        cursor_x += 0.6
            else:
                h_strokes = [np.asarray(s, np.float64) for s in self.f.strokes_for_text(text)]
                h_strokes = [s for s in h_strokes if len(s)]
                if h_strokes:
                    allp = np.concatenate(h_strokes)
                    x0 = allp[:, 0].min()
                    w = (allp[:, 0].max() - x0) / self.xh
                    for s in h_strokes:
                        sp = np.stack([(s[:, 0] - x0) / self.xh + cursor_x, (s[:, 1] - self.base) / self.xh], axis=1)
                        all_strokes.append(sp)
                    cursor_x += w + gap
                else:
                    cursor_x += 0.6 * len(text)

        if not all_strokes:
            return [], 0.6 * len(word)

        allp = np.concatenate(all_strokes)
        x0 = float(allp[:, 0].min())
        for s in all_strokes:
            s[:, 0] -= x0
        total_width = float(allp[:, 0].max())
        return all_strokes, max(total_width, 0.1)

    def width(self, word: str) -> float:
        if word not in self._cache:
            self._cache[word] = self.word(word)[1]
        return self._cache[word]


def _merge_touching(strokes: list[np.ndarray], p_join: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Hershey script glyphs touch end-to-start; join them (cursive) with probability p_join."""
    out: list[np.ndarray] = []
    for s in strokes:
        if out and np.linalg.norm(out[-1][-1] - s[0]) < 1e-6 and rng.random() < p_join:
            out[-1] = np.concatenate([out[-1], s[1:]])
        else:
            out.append(s)
    return out


def _deform(strokes: list[np.ndarray], amp: float, rng: np.random.Generator, n: int = 4) -> list[np.ndarray]:
    """Smooth random displacement field: same word never comes out identical."""
    w = rng.uniform(1.0, 4.5, (2, n))
    wy = rng.uniform(0.3, 2.0, (2, n))
    ph = rng.uniform(0, 2 * np.pi, (2, n))
    a = rng.normal(0.0, amp / np.sqrt(n), (2, n))
    out = []
    for s in strokes:
        x, y = s[:, :1], s[:, 1:2]
        dx = (a[0] * np.sin(w[0] * x + wy[0] * y + ph[0])).sum(1)
        dy = (a[1] * np.sin(w[1] * x + wy[1] * y + ph[1])).sum(1)
        out.append(s + np.stack([dx, dy], axis=1))
    return out


class ProceduralGenerator:
    name = "procedural"

    def __init__(self) -> None:
        self.fonts = {"cursive": _Font("scripts"), "print": _Font("futural")}

    def _font(self, style: StyleSpec) -> _Font:
        return self.fonts["cursive" if style.connectedness >= 0.4 else "print"]

    def estimate_width(self, text: str, style: StyleSpec) -> float:
        f = self._font(style)
        words = text.split()
        w = sum(f.width(wd) for wd in words) * style.width_scale * style.letter_spacing
        return float(w + 0.8 * style.word_spacing * max(len(words) - 1, 0))

    def generate_line(self, req: LineRequest) -> Ink:
        st, ls, rng = req.style, req.state, req.rng
        f = self._font(st)
        loose = 1.0 - ls.neatness
        all_strokes: list[np.ndarray] = []
        cursor = 0.0
        for word in req.text.split():
            strokes, width = f.word(word)
            if not strokes:
                cursor += width
                continue
            strokes = _merge_touching(strokes, 0.4 + 0.6 * st.connectedness, rng)
            strokes = [densify(s, 0.03) for s in strokes]
            if st.roundness > 0.05:
                sig = 1.0 + 3.0 * st.roundness
                strokes = [np.stack([gaussian_filter1d(s[:, 0], sig, mode="nearest"),
                                     gaussian_filter1d(s[:, 1], sig, mode="nearest")], 1) if len(s) > 4 else s
                           for s in strokes]
            strokes = _deform(strokes, 0.02 + 0.09 * loose, rng)
            # per-word jitter
            sc = float(np.exp(rng.normal(0.0, 0.02 + 0.04 * loose)))
            rot = np.radians(rng.normal(0.0, 0.5 + 1.5 * loose))
            slant = ls.slant_deg - f.slant + rng.normal(0.0, 1.0 + 2.5 * (1 - st.consistency))
            k = np.tan(np.radians(slant))
            dy = rng.normal(0.0, 0.03 + 0.08 * st.baseline_wobble)
            sx = st.width_scale * st.letter_spacing
            c, s_ = np.cos(rot), np.sin(rot)
            placed = []
            for s in strokes:
                x = s[:, 0] * sx * sc
                y = s[:, 1] * sc
                x = x - k * y
                placed.append(np.stack([c * x - s_ * y, s_ * x + c * y + dy], axis=1))
            allp = np.concatenate(placed)
            x0, x1 = allp[:, 0].min(), allp[:, 0].max()
            all_strokes.extend(p - np.array([x0 - cursor, 0.0]) for p in placed)
            cursor += (x1 - x0) + 0.9 * st.word_spacing * ls.word_spacing_scale * float(np.exp(rng.normal(0, 0.12)))
        ink = strokes_to_ink(all_strokes, st.speed, rng, text=req.text)
        if len(ink):
            # micro tremor along the stroke (smoothed noise)
            trem = gaussian_filter1d(rng.standard_normal((len(ink), 2)), 3.0, axis=0) * (0.004 + 0.01 * loose)
            ink.points[:, :2] += trem.astype(np.float32)
            ink.points[:, P] = np.clip(pseudo_pressure(ink) * (0.6 + 0.8 * st.pressure), 0.02, 1.0)
        ink.meta["generator"] = self.name
        return ink
