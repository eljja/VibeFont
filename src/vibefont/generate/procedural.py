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
from ..ink import P, X, Ink
from ..preprocess import pseudo_pressure
from ..vibe.spec import StyleSpec
from .base import LineRequest
from .hangul import WriterProfile, hangul_char_strokes, is_hangul, sample_writer_profile
from .kinematics import densify, strokes_to_ink
from .ligature import PlacedGlyph, connect_word


class _Font:
    def __init__(self, name: str):
        from HersheyFonts import HersheyFonts  # optional dependency: pip install Hershey-Fonts

        self.f = HersheyFonts()
        self.f.load_default_font(name)
        xs = np.concatenate([np.asarray(s, float) for s in self.f.strokes_for_text("x")])
        self.base = xs[:, 1].max()
        self.xh = xs[:, 1].max() - xs[:, 1].min()
        raw_l = [np.asarray(s, float) for s in self.f.strokes_for_text("lldhkbtf") if len(s)]
        norm_l = [np.stack([(s[:, 0]) / self.xh, (s[:, 1] - self.base) / self.xh], axis=1) for s in raw_l]
        self.slant = slant_deg(Ink.from_strokes(norm_l))
        self._k_unslant = float(np.tan(np.radians(self.slant)))
        self._cache: dict[str, float] = {}

    def _norm_hershey(self, strokes: list[np.ndarray], x0: float) -> list[np.ndarray]:
        out = []
        for s in strokes:
            nx = (s[:, 0] - x0) / self.xh
            ny = (s[:, 1] - self.base) / self.xh
            # Remove Hershey's intrinsic slant so both Latin and Hangul are in upright 0-deg space
            nx = nx + self._k_unslant * ny
            out.append(np.stack([nx, ny], axis=1))
        return out

    def place_glyphs(
        self,
        word: str,
        rng: np.random.Generator,
        connectedness: float = 0.5,
        neatness: float = 0.6,
        profile: WriterProfile | None = None,
    ) -> list[PlacedGlyph]:
        clumsy = max(0.45, 1.65 - 1.25 * float(np.clip(neatness, 0.0, 1.0)))
        prof_w = float(np.clip(profile.w_scale if profile is not None else 1.0, 0.88, 1.14))
        prof_h = float(np.clip(0.98 + 0.25 * ((profile.h_scale if profile is not None else 1.0) - 1.0), 0.92, 1.08))

        placed_glyphs: list[PlacedGlyph] = []
        cursor_x = 0.0
        for ch in word:
            ch_raw = [np.asarray(s, np.float64) for s in self.f.strokes_for_text(ch) if len(s)]
            if not ch_raw:
                cursor_x += 0.35
                continue
            all_c = np.concatenate(ch_raw)
            cx0 = float(all_c[:, 0].min())
            norm_c = self._norm_hershey(ch_raw, cx0)
            all_n = np.concatenate(norm_c)
            min_nx = float(all_n[:, 0].min())
            max_nx = float(all_n[:, 0].max())
            cw = max(max_nx - min_nx, 0.18)

            # Clean per-character proportion variation without CAPTCHA-like warping
            sx = prof_w * float(np.clip(np.exp(rng.normal(0.0, 0.038 * clumsy)), 0.90, 1.12))
            sy = prof_h * float(np.clip(np.exp(rng.normal(0.0, 0.038 * clumsy)), 0.90, 1.12))
            dy = float(np.clip(rng.normal(0.0, 0.018 * clumsy), -0.05, 0.05))
            d_slant = float(np.tan(np.radians(np.clip(rng.normal(0.0, 1.2 * clumsy), -3.0, 3.0))))

            ch_placed: list[np.ndarray] = []
            for s in norm_c:
                px = (s[:, 0] - min_nx) * sx
                py = s[:, 1] * sy + dy
                px = px - d_slant * py + cursor_x
                ch_placed.append(np.stack([px, py], axis=1))

            jitter = float(abs(np.log(sx)) + abs(np.log(sy)) + abs(dy) + abs(d_slant))
            placed_glyphs.append(PlacedGlyph(ch=ch, strokes=ch_placed, jitter=jitter))

            actual_max_x = max(float(s[:, 0].max()) for s in ch_placed)
            kern_jitter = float(np.clip(rng.normal(0.0, 0.018 * clumsy), -0.04, 0.05))
            kern_base = 0.01 if connectedness >= 0.4 else (0.24 if cw < 0.25 else 0.20)
            cursor_x = max(cursor_x + 0.20, actual_max_x + max(0.0, kern_base + kern_jitter))

        return placed_glyphs

    def _latin_word_strokes(
        self,
        word: str,
        rng: np.random.Generator | None = None,
        connectedness: float = 0.5,
        neatness: float = 0.6,
        profile: WriterProfile | None = None,
    ) -> tuple[list[np.ndarray], float]:
        """Character-by-character Latin synthesis with per-glyph structural & non-linear variation."""
        if rng is None:
            strokes = [np.asarray(s, np.float64) for s in self.f.strokes_for_text(word)]
            strokes = [s for s in strokes if len(s)]
            if not strokes:
                return [], 0.6 * len(word)
            allp = np.concatenate(strokes)
            x0 = allp[:, 0].min()
            out = self._norm_hershey(strokes, x0)
            all_norm = np.concatenate(out)
            min_x = float(all_norm[:, 0].min())
            for s in out:
                s[:, 0] -= min_x
            width = float(all_norm[:, 0].max() - min_x)
            return out, max(width, 0.1)

        glyphs = self.place_glyphs(word, rng=rng, connectedness=connectedness, neatness=neatness, profile=profile)
        if not glyphs:
            return [], max(0.6 * len(word), 0.1)

        if connectedness >= 0.4:
            p_join = float(np.clip((connectedness - 0.35) / 0.45, 0.0, 0.97))
            res = connect_word(glyphs, rng=rng, p_join=p_join)
            out_strokes = res.strokes
        else:
            out_strokes = [s for g in glyphs for s in g.strokes]

        if not out_strokes:
            return [], 0.6 * len(word)
        allp = np.concatenate(out_strokes)
        min_x = float(allp[:, 0].min())
        for s in out_strokes:
            s[:, 0] -= min_x
        return out_strokes, max(float(allp[:, 0].max() - min_x), 0.1)

    def word(
        self,
        word: str,
        rng: np.random.Generator | None = None,
        connectedness: float = 0.5,
        neatness: float = 0.6,
        profile: WriterProfile | None = None,
    ) -> tuple[list[np.ndarray], float]:
        if not any(is_hangul(c) for c in word):
            return self._latin_word_strokes(word, rng=rng, connectedness=connectedness, neatness=neatness, profile=profile)

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
        gap = 0.22
        clumsy = max(0.45, 1.65 - 1.25 * float(np.clip(neatness, 0.0, 1.0)))

        for is_h, text in runs:
            if is_h:
                for ch in text:
                    ch_strokes = hangul_char_strokes(ch, rng=rng, connectedness=connectedness, neatness=neatness, profile=profile)
                    if ch_strokes:
                        ch_pts = np.concatenate(ch_strokes, axis=0)
                        ch_x0 = float(ch_pts[:, 0].min())
                        ch_w = float(ch_pts[:, 0].max() - ch_x0)
                        sy_jit = float(np.clip(np.exp(rng.normal(0.0, 0.026 * clumsy)), 0.91, 1.07)) if rng is not None else 1.0
                        sx_jit = float(np.clip(np.exp(rng.normal(0.0, 0.035 * clumsy)), 0.89, 1.12)) if rng is not None else 1.0
                        dy_jit = float(np.clip(rng.normal(0.0, 0.018 * clumsy), -0.05, 0.05)) if rng is not None else 0.0
                        rot_jit = float(np.radians(np.clip(rng.normal(0.0, 1.0 * clumsy), -2.4, 2.4))) if rng is not None else 0.0
                        c_r, s_r = np.cos(rot_jit), np.sin(rot_jit)
                        for s in ch_strokes:
                            sp = s.copy()
                            px = (sp[:, 0] - ch_x0) * sx_jit
                            py = sp[:, 1] * sy_jit
                            rx = c_r * px - s_r * py + cursor_x
                            ry = s_r * px + c_r * py + dy_jit
                            all_strokes.append(np.stack([rx, ry], axis=1))
                        gap_jit = float(np.clip(rng.normal(0.0, 0.022 * clumsy), -0.05, 0.06)) if rng is not None else 0.0
                        cursor_x += max(ch_w * sx_jit, 1.24) + max(0.08, gap + gap_jit)
                    else:
                        cursor_x += 0.9
            else:
                norm_s, w = self._latin_word_strokes(text, rng=rng, connectedness=connectedness, neatness=neatness, profile=profile)
                if norm_s:
                    for s in norm_s:
                        sp = np.stack([s[:, 0] + cursor_x, s[:, 1]], axis=1)
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
    """Low-frequency smooth biomechanical posture field without high-frequency CAPTCHA distortion."""
    w = rng.uniform(0.5, 2.0, (2, n))
    ph = rng.uniform(0, 2 * np.pi, (2, n))
    a = rng.normal(0.0, amp / np.sqrt(n), (2, n))
    out = []
    for s in strokes:
        x = s[:, :1]
        dx = (a[0] * np.sin(w[0] * x + ph[0])).sum(1)
        dy = (a[1] * np.sin(w[1] * x + ph[1])).sum(1)
        out.append(s + np.stack([dx, dy], axis=1))
    return out


class ProceduralGenerator:
    name = "procedural"

    def __init__(self) -> None:
        self.fonts = {
            "fancy": _Font("scripts"),
            "cursive": _Font("cursive"),
            "print": _Font("futural"),
        }

    def _font(self, style: StyleSpec) -> _Font:
        if style.connectedness >= 0.65 and style.roundness >= 0.55:
            return self.fonts["fancy"]
        if style.connectedness >= 0.36:
            return self.fonts["cursive"]
        return self.fonts["print"]

    def estimate_width(self, text: str, style: StyleSpec) -> float:
        f = self._font(style)
        words = text.split()
        w = sum(f.width(wd) for wd in words) * style.width_scale * style.letter_spacing
        return float(w + 0.8 * style.word_spacing * max(len(words) - 1, 0))

    def generate_line(self, req: LineRequest) -> Ink:
        st, ls, rng = req.style, req.state, req.rng
        f = self._font(st)
        loose = 1.0 - ls.neatness
        w_seed = ls.writer_seed if getattr(ls, "writer_seed", 0) != 0 else int(rng.integers(1, 2**31 - 1))
        profile = sample_writer_profile(np.random.default_rng(w_seed), st)
        all_strokes: list[np.ndarray] = []
        cursor = 0.0
        for word in req.text.split():
            strokes, width = f.word(word, rng=rng, connectedness=st.connectedness, neatness=ls.neatness, profile=profile)
            if not strokes:
                cursor += width
                continue
            has_h = any(is_hangul(c) for c in word)
            strokes = _merge_touching(strokes, 0.4 + 0.6 * st.connectedness, rng)
            strokes = [densify(s, 0.03) for s in strokes]
            if st.roundness > 0.05:
                sig = (0.55 + 0.85 * st.roundness) if has_h else (0.9 + 2.1 * st.roundness)
                strokes = [np.stack([gaussian_filter1d(s[:, 0], sig, mode="nearest"),
                                     gaussian_filter1d(s[:, 1], sig, mode="nearest")], 1) if len(s) > 4 else s
                           for s in strokes]
            deform_amp = (0.010 + 0.020 * loose) if has_h else (0.014 + 0.030 * loose)
            strokes = _deform(strokes, deform_amp, rng)
            # per-word jitter
            sc = float(np.clip(np.exp(rng.normal(0.0, 0.032 + 0.042 * loose)), 0.86, 1.11))
            rot = np.radians(np.clip(rng.normal(0.0, 0.5 + 1.0 * loose), -1.5, 1.5))
            base_slant = 0.45 * ls.slant_deg if has_h else ls.slant_deg
            slant = base_slant + float(np.clip(rng.normal(0.0, 1.5 + 2.8 * (1 - st.consistency)), -6.5, 6.5))
            k = np.tan(np.radians(slant))
            dy = float(np.clip(rng.normal(0.0, 0.025 + 0.045 * st.baseline_wobble), -0.09, 0.09))
            sx = st.width_scale * st.letter_spacing
            c, s_ = np.cos(rot), np.sin(rot)
            placed = []
            for s in strokes:
                x = s[:, 0] * sx * sc
                y = s[:, 1] * sc
                x = x - k * y
                wy_rot = np.clip(s_ * x, -0.08, 0.08)
                placed.append(np.stack([c * x - s_ * y, wy_rot + c * y + dy], axis=1))
            allp = np.concatenate(placed)
            x0, x1 = allp[:, 0].min(), allp[:, 0].max()
            all_strokes.extend(p - np.array([x0 - cursor, 0.0]) for p in placed)
            cursor += (x1 - x0) + 0.9 * st.word_spacing * ls.word_spacing_scale * float(np.exp(rng.normal(0, 0.14)))
        ink = strokes_to_ink(all_strokes, st.speed, rng, text=req.text)
        if len(ink):
            # micro tremor along the stroke (smoothed noise)
            trem = gaussian_filter1d(rng.standard_normal((len(ink), 2)), 3.0, axis=0) * (0.005 + 0.012 * loose)
            ink.points[:, :2] += trem.astype(np.float32)
            xmin = float(ink.points[:, X].min())
            if xmin < 0.0:
                ink.points[:, X] -= xmin
            ink.points[:, P] = np.clip(pseudo_pressure(ink) * (0.6 + 0.8 * st.pressure), 0.02, 1.0)
        ink.meta["generator"] = self.name
        return ink
