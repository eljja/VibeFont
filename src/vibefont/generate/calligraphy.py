"""Expressive Calligraphy Art Generator & Layout Composer (Korean & English).

Transforms short poetic texts and aphorisms into genuine calligraphic artwork:
  1. Korean Brush Calligraphy (한글 붓펜/수묵 캘리그라피):
     - Stem vs. particle visual hierarchy (어간/핵심어는 크고 굵게, 조사/어미는 작고 가볍게 밀착 배치)
     - Calligraphic stroke sculpting: 기필 (angled entry press), 장획 (sweeping S-curve horizontal
       extensions), 삐침/파임 (long tapered vertical/diagonal tails), and interlocking syllable
       clusters (자간 밀착 & 지그재그 리듬감).
     - Traditional hand-carved vermilion stone seal (전각 낙관 인장) and subtle ink-wash accents.

  2. English Modern Brush & Copperplate Calligraphy:
     - Hero content word vs. function word scale hierarchy (1.42x hero words vs. 0.78x prepositions).
     - High-contrast downstroke shading (p -> 0.95 on downstrokes, p -> 0.06 on hairline upstrokes).
     - Entry/exit calligraphic swashes and symmetrical off-hand Copperplate under-flourishes.
"""

from __future__ import annotations

import numpy as np

from ..document.layout import Page, page_size
from ..ink import EOS, NCOLS, P, T, X, Y, Ink
from ..vibe.spec import VibeSpec
from .glyph_guard import MissingGlyphError, validate_text_glyphs
from .hangul import WriterProfile, _curve, hangul_char_strokes, is_hangul, sample_writer_profile
from .kinematics import ballistic_flourish, densify
from .procedural import ProceduralGenerator
from .semantics import SemanticArchetype, WordMorphology, analyze_semantics

# Korean grammatical particles/endings that should be rendered smaller & tucked beside the stem
_KO_PARTICLES = (
    "에서도", "에게는", "으로의", "이라는", "이라도", "처럼", "부터", "까지", "마다",
    "에서", "에게", "으로", "에는", "에도", "이라", "보다",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "로", "와", "과", "만",
)

# English short function words rendered smaller & lighter to emphasize hero words
_EN_FUNCTION_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "of", "for", "by", "with", "from",
    "into", "upon", "over", "and", "but", "or", "nor", "is", "are", "was", "were",
    "be", "been", "that", "this", "its", "our", "your", "my", "not", "only",
}


def _split_ko_stem_particle(word: str) -> tuple[str, str]:
    """Splits a Korean token into (content_stem, grammatical_particle) if applicable."""
    clean = word.rstrip(".,!?;:")
    punct = word[len(clean):]
    if len(clean) <= 1:
        return word, ""
    for part in _KO_PARTICLES:
        if clean.endswith(part) and len(clean) - len(part) >= 1:
            stem = clean[: len(clean) - len(part)]
            return stem, part + punct
    return word, ""


def _calligraphic_pressure_from_geometry(
    poly: np.ndarray,
    stroke_weight: float = 1.0,
    is_flourish: bool = False,
    is_tail_stroke: bool = False,
    has_ko: bool = True,
) -> np.ndarray:
    """Computes high-contrast calligraphic pressure along a dense 2D stroke polyline.

    - Flourishes are delicate hairlines with a gentle mid-curve swell.
    - Korean (has_ko=True): 횡세종태 (slender horizontals, bold vertical pillars, sharp 기필/수필).
    - English (has_ko=False): Copperplate / modern brush thick-thin shading with continuous
      unbroken hairline upstrokes (p >= 0.24) and rich downstroke shades (p ~ 0.90..1.00).
    """
    n = len(poly)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.array([0.45], dtype=np.float32)

    u = np.linspace(0.0, 1.0, n)
    if is_flourish:
        swell = np.sin(np.pi * u) ** 1.3
        return np.clip(0.16 + 0.30 * swell, 0.14, 0.46).astype(np.float32)

    dxy = np.gradient(poly, axis=0)
    ds = np.maximum(np.linalg.norm(dxy, axis=1), 1e-5)
    dy_norm = dxy[:, 1] / ds
    dx_norm = dxy[:, 0] / ds

    if has_ko:
        # Korean brush dynamics: firm vertical/diagonal strokes, lighter upward movements
        down_weight = np.clip(0.62 + 0.40 * dy_norm + 0.10 * np.abs(dx_norm), 0.30, 1.12)
        up_mask = np.clip((-dy_norm - 0.12) * 2.0, 0.0, 1.0)
        dir_p = down_weight * (1.0 - 0.52 * up_mask)

        # 기필 (entry attack peak near u=0.11) + 송필 (mid modulation) + 수필 (needle exit taper)
        entry = np.clip(0.45 + 0.55 * (u / 0.10), 0.45, 1.0) + 0.18 * np.exp(-((u - 0.11) / 0.08) ** 2)
        taper_span = 0.44 if is_tail_stroke else 0.26
        exit_taper = np.clip((1.0 - u) / taper_span, 0.04, 1.0) ** (1.10 if is_tail_stroke else 0.85)

        p = np.clip(stroke_weight * dir_p * entry * exit_taper, 0.05, 1.0)
        return p.astype(np.float32)
    else:
        # English Copperplate / Modern Brush: lush downstrokes, continuous unbroken hairline upstrokes
        down_weight = np.clip(0.58 + 0.46 * dy_norm, 0.28, 1.05)
        up_mask = np.clip((-dy_norm - 0.05) * 1.8, 0.0, 1.0)
        dir_p = np.clip(down_weight * (1.0 - 0.48 * up_mask), 0.26, 1.05)

        entry = np.clip(0.55 + 0.45 * (u / 0.08), 0.55, 1.0)
        if is_tail_stroke:
            exit_taper = np.clip((1.0 - u) / 0.35, 0.08, 1.0)
            p_min = 0.08
        else:
            exit_taper = np.clip(0.55 + 0.45 * ((1.0 - u) / 0.12), 0.55, 1.0)
            p_min = 0.24

        p = np.clip(stroke_weight * dir_p * entry * exit_taper, p_min, 1.0)
        return p.astype(np.float32)


def _sculpt_ko_syllable_strokes(
    ch: str,
    rng: np.random.Generator,
    is_hero: bool = False,
    is_word_end: bool = False,
    profile: WriterProfile | None = None,
    morph: WordMorphology | None = None,
) -> list[tuple[np.ndarray, float, bool]]:
    """Generates calligraphic Hangul syllable strokes with semantic emotion, 횡세종태, and 삐침/파임.

    Returns list of (stroke_xy, stroke_weight, is_tail_stroke).
    """
    raw = hangul_char_strokes(ch, rng=rng, connectedness=0.72, neatness=0.82, profile=profile)
    if not raw:
        return []

    if morph is not None:
        is_hero = morph.is_hero

    out: list[tuple[np.ndarray, float, bool]] = []
    n_st = len(raw)

    # Base stroke weight modulated by semantic archetype
    w_base = 1.00 if is_hero else 0.74
    if morph is not None:
        w_base = (1.00 + morph.weight_boost) if is_hero else (0.74 + 0.6 * morph.weight_boost)
        if morph.archetype == SemanticArchetype.BOLD_MASSIVE:
            w_base *= 1.30
        elif morph.archetype == SemanticArchetype.FLOWING_AIRY:
            w_base *= 0.88
        elif morph.archetype == SemanticArchetype.GENTLE_WARM:
            w_base *= 0.96
        elif morph.archetype == SemanticArchetype.SHARP_FIERCE:
            w_base *= 1.10
        elif morph.archetype == SemanticArchetype.SOARING_LUMINOUS:
            w_base *= 1.05

    complexity_damp = 0.84 if n_st >= 6 else (0.92 if n_st >= 5 else 1.0)

    for idx, s in enumerate(raw):
        pts = s.copy()
        if len(pts) < 3:
            out.append((pts, w_base * 0.55 * complexity_damp, False))
            continue

        # Semantic aspect ratio sculpting
        if morph is not None:
            if morph.archetype == SemanticArchetype.SOARING_LUMINOUS:
                pts[:, 1] *= 1.18
                pts[:, 0] *= 0.94
            elif morph.archetype == SemanticArchetype.BOLD_MASSIVE:
                pts[:, 1] *= 0.96
                pts[:, 0] *= 1.06

        span_x = float(pts[:, 0].max() - pts[:, 0].min())
        span_y = float(pts[:, 1].max() - pts[:, 1].min())
        is_last = idx == n_st - 1
        is_tail = False

        # Determine stroke weight using Korean calligraphy rule: 횡세종태 (horizontals thinner, verticals bolder)
        if span_x > span_y * 1.35:
            if span_x > 0.75 and n_st <= 4:
                s_weight = w_base * 0.76 * complexity_damp
            else:
                s_weight = w_base * 0.54 * complexity_damp
        elif span_y > 0.58:
            s_weight = w_base * 1.00 * complexity_damp
        else:
            s_weight = w_base * 0.58 * complexity_damp

        # 1. Add 기필 (angled brush entry attack) at the start of primary vertical/horizontal strokes
        ang = morph.angularity if morph is not None else 0.5
        if len(pts) >= 5 and (idx == 0 or span_y > 0.65) and ang > 0.15:
            hook_offset = np.array([-0.05 * ang, -0.05 * ang])
            hook = pts[0] + hook_offset
            pts = np.concatenate([[hook], pts], axis=0)

        # 2. Expressive S-curve wave on wide horizontal strokes (장획)
        if span_x > 0.72 and span_y < 0.42:
            u = np.linspace(0.0, 1.0, len(pts))
            ext = (1.16 if is_word_end else 1.08) if is_hero else 1.04
            if morph is not None and morph.archetype == SemanticArchetype.FLOWING_AIRY:
                ext *= 1.15
            x0 = pts[0, 0]
            pts[:, 0] = x0 + (pts[:, 0] - x0) * ext
            wave_amp = 0.06 if (morph and morph.archetype == SemanticArchetype.FLOWING_AIRY) else 0.04
            pts[:, 1] += -0.06 * u + wave_amp * np.sin(2.0 * np.pi * u)
            if is_hero and is_last:
                is_tail = True

        # 3. Sweeping brush tail on final strokes (삐침/파임 vs 가로끝 처리)
        elif is_last and (is_hero or is_word_end) and len(pts) >= 4:
            swash_mult = morph.swash_reach if morph is not None else 1.0
            end_pt = pts[-1]
            d_end = pts[-1] - pts[-3]
            norm_d = max(float(np.linalg.norm(d_end)), 1e-4)
            dir_vec = d_end / norm_d

            # Only genuine vertical or downward diagonal strokes (e.g. ㅣ, ㅏ, ㅓ, ㅅ, ㅈ) flourish downwards
            if dir_vec[1] > 0.40 and span_y > 0.45:
                tail_len = (0.50 if is_hero else 0.32) * swash_mult * float(rng.uniform(0.95, 1.15))
                sweep_dir = np.array([0.35 * dir_vec[0] - 0.10, max(0.70, dir_vec[1])])
                sweep_dir /= np.linalg.norm(sweep_dir)
                tail_pts, _ = ballistic_flourish(
                    start_pt=end_pt,
                    initial_direction=sweep_dir,
                    length_mm=tail_len,
                    inertia_curvature=-0.30 * float(rng.uniform(0.8, 1.2)),
                    n_pts=14,
                )
                pts = np.concatenate([pts[:-1], tail_pts], axis=0)
                is_tail = True
            elif dir_vec[0] > 0.55 and dir_vec[1] < 0.35 and is_hero and span_x > 0.65:
                # Horizontal strokes (ㄴ, ㄷ, ㄹ, ㅡ) taper horizontally/upward, NEVER downwards
                tail_len = 0.22 * swash_mult * float(rng.uniform(0.9, 1.1))
                sweep_dir = np.array([0.96, min(0.04, max(-0.12, dir_vec[1]))])
                sweep_dir /= np.linalg.norm(sweep_dir)
                tail_pts, _ = ballistic_flourish(
                    start_pt=end_pt,
                    initial_direction=sweep_dir,
                    length_mm=tail_len,
                    inertia_curvature=0.10,
                    n_pts=10,
                )
                pts = np.concatenate([pts[:-1], tail_pts], axis=0)
                is_tail = True

        out.append((pts, s_weight, is_tail))
    return out


def _build_offhand_flourish(
    cx_mm: float,
    y_mm: float,
    span_w_mm: float,
) -> list[tuple[np.ndarray, float, bool]]:
    """Creates a classic balanced calligraphic under-flourish in page millimetres."""
    hw = 0.42 * span_w_mm
    wave = _curve(
        [
            [cx_mm - hw, y_mm + 1.8],
            [cx_mm - 0.55 * hw, y_mm - 2.2],
            [cx_mm, y_mm + 0.4],
            [cx_mm + 0.55 * hw, y_mm + 2.8],
            [cx_mm + hw, y_mm - 1.4],
        ],
        n=60,
    )
    loop = _curve(
        [
            [cx_mm - 0.32 * hw, y_mm + 0.9],
            [cx_mm, y_mm - 1.4],
            [cx_mm + 0.32 * hw, y_mm + 0.9],
        ],
        n=36,
    )
    return [(wave, 0.0, True), (loop, 0.0, True)]


def _strokes_to_calligraphy_ink(
    placed_strokes: list[tuple[np.ndarray, float, bool]],
    text: str,
    has_ko: bool = True,
    glyphs: list[dict] | None = None,
) -> Ink:
    """Converts list of (xy_mm, stroke_weight, is_flourish_or_tail) in page mm into timed, pressure-sculpted Ink."""
    rows = []
    t = 0.0
    for xy_mm, s_weight, is_special in placed_strokes:
        if len(xy_mm) == 0:
            continue
        dense = densify(xy_mm, 0.08)
        if len(dense) < 2:
            dense = np.concatenate([xy_mm[:1], xy_mm[:1] + np.array([[0.05, 0.05]], dtype=np.float32)], axis=0)
        is_flourish = is_special and (s_weight <= 0.01)
        is_tail = is_special and (s_weight > 0.01)
        p = _calligraphic_pressure_from_geometry(
            dense,
            stroke_weight=max(s_weight, 0.30),
            is_flourish=is_flourish,
            is_tail_stroke=is_tail,
            has_ko=has_ko,
        )

        # 1. Neuromotor inflection curvature & corner hesitation (전절/기필 머무름)
        dxy = np.gradient(dense, axis=0)
        ds = np.maximum(np.linalg.norm(dxy, axis=1), 1e-4)
        tangent = dxy / ds[:, None]
        angles = np.arctan2(tangent[:, 1], tangent[:, 0])
        d_ang = np.abs((np.gradient(angles) + np.pi) % (2 * np.pi) - np.pi)
        curv = d_ang / ds

        # Deceleration at high curvature and momentary dwell pause at corners
        corner_slowdown = 1.0 / (1.0 + 2.5 * np.clip(curv / 0.8, 0.0, 4.0))
        corner_dwell = 0.045 * np.clip((curv - 0.4) / 1.2, 0.0, 1.0)

        # 2. Ballistic downstroke acceleration
        down_bias = np.clip(dxy[:, 1] / ds, -1.0, 1.0)
        cum_s = np.cumsum(ds)
        ballistic_boost = np.clip(down_bias, 0.0, 1.0) * np.clip(cum_s / 5.0, 0.0, 1.0)

        d = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        v_base = 56.0 * (1.25 - 0.70 * p[:-1])
        v_mm_s = np.clip(v_base * (1.0 + 1.85 * ballistic_boost[:-1]) * corner_slowdown[:-1], 6.0, 160.0)
        dt = np.maximum(d / v_mm_s, 1e-4) + corner_dwell[:-1]
        ts = np.concatenate([[0.0], np.cumsum(dt)])

        # 3. 10Hz involuntary physiological micro-tremor along extended strokes
        n_pts = len(dense)
        n_vec = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        phase = 2.0 * np.pi * 10.0 * ts
        tremor_fade = np.clip(np.minimum(np.arange(n_pts), n_pts - 1 - np.arange(n_pts)) / 4.0, 0.0, 1.0)
        tremor_amp = 0.013 * (1.0 - 0.45 * np.clip(p, 0.0, 1.0)) * tremor_fade
        dense_out = dense + (tremor_amp * np.sin(phase))[:, None] * n_vec

        r = np.zeros((len(dense_out), NCOLS), dtype=np.float32)
        r[:, X] = dense_out[:, 0]
        r[:, Y] = dense_out[:, 1]
        r[:, T] = t + ts
        r[:, P] = p
        r[:, EOS] = 0.0
        r[-1, EOS] = 1.0
        rows.append(r)
        t = float(r[-1, T]) + 0.06

    pts = np.concatenate(rows, axis=0) if rows else np.zeros((0, NCOLS), dtype=np.float32)
    meta = {"generator": "calligraphy"}
    if glyphs is not None:
        meta["glyphs"] = glyphs
    return Ink(pts, text=text, meta=meta)


def compose_calligraphy_page(
    text: str,
    spec: VibeSpec,
    rng: np.random.Generator | None = None,
) -> Page:
    """Composes a complete artistic calligraphy card page (Korean or English)."""
    rng = rng or np.random.default_rng(spec.seed)
    text = text.replace("\\n", "\n").strip()
    text = validate_text_glyphs(text, allow_hangul=True)
    spec.layout.page = "Card"
    w_mm, h_mm = page_size(spec)
    page = Page(w_mm, h_mm)

    has_ko = any(is_hangul(c) for c in text)
    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not raw_lines:
        return page

    proc = ProceduralGenerator()
    script_font = proc.fonts["fancy"]
    profile = sample_writer_profile(rng, spec.style)

    # 1. Semantic Emotion & Morphology Analysis
    semantic_lines = analyze_semantics(text)
    while len(semantic_lines) < len(raw_lines):
        semantic_lines.append([])

    # Compute line-level hierarchy across the entire piece
    line_scores: list[float] = []
    for m_list in semantic_lines:
        if not m_list:
            line_scores.append(1.0)
            continue
        sc = max(
            (0.15 if m.is_particle else (4.5 if m.archetype != SemanticArchetype.NEUTRAL else 1.0) * (1.35 if m.is_hero else 1.0))
            for m in m_list
        )
        line_scores.append(sc)

    max_sc = max(line_scores) if line_scores else 1.0
    min_sc = min(line_scores) if line_scores else 1.0

    line_scale_factors: list[float] = []
    for sc in line_scores:
        if len(raw_lines) <= 1:
            fac = 1.0
        elif max_sc > min_sc:
            t = (sc - min_sc) / (max_sc - min_sc)
            fac = 0.86 + 0.46 * t
        else:
            fac = 1.0
        line_scale_factors.append(fac)

    base_size_mm = 4.55 if has_ko else 3.20
    line_pitch = 17.0 if has_ko else 13.5

    # 2. Build normalized strokes for each line
    all_line_strokes_norm: list[list[tuple[np.ndarray, float, bool]]] = []
    all_line_glyphs: list[list[dict]] = []

    for line_idx, line_str in enumerate(raw_lines):
        words = line_str.split()
        m_list = semantic_lines[line_idx]
        line_fac = line_scale_factors[line_idx]
        line_strokes_norm: list[tuple[np.ndarray, float, bool]] = []
        line_glyphs: list[dict] = []
        cursor_x = 0.0

        if has_ko:
            for w_idx, word in enumerate(words):
                stem, particle = _split_ko_stem_particle(word)
                m_stem = next((m for m in m_list if m.word == stem and not m.is_particle), None)
                m_part = next((m for m in m_list if m.is_particle and (m.word == particle or particle.startswith(m.word))), None)
                is_hero_stem = (m_stem.is_hero if m_stem else False) or (line_fac > 1.15 and w_idx == 0)

                # Render stem syllables (dominant, sculpted, expressive)
                stem_sc = (m_stem.scale if m_stem else (1.42 if is_hero_stem else 1.12)) * line_fac
                for c_i, ch in enumerate(stem):
                    is_end = (c_i == len(stem) - 1) and not particle
                    sc = stem_sc * float(rng.uniform(0.97, 1.03))
                    dy = float(rng.uniform(-0.12, 0.12)) + (-0.08 if c_i % 2 == 1 else 0.04)
                    rot = np.radians(float(rng.uniform(-3.0, 2.5)))
                    c_rot, s_rot = np.cos(rot), np.sin(rot)
                    if is_hangul(ch):
                        sculpted = _sculpt_ko_syllable_strokes(
                            ch, rng, is_hero=is_hero_stem, is_word_end=is_end, profile=profile, morph=m_stem
                        )
                    else:
                        w_st, _ = script_font.word(ch, rng=rng, connectedness=0.0, neatness=0.88, profile=profile)
                        sculpted = [(s, 0.70, False) for s in w_st]
                    if not sculpted:
                        raise MissingGlyphError(ch, text=text)
                    all_p = np.concatenate([p for p, _, _ in sculpted], axis=0)
                    x0 = float(all_p[:, 0].min())
                    ch_w = max(float(all_p[:, 0].max() - x0), 0.20)
                    start_s = len(line_strokes_norm)
                    for pts, s_weight, is_tail in sculpted:
                        px = (pts[:, 0] - x0) * sc
                        py = pts[:, 1] * sc + dy
                        rx = c_rot * px - s_rot * py + cursor_x
                        ry = s_rot * px + c_rot * py
                        line_strokes_norm.append((np.stack([rx, ry], axis=1), s_weight, is_tail))
                    end_s = len(line_strokes_norm)
                    line_glyphs.append({"ch": ch, "strokes": (start_s, end_s)})
                    cursor_x += (ch_w * sc) + 0.26 * sc

                # Render trailing grammatical particle smaller & tucked tightly beside stem
                if particle:
                    part_sc = (m_part.scale if m_part else 0.72) * line_fac
                    for c_i, ch in enumerate(particle):
                        sc = part_sc * float(rng.uniform(0.96, 1.04))
                        dy = float(rng.uniform(-0.04, 0.10))
                        rot = np.radians(float(rng.uniform(-2.5, 2.0)))
                        c_rot, s_rot = np.cos(rot), np.sin(rot)
                        if is_hangul(ch):
                            sculpted = _sculpt_ko_syllable_strokes(
                                ch, rng, is_hero=False, is_word_end=(c_i == len(particle) - 1), profile=profile, morph=m_part
                            )
                        else:
                            w_st, _ = script_font.word(ch, rng=rng, connectedness=0.0, neatness=0.88, profile=profile)
                            sculpted = [(s, 0.70, False) for s in w_st]
                        if not sculpted:
                            raise MissingGlyphError(ch, text=text)
                        all_p = np.concatenate([p for p, _, _ in sculpted], axis=0)
                        x0 = float(all_p[:, 0].min())
                        ch_w = max(float(all_p[:, 0].max() - x0), 0.20)
                        start_s = len(line_strokes_norm)
                        for pts, s_weight, is_tail in sculpted:
                            px = (pts[:, 0] - x0) * sc
                            py = pts[:, 1] * sc + dy
                            rx = c_rot * px - s_rot * py + cursor_x
                            ry = s_rot * px + c_rot * py
                            line_strokes_norm.append((np.stack([rx, ry], axis=1), s_weight, is_tail))
                        end_s = len(line_strokes_norm)
                        line_glyphs.append({"ch": ch, "strokes": (start_s, end_s)})
                        cursor_x += (ch_w * sc) + 0.18 * sc

                cursor_x += 0.85 * line_fac
        else:
            for w_idx, word in enumerate(words):
                clean_w = word.strip(".,!?;:").lower()
                m_word = next((m for m in m_list if m.word.lower() == clean_w), None)
                is_hero_w = (m_word.is_hero if m_word else False) or (clean_w not in _EN_FUNCTION_WORDS and line_fac > 1.15)
                is_func = (clean_w in _EN_FUNCTION_WORDS) and not is_hero_w

                sc = ((0.76 if is_func else (1.38 if is_hero_w else 1.05))) * line_fac
                s_weight = 0.76 if is_func else (1.05 if is_hero_w else 0.92)
                w_strokes, _ = script_font.word(word, rng=rng, connectedness=0.92, neatness=0.88, profile=profile)
                if not w_strokes:
                    raise MissingGlyphError(word, text=text)
                slant_k = np.tan(np.radians(16.0))
                word_dy = float(rng.uniform(-0.05, 0.05))
                placed_w: list[np.ndarray] = []
                start_s = len(line_strokes_norm)
                for s_i, s in enumerate(w_strokes):
                    px = s[:, 0] * sc
                    py = s[:, 1] * sc + word_dy
                    px = px - slant_k * py + cursor_x
                    pts = np.stack([px, py], axis=1)
                    if w_idx == 0 and s_i == 0 and len(pts) >= 3:
                        p0 = pts[0]
                        d_in = pts[1] - pts[0]
                        swash_pts, _ = ballistic_flourish(
                            start_pt=p0,
                            initial_direction=d_in,
                            length_mm=0.75 * sc,
                            inertia_curvature=0.20,
                            is_entrance=True,
                            n_pts=14,
                        )
                        pts = np.concatenate([swash_pts[:-1], pts], axis=0)
                    elif w_idx == len(words) - 1 and s_i == len(w_strokes) - 1 and len(pts) >= 3:
                        pe = pts[-1]
                        d_out = pts[-1] - pts[-2]
                        swash_pts, _ = ballistic_flourish(
                            start_pt=pe,
                            initial_direction=d_out + np.array([0.25, -0.15]),
                            length_mm=0.90 * sc,
                            inertia_curvature=-0.22,
                            is_entrance=False,
                            n_pts=14,
                        )
                        pts = np.concatenate([pts, swash_pts[1:]], axis=0)
                    placed_w.append(pts)
                    is_tail = (w_idx == len(words) - 1 and s_i == len(w_strokes) - 1)
                    line_strokes_norm.append((pts, s_weight, is_tail))

                end_s = len(line_strokes_norm)
                line_glyphs.append({"ch": word, "strokes": (start_s, end_s)})
                all_wp = np.concatenate(placed_w, axis=0)
                cursor_x = float(all_wp[:, 0].max()) + 0.70 * sc

        all_line_strokes_norm.append(line_strokes_norm)
        all_line_glyphs.append(line_glyphs)

    # 3. Global layout scaling to fit card margins while preserving line hierarchy
    max_line_w_raw = 1e-3
    for lsn in all_line_strokes_norm:
        if lsn:
            all_pts = np.concatenate([p for p, _, _ in lsn], axis=0)
            w_raw = float(all_pts[:, 0].max() - all_pts[:, 0].min())
            max_line_w_raw = max(max_line_w_raw, w_raw)

    max_allowed_w_mm = w_mm - 28.0
    global_scale = min(1.0, max_allowed_w_mm / (max_line_w_raw * base_size_mm))
    eff_size_mm = base_size_mm * global_scale
    eff_pitch_mm = line_pitch * min(1.0, (h_mm - 28.0) / max(len(raw_lines) * line_pitch, 1e-3))

    line_inks: list[Ink] = []
    for line_idx, (line_str, lsn, l_glyphs) in enumerate(zip(raw_lines, all_line_strokes_norm, all_line_glyphs)):
        if not lsn:
            continue
        all_pts = np.concatenate([p for p, _, _ in lsn], axis=0)
        xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        lw_mm = (xmax - xmin) * eff_size_mm
        stagger_mm = (line_idx - 0.5 * (len(raw_lines) - 1)) * (3.6 if has_ko else 0.0)
        x_offset_mm = 0.5 * (w_mm - lw_mm) + stagger_mm
        x_offset_mm = float(np.clip(x_offset_mm, 8.0, w_mm - lw_mm - 8.0))
        y_base_mm = 0.5 * h_mm + (line_idx - 0.5 * (len(raw_lines) - 1)) * eff_pitch_mm

        mm_strokes: list[tuple[np.ndarray, float, bool]] = []
        for pts, s_weight, is_tail in lsn:
            xy_mm = np.stack(
                [
                    (pts[:, 0] - xmin) * eff_size_mm + x_offset_mm,
                    pts[:, 1] * eff_size_mm + y_base_mm,
                ],
                axis=1,
            )
            mm_strokes.append((xy_mm, s_weight, is_tail))

        line_inks.append(_strokes_to_calligraphy_ink(mm_strokes, text=line_str, has_ko=has_ko, glyphs=l_glyphs))

    page.lines = line_inks

    # 4. Center composition vertically and place seal / accents in optical void
    non_empty = [ln for ln in page.lines if len(ln)]
    if non_empty:
        ymin = min(ln.bbox()[1] for ln in non_empty)
        ymax = max(ln.bbox()[3] for ln in non_empty)
        xmin = min(ln.bbox()[0] for ln in non_empty)
        xmax = max(ln.bbox()[2] for ln in non_empty)
        dy = 0.5 * h_mm - 0.5 * (ymin + ymax) - (2.0 if not has_ko else 0.0)
        page.lines = [ln.translate(0.0, dy) for ln in page.lines]
        ymin += dy
        ymax += dy
        last_ln = page.lines[-1]
        last_bbox = last_ln.bbox()

        if has_ko:
            seal_size = 8.2
            all_pts_list = [ln.xy for ln in page.lines if len(ln)]
            page_all_pts = np.concatenate(all_pts_list, axis=0)

            candidates = [
                (min(w_mm - seal_size - 6.0, last_bbox[2] + 4.0), max(6.0, last_bbox[3] - seal_size)),
                (min(w_mm - seal_size - 8.0, xmax + 4.5), min(h_mm - seal_size - 8.0, ymax - seal_size + 1.0)),
                (min(w_mm - seal_size - 8.0, last_bbox[2] - seal_size), min(h_mm - seal_size - 8.0, last_bbox[3] + 4.0)),
                (w_mm - seal_size - 10.0, h_mm - seal_size - 10.0),
            ]
            best_cand = candidates[0]
            best_dist = -1.0
            for sx, sy in candidates:
                cx, cy = sx + 0.5 * seal_size, sy + 0.5 * seal_size
                dists = np.hypot(page_all_pts[:, 0] - cx, page_all_pts[:, 1] - cy)
                d_min = float(np.min(dists))
                if d_min > best_dist:
                    best_dist = d_min
                    best_cand = (sx, sy)

            sx0, sy0 = best_cand
            page.seal_box_mm = (sx0, sy0, sx0 + seal_size, sy0 + seal_size)

            ink_r, ink_g, ink_b = spec.ink.color
            wash_col = (
                min(0.95, 0.55 + 0.45 * ink_r),
                min(0.92, 0.52 + 0.45 * ink_g),
                min(0.92, 0.52 + 0.45 * ink_b),
            )
            page.wash_accents = [
                {"x_mm": max(10.0, xmin - 4.0), "y_mm": ymin + 2.0, "rx_mm": 6.5, "ry_mm": 5.0, "color": wash_col, "opacity": 0.11},
                {"x_mm": max(14.0, xmin + 2.5), "y_mm": ymin - 2.0, "rx_mm": 4.2, "ry_mm": 3.6, "color": wash_col, "opacity": 0.09},
            ]
        else:
            fl_y = min(h_mm - 12.0, ymax + 6.5)
            fl_strokes = _build_offhand_flourish(
                cx_mm=0.5 * w_mm,
                y_mm=fl_y,
                span_w_mm=min(80.0, (xmax - xmin) * 0.72),
            )
            page.lines.append(_strokes_to_calligraphy_ink(fl_strokes, text="", has_ko=False))

    # 5. Strict post-generation glyph integrity verification
    rendered_glyphs = [g["ch"] for lg in all_line_glyphs for g in lg]
    if has_ko:
        expected_chars = [c for c in text if not c.isspace()]
        if rendered_glyphs != expected_chars:
            missing = [c for c in expected_chars if c not in rendered_glyphs]
            raise MissingGlyphError(
                f"Calligraphy integrity check failed: expected {expected_chars}, got {rendered_glyphs}. Missing: {missing}",
                text=text,
            )
    else:
        expected_words = text.split()
        if rendered_glyphs != expected_words:
            missing = [w for w in expected_words if w not in rendered_glyphs]
            raise MissingGlyphError(
                f"Calligraphy integrity check failed: expected {expected_words}, got {rendered_glyphs}. Missing: {missing}",
                text=text,
            )

    for ln in page.lines:
        for g in ln.meta.get("glyphs", []):
            i0, i1 = g["strokes"]
            if i1 <= i0:
                raise MissingGlyphError(f"Glyph {g['ch']!r} registered with 0 strokes", text=text)

    return page
