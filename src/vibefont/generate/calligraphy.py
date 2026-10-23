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
from .hangul import _curve, hangul_char_strokes, is_hangul
from .kinematics import densify
from .procedural import ProceduralGenerator

# Korean grammatical particles/endings that should be rendered smaller & tucked beside the stem
_KO_PARTICLES = (
    "에서도", "에게는", "으로의", "이라는", "이라도", "처럼", "부터", "까지", "마다",
    "에서", "에게", "으로", "에는", "에도", "이라", "보다",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "로", "와", " 과", "만",
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
    is_hero: bool,
    is_word_end: bool,
) -> list[tuple[np.ndarray, float, bool]]:
    """Generates calligraphic Hangul syllable strokes with 횡세종태 contrast, 장획, and 삐침/파임.

    Returns list of (stroke_xy, stroke_weight, is_tail_stroke).
    """
    raw = hangul_char_strokes(ch, rng=rng, connectedness=0.72, neatness=0.85)
    if not raw:
        return []

    out: list[tuple[np.ndarray, float, bool]] = []
    n_st = len(raw)
    w_base = 1.00 if is_hero else 0.74
    complexity_damp = 0.84 if n_st >= 6 else (0.92 if n_st >= 5 else 1.0)

    for idx, s in enumerate(raw):
        pts = s.copy()
        if len(pts) < 3:
            out.append((pts, w_base * 0.55 * complexity_damp, False))
            continue

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

        # 1. Add subtle 기필 (angled brush entry hook) at the start of primary vertical/horizontal strokes
        if len(pts) >= 5 and (idx == 0 or span_y > 0.65):
            hook = pts[0] + np.array([-0.04, -0.04])
            pts = np.concatenate([[hook], pts], axis=0)

        # 2. Expressive S-curve wave on wide horizontal strokes (장획)
        if span_x > 0.72 and span_y < 0.42:
            u = np.linspace(0.0, 1.0, len(pts))
            ext = (1.16 if is_word_end else 1.08) if is_hero else 1.04
            x0 = pts[0, 0]
            pts[:, 0] = x0 + (pts[:, 0] - x0) * ext
            pts[:, 1] += -0.06 * u + 0.04 * np.sin(2.0 * np.pi * u)
            if is_hero and is_last:
                is_tail = True

        # 3. Sweeping downward/diagonal brush tail on final vertical or diagonal strokes (삐침/파임)
        elif is_last and (is_hero or is_word_end) and span_y > 0.55:
            tail_len = (0.34 if is_hero else 0.22) * float(rng.uniform(0.88, 1.15))
            end_pt = pts[-1]
            d_end = pts[-1] - pts[-3]
            norm_d = max(float(np.linalg.norm(d_end)), 1e-4)
            dir_vec = d_end / norm_d
            sweep_dir = np.array([0.30 * dir_vec[0] - 0.14, max(0.78, dir_vec[1])])
            sweep_dir /= np.linalg.norm(sweep_dir)
            ctrl1 = end_pt + 0.45 * tail_len * sweep_dir
            ctrl2 = end_pt + tail_len * (sweep_dir + np.array([-0.12, 0.10]))
            tail_curve = _curve([end_pt.tolist(), ctrl1.tolist(), ctrl2.tolist()], n=10)
            pts = np.concatenate([pts[:-1], tail_curve], axis=0)
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
) -> Ink:
    """Converts list of (xy_mm, stroke_weight, is_flourish_or_tail) in page mm into timed, pressure-sculpted Ink."""
    rows = []
    t = 0.0
    for xy_mm, s_weight, is_special in placed_strokes:
        dense = densify(xy_mm, 0.08)
        if len(dense) < 2:
            continue
        is_flourish = is_special and (s_weight <= 0.01)
        is_tail = is_special and (s_weight > 0.01)
        p = _calligraphic_pressure_from_geometry(
            dense,
            stroke_weight=max(s_weight, 0.30),
            is_flourish=is_flourish,
            is_tail_stroke=is_tail,
            has_ko=has_ko,
        )
        d = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        v_mm_s = np.clip(55.0 * (1.25 - 0.75 * p[:-1]), 14.0, 95.0)
        dt = np.maximum(d / v_mm_s, 1e-3)
        ts = np.concatenate([[0.0], np.cumsum(dt)])

        r = np.zeros((len(dense), NCOLS), dtype=np.float32)
        r[:, X] = dense[:, 0]
        r[:, Y] = dense[:, 1]
        r[:, T] = t + ts
        r[:, P] = p
        r[:, EOS] = 0.0
        r[-1, EOS] = 1.0
        rows.append(r)
        t = float(r[-1, T]) + 0.06

    pts = np.concatenate(rows, axis=0) if rows else np.zeros((0, NCOLS), dtype=np.float32)
    return Ink(pts, text=text, meta={"generator": "calligraphy"})


def compose_calligraphy_page(
    text: str,
    spec: VibeSpec,
    rng: np.random.Generator | None = None,
) -> Page:
    """Composes a complete artistic calligraphy card page (Korean or English)."""
    rng = rng or np.random.default_rng(spec.seed)
    text = text.replace("\\n", "\n").strip()
    spec.layout.page = "Card"
    w_mm, h_mm = page_size(spec)
    page = Page(w_mm, h_mm)

    has_ko = any(is_hangul(c) for c in text)
    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    proc = ProceduralGenerator()
    script_font = proc.fonts["fancy"]

    line_inks: list[Ink] = []
    base_size_mm = 4.85 if has_ko else 3.25
    line_pitch = 16.2 if has_ko else 13.2

    for line_idx, line_str in enumerate(raw_lines):
        words = line_str.split()
        line_strokes_norm: list[tuple[np.ndarray, float, bool]] = []
        cursor_x = 0.0

        # Identify the primary hero word in this line (longest content word)
        content_lengths = []
        for w in words:
            if has_ko:
                st_part, _ = _split_ko_stem_particle(w)
                content_lengths.append(len(st_part))
            else:
                cl = w.strip(".,!?;:").lower()
                content_lengths.append(0 if cl in _EN_FUNCTION_WORDS else len(cl))
        hero_word_idx = int(np.argmax(content_lengths)) if content_lengths else 0

        for w_idx, word in enumerate(words):
            is_line_hero = w_idx == hero_word_idx
            if has_ko:
                stem, particle = _split_ko_stem_particle(word)
                # Render stem syllables (bold, large, expressive, with clean breathing room)
                for c_i, ch in enumerate(stem):
                    is_end = (c_i == len(stem) - 1) and not particle
                    sc = (1.26 if is_line_hero else 1.08) * float(rng.uniform(0.96, 1.05))
                    # Dancing calligraphic baseline rhythm (지그재그 높낮이)
                    dy = float(rng.uniform(-0.14, 0.14)) + (-0.10 if c_i % 2 == 1 else 0.05)
                    rot = np.radians(float(rng.uniform(-2.8, 2.2)))
                    c_rot, s_rot = np.cos(rot), np.sin(rot)
                    sculpted = _sculpt_ko_syllable_strokes(ch, rng, is_hero=is_line_hero, is_word_end=is_end)
                    if not sculpted:
                        cursor_x += 1.0
                        continue
                    all_p = np.concatenate([p for p, _, _ in sculpted], axis=0)
                    x0 = float(all_p[:, 0].min())
                    ch_w = float(all_p[:, 0].max() - x0)
                    for pts, s_weight, is_tail in sculpted:
                        px = (pts[:, 0] - x0) * sc
                        py = pts[:, 1] * sc + dy
                        rx = c_rot * px - s_rot * py + cursor_x
                        ry = s_rot * px + c_rot * py
                        line_strokes_norm.append((np.stack([rx, ry], axis=1), s_weight, is_tail))
                    # Clean positive syllable kerning so brush strokes never collide into blobs
                    cursor_x += (ch_w * sc) + 0.34 * sc

                # Render trailing grammatical particle smaller & tucked beside the stem
                for c_i, ch in enumerate(particle):
                    sc = 0.82 * float(rng.uniform(0.96, 1.04))
                    dy = float(rng.uniform(-0.04, 0.12))
                    sculpted = _sculpt_ko_syllable_strokes(ch, rng, is_hero=False, is_word_end=(c_i == len(particle) - 1))
                    if not sculpted:
                        cursor_x += 0.55
                        continue
                    all_p = np.concatenate([p for p, _, _ in sculpted], axis=0)
                    x0 = float(all_p[:, 0].min())
                    ch_w = float(all_p[:, 0].max() - x0)
                    for pts, s_weight, is_tail in sculpted:
                        px = (pts[:, 0] - x0) * sc + cursor_x
                        py = pts[:, 1] * sc + dy
                        line_strokes_norm.append((np.stack([px, py], axis=1), s_weight, is_tail))
                    cursor_x += (ch_w * sc) + 0.28 * sc

                cursor_x += 0.92  # Inter-word breathing room
            else:
                clean_w = word.strip(".,!?;:").lower()
                is_func = clean_w in _EN_FUNCTION_WORDS and not is_line_hero
                sc = 0.86 if is_func else (1.22 if is_line_hero else 1.05)
                s_weight = 0.78 if is_func else (1.02 if is_line_hero else 0.92)
                w_strokes, w_width = script_font.word(word, rng=rng, connectedness=0.92, neatness=0.90)
                if not w_strokes:
                    cursor_x += 0.8
                    continue
                # Apply authentic Copperplate 16-deg slant and graceful baseline
                slant_k = np.tan(np.radians(16.0))
                word_dy = float(rng.uniform(-0.06, 0.06))
                placed_w: list[np.ndarray] = []
                for s_i, s in enumerate(w_strokes):
                    px = s[:, 0] * sc
                    py = s[:, 1] * sc + word_dy
                    px = px - slant_k * py + cursor_x
                    pts = np.stack([px, py], axis=1)
                    # Add entry swash on line-initial capital or exit flourish on line-final word
                    if w_idx == 0 and s_i == 0 and len(pts) >= 3:
                        p0 = pts[0]
                        swash = _curve(
                            [[p0[0] - 0.72, p0[1] + 0.28], [p0[0] - 0.36, p0[1] - 0.22], p0.tolist()],
                            n=14,
                        )
                        pts = np.concatenate([swash[:-1], pts], axis=0)
                    elif w_idx == len(words) - 1 and s_i == len(w_strokes) - 1 and len(pts) >= 3:
                        pe = pts[-1]
                        swash = _curve(
                            [pe.tolist(), [pe[0] + 0.48, pe[1] + 0.18], [pe[0] + 0.95, pe[1] - 0.24]],
                            n=14,
                        )
                        pts = np.concatenate([pts, swash[1:]], axis=0)
                    placed_w.append(pts)
                    is_tail = (w_idx == len(words) - 1 and s_i == len(w_strokes) - 1)
                    line_strokes_norm.append((pts, s_weight, is_tail))

                all_wp = np.concatenate(placed_w, axis=0)
                cursor_x = float(all_wp[:, 0].max()) + 0.72

        if not line_strokes_norm:
            continue

        # Convert normalized calligraphic line to millimetres and auto-fit to card margins
        all_pts = np.concatenate([p for p, _, _ in line_strokes_norm], axis=0)
        xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        stagger_mm = (line_idx - 0.5 * (len(raw_lines) - 1)) * (3.8 if has_ko else 0.0)
        raw_lw_mm = (xmax - xmin) * base_size_mm
        max_allowed_mm = w_mm - 26.0 - 2.0 * abs(stagger_mm)
        eff_size_mm = base_size_mm * min(1.0, max_allowed_mm / max(raw_lw_mm, 1e-3))
        lw_mm = (xmax - xmin) * eff_size_mm

        x_offset_mm = 0.5 * (w_mm - lw_mm) + stagger_mm
        y_base_mm = 0.5 * h_mm + (line_idx - 0.5 * (len(raw_lines) - 1)) * line_pitch

        mm_strokes: list[tuple[np.ndarray, float, bool]] = []
        for pts, s_weight, is_tail in line_strokes_norm:
            xy_mm = np.stack(
                [
                    (pts[:, 0] - xmin) * eff_size_mm + x_offset_mm,
                    pts[:, 1] * eff_size_mm + y_base_mm,
                ],
                axis=1,
            )
            mm_strokes.append((xy_mm, s_weight, is_tail))

        line_inks.append(_strokes_to_calligraphy_ink(mm_strokes, text=line_str, has_ko=has_ko))

    page.lines = line_inks

    # Center entire composition vertically on the card
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

        if has_ko:
            # Place traditional vermilion carved stone seal (전각 낙관 인장) beside lower-right of calligraphy
            seal_size = 8.2
            sx0 = min(w_mm - 16.0, xmax + 4.5)
            sy0 = min(h_mm - 15.0, ymax - seal_size + 1.5)
            page.seal_box_mm = (sx0, sy0, sx0 + seal_size, sy0 + seal_size)

            # Subtle watercolor / sumi-ink wash accents in negative space
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
            # Add balanced Copperplate off-hand flourish beneath the English stanza
            fl_strokes = _build_offhand_flourish(
                cx_mm=0.5 * w_mm,
                y_mm=min(h_mm - 11.0, ymax + 7.5),
                span_w_mm=min(85.0, (xmax - xmin) * 0.72),
            )
            page.lines.append(_strokes_to_calligraphy_ink(fl_strokes, text="", has_ko=False))

    return page
