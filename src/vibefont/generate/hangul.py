"""Organic procedural Hangul handwriting synthesizer.

Decomposes any Hangul syllable into Choseong (initial), Jungseong (medial),
and Jongseong (final), and composes them into natural, curved human handwriting
stroke paths in normalized units (x-height == 1, baseline y == 0, y down).

Key design features:
  1. Proportion balance with Latin: Hangul syllables match English cap/ascender height
     (y in [-1.72, +0.18], width ~1.42 x-heights).
  2. Smooth spline trajectories: every stroke is a multi-point organic curve with
     natural entry/exit hooks, slight upward horizontal tilt (+4 deg), and rounded corners.
  3. Sub-syllable stochasticity: each Jamo receives independent micro-variations in
     position, scale, and slant when an RNG is provided.
  4. Cursive ligatures: consecutive strokes softly link when connectedness is high.
"""

from __future__ import annotations

import numpy as np

CHOSEONG_NAMES = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNGSEONG_NAMES = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JUNGSEONG_VERTICAL = {0, 1, 2, 3, 4, 5, 6, 7, 20}  # ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅣ
JUNGSEONG_HORIZONTAL = {8, 12, 13, 17, 18}         # ㅗ ㅛ ㅜ ㅠ ㅡ
JUNGSEONG_COMPLEX = {9, 10, 11, 14, 15, 16, 19}    # ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ


def _curve(ctrl: list[list[float]] | np.ndarray, n: int = 12) -> np.ndarray:
    """Evaluates a smooth B-spline / quadratic Bezier chain through control points."""
    pts = np.asarray(ctrl, dtype=np.float64)
    m = len(pts)
    if m == 2:
        # Add a gentle mid-point bow for natural human stroke curvature
        mid = 0.5 * (pts[0] + pts[1])
        d = pts[1] - pts[0]
        perp = np.array([-d[1], d[0]]) * 0.03
        pts = np.array([pts[0], mid + perp, pts[1]])
        m = 3

    if m == 3:
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t ** 2 * pts[2]

    # For m >= 4: clamped uniform quadratic B-spline approximation (smooth corners)
    segments = []
    steps_per_seg = max(4, n // (m - 2))
    for i in range(m - 2):
        p0 = pts[i] if i == 0 else 0.5 * (pts[i] + pts[i + 1])
        p1 = pts[i + 1]
        p2 = pts[i + 2] if i == m - 3 else 0.5 * (pts[i + 1] + pts[i + 2])
        t = np.linspace(0.0, 1.0, steps_per_seg, endpoint=(i == m - 3))[:, None]
        seg = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        segments.append(seg)
    return np.concatenate(segments, axis=0)


def _ellipse_strokes(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    n: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Natural handwriting loop starting near 11 o'clock, sweeping CCW with slight overlap."""
    start_ang = -0.55 * np.pi
    span_ang = 2.07 * np.pi
    if rng is not None:
        start_ang += float(rng.normal(0.0, 0.14))
        span_ang += float(rng.normal(0.0, 0.12))
        rx *= float(np.exp(rng.normal(0.0, 0.04)))
        ry *= float(np.exp(rng.normal(0.0, 0.04)))
    th = np.linspace(start_ang, start_ang + span_ang, n)
    x = cx + rx * np.cos(th) + 0.03 * rx * np.sin(2 * th)
    y = cy + ry * np.sin(th)
    return np.stack([x, y], axis=1)


def choseong_strokes(
    idx: int,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
) -> list[np.ndarray]:
    """Returns list of smooth curved (M, 2) strokes in normalized [0, 1] x [0, 1] box."""
    if idx == 0:  # ㄱ: tilted horizontal -> rounded corner -> sweeping down-left stroke
        return [_curve([[0.06, 0.16], [0.55, 0.10], [0.90, 0.10], [0.86, 0.55], [0.72, 0.94]], 14)]
    elif idx == 1:  # ㄲ
        s1 = _curve([[0.04, 0.16], [0.44, 0.10], [0.42, 0.55], [0.34, 0.94]], 12)
        s2 = _curve([[0.52, 0.16], [0.94, 0.10], [0.92, 0.55], [0.82, 0.94]], 12)
        return [s1, s2]
    elif idx == 2:  # ㄴ: gentle vertical -> rounded bottom-left -> upward-tilted horizontal
        return [_curve([[0.16, 0.08], [0.12, 0.58], [0.15, 0.90], [0.55, 0.92], [0.94, 0.85]], 14)]
    elif idx == 3:  # ㄷ
        return [
            _curve([[0.12, 0.16], [0.52, 0.11], [0.90, 0.09]], 10),
            _curve([[0.16, 0.14], [0.12, 0.58], [0.15, 0.90], [0.55, 0.92], [0.94, 0.86]], 14),
        ]
    elif idx == 4:  # ㄸ
        return [
            _curve([[0.05, 0.16], [0.44, 0.10]], 8),
            _curve([[0.08, 0.14], [0.06, 0.88], [0.45, 0.86]], 12),
            _curve([[0.54, 0.16], [0.94, 0.10]], 8),
            _curve([[0.56, 0.14], [0.54, 0.88], [0.95, 0.86]], 12),
        ]
    elif idx == 5:  # ㄹ: style-responsive allographs (3-stroke print, 2-stroke standard, 1-stroke ribbon)
        if connectedness < 0.35:
            return [
                _curve([[0.08, 0.15], [0.88, 0.09], [0.85, 0.46]], 11),
                _curve([[0.14, 0.49], [0.50, 0.47], [0.85, 0.45]], 9),
                _curve([[0.14, 0.49], [0.12, 0.90], [0.55, 0.92], [0.94, 0.85]], 12),
            ]
        elif connectedness >= 0.68:
            return [
                _curve([[0.08, 0.15], [0.88, 0.09], [0.85, 0.45], [0.14, 0.49], [0.12, 0.90], [0.94, 0.85]], 20)
            ]
        return [
            _curve([[0.08, 0.15], [0.88, 0.09], [0.85, 0.46], [0.14, 0.48]], 14),
            _curve([[0.14, 0.48], [0.12, 0.90], [0.55, 0.92], [0.94, 0.85]], 12),
        ]
    elif idx == 6:  # ㅁ
        return [
            _curve([[0.14, 0.12], [0.12, 0.52], [0.16, 0.92]], 10),
            _curve([[0.14, 0.14], [0.88, 0.10], [0.85, 0.88], [0.16, 0.90]], 16),
        ]
    elif idx == 7:  # ㅂ
        return [
            _curve([[0.16, 0.08], [0.13, 0.52], [0.16, 0.92]], 10),
            _curve([[0.84, 0.08], [0.86, 0.52], [0.82, 0.92]], 10),
            _curve([[0.15, 0.52], [0.50, 0.49], [0.84, 0.47]], 8),
            _curve([[0.16, 0.90], [0.50, 0.88], [0.83, 0.86]], 8),
        ]
    elif idx == 8:  # ㅃ
        return [
            _curve([[0.06, 0.08], [0.06, 0.90], [0.44, 0.88], [0.44, 0.08]], 14),
            _curve([[0.06, 0.50], [0.44, 0.47]], 8),
            _curve([[0.54, 0.08], [0.54, 0.90], [0.92, 0.88], [0.92, 0.08]], 14),
            _curve([[0.54, 0.50], [0.92, 0.47]], 8),
        ]
    elif idx == 9:  # ㅅ
        by = 0.44 + (float(rng.normal(0.0, 0.03)) if rng is not None else 0.0)
        return [
            _curve([[0.52, 0.08], [0.36, 0.52], [0.08, 0.92]], 10),
            _curve([[0.38, by], [0.66, 0.68], [0.92, 0.90]], 10),
        ]
    elif idx == 10:  # ㅆ
        return [
            _curve([[0.28, 0.08], [0.18, 0.52], [0.04, 0.92]], 9),
            _curve([[0.20, 0.45], [0.35, 0.68], [0.46, 0.90]], 9),
            _curve([[0.76, 0.08], [0.66, 0.52], [0.52, 0.92]], 9),
            _curve([[0.68, 0.45], [0.82, 0.68], [0.95, 0.90]], 9),
        ]
    elif idx == 11:  # ㅇ
        return [_ellipse_strokes(0.50, 0.50, 0.39, 0.40, 18, rng=rng)]
    elif idx == 12:  # ㅈ
        return [
            _curve([[0.10, 0.16], [0.52, 0.10], [0.88, 0.10], [0.48, 0.52], [0.08, 0.92]], 14),
            _curve([[0.42, 0.48], [0.68, 0.70], [0.92, 0.90]], 10),
        ]
    elif idx == 13:  # ㅉ
        return [
            _curve([[0.04, 0.16], [0.44, 0.10], [0.24, 0.52], [0.04, 0.92]], 12),
            _curve([[0.22, 0.50], [0.46, 0.88]], 8),
            _curve([[0.54, 0.16], [0.94, 0.10], [0.74, 0.52], [0.54, 0.92]], 12),
            _curve([[0.72, 0.50], [0.96, 0.88]], 8),
        ]
    elif idx == 14:  # ㅊ
        return [
            _curve([[0.38, 0.04], [0.52, 0.09], [0.64, 0.12]], 8),
            _curve([[0.08, 0.26], [0.52, 0.20], [0.88, 0.20], [0.48, 0.58], [0.08, 0.94]], 14),
            _curve([[0.42, 0.54], [0.68, 0.74], [0.92, 0.92]], 10),
        ]
    elif idx == 15:  # ㅋ
        return [
            _curve([[0.08, 0.16], [0.55, 0.10], [0.90, 0.10], [0.86, 0.55], [0.74, 0.94]], 14),
            _curve([[0.12, 0.52], [0.50, 0.48], [0.84, 0.45]], 9),
        ]
    elif idx == 16:  # ㅌ
        return [
            _curve([[0.14, 0.15], [0.52, 0.10], [0.88, 0.08]], 9),
            _curve([[0.16, 0.50], [0.50, 0.47], [0.82, 0.45]], 9),
            _curve([[0.16, 0.14], [0.12, 0.56], [0.15, 0.90], [0.55, 0.92], [0.92, 0.85]], 14),
        ]
    elif idx == 17:  # ㅍ
        return [
            _curve([[0.08, 0.15], [0.50, 0.10], [0.92, 0.08]], 9),
            _curve([[0.32, 0.14], [0.30, 0.52], [0.28, 0.88]], 9),
            _curve([[0.68, 0.12], [0.66, 0.50], [0.64, 0.86]], 9),
            _curve([[0.06, 0.90], [0.50, 0.87], [0.94, 0.84]], 9),
        ]
    elif idx == 18:  # ㅎ
        return [
            _curve([[0.36, 0.04], [0.50, 0.08], [0.64, 0.11]], 8),
            _curve([[0.08, 0.25], [0.50, 0.20], [0.92, 0.18]], 9),
            _ellipse_strokes(0.50, 0.64, 0.34, 0.29, 16, rng=rng),
        ]
    return []


def jungseong_strokes(idx: int) -> list[np.ndarray]:
    """Returns list of smooth curved (M, 2) strokes in normalized [0, 1] x [0, 1] box."""
    v_main = lambda x: _curve([[x, 0.02], [x - 0.02, 0.52], [x, 0.98]], 12)
    h_main = lambda y: _curve([[0.04, y + 0.03], [0.50, y - 0.01], [0.96, y - 0.04]], 11)

    if idx == 0:  # ㅏ
        return [v_main(0.42), _curve([[0.42, 0.48], [0.70, 0.46], [0.94, 0.44]], 8)]
    elif idx == 1:  # ㅐ
        return [v_main(0.32), _curve([[0.32, 0.48], [0.54, 0.46], [0.74, 0.45]], 8), v_main(0.76)]
    elif idx == 2:  # ㅑ
        return [
            v_main(0.42),
            _curve([[0.42, 0.35], [0.68, 0.33], [0.94, 0.31]], 8),
            _curve([[0.42, 0.62], [0.68, 0.60], [0.94, 0.58]], 8),
        ]
    elif idx == 3:  # ㅒ
        return [
            v_main(0.32),
            _curve([[0.32, 0.35], [0.54, 0.33], [0.74, 0.32]], 8),
            _curve([[0.32, 0.62], [0.54, 0.60], [0.74, 0.59]], 8),
            v_main(0.76),
        ]
    elif idx == 4:  # ㅓ
        return [_curve([[0.06, 0.50], [0.34, 0.47], [0.60, 0.45]], 8), v_main(0.62)]
    elif idx == 5:  # ㅔ
        return [_curve([[0.04, 0.50], [0.22, 0.48], [0.40, 0.46]], 8), v_main(0.42), v_main(0.80)]
    elif idx == 6:  # ㅕ
        return [
            _curve([[0.06, 0.36], [0.34, 0.34], [0.60, 0.32]], 8),
            _curve([[0.06, 0.62], [0.34, 0.60], [0.60, 0.58]], 8),
            v_main(0.62),
        ]
    elif idx == 7:  # ㅖ
        return [
            _curve([[0.04, 0.36], [0.22, 0.34], [0.40, 0.33]], 8),
            _curve([[0.04, 0.62], [0.22, 0.60], [0.40, 0.59]], 8),
            v_main(0.42),
            v_main(0.80),
        ]
    elif idx == 8:  # ㅗ
        return [_curve([[0.50, 0.06], [0.49, 0.30], [0.50, 0.52]], 8), h_main(0.52)]
    elif idx == 9:  # ㅘ (left ㅗ sits below choseong in y=[0.54..0.78], right ㅏ spans full y=[0.02..0.98])
        return [
            _curve([[0.28, 0.54], [0.27, 0.66], [0.28, 0.78]], 8),
            _curve([[0.02, 0.80], [0.30, 0.76], [0.58, 0.72]], 9),
            v_main(0.70),
            _curve([[0.70, 0.48], [0.84, 0.46], [0.98, 0.44]], 8),
        ]
    elif idx == 10:  # ㅙ
        return [
            _curve([[0.24, 0.54], [0.23, 0.66], [0.24, 0.78]], 8),
            _curve([[0.02, 0.80], [0.28, 0.76], [0.52, 0.72]], 9),
            v_main(0.62),
            _curve([[0.62, 0.48], [0.76, 0.46], [0.88, 0.45]], 8),
            v_main(0.90),
        ]
    elif idx == 11:  # ㅚ
        return [
            _curve([[0.30, 0.54], [0.29, 0.66], [0.30, 0.78]], 8),
            _curve([[0.02, 0.80], [0.32, 0.76], [0.62, 0.72]], 9),
            v_main(0.76),
        ]
    elif idx == 12:  # ㅛ
        return [
            _curve([[0.35, 0.06], [0.34, 0.30], [0.35, 0.52]], 8),
            _curve([[0.65, 0.06], [0.64, 0.30], [0.65, 0.52]], 8),
            h_main(0.52),
        ]
    elif idx == 13:  # ㅜ
        return [h_main(0.46), _curve([[0.50, 0.46], [0.49, 0.72], [0.50, 0.96]], 9)]
    elif idx == 14:  # ㅝ (left ㅜ sits below choseong in y=[0.56..0.92], right ㅓ spans full y=[0.02..0.98])
        return [
            _curve([[0.02, 0.58], [0.32, 0.55], [0.60, 0.52]], 9),
            _curve([[0.30, 0.56], [0.29, 0.74], [0.30, 0.92]], 8),
            _curve([[0.48, 0.68], [0.62, 0.66], [0.76, 0.64]], 8),
            v_main(0.78),
        ]
    elif idx == 15:  # ㅞ
        return [
            _curve([[0.02, 0.58], [0.28, 0.55], [0.54, 0.52]], 9),
            _curve([[0.26, 0.56], [0.25, 0.74], [0.26, 0.92]], 8),
            _curve([[0.44, 0.68], [0.55, 0.66], [0.66, 0.64]], 8),
            v_main(0.68),
            v_main(0.90),
        ]
    elif idx == 16:  # ㅟ
        return [
            _curve([[0.02, 0.58], [0.32, 0.55], [0.62, 0.52]], 9),
            _curve([[0.32, 0.56], [0.31, 0.74], [0.32, 0.92]], 8),
            v_main(0.78),
        ]
    elif idx == 17:  # ㅠ
        return [
            h_main(0.46),
            _curve([[0.35, 0.46], [0.34, 0.72], [0.35, 0.96]], 8),
            _curve([[0.65, 0.46], [0.64, 0.72], [0.65, 0.96]], 8),
        ]
    elif idx == 18:  # ㅡ
        return [h_main(0.50)]
    elif idx == 19:  # ㅢ (left ㅡ sits below choseong at y=0.68, right ㅣ spans full y=[0.02..0.98])
        return [_curve([[0.02, 0.68], [0.34, 0.65], [0.66, 0.62]], 9), v_main(0.78)]
    elif idx == 20:  # ㅣ
        return [v_main(0.50)]
    return []


JONGSEONG_MAP = [
    None, 0, 1, None, 2, None, None, 3, 5, None,
    None, None, None, None, None, None, 6, 7, None, 9,
    10, 11, 12, 14, 15, 16, 17, 18,
]

COMPOSITE_JONGSEONG = {
    3: (0, 9),    # ㄳ
    5: (2, 12),   # ㄵ
    6: (2, 18),   # ㄶ
    9: (5, 0),    # ㄺ
    10: (5, 6),   # ㄻ
    11: (5, 7),   # ㄼ
    12: (5, 9),   # ㄽ
    13: (5, 16),  # ㄾ
    14: (5, 17),  # ㄿ
    15: (5, 18),  # ㅀ
    18: (7, 9),   # ㅄ
}


def jongseong_strokes(
    idx: int,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
) -> list[np.ndarray]:
    if idx == 0:
        return []
    base_idx = JONGSEONG_MAP[idx]
    if base_idx is not None:
        return choseong_strokes(base_idx, rng=rng, connectedness=connectedness)
    if idx in COMPOSITE_JONGSEONG:
        left_c, right_c = COMPOSITE_JONGSEONG[idx]
        s_left = _transform_strokes(choseong_strokes(left_c, rng=rng, connectedness=connectedness), 0.0, 0.0, 0.47, 1.0)
        s_right = _transform_strokes(choseong_strokes(right_c, rng=rng, connectedness=connectedness), 0.53, 0.0, 0.47, 1.0)
        return s_left + s_right
    return []


def _transform_strokes(
    strokes: list[np.ndarray],
    x: float,
    y: float,
    w: float,
    h: float,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    if rng is not None:
        x += float(rng.normal(0.0, 0.022 * w))
        y += float(rng.normal(0.0, 0.022 * h))
        w *= float(np.exp(rng.normal(0.0, 0.035)))
        h *= float(np.exp(rng.normal(0.0, 0.035)))
        rot = float(np.radians(rng.normal(0.0, 1.2)))
    else:
        rot = 0.0

    c, s_rot = np.cos(rot), np.sin(rot)
    cx, cy = x + 0.5 * w, y + 0.5 * h

    out = []
    for s in strokes:
        if len(s) == 0:
            continue
        p = s.copy()
        px = x + p[:, 0] * w
        py = y + p[:, 1] * h
        if rot != 0.0:
            dx, dy = px - cx, py - cy
            px = cx + c * dx - s_rot * dy
            py = cy + s_rot * dx + c * dy
        out.append(np.stack([px, py], axis=1))
    return out


def _soft_connect(strokes: list[np.ndarray], connectedness: float, rng: np.random.Generator | None) -> list[np.ndarray]:
    """Cursive ligature: occasionally connects nearby consecutive strokes within a syllable."""
    if connectedness < 0.35 or len(strokes) < 2:
        return strokes
    p_link = 0.55 * (connectedness - 0.3)
    out = [strokes[0]]
    for s in strokes[1:]:
        prev_end = out[-1][-1]
        cur_start = s[0]
        dist = float(np.linalg.norm(cur_start - prev_end))
        roll = float(rng.random()) if rng is not None else 0.5
        if dist < 0.55 and roll < p_link:
            bridge = _curve([prev_end.tolist(), cur_start.tolist()], n=6)
            out[-1] = np.concatenate([out[-1], bridge[1:-1], s], axis=0)
        else:
            out.append(s)
    return out


def syllable_strokes(
    char: str,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
) -> list[np.ndarray]:
    """Given a Hangul syllable, returns natural handwriting strokes.

    Normalized font box matches English cap/ascender height:
      x in [0.0, 1.42], y in [-1.72, +0.18] (baseline at y == 0, y down).
    """
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):
        return []

    s_idx = code - 0xAC00
    c_idx = s_idx // 588
    j_idx = (s_idx % 588) // 28
    f_idx = s_idx % 28

    has_jong = f_idx > 0
    c_st = choseong_strokes(c_idx, rng=rng, connectedness=connectedness)
    j_st = jungseong_strokes(j_idx)
    f_st = jongseong_strokes(f_idx, rng=rng, connectedness=connectedness)

    # Syllables without jongseong are slightly more compact vertically than 3-tier syllables with jongseong
    if has_jong:
        W = 1.60
        Y_TOP = -1.88
        H = 2.08
    else:
        W = 1.50
        Y_TOP = -1.78
        H = 1.82

    strokes: list[np.ndarray] = []

    if j_idx in JUNGSEONG_VERTICAL:
        if not has_jong:
            strokes += _transform_strokes(c_st, 0.04 * W, Y_TOP + 0.08 * H, 0.44 * W, 0.80 * H, rng)
            strokes += _transform_strokes(j_st, 0.50 * W, Y_TOP + 0.02 * H, 0.46 * W, 0.94 * H, rng)
        else:
            strokes += _transform_strokes(c_st, 0.05 * W, Y_TOP + 0.05 * H, 0.43 * W, 0.50 * H, rng)
            strokes += _transform_strokes(j_st, 0.50 * W, Y_TOP + 0.02 * H, 0.45 * W, 0.56 * H, rng)
            strokes += _transform_strokes(f_st, 0.16 * W, Y_TOP + 0.60 * H, 0.68 * W, 0.37 * H, rng)

    elif j_idx in JUNGSEONG_HORIZONTAL:
        if not has_jong:
            strokes += _transform_strokes(c_st, 0.16 * W, Y_TOP + 0.04 * H, 0.66 * W, 0.46 * H, rng)
            strokes += _transform_strokes(j_st, 0.05 * W, Y_TOP + 0.52 * H, 0.90 * W, 0.44 * H, rng)
        else:
            strokes += _transform_strokes(c_st, 0.18 * W, Y_TOP + 0.03 * H, 0.64 * W, 0.36 * H, rng)
            strokes += _transform_strokes(j_st, 0.06 * W, Y_TOP + 0.40 * H, 0.88 * W, 0.24 * H, rng)
            strokes += _transform_strokes(f_st, 0.16 * W, Y_TOP + 0.65 * H, 0.68 * W, 0.33 * H, rng)

    else:
        # Complex vowel (ㅘ, ㅙ, ㅚ, ㅝ, ㅞ, ㅟ, ㅢ): j_st spans from top (Y_TOP + 0.02*H) so right vertical stem is full-height!
        if not has_jong:
            strokes += _transform_strokes(c_st, 0.06 * W, Y_TOP + 0.05 * H, 0.44 * W, 0.42 * H, rng)
            strokes += _transform_strokes(j_st, 0.04 * W, Y_TOP + 0.02 * H, 0.92 * W, 0.94 * H, rng)
        else:
            strokes += _transform_strokes(c_st, 0.06 * W, Y_TOP + 0.03 * H, 0.44 * W, 0.30 * H, rng)
            strokes += _transform_strokes(j_st, 0.04 * W, Y_TOP + 0.02 * H, 0.92 * W, 0.58 * H, rng)
            strokes += _transform_strokes(f_st, 0.16 * W, Y_TOP + 0.63 * H, 0.68 * W, 0.35 * H, rng)

    return _soft_connect(strokes, connectedness, rng)


def is_hangul(char: str) -> bool:
    if not char:
        return False
    code = ord(char[0])
    return (
        (0xAC00 <= code <= 0xD7A3)
        or (0x3131 <= code <= 0x318E)
        or (0x1100 <= code <= 0x11FF)
    )


def hangul_char_strokes(
    char: str,
    rng: np.random.Generator | None = None,
    connectedness: float = 0.5,
) -> list[np.ndarray]:
    code = ord(char)
    if 0xAC00 <= code <= 0xD7A3:
        return syllable_strokes(char, rng=rng, connectedness=connectedness)
    if char in CHOSEONG_NAMES:
        idx = CHOSEONG_NAMES.index(char)
        st = _transform_strokes(choseong_strokes(idx), 0.1, -1.35, 1.15, 1.35, rng)
        return _soft_connect(st, connectedness, rng)
    if char in JUNGSEONG_NAMES:
        idx = JUNGSEONG_NAMES.index(char)
        st = _transform_strokes(jungseong_strokes(idx), 0.1, -1.35, 1.15, 1.35, rng)
        return _soft_connect(st, connectedness, rng)
    return []
