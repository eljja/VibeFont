"""Procedural single-stroke Hangul synthesizer.

Decomposes any Hangul syllable into Choseong (initial), Jungseong (medial),
and Jongseong (final), and composes them into natural handwriting stroke paths
in normalized units (x-height == 1, baseline y == 0, y down).
"""

from __future__ import annotations

import numpy as np

# Choseong: 19
# Jungseong: 21
# Jongseong: 28 (0 = None)

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


def _ellipse_strokes(cx: float, cy: float, rx: float, ry: float, n: int = 16) -> np.ndarray:
    """Natural handwriting ellipse starting from top-right and curving counter-clockwise."""
    th = np.linspace(np.pi / 2, -1.5 * np.pi, n)
    return np.stack([cx + rx * np.cos(th), cy + ry * np.sin(th)], axis=1)


def choseong_strokes(idx: int) -> list[np.ndarray]:
    """Returns list of (M, 2) strokes in normalized [0, 1] x [0, 1] box."""
    if idx == 0:  # ㄱ
        return [np.array([[0.05, 0.1], [0.95, 0.1], [0.95, 0.95]])]
    elif idx == 1:  # ㄲ
        s1 = np.array([[0.05, 0.1], [0.45, 0.1], [0.45, 0.95]])
        s2 = np.array([[0.55, 0.1], [0.95, 0.1], [0.95, 0.95]])
        return [s1, s2]
    elif idx == 2:  # ㄴ
        return [np.array([[0.1, 0.05], [0.1, 0.95], [0.95, 0.95]])]
    elif idx == 3:  # ㄷ
        return [
            np.array([[0.95, 0.1], [0.1, 0.1]]),
            np.array([[0.1, 0.1], [0.1, 0.95], [0.95, 0.95]]),
        ]
    elif idx == 4:  # ㄸ
        s1_top = np.array([[0.45, 0.1], [0.05, 0.1]])
        s1_bot = np.array([[0.05, 0.1], [0.05, 0.95], [0.45, 0.95]])
        s2_top = np.array([[0.95, 0.1], [0.55, 0.1]])
        s2_bot = np.array([[0.55, 0.1], [0.55, 0.95], [0.95, 0.95]])
        return [s1_top, s1_bot, s2_top, s2_bot]
    elif idx == 5:  # ㄹ
        return [
            np.array([[0.05, 0.1], [0.95, 0.1], [0.95, 0.5], [0.05, 0.5], [0.05, 0.95], [0.95, 0.95]])
        ]
    elif idx == 6:  # ㅁ
        return [
            np.array([[0.1, 0.05], [0.1, 0.95]]),
            np.array([[0.1, 0.05], [0.9, 0.05], [0.9, 0.95], [0.1, 0.95]]),
        ]
    elif idx == 7:  # ㅂ
        return [
            np.array([[0.1, 0.05], [0.1, 0.95]]),
            np.array([[0.9, 0.05], [0.9, 0.95]]),
            np.array([[0.1, 0.55], [0.9, 0.55]]),
            np.array([[0.1, 0.95], [0.9, 0.95]]),
        ]
    elif idx == 8:  # ㅃ
        # Two condensed ㅂ
        return [
            np.array([[0.05, 0.05], [0.05, 0.95]]),
            np.array([[0.45, 0.05], [0.45, 0.95]]),
            np.array([[0.05, 0.55], [0.45, 0.55]]),
            np.array([[0.05, 0.95], [0.45, 0.95]]),
            np.array([[0.55, 0.05], [0.55, 0.95]]),
            np.array([[0.95, 0.05], [0.95, 0.95]]),
            np.array([[0.55, 0.55], [0.95, 0.55]]),
            np.array([[0.55, 0.95], [0.95, 0.95]]),
        ]
    elif idx == 9:  # ㅅ
        return [
            np.array([[0.5, 0.05], [0.05, 0.95]]),
            np.array([[0.35, 0.45], [0.95, 0.95]]),
        ]
    elif idx == 10:  # ㅆ
        return [
            np.array([[0.25, 0.05], [0.05, 0.95]]),
            np.array([[0.18, 0.45], [0.45, 0.95]]),
            np.array([[0.75, 0.05], [0.55, 0.95]]),
            np.array([[0.68, 0.45], [0.95, 0.95]]),
        ]
    elif idx == 11:  # ㅇ
        return [_ellipse_strokes(0.5, 0.5, 0.42, 0.42)]
    elif idx == 12:  # ㅈ
        return [
            np.array([[0.05, 0.1], [0.95, 0.1]]),
            np.array([[0.5, 0.1], [0.05, 0.95]]),
            np.array([[0.35, 0.45], [0.95, 0.95]]),
        ]
    elif idx == 13:  # ㅉ
        return [
            np.array([[0.05, 0.1], [0.45, 0.1]]),
            np.array([[0.25, 0.1], [0.05, 0.95]]),
            np.array([[0.18, 0.45], [0.45, 0.95]]),
            np.array([[0.55, 0.1], [0.95, 0.1]]),
            np.array([[0.75, 0.1], [0.55, 0.95]]),
            np.array([[0.68, 0.45], [0.95, 0.95]]),
        ]
    elif idx == 14:  # ㅊ
        return [
            np.array([[0.5, 0.02], [0.5, 0.15]]),
            np.array([[0.05, 0.22], [0.95, 0.22]]),
            np.array([[0.5, 0.22], [0.05, 0.95]]),
            np.array([[0.35, 0.55], [0.95, 0.95]]),
        ]
    elif idx == 15:  # ㅋ
        return [
            np.array([[0.05, 0.1], [0.95, 0.1], [0.95, 0.95]]),
            np.array([[0.05, 0.5], [0.9, 0.5]]),
        ]
    elif idx == 16:  # ㅌ
        return [
            np.array([[0.95, 0.1], [0.1, 0.1]]),
            np.array([[0.1, 0.5], [0.85, 0.5]]),
            np.array([[0.1, 0.1], [0.1, 0.95], [0.95, 0.95]]),
        ]
    elif idx == 17:  # ㅍ
        return [
            np.array([[0.05, 0.1], [0.95, 0.1]]),
            np.array([[0.3, 0.1], [0.3, 0.95]]),
            np.array([[0.7, 0.1], [0.7, 0.95]]),
            np.array([[0.05, 0.95], [0.95, 0.95]]),
        ]
    elif idx == 18:  # ㅎ
        return [
            np.array([[0.5, 0.02], [0.5, 0.15]]),
            np.array([[0.1, 0.25], [0.9, 0.25]]),
            _ellipse_strokes(0.5, 0.65, 0.32, 0.30),
        ]
    return []


def jungseong_strokes(idx: int) -> list[np.ndarray]:
    """Returns list of (M, 2) strokes in normalized [0, 1] x [0, 1] box."""
    # ㅏ (0)
    if idx == 0:
        return [np.array([[0.5, 0.02], [0.5, 0.98]]), np.array([[0.5, 0.5], [0.95, 0.5]])]
    # ㅐ (1)
    elif idx == 1:
        return [
            np.array([[0.35, 0.02], [0.35, 0.98]]),
            np.array([[0.35, 0.5], [0.75, 0.5]]),
            np.array([[0.75, 0.02], [0.75, 0.98]]),
        ]
    # ㅑ (2)
    elif idx == 2:
        return [
            np.array([[0.5, 0.02], [0.5, 0.98]]),
            np.array([[0.5, 0.35], [0.95, 0.35]]),
            np.array([[0.5, 0.65], [0.95, 0.65]]),
        ]
    # ㅒ (3)
    elif idx == 3:
        return [
            np.array([[0.35, 0.02], [0.35, 0.98]]),
            np.array([[0.35, 0.35], [0.75, 0.35]]),
            np.array([[0.35, 0.65], [0.75, 0.65]]),
            np.array([[0.75, 0.02], [0.75, 0.98]]),
        ]
    # ㅓ (4)
    elif idx == 4:
        return [np.array([[0.05, 0.5], [0.5, 0.5]]), np.array([[0.5, 0.02], [0.5, 0.98]])]
    # ㅔ (5)
    elif idx == 5:
        return [
            np.array([[0.05, 0.5], [0.35, 0.5]]),
            np.array([[0.35, 0.02], [0.35, 0.98]]),
            np.array([[0.75, 0.02], [0.75, 0.98]]),
        ]
    # ㅕ (6)
    elif idx == 6:
        return [
            np.array([[0.05, 0.35], [0.5, 0.35]]),
            np.array([[0.05, 0.65], [0.5, 0.65]]),
            np.array([[0.5, 0.02], [0.5, 0.98]]),
        ]
    # ㅖ (7)
    elif idx == 7:
        return [
            np.array([[0.05, 0.35], [0.35, 0.35]]),
            np.array([[0.05, 0.65], [0.35, 0.65]]),
            np.array([[0.35, 0.02], [0.35, 0.98]]),
            np.array([[0.75, 0.02], [0.75, 0.98]]),
        ]
    # ㅗ (8)
    elif idx == 8:
        return [np.array([[0.5, 0.05], [0.5, 0.5]]), np.array([[0.05, 0.5], [0.95, 0.5]])]
    # ㅘ (9: ㅗ + ㅏ)
    elif idx == 9:
        return [
            np.array([[0.3, 0.1], [0.3, 0.5]]),
            np.array([[0.05, 0.5], [0.65, 0.5]]),
            np.array([[0.75, 0.05], [0.75, 0.95]]),
            np.array([[0.75, 0.5], [0.98, 0.5]]),
        ]
    # ㅙ (10: ㅗ + ㅐ)
    elif idx == 10:
        return [
            np.array([[0.25, 0.1], [0.25, 0.5]]),
            np.array([[0.05, 0.5], [0.55, 0.5]]),
            np.array([[0.65, 0.05], [0.65, 0.95]]),
            np.array([[0.65, 0.5], [0.9, 0.5]]),
            np.array([[0.9, 0.05], [0.9, 0.95]]),
        ]
    # ㅚ (11: ㅗ + ㅣ)
    elif idx == 11:
        return [
            np.array([[0.3, 0.1], [0.3, 0.5]]),
            np.array([[0.05, 0.5], [0.65, 0.5]]),
            np.array([[0.8, 0.05], [0.8, 0.95]]),
        ]
    # ㅛ (12)
    elif idx == 12:
        return [
            np.array([[0.35, 0.05], [0.35, 0.5]]),
            np.array([[0.65, 0.05], [0.65, 0.5]]),
            np.array([[0.05, 0.5], [0.95, 0.5]]),
        ]
    # ㅜ (13)
    elif idx == 13:
        return [np.array([[0.05, 0.5], [0.95, 0.5]]), np.array([[0.5, 0.5], [0.5, 0.95]])]
    # ㅝ (14: ㅜ + ㅓ)
    elif idx == 14:
        return [
            np.array([[0.05, 0.5], [0.65, 0.5]]),
            np.array([[0.35, 0.5], [0.35, 0.95]]),
            np.array([[0.55, 0.5], [0.8, 0.5]]),
            np.array([[0.8, 0.05], [0.8, 0.95]]),
        ]
    # ㅞ (15: ㅜ + ㅔ)
    elif idx == 15:
        return [
            np.array([[0.05, 0.5], [0.55, 0.5]]),
            np.array([[0.3, 0.5], [0.3, 0.95]]),
            np.array([[0.55, 0.5], [0.7, 0.5]]),
            np.array([[0.7, 0.05], [0.7, 0.95]]),
            np.array([[0.92, 0.05], [0.92, 0.95]]),
        ]
    # ㅟ (16: ㅜ + ㅣ)
    elif idx == 16:
        return [
            np.array([[0.05, 0.5], [0.65, 0.5]]),
            np.array([[0.35, 0.5], [0.35, 0.95]]),
            np.array([[0.8, 0.05], [0.8, 0.95]]),
        ]
    # ㅠ (17)
    elif idx == 17:
        return [
            np.array([[0.05, 0.5], [0.95, 0.5]]),
            np.array([[0.35, 0.5], [0.35, 0.95]]),
            np.array([[0.65, 0.5], [0.65, 0.95]]),
        ]
    # ㅡ (18)
    elif idx == 18:
        return [np.array([[0.05, 0.5], [0.95, 0.5]])]
    # ㅢ (19)
    elif idx == 19:
        return [np.array([[0.05, 0.5], [0.7, 0.5]]), np.array([[0.8, 0.05], [0.8, 0.95]])]
    # ㅣ (20)
    elif idx == 20:
        return [np.array([[0.5, 0.02], [0.5, 0.98]])]
    return []


JONGSEONG_MAP = [
    None,  # 0: None
    0,     # 1: ㄱ
    1,     # 2: ㄲ
    None,  # 3: ㄳ (composite handled separately)
    2,     # 4: ㄴ
    None,  # 5: ㄵ
    None,  # 6: ㄶ
    3,     # 7: ㄷ
    5,     # 8: ㄹ
    None,  # 9: ㄺ
    None,  # 10: ㄻ
    None,  # 11: ㄼ
    None,  # 12: ㄽ
    None,  # 13: ㄾ
    None,  # 14: ㄿ
    None,  # 15: ㅀ
    6,     # 16: ㅁ
    7,     # 17: ㅂ
    None,  # 18: ㅄ
    9,     # 19: ㅅ
    10,    # 20: ㅆ
    11,    # 21: ㅇ
    12,    # 22: ㅈ
    14,    # 23: ㅊ
    15,    # 24: ㅋ
    16,    # 25: ㅌ
    17,    # 26: ㅍ
    18,    # 27: ㅎ
]


def jongseong_strokes(idx: int) -> list[np.ndarray]:
    if idx == 0:
        return []
    base_idx = JONGSEONG_MAP[idx]
    if base_idx is not None:
        return choseong_strokes(base_idx)

    # Composite jongseong
    if idx == 3:  # ㄳ
        return choseong_strokes(0) + choseong_strokes(9)
    elif idx == 5:  # ㄵ
        return choseong_strokes(2) + choseong_strokes(12)
    elif idx == 6:  # ㄶ
        return choseong_strokes(2) + choseong_strokes(18)
    elif idx == 9:  # ㄺ
        return choseong_strokes(5) + choseong_strokes(0)
    elif idx == 10:  # ㄻ
        return choseong_strokes(5) + choseong_strokes(6)
    elif idx == 11:  # ㄼ
        return choseong_strokes(5) + choseong_strokes(7)
    elif idx == 12:  # ㄽ
        return choseong_strokes(5) + choseong_strokes(9)
    elif idx == 13:  # ㄾ
        return choseong_strokes(5) + choseong_strokes(16)
    elif idx == 14:  # ㄿ
        return choseong_strokes(5) + choseong_strokes(17)
    elif idx == 15:  # ㅀ
        return choseong_strokes(5) + choseong_strokes(18)
    elif idx == 18:  # ㅄ
        return choseong_strokes(7) + choseong_strokes(9)
    return []


def _transform_strokes(strokes: list[np.ndarray], x: float, y: float, w: float, h: float) -> list[np.ndarray]:
    out = []
    for s in strokes:
        if len(s) == 0:
            continue
        p = s.copy()
        p[:, 0] = x + p[:, 0] * w
        p[:, 1] = y + p[:, 1] * h
        out.append(p)
    return out


def syllable_strokes(char: str) -> list[np.ndarray]:
    """Given a Hangul character, returns natural handwriting strokes.

    Coordinates are in normalized font box: x in [0, 1], y in [-1, 0] (y down, baseline at 0).
    """
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):
        return []

    s_idx = code - 0xAC00
    c_idx = s_idx // 588
    j_idx = (s_idx % 588) // 28
    f_idx = s_idx % 28

    has_jong = f_idx > 0
    c_st = choseong_strokes(c_idx)
    j_st = jungseong_strokes(j_idx)
    f_st = jongseong_strokes(f_idx)

    strokes = []

    # Layout type
    if j_idx in JUNGSEONG_VERTICAL:
        # Vertical vowel: Choseong left, Jungseong right
        if not has_jong:
            strokes += _transform_strokes(c_st, 0.05, -0.92, 0.45, 0.85)
            strokes += _transform_strokes(j_st, 0.52, -0.98, 0.45, 0.96)
        else:
            strokes += _transform_strokes(c_st, 0.05, -0.92, 0.45, 0.50)
            strokes += _transform_strokes(j_st, 0.52, -0.98, 0.45, 0.58)
            strokes += _transform_strokes(f_st, 0.15, -0.38, 0.70, 0.36)

    elif j_idx in JUNGSEONG_HORIZONTAL:
        # Horizontal vowel: Choseong top, Jungseong bottom
        if not has_jong:
            strokes += _transform_strokes(c_st, 0.18, -0.95, 0.64, 0.48)
            strokes += _transform_strokes(j_st, 0.05, -0.48, 0.90, 0.45)
        else:
            strokes += _transform_strokes(c_st, 0.18, -0.95, 0.64, 0.38)
            strokes += _transform_strokes(j_st, 0.05, -0.58, 0.90, 0.30)
            strokes += _transform_strokes(f_st, 0.15, -0.32, 0.70, 0.30)

    else:
        # Complex vowel: Choseong top-left, Jungseong bottom & right
        if not has_jong:
            strokes += _transform_strokes(c_st, 0.08, -0.95, 0.45, 0.48)
            strokes += _transform_strokes(j_st, 0.05, -0.50, 0.90, 0.48)
        else:
            strokes += _transform_strokes(c_st, 0.08, -0.95, 0.45, 0.40)
            strokes += _transform_strokes(j_st, 0.05, -0.55, 0.90, 0.32)
            strokes += _transform_strokes(f_st, 0.15, -0.28, 0.70, 0.26)

    return strokes


def is_hangul(char: str) -> bool:
    if not char:
        return False
    code = ord(char[0])
    return (
        (0xAC00 <= code <= 0xD7A3)
        or (0x3131 <= code <= 0x318E)
        or (0x1100 <= code <= 0x11FF)
    )


def hangul_char_strokes(char: str) -> list[np.ndarray]:
    code = ord(char)
    if 0xAC00 <= code <= 0xD7A3:
        return syllable_strokes(char)
    if char in CHOSEONG_NAMES:
        idx = CHOSEONG_NAMES.index(char)
        return _transform_strokes(choseong_strokes(idx), 0.1, -0.9, 0.8, 0.8)
    if char in JUNGSEONG_NAMES:
        idx = JUNGSEONG_NAMES.index(char)
        return _transform_strokes(jungseong_strokes(idx), 0.1, -0.9, 0.8, 0.8)
    return []
