"""Pure-geometry ligature and cursive connection engine.

Handles glyph anchor detection, natural tail/head blending, delayed stroke extraction,
and cross-glyph / cross-jamo cursive joins without CAPTCHA-like distortion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class PlacedGlyph:
    ch: str
    strokes: list[np.ndarray]
    jitter: float = 0.0


@dataclass(frozen=True)
class GlyphAnchors:
    entry_stroke: int | None
    exit_stroke: int | None


@dataclass
class JoinInfo:
    prev_ch: str
    next_ch: str
    point: tuple[float, float]
    turn_deg: float
    path: np.ndarray | None = None


@dataclass
class ConnectResult:
    strokes: list[np.ndarray]
    joins: list[JoinInfo] = field(default_factory=list)


HIGH_EXIT = frozenset("bovw")


def connect_word(
    glyphs: list[PlacedGlyph],
    rng: np.random.Generator,
    p_join: float,
    writer_key: int = 0,
) -> ConnectResult:
    """Connect lowercase glyphs within a word into continuous cursive strokes."""
    if not glyphs:
        return ConnectResult([])

    glyph_bodies: list[tuple[str, list[np.ndarray]]] = []
    all_delayed: list[np.ndarray] = []

    for g in glyphs:
        if not g.strokes:
            continue
        merged = merge_glyph_internal(g.strokes)
        anchors = find_anchors(merged)
        body, delayed = split_delayed(merged, anchors)
        glyph_bodies.append((g.ch, body))
        all_delayed.extend(delayed)

    if not glyph_bodies:
        return ConnectResult(all_delayed)

    accum_strokes: list[np.ndarray] = []
    joins: list[JoinInfo] = []

    for idx, (ch, body) in enumerate(glyph_bodies):
        if not body:
            continue

        if not accum_strokes:
            accum_strokes.extend([s.copy() for s in body])
            continue

        prev_ch = glyph_bodies[idx - 1][0]
        can_join = (
            prev_ch.islower()
            and ch.islower()
            and (rng.random() < p_join)
            and len(accum_strokes[-1]) >= 2
            and len(body[0]) >= 2
        )

        joined = None
        if can_join:
            tail_a = accum_strokes[-1]
            head_b = body[0]
            meet_y = -0.98 if prev_ch in HIGH_EXIT else None
            joined = join_strokes(tail_a, head_b, rng, zone=0.20, max_gap=0.55, meet_y=meet_y)

        if joined is not None:
            j_idx = len(accum_strokes[-1]) - 1
            turn = turn_angle_deg(joined, j_idx, k=1)
            if turn <= 75.0:
                join_pt = (float(joined[j_idx, 0]), float(joined[j_idx, 1]))
                joins.append(JoinInfo(prev_ch=prev_ch, next_ch=ch, point=join_pt, turn_deg=turn))
                accum_strokes[-1] = joined
                if len(body) > 1:
                    accum_strokes.extend([s.copy() for s in body[1:]])
            else:
                accum_strokes.extend([s.copy() for s in body])
        else:
            accum_strokes.extend([s.copy() for s in body])

    accum_strokes.extend(all_delayed)
    return ConnectResult(strokes=accum_strokes, joins=joins)


def turn_angle_deg(path: np.ndarray, idx: int, k: int = 3) -> float:
    """Calculate the turning angle in degrees at path[idx] using step k."""
    if len(path) < 3 or idx <= 0 or idx >= len(path) - 1:
        return 0.0
    i_prev = max(0, idx - k)
    i_next = min(len(path) - 1, idx + k)
    v1 = path[idx] - path[i_prev]
    v2 = path[i_next] - path[idx]
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    cos_theta = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Check if line segment p1-p2 intersects p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def polylines_intersect(a: np.ndarray, b: np.ndarray, ignore_endpoints: bool = True) -> bool:
    """Check if any segment in polyline a intersects any segment in polyline b."""
    if len(a) < 2 or len(b) < 2:
        return False
    # Bounding box quick check
    min_a, max_a = a.min(axis=0), a.max(axis=0)
    min_b, max_b = b.min(axis=0), b.max(axis=0)
    if (max_a[0] < min_b[0] or min_a[0] > max_b[0] or
        max_a[1] < min_b[1] or min_a[1] > max_b[1]):
        return False

    n_a = len(a) - 1
    n_b = len(b) - 1
    for i in range(n_a):
        seg_a1, seg_a2 = a[i], a[i + 1]
        for j in range(n_b):
            seg_b1, seg_b2 = b[j], b[j + 1]
            if ignore_endpoints:
                # If segments share an endpoint, skip
                if (np.linalg.norm(seg_a1 - seg_b1) < 1e-6 or
                    np.linalg.norm(seg_a1 - seg_b2) < 1e-6 or
                    np.linalg.norm(seg_a2 - seg_b1) < 1e-6 or
                    np.linalg.norm(seg_a2 - seg_b2) < 1e-6):
                    continue
            if _segments_intersect(seg_a1, seg_a2, seg_b1, seg_b2):
                return True
    return False


def find_anchors(strokes: list[np.ndarray], connector_y: float = -0.56, tol: float = 0.45) -> GlyphAnchors:
    """Locate the entry stroke index and exit stroke index for a glyph."""
    valid_strokes = [(i, s) for i, s in enumerate(strokes) if len(s) > 0]
    if not valid_strokes:
        return GlyphAnchors(None, None)

    entries = [(i, s[0][0]) for i, s in valid_strokes if abs(s[0][1] - connector_y) <= tol]
    exits = [(i, s[-1][0]) for i, s in valid_strokes if abs(s[-1][1] - connector_y) <= tol]

    best_entry = min(entries, key=lambda x: x[1])[0] if entries else None
    best_exit = max(exits, key=lambda x: x[1])[0] if exits else None

    return GlyphAnchors(best_entry, best_exit)


def _arc_lengths(pts: np.ndarray) -> np.ndarray:
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(dists)])


def join_strokes(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    zone: float = 0.20,
    max_gap: float = 0.45,
    meet_y: float | None = None,
) -> np.ndarray | None:
    """Join tail of stroke a and head of stroke b by blending toward meeting point m."""
    if len(a) < 2 or len(b) < 2:
        return None

    tail_a = a[-1]
    head_b = b[0]
    gap = float(np.linalg.norm(head_b - tail_a))
    if gap > max_gap:
        return None

    mid = 0.5 * (tail_a + head_b)
    if meet_y is not None:
        mid[1] = meet_y

    arc_a = _arc_lengths(a)
    tot_a = arc_a[-1]
    z_a = max(min(tot_a * zone, 0.6), 1e-4)

    a_mod = a.copy()
    dist_from_tail = tot_a - arc_a
    mask_a = dist_from_tail <= z_a
    w_a = np.clip(1.0 - dist_from_tail[mask_a] / z_a, 0.0, 1.0)
    s_a = w_a * w_a * (3.0 - 2.0 * w_a)  # smoothstep
    a_mod[mask_a] += (mid - tail_a)[None, :] * s_a[:, None]

    arc_b = _arc_lengths(b)
    tot_b = arc_b[-1]
    z_b = max(min(tot_b * zone, 0.6), 1e-4)

    b_mod = b.copy()
    dist_from_head = arc_b
    mask_b = dist_from_head <= z_b
    w_b = np.clip(1.0 - dist_from_head[mask_b] / z_b, 0.0, 1.0)
    s_b = w_b * w_b * (3.0 - 2.0 * w_b)
    b_mod[mask_b] += (mid - head_b)[None, :] * s_b[:, None]

    joined = np.concatenate([a_mod, b_mod[1:]], axis=0)
    j_idx = len(a) - 1

    # Local smoothing over ±4 samples at junction
    k_smooth = 4
    i_start = max(j_idx - k_smooth, 0)
    i_end = min(j_idx + k_smooth + 1, len(joined))
    if i_end - i_start >= 3:
        sub = joined[i_start:i_end].copy()
        sub_smooth = gaussian_filter1d(sub, sigma=1.5, axis=0, mode="nearest")
        # Apply only to strictly modified samples
        joined[i_start:i_end] = sub_smooth

    return joined


def merge_glyph_internal(strokes: list[np.ndarray]) -> list[np.ndarray]:
    """Join internal strokes of multi-stroke glyphs (e.g. 'm', 'n', 'h', 'k')."""
    if len(strokes) <= 1:
        return strokes

    out: list[np.ndarray] = [strokes[0].copy()]
    for s in strokes[1:]:
        if len(s) == 0:
            continue
        prev = out[-1]
        gap = float(np.linalg.norm(prev[-1] - s[0]))
        if gap <= 1e-6:
            out[-1] = np.concatenate([prev, s[1:]], axis=0)
            continue

        # Check if s[0] touches any point along prev (within 0.05)
        dists = np.linalg.norm(prev - s[0], axis=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= 0.05:
            # Retrace from prev[-1] back to min_idx, then append s
            if min_idx < len(prev) - 1:
                retrace = prev[-1:min_idx:-1]
                out[-1] = np.concatenate([prev, retrace, s], axis=0)
            else:
                out[-1] = np.concatenate([prev, s], axis=0)
            continue

        if gap <= 0.35:
            # Direct link
            out[-1] = np.concatenate([prev, s], axis=0)
            continue

        out.append(s.copy())

    return out


def split_delayed(strokes: list[np.ndarray], anchors: GlyphAnchors) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split strokes into body strokes and delayed strokes (e.g. dots on i/j, t-crossbar)."""
    body: list[np.ndarray] = []
    delayed: list[np.ndarray] = []

    for i, s in enumerate(strokes):
        if len(s) == 0:
            continue
        arc = float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))
        max_y = float(s[:, 1].max())

        is_anchor = (i == anchors.entry_stroke or i == anchors.exit_stroke)
        if (not is_anchor) and max_y < -0.90 and arc < 0.80:
            delayed.append(s)
        else:
            body.append(s)

    return body, delayed
