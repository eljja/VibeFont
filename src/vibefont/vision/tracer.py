"""Offline-to-online handwriting vectorizer.

Converts offline handwriting images (IAM words, Korean OCR syllables/words) into
online stroke trajectories (x, y, t, p, eos) with natural writing order heuristics,
2/3 power-law kinematics, Gaussian path smoothing, and corpus-aware normalization.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from ..ink import EOS, NCOLS, P, T, X, Y, Ink
from ..preprocess import resample_time


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Pure NumPy implementation of the Zhang-Suen morphological thinning algorithm."""
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            p2 = np.pad(img[:-1, :], ((1, 0), (0, 0)))
            p3 = np.pad(img[:-1, 1:], ((1, 0), (0, 1)))
            p4 = np.pad(img[:, 1:], ((0, 0), (0, 1)))
            p5 = np.pad(img[1:, 1:], ((0, 1), (0, 1)))
            p6 = np.pad(img[1:, :], ((0, 1), (0, 0)))
            p7 = np.pad(img[1:, :-1], ((0, 1), (1, 0)))
            p8 = np.pad(img[:, :-1], ((0, 0), (1, 0)))
            p9 = np.pad(img[:-1, :-1], ((1, 0), (1, 0)))

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )
            c1 = (neighbors >= 2) & (neighbors <= 6)
            c2 = (transitions == 1)
            if step == 1:
                c3 = (p2 * p4 * p6 == 0)
                c4 = (p4 * p6 * p8 == 0)
            else:
                c3 = (p2 * p4 * p8 == 0)
                c4 = (p2 * p6 * p8 == 0)
            candidates = (img == 1) & c1 & c2 & c3 & c4
            if np.any(candidates):
                img[candidates] = 0
                changed = True
    return img


def _extract_strokes_from_skeleton(skel: np.ndarray, dist_map: np.ndarray) -> list[np.ndarray]:
    """Traces 1D paths from a binary skeleton image with smoothing and 2/3 power-law kinematics."""
    rows, cols = np.where(skel > 0)
    if len(rows) == 0:
        return []

    pts_set = set(zip(rows, cols))
    visited = set()

    def get_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in pts_set:
                    nbrs.append((nr, nc))
        return nbrs

    degrees = {pt: len(get_neighbors(pt[0], pt[1])) for pt in pts_set}

    # Prioritize endpoints (degree 1) sorted top-to-bottom, left-to-right
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    endpoints.sort(key=lambda pt: (pt[0] * 1.5 + pt[1]))

    strokes: list[list[tuple[int, int]]] = []

    for start in endpoints:
        if start in visited:
            continue
        cur = start
        path = [cur]
        visited.add(cur)

        while True:
            unvisited_nbrs = [n for n in get_neighbors(cur[0], cur[1]) if n not in visited]
            if not unvisited_nbrs:
                break
            if len(path) > 1:
                prev = path[-2]
                v_cur = (cur[0] - prev[0], cur[1] - prev[1])
                unvisited_nbrs.sort(key=lambda n: -((n[0] - cur[0]) * v_cur[0] + (n[1] - cur[1]) * v_cur[1]))
            else:
                unvisited_nbrs.sort(key=lambda n: (n[0] * 1.5 + n[1]))

            nxt = unvisited_nbrs[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt

        if len(path) >= 2:
            strokes.append(path)

    # Process remaining components (closed loops like 'o', 'e', 'ㅇ', 'ㅁ')
    remaining = [pt for pt in pts_set if pt not in visited]
    remaining.sort(key=lambda pt: (pt[0] * 1.5 + pt[1]))

    for start in remaining:
        if start in visited:
            continue
        cur = start
        path = [cur]
        visited.add(cur)

        while True:
            unvisited_nbrs = [n for n in get_neighbors(cur[0], cur[1]) if n not in visited]
            if not unvisited_nbrs:
                break
            nxt = unvisited_nbrs[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt

        if len(path) >= 2:
            strokes.append(path)

    # Sort strokes in natural writing order (left-to-right primary, top-to-bottom secondary)
    strokes.sort(key=lambda st: (st[0][1] * 0.8 + st[0][0] * 1.2))

    current_time = 0.0
    out_strokes = []

    for path in strokes:
        coords = np.array([[c, r] for r, c in path], dtype=np.float64)  # (x, y)
        n = len(coords)

        # Smooth 1-pixel staircase grid artifacts
        if n >= 4:
            coords[:, 0] = ndi.gaussian_filter1d(coords[:, 0], sigma=1.1, mode="nearest")
            coords[:, 1] = ndi.gaussian_filter1d(coords[:, 1], sigma=1.1, mode="nearest")

        d_vals = np.array([dist_map[r, c] for r, c in path], dtype=np.float64)
        median_d = np.median(d_vals) if len(d_vals) else 1.0
        p = np.clip(d_vals / max(median_d, 1e-4) * 0.5, 0.1, 0.95)
        if n >= 4:
            p = ndi.gaussian_filter1d(p, sigma=1.0, mode="nearest")

        # 2/3 power-law kinematics: speed slows down at high curvature
        diffs = np.diff(coords, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        if n >= 3:
            tangents = diffs / np.maximum(seg_lens[:, None], 1e-5)
            d_tan = np.linalg.norm(np.diff(tangents, axis=0), axis=1)
            curv = np.concatenate([[d_tan[0]], d_tan]) / np.maximum(seg_lens, 0.5)
        else:
            curv = np.zeros(len(seg_lens))

        v_base = 220.0
        v_inst = v_base * np.power(1.0 + 3.0 * np.clip(curv, 0.0, 10.0), -1.0 / 3.0)
        dt_segs = np.maximum(seg_lens / np.maximum(v_inst, 20.0), 0.004)
        times = np.concatenate([[current_time], current_time + np.cumsum(dt_segs)])
        current_time = times[-1] + 0.06  # 60ms pen-up transition

        eos = np.zeros(n, dtype=np.float64)
        eos[-1] = 1.0

        st_arr = np.empty((n, NCOLS), dtype=np.float32)
        st_arr[:, X] = coords[:, 0]
        st_arr[:, Y] = coords[:, 1]
        st_arr[:, T] = times
        st_arr[:, P] = p
        st_arr[:, EOS] = eos
        out_strokes.append(st_arr)

    return out_strokes


def _has_hangul(text: str) -> bool:
    return any(
        (0xAC00 <= ord(c) <= 0xD7A3) or (0x3130 <= ord(c) <= 0x318F) or (0x1100 <= ord(c) <= 0x11FF)
        for c in text
    )


def normalize_cropped_ink(ink: Ink, text: str = "") -> Ink:
    """Normalizes a horizontally-oriented cropped word/character Ink into model space.

    Model space: baseline at y == 0, x_min at x == 0, x-height == 1.0, y grows downward.
    Avoids unconstrained linear regression on extrema so single characters or short words
    with punctuation ('MR.', '가') never rotate wildly or explode in scale.
    """
    if len(ink) == 0:
        return ink

    pts = ink.points
    x = pts[:, X]
    y = pts[:, Y]

    if len(y) >= 20:
        y_top = float(np.percentile(y, 2))
        y_bot = float(np.percentile(y, 98))
    else:
        y_top = float(y.min())
        y_bot = float(y.max())

    span_y = max(y_bot - y_top, float(np.ptp(y)), 1.0)
    x_min = float(x.min())

    if _has_hangul(text):
        # Hangul syllable blocks sit from ~ -1.45 (top) to +0.10 (bottom) relative to Latin baseline
        base_y = y_top + 0.93 * span_y
        x_height = span_y / 1.55
    else:
        has_asc = any(c.isupper() or c in "bdfhklt0123456789!?" for c in text)
        has_desc = any(c in "gjpqy" for c in text)
        if has_asc and has_desc:
            base_y = y_top + (1.6 / 2.2) * span_y
            x_height = span_y / 2.2
        elif has_asc:
            base_y = y_bot - 0.04 * span_y
            x_height = span_y / 1.65
        elif has_desc:
            base_y = y_top + (1.0 / 1.6) * span_y
            x_height = span_y / 1.6
        else:
            base_y = y_bot - 0.04 * span_y
            x_height = span_y / 1.08

    x_height = max(x_height, 2.0)
    out = ink.translate(-x_min, -base_y).scale(1.0 / x_height)
    if len(out):
        out.points[:, T] -= out.points[0, T]
    return out


def trace_image_to_ink(
    img_or_path: Image.Image | np.ndarray | str | Path | bytes,
    text: str = "",
    hz: float = 100.0,
) -> Ink:
    """Converts a cropped handwriting image into a normalized online Ink object."""
    if isinstance(img_or_path, (str, Path)):
        img = Image.open(img_or_path).convert("L")
    elif isinstance(img_or_path, bytes):
        img = Image.open(io.BytesIO(img_or_path)).convert("L")
    elif isinstance(img_or_path, Image.Image):
        img = img_or_path.convert("L")
    else:
        img = Image.fromarray(img_or_path).convert("L")

    arr = np.array(img, dtype=np.float32)

    p20, p80 = np.percentile(arr, [20, 80])
    threshold = (p20 + p80) * 0.5
    binary_ink = (arr < threshold).astype(np.uint8)

    binary_ink = ndi.binary_opening(binary_ink, structure=np.ones((2, 2)))
    dist_map = ndi.distance_transform_edt(binary_ink)
    skel = zhang_suen_thinning(binary_ink)

    strokes = _extract_strokes_from_skeleton(skel, dist_map)
    if not strokes:
        return Ink(np.zeros((0, NCOLS), dtype=np.float32), text=text)

    all_pts = np.concatenate(strokes, axis=0)
    raw_ink = Ink(all_pts, text=text)

    resampled = resample_time(raw_ink, hz=hz)
    if len(resampled) < 2:
        return normalize_cropped_ink(raw_ink, text=text)

    return normalize_cropped_ink(resampled, text=text)
