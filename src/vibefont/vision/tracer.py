"""Offline-to-online handwriting vectorizer.

Converts offline handwriting images (IAM words, Korean OCR syllables/words) into
online stroke trajectories (x, y, t, p, eos) with natural writing order heuristics,
power-law kinematics, and uniform time resampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from ..ink import EOS, NCOLS, P, T, X, Y, Ink
from ..preprocess import normalize_line, resample_time


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
    """Traces 1D paths from a binary skeleton image."""
    rows, cols = np.where(skel > 0)
    if len(rows) == 0:
        return []

    pts_set = set(zip(rows, cols))
    visited = set()

    # Precompute degrees (8-connectivity)
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

    # Find starting points: prioritize endpoints (degree 1) sorted top-to-bottom, left-to-right
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
            # Choose neighbor closest to current direction or top-to-bottom
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

    # Sort strokes in natural writing sequence (left-to-right, top-to-bottom)
    strokes.sort(key=lambda st: (st[0][1] * 0.8 + st[0][0] * 1.2))

    # Convert paths to (M, 5) float32 strokes with simulated timestamps and pressure
    current_time = 0.0
    out_strokes = []

    for path in strokes:
        coords = np.array([[c, r] for r, c in path], dtype=np.float64)  # (x, y)
        n = len(coords)

        # Distance transform values at path pixels
        d_vals = np.array([dist_map[r, c] for r, c in path], dtype=np.float64)
        median_d = np.median(d_vals) if len(d_vals) else 1.0
        p = np.clip(d_vals / max(median_d, 1e-4) * 0.5, 0.1, 0.95)

        # Kinematics: arc length + curvature determines dt
        diffs = np.diff(coords, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)

        # Base writing speed ~ 200 pixels/sec
        v_base = 200.0
        dt_segs = np.maximum(seg_lens / v_base, 0.005)
        times = np.concatenate([[current_time], current_time + np.cumsum(dt_segs)])
        current_time = times[-1] + 0.08  # 80ms pen-up pause between strokes

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


def trace_image_to_ink(
    img_or_path: Image.Image | np.ndarray | str | Path,
    text: str = "",
    hz: float = 100.0,
) -> Ink:
    """Converts a cropped handwriting image into a normalized online Ink object."""
    if isinstance(img_or_path, (str, Path)):
        img = Image.open(img_or_path).convert("L")
    elif isinstance(img_or_path, bytes):
        import io
        img = Image.open(io.BytesIO(img_or_path)).convert("L")
    elif isinstance(img_or_path, Image.Image):
        img = img_or_path.convert("L")
    else:
        img = Image.fromarray(img_or_path).convert("L")

    arr = np.array(img, dtype=np.float32)

    # Automatic thresholding: Otsu or percentile
    # For scanned documents, paper is bright (>128) and ink is dark (<128)
    p20, p80 = np.percentile(arr, [20, 80])
    threshold = (p20 + p80) * 0.5
    binary_ink = (arr < threshold).astype(np.uint8)

    # Clean small isolated noise pixels
    binary_ink = ndi.binary_opening(binary_ink, structure=np.ones((2, 2)))

    # Compute Euclidean distance transform
    dist_map = ndi.distance_transform_edt(binary_ink)

    # Morphological skeletonization
    skel = zhang_suen_thinning(binary_ink)

    # Extract strokes
    strokes = _extract_strokes_from_skeleton(skel, dist_map)
    if not strokes:
        return Ink(np.zeros((0, NCOLS), dtype=np.float32), text=text)

    all_pts = np.concatenate(strokes, axis=0)
    raw_ink = Ink(all_pts, text=text)

    # Resample to uniform time and normalize
    resampled = resample_time(raw_ink, hz=hz)
    if len(resampled) < 2:
        return raw_ink

    normalized, _ = normalize_line(resampled)
    return normalized
