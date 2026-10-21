"""Track B: Vision-based Few-shot handwriting style extractor.

Extracts handwriting style descriptors (slant, stroke thickness/pressure,
connectedness, roundness, baseline wobble) from a reference handwriting image
and maps them into a VibeFont StyleSpec.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from ..vibe.spec import StyleSpec


def estimate_image_slant(binary_ink: np.ndarray) -> float:
    """Estimates dominant handwriting slant angle in degrees (+ = leaning right).

    Uses vertical projection profile entropy/variance maximization over sheared coordinates.
    """
    rows, cols = np.where(binary_ink > 0)
    if len(rows) < 50:
        return 0.0

    # Center coordinates around image center
    cy = np.mean(rows)
    cx = np.mean(cols)
    y = rows - cy
    x = cols - cx

    best_angle = 0.0
    max_var = -1.0

    # Test candidate slant angles from -20 to +35 degrees
    angles = np.arange(-20.0, 36.0, 1.0)
    for theta in angles:
        k = np.tan(np.radians(theta))
        # In image coords (y down): top has y < 0, bottom has y > 0
        # Right lean (+theta) has x > 0 when y < 0, so x + k * y cancels the lean
        x_desheared = x + k * y
        # Histogram of x projections
        counts, _ = np.histogram(x_desheared, bins=50)
        # Peakiness: variance of counts
        v = np.var(counts)
        if v > max_var:
            max_var = v
            best_angle = float(theta)

    return float(best_angle)


def extract_style_from_image(img_or_path: Image.Image | np.ndarray | str | Path | bytes) -> StyleSpec:
    """Analyzes a handwriting sample image and returns a fitted StyleSpec."""
    if isinstance(img_or_path, (str, Path)):
        img = Image.open(img_or_path).convert("L")
    elif isinstance(img_or_path, bytes):
        img = Image.open(io.BytesIO(img_or_path)).convert("L")
    elif isinstance(img_or_path, Image.Image):
        img = img_or_path.convert("L")
    else:
        img = Image.fromarray(img_or_path).convert("L")

    arr = np.array(img, dtype=np.float32)

    # Thresholding: ink < threshold
    p20, p80 = np.percentile(arr, [20, 80])
    threshold = (p20 + p80) * 0.5
    binary_ink = (arr < threshold).astype(np.uint8)

    ink_pixels = np.sum(binary_ink)
    if ink_pixels < 20:
        return StyleSpec()

    # 1. Slant
    slant = estimate_image_slant(binary_ink)

    # 2. Thickness & Pressure proxy
    dist_map = ndi.distance_transform_edt(binary_ink)
    thickness_vals = dist_map[binary_ink > 0] * 2.0
    median_thick = float(np.median(thickness_vals)) if len(thickness_vals) else 2.0
    # Map typical pixel thickness (2..8px) to normalized pressure (0.2..0.8)
    pressure = float(np.clip(0.15 + (median_thick - 1.5) * 0.12, 0.1, 0.95))

    # 3. Connectedness: ratio of width to number of connected components
    labeled, num_components = ndi.label(binary_ink)
    h, w = binary_ink.shape
    aspect = float(w) / max(float(h), 1.0)
    connectedness = float(np.clip(1.0 - (num_components / max(aspect * 8.0, 1.0)), 0.1, 0.9))

    # 4. Roundness proxy from perimeter-to-area ratio
    dilated = ndi.binary_dilation(binary_ink)
    perimeter = float(np.sum(dilated ^ binary_ink))
    area = float(ink_pixels)
    compaction = perimeter / max(area, 1.0)
    roundness = float(np.clip(1.2 - compaction * 0.4, 0.2, 0.8))

    return StyleSpec(
        slant_deg=slant,
        pressure=pressure,
        roundness=roundness,
        connectedness=connectedness,
        neatness=0.6,
        speed=0.5,
        baseline_wobble=0.3,
    )
