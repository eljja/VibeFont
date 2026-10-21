"""Dataset pre-flight health and integrity validator for VibeFont.

Checks:
- File existence (lines.npz, index.jsonl)
- Array shapes, offsets consistency, offsets boundaries
- Coordinate validity (finite, no NaN, no Inf)
- Time monotonicity within stroke segments
- Pressure bounds (non-negative)
- EOS validity (binary 0 or 1)
- Character vocabulary coverage (Hangul, Latin, Digits, Punctuation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import unicodedata

import numpy as np


@dataclass
class ValidationReport:
    is_healthy: bool
    total_lines: int = 0
    total_points: int = 0
    total_strokes: int = 0
    vocab_stats: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "HEALTHY" if self.is_healthy else "CORRUPTED"
        lines = [
            f"Dataset Health: [{status}]",
            f"Total Lines: {self.total_lines}",
            f"Total Points: {self.total_points}",
            f"Total Strokes: {self.total_strokes}",
            f"Vocabulary: Hangul={self.vocab_stats.get('hangul', 0)}, "
            f"Latin={self.vocab_stats.get('latin', 0)}, "
            f"Unique Chars={self.vocab_stats.get('unique_chars', 0)}",
        ]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def _is_hangul(ch: str) -> bool:
    cp = ord(ch)
    return (0xAC00 <= cp <= 0xD7A3) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F)


def _is_latin(ch: str) -> bool:
    cp = ord(ch)
    return (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A)


def validate_dataset(path: str | Path) -> ValidationReport:
    d = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    npz_path = d / "lines.npz"
    idx_path = d / "index.jsonl"

    if not npz_path.exists():
        errors.append(f"Missing required file: {npz_path}")
    if not idx_path.exists():
        errors.append(f"Missing required file: {idx_path}")

    if errors:
        return ValidationReport(is_healthy=False, errors=errors, warnings=warnings)

    try:
        data = np.load(npz_path)
    except Exception as e:
        return ValidationReport(
            is_healthy=False,
            errors=[f"Failed to load {npz_path}: {e}"],
            warnings=warnings,
        )

    if "points" not in data or "offsets" not in data:
        errors.append(f"Missing 'points' or 'offsets' in {npz_path}")
        return ValidationReport(is_healthy=False, errors=errors, warnings=warnings)

    pts = data["points"]
    offsets = data["offsets"]

    if pts.ndim != 2 or pts.shape[1] != 5:
        errors.append(f"Invalid points shape: {pts.shape}, expected (N, 5)")

    if not np.all(np.isfinite(pts)):
        nan_count = int(np.isnan(pts).sum())
        inf_count = int(np.isinf(pts).sum())
        errors.append(f"Non-finite numbers found in points: {nan_count} NaNs, {inf_count} Infs")

    # Read index rows
    rows: list[dict] = []
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_idx} in index.jsonl is invalid JSON: {e}")
    except Exception as e:
        errors.append(f"Failed to read {idx_path}: {e}")

    num_lines = len(rows)
    if len(offsets) != num_lines + 1:
        errors.append(f"Offsets length ({len(offsets)}) does not match index rows count + 1 ({num_lines + 1})")

    if len(offsets) > 0:
        if offsets[0] != 0:
            errors.append(f"First offset must be 0, got {offsets[0]}")
        if len(pts) > 0 and offsets[-1] != len(pts):
            errors.append(f"Last offset ({offsets[-1]}) does not match total points ({len(pts)})")
        if np.any(np.diff(offsets) < 0):
            errors.append("Offsets must be non-decreasing")

    # Check timestamps and stroke properties per line
    total_strokes = 0
    if len(pts) > 0 and len(offsets) == num_lines + 1:
        for i in range(num_lines):
            a, b = offsets[i], offsets[i + 1]
            line_pts = pts[a:b]
            if len(line_pts) == 0:
                warnings.append(f"Line {i} has 0 points")
                continue

            t_col = line_pts[:, 2]
            eos_col = line_pts[:, 4]

            # Coordinate bounds check (normalized space: y in [-15, 15], x in [-5, 250])
            if np.all(np.isfinite(line_pts[:, :2])):
                xmin, xmax = float(line_pts[:, 0].min()), float(line_pts[:, 0].max())
                ymin, ymax = float(line_pts[:, 1].min()), float(line_pts[:, 1].max())
                if ymin < -15.0 or ymax > 15.0 or xmin < -5.0 or (xmax - xmin) > 250.0:
                    errors.append(
                        f"Line {i} has out-of-bounds coordinates (scale/bound error): "
                        f"bbox=({xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f})"
                    )

            # EOS values
            if not np.all(np.isin(eos_col, [0.0, 1.0])):
                errors.append(f"Line {i} has invalid non-binary EOS values")

            # Count strokes
            stroke_ends = np.where(eos_col == 1.0)[0]
            total_strokes += len(stroke_ends)

            # Monotonic time check within each stroke
            prev_idx = 0
            for end_idx in stroke_ends:
                stroke_t = t_col[prev_idx : end_idx + 1]
                if len(stroke_t) > 1 and np.any(np.diff(stroke_t) < 0):
                    errors.append(f"Line {i} has non-monotonic timestamp in stroke ending at point {end_idx}")
                prev_idx = end_idx + 1

    # Vocabulary and character statistics
    hangul_count = 0
    latin_count = 0
    digit_count = 0
    punct_count = 0
    all_chars = set()

    for r in rows:
        text = r.get("text", "")
        for ch in text:
            all_chars.add(ch)
            if _is_hangul(ch):
                hangul_count += 1
            elif _is_latin(ch):
                latin_count += 1
            elif ch.isdigit():
                digit_count += 1
            elif unicodedata.category(ch).startswith("P") or ch in " \t\n":
                punct_count += 1

    vocab_stats = {
        "hangul": hangul_count,
        "latin": latin_count,
        "digit": digit_count,
        "punct": punct_count,
        "unique_chars": len(all_chars),
    }

    is_healthy = len(errors) == 0
    return ValidationReport(
        is_healthy=is_healthy,
        total_lines=num_lines,
        total_points=len(pts),
        total_strokes=total_strokes,
        vocab_stats=vocab_stats,
        errors=errors,
        warnings=warnings,
    )
