import numpy as np
import pytest

from vibefont.document.pipeline import write_document
from vibefont.generate.procedural import ProceduralGenerator
from vibefont.htr.metrics import dtw_distance
from vibefont.vibe.spec import VibeSpec


def test_repeated_generation_diversity_english():
    """Generating the same English text without fixed seed must yield non-identical ink."""
    gen = ProceduralGenerator()
    spec1 = VibeSpec()
    spec2 = VibeSpec()

    # When seed is None, distinct random seeds must be used
    pages1 = write_document("The quick brown fox jumps over the lazy dog.", spec1, gen)
    pages2 = write_document("The quick brown fox jumps over the lazy dog.", spec2, gen)

    ink1 = pages1[0].lines[0]
    ink2 = pages2[0].lines[0]

    # Trajectories must not be identical
    assert len(ink1) > 0 and len(ink2) > 0
    dist = dtw_distance(ink1.xy, ink2.xy)
    assert dist > 1e-4, f"Expected non-zero DTW distance between runs, got {dist}"


def test_repeated_generation_diversity_korean():
    """Generating the same Korean text without fixed seed must yield non-identical ink."""
    gen = ProceduralGenerator()
    spec1 = VibeSpec()
    spec2 = VibeSpec()

    pages1 = write_document("다람쥐 헌 쳇바퀴에 타고파", spec1, gen)
    pages2 = write_document("다람쥐 헌 쳇바퀴에 타고파", spec2, gen)

    ink1 = pages1[0].lines[0]
    ink2 = pages2[0].lines[0]

    assert len(ink1) > 0 and len(ink2) > 0
    dist = dtw_distance(ink1.xy, ink2.xy)
    assert dist > 1e-4, f"Expected non-zero DTW distance for Korean runs, got {dist}"


def test_reproducibility_with_explicit_seed():
    """When an explicit seed is provided, outputs must be 100% reproducible."""
    gen = ProceduralGenerator()
    spec1 = VibeSpec(seed=42)
    spec2 = VibeSpec(seed=42)

    pages1 = write_document("Consistent handwriting sample", spec1, gen)
    pages2 = write_document("Consistent handwriting sample", spec2, gen)

    ink1 = pages1[0].lines[0]
    ink2 = pages2[0].lines[0]

    np.testing.assert_allclose(ink1.points, ink2.points, atol=1e-5)


def test_intra_document_variation_same_word():
    """Within a single document, repeating the same word must yield distinct stroke shapes."""
    gen = ProceduralGenerator()
    spec = VibeSpec(seed=123)

    # Line 1 and Line 2 have the same word
    text = "VibeFont\nVibeFont"
    pages = write_document(text, spec, gen)

    assert len(pages[0].lines) == 2
    ink_line1 = pages[0].lines[0]
    ink_line2 = pages[0].lines[1]

    # Normalize both lines by moving minimum (x, y) to (0, 0)
    xy1 = ink_line1.xy - ink_line1.xy.min(axis=0)
    xy2 = ink_line2.xy - ink_line2.xy.min(axis=0)

    dist = dtw_distance(xy1, xy2)
    # The two instances of 'VibeFont' must have slight organic variation
    assert dist > 1e-4, f"Expected intra-document variation between identical words, got {dist}"
