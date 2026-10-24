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
    # The two instances of 'VibeFont' must have strong organic variation
    assert dist >= 0.08, f"Expected strong intra-document variation (>= 0.08), got {dist:.4f}"


def test_hangul_same_syllable_strong_allograph_and_structural_diversity():
    """Repeated Hangul syllables must show strong structural/allographic diversity, not tiny jitter."""
    from vibefont.generate.hangul import hangul_char_strokes

    rng = np.random.default_rng(2026)
    samples = []
    stroke_counts = set()
    for _ in range(12):
        st = hangul_char_strokes("하", rng=rng, connectedness=0.55, neatness=0.65)
        stroke_counts.add(len(st))
        pts = np.concatenate(st, axis=0)
        pts = pts - pts.min(axis=0)
        samples.append(pts)

    # Must produce multiple distinct stroke-count allographs (e.g., ㅎ hat variations / soft links)
    assert len(stroke_counts) >= 2, f"Expected multiple allograph stroke counts for '하', got {stroke_counts}"

    # Pairwise DTW distance across normalized instances of the exact same character must be large
    dists = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            dists.append(dtw_distance(samples[i], samples[j]))
    mean_dist = float(np.mean(dists))
    assert mean_dist >= 0.11, f"Expected mean pairwise DTW >= 0.11 for repeated Hangul char, got {mean_dist:.4f}"


def test_writer_archetypes_produce_distinct_structural_proportions():
    """Different writer seeds must yield distinct WriterProfile archetypes and syllable structures."""
    from vibefont.generate.hangul import WriterProfile, hangul_char_strokes, sample_writer_profile

    profiles: list[WriterProfile] = []
    aspect_ratios = []
    for seed in range(10, 24):
        rng = np.random.default_rng(seed)
        prof = sample_writer_profile(rng)
        profiles.append(prof)
        st = hangul_char_strokes("한", rng=np.random.default_rng(seed + 100), connectedness=0.55, neatness=0.70, profile=prof)
        pts = np.concatenate(st, axis=0)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        aspect_ratios.append(w / max(h, 1e-3))

    archetypes = {p.archetype for p in profiles}
    assert len(archetypes) >= 3, f"Expected at least 3 distinct writer archetypes, got {archetypes}"
    spread = (max(aspect_ratios) - min(aspect_ratios)) / np.mean(aspect_ratios)
    assert spread >= 0.25, f"Expected single-syllable aspect ratio spread >= 25% across writer profiles, got {spread * 100:.1f}%"


def test_hybrid_model_generator_strong_modulation():
    """ModelGenerator must apply strong hybrid modulation rather than staying tightly pinned to base_ink."""
    from pathlib import Path
    from vibefont.document.drift import DocumentDrift
    from vibefont.generate.base import LineRequest
    from vibefont.generate.model import ModelGenerator

    ckpt = Path("checkpoints/bilingual_5k_core")
    if not ckpt.exists():
        pytest.skip("Checkpoint not present")

    mgen = ModelGenerator(ckpt, device="cpu")
    spec = VibeSpec(seed=77)
    state = DocumentDrift(spec.style, np.random.default_rng(77), 5).next_line()

    req_base = LineRequest(text="같은 글자라도 매번 새롭게 써지는 손글씨", style=spec.style, state=state, rng=np.random.default_rng(77))
    base_ink = mgen._scaffold.generate_line(req_base)

    req_mod = LineRequest(text="같은 글자라도 매번 새롭게 써지는 손글씨", style=spec.style, state=state, rng=np.random.default_rng(77))
    mod_ink = mgen.generate_line(req_mod)

    n = min(len(base_ink), len(mod_ink))
    mean_disp = float(np.mean(np.linalg.norm(mod_ink.xy[:n] - base_ink.xy[:n], axis=1)))
    assert mean_disp >= 0.055, f"Expected strong hybrid neural modulation (>= 0.055), got {mean_disp:.4f}"

