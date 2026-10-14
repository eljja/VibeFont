import numpy as np
import pytest

from vibefont.ink import Ink, NCOLS, EOS, T, X, Y


def test_ink_basic():
    pts = np.array([
        [0.0, 0.0, 0.00, 0.5, 0.0],
        [1.0, 2.0, 0.01, 0.6, 0.0],
        [2.0, 1.0, 0.02, 0.5, 1.0],  # end of stroke 1
        [5.0, 5.0, 0.10, 0.5, 0.0],
        [6.0, 7.0, 0.11, 0.7, 1.0],  # end of stroke 2
    ], dtype=np.float32)
    ink = Ink(pts, text="hi")

    assert len(ink) == 5
    assert len(ink.stroke_slices()) == 2
    assert ink.stroke_slices()[0] == slice(0, 3)
    assert ink.stroke_slices()[1] == slice(3, 5)
    assert ink.has_pressure

    bbox = ink.bbox()
    assert bbox == (0.0, 0.0, 6.0, 7.0)


def test_ink_transforms():
    pts = np.zeros((3, NCOLS), dtype=np.float32)
    pts[:, :2] = [[0, 0], [1, 1], [2, 0]]
    ink = Ink(pts)

    # Translate
    t = ink.translate(5.0, -3.0)
    assert np.allclose(t.xy[0], [5.0, -3.0])

    # Scale
    s = ink.scale(2.0, 3.0)
    assert np.allclose(s.xy[1], [2.0, 3.0])


def test_ink_serialization(tmp_path):
    pts = np.zeros((4, NCOLS), dtype=np.float32)
    pts[:, :2] = [[0, 0], [1, 1], [2, 2], [3, 3]]
    pts[-1, EOS] = 1.0
    ink = Ink(pts, text="test")

    p = tmp_path / "test.json"
    ink.save(p)
    loaded = Ink.load(p)

    assert loaded.text == "test"
    assert len(loaded) == 4
    assert np.allclose(ink.xy, loaded.xy, atol=1e-3)
