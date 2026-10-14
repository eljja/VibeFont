import numpy as np
import pytest

from vibefont.ink import Ink, NCOLS, EOS
from vibefont.tokenizer import Bins, StrokeTokenizer, roundtrip_error


def test_bins_mulaw():
    b = Bins.mulaw(63, max_abs=10.0)
    assert b.n == 63
    # 0 should be exact center
    mid = b.centers[len(b.centers) // 2]
    assert abs(mid) < 1e-9

    idx = b.quantize(0.0)
    val = b.dequantize(idx)
    assert abs(val) < 1e-9


def test_tokenizer_roundtrip():
    # Generate a synthetic curved stroke
    t = np.linspace(0, 2 * np.pi, 100)
    x = t
    y = np.sin(t)
    pts = np.zeros((100, NCOLS), dtype=np.float32)
    pts[:, 0] = x
    pts[:, 1] = y
    pts[:, 2] = t / 10.0
    pts[-1, EOS] = 1.0

    ink = Ink(pts, text="curve")
    tok = StrokeTokenizer()

    err = roundtrip_error(tok, ink)
    # Target in review: mean error < 0.5px (or < 0.05 x-height)
    assert err["mean"] < 0.05
    assert err["max"] < 0.15
