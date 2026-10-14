"""Synthetic dataset from the placeholder generator.

Only for smoke-testing the training loop end to end before real data exists
(the model should learn to imitate Hershey-like strokes). Never mix into real training.

    python -m vibefont.data.synthetic --out data/processed/synthetic --docs 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..dataset import DatasetWriter
from ..document.drift import DocumentDrift
from ..generate.base import LineRequest
from ..generate.procedural import ProceduralGenerator
from ..vibe.spec import StyleSpec

WORDS = ("the of and to in is you that it he was for on are as with his they at be this have from or "
         "one had by word but not what all were we when your can said there use an each which she do how "
         "their if will up other about out many then them these so some her would make like him into time "
         "has look two more write go see number no way could people my than first water been call who oil "
         "its now find long down day did get come made may part over new sound take only little work know "
         "place year live me back give most very after thing our just name good sentence man think say "
         "great where help through much before line right too mean old any same tell boy follow came want "
         "show also around form three small set put end does another well large must big even such because "
         "turn here why ask went men read need land different home us move try kind hand picture again "
         "change off play spell air away animal house point page letter mother answer found study still "
         "learn should America world quick brown fox jumps lazy dog Dear friend Monday London").split()


def random_style(rng: np.random.Generator) -> StyleSpec:
    return StyleSpec(slant_deg=float(rng.uniform(-8, 25)), width_scale=float(rng.uniform(0.8, 1.25)),
                     roundness=float(rng.random()), connectedness=float(rng.random()),
                     neatness=float(rng.uniform(0.2, 1.0)), speed=float(rng.random()),
                     pressure=float(rng.random()), baseline_wobble=float(rng.random()),
                     consistency=float(rng.random()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/synthetic")
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--lines", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    gen = ProceduralGenerator()
    w = DatasetWriter(Path(a.out))
    for d in range(a.docs):
        style = random_style(rng)
        drift = DocumentDrift(style, rng, a.lines)
        for k in range(a.lines):
            text = " ".join(rng.choice(WORDS, int(rng.integers(2, 7))))
            ink = gen.generate_line(LineRequest(text, style, drift.next_line(), rng))
            ink.points[:, 0] -= ink.bbox()[0]
            w.add(ink, f"syn{d}", f"syn{d}", k, "synthetic")
    w.close()
    print("lines:", len(w.rows))


if __name__ == "__main__":
    main()
