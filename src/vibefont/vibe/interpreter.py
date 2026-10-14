"""Prompt -> VibeSpec.

Tiers (all produce the same VibeSpec contract):
  0. ``interpret``           offline keyword lexicon (always available, used as fallback)
  1. ``vibe.llm.interpret_llm`` any OpenAI-compatible endpoint returns a partial VibeSpec JSON
  2. (planned) learned text-embedding -> style-latent map trained on writer prototypes
"""

from __future__ import annotations

import re

from .lexicon import COLORS, DIMINISHERS, INTENSIFIERS, NEGATIONS, RULES
from .spec import VibeSpec

_CLAUSE_SPLIT = re.compile(r"[,.;\n]|\band\b|\bbut\b|그리고|하지만|그런데|면서")


def _intensity(clause: str) -> float:
    f = 1.0
    for w, m in INTENSIFIERS.items():
        if w in clause:
            f = max(f, m)
    for w, m in DIMINISHERS.items():
        if w in clause:
            f = min(f, m)
    if any(n in clause for n in NEGATIONS):
        f = -0.7 * f
    return f


def interpret(prompt: str, base: VibeSpec | None = None) -> VibeSpec:
    spec = base or VibeSpec()
    spec.prompt = prompt
    text = prompt.lower()
    for clause in _CLAUSE_SPLIT.split(text):
        clause = clause.strip()
        if not clause:
            continue
        f = _intensity(clause)
        for rule in RULES:
            if not any(k in clause for k in rule.keywords):
                continue
            spec.matched.append(f"{rule.name}x{f:+.1f}")
            for k, dv in rule.delta.items():
                setattr(spec.style, k, getattr(spec.style, k) + f * dv)
            if f > 0:
                spec.update(rule.set)
        for keys, rgb in COLORS.items():
            if any(k in clause for k in keys) and any(w in clause for w in ("잉크", "펜", "색", "ink", "pen", "colou")):
                spec.ink.color = rgb
                spec.matched.append(f"color:{keys[-1]}")
    m = re.search(r"seed\s*[:=]?\s*(\d+)", text)
    if m:
        spec.seed = int(m.group(1))
    spec.clamp()
    return spec
