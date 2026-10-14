"""Tier-1 prompt interpretation through any OpenAI-compatible chat endpoint (optional).

Configure with env vars:
    VIBEFONT_LLM_URL    e.g. https://api.openai.com/v1/chat/completions  or a local server
    VIBEFONT_LLM_MODEL  model name
    VIBEFONT_LLM_KEY    API key (optional for local servers)

Falls back to the offline lexicon on any error, so the program never depends on network.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from .interpreter import interpret
from .spec import PAPER_TYPES, PEN_TYPES, VibeSpec

SYSTEM = f"""You translate a description of handwriting ("vibe") into JSON.
Return ONLY a JSON object with any subset of these fields:
{{"style": {{"slant_deg": -25..40, "size_mm": 1.2..15, "width_scale": 0.6..1.8,
  "letter_spacing": 0.6..2, "word_spacing": 0.5..3, "roundness": 0..1, "connectedness": 0..1,
  "neatness": 0..1, "speed": 0..1, "pressure": 0..1, "baseline_wobble": 0..1,
  "consistency": 0..1, "fatigue": 0..1}},
 "pen": {{"type": one of {list(PEN_TYPES)}, "width_mm": 0.1..5, "nib_angle_deg": 0..90}},
 "ink": {{"color": [r, g, b] in 0..1}},
 "paper": {{"type": one of {list(PAPER_TYPES)}, "line_spacing_mm": number}}}}
Defaults: slant 6, size 3, neatness 0.6, speed 0.5, connectedness 0.5. Only include fields the
description implies."""


def interpret_llm(prompt: str, base: VibeSpec | None = None, timeout: float = 20.0) -> VibeSpec:
    url = os.environ.get("VIBEFONT_LLM_URL")
    model = os.environ.get("VIBEFONT_LLM_MODEL", "")
    if not url:
        return interpret(prompt, base)
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VIBEFONT_LLM_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VIBEFONT_LLM_KEY"]
    try:
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.S)
        spec = base or VibeSpec()
        spec.update(json.loads(m.group(0)) if m else {})
        spec.prompt = prompt
        spec.matched.append("llm")
        return spec
    except Exception as e:  # network, parsing, schema - never fatal
        spec = interpret(prompt, base)
        spec.matched.append(f"llm_failed:{type(e).__name__}")
        return spec
