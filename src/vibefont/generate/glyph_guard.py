"""Glyph guard and text normalization utilities.

Ensures that:
1. Common typographic Unicode characters (curly quotes, em-dashes, ellipses, CJK brackets)
   are mapped to canonical stroke representations instead of being silently dropped.
2. Truly unsupported characters (e.g. emojis or unknown scripts) raise MissingGlyphError
   immediately rather than failing silently.
"""

from __future__ import annotations

import string
from .hangul import is_hangul


class MissingGlyphError(ValueError):
    """Raised when text contains an unsupported character that cannot be rendered."""

    def __init__(self, char: str, text: str = ""):
        self.char = char
        self.text = text
        msg = f"Unsupported glyph {char!r} (U+{ord(char):04X}) cannot be rendered by VibeFont."
        if text:
            msg += f" Text: {text!r}"
        super().__init__(msg)


# Canonical typographic replacements for Hershey / Latin stroke renderers
TYPOGRAPHIC_REPLACEMENTS: dict[str, str] = {
    # Curly double quotes
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    # Curly single quotes & apostrophes
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    # Dashes & hyphens
    "—": "-",
    "–": "-",
    "―": "-",
    "‒": "-",
    "−": "-",
    # Ellipses & dots
    "…": "...",
    "·": ".",
    "•": ".",
    "‧": ".",
    # CJK brackets & quotes
    "「": "[",
    "」": "]",
    "『": "[",
    "』": "]",
    "〈": "<",
    "〉": ">",
    "《": "<",
    "》": ">",
    "【": "[",
    "】": "]",
    "〔": "[",
    "〕": "]",
    # Whitespace & zero-width
    "\u00a0": " ",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "～": "~",
    "∼": "~",
}

# Supported ASCII set for procedural & Latin fonts
_SUPPORTED_ASCII = set(string.printable)


def normalize_glyphs(text: str) -> str:
    """Replaces smart punctuation and typographic variants with canonical renderable characters."""
    out = []
    for ch in text:
        out.append(TYPOGRAPHIC_REPLACEMENTS.get(ch, ch))
    return "".join(out)


def check_glyph_support(ch: str, allow_hangul: bool = True) -> None:
    """Verifies that a single character is supported. Raises MissingGlyphError if not."""
    if ch.isspace():
        return
    if allow_hangul and is_hangul(ch):
        return
    # Check if supported ASCII
    if ch in _SUPPORTED_ASCII:
        return
    # Check if replaceable
    if ch in TYPOGRAPHIC_REPLACEMENTS:
        return
    raise MissingGlyphError(ch)


def validate_text_glyphs(text: str, allow_hangul: bool = True) -> str:
    """Validates all characters in text, normalizes typographic variants, and returns clean text.

    Raises MissingGlyphError if any unsupported character is found.
    """
    for ch in text:
        check_glyph_support(ch, allow_hangul=allow_hangul)
    return normalize_glyphs(text)
