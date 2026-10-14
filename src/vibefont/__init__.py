"""VibeFont: text + vibe prompt -> naturally varying handwriting.

Pipeline:
    prompt --(vibe.interpreter)--> VibeSpec
    text + VibeSpec --(document.pipeline)--> per-line requests with drifting state
    request --(generate.*)--> Ink (normalised units)
    Ink --(render.*)--> PNG / PDF / SVG
"""

from .ink import Ink

__all__ = ["Ink"]
__version__ = "0.1.0"
