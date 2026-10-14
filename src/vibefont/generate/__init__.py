"""Stroke generators. ``get_generator`` picks the best available implementation."""

from __future__ import annotations


def get_generator(name: str = "auto", model_path: str | None = None):
    if name in ("model", "auto") and model_path:
        from .model import ModelGenerator  # noqa: WPS433 (lazy: needs onnxruntime or torch)

        return ModelGenerator(model_path)
    if name in ("procedural", "auto"):
        from .procedural import ProceduralGenerator

        return ProceduralGenerator()
    raise ValueError(f"unknown generator {name!r} (model requires --model PATH)")
