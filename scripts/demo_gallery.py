"""Render the same text with several vibe prompts side by side (visual regression check).

    python scripts/demo_gallery.py  -> scratch/gallery_*.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vibefont.document.pipeline import write_document
from vibefont.generate import get_generator
from vibefont.render.raster import render_page
from vibefont.vibe.interpreter import interpret

TEXT = ("Dear friend, the same words never look the same twice.\n"
        "the same words, the same words, the same words.")
PROMPTS = {
    "fountain_elegant": "elegant fountain pen letter, blue ink",
    "ballpoint_hurried": "급하게 쓴 볼펜 메모, 줄노트",
    "pencil_child": "연필로 쓴 아이 글씨, 크게",
    "calligraphy": "calligraphy broad nib, very neat, black ink",
    "marker_bold": "bold marker, red ink, grid paper",
    "doctor": "의사 처방전처럼 엄청 휘갈겨서",
}


def main() -> None:
    out = Path("scratch")
    out.mkdir(exist_ok=True)
    gen = get_generator("procedural")
    for name, prompt in PROMPTS.items():
        spec = interpret(prompt)
        spec.seed = 7
        spec.layout.page = "A5"
        pages = write_document(TEXT, spec, gen, np.random.default_rng(spec.seed))
        img = render_page(pages[0], spec, dpi=200)
        img = img.crop((0, 0, img.width, img.height // 2))
        img.save(out / f"gallery_{name}.png")
        print(name, spec.matched)


if __name__ == "__main__":
    main()
