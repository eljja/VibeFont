"""Renders 600-dpi physical pen dynamics comparison across all 7 pen models:
  1. Fountain Pen (Liquid dye shading, pause dwell pooling, anisotropic fibre feathering, wet overlap bleed)
  2. Gel Pen (Smooth water-based liquid gel, pause pooling, subtle rim halo, wet bleed)
  3. Ballpoint Pen (Viscous oil paste socket accumulation goop at corners, rotational ball-period skips)
  4. Pencil (Graphite lead catching paper tooth micro-peaks, porous skip in tooth valleys)
  5. Marker (Chisel nib angle variation, wet bleed)
  6. Brush (Dynamic calligraphy pressure width, hair striations, wet bleed)
  7. Broad Nib (Chisel calligraphy angle shading, wet bleed)

Outputs:
  outputs/physics/{pen}.png (individual 600-dpi crops)
  outputs/physics/sheet.png (2x4 contact comparison sheet)
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vibefont.document.pipeline import write_document
from vibefont.generate import get_generator
from vibefont.render.raster import render_page
from vibefont.vibe.interpreter import interpret
from vibefont.vibe.spec import PenSpec, VibeSpec


PENS = [
    ("fountain", "Fountain Pen (만년필)", "아이보리 편지지에 블루블랙 만년필로 유려한 필기체로 쓴 글", (24, 43, 73)),
    ("gel", "Gel Pen (중성 젤펜)", "미색 노트에 검정 젤펜으로 또박또박 쓴 글", (28, 30, 34)),
    ("ballpoint", "Ballpoint Pen (유성 볼펜)", "줄노트에 파란 볼펜으로 빠르게 이어 쓴 글", (22, 60, 140)),
    ("pencil", "Pencil (연필)", "크라프트 종이에 4B 연필로 사각사각 쓴 글", (45, 45, 48)),
    ("marker", "Marker (마커)", "백색 종이에 빨간 마커펜으로 굵게 쓴 글", (180, 32, 40)),
    ("brush", "Brush (모필 붓)", "화선지에 먹색 붓으로 힘차고 유려하게 쓴 글", (25, 25, 27)),
    ("broad_nib", "Broad Nib (평촉 캘리펜)", "아이보리 종이에 버건디 평촉 캘리펜으로 우아하게 쓴 글", (110, 24, 45)),
]

SAMPLE_TEXT = "Quickly feathered ink 잉크 번짐"


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in [
        "malgun.ttf",
        "malgunbd.ttf",
        "arial.ttf",
        "seguiemj.ttf",
        "DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    out_dir = Path("outputs/physics")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = get_generator("model", "checkpoints/bilingual_5k_core")
    tiles = []

    print(f"Rendering 600-dpi physical pen comparisons to {out_dir}...", flush=True)
    t0 = time.perf_counter()

    for pen_type, label, vibe_prompt, ink_rgb in PENS:
        spec: VibeSpec = interpret(vibe_prompt)
        spec.pen = PenSpec(type=pen_type, width_mm=0.55 if pen_type not in ("brush", "broad_nib", "marker") else 0.75)
        spec.ink.color = tuple(float(c) / 255.0 for c in ink_rgb)
        spec.layout.page = "Card"
        spec.layout.width_mm = 90.0
        spec.layout.height_mm = 30.0
        spec.layout.margin_mm = (10.0, 8.0, 10.0, 8.0)
        spec.style.size_mm = 3.6
        spec.style.connectedness = 0.85
        spec.style.neatness = 0.88

        rng = np.random.default_rng(42)
        pages = write_document(SAMPLE_TEXT, spec, gen, rng)
        page0 = pages[0]

        img_600 = render_page(page0, spec, dpi=600, rng=np.random.default_rng(1042))
        single_path = out_dir / f"{pen_type}.png"
        img_600.save(single_path, dpi=(600, 600))
        print(f"  [+] Saved {single_path.name} ({img_600.width}x{img_600.height})", flush=True)

        tiles.append((pen_type, label, img_600))

    # Compose 2x4 Contact Sheet
    cols = 2
    rows = 4  # 7 pens + 1 summary tile
    card_w = 1000
    card_h = 360
    pad = 24
    header_h = 100

    sheet_w = pad * (cols + 1) + card_w * cols
    sheet_h = header_h + pad * (rows + 1) + card_h * rows
    sheet = Image.new("RGB", (sheet_w, sheet_h), (22, 24, 30))
    draw = ImageDraw.Draw(sheet)

    f_title = _load_font(32)
    f_sub = _load_font(18)
    f_card_label = _load_font(20)
    f_badge = _load_font(14)

    draw.text((pad + 10, 20), "VibeFont Analog Ink & Paper Physics Showcase (600 DPI)", fill=(245, 248, 255), font=f_title)
    draw.text((pad + 10, 62), "Dwell-time pooling, anisotropic fibre feathering, ballpoint socket goop & skip, wet-on-wet bleed", fill=(160, 172, 195), font=f_sub)

    for i, (pen_type, label, img) in enumerate(tiles):
        r = i // cols
        c = i % cols
        x0 = pad + c * (card_w + pad)
        y0 = header_h + pad + r * (card_h + pad)

        # Draw card container
        draw.rounded_rectangle(
            [x0, y0, x0 + card_w, y0 + card_h],
            radius=12,
            fill=(32, 35, 44),
            outline=(60, 66, 82),
            width=2,
        )

        # Header of card
        draw.text((x0 + 16, y0 + 12), label, fill=(235, 240, 250), font=f_card_label)

        # Thumbnail rendering inside card
        thumb_area_w = card_w - 32
        thumb_area_h = card_h - 60
        iw, ih = img.size
        scale = min(thumb_area_w / iw, thumb_area_h / ih)
        tw, th = int(iw * scale), int(ih * scale)
        resized = img.resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(resized, (x0 + 16 + (thumb_area_w - tw) // 2, y0 + 46 + (thumb_area_h - th) // 2))

    # Final summary card in the 8th slot
    x0 = pad + 1 * (card_w + pad)
    y0 = header_h + pad + 3 * (card_h + pad)
    draw.rounded_rectangle([x0, y0, x0 + card_w, y0 + card_h], radius=12, fill=(28, 38, 55), outline=(50, 80, 130), width=2)
    draw.text((x0 + 20, y0 + 20), "Physics Engine Highlights", fill=(140, 190, 255), font=f_card_label)
    summary_bullets = [
        "1. Dwell-Time Ink Pooling: Pause dwell pools liquid ink into paper",
        "2. Anisotropic Fibre Feathering: Lateral spread directed along cellulose fibres",
        "3. Ballpoint Socket Goop: High-shear turns deposit accumulated paste blobs",
        "4. Rotational Ball-Period Skips: Micro-gaps matching rolling ball circumference",
        "5. Wet-on-Wet Overlap Bleed: Crossing fresh wet lines bleeds capillaries",
    ]
    for b_idx, bullet in enumerate(summary_bullets):
        draw.text((x0 + 24, y0 + 64 + b_idx * 34), bullet, fill=(210, 225, 250), font=f_badge)

    sheet_path = out_dir / "sheet.png"
    sheet.save(sheet_path, dpi=(150, 150))
    print(f"Generated comparison sheet: {sheet_path} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
