"""Generates high-resolution physical pen-paper micro-physics closeups
and allograph persona comparison boards.

Showcases:
1. Micro-scale physical contact dynamics:
   - Pencil: particulate graphite adherence to microscopic paper tooth peaks
   - Fountain pen: flex-nib tine bifurcation tracks (slits) under heavy pressure
   - Ballpoint: viscous paste rim ridges along ball socket margin
2. Allograph skeletal diversity:
   - 고시체 (Gosi cursive shorthand: single-stroke Z-ㄹ, single-loop ㅁ, rapid forward slant)
   - 정갈한 정자체 (Classic Jeongja: orthogonal 90-degree corner stops, formal separate strokes)
   - 둥근 다이어리체 (Rounded Diary: exaggerated circular loops for ㅇ/ㅎ, playful bounce)
   - 명필 캘리그라피 (Master Calligraphy: dynamic stem/particle hierarchy & ballistic swashes)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vibefont.document.layout import Page, page_size
from vibefont.document.pipeline import write_document
from vibefont.generate.calligraphy import compose_calligraphy_page
from vibefont.render.raster import render_page
from vibefont.vibe.interpreter import interpret
from vibefont.vibe.spec import VibeSpec, PenSpec, InkSpec, PaperSpec, StyleSpec, LayoutSpec


def generate_microphysics_comparison(out_dir: Path, brain_dir: Path) -> Path:
    """Renders high-resolution samples of pencil, fountain pen, and ballpoint,

    and extracts 4x zoomed macro crops of the stroke boundary and texture.
    """
    text_sample = "물리적 펜촉과 종이 결의 상호작용\nPhysical Pen-Paper Contact Dynamics"

    # 1. Real Pencil on textured kraft/ivory paper
    spec_pencil = interpret("크라프트 종이에 연필로 쓴 글")
    spec_pencil.style.size_mm = 5.2
    spec_pencil.pen.width_mm = 0.55
    spec_pencil.layout.page = "Card"
    rng_p = np.random.default_rng(101)
    page_pencil = compose_calligraphy_page(text_sample, spec_pencil, rng=rng_p)
    img_pencil = render_page(page_pencil, spec_pencil, dpi=300, rng=rng_p)

    # 2. Real Flex Fountain Pen on Ivory paper
    spec_fountain = interpret("아이보리 종이에 남색 만년필로 강한 필압으로 쓴 글")
    spec_fountain.style.size_mm = 5.2
    spec_fountain.pen.width_mm = 0.70
    spec_fountain.pen.type = "fountain"
    spec_fountain.layout.page = "Card"
    rng_f = np.random.default_rng(202)
    page_fountain = compose_calligraphy_page(text_sample, spec_fountain, rng=rng_f)
    img_fountain = render_page(page_fountain, spec_fountain, dpi=300, rng=rng_f)

    # 3. Real Ballpoint Pen on Ruled paper
    spec_ball = interpret("줄노트에 검정 볼펜으로 쓴 글")
    spec_ball.style.size_mm = 5.2
    spec_ball.pen.width_mm = 0.50
    spec_ball.pen.type = "ballpoint"
    spec_ball.layout.page = "Card"
    rng_b = np.random.default_rng(303)
    page_ball = compose_calligraphy_page(text_sample, spec_ball, rng=rng_b)
    img_ball = render_page(page_ball, spec_ball, dpi=300, rng=rng_b)

    # Extract high-detail macro crops around dense stroke characters
    crop_w, crop_h = 360, 360
    cx, cy = img_pencil.width // 2, img_pencil.height // 2 - 30

    crop_pencil = img_pencil.crop((cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2))
    crop_fountain = img_fountain.crop((cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2))
    crop_ball = img_ball.crop((cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2))

    # Zoom in 2x with NEAREST so microscopic physical grain/tines are clearly visible without blur
    zoom_w, zoom_h = 440, 440
    crop_pencil_zoom = crop_pencil.resize((zoom_w, zoom_h), Image.Resampling.NEAREST)
    crop_fountain_zoom = crop_fountain.resize((zoom_w, zoom_h), Image.Resampling.NEAREST)
    crop_ball_zoom = crop_ball.resize((zoom_w, zoom_h), Image.Resampling.NEAREST)

    # Compose Macro Texture Board
    margin = 30
    spacing = 25
    header_h = 120
    panel_w = zoom_w
    panel_h = zoom_h + 120

    total_w = margin * 2 + 3 * panel_w + 2 * spacing
    total_h = header_h + margin * 2 + panel_h

    board = Image.new("RGB", (total_w, total_h), (245, 244, 240))
    draw = ImageDraw.Draw(board)

    # Header
    draw.rectangle([(0, 0), (total_w, header_h)], fill=(22, 28, 40))
    try:
        font_title = ImageFont.truetype("malgunbd.ttf", 32)
        font_sub = ImageFont.truetype("malgun.ttf", 18)
        font_panel_t = ImageFont.truetype("malgunbd.ttf", 20)
        font_panel_d = ImageFont.truetype("malgun.ttf", 14)
    except Exception:
        font_title = font_sub = font_panel_t = font_panel_d = ImageFont.load_default()

    draw.text((margin + 10, 24), "물리적 필기도구 마이크로 다이내믹스 검증 (Physical Pen-Paper Micro-Dynamics)", fill=(255, 255, 255), font=font_title)
    draw.text((margin + 12, 72), "단순 래스터 원형 도장이 아닌 실제 흑연 결정 고착 · 연성 닙 갈라짐 레일 · 볼펜 페이스트 융기 물리 구현", fill=(190, 205, 230), font=font_sub)

    panels = [
        (
            crop_pencil_zoom,
            "1. 연필 흑연 전단 & 종이 결(Tooth) 고착",
            "미세 종이 섬유 돌기 봉우리에만 흑연 입자가 깎여 묻어나며,\n골짜기에는 자연스러운 미세 공백(Micro-voids)이 형성됨.",
        ),
        (
            crop_fountain_zoom,
            "2. 연성 만년필 닙 갈라짐(Tine Split) & 염료 셰이딩",
            "필압 상승 시 2갈래 닙이 벌어지며 외곽에 짙은 듀얼 레일이 형성되고,\n중앙 슬릿과 획 방향에 따라 액상 염료 풀링(Pooling)이 일어남.",
        ),
        (
            crop_ball_zoom,
            "3. 유성 볼펜 점성 페이스트 림(Rim Paste)",
            "회전하는 볼 소켓 테두리를 따라 고점도 잉크 페이스트 릿지가 남고,\n내부는 잉크 점성에 의한 미세한 불균일 마찰 텍스처를 띰.",
        ),
    ]

    for i, (zoom_img, title, desc) in enumerate(panels):
        px = margin + i * (panel_w + spacing)
        py = header_h + margin

        # Container
        draw.rectangle([(px - 2, py - 2), (px + panel_w + 2, py + panel_h + 2)], fill=(220, 218, 212))
        draw.rectangle([(px, py), (px + panel_w, py + panel_h)], fill=(255, 255, 255))

        # Paste zoom image
        board.paste(zoom_img, (px, py))

        # Desc
        text_y = py + zoom_h + 14
        draw.text((px + 12, text_y), title, fill=(20, 25, 35), font=font_panel_t)
        draw.text((px + 12, text_y + 32), desc, fill=(90, 95, 105), font=font_panel_d)

    macro_path = out_dir / "gallery_pen_microphysics.png"
    board.save(macro_path, quality=95)
    if brain_dir.exists():
        board.save(brain_dir / "gallery_pen_microphysics.png")
    return macro_path


def generate_allograph_persona_comparison(out_dir: Path, brain_dir: Path) -> Path:
    """Renders identical literary text across 4 distinct handwriting personas

    and composes a side-by-side comparison board.
    """
    text = (
        "바람이 머물다 간 자리마다\n"
        "푸른 별의 기억이 고요히 피어난다"
    )

    personas = [
        {
            "id": "persona_01_gosi",
            "name": "고시체 (Gosi Cursive Shorthand)",
            "subtitle": "실제 수험생·법조인의 초고속 속기 필적",
            "desc": "단일 획 연속 Z-ㄹ, 한붓그리기 타원 ㅁ, 신속한 우상향 기울기와 유려한 어간 연결",
            "prompt": "고시체 수험생 속기 필기 남색 만년필",
            "seed": 881,
        },
        {
            "id": "persona_02_jeongja",
            "name": "정갈한 정자체 (Classic Jeongja Formal)",
            "subtitle": "교과서·궁서 기반의 반듯한 직교 정자",
            "desc": "90도 각진 모서리 일시멈춤, 4획 독립 ㅂ, 3획 ㅁ, 칼같이 단정한 직립 기하학",
            "prompt": "정갈한 정자체 교과서처럼 반듯하게 검정 볼펜",
            "seed": 882,
        },
        {
            "id": "persona_03_diary",
            "name": "둥근 다이어리체 (Bubbly Diary Cute)",
            "subtitle": "소녀 감성의 동글동글 아기자기 손글씨",
            "desc": "특대형 개방 원형 ㅇ/ㅎ, 둥근 모서리 ㅁ, 리듬감 있는 통통한 글자 비례와 줄노트 배치",
            "prompt": "둥근 다이어리체 동글동글 아기자기하게 초록 젤펜",
            "seed": 883,
        },
        {
            "id": "persona_04_calli",
            "name": "명필 캘리그라피 (Master Calligraphy)",
            "subtitle": "단어 의미를 형상화한 수묵 캘리그라피",
            "desc": "어간/조사 계층 구조, 횡세종태, 생체역학적 관성 삐침(Ballistic Flourish), 전각 낙관",
            "prompt": "명필 캘리그라피 붓펜 먹색",
            "seed": 884,
        },
    ]

    card_w, card_h = 600, 480
    rendered_cards: list[tuple[dict, Image.Image]] = []

    for item in personas:
        spec = interpret(item["prompt"])
        spec.layout.page = "Card"
        spec.style.size_mm = 4.8
        rng = np.random.default_rng(item["seed"])
        page = compose_calligraphy_page(text, spec, rng=rng)
        img = render_page(page, spec, dpi=220, rng=rng)

        # Save individual card
        card_p = out_dir / f"{item['id']}.png"
        img.save(card_p)
        if brain_dir.exists():
            img.save(brain_dir / f"{item['id']}.png")

        rendered_cards.append((item, img))

    # Compose Comparison Board (2 x 2 grid)
    header_h = 120
    margin = 30
    spacing = 25
    cols = 2
    rows = 2

    total_w = margin * 2 + cols * card_w + (cols - 1) * spacing
    total_h = header_h + margin * 2 + rows * card_h + (rows - 1) * spacing

    board = Image.new("RGB", (total_w, total_h), (246, 244, 240))
    draw = ImageDraw.Draw(board)

    # Header
    draw.rectangle([(0, 0), (total_w, header_h)], fill=(24, 30, 44))
    try:
        font_title = ImageFont.truetype("malgunbd.ttf", 32)
        font_sub = ImageFont.truetype("malgun.ttf", 18)
        font_card_t = ImageFont.truetype("malgunbd.ttf", 19)
        font_card_s = ImageFont.truetype("malgun.ttf", 14)
        font_card_d = ImageFont.truetype("malgun.ttf", 13)
    except Exception:
        font_title = font_sub = font_card_t = font_card_s = font_card_d = ImageFont.load_default()

    draw.text((margin + 10, 24), "한글 손글씨 골격 다양성 비교 (Allograph Skeletal Personas)", fill=(255, 255, 255), font=font_title)
    draw.text((margin + 12, 72), "동일한 문장이라도 화자의 성향과 목적에 따라 글자의 구조·획 분할·곡률이 완전히 달라지는 물리적 다양성", fill=(190, 205, 230), font=font_sub)

    for idx, (meta, img) in enumerate(rendered_cards):
        r = idx // cols
        c = idx % cols
        x = margin + c * (card_w + spacing)
        y = header_h + margin + r * (card_h + spacing)

        # Card container
        draw.rectangle([(x - 2, y - 2), (x + card_w + 2, y + card_h + 2)], fill=(225, 222, 215))
        draw.rectangle([(x, y), (x + card_w, y + card_h)], fill=(255, 255, 255))

        # Thumbnail
        thumb_area_h = card_h - 90
        img_thumb = img.copy()
        img_thumb.thumbnail((card_w - 20, thumb_area_h - 15), Image.Resampling.LANCZOS)
        tx = x + (card_w - img_thumb.width) // 2
        ty = y + 8 + (thumb_area_h - img_thumb.height) // 2
        board.paste(img_thumb, (tx, ty))

        # Banner at bottom
        bar_y = y + thumb_area_h + 2
        draw.line([(x + 14, bar_y), (x + card_w - 14, bar_y)], fill=(220, 218, 212), width=1)
        draw.text((x + 16, bar_y + 8), meta["name"], fill=(20, 25, 35), font=font_card_t)
        draw.text((x + 16, bar_y + 32), meta["subtitle"], fill=(40, 80, 160), font=font_card_s)
        draw.text((x + 16, bar_y + 54), meta["desc"], fill=(110, 115, 125), font=font_card_d)

    persona_board_path = out_dir / "gallery_allograph_personas.png"
    board.save(persona_board_path, quality=95)
    if brain_dir.exists():
        board.save(brain_dir / "gallery_allograph_personas.png")
    return persona_board_path


def main() -> None:
    out_dir = Path("outputs/physical_showcase")
    out_dir.mkdir(parents=True, exist_ok=True)
    brain_dir = Path("C:/Users/eljja/.gemini/antigravity/brain/d1c76613-c4b9-4b99-92f0-30fb9fa7c114")

    print("=== 1. Generating Micro-Physics Macro Comparison Board ===")
    p1 = generate_microphysics_comparison(out_dir, brain_dir)
    print(f"Generated: {p1}")

    print("=== 2. Generating Allograph Persona Comparison Board ===")
    p2 = generate_allograph_persona_comparison(out_dir, brain_dir)
    print(f"Generated: {p2}")

    print("=== All physical showcase assets generated successfully! ===")


if __name__ == "__main__":
    main()
