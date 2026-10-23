"""Generate 40 additional VibeFont Showcase images & build visual comparison boards + HTML gallery
combining the previous 20 showcase images with the new 40 images (60 total):

  1. Previous 20 Showcase (10 KO + 10 EN, >= 100 chars each)
  2. NEW Master Virtuoso Cursive ("명필 필기체"): 5 KO + 5 EN (>= 100 chars each)
  3. NEW Clumsy Elementary Kid Scrawl ("초딩 악필"): 5 KO + 5 EN (>= 100 chars each)
  4. NEW Expressive Calligraphy Short Works ("캘리그라피 단문"): 10 KO + 10 EN (centered cards)
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vibefont.document.pipeline import write_document
from vibefont.generate import get_generator
from vibefont.render.raster import render_page, save_images
from vibefont.vibe.interpreter import interpret


# ============================================================================
# 1. Master Virtuoso Cursive (5 Korean >= 100 chars + 5 English >= 100 chars)
# ============================================================================
MASTER_CURSIVE_KO = [
    {
        "id": "ko_master_01",
        "category": "master_cursive",
        "title": "창덕궁 후원 달빛 산책 (Moonlight in the Secret Garden)",
        "vibe": "아이보리 편지지에 남색 블루블랙 만년필로 명필처럼 아주 유려한 필기체로 우아하고 정성스럽게 쓴 글",
        "seed": 301,
        "text": (
            "부용지 잔잔한 수면 위로 둥근 가을달이 천천히 내려앉으면, 오래된 정자의 기와 끝마다 은빛 이슬이 맺힙니다.\n"
            "수백 년의 세월을 견뎌낸 아름드리 느티나무 사이로 불어오는 밤바람은 마치 옛 시인들이 남긴 나지막한 한숨처럼 "
            "부드럽게 옷자락을 스치며 지나갑니다."
        ),
    },
    {
        "id": "ko_master_02",
        "category": "master_cursive",
        "title": "오래된 바이올린 현의 기억 (Memories of an Old Violin)",
        "vibe": "미색 편지지에 버건디 와인색 만년필로 달필처럼 매우 우아하고 유려한 필기체로 천천히 쓴 편지",
        "seed": 302,
        "text": (
            "장인의 손끝에서 깎여 나온 단풍나무 울림통 속에는 백 년 전 어느 겨울밤에 연주되었던 야상곡의 잔향이 아직도 "
            "따뜻하게 잠들어 있습니다.\n"
            "활 끝이 현에 닿는 그 짧은 찰나에 침묵하던 시간은 다시 숨을 쉬기 시작하고, 잊혔던 이름들이 황금빛 선율로 "
            "조용히 피어오릅니다."
        ),
    },
    {
        "id": "ko_master_03",
        "category": "master_cursive",
        "title": "안개 낀 수목원의 아침 편지 (Misty Arboretum Morning)",
        "vibe": "줄노트 위에 숲색 초록 만년필로 명필의 멋진 필기체로 단정하고 우아하게 이어 쓴 기록",
        "seed": 303,
        "text": (
            "새벽 안개가 전나무 숲길을 하얗게 감싸 안은 아침, 축축한 이끼 위로 떨어진 솔방울에서 맑고 서늘한 풀내음이 "
            "번져 나갑니다.\n"
            "아무도 밟지 않은 오솔길을 따라 천천히 걸어가며 나뭇잎 사이로 부서지는 첫 햇살을 만년필 잉크 속에 고요히 "
            "담아 당신께 보냅니다."
        ),
    },
    {
        "id": "ko_master_04",
        "category": "master_cursive",
        "title": "고서점 창가에서의 필사록 (Transcribing in an Old Bookshop)",
        "vibe": "아이보리 종이에 세피아 갈색 만년필로 명필처럼 아주 유려한 필기체로 정성껏 쓴 필사 노트",
        "seed": 304,
        "text": (
            "빛바랜 양장본의 책장을 넘길 때마다 종이 섬유 속에 배어 있던 세월의 향기가 조용한 오후의 서점 안을 가득 "
            "채웁니다.\n"
            "누군가 밑줄을 그어 두었던 문장 위에 다시 만년필 촉을 얹어 한 자씩 따라 적다 보면, 시대를 건너온 사유의 "
            "온기가 손끝으로 고스란히 전해집니다."
        ),
    },
    {
        "id": "ko_master_05",
        "category": "master_cursive",
        "title": "겨울 바다 등대지기의 서신 (The Lighthouse Keeper's Letter)",
        "vibe": "크라프트 종이에 먹색 만년필로 달필의 멋진 필기체로 힘있고 우아하게 이어 쓴 편지",
        "seed": 305,
        "text": (
            "검푸른 수평선 너머로 거센 눈보라가 몰아치는 밤에도 등대의 불빛만은 흔들림 없이 머나먼 항로를 비추고 있습니다.\n"
            "어둠이 깊을수록 빛의 자리는 더욱 선명해지듯, 먼 바다를 항해하는 당신의 마음에도 언제나 돌아올 따뜻한 항구가 "
            "있음을 잊지 마시길 바랍니다."
        ),
    },
]

MASTER_CURSIVE_EN = [
    {
        "id": "en_master_01",
        "category": "master_cursive",
        "title": "Nocturne in the Winter Conservatory",
        "vibe": "ivory letter paper, burgundy fountain pen, virtuoso masterful spencerian cursive, very elegant slow",
        "seed": 401,
        "text": (
            "Beneath the vaulted glass roof of the winter conservatory, silver moonlight settles gently upon the "
            "sleeping camellias and stone fountains.\n"
            "A single cello sonata drifts from the distant parlor, weaving timeless grace through the quiet air "
            "until even the falling snow seems to pause and listen."
        ),
    },
    {
        "id": "en_master_02",
        "category": "master_cursive",
        "title": "Autumn Twilight in the Bodleian Library",
        "vibe": "cream paper, sepia brown fountain pen, copperplate masterful elegant cursive, careful",
        "seed": 402,
        "text": (
            "As October twilight deepens outside the tall leaded windows, the scent of cedarwood shelves and "
            "centuries-old vellum fills the quiet reading room.\n"
            "To trace these lines by lamplight is to converse with scholars across ages, united by the enduring "
            "beauty of patient ink and paper."
        ),
    },
    {
        "id": "en_master_03",
        "category": "master_cursive",
        "title": "Celestial Navigator's Starlight Log",
        "vibe": "ruled notebook, navy blue-black fountain pen, virtuoso masterful cursive, graceful neat",
        "seed": 403,
        "text": (
            "Two hours past midnight under a cloudless Atlantic sky: Orion rises clear above the starboard bow "
            "while phosphorescent waves glimmer along the hull.\n"
            "With sextant and compass aligned in calm harmony, our vessel holds her steady course toward the "
            "golden harbor of dawn."
        ),
    },
    {
        "id": "en_master_04",
        "category": "master_cursive",
        "title": "Botanist's Alpine Meadow Correspondence",
        "vibe": "ivory stationery, forest green fountain pen, masterful spencerian elegant cursive, slow",
        "seed": 404,
        "text": (
            "High above the timberline where glacier streams sing through beds of wild gentian, the morning air "
            "possesses a crystalline clarity found nowhere else on earth.\n"
            "I enclose a pressed blossom within this folded sheet, hoping its delicate azure petals bring you "
            "the tranquility of these mountain heights."
        ),
    },
    {
        "id": "en_master_05",
        "category": "master_cursive",
        "title": "Reflections by the Villa d'Este Fountains",
        "vibe": "kraft paper, charcoal sumi fountain pen, virtuoso copperplate graceful cursive, careful",
        "seed": 405,
        "text": (
            "Along the cypress-lined terraces of Tivoli, a hundred ancient fountains murmur in eternal counterpoint "
            "to the warm evening breeze.\n"
            "Time flows like water over mossy marble, teaching the observant heart that true nobility lies in "
            "unhurried poise and quiet gratitude."
        ),
    },
]

# ============================================================================
# 2. Clumsy Elementary Kid Scrawl (5 Korean >= 100 chars + 5 English >= 100 chars)
# ============================================================================
KID_SCRAWL_KO = [
    {
        "id": "ko_kid_01",
        "category": "kid_scrawl",
        "title": "치과 가기 싫은 날의 눈물 일기 (Dreading the Dentist)",
        "vibe": "줄공책에 연필로 엄청 못쓰는 초딩이 개발새발 삐뚤빼뚤하고 지저분하게 쓴 일기",
        "seed": 501,
        "text": (
            "오늘 엄마가 충치 치료하러 치과에 가자고 해서 식탁 밑에 숨었다가 결국 잡혀갔다! 위잉거리는 무서운 기계 소리가 "
            "날 때마다 두 눈을 꽉 감고 의자 손잡이를 부서질 만큼 잡았다.\n"
            "앞으로는 밤마다 초콜릿 몰래 안 먹고 양치질을 하루에 열 번씩 꼭 해야겠다고 다짐했다."
        ),
    },
    {
        "id": "ko_kid_02",
        "category": "kid_scrawl",
        "title": "체육시간 피구 대역전승 자랑 (Dodgeball Comeback Victory)",
        "vibe": "모눈 공책에 파란 볼펜으로 초딩이 엄청 못쓰는 악필로 급하고 삐뚤빼뚤하게 쓴 글",
        "seed": 502,
        "text": (
            "체육 시간에 옆 반이랑 피구 시합을 했는데 우리 편이 나 혼자 남아서 다들 졌다고 소리쳤다! 그런데 내가 날아오는 "
            "공을 연속으로 세 번이나 휙휙 피하고 마지막에 왕대포 슛을 던져서 상대편 에이스를 맞혔다!\n"
            "우리 반 애들이 전부 나한테 달려와서 하이파이브를 해서 어깨가 하늘 끝까지 올라갔다."
        ),
    },
    {
        "id": "ko_kid_03",
        "category": "kid_scrawl",
        "title": "하얀 옷에 떡볶이 국물 흘린 반성문 (Tteokbokki Stain Apology)",
        "vibe": "줄노트에 검정 사인펜 마커로 어린이가 개발새발 엄청 못쓰는 삐뚤빼뚤한 글씨로 쓴 반성문",
        "seed": 503,
        "text": (
            "엄마가 오늘 아침에 새로 사주신 하얀색 티셔츠를 입고 학교 앞 분식집에서 컵떡볶이를 먹다가 그만 빨간 국물을 "
            "왕창 흘리고 말았다.\n"
            "화장실에서 비누로 박박 문질렀는데 오히려 빨간 동그라미가 더 커져서 큰일 났다. 엄마 진짜 잘못했어요 다음부터는 "
            "앞치마를 두 개나 입고 먹을게요!"
        ),
    },
    {
        "id": "ko_kid_04",
        "category": "kid_scrawl",
        "title": "강아지가 수학 익힘책 물어뜯은 사건 (Dog Ate My Math Workbook)",
        "vibe": "모눈 방안에 연필로 초등 꼬마가 엄청 못쓰는 악필로 삐뚤빼뚤 지저분하게 쓴 일기",
        "seed": 504,
        "text": (
            "선생님 진짜 거짓말이 아니라 어제 밤에 분수 계산 숙제를 다 해놓고 방바닥에 뒀는데 우리 집 강아지 초코가 "
            "수학 익힘책 모서리를 질겅질겅 씹어 먹었어요!\n"
            "종이가 구멍이 뻥 뚫려서 답이 안 보이는 것뿐이지 저는 분명히 백 점짜리 답을 다 적어 놨었다고요!"
        ),
    },
    {
        "id": "ko_kid_05",
        "category": "kid_scrawl",
        "title": "비 오는 날 지렁이 구출 작전 (Rainy Day Earthworm Rescue)",
        "vibe": "줄공책에 초록 젤펜으로 초딩이 개발새발 엄청 못쓰는 삐뚤빼뚤한 큰 글씨로 쓴 관찰 일기",
        "seed": 505,
        "text": (
            "비가 엄청 많이 쏟아지는 하교길에 보도블록 위에서 길 잃은 대왕 지렁이 다섯 마리를 발견했다! 사람들이 모르고 "
            "밟을까 봐 나뭇잎에 조심조심 태워서 화단 흙 속으로 전부 이사를 시켜 주었다.\n"
            "장화 속에 빗물이 출렁출렁 들어가서 양말이 다 젖었지만 지렁이들을 구해 줘서 진짜 뿌듯했다."
        ),
    },
]

KID_SCRAWL_EN = [
    {
        "id": "en_kid_01",
        "category": "kid_scrawl",
        "title": "Why Boiled Broccoli Should Be Illegal",
        "vibe": "ruled notebook, pencil, clumsy kid terrible bad handwriting, messy wobbly scrawl",
        "seed": 601,
        "text": (
            "I think boiled broccoli looks like tiny green alien trees and tastes like wet grass from the soccer "
            "field! Tonight I tried hiding three pieces under my mashed potatoes but Mom has super X-ray vision.\n"
            "When I become president my first law will be pizza and ice cream every single Tuesday!"
        ),
    },
    {
        "id": "en_kid_02",
        "category": "kid_scrawl",
        "title": "Operation Rescue Hamster Under the Fridge",
        "vibe": "grid graph paper, blue ballpoint pen, child clumsy terrible wobbly messy print",
        "seed": 602,
        "text": (
            "During cartoon time my hamster Captain Fuzzball squeezed through the cage door and zoomed straight "
            "underneath the kitchen refrigerator!\n"
            "We had to build a trail of sunflower seeds and cheddar cheese cubes across the linoleum floor until "
            "he finally wiggled out covered in dust bunnies."
        ),
    },
    {
        "id": "en_kid_03",
        "category": "kid_scrawl",
        "title": "Top Secret Backyard Treehouse Club Rules",
        "vibe": "kraft brown paper, red marker pen, kid clumsy bad handwriting, messy wobbly scrawl",
        "seed": 603,
        "text": (
            "Official Rules of the Backyard Ninja Treehouse: Rule number one is nobody is allowed up the rope "
            "ladder without saying the secret dinosaur password!\n"
            "Rule number two is anyone who brings comic books or sour gummy worms automatically gets to be "
            "vice captain for the whole afternoon!"
        ),
    },
    {
        "id": "en_kid_04",
        "category": "kid_scrawl",
        "title": "The Giant Baking Soda Volcano Disaster",
        "vibe": "ruled paper, black ballpoint pen, clumsy child terrible wobbly messy handwriting",
        "seed": 604,
        "text": (
            "For my science fair volcano experiment I poured way too much vinegar and red food coloring into the "
            "clay mountain all at once!\n"
            "Giant pink bubbles exploded all over the kitchen ceiling and my little brother's hair turned bright "
            "strawberry pink before Dad could grab the paper towels."
        ),
    },
    {
        "id": "en_kid_05",
        "category": "kid_scrawl",
        "title": "My Pet Frog Jumped into the Teacher's Desk",
        "vibe": "grid paper, pencil, child clumsy bad handwriting, wobbly messy scrawl",
        "seed": 605,
        "text": (
            "I brought a super cool bullfrog in my lunchbox for Show and Tell, but right during spelling quiz "
            "he popped the lid open and hopped onto Mrs. Miller's dictionary!\n"
            "The whole third grade class laughed so hard that even the principal came in to help catch him."
        ),
    },
]

# ============================================================================
# 3. Expressive Calligraphy Short Works (10 Korean + 10 English, centered cards)
# ============================================================================
CALLIGRAPHY_KO = [
    {
        "id": "ko_calli_01",
        "category": "calligraphy",
        "title": "매화 향기 (Plum Blossom Fragrance)",
        "vibe": "아이보리 종이에 먹색 붓으로 힘차고 우아하게 쓴 캘리그라피",
        "seed": 701,
        "text": "눈보라 속에서도\n매화는 자신의 향기를\n팔지 않는다",
    },
    {
        "id": "ko_calli_02",
        "category": "calligraphy",
        "title": "별을 노래하는 마음 (Heart Singing of Stars)",
        "vibe": "미색 종이에 남색 블루블랙 평촉 캘리 펜으로 정성스럽고 우아하게 쓴 글",
        "seed": 702,
        "text": "별을 노래하는 마음으로\n모든 죽어가는 것을\n사랑해야지",
    },
    {
        "id": "ko_calli_03",
        "category": "calligraphy",
        "title": "고요한 호수의 달 (Moon on the Still Lake)",
        "vibe": "크라프트 종이에 먹물 붓으로 굵고 당당하며 유려하게 쓴 캘리",
        "seed": 703,
        "text": "바람이 그치면\n비로소 호수 위에\n둥근 달이 머문다",
    },
    {
        "id": "ko_calli_04",
        "category": "calligraphy",
        "title": "봄비와 꽃잎 (Spring Rain and Petals)",
        "vibe": "아이보리 편지지에 버건디 평촉 캘리 펜으로 부드럽고 우아하게 쓴 글",
        "seed": 704,
        "text": "가만히 내리는 봄비에\n세상 모든 나뭇가지마다\n연분홍 등불이 켜진다",
    },
    {
        "id": "ko_calli_05",
        "category": "calligraphy",
        "title": "대나무의 절개 (Bamboo in the Wind)",
        "vibe": "무지 종이에 먹색 붓으로 강하고 날카로우며 힘있게 쓴 캘리그라피",
        "seed": 705,
        "text": "바람에 흔들릴지언정\n푸른 대나무는\n결코 꺾이지 않는다",
    },
    {
        "id": "ko_calli_06",
        "category": "calligraphy",
        "title": "차 한 잔의 여유 (Warmth of a Cup of Tea)",
        "vibe": "크라프트 종이에 세피아 갈색 붓으로 따뜻하고 부드럽게 쓴 캘리",
        "seed": 706,
        "text": "따뜻한 찻잔 속에\n오늘 하루의 수고를\n조용히 내려놓는다",
    },
    {
        "id": "ko_calli_07",
        "category": "calligraphy",
        "title": "깊은 숲의 발걸음 (Steps in the Deep Forest)",
        "vibe": "아이보리 종이에 숲색 초록 평촉 캘리 펜으로 차분하고 단정하게 쓴 글",
        "seed": 707,
        "text": "천천히 걸어가는 사람만이\n숲이 들려주는 작은 노래를\n온전히 듣는다",
    },
    {
        "id": "ko_calli_08",
        "category": "calligraphy",
        "title": "새벽의 첫 빛 (First Light of Dawn)",
        "vibe": "미색 종이에 남색 붓으로 힘차고 자신감 있게 쓴 캘리그라피",
        "seed": 708,
        "text": "가장 어두운 밤의 끝에서\n가장 눈부신 새벽이\n시작된다",
    },
    {
        "id": "ko_calli_09",
        "category": "calligraphy",
        "title": "파도와 바위 (Waves and the Cliff)",
        "vibe": "무지 종이에 청록색 평촉 캘리 펜으로 유려하고 우아하게 쓴 글",
        "seed": 709,
        "text": "수천 번 부서지는 파도가\n마침내 가장 아름다운\n해안선을 조각한다",
    },
    {
        "id": "ko_calli_10",
        "category": "calligraphy",
        "title": "당신이라는 계절 (The Season Called You)",
        "vibe": "아이보리 종이에 보라색 붓으로 부드럽고 우아하게 이어 쓴 캘리",
        "seed": 710,
        "text": "당신이 웃던 그 순간\n나의 모든 시간은\n따뜻한 봄이 되었다",
    },
]

CALLIGRAPHY_EN = [
    {
        "id": "en_calli_01",
        "category": "calligraphy",
        "title": "Still Waters Run Deep",
        "vibe": "ivory paper, charcoal sumi brush, calligraphy, bold graceful slow cursive",
        "seed": 801,
        "text": "In the deepest silence\nthe clearest wisdom\nquietly awakens.",
    },
    {
        "id": "en_calli_02",
        "category": "calligraphy",
        "title": "Light Through Broken Clouds",
        "vibe": "cream paper, burgundy broad nib calligraphy pen, elegant masterful cursive",
        "seed": 802,
        "text": "Every shadow upon the earth\nis born only because\nthe light endures.",
    },
    {
        "id": "en_calli_03",
        "category": "calligraphy",
        "title": "Patience of the Oak",
        "vibe": "kraft brown paper, forest green broad nib calligraphy pen, careful bold confident",
        "seed": 803,
        "text": "The mighty oak was once\na tiny acorn that held\nits ground in the storm.",
    },
    {
        "id": "en_calli_04",
        "category": "calligraphy",
        "title": "Music of the Night Sky",
        "vibe": "ivory paper, navy blue-black broad nib calligraphy pen, virtuoso graceful cursive",
        "seed": 804,
        "text": "Dwell on the beauty of life;\nwatch the stars, and see\nyourself running with them.",
    },
    {
        "id": "en_calli_05",
        "category": "calligraphy",
        "title": "The Art of Beginning",
        "vibe": "plain paper, crimson red brush, calligraphy, bold confident slanted",
        "seed": 805,
        "text": "Courage is not the absence\nof fear, but the triumph\nof purpose over hesitation.",
    },
    {
        "id": "en_calli_06",
        "category": "calligraphy",
        "title": "Autumn Leaves in Gold",
        "vibe": "cream paper, sepia brown broad nib calligraphy pen, soft warm graceful cursive",
        "seed": 806,
        "text": "Autumn is a second spring\nwhen every leaf becomes\na golden flower.",
    },
    {
        "id": "en_calli_07",
        "category": "calligraphy",
        "title": "Voyage Beyond the Horizon",
        "vibe": "ivory paper, teal turquoise broad nib calligraphy pen, elegant airy cursive",
        "seed": 807,
        "text": "A ship in harbor is safe,\nbut that is not what\nships are built for.",
    },
    {
        "id": "en_calli_08",
        "category": "calligraphy",
        "title": "Simplicity and Grace",
        "vibe": "kraft paper, charcoal sumi broad nib calligraphy pen, neat careful bold",
        "seed": 808,
        "text": "Simplicity is the ultimate\nsophistication of a mind\nat peace with the world.",
    },
    {
        "id": "en_calli_09",
        "category": "calligraphy",
        "title": "Kindness in Every Season",
        "vibe": "ivory paper, lavender purple brush, calligraphy, graceful soft cursive",
        "seed": 809,
        "text": "Where there is kindness,\nthere is goodness, and\nevery winter turns to spring.",
    },
    {
        "id": "en_calli_10",
        "category": "calligraphy",
        "title": "Unwritten Tomorrow",
        "vibe": "plain paper, black broad nib calligraphy pen, virtuoso masterful elegant cursive",
        "seed": 810,
        "text": "Each morning offers a blank\npage waiting for a bold\nand generous hand.",
    },
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("malgunbd.ttf", "malgun.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact_sheet(
    entries: list[dict],
    out_path: Path,
    title: str,
    subtitle: str,
    cols: int = 5,
    thumb_w: int = 360,
    thumb_h: int = 460,
) -> None:
    """Renders a high-resolution labeled visual grid (contact sheet) of showcase images."""
    rows = (len(entries) + cols - 1) // cols
    pad = 24
    header_h = 110
    caption_h = 68
    cell_w = thumb_w + pad
    cell_h = thumb_h + caption_h + pad
    board_w = cols * cell_w + pad
    board_h = header_h + rows * cell_h + pad

    board = Image.new("RGB", (board_w, board_h), (24, 26, 32))
    draw = ImageDraw.Draw(board)
    f_title = _load_font(34)
    f_sub = _load_font(20)
    f_id = _load_font(17)
    f_desc = _load_font(14)

    draw.text((pad, 22), title, fill=(245, 247, 252), font=f_title)
    draw.text((pad, 68), subtitle, fill=(168, 178, 198), font=f_sub)

    for i, e in enumerate(entries):
        r = i // cols
        c = i % cols
        x0 = pad + c * cell_w
        y0 = header_h + r * cell_h

        # Card background
        draw.rounded_rectangle(
            [x0, y0, x0 + thumb_w, y0 + thumb_h + caption_h],
            radius=10,
            fill=(36, 40, 50),
            outline=(68, 75, 92),
            width=2,
        )

        img_path = Path(e["file"])
        if img_path.exists():
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                # Fit inside (thumb_w - 16, thumb_h - 16) preserving aspect ratio
                iw, ih = im_rgb.size
                scale = min((thumb_w - 16) / iw, (thumb_h - 16) / ih)
                nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
                resized = im_rgb.resize((nw, nh), Image.Resampling.LANCZOS)
                px = x0 + (thumb_w - nw) // 2
                py = y0 + (thumb_h - nh) // 2
                board.paste(resized, (px, py))

        # Caption
        cy = y0 + thumb_h + 6
        badge = f"[{e['id']}] {e['title'][:24]}"
        meta = f"pen={e['pen']} | paper={e['paper']} | {e['char_count']} chars"
        draw.text((x0 + 10, cy), badge, fill=(238, 242, 250), font=f_id)
        draw.text((x0 + 10, cy + 30), meta, fill=(150, 165, 190), font=f_desc)

    board.save(out_path, dpi=(150, 150))


def build_html_gallery(manifest: list[dict], out_html: Path) -> None:
    """Generates a standalone interactive HTML gallery viewing all 60 handwriting works."""
    cards_html = []
    cat_labels = {
        "original": ("지난 쇼케이스 20선 (Original 20)", "#3b82f6"),
        "master_cursive": ("명필 필기체 10선 (Virtuoso Cursive)", "#8b5cf6"),
        "kid_scrawl": ("초딩 악필 10선 (Clumsy Kid Scrawl)", "#f59e0b"),
        "calligraphy": ("캘리그라피 단문 20선 (Calligraphy Cards)", "#10b981"),
    }

    for item in manifest:
        cat = item.get("category", "original")
        cat_name, badge_color = cat_labels.get(cat, ("Showcase", "#64748b"))
        fname = Path(item["file"]).name
        lang_badge = "한국어 (KO)" if item["lang"] == "ko" else "English (EN)"
        r, g, b = [int(c * 255) for c in item["ink_rgb"]]
        cards_html.append(
            f"""
            <article class="card" data-category="{html.escape(cat)}" data-lang="{html.escape(item['lang'])}">
              <div class="thumb-wrap" onclick="openModal('{html.escape(fname)}', '{html.escape(item['title'])}')">
                <img src="{html.escape(fname)}" alt="{html.escape(item['title'])}" loading="lazy" />
              </div>
              <div class="card-body">
                <div class="badges">
                  <span class="badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}55">{html.escape(cat_name)}</span>
                  <span class="badge lang">{lang_badge}</span>
                  <span class="badge">{item['char_count']}자</span>
                </div>
                <h3>{html.escape(item['id'])} — {html.escape(item['title'])}</h3>
                <p class="vibe"><strong>Vibe:</strong> {html.escape(item['vibe'])}</p>
                <p class="meta">
                  <span class="ink-dot" style="background:rgb({r},{g},{b})"></span>
                  Pen: <code>{html.escape(item['pen'])}</code> &middot; Paper: <code>{html.escape(item['paper'])}</code>
                </p>
                <details>
                  <summary>원문 텍스트 보기 (View Text)</summary>
                  <pre>{html.escape(item['text'])}</pre>
                </details>
              </div>
            </article>
            """
        )

    page_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>VibeFont Handwriting Master Gallery (60 Works)</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #6366f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Pretendard', 'Malgun Gothic', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 28px 36px;
    }}
    header {{
      max-width: 1680px;
      margin: 0 auto 28px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      border-bottom: 1px solid #334155;
      padding-bottom: 20px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    .subtitle {{ margin: 0; color: var(--muted); font-size: 15px; }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    button.filter-btn {{
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #475569;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      transition: all 0.15s;
    }}
    button.filter-btn.active, button.filter-btn:hover {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .boards-bar {{
      max-width: 1680px;
      margin: 0 auto 24px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 14px 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    .boards-bar a {{
      color: #a5b4fc;
      text-decoration: none;
      font-size: 14px;
      font-weight: 600;
      padding: 6px 12px;
      background: #0f172a;
      border-radius: 6px;
      border: 1px solid #334155;
    }}
    .boards-bar a:hover {{ border-color: #818cf8; color: #fff; }}
    .grid {{
      max-width: 1680px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid #334155;
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s, border-color 0.15s;
    }}
    .card:hover {{
      transform: translateY(-3px);
      border-color: #6366f1;
    }}
    .thumb-wrap {{
      background: #0b1120;
      height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 12px;
      cursor: zoom-in;
    }}
    .thumb-wrap img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 6px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.45);
    }}
    .card-body {{ padding: 14px 16px 18px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .badge {{
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 99px;
      background: #334155;
      color: #e2e8f0;
      font-weight: 600;
    }}
    h3 {{ margin: 4px 0 8px; font-size: 16px; line-height: 1.35; }}
    p.vibe {{ margin: 0 0 8px; font-size: 13px; color: #cbd5e1; line-height: 1.4; }}
    p.meta {{ margin: 0 0 10px; font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }}
    .ink-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; border: 1px solid #fff4; }}
    details summary {{ cursor: pointer; font-size: 12px; color: #93c5fd; }}
    details pre {{
      white-space: pre-wrap;
      font-size: 12px;
      background: #0f172a;
      padding: 10px;
      border-radius: 8px;
      color: #e2e8f0;
      margin-top: 8px;
    }}
    #modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(5, 8, 18, 0.92);
      z-index: 9999;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      padding: 24px;
    }}
    #modal.open {{ display: flex; }}
    #modal img {{ max-width: 92vw; max-height: 85vh; border-radius: 8px; }}
    #modal-caption {{ margin-top: 12px; font-size: 18px; font-weight: 600; color: #f8fafc; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VibeFont Handwriting Showcase Gallery (총 60선 한눈에 보기)</h1>
      <p class="subtitle">이전 쇼케이스 20선 + 명필 필기체 10선 + 초딩 악필 10선 + 캘리그라피 단문 20선</p>
    </div>
    <div class="controls">
      <button class="filter-btn active" onclick="filterCards('all')">전체 보기 (60)</button>
      <button class="filter-btn" onclick="filterCards('original')">지난 쇼케이스 (20)</button>
      <button class="filter-btn" onclick="filterCards('master_cursive')">명필 필기체 (10)</button>
      <button class="filter-btn" onclick="filterCards('kid_scrawl')">초딩 악필 (10)</button>
      <button class="filter-btn" onclick="filterCards('calligraphy')">캘리그라피 단문 (20)</button>
      <button class="filter-btn" onclick="filterLang('ko')">한글 (30)</button>
      <button class="filter-btn" onclick="filterLang('en')">English (30)</button>
    </div>
  </header>

  <div class="boards-bar">
    <span><strong>한눈에 비교하는 종합 보드 (Contact Sheets):</strong></span>
    <a href="gallery_all_60_master_board.png" target="_blank">전체 60선 마스터 비교 보드</a>
    <a href="gallery_01_previous_20.png" target="_blank">1. 지난 쇼케이스 20선 보드</a>
    <a href="gallery_02_master_cursive_10.png" target="_blank">2. 명필 필기체 10선 보드</a>
    <a href="gallery_03_kid_scrawl_10.png" target="_blank">3. 초딩 악필 10선 보드</a>
    <a href="gallery_04_calligraphy_20.png" target="_blank">4. 캘리그라피 20선 보드</a>
  </div>

  <main class="grid" id="gallery-grid">
    {''.join(cards_html)}
  </main>

  <div id="modal" onclick="closeModal()">
    <img id="modal-img" src="" alt="" />
    <div id="modal-caption"></div>
  </div>

  <script>
    function filterCards(cat) {{
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      document.querySelectorAll('.card').forEach(c => {{
        c.style.display = (cat === 'all' || c.dataset.category === cat) ? 'flex' : 'none';
      }});
    }}
    function filterLang(lang) {{
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      document.querySelectorAll('.card').forEach(c => {{
        c.style.display = (c.dataset.lang === lang) ? 'flex' : 'none';
      }});
    }}
    function openModal(src, title) {{
      document.getElementById('modal-img').src = src;
      document.getElementById('modal-caption').textContent = title;
      document.getElementById('modal').classList.add('open');
    }}
    function closeModal() {{
      document.getElementById('modal').classList.remove('open');
    }}
  </script>
</body>
</html>
"""
    out_html.write_text(page_html, encoding="utf-8")


def main() -> None:
    out_dir = Path("outputs/showcase")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load previous 20 entries and include their original text from generate_showcase_20
    from generate_showcase_20 import ENGLISH_SHOWCASE, KOREAN_SHOWCASE

    prev_map = {item["id"]: item for item in (KOREAN_SHOWCASE + ENGLISH_SHOWCASE)}
    manifest_path = out_dir / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    original_entries = []
    for e in existing_manifest:
        if e["id"] in prev_map:
            e["category"] = "original"
            e["text"] = prev_map[e["id"]]["text"]
            original_entries.append(e)

    gen = get_generator("model", "checkpoints/bilingual_5k_core")
    new_items = MASTER_CURSIVE_KO + MASTER_CURSIVE_EN + KID_SCRAWL_KO + KID_SCRAWL_EN + CALLIGRAPHY_KO + CALLIGRAPHY_EN

    print(f"Generating {len(new_items)} new showcase images using generator='{gen.name}'...", flush=True)
    t0 = time.perf_counter()
    new_entries = []

    for i, item in enumerate(new_items, 1):
        char_len = len(item["text"].replace("\n", ""))
        cat = item["category"]
        if cat in ("master_cursive", "kid_scrawl"):
            assert char_len >= 100, f"{item['id']} has only {char_len} chars (< 100)!"

        spec = interpret(item["vibe"])
        spec.seed = item["seed"]

        if cat == "master_cursive":
            spec.layout.page = "B5"
            spec.layout.margin_mm = (18.0, 15.0, 18.0, 15.0)
            spec.style.neatness = max(spec.style.neatness, 0.90)
            spec.style.connectedness = max(spec.style.connectedness, 0.86)
            spec.style.consistency = max(spec.style.consistency, 0.90)
            spec.style.baseline_wobble = min(spec.style.baseline_wobble, 0.06)
        elif cat == "kid_scrawl":
            spec.layout.page = "B5"
            spec.layout.margin_mm = (18.0, 15.0, 18.0, 15.0)
            spec.style.neatness = min(spec.style.neatness, 0.10)
            spec.style.connectedness = min(spec.style.connectedness, 0.05)
            spec.style.consistency = 0.50
            spec.style.baseline_wobble = 0.34
            spec.style.size_mm = 3.25
            spec.pen.width_mm = min(spec.pen.width_mm, 0.62)
            if spec.paper.type == "grid":
                spec.paper.line_spacing_mm = 7.5
        elif cat == "calligraphy":
            spec.layout.page = "Card"
            spec.layout.margin_mm = (12.0, 10.0, 12.0, 10.0)
            spec.layout.align = "center"
            spec.layout.v_align = "center"
            spec.style.size_mm = 4.4 if item["id"].startswith("ko_") else 3.35
            spec.style.connectedness = max(spec.style.connectedness, 0.76)
            spec.style.neatness = max(spec.style.neatness, 0.84)

        rng = np.random.default_rng(spec.seed)
        pages = write_document(item["text"], spec, gen, rng)
        assert len(pages) == 1, f"{item['id']} overflowed onto {len(pages)} pages!"
        img = render_page(pages[0], spec, dpi=200, rng=np.random.default_rng(spec.seed + 1000))

        out_path = out_dir / f"{item['id']}.png"
        save_images([img], out_path, dpi=200)

        entry = {
            "index": len(original_entries) + i,
            "id": item["id"],
            "category": cat,
            "title": item["title"],
            "lang": "ko" if item["id"].startswith("ko_") else "en",
            "char_count": char_len,
            "vibe": item["vibe"],
            "text": item["text"],
            "matched_rules": spec.matched,
            "pen": spec.pen.type,
            "paper": spec.paper.type,
            "ink_rgb": [round(float(c), 3) for c in spec.ink.color],
            "file": str(out_path.resolve()),
        }
        new_entries.append(entry)
        print(
            f"[{i:02d}/40] {out_path.name} ({char_len} chars, {cat}) | pen={spec.pen.type} paper={spec.paper.type}",
            flush=True,
        )

    all_60 = original_entries + new_entries
    manifest_path.write_text(json.dumps(all_60, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build category contact sheets + master 60-image board
    print("Building visual comparison contact sheets & HTML gallery...", flush=True)
    build_contact_sheet(
        original_entries,
        out_dir / "gallery_01_previous_20.png",
        "1. Previous Showcase 20 Works (한글 10선 + 영어 10선)",
        "10 Korean & 10 English mood-matched handwriting documents (>= 100 chars each)",
        cols=5,
    )
    build_contact_sheet(
        [e for e in new_entries if e["category"] == "master_cursive"],
        out_dir / "gallery_02_master_cursive_10.png",
        "2. Master Virtuoso Cursive 10 Works (명필/달필 필기체: 한글 5선 + 영어 5선)",
        "High-discipline, flowing S-curve ligatures, flex-nib fountain pen shading (>= 100 chars each)",
        cols=5,
    )
    build_contact_sheet(
        [e for e in new_entries if e["category"] == "kid_scrawl"],
        out_dir / "gallery_03_kid_scrawl_10.png",
        "3. Clumsy Elementary Kid Scrawl 10 Works (초딩 악필: 한글 5선 + 영어 5선)",
        "High sub-syllable jitter, bouncing baseline, uneven character sizing (>= 100 chars each)",
        cols=5,
    )
    build_contact_sheet(
        [e for e in new_entries if e["category"] == "calligraphy"],
        out_dir / "gallery_04_calligraphy_20.png",
        "4. Expressive Calligraphy Short Cards 20 Works (캘리그라피 단문: 한글 10선 + 영어 10선)",
        "Centered artistic card composition with sumi brush & broad-nib calligraphy pens",
        cols=5,
        thumb_w=360,
        thumb_h=300,
    )
    build_contact_sheet(
        all_60,
        out_dir / "gallery_all_60_master_board.png",
        "VibeFont Complete 60 Handwriting Masterpieces — Side-by-Side Overview",
        "Rows 1-4: Previous 20 | Rows 5-6: Master Cursive 10 | Rows 7-8: Kid Scrawl 10 | Rows 9-12: Calligraphy 20",
        cols=5,
        thumb_w=300,
        thumb_h=360,
    )
    build_html_gallery(all_60, out_dir / "index.html")

    elapsed = time.perf_counter() - t0
    print(f"=== Completed 40 new images + 5 comparison boards + HTML gallery in {elapsed:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
