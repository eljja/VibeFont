"""Generate 20 VibeFont Showcase Masterpieces:
  - 10 Korean texts (>= 100 chars each) with distinct moods & matching handwriting vibes
  - 10 English texts (>= 100 chars each) with distinct moods & matching handwriting vibes
Saved locally as high-resolution PNG files in `outputs/showcase/`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from vibefont.document.pipeline import write_document
from vibefont.generate import get_generator
from vibefont.render.raster import render_page, save_images
from vibefont.vibe.interpreter import interpret


KOREAN_SHOWCASE = [
    {
        "id": "ko_01_love_letter",
        "title": "가을밤의 연서 (Romantic Autumn Love Letter)",
        "vibe": "아이보리 편지지에 버건디 만년필로 천천히 정성스럽고 우아하게 쓴 손편지",
        "seed": 101,
        "text": (
            "창밖에 늦가을 단풍잎이 조용히 내려앉는 밤입니다. 오래전 당신과 함께 걸었던 그 좁은 골목길의 공기마저 "
            "오늘따라 유난히 선명하게 떠오릅니다.\n"
            "말로는 다 전하지 못했던 고마움과 그리움을 한 글자씩 만년필 끝에 담아 보냅니다. 부디 다가오는 계절에도 "
            "당신의 모든 아침이 따뜻하고 평안하기를 진심으로 바랍니다."
        ),
    },
    {
        "id": "ko_02_lecture_notes",
        "title": "물리학 강의 필기 (Hurried Physics Lecture Notes)",
        "vibe": "줄노트 위에 파란 볼펜으로 급하고 빠르게 이어 쓴 강의 메모",
        "seed": 102,
        "text": (
            "제4장 열역학 제2법칙과 엔트로피의 비가역성 정리: 고립계에서 자발적인 모든 물리적 과정은 항상 총 엔트로피가 "
            "증가하는 방향으로만 진행된다.\n"
            "즉 시간의 화살은 열적 평형 상태를 향해 흐르며, 초기 조건의 미세한 불확실성은 비선형 동역학계를 거치면서 "
            "지수함수적으로 증폭된다는 점을 시험 전에 반드시 복습할 것."
        ),
    },
    {
        "id": "ko_03_rainy_diary",
        "title": "비 오는 새벽의 감성 일기 (Rainy Midnight Diary)",
        "vibe": "미색 일기장에 블루블랙 만년필로 차분하고 부드럽게 쓴 밤의 감성 일기",
        "seed": 103,
        "text": (
            "새벽 두 시, 창문을 두드리는 빗소리에 잠에서 깨어나 따뜻한 차 한 잔을 우렸다. 하루 동안 스쳐 지나갔던 "
            "수많은 생각들이 빗방울처럼 차분히 가라앉는다.\n"
            "완벽하지 않았던 오늘의 실수들도 이 조용한 밤안개 속에서는 조금씩 용서받는 기분이 든다. 내일은 조금 더 "
            "가벼운 마음으로 하루를 시작할 수 있을 것 같다."
        ),
    },
    {
        "id": "ko_04_child_journal",
        "title": "어린아이의 그림 관찰 일기 (Child's Playful Observation Journal)",
        "vibe": "모눈 공책에 진한 연필로 어린이가 삐뚤빼뚤하고 큰 글씨로 꾹꾹 눌러 쓴 일기",
        "seed": 104,
        "text": (
            "오늘 학교 끝나고 뒷산 공원에서 강아지랑 노란 나비를 따라 신나게 뛰어다녔다! 나비가 민들레 꽃잎 위에 "
            "살짝 앉았을 때 숨을 참으며 가까이 다가갔는데 바람이 불어서 하늘 높이 날아가 버렸다.\n"
            "집으로 돌아오는 길에 엄마가 사주신 딸기 아이스크림은 세상에서 제일 달콤하고 맛있었다."
        ),
    },
    {
        "id": "ko_05_brush_philosophy",
        "title": "묵향의 고전 사색록 (Classical Brush Meditation)",
        "vibe": "크라프트 종이 위에 먹색 붓으로 힘차고 당당하며 굵게 쓴 고전 사색",
        "seed": 105,
        "text": (
            "산이 높다고 하여 구름을 마다하지 않으며, 깊은 바다는 작은 시냇물을 가리지 않고 품는다. 사람의 마음 또한 "
            "고요한 호수와 같아서 바람이 불면 물결이 일지만 바람이 그치면 다시 본래의 맑은 빛을 되찾는다.\n"
            "눈앞의 빠른 성취에 흔들리지 말고 묵묵히 자신의 길을 걸어가는 자만이 마침내 깊은 숲의 이치를 깨닫는다."
        ),
    },
    {
        "id": "ko_06_pencil_margin_memo",
        "title": "새벽 독서 여백 메모 (Late-Night Pencil Reading Notes)",
        "vibe": "회색 샤프 연필로 작고 단정하며 또박또박 꼼꼼하게 적은 독서 노트",
        "seed": 106,
        "text": (
            "소설 속 주인공이 마주한 침묵은 단순한 체념이 아니라 세계를 다시 이해하려는 가장 능동적인 기다림이었다.\n"
            "저자는 화려한 수식어를 덜어내고 문장과 문장 사이의 여백을 통해 인물의 내면을 입체적으로 드러낸다. "
            "우리가 일상에서 쉽게 지나치는 사소한 몸짓 속에 오히려 가장 진실한 삶의 단서가 숨어 있음을 기억하자."
        ),
    },
    {
        "id": "ko_07_grandma_letter",
        "title": "할머니의 따뜻한 당부 편지 (Grandmother's Affectionate Letter)",
        "vibe": "아이보리 편지지에 세피아 갈색 볼펜으로 할머니가 천천히 정성스럽게 쓴 따뜻한 손편지",
        "seed": 107,
        "text": (
            "보고 싶은 우리 손주야, 요즘 일교차가 부쩍 커졌는데 아침저녁으로 옷은 따뜻하게 입고 다니는지 늘 마음이 쓰인다.\n"
            "바쁜 일상에 쫓기더라도 끼니만큼은 거르지 말고 따뜻한 국물에 밥을 꼭 챙겨 먹어라. 이번 주말에 집에 오면 "
            "네가 어릴 적부터 좋아하던 달콤한 단호박 식혜랑 따뜻한 갈비찜을 넉넉히 해놓고 기다리마."
        ),
    },
    {
        "id": "ko_08_fierce_manifesto",
        "title": "변화를 향한 결의문 (Fierce Red-Marker Manifesto)",
        "vibe": "빨간 마커 사인펜으로 화난 듯 강하고 날카로우며 빠르게 쓴 결의문",
        "seed": 108,
        "text": (
            "더 이상 안일한 타협과 침묵 뒤에 숨어 변화를 미룰 수는 없다! 지금 우리가 마주한 위기는 어제의 관성을 "
            "과감히 끊어낼 때만 비로소 극복될 수 있다.\n"
            "두려움을 넘어 현장의 목소리를 정면으로 직시하고, 선언에 그치지 않는 실질적인 행동과 단호한 책임으로 "
            "새로운 내일의 기준을 당장 오늘 여기서부터 증명하라!"
        ),
    },
    {
        "id": "ko_09_doctor_clinical_log",
        "title": "전문의 임상 회진 기록 (Physician's Rapid Clinical Chart)",
        "vibe": "모눈 종이 위에 검정 볼펜으로 의사가 처방전처럼 아주 빠르고 흘려 쓴 기록",
        "seed": 109,
        "text": (
            "환자 오전 회진 소견: 급성 인후통 및 미열 증상은 전일 대비 뚜렷한 호전 양상을 보이나, 야간 기침과 경미한 "
            "피로감이 잔존하여 수분 섭취 권장 및 경과 관찰 요망.\n"
            "호흡기 청진상 특이 수포음 없으며 혈압과 맥박 모두 정상 범위 내 안정적 유지 중. 3일분 대증 치료 처방 유지 후 "
            "외래 재내원 안내 완료."
        ),
    },
    {
        "id": "ko_10_cafe_postcard",
        "title": "봄날 제주 카페의 여행 엽서 (Sunny Jeju Cafe Travel Postcard)",
        "vibe": "크라프트 종이에 보라색 젤펜으로 귀엽고 동글동글하며 아기자기하게 쓴 여행 엽서",
        "seed": 110,
        "text": (
            "제주도의 작은 바닷가 마을 카페에서 창밖의 에메랄드빛 파도를 보며 이 엽서를 적어. 방금 구워낸 고소한 당근 케이크 "
            "한 조각이랑 향긋한 한라봉 에이드를 마시니까 그동안 쌓였던 피로가 눈 녹듯 사라지는 기분이야!\n"
            "돌담길 옆에 활짝 핀 유채꽃 사진도 잔뜩 찍어 두었으니 돌아가서 우리 같이 보면서 밤새 이야기 나누자."
        ),
    },
]

ENGLISH_SHOWCASE = [
    {
        "id": "en_01_romantic_letter",
        "title": "Autumn Evening Love Letter",
        "vibe": "ivory letter paper, burgundy fountain pen, very elegant and graceful slow cursive love letter",
        "seed": 201,
        "text": (
            "My dearest friend, as the golden autumn light fades softly across the garden wall, I find myself "
            "returning to the quiet conversations we shared last October.\n"
            "Every word you spoke still lingers like warm music in an old room, reminding me that true kindness "
            "never really leaves us, no matter how far the seasons carry us apart."
        ),
    },
    {
        "id": "en_02_lecture_notebook",
        "title": "Statistical Mechanics Lecture Notes",
        "vibe": "ruled notebook, blue ballpoint pen, hurried fast slanted lecture notes",
        "seed": 202,
        "text": (
            "Lecture 7 - Statistical Mechanics & Phase Transitions: Near the critical point, correlation length "
            "diverges and microscopic details become secondary to universal scaling laws.\n"
            "Remember to check the renormalization group flow equations and verify critical exponents before "
            "Friday's problem set deadline!"
        ),
    },
    {
        "id": "en_03_midnight_journal",
        "title": "Rainy Midnight Journal Entry",
        "vibe": "cream diary paper, navy blue-black fountain pen, soft calm reflective cursive",
        "seed": 203,
        "text": (
            "Midnight by the window: the rain has finally slowed to a gentle rhythm against the glass. Writing "
            "these lines at the end of a long week feels like setting down a heavy suitcase.\n"
            "Tomorrow does not need to be perfect; it only needs a quiet, honest beginning and a cup of warm tea."
        ),
    },
    {
        "id": "en_04_child_storybook",
        "title": "Backyard Rocket Ship Adventure",
        "vibe": "grid graph paper, pencil, child handwriting, large bubbly playful print",
        "seed": 204,
        "text": (
            "Today we built a giant cardboard rocket ship in the backyard and painted bright silver stars all "
            "over the wings!\n"
            "My dog Barnaby wore a little paper helmet and sat right in the pilot seat while we counted down "
            "from ten to blast off to the moon before dinner."
        ),
    },
    {
        "id": "en_05_calligraphy_meditation",
        "title": "Meditation on Deep Roots",
        "vibe": "kraft brown paper, charcoal ink, calligraphy broad nib pen, careful bold confident",
        "seed": 205,
        "text": (
            "Strength is rarely found in noise or haste; it grows quietly in the patience of deep roots beneath "
            "winter soil.\n"
            "When the wind rises across the mountain ridge, the tallest pines bend without breaking because they "
            "trust the ground that holds them fast."
        ),
    },
    {
        "id": "en_06_architect_field_notes",
        "title": "Architectural Daylight Survey",
        "vibe": "grid paper, gray pencil, small neat upright print letters, careful",
        "seed": 206,
        "text": (
            "Site survey note 14B: Morning sunlight enters the eastern clerestory at a twenty-eight degree angle, "
            "casting clean geometric bands across the limestone floor.\n"
            "Recommend widening the timber louver spacing by twelve millimeters to balance natural daylight "
            "throughout the reading hall."
        ),
    },
    {
        "id": "en_07_grandmother_recipe",
        "title": "Grandmother's Sunday Cinnamon Apples",
        "vibe": "ivory paper, sepia brown ballpoint pen, grandma elderly slow warm handwriting",
        "seed": 207,
        "text": (
            "For Sunday cinnamon apples: slice four crisp orchard apples thinly and toss them gently with brown "
            "sugar, a pinch of nutmeg, and plenty of sweet butter.\n"
            "Bake slowly until the whole kitchen fills with warmth, and always serve it while still bubbling hot "
            "with vanilla cream."
        ),
    },
    {
        "id": "en_08_urgent_manifesto",
        "title": "Call to Bold Action",
        "vibe": "plain paper, crimson red marker pen, angry bold sharp fast slanted",
        "seed": 208,
        "text": (
            "Do not wait for permission to build what the future demands! Every great leap forward began when "
            "someone refused to accept broken habits as permanent laws.\n"
            "Stand firm, speak with unmistakable clarity, and turn conviction into action starting right here today!"
        ),
    },
    {
        "id": "en_09_physician_chart",
        "title": "Cardiology Outpatient Follow-Up Note",
        "vibe": "ruled paper, black ballpoint pen, doctor prescription, very fast messy cursive",
        "seed": 209,
        "text": (
            "Clinical follow-up assessment: patient reports marked improvement in exercise tolerance and complete "
            "resolution of evening dizziness after dosage adjustment.\n"
            "Continue current regimen, monitor morning blood pressure daily, and recheck electrolyte panel in "
            "four weeks."
        ),
    },
    {
        "id": "en_10_travel_postcard",
        "title": "Postcard from the Coastal Train",
        "vibe": "kraft paper, forest green gel pen, cute bubbly wide airy handwriting",
        "seed": 210,
        "text": (
            "Greetings from the coastal train! We spent all afternoon wandering through tiny bookshops and watching "
            "fishing boats bob in the harbor while eating warm lemon pastries.\n"
            "Wish you were here to see the sunset painting the cliffs in gold and violet!"
        ),
    },
]


def main() -> None:
    out_dir = Path("outputs/showcase")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = get_generator("model", "checkpoints/bilingual_5k_core")
    all_items = KOREAN_SHOWCASE + ENGLISH_SHOWCASE
    manifest = []

    print(f"Generating {len(all_items)} showcase images using generator='{gen.name}'...", flush=True)
    t_start = time.perf_counter()

    for idx, item in enumerate(all_items, 1):
        char_len = len(item["text"].replace("\n", ""))
        assert char_len >= 100, f"{item['id']} has only {char_len} chars (< 100)!"

        spec = interpret(item["vibe"])
        spec.seed = item["seed"]
        spec.layout.page = "B5"
        spec.layout.margin_mm = (18.0, 15.0, 18.0, 15.0)

        rng = np.random.default_rng(spec.seed)
        pages = write_document(item["text"], spec, gen, rng)
        assert len(pages) == 1, f"{item['id']} overflowed onto {len(pages)} pages!"
        img = render_page(pages[0], spec, dpi=200, rng=np.random.default_rng(spec.seed + 1000))

        out_path = out_dir / f"{item['id']}.png"
        save_images([img], out_path, dpi=200)

        entry = {
            "index": idx,
            "id": item["id"],
            "title": item["title"],
            "lang": "ko" if item["id"].startswith("ko_") else "en",
            "char_count": char_len,
            "vibe": item["vibe"],
            "matched_rules": spec.matched,
            "pen": spec.pen.type,
            "paper": spec.paper.type,
            "ink_rgb": [round(float(c), 3) for c in spec.ink.color],
            "file": str(out_path.resolve()),
        }
        manifest.append(entry)
        print(
            f"[{idx:02d}/20] {out_path.name} ({char_len} chars) | pen={spec.pen.type} paper={spec.paper.type} "
            f"| matched={spec.matched}",
            flush=True,
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - t_start
    print(f"=== All 20 showcase images saved to {out_dir.resolve()} in {elapsed:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
