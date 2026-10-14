"""Tier-0 (offline, zero-dependency) vibe lexicon. Korean + English.

Each rule: keywords (substring match, lowercase), additive style deltas, absolute sets.
Deltas are scaled by intensity modifiers found in the same clause ("아주", "slightly", ...).
Extend freely - this is meant to be edited by hand and later distilled into a learned map.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rule:
    name: str
    keywords: tuple[str, ...]
    delta: dict[str, float] = field(default_factory=dict)  # style.* additive
    set: dict[str, dict] = field(default_factory=dict)  # {"pen": {...}, "paper": {...}, ...}


INTENSIFIERS = {
    "아주": 1.6, "매우": 1.6, "엄청": 1.8, "완전": 1.8, "너무": 1.6, "정말": 1.4, "많이": 1.4,
    "very": 1.6, "extremely": 1.9, "super": 1.7, "really": 1.4,
}
DIMINISHERS = {
    "약간": 0.5, "조금": 0.5, "살짝": 0.4, "다소": 0.6, "slightly": 0.5, "a bit": 0.5,
    "a little": 0.5, "somewhat": 0.6,
}
NEGATIONS = ("않", "안 ", "없", "not ", "no ", "n't")

COLORS: dict[tuple[str, ...], tuple[float, float, float]] = {
    ("검정", "검은", "까만", "black"): (0.06, 0.06, 0.07),
    ("파란", "파랑", "청색", "blue"): (0.08, 0.17, 0.55),
    ("남색", "navy", "블루블랙", "blue-black", "blue black"): (0.10, 0.12, 0.30),
    ("빨간", "빨강", "red"): (0.70, 0.08, 0.08),
    ("초록", "녹색", "green"): (0.08, 0.40, 0.18),
    ("보라", "purple", "violet"): (0.38, 0.14, 0.52),
    ("갈색", "세피아", "brown", "sepia"): (0.36, 0.22, 0.10),
    ("회색", "gray", "grey"): (0.35, 0.35, 0.37),
}

RULES: list[Rule] = [
    # ---------------------------------------------------------------- moods
    Rule("neat", ("단정", "깔끔", "정갈", "또박", "반듯", "neat", "tidy", "clean"),
         {"neatness": 0.3, "consistency": 0.25, "baseline_wobble": -0.2, "speed": -0.15}),
    Rule("messy", ("엉망", "지저분", "삐뚤", "휘갈", "악필", "messy", "sloppy", "scrawl"),
         {"neatness": -0.35, "consistency": -0.25, "baseline_wobble": 0.3, "speed": 0.2}),
    Rule("hurried", ("급하", "급히", "서둘", "바쁘", "빠르게", "메모", "hurried", "rushed", "quick", "fast"),
         {"speed": 0.35, "neatness": -0.2, "slant_deg": 6, "connectedness": 0.2, "width_scale": 0.1}),
    Rule("careful", ("천천히", "정성", "공들", "꼼꼼", "careful", "slow", "deliberate"),
         {"speed": -0.3, "neatness": 0.25, "pressure": 0.1}),
    Rule("elegant", ("우아", "고급", "세련", "품격", "elegant", "graceful", "classy", "fancy"),
         {"slant_deg": 8, "connectedness": 0.3, "neatness": 0.2, "roundness": 0.15, "width_scale": -0.08}),
    Rule("cute", ("귀여", "깜찍", "아기자기", "동글", "cute", "bubbly", "kawaii"),
         {"roundness": 0.35, "slant_deg": -6, "connectedness": -0.3, "size_mm": 0.6, "width_scale": 0.12}),
    Rule("child", ("어린이", "아이처럼", "아이 글씨", "아이가", "초등", "꼬마", "child", "kid"),
         {"size_mm": 1.5, "neatness": -0.25, "connectedness": -0.45, "baseline_wobble": 0.35,
          "slant_deg": -4, "pressure": 0.2, "consistency": -0.2}),
    Rule("doctor", ("의사", "처방", "doctor", "prescription"),
         {"neatness": -0.5, "speed": 0.45, "connectedness": 0.4, "slant_deg": 10, "width_scale": 0.15}),
    Rule("bold_confident", ("힘있", "힘찬", "자신감", "당당", "굵", "bold", "confident", "strong"),
         {"pressure": 0.3, "size_mm": 0.5, "speed": 0.1, "consistency": 0.1}),
    Rule("soft", ("부드러", "잔잔", "차분", "따뜻", "soft", "gentle", "calm", "warm"),
         {"roundness": 0.25, "pressure": -0.15, "speed": -0.1}),
    Rule("sharp", ("날카", "각진", "딱딱", "sharp", "angular", "spiky"),
         {"roundness": -0.35, "pressure": 0.1}),
    Rule("small", ("작은", "작게", "깨알", "small", "tiny"), {"size_mm": -0.8}),
    Rule("large", ("큰 글씨", "크게", "큼직", "large", "big"), {"size_mm": 1.2}),
    Rule("narrow", ("좁은", "홀쭉", "narrow", "condensed"), {"width_scale": -0.2, "letter_spacing": -0.1}),
    Rule("wide", ("넓은", "넉넉", "wide", "airy"), {"width_scale": 0.2, "letter_spacing": 0.15, "word_spacing": 0.2}),
    Rule("upright", ("똑바로", "세로로", "upright", "vertical"), {"slant_deg": -8}),
    Rule("slanted", ("기울", "이탤릭", "slanted", "italic"), {"slant_deg": 10}),
    Rule("cursive", ("필기체", "흘려", "이어", "cursive", "joined", "script"), {"connectedness": 0.4}),
    Rule("print", ("정자", "인쇄체", "또박또박", "print", "block letters"), {"connectedness": -0.4, "neatness": 0.15}),
    Rule("tired", ("피곤", "졸린", "지친", "tired", "sleepy", "exhausted"),
         {"fatigue": 0.5, "neatness": -0.2, "baseline_wobble": 0.2, "pressure": -0.15, "slant_deg": 3}),
    Rule("angry", ("화난", "분노", "격앙", "angry", "furious"),
         {"pressure": 0.4, "speed": 0.3, "roundness": -0.25, "neatness": -0.2, "slant_deg": 8}),
    Rule("old", ("할머니", "할아버지", "어르신", "노인", "elderly", "grandma", "grandpa"),
         {"baseline_wobble": 0.3, "speed": -0.3, "connectedness": 0.2, "neatness": 0.0, "size_mm": 0.4}),
    Rule("formal_letter", ("편지", "연서", "손편지", "letter", "love letter"),
         {"neatness": 0.15, "connectedness": 0.15},
         {"paper": {"type": "ivory"}}),
    Rule("diary", ("일기", "다이어리", "노트", "diary", "journal", "notebook"),
         {}, {"paper": {"type": "ruled"}}),
    Rule("exam", ("시험", "답안", "exam", "answer sheet"), {"speed": 0.2}, {"paper": {"type": "ruled"}}),
    # ----------------------------------------------------------------- pens
    Rule("pen_fountain", ("만년필", "fountain"), {"connectedness": 0.1},
         {"pen": {"type": "fountain", "width_mm": 0.55}}),
    Rule("pen_ballpoint", ("볼펜", "ballpoint", "biro"), {}, {"pen": {"type": "ballpoint", "width_mm": 0.45}}),
    Rule("pen_gel", ("젤펜", "중성펜", "gel pen", "gel"), {}, {"pen": {"type": "gel", "width_mm": 0.5}}),
    Rule("pen_pencil", ("연필", "샤프", "pencil"), {}, {"pen": {"type": "pencil", "width_mm": 0.5},
                                                         "ink": {"color": [0.22, 0.22, 0.24]}}),
    Rule("pen_marker", ("마커", "사인펜", "네임펜", "marker", "felt"), {"size_mm": 0.8},
         {"pen": {"type": "marker", "width_mm": 1.2}}),
    Rule("pen_brush", ("붓", "brush"), {"size_mm": 2.0, "pressure": 0.2}, {"pen": {"type": "brush", "width_mm": 2.0}}),
    Rule("pen_calligraphy", ("캘리", "calligraphy", "broad nib", "평촉"), {"neatness": 0.2, "size_mm": 1.5},
         {"pen": {"type": "broad_nib", "width_mm": 1.6, "nib_angle_deg": 35}}),
    # ---------------------------------------------------------------- paper
    Rule("paper_ruled", ("줄노트", "줄 노트", "줄공책", "ruled", "lined"), {}, {"paper": {"type": "ruled"}}),
    Rule("paper_grid", ("모눈", "방안", "grid", "graph paper"), {}, {"paper": {"type": "grid", "line_spacing_mm": 5.0}}),
    Rule("paper_kraft", ("크라프트", "갱지", "kraft", "brown paper"), {}, {"paper": {"type": "kraft"}}),
    Rule("paper_ivory", ("미색", "아이보리", "ivory", "cream"), {}, {"paper": {"type": "ivory"}}),
]
