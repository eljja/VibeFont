"""Semantic Emotion & Morphology Analysis Engine for Calligraphy and Expressive Handwriting.

In human calligraphy, letter anatomy, scale, weight, and stroke rhythm are deeply
shaped by the meaning and emotional resonance of the words (단어의 뜻을 살린 조형미).
This engine parses Korean and English text to assign semantic archetypes and
morphological directives to guide calligraphic stroke sculpting and layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import numpy as np


class SemanticArchetype(str, Enum):
    BOLD_MASSIVE = "bold_massive"          # 바위, 절벽, 대지, 거목 / rock, oak, deep, mountain
    SHARP_FIERCE = "sharp_fierce"          # 눈보라, 서리, 번개, 폭풍 / storm, frost, blade, fierce
    FLOWING_AIRY = "flowing_airy"          # 파도, 바람, 안개, 물결, 침묵 / wave, wind, mist, silence
    GENTLE_WARM = "gentle_warm"            # 봄, 사랑, 꽃, 온기, 미소 / spring, love, gentle, bloom
    SOARING_LUMINOUS = "soaring_luminous"  # 별, 하늘, 빛, 새벽, 꿈 / star, light, sky, dawn, soar
    NEUTRAL = "neutral"


# Korean keyword roots for each archetype
_KO_LEXICON = {
    SemanticArchetype.BOLD_MASSIVE: {
        "바위", "절벽", "거대", "웅장", "대지", "거목", "단단", "굳건", "무겁", "뿌리",
        "성벽", "기둥", "침묵", "태산", "쇠", "돌", "벽", "깊은", "깊이", "묵직",
    },
    SemanticArchetype.SHARP_FIERCE: {
        "눈보라", "폭풍", "서리", "칼", "번개", "날카", "매섭", "사자", "갈기", "맹렬",
        "불꽃", "겨울", "추위", "얼음", "벼락", "칼날", "가시", "싸늘", "거친", "돌풍",
    },
    SemanticArchetype.FLOWING_AIRY: {
        "파도", "바람", "강", "물", "바다", "흐름", "물결", "안개", "구름", "잔물결",
        "향기", "시내", "호수", "여울", "물방울", "숨", "공기", "자유", "흘러", "날아",
    },
    SemanticArchetype.GENTLE_WARM: {
        "봄", "사랑", "미소", "꽃", "온기", "포근", "햇살", "다정", "연인", "달콤",
        "아이", "매화", "꽃잎", "따뜻", "봄날", "품", "어머니", "심장", "고운", "부드럽",
    },
    SemanticArchetype.SOARING_LUMINOUS: {
        "별", "하늘", "빛", "꿈", "비상", "새벽", "은하", "높이", "희망", "영원",
        "천상", "태양", "아침", "달", "달빛", "등대", "솟아", "푸른", "눈부신", "환한",
    },
}

# English keyword roots for each archetype
_EN_LEXICON = {
    SemanticArchetype.BOLD_MASSIVE: {
        "oak", "rock", "stone", "cliff", "ground", "mountain", "earth", "massive",
        "fortress", "pillar", "iron", "deep", "heavy", "mighty", "hold", "root",
    },
    SemanticArchetype.SHARP_FIERCE: {
        "storm", "winter", "frost", "blizzard", "sharp", "fierce", "lightning", "thunder",
        "rage", "blade", "tempest", "sword", "claw", "cold", "wild", "frozen",
    },
    SemanticArchetype.FLOWING_AIRY: {
        "wave", "wind", "water", "sea", "river", "mist", "stream", "cloud",
        "breeze", "ocean", "flow", "flight", "breath", "silence", "silent", "whisper",
    },
    SemanticArchetype.GENTLE_WARM: {
        "spring", "love", "gentle", "warm", "bloom", "soft", "tender", "smile",
        "blossom", "flower", "sweet", "heart", "sunshine", "golden", "peace", "grace",
    },
    SemanticArchetype.SOARING_LUMINOUS: {
        "star", "stars", "light", "sky", "dream", "dawn", "sun", "soar",
        "high", "fly", "hope", "heaven", "eternal", "radiance", "spark", "shine",
    },
}

_KO_PARTICLES = (
    "에서도", "에게는", "으로의", "이라는", "이라도", "처럼", "부터", "까지", "마다",
    "에서", "에게", "으로", "에는", "에도", "이라", "보다", "이라서", "면서", "한테",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "로", "와", "과", "만",
)

_EN_FUNCTION_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "of", "for", "by", "with", "from",
    "into", "upon", "over", "and", "but", "or", "nor", "is", "are", "was", "were",
    "be", "been", "that", "this", "its", "our", "your", "my", "not", "only", "as",
}


@dataclass
class WordMorphology:
    word: str
    archetype: SemanticArchetype
    scale: float = 1.0
    weight_boost: float = 0.0
    angularity: float = 0.5
    slant_drift_deg: float = 0.0
    swash_reach: float = 1.0
    is_hero: bool = False
    is_particle: bool = False


def _clean_token(w: str) -> str:
    return re.sub(r"[^\w\s가-힣]", "", w).strip().lower()


def classify_word_archetype(word: str) -> SemanticArchetype:
    clean = _clean_token(word)
    if not clean:
        return SemanticArchetype.NEUTRAL

    # 1. Exact or prefix match against Korean lexicon
    for arch, words in _KO_LEXICON.items():
        for kw in words:
            if clean.startswith(kw) or kw in clean:
                return arch

    # 2. English match
    for arch, words in _EN_LEXICON.items():
        if clean in words:
            return arch
        # Stem match (e.g. storms -> storm)
        for kw in words:
            if clean.startswith(kw) and len(clean) <= len(kw) + 2:
                return arch

    return SemanticArchetype.NEUTRAL


def _split_ko_token(token: str) -> tuple[str, str]:
    clean = token.rstrip(".,!?;:")
    punct = token[len(clean):]
    if len(clean) <= 1:
        return token, ""
    for part in _KO_PARTICLES:
        if clean.endswith(part) and len(clean) - len(part) >= 1:
            stem = clean[: len(clean) - len(part)]
            return stem, part + punct
    return token, ""


def analyze_semantics(text: str) -> list[list[WordMorphology]]:
    """Analyzes text line-by-line and returns structured calligraphic morphology for every word."""
    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    result: list[list[WordMorphology]] = []

    for line in raw_lines:
        tokens = line.split()
        if not tokens:
            continue

        line_morphs: list[WordMorphology] = []
        scores: list[float] = []

        for token in tokens:
            stem, particle = _split_ko_token(token)
            clean_stem = _clean_token(stem)
            clean_token = _clean_token(token)

            is_func = (clean_token in _EN_FUNCTION_WORDS)
            arch = classify_word_archetype(stem if stem else token)

            # Score hero importance: resonance words get strong boost
            score = 0.0
            if is_func:
                score = 0.1
            elif arch != SemanticArchetype.NEUTRAL:
                score = 5.0 + len(clean_stem)
            else:
                score = 1.0 + len(clean_stem)
            scores.append(score)

        hero_idx = int(np.argmax(scores)) if scores else 0

        for idx, token in enumerate(tokens):
            stem, particle = _split_ko_token(token)
            clean_token = _clean_token(token)
            is_hero = (idx == hero_idx) and scores[idx] > 0.5
            is_func = (clean_token in _EN_FUNCTION_WORDS)

            if particle:
                # Add stem as hero/content
                arch = classify_word_archetype(stem)
                m_stem = _create_morphology(stem, arch, is_hero=is_hero, is_particle=False)
                line_morphs.append(m_stem)
                # Add particle as secondary compact
                m_part = _create_morphology(particle, SemanticArchetype.NEUTRAL, is_hero=False, is_particle=True)
                line_morphs.append(m_part)
            else:
                arch = classify_word_archetype(token)
                m = _create_morphology(token, arch, is_hero=is_hero, is_particle=is_func)
                line_morphs.append(m)

        result.append(line_morphs)

    return result


def _create_morphology(
    word: str,
    arch: SemanticArchetype,
    is_hero: bool = False,
    is_particle: bool = False,
) -> WordMorphology:
    if is_particle:
        return WordMorphology(
            word=word,
            archetype=SemanticArchetype.NEUTRAL,
            scale=0.72,
            weight_boost=-0.20,
            angularity=0.30,
            slant_drift_deg=0.0,
            swash_reach=0.40,
            is_hero=False,
            is_particle=True,
        )

    # Base scale & weight by archetype
    if arch == SemanticArchetype.BOLD_MASSIVE:
        base_sc = 1.68 if is_hero else 1.32
        w_boost = 0.36 if is_hero else 0.22
        ang = 0.80
        slant = 0.0
        swash = 0.70
    elif arch == SemanticArchetype.SHARP_FIERCE:
        base_sc = 1.58 if is_hero else 1.28
        w_boost = 0.18 if is_hero else 0.08
        ang = 0.95
        slant = 4.5
        swash = 1.65
    elif arch == SemanticArchetype.FLOWING_AIRY:
        base_sc = 1.52 if is_hero else 1.22
        w_boost = -0.06 if is_hero else -0.10
        ang = 0.12
        slant = 2.0
        swash = 1.70
    elif arch == SemanticArchetype.GENTLE_WARM:
        base_sc = 1.48 if is_hero else 1.20
        w_boost = 0.08 if is_hero else 0.02
        ang = 0.08
        slant = -1.5
        swash = 1.25
    elif arch == SemanticArchetype.SOARING_LUMINOUS:
        base_sc = 1.62 if is_hero else 1.26
        w_boost = 0.14 if is_hero else 0.04
        ang = 0.55
        slant = 3.2
        swash = 1.50
    else:  # NEUTRAL
        base_sc = 1.32 if is_hero else 1.08
        w_boost = 0.06 if is_hero else 0.0
        ang = 0.50
        slant = 0.0
        swash = 1.0

    return WordMorphology(
        word=word,
        archetype=arch,
        scale=float(np.clip(base_sc, 0.65, 2.20)),
        weight_boost=float(np.clip(w_boost, -0.25, 0.45)),
        angularity=float(np.clip(ang, 0.0, 1.0)),
        slant_drift_deg=float(np.clip(slant, -8.0, 8.0)),
        swash_reach=float(np.clip(swash, 0.20, 2.00)),
        is_hero=is_hero,
        is_particle=False,
    )
