"""Fixed (no fitting needed) character vocabulary for English + Korean.

Korean syllables are decomposed with Unicode NFD into position-specific conjoining jamo
(choseong U+1100.., jungseong U+1161.., jongseong U+11A8..). 11,172 syllables therefore
share ~70 symbols, and since every syllable starts with a choseong, syllable boundaries
stay implicit. This is what makes Korean learnable from a few hours of collected data.
"""

from __future__ import annotations

import unicodedata

PAD, BOS, EOS, UNK = 0, 1, 2, 3
_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
_ASCII = [chr(c) for c in range(32, 127)]
_EXTRA = list("‘’“”…·–—°€£¥")
_JAMO = [chr(c) for c in range(0x1100, 0x1200)]

VOCAB: list[str] = _SPECIALS + _ASCII + _EXTRA + _JAMO
_INDEX = {ch: i for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def encode(text: str, add_bos_eos: bool = True) -> list[int]:
    ids = [_INDEX.get(ch, UNK) for ch in normalize(text)]
    return [BOS] + ids + [EOS] if add_bos_eos else ids


def decode(ids: list[int]) -> str:
    s = "".join(VOCAB[i] for i in ids if i >= len(_SPECIALS))
    return unicodedata.normalize("NFC", s)
