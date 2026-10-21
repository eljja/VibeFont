"""Build 5,000-sample clean bilingual dataset:
  - 1,500 IAM English real traced words
  - 1,500 AI Hub Korean OCR real traced words (750 multi-syllable words + 750 syllables)
  - 2,000 Canonical online bilingual stroke sequences (1,000 English + 1,000 Korean across 20 writer styles)
"""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
import time
import zipfile

import numpy as np

from vibefont.dataset import DatasetWriter
from vibefont.data.iam_words import parse_words_txt
from vibefont.data.korean_ocr import extract_text_from_json
from vibefont.document.drift import DocumentDrift
from vibefont.generate.base import LineRequest
from vibefont.generate.procedural import ProceduralGenerator
from vibefont.vibe.spec import StyleSpec
from vibefont.vision.tracer import trace_image_to_ink


EN_WORDS = [
    "Hello", "VibeFont!", "handwriting", "natural", "bilingual", "synthesis", "every", "letter",
    "unique", "flowing", "rhythm", "fountain", "notebook", "diary", "morning", "coffee",
    "thoughtful", "creative", "journey", "whisper", "sunlight", "library", "memories", "promise",
    "gentle", "breeze", "sketchbook", "harmony", "graceful", "authentic", "character", "expression",
    "paper", "inkwell", "signature", "story", "dreamer", "horizon", "tranquil", "melody",
]

KO_WORDS = [
    "자연스러운", "한글과", "영어", "필기체.", "모든", "글자가", "다르게", "작성됩니다.",
    "따뜻한", "봄날의", "기억을", "종이", "위에", "천천히", "적어봅니다.", "아름다운",
    "손글씨", "감성적인", "다이어리", "오늘의", "기록", "바람이", "불어오는", "창가에서",
    "차분한", "마음으로", "편지를", "씁니다.", "행복한", "순간들", "소중한", "이야기",
    "은은한", "만년필", "잉크의", "번짐과", "부드러운", "곡선", "리듬감", "있는",
]


def main():
    out_dir = Path("data/processed/bilingual_5k")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(out_dir)

    # 1. Ingest 1,500 IAM English words
    print("=== 1. Ingesting English IAM Words (target: 1,500) ===", flush=True)
    t0 = time.perf_counter()
    iam_zip = Path("words.tgz.zip")
    words_txt = Path("data/raw/iam_words/words.txt")
    meta = parse_words_txt(words_txt)

    count_en = 0
    with zipfile.ZipFile(iam_zip) as zf:
        tgz_name = [n for n in zf.namelist() if n.endswith(".tgz") or n.endswith(".tar.gz")][0]
        with zf.open(tgz_name) as tf_stream:
            with tarfile.open(fileobj=tf_stream, mode="r|gz") as tf:
                for member in tf:
                    if not member.name.endswith(".png") or member.isdir():
                        continue
                    word_id = Path(member.name).stem
                    if word_id not in meta:
                        continue
                    word_info = meta[word_id]
                    img_data = tf.extractfile(member).read()
                    try:
                        ink = trace_image_to_ink(img_data, text=word_info["text"], hz=100.0)
                        xmin, ymin, xmax, ymax = ink.bbox()
                        if len(ink) >= 4 and ymin >= -12.0 and ymax <= 12.0 and (xmax - xmin) <= 120.0:
                            writer.add(
                                ink,
                                writer=f"iam_{word_info['writer']}",
                                doc=word_info["doc"],
                                line_no=count_en,
                                source="iam_words",
                            )
                            count_en += 1
                            if count_en >= 1500:
                                break
                    except Exception:
                        continue
    t_en = time.perf_counter() - t0
    print(f"Ingested {count_en} English words in {t_en:.1f}s ({count_en / max(t_en, 0.01):.1f} words/s)", flush=True)

    # 2. Ingest 1,500 Korean handwriting samples (750 words + 750 syllables)
    print("=== 2. Ingesting Korean Handwriting Samples (target: 1,500: 750 words + 750 chars) ===", flush=True)
    t1 = time.perf_counter()
    korean_root = list(Path(".").glob("*한글*"))[0]
    val_dir = korean_root / "Validation"

    lbl_zip = None
    img_zip = None
    for p in val_dir.glob("*.zip"):
        p_name = p.name
        if "라벨" in p_name and "필기체" in p_name:
            lbl_zip = p
        elif "원천" in p_name and "필기체" in p_name:
            img_zip = p

    if not lbl_zip or not img_zip:
        raise FileNotFoundError(f"Could not locate Korean validation handwriting zip files in {val_dir}")

    count_ko_words = 0
    count_ko_chars = 0
    with zipfile.ZipFile(lbl_zip) as z_lbl, zipfile.ZipFile(img_zip) as z_img:
        img_names_set = set(z_img.namelist())
        json_names = [n for n in z_lbl.namelist() if n.endswith(".json") and not z_lbl.getinfo(n).is_dir()]
        # Stride through the entire archive so we hit both 1.글자 (first half) and 2.단어 (second half)
        stride = max(1, len(json_names) // 4000)
        strided_names = json_names[::stride] + json_names[1::stride]

        for name in strided_names:
            if count_ko_words >= 750 and count_ko_chars >= 750:
                break
            target_jpg = name[:-5] + ".jpg"
            if target_jpg not in img_names_set:
                continue
            try:
                lbl_bytes = z_lbl.read(name)
                data = json.loads(lbl_bytes.decode("utf-8"))
                text = extract_text_from_json(data)
                if not text:
                    continue
                is_word = len(text.strip()) >= 2
                if is_word and count_ko_words >= 750:
                    continue
                if not is_word and count_ko_chars >= 750:
                    continue

                img_bytes = z_img.read(target_jpg)
                ink = trace_image_to_ink(img_bytes, text=text, hz=100.0)
                xmin, ymin, xmax, ymax = ink.bbox()
                if len(ink) >= 4 and ymin >= -12.0 and ymax <= 12.0 and (xmax - xmin) <= 120.0:
                    parts = name.replace("\\", "/").split("/")
                    writer_id = parts[-2] if len(parts) >= 2 else "korean"
                    writer.add(
                        ink,
                        writer=f"ko_{writer_id}",
                        doc=Path(name).stem,
                        line_no=count_ko_words + count_ko_chars,
                        source="korean_ocr_word" if is_word else "korean_ocr_char",
                    )
                    if is_word:
                        count_ko_words += 1
                    else:
                        count_ko_chars += 1
            except Exception:
                continue

    t_ko = time.perf_counter() - t1
    count_ko = count_ko_words + count_ko_chars
    print(f"Ingested {count_ko} Korean samples ({count_ko_words} words, {count_ko_chars} chars) in {t_ko:.1f}s", flush=True)

    # 3. Generate 2,000 canonical online bilingual words across 20 writer styles
    print("=== 3. Generating 2,000 Canonical Online Bilingual Word Trajectories ===", flush=True)
    t2 = time.perf_counter()
    proc_gen = ProceduralGenerator()
    rng = np.random.default_rng(2026)

    count_online = 0
    for i in range(2000):
        writer_idx = i % 20
        slant = float(np.linspace(-2.0, 16.0, 20)[writer_idx])
        conn = float(np.linspace(0.15, 0.85, 20)[writer_idx])
        roundness = float(np.linspace(0.35, 0.90, 20)[(writer_idx * 7) % 20])
        speed = float(np.linspace(0.4, 0.85, 20)[(writer_idx * 3) % 20])
        neatness = float(np.linspace(0.55, 0.92, 20)[(writer_idx * 11) % 20])

        style = StyleSpec(
            slant_deg=slant,
            connectedness=conn,
            roundness=roundness,
            speed=speed,
            neatness=neatness,
            baseline_wobble=0.18 * (1.1 - neatness),
        )
        drift = DocumentDrift(style, rng)
        state = drift.next_line()

        word = EN_WORDS[i % len(EN_WORDS)] if i < 1000 else KO_WORDS[i % len(KO_WORDS)]
        req = LineRequest(text=word, style=style, state=state, rng=rng, context=[])
        ink = proc_gen.generate_line(req)
        if len(ink) >= 4:
            writer.add(
                ink,
                writer=f"online_w{writer_idx:02d}",
                doc=f"online_doc_{writer_idx:02d}",
                line_no=i // 20,
                source="canonical_online",
            )
            count_online += 1

    t_on = time.perf_counter() - t2
    print(f"Generated {count_online} canonical online samples in {t_on:.1f}s", flush=True)

    writer.close()
    total = count_en + count_ko + count_online
    print(f"=== Successfully saved {total} clean bilingual samples to {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
