"""Build 5,000-sample balanced bilingual dataset (2,500 IAM English + 2,500 AI Hub Korean handwriting)."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
import time
import zipfile

from vibefont.dataset import DatasetWriter
from vibefont.data.iam_words import parse_words_txt
from vibefont.data.korean_ocr import extract_text_from_json
from vibefont.vision.tracer import trace_image_to_ink


def main():
    out_dir = Path("data/processed/bilingual_5k")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(out_dir)

    # 1. Ingest 2,500 IAM English words
    print("=== Ingesting English IAM Words (target: 2,500) ===")
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
                        if len(ink) >= 4:
                            writer.add(
                                ink,
                                writer=f"iam_{word_info['writer']}",
                                doc=word_info["doc"],
                                line_no=count_en,
                                source="iam_words",
                            )
                            count_en += 1
                            if count_en >= 2500:
                                break
                    except Exception:
                        continue
    t_en = time.perf_counter() - t0
    print(f"Ingested {count_en} English words in {t_en:.1f}s ({count_en / max(t_en, 0.01):.1f} words/s)")

    # 2. Ingest 2,500 Korean handwriting samples
    print("=== Ingesting Korean Handwriting Samples (target: 2,500) ===")
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

    count_ko = 0
    with zipfile.ZipFile(lbl_zip) as z_lbl, zipfile.ZipFile(img_zip) as z_img:
        img_names_set = set(z_img.namelist())
        for name in z_lbl.namelist():
            if not name.endswith(".json") or z_lbl.getinfo(name).is_dir():
                continue
            target_jpg = name[:-5] + ".jpg"
            if target_jpg not in img_names_set:
                continue
            try:
                lbl_bytes = z_lbl.read(name)
                data = json.loads(lbl_bytes.decode("utf-8"))
                text = extract_text_from_json(data)
                if not text:
                    continue
                img_bytes = z_img.read(target_jpg)
                ink = trace_image_to_ink(img_bytes, text=text, hz=100.0)
                if len(ink) >= 4:
                    parts = name.replace("\\", "/").split("/")
                    writer_id = parts[-2] if len(parts) >= 2 else "korean"
                    writer.add(
                        ink,
                        writer=f"ko_{writer_id}",
                        doc=Path(name).stem,
                        line_no=count_ko,
                        source="korean_ocr",
                    )
                    count_ko += 1
                    if count_ko >= 2500:
                        break
            except Exception:
                continue

    t_ko = time.perf_counter() - t1
    print(f"Ingested {count_ko} Korean handwriting samples in {t_ko:.1f}s ({count_ko / max(t_ko, 0.01):.1f} samples/s)")

    writer.close()
    print(f"=== Successfully saved {count_en + count_ko} bilingual samples to {out_dir} ===")


if __name__ == "__main__":
    main()
