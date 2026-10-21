"""AI Hub Korean Handwriting OCR dataset ingestion -> VibeFont dataset format.

Converts Korean handwriting image samples (letters and words) from AI Hub OCR dataset
into normalized online stroke trajectories (x, y, t, p, eos) with ground truth Hangul text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import zipfile

from ..dataset import DatasetWriter
from ..vision.tracer import trace_image_to_ink


def extract_text_from_json(data: dict[str, Any]) -> str:
    """Extracts ground truth Korean text from AI Hub OCR JSON schema."""
    if "info" in data and "text" in data["info"] and data["info"]["text"]:
        return data["info"]["text"].strip()
    if "text" in data:
        t_block = data["text"]
        if "letter" in t_block and "value" in t_block["letter"]:
            return t_block["letter"]["value"].strip()
        if "word" in t_block and "value" in t_block["word"]:
            return t_block["word"]["value"].strip()
    return ""


def ingest_korean_ocr(
    label_zip_path: str | Path,
    image_zip_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    hz: float = 100.0,
) -> int:
    """Ingests paired Korean handwriting images and JSON labels from ZIP archives."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = DatasetWriter(out)

    lbl_zip_p = Path(label_zip_path)
    img_zip_p = Path(image_zip_path)

    count = 0
    with zipfile.ZipFile(lbl_zip_p) as z_lbl, zipfile.ZipFile(img_zip_p) as z_img:
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
                ink = trace_image_to_ink(img_bytes, text=text, hz=hz)

                if len(ink) >= 4:
                    # Form writer and doc ID from path, e.g. '1.글자/138/...' -> writer='138'
                    parts = name.replace("\\", "/").split("/")
                    writer_id = parts[-2] if len(parts) >= 2 else "korean"
                    doc_id = Path(name).stem

                    w.add(
                        ink,
                        writer=writer_id,
                        doc=doc_id,
                        line_no=count,
                        source="korean_ocr",
                    )
                    count += 1
                    if limit is not None and count >= limit:
                        break
            except Exception:
                continue

    w.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest AI Hub Korean OCR handwriting into VibeFont LineStore")
    parser.add_argument("--labels", required=True, help="Path to [라벨]*.zip")
    parser.add_argument("--images", required=True, help="Path to [원천]*.zip")
    parser.add_argument("--out", default="data/processed/korean_ocr", help="Output directory")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum samples to ingest")
    args = parser.parse_args()

    n = ingest_korean_ocr(args.labels, args.images, args.out, limit=args.limit)
    print(f"Successfully ingested {n} Korean handwriting samples -> {args.out}")


if __name__ == "__main__":
    main()
