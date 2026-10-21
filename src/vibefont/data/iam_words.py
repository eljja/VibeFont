"""IAM Offline Word dataset ingestion -> VibeFont dataset format.

Traces offline handwritten word PNGs from IAM into online stroke trajectories (x, y, t, p, eos),
aligns with words.txt ground truth transcriptions, and saves to LineStore.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
from typing import Any
import zipfile

from ..dataset import DatasetWriter
from ..vision.tracer import trace_image_to_ink


def parse_words_txt(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parses IAM words.txt into a dict keyed by word_id."""
    meta: dict[str, dict[str, Any]] = {}
    path = Path(path)
    if not path.exists():
        return meta

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ", 8)
        if len(parts) < 9:
            continue
        word_id, status = parts[0], parts[1]
        if status != "ok":
            continue
        transcription = parts[8]
        # Form ID is e.g. a01-000u from a01-000u-00-00
        id_parts = word_id.split("-")
        form_id = f"{id_parts[0]}-{id_parts[1]}" if len(id_parts) >= 2 else word_id
        writer_id = id_parts[0]

        meta[word_id] = {
            "status": status,
            "text": transcription,
            "writer": writer_id,
            "doc": form_id,
        }
    return meta


def ingest_iam_words(
    zip_path: str | Path,
    words_txt_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    hz: float = 100.0,
) -> int:
    """Ingests word images from words.tgz (inside words.tgz.zip) into LineStore."""
    meta = parse_words_txt(words_txt_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = DatasetWriter(out)

    zip_p = Path(zip_path)
    count = 0

    with zipfile.ZipFile(zip_p) as zf:
        # Find words.tgz inside the zip
        tgz_name = [n for n in zf.namelist() if n.endswith(".tgz") or n.endswith(".tar.gz")]
        if not tgz_name:
            raise FileNotFoundError("No .tgz archive found inside zip")

        with zf.open(tgz_name[0]) as tf_stream:
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
                        ink = trace_image_to_ink(img_data, text=word_info["text"], hz=hz)
                        if len(ink) >= 4:
                            w.add(
                                ink,
                                writer=word_info["writer"],
                                doc=word_info["doc"],
                                line_no=count,
                                source="iam_words",
                            )
                            count += 1
                            if limit is not None and count >= limit:
                                break
                    except Exception as e:
                        continue

    w.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest IAM offline words into VibeFont LineStore")
    parser.add_argument("--zip", default="words.tgz.zip", help="Path to words.tgz.zip")
    parser.add_argument("--words-txt", default="data/raw/iam_words/words.txt", help="Path to words.txt")
    parser.add_argument("--out", default="data/processed/iam_words", help="Output directory")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum words to ingest")
    args = parser.parse_args()

    n = ingest_iam_words(args.zip, args.words_txt, args.out, limit=args.limit)
    print(f"Successfully ingested {n} words -> {args.out}")


if __name__ == "__main__":
    main()
