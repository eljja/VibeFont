"""IAM-OnDB ingestion -> VibeFont dataset format.

Expected layout after extracting the official archives into ``data/raw/iamondb``:
    lineStrokes/a01/a01-000/a01-000u-01.xml    (lineStrokes-all.tar.gz)
    ascii/a01/a01-000/a01-000u.txt             (ascii-all.tar.gz)
    original/a01/a01-000/strokesu.xml          (original-xml-all.tar.gz, writer ids) [optional]

!! Written from the published format description and NOT yet run on the real data.
!! First task when the data arrives: run ``python -m vibefont.data.iamondb --check`` and fix.

Line XML:  <StrokeSet><Stroke ...><Point x=".." y=".." time=".."/>...
ASCII txt: header ..., then "CSR:" followed by one transcription line per lineStrokes file.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..dataset import DatasetWriter
from ..ink import Ink
from ..preprocess import normalize_line, resample_time


def parse_line_xml(path: Path) -> list[np.ndarray]:
    root = ET.parse(path).getroot()
    strokes = []
    for st in root.iter("Stroke"):
        pts = [(float(p.get("x")), float(p.get("y")), float(p.get("time"))) for p in st.iter("Point")]
        if pts:
            strokes.append(np.asarray(pts, np.float64))
    return strokes


def parse_ascii(path: Path) -> list[str]:
    txt = path.read_text(encoding="latin-1")
    if "CSR:" not in txt:
        return []
    body = txt.split("CSR:", 1)[1]
    return [ln.strip() for ln in body.strip().splitlines() if ln.strip()]


def writer_id(original_dir: Path, form: str) -> str:
    """Best effort: look for a writerID attribute in original/<a01>/<a01-000>/strokes*.xml."""
    sub = original_dir / form[:3] / form[:7]
    for f in sorted(sub.glob("strokes*.xml")) if sub.exists() else []:
        m = re.search(r'writerID="([^"]+)"', f.read_text(encoding="latin-1", errors="ignore"))
        if m:
            return m.group(1)
    return "unknown"


def iter_lines(root: Path):
    """Yield (ink_raw, writer, doc, line_no)."""
    ls_root, asc_root, orig = root / "lineStrokes", root / "ascii", root / "original"
    for txt in sorted(asc_root.rglob("*.txt")):
        form = txt.stem  # e.g. a01-000u
        lines = parse_ascii(txt)
        xml_dir = ls_root / form[:3] / form[:7]
        xmls = sorted(xml_dir.glob(f"{form}-*.xml")) if xml_dir.exists() else []
        if not xmls:
            continue
        wid = writer_id(orig, form) if orig.exists() else "unknown"
        for k, xml in enumerate(xmls):
            if k >= len(lines):
                break
            strokes = parse_line_xml(xml)
            if strokes:
                yield Ink.from_strokes(strokes, text=lines[k]), wid, form, k


def build(root: Path, out: Path, hz: float = 100.0, limit: int | None = None) -> int:
    w = DatasetWriter(out)
    n = 0
    for ink, wid, doc, k in iter_lines(root):
        ink = resample_time(ink, hz)
        ink, _ = normalize_line(ink)
        w.add(ink, wid, doc, k, "iamondb")
        n += 1
        if limit and n >= limit:
            break
    w.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/iamondb")
    ap.add_argument("--out", default="data/processed/iamondb")
    ap.add_argument("--check", action="store_true", help="parse 20 lines, print stats, write nothing big")
    a = ap.parse_args()
    root = Path(a.root)
    if a.check:
        for i, (ink, wid, doc, k) in enumerate(iter_lines(root)):
            x0, y0, x1, y1 = ink.bbox()
            dt = np.diff(ink.points[:, 2])
            print(f"{doc}#{k} writer={wid} pts={len(ink)} strokes={len(ink.stroke_slices())} "
                  f"bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) median_dt={np.median(dt):.4f}s text={ink.text!r}")
            if i >= 19:
                break
        return
    print("lines:", build(root, Path(a.out)))


if __name__ == "__main__":
    main()
