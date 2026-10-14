"""Web Collector dataset ingestion -> VibeFont dataset format.

Converts JSON files exported by ``web/collector/index.html`` (format: vibefont-collector/1)
into the unified LineStore format with full stylus pressure, timestamp resampling,
and baseline/x-height normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import DatasetWriter
from ..ink import EOS, NCOLS, P, T, X, Y, Ink
from ..preprocess import normalize_line, resample_time


def parse_collector_stroke(stroke_dict: dict[str, Any]) -> np.ndarray | None:
    """Parses a single stroke dict into (N, 5) float32 array: [x, y, t, p, eos]."""
    xs = stroke_dict.get("x", [])
    ys = stroke_dict.get("y", [])
    ts = stroke_dict.get("t", [])
    ps = stroke_dict.get("p", [])

    n = len(xs)
    if n == 0 or len(ys) != n:
        return None

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    t = np.asarray(ts, dtype=np.float64) if len(ts) == n else np.arange(n, dtype=np.float64) * 0.01
    p = np.asarray(ps, dtype=np.float64) if len(ps) == n else np.full(n, 0.5, dtype=np.float64)

    eos = np.zeros(n, dtype=np.float64)
    eos[-1] = 1.0

    pts = np.empty((n, NCOLS), dtype=np.float64)
    pts[:, X] = x
    pts[:, Y] = y
    pts[:, T] = t
    pts[:, P] = p
    pts[:, EOS] = eos
    return pts


def parse_collector_item(item: dict[str, Any], hz: float = 100.0) -> Ink | None:
    """Parses a single item from the collector export into a normalized Ink object."""
    strokes_data = item.get("strokes", [])
    if not strokes_data:
        return None

    stroke_pts = []
    for s in strokes_data:
        pts = parse_collector_stroke(s)
        if pts is not None and len(pts) > 0:
            stroke_pts.append(pts)

    if not stroke_pts:
        return None

    all_pts = np.concatenate(stroke_pts, axis=0)
    text = item.get("text", "")
    writer = item.get("writer", "unknown")

    raw_ink = Ink(all_pts.astype(np.float32), text=text, meta={"writer": writer})
    resampled = resample_time(raw_ink, hz=hz)
    if len(resampled) < 2:
        return None

    normalized, _ = normalize_line(resampled)
    return normalized


def ingest_collector_json(
    json_path: str | Path,
    out_dir: str | Path,
    hz: float = 100.0,
) -> int:
    """Ingests a single collector export JSON file into a LineStore directory."""
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    fmt = data.get("format", "")
    if fmt != "vibefont-collector/1":
        raise ValueError(f"Unsupported collector format: {fmt!r} in {path}")

    writer_default = data.get("writer", "unknown")
    doc_id = path.stem
    items = data.get("items", [])

    w = DatasetWriter(out_dir)
    count = 0
    for line_no, item in enumerate(items):
        ink = parse_collector_item(item, hz=hz)
        if ink is not None and len(ink) > 0:
            writer = item.get("writer", writer_default)
            w.add(ink, writer=writer, doc=doc_id, line_no=line_no, source="collector")
            count += 1

    w.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest web collector exports into VibeFont LineStore")
    parser.add_argument("input", help="Path to collector JSON file or directory")
    parser.add_argument("output", help="Output directory for processed LineStore")
    parser.add_argument("--hz", type=float, default=100.0, help="Resampling rate in Hz (default: 100)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_file():
        n = ingest_collector_json(in_path, args.output, hz=args.hz)
        print(f"Processed {n} lines from {in_path} -> {args.output}")
    elif in_path.is_dir():
        files = sorted(in_path.glob("*.json"))
        total = 0
        w = DatasetWriter(args.output)
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("format") != "vibefont-collector/1":
                    continue
                writer_default = data.get("writer", "unknown")
                doc_id = f.stem
                for line_no, item in enumerate(data.get("items", [])):
                    ink = parse_collector_item(item, hz=args.hz)
                    if ink is not None and len(ink) > 0:
                        writer = item.get("writer", writer_default)
                        w.add(ink, writer=writer, doc=doc_id, line_no=line_no, source="collector")
                        total += 1
            except Exception as e:
                print(f"Skipping {f.name}: {e}")
        w.close()
        print(f"Processed {total} lines from {len(files)} files -> {args.output}")


if __name__ == "__main__":
    main()
