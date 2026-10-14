"""Command line: ``vibefont write "text" -p "vibe prompt" -o out.png``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _read_text(args) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text == "-":
        return sys.stdin.read()
    return args.text


def cmd_write(args) -> None:
    from .document.pipeline import write_document
    from .generate import get_generator
    from .render.raster import render_page, save_images
    from .render.svg import save_svg
    from .vibe.interpreter import interpret
    from .vibe.llm import interpret_llm

    spec = interpret_llm(args.prompt) if args.llm else interpret(args.prompt)
    if args.seed is not None:
        spec.seed = args.seed
    if spec.seed is None:
        spec.seed = int(np.random.SeedSequence().entropy % (2**31))
    if args.page:
        spec.layout.page = args.page
    gen = get_generator(args.generator, args.model)
    t0 = time.perf_counter()
    rng = np.random.default_rng(spec.seed)
    pages = write_document(_read_text(args), spec, gen, rng)
    t1 = time.perf_counter()
    out = Path(args.output)
    if out.suffix.lower() == ".svg":
        files = save_svg(pages, spec, out)
    else:
        imgs = [render_page(p, spec, args.dpi, np.random.default_rng(spec.seed + i)) for i, p in enumerate(pages)]
        files = save_images(imgs, out, args.dpi)
    t2 = time.perf_counter()
    print(f"generator={gen.name} seed={spec.seed} matched={spec.matched}")
    print(f"pages={len(pages)} lines={sum(len(p.lines) for p in pages)} gen={t1 - t0:.2f}s render={t2 - t1:.2f}s")
    for f in files:
        print(f"wrote {f}")
    if args.save_spec:
        Path(args.save_spec).write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_interpret(args) -> None:
    from .vibe.interpreter import interpret
    from .vibe.llm import interpret_llm

    spec = interpret_llm(args.prompt) if args.llm else interpret(args.prompt)
    print(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vibefont", description="text + vibe prompt -> handwriting")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="write text by hand")
    w.add_argument("text", nargs="?", default="", help="text to write ('-' = stdin)")
    w.add_argument("-f", "--file", help="read text from a UTF-8 file")
    w.add_argument("-p", "--prompt", default="", help='vibe, e.g. "급하게 쓴 만년필 메모"')
    w.add_argument("-o", "--output", default="out.png", help=".png / .pdf / .svg")
    w.add_argument("--seed", type=int, default=None, help="same seed -> same document")
    w.add_argument("--dpi", type=int, default=200)
    w.add_argument("--page", default=None, help="A4 | A5 | B5 | Letter")
    w.add_argument("--generator", default="auto", choices=["auto", "procedural", "model"])
    w.add_argument("--model", default=None, help="path to exported model directory")
    w.add_argument("--llm", action="store_true", help="interpret prompt via VIBEFONT_LLM_URL")
    w.add_argument("--save-spec", default=None, help="also dump the resolved VibeSpec JSON")
    w.set_defaults(func=cmd_write)

    i = sub.add_parser("interpret", help="show how a prompt is understood")
    i.add_argument("prompt")
    i.add_argument("--llm", action="store_true")
    i.set_defaults(func=cmd_interpret)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
