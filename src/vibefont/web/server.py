"""VibeFont Interactive Web Studio & REST API Server.

Zero external dependencies: uses Python standard library `http.server`.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import socket
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..document.pipeline import write_document
from ..generate import get_generator
from ..render.raster import render_page
from ..vibe.interpreter import interpret
from ..vibe.llm import interpret_llm
from ..vision.extractor import extract_style_from_image


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class VibeStudioHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            studio_html_path = Path(__file__).resolve().parent.parent.parent.parent / "web" / "studio" / "index.html"
            if not studio_html_path.exists():
                self.send_error(404, "Studio UI file not found")
                return
            content = studio_html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/interpret":
            try:
                body = self._read_json()
                prompt = body.get("prompt", "")
                use_llm = body.get("llm", False)
                spec = interpret_llm(prompt) if use_llm else interpret(prompt)
                self._send_json(spec.to_dict())
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)

        elif path == "/api/extract-style":
            try:
                body = self._read_json()
                b64_img = body.get("image", "")
                if not b64_img:
                    self._send_json({"error": "Missing 'image' parameter"}, status=400)
                    return
                # Handle optional data URI prefix
                if "," in b64_img:
                    b64_img = b64_img.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_img)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
                style = extract_style_from_image(pil_img)
                self._send_json({"style": dataclasses.asdict(style)})
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)

        elif path == "/api/generate":
            try:
                body = self._read_json()
                text = body.get("text", "")
                prompt = body.get("prompt", "")
                gen_name = body.get("generator", "auto")
                model_path = body.get("model", None)
                dpi = int(body.get("dpi", 150))
                page_fmt = body.get("page", None)
                seed = body.get("seed", None)

                use_llm = body.get("llm", False)
                spec = interpret_llm(prompt) if use_llm else interpret(prompt)

                if seed is not None:
                    spec.seed = int(seed)
                if spec.seed is None:
                    spec.seed = int(np.random.SeedSequence().entropy % (2**31))

                if page_fmt:
                    spec.layout.page = page_fmt

                # Merge custom style if provided in body
                if "style" in body and isinstance(body["style"], dict):
                    for k, v in body["style"].items():
                        if hasattr(spec.style, k) and v is not None:
                            setattr(spec.style, k, float(v))

                gen = get_generator(gen_name, model_path)
                rng = np.random.default_rng(spec.seed)
                pages = write_document(text, spec, gen, rng)

                images_b64: list[str] = []
                for i, p in enumerate(pages):
                    page_rng = np.random.default_rng(spec.seed + i)
                    img = render_page(p, spec, dpi, page_rng)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

                self._send_json({
                    "images": images_b64,
                    "spec": spec.to_dict(),
                    "pages": len(pages),
                    "generator": gen.name,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        else:
            self.send_error(404, "Endpoint Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard logging during unit testing
        pass


def create_server(host: str = "127.0.0.1", port: int = 8080) -> HTTPServer:
    return HTTPServer((host, port), VibeStudioHandler)
