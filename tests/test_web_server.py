import base64
import json
import threading
import time
import urllib.request
import pytest
from vibefont.web.server import create_server, find_free_port


@pytest.fixture(scope="module")
def server_url():
    port = find_free_port()
    server = create_server("127.0.0.1", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    url = f"http://127.0.0.1:{port}"
    yield url
    server.shutdown()
    server.server_close()


def test_get_index_html(server_url):
    req = urllib.request.Request(f"{server_url}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "VibeFont" in html


def test_api_interpret(server_url):
    payload = json.dumps({"prompt": "만년필로 단정하게 쓴 편지"}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/interpret",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "matched" in data
        assert "pen" in data
        assert data["pen"]["type"] == "fountain"


def test_api_generate(server_url):
    payload = json.dumps({
        "text": "Hello 한글",
        "prompt": "볼펜",
        "generator": "procedural",
        "dpi": 72,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "images" in data
        assert len(data["images"]) >= 1
        # Validate base64 decoded PNG header
        img_bytes = base64.b64decode(data["images"][0])
        assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_api_extract_style(server_url):
    import io
    from PIL import Image, ImageDraw

    # Create dummy 100x100 slanted stroke image
    img = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(img)
    draw.line([(20, 80), (80, 20)], fill=0, width=5)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = json.dumps({"image": b64_img}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/extract-style",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "style" in data
        assert "slant_deg" in data["style"]
