# VibeFont Production-Readiness & Complete Pre-Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all remaining architectural and functional pillars of VibeFont before real data ingestion: (1) Vision-based Few-shot handwriting style extractor (Track B foundation), (2) Local Interactive Web Studio & REST API (`vibefont serve`), (3) Dataset validation and pre-flight health checker (`vibefont validate-data`), and (4) Comprehensive CLI end-to-end integration tests.

**Architecture:** 
1. **Vision Style Extractor (`src/vibefont/vision/extractor.py`)**: A lightweight vision analysis pipeline that inspects an uploaded handwriting sample image (or scanned note), extracts contour kinematics, stroke width distribution, slant angle via Radon/PCA, baseline wobble, and connects into `StyleSpec` to allow "write like this image".
2. **Local Web Studio & Server (`src/vibefont/web/server.py` & `web/studio/index.html`)**: A zero-dependency Python `http.server` backend exposing REST endpoints (`/api/generate`, `/api/interpret`, `/api/extract-style`) serving an interactive single-page studio where users type Korean/English, enter vibes ("만년필로 흘려 쓴 편지"), preview realistic ink rendering live on ruled/kraft/ivory paper, and download PNG/PDF/SVG.
3. **Data Health & Pre-flight Validator (`src/vibefont/data/validator.py`)**: Automated verification for incoming IAM-OnDB and Korean datasets, detecting corrupt XML, non-monotonic timestamps, zero-length strokes, and computing Hangul/Latin character coverage.
4. **End-to-End CLI & Full Pipeline Verification (`tests/test_cli_e2e.py`)**: Complete integration tests exercising all CLI subcommands (`write`, `interpret`, `export-onnx`, `validate-data`, `serve`).

**Tech Stack:** Python 3.11, PyTorch 2.6+, NumPy, SciPy, Pillow, onnxruntime, pytest.

**Spec:** `docs/MASTER_PLAN.md` and user requirements from conversation history (Dual-track architecture, non-deterministic human handwriting, bilingual English + Korean, low-spec CPU & mobile friendly, RTX 5070 Ti training).

## Global Constraints

- Must run on CPU without GPU requirements for inference, with RTX 5070 Ti utilized for neural model training.
- No static glyph repetition: repeated letters or words must always have natural non-zero variance ($DTW > 0$).
- Fully bilingual (English ASCII + Korean Hangul syllables and standalone Jamo).
- Preserves physical stroke dynamics: uniform 100Hz time resampling, optical density with Beer-Lambert absorption.

## Review Focus

1. **Mixed English and Korean in single word** (e.g., `VibeFont한글1호`): must render without glyph overlap or negative cursor offsets.
2. **Empty or blank lines in multi-paragraph input**: must create vertical spacing without causing zero-point division or NaN in layout.
3. **Style extraction on non-ideal images**: noisy backgrounds, low contrast, or skewed photos must fall back gracefully to sensible default `StyleSpec` instead of throwing exceptions.
4. **Port collision on web studio server**: `vibefont serve` must handle port conflicts gracefully and report the active URL.
5. **Corrupted lines in dataset archives**: validator must flag corrupted strokes without aborting the entire dataset scan.

---

### Task 1: Vision-based Few-shot Handwriting Style Extractor (Track B Foundation)

**Files:**
- Create: `src/vibefont/vision/extractor.py`
- Create: `src/vibefont/vision/__init__.py`
- Test: `tests/test_vision_extractor.py`

**Interfaces:**
- Consumes: `vibefont.vibe.spec.StyleSpec`, `PIL.Image`
- Produces: `extract_style_from_image(image: Image.Image | Path | str) -> StyleSpec`

- [x] **Step 1: Write the failing test**
  In `tests/test_vision_extractor.py`, create a synthetic black-and-white handwritten line image (slanted strokes with varied thickness), call `extract_style_from_image(img)`, and assert that it returns a valid `StyleSpec` with measured slant, thickness, and roundness.

- [x] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\pytest tests\test_vision_extractor.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'vibefont.vision'`

- [x] **Step 3: Implement `src/vibefont/vision/extractor.py`**
  Implement binarization (Otsu threshold), contour extraction, stroke width estimation via distance transform, slant angle detection via horizontal run-length / PCA on stroke segments, and map to `StyleSpec`.

- [x] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\pytest tests\test_vision_extractor.py -v`
  Expected: PASS

- [x] **Step 5: Commit**
  ```bash
  git add src/vibefont/vision tests/test_vision_extractor.py
  git commit -m "feat(vision): add few-shot handwriting style extractor from reference images"
  ```

---

### Task 2: Dataset Pre-flight Health & Integrity Validator

**Files:**
- Create: `src/vibefont/data/validator.py`
- Modify: `src/vibefont/cli.py`
- Test: `tests/test_data_validator.py`

**Interfaces:**
- Consumes: `vibefont.dataset.LineStore`, raw dataset directories
- Produces: `validate_dataset(path: Path) -> ValidationReport`

- [x] **Step 1: Write the failing test**
  In `tests/test_data_validator.py`, construct a valid `LineStore` and a corrupted `LineStore` (with NaN coordinates and non-monotonic time). Assert that `validate_dataset` returns `is_healthy=True` for valid and `is_healthy=False` with descriptive errors for corrupt.

- [x] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\pytest tests\test_data_validator.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'vibefont.data.validator'`

- [x] **Step 3: Implement `src/vibefont/data/validator.py`**
  Implement validation logic: checks offsets consistency, bounds of coordinates, presence of NaNs/Infs, time monotonicity, text length vs stroke count ratio, and character vocabulary statistics. Wire `validate-data` into `cli.py`.

- [x] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\pytest tests\test_data_validator.py -v`
  Expected: PASS

- [x] **Step 5: Commit**
  ```bash
  git add src/vibefont/data/validator.py src/vibefont/cli.py tests/test_data_validator.py
  git commit -m "feat(data): add pre-flight dataset integrity validator"
  ```

---

### Task 3: Interactive Local Web Studio & API Server

**Files:**
- Create: `src/vibefont/web/server.py`
- Create: `src/vibefont/web/__init__.py`
- Create: `web/studio/index.html`
- Modify: `src/vibefont/cli.py`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `vibefont.document.pipeline.write_document`, `vibefont.render.raster.render_page`, `vibefont.vibe.interpreter.interpret`
- Produces: `create_server(host: str, port: int) -> HTTPServer`, CLI command `vibefont serve`

- [x] **Step 1: Write the failing test**
  In `tests/test_web_server.py`, test the HTTP request handler:
  - `POST /api/generate` with `{"text": "테스트", "prompt": "만년필"}` returns PNG base64 and VibeSpec.
  - `POST /api/interpret` returns parsed VibeSpec JSON.
  - `GET /` serves the Web Studio HTML.

- [x] **Step 2: Run test to verify it fails**
  Run: `.venv\Scripts\pytest tests\test_web_server.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'vibefont.web'`

- [x] **Step 3: Implement `src/vibefont/web/server.py` and `web/studio/index.html`**
  - Implement Python `http.server.BaseHTTPRequestHandler` handling JSON API and static files.
  - Create a modern, responsive Web Studio UI in `web/studio/index.html` with:
    - Text input with bilingual presets (Hangul pangrams, English quotes).
    - Vibe prompt bar with quick pills ("만년필", "급하게 흘려 씀", "단정한 볼펜", "연필").
    - Real-time high-resolution paper canvas preview.
    - Export buttons (Download PNG, PDF, SVG).
  - Add `serve` command to `src/vibefont/cli.py`.

- [x] **Step 4: Run test to verify it passes**
  Run: `.venv\Scripts\pytest tests\test_web_server.py -v`
  Expected: PASS

- [x] **Step 5: Commit**
  ```bash
  git add src/vibefont/web web/studio src/vibefont/cli.py tests/test_web_server.py
  git commit -m "feat(web): add local interactive web studio and REST API server"
  ```

---

### Task 4: End-to-End CLI & Pipeline Verification

**Files:**
- Create: `tests/test_cli_e2e.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- Consumes: `vibefont.cli.main`
- Produces: Verified end-to-end user workflows

- [x] **Step 1: Write the failing test**
  In `tests/test_cli_e2e.py`, test the complete CLI invocations via `subprocess`:
  - `vibefont write "bilingual test 다람쥐" -p "pencil" -o output.png`
  - `vibefont write "vector test" -o output.svg`
  - `vibefont interpret "neat fountain pen blue ink"`
  - `vibefont export-onnx --checkpoint checkpoints/test_run/model.pt --output tmp_onnx/`
  - `vibefont validate-data data/processed/synthetic`

- [x] **Step 2: Run test to verify it passes all subcommands**
  Run: `.venv\Scripts\pytest tests\test_cli_e2e.py -v`
  Expected: PASS

- [x] **Step 3: Commit**
  ```bash
  git add tests/test_cli_e2e.py
  git commit -m "test: add comprehensive end-to-end CLI integration test suite"
  ```

---

### Task 5: Final Comprehensive Verification & Production Readme

**Files:**
- Create: `docs/USER_GUIDE.md`
- Test: Full pytest suite

- [x] **Step 1: Run complete pytest suite**
  Run: `.venv\Scripts\pytest -v`
  Confirm 0 failures across all tests.

- [x] **Step 2: Create comprehensive `docs/USER_GUIDE.md`**
  Document the complete workflow:
  - Running CLI `vibefont write` with examples.
  - Launching Web Studio `vibefont serve`.
  - Tablet handwriting collection with `web/collector/index.html`.
  - Dataset preparation with IAM-OnDB and Web Collector exports.
  - Training on local RTX 5070 Ti with `train_core.py`.
  - Low-spec CPU & ONNX deployment with `export_onnx.py`.

- [x] **Step 3: Commit**
  ```bash
  git add docs/USER_GUIDE.md
  git commit -m "docs: add comprehensive VibeFont user and operational guide"
  ```
