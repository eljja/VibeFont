# VibeFont 사용 가이드 (User & Operational Guide)

VibeFont는 디지털 폰트의 한계(모든 글자가 동일한 형태)를 넘어, **사람이 쓴 자연스러운 필기(Handwriting)**를 생성하는 듀얼 트랙 이중 언어(한국어 + 영어) 손글씨 합성 시스템입니다.

---

## 🌟 핵심 철학 및 특징

1. **절대 같지 않은 글씨 (Non-Deterministic Handwriting, $DTW > 0$)**:
   - 동일한 문장이나 단어를 다시 요청하더라도, 사람이 손으로 쓸 때 발생하는 자연스러운 미세 변화(근육 피로도, 펜의 가속도, 글자 간 연결 변형)가 반영되어 매번 새로운 고유 필적이 생성됩니다.
2. **한국어 + 영어 자연스러운 조화 (Bilingual Support)**:
   - 한글 음절(초성/중성/종성 조합), 단독 자모, 영문 알파벳(대소문자), 숫자 및 문장 부호가 한 줄에 섞여도 자간과 베이스라인이 끊김 없이 연결됩니다.
3. **분위기 맞춤형 바이브 프롬프트 (Vibe Prompting)**:
   - `"만년필로 단정하게 쓴 편지"`, `"급하게 휘갈겨 쓴 메모"`, `"4B 연필 스케치"` 등의 자연어 프롬프트 또는 사전 정의 스타일을 통해 필기 속도, 기울기(Slant), 자간, 굵기, 용지 흡수율을 제어합니다.
4. **사진 필체 복제 (Track B Reference Style Extraction)**:
   - 종이에 손으로 쓴 사진이나 스캔본을 업로드하면, 컴퓨터 비전 알고리즘(Zhang-Suen 세선화, Radon 투영 분산)을 통해 기울기, 필압, 연결성을 자동으로 추출하여 해당 필체로 글씨를 작성합니다.
5. **저사양 CPU / 모바일 구동 & 로컬 GPU 학습 지원**:
   - 추론 및 렌더링은 경량 절차적 물리 모형과 ONNX Runtime을 지원하여 별도의 GPU가 없는 저사양 노트북이나 스마트폰에서도 즉시 실행 가능합니다.
   - 고품질 신경망 코어(CoreModel) 학습은 NVIDIA GeForce RTX 5070 Ti(CUDA) 환경을 완벽하게 활용합니다.

---

## 🚀 빠른 시작 (Quick Start)

### 1. 손글씨 생성 (CLI `vibefont write`)

```bash
# 기본 손글씨 생성 (PNG 출력)
python -m vibefont write "다람쥐 헌 쳇바퀴에 타고파. VibeFont handwriting." -o output.png

# 바이브 프롬프트 및 용지 설정
python -m vibefont write "오랜만에 펜을 들었어. 바람이 참 선선한 가을 저녁이네." \
  -p "만년필로 정성스럽게 쓴 편지" \
  --page A4 \
  --dpi 200 \
  -o letter.png

# 벡터 포맷(SVG) 출력 (플로터나 레이저 각인기, 인쇄소 전달용)
python -m vibefont write "VibeFont Vector Handwriting" -o output.svg

# 신경망 모델(Model Generator) 지정 생성
python -m vibefont write "신경망 코어로 쓴 한글 필기" \
  --generator model \
  --model checkpoints/bilingual_5k_core \
  -o neural_output.png
```

### 2. 바이브 프롬프트 해석 디버깅 (`vibefont interpret`)

```bash
python -m vibefont interpret "급하게 쓴 파란색 4B 연필 메모"
```
출력:
```json
{
  "style": {
    "slant_deg": 12.0,
    "speed": 0.8,
    "neatness": 0.3,
    "roundness": 0.5
  },
  "pen": {
    "type": "pencil",
    "width_mm": 0.7
  },
  "ink": {
    "color": [0.1, 0.15, 0.45]
  }
}
```

### 3. 데이터셋 무결성 검증 (`vibefont validate-data`)

```bash
python -m vibefont validate-data data/processed/bilingual_5k
```
출력:
```text
Dataset Health: [HEALTHY]
Total Lines: 5000
Total Points: 1699426
Total Strokes: 159172
Vocabulary: Hangul=2500, Latin=11103, Unique Chars=2358
```

---

## 🎨 대화형 웹 스튜디오 (`vibefont serve`)

별도의 복잡한 웹 프레임워크 설치 없이, 표준 라이브러리 기반의 웹 스튜디오를 브라우저에서 바로 사용할 수 있습니다.

```bash
python -m vibefont serve --port 8080
```

1. 웹 브라우저에서 `http://127.0.0.1:8080` 접속.
2. 좌측 패널에 한국어 및 영어 텍스트 입력.
3. 바이브 프롬프트 선택 (만년필, 빠른 볼펜, 4B 연필 등).
4. **[사진 필체 복제]**: 내 손글씨 사진을 드래그하여 업로드하면 필체 스타일 즉시 추출.
5. **[손글씨 작성하기]** 클릭 -> 우측에 실시간 고해상도 종이 질감 렌더링 확인.
6. **[현재 페이지 다운로드]**를 통해 고해상도 PNG 파일 저장.

---

## 📊 데이터셋 및 모델 학습 파이프라인

### 1. IAM 영어 필기 데이터 변환
```bash
python -m vibefont.data.iam_words --zip words.tgz.zip --words-txt data/raw/iam_words/words.txt --out data/processed/iam_words --limit 5000
```

### 2. AI Hub 한글 필기 OCR 데이터 변환
```bash
python -m vibefont.data.korean_ocr \
  --labels "다양한 형태의 한글 문자 OCR/Validation/[라벨]validation_필기체.zip" \
  --images "다양한 형태의 한글 문자 OCR/Validation/[원천]validation_필기체.zip" \
  --out data/processed/korean_ocr \
  --limit 5000
```

### 3. NVIDIA RTX 5070 Ti 로컬 학습
```bash
python -m vibefont.train_core \
  --data data/processed/bilingual_5k \
  --out-dir checkpoints/bilingual_5k_core \
  --preset xs \
  --epochs 15 \
  --batch-size 32 \
  --device cuda
```

### 4. 저사양 CPU 배포용 ONNX 변환
```bash
python -m vibefont.export_onnx \
  --checkpoint checkpoints/bilingual_5k_core/best_model.pt \
  --output models/onnx_cpu/ \
  --device cpu
```

---

## 📁 주요 코드 구조 및 역할

- [`src/vibefont/vision/tracer.py`](file:///d:/Code/VibeFont/src/vibefont/vision/tracer.py): 오프라인 이미지(PNG/JPG)를 100Hz 온라인 스트로크 좌표 `(x, y, t, p, eos)`로 복원하는 형태학적 세선화 추적기.
- [`src/vibefont/vision/extractor.py`](file:///d:/Code/VibeFont/src/vibefont/vision/extractor.py): 참조 손글씨 사진에서 기울기, 필압, 자간, 원형도를 추출하는 Track B 비전 분석기.
- [`src/vibefont/document/pipeline.py`](file:///d:/Code/VibeFont/src/vibefont/document/pipeline.py): 줄바꿈, 문단 여백, 피로도 누적 및 베이스라인 흔들림을 주입하는 물리 조판 파이프라인.
- [`src/vibefont/render/raster.py`](file:///d:/Code/VibeFont/src/vibefont/render/raster.py): Beer-Lambert 흡수율과 미세 잉크 튐, 종이 질감을 시뮬레이션하는 래스터 렌더러.
- [`src/vibefont/render/svg.py`](file:///d:/Code/VibeFont/src/vibefont/render/svg.py): 무손실 벡터 패스 SVG 생성기.
- [`src/vibefont/web/server.py`](file:///d:/Code/VibeFont/src/vibefont/web/server.py): 경량 로컬 웹 스튜디오 백엔드 서버.
- [`web/studio/index.html`](file:///d:/Code/VibeFont/web/studio/index.html): 인터랙티브 웹 스튜디오 UI.
- [`src/vibefont/data/validator.py`](file:///d:/Code/VibeFont/src/vibefont/data/validator.py): 데이터셋 무결성, 타임스탬프 단조성, 결측치, 어휘 분포 검사기.
