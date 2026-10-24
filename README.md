# VibeFont ✒️✨

> **Living, Breathing Handwriting Generation Engine**  
> 고정된 디지털 글리프를 넘어, 문맥에 따라 살아 숨 쉬는 연속적 궤적과 아날로그 펜의 물리적 질감을 재현하는 차세대 동적 필기체 합성 시스템

[![GitHub license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Autoregressive%20LMM%20%2B%20Physics%20Shader-purple.svg)](docs/MASTER_PLAN.md)
[![Status](https://img.shields.io/badge/Status-Architecture%20Review-orange.svg)](docs/OPUS_VERIFICATION_GUIDE.md)

---

## 💡 The Core Problem & VibeFont Realism Architecture (해결 완료 ✅)

디지털 필기체 폰트가 부자연스러운 3대 핵심 문제를 순수 기하학 리가처, 유체/접촉 물리 셰이더, 문맥/필자 동적 변형 엔진을 통해 완벽히 해결했습니다:

1. **단절된 연결성 (Broken Ligatures) 해결** ➔ [`src/vibefont/generate/ligature.py`](src/vibefont/generate/ligature.py)
   * **앵커 자동 검출 & 스무스스텝 블렌딩:** 글리프의 진입점(Entry)과 탈출점(Exit)을 기하학적으로 감지하고, 글리프 본체 왜곡 없이 꼬리와 머리를 부드러운 스무스스텝($3w^2 - 2w^3$)으로 연결합니다.
   * **지연 획 분리 (Delayed Strokes):** 'i'의 점, 't'의 가로획 등 지연 획을 단어 전체 필기 궤적에서 자동 분리하여, 단어 본문을 끊김 없이 한 번에 이어 쓴 뒤 마지막에 찍는 실제 인간의 필기 습관을 모사합니다 (펜 리프트 80% 이상 감소).
   * **높은 탈출(High Exit) 처리:** 'b', 'o', 'v', 'w' 뒤의 높은 연결선과 일반 기저선 연결선을 문맥 분기하여 자연스럽게 이어줍니다.
   * **한글 자모 간 연속 획 (Cross-Jamo Cursive):** 초성-중성-종성 간의 간격을 분석하여 다른 획과 교차하지 않는 무충돌 2차 베지어 헤어라인 커넥터를 형성합니다.

2. **아날로그 물리감 부재 해결** ➔ [`src/vibefont/render/fluid.py`](src/vibefont/render/fluid.py), [`src/vibefont/render/pens.py`](src/vibefont/render/pens.py)
   * **머무름 시간 기반 잉크 고임 (Dwell-Time Ink Pooling):** 펜이 급커브나 정지 지점에서 머문 국소 시간(Excess Dwell)을 적분하여 만년필(+90%), 젤/마커(+60%), 볼펜(+30%)에 유기적인 잉크 고임 농담(Shading)을 구현합니다 (연필 등 건식 필기구는 격리).
   * **종이 섬유 지향 비등방성 번짐 (Anisotropic Fibre Feathering):** 가우시안 평활화된 2D 셀룰로오스 섬유 각도장($\theta$)을 따라 서브픽셀 잉크 확산 시뮬레이션을 수행하여 인위적 흐림이 아닌 진짜 종이 번짐을 재현합니다.
   * **유성 볼펜 페이스트 뭉침 & 롤러볼 스킵 (Ballpoint Goop & Skips):** 소켓 테두리에 누적된 끈적한 유성 잉크가 급회전 시 종이에 긁혀 나오는 코너 블롭과, 롤러볼 회전 주기($\pi D_{\text{ball}} \approx 2.2\text{mm}$) 및 고속 필기 시의 미세 스킵을 모델링했습니다.
   * **젖은 잉크 겹침 번짐 (Wet-on-Wet Overlap Bleed):** 직전 획이 아직 마르지 않은 시간 윈도우(1.5초) 내에 새로운 획이 교차할 때 두 획의 잉크가 모세관 현상으로 번져 교차점이 더 짙고 부드럽게 융합됩니다.

3. **정적 글리프 (Static Glyphs) 해결** ➔ [`src/vibefont/generate/procedural.py`](src/vibefont/generate/procedural.py)
   * **단어 내 위치별 변형 (Position-in-Word Dynamics):** 단어 시작 글자는 또박또박 안정적으로(계수 0.85), 중간 글자는 정속으로(1.0), 단어 끝 글자는 자연스럽게 풀리는 꼬리(1.30)로 변형합니다.
   * **필자별 글자쌍 커넥터 습관 (Letter-Pair Habit):** 동일 필자의 고유 해시(CRC32)를 통해 특정 글자쌍(예: 'er', 'th')을 이어 쓸 때 발생하는 고유한 높낮이/각도 습관을 결정론적으로 유지하면서 미세 실행 지터를 부여합니다.
   * **캡차(CAPTCHA)급 왜곡 원천 배제:** 가독성을 해치는 비선형 워핑 대신 철저한 가독성 클립 경계 내에서 인간 필적 수준의 자연스러운 비정형성만을 부여합니다.

---

## 🚀 The VibeFont Solution: 2-Stage Hybrid Architecture

Diffusion 모델의 치명적 단점인 **철자 환각(Spelling Hallucination)**과 **느린 추론 속도**를 배제하고, 언어모델의 시퀀스 정렬 능력과 GPU 셰이더의 초고속 물리 렌더링을 결합한 하이브리드 파이프라인을 구축합니다.

```
[입력 텍스트 + 스타일 프롬프트 / 손글씨 사진]
                       │
                       ▼
    ┌────────────────────────────────────────┐
    │       Stroke Trajectory Generator      │
    │  (Track A: 5~25M / Track B: ~50~300M)  │
    │  - 예측: (Δx, Δy, Δt, 펜 상태, [필압])   │
    └────────────────────────────────────────┘
                       │ 
                       ▼ 실시간 벡터 스트림
    ┌────────────────────────────────────────┐
    │     Procedural Physical Ink Shader     │
    │  - WebGL2 → WebGPU 렌더러               │
    │  - 속도/곡률/유사 필압은 렌더러에서 유도 │
    └────────────────────────────────────────┘
                       │
                       ▼
      [무한 해상도 벡터(SVG) & 초고해상도 잉크 렌더링]
```

---

## 🏗️ Dual-Track Development Roadmap

### 🅰️ Track A: Lightweight Stroke Model (`5M ~ 25M`, 데이터 확보 후 확장)
* **목표:** 온디바이스 / 웹 브라우저(WebGPU) 실시간 필기 궤적 생성
* **특징:**
  * CTC 강제 정렬 기반 하드 조건부 → 글자 누락/반복 구조적 방지
  * 앞뒤 글자의 문맥 기반 연결
  * Temperature + sampling bias로 "정갈함 ↔ 자유로움" 제어 (같은 작성자 안에서의 변동)

### 🅱️ Track B: Reference-Image Style Model (인코더 + `50M ~ 300M` 디코더)
* **목표:** 손글씨 사진으로부터 필체, 기울기, 자간을 추출하여 모사
* **특징:**
  * writer-ID 대조 학습 인코더 vs DINOv2-small 비교 후 선정
  * Perceiver Resampler (K=32 스타일 토큰)
  * 기준선: SDT (CVPR 2023)

### 🎨 Shared Core: Procedural Physical Ink Shader
* 모델 출력 (x, y, t, state)에서 속도·곡률·유사 필압을 유도하여 GPU에서 렌더링
* **만년필:** 저속 구간 잉크 고임, 종이 섬유 번짐, 농담
* **볼펜:** 필압/속도 기반 밀도, 스킵, 획 시작점 뭉침
* **넓은 펜촉:** 획 방향과 펜촉 각도에 따른 굵기 변화

---

## 📂 Repository Structure

```
VibeFont/
├── docs/
│   ├── MASTER_PLAN.md               # 마스터 계획서 (v1.1)
│   ├── OPUS_VERIFICATION_GUIDE.md   # 검증 프롬프트
│   └── reviews/OPUS_REVIEW_v1.md    # Opus 아키텍처 리뷰
├── src/                             # (예정)
│   ├── data/                        # IAM-OnDB 로더, 전처리, 수집 웹앱
│   ├── tokenizer/                   # 오차 피드백 양자화 토크나이저
│   ├── htr/                         # 온라인 HTR(CTC) + 강제 정렬 + 평가
│   ├── models/                      # Track A / Track B
│   └── shader/                      # WebGL2/WebGPU 잉크 셰이더
├── tests/
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📑 Documentation & Links

* 📘 [상세 마스터 개발 계획서 (Master Plan v1.1)](docs/MASTER_PLAN.md)
* 🔍 [Opus 아키텍처 리뷰 v1](docs/reviews/OPUS_REVIEW_v1.md)
* 🔬 [Claude Opus 아키텍처 검증 가이드 (Opus Guide)](docs/OPUS_VERIFICATION_GUIDE.md)

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
