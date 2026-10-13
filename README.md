# VibeFont ✒️✨

> **Living, Breathing Handwriting Generation Engine**  
> 고정된 디지털 글리프를 넘어, 문맥에 따라 살아 숨 쉬는 연속적 궤적과 아날로그 펜의 물리적 질감을 재현하는 차세대 동적 필기체 합성 시스템

[![GitHub license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Autoregressive%20LMM%20%2B%20Physics%20Shader-purple.svg)](docs/MASTER_PLAN.md)
[![Status](https://img.shields.io/badge/Status-Architecture%20Review-orange.svg)](docs/OPUS_VERIFICATION_GUIDE.md)

---

## 💡 The Core Problem: Why Digital Script Fonts Feel Fake

1. **정적 글리프 (Static Glyphs):** 같은 문맥에서 동일한 글자는 항상 100% 동일한 형태로 찍혀 나옵니다.
2. **단절된 연결성 (Broken Ligatures):** 앞 글자의 꼬리(Exit stroke)와 뒷 글자의 머리(Entry stroke)가 단어와 문맥에 따라 자연스럽게 이어지지 않습니다.
3. **아날로그 물리감 부재:** 진짜 만년필의 잉크 번짐(Feathering)과 농담(Shading), 볼펜의 미세한 스킵과 뭉침(Goop) 같은 유체/접촉 물리가 없습니다.

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
