# VibeFont ✒️✨

> **Living, Breathing Handwriting Generation Engine**  
> 고정된 디지털 글리프를 넘어, 문맥에 따라 살아 숨 쉬는 연속적 궤적과 아날로그 펜의 물리적 질감을 재현하는 차세대 동적 필기체 합성 시스템

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
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
    │     LMM Stroke Trajectory Generator    │
    │  (Track A: 50M~300M / Track B: 0.5~1.5B)│
    │  - 예측: (Δx, Δy, 압력 p, 펜 상태 s)       │
    └────────────────────────────────────────┘
                       │ 
                       ▼ 60fps 실시간 벡터 스트림
    ┌────────────────────────────────────────┐
    │     Procedural Physical Ink Shader     │
    │  - WebGL / WebGPU / Canvas2D 렌더러     │
    │  - 만년필 농담, 잉크 뭉침, 볼펜 릿지 계산 │
    └────────────────────────────────────────┘
                       │
                       ▼
      [무한 해상도 벡터(SVG) & 초고해상도 잉크 렌더링]
```

---

## 🏗️ Dual-Track Development Roadmap

### 🅰️ Track A: Stroke-based Lightweight LMM (`50M ~ 300M`)
* **목표:** 초경량 온디바이스 / 웹 브라우저(WebGPU) 실시간 필기 궤적 생성
* **특징:**
  * 텍스트 1:1 보존 (오탈자 0%)
  * 앞뒤 글자의 동적 합자(Ligature) 유기적 보간
  * Temperature 파라미터 제어로 같은 단어도 매번 미세하게 다른 형태로 생성

### 🅱️ Track B: Vision-Conditioned Few-Shot LMM (`0.5B ~ 1.5B`)
* **목표:** 손글씨 사진 1장으로부터 고유의 필체, 기울기, 자간, 필압을 추출하여 모사
* **특징:**
  * DINOv2 기반 Visual Style Feature 추출
  * Perceiver Resampler를 통한 고밀도 Style Token 압축
  * 클라우드 경량 GPU(T4/L4/A10G) 단일 배치 서빙

### 🎨 Shared Core: Procedural Physical Ink Shader
* AI가 출력한 좌표와 속도($v = \Delta d / \Delta t$), 필압($p$) 데이터를 받아 GPU 프래그먼트 셰이더로 실시간 코팅
* **만년필 모드:** 저속 구간 잉크 뭉침(Pooling), 종이 섬유 흡수 번짐(Bleeding), 잉크 농담(Shading)
* **볼펜 모드:** 압력 분산에 따른 양쪽 릿지(Ridge)와 중심부 빈틈, 잉크 찌꺼기(Goop)

---

## 📂 Repository Structure

```
VibeFont/
├── docs/
│   ├── MASTER_PLAN.md               # 마스터 아키텍처 및 상세 개발 계획서
│   └── OPUS_VERIFICATION_GUIDE.md    # Claude Opus 아키텍처 검증 가이드
├── src/                             # 핵심 소스 코드 (예정)
│   ├── tokenizer/                   # 궤적 토크나이저 (Δx, Δy, Bézier)
│   ├── models/                      # Track A (Nano) & Track B (Vision) 아키텍처
│   ├── dataset/                     # IAM-OnDB 및 폰트 스켈레톤 변환기
│   └── shader/                      # WebGL/WebGPU 잉크 물리 셰이더
├── tests/                           # 단위 테스트 및 검증 스크립트
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📑 Documentation & Links

* 📘 [상세 마스터 개발 계획서 (Master Plan)](docs/MASTER_PLAN.md)
* 🔬 [Claude Opus 아키텍처 검증 가이드 (Opus Guide)](docs/OPUS_VERIFICATION_GUIDE.md)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
