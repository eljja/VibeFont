# VibeFont: 차세대 동적 필기체 합성 시스템 마스터 개발 계획서

**Version:** 1.0.0  
**Target Architecture:** Autoregressive Stroke LMM (No Diffusion) + Procedural Physical Ink Shader  
**Repository:** [https://github.com/eljja/VibeFont](https://github.com/eljja/VibeFont)  
**Primary Reviewer:** Claude Opus (Architecture & Mathematics Verification)  

---

## 1. 프로젝트 비전 및 핵심 패러다임

기존 디지털 필기체 폰트의 근본적인 한계:
1. **정적 글리프(Static Glyphs):** 동일한 글자는 문맥과 상관없이 항상 100% 동일한 형태로 출력됨.
2. **단절된 연결(Broken Ligatures):** 앞 글자의 꼬리(Exit stroke)와 뒷 글자의 머리(Entry stroke)가 물리적으로 이어지지 않음.
3. **물리적 질감 부재:** 펜촉의 필압, 속도에 따른 잉크 뭉침(Pooling), 종이 번짐(Bleeding) 등의 아날로그 감성이 없음.

**VibeFont의 해결책:**
텍스트를 픽셀 이미지가 아닌 **'연속적 펜 궤적(Continuous Pen Trajectory)'**으로 모델링하고, 질감은 **'물리 셰이더(Procedural Shader)'**로 분리 렌더링하는 2단계 하이브리드 파이프라인.

```
[User Text + Style Prompt/Image]
               │
               ▼
   ┌────────────────────────────────────────┐
   │         LMM Trajectory Generator       │
   │  (Track A: 50M~300M / Track B: 0.5~1.5B)│
   └────────────────────────────────────────┘
               │ Stroke Coordinates: (x, y, p, state)
               ▼
   ┌────────────────────────────────────────┐
   │    Procedural Physics Ink Shader       │
   │  (WebGL / WebGPU / Canvas2D Engine)    │
   └────────────────────────────────────────┘
               │
               ▼
[최종 출력: 벡터(SVG) 또는 초고해상도 잉크 렌더링]
```

---

## 2. Track A: 스트로크 궤적 기반 경량 LMM (50M ~ 300M)

### 2.1. 목적 및 요구사항
* **목표:** 텍스트와 텍스트 스타일 임베딩(Style ID/Vibe Vector)을 입력받아 실시간으로 사람이 쓴 것 같은 필기 궤적을 자기회귀(Autoregressive) 방식으로 생성.
* **타겟 환경:** 스마트폰 온디바이스, 웹 브라우저(WebGPU / ONNX Runtime Web)에서 60fps 이상 실시간 스트로크 드로잉.
* **파라미터 규모:** 80M ~ 200M (Sweet Spot: ~120M).

### 2.2. 데이터 표현 및 토크나이징 설계 (Tokenization)
연속적인 펜 움직임을 트랜스포머가 처리할 수 있도록 이산 토큰화(Discretization)합니다.

* **스트로크 시퀀스 단위 정의:**
  $$T_i = (\Delta x_i, \Delta y_i, p_i, s_i)$$
  * $\Delta x_i, \Delta y_i$: 이전 좌표 대비 상대 변위 (8-bit or 10-bit Quantization, -512 ~ +512 bin).
  * $p_i$: 필압 (Pressure, 0~255).
  * $s_i$: 펜 상태 (State: `PEN_DOWN`, `PEN_UP`, `END_OF_CHAR`, `END_OF_TEXT`).
* **대안 아키텍처 검토 (Bézier Segment Tokenization):**
  * 조밀한 점 시퀀스 대신 **3차 베지어 곡선(Cubic Bézier) 제어점 ($P_0, P_1, P_2, P_3$)** 단위로 토큰화하여 시퀀스 길이를 1/5로 압축하는 방안.

### 2.3. 모델 아키텍처
* **Text Encoder (Prefix):**
  * 6-Layer Small Transformer (Hidden Dim: 384, Char/Byte 단위 BPE).
  * 텍스트 철자를 1:1 보존하기 위해 서브워드가 아닌 문자(Character) 단위 인코딩 필수.
* **Trajectory Decoder:**
  * 12-Layer Decoder-only Transformer (Hidden Dim: 512, Heads: 8, Intermediate Dim: 2048, ~120M params).
  * **Attention 구조:**
    * Causal Self-Attention (Trajectory History 반영).
    * Cross-Attention with Localized Monotonic Window: 글씨는 왼쪽에서 오른쪽으로 순차적으로 쓰여지므로, 텍스트 인코더와의 정렬이 역전되지 않도록 모노토닉(단조) 어텐션 유도 또는 윈도우 마스크 적용.
* **Output Heads:**
  * Quantized Coordinate Classification Head (Cross-Entropy Loss) 또는
  * GMM/MDN Head (Mixture Density Network Head for continuous coordinate regression).
  * *권장:* 안정적인 학습과 모드 붕괴 방지를 위해 이산 분류(Cross-Entropy) 헤드 우선 검증.

### 2.4. 데이터셋 구축 파이프라인
1. **기존 온라인 필기 데이터셋:**
   * **IAM-OnDB (IAM On-Line Handwriting Database):** 영문 필기 궤적 표준 (86,272개 텍스트 라인).
   * **AI Hub 한글 필기체 궤적 데이터:** 한글 초성/중성/종성 궤적 데이터.
2. **오프라인 폰트 역벡터화 (Font Skeletonization Synthesis):**
   * 기존 필기체 폰트의 TTF/OTF 외곽선에서 중심선(Medial Axis / Voronoi Skeleton)을 추출하고, 휴리스틱 오더링(TSP/위상 정렬) 알고리즘을 거쳐 대규모 합성 온라인 궤적 데이터 자동 생성 (수백만 단어 규모 증강).

---

## 3. Track B: 스타일 비전 인코더 결합형 LMM (0.5B ~ 1.5B)

### 3.1. 목적 및 요구사항
* **목표:** 사용자가 업로드한 손글씨 사진 1장(Few-shot / In-Context Visual Reference)을 입력받아, 그 사람 고유의 자간, 기울기, 필압 분포, 합자(연결) 습관을 그대로 모사하여 지정된 텍스트를 작성.
* **타겟 환경:** 클라우드 GPU API 서버 (A10G, T4, L4 등 경량 GPU에서 단일 배치 추론 가능).
* **파라미터 규모:** 0.8B ~ 1.3B.

### 3.2. 모델 아키텍처 설계
```
[Reference Handwriting Image]          [Target Text: "VibeFont Engine"]
            │                                         │
            ▼                                         ▼
   ┌─────────────────┐                       ┌─────────────────┐
   │  Vision Encoder │                       │   Text Encoder  │
   │ (SigLIP / DINO) │                       │  (Transformer)  │
   └─────────────────┘                       └─────────────────┘
            │                                         │
            ▼ (Perceiver / MLP Resampler)              │
    [Style Tokens: S_1...S_k]                         │
            └────────────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────┐
                    │ Trajectory Decoder (0.8B) │
                    │   (Track A Scaled Up)     │
                    └───────────────────────────┘
                                 │
                                 ▼
                     [Predicted Stroke Stream]
```

* **Visual Style Encoder:**
  * **후보 A:** `DINOv2-small/base` (저수준 기하학, 획의 질감, 선 굵기 등 자기지도 시각 특징 추출에 가장 우수).
  * **후보 B:** `SigLIP-So400M` (텍스트-이미지 정렬이 우수하나 고주파 선 기하학 특징에서는 DINO 대비 검토 필요).
  * **Style Resampler:** ViT의 패치 임베딩을 고정된 $K=32$개 또는 $64$개의 **Style Tokens**로 압축하는 Perceiver Resampler 레이어.
* **Trajectory Decoder Backbone:**
  * 24-Layer Transformer (~800M params, Hidden Dim: 1024, Heads: 16).
  * Prefix에 Style Token들을 위치시키고, Text에 Cross-Attention을 걸며 Stroke를 자기회귀 생성.

### 3.3. 학습 전략 (3단계 커리큘럼)
1. **Stage 1: Trajectory Backbone 사전학습**
   * Track A에서 검증된 텍스트-궤적 생성 모델을 대규모 합성 궤적 데이터셋으로 사전학습.
2. **Stage 2: 스타일 대조 학습 (Style Contrastive Pretraining)**
   * 동일 작성자의 다른 문장 궤적 / 다른 작성자의 동일 문장 궤적을 이용하여 스타일 인코더가 내용(Text Content)과 무관하게 필체(Style Invariant)만을 추출하도록 대조 손실(InfoNCE) 학습.
3. **Stage 3: End-to-End Few-shot 파인튜닝**
   * (손글씨 이미지 패치, 텍스트) $\to$ (정답 궤적 토큰) 형태로 전과정 파인튜닝.
   * 이미지 증강: 조명 변화, 종이 구김, 렌즈 왜곡, 크롭 노이즈 적용.

---

## 4. 공통 코어: 물리 잉크 렌더러 (Procedural Ink Shader)

AI가 출력한 궤적에 '진짜 만년필/볼펜'의 감성을 불어넣는 렌더링 엔진입니다.

```
AI 출력: (x, y, p, t)
   ↓
물리 연산: 속도 v = Δd / Δt, 가속도 a, 압력 p
   ↓
기하 메쉬(Mesh) 생성: 펜촉 형태(Nib Shape & Angle)에 따른 버텍스 확장
   ↓
프래그먼트 셰이더(Fragment Shader):
   - 만년필: v가 낮으면 Pooling(잉크 뭉침), 외곽선 Feathering(종이 스며듬), 농담(Shading)
   - 볼펜: 선 양끝의 릿지(Ridge), 중앙의 미세 스킵(Skip), 잉크 찌꺼기(Goop)
```

* **장점:** 렌더링에 AI 연산이 들지 않으므로, 생성 즉시 4K/벡터 해상도로 초당 60프레임 무지연 렌더링 보장.

---

## 5. 단계별 마일스톤 및 로드맵

| 단계 | 기간 | 주요 마일스톤 (Track A & B 연계) | 산출물 |
| :---: | :---: | :--- | :--- |
| **Phase 1** | 1~3주차 | **수학적 모델 검증 & 토크나이저 구축**<br>• IAM-OnDB 기반 전처리 및 토크나이저 구현<br>• 베지어 곡선 vs 포인트 양자화 비교 실험<br>• 웹 캔버스 베이직 셰이더 프로토타입 | Stroke Tokenizer, Web Shader v0.1 |
| **Phase 2** | 4~7주차 | **Track A (경량 LMM) 학습 및 온디바이스 포팅**<br>• 100M 트랜스포머 아키텍처 학습<br>• Temperature 노이즈 기반 인스턴스별 변형 검증<br>• ONNX/WebGPU 변환 및 웹 데모 구현 | VibeFont-Nano (100M), WebGPU Demo |
| **Phase 3** | 8~12주차 | **Track B (스타일 비전 결합) 개발**<br>• DINOv2 기반 Style Resampler 설계<br>• 멀티모달 Few-shot 데이터 파이프라인 가동<br>• 0.8B 모델 학습 및 스타일 모사율 평가 | VibeFont-Vision (0.8B), Style Cloner API |
| **Phase 4** | 13~15주차 | **통합 및 상용화 파이프라인 완성**<br>• 셰이더 고급화 (만년필 번짐, 수채화, 볼펜 완성도)<br>• 시스템 폰트(OpenType-SVG) 내보내기 익스포터 개발 | VibeFont Suite v1.0 |
