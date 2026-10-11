# Claude Opus Verification & Architectural Review Guide

이 문서는 **VibeFont** 프로젝트의 기술 아키텍처 및 수학적 타당성을 Claude Opus에게 검토받기 위한 질의서입니다.

---

## 📌 Claude Opus 입력용 프롬프트 (그대로 복사하여 사용)

```markdown
당신은 생성형 AI, 컴퓨터 비전, 시퀀스 모델링 및 디지털 타이포그래피 분야의 최고 아키텍트입니다.
아래에 제시된 차세대 동적 필기체 합성 시스템 [VibeFont Master Plan]을 면밀히 검토하고, 시스템의 수학적/아키텍처적 타당성, 잠재적 리스크, 그리고 구체적인 개선안을 제시해 주십시오.

### [검토 배경]
기존 디지털 필기체 폰트는 고정된 글리프(Glyph)로 인해 같은 글자가 항상 동일하게 출력되고, 문맥에 따른 앞뒤 연결(Ligature)이 부자연스러우며, 만년필/볼펜 같은 아날로그 펜 물리 질감을 담지 못합니다.
VibeFont는 Diffusion 모델을 배제하고, "Autoregressive Stroke LMM (50M~1.5B) + Procedural Physical Ink Shader"라는 2단계 하이브리드 파이프라인으로 이를 해결하고자 합니다.

---

### [집중 검토 요청 사항 (Checkpoints)]

1. **스트로크 시퀀스 토크나이징 기법 (Quantized Points vs. Bézier Segments):**
   - 펜 궤적을 고정 격자 델타 좌표(Δx, Δy, p, state)로 이산화하는 방식과 3차 베지어 제어점(Cubic Bézier Segments)으로 분할하는 방식 중, 트랜스포머의 수렴 속도, 시퀀스 길이 최적화, 손떨림 노이즈(Jitter) 표현력 관점에서 어느 접근이 더 우수한가?
   - MDN(Mixture Density Networks) 같은 연속 회귀 헤드 대비 완전 이산 분류(Cross-Entropy) 헤드가 가지는 실질적 장단점은?

2. **텍스트-스트로크 정렬과 어텐션 구조 (Monotonic Attention):**
   - 텍스트 인코더와 궤적 디코더 간의 정렬에서 문장이 길어질 때 글자를 누락(Skipping)하거나 무한 반복하는 문제를 방지하기 위해, 일반 Cross-Attention 대신 Monotonic Attention 또는 Localized Window Masking을 도입해야 하는가? 구체적인 수학적 메커니즘을 추천한다면?

3. **Track B: 스타일 비전 인코더 선정 (DINOv2 vs. SigLIP / CLIP):**
   - 손글씨 사진 1장으로부터 의미적(Semantic) 정보가 아닌 선의 곡률, 기울기, 필압 분포, 자간 같은 '저수준 기하학적 필체 스타일'을 추출할 때 DINOv2가 적합한가, 아니면 SigLIP 계열이 적합한가?
   - ViT 패치 임베딩을 고정된 K개의 Style Prompt Token으로 압축하는 Perceiver Resampler 구조에 대해 잠재적 병목은 없는가?

4. **합성 궤적 생성을 위한 폰트 스켈레톤화(Skeletonization)의 위상학적 모호성:**
   - 대규모 학습 데이터를 위해 기존 필기체 폰트의 외곽선에서 중심선(Medial Axis)을 추출할 때, 교차점(Junction)에서 획 순서(Stroke Order)가 뒤엉키는 문제를 해결할 수 있는 가장 신뢰성 높은 그래프/위상학적 알고리즘은 무엇인가?

5. **물리 잉크 셰이더(Procedural Shader)와의 인터페이스:**
   - LMM이 출력해야 하는 최소 물리 파라미터 세트(예: 속도, 가속도, 압력, 펜 각도)와 이를 WebGL/WebGPU 셰이더에서 만년필 잉크 번짐(Pooling/Bleeding) 및 볼펜 질감으로 변환하는 수식적 접근에 대한 평가.
```

---

## 📎 첨부할 계획서 전문
[docs/MASTER_PLAN.md](file:///d:/Code/VibeFont/docs/MASTER_PLAN.md)
