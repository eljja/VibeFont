# Human Calligraphy Realism & Semantic Expressiveness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform VibeFont's calligraphy and expressive handwriting from generic, computer-generated digital scripts into indistinguishable, human-grade calligraphy where word meaning drives stroke anatomy (단어 뜻을 살린 파격적 조형미), layout clusters organically (유기적 비대칭 여백), brush physics produce authentic dry bristles (갈필/飛白), and neuromotor hesitation removes all digital sterility.

**Architecture:** A 4-pillar human calligraphy engine:
1. `generate.semantics`: Analyzes Korean & English text for emotional/physical semantic archetypes (`BOLD_MASSIVE`, `SHARP_FIERCE`, `FLOWING_AIRY`, `GENTLE_WARM`, `SOARING_LUMINOUS`) and assigns stroke morphology directives (scale, weight, angularity, baseline drift, swash reach).
2. `generate.calligraphy`: Sculpting engine that uses semantic directives to transform stroke geometry—dramatic hero vs particle contrast (up to 2.2x : 0.65x), expressive 장획/삐침/파임, and asymmetric interlocking convex-hull layout.
3. `render.fluid` & `render.raster`: Authentic dry-brush (갈필/飛白) multi-bristle physics with longitudinal hair splits, speed starvation, and paper-tooth interaction.
4. `generate.kinematics`: Neuromotor hesitation, inflection pauses, physiological micro-tremor, and extreme calligraphic pressure contrast ($p \in [0.06, 0.98]$).

**Tech Stack:** Python 3.11+, NumPy, SciPy (ndimage), PIL (Pillow), pytest.

**Spec:** In-session requirements from user request ("캘리그래피는 너무 다 비슷해. 캘리그래피의 많은 경우는 단어의 뜻을 살린 글씨체를 적용하는게 많아. 그리고 아직도 컴퓨터가 만들어낸 글씨체라는게 티가나. 내가 원하는건 사람이 쓴건지 컴퓨터가 쓴건지 헷갈릴 정도의 글씨체야. 너가 직접 다 개선해내. 좀 그냥 해.")

## Global Constraints

- **No CAPTCHA Distortion:** Every single character must remain 100% human-readable and aesthetically pleasing. Distortion must follow calligraphic brush conventions (장획, 파임, 갈필), never random noise warping.
- **Strict TDD:** Every task must write failing tests first (Red), implement minimal code (Green), and verify regression-free full test suite before committing.
- **Performance Budget:** 60-image showcase generation must complete within $\le 94$s (baseline: ~68s).
- **Zero Broken Pre-existing Tests:** All 90 existing tests must remain green at every commit.

## Review Focus

1. *Unmapped words or neutral words:* Must default gracefully to balanced human calligraphy without errors or vanishing strokes.
2. *Extreme scale contrast:* When a hero word is 2.0x larger than particles, bounding boxes and kerning must never cause stroke overlap collisions or text truncation.
3. *Dry-brush stability:* Longitudinal bristle splits must never create visual artifacts or infinite loops when strokes are short or have zero length.
4. *Mixed Korean-English texts:* Semantic classification and layout nesting must seamlessly support both Hangul and Latin.
5. *Visual authenticity test:* Generated calligraphy must pass visual inspection as genuine hand-brushed artwork with distinct character per poem.

---

### Task 1: Semantic Emotion & Morphology Engine

**Files:**
- Create: `src/vibefont/generate/semantics.py`
- Test: `tests/test_semantics.py`

**Interfaces:**
- `SemanticArchetype`: Enum with `BOLD_MASSIVE`, `SHARP_FIERCE`, `FLOWING_AIRY`, `GENTLE_WARM`, `SOARING_LUMINOUS`, `NEUTRAL`.
- `WordMorphology`: Dataclass containing:
  - `archetype: SemanticArchetype`
  - `scale: float` (e.g. 1.0 to 2.2)
  - `weight_boost: float` (-0.25 to +0.45)
  - `angularity: float` (0.0 soft curve to 1.0 sharp chisel/hook)
  - `slant_drift_deg: float` (-8.0 to +8.0)
  - `swash_reach: float` (0.0 to 1.8)
  - `is_hero: bool`
  - `is_particle: bool`
- `analyze_semantics(text: str) -> list[list[WordMorphology]]`: Parses text line-by-line, identifies emotional resonance words in Korean and English, recognizes particles/function words, and computes proportional morphology.

- [ ] **Step 1: Write failing tests** in `tests/test_semantics.py`:
  - `test_korean_semantic_archetypes`: "눈보라" -> `SHARP_FIERCE`, "바위" -> `BOLD_MASSIVE`, "봄" -> `GENTLE_WARM`, "별" -> `SOARING_LUMINOUS`, "파도" -> `FLOWING_AIRY`.
  - `test_english_semantic_archetypes`: "storm" -> `SHARP_FIERCE`, "oak" -> `BOLD_MASSIVE`, "silence" -> `FLOWING_AIRY`, "stars" -> `SOARING_LUMINOUS`.
  - `test_hero_particle_scale_contrast`: Content words in a poem have `scale >= 1.30 * particle.scale`.
  - `test_morphology_clipping_bounds`: Scale is bounded in `[0.65, 2.20]`, weight in `[-0.25, 0.45]`.
- [ ] **Step 2:** Run `pytest tests/test_semantics.py` -> FAIL (`ModuleNotFoundError: vibefont.generate.semantics`).
- [ ] **Step 3:** Implement `src/vibefont/generate/semantics.py` with Korean and English semantic lexicons, particle detectors, and morphology scoring.
- [ ] **Step 4:** Run `pytest tests/test_semantics.py` -> PASS. Full suite -> 94/94 green.
- [ ] **Step 5: Commit** `feat(semantics): semantic emotion and morphology engine for expressive calligraphy`.

---

### Task 2: Semantic-Driven Stroke Sculpting for Korean & English

**Files:**
- Modify: `src/vibefont/generate/calligraphy.py` (`_sculpt_ko_syllable_strokes`, `_calligraphic_pressure_from_geometry`, English word swashes)
- Test: `tests/test_calligraphy_morphology.py`

**Interfaces:**
- `_sculpt_ko_syllable_strokes(ch: str, rng: np.random.Generator, morph: WordMorphology, profile: WriterProfile | None = None) -> list[tuple[np.ndarray, float, bool]]`:
  - `BOLD_MASSIVE`: Heavy blunt 기필, thick vertical pillars ($1.3\times$ weight), grounded horizontal bases.
  - `SHARP_FIERCE`: Elongated 45-deg angled entry attacks, accelerated sweeping 삐침/파임 ($1.4\times$ tail reach).
  - `FLOWING_AIRY`: Graceful S-curve horizontal 장획 extensions, smooth undulating ligatures.
  - `GENTLE_WARM`: Rounded corners (bezier filleting), relaxed curves, reduced 기필 hook.
  - `SOARING_LUMINOUS`: Soaring vertical ascenders ($1.25\times$ height) and high-contrast needle exits.
- English script enhancement:
  - Hero word ascender/descender loop expansions based on morphology swash reach.
  - High-contrast Copperplate & Modern Brush thick-thin ratio scaling ($p_{\text{down}} \sim 0.95$, $p_{\text{up}} \sim 0.08$).

- [ ] **Step 1: Write failing tests** in `tests/test_calligraphy_morphology.py`:
  - `test_massive_syllable_heavier_than_airy`: Syllable strokes for `BOLD_MASSIVE` ("바위") have higher average stroke weight than `FLOWING_AIRY` ("바람").
  - `test_sharp_syllable_longer_tail_than_gentle`: Tail stroke length for `SHARP_FIERCE` ("눈") is longer than `GENTLE_WARM` ("봄").
  - `test_soaring_syllable_aspect_ratio`: Height-to-width ratio of `SOARING_LUMINOUS` ("별") is taller than neutral.
  - `test_all_strokes_remain_finite_and_legible`: No NaN/inf, stroke coordinates bounded within card layout.
- [ ] **Step 2:** Run `pytest tests/test_calligraphy_morphology.py` -> FAIL.
- [ ] **Step 3:** Implement semantic morphology in `src/vibefont/generate/calligraphy.py`.
- [ ] **Step 4:** Run `pytest tests/test_calligraphy_morphology.py` -> PASS. Full suite -> green.
- [ ] **Step 5: Commit** `feat(calligraphy): semantic morphology sculpting for Hangul and English script`.

---

### Task 3: Organic Asymmetric Composition & Interlocking Layout

**Files:**
- Modify: `src/vibefont/generate/calligraphy.py` (`compose_calligraphy_page`)
- Test: `tests/test_calligraphy_layout.py`

**Interfaces:**
- `compose_calligraphy_page(text: str, spec: VibeSpec, rng: np.random.Generator | None = None) -> Page`:
  - Analyzes entire text semantics with `analyze_semantics`.
  - Determines primary focal center (핵심 구절 / Focal Point).
  - Interlocking syllable clustering: Grammatical particles nest into the concave negative space beneath or beside the preceding stem rather than marching in a flat line.
  - Asymmetric optical balance: Dynamically balances ink mass across the card, leaving breathing room (여백의 미).
  - Dynamic seal stamp placement: Positions the hand-carved stone seal (`seal_box_mm`) in the primary optical void (top-right or bottom-left depending on ink gravity), not always fixed bottom-right.

- [ ] **Step 1: Write failing tests** in `tests/test_calligraphy_layout.py`:
  - `test_hero_line_dominates_secondary_line`: Height and visual weight of the primary focal line is $\ge 1.35\times$ secondary lines.
  - `test_particle_tucked_into_stem_negative_space`: X-coordinate gap between stem and particle is smaller than inter-word spacing.
  - `test_seal_stamp_placed_in_optical_void`: Seal box does not overlap any ink strokes.
  - `test_no_strokes_cross_card_margins`: All ink points strictly contained inside card boundaries.
- [ ] **Step 2:** Run `pytest tests/test_calligraphy_layout.py` -> FAIL.
- [ ] **Step 3:** Implement organic asymmetric layout in `src/vibefont/generate/calligraphy.py`.
- [ ] **Step 4:** Run `pytest tests/test_calligraphy_layout.py` -> PASS. Full suite -> green.
- [ ] **Step 5: Commit** `feat(calligraphy): organic asymmetric composition with interlocking syllable clustering`.

---

### Task 4: Authentic Dry-Brush (갈필 / 飛白) & Multi-Bristle Physics

**Files:**
- Modify: `src/vibefont/render/fluid.py` (add `bristle_striations`)
- Modify: `src/vibefont/render/raster.py` (`stamp_stroke` bristle simulation)
- Modify: `src/vibefont/render/pens.py` (`brush` pen model)
- Test: `tests/test_dry_brush.py`

**Interfaces:**
- `bristle_striations(xy: np.ndarray, n_vec: np.ndarray, v: np.ndarray, dry: np.ndarray, rng: np.random.Generator) -> np.ndarray`:
  - Longitudinal multi-bristle hair simulation: 7 to 11 discrete bristle filaments tracked along the stroke path.
  - Speed-dependent hair separation: At high velocities ($v > 65$ mm/s) or dry stroke ends, individual hairs spread apart and ink transfer breaks up (갈필).
  - Paper tooth interaction: Dry bristles deposit ink only on paper micro-peaks, leaving paper texture visible in the hair gaps.
- `stamp_stroke` in `src/vibefont/render/raster.py`:
  - Integrates `bristle_striations` into coverage calculation when `pen.type == "brush"`.

- [ ] **Step 1: Write failing tests** in `tests/test_dry_brush.py`:
  - `test_fast_brush_stroke_exhibits_dry_striations`: High-velocity brush stroke has significantly higher transverse density variance (bristle lines) than slow stroke.
  - `test_slow_brush_stroke_remains_saturated`: Slow brush stroke has continuous coverage without premature hair splitting.
  - `test_dry_brush_interacts_with_paper_tooth`: Valleys in paper tooth have lower ink density on dry-brush strokes.
- [ ] **Step 2:** Run `pytest tests/test_dry_brush.py` -> FAIL.
- [ ] **Step 3:** Implement multi-bristle striations in `fluid.py`, `pens.py`, and `raster.py`.
- [ ] **Step 4:** Run `pytest tests/test_dry_brush.py` -> PASS. Full suite -> green.
- [ ] **Step 5: Commit** `feat(render): authentic dry-brush (갈필/비백) multi-bristle physics simulation`.

---

### Task 5: Neuromotor Hesitation & Micro-Rhythm Dynamics

**Files:**
- Modify: `src/vibefont/generate/calligraphy.py` (`_strokes_to_calligraphy_ink`)
- Modify: `src/vibefont/generate/kinematics.py`
- Test: `tests/test_neuromotor_calligraphy.py`

**Interfaces:**
- `_strokes_to_calligraphy_ink`:
  - Inflection hesitation: At sharp corners ($\kappa > 2.5$), the virtual hand pauses briefly ($dt \sim 0.04-0.07$s), allowing ink to dwell and pool naturally.
  - Strike acceleration: Downstrokes following a pause exhibit sudden ballistic acceleration ($v > 120$ mm/s).
  - Physiological tremor: Subtle $10\text{Hz}$ neuromotor oscillation ($\sigma \sim 0.012$mm) along extended sweeping strokes that eliminates sterile digital splines.
  - Extreme dynamic pressure range: $p \in [0.06, 0.98]$ smoothly interpolated via cubic Hermite tangents.

- [ ] **Step 1: Write failing tests** in `tests/test_neuromotor_calligraphy.py`:
  - `test_corner_dwell_hesitation`: Time spent at high curvature inflection point is $\ge 1.8\times$ time on straight segment of equal length.
  - `test_downstroke_ballistic_acceleration`: Downward stroke following hesitation accelerates to peak velocity.
  - `test_tremor_removes_perfect_bezier_collinearity`: 3 consecutive points on a long flourish deviate from a straight line by natural micro-wobble ($0.005 \le \delta \le 0.025$ mm).
- [ ] **Step 2:** Run `pytest tests/test_neuromotor_calligraphy.py` -> FAIL.
- [ ] **Step 3:** Implement neuromotor hesitation and physiological micro-rhythm.
- [ ] **Step 4:** Run `pytest tests/test_neuromotor_calligraphy.py` -> PASS. Full suite -> green.
- [ ] **Step 5: Commit** `feat(kinematics): neuromotor hesitation and calligraphic micro-rhythm dynamics`.

---

### Task 6: Visual Quality Gate, 60-Showcase Regeneration & Final Verification

**Files:**
- Modify: `scripts/generate_showcase_60.py` (refresh calligraphy configurations with distinct poetic vibes)
- Run: `scripts/generate_showcase_60.py`
- Inspect: All 20 calligraphy artworks + master comparison board
- Document: `docs/CALLIGRAPHY_REALISM.md`

- [ ] **Step 1:** Run full test suite `pytest -q` -> all tests pass.
- [ ] **Step 2:** Run `python scripts/generate_showcase_60.py` -> verify elapsed time $\le 94$s.
- [ ] **Step 3:** Visually inspect all 20 calligraphy cards:
  - Korean poems: "눈보라 속 매화" (sharp winter thorns + soft blossom), "별을 노래하는 마음" (soaring starry strokes), "파도와 바위" (crashing wave surges vs heavy block rock).
  - English poems: "Still Waters", "Light Through Broken Clouds", "Patience of the Oak".
  - Confirm authentic 갈필 (dry-brush splits), extreme pressure drama, human hesitation, and organic asymmetric layout.
- [ ] **Step 4: Commit** `feat(showcase): refresh 60-showcase with semantic calligraphy and human realism`.
- [ ] **Step 5:** Present visual proof and gallery links to the user.
