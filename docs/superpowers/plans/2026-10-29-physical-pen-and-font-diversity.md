# Physical Pen Dynamics and Font Diversity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

This plan resolves the synthetic "computer-generated pixel" appearance and font monotony in VibeFont by replacing 2D circle-stamping with authentic tool-paper contact physics, replacing artificial Bézier spline swashes with biomechanical neuromotor release, and introducing distinct handwriting skeletal topologies (Allograph Personas).

---

## Architecture & Decomposition Overview

1. **`src/vibefont/render/pens.py` & `src/vibefont/render/raster.py` (Physical Pen Mechanics & Micro-Texture)**
   - **Fountain Pen Slit/Tine Bifurcation**: Under flex/pressure, metal nib prongs spread apart, producing dual-rail edge darkening with lighter slit centers and liquid dye pooling.
   - **Pencil Graphite Particulate Adhesion**: Microscopic graphite shearing onto paper tooth peaks along movement tangent with realistic particulate void micro-texture.
   - **Ballpoint Socket Rolling**: Viscous paste edge tracks from the rolling metal sphere and microscopic chatter gaps.
   - **Capillary Fiber Diffusion**: Realistic anisotropic paper fiber ink absorption replacing artificial uniform blur.

2. **`src/vibefont/generate/calligraphy.py` & `src/vibefont/generate/kinematics.py` (Biomechanical Neuromotor Lift-Off)**
   - Remove hardcoded Bézier curves (`_curve([[p0[0]...]])`).
   - Implement neuromotor ballistic exit: dynamic velocity decay $v(t) = v_0 e^{-kt}$, forearm/wrist pivot inertia curvature, and natural pressure lift-off $p(t) \to 0$.

3. **`src/vibefont/generate/hangul.py` & `src/vibefont/vibe/lexicon.py` (Handwriting Skeletal Topologies / Allograph Personas)**
   - Implement distinct allograph topologies:
     - `gosi` (고시체 / 속기체): Rapid, continuous cursive linkages, simplified ㄹ (z-form), single-stroke circular ㅁ, streamlined ㅈ/ㅊ.
     - `classic_jeongja` (정자체 / 반듯한 명필): Crisp angular stops, authoritative vertical pillars, balanced proportions.
     - `rounded_diary` (둥근 일기체): Bubbly circular consonant heads, relaxed open loops, buoyant baseline rhythm.
     - `rough_hand` (투박한 손글씨): Heavy blunt pressure, sharp abrupt directional transitions.
   - Wire into `interpret()` so prompts for "고시체", "정갈한 정자체", "둥근 다이어리체", "거친 연필 필체" select fundamentally distinct letter skeletons.

---

## Proposed File Map

- Modify `src/vibefont/render/pens.py`: Add tine bifurcation parameter, graphite particulate adhesion scale, ballpoint rim paste profile.
- Modify `src/vibefont/render/raster.py`: Update `stamp_stroke` to render tool-specific micro-footprints (slit bifurcation, graphite particle tooth adhesion, fiber wicking).
- Modify `src/vibefont/generate/calligraphy.py`: Replace static Bézier `_curve` swashes with biomechanical ballistic release.
- Modify `src/vibefont/generate/hangul.py`: Add allograph topologies for `gosi`, `classic_jeongja`, and `rounded_diary`.
- Modify `src/vibefont/vibe/lexicon.py` & `src/vibefont/vibe/interpreter.py`: Map new handwriting persona keywords to allograph styles.
- Create `tests/test_pen_microphysics.py`: Test suite verifying physical slit tracks, graphite particle distribution, and fiber wicking.
- Create `tests/test_allograph_personas.py`: Test suite verifying distinct letter skeletal topologies across handwriting personas.

---

## Tasks

### Task 1: Physical Pen Micro-Footprints & Texture Simulation

**Files:**
- Modify: `src/vibefont/render/pens.py`
- Modify: `src/vibefont/render/raster.py`
- Test: `tests/test_pen_microphysics.py`

- [ ] **Step 1.1**: Write failing tests in `tests/test_pen_microphysics.py` asserting:
  - Flex fountain pen stroke produces a transverse tine profile with higher edge density (bifurcation rail tracks) under high pressure.
  - Pencil stroke on paper tooth produces non-zero particulate micro-voids (graphite flakes adhering only to peaks, leaving tooth valleys light).
  - Ballpoint stroke exhibits dual paste margin accumulation.
- [ ] **Step 1.2**: Run `pytest tests/test_pen_microphysics.py` and confirm it fails (Red).
- [ ] **Step 1.3**: Update `PenSample` in `src/vibefont/render/pens.py` to include:
  - `tine_split: float`: degree of nib tine parting under flex pressure (0 for rigid, up to 0.75 for flex fountain).
  - `graphite_shear: float`: particulate graphite flakes peeling on paper tooth peaks.
  - `ball_rim_paste: float`: viscous ink paste ridge along ball socket margins.
- [ ] **Step 1.4**: Update `stamp_stroke` in `src/vibefont/render/raster.py`:
  - When `tine_split > 0.05` and pressure is high: modulate transverse coverage `u_perp` so the two tines leave distinct darker outer rail tracks with a slightly lighter center slit channel.
  - When `graphite_shear > 0.05`: modulate density using the local high-frequency paper tooth gradient, depositing graphite predominantly on the leading tooth ridges and leaving micro-voids in deep valleys.
  - When `ball_rim_paste > 0.05`: create dual margin ridges along the stroke path.
- [ ] **Step 1.5**: Run `pytest tests/test_pen_microphysics.py` and confirm all tests pass (Green).
- [ ] **Step 1.6**: Run the full test suite `pytest tests/` to confirm zero regressions.
- [ ] **Step 1.7**: Commit changes with message `feat(render): implement physical pen micro-footprints and graphite particulate adhesion`.

---

### Task 2: Biomechanical Neuromotor Release (Replace Artificial Bézier Swashes)

**Files:**
- Modify: `src/vibefont/generate/calligraphy.py`
- Modify: `src/vibefont/generate/kinematics.py`
- Test: `tests/test_calligraphy_layout.py`

- [ ] **Step 2.1**: Write test in `tests/test_calligraphy_layout.py` verifying:
  - Calligraphy flourish entrance and exit strokes are generated through biomechanical ballistic momentum (exponential velocity decay and continuous second derivative curvature) rather than static Bézier control points.
  - Exit strokes terminate smoothly with decaying pressure ($p \to 0$) without clipping or sharp unnatural kinks.
- [ ] **Step 2.2**: Run `pytest tests/test_calligraphy_layout.py` and confirm failure (Red).
- [ ] **Step 2.3**: In `src/vibefont/generate/kinematics.py`, implement `ballistic_flourish(start_pt, initial_direction, initial_speed, inertia_curvature, length_mm, n_pts=20)` simulating hand inertia, wrist pivot arc, and natural deceleration.
- [ ] **Step 2.4**: In `src/vibefont/generate/calligraphy.py`, remove all hardcoded `_curve([[...]])` Bézier spline definitions and replace them with `ballistic_flourish`.
- [ ] **Step 2.5**: Run `pytest tests/test_calligraphy_layout.py` and confirm tests pass (Green).
- [ ] **Step 2.6**: Run `pytest tests/test_glyph_integrity.py` to ensure 100% glyph integrity remains intact.
- [ ] **Step 2.7**: Commit changes with message `refactor(calligraphy): replace synthetic bezier swashes with biomechanical ballistic release`.

---

### Task 3: Handwriting Skeletal Topologies (Allograph Personas)

**Files:**
- Modify: `src/vibefont/generate/hangul.py`
- Modify: `src/vibefont/vibe/lexicon.py`
- Modify: `src/vibefont/vibe/interpreter.py`
- Test: `tests/test_allograph_personas.py`

- [ ] **Step 3.1**: Write failing tests in `tests/test_allograph_personas.py` verifying:
  - Specifying archetype `gosi` produces simplified cursive allographs (e.g. ㄹ rendered as single continuous z-stroke, ㅁ rendered as single-loop oval).
  - Specifying archetype `classic_jeongja` produces crisp 90-degree angular corner stops for ㄱ, ㄷ, ㅁ, ㅂ.
  - Specifying archetype `rounded_diary` produces exaggerated circular loops for ㅇ, ㅎ, and curved stroke terminals.
  - Prompts with "고시체", "정갈한 정자체", "둥근 다이어리" in `interpret()` select these distinct structural archetypes.
- [ ] **Step 3.2**: Run `pytest tests/test_allograph_personas.py` and confirm failure (Red).
- [ ] **Step 3.3**: In `src/vibefont/generate/hangul.py`, extend Jamo stroke generators (`choseong_strokes`, `jungseong_strokes`, `jongseong_strokes`):
  - Add topology handlers for `gosi` (cursive single-stroke allographs).
  - Add topology handlers for `classic_jeongja` (formal upright calligraphy with sharp corner pauses).
  - Add topology handlers for `rounded_diary` (curved, bubbly open loops).
- [ ] **Step 3.4**: In `src/vibefont/vibe/lexicon.py` and `interpreter.py`, add lexicon rules mapping Korean & English prompt keywords ("고시체", "수험생", "정자체", "교과서체", "다이어리체", "동글동글", "shorthand", "block letters", "bubbly") to the respective structural archetype.
- [ ] **Step 3.5**: Run `pytest tests/test_allograph_personas.py` and confirm tests pass (Green).
- [ ] **Step 3.6**: Commit changes with message `feat(hangul): introduce distinct handwriting skeletal allograph personas`.

---

### Task 4: Regenerate Comparison Showcase & Visual Verification

**Files:**
- Modify: `scripts/generate_showcase_60.py`
- Execute: Generate updated visual comparison boards showing:
  - Real pencil texture with microscopic paper tooth graphite adherence.
  - Real flex fountain pen with tine slit bifurcation and authentic liquid dye pooling.
  - Side-by-side comparison of the new skeletal allographs (고시체 vs 정자체 vs 다이어리체 vs 명필 흘림체).
- Update Brain Artifacts: Copy fresh images to artifact directory and update `showcase_60_gallery.md`.

- [ ] **Step 4.1**: Execute showcase script and generate close-up crops of pencil graphite tooth and fountain pen tine slit tracks.
- [ ] **Step 4.2**: Verify that all 128+ tests in test suite pass without warnings or regressions.
- [ ] **Step 4.3**: Copy generated artifacts to brain directory and present visual proof to the user.
