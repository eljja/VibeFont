# Handwriting Realism Gaps (Ligatures → Ink Physics → Contextual Variation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three README "Core Problem" gaps in order — (A) context-aware cursive ligatures, (B) physically-motivated ink effects, (C) context-dependent glyph variation — without ever sacrificing legibility.

**Architecture:** (A) A new pure-geometry module `generate/ligature.py` joins placed glyphs by *reshaping* exit tails / entry heads (never by adding long bridges), re-orders delayed strokes (i-dots, t-crosses), and is consumed by the Latin path and by Hangul cross-jamo joining. (B) Pen physics gains dwell-time pooling, ballpoint paste accumulation/ball-period skips, fibre-directed anisotropic feathering (`render/fluid.py`) and wet-on-wet bleed. (C) Glyph jitter depends on position in word, and connector shapes carry per-writer, per-letter-pair habits.

**Tech Stack:** Python 3, numpy, scipy (`gaussian_filter`, `map_coordinates`), Hershey-Fonts, pytest. No new dependencies.

**Spec:** README.md §"The Core Problem" (items 1–3) + conversation decision 2026-10-24: order 2 → 3 → 1, legibility first (user: "캡차 수준으로 읽기 어려울 정도로 이상해지면 안돼").

## Global Constraints

- Legibility first: ligature code may not move any glyph-body point (outside the connector zone = last/first 20 % of arc length of exit/entry strokes, capped at 0.6 x-height) at all — displacement there must be exactly 0.
- ㄷ must never merge into a single stroke with itself through cross-jamo joining (ㄷ/ㄹ confusion rule from 2026-10-24); joins only happen *between* jamo groups.
- No connector may cross another stroke of the same word/syllable (anti-CAPTCHA guard).
- Units: generator geometry is normalised (x-height = 1.0, baseline y = 0, **y grows downward**, x-height top = −1.0). Pen/raster physics is in page mm and seconds.
- Deterministic for a fixed seed; every random draw goes through the passed `np.random.Generator`.
- All pre-existing tests (60 at baseline) stay green after every task: `.venv\Scripts\pytest.exe -q`.
- Showcase render time (`scripts/generate_showcase_60.py`, baseline 72.3 s) may grow by at most 30 %.
- Out of scope: WebGL2/WebGPU shader port; learned-model ligatures (need licensed data).

## Review Focus

1. Words with capitals, digits or punctuation (`"Hello,"`, `"don't"`, `"A4"`) — expected: no crash, no connector into/out of punctuation or capitals' non-anchored ends. → test in Task A2.
2. One-letter words (`"a"`, `"I"`) and empty strings — expected: unchanged glyph, no join. → test in Task A2.
3. Mixed Hangul+Latin word (`"VibeFont한글"`) — expected: Latin run joined, Hangul run cross-jamo joined, no join across the script boundary. → test in Task A3.
4. Print style (`connectedness < 0.4`) — expected: byte-identical stroke count to today. → test in Task A2.
5. Pens other than the modified one (pencil, brush, broad_nib) — expected: dwell/goop/wet effects do not leak into them. → tests in Tasks B1, B3, B4.

---

## Phase 0

### Task 0: Baseline commit

**Files:**
- Modify: `.gitignore` (add `outputs/`)

- [ ] **Step 1:** Run `.venv\Scripts\pytest.exe -q` → expect `60 passed`.
- [ ] **Step 2:** Add `outputs/` to `.gitignore`.
- [ ] **Step 3: Commit** the pending legibility/showcase work plus this plan:
```bash
git add -A
git commit -m "fix(hangul,latin): restore legibility, split showcase into ko/en; add realism plan"
```

---

## Phase A — Broken Ligatures

### Task A1: Ligature geometry primitives

**Files:**
- Create: `src/vibefont/generate/ligature.py`
- Test: `tests/test_ligature.py`

**Interfaces:**
- Produces:
  - `@dataclass PlacedGlyph: ch: str; strokes: list[np.ndarray]; jitter: float = 0.0`
  - `@dataclass(frozen=True) GlyphAnchors: entry_stroke: int | None; exit_stroke: int | None` — entry point = `strokes[entry_stroke][0]`, exit point = `strokes[exit_stroke][-1]`.
  - `@dataclass JoinInfo: prev_ch: str; next_ch: str; point: tuple[float, float]; turn_deg: float; path: np.ndarray | None = None` (`path` = added connector polyline, `None` for blend joins).
  - `find_anchors(strokes: list[np.ndarray], connector_y: float = -0.56, tol: float = 0.25) -> GlyphAnchors` — entry: leftmost stroke start with `|y - connector_y| <= tol` and `x <= glyph_min_x + 0.25`; exit: rightmost stroke end with same y rule and `x >= glyph_max_x - 0.25`.
  - `join_strokes(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, *, zone: float = 0.20, max_gap: float = 0.45, meet_y: float | None = None) -> np.ndarray | None` — returns `None` if `|a[-1]-b[0]| > max_gap`; otherwise meeting point `m` = midpoint (y replaced by `meet_y` if given), a's tail and b's head displaced toward `m` with smoothstep weight over the zone (weight exactly 0 outside), concatenated `a + b[1:]`, then a 1-D Gaussian (σ = 1.5 samples) applied only to the ±4 samples around the junction.
  - `merge_glyph_internal(strokes: list[np.ndarray]) -> list[np.ndarray]` — consecutive strokes k, k+1: gap ≤ 1e-6 → concatenate; start of k+1 within 0.05 of a point of stroke k → retrace (append reversed segment of k from its end back to that point, then k+1); gap ≤ 0.35 → concatenate directly; else keep separate.
  - `split_delayed(strokes: list[np.ndarray], anchors: GlyphAnchors) -> tuple[list[np.ndarray], list[np.ndarray]]` — delayed = not entry/exit stroke AND `max(y) < -0.9` AND arc length < 0.8.
  - `turn_angle_deg(path: np.ndarray, idx: int, k: int = 3) -> float` and `polylines_intersect(a: np.ndarray, b: np.ndarray, ignore_endpoints: bool = True) -> bool`.

- [ ] **Step 1: Write failing tests** in `tests/test_ligature.py` (helper `glyph(font, ch)` returns `_Font._norm_hershey` strokes of one char):
```python
def test_find_anchors_cursive_lowercase():
    f = _Font("cursive")
    for ch in "acehilmnrsuz":
        a = find_anchors(glyph(f, ch))
        assert a.entry_stroke is not None and a.exit_stroke is not None, ch

def test_join_strokes_meets_smoothly():
    a = line((0, -0.56), (1, -0.56), 21); b = line((1.2, -0.56), (2.2, -0.56), 21)
    m = join_strokes(a, b, np.random.default_rng(0))
    assert m is not None and len(m) == 41
    assert max(turn_angle_deg(m, i) for i in range(3, 38)) <= 20

def test_join_strokes_respects_max_gap():
    a = line((0, -0.56), (1, -0.56), 21); b = line((1.8, -0.56), (2.8, -0.56), 21)
    assert join_strokes(a, b, np.random.default_rng(0)) is None

def test_join_strokes_leaves_glyph_body_untouched():
    # first 80 % of a's arc and last 80 % of b's arc are bit-identical
    ...  # assert np.max(np.abs(m[:17] - a[:17])) <= 1e-9 and same for b tail

def test_merge_glyph_internal_m_is_one_stroke_and_shape_preserved():
    g = glyph(_Font("cursive"), "m"); out = merge_glyph_internal(g)
    assert len(out) == 1
    assert max(point_to_polyline(p, out[0]) for s in g for p in s) <= 0.02

def test_split_delayed_i_and_t():
    f = _Font("cursive")
    for ch in "it":
        g = glyph(f, ch); body, delayed = split_delayed(g, find_anchors(g))
        assert len(delayed) == 1, ch
```
- [ ] **Step 2:** `.venv\Scripts\pytest.exe tests/test_ligature.py -v` → FAIL (`ModuleNotFoundError: vibefont.generate.ligature`).
- [ ] **Step 3:** Implement the interfaces above in `ligature.py`.
- [ ] **Step 4:** Run the file → PASS; run full suite → all green.
- [ ] **Step 5: Commit** `feat(ligature): anchor detection, tail/head blend joins, delayed-stroke split`.

### Task A2: Contextual Latin cursive joining

**Files:**
- Modify: `src/vibefont/generate/procedural.py` (`_Font._latin_word_strokes` L50-118 → split into `place_glyphs` + connect)
- Modify: `src/vibefont/generate/ligature.py` (add `connect_word`)
- Test: `tests/test_ligature.py`

**Interfaces:**
- Consumes: Task A1 API.
- Produces:
  - `_Font.place_glyphs(word: str, rng: np.random.Generator, connectedness: float, neatness: float, profile: WriterProfile | None) -> list[PlacedGlyph]` (today's loop body; sets `PlacedGlyph.jitter = |ln sx| + |ln sy| + |dy| + |d_slant|`).
  - `HIGH_EXIT = frozenset("bovw")` — joins out of these meet at `meet_y = -0.95`; all others meet at the anchors' mean y.
  - `@dataclass ConnectResult: strokes: list[np.ndarray]; joins: list[JoinInfo]`
  - `connect_word(glyphs: list[PlacedGlyph], rng: np.random.Generator, p_join: float, writer_key: int = 0) -> ConnectResult` — per glyph `merge_glyph_internal`, `split_delayed`; adjacent glyphs joined with probability `p_join` via `join_strokes`; delayed strokes appended after all body strokes in glyph order. Glyphs whose char is not `str.islower()` never join on either side.
  - `_latin_word_strokes` keeps its signature; when `rng is not None and connectedness >= 0.4` it returns `connect_word(..., p_join=clip((connectedness-0.35)/0.45, 0, 0.97)).strokes`, else today's flattened strokes.

- [ ] **Step 1: Write failing tests:**
```python
def _strokes(word, conn, seed):
    return _Font("cursive")._latin_word_strokes(word, rng=np.random.default_rng(seed), connectedness=conn, neatness=0.7)[0]

def test_cursive_word_pen_lifts_drop():
    hi = np.mean([len(_strokes("minimum", 0.95, s)) for s in range(5)])
    lo = np.mean([len(_strokes("minimum", 0.30, s)) for s in range(5)])
    assert hi <= 4.5 and hi <= 0.35 * lo

def test_print_style_stroke_count_unchanged():
    assert len(_strokes("minimum", 0.2, 0)) == 17

def test_delayed_strokes_drawn_last():
    st = _strokes("kitten", 0.95, 0)
    is_del = [s[:, 1].max() < -0.9 and arclen(s) < 0.8 for s in st]
    assert is_del[-3:] == [True, True, True] and not any(is_del[:-3])

def test_high_join_after_o_and_baseline_join_after_a():
    f = _Font("cursive")
    for seed in range(5):
        rng = np.random.default_rng(seed)
        r_on = connect_word(f.place_glyphs("on", rng, 0.95, 0.7, None), rng, 0.97)
        r_an = connect_word(f.place_glyphs("an", rng, 0.95, 0.7, None), rng, 0.97)
        if r_on.joins: assert r_on.joins[0].point[1] <= -0.85
        if r_an.joins: assert -0.75 <= r_an.joins[0].point[1] <= -0.40

def test_join_turn_angles_bounded_over_corpus():
    turns = [...]  # all JoinInfo.turn_deg from "the quick brown fox jumps over the lazy dog" seeds 0..4
    assert max(turns) <= 75 and np.mean(turns) <= 40

def test_edge_words_capitals_punctuation_single_letters():
    for w in ["Hello,", "don't", "A4", "a", "I", ""]:
        st, width = _Font("cursive")._latin_word_strokes(w, rng=np.random.default_rng(1), connectedness=0.95, neatness=0.7)
        assert width > 0
    # no join touches a capital or punctuation glyph
    r = connect_word(_Font("cursive").place_glyphs("Hello,", np.random.default_rng(1), 0.95, 0.7, None), np.random.default_rng(1), 0.97)
    assert all(j.prev_ch.islower() and j.next_ch.islower() for j in r.joins)
```
- [ ] **Step 2:** Run → FAIL (`AttributeError: place_glyphs` / pen-lift assertion).
- [ ] **Step 3:** Implement `place_glyphs`, `HIGH_EXIT`, `ConnectResult`, `connect_word`, and wire into `_latin_word_strokes`.
- [ ] **Step 4:** Run file → PASS; full suite → green.
- [ ] **Step 5: Commit** `feat(latin): contextual cursive ligatures with high/baseline joins and delayed strokes`.

### Task A3: Hangul cross-jamo joining

**Files:**
- Modify: `src/vibefont/generate/hangul.py` (`syllable_strokes` L582-653: keep cho/jung/jong placed lists separate; replace final `_soft_connect`)
- Test: `tests/test_ligature.py`

**Interfaces:**
- Consumes: `join_strokes`, `JoinInfo`, `polylines_intersect`, `turn_angle_deg`.
- Produces:
  - `join_jamo_groups(groups: list[list[np.ndarray]], connectedness: float, rng: np.random.Generator | None) -> tuple[list[np.ndarray], list[JoinInfo]]` — inside each group apply today's `_soft_connect`; between group g's last stroke and group g+1's first stroke, with probability `clip(1.2*(connectedness-0.45), 0, 0.8)`: gap ≤ 0.30 → blend join (`join_strokes(max_gap=0.30)`); 0.30 < gap ≤ 0.70 and `connectedness >= 0.7` → hairline connector (quadratic curve, mid-offset ≤ 0.08 perpendicular, recorded in `JoinInfo.path`) **only if** it intersects no other stroke of the syllable; otherwise no join.
  - `syllable_strokes_with_joins(char, rng, connectedness, neatness, profile) -> tuple[list[np.ndarray], list[JoinInfo]]`; `syllable_strokes` returns `[0]` of it.

- [ ] **Step 1: Write failing tests:**
```python
def test_join_jamo_groups_never_joins_inside_a_group_beyond_touching():
    g1 = [line((0,0),(1,0),11), line((0,0.5),(1,0.5),11)]   # ㄷ-like: 2 strokes, gap 0.5
    g2 = [line((1.1,0.5),(1.1,-1),11)]
    out, joins = join_jamo_groups([g1, g2], 0.95, np.random.default_rng(0))
    assert len(out) >= 2 and len(joins) <= 1

def test_hangul_cursive_reduces_pen_lifts():
    txt = "가나다라마바사아자차"
    def mean_strokes(conn): return np.mean([len(syllable_strokes(c, rng=np.random.default_rng(i), connectedness=conn, neatness=0.7)) for i in range(10) for c in txt])
    assert mean_strokes(0.9) <= 0.85 * mean_strokes(0.3)

def test_hangul_connectors_never_cross_other_strokes():
    for seed in range(20):
        for ch in "강물이흐르는밤":
            st, joins = syllable_strokes_with_joins(ch, np.random.default_rng(seed), 0.95, 0.7, None)
            for j in joins:
                if j.path is not None:
                    assert not any(polylines_intersect(j.path, s) for s in st if not shares_endpoint(s, j.path))

def test_mixed_script_word_no_join_across_boundary():
    st, _ = _Font("cursive").word("VibeFont한글", rng=np.random.default_rng(0), connectedness=0.95, neatness=0.7)
    # no stroke spans the x-gap between the Latin run's max x and the Hangul run's min x
    ...
```
- [ ] **Step 2:** Run → FAIL (`ImportError: join_jamo_groups`).
- [ ] **Step 3:** Implement; ensure `test_hangul_kid_mode_vs_master_cursive_and_calligraphy` and the 다/라, 모/오 legibility tests in `tests/test_korean_stroke.py` stay green.
- [ ] **Step 4:** Run file → PASS; full suite → green.
- [ ] **Step 5: Commit** `feat(hangul): cross-jamo cursive joins with non-crossing hairline connectors`.

### Task A4: Phase A visual gate

- [ ] **Step 1:** `.venv\Scripts\python.exe scripts/generate_showcase_60.py` → completes; note elapsed time (≤ 94 s).
- [ ] **Step 2:** Inspect `outputs/showcase/en/en_master_01.png`, `en_kid_01.png`, `ko/ko_master_03.png`, `ko/ko_kid_01.png`: master cursive visibly joined, kid/print unjoined, no letter unreadable. Record findings in the PR/commit message.
- [ ] **Step 3: Commit** `chore(showcase): regenerate after ligature phase` (code only; outputs are git-ignored).

---

## Phase B — Analog Ink Physics

### Task B1: Dwell-time ink pooling

**Files:**
- Modify: `src/vibefont/render/pens.py` (new helper + fountain/gel/ballpoint/marker branches)
- Test: `tests/test_ink_physics.py`

**Interfaces:**
- Produces: `_dwell_pool(ink: Ink, radius_mm: float = 0.25, tau_s: float = 0.25) -> np.ndarray` — per point, `1 - exp(-T_local / tau_s)` where `T_local` = time the pen spent within ±`radius_mm` of arc length around that point. Gains: fountain `dens *= 1 + 0.9*pool`, `w *= 1 + 0.25*pool`; gel/marker `dens *= 1 + 0.6*pool`; ballpoint `dens *= 1 + 0.30*pool`; pencil/brush/broad_nib unchanged.

- [ ] **Step 1: Write failing tests** (helper `ink_with_pause(pause_s)`: 20 mm straight line at 25 mm/s with the pen held still for `pause_s` at x = 10 mm, plus a second pause of 0.1 s at x = 5 mm):
```python
def test_longer_pause_pools_more_ink_fountain():
    ink = ink_with_pause(0.6)
    d = pen_samples(ink, PenSpec(type="fountain", width_mm=0.5), StyleSpec(), np.random.default_rng(0)).dens
    assert d[idx_at(ink, 10.0)] >= 1.2 * d[idx_at(ink, 5.0)]

def test_dwell_pooling_does_not_leak_into_pencil():
    ink = ink_with_pause(0.6)
    d = pen_samples(ink, PenSpec(type="pencil", width_mm=0.5), StyleSpec(), np.random.default_rng(0)).dens
    assert d[idx_at(ink, 10.0)] <= 1.15 * np.median(d)
```
- [ ] **Step 2:** Run → first test FAILS (speed-only model gives equal density at both pauses).
- [ ] **Step 3:** Implement `_dwell_pool` and the per-pen gains.
- [ ] **Step 4:** Run file → PASS; full suite → green.
- [ ] **Step 5: Commit** `feat(pens): dwell-time ink pooling`.

### Task B2: Fibre-directed anisotropic feathering

**Files:**
- Create: `src/vibefont/render/fluid.py`
- Modify: `src/vibefont/render/raster.py:228-231` (replace isotropic feather)
- Test: `tests/test_ink_physics.py`

**Interfaces:**
- Produces:
  - `fibre_field(shape: tuple[int, int], rng: np.random.Generator, px_per_mm: float, mean_angle_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]` — `(theta, strength)`; theta = mean angle (random if None) + Gaussian-smoothed noise (σ = 0.6 mm, ±35°); strength in [0, 1] from smoothed noise.
  - `feather_anisotropic(D: np.ndarray, theta: np.ndarray, strength: np.ndarray, amount: float, px_per_mm: float, iters: int = 3) -> np.ndarray` — each iteration samples D at ±step along (cosθ, sinθ) with `scipy.ndimage.map_coordinates(order=1)`, step = 0.06 mm, and adds `amount * strength * max(avg - D, 0)`; `amount == 0` returns `D` unchanged (same object values).

- [ ] **Step 1: Write failing tests:**
```python
def test_feather_spreads_along_fibres():
    D = gaussian_dot(64, sigma=3.0); th = np.zeros_like(D); s = np.ones_like(D)
    out = feather_anisotropic(D, th, s, amount=0.8, px_per_mm=11.8)
    Ixx, Iyy = second_moments(out)
    assert Ixx / Iyy >= 1.4

def test_feather_amount_zero_is_identity():
    D = gaussian_dot(64, 3.0)
    assert np.array_equal(feather_anisotropic(D, np.zeros_like(D), np.ones_like(D), 0.0, 11.8), D)

def test_feather_mass_bounded():
    D = gaussian_dot(64, 3.0)
    out = feather_anisotropic(D, np.zeros_like(D), np.ones_like(D), 0.8, 11.8)
    assert D.sum() <= out.sum() <= 1.25 * D.sum()
```
- [ ] **Step 2:** Run → FAIL (`ModuleNotFoundError: vibefont.render.fluid`).
- [ ] **Step 3:** Implement `fluid.py`; in `render_page` replace step 3 with `feather_anisotropic(D, *fibre_field(D.shape, rng, px), amount=feather*0.65, px_per_mm=px)`.
- [ ] **Step 4:** Run file → PASS; full suite → green (incl. `test_physical_pen_dynamics_and_contrast`).
- [ ] **Step 5: Commit** `feat(render): fibre-directed anisotropic ink feathering`.

### Task B3: Ballpoint paste accumulation and ball-period skips

**Files:**
- Modify: `src/vibefont/render/pens.py` (ballpoint branch L111-129: replace random `corner_blob` and arc-noise skip)
- Test: `tests/test_ink_physics.py`

**Interfaces:**
- Produces: `_ballpoint_goop(ink: Ink, rng: np.random.Generator, accum_per_mm: float = 0.06, release_curv: float = 2.2, ball_d_mm: float = 0.7) -> tuple[np.ndarray, np.ndarray]` → `(blob, skip)`, each in [0, 1]. `blob`: paste accumulates `accum_per_mm` per mm travelled (cap 1), released (value written at that point, accumulator reset) where curvature > `release_curv` or at stroke end. `skip`: `clip((v-40)/150, 0, 0.75) * periodic` where `periodic` has period `π * ball_d_mm` along arc length with per-stroke random phase and ±10 % jitter. Ballpoint `dens += 0.9*blob`, `w *= 1 + 0.22*blob`, `dryness = max(dryness_existing_downstroke_term, skip)`.

- [ ] **Step 1: Write failing tests:**
```python
def test_goop_grows_with_travel_before_release():
    ink = zigzag_ink(corners_at_mm=[5.0, 25.0], speed=60)  # sharp 90° corners
    blob, _ = _ballpoint_goop(ink, np.random.default_rng(0))
    assert blob[idx_at(ink, 25.0)] > blob[idx_at(ink, 5.0)] + 0.3

def test_skip_increases_with_speed():
    _, s_fast = _ballpoint_goop(line_ink(60, speed=200), np.random.default_rng(0))
    _, s_slow = _ballpoint_goop(line_ink(60, speed=30), np.random.default_rng(0))
    assert s_fast.mean() >= 3 * s_slow.mean() + 0.02

def test_skip_has_ball_period():
    ink = line_ink(60, speed=200)
    _, s = _ballpoint_goop(ink, np.random.default_rng(0))
    assert abs(dominant_period_mm(s, ink) - np.pi * 0.7) <= 0.4
```
- [ ] **Step 2:** Run → FAIL (`ImportError: _ballpoint_goop`).
- [ ] **Step 3:** Implement and wire into ballpoint branch (keep the existing dry-start gate).
- [ ] **Step 4:** Run file → PASS; full suite → green.
- [ ] **Step 5: Commit** `feat(pens): ballpoint paste accumulation and ball-rotation skips`.

### Task B4: Wet-on-wet overlap bleed

**Files:**
- Modify: `src/vibefont/render/raster.py` (`stamp_stroke` returns its bbox buffer; `render_page` keeps a last-deposit time map for liquid pens)
- Modify: `src/vibefont/render/fluid.py` (add `wet_overlap_bleed`)
- Test: `tests/test_ink_physics.py`

**Interfaces:**
- Produces:
  - `stamp_stroke(...) -> tuple[int, int, np.ndarray] | None` — `(y0, x0, B)` of the stroke's own density buffer (already added into `D`), `None` if nothing stamped.
  - `wet_overlap_bleed(D_old: np.ndarray, B_new: np.ndarray, age_s: np.ndarray, window_s: float = 1.5, gain: float = 0.35, sigma_px: float = 1.4) -> np.ndarray` — `gain * gaussian_filter(min(D_old, B_new) * clip(1 - age_s/window_s, 0, 1), sigma_px)`; zero where `age_s >= window_s`.
  - Liquid pens = `{"fountain", "gel", "marker", "brush", "broad_nib"}`; stroke time = mean `T` of the stroke.

- [ ] **Step 1: Write failing tests:**
```python
def test_bleed_zero_outside_window():
    D_old = np.ones((20, 20)); B = np.ones((20, 20))
    assert wet_overlap_bleed(D_old, B, np.full((20, 20), 5.0)).sum() == 0

def test_quick_crossing_is_darker_than_slow_crossing():
    dark_quick = intersection_density(render_cross(pen="fountain", dt=0.3))
    dark_slow = intersection_density(render_cross(pen="fountain", dt=3.0))
    assert dark_quick >= 1.10 * dark_slow

def test_ballpoint_has_no_wet_bleed():
    assert intersection_density(render_cross(pen="ballpoint", dt=0.3)) == pytest.approx(
        intersection_density(render_cross(pen="ballpoint", dt=3.0)), rel=0.02)
```
(`render_cross` builds a `Page(60, 40, lines=[Ink...])` with two perpendicular 20 mm strokes whose start times differ by `dt`, renders with `render_page(page, VibeSpec(pen=PenSpec(type=pen)), dpi=300, rng=default_rng(0))` and returns the image.)
- [ ] **Step 2:** Run → FAIL (`ImportError: wet_overlap_bleed`).
- [ ] **Step 3:** Implement; `Tm` map initialised to `-inf`, updated over each stroke's bbox where `B > 0.05`.
- [ ] **Step 4:** Run file → PASS; full suite → green.
- [ ] **Step 5: Commit** `feat(render): wet-on-wet overlap bleed for liquid inks`.

### Task B5: Phase B visual + performance gate

**Files:**
- Create: `scripts/render_physics_compare.py` — renders `"Quickly feathered ink 잉크 번짐"` with each of the 7 pens at 600 dpi to `outputs/physics/{pen}.png` plus a 2×4 contact sheet `outputs/physics/sheet.png`.

- [ ] **Step 1:** Run the script → 8 PNGs produced.
- [ ] **Step 2:** Inspect `fountain.png` (directional feather, pooled stroke ends, darker crossings), `ballpoint.png` (blobs after long runs, periodic skips on fast strokes), `pencil.png` (unchanged character).
- [ ] **Step 3:** Run `scripts/generate_showcase_60.py`; elapsed ≤ 94 s.
- [ ] **Step 4: Commit** `chore(physics): comparison sheet script`.

---

## Phase C — Contextual Glyph Variation

### Task C1: Position-in-word variation

**Files:**
- Modify: `src/vibefont/generate/procedural.py` (`place_glyphs`; Hangul loop in `_Font.word` L152-175)
- Test: `tests/test_context_variation.py`

**Interfaces:**
- Produces: `position_factor(i: int, n: int) -> float` in `procedural.py` — `0.85` for `i == 0` (when `n >= 2`), `1.30` for `i == n-1` (when `n >= 3`), else `1.0`. Multiplies the per-glyph `clumsy` before sampling sx/sy/dy/d_slant (Latin) and sx_jit/sy_jit/dy_jit/rot_jit (Hangul). Existing clip bounds unchanged (legibility).

- [ ] **Step 1: Write failing tests:**
```python
def test_position_factor_values():
    assert position_factor(0, 5) == 0.85 and position_factor(4, 5) == 1.30 and position_factor(2, 5) == 1.0
    assert position_factor(0, 1) == 1.0

def test_word_final_glyph_looser_than_initial():
    f = _Font("cursive"); first, last = [], []
    for s in range(100):
        g = f.place_glyphs("letters", np.random.default_rng(s), 0.95, 0.5, None)
        first.append(g[0].jitter); last.append(g[-1].jitter)
    assert np.mean(last) >= 1.3 * np.mean(first)
```
- [ ] **Step 2:** Run → FAIL (`ImportError: position_factor`).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run file → PASS; full suite → green.
- [ ] **Step 5: Commit** `feat(variation): word-position dependent glyph looseness`.

### Task C2: Per-writer letter-pair connector habits

**Files:**
- Modify: `src/vibefont/generate/ligature.py` (`connect_word` uses `writer_key`)
- Modify: `src/vibefont/generate/procedural.py` (`_latin_word_strokes` passes `writer_key = zlib.crc32(repr(profile).encode())`, 0 when `profile is None`)
- Test: `tests/test_context_variation.py`

**Interfaces:**
- Produces: `pair_bias(prev_ch: str, next_ch: str, writer_key: int) -> tuple[float, float]` in `ligature.py` — deterministic `(dx, dy)` in `[-0.06, 0.06]²` from a stable hash (`zlib.crc32(f"{writer_key}:{prev_ch}{next_ch}")`, not Python `hash`). `connect_word` adds it to the meeting point, plus rng noise with σ = 0.012.

- [ ] **Step 1: Write failing test:**
```python
def test_connector_habit_consistent_within_pair_distinct_across_pairs():
    f = _Font("cursive"); pairs = ["an", "er", "im", "us"]; means, stds = [], []
    for p in pairs:
        ys = []
        for s in range(30):
            rng = np.random.default_rng(s)
            r = connect_word(f.place_glyphs(p, rng, 0.95, 0.9, None), rng, 0.97, writer_key=7)
            if r.joins: ys.append(r.joins[0].point[1])
        means.append(np.mean(ys)); stds.append(np.std(ys))
    assert np.mean(stds) <= 0.5 * np.std(means)
```
- [ ] **Step 2:** Run → FAIL (pair means equal, all anchors at y = −0.56).
- [ ] **Step 3:** Implement `pair_bias` and wire it.
- [ ] **Step 4:** Run file → PASS; full suite → green.
- [ ] **Step 5: Commit** `feat(ligature): per-writer letter-pair connector habits`.

### Task C3: Final verification

- [ ] **Step 1:** `.venv\Scripts\pytest.exe -q` → all pass (60 + new).
- [ ] **Step 2:** `scripts/generate_showcase_60.py` → ≤ 94 s; inspect `ko/gallery_ko_all_30.png` and `en/gallery_en_all_30.png` for legibility.
- [ ] **Step 3:** Update README §"The Core Problem" with a short "Status" line per item pointing at the implementing modules.
- [ ] **Step 4: Commit** `docs: record realism-gap status in README`.
