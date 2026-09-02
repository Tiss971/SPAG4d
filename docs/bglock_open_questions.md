# bglock — audit backlog and parallel work plan

Working document for a Claude Code session. Candidate bugs, tests, and fixes for the
background-locked depth stabilization stage of SPAG4D. Nearly everything below is now closed —
see §12 for the compact status table; the numbered sections keep the original problem framing plus
a short result summary. Full evidence trails live in the linked benchmark docs/scripts, not inline.

Section numbers are stable identifiers — other docs in this repo link to them by number (e.g.
`§8.7`, `§4.1`) — so they are **not** reordered to match reading priority; use the map below or
§12 to jump straight to a topic instead of reading top to bottom.

## Contents (what each section actually covers)

**Meta / how to use this doc** — §1 rules for working this backlog (verify before acting; closes
here, not with a new fix); §2 shared-GPU etiquette; §3 the original task-packet → section mapping
(historical, all packets closed).

**Core propagation bugs** — §4 P0 confidence bugs in flow-propagation (4.1 confidence absorbing at
zero, 4.2 `prev_confidence` not warped by flow — both fixed 2026-08-17, one fix under the hood of
the other).

**Reference background plate** — §5 `D_ref` (the one-time temporal-median background depth) plate
*coverage*: thin-sample pixels, occluder holes, staleness — fixed 2026-08-18.

**Methodology for validating with no ground truth** — §6, the largest section: 6.0 the "freezing
gives perfect stability for free" trap that invalidates any stability number read alone; 6.1 stride
invariance; 6.2 time-reversal consistency (open); 6.3 mask-injection proxy ground truth; 6.4
synthetic-motion ground truth; 6.5 free consistency signals (open, not run); 6.6 the ablation-ladder
reporting frame; 6.7 instrument-error floors; 6.8 an open watch-item; 6.9 what a 30s clip can and
can't support statistically.

**Masking** — §7 SAM3/`detect.py` mask coverage, false-positive audit, tracking-budget behavior.

**Compositing / point-cloud output** — §8: 8.1 feathering producing geometrically-invalid "phantom
slab" depth (fixed, opt-in `SPAG_HARD_DEPTH_CUTOVER`); 8.2 NaN propagation (audited, already
correct); 8.3 dilation radius should scale with resolution (open); 8.4 ERP-anisotropic kernels
(open); 8.5 alpha=1.0 hole-forcing; 8.6 `freeze_bg` freezing color along with geometry (fixed,
`freeze_bg_live_color`); 8.7 the scene01 "waiter" depth-bleed trail — **reopened**, the
longest-running unresolved item in this doc.

**Instrumentation** — §9 the diagnostics/CSV-dump pass.

**Depth backends** — §10: 10.1 the shared backend contract; 10.2 outlier contamination of the
affine fit (fixed); 10.3 PaGeR cube-face seam steps (open); 10.4 PaGeR indoor/outdoor
reclassification per frame (audited, defensive fix applied); 10.5 DA360 residual-vs-latitude
(resolved — content confound, not a bug); 10.6 MapAnything migration (plan only, not executed).

**Sanity check** — §11 confirming the camera is actually static (confirmed).

**Status rollup** — §12 priority index: every item above in one flat table, sorted by priority and
current status, not by section order — start here if you just want "what's still open."

---

## 0. Context

**What the pipeline does.** Fixed-camera 360° equirectangular (ERP) video → per-frame monocular
depth → stabilization → dynamic point cloud / 3D Gaussian splats. Static background, a small
number of moving people/objects.

**What bglock does.** A monocular depth model has no memory, so static background pixels still get
a slightly different depth every frame → flicker. bglock fixes the background to a reference depth
map `D_ref` and flow-propagates only the dynamic regions:

1. **Align** — per-frame scale/shift fit against static pixels.
2. **Mask** — SAM3 marks dynamic regions per frame.
3. **Propagate** — `propagate_depth_via_flow` warps last frame's stabilized depth by WAFT
   bidirectional flow, blends with the fresh estimate under a confidence map.
4. **Composite** — `composite_bg_locked` mixes `alpha * object_depth + (1-alpha) * D_ref` with a
   dilated + feathered alpha.

`D_ref` is a per-pixel temporal **median of the RGB frames** (excluding moving content), depth
model run **once** on that clean plate.

**Components.**  `spag4d/da360_model.py` (DA360 backend,
scale-invariant disparity inverted to depth), `spag4d/pager_model.py` (PaGeR backend, ERP→cubemap→
per-face→stitch), WAFT (optical flow).


---

## 1. Rules

**Verify before acting.** Check every candidate against live code first; several early premises in
this doc turned out false on inspection. "Checked, not a problem, here's why" closes an item.

**Do not implement speculative fixes** for premises that don't hold. **Report per item**: premise
held y/n, evidence, action taken.

---

## 2. GPU usage

Shared server. Before launching:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
CUDA_VISIBLE_DEVICES=<free_idx> python <script>
```
One task packet per GPU; re-check occupancy immediately before launch; record which GPU each run
used so timing is comparable. `detect.py`/current scripts follow `CUDA_VISIBLE_DEVICES` — no code
change needed.

---

## 3. Parallel task packets (historical — work is done, kept for packet→section mapping)

| ID | Goal | Section |
|---|---|---|
| T1 | Confidence semantics in propagation | §4.1, §4.2 |
| T2 | Backend validity + outliers | §10.1–10.5 |
| T3 | Diagnostics pass | §9, §10.5, §11 |
| T4 | `D_ref` plate coverage | §5 |
| T5 | Compositing | §8 |
| T6 | Mask coverage in `detect.py` | §7 |
| T7 | Validation harness | §6 |
| T8 | Backend contract refactor | §10.1 |

All waves closed; see §12.

---

## 4. P0 — suspected bugs

### 4.1 `running_confidence` absorbing at zero
`running_confidence = min(frame_confidence, prev_confidence * 0.85)` — once a pixel hits 0 it's
pinned there forever, falling back to the raw per-frame estimate for the rest of the clip.

> **Fixed 2026-08-17.** Real bug differed from the premise: state was pinned to exactly
> `{0.0, 0.85}`, never actually decaying. Now a per-pixel **age** field:
> `confidence = frame_confidence * max(conf_floor, decay**age)` (`SPAG_CONF_DECAY=0.85`,
> `SPAG_CONF_FLOOR=0.5`; `SPAG_CONF_LEGACY=1` restores the old behavior). The old path suppressed
> ~31% of real foreground motion (`fid_ratio` 0.692 → 0.923). Synthetic validation in §6.4.

### 4.2 `prev_confidence` not warped by flow
Confidence describes a surface point, not a grid location; reading it un-warped means a pixel just
uncovered by a moving person inherits the person's edge confidence instead of the background's.

> **Fixed 2026-08-17, folded into §4.1's fix.** The age field rides in the same batched
> `warp_backward_multi(_torch)` call that already warps `depth_prev_final` — free correctness, zero
> extra VRAM.

Backend-specific bugs that surfaced during this audit (outlier contamination of the affine fit,
PaGeR's CLIP reclassification) are backend properties, not propagation bugs — they moved to §10.2
and §10.4 to keep this section's split consistent (P0 propagation-state bugs only).

---

## 5. `D_ref` — plate coverage

Median-RGB-plate is sound (one coherent inference beats stitching per-pixel samples from N
independently-scaled maps). Risk moved from depth to plate *coverage*: thin-sample pixels can land
on motion blur/mask edges; permanent occluders create holes; no refresh means sun/furniture drift
is never corrected.

> **Measured + fixed 2026-08-18.** Coverage is good (96.7% mean, 0.34% zero-sample, already
> NaN-guarded) and the <2% thin-coverage tail is **strongly clustered** — one connected component
> holds 85% of it, i.e. a single recurring occluded region rather than scattered noise, which is
> what made the fix cheap. `compute_temporal_median` gained `min_sample_frames` (opt-in via
> `SPAG_DREF_MIN_SAMPLES=K`), giving below-threshold pixels the inpaint/NaN treatment zero-sample
> pixels already had; default `0` preserves old behavior. **Not done:** two-pass plate build, and
> an end-to-end run on a clip with real thin coverage (the smoke-test clip never tripped the
> threshold, so integration is confirmed but the effect isn't).

---

## 6. Validation without ground truth

No GT depth exists. Clips are ~30s (~900 frames), frame-stepping available.

### 6.0 The trap: every stability metric is gamed by freezing

`output = D_ref` for every pixel scores perfectly on any stability metric — background flicker of
0.0000 is true by construction, not evidence. **No stability number without a paired fidelity
number** showing motion was preserved (temporal-derivative magnitude vs. raw; flatter-not-smoother
= motion suppression). This governs every result below.

### 6.1 Stride invariance
True depth at a timestamp doesn't depend on frame stride. Run at stride 1/2/3/5, compare shared
timestamps — disagreement is propagation error measured against nothing but itself; also tests
whether the 0.85 decay constant is fps-coupled.

> **Done 2026-08-18.** Strides 1/2/3/5 compared at 6 shared timestamps: **no measurable growth in
> disagreement at any stride** (flat to ±0.0003m over up to 30 propagation steps), so bounded-drift
> holds on real data. The disagreement that exists is present from frame 0, before any propagation
> — each stride builds `D_ref` from a different frame subset, a plate-composition confound rather
> than drift. **fps-coupling still open**: that confound dominates any subtler decay signal;
> isolating it needs a pinned shared plate.

### 6.2 Time reversal
Run forward and on the reversed sequence, compare depth per timestamp — exposes the disocclusion
path (forward-disocclusion = backward-occlusion). **Open — not run.**

### 6.3 Mask injection — proxy ground truth for the dynamic path
Inject a genuinely-static background patch into the dynamic mask so bglock runs the full
flow-propagation blend on it. The correct answer is known: `D_ref`. Converts the untestable dynamic
branch into a measurable one; vary by latitude for the latitude-resolved error profile.

> **Done 2026-08-18.** `SPAG_MASK_INJECT` (JSON `[lat_frac, lon_frac, half_px]`) OR's patches into
> `lock_mask` per frame without mutating the underlying masks; scored by
> `scripts/mask_injection_analysis.py`. One 300-frame run (`bg_depth_mean≈5.18m`), 3 latitudes:
>
> | patch | RMSE (m) | mean\|e\| (m) | p95\|e\| (m) | max\|e\| (m) |
> |---|---|---|---|---|
> | north_pole (0.10) | 0.071 | 0.067 | 0.106 | 0.147 |
> | equator (0.50) | 0.561 | 0.445 | 1.046 | 2.044 |
> | south_pole (0.90) | 0.013 | 0.010 | 0.023 | 0.035 |
>
> Error is **not latitude-uniform**: ~9% mean at the equator vs ~1.3%/0.2% at the poles, which are
> low-texture/near-zero-flow and therefore the easy case, not general accuracy evidence. The
> equator number (~0.44m mean, 2.04m max) is the first hard accuracy bound on the dynamic path,
> where real motion lives — so don't calibrate FB-thresholds at the poles. **Not done:**
> object-adjacent feather-zone variant (§8.1).

### 6.4 Synthetic objects — the only real ground truth obtainable
Composite a rendered object with known depth/trajectory into the plate. Priority: **radial motion**
(invisible to 2D flow, the only test of the decay constant), then lateral, seam-crossing, pole
transit.

> **Closed 2026-08-18.** Synthesized at the `propagate_depth_via_flow` level, four arms (raw /
> production decay 0.85+0.5 / legacy / frozen). On **radial** motion — zero flow by construction,
> the case this mechanism exists for — decay scores RMSE 0.198m vs. legacy 0.907m and frozen
> 3.459m, i.e. **4.6x better than legacy**. A `conf_floor` sweep (0.1–0.95) is convex with
> production's 0.5 right at the elbow (0.9 is 6x worse). On **lateral** motion all four arms track
> within ~1cm, and **combined** radial+lateral stays within ~4% of radial-only — so decay/floor is
> neither needed nor harmful where flow can already see the motion, and the two axes don't
> cross-talk. Seam-crossing/pole-transit and two-plane feathering (§8.1) deliberately out of scope.

### 6.5 Free consistency signals already in the pipeline — open, not run
PaGeR normals vs. depth-gradient agreement; ordinal (which-is-nearer) consistency over time;
rigid-object internal shape stability; flow/depth residual on static pixels.

### 6.6 Ablation ladder is the reporting frame
raw → affine-only → +bglock → +flow-propagation → +decay, each rung showing stability improves
without fidelity collapsing (§6.0).

> **Partial 2026-08-18.** A 3/5-rung slice (noprop/legacy/decay); `raw`/`affine-only` not run. It
> caught §6.0's trap live: on one clip `legacy` posts the *best*-looking stability number
> (`fg_depth_cv` 0.131 vs decay's 0.158) at only 0.69 fidelity — buying stability by suppressing
> 31% of real motion — while decay looks worse but holds 0.92. On the second clip neither arm
> suppresses motion at all. So the mechanism is stability-positive with no systematic fidelity
> cost, and reading the stability column alone would have picked the wrong config.

### 6.7 Instrument-error floors
fp16 quantization step (~0.016 near 20m) vs. the claimed 0.037 flicker figure; whether outliers
(§10.2) were excluded when that figure was computed.

> **Done 2026-08-18.** The original comparison was apples-to-oranges: fp16 has constant *relative*
> precision (~0.0008 of depth) and `bg_depth_cv` is dimensionless. Put on the same footing,
> **3/10 baseline clips sit at or below the fp16 floor** — their measured "flicker" is
> indistinguishable from float noise — while the other 7 sit clearly above it (2.8–31x). Caveat:
> that baseline predates the §10.2 outlier fix and hasn't been re-run. The "0.037" headline figure
> matches no number in the baseline JSON; treat it as unverified.

### 6.8 Watch it — open
Render from a novel viewpoint as video + a `|depth_t − depth_{t−1}|` heatmap video. Feathering
smear and mask-edge popping are obvious to the eye, near-invisible to per-pixel metrics.

### 6.9 What a 30s clip supports
900 frames covers a stride sweep (1/2/3/5) and ~7 chunks against a 128-frame tracking cap; not
enough for slow-drift EMA-refresh effects, which need longer footage.

---

## 7. Masks and mask coverage

**False positives** (auto-exposure shifts, compression artifacts, seam wobble, SAM3
over-extension) silently break the zero-flicker guarantee and the metric can't see them. Proposed
fix: auto-demote objects whose in-mask flow stays below threshold for N frames.
**False negatives** (foliage, water, screens) are an accepted trade-off, should be quantified via
inverse audit (high flow + static flag).
**Coverage gaps:** does `max_frame_num_to_track=128` split symmetrically? Does chunking reset
`depth_prev_final`/`running_confidence` at seams? Is `len(seen)<10` still the drop threshold?

> **Code-audited 2026-08-18 (no new pipeline run).** The whole coverage-gap premise is false
> against live code: `max_frame_num_to_track=128` is **not** the default (unset
> `SPAG_SAM3_MAX_FRAMES` means the whole clip, no cap), so the "symmetric split" question is moot;
> where a cap *is* opted in, chunked `propagate_in_video` already re-seeds tracked objects at
> window boundaries, so masks can't stop past it; and the predicted depth-state reset doesn't apply
> since `PropagationState` is driven continuously by the outer per-frame loop, on a separate axis
> from mask chunking. One stale fact corrected: the drop threshold is `min_times_seen=3`
> (`video.py:1880`), not `len(seen)<10`. **Not done:** false-negative inverse audit and the
> auto-demote fix — both need a real masks+flow run.
>
> **Follow-up, closed:** the missing default cap does *not* explain `trop_long_embouteillage`'s
> unmet <16GB VRAM target (see `.claude/B1_VRAM_TIME_REDUCTION_PLAN.md`). VRAM is SAM3-bound by
> memory-bank growth over frame count, and chunking was tested then **rejected by explicit user
> decision (2026-07-31)**: re-seeded windows on one session don't free memory-bank state, only
> fresh sessions would, which reintroduces the tracking-continuity regression this section's fix
> exists to avoid. Accepted limitation.

---

## 8. Compositing and point-cloud consequences

### 8.1 Feathering creates geometrically invalid depth
Across large discontinuities (person at 2m, wall at 10m) a linear blend emits mid-range values
with no corresponding surface — a visible smear in the point cloud.

> **Confirmed, fixed, benchmarked 2026-08-18.** Premise held: `composite_bg_locked(_torch)` is a
> literal `alpha*object_depth + (1-alpha)*depth_ref` blend, and the default ~28px feather band emits
> pure-interpolation depth spanning 2.6–9.4m for a 2m/10m step. **Fix, opt-in:
> `SPAG_HARD_DEPTH_CUTOVER=1`** — alpha stays soft so color/opacity still feather smoothly, but the
> depth *value* is nearest-source selected (`alpha>=0.5 → object_depth else depth_ref`) rather than
> interpolated; synthetically this takes phantom-depth pixels in the band from 4800/4800 to 0/4800.
> On real clips the effect is confined to 0.8–2.1% of frame (mean ~2–3cm, ~1.8m tail), invisible to
> whole-frame stability metrics. **Verdict: keep opt-in** — a real fix, but with no fidelity/GT
> evidence it *improves* real clips, and a hard silhouette could hurt shallow legitimate gaps.

### 8.2 NaN propagation
Undefined `D_ref` NaN could spread through blur/warp/fit kernels — a Gaussian blur turns one bad
pixel into a blurred disc, compounding if it feeds back into state.

> **Audited 2026-08-18 — premise false, already handled.** `depth_ref` NaN is explicitly guarded at
> composite (`ref_undefined → safe_alpha=1.0`), no NaN reaches the flow-warp sampler, and alignment
> fits gate on `isfinite`. Only missing: a belt-and-suspenders `isfinite` assert before the splat
> stage.

### 8.3 Dilation radius should be derivable, not magic
Depth carries no detail beyond the backend's native resolution, so the dilation covering the
depth-bleed halo is largely a bilinear-upsampling halo — derive its width from
`W_full / native_W` instead of a fixed constant. **P2, not touched.**

### 8.4 ERP-anisotropic kernels
Dilation and blur operate in pixel space, but ERP pixel density per solid angle explodes toward
the poles, so the feather is angularly wider at high latitude than at the equator. Scale kernel
width by `cos(latitude)`. **P2, not touched.**

### 8.5 Alpha=1.0 hole-forcing
Correct while an object is present, undefined for the frame it leaves. Not separately
investigated.

### 8.6 `freeze_bg`'s color-vs-geometry conflation
`freeze_bg=True` was built to solve a real point-cloud gap (bglock-alone has no depth value for
a pixel it can't currently see, so parallax/temporal novel-view rendering opens a hole behind a
moving object — the current-frame depth map is single-layer, not occlusion-aware), but the fix as
shipped froze **color together with geometry**: the background Gaussian block is built once from
the temporal-median plate and concatenated unchanged into every frame, so shadows and (per D1's
open question) video screens never update for the whole clip once `freeze_bg` is on.

> **Fixed 2026-08-21.** New `freeze_bg_live_color` param (default `True` whenever `freeze_bg=True`)
> decouples the two: *geometry* stays frozen — same `means`/`scales`/`quats`, same hole-free
> coverage guarantee — while *color* is re-sampled every frame from the live image at each
> background Gaussian's stored pixel index wherever that pixel is currently visible
> (`sam_mask==0`), falling back to the frozen median color only under a dynamic object. Implemented
> via a `pixel_idx` key on the Gaussian dict, which survives `scene_filter`'s prune chain for free
> since all three prune fns generalize over `dict.items()`. Validated on two clips chosen for
> moving floor shadows and flat screens (`BENCHMARK_RULES.md` §17) at ~0–40s / ~250–300MB overhead
> — cheap enough to gate on `freeze_bg` alone rather than a separate flag. Answers the color half of
> D1's "can FreeTimeGS fit a video screen on static Gaussians?"; geometry staying frozen under a
> moving-but-unmasked screen is unchanged, still the accepted `SPAG_LOCK_ACTIVITY` trade-off (§7).

### 8.7 scene01 "waiter" depth-bleed trail — reopened
Under `bglock`, scene01 shows a smeared/ghosted depth trail behind the moving waiter instead of
clean per-frame geometry. Investigated sequentially (`--stride 4 --skip-step 2 --freeze-bg
--outlier-pruning 0.02 --grazing-angle 85.0 --sparse-pruning 0.1`, waiter visible frames 140-190):

> **#2–#4 — three null results.** *#2 mask-scope gap:* a real bug, found and shipped —
> `align_depth_frame(_gpu)` fit *and* applied the affine correction only outside `fused_mask`, while
> `composite_bg_locked` blends over a wider dilated+feathered footprint (37% of it sat outside
> `fused_mask`). Fixed via `apply_everywhere` (the fit still excludes that footprint from its static
> sample; the apply now covers the whole frame under bglock) — correct, but it does not move the
> trail. *#3 dilate/feather retune:* swept 6/4 vs 12/9 vs 20/15 (OOM); narrow halves peak
> `depth_delta_feather` but renders differ by only 0.0006–0.009 mean abs, i.e. noise. Defaults
> (12/9) kept. *#4 confidence-decay compounding:* `SPAG_CONF_LEGACY=1` moves waiter-window
> `depth_delta` ~10% either way with no consistent sign — not a contributor.
>
> **#5 `SPAG_BLEED_REJECT` — worked, then reverted.** This one found the right mechanism:
> per-frame monocular noise at the silhouette boundary bleeding into the frozen background plate
> during bilinear flow-warp interpolation across the depth cliff. Rejecting per-pixel depth that
> collapsed within `bleed_frac=0.35` of the mask/background gap measurably reduced the trail — mean
> abs diff 0.09–0.42, an order of magnitude above #2–#4, concentrated at the leg/floor boundary,
> with floors rendering clean and legs terminating sharply. **Reverted anyway**: the test is a
> single global per-frame median-gap threshold, so it cannot distinguish a real bleed artifact from
> a subject genuinely touching the background (feet on the floor legitimately converge toward
> `depth_ref` there). Real contact points were silently deleted — unacceptable information loss.
> Removed from `spag4d/video.py`.
>
> **Net: reopened.** The direction is confirmed (silhouette-edge bleed during flow warp); this
> implementation is known-bad. The failure mode any future fix must avoid is **deleting genuine
> foreground pixels merely because their depth value sits close to the adjacent background value**.
> A correct fix has to read local *spatial structure* — a bleed ring is contiguous with the
> silhouette edge and unbounded along the depth ramp, whereas true contact is a small stable
> footprint — rather than thresholding a single global depth-value distance. Likely shares its
> underlying fix with §8.3's derivable dilation radius.

---

## 9. Instrumentation — one diagnostics pass

Per-frame affine residual + fitted scale, median flow over static pixels, confidence histogram,
depth variance by region (static/dynamic/feather), per-object flow magnitude, `D_ref` sample-count
map, residual-vs-latitude.

> **Implemented + full-clip-verified 2026-08-18.** `SPAG_DIAG_CSV=<path.csv>` (unset = zero
> overhead); all eight signals land as one row per frame, reusing existing accumulators rather than
> recomputing. Caught a real bug on the way: `alpha_t.cpu().numpy()` threw on bf16 tensors in the
> GPU composite path. **Not done:** any QA plot — output was only spot-checked numerically.

---

## 10. Backends

### 10.1 The contract both wrappers should satisfy
`predict()` returns `(depth, mask)` for both, but they differ in convention, native-resolution cap,
invalid-pixel signal, scale semantics, and value range — none of it declared.

> **Done 2026-08-18.** Both wrappers now declare `depth_convention`, `native_resolution`,
> `invalid_pixel_signal`, `metric`, `value_range` as instance attributes. DA360:
> `radial` / `(518,1036)` / `None` / `metric=False` (median-anchored to 5.0m, not true metric) /
> `(0.0, inf)`. PaGeR gained the two it lacked: `invalid_pixel_signal="sky_mask"`,
> `value_range=(1e-2, 200.0)`. **Declarative half only** — bglock still hardcodes its thresholds
> rather than reading these; that wiring is follow-on work.

### 10.2 Backend outliers destroy the affine fit
DA360 inverts disparity (`1/(abs(d)+eps)`) — sky/poles → ~10⁶. PaGeR clamps to `[1e-2, 200]` with
sky at exactly 200.0. Both poison the least-squares affine fit and deflate flicker metrics computed
over them.

> **Fixed 2026-08-17 (fit) + 2026-08-18 (metrics).** Used a generic magnitude gate rather than
> per-backend validity masks: exclude any pixel beyond 50x the reference plate's own median from
> both the static-pixel fit (`align_depth_frame(_gpu)`) and the `bg_depth_cv`/`fg_depth_cv` medians
> (`_valid_metric_mask` in `video.py`) — one threshold catches DA360's 1e6 and PaGeR's clamped 200.0
> alike, verified by synthetic injection on both numpy and torch paths. Per-backend masks (§10.1)
> deferred unless a future backend lacks a magnitude-gate-friendly outlier signature.

### 10.3 PaGeR — cube-face seam steps — open
Each of 6 faces predicted independently and stitched; a moving object crossing a face boundary can
step in depth, periodic in *position* not time.

### 10.4 PaGeR re-classifies indoor/outdoor every frame
With `metric=True`, a per-frame CLIP call can flip the scale head mid-sequence → discontinuous
global scale jump.

> **Audited, premise false; defensive fix applied 2026-08-17.** `metric=False` short-circuits
> before CLIP runs, and production uses DA360 — PaGeR isn't in the path. Fixed anyway ahead of need
> (E1's pager plan puts PaGeR on the critical path): `_skip_heads` now caches the label in
> `self._scale_label` on first call. Inert today, not benchmarked.

### 10.5 DA360
Affine fit is in depth space (defensible if DA360's shift MLP truly removes shift — else Möbius,
not affine, and biased at far range). Per-frame median renormalization to 5.0m is content-dependent
(entering/leaving objects shift every background pixel's scale). Depth convention (radial/planar)
never declared.

> **Done 2026-08-18** (`scripts/residual_vs_latitude_test.py`): scale-only DA360-vs-PaGeR fit on
> static content, residual binned by latitude, two plates. Resolved as a **content confound, not a
> convention bug** — one plate's residual falls monotonically equator→south-pole, the other *peaks*
> at the equator and falls toward both poles. A content-independent convention bug would produce
> the same shape in both rooms; two different shapes instead matches each room's difficult content
> (windows/lights) sitting at different latitudes. A third room would rule out coincidence.

### 10.6 MapAnything migration — plan, not yet executed
Cubemap plumbing (`erp_to_cubemap`/`cubemap_to_erp`, per-face intrinsics/extrinsics) already exists
in PaGeR — reuse rather than adding a second implementation. One frame's 6 faces are a valid
multi-view set but share an optical centre (zero baseline, no parallax) — benefit is cross-face
consistency, not triangulation. **Will not replace bglock** — fixed camera means no temporal
baseline; should improve per-frame geometry, not frame-to-frame stability. Carry over
`native_resolution` cap; re-run the §10.3 seam test after the switch.

---

## 11. Verify the camera is actually static

Zero-flicker guarantee assumes frame-to-frame background correspondence is the identity. Real rigs
can violate this (vibration, wind, rolling shutter). Test: median WAFT flow over static pixels.

> **Confirmed static, no action needed — 2026-08-18.** Per-frame median WAFT flow over `mask==0`
> pixels on two canary clips came in at 0.011–0.031px mean (max 0.062px) — consistently sub-0.1px,
> two orders of magnitude below anything that would move a depth alignment. The consistent dx/dy
> sign across frames reads as WAFT's own correlation-window bias on a static scene, not physical
> motion. No per-frame global warp needed; closed without a third clip.

---

## 12. Priority index

| Priority | Item | Status |
|---|---|---|
| P0 | Confidence absorbing at zero (§4.1) | ✅ fixed 2026-08-17 |
| P0 | Warp `prev_confidence` by flow (§4.2) | ✅ fixed 2026-08-17 |
| P0 | Validity masks + robust fit (§10.2) | ✅ fixed 2026-08-17/18 (magnitude gate, fit + metrics) |
| P0 | Pin CLIP indoor/outdoor label (§10.4) | ✅ audited false, defensive fix applied 2026-08-17 |
| P1 | Diagnostics pass (§9) | ✅ implemented + full-clip-verified 2026-08-18 |
| P1 | Confirm camera is static (§11) | ✅ confirmed static 2026-08-18 |
| P1 | Residual-vs-latitude test (§10.5) | ✅ resolved 2026-08-18 — content confound, not convention bug |
| P1 | fp16 + outlier error floors (§6.7) | ✅ done 2026-08-18 — 3/10 baseline clips at/below fp16 floor |
| P1 | Paired stability/fidelity + ablation ladder (§6.0, §6.6) | ✅ partial 2026-08-18 — 3/5 rungs, caught the freezing trap live |
| P1 | Stride-invariance sweep (§6.1) | ✅ run 2026-08-18 — no accumulated propagation error; fps-coupling still open |
| P1 | Mask-injection proxy GT (§6.3) | ✅ done 2026-08-18 — equator RMSE 0.56m vs pole 0.01–0.07m |
| P1 | Mask false-positive audit + tracking-budget behaviour (§7) | ✅ code-audited 2026-08-18 — 128-cap premise false for default path |
| P1 | Sample-count floor + two-pass plate (§5) | ✅ measured + fixed 2026-08-18 (`SPAG_DREF_MIN_SAMPLES`); two-pass build not done |
| P1 | Feathering smear across discontinuities (§8.1) | ✅ fixed 2026-08-18, opt-in `SPAG_HARD_DEPTH_CUTOVER=1`; stays opt-in |
| P1 | NaN propagation (§8.2) | ✅ audited 2026-08-18 — already correctly guarded |
| P1 | Backend declares all five properties (§10.1) | ✅ implemented 2026-08-18; bglock reading them dynamically is follow-on |
| P1 | `freeze_bg` freezes color along with geometry (§8.6) | ✅ fixed 2026-08-21, `freeze_bg_live_color` default True |
| P1 | scene01 "waiter" depth-bleed trail (§8.7) | 🔴 reopened 2026-08-21 — mechanism confirmed, fix reverted (deleted real bg-contact points) |
| P2 | Time-reversal consistency (§6.2) | open |
| P2 | Synthetic radial/lateral/combined motion + floor sweep (§6.4) | ✅ closed 2026-08-18 |
| P2 | Normals-vs-depth-gradient consistency (§6.5) | open |
| P2 | Cube-face seam steps (§10.3) | open |
| P2 | `scale_anchor` from `D_ref` median (§10.5) | open |
| P2 | Dilation radius from native resolution (§8.3) | open |
| P2 | Auto-demote low-flow objects (§7) | open |
| P2 | FB threshold in angular units | open |
| P2 | Decay constant tied to fps / half-life | open |
| P3 | `D_ref` slow EMA refresh (§5) | open |
| P3 | ERP-anisotropic dilate/blur kernels (§8.4) | open |
| P3 | Cubemap decomposition of *flow* | open, deferred |

**Deferred.** Cubemap decomposition of optical flow is premature until latitude-resolved FB-error
shows how bad mid-latitude flow actually is — widening the pole margin may capture most of the
benefit. Separate from the cubemap projection used for *depth* (§10.6); reuse the same in-repo
helpers if it happens.

**Other WAFT/ERP notes.** WAFT is trained on rectilinear imagery — mid-latitude flow (~45–70°) is
degraded but not gated; measure FB-error vs. latitude to place a real cutoff instead of picking a
pole margin by eye. The 1.5px FB threshold is resolution-dependent — express in degrees of arc.
0.85 decay is fps-dependent — derive from a target half-life: `decay = 0.5 ** (1 / (fps *
half_life_s))`. Confirm ERP seam wrapping in every operation (FB consistency, pole-mask
construction, mask dilation, `prev_confidence` warp) — one unwrapped op produces a persistent
artifact at longitude 0.
