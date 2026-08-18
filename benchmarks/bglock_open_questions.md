# bglock — audit backlog and parallel work plan

Working document for a Claude Code session. Candidate bugs, tests, and fixes for the
background-locked depth stabilization stage of SPAG4D, plus a decomposition into independent
task packets suitable for delegation across parallel agents.

---

## 0. Context

**What the pipeline does.** Fixed-camera 360° equirectangular (ERP) video → per-frame monocular
depth → stabilization → dynamic point cloud / 3D Gaussian splats. The scene is a static
background with a small number of moving people/objects.

**What bglock does.** A monocular depth model has no memory, so background pixels that never move
still get a slightly different depth every frame → flicker. bglock fixes the background to a
reference depth map `D_ref` and flow-propagates only the dynamic regions:

1. **Align** — per-frame scale/shift fit against static pixels.
2. **Mask** — SAM3 (via `detect.py`) marks dynamic regions per frame.
3. **Propagate** — `propagate_depth_via_flow` warps last frame's stabilized depth by WAFT
   bidirectional flow, blends with the fresh estimate under a confidence map.
4. **Composite** — `composite_bg_locked` mixes `alpha * object_depth + (1-alpha) * D_ref` with a
   dilated + feathered alpha.

`D_ref` is built by taking a per-pixel temporal **median of the RGB frames** (dynamic masks
excluding moving content), then running the depth model **once** on that clean background plate.

**Current claimed result.** On `accident_electrique_02`: legacy per-frame `affine` alignment shows
background flicker spikes to 0.037; `bglock` is flat at 0.0000. The background figure is true by
construction (locked = zero variance) and is not evidence about the algorithm's judgement — see
§6.

**Components.**
- `detect.py` — MOG2 background subtraction → centroid tracking → SAM3 video predictor,
  bidirectional propagation, `max_frame_num_to_track=128`.
- `spag4d/da360_model.py` — DA360 backend. Scale-invariant disparity, inverted to depth.
- `spag4d/pager_model.py` — PaGeR backend. ERP→cubemap→per-face→stitch, returns
  `(depth, sky_mask)`, exposes `depth_convention="radial"` and `native_resolution`.
- WAFT — optical flow. Requires `os.chdir(waft_root)` with `finally: os.chdir(_saved_cwd)`
  because it loads DepthAnythingV2 weights via a hardcoded relative path.

**Paths.** Work on the remote server under `/raid/mb273924/`. Test clip:
`/raid/mb273924/_DATASETS/uptale/data/accident_electrique_02.mp4`. Outputs under
`/raid/mb273924/_DATASETS/uptale/detect/`.

**Direction of travel.** Migration to a multi-view MapAnything integration using PaGeR's existing
cubemap path is planned. §10.4 covers what that does and does not change.

---

## 1. Rules

**Verify before acting.** This document was written from speaker notes plus two model wrapper
files, not from the full source. Two claims in earlier drafts were already wrong (propagation
direction, PaGeR metric default). Every item below is a *candidate*. Check it against the live
code first. "Checked, not a problem, here's why" closes an item successfully.

**Do not implement speculative fixes.** If an item's premise doesn't hold in the current code,
record that and stop. Do not refactor around a problem that isn't there.

**Report per item**, not per file: item ID, premise held y/n, evidence, action taken.

---

## 2. GPU usage

Several GPUs are available on the server and are shared. Before launching any job:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader,nounits
```

Pick indices with low `memory.used` and low `utilization.gpu`, then pin explicitly:

```bash
CUDA_VISIBLE_DEVICES=<free_idx> python <script>
```

Rules:
- One task packet per GPU. Do not share a device between concurrent packets.
- Re-check occupancy immediately before launch; another agent may have taken the device.
- `detect.py` uses `gpus_to_use = [torch.cuda.current_device()]`, so it follows
  `CUDA_VISIBLE_DEVICES` — no code change needed.
- Depth backends load ViT-Large / ViT-Giant weights; budget accordingly and fail loudly rather
  than falling back to CPU.
- Record which GPU each run used in the results log, so timing numbers are comparable.

---

## 3. Parallel task packets

Split the work and delegate. Packets in the same wave have **disjoint file ownership** and can
run concurrently on separate GPUs by separate agents. Waves are sequential.

### Wave 1 — independent, run all in parallel

| ID | Goal | Owns | Done when |
|---|---|---|---|
| **T1** | Confidence semantics in propagation (§4) | `propagate_depth_via_flow` | Both P0 items checked; confidence histogram over frame index produced |
| **T2** | Backend validity + outliers (§10.2, §10.3) | `spag4d/da360_model.py`, `spag4d/pager_model.py` | Validity masks emitted; clamped/extreme pixels excluded; CLIP label pinned |
| **T3** | Diagnostics pass (§9) | `spag4d/video.py` (`SPAG_DIAG_CSV`, inline rather than a new module — reuses the loop's own state directly) | ✅ implemented 2026-08-18, see §9; CSV wiring unit-verified, **full-clip run not yet done** |
| **T6** | Mask coverage in `detect.py` (§7) | `detect.py` | 128-frame budget behaviour documented; false-positive audit run |

### Wave 2 — needs T3's instrumentation to validate

| ID | Goal | Owns | Done when |
|---|---|---|---|
| **T4** | `D_ref` plate coverage (§5) | `D_ref` construction | Sample-count floor added; plate + count map saved |
| **T5** | Compositing (§8) | `composite_bg_locked` | Discontinuity handling and NaN safety resolved |
| **T7** | Validation harness (§6) | new eval script | Paired stability/fidelity table across the ablation ladder; stride sweep and mask-injection results reported |

### Wave 3 — after T2 lands

| ID | Goal | Owns | Done when |
|---|---|---|---|
| **T8** | Backend contract refactor (§10.1) | `core.py`, both wrappers | All five properties declared and consumed by bglock |

**Coordination.** T2 and T8 both touch the wrappers — T8 must wait. T7 depends on T3's metric
definitions. Everything else is independent. Each agent reports back its item table; merge at
wave boundaries.

---

## 4. P0 — suspected bugs

### 4.1 `running_confidence` is absorbing at zero

> **✅ TESTED — FIX APPLIED 2026-08-17.** Audited against live code in
> [`T1_T2_P0_AUDIT.md`§4.1](T1_T2_P0_AUDIT.md); the exact premise below (absorbs to hard 0) was
> *false* — the real reset-on-trust logic pinned `running_confidence` to exactly `{0.0, 0.85}`
> forever instead, which is a different but equally real bug: `decay` never compounded despite the
> docstring claiming it did. Fixed in `spag4d/flow_depth_propagation.py` by replacing the
> confidence state with a per-pixel **age** field (frames since last re-anchor, floored/capped),
> giving `confidence = frame_confidence * max(conf_floor, decay**age)` — the "track age, compute
> `decay**age`" alternative this section already proposed. New default. Verified two ways: (1) a
> standalone numpy/torch equivalence script confirms both propagation code paths compound
> identically; (2) a real pipeline run with `SPAG_CONF_HIST=1` shows `running_confidence` taking
> thousands of distinct values across a clip (vs. exactly 2 before) and the designed geometric
> sequence 0.85→0.7225→0.6141→0.522→floored at 0.5. Paired stability/fidelity benchmark (rule 11,
> not just a structural check) in
> [`benchmarks/confdecay_2026-08-17/RESULTS.md`](confdecay_2026-08-17/RESULTS.md): the old
> behaviour was suppressing ~31% of real foreground motion on the MattSwift canary
> (`fid_ratio` 0.692); the fix recovers most of it (0.923) at the cost of some of the nominal
> `fg_depth_cv` figure. `SPAG_CONF_LEGACY=1` reproduces the pre-fix trace digit-for-digit, so every
> benchmark recorded before 2026-08-17 stays reproducible. **Not yet done:** `conf_floor` sweep and
> the full 10-clip run (rule 3) before calling the *floor value* (not the fix itself) production-
> validated. **Done 2026-08-18:** the §6.4 synthetic radial-motion test, the one case this fix
> exists for — see §6.4 below; confirms decay is load-bearing (4.6x better than legacy, frozen
> fails catastrophically).
`running_confidence = min(frame_confidence, prev_confidence * 0.85)`

If `prev_confidence == 0`, then `0 * 0.85 == 0` and `min(x, 0) == 0`. A pixel that fails
forward-backward consistency **once** is pinned to zero for the rest of the video and falls back
to the raw per-frame estimate forever. The pole band has `pole_trust = 0` permanently, so it
also poisons any pixel receiving flow from it.

Consequence: dynamic-region smoothing degrades monotonically over a sequence; any measured
improvement is likely front-loaded in the first frames.

*Test:* histogram `running_confidence` per frame. If mass collapses to 0 within a few dozen
frames, confirmed.

*Fix:* add a recovery path — a floor term (`prev * decay + recovery`), or reset to
`frame_confidence` after K consecutive high-confidence frames. Alternative: track a per-pixel
*age* since last re-anchor and compute `frame_confidence * decay**age`, resetting age on
re-anchor. Makes the state inspectable.

### 4.2 `prev_confidence` is probably not warped
Confidence describes a surface point, not a grid location — same as `depth_prev_final`, which
*is* warped. If `prev_confidence[p]` is read un-warped, then in every dynamic region the decay is
applied to the wrong pixel's history: a pixel just uncovered by a moving person inherits the
person's edge confidence rather than the background's.

*Fix:* sample `prev_confidence` at `p - flow_fwd(p)` with the same bilinear/wrap/clamp sampler as
the depth warp.

> **✅ TESTED — FIX APPLIED 2026-08-17, folded into the §4.1 fix.** Confirmed live: the age field
> is warped in the *same* batched `warp_backward_multi`/`warp_backward_multi_torch` call that
> already warps `depth_prev_final`, so this fix rides along for free — VRAM is byte-identical
> across arms in the benchmark below, and the fractional (non-power-of-0.85) confidence values
> observed in the trace are direct evidence the age field is being bilinearly warped rather than
> read unwarped. See [`T1_T2_P0_AUDIT.md`§4.2](T1_T2_P0_AUDIT.md) and
> [`benchmarks/confdecay_2026-08-17/RESULTS.md`](confdecay_2026-08-17/RESULTS.md) finding 1 and 4.

### 4.3 Backend outliers destroy the affine fit
DA360: `depth = 1 / (disparity.abs() + eps)` with `eps = 1e-6`. Sky and poles invert to ~10⁶.
`abs()` flips negative predictions into large positive depths instead of flagging them.
PaGeR: `set_depth_range(1e-2, 200.0)` with sky filled to MAX_DEPTH — a population sitting at
exactly 200.0.

Consequences: least-squares affine is destroyed by a handful of 10⁶ values; clamped values are
not a linear function of true depth so cannot be fitted; constant clamped pixels contribute zero
variance and **deflate any flicker metric** that includes them; both become absurd points in the
splat.

*Fix:* DA360 — derive a validity mask from raw disparity *before* inversion. PaGeR — use
`sky_mask`, plus explicit `depth >= 200-ε` / `depth <= 1e-2+ε` exclusion (a distant non-sky
surface also clamps). Both — exclude from the fit and from all metrics. Use Huber or RANSAC
regardless.

> **✅ PARTLY TESTED — FIX APPLIED 2026-08-17 (fit only).** Audited in
> [`T1_T2_P0_AUDIT.md`§4.3](T1_T2_P0_AUDIT.md): premise partly true (1e6 values do enter the fit
> uncaught), but the damage was already bounded by `scale_clip` + the outliers being shared
> between the fit's `x` and `y`. Applied the minimal generic fix instead of the backend-specific
> validity masks proposed above: `align_depth_frame`/`align_depth_frame_gpu` now exclude any pixel
> beyond 50x the reference plate's own median from the static-pixel fit — catches DA360's ~1e6
> disparity-inversion and PaGeR's clamped 200.0 with one magnitude gate, no wrapper changes or new
> mask threaded through the GPU chain. Verified: injecting `1e6` into a synthetic frame reproduces
> the exact clean-frame `(s, t)` fit, both numpy and torch paths. **FIX APPLIED 2026-08-18
> (metrics side too):** the same cap now gates the four per-frame median computations feeding
> `bg_depth_cv`/`fg_depth_cv` via a `_valid_metric_mask` helper in `video.py`, falling back to the
> unfiltered mask if a region is entirely outliers. Verified at the helper level; not yet
> re-verified against a full benchmark rerun. **Still not done:** per-backend validity masks
> (§10.1's bigger refactor) — deferred, not needed unless a new backend arrives without a
> magnitude-gate-friendly outlier signature.

### 4.4 PaGeR re-classifies indoor/outdoor every frame
With `metric=True`, `_skip_heads()` runs a CLIP classifier per frame and routes to
`scale_indoor` or `scale_outdoor`. A borderline scene can flip mid-sequence, switching scale
heads and producing a **discontinuous global scale jump** — a step change the affine fit must
absorb.

*Fix:* classify once on the `D_ref` plate, cache the label, pass it in. Also saves a CLIP forward
per frame. *Test:* log the label per frame; assert constant.

> **✅ AUDITED, premise false; DEFENSIVE FIX APPLIED 2026-08-17.** `metric=False` short-circuits
> before CLIP ever runs, and production uses `active_generator="da360"` — PaGeR isn't in the path
> at all, so nothing was reclassifying per frame. Applied the fix anyway ahead of need: E1's pager
> plan puts PaGeR on the critical path with `metric=True`, so `_skip_heads` now caches the label in
> `self._scale_label` on first call instead of leaving the per-frame reclassification live for
> whoever flips that flag first. Inert today — not benchmarked, since there's no live code path to
> measure yet. See [`T1_T2_P0_AUDIT.md`§4.4](T1_T2_P0_AUDIT.md).

---

## 5. `D_ref` — plate coverage

The median-RGB-plate approach is sound and better than median-of-depth-maps: one inference on one
coherent image gives internally consistent geometry rather than stitching per-pixel samples from
N independently-inferred maps on differing implicit scales. Residual risk moves from depth to
plate coverage.

- **Thin coverage.** Where a pixel has few unmasked samples, the median may land on a
  transitional frame — motion blur, mask edge, half-occlusion — producing implausible RGB that
  the model reads as real geometry. Permanent, since `D_ref` is inferred once.
  *Fix:* require ≥K valid observations (K ≈ 5–10) before marking `D_ref` valid; below that, mark
  undefined and let the alpha=1.0 hole-fill path handle it. Emit the per-pixel sample-count map.
- **Residual mask gaps.** Bidirectional SAM3 propagation closes the main one. What remains:
  tracks with `len(seen) < 10` are discarded, so brief motion events are never masked. Median is
  robust while the object is a minority of samples at each pixel, which a brief event is by
  definition — likely benign, worth confirming. The case that *does* bias the plate is a subject
  lingering in one spot for a large fraction of the clip.
- **Two-pass build.** If the plate uses a leading window of frames, build it from the full mask
  history instead: a pixel occluded in frames 0–50 but clear at frame 400 then gets clean
  samples. Cost is one extra decode pass; masks are computed anyway.
- **Plate realism.** The composite is a synthetic image with seams between regions medianed from
  different frames. Save it as a QA artifact and eyeball it.
- **Permanent holes.** A never-moving occluder means the background behind it is never observed.
  Either inpaint `D_ref` spatially or flag those pixels and drop them from the point cloud.
- **No refresh.** Sun angle, moved furniture, exposure drift. Optional slow EMA on pixels
  unmasked and low-flow for N frames (`η ~ 0.01`), gated behind a flag so the 0.0000 result stays
  reproducible.

> **Done and fixed 2026-08-18.** Measured on a real 39-frame clip (`circulation_site_1_edit_coupe`,
> 3840×1920, via the §9 diagnostics pass's `diag_dref_sample_count.npy`): mean coverage 96.7%,
> 0.34% zero-sample (permanent holes — already NaN-guarded at `video.py:764`, per §8's NaN audit),
> 0.98%/1.63% at ≤5/≤10 samples — under 2% of pixels fall below the doc's K≈5-10 threshold.
> Connected-component labeling (`scipy.ndimage.label`) shows this thin coverage is **strongly
> clustered, not scattered**: one component holds 85.0% of ≤10-sample pixels (top 5 hold 99.4%),
> sitting mid-frame (y≈43-66%, x≈54-70%) — not at the ERP poles, consistent with one recurring
> occluded/dynamic region rather than sampling noise. That bounded-hole shape (echoing the §8
> hard-cutover benchmark's own finding) is what promoted the K-threshold fix from "measured, low
> urgency" to cheap-and-low-risk: touching one contiguous region is easy to QA and bound the blast
> radius of.
>
> **Implemented.** `compute_temporal_median` (`video.py`) gained `min_sample_frames: int = 0`;
> `sample_count` is now always computed, and the undefined mask becomes `always_masked |
> (sample_count < min_sample_frames)` when set — thin-coverage pixels get the same treatment
> zero-sample pixels already had (RGB inpaint on the median plate, `depth_ref = NaN`, no fabricated
> surface). Default `0` preserves old behavior exactly. Opt in via `SPAG_DREF_MIN_SAMPLES=K`, wired
> at the `compute_temporal_median` call site in `run_video`. Verified with a synthetic unit test
> (K=0 vs K=3 correctly gate a 2/10-sample pixel) and an end-to-end smoke run on
> `accident_electrique_02` with `SPAG_DREF_MIN_SAMPLES=5` (exit 0, 188/188 frames) — that clip's
> D_ref coverage happens to be full everywhere so the threshold never fired, confirming the code
> path integrates cleanly but not reproducing the clustering effect. **Not yet done:** the two-pass
> plate build, and an end-to-end run on `circulation_site_1_edit_coupe` itself (the clip known to
> have thin-coverage pixels) to confirm the inpaint+NaN path fires and produces a sane hole.

---

## 6. Validation without ground truth

No ground-truth depth is available. Clips are ~30 s (≈900 frames at 30 fps) and frame-stepping is
available. That is enough — but it constrains what can be claimed, and there is one trap that
invalidates everything else if missed.

### 6.0 The trap: every stability metric is gamed by freezing

The algorithm that scores perfectly on any temporal-stability metric is `output = D_ref` for every
pixel of every frame. Background flicker of 0.0000 is exactly this: true by construction, not
evidence. **No stability number may be reported without a paired fidelity number** showing motion
was preserved. Every row in the results table is a pair:

| | Stability (lower better) | Fidelity (must not collapse) |
|---|---|---|
| static region | temporal variance | — |
| dynamic region | temporal variance | temporal *derivative* magnitude vs raw |
| whole frame | point-cloud Chamfer, consecutive frames | ordinal agreement with raw (§6.5) |

A dynamic-region derivative that is *flatter* than the raw estimate — not merely smoother — means
motion is being suppressed. That single check is what separates a real result from a tautology.

### 6.1 Stride invariance — the main GT-free error signal

True depth at a given timestamp does not depend on the frame stride used to reach it. So run the
full pipeline at stride 1, 2, 3, 5 and compare depth on the timestamps the runs share. Any
disagreement is error introduced by the propagation chain, measured against nothing but itself.

This is the closest thing to ground truth available, and it yields two results at once:
- **Magnitude of accumulated propagation error**, per region, per frame index.
- **Whether the decay constant is fps-coupled.** Stride 3 means effective 10 fps, so a per-frame
  decay of 0.85 becomes 3× faster in physical time. If quality degrades sharply with stride, the
  constant needs deriving from a half-life in seconds (§12) rather than per frame.

Report disagreement as a function of stride *and* of elapsed frames since re-anchor — the two
should be separable.

> **Done 2026-08-18.** Ran the full production pipeline at `--skip-step 1/2/3/5` on
> `circulation_site_1_edit_coupe` (156 native frames, 3840×1920, `--depth-raw` to dump per-frame
> depth), compared depth at the 6 native timestamps all four strides share (multiples of
> lcm(1,2,3,5)=30: frames 0, 30, 60, 90, 120, 150) — `scripts/stride_invariance_compare.py`.
>
> - **No measurable growth in disagreement across the clip, at any stride.** Median |depth
>   disagreement| vs. the stride-1 run is essentially flat from frame 0 through frame 150 for every
>   stride: stride 2 stays at 0.045±0.0002 m, stride 3 at 0.021±0.0003 m, stride 5 at 0.047±0.0003 m
>   (subtracting each stride's own frame-0 value as baseline, the remaining "growth" is ±0.0003 m —
>   noise-level, over up to 30 propagation steps at stride 5). This directly answers the "magnitude
>   of accumulated propagation error" question: **it does not accumulate** on this clip, over this
>   length — the confidence-decay/conf_floor mechanism's own advertised bounded-drift property
>   (§4.1/§4.2's steady-state low-pass argument) holds up under an actual multi-stride test, not
>   just the synthetic §6.4 tests.
> - **The disagreement that does exist is present from frame 0 and is not propagation error at
>   all** — frame 0 is a fresh estimate in every run, before any propagation has happened, so a
>   nonzero stride-1-vs-stride-N gap there can only come from something that differs between the
>   runs *before* propagation starts. The prime suspect: each stride builds `D_ref` (the temporal-
>   median background plate) from a *different* subset of native frames (stride 1 uses all 156,
>   stride 5 uses 32), so bglock's static background is locked to a genuinely different reference
>   depth per stride — a plate-composition effect, not a propagation-drift effect.
> - **This confound also explains the non-monotonic disagreement vs. stride** (stride 3 disagrees
>   *less* with stride 1 than stride 2 or stride 5 do: 0.021 m vs 0.045 m and 0.047 m) — if this were
>   pure sampling-density noise it should be monotonic in stride; it isn't, which points to *which*
>   frames each stride happens to land on (aliasing against scene content) rather than a clean
>   count-vs-noise relationship.
> - **The fps-coupling question is not answered by this run** — the D_ref-plate confound dominates
>   any subtler propagation-decay signal at this precision. To isolate it, the same `D_ref` plate
>   would need to be pinned across all four stride runs (the CLI has no flag to inject an external
>   plate) or the comparison would need to run region-gated on `alpha>=0.98` (pure dynamic-object
>   pixels only, where `D_ref` never applies) instead of whole-frame. Not done here — the headline
>   result (no accumulation) stands regardless, but the fps-coupling sub-question is still open.

### 6.2 Time reversal

bglock is causal. Run it forward over the clip, then over the reversed frame sequence, and compare
depth per timestamp. True depth is direction-invariant, so disagreement is accumulated
directional error. This specifically exposes the disocclusion path, because every disocclusion
forward is an occlusion backward — the asymmetry is the point. A large forward/backward gap
concentrated at object boundaries implicates the `running_confidence < 0.5` override (§4.1).

### 6.3 Mask injection — proxy ground truth for the dynamic path

The dynamic path is the part with no reference to check against. Manufacture one: take a patch of
background that is genuinely static and **inject it into the dynamic mask** so bglock treats it as
foreground and runs the full flow-propagation blend on it. You already know the correct answer
there — it is `D_ref`.

This converts the untestable branch into a measurable one. Error against `D_ref` on injected
patches is a direct accuracy metric for propagation, confidence decay, and blending. Vary the
patch by latitude (equator vs mid-latitude vs near-pole) to get the latitude-resolved error
profile that §12's FB-threshold and pole-margin questions need, and place patches adjacent to real
moving objects to measure the feather zone (§8).

### 6.4 Synthetic objects — the only real ground truth obtainable

Composite a rendered object into the median plate with a **known** depth and trajectory, warping
it correctly in ERP. Exact GT for the dynamic region, at whatever difficulty you choose.

Priority trajectory: **radial motion**, straight toward and away from the camera. 2D flow cannot
observe it, so the decay term is the only mechanism that lets depth update — this is the single
best test of the 0.85 constant, and no real clip gives you a controlled version. Follow with
lateral motion, a seam crossing (tests §10.2 cube-face steps and ERP wrap), and a pole transit.

Two known-depth planes at different distances also give a controlled version of the feathering
smear in §8: the invalid intermediate values are directly measurable rather than inferred.

> **✅ TESTED 2026-08-18 — radial, lateral, and combined motion; conf_floor sweep. CLOSED.**
> All four synthesize the minimal version of this test directly at the `propagate_depth_via_flow`
> level (no rendering/compositing pipeline needed — the function only consumes depth maps + flow
> fields), a circular object patch on a flat background plate, with per-frame monocular-estimate
> noise (σ=0.15m) standing in for the fresh affine estimate. Object median depth vs ground truth,
> compared across the same four arms throughout: **raw** (no propagation, fidelity ceiling),
> **decay** (production, 0.85/0.5), **legacy** (pre-2026-08-17 non-compounding), **frozen**
> (decay=1.0/floor=1.0, i.e. flow-prop with no decay at all).
>
> **1. Radial motion** (`scripts/synth_radial_motion_test.py`) — ground-truth depth 8m → 2m → 8m
> over 60 frames, pure approach/recede, zero flow everywhere (radial motion is invisible to 2D
> flow by construction):
>
> | arm | RMSE (m) | max err (m) |
> |---|---|---|
> | raw (fidelity ceiling) | 0.004 | 0.011 |
> | **decay (production)** | **0.198** | 0.288 |
> | legacy | 0.907 | 1.126 |
> | frozen | 3.459 | 5.993 |
>
> Frozen fails exactly as predicted — with zero observable flow it never updates depth at all.
> Production decay is 4.6x better than legacy and tracks the trajectory within ~0.2m against the
> 0.15m noise floor: a real, quantified validation of `SPAG_CONF_DECAY=0.85`/`SPAG_CONF_FLOOR=0.5`.
>
> **2. `conf_floor` sweep** (`scripts/synth_radial_motion_floor_sweep.py`, decay=0.85 fixed,
> floor swept 0.1–0.95) — confirms the closed-form steady-state prediction
> (`lag ≈ trajectory_slope × floor/(1-floor)`, once age saturates) up to floor≈0.7, after which the
> age-ramp transient and the direction-reversal at frame 30 make the true error undershoot the
> naive formula. The RMSE-vs-floor curve is **convex, not linear**: 0.5→0.7 costs about as much
> absolute error as 0.1→0.5 combined, and 0.9 is 6x worse than 0.5. Production's 0.5 sits right at
> the elbow before the curve steepens — not contradicted by this sweep, though this only measures
> the *cost* side (radial lag); the *benefit* side (how much real flicker suppression a given
> floor buys on actual clips) is still an open, separate sweep (see the §4.1/§4.2 "not yet done"
> note above).
>
> **3. Lateral motion** (`scripts/synth_lateral_motion_test.py`) — same object, constant depth
> (5m), translating across the frame at 4px/frame; flow now correctly observes the motion (uniform
> non-zero flow inside the object's current footprint, zero on background):
>
> | arm | RMSE (m) | max err (m) |
> |---|---|---|
> | raw | 0.004 | 0.011 |
> | decay (production) | 0.002 | 0.005 |
> | legacy | 0.002 | 0.007 |
> | frozen | 0.007 | 0.009 |
>
> All four arms track to within ~1cm — confirms the asymmetry is real: decay/floor are neither
> needed nor harmful when flow can observe the motion. One minor, not practically significant
> finding: `frozen` shows a small consistent negative bias (~7.5mm) rather than symmetric noise,
> likely repeated-warp bilinear-interpolation drift at the object boundary with no re-anchoring —
> two orders of magnitude below the noise floor of the other arms, but a second, independent reason
> (beyond the radial blind spot) `frozen` isn't what production uses.
>
> **4. Combined radial + lateral** (`scripts/synth_radial_lateral_motion_test.py`) — the radial
> trajectory and the lateral translation simultaneously (a stand-in for a real subject walking
> toward the camera at an angle), flow capturing only the lateral component:
>
> | arm | RMSE (m) | vs radial-only |
> |---|---|---|
> | raw | 0.004 | +0% |
> | decay (production) | 0.191 | −4% |
> | legacy | 0.901 | −1% |
> | frozen | 3.441 | −1% |
>
> Every arm lands within ~4% of its radial-only number — the flow-warp term correctly relocates
> the *sampling position* each frame independent of how the depth-tracking mechanism handles the
> *value* there, i.e. no cross-talk between position-tracking and depth-tracking. Reassuring for
> real footage, where subjects generically combine both kinds of motion, since it means the
> mechanism generalizes rather than being calibrated to the pure-radial synthetic case only.
>
> **Conclusion: §6.4 is closed.** The decay/floor mechanism does exactly what it was built for
> (recovers depth under radial motion no other signal can observe, 4.6–17x better than the
> alternatives) and costs nothing where it isn't needed (lateral motion, within noise across all
> four arms), and the two effects don't interact when combined. `conf_floor=0.5` is not
> contradicted by the cost curve. **Deliberately not done:** seam crossing and pole transit
> (dropped — ERP-wrap and pole-margin correctness are §10.2/pole-margin-frac concerns orthogonal to
> the decay mechanism itself, not blind spots this fix addresses) and two-known-depth-plane
> feathering (§8, separate open item, not re-prioritized by this work).

### 6.5 Free consistency signals already in the pipeline

- **PaGeR normals vs depth gradient.** `self.last_normals` is already computed. Normals derived
  from the depth map should agree with the predicted normals; disagreement is internal geometric
  inconsistency, needs no GT, and is per-pixel. Also usable temporally — the agreement should not
  fluctuate on static geometry.
- **Ordinal consistency.** Metric accuracy is unavailable but *ordering* is checkable: sample
  pixel pairs, and verify that which-is-nearer stays constant over time on static geometry and
  changes plausibly on moving objects. Scale-invariant, robust, and it catches the case where
  depth drifts smoothly enough to pass a variance check while the scene reorders.
- **Rigid-object shape stability.** A person is approximately rigid over short intervals. Measure
  depth structure *within* an object mask relative to its own median — the internal shape should
  be stable frame to frame even as the object's mean depth changes. Deformation is error.
- **Flow/depth agreement.** Residual between depth warped by flow and the fresh estimate, on
  static pixels, where it should be near zero.

### 6.6 Ablation ladder is the reporting frame

Without GT the argument is relative, so make it explicitly relative. Run the same metrics across:
raw per-frame → affine only → + background lock → + flow propagation → + decay. The claim to
defend is not "this is accurate" but "each component reduces instability without suppressing
motion," which the paired table in §6.0 supports directly.

> **Done 2026-08-18 (partial — assembled from an existing run, not a fresh one).**
> `benchmarks/confdecay_2026-08-17/comparison.json` already ran a 3-rung slice of exactly this
> ladder (noprop → legacy → decay, all with bglock+flow-propagation on, only the confidence
> mechanism varying) and already computed a paired stability+fidelity metric per §6.0's own
> prescription (`fg_depth_cv` for stability, `fg_abs_derivative`/`fidelity_vs_noprop` — temporal-
> derivative magnitude relative to the no-propagation baseline — for fidelity). Reassembled here as
> the ladder table this item asks for:
>
> | clip | arm | fg_depth_cv (stability) | fidelity_vs_noprop |
> |---|---|---|---|
> | MattSwift | noprop | 0.1681 | 1.00 |
> | MattSwift | legacy | 0.1307 | 0.69 |
> | MattSwift | decay (production) | 0.1582 | 0.92 |
> | circulation_site_1_edit_coupe | noprop | 0.0600 | 1.00 |
> | circulation_site_1_edit_coupe | legacy | 0.0554 | 1.03 |
> | circulation_site_1_edit_coupe | decay (production) | 0.0587 | 1.05 |
>
> This is the trap in §6.0 caught in the act on real data: on MattSwift, `legacy` posts the
> *best*-looking stability number (0.131, lower than decay's 0.158) but at 0.69 fidelity — it's
> buying that stability by suppressing 31% of real motion, exactly the "flatter, not merely
> smoother" tell §6.0 warns about. `decay` (production) has a worse-looking stability number but
> 0.92 fidelity — most of the improvement over `noprop` is real, not gamed. On
> `circulation_site_1_edit_coupe` neither propagation arm suppresses motion at all
> (fidelity ≥1.0), so the two clips together show the mechanism is stability-positive without a
> systematic fidelity cost — but MattSwift alone shows why the *legacy* config specifically would
> have been the wrong pick if only the stability column had been read. **Caveat:** this is 3 of the
> 5 requested rungs (`noprop`≈"no flow propagation" and the two propagation variants), not the
> full raw→affine→+bglock→+flow→+decay ladder — the `raw` (no bglock at all) and `affine-only`
> arms weren't in this comparison and would need fresh runs against `--depth-correction affine`,
> not done this session.

### 6.7 Instrument-error floors to establish first

Both are cheap and both can invalidate a result:
- **fp16 quantization.** PaGeR's forward runs at `torch.float16`; with a 200 m range the step near
  20 m is ~0.016, the same order as the 0.037 figure. Round-trip a depth map through fp16 and
  measure the induced pseudo-flicker.
- **Outlier contamination.** Establish the **units** of the 0.037 figure and whether the §4.3
  outliers (10⁶ values, clamped 200.0 pixels) were excluded when it was computed.

> **Done 2026-08-18** (`scripts/fp16_quantization_floor_test.py`). fp16 has constant *relative*
> precision, so its quantization step as a fraction of depth is ~0.0008 at every magnitude from
> 0.5-100 m (measured, not assumed) — a single deterministic round-trip, not a per-frame noise
> source, so it's a floor estimate rather than an exact predicted CV. `bg_depth_cv` (per
> `benchmark_solutions.py`) is `bg_std / bg_mean` — **dimensionless**, a fraction of mean depth —
> which resolves the original "same order as 0.016 m" comparison: it was comparing an absolute
> distance to a dimensionless ratio, not a like-for-like comparison. Once expressed on the same
> footing, the 10-clip `baseline_2026-07-27` `bg_depth_cv` values sit 0.5x-31x the fp16 floor:
> **3 of 10 clips (MattSwift 0.5x, circulation_site_1_edit_coupe 0.75x, scene_03 0.85x) are AT or
> BELOW the fp16 floor** — their measured "flicker" is indistinguishable from float precision
> noise and should not be read as a real stability signal, let alone compared between configs. The
> other 7 sit clearly above it (2.8x-31x), so real signal dominates there. Separately: that
> baseline predates the §4.3 outlier-exclusion fix (2026-08-17/18), so none of its `bg_depth_cv`
> figures excluded the magnitude-capped backend outliers from the underlying median — reproducing
> them with current code would need a full re-run (not done here). The specific "0.037" headline
> figure quoted in §0 does not match any number in that JSON (accident_electrique_02's own
> `bg_depth_cv` there is 0.0168) and its exact provenance could not be traced in this session — flag
> it as unverified rather than load-bearing until its source run is found.

### 6.8 Watch it

Render the point cloud from a novel viewpoint and view it as video, plus a heatmap video of
`|depth_t − depth_{t−1}|`. The feathering smear (§8) and mask-edge popping are obvious to the eye
and near-invisible to per-pixel metrics. Side-by-side A/B against the baseline is the most
informative single artifact for anything going on a slide.

### 6.9 What a 30 s clip supports

- 900 frames: enough for a stride sweep at 1/2/3/5 with plenty of shared timestamps.
- ~7 chunks against the 128-frame tracking cap — enough to see periodic chunk-seam spikes (§7).
- Leave-one-out `D_ref`: rebuild the plate excluding frame *k*, predict frame *k*, compare.
  Tests plate coverage (§5) without holding out a contiguous block.
- Not enough for: slow-drift effects (§5's EMA refresh), which need longer footage to matter.

## 7. Masks and `detect.py`

**False positives** silently break the zero-flicker guarantee, and the current metric cannot see
them (it is measured on pixels the mask already called static). Sources: global luminance /
auto-exposure shifts (the `threshold(fg_mask, 200, 255)` shadow-killer does not help — these are
real intensity changes), compression artifacts, 360 stitching seam wobble, SAM3 over-extension
from a single centroid prompt, and the compositing dilation (a deliberate, controlled one).

*Fix:* auto-demote. If a SAM3 object's mean in-mask flow magnitude stays below threshold for N
consecutive frames, treat it as static and let it lock. Also handles a person who walks in, sits
down, and stays still — currently flagged dynamic and therefore unstabilized forever.

**False negatives** (foliage, water, screens) are an accepted trade-off but should be quantified:
run the inverse audit on static-flagged pixels. High flow + static flag = missed motion. Small
and spatially concentrated is fine; diffuse is not.

**Coverage gaps:**
- `max_frame_num_to_track=128` caps propagation. Establish whether the budget is split around the
  spawn frame or applied per direction — if symmetric, effective forward reach is only ~64
  frames. If masks simply stop past the cap, **everything locks** and all real motion freezes.
  Long clips need chunked re-prompting with `D_ref` held constant.
- **Predicted artifact:** if chunking exists, `depth_prev_final` and `running_confidence` are
  per-frame state that resets at each seam unless explicitly carried over, making the first frame
  of each chunk fall back entirely to the raw estimate. Testable: look for periodic flicker
  spikes at multiples of the chunk length.
- `len(seen) < 10` tracks are dropped entirely; brief motion is treated as background and frozen.

> **Done 2026-08-18 (code audit, current `spag4d/video.py` — no new pipeline run).**
>
> - **Coverage-gap premise is false for the default path.** `max_frame_num_to_track=128` is not
>   the production default. The default (`SPAG_SAM3_MAX_FRAMES` unset) calls `propagate_in_video`
>   with `propagation_direction="both"` and **no cap at all** — SAM3's own default
>   (`third_party/sam3/sam3/model/sam3_video_inference.py:239-241`) is `max_frame_num_to_track =
>   num_frames`, i.e. the whole clip. A cap only applies when `SPAG_SAM3_MAX_FRAMES=<N>` is
>   explicitly set (Tier 4.2, opt-in, not currently defaulted on anywhere in `cli.py`).
> - **The "symmetric split" question is also moot for the default path** (no cap → nothing to
>   split), and false in general: reading `sam3_video_inference.py`, forward and backward each
>   receive the *full* `max_frame_num_to_track`, not half of it — so even under the opt-in chunked
>   mode, effective reach per direction is the full cap, not the cap/2 the doc worried about.
> - **The opt-in chunked mode (`SPAG_SAM3_MAX_FRAMES`) already has the fix this section asks for.**
>   `run_video`'s Tier 4.2 code (`video.py:2459-2537`, comment dated to a prior session) chunks
>   `propagate_in_video` into successive capped windows, re-seeds tracked objects at each window
>   boundary (`_reseed_window`), and falls back to the nearest tracked frame's mask for any needed
>   frame chunking still misses — so "masks simply stop past the cap → everything locks" cannot
>   happen even in the capped mode. Real regression this fixed is documented inline: an uncapped-
>   window baseline on a 75-frame test clip covered only 23/38 sampled frames (68% missing) before
>   this chunking was added.
> - **The "predicted artifact" (depth_prev_final/running_confidence resetting at chunk seams) does
>   not apply.** That state lives in `PropagationState`/`flow_state`, which is driven by `run_video`'s
>   outer per-frame loop continuously — it has no dependency on how the *mask* was chunked. SAM3
>   window boundaries and depth-propagation frames are on separate axes; a mask-chunk seam does not
>   reset depth state. (A *mask*-level seam artifact is still possible — the nearest-frame fallback
>   at an uncovered boundary frame is a real, if rare, discontinuity — but that's a masking artifact,
>   not the depth-state-reset this item predicted.)
> - **`len(seen) < 10` is stale — the actual threshold is `min_times_seen=3`**
>   (`video.py:1880`, used at `video.py:2260`, `2377`). `detect.py` and the `threshold(fg_mask, 200,
>   255)` shadow-killer referenced above do not exist in the current codebase (consistent with
>   CLAUDE.md's note that some doc-referenced files are historical) — false-positive source list is
>   otherwise still plausible (luminance shifts, compression, seam wobble, SAM3 over-extension) but
>   unverified against current code.
> - **Not done:** the false-negative inverse audit (high-flow-but-static-flagged pixels) and the
>   false-positive auto-demote fix (§7's own suggested fix, tracked separately as P2). Both need a
>   real run with masks + flow dumped and per-pixel comparison — out of scope for a code-only audit.
>
> **Follow-up 2026-08-18: does the missing default cap explain the unmet <16GB VRAM target on
> long clips (`trop_long_embouteillage`, 18.2GB as of 2026-07-31)?** Already investigated and
> answered in `.claude/B1_VRAM_TIME_REDUCTION_PLAN.md` (lines ~290-338) — **no, chunking is not a
> viable lever, and this was tested/decided, not just theorized:**
> - VRAM is SAM3-bound (memory-bank + image-encoder activations, not WAFT/depth). On
>   `trop_long_embouteillage`, resolution levers (`SPAG_SAM3_SCALE`/`MAXSIZE`) plateau at ~15.3GB
>   while collapsing mask IoU to 0.28 on every frame — evidence the floor scales with frame count /
>   tracked-object count (SAM3 memory-bank growth over 1349 frames), not per-frame resolution.
> - `SPAG_SAM3_MAX_FRAMES` chunking (the mechanism this section is about) was evaluated as a VRAM
>   lever for exactly this clip and **rejected by explicit user decision (2026-07-31)**: chunked/
>   re-seeded `propagate_in_video` windows on **one SAM3 session do not free memory-bank state** —
>   re-seeding costs extra inference and returns VRAM to baseline, it doesn't bound cumulative
>   growth. It's documented in B1 as "correctness fix only, not a useful [VRAM] lever."
> - The only mechanism that would actually bound memory-bank growth is periodic **fresh SAM3
>   sessions** (not windowed calls on one session) — but that reintroduces the same tracking-
>   continuity risk the original uncapped/unchunked attempt hit (23/38 frames covered, the same
>   regression this section's fix addresses). Judged not worth building for one outlier clip.
> - **Current accepted state:** `trop_long_embouteillage` (and other very-long clips) stay over the
>   16GB target under the best shipped config (bf16 + maxsize1536); the other 9/10 clips are
>   comfortably under (5.7–11.5GB). This is a known, accepted limitation, not an open item.

---

## 8. Compositing and point-cloud consequences

- **Feathering creates geometrically invalid depth.** Across a large discontinuity (person at
  2 m, wall at 10 m) the linear blend produces 4/6/8 m values corresponding to no surface. In a
  depth map this reads as harmless smoothing; in a point cloud it emits a visible smear
  connecting subject to background. Likely the most visually damaging item here.
  *Fix:* detect large `|object_depth - D_ref|` and switch to nearest/hard selection there,
  feathering only across small gaps; or keep the feather and emit per-pixel confidence, dropping
  feathered-across-discontinuity pixels from the splat.
- **NaN propagation.** If undefined `D_ref` is NaN, any spatial operator touching its
  neighbourhood spreads it across the whole kernel — a Gaussian blur turns one bad pixel into a
  blurred disc, and it compounds if the result feeds back into state. Same hazard in the flow
  warp (bilinear sampling) and the affine fit (one NaN → NaN solution).
  *Fix:* explicit boolean validity mask plus a finite sentinel, never NaN. Assert
  `np.isfinite(...).all()` before the splat stage.
- **Dilation radius is derivable, not magic.** Depth carries no detail beyond the backend's
  native resolution (DA360 518×1036; PaGeR `native_resolution` = 2·504 × 4·504), so the
  "depth-bleed halo" the dilation covers is largely a bilinear-upsampling halo, with width in
  full-res pixels scaling as `W_full / native_W`. Derive it from that ratio and it transfers
  across the `max_length` downscaling automatically.
- **ERP-anisotropic kernels.** Dilation and blur operate in pixel space, but ERP pixel density
  per solid angle explodes toward the poles, so the feather is much wider in angular terms at
  high latitude. Scale kernel width by `cos(latitude)`.
- **Alpha=1.0 hole-forcing** is correct while an object is present but undefined for the frame it
  leaves.

> **Confirmed, fixed, and benchmarked 2026-08-18.** `composite_bg_locked`/`_torch`
> (`flow_depth_propagation.py:227-256`) is a literal `alpha*object_depth + (1-alpha)*depth_ref`
> blend; `feather_dynamic_mask`'s defaults (`dilate_px=12, feather_px=9`) produce a ~28px transition
> band that, for a 2m/10m step, emits 20px of pure-interpolation depth spanning 2.6-9.4 m — a smear
> with no corresponding surface, as predicted.
>
> **Fix (opt-in): `SPAG_HARD_DEPTH_CUTOVER=1`.** Added `hard_depth_cutover` to both
> `composite_bg_locked` and `composite_bg_locked_torch` — simpler than the doc's own suggestions
> (no threshold to tune, no splat-stage filter): `alpha` stays soft for any other consumer, but the
> depth *value* is selected by nearest source (`alpha>=0.5 -> object_depth else depth_ref`) instead
> of interpolated. Wired into `run_video` at both composite call sites, env-gated, default off.
> Synthetic 2m/10m step test (`scripts/feather_hardcutover_test.py`): default shows 4800/4800
> phantom-depth px in the feather band (range 2.33-9.67m); with the flag, **0/4800**, range
> collapses exactly to `[2.00, 10.00]`.
>
> **Real-clip benchmark** (`SPAG_SAM3_MAXSIZE=768 SPAG_DETERMINISTIC=1 SPAG_OCCL_FBGATE=1`, off vs
> on, `scripts/run_bglock_audit.py`) on **MattSwift** (300f, moderate fg/bg gap) and
> **accident_electrique_02** (188f, larger gap):
> - Whole-frame stability metrics (`bg_depth_cv`, `fg_depth_cv`, `bg_spikes_per_frame`,
>   `fg_delta_mean`) are ~unaffected on both clips (differ only in the 3rd-4th significant digit) —
>   expected, since they average over solid mask regions, not the thin feather-band ring the fix
>   touches (consistent with "rule 11": no stability number without a paired fidelity number).
> - Direct per-pixel diff between off/on `depth_{idx}.npy`, all frames, is the metric that shows
>   the real effect:
>
>   | clip | frames affected | mean px differing | mean \|diff\| in differing px | max \|diff\| |
>   |---|---|---|---|---|
>   | MattSwift | 300/300 | 2.11% of frame | 0.030 m | 1.87 m |
>   | accident_electrique_02 | 188/188 | 0.84% of frame (0.45-1.20% range) | 0.020 m | 1.75 m |
>
>   Every frame in both clips affected; footprint stays a small single-digit-percent slice of the
>   frame (the feather band itself); the ~1.8m max tail on both clips is exactly the phantom-slab
>   effect being removed. Both clips agree closely despite the gap-size difference, suggesting the
>   effect size is governed by feather-band geometry more than scene depth range.
> - Visual artifacts saved (fixed-range colormap videos + a diff heatmap per project convention;
>   PLYs already saved per-arm by `run_bglock_audit.py`) in the scratchpad's `hardcut_bench/`
>   (session-local, not committed).
> - **Verdict: keep opt-in, do not flip the default.** The fix is real, confined to the feather
>   band as designed, and costs nothing in the stability metrics that gate production quality — but
>   nothing here shows it's an *improvement* (no fidelity/GT comparison exists for real clips), and
>   a hard silhouette boundary is a genuine behavior change that could hurt shallow, legitimately-
>   gradual depth gaps (a hand near a table) even as it helps large discontinuities. Ship as a
>   documented opt-in for clips with known hard fg/bg discontinuities, same scoping as
>   `SPAG_SINGLE_PASS`/`freeze_bg`.
>
> **NaN propagation is already handled correctly at the composite step**, contrary to the concern
> as stated. `depth_ref` can contain NaN (`video.py:764`, permanent-holes pixels only), but both
> composite functions guard it explicitly (`ref_undefined` → `safe_alpha=1.0, safe_ref=0.0`, so
> `depth_final = object_depth`, always finite). No NaN reaches the flow-warp bilinear sampling; the
> alignment fits also already gate on `isfinite`. Only missing: a blanket `isfinite` assert before
> the splat stage — low priority, belt-and-suspenders only.
>
> Dilation-radius-from-native-resolution and ERP-anisotropic kernels are P2, not touched.

---

## 9. Instrumentation — one diagnostics pass (T3)

These share most of their computation and are cheap once depth and flow are in memory. Dump to a
single CSV per run plus a few images:

- per-frame affine residual on static pixels — broadest single canary; rises for model drift,
  mask misses, and rig motion alike
- per-frame fitted **scale**, logged separately from the residual — for DA360 a scale trace that
  jumps when subjects enter frame is expected (see §10.3); a residual that jumps is not
- per-frame median flow vector over static pixels — detects rig motion (§11)
- `running_confidence` histogram per frame — detects §4.1
- depth variance split into static / dynamic / feather-band regions
- per-object mean in-mask flow magnitude — feeds the false-positive audit and the auto-demote rule
- `D_ref` sample-count map + the median RGB plate, saved once
- residual **against latitude** — see §10.3; answers two questions in one plot

> **Implemented and full-clip-verified 2026-08-18 (T3).** `SPAG_DIAG_CSV=<path.csv>` (rule 13:
> unset = zero new work/allocations/behavior change). All eight signals land as one row per frame:
> - `align_depth_frame`/`_gpu` gained an optional `diag_out: dict | None` — fitted `scale`/`shift`,
>   `n_static`, residual median/p95, and a full-frame `resid_full` + `static_mask` for latitude
>   binning. Numpy/GPU paths verified bit-identical on a synthetic frame.
> - Median flow over static pixels and `running_confidence` histogram both reuse existing
>   arrays/accumulators (§11 logic, `SPAG_CONF_HIST`'s trace) — no duplicate computation.
> - Depth variance by region: frame-to-frame median delta within `alpha<=0.02`/`>=0.98`/else
>   (static/dynamic/feather), using the actual composite weight rather than the coarser fused mask.
> - Per-object flow magnitude (mean `|flow|` per SAM3 object), `D_ref` sample-count map (saved once
>   from `compute_temporal_median`, feeds §5), and residual-vs-latitude (8 row bands, feeds §10.3).
>
> Full-clip run on `circulation_site_1_edit_coupe` (39 frames, 3840×1920) produced a complete
> 39-row CSV, all columns populated. Caught a **real bug** along the way: `alpha_t.cpu().numpy()`
> in the GPU composite path threw on bf16 tensors (`TypeError: Got unsupported ScalarType
> BFloat16`); fixed by casting to float32 before capture — exactly the kind of thing a synthetic
> unit test alone wouldn't have caught. Still not done: a QA plot/visualization of the CSV contents
> (only the raw CSV + sample-count `.npy` were produced and spot-checked numerically).

---

## 10. Backends

### 10.1 The contract both wrappers should satisfy

Both `predict()` methods return `(depth, mask)`, which makes them look interchangeable to
`core.py`. They are not:

| Property | DA360 | PaGeR | bglock assumes |
|---|---|---|---|
| Depth convention | undeclared | `"radial"` (declared) | radial, implicitly |
| Native detail cap | 518×1036 (implicit) | `native_resolution` (2·504, 4·504) | full ERP res |
| Invalid-pixel signal | `None` | `sky_mask` | all pixels valid |
| Scale semantics | median-anchored to 5.0 m | scale-invariant (`metric=False` default) | comparable across frames |
| Value range | unbounded (→10⁶) | clamped `[1e-2, 200]` | unbounded, finite |

**Neither backend returns metres.** DA360's "approximate meters" is anchored to a hardcoded 5.0 m
median; PaGeR skips both scale heads by default. Every absolute-valued threshold in bglock
(notably §8's discontinuity threshold) must be relative to the scene's own depth distribution.

*Fix (T8):* every backend declares all five as instance attributes; bglock reads them instead of
hardcoding. PaGeR already exposes two — mirror them on DA360 and add the rest to both. This is
the same seam MapAnything will plug into.

> **Done 2026-08-18.** Both wrappers now declare all five as instance attributes
> (`spag4d/da360_model.py`, `spag4d/pager_model.py`): `depth_convention`, `native_resolution`,
> `invalid_pixel_signal`, `metric`, `value_range`. DA360 gets `depth_convention="radial"` (its
> ERPCircularConv2d geometry already matches ERP ray directions — same family as PaGeR's declared
> convention, it just never said so), `native_resolution=(518,1036)`, `invalid_pixel_signal=None`
> (predict() truly has no validity output), `metric=False` (median-anchored to 5.0 m, not true
> metric), `value_range=(0.0, inf)` (confirmed unbounded, no clamp in the code). PaGeR gets the two
> it was missing: `invalid_pixel_signal="sky_mask"`, `value_range=(1e-2, 200.0)` (the
> `set_depth_range()` clamp applied in `.load()`). Declarative half only — bglock's hardcoded
> thresholds (e.g. §8's discontinuity threshold) still don't *read* these attributes yet; wiring
> that up is separate follow-on work, not done here.

### 10.2 PaGeR

Beyond §4.3 and §4.4:

- **Cube-face seam steps on moving objects.** Each of 6 faces is predicted from its own context
  and stitched afterwards, so an object crossing a face boundary is estimated by two different
  contexts and can step in depth. Static seams don't matter (locked to `D_ref` anyway) — but this
  is a **dynamic-region flicker source unique to PaGeR**, periodic in object *position* rather
  than time, so it won't look like normal flicker.
  *Test:* plot a tracked object's centroid depth against its longitude; look for steps near face
  boundaries (±45°, ±135° at the default 90° FOV). *Fix if present:* `cube_fov` above 90° with
  blended stitching.
- **Confirm the unprojection honours `depth_convention = "radial"`** — distance along the ray, no
  planar-depth correction. The risk is that the unprojection was written against DA360 and
  silently assumes whatever DA360 produces.

### 10.3 DA360

- **Affine space — resolved, with one open test.** `predict()` returns depth, so the fit is in
  depth space. That is defensible: a pure *scale* survives inversion, and DA360's shift MLP claims
  to remove the shift, so **scale-only in depth space** is correct. Residual risk: if any shift
  survives, `depth = 1/(s·disp + t)` is Möbius, not affine, and no depth-space fit corrects it —
  biased at far range where it's hardest to notice.
  *Test:* fit scale-only, full-affine-in-depth, and in disparity space; compare static residuals.
  If full-affine beats scale-only by a lot, the shift is leaking and the fit belongs in disparity
  space.
- **Per-frame median renormalization is content-dependent.** Each frame is rescaled so its median
  depth is 5.0 m, taken over *all* pixels — so when a large object enters or leaves, the median
  moves and the global scale of **every background pixel** changes with it. The affine fit absorbs
  it, but that means the fit is doing essential per-frame work rather than correcting small drift.
  *Fix:* expose a `scale_anchor` and normalize every frame to the `D_ref` plate's median instead.
  Removes the jitter at source and makes the fitted scale a meaningful diagnostic.
- **Declare and verify the depth convention.** The wrapper never says whether output is radial or
  planar.
  *Test — run this regardless:* both backends on the same static plate, scale-only fit, residual
  plotted **against latitude**. A convention mismatch appears as a smooth systematic
  latitude-dependent trend; genuine model disagreement appears as noise. The same plot answers
  whether a global affine is too coarse for ERP (a latitude-banded or low-order spherical-harmonic
  fit being the alternative). Two results, one experiment.

> **Done 2026-08-18** (`scripts/residual_vs_latitude_test.py`), two plates — resolved as a content
> confound, not a convention bug. Scale-only fit of DA360 to PaGeR on each plate's static content,
> residual binned into 16 latitude bands:
> - **MattSwift** (real indoor room): residual falls roughly monotonically from equator-ish (+17°,
>   0.97 m) toward the south pole (-84°, 0.30 m); correlation -0.64. Overall median residual 0.595 m
>   (~12% of the plate's ~5 m median depth — an order of magnitude above the fp16 floor from §6.7,
>   so real signal, not measurement noise). The residual *map* (not just the curve) is dominated by
>   scene content — window/ceiling-light hotspots (near-saturated, least reliable for both models)
>   carry the largest residuals, and this room's windows sit in the upper-middle latitude bands.
> - **`circulation_site_1_edit_coupe`** (different room, same script): residual **peaks sharply at
>   the equator** (-5.6°, 2.29 m) and falls off toward *both* poles (north 84°: 0.27 m, south -84°:
>   0.59 m) — a symmetric bump, not a monotonic slope. Plain linear correlation is only 0.19 (looks
>   like noise by that metric alone), but correlation against |latitude| (catches symmetric
>   U/peak shapes) is **-0.88** — just as latitude-structured as MattSwift, only shaped differently.
>
> **Verdict:** both plates show strong latitude structure (|corr| 0.64 and 0.88), but the two curves
> have different shapes (monotonic vs. equator-peaked). A single content-independent
> convention/geometry bug would produce the *same* shape regardless of room; two different shapes
> is exactly the scene-content-confound outcome, consistent with each room's difficult content
> (windows/lights) sitting at different latitudes. **Resolves the open question for these two
> plates**, though a 3rd room would still be worth it to rule out two-sample coincidence.



### 10.4 MapAnything migration

- **The cubemap plumbing already exists.** PaGeR vendors `erp_to_cubemap` / `cubemap_to_erp` from
  `src.utils.geometry_utils`, and `get_intrinsics_extrinsics(image_size=face_size, fov=cube_fov)`
  produces per-face intrinsics and relative extrinsics. Reuse these rather than adding
  `py360convert` alongside — a second implementation invites face-ordering and FOV mismatches.
- **One frame's 6 faces are a valid multi-view set** with known relative extrinsics, which is the
  input shape MapAnything wants. Caveat: they share an optical centre, so the **baseline is
  zero** — no parallax, only seam overlap. Methods relying on triangulation get nothing from
  this; what they get is cross-face consistency. Be explicit about which benefit is expected
  before investing.
- **MapAnything will not replace bglock.** The camera is fixed, so the background is the same view
  every frame: no temporal baseline, no temporal parallax, multi-view across time degenerates.
  It should improve per-frame geometry and cross-face consistency, not frame-to-frame stability.
  Everything in §4–§9 survives the backend change.
- **Carry over:** the `native_resolution` cap still bounds ERP output; and re-run the §10.2 seam
  test after the switch — a multi-view method may fix face-boundary steps (if it enforces
  cross-face consistency) or worsen them (if it adds per-face pose refinement). That test is the
  cleanest single indicator of whether the integration works.

---

## 11. Verify the camera is actually static

The entire zero-flicker guarantee rests on frame-to-frame background correspondence being the
identity. Real 360 rigs violate this: mount vibration, wind, thermal drift, rolling shutter,
in-camera stabilization.

*Test:* median WAFT flow vector over static-flagged pixels, per frame. Consistently non-zero means
`D_ref` needs a per-frame global warp (or at minimum a global 2D shift) before compositing. Cheap,
and worth doing once on `accident_electrique_02` before anything else is trusted.

> **✅ TESTED 2026-08-18 — camera confirmed static, no action needed.** Ran on the two canary
> clips (rule 15) with `depth_npy_dir` enabled and computed the per-frame median WAFT flow over
> `mask==0` pixels (`scripts/check_camera_static.py`):
>
> | clip | frames | mean \|median flow\| (px) | max (px) |
> |---|---|---|---|
> | `circulation_site_1_edit_coupe` | 77 | 0.031 | 0.062 |
> | `MattSwift` | 299 | 0.011 | 0.021 |
>
> Both consistently sub-0.1px — two orders of magnitude below anything that would move a depth
> alignment. dx/dy have a consistent sign across frames on both clips, but at this magnitude that
> reads as WAFT's own correlation-window bias on a static scene, not physical rig motion; not
> actionable regardless of cause. No per-frame global warp needed. Not run on
> `accident_electrique_02` as originally suggested — the two canary clips already cover a
> short/long, low/high-motion pair and agree, so a third clip wasn't needed to close this.

---

## 12. Priority index

| Priority | Item | Packet | Effort | Status |
|---|---|---|---|---|
| P0 | Confidence absorbing at zero (§4.1) | T1 | small | ✅ tested, fixed 2026-08-17 |
| P0 | Warp `prev_confidence` by flow (§4.2) | T1 | small | ✅ tested, fixed 2026-08-17 |
| P0 | Validity masks + robust fit (§4.3) | T2 | small | ✅ fit fixed 2026-08-17, metrics-side exclusion fixed 2026-08-18 (both magnitude gate); per-backend validity masks (§10.1) deferred, not needed |
| P0 | Pin CLIP indoor/outdoor label (§4.4) | T2 | small | ✅ audited false, defensive fix applied 2026-08-17 (not yet benchmarked, no live path) |
| P1 | Diagnostics pass (§9) | T3 | medium | ✅ implemented 2026-08-18, full-clip run 2026-08-18 (39-frame clip, 3840×1920) — found+fixed a real bfloat16→numpy crash in the GPU composite-alpha capture path along the way |
| P1 | Confirm camera is static (§11) | T3 | small | ✅ tested 2026-08-18, confirmed static (sub-0.1px median flow on both canary clips) |
| P1 | Residual-vs-latitude test (§10.3) | T3 | small | ✅ resolved 2026-08-18: 2 plates run — MattSwift monotonic (corr -0.64), circulation_site_1 equator-peaked (corr vs \|lat\| -0.88) — different shapes per room = content confound, not a room-independent convention bug |
| P1 | fp16 + outlier error floors (§6.7) | T7 | small | ✅ done 2026-08-18: fp16 floor ~0.0008 CV; 3/10 baseline clips are at/below it (noise, not signal); baseline predates §4.3 outlier fix; "0.037" headline figure unverified (doesn't match any recorded number) |
| P1 | Paired stability/fidelity metrics + ablation ladder (§6.0, §6.6) | T7 | medium | ✅ partial 2026-08-18: assembled 3/5-rung ladder (noprop/legacy/decay) from existing confdecay data — caught legacy's freezing-gaming trap in the act on MattSwift (0.69 fidelity); raw/affine-only rungs not run |
| P1 | Stride-invariance sweep (§6.1) | T7 | medium | ✅ run 2026-08-18: full pipeline at skip_step 1/2/3/5 on real 156-frame clip — **no accumulated propagation error** over 150 native frames (growth over baseline ±0.0003m, noise-level); disagreement present from frame 0 traced to differing D_ref plate composition per stride, not propagation drift; fps-coupling sub-question still open (confound dominates) |
| P1 | Mask-injection proxy GT (§6.3) | T7 | medium | open — needs a mask-harness modification + real run, heavier than time allowed this pass |
| P1 | Mask false-positive audit (§7) | T6 | small | ✅ code-audited 2026-08-18: coverage-gap premise (128-frame default cap, symmetric split) is false for the default path (no cap unless `SPAG_SAM3_MAX_FRAMES` set; chunked mode already re-seeds+falls back); `len(seen)<10` is stale, actual is `min_times_seen=3`; false-negative audit itself not run |
| P1 | Behaviour past `max_frame_num_to_track=128` (§7) | T6 | medium | ✅ resolved by the same audit above — folded into the §7 entry |
| P1 | Sample-count floor + two-pass plate (§5) | T4 | medium | ✅ measured 2026-08-18 on real clip: 96.7% mean coverage, 0.34% zero-sample (permanent holes, already NaN-guarded), <2% below the doc's K≈5-10 thin-coverage threshold; thin-coverage pixels confirmed **clustered, not scattered** (one component holds 85% of ≤10-sample pixels, mid-frame not at poles). **Fixed 2026-08-18**: `SPAG_DREF_MIN_SAMPLES=K` opt-in flag extends the existing zero-sample undefined/inpaint/NaN treatment to thin-coverage pixels (`compute_temporal_median`, default K=0 = old behavior); unit-verified, smoke-run end-to-end on a real clip (no regression, but that clip had full coverage so the threshold never fired); two-pass plate build still not implemented |
| P1 | Feathering smear across discontinuities (§8) | T5 | medium | ✅ confirmed+quantified+**fixed** 2026-08-18: ~28px band emitted 20px of pure-interpolation depth (2.6-9.4m) for a 2m/10m step; fix shipped opt-in as `SPAG_HARD_DEPTH_CUTOVER=1` (nearest-source depth select, soft alpha kept), verified 4800/4800→0/4800 phantom-depth px on synthetic test. **Real-clip benchmark done** (MattSwift + accident_electrique_02): whole-frame stability metrics unaffected (as expected, feather band is too thin to move them); direct per-pixel diff shows real effect confined to 0.8-2.1% of frame, mean ~2-3cm/max ~1.8m. **Verdict: stays opt-in**, not promoted to default — no fidelity/GT evidence it's an improvement, and it's a real geometry-behavior change (hard vs soft silhouette) that could hurt shallow legitimate gaps even as it helps large ones |
| P1 | NaN propagation (§8) | T5 | small | ✅ audited 2026-08-18: already correctly guarded at the composite step (`ref_undefined` fallback), NaN never reaches the flow warp; only the suggested blanket `isfinite` assert before splat is missing (low-priority belt-and-suspenders) |
| P1 | Backend declares all five properties (§10.1) | T8 | medium | ✅ implemented 2026-08-18: both `DA360Model`/`PaGeRModel` now declare `depth_convention`/`native_resolution`/`invalid_pixel_signal`/`metric`/`value_range`; bglock reading them dynamically instead of hardcoding is separate follow-on work |
| P2 | Time-reversal consistency (§6.2) | T7 | small | open |
| P2 | Synthetic radial/lateral/combined motion + floor sweep (§6.4) | T7 | medium | ✅ CLOSED 2026-08-18 — decay mechanism confirmed load-bearing on radial (4.6-17x better than legacy/frozen), inert-cost on lateral, no interaction when combined; conf_floor=0.5 not contradicted by cost-side sweep |
| P2 | Normals-vs-depth-gradient consistency (§6.5) | T3 | small | open |
| P2 | Cube-face seam steps (§10.2) | T2 | medium | open |
| P2 | `scale_anchor` from `D_ref` median (§10.3) | T2 | small | open |
| P2 | Dilation radius from native resolution (§8) | T5 | small | open |
| P2 | Auto-demote low-flow objects (§7) | T6 | medium | open |
| P2 | FB threshold in angular units | T1 | small | open |
| P2 | Decay constant tied to fps / half-life | T1 | small | open |
| P3 | `D_ref` slow EMA refresh (§5) | T4 | small | open |
| P3 | ERP-anisotropic dilate/blur kernels (§8) | T5 | medium | open |
| P3 | Cubemap decomposition of *flow* | — | large | open |

**Deferred.** Cubemap decomposition of the optical flow is the largest item and is premature
until the latitude-resolved FB-error measurement shows how bad mid-latitude flow actually is —
widening the pole margin may capture most of the benefit. Separate question from the cubemap
projection used for *depth* (§10.4); if it ever happens, use the same in-repo helpers.

**Other WAFT/ERP notes** (fold into T1): WAFT is trained on rectilinear imagery, so mid-latitude
flow (~45–70°) is degraded but not gated — measure FB-error against latitude to place the real
cutoff instead of picking a pole margin by eye. The 1.5 px FB threshold is resolution-dependent;
express it in degrees of arc so it transfers across `max_length` downscaling. The 0.85 decay is
fps-dependent; derive it from a target half-life in seconds:
`decay = 0.5 ** (1 / (fps * half_life_s))`. Confirm ERP seam wrapping in *every* operation — FB
consistency, pole-mask construction, mask dilation, and the proposed `prev_confidence` warp — as
one unwrapped operation produces a persistent artifact at longitude 0.
