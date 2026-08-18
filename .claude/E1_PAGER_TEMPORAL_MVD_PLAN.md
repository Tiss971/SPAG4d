# E1 — "pager" generator: cubemap + DA3 (MVD) over spatial + temporal windows

_2026-07-31. Written after investigating vipe's own MVD/world_lock code to ground the
design in what's actually proven to work (and not work) there._

**Correction (2026-07-31, same day):** the spatial half of this plan is not a future
design — it already exists and ships. `spag4d/pager_model.py` (`PaGeRModel`) wraps a
vendored `prs-eth/PaGeR` (DA3-Giant) checkpoint and is wired into `core.py` as
`active_generator="pager"` today: ERP → 6-face cubemap → DA3 → stitched-back ERP
depth/normals/sky-mask. That covers exactly "Step 1" below (spatial-only, 6 faces × 1
frame). It has `global_scale_factor`/`temporal_consistency` params, but those are a
post-hoc scalar rescale + a CLIP-scale-head toggle, **not** joint multi-frame DA3
inference — no batching across timestamps happens anywhere in the current wrapper.
The upstream model does support it natively (`DepthAnything3.forward()`,
`pager_arch/PaGeR/src/depth_anything_3/api.py:59`, takes `image: (B, N, 3, H, W)` — N
is a flat view axis, faces and frames are interchangeable to it), so extending the
wrapper to pass N = 6 faces × T frames per call is an extension of existing code, not
new architecture. **Everything from "Option A" onward in this doc is still unbuilt and
is the real remaining work; the "Recommended sequence" below should start at step 2,
not step 1.**

## Motivation

`active_generator="da360"` (current default) hallucinates badly on the human subject over
a *batch* of consecutive frames in `accident_electrique_02` — visible as the huge
flicker/jerk spike in that scene's `comparison.html` (see benchmarks/accident_electrique_02).
This is not the kind of error `bglock` or the foreground WAFT-flow-propagated smoothing can
fix: `bglock` is background-only by construction, and smoothing a *systematically wrong*
depth sequence (the model is wrong for many consecutive frames, not just noisy) just
produces a smoothly-wrong result.

Two candidate root causes, not mutually exclusive:
1. **ERP distortion.** `da360` runs the depth model directly on the equirectangular image;
   a human standing away from the equator is geometrically warped in a way the model's
   training distribution (ordinary perspective photos) never saw.
2. **No multi-view/temporal signal at all.** `da360` is purely per-frame monocular, so it
   has nothing to anchor scale/geometry against besides its own single-image prior.

## Design: pager (ERP → cubemap) + DA3 batched over faces × frames

Reproject each ERP frame to 6 cubemap faces (known, fixed intrinsics — pinhole, same for
every face) and run **DA3** ("mvd" — Depth Anything 3, the same model vipe calls
`mvd`/`mvd_giant` via `MultiviewDepthProcessor`,
`vipe/priors/depth/dav3/api.py::DepthAnything3Model`) jointly across:

- **Spatial axis** — the 6 faces of one timestamp. Fixes root cause 1 (perspective input,
  known cross-face poses give genuine multi-view geometric constraint).
- **Temporal axis** — N consecutive frames of the *same face*. Fixes root cause 2.

**Why this is easier for SPAG4D than for vipe:** `DepthAnything3Model.inference()` takes a
flat `list[image]` + `(N,4,4)` extrinsics + `(N,3,3)` intrinsics — architecturally it makes
no distinction between "different camera, same instant" and "same camera, different
instant," so a joint `6 × N`-image batch is exactly the same call shape vipe already uses,
just with a richer pose graph. vipe needs SLAM to get relative poses across its temporal
window because its camera moves; SPAG4D's camera is fixed, so every frame of a given face
shares **identical, exactly-known** extrinsics/intrinsics — no pose estimation, no SLAM
noise. This is a strictly better-conditioned version of what vipe already ships.

## Confirmed risk: DA3 has no dynamic-object awareness — investigated 2026-07-31

Before assuming temporal batching is free, I checked whether DA3 (or vipe's use of it) has
*any* mechanism to keep a moving subject's geometry from bleeding across the temporal
frames in a batch. It does not:

- `DepthAnything3Model.inference()` and the underlying `DepthAnything3Net.forward()`
  (`vipe/priors/depth/dav3/api.py:247`, `.../dav3/model/da3.py:96`) take only
  `image, extrinsics, intrinsics` — no mask parameter at any level of the call chain.
- Architecture is standard VGGT/DUSt3R-style alternating local/global attention
  (`vision_transformer.py:281`): local blocks attend within one frame, **global blocks
  attend jointly across every token from every view supplied**. Nothing distinguishes a
  spatial-view-difference from a temporal-view-difference — a moving human's tokens across
  the temporal window can freely attend to and blend with each other.
- `depth_conf` output exists but is used for sky-masking / metric-scale alignment
  (`heads.py:263`, `da3.py:308`), never documented or wired as a dynamics/motion signal.
- vipe itself never feeds a dynamic mask *into* DA3 — `static_mask` is applied only
  *downstream*, in `world_lock.py::consensus_log_residual` (lines 73–101), to exclude
  dynamic pixels from the **background** consensus. DA3's own output for dynamic regions is
  whatever the model produced, unguarded. `world_lock` is a static-background-only
  stabilizer by design (docstring: "dynamic pixels genuinely move, so a static-scene
  consensus does not apply") and vipe's own docs (`docs/world_lock.md §8`) list
  per-object temporal consistency as explicitly out of scope/deferred — vipe hit the same
  wall and did not solve it.
- One unexploited hook found: an `attn_mask` kwarg is plumbed all the way down to the
  attention layer (`vision_transformer.py:305` → `heads.py:88` → `dinov2/layers.py:47`) but
  **no caller in the vendored `dav3/` tree ever populates it** — architecturally available,
  not currently used by anyone.

**Net: naively batching N frames of a face where the human actually moves is a real risk,
not just theoretical** — the model could average/smear the person's geometry across
timesteps exactly the way this project is trying to avoid, since it has no notion that
"person at t" and "person at t+k" are not the same static point.

## Mitigation options (ranked by effort)

### Option A — mask-gate the batch composition per face (recommended starting point)

Use the SAM dynamic mask (already computed every run) to decide, per face per frame,
whether that face's DA3 batch includes temporal neighbors:

- **Face has no dynamic content at time t:** full spatiotemporal batch (6 faces × N
  frames) — safe, maximum stability, no identity-blend risk since nothing moves.
- **Face has dynamic content (human) at time t:** spatial-only batch (6 faces, same
  timestamp) for that frame — pure multi-view geometric constraint, zero cross-time
  mixing, so the identity-blend risk is structurally impossible. Still fixes root cause 1
  (ERP distortion) for the frames that need it most; does not get the temporal-averaging
  benefit on exactly the region that needs it, which stays an open gap (see below).

**Effort/risk: low.** Pure scheduling/batch-selection logic ahead of the existing DA3
call — no model/architecture changes, no forking DA3's wrapper. This should be the first
thing built and benchmarked.

### Option B — flow-magnitude-adaptive temporal window

For dynamic faces, don't drop to spatial-only outright — shrink the temporal window based
on the subject's own measured displacement (SPAG4D already computes dense WAFT flow every
run): large per-frame flow magnitude in the dynamic mask → short/no temporal window (falls
back toward Option A); small magnitude (person nearly still) → longer window, since a
nearly-static subject poses little identity-blend risk and gets real temporal-averaging
benefit. Reuses flow SPAG4D already has; no new signal needed.

**Effort/risk: medium.** Requires picking a flow-magnitude → window-size mapping and
validating it doesn't reintroduce the blend artifact at the boundary case (small-but-real
motion). Worth doing only after Option A is measured and shows a residual gap on
near-static-but-present humans.

### Option C — populate `attn_mask` to fork DA3's attention

Exploit the already-plumbed-but-unused `attn_mask` hook to explicitly block cross-frame
(temporal) attention for tokens inside the dynamic mask region, while leaving cross-face
(spatial, same-timestamp) attention and all-frame attention for static tokens untouched.
This would give the best of both: full temporal averaging for background/static content in
every batch, and zero identity-blend risk for the dynamic subject, in a *single* unified
batch (no per-face branching needed).

**Effort/risk: high.** Requires forking/patching the vendored DA3 wrapper (or vipe's copy
of it) to actually construct and pass a per-token attention mask derived from the
projected dynamic-region tokens — real model-internals surgery, plus verifying the mask
shape/semantics match what `Attention.forward` expects (`dinov2/layers.py:47`). Only worth
it if A/B prove insufficient on the actual problem clips.

## Recommended sequence

```
1. DONE — pager (ERP → 6-face cubemap reprojection), spatial-only batch (6 faces × 1
   frame), ships as active_generator="pager" (spag4d/pager_model.py + core.py).
   Root cause 1 in isolation is already testable today.
2. Extend PaGeRModel.predict() to accept a list/window of frames and pass N = 6 faces
   × T frames to DepthAnything3.forward() in one call, gated per Option A (mask-gated
   spatial-only-for-dynamic-faces, full spatiotemporal otherwise) — cheapest temporal
   extension, no DA3 model modification, just wrapper + batch-composition changes.
3. Benchmark against accident_electrique_02 (the motivating clip) using the existing
   eval_fg_point_stability.py / eval_bg_point_stability.py harness — same instrument
   already used for the affine-vs-bglock comparisons, so results are directly comparable.
   Also worth a baseline run of plain `active_generator="pager"` (spatial-only, no
   temporal) against that clip first, since that's a zero-effort check of whether root
   cause 1 alone already explains most of the hallucination.
4. Only if the human still flickers materially within its own dynamic window (i.e. Option
   A's spatial-only fallback isn't enough even though cross-time blending is avoided):
   consider Option B, then C.
```

## Open questions (not yet answered)

- **Does spatial-only-per-frame already fix most of the hallucination?** If root cause 1
  (ERP distortion) dominates, Option A alone might be sufficient and B/C are unnecessary
  complexity — this needs to be measured, not assumed, before investing further.
- **Cross-window stitching for the static-content batches.** vipe's own independent
  10-frame windows produced a ~7-frame periodic scale bias even with SLAM-quality poses
  (`docs/world_lock.md §5`); SPAG4D's fixed camera removes the SLAM-noise contribution but
  independent-window stitching may still need an analogous overlap/log-scale reconciliation
  step for the static-face batches. Not investigated yet.
- **Window size vs. compute/VRAM.** vipe capped its purely-temporal window at 10 (with
  3-frame overlap) for a reason; SPAG4D's spatial axis multiplies that (6 faces), so a
  full `6×10` batch is a much bigger single DA3 call than anything vipe runs today. Needs
  its own VRAM benchmark once Option A is working end-to-end, independent of the
  `SPAG_SAM3_SCALE` VRAM lever ([B1](B1_VRAM_TIME_REDUCTION_PLAN.md)) which is a different
  model.

## Explicitly not this plan's job

- Per-Gaussian temporal identity / anti-popping — FreeTimeGS's job, unchanged from
  [D1](D1_FUTURE_WORK_PLAN.md).
- Reusing vipe's `world_lock` directly — confirmed static-background-only; SPAG4D's own
  `bglock` already covers that ground for background. This plan is specifically about the
  foreground/dynamic gap neither `world_lock` nor `bglock` addresses.
