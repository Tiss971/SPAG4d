# E1 — "pager" generator: cubemap + DA3 (MVD) over spatial + temporal windows

Written 2026-07-31. **Status: spatial-only pager (`active_generator="pager"`) ships in production; the temporal/MVD batching design below remains unbuilt.**

**Spatial-only already ships**: `spag4d/pager_model.py` (`PaGeRModel`, vendored `prs-eth/PaGeR`
DA3-Giant) wraps ERP → 6-face cubemap → DA3 → stitched-back ERP depth/normals/sky-mask, wired as
`active_generator="pager"`. No joint multi-frame batching yet — the upstream model supports it
natively (`DepthAnything3.forward()` takes `image: (B, N, 3, H, W)`, N a flat view axis, faces and
frames interchangeable), so extending the wrapper to N = 6 faces × T frames is an extension of
existing code. **Everything below is the real unbuilt work.**

## Motivation

`da360` (default) hallucinates on the human subject over consecutive frames in
`accident_electrique_02` (visible flicker/jerk spike). Not fixable by `bglock` (background-only) or
foreground flow-smoothing (smooths a systematically-wrong sequence into a smoothly-wrong one). Two
candidate causes: (1) ERP distortion — `da360` runs directly on equirectangular images, off the
depth model's training distribution; (2) no multi-view/temporal signal — `da360` is purely
per-frame monocular.

## Design

Reproject each ERP frame to 6 cubemap faces (known fixed pinhole intrinsics) and run DA3 jointly
across the **spatial axis** (6 faces, one timestamp — fixes cause 1) and the **temporal axis** (N
consecutive frames, same face — fixes cause 2). Easier for SPAG4D than vipe: the camera is fixed,
so every frame of a given face shares identical, exactly-known extrinsics/intrinsics — no SLAM, no
pose noise, unlike vipe's moving-camera case.

## Confirmed risk: DA3 has zero dynamic-object awareness

Checked the vendored DA3/vipe call chain (`vipe/priors/depth/dav3/api.py`, `model/da3.py`,
`vision_transformer.py`): no mask parameter anywhere in `inference()`/`forward()`; global attention
blocks attend jointly across every token from every view supplied, with nothing distinguishing a
spatial-view difference from a temporal one — a moving subject's tokens can freely blend across the
temporal window. vipe itself never feeds a dynamic mask into DA3 either; `world_lock.py` applies
`static_mask` only downstream to exclude dynamic pixels from *background* consensus, and vipe's own
docs list per-object temporal consistency as out of scope. One unexploited hook: `attn_mask` is
plumbed to the attention layer but no caller populates it. **Net: naively batching frames where the
subject moves risks smearing its geometry across time exactly as this project is trying to avoid.**

## Mitigation options (ranked by effort)

- **A — mask-gate batch composition per face (recommended start).** Use the existing SAM dynamic
  mask: static face → full 6×N spatiotemporal batch; dynamic face → spatial-only (6 faces, same
  timestamp), structurally immune to cross-time blending. Low effort — pure batch-selection logic,
  no model changes. Leaves the temporal-averaging benefit ungotten on exactly the region that needs
  it most (open gap).
- **B — flow-magnitude-adaptive temporal window (medium effort).** For dynamic faces, shrink the
  window by the subject's own WAFT flow magnitude instead of dropping to spatial-only outright.
  Only worth building after A is measured and shows a residual gap on near-static-but-present
  subjects.
- **C — populate `attn_mask` to block cross-frame attention on dynamic tokens (high effort).**
  Best of both (full temporal averaging for static content, zero blend risk for dynamic subjects) in
  one unified batch, but requires forking the vendored DA3 attention internals. Only worth it if
  A/B prove insufficient.

## Recommended sequence

1. Extend `PaGeRModel.predict()` to accept a frame window and pass N = 6×T to
   `DepthAnything3.forward()`, gated per Option A.
2. Benchmark against `accident_electrique_02` with the existing `eval_fg_point_stability.py` /
   `eval_bg_point_stability.py` harness; also run plain spatial-only `pager` as a zero-effort
   baseline check of whether cause 1 alone explains most of the hallucination.
3. Only if the subject still flickers within its own dynamic window, consider B then C.

## Open questions

- Does spatial-only-per-frame already fix most of the hallucination? Unmeasured.
- Cross-window stitching for static-content batches — vipe's independent 10-frame windows showed a
  ~7-frame periodic scale bias even with SLAM poses; SPAG4D removes the SLAM-noise contribution but
  may still need an overlap/log-scale reconciliation step. Not investigated.
- Window size vs. VRAM — a full 6×10 batch is much bigger than anything vipe runs; needs its own
  VRAM benchmark once Option A works, independent of the `SPAG_SAM3_SCALE`/`SPAG_SAM3_MAXSIZE`
  levers in `B1_VRAM_TIME_REDUCTION_PLAN.md` (different model).

## Explicitly not this plan's job

Per-Gaussian temporal identity/anti-popping (FreeTimeGS's job, see `D1_FUTURE_WORK_PLAN.md`);
reusing vipe's `world_lock` directly (confirmed static-background-only, `bglock` already covers
that ground) — this plan is specifically about the foreground/dynamic gap neither addresses.
