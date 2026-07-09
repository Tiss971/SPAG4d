# Depth-stability rework + benchmark (fixed 360 ERP camera)

## Problem (user)
Per-frame monocular depth (DA360) is inconsistent **frame-to-frame** (same
pixel, different depth) and **pixel-to-pixel within a frame**. Any post-hoc
alignment of such an estimate is fighting noise, so it "will always be hard".
Goal: **background must be fixed, moving objects must move correctly in it.**

## Key insight
The camera never moves. So every pixel outside a moving object is static
background whose true depth is **constant in time** and already known
artifact-free from the reference depth `D_ref` (DA360 on the temporal-median
background). Re-estimating + affine/flow-correcting it each frame only
reinjects monocular noise as flicker. Don't re-estimate it — **lock it**.

## Three strategies compared
- **A. affine** (current pipeline): fresh DA360 every frame, `s·D+t` aligned to
  `D_ref` over static pixels.
- **B. flowprop** (`propagate_depth_via_flow`): warp previous stabilized depth
  forward with WAFT flow, blend back to monocular by a decaying confidence.
- **C. bg-locked** (`composite_bg_locked`, new): static pixels = `D_ref`
  exactly (zero temporal variance by construction); the flow-propagated object
  depth is used **only** inside a feathered dynamic mask. Directly encodes
  "background fixed, objects move".

## Result — `9_MattSwift.mp4`, 150 frames, native 2048×1024, PLY stride 4
(dynamic mask ≈ 3% of frame, i.e. a person moving in a room)

| method | bg_temporal_std (m) | bg_flicker p2p (m) | fg_temporal_std (m) | spatial_rough |
|--------|--------:|--------:|--------:|--------:|
| A. affine (current) | 0.0974 | 0.0322 | 0.3536 | 0.02535 |
| B. flow-prop        | 0.0847 | 0.0059 | 0.2797 | 0.02248 |
| **C. bg-locked (new)** | **0.0001** | **0.0000** | **0.2367** | 0.03095 |

- **Background is now fixed**: temporal std 0.097 m → 0.0001 m (~1000×), p2p
  flicker 0.032 m → 0.000 m.
- **Objects still move**: foreground std stays nonzero (0.24 m) — and *lower*
  flicker than affine (0.35 m), because the object depth is flow-propagated.
- bg-locked's slightly higher spatial roughness is a boundary artifact: the
  metric's static region includes the feathered mask edge where the smooth
  `D_ref` meets the noisier object depth. Widening `feather_px` trades this off.

## Reproduce
```bash
conda run -n spag4d python benchmark_depth_stability.py \
  --video /raid/mb273924/SPAG4d/TestImage/9_MattSwift.mp4 \
  --output ./benchmark_depth_stability_mattswift \
  --max-frames 150 --ply-stride 4 --ply-export-count 5
```
Outputs `stats.json`, `comparison.png`, and stride-4 PLY sequences for
`affine` and `bglock` (viewer A/B check of background stillness vs object
motion). Flow runs at ≤1024px (`--flow-max-size`) then upscaled; depth /
compositing / PLY run at native resolution.

## Caveats / next
- The dynamic mask here is a WAFT-flow-magnitude proxy, not SAM3. bg-locking
  is only as good as the mask: a missed moving-object pixel gets frozen to
  background depth. In the real pipeline, feed the SAM3 mask into
  `composite_bg_locked` (it already dilates+feathers to absorb mask lag/halo).
- Radial motion (object moving straight toward the fixed camera) has ~zero
  flow; flowprop's decaying confidence re-anchors it to monocular depth, so
  bg-locked inherits that correction inside the mask.
- Source is 2048×1024 (the available "4K-class" ERP test clip); the pipeline is
  resolution-agnostic — pass true 4K footage unchanged.
