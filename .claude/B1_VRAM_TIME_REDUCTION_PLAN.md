# VRAM & time reduction — winner config `bglock_sol1_median_w5`

_Started 2026-07-20, last consolidated 2026-08-25. Target: the production winner
(`depth_correction=bglock` + temporal median smoother, window=5, `da360`). Compacted to
standing conclusions; full sweep tables live in git history / `benchmarks/`._

## Bottom line

- **VRAM is SAM3-bound.** After all shipped lossless work, the whole-run peak lives entirely in
  SAM3 segmentation (image-encoder activations + memory bank); WAFT flow and the per-frame depth
  loop are each only a few GB and are not the peak.
- **Cumulative shipped result (winner config), all lossless:** peak VRAM 37,092→20,353 MB (−45%),
  wall time (fast5) 148.8→88.7s (−40%), stability metrics unchanged to float32 rounding.
- **Target <16GB for any clip length is not fully met** — see the long-clip item below.

## Shipped & default-on (lossless unless noted)

All in `spag4d/video.py` / `spag4d/flow_depth_propagation.py`.

**VRAM**: float32 cast fix in `TemporalDepthSmoother` (was doubling to float64); free WAFT before
SAM3/depth loop (`del model; gc.collect(); empty_cache()`); `empty_cache()` at phase boundaries;
SAM3 state offload to CPU (`offload_state_to_cpu=True`, −10-15% tracking fps, −14% to more on
long/many-object clips).

**Time** (fast5 148.8→88.7s): batched ERP warp (32.3→9.6s); cached meshgrid/pole-trust-mask;
partition median in the smoother; closed-form normal-equations lstsq; GPU-resident depth chain
(`align_depth_frame_gpu`, torch smoother, `propagate_depth_via_flow_torch` — biggest single win,
guarded to the bglock+lstsq+median hot path, numpy fallback otherwise); composite-on-GPU
(`SPAG_GPU_COMPOSITE`, default on, depth-align block 11.4→5.8s, near-lossless mean|Δ| 0.0036m
confined to feather band); DA360 prefetch + async PLY writes (bit-identical, off critical path).

## `SPAG_SINGLE_PASS` — opt-in, default OFF

Derives the SAM magnitude mask from a unified bidirectional WAFT pass instead of separate
forward-mask + bidirectional-propagation passes. Real time win (−38% flow-phase wall on a
962-pair clip, −31% to −34% production gate), VRAM-neutral. **Real fg-depth regression**, not a
SAM3 tracking issue (mask + prompt schedule byte-identical to baseline): the shared flow array
needs `seam_pad=0` for mask derivation but `seam_pad=64` for ERP-wraparound propagation, so
foreground depth loses seam correction on seam-adjacent frames (drops fg entirely on ~34/158
scene01 frames; background untouched, locked to `depth_ref`). Full fix (`SPAG_SP_FIX_SEAMGAP`,
recompute a `seam_pad=64` flow for propagation) restores exactness but is +15% slower than
baseline — proven dead end, kept as scaffolding only. Conditional seam padding (re-pad only near
the seam) is the only remaining avenue with any win; see `docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md`.
**Verdict: stays opt-in, safe only when there's no seam-crossing foreground motion.**

## SAM3 VRAM levers (the peak)

- **`SPAG_SAM3_BF16` (opt-in, default off)** — casts SAM3's own weights to bf16 (norm layers kept
  fp32-compute) in the vendored `third_party/sam3` fork. Solo: VRAM −17.5% (fast5), time −14%
  (bf16 tensor-core throughput), all stability metrics within noise, mask IoU 0.99 vs baseline at
  full production scale (MattSwift). **Safe alone.**
- **`SPAG_SAM3_SCALE` (opt-in, default off)** — downscales frames SAM3 decodes; real VRAM lever
  (−51% to −77% at 0.5). **Clip-dependent mask regression**: thin/articulated foreground (a
  person's limbs, workshop tools) can fall below the downscaled input's effective resolution and
  get cropped — confirmed on MattSwift (mean IoU 0.34, legs lost every frame) and atelier_1 (IoU
  0.38, erratic), while blockier-subject clips (scene01, Dispo_RDV) stay clean (IoU 0.97-0.78).
  Root cause: `SPAG_SAM3_SCALE` multiplies the already-WAFT-shrunk `viz_frames` shape, not native
  resolution, so 0.5 was actually a 0.2-0.25× factor off true native res. **No safe blanket
  default — per-clip-gated only, and the failure isn't predictable from resolution or object count,
  only from the clip's foreground subject shape.**
- **`SPAG_SAM3_MAXSIZE` (opt-in, default off)** — caps the longer edge of *true native* resolution
  instead of the WAFT-shrunk shape; recovers most mask quality lost to `SCALE` (MattSwift IoU
  0.34→0.93, atelier_1 0.38→0.87) for ~1.1-1.3GB less VRAM saving. **Do not stack with
  `SPAG_SAM3_BF16`**: full 10-clip validation showed the combination reintroduces the exact
  MattSwift/atelier_1 leg-loss regression (IoU back to 0.35/0.40, deterministic, confirmed
  bit-for-bit reproducible on the same GPU) even though each flag is safe solo (bf16 alone IoU
  0.99 at full res; maxsize alone IoU 0.93 at fp32) — bf16's precision loss removes just enough
  margin on an already-downscaled thin-limb input to flip leg detection.
- **VRAM predictive model**: `vram_max_mb ≈ 9629 + 427·(Mpx) + 2626·n_objects` (R²=0.66, 10-clip
  fit) — resolution and tracked-object count are the two real levers, ~2.6GB per tracked object.
  Rough triage only (±20-40% per clip); `circulation_site_1_edit_coupe`'s VRAM sits far below what
  resolution alone predicts (shortest clip, 156 frames) — evidence the memory-bank footprint grows
  with frame count toward a plateau, not purely resolution/object-count.
- **Long-clip VRAM floor — target not met, accepted (user decision 2026-07-31).**
  `trop_long_embouteillage` (1349 frames, modest native res) plateaus at ~15.3GB from
  `SPAG_SAM3_MAXSIZE=1024` down to 768 (mask IoU collapses to 0.28 with zero further VRAM gain) —
  the floor is driven by SAM3 memory-bank/tracking-state growth over frame count, not per-frame
  resolution, so resolution levers can't close the gap. Stays at 18.2GB under bf16+maxsize1536,
  over the <16GB target. Fixing it would need periodic fresh SAM3 sessions (not just windowed
  `propagate_in_video` — that doesn't free memory-bank state), which carries the same
  tracking-continuity risk the original uncapped chunking attempt hit. **Judged not worth it for a
  single outlier clip; the other 9/10 clips are comfortably under (5.7-11.5GB).**
- **`SPAG_SAM3_MAX_FRAMES` chunking** (shipped opt-in) — correctness fix only, not a VRAM lever.
  Re-seeds each window via a centroid `add_prompt` because chaining `propagate_in_video` on one
  session silently drops tracked objects without it.
- **`SPAG_SAM3_FP16`** — removed. `autocast` only casts activations; weights (the dominant VRAM
  cost) stay fp32, no benefit.
- **`SPAG_MASK_SCALE`** — removed. Premise was wrong: WAFT is freed before SAM3 runs (measured
  WAFT 2.7GB vs SAM3 23.8GB, ~8.8× peak), so shrinking already-freed WAFT tensors can't lower a
  later peak. Where it ever helped, the mechanism was coarser SAM3 contours → smaller memory bank,
  a quality-coupled side effect, not a clean knob; item 3 (`SCALE`) is the real lever.

## Not pursued / deferred

fp16 depth (targets a ~4s stage, zero VRAM benefit now the peak is SAM3-bound); CUDA graph capture
of the depth chain (not branch-free at the Python level, would need a correctness-risk rewrite);
depth-loop parallelism (causal — propagation consumes the previous frame's composited depth);
host-RAM levers (relevant to RAM on long clips, not the VRAM metric).

## Notes / gotchas

- `freeze_bg` has no effect on stability/VRAM (identical to 6 sig figs); `True` (default) is just
  +35-40s/video cheaper.
- Multi-GPU: set `CUDA_DEVICE_ORDER=PCI_BUS_ID` before `CUDA_VISIBLE_DEVICES`, or an index can
  silently resolve to the DGX Display device → silent OOM.
- Back-to-back single-GPU runs throttle (+20s on a short clip); cool to ≤46°C between runs.
