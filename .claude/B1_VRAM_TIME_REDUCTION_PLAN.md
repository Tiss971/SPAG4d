# Plan: Reduce VRAM (and time) for the winner config `bglock_sol1_median_w5`

_Dated 2026-07-20. Target: the production winner from `benchmark_solutions/SUMMARY.json`._

## Context

The 14-video benchmark picked `bglock_sol1_median_w5` (depth_correction=`bglock` +
temporal median depth smoothing, window=5) as the best temporal-stability config,
but at **+28% peak VRAM (35.6 → 45.6 GB)** and +70% time vs baseline. Goal: bring VRAM
down first (time is a welcome secondary); fp16 depth is acceptable **if** a re-benchmark
confirms the stability metrics still hold.

Investigation of `spag4d/video.py`, `spag4d/flow_depth_propagation.py`,
`spag4d/da360_model.py`, and `spag4d/detect_opticalflow.py` found the VRAM/time is
**not** in the smoother itself (it's a bounded 5-frame CPU deque). The real costs are:

- **A suspicious +6.7 GB from `sol1` alone** even though smoothing is CPU-only — prime
  suspect is `np.median`/`np.tensordot` promoting depth from **float32 → float64**
  (`TemporalDepthSmoother.__call__`, `video.py:724-731`), which then flows into the GPU
  tensors built by `to_gaussians` (`video.py:872`), doubling their size and forcing fp64
  CUDA math. (Note the SUMMARY VRAM deltas are near-perfectly additive: baseline 35610,
  +3307 bglock, +6666 sol1, +9981 combined — so both are real allocations, not noise.)
- **WAFT never released**: `WAFTWrapper` is loaded at `video.py:457` and kept resident on
  GPU through SAM3 and the entire per-frame depth loop, though unused after the flow
  phase ends (`video.py:500`). No `del`/`empty_cache()` anywhere in `video.py`.
- **DA360 runs fp32**: it has a built-in `autocast(enabled=self.mixed_precision)` path
  (`da360_arch/.../da360.py:74`) but `mixed_precision` defaults `False` and `DA360Model`
  never enables it.
- **Redundant WAFT double-pass** (time, not VRAM): `model.run` computes forward flow then
  discards it, keeping only magnitude masks (`detect_opticalflow.py:205-231`); bglock then
  runs a *second* bidirectional pass (`video.py:485-500`, in-code
  `TODO: FACTORISE TO HAVE ONE PASS ONLY` at line 490).

## Approach (ordered; VRAM-first)

### Tier 0 — Instrument (prerequisite, ~30 min)
Add lightweight peak-VRAM probes at phase boundaries so every change below is measured,
not guessed. Wrap key phases with `torch.cuda.reset_peak_memory_stats()` +
`torch.cuda.max_memory_allocated()` logging: after the flow phase (`video.py:500`), after
SAM teardown, and inside the per-frame depth loop. Confirm the current per-phase peak
split (WAFT-resident vs depth-loop) before touching code.

### Tier 1 — Lossless VRAM wins (no re-validation needed; output should be identical)
1. **Fix the float64 promotion in the smoother.** In `TemporalDepthSmoother.__call__`
   (`video.py:721-731`) cast the returned array back to `float32`
   (`return np.median(stack, axis=0).astype(np.float32, copy=False)`, same for the
   gaussian/tensordot branch and the passthrough). **First verify** by printing
   `aligned_depth_np2.dtype` with/without smoothing; if float64, this is likely the bulk
   of the +6.7 GB `sol1` VRAM. Cheapest, highest-value VRAM fix.
2. **Free WAFT before SAM3/depth loop.** After the flow phase (`video.py:500`, inside the
   `try`) add `del model; gc.collect(); torch.cuda.empty_cache()` (free only after the
   bglock second pass, or after the Tier-3 factorization). Removes WAFT's resident
   footprint during the depth loop.
3. **`empty_cache()` at phase boundaries.** After WAFT free and after SAM3
   `close_session`/`shutdown` (`video.py:1606-1612` / `1715-1720`) so cached blocks are
   returned before the depth loop's peak.

### Tier 2 — Validated fp16 depth (gated behind re-benchmark)
4. **Enable DA360 mixed precision.** Thread a `mixed_precision=True` flag through
   `DA360Model.load`/construction (`da360_model.py:111`) into the arch's existing
   `autocast` path (`da360.py:74`). Roughly halves depth-model activation VRAM and speeds
   the per-frame hot path. **Gate:** re-run the 14-video benchmark on the winner config
   and confirm `bg_depth_cv`, `fg_depth_cv`, `bg_spikes_per_frame`, `fg_delta_mean` stay
   within ~5% of the current winner values in `benchmark_solutions/SUMMARY.json`.

### Tier 3 — Time win + host-RAM (secondary)
5. **Factorize the WAFT double-pass** (`video.py:478` + `485-500`): make `model.run`
   optionally return per-pair forward flow, and derive the magnitude mask from the
   seam-padded bidirectional pass so WAFT runs the frames **once** instead of ~3×.
   Recovers most of bglock's +220s. Also lets Tier-1 step 2 free WAFT immediately after a
   single unified pass. (Time, not VRAM.)
6. **Keep `flows_fwd/bwd` at flow-res** (≤1024) and `upscale_flow` lazily per-frame in the
   loop instead of accumulating native-res arrays for all pairs (`video.py:495-499`).
   Host-RAM only (O(N·H·W) → O(N·h·w)); relevant for long clips, not the VRAM metric.
7. **Make `aligned_depth_list_cpu` optional** (`video.py:633, 853, 934-937`) — it holds
   every frame's full-res depth in host RAM only to compute an end-of-run std map; gate it
   behind a `--depth-std-diagnostic` flag. Host-RAM only.

## Critical files
- `spag4d/video.py` — smoother dtype (721-731), WAFT load/free (457, 500), second flow
  pass (485-500), flows accumulation (495-499), depth loop (751-897), depth-list
  (633/853/934-937).
- `spag4d/da360_model.py` — `load`/`predict` (111, 163+) for the `mixed_precision` flag.
- `da360_arch/.../da360.py:74` — existing autocast path to activate.
- `spag4d/detect_opticalflow.py:205-231` — `WAFTWrapper.run` (for the factorization).
- `spag4d/cli.py` — expose new flags (`--depth-mixed-precision`, `--depth-std-diagnostic`).

## Results — Tier 0+1 implemented & validated (2026-07-20)

Implemented in `spag4d/video.py`: `import gc`; float32 cast in `TemporalDepthSmoother`
(Tier 1.1); `del model; gc.collect(); torch.cuda.empty_cache()` after the flow phase
(Tier 1.2); `empty_cache()` after SAM3 (Tier 1.3); cumulative `[VRAM]` phase probes
(Tier 0, no `reset_peak_memory_stats()` — an initial reset corrupted the benchmark's
whole-run peak read and was removed).

Single-video regression on **MattSwift** (300 frames, `bglock_sol1_median_w5`, da360):

| phase | cumulative peak VRAM |
|---|---|
| flow (WAFT freed) | 2,717 MB |
| **SAM3 segmentation** | **23,758 MB ← whole-run peak** |
| depth loop | no further increase |

- **Whole-run peak: 37,092 MB → 23,758 MB (−36%), lossless.** The old peak lived in the
  **depth loop**, inflated by float64 gaussian tensors (Tier 1.1), resident WAFT (1.2),
  and un-released cache (1.3); it is now ~2.6 GB. The binding constraint is now **SAM3**.
- **Losslessness confirmed**: MattSwift stability metrics identical to old to 3–4 sig
  figs (float32 rounding): bg_depth_cv 0.0004627→0.0004602, bg_delta_mean
  0.0009239→0.0009251, fg_delta_mean 0.0070054→0.0069700, fg_depth_cv 0.8264→0.8236,
  spikes 0→0.
- **Time**: 462 s (batch) → 549 s (single-video) is a **cold-start artifact** (DA360
  load + cuDNN autotune amortized in the 14-video batch, paid inline here), not a
  regression — the loop is byte-identical.

### Revised next step (supersedes Tier 2 for the VRAM goal)
Since the peak is now **SAM3 (23.8 GB)**, fp16 depth (Tier 2) would **not** lower the
peak — the depth loop is only ~2.6 GB. To cut VRAM further, target SAM3 instead:
- **Tier 1.5 — SAM3 VRAM**: inspect `segment_with_flows`/`segment_with_sam`
  (`video.py:~1223/1638`). `offload_video_to_cpu=True` is already set. Look at the model
  size/precision, the tracked-object window, and whether the whole clip is held in the
  predictor session at once; consider chunking long clips through `propagate_in_video`.
Keep Tier 2 (fp16) only as a **time** optimization for the depth loop, and Tier 3
(WAFT factorization) as the main time lever, if/when time becomes the priority.

### Tier 1.5 — SAM3 state offload (implemented & validated 2026-07-20)
SAM3's per-frame inference-state memory bank lives on GPU by default and scales with
clip length × tracked objects; with `propagation_direction="both"` and no
`max_frame_num_to_track` cap the whole clip's state sits in VRAM. `start_session`
accepts `offload_state_to_cpu` (plumbed to `sam3_tracking_predictor.py:79-82`, storage
device → CPU) but the pipeline never passed it. Set `offload_state_to_cpu=True` on both
`start_session` calls (`segment_with_flows`, `segment_with_sam`). Lossless — only moves
where state is stored; masks/outputs unchanged. Docstring cost ~10–15% tracking fps.

MattSwift (winner config): SAM3-phase / whole-run peak **23,758 → 20,353 MB (−14%)**,
time flat (549→552 s), metrics byte-identical. **Cumulative Tier 1 + 1.5: 37,092 →
20,353 MB = −45%, lossless.** The −14% here is a floor: MattSwift tracks ~1 object so
the memory bank is small and the fixed image-encoder activations dominate; on
many-object / long clips (e.g. boutique1_HQ 70 GB, vid360_bruit_operatrice 77 GB) the
memory bank is a far larger share, so the offload should help substantially more —
**validate on a high-VRAM clip next.**

## Results — Time optimization (lossless, 2026-07-20)

After VRAM was solved (SAM3-bound), the peak stopped moving, so the remaining
plan tiers (Tier 2 fp16, Tier 3 WAFT factorization) are **time-only**. A fast
single-video harness was set up for iteration: **`accident_electrique_fast5`**
(75 frames @ 3840×1920, ~4× faster than MattSwift), isolated at
`/raid/mb273924/_DATASETS/uptale/_bench_fast5/`, output `./benchmark_fast5/`,
`SUMMARY_fast5.json`. (Bench caches completed configs — `rm -rf
./benchmark_fast5/<config>` before re-running.)

**Profiling the depth loop overturned the plan's assumptions.** DA360 depth is
only ~4–5 s (3%) and PLY saving 1.5 s — neither is a bottleneck. The real
per-frame hogs were CPU/GPU depth post-processing:
`propagate_depth_via_flow` 32 s, temporal-median smoother 16 s,
`align_depth_frame` 15 s.

Lossless fixes implemented (`flow_depth_propagation.py`, `video.py`):
- **Batched ERP warp** (`warp_backward_multi`): the depth warp and the two
  FB-consistency warps all use the same `flow_fwd` grid → one batched
  `grid_sample` (one CPU↔GPU transfer) instead of three. **32.3 → 9.6 s (−70%)**.
- **Cached** base meshgrid (`_base_meshgrid`) and `pole_trust_mask` — constant
  for fixed (H,W) but were rebuilt every warp/frame.
- **Partition median** in `TemporalDepthSmoother` instead of `np.median`'s full
  sort (exact same result). 15.8 → 13.4 s.
- **Closed-form normal equations** in `align_depth_frame(method="lstsq")`
  instead of `np.linalg.lstsq` SVD over ~7M pixels. Align block 44 → 38 s.

**Whole-run wall time 148.8 → 113.2 s (−24%), fully lossless** — fast5
`bg_depth_cv` (0.01854882342996768) and spikes/frame (0.027027…) byte-identical
across every step. No VRAM change (peak stays SAM3-bound at 9,234 MB on fast5).
Commits: `d7c833f` (batched warp + caches + partition median), `36fc85e`
(align closed-form).

**Tier 2 (fp16): dropped** — targets only the ~4 s depth model, zero VRAM
benefit now that the peak is SAM3-bound, and carries re-validation risk.
**Tier 3 (WAFT factorization): not pursued** — the pass-1 (mask) forward flow
is *not* seam-padded while the pass-2 (propagation) forward flow *is*; they
differ near the ERP seam, so collapsing them would change the motion mask →
SAM3 segmentation → output. It is a correctness change, not a lossless win.
Remaining lossless levers are minor (smoother `np.stack` copy ~13 s).

## Results — GPU-resident depth chain (2026-07-20)

**Can ops move to GPU without hurting VRAM?** Yes — and it was the biggest time
win. Depth is born on the GPU (DA360) but was moved to CPU immediately, so
align/smooth/propagate ran as numpy on 3840×1920 arrays. The whole chain now
stays on the GPU for the winner hot path (guarded: bglock + lstsq +
median/none smoother; any other config falls back to the numpy path):
`align_depth_frame_gpu`, `TemporalDepthSmoother.call_torch` (kthvalue median,
avoids `torch.quantile`'s 2^24 cap), `propagate_depth_via_flow_torch` (+
`warp_backward_multi_torch`, cached torch meshgrid/pole mask). Composite stays
CPU (numpy boundary for `to_gaussians`); flows transfer per-frame (cheap).

fast5 winner: depth post-processing block **38 → 11 s**, whole-run
**113 → 88.7 s**. **VRAM unchanged at 9,234 MB** — the peak is SAM3-bound and
the chain adds only a few depth-sized tensors (~a few hundred MB), well under
it. **Metrics byte-identical** (bg_depth_cv 0.01854882342996768, spikes
0.027027…): align leaves background pixels untouched and kthvalue selects the
same element as np.median, so the bg-only stability metrics don't move.
Commit `e5b1681`.

**Can it be multi-threaded/processed?** Only marginally: the loop is *causal*
(propagation consumes the previous frame's composited depth; the smoother is a
causal window), so frames can't be parallelized. Off-critical-path overlap
(prefetch DA360 depth ~4 s, async PLY writes ~1.5 s) is ~5–7 s — not pursued;
the GPU chain already removed the CPU hogs that threading would have hidden.

### Cumulative result (whole effort, winner config)
Wall time **148.8 → 88.7 s (−40%)**; peak VRAM **37,092 → 20,353 MB (−45%)**;
stability metrics lossless throughout. VRAM is SAM3-bound; further time levers
(align composite on GPU, Tier 3 seam-mask factorization) are minor and/or not
lossless.

### Composite-on-GPU + Tier-3 single-pass (2026-07-21, commit `d23cb01`)
Follow-up on the two "minor/not-lossless" levers noted above. Measured on
`accident_electrique_fast5`, winner config `bglock + median-w5`, da360, single GPU.
Harness: `run_capture.py` + `make_diff_figures.py`. Baseline (both off): wall **94.0 s**,
VRAM **9,234 MB**, bg_cv 0.0185488, spikes 0.027027, fg_cv 0.0129706, fg_delta 0.0288694.

**Composite-on-GPU — default ON** (`SPAG_GPU_COMPOSITE=0` to disable). Ports
`feather_dynamic_mask` + `composite_bg_locked` to torch (`composite_bg_locked_torch`):
square max-pool dilate + separable Gaussian conv (cv2 sigma formula) vs cv2 elliptical
dilate/GaussianBlur. Keeps the final bg-locked composite on the GPU in the winner path.

| metric | baseline | gpucomp | Δ |
|---|---|---|---|
| wall | 94.0 s | **88.4 s** | −6% |
| depth_alignement block | 11.45 s | **5.81 s** | −49% (the cv2 feather was the cost) |
| VRAM | 9,234 MB | 9,234 MB | flat |
| bg_cv / spikes / fg_delta | — | — | **identical** |
| fg_cv | 0.01297065 | 0.01297029 | +3.6e-7 |

Depth diff: mean |Δ| **0.0036 m**, confined to the feather band (10.7% of pixels, max 12 m
at one edge speck); SAM mask **0%** changed (composite is post-segmentation). → near-lossless,
+6% at zero VRAM → shipped default-on.

**Tier-3 single-pass — opt-in** (`SPAG_SINGLE_PASS=1`). Derives the SAM magnitude mask from
the unified bidirectional pass so WAFT runs **once** instead of twice. Seam-band mask shift
is governed by `SPAG_SP_SEAMPAD` (default **0**):

| metric | baseline | pad=64 | **pad=0 (default)** |
|---|---|---|---|
| wall | 94.0 s | 83.1 s | 85.5 s (−9%) |
| VRAM | 9,234 MB | 8,468 MB | 9,234 MB |
| bg_cv | 0.0185488 | 0.0179007 | **0.0185488 identical** |
| spikes / fg_delta | — | shifts | **identical** |
| fg_cv | 0.0129706 | 0.0190383 (+47%) | **0.0131906 (+1.7%)** |
| % mask chg | — | 0.32% (grows over time) | **0.00%** |
| % depth chg>1e-3 | — | 95.7% | **9.7%** |

Root cause of the pad=64 mask shift: with WAFT's `pad_to_train_size=False`/`tiling=False`
it runs on the raw input size, so seam-padding (W→W+2·seam_pad) changes the input width and
— via WAFT's global receptive field — perturbs the flow *everywhere*, not just the seam.
Fix is to drop the seam padding, not the mask pass: `seam_pad=0` makes the forward flow the
same `infer_pair` the two-pass baseline used → mask byte-identical, and the only residual is
fg_cv +1.7% from non-padded seam propagation of objects crossing the ERP seam (bg is locked
to ref regardless). Getting *both* an exact mask and exact seam-padded propagation genuinely
needs two differently-shaped WAFT inputs, so it stays opt-in behind the full re-benchmark gate.

## Tier 4 — Ideas not yet tried (2026-07-21)

Follow-up on the completed work above (VRAM now SAM3-bound at 20,353 MB; time now
dominated by SAM3 segmentation + WAFT flow, since the depth loop is mostly
GPU-resident and composite-on-GPU/Tier-3 single-pass already shipped). None of these
are implemented or measured yet — each needs the same re-benchmark/diff-figure
discipline used for the earlier tiers before being trusted.

**VRAM (SAM3 is the current ceiling)**
1. **fp16/bf16 SAM3 image-encoder weights** — same idea as the dropped DA360
   mixed-precision tier, but aimed at the actual current bottleneck. Halves encoder
   activation memory. Gate behind mask-quality / stability-metric re-benchmark, same
   as Tier 2.
2. **Cap `max_frame_num_to_track` / chunk `propagate_in_video`** — the whole clip's
   tracking state currently lives in one session (offloaded to CPU, but unbounded by
   clip length). Windowed/re-seeded propagation would bound peak state size instead
   of letting it grow with clip length — expected to matter most on the long/
   high-VRAM clips already flagged as needing follow-up (`boutique1_HQ` 70 GB,
   `vid360_bruit_operatrice` 77 GB).
3. **Downscale the SAM3 segmentation pass** — segment on a lower-res proxy, then
   upsample the mask before feathering/compositing. Mask boundaries are already
   feathered downstream, so some softening is plausibly tolerable; validate with the
   same mask-diff methodology (`make_diff_figures.py`) used for Tier-3.

**Time**
4. **Async/prefetch DA360 depth for frame i+1** while post-processing frame i's
   align/smoother/propagate/composite — noted earlier as "~5–7 s, not pursued," but
   now that post-processing is GPU-resident and fast (~11 s vs the old 38 s), the
   relative overlap opportunity is proportionally larger.
5. **Async PLY writes** — same rationale as #4, also previously shelved for the same
   reason (small in absolute terms, now a bigger relative share of per-frame time).
6. **CUDA graph capture for the per-frame depth chain** — align/smoother/propagate/
   composite is now a fixed, identical sequence of small GPU ops every frame;
   capturing it as a CUDA graph would cut kernel-launch overhead, which matters more
   now that each op is individually fast.
7. **Push WAFT `scale` down further for the mask-only pass** — SAM3 prompting only
   needs a magnitude mask, not sub-pixel flow; a lower-res single-pass estimate for
   the mask (keeping full-res flow for propagation) could shave more off the WAFT
   phase beyond the Tier-3 single-pass win already shipped.

## Verification (no-regression)
- **Per-tier VRAM**: re-run `benchmark_solutions.py` on the `bglock_sol1_median_w5` config
  over a 3–4 video subset after each tier; compare `mean_vram_max_mb` against the current
  45591 MB in `benchmark_solutions/SUMMARY.json`. Tier 1 alone should recover most of the
  sol1 +6.7 GB and the WAFT residency; report the new split from the Tier-0 probes.
- **Losslessness (Tier 1)**: on one video, confirm output PLYs / `depth_metrics` are
  unchanged (bit-identical or within float32 rounding) vs the current winner — Tier 1 is
  meant to change memory, not results.
- **fp16 gate (Tier 2)**: full 14-video benchmark; the four stability metrics
  (`bg_depth_cv`, `fg_depth_cv`, `bg_spikes_per_frame`, `fg_delta_mean`) must stay within
  ~5% of current winner values, else keep fp16 off by default.
- **Time (Tier 3)**: compare `mean_time_seconds` vs current 737.5 s; factorization should
  move it toward the `sol1_median_w5`-alone figure (506 s).
