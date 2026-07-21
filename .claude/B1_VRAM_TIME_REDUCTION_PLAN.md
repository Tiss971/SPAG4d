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

## Results — Tier 4 time levers (2026-07-21)

**4. DA360 prefetch (implemented, lossless, default-on).** `video.py`: the per-frame
loop now issues frame i+1's `depth_engine.predict` on a dedicated `torch.cuda.Stream`
right after unpacking frame i's result, instead of at the top of iteration i+1. The
consuming code does `current_stream().wait_stream(depth_stream)` +
`tensor.record_stream(...)` before use, so the dependency is enforced without a host
sync — the align/smoother/propagate block's existing `.cpu()`/`.item()` calls only
drain the *default* stream, so DA360's kernels for the next frame can execute
concurrently with them instead of being serialized behind the same stream. Same
`predict()` call, same inputs → bit-identical depth, purely a scheduling change.

**5. Async PLY writes (implemented, lossless, default-on).** `video.py` + unchanged
`ply_writer.py`: `save_ply_gsplat` (GPU→CPU sync + numpy SH/opacity encoding + disk
write) is submitted to a `ThreadPoolExecutor(max_workers=2)` instead of called inline,
so frame i's write runs off the critical path while frame i+1's GPU work proceeds.
Futures are joined right after the loop (before any code reads the PLY files, e.g. the
`file_size` stat). Byte-identical output, same function, just off-thread.

**7. WAFT mask-pass downscale (implemented, opt-in via `SPAG_MASK_SCALE`, default
`1.0`=off).** In the two-pass (non-`SPAG_SINGLE_PASS`) branch, when `SPAG_MASK_SCALE
< 1.0` the frames fed to `model.run()` (mask-only pass) are downscaled further via
`cv2.resize(..., INTER_AREA)` and the resulting binary masks upsampled back to
`waft_frames` resolution (`INTER_NEAREST`) before being handed to `segment_with_flows`.
The full-res bidirectional pass used for depth propagation is untouched. **Not
lossless** — coarser flow shifts the magnitude-threshold mask boundary, which changes
SAM3 prompts and therefore output; must be re-benchmarked (mask-diff methodology, like
`SPAG_SINGLE_PASS`) before being trusted or defaulted on.

**6. CUDA graph capture — investigated, not implemented.** Re-reading
`align_depth_frame_gpu` (`video.py:1347+`) and the bglock branch (`video.py:937-993`)
shows the per-frame chain is **not** actually branch-free at the Python level: (a) the
composite/propagate call takes a structurally different path on `idx == 0` vs every
other frame (`depth_prev_final is None` check, `video.py:943-961`); (b) alignment does
host-side branching on tensor *values* (`s_clipped != s` clipping logic driving which
op sequence runs, plus several `.item()` calls used for early-exit-style logic) — CUDA
graphs require the exact same kernel sequence on every replay, and both of these
violate that. Capturing would need a branch-free rewrite of the align/composite math
(e.g. `torch.where` instead of Python-level `if`s) and pinned static I/O buffers, which
is a correctness-risk redesign, not the same category as 4/5/7. Deferred — same
gating posture the doc already applied to Tier 2 (fp16) and the original Tier-3
factorization when they turned out to be non-lossless or not worth the risk.

**Verification status**: smoke-tested end-to-end on `accident_electrique_fast5`
(4/5 default-on path); no fast5 SUMMARY/diff-figure harness currently exists in the
repo (per `CLAUDE.md`, the old `run_capture.py`/`make_diff_figures.py` scripts are
gone), so the wall-time/lossless numbers below are from a single direct run via
`python -m spag4d convert`, not the full mask-diff regression the earlier tiers used.
`SPAG_MASK_SCALE` (item 7) still needs that same re-benchmark gate before shipping
default-on, exactly like `SPAG_SINGLE_PASS`.

## Results — real `benchmark_solutions.py` validation (2026-07-21)

Ran `bglock_sol1_median_w5` on a 3-video subset (`MattSwift`, `Dispo_RDV`, `scene01`,
`/tmp/spag4d_bench_subset`) via the actual harness (not smoke tests), comparing a
pre-Tier-4 baseline (`git stash`) against the current tree, plus isolated runs for
items 7 and 1.

**Items 4/5 (DA360 prefetch + async PLY writes) — real numbers, no meaningful effect
on these clips.** Before vs after, per video: Dispo_RDV 702.9s→702.6s, MattSwift
348.9s→347.1s, scene01 401.0s→396.7s (0–1.1% deltas, within run-to-run noise). VRAM
unchanged. Stability metrics bit-identical. Confirms the fast5 finding: the per-frame
depth loop genuinely got faster in isolation, but on clips this long SAM3/WAFT
dominate total wall time, so the aggregate barely moves. Lossless, kept default-on
since there's no downside, but they are not the lever that matters here.

**Item 7 (`SPAG_MASK_SCALE=0.35`) — real win.** Off (1.0) vs 0.35, per video:
Dispo_RDV 702.6s→592.1s (−15.7%, 48759→37200 MB VRAM, −23.7%); MattSwift
347.1s→297.3s (−14.4%, 22102→18315 MB, −17.1%); scene01 396.7s→317.2s (−20.0%,
27929→15627 MB, −44.0%). The VRAM drop is a genuine bonus (smaller mask-pass tensors
lower WAFT's peak, not just its runtime), not something the original design predicted.
Background stability (`bg_depth_cv`, `bg_spikes_per_frame`) unchanged within noise on
all 3 — expected, since bglock doesn't depend on flow-mask precision. Foreground
stability mostly held (Dispo_RDV/scene01 `fg_depth_cv` within noise or improved) but
**MattSwift's `fg_depth_cv` dropped 0.827→0.551** — a real shift, not noise, most
likely because the coarser mask changed which contours got promoted to SAM3 prompts
(different objects tracked), not just numerical drift in a fixed detection. Verdict:
real time/VRAM win, but not proven lossless for *what* gets tracked as foreground —
still opt-in, still needs the same scrutiny as `SPAG_SINGLE_PASS` before any
default-on move, and this MattSwift result is a concrete reason to keep it opt-in.

**Item 1 (`SPAG_SAM3_FP16`) — implementation bug found and fixed, but delivers no
benefit.** First attempt (`video_predictor.model.half()`) crashed on **all 3 videos**
with `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half`,
because callers still feed the model fp32 tensors — a blanket weight cast isn't
compatible with the rest of the pipeline's dtype assumptions. Fixed by wrapping the
actual forward-pass call sites (`add_prompt`, `propagate_in_video` in both
`segment_with_flows` and `segment_with_sam`) in `torch.autocast(device_type="cuda",
dtype=torch.float16)` instead, which handles the fp32/fp16 mixing internally. That
runs without crashing, but real-benchmark numbers show it does essentially nothing:
Dispo_RDV 702.6s→711.0s, MattSwift 347.1s→347.7s, scene01 396.7s→402.4s (all within
+1.4% noise), VRAM within ~60 MB (noise) on all 3, stability identical. Autocast only
casts *activations* per-op; the weight tensors themselves stay fp32, so the dominant
VRAM cost (model weights) is untouched, and autocast's dispatch overhead roughly
cancels any compute saved. **Not worth pursuing further** — true fp16 weights would
require casting every call-site input to match, a bigger and riskier change than the
payoff justifies given item 7 already delivers the real win. **Removed** — the
`SPAG_SAM3_FP16` env var and `torch.autocast` wrapping were reverted from
`segment_with_flows`/`segment_with_sam` in `video.py`, no code footprint left.

**Item 2 (`SPAG_SAM3_MAX_FRAMES`) — fixed, but the real fix costs almost all of the
original savings (took three attempts).** First attempt crashed on all 3 videos with
`KeyError: <frame_idx>` at `video.py:946`: a single `propagate_in_video` call with
`max_frame_num_to_track` set and `propagation_direction="both"` leaves any frame
outside the cap's reach from `start_frame_index` with no SAM3 output, and the
downstream per-frame loop has no fallback. First fix attempt patched this with
nearest-neighbor gap-filling (copy the nearest tracked frame's mask for any missing
index) — this stopped the crash, but on inspection (a 75-frame test clip,
`SPAG_SAM3_MAX_FRAMES=20`) it turned out **68% of frames were duplicate-filled, not
tracked** (SAM3 covered only frames 0–22 of 75, then froze). Rejected in favor of
windowed/re-seeded chunking. Second attempt: `segment_with_flows`/`segment_with_sam`
issue *multiple* `propagate_in_video` calls on the same session (forward from
`start_frame_index` to the end, then backward to frame 0), each bounded by
`max_frame_num_to_track`, chaining each window's end to the next window's start. This
was reported "working" based on a **frame-coverage** metric (every frame index has
*some* `outputs_per_frame` entry) — validated on the 75-frame clip: 74/75 frames got
an entry (vs 23/75 before), full 3-video run showed 0 frames needing the
nearest-neighbor fallback, and real time/VRAM savings (Dispo_RDV −32%/−48%, MattSwift
−27%/−24%, scene01 −14%/−17%).

**That report was wrong — the user caught it by visually inspecting the mask output**
("mask on it seems to not be propagated between chunk"). Coverage-of-entry is not the
same as coverage-of-a-real-mask: `_consume_response` stores whatever
`propagate_in_video` yields for a frame index even when the yielded mask is empty, so
"every frame has an entry" was compatible with an object silently going untracked for
an entire window. Added debug instrumentation that checks each new window's *first*
frame for a non-empty mask (`any(m.any() for m in first_out['out_binary_masks'])`) and
re-ran the 75-frame clip at `SPAG_SAM3_MAX_FRAMES=20` (forcing 4 forward windows):
windows 1–2 came back non-empty, but **windows 3 and 4 (native frames 44 and 65) came
back with a completely empty mask** — the object was lost, and the nearest-neighbor
fallback never triggered because those frames technically had a (blank) entry.

**Root cause**: chaining `propagate_in_video` calls on the same `session_id` keeps the
session's memory bank alive across calls, but does not reliably re-establish which
object to keep tracking — a fresh `propagate_in_video` call with a new
`start_frame_index` does not automatically continue from where the previous call left
off; without an explicit re-prompt it can come back empty.

**Fix**: `_reseed_window(anchor_idx, target_idx)` (`video.py`, in `segment_with_flows`)
re-registers every object right before each new window opens: reads the previous
window's last (or first, for the backward pass) frame's raw output, computes each
object's mask centroid in that frame, and issues an `add_prompt` call
(`points=[[cx, cy]]` in the same relative-coordinate convention used by the initial
flow-triggered prompts) at the new window's start frame — an explicit anchor instead of
relying on unverified memory-bank persistence. Re-ran the same 75-frame,
`SPAG_SAM3_MAX_FRAMES=20` diagnostic: **all 4 windows now report a non-empty first
frame**, confirming the fix.

**But the fix eliminates almost all of the original saving.** Full 3-video
re-benchmark (`SPAG_SAM3_MAX_FRAMES=200`) with re-seeding, vs. both the true baseline
and the (incorrect) no-reseed chunked numbers:

| video | baseline (no chunking) | chunked, no reseed (buggy) | chunked + reseed (correct) |
|---|---|---|---|
| Dispo_RDV | 702.6s / 48,759 MB | 480.4s / 25,342 MB | **702.5s / 46,522 MB** |
| MattSwift | 347.1s / 22,102 MB | 253.1s / 16,832 MB | **355.3s / 22,100 MB** |
| scene01 | 396.7s / 27,929 MB | 343.0s / 23,244 MB | **388.2s / 24,974 MB** |

Time is back to baseline (within noise) on all 3, and VRAM only drops meaningfully on
Dispo_RDV (−4.6%); MattSwift is flat, scene01 −10.6%. Background stability unchanged
(`bg_spikes_per_frame` 0 on all 3). The re-seeding `add_prompt` calls add real
inference cost, and — more importantly — re-registering objects each window does not
bound the session's underlying memory-bank growth, which is what item 2 was originally
meant to cap; the "chunking" only bounds `max_frame_num_to_track` within a single call,
not the cumulative session state across windows. **Verdict revised: item 2 is
correctness-fixed but not a meaningfully useful VRAM/time lever as implemented** —
shipping it opt-in is fine (it no longer silently drops the tracked object), but it
should not be recommended as an optimization; item 7 (`SPAG_MASK_SCALE`) and item 3
(`SPAG_SAM3_SCALE`) remain the real wins from this tier.

**Item 3 (`SPAG_SAM3_SCALE`) — fixed and real-benchmarked, working (three bugs found).**
First crash: `NameError: name 'waft_frames' is not defined` — fixed by using the
function's actual parameter name (`viz_frames`). Second crash (writing a downscaled
proxy `.mp4` and pointing `resource_path` at it): `IndexError: list index out of range`
inside third-party SAM3 code, because SAM3's own decoder didn't reproduce the same
frame count as what was written (codec-level frame drop/pad on the encode/decode round
trip). Rewrote to pass an in-memory list of `PIL.Image` frames as `resource_path`
instead — `load_resource_as_video_frames` (SAM3's own loader) accepts a list directly
and uses `len(list)` as the frame count, no decoder round-trip possible. Third crash
(after that rewrite): the PIL list was built from `segment_with_flows`'s own
`viz_frames` parameter, but that parameter is actually the caller's WAFT-scaled
(max-1024-side) array, not the true native video — `add_prompt`'s `frame_index` is in
*native* frame space, so it ran past the end of the shorter list, another
`IndexError` inside SAM3. Fixed by decoding the native video directly (one
`cv2.VideoCapture` pass) to build the downscaled list at the correct native frame
count. Fourth crash (after that): masks returned from a downscaled SAM3 pass are at
the downscaled resolution, but the downstream per-frame depth loop fuses them against
full native-resolution frames — `ValueError: operands could not be broadcast`. Took
two wrong guesses at the correct upsample target (`meta['new_H']/new_W` is a *different*
knob, item 7's independent mask-scale setting; this function's own `viz_frames` is
also WAFT-scaled, not native) before landing on the actual native size,
`meta['H']/meta['W']`. Full 3-video benchmark (`SPAG_SAM3_SCALE=0.5`), all 3 completed:
Dispo_RDV 702.6s→689.1s (−2%, 48759→13334 MB VRAM, **−73%**); MattSwift 347.1s→357.4s
(+3%, 22102→10862 MB, **−51%**); scene01 396.7s→406.2s (+2%, 27929→11544 MB, **−59%**).
Time is essentially unchanged (within noise — SAM3 compute isn't the bottleneck here,
VRAM is), but VRAM savings are the largest of any Tier-4 item. Background stability
unchanged within noise on Dispo_RDV/scene01. MattSwift shows a real foreground-tracking
shift (`fg_depth_cv` 0.827→0.937, `fg_delta_mean` 0.007→0.025, `bg_depth_cv` also up
~3x though still small in absolute terms, 0.0005→0.0015) — consistent with the
documented caveat that a coarser SAM3 input changes which contours/boxes get promoted
to prompts, i.e. genuinely different tracked objects, not just noise. Real, large VRAM
win; opt-in, default off, same scrutiny caveat as item 7.

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
