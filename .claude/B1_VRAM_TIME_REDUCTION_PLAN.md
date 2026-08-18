# VRAM & time reduction — winner config `bglock_sol1_median_w5`

_Started 2026-07-20, last consolidated 2026-07-24. Target: the production winner
(`depth_correction=bglock` + temporal median depth smoother, window=5, `da360`). This
doc was a running changelog; it has been compacted to the current truth. Per-tier raw
tables from intermediate steps are dropped — only the standing conclusions remain._

## Bottom line (current state)

- **VRAM is SAM3-bound.** After the shipped lossless work the whole-run peak lives
  entirely in **SAM3 segmentation** (image-encoder activations + memory bank); the WAFT
  flow phase and the per-frame depth loop are each only a few GB and are *not* the peak.
- **Cumulative shipped result (winner config), all lossless:**
  - Peak VRAM **37,092 → 20,353 MB (−45%)** (fast5: unchanged at ~9,234 MB, already SAM3-bound).
  - Wall time (fast5) **148.8 → 88.7 s (−40%)**.
  - Stability metrics (`bg_depth_cv`, `fg_depth_cv`, `bg_spikes_per_frame`, `fg_delta_mean`)
    unchanged to float32 rounding throughout.
- **The remaining VRAM lever is SAM3, not depth or WAFT.** fp16 depth (old Tier 2) was
  dropped — the depth loop is ~2.6 GB, so halving it can't move a 20 GB SAM3 peak.

## Shipped & default-on (lossless unless noted)

All in `spag4d/video.py` / `spag4d/flow_depth_propagation.py`. Each was VRAM/diff-checked
against the winner config before shipping.

**VRAM (Tier 1 + 1.5, commits around 2026-07-20):**
- **float32 cast in `TemporalDepthSmoother`** — the median/tensordot branch was promoting
  depth float32→float64, doubling the gaussian tensors in `to_gaussians`. Casting back was
  the bulk of the old depth-loop peak.
- **Free WAFT before SAM3/depth loop** — `del model; gc.collect(); torch.cuda.empty_cache()`
  after the flow phase (WAFT was resident but unused through SAM3 + the whole depth loop).
- **`empty_cache()` at phase boundaries** (after WAFT free, after SAM3 teardown).
- **SAM3 state offload to CPU** — `offload_state_to_cpu=True` on both `start_session` calls
  (`segment_with_flows`, `segment_with_sam`); moves the per-frame memory bank off GPU.
  Costs ~10–15% tracking fps, lossless. −14% on a 1-object clip (encoder-activation floor);
  larger share on many-object/long clips.

**Time (fast5 148.8 → 88.7 s):**
- **Batched ERP warp** (`warp_backward_multi`): depth warp + 2 FB-consistency warps share
  one `flow_fwd` grid → one batched `grid_sample`/transfer instead of three (32.3→9.6 s).
- **Cached** `_base_meshgrid` + `pole_trust_mask` (constant per (H,W), were rebuilt every frame).
- **Partition median** in the smoother instead of full `np.median` sort (exact same result).
- **Closed-form normal equations** in `align_depth_frame(method="lstsq")` (vs SVD lstsq).
- **GPU-resident depth chain** (biggest time win): `align_depth_frame_gpu`,
  `TemporalDepthSmoother.call_torch` (kthvalue median — avoids `torch.quantile`'s 2^24 cap),
  `propagate_depth_via_flow_torch` + `warp_backward_multi_torch`. Guarded to the winner hot
  path (bglock + lstsq + median/none); any other config falls back to numpy. VRAM unchanged
  (adds only a few depth-sized tensors, well under the SAM3 peak).
- **Composite-on-GPU** (`SPAG_GPU_COMPOSITE`, default ON; `=0` to disable):
  `composite_bg_locked_torch` ports `feather_dynamic_mask` + `composite_bg_locked` to torch.
  The cv2 feather was the cost (depth-align block 11.4→5.8 s). Near-lossless — mean |Δ|
  0.0036 m confined to the feather band, SAM mask 0% changed (composite is post-segmentation).
- **DA360 prefetch** (frame i+1's `predict` on a dedicated CUDA stream) and **async PLY
  writes** (`ThreadPoolExecutor`). Both bit-identical, off critical path. Real-benchmark:
  negligible on long clips (SAM3/WAFT dominate wall time) but no downside, kept on.

## `SPAG_SINGLE_PASS` — the WAFT time lever (opt-in, default OFF)

Derives the SAM magnitude mask from the unified **bidirectional** WAFT pass so WAFT runs
each pair **once** (vs baseline's forward mask pass + bidirectional propagation pass).
Seam handling via `SPAG_SP_SEAMPAD` (default 0). **Full investigation and the
decoupled-fix experiment: [C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md](C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md).**

- **Time win real, large, VRAM-neutral:** −38% flow-phase wall (962-pair clip 564.9→351.0 s;
  −31% to −34% on the 3-clip production gate). Short clips show no win (cuDNN autotune warmup
  dominates). Not a VRAM lever (SAM3-bound).
- **Regression real on foreground depth, NOT SAM3 tracking.** Mask + add_prompt schedule
  byte-identical to baseline (proven pixel-diff + identical SAM3 registration logs); every
  `depth_metrics` array except `stabilized_fg_median` matches exactly. Root cause: the
  single-pass flow array is shared between mask derivation (needs `seam_pad=0`) and
  `propagate_depth_via_flow` (needs `seam_pad=64` for ERP wraparound), so foreground depth
  loses seam correction — dropping the foreground entirely on ~34/158 scene01 frames. Background
  untouched (locked to `depth_ref`), so `bg_depth_cv`/spikes/`fg_delta_mean` identical.
- **The full seam-pad fix is a proven dead end for the time win.** `SPAG_SP_FIX_SEAMGAP=1`
  (recompute a `seam_pad=64` flow for propagation only) restores `stabilized_fg_median`
  byte-identical to baseline — but is +15% *slower* than baseline (4 vs 3 `infer_pair`/pair);
  even best-case it only equals baseline, because the two consumers genuinely need different
  `seam_pad` on the same frames. Kept in-tree as proof-of-root-cause / scaffolding only.
- **Only remaining avenue with any win: conditional** seam padding (re-pad only when the
  tracked object's bbox is near the seam) — gated on a corpus question (how common is
  seam-adjacent foreground motion), not more scene01 work. See C1.

**Verdict: default OFF. Safe per-clip only when there is no seam-crossing foreground motion,
or for preview passes. Do not flip the default.**

## SAM3 VRAM levers (the peak) — status

- **Item 8 `SPAG_SAM3_BF16` (new, opt-in, default off) — casts SAM3's own weights to
  bf16** (norm layers — `LayerNorm`/`GroupNorm`/`BatchNorm*` — kept fp32-parameter, wrapped
  to compute in fp32 to avoid dtype-mismatch kernel errors). Unlike item 1's `autocast`
  (activations only, weights stay fp32), this hits the actual dominant cost. Implemented in
  the vendored `third_party/sam3/sam3/model/sam3_video_predictor.py` (only insertion point —
  model construction happens entirely inside the package). Deterministic A/B on fast5:
  **VRAM 7,703 → 6,352 MB (−17.5%)**, wall time **131.9 → 112.8 s** (bf16 tensor-core
  throughput, free win), all stability metrics within noise (`bg_depth_cv` +0.1%,
  `fg_depth_cv` +0.1%, `fg_delta_mean` +0.4%). **Stacks with item 3** (`SPAG_SAM3_SCALE=0.5`):
  combined **4,872 MB (−36.8% vs baseline)** on fast5 alone. **Full 10-clip production
  re-benchmark (2026-07-31, exact `benchmarks/baseline_2026-07-27/BASELINE_2026-07-27.md` config + env, stacked
  bf16+scale0.5, 3-way GPU-parallel):** VRAM **31,921 → 9,963 MB mean (−68.8%)** — a much
  bigger win than fast5 suggested, because larger clips are more purely SAM3-activation-
  bound (less per-run overhead diluting the encoder-activation savings). Per-clip range
  −50% to −75%. Time flat overall (390→393s mean, bf16 throughput roughly cancels
  scale0.5's altered contour-promotion / tracking-iteration cost). `bg_depth_cv` matches
  baseline to the 3rd–4th decimal on **every** clip — background path (bglock) is
  completely untouched by this lever. **Not yet safe to default on — two open flags:**
  (1) **MattSwift `fg_depth_cv` collapsed 0.28→0.0094 (30×)**, the exact clip D1 P3 already
  flagged for real fg-tracking drift under `SPAG_SAM3_SCALE=0.5` alone; frame-to-frame
  `fg_delta_max`/`fg_delta_mean` stayed close to baseline (0.930 vs 0.925, 0.040 vs 0.044)
  so the object still moves similarly frame-to-frame, but the overall depth distribution
  tightened drastically — could be a genuine steadier lock or a mask/coordinate artifact;
  needs a visual check of the SAM3 mask preview before trusting the number. (2)
  **boutique1_HQ time regressed +54.8%** (417→646s), the only clip to regress meaningfully
  against an otherwise flat-to-improved time picture — likely GPU contention from running
  3 clips in parallel across GPUs during this benchmark, not a real effect, but unconfirmed
  pending a solo re-run. Dispo_RDV/scene01 `fg_cv` also rose ~50%, smaller and more
  consistent with normal noise.

  **MattSwift root-caused (2026-07-31): real mask regression, not metric noise.** Solo
  same-GPU re-run with `depth_npy_dir` set (dumps the raw per-frame fused SAM mask) and
  direct pixel diff against a same-session baseline: **mean SAM mask IoU 0.34** across all
  150 frames, new mask area a strikingly uniform **51.4% of baseline** (std 0.035 — not
  scattered noise, a systematic effect). Per-frame bbox inspection shows why: the top
  y-boundary matches baseline almost exactly every frame (~344 vs ~348) while the bottom
  is consistently ~80px short (~600 vs ~682) — **SAM3 is losing the tracked person's legs**
  under the downscaled input, every frame, not intermittently. Visual overlay (frame 50,
  red=baseline-only/yellow=overlap/green=new-only) also shows a spurious extra green blob
  with no baseline counterpart — a genuine extra mis-detection, not just a smaller box. This
  explains the `fg_depth_cv` collapse mechanistically: a mask cropped to the tighter,
  more-uniform upper body produces a spatially tighter (falsely "steadier") depth
  distribution — the metric improved because the segmentation got worse, not better. Root
  cause is `SPAG_SAM3_SCALE=0.5`, not `SPAG_SAM3_BF16` (bf16 alone showed no such pattern on
  fast5, and the mechanism — thin limbs falling below the downscaled 512×256 SAM3 input's
  effective resolution — is exactly the item-3 caveat already on record, now confirmed with
  pixel evidence instead of just an aggregate metric). **Verdict: `SPAG_SAM3_SCALE=0.5`
  stays opt-in / per-clip-gated, do not default on for clips with thin/articulated
  foreground subjects. `SPAG_SAM3_BF16` alone remains the cleaner, lower-risk lever of the
  two** — this investigation isolated scale as the culprit but didn't clear bf16 to the same
  evidence bar (mask IoU) at full production scale, only on the fast5 spot-check.

  boutique1_HQ's time regression is still unexplained — not yet re-run solo to rule out
  GPU contention from the 3-way parallel benchmark.

  **`SPAG_SAM3_BF16` cleared at full production scale (2026-07-31).** Solo bf16-only
  (no scale) re-run on MattSwift, mask-diffed against the same raw baseline masks used
  above: **mean SAM mask IoU 0.991** (min 0.954 across all 150 frames, none below 0.5),
  area ratio 1.003±0.010 — masks are essentially identical to baseline. `fg_depth_cv`
  0.123 vs the matched-baseline 0.134, within run-to-run noise (see determinism caveat
  below). Confirms the MattSwift regression is 100% attributable to `SPAG_SAM3_SCALE`;
  `SPAG_SAM3_BF16` alone is safe to default on.

  **Baseline non-determinism note:** re-running the fp32 baseline solo under the documented
  `SPAG_DETERMINISTIC=1 SPAG_OCCL_FBGATE=1` gave `fg_depth_cv=0.134` on one GPU vs `0.28`
  recorded in `benchmarks/baseline_2026-07-27/baseline_2026-07-27.json` on another — `SPAG_DETERMINISTIC=1` does not
  guarantee bit-exact results across different GPU hardware. Use matched-environment
  (same-GPU, same-session) comparisons, not cross-run JSON diffs, when chasing small
  metric deltas.

  **VRAM vs (resolution, n_objects, n_frames) — rough predictive model (2026-07-31).**
  Correlated the 10-clip baseline `vram_max_mb` against native resolution and the SAM3
  object-track count logged per clip (`--- Propagating N moving object ---`):

  | clip | res (Mpx) | frames | n_objects | vram_max_mb |
  |---|---|---|---|---|
  | CIELE | 7.37 | 816 | 4 | 38,656 |
  | Dispo_RDV | 3.28 | 840 | 9 | 43,016 |
  | MattSwift | 2.10 | 600 | 5 | 18,080 |
  | accident_electrique_02 | 7.37 | 375 | 2 | 15,438 |
  | atelier_1 | 3.28 | 630 | 4 | 21,201 |
  | boutique1_HQ | 3.28 | 597 | 16 | 44,283 |
  | circulation_site_1_edit_coupe | 7.37 | 156 | 4 | 12,823 |
  | scene01 | 3.28 | 630 | 8 | 29,670 |
  | scene_03 | 3.28 | 617 | 12 | 37,603 |
  | trop_long_embouteillage | 1.84 | 1349 | 14 | 58,443 |

  Linear fit `vram_max_mb ≈ 9629 + 427·(Mpx) + 2626·n_objects` gives R²=0.66 (adding a
  px×n_objects interaction term only gets to R²=0.70). Resolution and object-track count
  are the two real levers — each additional tracked object costs roughly **2.6 GB**. The
  fit's worst miss is `circulation_site_1_edit_coupe`: same 7.37 Mpx class as CIELE and
  accident_electrique_02 but only 156 frames (shortest clip by far) and VRAM far below
  what resolution alone predicts — suggesting SAM3's memory-bank footprint grows with
  frames processed up to a plateau that longer clips reach and short clips don't, rather
  than being purely resolution/object-count driven. **Usable as a rough triage heuristic
  (±20-40% per clip), not a precise predictor** — a real dry run is still needed to know
  if a specific clip will blow VRAM.

  **`SPAG_SAM3_SCALE=0.5` mask regression is clip-dependent, not universal (2026-07-31).**
  Same solo baseline-vs-scale0.5 mask-IoU methodology run on 3 more clips (matched
  same-GPU, same-session pairs, not cross-run JSON):

  | clip | mean IoU | min IoU | frames IoU<0.5 | area ratio (new/base) |
  |---|---|---|---|---|
  | MattSwift (recap) | 0.34 | 0.28 | 150/150 | 0.51 ± 0.04 (uniform leg-crop) |
  | atelier_1 | 0.38 | 0.06 | 107/158 | 0.42 ± 0.26 (erratic, worse than MattSwift) |
  | scene01 | 0.78 | 0.55 | 0/158 | 1.01 ± 0.05 (mild boundary jitter only) |
  | Dispo_RDV | 0.97 | 0.96 | 0/210 | 1.00 ± 0.01 (essentially clean) |

  No correlation with object-track count (atelier_1 has 4 objects and is badly hit;
  Dispo_RDV has 9 and is clean) or resolution — both bad clips are at 2560×1280, same as
  the two clean ones. The dividing line looks qualitative: MattSwift and atelier_1 both
  have thin/small/articulated foreground structure (a person's limbs; workshop tools/arms)
  that falls below the downscaled 512×256 SAM3 input's effective resolution, matching the
  original item-3 caveat. scene01 and Dispo_RDV's subjects are comparatively larger/blockier
  in-frame. **Verdict unchanged and now better evidenced: `SPAG_SAM3_SCALE=0.5` must stay
  opt-in / per-clip-gated — there is no safe blanket default, and the failure mode is not
  predictable from resolution or object count alone, only from inspecting the clip's
  foreground subject shape.**

  **`SPAG_SAM3_SCALE` root cause found, and `SPAG_SAM3_MAXSIZE` added as the fix
  (2026-07-31).** `SPAG_SAM3_SCALE` multiplies `segment_with_flows`'s `viz_frames`
  parameter shape — but that parameter is actually `waft_frames`, already capped to
  ~1024px on the long side by WAFT, *not* native resolution (same confusable-naming
  trap flagged in the mask-upsample code comment at `video.py:2301-2311`). Consequence,
  confirmed from the run logs: every one of MattSwift/atelier_1/scene01/Dispo_RDV logged
  `downscaled to 512x256` under `SPAG_SAM3_SCALE=0.5` regardless of native resolution
  (2048px-2560px) — the real native→SAM3 factor was **0.2-0.25×, not 0.5×**. That's why
  it hurt MattSwift/atelier_1: their subjects' limbs collapse below detectability at
  512×256, while scene01/Dispo_RDV's larger subjects survive the same crush.

  Added `SPAG_SAM3_MAXSIZE` (`video.py:1678-1707`) as an absolute-pixel-cap alternative:
  caps the longer edge of the **true native** resolution (`meta['H']`/`meta['W']`, not
  the WAFT-shrunk `viz_frames`) instead of applying a relative factor to an
  already-shrunk shape. Re-tested MattSwift and atelier_1 at `SPAG_SAM3_MAXSIZE=1536`
  (matched same-GPU mask-IoU methodology):

  | clip | config | target res | mean IoU | min IoU | frames IoU<0.5 | vram_max_mb |
  |---|---|---|---|---|---|---|
  | MattSwift | scale=0.5 | 512×256 | 0.34 | 0.28 | 150/150 | 9,019 |
  | MattSwift | maxsize=1536 | 768×384 | **0.93** | 0.89 | 0/150 | 10,142 |
  | atelier_1 | scale=0.5 | 512×256 | 0.38 | 0.06 | 107/158 | 8,877 |
  | atelier_1 | maxsize=1536 | 614×306 | **0.87** | 0.77 | 0/158 | 10,247 |

  `SPAG_SAM3_MAXSIZE=1536` recovers most of the mask quality (IoU 0.87-0.93 vs
  0.34-0.38) while giving up only ~1.1-1.3 GB of the VRAM saving vs `SPAG_SAM3_SCALE=0.5`
  on these two clips (both still ~44-51% below fp32-native baseline). Not yet run across
  the other 8 clips or validated against scene01/Dispo_RDV (which weren't broken by
  scale=0.5 to begin with, so the interesting check there is whether maxsize=1536 costs
  them meaningfully less VRAM savings than scale=0.5 did — expected yes, since
  maxsize=1536 is a smaller relative shrink than 0.5-off-WAFT-shape was for these
  2560px-class clips too). **Leading candidate to replace `SPAG_SAM3_SCALE` as the
  default-off VRAM lever, pending full 10-clip validation** — needs a full-benchmark
  re-run (with bf16 stacked) before considering any default-on change.

  **Full 10-clip stacked (bf16+maxsize1536) validation — regression reappears when
  stacked (2026-07-31).** Ran matched same-GPU baseline+stacked pairs for all 10 clips
  (`SPAG_SAM3_BF16=1 SPAG_SAM3_MAXSIZE=1536`, results in
  `benchmarks/b1_vram_2026-07-31/bf16_maxsize1536_2026-07-31.json`, compared against
  `benchmarks/baseline_2026-07-27/baseline_2026-07-27.json`):

  | clip | vram Δ | time Δ | mean mask IoU | min IoU | frames IoU<0.5 |
  |---|---|---|---|---|---|
  | CIELE | −75.1% | −10.0% | 0.94 | 0.90 | 0/816 |
  | Dispo_RDV | −71.9% | −7.9% | 0.98 | 0.96 | 0/840 |
  | **MattSwift** | −49.7% | −11.9% | **0.35** | 0.30 | **150/150** |
  | accident_electrique_02 | −55.9% | −0.5% | 0.93 | 0.87 | 0/375 |
  | **atelier_1** | −58.0% | −16.0% | **0.40** | 0.06 | **105/158** |
  | boutique1_HQ | −72.5% | −10.6% | 0.87 | 0.76 | 0/597 |
  | circulation_site_1_edit_coupe | −53.1% | −9.5% | 0.65 | 0.59 | 0/156 |
  | scene01 | −63.6% | −9.3% | 0.83 | 0.64 | 0/630 |
  | scene_03 | −66.6% | −10.1% | 0.70 | 0.57 | 0/617 |
  | trop_long_embouteillage | −74.1% | −11.2% | 0.75 | 0.65 | 0/1349 |
  | **mean** | **−67.9%** | **−10.0%** | 0.738 | — | — |

  Mean VRAM/time/`bg_depth_cv` numbers look excellent — matches the earlier
  bf16+scale0.5 stacked win almost exactly (−67.9% vs −68.8%) with time now also down
  (−10.0%, vs flat before). **But MattSwift and atelier_1's mask IoU collapsed right back
  to the old scale=0.5 failure values (0.35/0.40, essentially identical to 0.34/0.38),
  not the clean maxsize-alone values (0.93/0.87) found just above.** Re-ran MattSwift's
  exact stacked config pinned to the same GPU used for the clean maxsize-alone result to
  rule out cross-GPU non-determinism (the known confound from the earlier
  baseline-noise finding) — **got the identical IoU (0.3487, bit-for-bit) both times**,
  so this is deterministic and real, not run-to-run noise. **Conclusion: `SPAG_SAM3_BF16`
  and `SPAG_SAM3_MAXSIZE` are each individually safe, but interact badly when stacked**
  — bf16's weight-precision loss apparently removes just enough signal margin on an
  already-downscaled (768×384) thin-limb input to flip leg detection, even though bf16
  alone at full native resolution had ample margin (IoU 0.99) and maxsize alone at fp32
  had ample margin too (IoU 0.93). **Do not stack `SPAG_SAM3_BF16` with
  `SPAG_SAM3_MAXSIZE` (or `SPAG_SAM3_SCALE`) as a default — each is safe solo, the
  combination is not, and the failure is clip-content-dependent (thin/articulated
  subjects) so it won't show up on every clip's aggregate metrics.** A colored-depthmap
  render of the stacked run makes the MattSwift/atelier_1 leg loss directly visible
  (`scripts/render_depth_colormap.py`).
- **Target: peak VRAM <16GB for any clip length (user-stated, see memory
  `project_vram_target_16gb`) — `trop_long_embouteillage` (1349 frames, native 1920×960,
  by far the longest clip in the 10-clip set) is the sole failure at 18.2GB under
  bf16+maxsize1536.** Since its native resolution is already modest, `maxsize=1536` only
  gives it a mild 0.8× downscale (818×408) — little headroom left on that lever for this
  clip specifically. Swept `SPAG_SAM3_MAXSIZE` more aggressively on this clip alone
  (1536/1024/768, bf16 on, isolated single-clip runs so not directly comparable in
  absolute VRAM to the batched table above, but internally consistent):

  | maxsize | target res | VRAM | mean IoU vs maxsize1536 | frames IoU<0.5 |
  |---|---|---|---|---|
  | 1536 | 818×408 | 20,098 MB | (reference) | — |
  | 1024 | 546×272 | 15,294 MB | **0.28** | **338/338** |
  | 768  | 408×204 | 15,539 MB | **0.28** | **338/338** |

  **VRAM plateaus at ~15.3GB going from maxsize 1024→768 (flat, not falling further)
  while mask IoU collapses to ~0.28 with every single frame below 0.5** — i.e. pushing
  the resolution lever past 1024 buys no further VRAM reduction on this clip and destroys
  segmentation. This is strong evidence the VRAM floor here is **not** driven by
  per-frame resolution but by something that scales with frame count / tracked-object
  count instead — most likely SAM3's memory-bank/tracking-state accumulating over 1349
  frames (this clip's frame count is 1.6-3.6x every other clip in the set). Consistent
  with the `circulation_site_1_edit_coupe` outlier noted in the VRAM-prediction-model fit
  above (shortest clip, VRAM far below what resolution alone predicted). **Conclusion:
  resolution-based levers (`SPAG_SAM3_SCALE`/`SPAG_SAM3_MAXSIZE`) cannot close this gap
  for long clips — hitting <16GB on `trop_long_embouteillage` needs an orthogonal lever
  that bounds memory-bank growth over long sequences** (e.g. periodic memory-bank
  pruning/reset, or chunking very long clips the way `SPAG_SAM3_MAX_FRAMES` already does
  for correctness — see item 2 below — but tuned/measured explicitly as a VRAM lever
  rather than just a correctness fix). **Decision (2026-07-31, user): stop here — accept
  18.2GB on this clip as a known limitation rather than build session-reset chunking.**
  The real fix would need periodic fresh SAM3 sessions (not just windowed
  `propagate_in_video` calls on one session — those don't free memory-bank state, see
  item 2) to actually bound growth, which carries the same tracking-continuity risk at
  reset boundaries that the original uncapped/unchunked attempt hit (23/38 frames
  covered) — judged not worth it for a single outlier clip. `trop_long_embouteillage`
  (and other very-long clips) stay over the 16GB target under the current best config
  (bf16+maxsize1536); the other 9/10 clips are comfortably under (5.7-11.5GB).
- **Item 3 `SPAG_SAM3_SCALE` (opt-in, default off) — the real VRAM lever.** Downscales the
  frames SAM3 decodes → shrinks the image-encoder activations that *are* the peak. 3-clip
  subset @ 0.5: VRAM **−51% to −77%**, time flat (SAM3 compute isn't the time bottleneck),
  all four metrics inside the ±5% gate on all 3. Caveat: a coarser SAM3 input changes which
  contours get promoted to prompts → can track genuinely different objects; MattSwift (not
  in that subset) showed real fg-tracking drift (`fg_depth_cv` 0.827→0.937) in an earlier
  run. Best candidate for eventual default-on, pending a wider re-benchmark including a
  drift-prone clip. Implementation notes: build the native-res PIL frame list from a direct
  `cv2.VideoCapture` pass; upsample masks to `meta['H']/meta['W']` (native), not the
  WAFT-scaled `viz_frames`.
- **Item 2 `SPAG_SAM3_MAX_FRAMES` chunking (shipped opt-in, commit `afc7b49`) — correctness
  fix only, not a useful lever.** Chunked/re-seeded `propagate_in_video`: each new window is
  re-seeded via `_reseed_window` (centroid `add_prompt` at the window start) because chaining
  `propagate_in_video` on one session does *not* reliably continue tracking — without a
  re-prompt, later windows come back with empty masks and the object is silently lost
  (caught by visual mask inspection; "every frame has an entry" ≠ "every frame has a real
  mask"). The correct re-seeded version returns time/VRAM to baseline (re-seeding costs
  inference and doesn't bound cumulative session memory). Ship opt-in so it no longer drops
  the object; do **not** recommend as an optimization.
- **Item 1 `SPAG_SAM3_FP16` — removed, no benefit.** `torch.autocast` casts only activations
  per-op; weights stay fp32 (the dominant VRAM cost), and dispatch overhead cancels any
  compute saving. Real weight fp16 needs casting every call-site input — bigger/riskier than
  the payoff. Env var + autocast wrapping reverted; no code footprint.
- **Item 7 `SPAG_MASK_SCALE` — REMOVED from code (2026-07-24).** Its premise was wrong.
  Direct per-phase measurement on a long clip (`vid360_bruit_operatrice`, 963 frames,
  `SPAG_MASK_SCALE=1.0`, reset peak after WAFT freed): **WAFT flow 2,717 MB vs SAM3 23,808 MB
  — SAM3 is the peak by ~8.8×, at 1 tracked object.** WAFT is freed *before* SAM3 runs, so
  shrinking the (already-freed) WAFT mask-pass tensors cannot lower a later peak. Where item 7
  ever cut peak, the mechanism was SAM3 (a coarser mask → fewer/different contours promoted →
  smaller memory bank) — a quality-coupled side effect, not a clean knob, and it does nothing
  on a clip already at its 1-object floor. The real SAM3-peak lever is item 3. Removed rather
  than kept as a misleading opt-in. (The old 40–77 GB clip flags in this doc's history were
  always SAM3, never WAFT.)

## Not pursued / deferred

- **fp16 depth (old Tier 2)** — dropped; targets only the ~4 s depth model, zero VRAM benefit
  now the peak is SAM3-bound, carries re-validation risk.
- ~~fp16/bf16 SAM3 *weights*~~ — **attempted, see item 8 above** (`SPAG_SAM3_BF16`).
- **CUDA graph capture of the per-frame depth chain** — the chain is *not* branch-free at the
  Python level (frame-0 special-case in composite/propagate; value-dependent host branching in
  align, `.item()` early-exits). Would need a branch-free rewrite (`torch.where`) + pinned
  static buffers — a correctness-risk redesign. Deferred.
- **Depth-loop parallelism** — the loop is causal (propagation consumes the previous frame's
  composited depth; smoother is a causal window), so frames can't be parallelized. Off-path
  overlap already captured by prefetch + async PLY.
- **Host-RAM levers** (keep flows at flow-res + lazy per-frame upscale; make
  `aligned_depth_list_cpu` optional) — relevant only to host RAM on long clips, not the VRAM
  metric; not implemented.

## Notes / gotchas

- **`freeze_bg` has no effect on stability/VRAM** — `False` vs `True` is identical to 6 sig
  figs on all metrics and VRAM; `False` just costs +35–40 s/video of bg-Gaussian bookkeeping.
  `freeze_bg=True` (default) stays strictly better for this metric set.
- **Multi-GPU benchmark ordering** — set `CUDA_DEVICE_ORDER=PCI_BUS_ID` *before*
  `CUDA_VISIBLE_DEVICES`. Without it, index N under plain CUDA ordering can resolve to the
  4 GB "NVIDIA DGX Display" device instead of the A100 `nvidia-smi` shows at that index →
  silent OOM.
- **Benchmarking on one GPU** — back-to-back runs throttle (2nd-run penalty up to +20 s on a
  short clip); cool to ≤46 °C between runs, or the ordering effect swamps small deltas. Clock
  locking needs perms this box's user lacks.
- **Harness scripts referenced in old versions of this doc** (`run_capture.py`,
  `make_diff_figures.py`, `benchmark_c1_*`) no longer exist — see `CLAUDE.md`. Current
  one-offs: `benchmark_solutions.py`, `benchmark_tier4_items.py`, `batch.sh` at repo root.
