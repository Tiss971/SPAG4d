# Dynamic 360° video reconstruction — project resume (Mathis BOSTON)

Onboarding / status doc: what this project does, what's been built, what's proven to work, and
what's still open. Scope: everything authored by Mathis BOSTON from `4cd2393` ("feat: prototype
flow-based depth propagation for fixed 360 ERP video", 2026-07-09) through the current uncommitted
working tree (2026-08-25) — 32 commits, all his (`git log --format='%an' 4cd2393..HEAD`), plus
work in progress.

## Executive summary

**Before this work**, SPAG4D only handled the static case: one 360° (ERP) image → monocular depth
→ 3D Gaussian Splat (3DGS) projection. No notion of time, motion, or moving subjects.

**What was built**: a per-frame *video* pipeline for a **fixed** 360° camera filming a **dynamic**
scene. The core idea is to decouple the scene into two parts that are estimated completely
differently, then recompose them every frame:

- **Background** — assumed static (fixed camera). Its geometry is computed **once** from a
  temporal-median reference depth and locked ("bglock"), which eliminates the flicker that
  independent per-frame depth estimation would otherwise produce on a scene that never actually
  moves.
- **Foreground** — moving subjects, detected and tracked with SAM3 video segmentation, with their
  depth re-estimated and temporally smoothed every frame (optical flow propagation + a confidence
  model that decays trust the longer a pixel goes without a fresh estimate).

This decoupling is what makes the reconstruction usable for the downstream goal: SPAG4D is stage 1
of a two-stage pipeline feeding **FreeTimeGS**, a 4D Gaussian trainer, which needs exactly this
static/dynamic split to train a navigable, temporally-consistent scene.

**Status**: the bglock + SAM3 + flow-propagation pipeline is shipped and is the default production
config, validated across a 10-clip reference baseline. Two things remain open: (1) one specific
visual artifact (a depth "trail" behind fast-moving subjects) that five fix attempts have not fully
solved, and (2) a second uncommitted round of fixes/features (see §6) not yet merged.

## Models used — all pretrained, zero-shot, nothing fine-tuned

No model was trained or fine-tuned as part of this work. Every model below is used off-the-shelf;
all of the actual engineering is **post-model, classical processing** on top of frozen model
outputs (compositing, confidence-weighted temporal blending, mask logic, pruning heuristics) —
not weight updates.

| Role | Model | Status |
|---|---|---|
| Monocular depth (main, production) | **DA360** (Depth Anything, ERP-adapted) | Shipped default (`active_generator="da360"`) |
| Monocular depth (alt) | **DAP** | Available, same machinery as DA360 |
| Monocular depth (proposed) | **PaGeR** | Panoramic depth+normals+sky-mask model; wrapper exists (`pager_model.py`) but inert in production — design not fully landed |
| Optical flow | **WAFT** | Checkpoint `tar-c-t.pth`; drives temporal depth propagation + motion masks |
| Video segmentation/tracking | **SAM3** | Object-aware moving-subject masks, tracked across the whole clip |
| Alternate full pipeline | **UniSHARP** | External subprocess producing a complete 3DGS reconstruction per frame; architecturally separate branch, incompatible with bglock/SAM3/WAFT |

## What's tested and validated

- Temporal-stability solutions 1/4/5/6 compared via a purpose-built benchmark harness →
  `bglock_sol1_median_w5` won and became the shipped approach.
- **10-clip reference baseline** recorded over the shipped production config (the closest thing to
  a regression suite this project has).
- Confidence-decay compounding audited against live code (found genuinely dead, then fixed).
  An ablation-ladder pass caught a real trap in the process: on MattSwift, the **old** (pre-fix)
  confidence path posted a *better-looking* stability number (`fg_depth_cv` 0.131 vs. the fix's
  0.158) — but only because it was suppressing 31% of the subject's real motion. Any stability
  metric read in isolation, without a paired motion-fidelity check, can pick exactly the wrong
  config.
- **Mask-injection proxy ground truth** for the flow-propagation dynamic path (`SPAG_MASK_INJECT`):
  since no real depth ground truth exists, a genuinely-static patch is injected into the dynamic
  mask so bg-lock runs its full propagation/blend on a pixel whose correct answer (`D_ref`) is
  known. Result (MattSwift): RMSE 0.56m at the equator vs. 0.01-0.07m at the poles — the only real
  accuracy number that exists for this path, and it shows error is concentrated exactly where real
  motion actually lives (low-texture/near-zero-flow poles are the easy case, not evidence of
  general accuracy).
- Scene01 "waiter trail" artifact: 5 fix attempts benchmarked with matched-config visual +
  pixel-diff validation; one (`SPAG_BLEED_REJECT`) measurably reduced the artifact but was
  **reverted** after it was found to delete real background-contact points (e.g. feet on the
  floor) as false positives.
- Dispo_RDV close-range-people-invisible bug: isolated simulation + full end-to-end rerun
  confirmed the fix (a background-derived depth floor was wrongly gating out nearby dynamic
  subjects).
- `freeze_bg_live_color` (frozen geometry, live-resampled color): 3-arm comparison
  (bglock / frozen_bg / frozen_bg_live_color) on two clips chosen to stress shadows and screens.
- UniSHARP vs DA360/PaGeR: compared on a background-stability metric only so far, favorable to
  UniSHARP among non-bglock generators — **no direct comparison against bglock exists yet**.

## What's working (shipped defaults)

- `depth_correction="bglock"` — background geometry locked to a one-time reference; near-zero
  background flicker.
- GPU-resident depth chain (align → smooth → propagate → composite, all on GPU) — practical
  runtime, not just correctness.
- SAM3 tracking: prompts spread across the body + fragment merging, plus re-seeding across
  chunked tracking windows — no more identity loss/fragmentation over a clip.
- Confidence-decay compounding over a per-pixel age field — propagated depth trust actually decays
  with distance from the last fresh estimate, instead of being pinned to a fixed blend.
- `freeze_bg` + `freeze_bg_live_color` (opt-in) — frozen geometry with live-updating color, fixing
  the "dead" look of frozen shadows/screens.
- Several correctness bugs fixed along the way: mirrored pixel lookup in grazing-angle pruning,
  depth/sky-mask sampled at the wrong sub-pixel location, sky threshold clipping real far
  background, close-range dynamic subjects being invisibly pruned.
- VRAM/perf: ~36% VRAM reduction and ~40% cumulative wall-time reduction on the production config,
  without any accuracy tradeoff (closed-form fits, batching, GPU residency — all lossless).

## What's still open

- **Scene01 depth "trail"** behind fast-moving subjects under bglock — root cause (per-pixel depth
  bleeding across the silhouette edge during flow warp) is understood, but every fix tried so far
  either doesn't help or introduces a worse regression. A sixth attempt is coded but not yet
  validated.
- **UniSHARP vs bglock**: no head-to-head comparison exists; UniSHARP's structural advantage
  (native per-frame 3DGS, no cubemap seams) vs bglock's near-perfect background stability is
  unresolved.
- **PaGeR**: designed and partially wired, not active in production.
- Forward-backward flow consistency isn't yet linked to SAM3's tracking/memory during detected
  occlusions (known gap, not yet addressed).
- **VRAM target (<16GB, any clip length) is not fully met.** The best shipped config still peaks at
  18.2GB on the longest clip in the 10-clip set (`trop_long_embouteillage`, 1349 frames). Pushing
  the SAM3 input-resolution levers further doesn't help past a point — VRAM plateaus at ~15.3GB
  even at very aggressive downscaling (while segmentation quality collapses), showing the floor on
  long clips is driven by SAM3's memory-bank growth over frame count, not per-frame resolution.
  Fixing it for real would need periodic fresh SAM3 sessions to bound that growth, which
  reintroduces a tracking-continuity risk at reset boundaries — judged not worth it for one outlier
  clip; accepted as a known limitation rather than solved.
- A second round of fixes (§6 below) is uncommitted and needs review before shipping.

---

## Technical deep-dive: six key mechanisms

Full mechanism detail lives in the `.claude/` files (kept there as the load-bearing reference,
compacted 2026-08-25) — this section is now just a map of where to look:

1. **Temporal background creation (`D_ref`)** — `.claude/pipeline_overview.md` steps [4]-[5].
2. **SAM3 propagation inputs (text prompt + flow mask)** — `.claude/pipeline_overview.md` step [3].
3. **Edges of dynamic content / the "trail" artifact** — `docs/bglock_open_questions.md` §8.7
   (full #2-#5 fix-attempt history) + `.claude/HOW_TO_HAVE_TEMPORAL_STABILITY.md`'s 2026-08-20→21
   chronology row; the related-but-distinct feathering "phantom slab" fix
   (`SPAG_HARD_DEPTH_CUTOVER`) is `docs/bglock_open_questions.md` §8.1.
4. **`freeze_bg_live_color` vs `freeze_bg` vs no freeze** — `.claude/pipeline_overview.md` step
   [6.4] and `docs/bglock_open_questions.md` §8.6.
5. **Affine vs. bglock (`depth_correction`)** — `.claude/pipeline_overview.md` step [6.2] +
   `.claude/HOW_TO_HAVE_TEMPORAL_STABILITY.md`'s A/B/C benchmark table.
6. **VRAM reduction (SAM3 input size, bf16, WAFT input size)** — `.claude/B1_VRAM_TIME_REDUCTION_PLAN.md`
   in full (SAM3 VRAM levers section).

---

# Detailed chronology (commit-by-commit)

## 1. Foundational prototype (2026-07-09 → 07-10)

- **`4cd2393` — flow-based depth propagation, prototype.** The seed idea: instead of running
  monocular depth fresh on every frame (temporally unstable — independent per-frame estimates
  flicker even on a perfectly static scene), warp the *previous* frame's depth forward using
  optical flow (WAFT) and blend it with the fresh estimate, weighted by forward-backward
  flow-consistency error as a trust signal.
- **`b48b1bb` — validated on real 4K ERP footage.** Confirms the prototype survives contact with
  real fixed-camera 360 video, not just synthetic tests.
- **`33eb9c5` — background-locked compositing ("bglock").** The key architectural split: for a
  *fixed* camera, the background's true geometry never changes, so lock it to a one-time reference
  depth (temporal median) instead of re-deriving it per frame — eliminates background flicker
  entirely, at the cost of needing a reliable mask of what counts as "background" vs. "moving
  foreground" to know where to apply the lock vs. where to let per-frame estimation through. This
  mask is where SAM enters the pipeline.
- **`e3ed4a7` — baseline snapshot** before a round of temporal-stability experiments (safety
  checkpoint, not a feature).
- **`bf36872` — temporal-stability solutions 1/4/5/6 + benchmark harness.** Multiple competing
  stabilization strategies implemented side by side with a harness to compare them
  quantitatively — this benchmark-first approach (documented later in
  `.claude/HOW_TO_HAVE_TEMPORAL_STABILITY.md`) becomes the standard way every subsequent stability
  change in this repo gets evaluated.
- **`f6f899f` — parallel-safe benchmark reporting** (infra for the above, so concurrent benchmark
  runs don't clobber each other's summaries).
- **`b853e58`, `8d3bcea` — robustness fixes**: WAFT flow failures fall back to SAM cleanly instead
  of crashing; malformed SAM masks are skipped rather than corrupting the temporal median used for
  bg-lock.

## 2. SAM integration + video path hardening (2026-07-15)

- **`23eecc8` — background-locked depth correction + `unisharp360` video path**, the commit that
  makes bg-lock a first-class option in the main video conversion path (not just an experiment),
  alongside a second sharpening-focused ERP generator (`unisharp360`) as an alternative to the
  primary `da360` depth model.
- **`82a1d29` — fix a mirrored pixel-column lookup in `prune_grazing_angle`** (scene_filter bug:
  wrong column indexing was silently corrupting which points got pruned near grazing angles).
- **`8600be1` — `unisharp360` always passes `--no-render`**, dropping a dead env-var gate — cleanup
  after the generator had stabilized.
- **`d7487f0` — modernize type hints, unisharp defaults cleanup** (housekeeping).
- **`6ec2433` — dev tool: reconstruct ERP depth maps from a 3DGS PLY** (a debugging/inspection
  utility — given an output splat file, recover what depth map produced it, for validating the
  projection math independent of the video loop).
- **`b01eff4` — drop a malformed permission entry** (repo hygiene, unrelated to the pipeline).
- **`d7678d0` — merge**: `bg-locked depth stability + unisharp360 video path` lands on the main
  line from a worktree branch, consolidating everything above.

## 3. Downstream integration + SAM3 tracking quality (2026-07-17 → 07-20)

- **`6633a5a` — export moving-mask + freeze_bg point count for FreeTimeGsVanilla**. This is the
  first explicit acknowledgment that SPAG4D is stage 1 of a two-stage pipeline: the moving/static
  mask produced here is exported specifically to support background regularization in
  FreeTimeGS (the downstream 4D Gaussian trainer).
- **`52bc130` — one-off export script** for a specific clip/config combination
  (`bglock_sol1_median_w5` on `accident_electrique_fast2`) — the stability solution from `bf36872`
  that ultimately won the benchmark comparison.
- **`9408818` — spread SAM3 prompts across the body + merge fragment boxes.** SAM3 tracking
  quality fix: previously a single prompt point/box per subject could fragment into multiple
  disconnected mask pieces (e.g. tracking loses the legs but keeps the torso); spreading prompts
  across the body and merging fragments back together produces one coherent subject mask instead
  of several unstable fragments — directly improves the reliability of the fg/bg decoupling this
  whole pipeline depends on.
- **`4daa3d1`, `61dda17`, `d7c833f`, `36fc85e` — VRAM/perf pass on the bg-lock + smoothing config**:
  −36% whole-run VRAM, −14% peak VRAM via offloading SAM3 inference-state to CPU, −19% wall time
  via batched ERP warp + cached grids + partition median, plus a closed-form (vs. iterative)
  least-squares solve for the per-frame affine scale/shift depth-alignment fit. These make the
  temporally-stable pipeline actually practical to run at scale, not just correct.
- **`7057fc5`, `e5b1681`, `2312810` — GPU-resident depth chain**: moves the align/smooth/propagate/
  composite steps to stay GPU-resident across the per-frame loop instead of round-tripping to CPU
  numpy between steps, for a further −24% then −40% cumulative wall-time reduction (documented as
  it landed).

## 4. Single-pass flow + scene_filter/SAM3 correctness (2026-07-21)

- **`d23cb01` — GPU-native bg-locked composite (default on) + single-pass WAFT flow.** Two
  changes: the bg-lock compositing step itself moves fully onto GPU (removing another CPU
  round-trip), and an opt-in `SPAG_SINGLE_PASS` mode runs WAFT once instead of twice per frame
  pair (later found to degrade foreground depth on seam-crossing clips — stays opt-in, not
  default, per `CLAUDE.md`).
- **`c53d3f6` — docs**: folds the composite-on-GPU + single-pass results into the VRAM reduction
  plan.
- **`460a582` — remove stale benchmark files/scripts** related to flow propagation and
  background-locking (cleanup of experiment artifacts once the winning approach was settled).
- **`1d79c80` — fix(scene_filter): sample depth/sky mask at cell center**, matching where the
  Gaussian is actually placed (previously sampled at a cell corner, causing a subtle
  misregistration between the pruning decision and the point it was deciding about); also restores
  the benchmark harness and exposes more CLI options.
- **`afc7b49` — fix(sam3): re-seed tracked objects across chunked `propagate_in_video` windows.**
  SAM3's video propagation runs in fixed-size chunks for memory reasons; without re-seeding, a
  tracked object's identity could silently drop or drift across a chunk boundary — this fix keeps
  tracking continuous across the whole clip, not just within one chunk.

## 5. Occlusion handling + confidence-decay correctness (2026-07-27 → 2026-08-18)

- **`5e2b571` — FB-error re-anchor gate with skip-cap second signal.** Addresses trailing-edge
  occlusion: when a moving foreground subject uncovers background that was hidden in the previous
  frame, forward-backward flow-consistency error alone can be fooled into trusting a bad
  propagated value; adding a second signal (a cap on how many frames a pixel can go without being
  re-anchored to a fresh estimate) forces a periodic reset independent of what FB-error alone says.
- **`26b0309` — reference baseline over 10 clips**, recording the shipped production config's
  numbers across a broader clip set than the single-scene validations used during development.
- **`6712c39` — P0 confidence-decay compounding fix + backend contract + D_ref thin-coverage
  floor.** Fixes flow-propagation confidence to actually compound over a per-pixel *age* field
  warped along the flow (`SPAG_CONF_DECAY=0.85`, `SPAG_CONF_FLOOR=0.5`) — before this, decay was
  pinned to a fixed blend factor and the "decay" knob was dead code (confidence was reset to 1.0
  every trusted frame instead of decaying with distance from the last trusted anchor). Also adds a
  documented backend contract and a floor for thin-coverage reference-depth regions (`D_ref`).
  This is the current `HEAD` of the committed history.

## 6. Uncommitted work in progress (as of 2026-08-25)

A second phase of work sits uncommitted on top of `6712c39`, continuing the same
decoupling-and-recompose approach. Summary:

- **Scene-defaults/pruning fixes**: `sky_threshold` moved from `p95`→`p99` (stopped clipping real
  far background as sky); `outlier_pruning` CLI/API default lowered to match what the video path
  already used; a background-derived `depth_min` was wrongly gating out close-range *dynamic*
  foreground subjects entirely (root-caused and fixed — see
  [`docs/DISPO_RDV_CLOSE_RANGE_PRUNING_FIX.md`](DISPO_RDV_CLOSE_RANGE_PRUNING_FIX.md)).
  A related mask-scope gap in the affine depth-correction fit vs. the wider bg-lock compositing
  footprint was found and closed too — see `docs/bglock_open_questions.md` §8.7.
- **`freeze_bg_live_color`**: a hybrid background mode — geometry frozen (from the one-time
  temporal median, avoiding trailing-edge occlusion holes) but color re-sampled live every frame
  wherever that background pixel is currently visible. Fixes frozen backgrounds looking dead on
  shadows/screens. Implemented via a new `pixel_idx` field threaded through the Gaussian dict so it
  survives the existing generic prune chain. See `.claude/pipeline_overview.md` step [6.4].
- **ERP compositor fix**: cubemap face rendering switched from per-face push-splat directly into
  the ERP buffer to render-then-pull-sample, fixing floor/ceiling grazing-angle artifacts.
- **Scene01 "waiter trail" investigation** (the longest-running open thread): a visible
  depth-bleed trail behind a moving subject under bg-lock. Five fix attempts (#1-#5) were tried and
  four were null results or non-contributing; #5 (`SPAG_BLEED_REJECT`, a global per-frame
  median-gap threshold) measurably reduced the trail but was reverted for deleting genuine
  background-contact points (e.g. feet on the floor) as false positives. A sixth attempt is now
  in progress, uncommitted: per-pixel depth-discontinuity detection via local gradient structure
  (not a global threshold) gating nearest-vs-bilinear flow-warp sampling and/or a confidence
  penalty at silhouette edges — three independent env-gated knobs, all currently OFF by default,
  not yet validated against this specific clip.


## Overall throughline

Every commit in this range serves one of three purposes: (1) **decoupling** — separating a fixed
camera's genuinely-static background from genuinely-moving foreground, first via flow-based
temporal propagation, then via SAM-driven spatial masking, then via bg-lock compositing that
recombines the two; (2) **stability/correctness** of that decoupling under real footage (SAM3
fragment/re-seed fixes, mask-scope gaps, sky/depth-range pruning bugs, confidence-decay
compounding); or (3) **making it fast enough to run** (VRAM offloading, GPU-resident chains,
single-pass flow, closed-form fits). The repo went from "one 360 image → one static splat" to "a
per-frame video pipeline that holds a fixed camera's background rock-steady while tracking and
re-estimating every moving subject independently," feeding a downstream 4D Gaussian trainer
(FreeTimeGS) that consumes exactly that static/dynamic split.
