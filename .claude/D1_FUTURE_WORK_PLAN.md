# D1 — Future work toward the project goal

_2026-07-24, revised same night after finding the downstream consumer. Mostly historical now —
P1/P2/occlusion/`SPAG_LOCK_ACTIVITY` are shipped, P3/P4 were benchmarked and rejected. Kept for the
decision trail and the one item still open (`SPAG_LOCK_ACTIVITY`'s screen/appearance gap below).
High-level summary of all of this also lives in `docs/WORK_LOG_DYNAMIC_360_RECONSTRUCTION.md`._

## Pipeline reality — SPAG4D is stage 1 of two

[goal.md](goal.md) says "ERP video → navigable 3DGS scene," but SPAG4D doesn't produce the
navigable scene itself — it's the front half feeding **FreeTimeGS**
(`/raid/mb273924/FreeTimeGsVanilla`), which trains **one** set of 4D Gaussians (position, canonical
time, duration, velocity) from SPAG4D's per-frame depth/mask/flow as init + persistent supervision.

**This settles the old open question about per-Gaussian temporal identity / popping: it's not
SPAG4D's job.** That's exactly what FreeTimeGS's 4D representation (one Gaussian set + velocity,
not N independent per-frame sets) exists to solve. SPAG4D's job is to feed it the best possible
**per-frame** signals — stable metric depth, clean masks, optical flow — and every item below is
ranked by whether it improves that downstream reconstruction.

**The downstream bottleneck**: a fixed 360° camera means zero parallax, so photometric optimization
alone drives Gaussians into ray-aligned "needles" that dislocate under any camera translation.
FreeTimeGS's fix is to use SPAG4D's depth (Phase 1) and **flow** (Phase 2) as persistent training
supervision. Phase 2 was blocked on a flow export that didn't exist — that's P1 below.

## P1 — Export per-frame WAFT flow — ✅ shipped

`flow_{idx}.npy` (H×W×2 float32, native res, forward flow t→t+1, `idx>=1`) dumped alongside
`depth_{idx}.npy`/`mask_{idx}.npy` whenever `depth_npy_dir` is set, on the bglock flow path
(`video.py`). No production-path change — only active when `depth_npy_dir` is already on. Feeds
`FreeTimeGsVanilla/docs/FLOW_DEPTH_MASK_TRAINING_PLAN.md` Phase 2 (velocity supervision).

## P2 — `SPAG_DETERMINISTIC` — ✅ shipped

**Why**: `fg_depth_cv` swung 3.75× between identical reruns (0.0955 vs 0.359, MattSwift) because
nothing seeds torch/numpy/cuDNN and a marginal contour threshold flips which frames register a
track — made every foreground-quality comparison untrustworthy.

**What it does**: `SPAG_DETERMINISTIC=1` (opt-in, default off) seeds np/torch/CUDA (`SPAG_SEED`,
default 42), sets `cudnn.deterministic=True` / `cudnn.benchmark=False`. Deliberately *not*
`torch.use_deterministic_algorithms(True)` (raises on WAFT/SAM3 ops without deterministic kernels).

**Validated**: two deterministic runs on different GPUs reproduce byte-identical
`stabilized_fg_median` (max|Δ|=0). Cross-GPU bit-exactness is **not** guaranteed — always compare
both sides of a benchmark under the same flag, never a deterministic run against a non-deterministic
baseline (they can differ by ~0.075 CV).

## P3 — `SPAG_SAM3_SCALE` default-on — ❌ rejected, stays opt-in

Background is scale-invariant (bg-locked → `bg_depth_cv` moves <3.2% at scale 0.5), but foreground
depth stability degrades materially on small/seam-adjacent objects (`fg_depth_cv` +49.8% MattSwift,
+27.9% scene01 at scale 0.5) — real signal, confirmed under P2's determinism, not measurement noise.
VRAM win is real (−46% to −69%) but only lossless when the foreground is large/central. **Verdict:
per-clip opt-in only** — never default it on drift-prone or seam-adjacent-foreground clips. (Later
superseded in practice by `SPAG_SAM3_MAXSIZE`, see `B1_VRAM_TIME_REDUCTION_PLAN.md`.)

## P4 — Conditional seam padding — ❌ confirmed dead end

Idea: recompute the expensive seam-padded flow only on the ~20% of frames where moving foreground
touches the ERP seam, keeping `SPAG_SINGLE_PASS`'s time win everywhere else. **Root cause the idea
missed**: `pad_circular_horizontal` pads the whole frame and feeds it to WAFT (a full-frame
network), so `seam_pad=0` vs `64` perturbs interior flow just as much as seam-band flow (measured:
interior mean|Δflow| 0.0351 vs seam-band 0.0349 — uniform, not localized). Seam padding can't be
conditionally localized; only recomputing *all* frames matches baseline, and that's already the
proven-slower `SPAG_SP_FIX_SEAMGAP` dead end. `SPAG_SINGLE_PASS` stays per-clip opt-in for clips
with no seam-crossing foreground motion, full stop.

## Occlusion / track-drift handling — ✅ shipped

[goal.md](goal.md) specified an FB flow-error threshold to prevent track drift on occlusion. Since
SAM3's internal memory accumulation has no per-frame skip hook, the lever is *which frame anchors
each track*: `SPAG_OCCL_FBGATE=1` picks a track's first occurrence with FB-consistency error below
`SPAG_OCCL_FB_THRESH` (tuned to **3.0px**, swept 1.5/3.0/4.0) instead of blindly using the very
first occurrence, which can land on an occluded/unreliable frame.

**Net result** (jam clip, 1349 frames, per-track present-frame span via SAM3 mask persistence):
threshold 3.0 + `SPAG_OCCL_MAX_SKIP=8` (a second signal capping how many occurrences a track may
skip before re-anchoring, added to fix one regression — see below) gives **+1428 present-frames
net (+6.4%)**, with genuine occlusion recoveries at full strength (obj 17: 105→751 frames, obj 2:
552→1349) and no track regressing by more than ~10 frames.

**Why the skip cap was needed**: FB-error alone can't distinguish "noisy local flow on a
continuously-tracked object" (obj 1: FB-error 10.6px on frame 0, but *not* actually occluded — just
noisy, with 16 dense detections before its "clean" anchor) from "genuinely occluded, barely detected
until it clears" (obj 17: zero detections until frame 432, clean within 5 occurrences). Since
re-anchoring discards every occurrence before the clean one, capping the discard budget
(`SPAG_OCCL_MAX_SKIP=8`) fixes obj-1-style false positives while preserving genuine recoveries.

Ships as a committed opt-in flag, inert (byte-identical) when off.

## `SPAG_LOCK_ACTIVITY` — decouple activity mask from bg-lock decision — ✅ shipped default-on

**The bug**: `alignement_mask="sam_and_activity"` built one `fused_mask = activity | sam` and used
it for two jobs needing different masks — excluding untrustworthy pixels from the alignment fit
(correct use of activity, a whole-video statistic) vs. deciding which pixels are moving *right now*
in `composite_bg_locked` (category error: one frame of motion anywhere in the clip unlocks a pixel
for the *entire* clip, undoing exactly the flicker bg-lock exists to remove).

**Fix**: `lock_mask = sam_mask` for compositing/disocclusion; the alignment fit and reported metrics
keep using `fused_mask` (so metrics stay comparable). Default on; `SPAG_LOCK_ACTIVITY=0` restores
old behavior; no-op under `alignement_mask="sam"`.

**Measured** (`circulation_site_1_edit_coupe`): activity-only pixels' drift CV dropped 5.40%→0.245%,
pure-static drift CV 0.230%→0.004%, foreground motion preserved (peak `in/std_max` 13.49→14.98,
i.e. rose, not suppressed). Downstream NPZ background point count dropped 3.84M→576K (voxel dedup
5×→26×) with foreground bit-identical — previously-unlocked activity pixels were jittering across
voxel boundaries and defeating dedup.

**Known cost, accepted**: genuinely-moving background the mask doesn't catch (foliage, water,
rippling fabric, a video screen) gets its *geometry* frozen to the temporal median. Downstream
forces background velocity to zero regardless, so that motion was never reaching the 4D model
anyway.

**Open item, not yet built — static geometry / dynamic appearance (video screens).** For a screen
the frozen *geometry* is correct (it really is a static plane), but nothing downstream can represent
its changing color: FreeTimeGS Gaussians have no time-varying `sh0`/`shN` — the only time-varying
mechanisms are position (velocity) and a Gaussian temporal-opacity envelope
(`sigma(t)=exp(-0.5*((t-t_center)/duration)^2)`). Screen pixels currently go down the background
path, which voxel-dedups and *medians color* across all keyframes into one always-on Gaussian — a
grey smear. This predates `SPAG_LOCK_ACTIVITY` (the flag changes depth, not the fg/bg split); the
flag just makes the smear geometrically clean instead of also jittering in depth. Proposed fix
(not built): a third mask class routed to a non-deduped, per-keyframe short-duration path so the
opacity-stack mechanism can encode a fading sequence of fixed colors instead of one median. Only
worth doing if a screen-heavy clip (`tissc`, `scene01` are candidates) actually looks wrong.

## Dependency order

```
P1 (flow export)   ── independent, cheapest, highest downstream value ── done first
P2 (determinism)   ── unblocked P3, P4, occlusion, and any FG change
   ├─ P3 (SAM3_SCALE default-on)    — rejected
   ├─ P4 (conditional seam padding) — dead end
   └─ occlusion                     — shipped
```

## Explicitly NOT SPAG4D's job

- **Per-Gaussian temporal identity / anti-popping** — FreeTimeGS's 4D representation exists
  specifically to solve this; SPAG4D feeds depth/mask/flow, doesn't track Gaussians.
- **Free-viewpoint parallax from a fixed camera** — physically absent; the realistic downstream
  target is graceful degradation under small translations via depth+flow supervision, not
  synthesizing parallax that was never captured.

## Dead ends confirmed on this list

Background stabilization (solved, bg-lock + `SPAG_LOCK_ACTIVITY`); robust affine RANSAC/Siegel (no
value while bg is locked); `sol4` multi-frame reference (no-op); vipe replacement via Open-d4rt
(not viable, see `Open-d4rt/OPEN_D4RT_EVALUATION.md`); full `SPAG_SP_FIX_SEAMGAP` (correct but
slower than baseline); conditional seam padding (P4 above).
