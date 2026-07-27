# D1 — Future work toward the project goal

_2026-07-24. Forward-looking plan. **Revised the same night** after discovering the
downstream consumer (`/raid/mb273924/FreeTimeGsVanilla`) — that changes the framing of
what SPAG4D's future work is *for*. Read the "Pipeline reality" section first._

## Pipeline reality — SPAG4D is stage 1 of two

The goal ([goal.md](goal.md)) says "convert ERP video → consistent 6-DoF-navigable 3DGS
scene." SPAG4D does **not** produce the navigable scene by itself. It is the **front half**
of a two-stage pipeline:

```
SPAG4D  ──(per-frame ERP depth_{idx}.npy + mask_{idx}.npy + ply)──►  FreeTimeGS  ──► 4D scene
(this repo)                                                          (/raid/mb273924/FreeTimeGsVanilla)
flicker-free per-frame depth                                        one 4D Gaussian set
                                                                    (pos, canonical time,
                                                                     duration, velocity)
```

FreeTimeGS (`FreeTimeGsVanilla`, CLAUDE.md there) trains **one** set of 4D Gaussians —
each carrying position, canonical time, duration, and velocity — so a single set renders
the whole sequence. **SPAG4D's per-frame output is consumed as the init** (and, per their
active plan, increasingly as *persistent supervision*).

**This resolves the biggest open question in the old version of this plan.** The
"per-Gaussian temporal identity / popping" problem is **not SPAG4D's job** — it is exactly
what FreeTimeGS's 4D representation exists to solve (one Gaussian set with velocity, not N
independent per-frame sets). SPAG4D should **not** build 4D tracking. Its contribution to
temporal consistency is to feed FreeTimeGS the best possible **per-frame** signals: stable
metric depth, clean masks, and — the concrete missing piece — **optical flow**.

**So the operative question for SPAG4D future work is: does it improve the downstream 4D
reconstruction?** Items below are re-ranked on that basis.

### The downstream bottleneck (drives everything)

FreeTimeGS's dominant failure mode (their `VIEWER_QUALITY_DIAGNOSIS.md`,
`FLOW_DEPTH_MASK_TRAINING_PLAN.md`): the capture is a **fixed 360° camera → zero
parallax**. With no parallax and only monocular depth, photometric optimization drives
Gaussians into ray-aligned "needles" — perfect on trained views, **dislocating on any
camera translation** ("sweep d'éloignement", worse at 20k steps than 5k). Loss/anisotropy
tuning attacks the symptom, not the cause. Their fix: use SPAG4D's depth (Phase 1) and
**flow** (Phase 2) as *persistent supervision* during training, anchoring geometry to the
mono-depth surface and the velocity field to observed motion. **Phase 2 is blocked on a
SPAG4D export that doesn't exist yet** — that's Priority 1 below.

## Priority 1 — Export per-frame WAFT flow for downstream velocity supervision — ✅ IMPLEMENTED 2026-07-24 (uncommitted, needs downstream validation)

**Status:** done in [video.py:1073-1082](../spag4d/video.py#L1073) — dumps
`flow_{idx}.npy` (H×W×2 float32, native res, `idx>=1`) alongside the existing
`depth_{idx}.npy`/`mask_{idx}.npy` when `depth_npy_dir` is set, guarded on the bglock flow
path (`flows_fwd` populated). Docstring updated. Parses clean. **Not yet run end-to-end nor
consumed by FreeTimeGS Phase 2** — that's the open validation step (sanity-load a
`flow_{idx}.npy`; wire it into `FreeTime_dataset.py` per their plan).

**Why it's first now:** it's the one concrete thing the downstream **explicitly needs from
SPAG4D and cannot get elsewhere**, it directly attacks the pipeline's dominant failure
mode (needle dislocation), the consuming plan is already designed and approved
(`FreeTimeGsVanilla/docs/FLOW_DEPTH_MASK_TRAINING_PLAN.md` Phase 2), and it is nearly
free to add. Highest downstream-benefit-per-effort item available.

**What exists:** SPAG4D already dumps `depth_{idx}.npy` + `mask_{idx}.npy` per frame when
`depth_npy_dir` is set ([video.py:1073-1075](../spag4d/video.py#L1073)). The dense WAFT
flow (`flows_fwd`/`flows_bwd`) is already computed and held in memory
([video.py:462, 545-570](../spag4d/video.py#L462)), already upscaled to native
`meta['H']×meta['W']`. Today it survives only as a preview `flow_mag.mp4`.

**Approach:** in the per-frame dump block ([video.py:1073](../spag4d/video.py#L1073)), add
`np.save(depth_npy_dir / f"flow_{idx}.npy", flows_fwd[idx-1].astype(np.float32))` (H×W×2).
Watch the off-by-one: `flows_fwd` has `n_total_frames-1` entries; frame `idx` uses
`flows_fwd[idx-1]`, and frame 0 has no forward flow — export for `idx>=1` only (matches how
`depth`/`mask` already index). Document the convention (forward flow t→t+1, native res,
original-pixel displacement with x-wrap at the seam — same convention
`propagate_depth_via_flow` consumes). Consider gating behind the existing `depth_npy_dir`
so it's free when that's already on.

**Validate:** sanity-load one `flow_{idx}.npy` (shape H×W×2, finite, magnitudes sane), and
overlay against the `flow_mag.mp4` preview for the same frame. Downstream validation is
FreeTimeGS's Phase 2 (velocity loss → sweep improvement).

**Effort/risk:** very low / very low. ~2 lines + a doc note. No production-path change (only
active when `depth_npy_dir` set). **Do this first.**

## Priority 2 — Make foreground evaluation deterministic — ✅ IMPLEMENTED + VALIDATED 2026-07-27 (uncommitted)

**Why:** every foreground quality change SPAG4D might make needs a trustworthy before/after,
and the primary FG metric is noise-prone — `fg_depth_cv` swung 3.75× between identical
reruns (B1, MattSwift, 0.0955 vs 0.359), because nothing seeds torch/numpy/cuDNN and the
marginal `found_object` contour threshold ([video.py:~1802](../spag4d/video.py)) flips which
frames register a track.

**Implemented:** `SPAG_DETERMINISTIC=1` (opt-in, default off) at the top of `run_video()`
([video.py:~401](../spag4d/video.py#L401)) seeds np/torch/CUDA (`SPAG_SEED`, default 42) and
sets `cudnn.deterministic=True` + `cudnn.benchmark=False`. Deliberately *not*
`torch.use_deterministic_algorithms(True)` — it raises on WAFT/SAM3 ops lacking deterministic
kernels; seeds + cudnn.deterministic is the safe subset. `cudnn.benchmark=False` also drops
the autotune warmup, which is the throughput cost gated behind the flag.

**Validated (MattSwift, `bglock_sol1_median_w5`, 4 runs on 4 GPUs):**
- **Deterministic mode reproduces exactly:** two `SPAG_DETERMINISTIC=1` runs on *different*
  GPUs → `fg_depth_cv` 0.28158 == 0.28158, `stabilized_fg_median` **byte-identical**
  (max|Δ|=0, 0 NaN-pattern mismatches). The flag delivers true reproducibility.
- **The flag is not a no-op:** deterministic vs control differ on all 40 valid frames
  (max|Δ|=0.275) — it forces different (deterministic) kernels, so the deterministic value
  (0.282) is its own new reference, distinct from the autotuned value (0.359). Expected;
  the point is reproducibility for gating, not matching the nondeterministic number.
- **Caveat — the control swing is intermittent, didn't manifest this run:** both control
  (no-flag) runs *also* matched each other (0.35885 == 0.35885), landing exactly in B1's
  0.359 basin (not its 0.0955 one). Near-simultaneous runs autotuned into the same basin, so
  the swing is timing/load-dependent and couldn't be reproduced on demand here. It is real
  (B1) but sporadic — which is *why* the flag is worth having: it removes a rare-but-severe
  gating hazard rather than a constant one.

**How to use it:** run **all** sides of a comparison with `SPAG_DETERMINISTIC=1` (never a
deterministic run against a non-deterministic baseline — they differ by ~0.075 CV here). The
deterministic value is the reference for FG gating going forward.

**Effort/risk:** low / low. Opt-in, no production-path change (production keeps autotune).

## Priority 3 — `SPAG_SAM3_SCALE` default-on — ❌ BENCHMARKED 2026-07-27, STAYS OPT-IN

**Why it was proposed:** VRAM is SAM3-bound; this is the real lever, letting longer/higher-res
clips run through stage 1. Opt-in pending a wider benchmark, because a coarser SAM3 input can
promote different contours → track different objects.

**Benchmarked (scale 0.5 vs 1.0, `bglock_sol1_median_w5`, `SPAG_DETERMINISTIC=1` on both sides
— P2 made this reading trustworthy; 3 clips, GPU 0/1/2):**

| clip | peak VRAM 1.0→0.5 | bg_depth_cv Δ | fg_depth_cv 1.0→0.5 | fg_delta_mean Δ |
|---|---|---|---|---|
| dispo (fg simple/large) | 41.9→12.9 GB (**−69%**) | +0.1% | 0.0514→0.0513 (**−0.1%**) | +1.7% |
| scene01 (seam-crossing) | 28.7→11.6 GB (**−60%**) | +0.4% | 0.0452→0.0577 (**+27.9%**) | +5.6% |
| mattswift (drift-prone) | 19.0→10.3 GB (**−46%**) | +3.2% | 0.2816→0.4219 (**+49.8%**) | −7.2% |

**Verdict — do NOT promote to default; keep opt-in.** Clean pattern:
- **Background is scale-invariant** (bg-locked → coarser SAM3 barely moves it; `bg_depth_cv`
  within +3.2% everywhere, the bg being what VRAM-conscious long clips most care about).
- **Foreground depth stability degrades materially** where the object is small or seam-adjacent:
  mattswift `fg_depth_cv` +49.8%, scene01 +27.9% — this is the "coarser mask promotes different
  contours / tracks a different object" hazard from B1, now **confirmed as real signal** (P2
  determinism, both sides seeded) rather than the old 3.75× measurement noise. It is lossless
  only when foreground is simple/large (dispo: fg unchanged).

**So it stays a per-clip opt-in**, not a default: a big VRAM win (−46% to −69%) that the
operator turns on when a clip is too large/long to fit at scale 1.0 *and* its foreground is
large/central — never blindly on drift-prone or seam-adjacent-foreground clips. Document this
guidance next to the flag. (Benchmark artifacts: scratchpad `p3_*.npz`, `compare_p3.py`; the
`bg_spikes_per_frame` column reads FAIL only as a 0/0=nan gate artifact — both scales = 0.0,
i.e. a pass.)

**Effort/risk:** done. No production-path change (flag remains default off).

## Priority 4 — Conditional seam padding — ❌ CONFIRMED DEAD END 2026-07-27

**Idea (was):** `SPAG_SINGLE_PASS` is a real flow-phase win blocked only by the foreground
seam-pad regression (C1); the full fix (`SPAG_SP_FIX_SEAMGAP`, recompute the padded flow on
*every* frame) is +15% vs baseline. The conditional variant would recompute the padded flow
**only** on frames where moving foreground touches the ERP seam band — on the assumption that
seam padding is a *local* correction, so most frames keep the cheap flow and the win survives.

**Built + tested (`SPAG_SP_COND_SEAMPAD=1`, [video.py single-pass loop](../spag4d/video.py#L560),
scene01, `SPAG_DETERMINISTIC=1`):**
- The conditional flagged **29/157** pairs as seam-adjacent (corpus measurement obtained).
- Recomputing those 29 padded flows moved `fg_depth_cv` only 0.01553 → **0.01536** — i.e. it
  stayed at the *plain single-pass* (degraded) value, **not** the two-pass baseline **0.04515**.
  No foreground recovery.

**Root cause (why the premise is false):** `pad_circular_horizontal` pads the *whole* frame
and feeds the wider image to WAFT, a full-frame network. `seam_pad=0` vs `64` therefore
perturbs **interior** flow just as much as seam-band flow — measured directly on a scene01
pair: mean|Δflow| interior **0.0351** vs seam-band **0.0349** (essentially uniform, not
localized). So *every* frame's propagation flow differs between pad=0 and pad=64, and
`stabilized_fg_median` diverges across the whole propagation chain regardless of whether the
object is near the seam. Seam padding cannot be conditionally localized. Only recomputing
**all** frames (= `SPAG_SP_FIX_SEAMGAP`, the +15% dead end) matches baseline.

**Verdict:** no conditional path recovers the single-pass win. `SPAG_SP_COND_SEAMPAD` kept
opt-in as scaffolding/proof (with the measurement instrumentation), not shippable. This
closes the single-pass foreground-regression avenue: single_pass stays per-clip opt-in for
clips with no seam-crossing foreground motion (as CLAUDE.md already states), full stop.

**Effort/risk:** done (negative result).

## Occlusion / track-drift handling — ✅ IMPLEMENTED 2026-07-27 (opt-in, uncommitted)

[goal.md](goal.md) Step 2 specifies a bidirectional FB flow-error threshold to prevent track
drift on occlusion. SAM3's *internal* memory accumulation is a black box (no per-frame hook to
"skip this frame's memory update"), so the lever we actually control is **which frame anchors
each track**. The flow tracker registers every object at its *first-seen* frame; if that frame
is an occlusion/disocclusion event the flow (hence box/points) is unreliable and SAM3 gets
anchored onto a bad location → drift for the rest of the clip.

**Implemented (`SPAG_OCCL_FBGATE=1`, [video.py](../spag4d/video.py#L1889)):** compute the FB
consistency error (`fb_consistency_error`, already used by `propagate_depth_via_flow`) over
each candidate anchor's bbox; pick the track's first occurrence with mean FB-error below
`SPAG_OCCL_FB_THRESH` (default 1.5 px), falling back to the plain first occurrence if all are
occluded. flows are now plumbed into `segment_with_flows`; inert (byte-identical) when the flag
is off. `fb_err` maps cached per frame.

**Confirmed on occlusion-heavy clips (traffic: `circulation_site_1_edit_coupe`,
`trop_long_embouteillage`, gate off vs on, `SPAG_DETERMINISTIC=1`, `SPAG_OCCL_DEBUG=1`):**
- **Decision correctness — solid.** The gate re-anchored **6 tracks** across the two clips,
  every time skipping genuinely high FB-error occurrences and landing on cleanly-separated
  clean ones: jam obj 1 skipped 17 occluded frames (**10.61 px** → 0.96), obj 17 skipped 5
  (**43.87 px** → 1.42), obj 2 (4.85→0.83), obj 3 (2.76→0.21), obj 6 (3.98→1.31); circulation
  obj 4 (1.66→1.17). Occluded-vs-clean FB-error distributions are 3–40× apart — real occlusion
  signal, not noise. The gate demonstrably prevents anchoring a SAM3 track on a catastrophic
  (e.g. 43.9 px) frame.
- **A downstream quality win was NOT demonstrable from aggregate depth metrics.** Per-frame
  track survival (non-None `stabilized_fg_median`) was 100% both ways — too coarse for
  multi-object scenes (some foreground is always present, so it can't isolate per-*track*
  drift). Aggregate `fg_depth_cv` even ticked up slightly (circulation +3.1%, jam +6.0%), but
  that's a whole-scene number dominated by real inter-vehicle depth spread, not a per-track
  occlusion measure — wrong instrument.

**Per-track quality confirmed (mask-persistence harness, `SPAG_OCCL_MASKDUMP=<json>` →
[video.py end of `segment_with_flows`](../spag4d/video.py#L2226); jam clip, gate off vs on).**
Metric: per-obj_id **present-frame span** (# frames the track exists with a non-empty SAM3 mask;
coverage-within-span is ~1.0 because SAM3 tracks persistently, so span = the real drift signal).
Same clip ⇒ same obj_ids, so tracks compare 1:1. Result on the 5 re-anchored tracks:

| obj | FB-err skipped | present off → on | Δ |
|---|---|---|---|
| 17 | **43.87 px** | 105 → **820** | **+681%** |
| 2 | 4.85 px | 552 → 1342 | +143% |
| 1 | 10.61 px | 1026 → 976 | −5% |
| 6 | 3.98 px | 1349 → 1349 | 0 |
| 3 | 2.76 px | 1349 → 1126 | −17% |

Net **+1232 present-frames** across the re-anchored tracks. **The gate delivers big wins exactly
where the anchor was catastrophic** (obj 17: a track that died after 105 frames when anchored on
a 43.9 px occlusion survives 820 frames from a clean anchor; obj 2 +143%). **Two honest caveats:**
(1) *marginal* skips (FB-err just over 1.5 px, e.g. obj 3 at 2.76) can slightly regress the track
— not worth re-anchoring; (2) re-anchoring shifts `start_frame_index`/the joint multi-object
propagation schedule, so non-re-anchored neighbours also move (obj 8 −28%, obj 14 −18%).

**Threshold sweep + collateral damp — done 2026-07-27 → default is now 3.0 px.** Two changes
landed together: (a) swept `SPAG_OCCL_FB_THRESH` ∈ {1.5, 3.0, 4.0} on the jam clip, and (b) damped
the schedule collateral — re-anchoring now pins the global `start_frame_index` to the *earliest*
re-anchor rather than letting each object drag the joint start. Per-track present-frame span
(SAM3 mask-persistence, 1349-frame clip, gate-off baseline vs each threshold):

| track | off | t1.5 | t3.0 | t4.0 | note |
|---|---|---|---|---|---|
| obj 2  | 552  | 1349 | 1349 | 1311 | genuine occlusion recovery — **+797** |
| obj 17 | 105  | 859  | 851  | 862  | genuine occlusion recovery — **+746** |
| obj 3  | 1349 | 1287 | 1349 | 1349 | marginal skip (2.76 px): regresses at 1.5, **clean at ≥3.0** |
| obj 4  | 1265 | 1200 | 1271 | 1265 | collateral at 1.5 only, gone at 3.0 |
| obj 14 | 1346 | 1258 | 1344 | 1346 | collateral at 1.5 only, gone at 3.0 |
| obj 1  | 1026 | 851  | 686  | 602  | **bad re-anchor at every threshold** — see below |
| **net** | — | **+1145** | **+1198** | **+1090** | t3.0 best |

**Verdict: 3.0 px is the win-only regime.** It keeps both genuine occlusion recoveries (obj 2, obj
17) at full strength while every marginal-skip collateral (obj 3/4/14) snaps back to baseline, and
(b) removes the obj-8/obj-14 schedule perturbation seen in the first pass. Best net (**+1198
present-frames, +5.4 %**). Default set to `3.0` in [video.py:1910](../spag4d/video.py).

**obj-1 caveat — RESOLVED 2026-07-27 with a second signal (`SPAG_OCCL_MAX_SKIP`, default 8).**
Root cause, from the per-occurrence FB-err trajectories: obj 1's first-frame FB-err (10.61 px) was
*noisy local flow on a continuously well-tracked object*, not an occlusion — it is detected at 16
dense occurrences (f0..f88, FB-err 10→20→6→4) before its "clean" frame. A genuinely occluded track
is barely detected during the occlusion (obj 17: **zero** detections before it emerges at f432,
then clean within 5 occurrences), so its clean anchor is only a few occurrences in. FB-error alone
can't separate these (obj 1's chosen anchor reads a clean 1.84 px); **the skip count can.** Since
re-anchoring discards every occurrence before the clean one, its cost is exactly those skipped good
frames — so cap it: only re-anchor when the clean anchor is within `SPAG_OCCL_MAX_SKIP` occurrences
of the first. obj 2 (skip 2), obj 6 (skip 2), obj 17 (skip 5) still re-anchor; obj 1 (skip 16) no
longer does.

Validated on the jam clip (thresh 3.0, gate-off baseline → cap 8):

| track | off | no cap | **cap 8** | |
|---|---|---|---|---|
| obj 1  | 1026 | 686  | **1016** | recovered — was the only regression |
| obj 2  | 552  | 1349 | 1349 | genuine occlusion win, kept |
| obj 17 | 105  | 851  | 751  | genuine occlusion win, kept |
| **net** | 22293 | +1198 (+5.4 %) | **+1428 (+6.4 %)** | |

No track now regresses by more than ~10 present-frames (determinism/schedule noise). The gate is a
clean net win with no meaningful downside on the tested clips.

**Status:** implemented, safe (opt-in, inert off), tuned (default 3.0 px FB threshold + max-skip 8),
obj-1 caveat resolved, and confirmed to materially improve tracking through severe occlusion
(**+6.4 % present-frames net** on the jam clip, no regressions). Ready to ship as a committed
opt-in flag.

## Dependency order

```
P1 (flow export)   ── independent, cheapest, highest downstream value ── DO FIRST
P2 (determinism)   ── unblocks P3, P4, occlusion, and any FG change
   ├─ P3 (SAM3_SCALE default-on)   — pure throughput win once gated
   ├─ P4 (conditional seam padding) — gated also on a corpus measurement
   └─ occlusion (spec completion)
```

P1 stands alone and pays off immediately downstream — do it first. P2 is the prerequisite
for trusting every foreground/scale change; do it next. P3 is the low-risk throughput win;
P4 and occlusion follow.

## Explicitly NOT SPAG4D's job (belongs downstream in FreeTimeGS)

- **Per-Gaussian temporal identity / anti-popping** — the entire reason FreeTimeGS's 4D
  representation exists (one Gaussian set + velocity). SPAG4D feeds it depth/mask/flow; it
  does not track Gaussians. (This corrects the old D1 draft, which listed it as SPAG4D P2.)
- **Free-viewpoint parallax from a fixed camera** — physically absent; the realistic target
  (their plan) is *graceful* degradation under small translations, via depth+flow
  supervision, not synthesizing parallax that was never captured.

## Explicitly not on this list (dead ends, confirmed)

- **Background stabilization** — solved (bg-lock, variance ~0). No further work.
- **Robust affine (RANSAC/Siegel)** — no value while bg is locked, not re-estimated.
- **`sol4` multi-frame reference** — confirmed no-op, abandoned.
- **vipe replacement via Open-d4rt** — evaluated, not viable (Open-d4rt/OPEN_D4RT_EVALUATION.md).
- **Full `SPAG_SP_FIX_SEAMGAP`** — correct but slower than baseline (C1).
- **Conditional seam padding (`SPAG_SP_COND_SEAMPAD`)** — P4; confirmed dead end 2026-07-27,
  seam padding is a global (not localizable) WAFT input transform, so no conditional recompute
  recovers the single-pass foreground regression.
