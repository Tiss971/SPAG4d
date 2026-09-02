# Rules for running/comparing SPAG-4D benchmarks

Established the hard way across the B1 VRAM-lever work (2026-07-31). Follow these for any
new benchmark comparison in this folder, not just VRAM levers.

## 1. Never compare metrics across separate runs/GPUs — only matched pairs

`SPAG_DETERMINISTIC=1` gives bit-for-bit reproducibility **within the same GPU**, but
different GPU hardware can diverge (`fg_depth_cv` seen at 0.134 vs 0.28 for the identical
config/seed on two different GPUs). Consequence:

- **Always run baseline and the config-under-test back-to-back, same process, same GPU**,
  and diff *that* pair. Never diff a new run's numbers against an old JSON file's numbers
  (e.g. `baseline_2026-07-27/baseline_2026-07-27.json`) as if they were controlled — that file is a reference
  point for eyeballing magnitude, not a rigorous comparison baseline.
- If a result looks surprising (e.g. a regression reappearing that was previously fixed),
  the first suspect is cross-GPU noise — rule it out by re-running both configs pinned to
  the *same* GPU (`CUDA_VISIBLE_DEVICES=<n>`) before concluding the effect is real. This
  caught a real bf16+maxsize interaction effect only after a same-GPU re-run reproduced
  the identical IoU (0.3487, bit-for-bit) twice.

## 2. Aggregate metrics can hide localized failures — always add mask IoU

`bg_depth_cv`/`fg_depth_cv`/spike-rate are frame-averaged over the whole clip. A lever that
wrecks segmentation on a handful of frames (e.g. a thin/articulated-subject clip losing legs
for 150/150 frames) can still post a fine aggregate `fg_depth_cv` if the rest of the clip is
stable. **Any lever touching SAM3/masking must be validated with per-frame mask IoU**
against the matched baseline, not aggregate depth-CV metrics alone:

```python
b = np.load(f"{base_dir}/mask_{i}.npy") > 0
m = np.load(f"{new_dir}/mask_{i}.npy") > 0
iou = np.logical_and(b, m).sum() / np.logical_or(b, m).sum()
```

Report `mean_iou`, `min_iou`, and `frames_iou_below_0.5` (count, not just the mean) — a
lever that's fine on average but has even one clip with many frames <0.5 IoU is not safe to
default on. Requires the run to be launched with `depth_npy_dir=<dir>` so `mask_{idx}.npy`
(fused SAM mask, uint8) and `depth_{idx}.npy` (float32 meters) get dumped per frame.

## 3. VRAM/quality targets are gated on the max across clips, not the mean

A lever's mean VRAM/time win across the 10-clip set is not sufficient — check every
individual clip against the target. The 10-clip set spans 39–1349 frames and 1920×960–
3840×1920 native resolution; a lever that's clearly a win on the mean can still leave an
outlier (typically the longest or highest-res clip) over target. See
`project_vram_target_16gb` memory: bf16+maxsize1536 won on 9/10 clips but left the longest
clip (`trop_long_embouteillage`, 1349 frames) over the 16GB target — a length-driven
outlier that the mean completely masked.

## 4. Persist config + comparison-baseline metadata in every result JSON

Every benchmark JSON should carry a `config` block recording exactly what was run, not just
the numbers, so a JSON found later is self-describing:

```json
{
  "config": {
    "depth_correction": "bglock", "depth_smoothing": true,
    "depth_smoothing_window": 5, "depth_smoothing_method": "median",
    "generator": "da360",
    "env": {"SPAG_DETERMINISTIC": 1, "SPAG_OCCL_FBGATE": 1},
    "commit_base": "<git sha>",
    "compared_against": "<the file/run this was diffed against, or 'matched same-session pair'>"
  },
  "videos": { "<clip>": { "baseline_matched": {...}, "stacked": {...}, "mask_iou": {...} } }
}
```

## 5. Always capture object-track count

`n_objects` (how many moving objects SAM3 ends up tracking) is a real VRAM/quality
covariate — see the VRAM-prediction-model section in `.claude/B1_VRAM_TIME_REDUCTION_PLAN.md`
(~2.6 GB per additional tracked object) and the `SPAG_SAM3_SCALE`/`SPAG_SAM3_MAXSIZE`
findings (mask regressions were content-dependent, not resolution-dependent, and object
count is one of the few available signals for that). It is **not** returned by `run_video()`
or `ConversionResult` — it only appears in stdout:

```
--- Propagating {found_object} moving object starting at {start_frame_index} ---
```

(`spag4d/video.py`, `segment_with_flows`). Capture it by tee-ing/redirecting each run's
stdout to a log file and parsing that line (`grep "Propagating.*moving object"`), then
persist it as `n_objects` alongside the other per-clip fields in the result JSON. Don't
skip this even when it feels like just a curiosity — it's the field most likely to explain
why an otherwise-safe lever fails on one specific clip.

## 6. `run_video()` result object — correct field extraction

`res` has no `.n_frames`/`.mean_splats` attributes. Use:

```python
n_frames = len(res.splat_count) if res.splat_count else 0
mean_splats = float(np.mean(res.splat_count)) if res.splat_count else None
stab = calculate_temporal_stability(res.depth_metrics) if res.depth_metrics else None
```

(`calculate_temporal_stability`, `BASE_KWARGS` from `benchmark_solutions.py` — reuse them,
don't reimplement.)

## 7. GPU load-balancing across the 10-clip set

Real GPUs available: indices 0, 1, 2, 4 (index 3 is a small 4GB display device — avoid it).
Clip runtimes vary ~4x (39 to 1349 frames); split into 4 groups with LPT
(longest-processing-time-first greedy bin-packing) using recorded `time_seconds` from
`baseline_2026-07-27/baseline_2026-07-27.json`, not an even split by clip count, to keep wall-clock roughly even
across groups. Launch each group as a background job, then wait on all four via a single
polling loop that greps for a sentinel string (`^EXIT`/`^FINAL:`) in each log — don't use a
tool that treats every log line as an event.

## 8. Visual sanity check

For any lever that touches mask/depth quality, render a colored-depth video
(`scripts/render_depth_colormap.py <npy_dir> <out.mp4>`) for at least the clips flagged by
mask IoU — aggregate numbers can look plausible while a mask-contour overlay makes a
lost-limb or wrong-object failure obvious at a glance.

## 9. The 10-clip reference set

CIELE, Dispo_RDV, MattSwift, accident_electrique_02, atelier_1, boutique1_HQ,
circulation_site_1_edit_coupe, scene01, scene_03, trop_long_embouteillage
(`/raid/mb273924/_DATASETS/uptale/data/videos`). Chosen to span resolution (1920×960 to
3840×1920), frame count (39–1349), and subject type (thin/articulated to blocky/static) —
don't narrow the set for a new lever unless you have a specific reason, since several real
regressions in this set only showed up on 1-2 of the 10 clips (MattSwift/atelier_1 for
thin-limb loss, `trop_long_embouteillage` for length-driven VRAM).

---

# Addendum — fast-iteration rules for the bglock audit (2026-08-17)

Added for the `bglock_open_questions.md` audit session, whose needs differ from the VRAM-lever
work above: many short iterations on algorithm behaviour, not a few long runs on a shipping
decision. Rules 1-9 still apply; these narrow them.

## 10. The fast-loop config, and the one flag combination to never use

Iterating quickly is only useful if the cheap config doesn't itself break segmentation or flow —
otherwise every measurement is confounded by a mask failure. The validated fast config is:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> \
  SPAG_SAM3_MAXSIZE=768 SPAG_DETERMINISTIC=1 SPAG_OCCL_FBGATE=1 \
  /home/mb273924/miniforge3/envs/spag4d/bin/python \
  scripts/run_bglock_audit.py <clip.mp4> <out_dir>
```

**`SPAG_SAM3_MAXSIZE=768`, not 1536** — before trusting any recorded maxsize number, know that the
flag applied its factor to the WAFT-shrunk shape rather than native
(fixed 2026-08-17), so every "maxsize=1536" result in the B1 doc was actually measured at
**614-818 px effective**, and its "target res" column is the buggy composed value. Post-fix,
`768` reproduces the validated-safe operating point uniformly across clips (MattSwift's IoU-0.93
config was exactly 768 px effective); `1536` is a much weaker downscale than anything validated
and will cost far more VRAM than any recorded maxsize figure.

Use it **alone, with `SPAG_SAM3_BF16` OFF.** From
`.claude/B1_VRAM_TIME_REDUCTION_PLAN.md` and `b1_vram_2026-07-31/bf16_maxsize1536_2026-07-31.json`:

| config (as recorded — see res caveat above) | MattSwift mask IoU | atelier_1 mask IoU |
|---|---|---|
| maxsize alone, fp32 (eff. 768 / 614 px) | **0.93** | **0.87** |
| `SPAG_SAM3_BF16` alone (native res) | **0.99** | — |
| `SPAG_SAM3_SCALE=0.5` alone (eff. 512 px) | 0.34 | 0.38 |
| **bf16 + maxsize stacked** (eff. 768 / 614 px) | **0.35** (150/150 frames <0.5) | **0.40** (105/158 <0.5) |

**Each flag is safe solo; the stack is not.** Confirmed deterministic — the stacked MattSwift IoU
reproduced bit-for-bit (0.3487) across two same-GPU runs, so it is not the cross-GPU noise of
rule 1. Mechanism: bf16's weight-precision loss removes just enough margin on an already-
downscaled 768×384 input to lose thin limbs. Do not stack them for speed, and never use
`SPAG_SAM3_SCALE` (it anchors off the already-WAFT-shrunk shape, giving a real 0.2-0.25× factor
rather than the 0.5 it advertises).

**2026-08-18 defaults + isolated VRAM check (single-clip, MattSwift, same-GPU pair):** `SPAG_SAM3_BF16`
default flipped on, `SPAG_SAM3_MAXSIZE` default flipped off (0/native) so the two stay solo by
default — see CLAUDE.md. This also resolved a live mystery: `benchmarks/rebenchmark_2026-08-18/`
showed a ~46% mean VRAM drop vs `baseline_2026-07-27` under the *old* defaults (maxsize=1536,
bf16 off then), initially suspected to be bf16 (it wasn't wired into any default at the time).
Isolated re-test on MattSwift alone: baseline 18,080 MB → bf16-alone-native 16,468 MB (−8.9%) →
**maxsize=1536-alone (bf16 off) 12,377 MB (−31.5%)**, closely matching the rebenchmark's 13,883 MB
for the same combo (residual gap = normal cross-GPU noise, rule 1). **The maxsize downscale, not
bf16, was responsible for nearly all of the rebenchmark's VRAM drop.** Net: maxsize=1536-alone
saves more VRAM than bf16-alone-native (12.4GB vs 16.5GB) at a real fidelity cost (IoU 0.93 vs
0.99, table above) — an open choice between the two as the shipped default, not yet decided;
today's default keeps bf16 (fidelity-first). Revisit once VRAM pressure (`project_vram_target_16gb`
memory: max across clips must stay <16GB) is checked against more than one clip.

## 11. Never quote a stability number without its paired fidelity number

This is `bglock_open_questions.md` §6.0 promoted to a benchmark rule, because it invalidates
everything else if missed. `output = D_ref` for every pixel of every frame scores *perfectly* on
every temporal-stability metric in this repo — background `bg_depth_cv` of 0.0000 under bglock is
exactly that, true by construction and not evidence about the algorithm's judgement. So:

- Every reported stability delta must come with a **dynamic-region temporal-derivative**
  comparison against the raw per-frame estimate. Flatter-than-raw means motion was suppressed,
  not that stability improved.
- `bg_depth_cv` is meaningless as a *ranking* signal between bglock variants (all are ~0 by
  construction). It is only useful as a regression tripwire — if it moves off ~0, something broke.
- Rule 2's mask IoU is the fidelity floor for anything touching masks; the derivative check is the
  fidelity floor for anything touching the propagation/confidence/composite chain.

Related, from `project_vipe_metric_scale_confound` memory: consistency-style metrics have already
inverted a verdict twice on the vipe side. Treat any metric that a freeze would ace as
non-decisional.

## 12. Verify the premise in the source before benchmarking it

`bglock_open_questions.md` §1, worth repeating here because it saved a full GPU cycle in this
session: of the four P0 items, **two premises were false on inspection** (§4.1 confidence
absorbing at zero — a per-frame reset already prevents it; §4.4 PaGeR per-frame CLIP routing —
`metric=False` short-circuits, and `da360` is the production generator). Neither needed a run.
See `bglock_open_questions.md` §4.1 and §10.4. Read the code path end-to-end first; a benchmark that
confirms a non-existent bug is pure cost.

The corollary is not "skip the run": reading §4.1's path closely enough to disprove it surfaced a
*different*, real defect one line below (the reset that prevented the absorbing state also made
`decay` inert), which then did earn a benchmark — `confdecay_2026-08-17/`. Disproving the stated
premise is where the useful finding starts, not where it ends.


## 16. Every arm behind the same env-var seam, including the "raw" reference

`confdecay_2026-08-17` got its no-propagation fidelity reference for free by pushing the existing
knobs to a degenerate setting (`SPAG_CONF_DECAY=0 SPAG_CONF_FLOOR=0` → confidence ≡ 0 → the fresh
per-frame estimate) rather than adding an ablation code path. Same binary, same call graph, one
env var — so the reference arm cannot drift from the arms it is scoring, and rule 11's fidelity
denominator costs a run instead of a refactor. Look for that degenerate setting before writing an
ablation branch. Pair it with a `*_LEGACY=1` arm that restores the pre-change behaviour and verify
it reproduces a pre-change trace digit-for-digit; that is what makes older recorded numbers still
reproducible after a default flips.

## 13. Instrumentation goes behind an env var, inert when unset

Pattern used by `SPAG_CONF_HIST`, `SPAG_OCCL_MASKDUMP`, `SPAG_OCCL_DEBUG`, `depth_npy_dir`: the
diagnostic writes nothing and allocates nothing unless its var is set, so the production path
stays byte-identical and the same build serves both. Verify inertness explicitly (a matched run
with the var unset must reproduce the baseline exactly) — an instrumentation-induced delta is
indistinguishable from the effect you are measuring.

## 14. Pin the interpreter

The repo's deps live in the `spag4d` conda env
(`/home/mb273924/miniforge3/envs/spag4d/bin/python`; cv2 4.13, torch 2.11+cu128). A bare `python`
resolves to miniforge `base`, which has no `cv2` — the run dies at import after the GPU is already
claimed. Invoke the env's interpreter by absolute path in every launcher rather than relying on an
activated shell, since background jobs don't inherit one.

## 15. Short clips for the loop, but the canary comes along

`circulation_site_1_edit_coupe` (156 frames) is the cheapest clip in the set and the natural loop
target. Pair every loop iteration with **`MattSwift`** (600 frames, thin/articulated subject):
it is the clip that caught the `SPAG_SAM3_SCALE` leg-crop, the bf16+maxsize interaction, and the
`fg_depth_cv` determinism swing. A change that looks clean on the short clip alone has not been
tested against the failure mode this repo actually has. Full 10-clip runs stay reserved for
ship decisions (rule 3).

## 17. Appearance-only changes need their own canary pair, and their own metric

Rules 2/8/11 cover geometry fidelity (mask IoU, visual overlay, dynamic-region derivative), but
none of them would catch a background-*color* regression — a lever that freezes a Gaussian's
color as well as its position can post a perfect `bg_depth_cv` and a clean mask IoU while quietly
making shadows/screens dead. Caught this validating `freeze_bg_live_color` (2026-08-21, see
CLAUDE.md "Changed 2026-08-21"): use a matched clip pair chosen for appearance-vs-geometry
divergence specifically —

- `circulation_site_1_edit_coupe` — moving shadows on an otherwise-static floor (appearance
  changes, geometry doesn't).
- `boutique1_HQ` — flat screens, some SAM-tracked as dynamic, some not (the untracked ones are
  exactly the case a background-locking lever can accidentally freeze).

Neither depth-CV nor mask IoU says anything about color, so the check has to read the PLY color
channel directly: dump `_freeze_bg_meta.json`'s `n_bg_points` to slice just the frozen-background
block (it's always the first N points when `freeze_bg` concatenates it), then diff that block's
color between two frames far apart in the clip. A working live-color lever shows nonzero,
growing `mean|Δ|` there; a fully-frozen lever shows exactly `0.0`, and a lever that only *claims*
to update color but doesn't wire the mask/pixel-index correctly will silently show `0.0` too —
this is the way a broken implementation gets caught, since it can't be told apart from "working"
by geometry metrics alone:

```python
n_bg = json.load(open(f"{arm_dir}/gaussians/_freeze_bg_meta.json"))["n_bg_points"]
c0  = load_ply_colors(f"{arm_dir}/gaussians/frame_0.ply")[:n_bg]
cN  = load_ply_colors(f"{arm_dir}/gaussians/frame_N.ply")[:n_bg]
mean_abs_delta = np.abs(c0 - cN).mean()  # ~0 = frozen (bug if the lever claims otherwise); >0 growing = live
```
