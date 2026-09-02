# SPAG-4D RAM-fix / stride=4 / no-SOR sweep — 2026-09-02

Full 15-clip run of the shipped RAM-fix pipeline (viz_tensor/flow/mask buffer chunking +
`compute_temporal_median`/`nanstd` row-chunking, all unconditional in `spag4d/video.py`,
see `docs/RAM_USAGE_INVESTIGATION_PLAN.md`), at denser Gaussian sampling (`stride=4` vs the
reference sweep's `stride=8`) with SOR outlier pruning disabled. Config and the exact script
used are saved alongside this report (`config.json`, `run_full_clip_postfix_sweep.py`).

## Config (frozen)

- **Pipeline config:** `bglock_sol1_median_w5_livebg_sam3bf16_maxsize1536_ramfix_stride4_nosor`
  — `depth_correction="bglock"`, `depth_smoothing=True`, `depth_smoothing_window=5`,
  `depth_smoothing_method="median"`, `freeze_bg=True`, `freeze_bg_live_color=True`.
- **Generator:** `da360`.
- **Base kwargs:** `skip_step=4, stride=4, temporal_consistency=False, outlier_pruning=0.0,
  grazing_angle=85.0, sparse_pruning=0.1`.
- **Env:** `SPAG_SAM3_BF16=1, SPAG_SAM3_MAXSIZE=1536, SPAG_RAM_TRACE=1`.
- **Difference vs. reference sweep** (`benchmarks/new_bchmk_2026-08-31/bglock_sol1_median_w5_livebg_sam3bf16_maxsize1536`):
  `stride` 8→4 (denser Gaussian sampling) and `outlier_pruning` 0.1→0.0 (SOR disabled).
  SOR (`prune_outliers`) uses a single GLOBAL mean/std of nearest-neighbor distance across the
  whole point cloud; at stride=4's higher point density, far background/sky geometry becomes
  relatively sparser vs. dense nearby geometry than it already was at stride=8, making SOR's
  global threshold more likely to misclassify legitimate far/background points as outliers and
  delete them permanently (matches prior finding: SOR is unsafe under bglock). `grazing_angle`
  (relative depth-gradient, distance-independent) and `sparse_pruning` (per-splat scale-adaptive
  radius, designed to preserve sparse distant backgrounds) are NOT distance-biased the same way
  and were left unchanged. Full rationale in the script's module docstring.
- **Each clip runs in its own subprocess** so `ram_max_mb` (`resource.getrusage` peak RSS) and
  VRAM peak are per-clip, not cumulative across the sweep.

## Metrics

time · peak RAM (RSS) · peak VRAM (`torch.cuda.max_memory_allocated`) · frames analysed ·
`n_tracked_objects` (distinct SAM3 object ids across the whole clip) · mean concurrent tracked
objects/frame · mean splats/frame.

| video | frames | time (s) | RAM (MB) | VRAM (MB) | n_tracked_objs | objs/frame | mean splats |
|---|--:|--:|--:|--:|--:|--:|--:|
| accident_electrique_02 | 94 | 246.1 | 15181 | 6734 | 1 | 1.00 | 291204 |
| atelier_1 | 158 | 401.2 | 23552 | 13245 | 7 | 4.62 | 121617 |
| boutique1_HQ | 150 | 502.0 | 35268 | 18693 | 15 | 13.43 | 134807 |
| CIELE | 204 | 635.1 | 31088 | 11887 | 3 | 2.92 | 286080 |
| circulation_site_1_edit_coupe | 39 | 140.9 | 11669 | 6212 | 6 | 5.08 | 298718 |
| Dispo_RDV | 210 | 527.8 | 31671 | 16331 | 7 | 7.00 | 135540 |
| MattSwift | 150 | 307.2 | 16473 | 9811 | 3 | 3.00 | 82126 |
| productionPont_2_MAX | 163 | 437.7 | 25986 | 10291 | 3 | 3.00 | 271536 |
| projections_yeux_2 | 75 | 222.8 | 14432 | 6874 | 3 | 2.38 | 284263 |
| risque_securite_machine_2 | 173 | 441.4 | 24159 | 9076 | 1 | 1.00 | 283210 |
| scene01 | 158 | 395.3 | 24702 | 13240 | 7 | 6.21 | 127023 |
| scene_03 | 155 | 391.1 | 23893 | 13029 | 7 | 6.30 | 130228 |
| tissc | 105 | 285.6 | 19018 | 8060 | 3 | 2.94 | 296382 |
| trop_long_embouteillage | 338 | 902.8 | 52759 | 29548 | 10 | 7.65 | 72323 |
| vid360_bruit_operatrice | 241 | 739.4 | 36608 | 12066 | 2 | 1.97 | 283044 |
| **MEAN (n=15)** | **160.9** | **438.4** | **25764** | **12340** | **4.6** | | **206540** |
| **MAX** | 338 | 902.8 | 52759 | 29548 | 15 | | |

## Reading the numbers

- **RAM stays well under the pre-fix CIELE reference** (28,301 MB post-fix at the old stride=8
  config, `docs/RAM_USAGE_INVESTIGATION_PLAN.md`) even at stride=4's higher point density — CIELE
  here is 31,088 MB, up from that but still far below the pre-fix 64,053 MB baseline. The RAM
  fixes hold at higher Gaussian density.
- **`trop_long_embouteillage` is both the longest clip (338 frames) and the peak on every cost
  metric** (902.8 s, 52,759 MB RAM, 29,548 MB VRAM) — consistent with it being the known VRAM-floor
  stress case (`docs/D1_FUTURE_WORK_PLAN.md`). No other clip is close on any axis.
- **`boutique1_HQ` has by far the most tracked objects** (15 distinct ids, mean 13.4 concurrent/frame)
  yet its cost (502 s, 35,268 MB RAM, 18,693 MB VRAM) is mid-pack, not a peak — tracked-object count
  alone doesn't predict RAM/VRAM cost here, frame count and scene depth-range do.
- **Single-object clips** (`accident_electrique_02`, `risque_securite_machine_2`) sit at the low end
  of RAM/VRAM as expected — less SAM3 mask/tracking state to hold.
- **No results.json failures** — all 15 clips completed and produced `depth_maps/` npy output.

## Reconstruction consistency (added 2026-09-02, bg/fg split 2026-09-02)

**Depth reprojection consistency** — a self-referential reconstruction-quality metric that
needs no ground truth (see `spag4d/reconstruction_metrics.py`, wired into `ConversionResult`
as `depth_reproj_consistency` whenever `depth_npy_dir` is set): for each frame, flow-warp the
previous frame's depth into the current grid (WAFT flow, edge-aware `warp_backward`) and
compare to the depth actually estimated that frame: `pred[p] = depth_prev[p - flow(p)]`,
`err = |pred - depth|`.

**Split by SAM mask** (added after the first pass conflated two different failure modes —
see the `atelier_1` frame-84 case study below):
- **`bg_*`** — background only (`mask==0`). This is the primary signal: bglock background
  should sit still, so any warp disagreement here is real depth instability, not scene motion.
  `bg_flagged_frames` (>2σ above the clip's own bg mean) is what actually matters.
- **`fg_*`** — foreground only, on the mask's **eroded interior** (9px erosion, boundary
  stripped). Foreground silhouette edges disagree with the warp *by construction* even for a
  perfectly consistent reconstruction (occlusion/dis-occlusion as a moving object's edge
  reveals/hides background) — eroding to the interior isolates "is this object's own depth
  internally coherent as it moves" from that expected edge noise. Informational only, not
  used for flagging — no principled threshold for "how much foreground depth wobble is fine"
  yet.

| video | bg_mean (m) | bg_max (m) | n bg-flagged | fg_mean (m) | fg_max (m) |
|---|--:|--:|--:|--:|--:|
| accident_electrique_02 | 0.0042 | 0.0090 | 8 | 0.2597 | 2.3455 |
| **atelier_1** | **0.0255** | **0.4420** | 6 | 0.4827 | 3.3557 |
| boutique1_HQ | 0.0058 | 0.0200 | 11 | 0.0900 | 1.3880 |
| CIELE | 0.0022 | 0.0057 | 9 | 0.1161 | 0.6472 |
| circulation_site_1_edit_coupe | 0.0103 | 0.0158 | 2 | 0.4398 | 0.6501 |
| Dispo_RDV | 0.0047 | 0.0132 | 9 | 0.0364 | 0.2523 |
| MattSwift | 0.0027 | 0.0043 | 4 | 0.0361 | 0.1114 |
| productionPont_2_MAX | 0.0130 | 0.0328 | 11 | 0.1367 | 0.7053 |
| projections_yeux_2 | 0.0051 | 0.0181 | 3 | 0.2318 | 0.8207 |
| risque_securite_machine_2 | 0.0027 | 0.0036 | 10 | 0.4945 | 3.5493 |
| scene01 | 0.0031 | 0.0210 | 3 | 0.0683 | 0.5530 |
| scene_03 | 0.0057 | 0.0173 | 11 | 0.0570 | 0.2393 |
| tissc | 0.0051 | 0.0239 | 6 | 1.1290 | 3.5171 |
| trop_long_embouteillage | 0.0127 | 0.0479 | 17 | 0.4813 | 2.8936 |
| vid360_bruit_operatrice | 0.0035 | 0.0071 | 14 | 0.4247 | 2.9460 |
| **MEAN (n=15)** | **0.0071** | **0.0454** | | **0.2990** | **1.5983** |

### Reading the numbers

- **`atelier_1` is a genuine background outlier, not just foreground edge noise.** Its
  `bg_max_mean_err` (0.442 m) is ~9x the next-worst clip's *background* score
  (`trop_long_embouteillage` at 0.048 m), and the flagged frames (82-87) are unchanged from
  before the bg/fg split. The case-study visualization (frame 84: forklift driving through the
  scene) showed the raw error concentrated almost entirely on the forklift's own silhouette —
  the working hypothesis was that this was purely expected occlusion-edge disagreement, which
  the fg/bg split was built to test. **That hypothesis is wrong**: masking the forklift out
  (even with a 9px erosion margin) still leaves a large background-classified spike. The most
  likely explanation is a **SAM mask miss** — the forklift's cast shadow, a reflection off the
  epoxy floor, or a thin sliver of the object silhouette (antenna, forks, mirror) not fully
  covered by the tracked mask — leaking real object motion into what the metric (and the
  pipeline's own bglock compositing) treats as static background. Worth pulling the SAM mask
  for frames 82-87 and overlaying it on the RGB to check for exactly this.
- **`accident_electrique_02` confirms the originally-reported bug**: frames 10-13 (and a
  smaller bump at 61/64/65) are bg-flagged, matching the "successive frames between 0 and 20
  look bad" report — and now isolated to background specifically, ruling out "it's just the
  moving person" as an explanation.
- **`fg_mean_err` is an order of magnitude noisier than `bg_mean_err` across every clip**
  (mean 0.299 m vs 0.071 m) — expected, since even a well-tracked moving object's depth is
  inherently harder for a single flow-warp to predict than static geometry. `tissc` and
  `risque_securite_machine_2` have the highest `fg_max` (3.5 m) despite unremarkable
  backgrounds — worth a look if foreground quality specifically matters downstream, but not
  comparable to the bg severity scale above.
- **`trop_long_embouteillage` has the most bg-flagged frames (17)** but a middling
  `bg_max_mean_err` (0.048 m) — consistent with it being the longest clip (338 frames), so
  more opportunities for isolated disagreement rather than one severe event like `atelier_1`.

## Reproduce

```bash
conda activate spag4d
export SPAG_SAM3_BF16=1 SPAG_SAM3_MAXSIZE=1536 SPAG_RAM_TRACE=1
python scripts/run_full_clip_postfix_sweep.py   # all 15 clips, one subprocess each
```

The exact script used for this run is saved alongside this report:
[`run_full_clip_postfix_sweep.py`](run_full_clip_postfix_sweep.py), and the exact
kwargs/env/video-list are in [`config.json`](config.json).

The 15 clips: vid360_bruit_operatrice, CIELE, trop_long_embouteillage, productionPont_2_MAX,
risque_securite_machine_2, Dispo_RDV, boutique1_HQ, tissc, scene01, scene_03, atelier_1,
accident_electrique_02, projections_yeux_2, MattSwift, circulation_site_1_edit_coupe
(from `/raid/mb273924/_DATASETS/uptale/data/videos`).
