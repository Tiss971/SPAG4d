# scripts/

One-off analysis, benchmark, and tooling scripts. Each file's own docstring
is the authoritative description — this table is a navigation index, not a
replacement. Read the script before running it; most take `--help`.

## Regression-test tooling

| Script | Purpose |
|---|---|
| `capture_golden_fixtures.py` | Regenerates the golden-fixture `.npy` arrays and `manifest.json` for `tests/test_regression_golden.py`. Run locally whenever fixtures are missing or stale — see `tests/README.md`. |
| `make_fixture_clips.py` | Builds the trimmed source clips (`scene01_trim25.mp4`, `circulation_trim25.mp4`) that the golden fixtures are captured from. |

## Stability / benchmark analysis

| Script | Purpose |
|---|---|
| `analyze_flow_latitude.py` | Analyzes optical-flow behavior as a function of ERP latitude. |
| `check_camera_static.py` | Checks whether a clip's camera is genuinely static (fixed-camera assumption). |
| `compare_conf_arms.py` | Compares confidence-decay benchmark arms (see `SPAG_CONF_DECAY`/`SPAG_CONF_LEGACY`). |
| `compute_mask_iou.py` | Computes IoU between two mask sets (e.g. SAM3 bf16 vs fp32 regression checks). |
| `depth_reprojection_consistency.py` | Reconstruction-consistency metric (bg/fg split), added in commit `73e2a0f`. |
| `eval_bg_point_stability.py` | Evaluates background point-cloud stability across frames. |
| `eval_fg_point_stability.py` | Evaluates foreground point-cloud stability across frames. |
| `export_flow_only.py` | Exports per-frame WAFT flow (`flow_{idx}.npy`) without a full pipeline run — recovery path for datasets missing flow, see project memory. |
| `feather_hardcutover_test.py` | Tests bg-lock feather vs. `SPAG_HARD_DEPTH_CUTOVER` compositing. |
| `fp16_quantization_floor_test.py` | Tests fp16 quantization floor effects on depth output. |
| `mask_injection_analysis.py` | Mask-injection RMSE ground-truth analysis, see `docs/bglock_open_questions.md` §6.3. |
| `render_depth_colormap.py` | Renders a depth map as a colormap image/video for visual inspection. |
| `render_fg_depth_flicker.py` | Renders foreground-depth flicker visualization. |
| `render_indepscale_fg_flicker.py` | Renders foreground-depth flicker at independent per-frame scale. |
| `render_translated_rgb.py` | Renders RGB translated by estimated motion, for visual QA. |
| `residual_vs_latitude_test.py` | Tests residual depth error as a function of ERP latitude. |
| `run_atelier1_bglock_sol1.py` | Runs the `bglock_sol1_median_w5` solution benchmark arm. |
| `run_bglock_audit.py` | Runs the bg-lock audit checklist against a clip. |
| `run_bg_stability_bench.py` | Runs the background-stability benchmark suite. |
| `run_flowfix_trace.py` | Traces the single-pass seam-fix flow correction (`SPAG_SP_FIX_SEAMGAP`/`SPAG_SP_COND_SEAMPAD` investigation). |
| `run_full_clip_postfix_sweep.py` | Sweeps postfix parameters over a full clip. |
| `run_pager_accident_bench.py` | Runs the pager-generator benchmark on the `accident_electrique_02` hallucination case, see `.claude/E1_PAGER_TEMPORAL_MVD_PLAN.md`. |
| `run_ram_decompose.py` | Decomposes RAM usage across pipeline phases. |
| `run_scene01_bg_stability_bench.py` | Runs the background-stability benchmark suite on `scene01`. |
| `run_segmentation_only.py` | Runs only the SAM3 segmentation/tracking stage, without full reconstruction. |
| `stride_invariance_compare.py` | Compares outputs across different frame-stride settings. |
| `synth_lateral_motion_test.py` | Synthetic test: lateral camera/object motion. |
| `synth_radial_lateral_motion_test.py` | Synthetic test: combined radial + lateral motion. |
| `synth_radial_motion_floor_sweep.py` | Synthetic test: radial motion, sweeping floor-depth parameters. |
| `synth_radial_motion_test.py` | Synthetic test: radial camera/object motion. |
| `view_depth_sequence.py` | Interactive viewer for a depth-map sequence. |

## Reporting / misc

| Script | Purpose |
|---|---|
| `build_comparison_html.py` | Builds an HTML comparison page from benchmark results. |
| `run.py` | Generic pipeline runner entry point. |

## Windows / external-tool setup

| Script | Purpose |
|---|---|
| `pager_smoke.py` | Phase-1 smoke test for the PaGeR backend, native Windows. |
| `patch_unisharp_no_render.py` | Idempotent patcher adding a `--no-render` flag to UniSHARP's `infer_unisharp.py`. |
| `setup_unisharp_windows.bat` | Windows batch setup script for UniSHARP. |
| `reconvert_lockactivity.sh` | Reconverts clips under `SPAG_LOCK_ACTIVITY`. |

## External-dependency scripts (unverifiable from this repo)

These two depend on the external `OmniRoam` repo and a `conda run -n omniroam`
environment that this repo doesn't control or vendor. Their liveness can't be
confirmed by reading this repo alone — do not assume dead, do not delete
without checking the OmniRoam side first.

| Script | Purpose |
|---|---|
| `regenerate_trajectory_snapshots.sh` | Regenerates OmniRoam trajectory-snapshot test fixtures via WSL + `conda run -n omniroam`. |
| `setup_omniroam_wsl.sh` | Sets up OmniRoam + its conda env in WSL2. |

## Root-level scripts (outside `scripts/`)

| Script | Purpose |
|---|---|
| `benchmark_solutions.py` | Benchmarks temporal-stability solutions vs. baseline (`bg_depth_cv`, `fg_depth_cv`, etc.). Output defaults to `./benchmark_solutions/<config_name>/`. |
| `report_solutions.py` | Combines per-config `results.json` outputs from `benchmark_solutions.py` into a comparison table. |
| `batch.sh` | (removed in commit `73e2a0f` — no longer present.) |
