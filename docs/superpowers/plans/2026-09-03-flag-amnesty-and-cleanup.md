# Flag Amnesty and Script/Doc Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the ~45 `SPAG_*` env flags in `spag4d/video.py` and `spag4d/flow_depth_propagation.py`, delete the confirmed-dead ones, document the survivors, then clean up script/doc drift: index all root/`scripts/` tooling, fix the pager-plan status wording, and fix `report_solutions.py`'s stale `--dir` default.

**Architecture:** Two independent tracks gated by the existing golden regression harness (`tests/test_regression_golden.py`, 11 tests, ~162s). Track A (flag amnesty) is a sequence of small, individually-gated code deletions/renames plus one new reference doc. Track B (script/doc cleanup) is pure documentation — no code deletions, since research found no confirmed-dead scripts, only two externally-dependent ones (OmniRoam) that are unverifiable from this repo and therefore get documented as such, not removed.

**Tech Stack:** Python 3.12, pytest, existing `spag4d/` codebase. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-03-spag4d-codebase-health-design.md` — sections 3 (Flag amnesty) and 4 (Script/doc cleanup) only. Section 2 (`run_video()` extraction) is out of scope for this plan.

## Global Constraints

- Every code-touching task (Tasks 2-5) must run `tests/test_regression_golden.py` **before and after** its change and confirm 11/11 pass both times, per the spec's harness-gating requirement.
- Test invocation (from repo root, per `tests/README.md`): `/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v`
- No script or doc file gets deleted in this plan. Research (2026-09-03) found zero confirmed-dead scripts — only two OmniRoam-dependent shell scripts (`scripts/regenerate_trajectory_snapshots.sh`, `scripts/setup_omniroam_wsl.sh`) that are unverifiable from this repo alone; they get an explicit "external dependency, unverifiable" note in the new `scripts/README.md`, not removal.
- Flag deletions in this plan are limited to the three the research pass actually confirmed dead: `SPAG_OCCL_MASKDUMP`, `SPAG_TRACK_DEDUP_DEBUG`, `SPAG_SP_FIX_SEAMGAP`. Two flags the original spec named as deletion candidates — `SPAG_SP_COND_SEAMPAD` and `SPAG_CONF_LEGACY` — are explicitly **kept and documented as live/required** per the user's 2026-09-03 ruling (research found `SPAG_SP_COND_SEAMPAD` is the still-open seam-pad approach, and `SPAG_CONF_LEGACY` is required by `docs/bglock_open_questions.md` and CLAUDE.md to reproduce pre-2026-08-17 benchmark numbers).
- `SPAG_SAM3_SCALE_DISABLE` is a naming bug (code reads that name; every comment/doc refers to `SPAG_SAM3_SCALE`). Per the user's 2026-09-03 ruling, this plan renames the code to match the docs, not the reverse.
- This plan supersedes commit `73e2a0f`'s "pending doc/script cleanup" WIP note, per the user's 2026-09-03 confirmation — no separate follow-up commit is needed for that note.

---

### Task 1: `docs/SPAG_ENV_FLAGS.md` reference table

**Files:**
- Create: `docs/SPAG_ENV_FLAGS.md`

**Interfaces:**
- Consumes: nothing (pure documentation, first task).
- Produces: the flag inventory that Tasks 2-5 update in place as they delete/rename flags — later tasks edit this same file's rows rather than creating a new doc.

- [ ] **Step 1: Write the full flag table**

Create `docs/SPAG_ENV_FLAGS.md` with this exact content:

```markdown
# SPAG_* Environment Flags Reference

Audited 2026-09-03. Every `SPAG_*` flag read by `spag4d/video.py` or
`spag4d/flow_depth_propagation.py`, its default, and its status. When a
flag's default value changes, or a flag is added/removed, update this table
in the same commit.

Status legend:
- **default-on** — shipped, on by default, no action needed to use it.
- **opt-in** — shipped, off by default, safe to enable per the linked doc.
- **debug** — developer-only diagnostic output/dump, no functional effect
  on pipeline output.
- **required-for-repro** — off by default, but needed to reproduce a
  specific historical benchmark; do not delete even though "dead" by usage.
- **experimental-open** — an unfinished investigation track, not yet
  resolved as dead or shippable; do not delete.

## spag4d/flow_depth_propagation.py

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_FLOW_EDGE_NEAREST` | `0` | opt-in | Nearest-neighbor edge handling in flow-based depth propagation. |
| `SPAG_FLOW_EDGE_CONF_PENALTY` | `0.0` | opt-in | Confidence penalty applied at flow edges. |
| `SPAG_FLOW_EDGE_ZERO_CONF` | `0` | opt-in | Zeroes confidence at edges. |
| `SPAG_FLOW_EDGE_THRESH` | `0.15` | opt-in | Relative threshold for edge detection. |

## spag4d/video.py

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_BGLOCK_DILATE_PX` | `12` | opt-in | Overrides bg-lock dilation px, only applied if changed from default. |
| `SPAG_BGLOCK_FEATHER_PX` | `9` | opt-in | Overrides bg-lock feather px, only applied if changed from default. |
| `SPAG_DIAG_CSV` | unset | opt-in | Path to write a diagnostics CSV. See `docs/bglock_open_questions.md` §9. |
| `SPAG_DETERMINISTIC` | `0` | default-on for tests | Enables deterministic seeding; forced to `1` by `tests/conftest.py` for the golden regression harness. |
| `SPAG_SEED` | `42` | opt-in | Seed value used when `SPAG_DETERMINISTIC=1`. |
| `SPAG_RAM_TRACE` | `0` | debug | Enables RAM tracing. |
| `SPAG_SINGLE_PASS` | `0` | opt-in | Single-pass WAFT flow (skip second pass). Degrades foreground depth on seam-crossing clips — see `docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md`. Not default. |
| `SPAG_SP_SEAMPAD` | `0` | opt-in | Seam pad amount used with `SPAG_SINGLE_PASS`. |
| `SPAG_SP_COND_SEAMPAD` | `0` | experimental-open | Conditional seam-padding variant for single-pass mode. Still the only unexplored avenue for fixing single-pass's seam regression per `docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md` and CLAUDE.md — **not dead, do not delete.** |
| `SPAG_DREF_MIN_SAMPLES` | `0` | opt-in | Min sample threshold for depth-reference. See `docs/bglock_open_questions.md` §5. |
| `SPAG_BGLOCK_NOFLOW` | `0` | opt-in | Disables flow use in bg-lock. |
| `SPAG_BGLOCK_NOFLOW_BLEND` | `0.5` | opt-in | Blend factor for no-flow bg-lock mode. |
| `SPAG_CONF_DECAY` | `0.85` | default-on | Per-pixel confidence decay rate, compounding over a warped age field. Shipped 2026-08-17. |
| `SPAG_CONF_FLOOR` | `0.5` | default-on | Floor for decayed confidence. |
| `SPAG_CONF_LEGACY` | `0` | required-for-repro | Restores the pre-2026-08-17 non-compounding confidence-decay behavior. Required to reproduce any benchmark recorded before 2026-08-17 — see CLAUDE.md and `docs/bglock_open_questions.md` §4.1/§4.2. **Not dead, do not delete.** |
| `SPAG_CONF_HIST` | unset | debug | Path to write confidence-history data. |
| `SPAG_COMPOSITE_LOWRES` | `0` | opt-in | Low-res compositing path. |
| `SPAG_GPU_COMPOSITE` | `1` | default-on | GPU-resident compositing. |
| `SPAG_HARD_DEPTH_CUTOVER` | `0` | opt-in | Hard cutover mode for depth compositing. See `docs/bglock_open_questions.md` §8.1. |
| `SPAG_LOCK_ACTIVITY` | `1` | default-on | Bg-lock composite keys off `sam_mask` alone, not the fused SAM|activity mask. Shipped 2026-07-28, see project CLAUDE.md. |
| `SPAG_MASK_INJECT` | unset | opt-in | Optional mask injection override path. See `docs/bglock_open_questions.md` §6.3. |
| `SPAG_SAM3_SCALE` | `1.0` | opt-in | Downscale factor for the SAM3 segmentation pass input. Renamed from `SPAG_SAM3_SCALE_DISABLE` in this plan (Task 5) to match its actual usage everywhere else in code comments and docs. Do not combine with `SPAG_SAM3_BF16` — confirmed-deterministic regression, see project CLAUDE.md. |
| `SPAG_SAM3_MAXSIZE` | `0` (disabled) | default-off | Absolute cap on SAM3 input's longer edge. Flipped `1536→0` 2026-08-18 to keep `SPAG_SAM3_BF16` solo — see project CLAUDE.md. |
| `SPAG_TRACK_DEDUP_GAP` | `40` | opt-in | Track-dedup temporal gap threshold. |
| `SPAG_TRACK_DEDUP_OVERLAP` | `0.6` | opt-in | Track-dedup overlap threshold. |
| `SPAG_TRACK_DEDUP_SIZE_RATIO` | `0.3` | opt-in | Track-dedup size-ratio threshold. |
| `SPAG_TRACK_DEDUP_MIN_COOCCUR` | `2` | opt-in | Min co-occurrence count for track dedup. |
| `SPAG_TRACK_DEDUP_IDX_TOL` | `3` | opt-in | Frame-index tolerance for track dedup matching. |
| `SPAG_TRACK_REID_MAX_GAP` | `40` | opt-in | Max frame gap for track re-identification via anchor proximity. |
| `SPAG_OCCL_FBGATE` | `0` | opt-in | Forward-backward consistency gate for occlusion handling. |
| `SPAG_OCCL_FB_THRESH` | `3.0` | opt-in | FB-consistency threshold for occlusion gate. |
| `SPAG_OCCL_MAX_SKIP` | `8` | opt-in | Max frame skip for occlusion handling. |
| `SPAG_SAM3_MAX_FRAMES` | unset | opt-in | Optional cap on SAM3 frame batch size. |

## Deleted flags (removed by this plan, Task 2-4)

| Flag | Was | Removed because |
|---|---|---|
| `SPAG_OCCL_MASKDUMP` | debug | Pure debug/dump utility, no doc references it as load-bearing. |
| `SPAG_TRACK_DEDUP_DEBUG` | debug | Pure debug print flag, no doc references it as load-bearing. |
| `SPAG_SP_FIX_SEAMGAP` | opt-in | Confirmed dead end 2026-07/08 — correct but +15% slower than baseline, erasing `SPAG_SINGLE_PASS`'s whole time win. See `docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md`. |
```

- [ ] **Step 2: Commit**

```bash
git add docs/SPAG_ENV_FLAGS.md
git commit -m "docs: add SPAG_* env flag reference table"
```

---

### Task 2: Delete `SPAG_OCCL_MASKDUMP`

**Files:**
- Modify: `spag4d/video.py:3165-3178` (and any further lines of the same block, see Step 2)
- Modify: `docs/SPAG_ENV_FLAGS.md` (already lists it as deleted from Task 1 — no change needed here)

**Interfaces:**
- Consumes: `tests/test_regression_golden.py` (existing, run before and after).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Run the harness before touching anything**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`.

- [ ] **Step 2: Read the full dead block and delete it**

Read `spag4d/video.py` starting at line 3165 (the comment `# Opt-in per-object mask-persistence dump (SPAG_OCCL_MASKDUMP=<path.json>):`) through to the end of the `if _maskdump:` block (find the matching dedent — the block writes `summary` to a JSON file at `_maskdump` and closes with a `print` statement; read to that point with `sed -n '3160,3210p' spag4d/video.py` to find the exact closing line before deleting, since the excerpt available during planning was truncated at line 3178).

Delete the entire block: the leading comment lines, the `_maskdump = os.environ.get("SPAG_OCCL_MASKDUMP")` assignment, and the full `if _maskdump:` body.

- [ ] **Step 3: Verify no other references**

```bash
grep -rn "SPAG_OCCL_MASKDUMP" spag4d/ scripts/ *.py docs/ .claude/ 2>/dev/null
```
Expected: no matches (this doc audit found none outside `video.py` already).

- [ ] **Step 4: Run the harness after**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`. This flag was never exercised in a default run, so the fixtures should be untouched — if any test fails, stop and investigate before continuing (do not touch fixtures).

- [ ] **Step 5: Commit**

```bash
git add spag4d/video.py
git commit -m "chore: remove dead SPAG_OCCL_MASKDUMP debug flag"
```

---

### Task 3: Delete `SPAG_TRACK_DEDUP_DEBUG`

**Files:**
- Modify: `spag4d/video.py:2594-2596` and `spag4d/video.py:2625-2628`

**Interfaces:**
- Consumes: `tests/test_regression_golden.py`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Run the harness before**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`.

- [ ] **Step 2: Remove the first debug print (video.py:2594-2596)**

Current code:
```python
                if os.environ.get("SPAG_TRACK_DEDUP_DEBUG", "0") == "1":
                    print(f"[Track dedup][dbg] obj{active_tracks[a]['obj_id']} vs obj{active_tracks[b]['obj_id']}: "
                          f"{len(pairs)} matched pairs, mean overlap={mean_ov:.3f}, mean gap={mean_gap:.1f}px -> DUP")
                _dedup_union(a, b)
```
Replace with:
```python
                _dedup_union(a, b)
```

- [ ] **Step 3: Remove the second debug print (video.py:2625-2628)**

Current code:
```python
                        if os.environ.get("SPAG_TRACK_DEDUP_DEBUG", "0") == "1":
                            print(f"[Track dedup][dbg] obj{active_tracks[a]['obj_id']} <-> obj{active_tracks[b]['obj_id']}: "
                                  f"reid anchor gap={time_gap} decimated frames, "
                                  f"overlap={spatial_ov:.3f}, gap={spatial_gap:.1f}px -> DUP")
                        _dedup_union(a, b)
                        break
```
Replace with:
```python
                        _dedup_union(a, b)
                        break
```

- [ ] **Step 4: Verify no other references**

```bash
grep -rn "SPAG_TRACK_DEDUP_DEBUG" spag4d/ scripts/ *.py docs/ .claude/ 2>/dev/null
```
Expected: no matches.

- [ ] **Step 5: Run the harness after**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`.

- [ ] **Step 6: Commit**

```bash
git add spag4d/video.py
git commit -m "chore: remove dead SPAG_TRACK_DEDUP_DEBUG debug flag"
```

---

### Task 4: Delete `SPAG_SP_FIX_SEAMGAP`

**Files:**
- Modify: `spag4d/video.py:629-634`

**Interfaces:**
- Consumes: `tests/test_regression_golden.py`.
- Produces: nothing consumed by later tasks. Note for Task 5's implementer: this task and Task 5 both touch the `single_pass` block in `video.py` (lines ~618-645) but at disjoint line ranges (629-634 here vs. 2206+ for Task 5) — no overlap, safe to run sequentially in either order, but this plan orders Task 4 before Task 5.

- [ ] **Step 1: Run the harness before**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`.

- [ ] **Step 2: Remove `fix_seam_gap` and simplify `cond_seam`**

Current code (`spag4d/video.py:624-641`):
```python
            sp_seam = int(os.environ.get("SPAG_SP_SEAMPAD", "0"))
            # Decoupled-pass fix (opt-in, dead end -- see C1): redo flow at
            # flow_seam_pad for propagation only. Functionally perfect but +15%
            # slower than baseline, erasing single_pass's whole time win.
            fix_seam_gap = (
                os.environ.get("SPAG_SP_FIX_SEAMGAP", "0") == "1"
                and sp_seam != flow_seam_pad
            )
            # P4 conditional variant of fix_seam_gap -- CONFIRMED DEAD END
            # 2026-07-27, kept as scaffolding only. Seam padding isn't local
            # (pads the whole frame into WAFT), so recomputing only near-seam
            # pairs doesn't recover baseline; see D1 Priority 4.
            cond_seam = (
                os.environ.get("SPAG_SP_COND_SEAMPAD", "0") == "1"
                and sp_seam != flow_seam_pad
                and not fix_seam_gap
            )
            seam_band = flow_seam_pad  # px from each vertical edge (at waft res)
            Hw, Ww = waft_frames.shape[1], waft_frames.shape[2]
            flow_masks = np.zeros((len(waft_frames), Hw, Ww), dtype=np.uint8)
            print(f"[SPAG4D] Single-pass WAFT (seam_pad={sp_seam}, fix_seam_gap={fix_seam_gap}, "
                  f"cond_seampad={cond_seam}): bidirectional flow + derived SAM mask...")
```

Replace with:
```python
            sp_seam = int(os.environ.get("SPAG_SP_SEAMPAD", "0"))
            # P4 conditional seam-padding -- still the only unexplored avenue
            # for single_pass's seam regression, see D1 Priority 4 / C1.
            cond_seam = (
                os.environ.get("SPAG_SP_COND_SEAMPAD", "0") == "1"
                and sp_seam != flow_seam_pad
            )
            seam_band = flow_seam_pad  # px from each vertical edge (at waft res)
            Hw, Ww = waft_frames.shape[1], waft_frames.shape[2]
            flow_masks = np.zeros((len(waft_frames), Hw, Ww), dtype=np.uint8)
            print(f"[SPAG4D] Single-pass WAFT (seam_pad={sp_seam}, "
                  f"cond_seampad={cond_seam}): bidirectional flow + derived SAM mask...")
```

- [ ] **Step 3: Check for any other use of `fix_seam_gap` in the same function**

```bash
grep -n "fix_seam_gap" spag4d/video.py
```
If any remaining reference exists below the block just edited (within the same `run_video`/single-pass code path), read that section and remove or adapt the reference so `fix_seam_gap` is no longer referenced anywhere — it should have been local to this block only, but confirm before committing.

- [ ] **Step 4: Verify no other references to the flag itself**

```bash
grep -rn "SPAG_SP_FIX_SEAMGAP" spag4d/ scripts/ *.py 2>/dev/null
```
Expected: no matches in code. (`docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md` and CLAUDE.md still name it in past tense as a historical record of the dead-end investigation — leave those doc references as-is, they describe what was tried, not a live code pointer.)

- [ ] **Step 5: Run the harness after**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`. `SPAG_SINGLE_PASS` is off by default in both golden fixtures' capture command, so this path isn't exercised by the harness — if a failure occurs here, it indicates an unrelated regression; stop and investigate.

- [ ] **Step 6: Commit**

```bash
git add spag4d/video.py
git commit -m "chore: remove dead SPAG_SP_FIX_SEAMGAP experimental flag"
```

---

### Task 5: Rename `SPAG_SAM3_SCALE_DISABLE` → `SPAG_SAM3_SCALE`

**Files:**
- Modify: `spag4d/video.py:2206`

**Interfaces:**
- Consumes: `tests/test_regression_golden.py`.
- Produces: `SPAG_SAM3_SCALE` as the real env var name, matching `docs/SPAG_ENV_FLAGS.md` (Task 1) and every existing code comment/doc reference.

- [ ] **Step 1: Run the harness before**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`.

- [ ] **Step 2: Rename the env var read**

Current code (`spag4d/video.py:2206`):
```python
    sam3_scale = float(os.environ.get("SPAG_SAM3_SCALE_DISABLE", "1.0"))
```
Replace with:
```python
    sam3_scale = float(os.environ.get("SPAG_SAM3_SCALE", "1.0"))
```

- [ ] **Step 3: Verify no other code reference to the old name**

```bash
grep -rn "SPAG_SAM3_SCALE_DISABLE" spag4d/ scripts/ *.py docs/ .claude/ 2>/dev/null
```
Expected: no matches — every other reference in the codebase (comments at `video.py:2201`, `2240`, `3125`, plus CLAUDE.md/B1/D1 docs) already uses `SPAG_SAM3_SCALE` without the suffix, per the 2026-09-03 research pass. If this grep finds a match, read that line and rename it too before committing.

- [ ] **Step 4: Run the harness after**

```bash
/home/mb273924/miniforge3/envs/spag4d/bin/python -m pytest tests/test_regression_golden.py -v
```
Expected: `11 passed`. `SPAG_SAM3_SCALE` defaults to `1.0` (no-op) in both golden fixtures, so behavior is unchanged for the harness's own runs — this is purely a naming fix.

- [ ] **Step 5: Commit**

```bash
git add spag4d/video.py
git commit -m "fix: rename SPAG_SAM3_SCALE_DISABLE to SPAG_SAM3_SCALE to match docs/comments

The env var name never matched any comment or doc reference to it (all of
which say SPAG_SAM3_SCALE) -- a stray _DISABLE suffix. No behavior change:
default remains 1.0 (no-op)."
```

---

### Task 6: `scripts/README.md` index

**Files:**
- Create: `scripts/README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks (independent of Track A).

- [ ] **Step 1: Confirm the current script inventory matches this plan's list**

```bash
ls scripts/*.py scripts/*.sh scripts/*.bat 2>/dev/null
```
Expected output (40 files, per the 2026-09-03 research pass): `analyze_flow_latitude.py`, `build_comparison_html.py`, `capture_golden_fixtures.py`, `check_camera_static.py`, `compare_conf_arms.py`, `compute_mask_iou.py`, `depth_reprojection_consistency.py`, `eval_bg_point_stability.py`, `eval_fg_point_stability.py`, `export_flow_only.py`, `feather_hardcutover_test.py`, `fp16_quantization_floor_test.py`, `make_fixture_clips.py`, `mask_injection_analysis.py`, `pager_smoke.py`, `patch_unisharp_no_render.py`, `reconvert_lockactivity.sh`, `regenerate_trajectory_snapshots.sh`, `render_depth_colormap.py`, `render_fg_depth_flicker.py`, `render_indepscale_fg_flicker.py`, `render_translated_rgb.py`, `residual_vs_latitude_test.py`, `run_atelier1_bglock_sol1.py`, `run_bglock_audit.py`, `run_bg_stability_bench.py`, `run_flowfix_trace.py`, `run_full_clip_postfix_sweep.py`, `run_pager_accident_bench.py`, `run.py`, `run_ram_decompose.py`, `run_scene01_bg_stability_bench.py`, `run_segmentation_only.py`, `setup_omniroam_wsl.sh`, `setup_unisharp_windows.bat`, `stride_invariance_compare.py`, `synth_lateral_motion_test.py`, `synth_radial_lateral_motion_test.py`, `synth_radial_motion_floor_sweep.py`, `synth_radial_motion_test.py`, `view_depth_sequence.py`.

If the actual `ls` output differs from this list (files added/removed since the research pass), use the actual current output for Step 2 instead — read each new/changed file's docstring before writing its row.

- [ ] **Step 2: Write the index**

Create `scripts/README.md`:

```markdown
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
| `render_debug.py` | Debug rendering utility for pipeline intermediate outputs. |
| `batch.sh` | (removed in commit `73e2a0f` — no longer present.) |
```

- [ ] **Step 2: Verify `render_debug.py` and `batch.sh` status**

```bash
ls render_debug.py 2>/dev/null; git log --oneline -1 -- batch.sh
```
If `render_debug.py` doesn't exist, remove that row from the table before committing. The `batch.sh` row documents it as removed intentionally — leave that row as historical context; do not re-add the file.

- [ ] **Step 3: Commit**

```bash
git add scripts/README.md
git commit -m "docs: add scripts/ index"
```

---

### Task 7: Fix pager doc-drift and `report_solutions.py` `--dir` default

**Files:**
- Modify: `.claude/E1_PAGER_TEMPORAL_MVD_PLAN.md:1-9`
- Modify: `report_solutions.py:73`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Read the current top-of-file status block**

```bash
sed -n '1,12p' .claude/E1_PAGER_TEMPORAL_MVD_PLAN.md
```

- [ ] **Step 2: Fix the stale status line**

The current top-of-file line reads (per the 2026-09-03 research pass) something like `"...dated 2026-07-31. Design plan, not yet implemented."` followed by a bolded note that spatial-only pager already ships. Replace the leading status line so it no longer claims "not yet implemented" as the file's overall status — the file already has the correct nuance below it (spatial-only ships, temporal/MVD doesn't), so this step only fixes the top-line summary to match. Change the first status line to:

```
Written 2026-07-31. **Status: spatial-only pager (`active_generator="pager"`) ships in production; the temporal/MVD batching design below remains unbuilt.**
```

Keep the existing bolded "Spatial-only already ships" paragraph immediately below unchanged — it already correctly documents `spag4d/pager_model.py` / `PaGeRModel` as real and wired.

- [ ] **Step 3: Fix `report_solutions.py`'s `--dir` default**

Current code (`report_solutions.py:73`):
```python
    ap.add_argument("--dir", default="benchmark_solutions_pager")
```
Replace with:
```python
    ap.add_argument("--dir", default="benchmark_solutions")
```
This matches `benchmark_solutions.py`'s own `--output` default (`./benchmark_solutions`, confirmed 2026-09-03) — `benchmark_solutions_pager` does not exist anywhere in the repo and never has.

- [ ] **Step 4: Fix the matching docstring example in `report_solutions.py`**

```bash
grep -n "benchmark_solutions_pager" report_solutions.py
```
Update any remaining occurrence (the usage-example line near the top of the file) from `benchmark_solutions_pager` to `benchmark_solutions` to match the new default.

- [ ] **Step 5: Verify no other stale references**

```bash
grep -rln "benchmark_solutions_pager" . --include="*.py" --include="*.md" 2>/dev/null
```
Fix any remaining match the same way (replace `benchmark_solutions_pager` with `benchmark_solutions`).

- [ ] **Step 6: Commit**

```bash
git add .claude/E1_PAGER_TEMPORAL_MVD_PLAN.md report_solutions.py
git commit -m "docs: fix pager doc-drift status line and report_solutions.py --dir default

E1_PAGER_TEMPORAL_MVD_PLAN.md's status line said 'not yet implemented' when
spatial-only pager has shipped since; the file's own body already said so.
report_solutions.py --dir defaulted to a directory (benchmark_solutions_pager)
that never existed -- benchmark_solutions.py's own default is benchmark_solutions."
```

---

### Task 8: Update the pager-plan memory file and close out the 73e2a0f WIP note

**Files:**
- Modify: `/home/mb273924/.claude/projects/-raid-mb273924-SPAG4d/memory/project_e1_pager_plan.md` (outside the repo — user's global memory store, not git-tracked)

**Interfaces:**
- Consumes: Task 7's doc fix (this task's memory-file wording should match the corrected `.claude/E1_PAGER_TEMPORAL_MVD_PLAN.md` status line).
- Produces: nothing (terminal task).

- [ ] **Step 1: Read the current memory file**

```bash
cat "/home/mb273924/.claude/projects/-raid-mb273924-SPAG4d/memory/project_e1_pager_plan.md"
```

- [ ] **Step 2: Update the description to match the corrected doc status**

The memory file's `description:` frontmatter field and body currently say the pager plan is "not implemented." Update the body to state: spatial-only pager (`active_generator="pager"`) ships in production via `spag4d/pager_model.py`; only the temporal/MVD batching extension described in `.claude/E1_PAGER_TEMPORAL_MVD_PLAN.md` remains unbuilt. Keep the existing "start from mask-gated batching (Option A), not attn_mask fork" guidance line unchanged — that's still the right next-step guidance for the unbuilt part.

- [ ] **Step 3: Update `MEMORY.md`'s index line for this entry**

```bash
grep -n "e1_pager_plan" "/home/mb273924/.claude/projects/-raid-mb273924-SPAG4d/memory/MEMORY.md"
```
If the one-line index entry says "not implemented," update it to say "spatial-only ships, temporal/MVD unbuilt" to match.

- [ ] **Step 4: No git commit for this task**

This file lives outside the repo (`/home/mb273924/.claude/...`), so there is nothing to `git add`/`commit` — the edit is complete once saved. This closes out the pager doc-drift item and, combined with Tasks 1-7, supersedes commit `73e2a0f`'s "pending doc/script cleanup" WIP note per the user's 2026-09-03 confirmation — no separate commit or follow-up is needed for that note.

---

## Self-Review

**Spec coverage:**
- Section 3 (Flag amnesty): `docs/SPAG_ENV_FLAGS.md` (Task 1) ✓, delete confirmed-dead flags gated on the harness (Tasks 2-4) ✓, `SPAG_CONF_LEGACY`/`SPAG_SP_COND_SEAMPAD` kept per corrected research + user ruling (documented in Global Constraints and Task 1's table) ✓, `SPAG_SAM3_SCALE_DISABLE` naming bug fixed per user ruling (Task 5) ✓.
- Section 4 (Script/doc cleanup): `scripts/README.md` index (Task 6) ✓, pager doc-drift fix (Task 7) ✓, `report_solutions.py --dir` default fix (Task 7) ✓, `73e2a0f` pending-cleanup question resolved (Task 8 + Global Constraints, per explicit user confirmation) ✓.
- Original spec also implicitly expected script *deletions* for genuinely dead scripts — none were found in the 2026-09-03 research pass, so this plan documents rather than deletes, consistent with the spec's own "confirm with user before deleting" instruction and the fact there's nothing confirmed-dead to confirm deleting.

**Placeholder scan:** No "TBD"/"TODO"/"add appropriate X" found — every step has literal file content, exact line numbers, or exact shell commands. Task 2's exact end-line for the `SPAG_OCCL_MASKDUMP` block was not fully visible during planning (truncated at line 3178 in the research excerpt); Step 2 of that task explicitly instructs the implementer to `sed -n` the surrounding range and find the real closing line rather than guessing — this is a legitimate "find the exact boundary before editing" instruction, not a placeholder, since the deletion target (the whole `if _maskdump:` block) and its start line are both exact.

**Type consistency:** No functions/types are introduced by this plan — all tasks are deletions, one rename, and documentation. The renamed identifier (`SPAG_SAM3_SCALE_DISABLE` → `SPAG_SAM3_SCALE`) is referenced consistently across Task 5's steps and Task 1's table.
