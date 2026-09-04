# SPAG4D codebase-health roadmap — design spec

_2026-09-03_

## Context

SPAG4D's temporal-stability work (bglock, occlusion re-anchoring, `SPAG_LOCK_ACTIVITY`, single-pass
flow, confidence decay) has been validated one full-clip benchmark campaign at a time
(`benchmarks/<topic>_<date>/`), because there is no cheaper way to answer "is this change inert?".
That is now the bottleneck on further work, not any specific algorithm:

- `run_video()` in [spag4d/video.py](../../../spag4d/video.py) is a single **3017-line** function
  (the file is 3350 lines total) — every feature since bglock landed as a new branch inside it.
- **45 distinct `SPAG_*` environment flags** gate behavior, several already documented as rejected
  dead ends (`SPAG_SP_FIX_SEAMGAP`, `SPAG_SP_COND_SEAMPAD`, `SPAG_CONF_LEGACY`,
  `SPAG_SAM3_SCALE_DISABLE`) or pure debug dumps (`SPAG_OCCL_MASKDUMP`, `SPAG_TRACK_DEDUP_DEBUG`,
  `SPAG_RAM_TRACE`), still live in the code.
- **Zero automated tests** anywhere outside `third_party/`. No `pytest`, no `conftest.py`.
- ~40 one-off scripts at the repo root and in `scripts/` with no index; CLAUDE.md's own "Note"
  section already documents one round of scripts going stale and unrunnable.
- HEAD (`73e2a0f`) is an explicit WIP checkpoint ("pending doc/script cleanup") left mid-session.

Two prior candidate directions were explicitly scoped out for this cycle:
- **Pager** (`spag4d/pager_model.py` + `pager_arch/`, wired through `core.py`/`cli.py`/`video.py`)
  is implemented and importable but was never benchmarked against `da360`. Decision: leave the code
  as-is (do not delete, do not invest further), remove it from the roadmap, and fix the stale docs
  that currently claim it is "not yet implemented" (`.claude/E1_PAGER_TEMPORAL_MVD_PLAN.md`, project
  memory `project_e1_pager_plan.md`).
- Depth-backbone swap (MoGe-2/MapAnything) and the screen appearance gap are deferred to a future
  cycle, once changes are cheap enough to validate.

## Goal

Make future changes to `run_video()` and its flag surface cheap and safe to validate, without
changing any production numerical output. This is infrastructure, not a feature: success is
"the same regression check that used to require a full-clip benchmark run now runs in seconds,"
not any new pipeline capability.

A secondary, forward-looking constraint on the harness (not a deliverable this cycle): the fixtures
and any future metrics built on them should be able to check foreground depth/mask **coherence
under camera translation to novel viewpoints**, not just source-view frame-to-frame stability —
mirroring what `scripts/eval_fg_point_stability.py` / `scripts/depth_reprojection_consistency.py`
already probe. This shapes what the harness captures (reprojectable per-frame depth+mask, not just
final PLYs) even though building that specific check is out of scope for this cycle.

## Non-goals

- No algorithmic changes to depth/mask/flow behavior. Every step must be verified byte-identical
  (or exactly regression-covered where intentionally changed) against pre-refactor output.
- No pager deletion, no depth-backbone swap, no screen-appearance-gap work this cycle.
- No CI system setup — local `pytest` only, matching the repo's current no-CI reality.

## Design

### 1. Golden-output regression harness

Two fixture clips, chosen because they exercise different code paths already known to be fragile:

- `scene01.mp4` — seam-crossing foreground motion (the clip `SPAG_SINGLE_PASS`/`SPAG_SP_*` were
  benchmarked and rejected against).
- `circulation_site_1_edit_coupe.mp4` — mostly-static background with ground shadows (the clip
  `SPAG_LOCK_ACTIVITY` and `freeze_bg_live_color` were validated against).

Both sourced from `/raid/mb273924/_DATASETS/uptale/data/videos/`. Fixtures cover only the **first
~20-30 frames** of each (not full clips) to keep runtime and fixture size small — long enough to
exercise SAM3 tracking init, bglock reference-frame selection, and at least one seam crossing on
`scene01`, short enough to run in well under a minute per test.

Run under `SPAG_DETERMINISTIC=1` (already shipped, seeds np/torch/CUDA/cuDNN). Capture and pin as
fixtures: `depth_*.npy`, `mask_*.npy`, `flow_*.npy` for the frame range, plus the final PLY's
Gaussian count and a hash of its vertex buffer (cheap smoke check that export didn't silently
change). Store fixtures in-repo under `tests/fixtures/` — verify total size stays in the tens-of-MB
range before committing; if not, revisit (git-lfs, or fewer frames).

New `tests/` directory:
- `tests/conftest.py` — fixture paths, a `run_pipeline_on(clip, frames)` helper.
- `tests/test_regression_golden.py` — byte-identity assertions against the pinned fixtures, one
  parametrized test per clip.
- `tests/test_helpers.py` — unit tests for the pure-function helpers that don't need a full
  pipeline run: `_estimate_scale_shift`, `_fit_outlier_cap`, `_activity_mask_from_std`,
  `pad_circular_horizontal` (if defined standalone), ERP↔cubemap round-trip in
  `spag4d/spherical_grid.py`.

Cross-GPU note (already known from `SPAG_DETERMINISTIC`'s validation): byte-exactness holds across
GPUs for this pipeline, but if a future run on different hardware breaks the fixtures, that is a
signal to investigate, not to loosen the assertion casually.

### 2. `run_video()` extraction

Once the harness is green, split `run_video()` into phase functions along its existing conceptual
boundaries (visible today as comment-delimited sections and the nested helper closures at
`video.py:488` `_ram_checkpoint`, `:1171` `_predict_depth`, `:1705` `stats_it`, etc.):

1. Frame extraction / temporal-median build
2. Depth prediction + alignment (`align_depth_frame` / `align_depth_frame_gpu`)
3. Segmentation/tracking (`segment_with_sam`, `segment_with_flows`)
4. Stabilization (flow propagation, bglock compositing)
5. Gaussian construction + export (`to_gaussians`, scene filtering)

Each extraction is its own commit, gated on `tests/test_regression_golden.py` passing before the
commit and immediately after. Nested nonlocal-closure helpers get promoted to module-level
functions taking explicit arguments only where that doesn't require touching the numerical path —
where a helper's closure-captured state is entangled with the parent function's control flow in a
way that's risky to unwind quickly, leave it in place and note it rather than force the extraction.

### 3. Flag amnesty

After extraction (phases make it easier to see which flags belong to which phase). Delete
confirmed-dead flags per the docs' own verdicts: `SPAG_SP_FIX_SEAMGAP`, `SPAG_SP_COND_SEAMPAD`,
`SPAG_CONF_LEGACY` (only if no open benchmark still needs it — check `docs/bglock_open_questions.md`
first), `SPAG_SAM3_SCALE_DISABLE`, and pure-debug flags with no production use
(`SPAG_OCCL_MASKDUMP`, `SPAG_TRACK_DEDUP_DEBUG` — confirm no active debugging session depends on
them). Each deletion re-runs the golden harness. Remaining flags get a short reference table (one
new doc, `docs/SPAG_ENV_FLAGS.md`) listing name, default, status (shipped-default-on /
opt-in-validated / experimental), and the doc section that justifies it — so the next audit doesn't
require re-reading every `.claude/*.md` file.

### 4. Script/doc cleanup

- Add `scripts/README.md` indexing the ~40 root/scripts files by purpose (or move genuinely dead
  ones to `scripts/archive/` — confirm with the user before deleting anything, per the git-safety
  default).
- Fix pager doc drift: update `.claude/E1_PAGER_TEMPORAL_MVD_PLAN.md` and the project memory to say
  implemented-but-not-pursued, not "not yet implemented."
- Fix `report_solutions.py`'s stale `--dir` default (`benchmark_solutions_pager`).
- Resolve the pending state at HEAD (`73e2a0f`) — confirm with the user whether "pending doc/script
  cleanup" mentioned in that commit message is superseded by this plan or still needs separate
  handling before starting.

## Testing strategy

- `tests/test_helpers.py` runs on every change, no GPU/clip dependency, seconds.
- `tests/test_regression_golden.py` runs before/after each extraction or flag deletion; requires
  GPU + the two fixture clips; this is the safety net the whole plan depends on.
- No new numerical output is introduced — verification is entirely "did this stay byte-identical,"
  except where a flag deletion intentionally changes default behavior (none planned; all deletions
  target flags already off or already proven dead).

## Order of work

1. Harness (fixtures + `tests/`) — blocks everything else.
2. `run_video()` phase extraction, one phase per commit.
3. Flag amnesty.
4. Script/doc cleanup (can interleave with 2-3 where independent, e.g. the pager doc fix and
   `report_solutions.py` default can happen anytime).
