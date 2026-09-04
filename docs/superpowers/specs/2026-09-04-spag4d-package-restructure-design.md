# SPAG4D package restructure — grouped subfolders

Date: 2026-09-04
Status: approved, pending implementation plan

## Goal

`spag4d/` is 25 flat files (~9,850 lines), dominated by `video.py` (3,187
lines) and `sharp360.py` (1,184 lines). This is in-place cleanup to group
the package into logical subfolders, in prep for an eventual port of
this code into another repo. Not a rewrite: behavior must not change.

## Target structure

```
spag4d/
  depth/        depth_blend.py, depth_refiner.py, flow_depth_propagation.py, scene_filter.py
  geometry/     spherical_grid.py, projection.py, ply_writer.py, ply_depth_render.py
  generators/   da360_model.py, da360_arch/, pager_model.py, pager_arch/,
                sharp360.py, sharp_arch/, unisharp360.py, unisharp_adapter.py, unisharp_format.py
  motion/       detect_opticalflow.py, detect_opticalflow_utils.py
  pipeline/     video.py, core.py, spag_converter.py
  analysis/     analyze_mask_quality.py, reconstruction_metrics.py, scene_analysis.py
  cli.py, progress.py, __init__.py, __main__.py   # stay top-level
```

Each new subfolder gets an `__init__.py` re-exporting its public symbols
(what's currently importable from the flat file), so intra-package
imports read naturally (`from spag4d.depth import scene_filter` or
`from spag4d.depth.scene_filter import SkyMode`, whichever matches how
call sites already use it — decide per-symbol during implementation,
favor `from spag4d.<pkg>.<module> import X` explicit paths over
re-exports where an import is used in only one place).

`da360_arch/`, `pager_arch/`, `sharp_arch/` move as whole directories
under `generators/` alongside their respective `*_model.py`/`sharp360.py`
files, unchanged internally.

## Current internal dependency graph (must be preserved)

```
__main__.py         -> cli
cli.py               -> pager_model (lazy, inside a function)
detect_opticalflow.py -> progress
spag_converter.py    -> scene_filter, spherical_grid
pager_model.py        -> pager_arch
reconstruction_metrics.py -> flow_depth_propagation
video.py              -> core, detect_opticalflow, flow_depth_propagation,
                          ply_writer, progress, scene_analysis,
                          reconstruction_metrics (lazy, inside a function)
__init__.py           -> core
core.py, sharp360.py, cli.py, video.py -> third_party/sam3 (external, non-portable)
```

No cycles. This graph is small enough to remap by hand; no dependency
restructuring is needed, only import path updates.

## External callers to update (no compat shims)

This is in-place cleanup of an internal package, not a published
library — old flat import paths will NOT be kept as re-export shims.
Every caller gets rewritten to the new path. Known callers today:

- `tests/test_helpers.py`, `tests/test_regression_golden.py`
- `benchmark_solutions.py`, `report_solutions.py` (repo root)
- `scripts/*.py` (~15 files): `run.py`, `run_flowfix_trace.py`,
  `run_ram_decompose.py`, `run_bg_stability_bench.py`,
  `feather_hardcutover_test.py`, `run_bglock_audit.py`,
  `run_atelier1_bglock_sol1.py`, `run_segmentation_only.py`,
  `pager_smoke.py`, `depth_reprojection_consistency.py`,
  `capture_golden_fixtures.py`, `run_full_clip_postfix_sweep.py`,
  `residual_vs_latitude_test.py`, `export_flow_only.py`,
  `run_pager_accident_bench.py`, `run_scene01_bg_stability_bench.py`

Note: `scripts/render_translated_rgb.py` already imports a nonexistent
path (`spag4d.refine.geometric.render_utils`) — pre-existing breakage,
unrelated to this move; leave as-is (or flag separately), don't try to
fix it as part of this restructure.

## Non-portable dependency boundary (SAM3/third_party)

`core.py`, `sharp360.py`, `cli.py`, `video.py` import from
`third_party/sam3`. Per decision: isolate by documentation only, no code
movement. Add a short comment at each import site (or a single
`generators/README.md` note where `sharp360.py` lives, whichever reads
cleaner in context) marking these as the non-portable boundary a future
port needs to swap out. No behavior change.

## Dead code

Checked: the flow-warp propagation deletion (commit `5800d6e`) was
already thorough. Only harmless historical comments remain
(`video.py:658,1000,1444`) referencing the old mechanism — these read
fine as documentation of why the current approach was chosen and should
NOT be deleted as part of this restructure. No further dead-code hunt
is in scope here.

## Docs/benchmarks triage (flag, don't auto-delete)

Produce a triage list (keep / stale-candidate / unclear) covering:
- `docs/*.md` (5 files, excluding `docs/superpowers/`)
- `scripts/*.py` and `*.sh`/`*.bat` (~30 files)
- `benchmarks/*_2026-*` run-output subfolders

Criteria for "stale-candidate": one-off script tied to a specific
already-closed investigation (per CLAUDE.md's chronology docs), not
referenced by any currently-relevant doc, not run in the last ~30 days
per file mtime. Present the list to the user; only delete what they
confirm. This restructure's code changes do not depend on the triage
outcome — it can land as a separate, later step.

## Testing / verification

- Run `tests/` (including the golden-output regression harness) after
  the move — must pass unchanged.
- `python -m spag4d --help` and a basic `import spag4d` smoke check.
- `grep -rn "from spag4d\.\|import spag4d" --include=*.py .` afterward
  to confirm no stale flat-path imports remain outside intentionally
  unrelated pre-existing breakage (`render_translated_rgb.py`).

## Out of scope

- Any behavior/logic change inside moved files.
- Actually extracting/porting code to another repo (this is prep only).
- Deleting docs/scripts/benchmarks without explicit user confirmation
  on the triage list.
- Removing or restructuring the `third_party/sam3` coupling itself.
