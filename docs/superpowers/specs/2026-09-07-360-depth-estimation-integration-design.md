# SPAG4D → 360_depth_estimation Integration Design

## Goal

Transfer SPAG4D's work into a branch of `/raid/mb273924/360_depth_estimation`
(maintained by a colleague), which already vendors several 360°-depth model
backbones (DA360, DA-2, GemDepth, map-anything, midas, SAM2, ZoeDepth) behind
a shared `depth360` CLI (`main.py` + `utils/`). Both repos serve the same
domain — 360° depth estimation — and SPAG4D currently duplicates part of
that work in its own vendor tree.

This spec covers two phases:

- **Phase 1** (in scope for the implementation plan that follows this spec):
  mechanically copy SPAG4D's files into a new branch of `360_depth_estimation`
  and wire them so everything imports and runs correctly from the new
  location, with the one confirmed literal duplication (DA360) removed.
- **Phase 2** (roadmap only, not planned in detail here): dilute SPAG4D's
  own modules into `360_depth_estimation`'s existing shared modules
  (`utils/`, `main.py`'s subcommand pattern) over a series of later,
  independently-reviewed changes.

No git history is preserved in the move — this is a plain file copy into
the target repo's working tree, not a `git subtree`/history-preserving merge.

## Current State

**`360_depth_estimation`** — flat layout: vendored model backbones as
top-level directories (`DA360/`, `DA-2/`, `GemDepth/`, `map-anything/`,
`midas/`, `SAM2/`, `zoedepth/`), a thin `utils/` package (projection,
motion tracking, bilateral filtering, model loaders, generic utils), and
`main.py` dispatching to `argparse` subcommands (`erp`, `perspectives`,
`single`) that each restrict `--model` to whichever backbones apply. One
`pyproject.toml`, package name `depth360`, console script `depth360`.

**SPAG4D** — a `spag4d/` package (generators, depth, geometry, motion,
pipeline, refine, analysis submodules) plus `third_party/sam3` and
`third_party/UniSHARP` vendor trees, its own `pyproject.toml` (package
name `spag4d`, console script `spag4d`), `scripts/`, `tests/`, `docs/`,
`benchmarks/`.

**Confirmed duplication:** `spag4d/generators/da360_arch/DA360/` is a
full separate git clone of the same upstream Insta360 DA360 repository
that `360_depth_estimation/DA360/` already vendors — identical upstream
code, two independent copies with independent `.git` directories.

**Related but NOT duplicated** (different algorithms/interfaces solving
related problems — left alone in Phase 1, listed as Phase 2 dilution
candidates below):
- `spag4d/geometry/projection.py` (cubemap + icosahedral ERP↔tangent-plane
  projector, custom implementation) vs. `utils/projection.py` (ERP↔cubemap
  via the `pytorch360convert` library, `e2c`/`e2p`).
- `spag4d/motion/detect_opticalflow.py` (optical-flow-based motion
  detection) vs. `utils/motion_tracking.py` (MOG2 background-subtractor-based
  motion detection for a fixed-camera 360 video).

## Phase 1: Copy + Wire (this spec's scope)

### 1. What moves

Copied as-is into `360_depth_estimation/` at the top level, alongside the
existing vendor directories:
- `spag4d/` (the package)
- `third_party/sam3`, `third_party/UniSHARP`
- `scripts/`, `tests/`, `docs/`, `benchmarks/`

### 2. DA360 de-duplication

- Delete `spag4d/generators/da360_arch/DA360/` (the duplicate clone) from
  the copied tree.
- Update `spag4d/generators/da360_model.py`'s imports to reach the
  top-level `DA360/` package (the one `360_depth_estimation` already
  vendors) instead of the deleted `da360_arch/DA360` copy.
- Verify DA360 checkpoint loading still resolves — `360_depth_estimation`
  expects weights under `checkpoints/DA360_large.pth` per its README;
  confirm SPAG4D's checkpoint path config points at the same location or
  is updated to.

### 3. Import-path wiring

SPAG4D's code currently assumes it sits at repo root (e.g.
`sys.path.insert(0, ...)` calls keyed off `__file__` depth, relative
imports assuming `third_party/` is a repo-root sibling). Once nested
under `360_depth_estimation/`, every such path assumption must be
re-verified against the new nesting depth. This is the precise-wiring
work: not a mechanical find-replace, since path depth differs per call
site — each `sys.path` manipulation and cross-package import in
`spag4d/`, `third_party/sam3`, and `third_party/UniSHARP` needs to be
checked against the new location and fixed where it breaks.

### 4. `pyproject.toml` reconciliation

Merge into one `pyproject.toml` at the `360_depth_estimation` root:
- Keep both `[project.scripts]` entries (`depth360` and `spag4d`) so both
  CLIs work from the same installed environment.
- Union the two dependency lists. Watch for version conflicts — both
  repos pin `torch`/`torchvision`; `360_depth_estimation`'s README pins
  `torch==2.7.1`+`torchvision==0.22.1` for specific CUDA wheel URLs, while
  SPAG4D's `pyproject.toml` only floors `torch>=2.0`. Resolve to the
  stricter constraint if they don't already agree.
- SPAG4D's `[project.optional-dependencies]` groups (`server`, `download`,
  `sharp`, `dev`) carry over unchanged; no reason to fold them into
  `360_depth_estimation`'s (which currently has no optional-dependency
  groups at all).

### 5. Verification

Before considering Phase 1 done, from the merged tree:
- SPAG4D's existing test suite (`tests/`) passes.
- `360_depth_estimation`'s existing test (`tests/test_cli_smoke.py`)
  still passes (unaffected by the addition, but confirms nothing broke).
- A real `spag4d` CLI run against a short test clip succeeds end-to-end
  (not just imports — the DA360 rewiring in particular needs a live run,
  since a stale/wrong checkpoint path only fails at model-load time).
- A real `depth360` CLI run (any one subcommand) still succeeds, confirming
  the merge didn't disturb the existing repo's own entry point.

## Phase 2: Dilution Roadmap (not planned in detail here)

Each item below is its own future change, reviewed independently, in
roughly this priority order (highest-leverage / lowest-risk first):

1. **`da360_model.py` → inline `--model` branch in `main.py`.** Once
   Phase 1's rewiring is done and DA360 is a single shared copy, SPAG4D's
   separate generator-wrapper layer for DA360 becomes redundant with
   `main.py`'s existing `DA360`/`MAP_ANYTHING`/`DA2_PANO` dispatch — fold
   it in rather than keeping two call paths to the same model.
2. **Projection module reconciliation.** `spag4d/geometry/projection.py`
   and `utils/projection.py` solve overlapping problems differently, do
   not attempt to unify them until a concrete need — either the DA360
   folding above or a shared cubemap-projection consumer — forces the
   comparison.
3. **Motion-detection module reconciliation.** Same caution as
   projection: `spag4d/motion/detect_opticalflow.py` and
   `utils/motion_tracking.py` stay separate until something concretely
   needs one to call the other.
4. **`spag4d/pipeline/` → new `main.py` subcommand.** SPAG4D's video
   pipeline orchestration becomes a `depth360 video-360`-style subcommand
   built on the (by-then) shared model/projection/motion functions,
   retiring the separate `spag4d` console-script entry point. This is the
   end state implied by "dilution" but depends on items 1-3 landing first.

No task breakdown for Phase 2 belongs in the implementation plan that
follows this spec — each item gets its own brainstorm/spec/plan cycle
when it's taken up.

## Out of Scope

- Preserving SPAG4D's git history in the target repo (explicitly declined
  by the user — plain copy only).
- Any Phase 2 dilution work (see above — roadmap only).
- Renaming either package (`depth360`/`spag4d` both keep their names and
  console scripts through Phase 1).
