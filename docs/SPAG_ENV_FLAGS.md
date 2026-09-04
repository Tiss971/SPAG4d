# SPAG_* Environment Flags Reference

Audited 2026-09-03, regrouped by subsystem 2026-09-04. Every `SPAG_*` flag
read by `spag4d/video.py` or `spag4d/flow_depth_propagation.py`, its
default, and its status. When a flag's default value changes, or a flag is
added/removed, update this table in the same commit.

Status legend:
- **default-on** — shipped, on by default, no action needed to use it.
- **opt-in** — shipped, off by default, safe to enable per the linked doc.
- **debug** — developer-only diagnostic output/dump, no functional effect
  on pipeline output.
- **required-for-repro** — off by default, but needed to reproduce a
  specific historical benchmark; do not delete even though "dead" by usage.
- **experimental-open** — an unfinished investigation track, not yet
  resolved as dead or shippable; do not delete.

## Bg-lock / depth compositing

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_LOCK_ACTIVITY` | `1` | default-on | Bg-lock composite keys off `sam_mask` alone, not the fused SAM\|activity mask. Shipped 2026-07-28, see project CLAUDE.md. |
| `SPAG_BGLOCK_DILATE_PX` | `12` | opt-in | Overrides bg-lock dilation px, only applied if changed from default. |
| `SPAG_BGLOCK_FEATHER_PX` | `9` | opt-in | Overrides bg-lock feather px, only applied if changed from default. |
| `SPAG_BGLOCK_NOFLOW` | `0` | opt-in | Disables flow use in bg-lock. |
| `SPAG_BGLOCK_NOFLOW_BLEND` | `0.5` | opt-in | Blend factor for no-flow bg-lock mode. |
| `SPAG_DREF_MIN_SAMPLES` | `0` | opt-in | Min sample threshold for depth-reference. See `docs/bglock_open_questions.md` §5. |
| `SPAG_HARD_DEPTH_CUTOVER` | `0` | opt-in | Hard cutover mode for depth compositing. See `docs/bglock_open_questions.md` §8.1. |
| `SPAG_MASK_INJECT` | unset | opt-in | Optional mask injection override path. See `docs/bglock_open_questions.md` §6.3. |
| `SPAG_COMPOSITE_LOWRES` | `0` | opt-in | Low-res compositing path. |
| `SPAG_GPU_COMPOSITE` | `1` | default-on | GPU-resident compositing. |
| `SPAG_DIAG_CSV` | unset | opt-in | Path to write a diagnostics CSV. See `docs/bglock_open_questions.md` §9. |

## Flow-propagation confidence decay (spag4d/flow_depth_propagation.py + video.py)

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_CONF_DECAY` | `0.85` | default-on | Per-pixel confidence decay rate, compounding over a warped age field. Shipped 2026-08-17. |
| `SPAG_CONF_FLOOR` | `0.5` | default-on | Floor for decayed confidence. |
| `SPAG_CONF_LEGACY` | `0` | required-for-repro | Restores the pre-2026-08-17 non-compounding confidence-decay behavior. Required to reproduce any benchmark recorded before 2026-08-17 — see CLAUDE.md and `docs/bglock_open_questions.md` §4.1/§4.2. **Not dead, do not delete.** |
| `SPAG_CONF_HIST` | unset | debug | Path to write confidence-history data. |
| `SPAG_FLOW_EDGE_NEAREST` | `0` | opt-in | Nearest-neighbor edge handling in flow-based depth propagation. |
| `SPAG_FLOW_EDGE_CONF_PENALTY` | `0.0` | opt-in | Confidence penalty applied at flow edges. |
| `SPAG_FLOW_EDGE_ZERO_CONF` | `0` | opt-in | Zeroes confidence at edges. |
| `SPAG_FLOW_EDGE_THRESH` | `0.15` | opt-in | Relative threshold for edge detection. |

## Single-pass WAFT / seam padding

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_SINGLE_PASS` | `0` | opt-in | Single-pass WAFT flow (skip second pass). Degrades foreground depth on seam-crossing clips — see `docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md`. Not default. |
| `SPAG_SP_SEAMPAD` | `0` | opt-in | Seam pad amount used with `SPAG_SINGLE_PASS`. |
| `SPAG_SP_COND_SEAMPAD` | `0` | experimental-open | Conditional seam-padding variant for single-pass mode. Still the only unexplored avenue for fixing single-pass's seam regression per `docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md` and CLAUDE.md — **not dead, do not delete.** |

## SAM3 segmentation pass

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_SAM3_BF16` | `1` | default-on | Selective bf16 weight-cast on SAM3 (lives in the gitignored `third_party/sam3` vendor fork). Validated safe solo at native res (MattSwift IoU 0.99). |
| `SPAG_SAM3_MAXSIZE` | `2048` | default-on | Absolute cap on SAM3 input's longer edge. Was `0`→disabled 2026-08-18 to keep `SPAG_SAM3_BF16` solo (1536 stack regressed); changed to `2048` 2026-09-04, stacked with `SPAG_SAM3_BF16=1` — pending formal MattSwift IoU re-validation, see project CLAUDE.md. |
| `SPAG_SAM3_SCALE` | `1.0` | opt-in | Downscale factor for the SAM3 segmentation pass input. Renamed from `SPAG_SAM3_SCALE_DISABLE` in this plan (Task 5) to match its actual usage everywhere else in code comments and docs. Superseded by `SPAG_SAM3_MAXSIZE` as the preferred resolution lever — see project CLAUDE.md. |
| `SPAG_SAM3_MAX_FRAMES` | unset | opt-in | Optional cap on SAM3 frame batch size. |

## Track dedup / re-identification

**Changed 2026-09-04:** the 5 individual threshold flags (`_GAP`, `_OVERLAP`,
`_SIZE_RATIO`, `_MIN_COOCCUR`, `_IDX_TOL`) are **removed**. A single
`SPAG_TRACK_DEDUP_PROFILE` (`strict`\|`default`\|`loose`) now sets all 5 at
once — no per-value override remains; edit the preset table in
`spag4d/video.py` (`_dedup_profiles`) directly for one-off tuning. Only
`default` is production-validated (it's the prior individual defaults,
byte-for-byte unchanged — no default behavior change from this flag's
addition). `strict`/`loose` are directional presets (tighter/looser dedup
matching) that have **not** been benchmarked — treat them as untested until
run through the golden regression harness on a track-dedup-heavy clip
(e.g. MattSwift).

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_TRACK_DEDUP_PROFILE` | `default` | opt-in | Sets all 5 dedup thresholds (GAP=40, OVERLAP=0.6, SIZE_RATIO=0.3, MIN_COOCCUR=2, IDX_TOL=3 for `default`). See note above. |
| `SPAG_TRACK_REID_MAX_GAP` | `40` | opt-in | Max frame gap for track re-identification via anchor proximity. Different mechanism (anchor-proximity re-ID, not fragment-fusion dedup) — not part of the profile. |

## Deleted flags (removed 2026-09-04, folded into `SPAG_TRACK_DEDUP_PROFILE`)

| Flag | Was | Removed because |
|---|---|---|
| `SPAG_TRACK_DEDUP_GAP` | opt-in, `40` | Folded into `SPAG_TRACK_DEDUP_PROFILE`'s preset table; no per-value override kept. |
| `SPAG_TRACK_DEDUP_OVERLAP` | opt-in, `0.6` | Folded into `SPAG_TRACK_DEDUP_PROFILE`'s preset table; no per-value override kept. |
| `SPAG_TRACK_DEDUP_SIZE_RATIO` | opt-in, `0.3` | Folded into `SPAG_TRACK_DEDUP_PROFILE`'s preset table; no per-value override kept. |
| `SPAG_TRACK_DEDUP_MIN_COOCCUR` | opt-in, `2` | Folded into `SPAG_TRACK_DEDUP_PROFILE`'s preset table; no per-value override kept. |
| `SPAG_TRACK_DEDUP_IDX_TOL` | opt-in, `3` | Folded into `SPAG_TRACK_DEDUP_PROFILE`'s preset table; no per-value override kept. |

## Occlusion handling (FB-consistency gate)

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_OCCL_FBGATE` | `0` | opt-in | Forward-backward consistency gate for occlusion handling. |
| `SPAG_OCCL_FB_THRESH` | `3.0` | opt-in | FB-consistency threshold for occlusion gate. |
| `SPAG_OCCL_MAX_SKIP` | `8` | opt-in | Max frame skip for occlusion handling. |
| `SPAG_OCCL_DEBUG` | `0` | debug | Verbose per-occurrence FB-error logging for the occlusion gate above. |

## Determinism / misc debug

| Flag | Default | Status | Notes |
|---|---|---|---|
| `SPAG_DETERMINISTIC` | `0` | default-on for tests | Enables deterministic seeding; forced to `1` by `tests/conftest.py` for the golden regression harness. |
| `SPAG_SEED` | `42` | opt-in | Seed value used when `SPAG_DETERMINISTIC=1`. |
| `SPAG_RAM_TRACE` | `0` | debug | Enables RAM tracing. |

## Deleted flags (removed 2026-09-03, Task 2-4)

| Flag | Was | Removed because |
|---|---|---|
| `SPAG_OCCL_MASKDUMP` | debug | Pure debug/dump utility, no doc references it as load-bearing. |
| `SPAG_TRACK_DEDUP_DEBUG` | debug | Pure debug print flag, no doc references it as load-bearing. |
| `SPAG_SP_FIX_SEAMGAP` | opt-in | Confirmed dead end 2026-07/08 — correct but +15% slower than baseline, erasing `SPAG_SINGLE_PASS`'s whole time win. See `docs/C1_SINGLE_PASS_SEAMPAD_ROOTCAUSE.md`. |

## Consolidation candidates (not yet acted on)

Groups of flags that are individually live and load-bearing but could
plausibly collapse into fewer knobs. Listed here as candidates only — none
of these have been designed or approved, this is not a plan.

- ~~`SPAG_TRACK_DEDUP_*` (5 flags)~~ — **shipped 2026-09-04** as
  `SPAG_TRACK_DEDUP_PROFILE`, see the Track dedup section above.
- **Resolved, not a candidate: `SPAG_OCCL_FBGATE` / `_FB_THRESH` /
  `_MAX_SKIP`** — same gate-plus-dependent-params shape as
  `SPAG_SINGLE_PASS`/`SPAG_SP_SEAMPAD` below, not a redundant threshold set
  like track-dedup was. Only 2 dependent knobs; leaving as-is.
- **Resolved, not a candidate: `SPAG_FLOW_EDGE_NEAREST` / `_CONF_PENALTY` /
  `_ZERO_CONF`** — read the code comment at
  `flow_depth_propagation.py:60-77`: these are explicitly documented as
  "three independent experiments," i.e. mutually exclusive alternative
  fixes for the same silhouette-bleed bug, not a set meant to be combined
  or co-tuned. Merging them into one flag would hide that they're
  alternatives, not settings. `SPAG_FLOW_EDGE_THRESH` is the one genuinely
  shared parameter (the edge-detection threshold all three read) and stays
  separate since it's orthogonal to which of the three is active.
- **`SPAG_SP_SEAMPAD` / `SPAG_SP_COND_SEAMPAD`** — both only matter when
  `SPAG_SINGLE_PASS=1`; already effectively a 3-flag mini-namespace. Not
  worth merging further, but worth naming as an `SPAG_SINGLE_PASS_*`
  prefix if `SPAG_SP_COND_SEAMPAD`'s open investigation ever ships.
- **Not a candidate:** `SPAG_SAM3_BF16` / `SPAG_SAM3_MAXSIZE` /
  `SPAG_SAM3_SCALE` — these interact (see notes above) but are genuinely
  orthogonal levers (precision vs. two different resolution mechanisms),
  not duplicates of the same knob.

No further consolidation candidates remain open as of 2026-09-04: every
group of 3+ flags in this file has now been either merged
(`SPAG_TRACK_DEDUP_PROFILE`) or explicitly ruled out with a stated reason.
