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
