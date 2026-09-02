# SPAG_BLEED_REJECT scene01 validation (2026-08-21) — superseded, fix reverted

**Status: the fix validated here was reverted for a correctness bug — see `docs/bglock_open_questions.md`
§8.7 for the full scene01 "waiter" trail investigation (#2-#5). Kept as a record that this class of
fix (rejecting silhouette-edge depth bleed) does measurably reduce the trail artifact — the
mechanism was right, this specific global-threshold implementation was not.**

## What was tested

`SPAG_BLEED_REJECT` (`spag4d/video.py`, since removed): under `freeze_bg`, tested how close each
in-SAM-mask pixel's depth had collapsed toward the frozen background plate (`depth_ref`) and
dropped (NaN'd) pixels within `bleed_frac=0.35` of that gap — theory being that outermost
silhouette pixels pick up background depth via bilinear `grid_sample` interpolation across the
depth cliff during flow warp.

Two matched scene01 conversions (`--stride 4 --skip-step 2 --freeze-bg --depth-correction bglock
--outlier-pruning 0.02 --grazing-angle 85.0 --sparse-pruning 0.1`): `SPAG_BLEED_REJECT=0` vs.
`=0.35`, rendered via `gsplat.rasterization()` at the waiter window (frames 140-190).

## Result

Pixel diff (with vs. without), full 768×768 render: mean abs diff 0.09-0.42, up to 1.5% of pixels
— an order of magnitude larger than every prior arm in the #1-#4 investigation (fix2b/#3 both
measured 0.0006-0.009, noise-level). Diff concentrated at the leg/floor boundary: **without** the
fix, a skin-toned smear bridges from the waiter's legs into the floor; **with** it, the floor
renders clean and legs terminate sharply — consistent across 6 sampled frames.

`SPAG_DIAG_CSV` deltas were identical between arms — expected, since that diagnostic is written
before the bleed-reject step runs in the per-frame loop, so it structurally can't see this class
of fix (same blind spot as fix#2/2b's numeric check). The visual sequence comparison was the only
instrument able to observe the artifact.

## Why it was reverted

The rejection test used a single global per-frame median-gap threshold, which can't distinguish
an actual bleed artifact from a subject genuinely touching the background — e.g. feet in contact
with the floor legitimately have depth converging toward `depth_ref` at exactly that location.
Real contact points were silently deleted along with the intended bleed ring — unacceptable
information loss, not a shippable tradeoff. Block removed from `spag4d/video.py`.
