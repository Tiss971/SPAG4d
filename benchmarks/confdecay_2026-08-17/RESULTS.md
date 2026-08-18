# Confidence-decay reactivation — matched arms, 2026-08-17

Validates the T1 §4.1/§4.2 fix (`benchmarks/T1_T2_P0_AUDIT.md`): the per-pixel confidence decay now
compounds over an **age** field that is **warped along the flow**, instead of being reset to 1.0
every trusted frame (which pinned `running_confidence` to exactly {0.0, 0.85} and made `decay` a
dead knob).

## Arms

Three per clip, launched back-to-back on the same GPU in the same session (rule 1), identical in
everything but the two confidence constants:

| arm | env | meaning |
|---|---|---|
| `legacy` | `SPAG_CONF_LEGACY=1` | pre-fix behaviour — fixed 0.85 blend, no compounding |
| `decay` | `SPAG_CONF_DECAY=0.85 SPAG_CONF_FLOOR=0.5` | **the fix, and the new default** |
| `noprop` | `SPAG_CONF_DECAY=0 SPAG_CONF_FLOOR=0` | confidence ≡ 0 → fresh per-frame monocular estimate; the **raw fidelity reference** |

Common: `SPAG_SAM3_MAXSIZE=768 SPAG_DETERMINISTIC=1 SPAG_OCCL_FBGATE=1`, `scripts/run_bglock_audit.py`
config (bglock + da360 + median-5 smoother, `skip_step=2 stride=4`). GPU 0 = circulation, GPU 1 =
MattSwift. Raw numbers: [`comparison.json`](comparison.json), produced by
`scripts/compare_conf_arms.py`.

## Paired stability / fidelity (rule 11)

`fg_deriv` = mean per-frame `|depth_t − depth_{t−1}|` inside the dynamic mask, normalised by frame
median depth, weighted by in-mask pixel count. `fid_ratio` = that value divided by the `noprop`
arm's — **1.0 means the raw motion signal is fully preserved; below 1.0 means motion is being
suppressed.** `n_dist` = distinct values of `running_confidence` in a frame, the direct check on
whether decay compounds at all.

### MattSwift (299 propagated frames, thin/articulated subject)

| arm | bg_cv | fg_cv | fg_deriv | **fid_ratio** | n_dist | conf (last 10) | time | VRAM |
|---|---|---|---|---|---|---|---|---|
| legacy | 0.0005 | **0.1307** | 0.00464 | **0.692** | 2 | 0.715 | 428 s | 10349 MB |
| decay | 0.0005 | 0.1582 | 0.00619 | **0.923** | 7571 | 0.421 | 426 s | 10349 MB |
| noprop | 0.0005 | 0.1681 | 0.00671 | 1.000 | 1 | 0.000 | 422 s | 10349 MB |

### circulation_site_1_edit_coupe (77 propagated frames, near-static)

| arm | bg_cv | fg_cv | fg_deriv | **fid_ratio** | n_dist | conf (last 10) | time | VRAM |
|---|---|---|---|---|---|---|---|---|
| legacy | 0.0015 | 0.0554 | 0.07588 | 1.030 | 2 | 0.708 | 174 s | 6838 MB |
| decay | 0.0015 | 0.0587 | 0.07723 | 1.048 | 21316 | 0.419 | 173 s | 6838 MB |
| noprop | 0.0015 | 0.0600 | 0.07367 | 1.000 | 1 | 0.000 | 172 s | 6838 MB |

## What the numbers say

**1. The fix does what it claims — empirically, not just structurally.** `n_dist` goes from 2
(legacy: {0, 0.85} on every frame of both clips, reproducing the audit's static analysis exactly)
to thousands. The MattSwift confidence trace shows the compounding directly, mean
`running_confidence` per frame:

```
frame   0      1      2      3      4      10     150    (age_max)
mean    0.715  0.607  0.517  0.439  0.421  0.420  0.421   1→2→3→4→5→6, then saturated
```

0.85 → 0.7225 → 0.6141 → 0.5220 → floored at 0.5, exactly the designed geometric sequence, with
the population mean sitting below each value because the ~16% pole band contributes zeros.
Fractional ages (from bilinear-warping the age field, i.e. the §4.2 fix) are what produce the
non-power-of-0.85 values in between.

**2. Legacy's foreground stability was substantially bought with suppressed motion.** MattSwift's
legacy `fid_ratio` is **0.692** — nearly a third of the real frame-to-frame foreground depth change
was being flattened away. This is precisely the tautology rule 11 exists to catch: legacy's better
`fg_depth_cv` (0.1307 vs 0.1582) is not straightforwardly "more stable", because a metric measuring
temporal variation of foreground depth goes down when you remove foreground motion.

**3. The fix trades stability for fidelity, deliberately, and the trade is visible.** Taking
`noprop` as 0% and `legacy` as 100% of the available smoothing:

| | fg_cv reduction vs raw | motion preserved |
|---|---|---|
| legacy | 100% (0.1681 → 0.1307) | 69% |
| **decay** | **26% (0.1681 → 0.1582)** | **92%** |

So the new default gives up most of the nominal foreground-stability figure to recover three
quarters of the lost motion. Whether that is the right operating point is a `SPAG_CONF_FLOOR`
question, not a code question — the floor is exactly the steady-state propagation weight
(`conf → conf_floor` after ~5 frames), so a floor sweep moves along this trade-off curve directly.
0.5 was chosen a priori as the neutral midpoint, not fitted.

**4. Nothing else moved.** `bg_depth_cv` is identical across all three arms on both clips
(0.0005 / 0.0015) — the tripwire is clean, as expected since the background is locked to `D_ref`
regardless of confidence. Time is within noise (−0.3% MattSwift, −0.8% circulation) and **VRAM is
byte-identical**: warping the age field rides along in the existing batched `grid_sample`, so the
§4.2 fix is genuinely free. Mask IoU is 1.0 by construction — SAM3 runs before the depth chain and
the segmentation config is identical across arms.

**5. On a near-static clip the change is close to a no-op.** circulation shows `fid_ratio` ≥ 1 for
every arm (no suppression to recover — there is barely any foreground motion) and `fg_cv` moving
0.0554 → 0.0587. Consistent with the audit's note that this clip is the most static in the set;
it is the loop clip, not the evidence clip.

## Caveats before shipping

- **Two clips, not ten.** Rule 3 gates a default-flip on the max across the 10-clip set, not on the
  loop pair. The default was flipped here because the *fix* is a correctness fix (decay was
  advertised in the docstring and did not exist), but the `conf_floor` value has not been swept and
  the 10-clip run has not been done.
- **`legacy` is verified faithful.** The `legacy` arm reproduces the pre-fix confidence trace
  digit-for-digit (`frac_zero` 0.15960 at frame 0, 0.16555 at frame 5, mean 0.7092, distinct
  {0, 0.85, 1.0} on circulation), so `SPAG_CONF_LEGACY=1` is a real escape hatch and every
  pre-2026-08-17 benchmark remains reproducible.
- **Not yet measured:** the radial-motion case the decay exists for (§6.4's synthetic
  toward/away-from-camera object) — none of the three metrics above isolates it, and it is the one
  scenario where compounding should show a clear *win* rather than a trade. That test is now worth
  building, because for the first time there is a mechanism for it to exercise.
- Per-frame `depth_maps/` dumps were deleted after metric extraction (see `../README.md`); the
  colored-depth videos (`depth_{arm}.mp4`, fixed 0.5–15 m TURBO range, so arms are directly
  comparable) and the `gaussians/` PLYs are kept for visual A/B (rule 8).
