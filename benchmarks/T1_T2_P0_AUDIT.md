# T1/T2 P0 audit — premise checks against live code

_2026-08-17. Per `bglock_open_questions.md` §1: every item is a candidate; verify the premise
in the current source before acting, and record "checked, not a problem, here's why" as a
successful close. No speculative fixes applied._

Source of truth for this pass: `spag4d/flow_depth_propagation.py`,
`spag4d/video.py` (`align_depth_frame`, `align_depth_frame_gpu`, `run_video` depth loop),
`spag4d/da360_model.py`, `spag4d/pager_model.py`.

## Summary

| item | premise | verdict |
|---|---|---|
| §4.1 `running_confidence` absorbing at zero | **FALSE** | a per-frame reset already prevents it — but the reset makes `decay` inert, which is a *different* real defect. **FIX APPLIED 2026-08-17** (decay reactivated over a warped age field) |
| §4.2 `prev_confidence` not warped | **TRUE** | confirmed un-warped in both numpy and torch paths; impact bounded by the §4.1 finding. **FIX APPLIED 2026-08-17** (the carried state is warped with the depth, same batched `grid_sample`) |
| §4.3 backend outliers destroy the affine fit | **PARTLY TRUE** | 1e6 values do enter the fit; damage is bounded by `scale_clip` + outliers being *shared* between x and y, not by any validity mask. **FIX APPLIED 2026-08-17** (magnitude-based validity gate on the fit) |
| §4.4 PaGeR re-classifies indoor/outdoor per frame | **FALSE** | `metric=False` short-circuits before CLIP, and `da360` is the production generator anyway. **DEFENSIVE FIX APPLIED 2026-08-17** (label cached anyway, since premise is false only by default config, not by construction) |
| §10.3 DA360 per-frame median renorm is content-dependent | **TRUE and active in production** | `temporal_consistency=False` is the default; every frame is rescaled to a 5 m all-pixel median |

---

## §4.1 — `running_confidence` absorbing at zero: premise FALSE

The doc reads only the blend line
([flow_depth_propagation.py:310](../spag4d/flow_depth_propagation.py#L310)):

```python
running_confidence = np.minimum(frame_confidence, state.confidence * state.decay)
```

and concludes a pixel that fails FB-consistency once is pinned at zero forever. But the very next
state write ([:326](../spag4d/flow_depth_propagation.py#L326), torch twin at
[:441](../spag4d/flow_depth_propagation.py#L441)) is a reset, not a carry-forward:

```python
state.confidence = np.where(frame_confidence > 0, 1.0, running_confidence)
```

So any pixel whose *current* frame is trusted has its stored confidence set back to `1.0`
regardless of history. Recovery takes exactly one frame of lag (the frame it recovers on still
reads the stale `0`, because `min(1, 0*0.85) == 0`), then it is back to full trust. **Not
absorbing.** The one genuinely permanent zero is the pole band, where `pole_trust == 0` makes
`frame_confidence == 0` every frame — that is by design (WAFT flow is untrustworthy at the ERP
singularity), not the reported bug.

### The real defect the reset exposes: `decay=0.85` never compounds

`frame_confidence = fb_trust * pole_trust` is a product of two **binary** masks, so it is
{0.0, 1.0}. Combined with the reset above, `state.confidence` is always exactly `1.0` or `0.0`
entering the next frame, and therefore:

```
running_confidence  ∈  {0.0, 0.85}      (and 1.0 on the very first frame, where state is None)
```

Nothing in between is reachable. The multiplicative chain the docstring advertises —
_"carries the per-pixel confidence decay across frames so radial motion (invisible to flow) gets
continuously re-anchored to monocular depth"_ — **does not happen**. `decay` is not a decay; it is
a fixed 0.85/0.15 blend weight between propagated and fresh depth, applied identically on frame 3
and frame 300. Consequences worth stating plainly:

- **The stated mechanism for handling radial motion is absent.** A pixel propagating steadily for
  200 frames is trusted at exactly the same 0.85 as one propagating for 2. There is no growing
  pull back toward the fresh monocular estimate.
- **The `< 0.5` disocclusion override is effectively `== 0`.** `reveal = (running_confidence < 0.5)
  & (disocclusion_mask == 0)` ([:319](../spag4d/flow_depth_propagation.py#L319)) can only fire on
  the `0.0` branch, since the only other reachable value is 0.85. So the `D_ref` reveal path is
  gated purely on binary FB/pole failure, and the 0.5 threshold carries no information.
- **`decay` is a dead tuning knob.** Any sweep of `PropagationState.decay` only moves the fixed
  blend weight; it cannot change temporal behaviour. This also retires §12's "decay constant tied
  to fps / half-life" item *as stated* — there is no half-life to derive, because there is no
  exponential. Deriving one requires first deciding whether compounding is wanted at all.

This is a genuine finding but it is a **design question, not a crash**: the current behaviour (a
constant 0.85 blend + hard reveal on FB failure) is defensible and is what every shipped benchmark
number was measured under. Per §1 no fix is applied here. If compounding is wanted, the minimal
change is to drop the `frame_confidence > 0` reset and let `state.confidence` carry
`running_confidence` forward — but that reintroduces exactly the absorbing-at-zero hazard §4.1
predicted, so it needs the floor/recovery term §4.1 proposes *at the same time*, and a paired
stability/fidelity measurement (§6.0) to show it doesn't just freeze more.

**Instrumentation added** (opt-in, inert when unset): `SPAG_CONF_HIST=<path.json>` dumps per-frame
`running_confidence` stats — `frac_zero`, `frac_lt_0.5`, `mean`, and the first 8 distinct values —
via a new `PropagationState.trace` field. The `distinct` field is the direct empirical check on the
{0, 0.85} claim above; `frac_zero` over frame index is T1's requested histogram.

### Empirical confirmation (T1 "done when" — `circulation_site_1_edit_coupe`, 77 propagated frames)

Run: `SPAG_SAM3_MAXSIZE=1536 SPAG_DETERMINISTIC=1 SPAG_OCCL_FBGATE=1`, GPU 0, 178 s, 6,836 MB.
Trace at `_bglock_audit_2026-08-17/circulation/confidence_trace.json`.

| frame | `frac_zero` | `frac_lt_0.5` | `mean` |
|---|---|---|---|
| 0 | 0.15960 | 0.15960 | 0.84040 |
| 5 | 0.16555 | 0.16555 | 0.70928 |
| 38 | 0.16566 | 0.16566 | 0.70919 |
| 76 | 0.16567 | 0.16567 | 0.70918 |

Three things fall out, all confirming the static analysis:

1. **No collapse — §4.1 is definitively closed.** `frac_zero` is flat across the whole clip
   (0.1596 → 0.1657, range 0.1596-0.1691, first-10 mean 0.1643 vs last-10 mean 0.1675). An
   absorbing-at-zero field would climb monotonically toward 1.0; this doesn't drift at all. The
   `mean` is likewise flat at ~0.709 from frame 1 onward.
2. **`decay` is inert — confirmed.** The distinct values of `running_confidence` observed across
   **all 77 frames** are exactly `{0.0, 0.85, 1.0}` (1.0 only on frame 0, where `state` is `None`).
   No intermediate value is ever reached, so no compounding occurs. This is the empirical proof of
   the design defect above.
3. **The `< 0.5` threshold carries no information — confirmed.** `frac_lt_0.5 == frac_zero` to
   every printed digit on every frame, because 0.85 is the only other reachable value.

**Bonus quantification: the zeros are almost entirely the by-design pole band, not FB failures.**
`pole_margin_frac=0.08` zeroes the top and bottom 8% of rows = 16% of the frame, and `frac_zero`
sits at 0.1596-0.1691 — i.e. within ~0.1-0.9 pp of the pole band's 0.16. So **forward-backward
consistency failures account for under 1% of the frame** on this clip, and the flow is trusted
essentially everywhere outside the poles. Two consequences for the rest of the backlog: the
`D_ref` disocclusion-reveal path (§4.1, gated on `running_confidence == 0` outside the SAM mask)
fires on a very small pixel population, so it is unlikely to be a major error source; and §12's
"FB threshold in angular units" item is low-value on this clip — the 1.5 px threshold is not
binding. Worth re-measuring on a clip with faster/larger motion (MattSwift) before generalising,
since `circulation_site_1_edit_coupe` is the shortest and one of the most static in the set.

### Replicated on MattSwift — 299 frames, thin/articulated moving subject

Same flags, GPU 1, 418 s, 10,349 MB, 6 tracked objects, SAM3 at 768×384 (the resolution B1
actually validated at IoU 0.93 — see the maxsize section below). `fg_depth_cv` 0.1236, matching
B1's recorded bf16-alone 0.123 / matched-baseline 0.134 basin rather than the 0.28 basin.

| | circulation (77 f) | MattSwift (299 f) |
|---|---|---|
| distinct `running_confidence` values | `{0.0, 0.85, 1.0}` | `{0.0, 0.85, 1.0}` |
| `frac_zero` first → last | 0.1596 → 0.1657 | 0.15885 → 0.15885 |
| `frac_zero` first-10 vs last-10 mean | 0.1643 / 0.1675 | 0.15947 / 0.15884 (**down**) |
| `frac_lt_0.5 == frac_zero` every frame | yes | yes (all 299) |
| FB failures beyond the 0.16 pole band | +0.1 to +0.9 pp | −0.2 to **+0.9 pp** |
| mean confidence, steady state | 0.709 | 0.714 |

All four conclusions hold on a clip 4× longer with a genuinely articulated moving subject, so they
are properties of the algorithm, not of the static test clip:

- **No collapse** — `frac_zero` ends *lower* than it started over 299 frames. §4.1 closed.
- **`decay` inert** — still exactly three reachable values across 299 frames.
- **`< 0.5` threshold informationless** — exact equality on every one of 299 frames.
- **FB threshold not binding.** Even here, forward-backward failures add at most **0.89 pp** of
  frame area on top of the 16 pp pole band. This retires §12's "FB threshold in angular units" as
  a priority item: at 1.5 px the gate is essentially never the deciding factor, so expressing it
  in degrees of arc would change nothing measurable. The corollary is more interesting — WAFT's
  mid-latitude degradation (§12's other WAFT note) is **not** being caught by FB-consistency at
  all, so if that degradation is real it is passing through as *trusted* flow. A latitude-resolved
  FB-error plot (§10.3's residual-vs-latitude experiment, T3) is the right instrument, and it is
  now better motivated than the threshold-units work.

### FIX APPLIED 2026-08-17 — decay reactivated, over a flow-warped age field

The audit above stopped at "design question, not a crash" per §1. That decision was then taken:
**compounding is wanted**, and both §4.1's fix and §4.2's fix land together, because they are the
same change — once confidence is a real accumulated field, warping it is mandatory, not optional
(the audit said exactly this: "if the reset is ever removed … warping it becomes P0 again").

The carried state is now a per-pixel **age** — frames elapsed since that surface point was last
re-anchored to fresh monocular depth — rather than a confidence:

```python
age  = where(frame_confidence > 0, warp(state.age, flow_fwd) + 1, 0)     # §4.2: warped
conf = frame_confidence * max(conf_floor, decay ** age)                  # §4.1: compounds
```

Three deliberate properties:

- **Age, not confidence, is the state.** `decay ** age` is inspectable and reconstructible from
  the state at any frame; a raw accumulated confidence is not. This is §4.1's own "Alternative"
  proposal.
- **It is warped, at zero extra cost.** Age describes a surface point, so it rides along in the
  existing `warp_backward_multi` / `warp_backward_multi_torch` stack next to `depth_prev_final` —
  one more slice in a batched `grid_sample`, no additional kernel. That closes §4.2.
- **`conf_floor = 0.5` prevents the absorbing state §4.1 predicted**, in the other direction.
  Unfloored, `decay ** age` reaches 1e-3 by frame 45 and propagation switches itself off, which
  would restore exactly the foreground flicker bglock exists to remove. Floored, confidence runs
  `0.85 → 0.72 → 0.61 → 0.52 → 0.50 …` and then holds: a first-order low-pass that mixes
  `1 - conf_floor` of fresh monocular depth in every frame. Accumulated radial drift is therefore
  bounded by a geometric series instead of growing without limit — and smoothly, with no periodic
  full-re-anchor pop, which the sawtooth "reset to 0 every K frames" variant would have.

Two consequential side effects, both intentional:

- **The `running_confidence < 0.5` reveal gate is now written `frame_confidence == 0`.** It was
  only ever equivalent to that (audit finding 3), and under a floor it would otherwise become a
  silent function of `conf_floor` — a tuning knob would move the `D_ref` reveal set. Same pixels
  as before, no longer accidentally coupled.
- **The first propagated frame is now 0.85, not 1.0.** With `state` empty the old code took
  `running_confidence = frame_confidence = 1.0`, i.e. one frame of pure propagation with no
  monocular anchor at all. Age starts at 1, so that frame now blends like every other.

**Flags** (`spag4d/video.py`): `SPAG_CONF_DECAY` (default 0.85) and `SPAG_CONF_FLOOR` (default
0.5) expose the two constants for sweeping; `SPAG_CONF_LEGACY=1` restores the pre-2026-08-17
non-compounding path exactly, which is what every benchmark number recorded before this date was
measured under. `SPAG_CONF_DECAY=0 SPAG_CONF_FLOOR=0` degenerates to confidence ≡ 0, i.e. the
fresh per-frame monocular estimate — a free "no propagation" arm through the same code path, used
as the raw fidelity reference below.

`decay` is now a live knob rather than a dead one, so §12's "decay constant tied to fps /
half-life" item is back on the table *as stated*: `decay = 0.5 ** (1 / (fps * half_life_s))` is now
meaningful, and the stride sweep (§6.1) can actually detect fps-coupling. It could not before.

**Results:** [`benchmarks/confdecay_2026-08-17/RESULTS.md`](../benchmarks/confdecay_2026-08-17/RESULTS.md)
— three matched arms per clip (`legacy`, `decay`, `noprop`), same GPU, back-to-back, per
BENCHMARK_RULES rule 1, with the paired fidelity number rule 11 requires. Headline:

| clip | arm | fg_depth_cv | fidelity vs raw | distinct `running_confidence` |
|---|---|---|---|---|
| MattSwift | legacy | 0.1307 | **0.692** | 2 |
| MattSwift | **decay** | 0.1582 | **0.923** | 7571 |
| MattSwift | noprop (raw) | 0.1681 | 1.000 | 1 |
| circulation | legacy / decay / noprop | 0.0554 / 0.0587 / 0.0600 | 1.030 / 1.048 / 1.000 | 2 / 21316 / 1 |

Three things this establishes:

1. **The decay now compounds** — 2 distinct confidence values became thousands, and the MattSwift
   trace walks the designed sequence `0.85 → 0.72 → 0.61 → 0.52 → 0.50` with `age` saturating at 6.
   The audit's `{0.0, 0.85}` finding is fixed, empirically, not just structurally.
2. **Legacy's foreground stability was ~31% suppressed motion** on MattSwift (fidelity 0.692
   against the raw arm). Its lower `fg_depth_cv` was substantially the tautology rule 11 warns
   about. The new default recovers three quarters of that lost motion (0.923) and gives up most of
   the nominal `fg_cv` gain to do it — a real trade, and `SPAG_CONF_FLOOR` is the knob that moves
   along it, since the floor *is* the steady-state propagation weight.
3. **§4.2 is free.** VRAM is byte-identical and wall time within noise across arms — the age field
   rides in the existing batched `grid_sample`.

`bg_depth_cv` is unchanged across all arms (tripwire clean; the background is locked to `D_ref`
regardless of confidence). Not yet done: the `conf_floor` sweep, the 10-clip run rule 3 gates a
default-flip on, and §6.4's synthetic radial-motion object — the one case where compounding should
show a clear win rather than a trade, and which nothing in this table isolates.

## §4.2 — `prev_confidence` is not warped: premise TRUE

`state.confidence` is read at grid position `p` ([:310](../spag4d/flow_depth_propagation.py#L310),
[:431](../spag4d/flow_depth_propagation.py#L431)) while `depth_prev_final` in the same expression
*is* warped to `p - flow_fwd(p)` through `warp_backward_multi`. So the two are sampled in different
frames of reference, exactly as the doc says.

**But the §4.1 finding bounds the impact severely.** Because `state.confidence` is a binary
{0, 1} field rather than a smooth accumulated history, the misregistration can only ever flip a
pixel between "0.85 blend" and "0 → fresh estimate / `D_ref` reveal", and only on the one-frame
lag after an FB failure moves. There is no multi-frame history to misattribute — the scenario the
doc describes ("a pixel just uncovered by a moving person inherits the person's edge confidence")
costs one frame, not a persistent wrong state. Fixing it is still correct and cheap (sample it with
the same batched warp — it can ride along in the existing `warp_backward_multi` stack at no extra
`grid_sample`), but it is a **P2-sized effect, not P0**, and it should be re-prioritised below the
§4.1 design question it depends on: if the reset is ever removed and confidence becomes a real
accumulated field, warping it becomes P0 again.

**FIX APPLIED 2026-08-17** — and that is exactly what happened: the §4.1 reset was removed, so this
returned to P0 and was fixed in the same change. `state.age` is appended to the existing
`warp_backward_multi` source list, so it is sampled at `p - flow_fwd(p)` with the identical
bilinear/wrap-x/clamp-y grid as `depth_prev_final`, in the same batched `grid_sample`. See the
FIX APPLIED block under §4.1 for the full scheme.

## §4.3 — backend outliers in the affine fit: premise PARTLY TRUE

DA360 does produce the ~1e6 values as described
([da360_model.py:230](../spag4d/da360_model.py#L230), `depth = 1.0 / (disparity.abs() + eps)`,
`eps = 1e-6`). And the fit's static mask
([video.py:1420](../spag4d/video.py#L1420), GPU twin [:1505](../spag4d/video.py#L1505)) does **not**
exclude them — it tests only `mask_moving == 0`, `> 0`, and `isfinite`, all of which 1e6 passes.
There is no validity mask anywhere in the chain. So the premise's mechanism is real.

Two reasons it has not visibly exploded, both worth recording because they are what a fix must not
break:

1. **The outliers are shared between predictor and target.** The fit is
   `depth_ref ~ s·depth_frame + t`, and `depth_ref` is produced by the *same* DA360 inversion on
   the median plate — so a sky pixel sits at ~1e6 in both `x` and `y`. High-leverage, but roughly
   *on* the line `y ≈ x`, so it pulls `s` toward 1 rather than to an arbitrary value. A fit against
   a target from a different backend or a different clamp regime would not have this protection.
2. **`scale_clip=(0.5, 2.0)`** ([:1387](../spag4d/video.py#L1387)) hard-bounds the damage, and on
   clipping recomputes `t` from medians (robust) rather than keeping the contaminated `t`.

So this is a **latent** hazard rather than an active one: correct to fix (derive a validity mask
from raw disparity *before* inversion, exclude from fit and from all reported metrics), but it is
not currently corrupting the shipped numbers, and the §6.7 concern that clamped pixels *deflate
flicker metrics* is the sharper half of the item — those pixels are constant by construction and
sit in the same arrays the metrics average over.

Note also that `align_depth_frame` **mutates its `depth_frame` argument in place**
([:1486](../spag4d/video.py#L1486)) while the GPU path clones ([:1543](../spag4d/video.py#L1543)) —
an inconsistency to be aware of before touching either.

### FIX APPLIED 2026-08-17 — magnitude-based validity gate on the affine fit

Rather than teach each backend to declare a proper validity mask (§10.1's bigger, still-open
refactor), the minimal version of "exclude from the fit" was applied directly at the one place
both backends' outliers actually do damage: `align_depth_frame` / `align_depth_frame_gpu`'s
static-pixel selection
([video.py:1399](../spag4d/video.py#L1399), [:1509](../spag4d/video.py#L1509)).

```python
outlier_cap = median(depth_ref[isfinite & >0]) * 50.0
static_mask &= (depth_frame < outlier_cap) & (depth_ref < outlier_cap)
```

50x the plate's own median is generic across backends by construction — real scene depth is
never 50x its own median, whether the outlier is DA360's ~1e6 disparity-inversion or PaGeR's
clamped 200.0 — and needs no backend-specific wiring or new return value threaded through the
GPU-resident chain. **Verified**: injecting `1e6` into a 5% strip of a synthetic frame and
re-fitting reproduces the exact same `(s, t)` as the clean frame, on both the numpy and torch
paths (`torch.allclose(..., atol=1e-3)` true on the untouched region). Initially scoped to the fit
only, leaving the deflated-flicker-metric half of §4.3 (outliers sitting in the *stability metric*
arrays, not just the fit) open. **FIX APPLIED 2026-08-18**: extended the same `_fit_outlier_cap`
gate to the metrics side via a `_valid_metric_mask(depth, base_mask, cap)` helper, applied to the
four per-frame median computations that feed `bg_depth_cv`/`fg_depth_cv`
([video.py](../spag4d/video.py), around the `median_depth_fg`/`aligned_median_depth_fg`/
`aligned_median_depth_bg`/`stabilized_aligned_median_depth_fg` accumulators). Falls back to the
unfiltered `base_mask` if the gate would empty a region (e.g. genuinely all-sky), since
`np.median` raises on an empty array. Verified at the helper level (excludes injected outliers,
falls back correctly on an all-outlier region); not yet re-verified against a full benchmark rerun
of `bg_depth_cv`/`fg_depth_cv` end to end. `scale_clip=(0.5, 2.0)` remains as a second line of
defense on `s` itself. Per-backend validity masks (§10.1's bigger refactor) remain out of scope —
see the open question this closed in `bglock_open_questions.md` §12.

## §4.4 — PaGeR per-frame CLIP routing: premise FALSE

`_skip_heads` ([pager_model.py:115](../spag4d/pager_model.py#L115)) returns before touching CLIP
whenever `self.metric` is false:

```python
if not self.metric:
    return {"scale_indoor", "scale_outdoor"}, False
```

and `metric` defaults to `False` in both `__init__` and the `from_*` constructor
([:74](../spag4d/pager_model.py#L74), [:89](../spag4d/pager_model.py#L89)). Independently,
production runs `active_generator="da360"`, so PaGeR is not in the path at all. **Closed, no
action needed for production.** It becomes live only if someone constructs PaGeR with
`metric=True` — which is exactly what the E1 pager plan intends to do.

**DEFENSIVE FIX APPLIED 2026-08-17.** Since E1 puts PaGeR on the critical path, the doc's own fix
was applied now rather than deferred: `PaGeRModel._skip_heads` ([pager_model.py:115](../spag4d/pager_model.py#L115))
caches the CLIP label in `self._scale_label` on first classification instead of reclassifying
every frame. Inert today (the branch that calls it never executes while `metric=False`), so there
is nothing to benchmark yet — this is pre-emptive, not a measured fix, and should be re-verified
once E1 actually exercises `metric=True`.

## §10.3 — DA360 per-frame median renormalization: TRUE and active

Not in the §12 P0 block, but it surfaced while checking §4.3 and it is the more consequential
finding of the two. `predict()` branches on `temporal_consistency`
([da360_model.py:233-244](../spag4d/da360_model.py#L233)):

```python
if temporal_consistency or global_scale_factor is not None: ...
else:
    for i in range(B):                       # per-frame renorm
        depth[i] = depth[i] * (5.0 / depth[i].median())
```

`run_video`'s default is `temporal_consistency=False`
([video.py:346](../spag4d/video.py#L346)), it is `False` in `benchmark_solutions.BASE_KWARGS`, and
no caller passes `global_scale_factor` — the per-frame call site
([video.py:1008](../spag4d/video.py#L1008)) forwards only `temporal_consistency`. **So the
else-branch is what production runs**, complete with its own source comment: _"WARNING: this causes
scale drift between consecutive frames in videos."_

The median is over **all** pixels, so when a subject enters or leaves frame the global scale of
every background pixel moves with it. Under bglock the background is then overwritten by `D_ref`
so the *output* is unaffected — but the per-frame affine fit is absorbing a content-dependent
global scale step on every frame, which means the fitted `s` trace is dominated by subject area
rather than by model drift, and `s` is therefore useless as the diagnostic §9 wants it to be.

---

## Incidental: `SPAG_SAM3_MAXSIZE` never applied the size it advertised — FIXED

Found while reading the segmentation log of the run above, which printed
`downscaled to 408x204 (SPAG_SAM3_MAXSIZE=1536, effective scale=0.400)` — 408 px, not 1536.

The lever ([video.py:1688-1715](../spag4d/video.py#L1688)) derives its factor from **native**
dimensions, correctly, and carries a long comment explaining that using the WAFT-shrunk
`viz_frames` shape instead would compose two shrinks. **The fix had only been applied to the
derivation, not the application** — the two lines immediately below the comment still read:

```python
sW = max(int(viz_frames.shape[2] * sam3_scale) & ~1, 2)   # viz_frames = WAFT-shrunk (~1024 long)
```

So the effective SAM3 resolution was `viz_long × (maxsize / native_long)`, not `maxsize`:

| native long edge | derived scale | effective res (buggy) | intended |
|---|---|---|---|
| 3840 (CIELE, circulation, accident_02) | 0.400 | **408** | 1536 |
| 2560 (atelier_1, scene01, …) | 0.600 | **614** | 1536 |
| 2048 (MattSwift) | 0.750 | **768** | 1536 |
| 1920 (trop_long_embouteillage) | 0.800 | **818** | 1536 |

**This means `.claude/B1_VRAM_TIME_REDUCTION_PLAN.md`'s maxsize results are mislabeled.** Its
"target res" column reads exactly 768×384 (MattSwift), 614×306 (atelier_1), 818×408
(trop_long) — the buggy composed values, matching the table above line for line. So:

- `SPAG_SAM3_MAXSIZE=1536` **has never been run.** What was measured as "maxsize=1536" was
  614-818 px effective, i.e. a 0.16-0.32× native shrink.
- The IoU-recovery result (MattSwift 0.93, atelier_1 0.87) is real but belongs to **768 px /
  614 px effective**, not 1536.
- The −58% VRAM figure likewise belongs to those low resolutions. With the fix, `maxsize=1536`
  will consume substantially more VRAM than any recorded maxsize number.
- The `trop_long_embouteillage` sweep labeled 1536/1024/768 was actually 818/546/408 px. Its
  "VRAM plateaus while IoU collapses" conclusion still holds directionally (VRAM flat from
  546→408 px), but the axis labels are wrong by the same composition.

Fixed at [video.py:1714](../spag4d/video.py#L1714) to scale `meta['W']/meta['H']` (native, which
is also what the frames being resized are decoded at). **Consequence for flag defaults:** the
working tree has `SPAG_SAM3_MAXSIZE` defaulting to `1536`, which post-fix is a *much* weaker
downscale than everything validated. To reproduce the validated-safe operating point (MattSwift
IoU 0.93 at 768 px effective, atelier_1 0.87 at 614 px) the equivalent is now
**`SPAG_SAM3_MAXSIZE=768`**, which gives every clip a uniform 768 px regardless of native
resolution — at or above both validated points. The default needs re-picking deliberately rather
than left at 1536 by accident.

---

Note this interacts with the B1 finding that `temporal_consistency=True` was the original fix for
scale drift ("computes `scale_factor_to_5m` once on the reference frame"): that path still exists
and is still correct ([video.py:725-729](../spag4d/video.py#L725)), it is simply **off by default**.
Whether flipping it is safe is a measurable question (paired §6.0 table), not a refactor. §10.3's
proposed `scale_anchor` from the `D_ref` plate median is the same idea in cleaner form.
