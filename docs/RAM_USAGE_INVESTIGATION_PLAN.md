# RAM usage investigation plan

## TL;DR — current understanding (2026-09-01)

RAM is dominated by **three whole-clip buffers**, each built up-front and held
resident for the entire run before the per-frame loop even starts. The
per-frame loop itself costs **zero** additional RAM — verified flat on every
clip traced so far.

| buffer | scales with | formula (bytes) | CIELE example (3840×1920, 204 frames) |
|---|---|---|---:|
| `viz_frames` (decoded video) | frames × native res | `n_frames × W × H × 3` | 4.2 GiB |
| `waft_frames` (WAFT input, downscaled) | frames × working res | `n_frames × new_W × new_H × 3` | 0.3 GiB |
| **flow buffer** `flows_fwd`/`flows_bwd` | frame-pairs × res² | `n_pairs × 2 × W × H × 2ch × 4B` | **22.3 GiB → 1.6 GiB (fixed)** |
| **mask buffer** `outputs_per_frame` (SAM3) | frames × tracked objects × res | `n_frames × W×H × n_objects` | **16.4 GiB (not yet fixed)** |

**Key facts:**
- **Resolution cost is squared** (`W×H` term) — halving resolution quarters
  the buffer. Frame count is only linear. This is why the flow-buffer fix
  (native → ≤1024px working res) gave a clean **~13.8× reduction**, matching
  the **~13.7× pixel-count** reduction almost exactly.
- **Per-frame loop cost is zero.** `ru_maxrss` is flat from `before_frame_loop`
  through the last frame on every clip traced (control clip, CIELE, and
  `vid360_bruit_operatrice`). All RAM is committed during setup, not the loop.
- **The flow buffer's 22.3 GiB was never WAFT's compute cost** — WAFT always
  runs at the small downscaled resolution. The 22.3 GiB was a *post-processing*
  bug: every flow field got upscaled to native res and kept in memory for the
  whole clip, immediately after WAFT produced it. The fix defers that upscale
  to a lazy, per-frame, throwaway operation at the point of use instead of
  doing it once for the whole clip and retaining the result — **shipped, not
  yet defaulted on** (needs the standard benchmark pass).
- **`ram_trace.csv`'s checkpoint deltas overstate retained cost** — they
  include transient allocations that get freed again, and miss non-Python-heap
  memory entirely (PyTorch/SAM3 native allocations). Trust the final
  `ram_max_mb` (real OS RSS) over per-checkpoint deltas for "how much did X
  cost."

**Status:**
- ✅ Mask-buffer fix (lazy resize instead of eager whole-clip upsample) — done
- ✅ Flow-buffer fix (lazy upscale instead of eager whole-clip upscale) — done
- ✅ Verified on `circulation_site_1_edit_coupe` (−26.7% combined, byte-identical output) and `CIELE` (−34.1%, `after_flow_waft` −44.9%)
- ✅ `viz_tensor` chunked downscale fix (see "Whole-clip downscale tensor fix" below) — shipped, default (`_chunk = 8`, no env var)
- ✅ `compute_temporal_median` chunked fix and post-loop `nanstd` chunked fix (see "Two more whole-clip float buffers found and fixed" below) — shipped
- ✅ Combined effect on `CIELE`: **64,053 MB → 28,301 MB peak RSS (−55.8%)**, no time/VRAM cost
- ⬜ Standard `benchmarks/BENCHMARK_RULES.md` pass before becoming default
- ⬜ Mask buffer (`outputs_per_frame`, SAM3) is stored at SAM3's own inference
  resolution; capping that via `SPAG_SAM3_MAXSIZE=1024` (matching WAFT's
  working res) only bought ~6.9% RAM on CIELE — most of what looked like
  "mask buffer cost" was actually the two `temporal_median`/`nanstd` bugs
  below, now fixed. Whether the mask buffer itself is still worth capping
  is unresolved — `SPAG_SAM3_BF16` + `SPAG_SAM3_MAXSIZE` stacking is a
  known-bad combo (see root CLAUDE.md §SAM3_BF16), so this needs its own
  isolated pass, not bundled with bf16.
- ⬜ ~20 GB gap between `tracemalloc`'s accounted allocations and actual `ru_maxrss` on CIELE, still unexplained (likely native PyTorch/SAM3 allocations) — would need different instrumentation (`torch.profiler`, `/proc/self/smaps`) to pin down
- ⬜ Chunk-sweep's chunk=39 anomaly (~3 GB / ~18% run-to-run variance vs. an earlier one-shot baseline measurement on the same code path) — flagged, not chased down

Full chronological findings, code-site detail, and numbers below.

---

## Why

The `bglock_sol1_median_w5` config (+ `freeze_bg_live_color`, SAM3 bf16/maxsize=1536, `skip_step=4`)
was benchmarked per-clip with isolated `ru_maxrss` (one subprocess per video, so the peak
is that video's own footprint — see `scripts/run_atelier1_bglock_sol1.py`). Results across
15 clips, sorted by RAM:

| video | frames | time (s) | VRAM (GB) | RAM (GB) |
|---|---:|---:|---:|---:|
| vid360_bruit_operatrice | 241 | 567.3 | 11.79 | 101.47 |
| CIELE | 204 | 491.5 | 11.60 | 94.21 |
| trop_long_embouteillage | 338 | 758.8 | 28.88 | 79.65 |
| productionPont_2_MAX | 163 | 391.4 | 10.04 | 77.05 |
| risque_securite_machine_2 | 173 | 401.3 | 8.86 | 68.83 |
| Dispo_RDV | 210 | 452.0 | 15.95 | 67.38 |
| boutique1_HQ | 150 | 643.1 | 18.25 | 67.11 |
| tissc | 105 | 255.1 | 7.86 | 50.04 |
| scene01 | 158 | 339.2 | 12.92 | 49.90 |
| scene_03 | 155 | 339.5 | 12.75 | 48.71 |
| atelier_1 | 158 | 345.4 | 12.93 | 46.84 |
| accident_electrique_02 | 94 | 225.0 | 6.58 | 38.93 |
| projections_yeux_2 | 75 | 198.5 | 6.71 | 35.90 |
| MattSwift | 150 | 265.4 | 9.57 | 28.42 |
| circulation_site_1_edit_coupe | 39 | 123.8 | 6.05 | 24.08 |

RAM is the binding resource for capacity planning here, not VRAM (max 28.9GB vs max
101.5GB). It does **not** track frame count cleanly: `trop_long_embouteillage` is the
longest clip (338 frames) but ranks 3rd in RAM, while `vid360_bruit_operatrice` (241
frames) and `CIELE` (204 frames) rank 1st/2nd. Whatever drives RAM, it isn't just
duration — it's correlated with content (SAM3 mask/track count, `freeze_bg` pixel-index
retention, dynamic-object density).

Full per-clip `results.json` under
`benchmarks/new_bchmk_2026-08-31/bglock_sol1_median_w5_livebg_sam3bf16_maxsize1536/<clip>/`.

## Goal

Explain the RAM variance well enough to answer: "given a RAM budget, how many seconds of
video (of what kind of content) can I process?" Currently we can't answer that beyond "it
depends on content, not just length."

## Hypotheses to check, roughly in order of suspected cost

1. **SAM3 tracking state accumulation.** SAM3 keeps per-object memory/track history across
   frames. Clips with more distinct tracked objects (CIELE, vid360_bruit_operatrice —
   likely crowded/high-activity scenes) may accumulate more track memory than long but
   visually static clips like trop_long_embouteillage's static-camera traffic. Check:
   correlate RAM with mean/max concurrent SAM3 track count per clip (loggable from the
   tracker's internal state or by instrumenting `spag4d/detect_*.py`'s SAM3 call sites).

2. **`freeze_bg` pixel_idx retention.** `freeze_bg_live_color` (CLAUDE.md, 2026-08-21 entry)
   adds a `pixel_idx` key surviving the whole Gaussian dict through prune chains. If any
   structure holding per-Gaussian dicts across frames isn't released/overwritten each
   frame (e.g. accumulates history instead of being replaced), RAM would grow with frame
   count AND with per-frame Gaussian count (dynamic-object-heavy scenes producing more
   Gaussians per frame). Check: rerun one heavy clip (CIELE or vid360_bruit_operatrice)
   with `freeze_bg=False` and compare RAM directly — isolates this variable per the
   `benchmark_solutions.py` convention (`BASE_KWARGS` default is `freeze_bg=False`).

3. **Depth/flow/mask buffers not freed per frame.** `depth_npy_dir` was set for this sweep
   (saves `depth_{idx}.npy`, `mask_{idx}.npy`, `flow_{idx}.npy` to disk each frame) — check
   whether the in-memory arrays behind those writes are held onto (e.g. appended to a list
   for later use) rather than freed after the `np.save` call. Check: rerun one heavy clip
   without `depth_npy_dir` set and compare RAM; if it drops significantly, the npy-dump
   path is retaining references it shouldn't.

4. **Per-frame Python object growth (non-tensor).** Gaussian dicts, mask metadata, or
   diagnostic accumulators (anything analogous to `SPAG_DIAG_CSV` from
   `docs/bglock_open_questions.md` §9) that append across the whole run instead of being
   scoped per-frame. Check: `tracemalloc` snapshot diff between frame 10 and frame N-10 on
   a heavy clip (avoids startup-cost noise), or a periodic `objgraph.show_growth()` call
   inside the per-frame loop in `run_video()`.

5. **bf16 SAM3 + maxsize=1536 interaction.** This sweep ran the flagged non-default combo
   (CLAUDE.md 2026-08-18 entry, IoU regression). Unclear if it also has a *host RAM* side
   effect distinct from its known IoU/VRAM issue — check by rerunning one heavy clip at
   default SAM3 settings (`SPAG_SAM3_BF16=0`, `SPAG_SAM3_MAXSIZE=1536` default-disabled →
   0) and comparing RAM. Low prior, but cheap to rule out since it's a one-line env change.

## Method

- Instrument, don't guess: add `tracemalloc` (Python-object-level) and periodic
  `resource.getrusage().ru_maxrss` snapshots at fixed frame intervals (e.g. every 20
  frames) inside `run_video()`'s main per-frame loop, gated behind an env var
  (`SPAG_RAM_TRACE=1`) so it stays zero-cost by default — follow the existing
  `ram_max_mb` / `ConversionResult` pattern (in-code, not a wrapper) per prior session
  guidance ("add ram tracking in code not around").
- Run the trace on the two RAM outliers (`vid360_bruit_operatrice`, `CIELE`) and one
  low-RAM control (`circulation_site_1_edit_coupe`) to get a growth-curve comparison:
  linear-with-frames (leak/accumulation) vs. flat-with-spikes (peak driven by per-frame
  content, e.g. a burst of SAM3 tracks) look very different in a snapshot series and will
  immediately point at which hypothesis above is live.
- Once the growth pattern is identified, isolate the responsible kwarg/code path via a
  single-variable rerun (flip one of: `freeze_bg`, `depth_npy_dir`, `SPAG_SAM3_BF16`) on
  the same outlier clip, matching the ablation convention already used elsewhere in this
  repo (see `benchmarks/BENCHMARK_RULES.md`).

## Non-goals

- Not attempting a general memory-profiling pass over the whole pipeline — scoped strictly
  to explaining the observed RAM variance across these 15 clips.
- Not changing the default config based on this investigation alone; any fix candidate
  goes through the normal benchmark-and-compare convention before becoming a default.

## Findings (short-clip debug pass, 2026-09-01)

Instrumented `spag4d/video.py`'s `run_video()` per the Method section above:
`SPAG_RAM_TRACE=1` (zero-cost when unset) starts `tracemalloc` and writes `ru_maxrss` +
`tracemalloc` snapshots to `<output_folder>/ram_trace.csv` at phase boundaries
(`start`, `after_frame_extraction`, `after_flow_waft`, `after_sam3_segmentation`,
`before_frame_loop`) and every 20th frame inside the main per-frame loop, with the
top-3 `tracemalloc.compare_to(...)` growers since the previous checkpoint.

Debugged on `circulation_site_1_edit_coupe` (39 frames, the shortest/lowest-RAM clip
in the sweep) using the exact config from `scripts/run_atelier1_bglock_sol1.py`.
`ru_maxrss` checkpoint sequence:

| checkpoint | ru_maxrss (MB) | delta | top tracemalloc grower |
|---|---:|---:|---|
| start | 3160 | — | — |
| after_frame_extraction | 3160 | +0 | frame buffer (+823 MiB) |
| after_flow_waft | 10859 | **+7699** | `flow_depth_propagation.py:109` (`upscale_flow`'s `cv2.resize`), +4275 MiB / 152 arrays |
| after_sam3_segmentation | 21504 | **+10645** | `video.py:2963` (native-res mask resize in `segment_with_sam`), +5576 MiB / 1587 arrays |
| before_frame_loop | 25068 | +3564 | background/median build |
| frame_0 … frame_38 | 25068 | **+0 (flat)** | noise only |

**The entire ~25GB peak is resident before the per-frame loop starts, and stays
byte-flat across all 39 frames of the loop.** This rules out hypotheses #1 (SAM3
per-frame track memory growth) and #4 (per-frame Python object growth) for this
clip — both predict frame-loop growth, and there is none. It isn't #2
(`freeze_bg` pixel_idx) either — that structure only exists inside the (flat) loop.

### Root cause: two full-clip buffers built up-front, sized by frame_count × resolution × object_count

1. **`flows_fwd` / `flows_bwd`** (`spag4d/video.py:550`, populated at lines
   612-654): every bidirectional WAFT flow field for the *entire* clip (N-1
   pairs, forward + backward, native ERP resolution) is computed once before the
   frame loop and kept as a live Python list for the whole run, because the loop
   indexes into it later by `idx-1` (`video.py:1280`, `1353`, `1393`, `1433` —
   propagation, static-flow diagnostics, `flow_npy` export). 4.3GB already on a
   39-frame clip; scales with `frame_count × resolution`.

2. **`outputs_per_frame`** (built in `segment_with_sam`, `video.py:2989+`): every
   SAM3 object mask, per tracked object, per frame, upsampled to **native ERP
   resolution** (`video.py:2963`) and held in a dict for the whole run, because
   the loop looks it up later by `idx * skip_step` (`video.py:1113`). 5.6GB
   across 1587 mask arrays on this clip; scales with
   `frame_count × tracked_object_count`, not just frame count — this is exactly
   the content-dependence the 15-clip sweep observed (crowded scenes like
   `CIELE`/`vid360_bruit_operatrice` outranking the longest-but-static
   `trop_long_embouteillage` in RAM). SAM3's own video-predictor state is
   already freed after `segment_with_sam` returns — it's this post-hoc
   native-resolution mask cache that's retained, not internal SAM3 memory.

This is hypothesis #3 as originally suspected (a "buffer not freed per frame"),
but **not** the `depth_npy_dir` npy-dump path named in the plan — the
`flow_{idx}.npy`/`mask_{idx}.npy` writes read from data already resident in
`flows_fwd`/`outputs_per_frame`; the dump doesn't add its own retention. The
real culprits are the two full-clip precompute buffers themselves.

### Capacity-planning implication

RAM scales with `frame_count × resolution` (flow buffer) **plus**
`frame_count × tracked_object_count × resolution` (mask buffer) — not duration
alone. That's why duration under-predicts RAM for content-heavy clips and
over-predicts it for static ones, matching the sweep's original observation,
now with a mechanism attached.

### Fix directions (not implemented — needs the normal benchmark pass)

Both buffers are only needed for random access by frame index later in the
loop, not simultaneous whole-clip residency. Roughly in order of invasiveness:

- Keep flow fields at WAFT's native inference resolution and upsample lazily
  per-frame at point of use, instead of upsampling all N-1 pairs up front
  (relevant when `SPAG_SAM3_SCALE`/`SPAG_SAM3_MAXSIZE` are active; needs
  checking whether this trace's native-res run even hit the resize path or
  short-circuited on matching shapes).
- Same idea for `outputs_per_frame`: keep masks at SAM3's working resolution,
  upsample lazily at `video.py:1113-1117` instead of eagerly for the whole dict
  in `segment_with_sam`.
- More invasive: stream flow/mask results to disk (or a small rolling window)
  instead of materializing the whole clip, since the loop's only real need is
  `idx-1` lookback, not arbitrary random access. Would need the
  `flow_npy`/mask export path re-plumbed too.

Any of these is a candidate, not a decision — standard
`benchmarks/BENCHMARK_RULES.md` benchmark-and-compare pass required before
becoming a default, per Non-goals above.

### Instrumentation added

`SPAG_RAM_TRACE=1` in `spag4d/video.py`: `tracemalloc` started at the top of
`run_video()`; a `_ram_checkpoint(label, prev_snapshot)` helper writes
`ru_maxrss_mb`, `tracemalloc_current_mb`, `tracemalloc_peak_mb`, and the top-3
`tracemalloc.compare_to(...)` diffs to `<output_folder>/ram_trace.csv` at the
phase boundaries and every-20-frames points listed above.

## Cross-clip confirmation (2026-09-01)

Reran the same `SPAG_RAM_TRACE=1` trace on the two RAM outliers,
`vid360_bruit_operatrice` (241 frames, rank 1) and `CIELE` (204 frames, rank 2),
same config as the short-clip debug pass above. Both reproduce the
plateau-before-loop pattern exactly: `ru_maxrss` is fully flat from
`before_frame_loop` through the last frame in both cases (verified at every
20th frame, no drift). This rules out per-frame accumulation as the driver for
the top-2 outliers too, not just the short control clip.

| clip | frames | final ru_maxrss (GB) | after_frame_extraction (GB) | flow buffer (`flows_fwd`/`flows_bwd`, GiB) | mask buffer (`outputs_per_frame`, GiB) | dominant buffer |
|---|---:|---:|---:|---:|---:|---|
| `circulation_site_1_edit_coupe` | 39 | 25.1 | 3.2→3.2 (+0) | 4.3 | 5.6 | mask (1.3x) |
| `CIELE` | 204 | 98.4 | 3.2→6.4 (+3.2) | 22.3 | 16.4 | flow (1.4x) |
| `vid360_bruit_operatrice` | 241 | 106.5 | 3.1→7.2 (+4.1) | 26.4 | 13.0 | flow (2.0x) |

(sweep's original `results.json` RAM numbers: 24.08 / 94.21 / 101.47 GB
respectively — this trace's runs land within ~3-5% of those, consistent
run-to-run.)

**Key finding: which buffer dominates varies by clip, so there isn't one single
root cause — there are two independent linear-scaling buffers that both get
built up-front and held for the whole run.**

- The **flow buffer** (`flows_fwd`/`flows_bwd`) scales with `frame_count`
  roughly regardless of scene content — it's one WAFT pair per adjacent frame
  index, so it's the more *predictable* of the two, and dominates on
  `CIELE`/`vid360_bruit_operatrice` (both long-ish, and — per the plan's
  framing — high-activity/crowded scenes with a lot of dense flow, though scene
  content doesn't change flow array *count*, only whether they're needed at
  all for `alignement_mask` paths that use flow).
- The **mask buffer** (`outputs_per_frame`) scales with
  `frame_count × tracked_object_count`, so it's the content-dependent term —
  it dominates on the short, static-camera control clip
  (`circulation_site_1_edit_coupe`) where frame count is low but there's still
  a nontrivial number of tracked SAM3 objects per frame, and its share drops
  relative to flow as frame count grows on the two longer outliers (masks grew
  ~3x from the short clip to the outliers, while flow grew ~5-6x — consistent
  with flow's per-frame-pair scaling outpacing whatever plateau
  object-count growth hits on longer clips).
- `after_frame_extraction`'s raw-frame buffer (`video.py:86`, the
  `viz_frames` array) also scales with frame count and is non-trivial on its
  own (3.2GB → 4.1-4.3GB across the three clips) but is dwarfed by the other two.

This confirms the plan's original intuition that RAM correlates with "SAM3
mask/track count" (hypothesis #1) and "content, not just duration" — but
relocates the mechanism: it's not SAM3's internal tracking state (freed after
`segment_with_sam` returns), and it's not pure per-frame accumulation in the
loop (there is none, on all three clips) — it's these two precomputed,
whole-clip, held-for-random-access buffers, with their *relative* weight
shifting by clip.

### Next steps

- Extract `n_tracked_objects` / `n_masks_total` as an explicit per-clip metric
  (from `after_sam3_segmentation`'s tracemalloc count field — 1587/3791/4773
  mask arrays for the 39/241/204-frame clips respectively here) to correlate
  directly against the sweep's per-clip RAM numbers instead of inferring it
  from tracemalloc, and to get a real per-clip "average tracked objects/frame"
  figure (roughly 40.7, 15.7, and 23.4 masks/frame for
  `circulation_site_1_edit_coupe`, `vid360_bruit_operatrice`, `CIELE`
  respectively, from raw count/frame_count — surprisingly not higher on the
  outliers, suggesting mask *resolution*/array size, not object count, may
  need a closer look before assuming "tracked-object density" is the whole
  story for the mask buffer).
- Any fix candidate still needs the standard benchmark-and-compare pass
  (`benchmarks/BENCHMARK_RULES.md`) before becoming a default, per Non-goals.

## Mask-buffer fix implemented (2026-09-01)

Implemented the first "lazy per-frame upsampling" fix direction above, scoped
to the **mask buffer** (`outputs_per_frame`) only, iterated on
`circulation_site_1_edit_coupe` (fastest clip, ~140s/run).

**Change:** `segment_with_sam` (`spag4d/video.py`, formerly ~line 2957-2965) no
longer eagerly `cv2.resize`s every SAM3 mask to native ERP resolution for the
whole clip up front. Masks are left at SAM3's working resolution (native
resolution when `SPAG_SAM3_MAXSIZE`/`SPAG_SAM3_SCALE` are at their disabled
defaults — a no-op then — but a real reduction under this sweep's
`SPAG_SAM3_MAXSIZE=1536`). The resize is now done lazily, once per read, at
the two places that actually consume `outputs_per_frame`/`masks_dict` at
native resolution:

- `run_video`'s per-frame loop, at the `output = outputs_per_frame[idx *
  skip_step]` mask-fusion site (was already resolution-agnostic in shape but
  assumed native; now resizes per-object with `INTER_NEAREST` before
  `np.maximum`-ing into `sam_mask`).
- `compute_temporal_median`'s per-frame mask-fusion loop, which previously
  **silently skipped** (`continue`d) any mask whose shape didn't match `(H,
  W)` — a correctness bug this change would otherwise have introduced (masks
  silently dropped from the background-median computation). Fixed to resize
  instead of skip.

Also fixed the same latent resize gap in `analyze_mask_quality.py` (opt-in
`mask_diagnostics` diagnostic path) for consistency, since it also assumed
native-resolution masks.

**Left untouched:** the flow buffer (`flows_fwd`/`flows_bwd`). Its consumers
(`propagate_depth_via_flow`/`_torch`) do direct pixel-offset depth warping
that requires flow and depth to already be the same resolution — making this
one lazy would mean threading an upsample through 4+ call sites
(`video.py:1280-1433`) with correctness risk closer to the mask-buffer bug
above than the mask fix's single clean insertion point. Deferred; still the
larger of the two buffers on the two long-clip outliers (flow > mask on both
`CIELE` and `vid360_bruit_operatrice` per the cross-clip table above), so this
fix alone does **not** address the worst-case clips — only the short control
clip where the mask buffer happened to dominate.

**Result on `circulation_site_1_edit_coupe`** (same `SPAG_RAM_TRACE=1` trace,
identical config/env otherwise):

| | before | after | delta |
|---|---:|---:|---:|
| final `ram_max_mb` | 25,067.6 | 20,424.2 | **−4,643 MB (−18.5%)** |
| `after_sam3_segmentation` delta | +10,645 MB | +6,729 MB | −3,916 MB |
| `after_flow_waft` delta (unchanged, control) | +7,699 MB | +7,699 MB | 0 (as expected) |

Per-frame loop still plateaus flat post-fix (`before_frame_loop` →
`frame_38`: 20,199.5 MB unchanged) — no new per-frame growth introduced by the
lazy-resize path. Gaussian output verified identical to the pre-fix baseline
(frame_37: 75,001 Gaussians, frame_38: 73,776 Gaussians — exact match), so this
is a lossless memory-layout change, not an approximation.

### Remaining work

- The flow buffer is still unaddressed and dominates on the two real-world
  outliers (`CIELE`, `vid360_bruit_operatrice`) — this fix's ~18.5% win on the
  control clip likely does **not** generalize to a similar percentage on those
  two; need to rerun the same trace on them to measure the actual delta now
  that the mask buffer's contribution is smaller (their mask/flow ratio was
  already flow-dominant, so the mask fix's *relative* impact there is smaller
  than on this clip).
- Flow-buffer fix (lazy upsample or streaming) is the next candidate but
  needs more careful call-site auditing given the depth-warping resolution
  coupling noted above — not started.
- Not yet run through `benchmarks/BENCHMARK_RULES.md`'s standard
  benchmark-and-compare pass (output-identity check above is a smoke test on
  one clip, not a full validation) — required before this becomes a default,
  per Non-goals.

## Flow-buffer fix implemented (2026-09-01)

Implemented the second half of the fix, scoped to the **flow buffer**
(`flows_fwd`/`flows_bwd`), same methodology and clip as the mask-buffer fix
above (`circulation_site_1_edit_coupe`).

**Change:** the two population sites in `run_video` (the default `bglock`
bidirectional-flow pass and the opt-in `SPAG_SINGLE_PASS` pass) no longer call
`upscale_flow(ff, meta["H"], meta["W"])` eagerly when appending to
`flows_fwd`/`flows_bwd`. Flow is left at WAFT's working resolution (native res
when `scale >= 1.0`, i.e. a no-op — but real, since `MAX_SIZE=1024` always
downscales real ERP frames wider than 1024px, which every clip in this sweep
is). `upscale_flow` already no-ops when shapes already match, making it a safe
drop-in for a lazy call at each read site instead of the populate site. Fixed
all 5 identified consumers in `spag4d/video.py`:

- **GPU hot path** (`propagate_depth_via_flow_torch`, `gpu_depth_chain`
  branch): `flow_fwd_native = upscale_flow(flows_fwd[idx - 1], meta["H"],
  meta["W"])` (and same for `flows_bwd`) before building the CUDA tensors.
- **CPU hot path** (`propagate_depth_via_flow`, non-GPU branch): same
  `upscale_flow` call before passing into the function — both hot paths
  required this since neither function has internal resolution-matching
  logic (confirmed via their signatures in `flow_depth_propagation.py`).
- **Diagnostic site (e)** (`SPAG_DIAG_CSV`, median flow vector over static
  pixels): `flow_static = upscale_flow(flows_fwd[idx - 1], meta["H"],
  meta["W"])[lock_mask == 0]` — `lock_mask` is native res, so flow must be
  upscaled here too.
- **Diagnostic site (f)** (`SPAG_DIAG_CSV`, per-object mean flow magnitude):
  the opposite direction from the other sites — `output`'s per-object masks
  are SAM3's *working* resolution (unaffected by this change, already fixed
  at working res by the mask-buffer fix), not necessarily WAFT's working
  resolution. Rather than upscale the large flow field to native res just to
  index it with a working-res mask, resized the small per-object mask to
  match `flow_mag_full`'s shape instead — cheaper and avoids a spurious
  upscale-then-downscale-by-indexing round trip.
- **`flow_npy` export** (`depth_npy_dir`, opt-in FreeTimeGS velocity-
  supervision export): upscales to native res before `np.save`, preserving
  the existing on-disk contract (`flow_{idx}.npy` must stay pixel-aligned
  with the native-res depth maps saved alongside it — this consumer, unlike
  the others, cannot silently change its output resolution).

**Result on `circulation_site_1_edit_coupe`** (same `SPAG_RAM_TRACE=1` trace,
identical config/env, run on top of the mask-buffer fix — i.e. both fixes
combined):

| | mask-fix-only | mask+flow fix | delta |
|---|---:|---:|---:|
| final `ram_max_mb` | 20,424.2 | 18,363.4 | **−2,060.8 MB (−10.1%)** |
| `after_flow_waft` | 10,861.1 | 6,999.3 | **−3,861.8 MB** |
| `after_sam3_segmentation` | 17,590.3 | 15,077.5 | −2,512.8 MB |
| `before_frame_loop` | 20,199.5 | 18,363.4 | −1,836.1 MB |

Combined against the original unfixed baseline: **25,067.6 MB → 18,363.4 MB
(−26.7%)**. Note `after_sam3_segmentation`'s delta (−2,512.8 MB) is *larger*
than `after_flow_waft`'s own drop (−3,861.8 MB minus some noise) would predict
in isolation — `tracemalloc`'s peak accounting overlaps phases, so read the
final `ram_max_mb` row as the authoritative number, the per-checkpoint deltas
as directional confirmation of which phase improved.

Per-frame loop still plateaus flat post-fix (`before_frame_loop` → `frame_38`:
18,363.4 MB unchanged) — no new per-frame growth introduced. Gaussian output
verified byte-identical to both the pre-fix baseline and the mask-fix-only
run (all 39 `.ply` files, exact byte-for-byte match), confirming this is also
a lossless memory-layout change.

### Reverified on `CIELE` (2026-09-01)

Ran the same `scripts/run_flowfix_trace.py CIELE` trace (both fixes stacked,
same config) against the original unfixed CIELE baseline (204 frames,
`docs/RAM_USAGE_INVESTIGATION_PLAN.md` top table / cross-clip confirmation
section, `ram_max_mb` corresponding to 94.21 GB total, `after_flow_waft` =
50,746.7 MB):

| | baseline (unfixed) | mask+flow fix | delta |
|---|---:|---:|---:|
| final `ram_max_mb` | 96,470* | 63,579.6 | **−32,890 MB (−34.1%)** |
| `after_flow_waft` | 50,746.7 | 27,989.3 | **−22,757.4 MB (−44.9%)** |

*(94.21 GB from the top-table summary, converted to MB for direct comparison.)*

Confirms the prediction from the short-clip write-up: the flow-buffer fix's
win is resolution-driven, not content-driven, so it scales even better in
absolute and percentage terms on CIELE (a long, high-resolution real-world
clip) than on `circulation_site_1_edit_coupe`'s short synthetic trace
(−26.7% combined there vs −34.1% here). `after_flow_waft` remains the single
largest checkpoint drop, confirming flow was and still is the dominant buffer
on this clip class — the fix directly attacks the thing that mattered here.

No errors, 204 frames (matches original sweep), splat counts sane
(~70–71K/frame, smooth frame-to-frame variation, no spikes or collapse). No
pre-fix `gaussians/` directory exists for CIELE at these settings, so
byte-identical comparison wasn't done here (only sanity-checked); the
short-clip run already established losslessness of the memory-layout change,
and nothing in this run's logic path differs.

### Remaining work

- Not yet run through `benchmarks/BENCHMARK_RULES.md`'s standard
  benchmark-and-compare pass on either clip class — required before this
  becomes a default, per Non-goals.
- `scripts/run_flowfix_trace.py` is the reusable driver script used for both
  verification runs; not yet wired into `run_atelier1_bglock_sol1.py` or any
  other reusable benchmark entry point.

## Further flow-buffer levers scoped, not implemented (2026-09-01)

Two additional directions were scoped after the lazy-upscale fix above, to see
if the flow buffer's now-small retained size (1.6 GiB CIELE / 304 MiB
circulation) could be reduced further or eliminated.

### Recompute-per-frame instead of caching — scoped, rejected

Idea: since every consumer (`video.py:1306,1325,1388,1434,1481`) reads
`flows_fwd[idx-1]`/`flows_bwd[idx-1]` strictly sequentially (confirmed — no
consumer does random access), the whole-clip buffer could be replaced with a
rolling single-pair cache, computing each flow pair lazily inside the
per-frame loop instead of precomputing all `n_pairs` up front.

**Why this doesn't pencil out:** the barrier isn't RAM, it's the WAFT model's
lifetime. `model` (`WAFTWrapper`) is deliberately released right after flow
computation (`video.py:678-681`: `del model; gc.collect();
torch.cuda.empty_cache()`), *before* SAM3 loads and before the
depth/compositing loop runs — freeing VRAM for those later stages. Moving flow
computation into the per-frame loop means keeping WAFT resident in VRAM
simultaneously with SAM3 and the whole depth/Gaussian-generation pipeline,
which currently never overlap.

Measured VRAM headroom to check whether this overlap would actually be a
problem (`circulation_site_1_edit_coupe` run log): WAFT's own peak is only
**2,717 MB**; the whole-run peak (SAM3 + depth loop dominated) is **6,218
MB**, on an 80GB GPU with ~81GB free at idle — i.e. headroom is not remotely
the constraint here. So the VRAM-overlap risk this lever originally seemed to
trade against turned out to be a non-issue at this GPU's scale.

The actual blocker is **implementation surface, not risk**: doing this
correctly means threading the live WAFT model through the rest of
`run_video()` and touching every one of: the default `bglock` hot path (GPU
and CPU variants), the `single_pass` variant (with its `sp_seam`/`cond_seam`
seam-recompute branching), and all 5 `SPAG_DIAG_CSV` diagnostic read sites —
a much larger, more error-prone surface than the mask/flow lazy-upscale fixes
touched (2 population sites, 5 read sites, no model-lifetime coupling).
Weighed against the capped upside (≤1.6 GiB on the biggest clip traced,
already a small buffer after the upscale fix), **not worth the refactor risk
— rejected, not implemented.** If GPU headroom later becomes a real
constraint (different hardware, larger batch concurrency), the VRAM-overlap
concern would need re-measuring, but the win/risk verdict is unlikely to flip
purely from VRAM headroom shrinking a bit, given how large the implementation
surface is on its own.

### Composite-at-low-res instead of upscaling flow to native — prototyped, measured, not worth defaulting on

Idea: `propagate_depth_via_flow(_torch)` currently requires native-res flow
and depth because it does direct pixel-offset warping with no internal
resolution-matching. If depth compositing instead ran entirely at WAFT's
working resolution (downscaling the native-res depth inputs to match, instead
of upscaling flow to match them), the flow buffer would never need a
native-res copy at all — not even the current lazy per-frame upscale
transient (~112.5 MB/pair on CIELE, freed each frame but still allocated).

**This is a genuine algorithmic change, not a memory-layout change** — compositing
at lower resolution changes the numerical behavior of the depth warp at
object/background boundaries (fewer pixels to feather across, different
aliasing), unlike every fix shipped so far in this doc, all of which were
verified byte-identical.

**Scoping** (reading `flow_depth_propagation.py` in full): no code changes
needed inside that module — `H, W = depth_curr_affine.shape` is derived from
the input array's own shape (not a separate resolution parameter or a
native-res assumption), and every cache (`_BASE_GRID_CACHE`,
`_POLE_MASK_CACHE`, and the `_torch` twins) is keyed by `(H, W[, device])`,
so it will simply build a smaller cache for smaller inputs. The change is
entirely local to the `video.py` call site: downscale the four inputs
(`depth_prev_t`, `aligned_t`, `depth_ref_t`, `lock_t`) to
`meta["new_W"]/["new_H"]` right before the call, and upscale the single
`depth_object_t` output back to native res right after. Implemented behind
`SPAG_COMPOSITE_LOWRES=1` (default off), scoped to the `gpu_depth_chain`
branch only (`video.py` around the `propagate_depth_via_flow_torch` call);
`flows_fwd`/`flows_bwd` are already stored at working res so no flow upscale
is needed either — the existing `upscale_flow` call is skipped entirely on
this path.

**Accuracy validation** (offline replay against real per-frame depth/mask/flow
dumps from the fixed-buffer circulation trace, comparing native-res
compositing vs. low-res compositing at the same `propagate_depth_via_flow`
call, 39 frames): full-frame mean RMSE 0.1998, mean PSNR 45.99 dB
(depth data range ≈39.6). At dynamic-object silhouette boundaries (SAM mask
edges, dilated 5px) specifically: mean RMSE 1.363, mean PSNR 29.11 dB — ~7×
worse than the full-frame average, confirming the risk is concentrated
exactly where the module's own `_EDGE_NEAREST`/`hard_depth_cutover`
documentation already flags fragility (silhouette-bleed / "phantom slab").
Caveat: this offline replay only had single-direction per-frame flow dumps
available, so `flow_bwd` was approximated as `-flow_fwd` for both the
baseline and candidate sides — valid for a same-vs-same relative comparison
but not an absolute-accuracy number against true bidirectional WAFT flow.

**Measured cost/benefit** (real pipeline run, `circulation_site_1_edit_coupe`,
39 frames, `SPAG_COMPOSITE_LOWRES=1` vs. the fixed-buffer baseline):

| metric | baseline | composite-at-low-res | delta |
|---|---:|---:|---:|
| wall time | 140.0s | 141.6s | +1.6s (~+1%, noise-level) |
| VRAM peak | 6217.5 MB | 6217.4 MB | ~0 |
| RAM peak | 18,363 MB | 16,633 MB | **−1,730 MB (−9.4%)** |

VRAM is unaffected because SAM3 already dominates the whole-run peak, so
shrinking this one compositing step's transient tensors doesn't move the
reported ceiling at all. Speed is a wash — the win from running the warp math
on ~14× fewer pixels is offset by the added downscale (×4 inputs) and
upscale (×1 output) `F.interpolate` calls each frame. RAM shows a real but
modest win (−1.7 GiB), smaller in kind than the flow-buffer/mask-buffer
fixes because this only shrinks *transient per-frame* tensors, not a
retained whole-clip buffer.

**Verdict: not worth defaulting on.** No VRAM or speed benefit, a modest RAM
win, traded against a real (if scoped/expected) accuracy cost concentrated at
dynamic-object boundaries — exactly the area FreeTimeGS's downstream 4D
reconstruction is most sensitive to. Code kept behind `SPAG_COMPOSITE_LOWRES=1`
(default off) for future revisit if VRAM ever does become the binding
constraint on this call (e.g. a different SAM3 config that no longer
dominates peak).

## Cost model: what per-frame and per-resolution actually costs (2026-09-01)

Worked from CIELE (3840×1920 native, 816 native frames on disk, 204 kept after
`skip_step=4` → 203 forward flow-pairs), comparing hand-computed buffer sizes
against the measured `ram_trace.csv` numbers.

### The three whole-clip buffers, and what drives each one's size

| buffer | held at | formula | CIELE size |
|---|---|---|---:|
| `viz_frames` (decoded frames) | **native res** | `n_kept_frames × W × H × 3 bytes` (uint8 RGB) | 204 × 3840×1920×3 = **4.20 GiB** |
| `waft_frames` (WAFT input) | **downscaled** to `MAX_SIZE=1024` long edge | `n_kept_frames × new_W × new_H × 3 bytes`, `new_W,new_H` from `scale = min(1024/max(W,H), 1)` | 204 × 1024×512×3 ≈ **300 MiB** |
| `flows_fwd`/`flows_bwd` (**retained** flow buffer — NOT WAFT's own compute cost, see note below) | **baseline: native res** (bug); **fixed: WAFT working res** | `n_pairs × 2 (fwd+bwd) × W × H × 2 (channels) × 4 bytes` (float32) | baseline: 203×2×3840×1920×2×4 = **22.30 GiB**; fixed: 203×2×1024×512×2×4 ≈ **1.62 GiB** |
| mask buffer (`outputs_per_frame`, SAM3) | native-ish res × tracked-object count | `n_frames × (W×H) × bytes_per_object_mask × n_tracked_objects` | observed 16.4 GiB ÷ 204 frames ÷ (3840×1920) ≈ **11.7×** a single-frame mask — consistent with ~11-12 tracked objects (1 byte/px each) retained per frame, not one flattened mask |

**The lever that matters most: resolution, squared.** Doubling either W or H
quadruples every one of these buffers (`W×H` is the base unit everywhere).
Frame *count* only scales buffers linearly. That's why the flow-buffer fix's
win (pure resolution truncation, `native → ≤1024px long edge`, a ~13.7×
reduction in pixel count for CIELE: `(3840×1920)/(1024×512) ≈ 13.7`) tracks so
closely with the computed 22.30 → 1.62 GiB ratio (≈13.8×) — this is a clean,
predictable, resolution-squared effect, not something content- or
frame-count-dependent.

**This is not "WAFT now only costs 1.6 GiB."** WAFT's own compute — model
forward passes, VRAM footprint, working resolution internally — is completely
unchanged by the fix; it always ran at the downscaled `waft_frames` resolution
and always will. The 22.30 GiB was never WAFT's compute cost either — it was
the cost of a *post-processing* step layered on top of WAFT's output:
upscaling every flow field to native res immediately, then retaining that
native-res copy for the whole clip (`n_pairs × 2 × native_W × native_H`,
needed later by `propagate_depth_via_flow(_torch)`). The fix doesn't touch
WAFT or skip that upscale — it only defers it (upscale lazily, per-frame, at
the point of use) instead of doing it once for all 203 pairs and keeping the
result around. So "1.62 GiB" is the size of the *retained* buffer post-fix,
not a claim about how much memory flow computation costs overall.

### Per-frame cost inside the loop: zero

Once `before_frame_loop` is reached, `ru_maxrss` is **flat** in both the
baseline and fixed traces (`frame_0` → `frame_203` unchanged to the decimal in
every trace recorded so far). The per-frame loop only *reads* — `idx-1`
lookback into the flow buffer, indexed reads into the mask buffer — it never
appends or grows either structure. **All RAM cost is committed before frame 0
even starts**; running a longer clip costs more only because the three
buffers above are each sized `n_frames × (something)`, not because the loop
itself leaks.

### Checkpoint deltas overstate the retained-buffer cost — important caveat

Reading `ram_trace.csv`'s checkpoint-to-checkpoint deltas as "the cost of that
phase" is misleading on two counts, both visible on CIELE:

1. **`tracemalloc`'s peak vs current.** `tracemalloc_peak_mb` (used to compute
   `ram_trace`'s deltas) captures the transient high-water mark *during* a
   phase, not what's left resident after it. WAFT running twice (magnitude
   pass + bidirectional pass) allocates and frees large transient tensors;
   those show up in the `after_flow_waft` delta even though they're not part
   of the retained `flows_fwd`/`flows_bwd` buffer. This is why the
   `after_flow_waft` delta (baseline 44.3 GB jump from the prior checkpoint)
   is roughly double the actual retained flow buffer (22.3 GiB) — about half
   the delta is transient WAFT working memory, already freed by the next
   checkpoint.
2. **`tracemalloc` only sees the Python heap.** It has no visibility into
   PyTorch's native tensor allocator, CUDA host-staging/pinned buffers, or
   SAM3's C++-side allocations. On CIELE, `tracemalloc_current_mb` barely
   moves between the baseline and fixed `after_flow_waft` checkpoints (5,118
   → 4,406 MB) even though `ru_maxrss` drops by 22.8 GB — the *actual* saved
   memory (the 22.3 GiB flow buffer minus the 1.6 GiB it now costs) is real
   and lands in the ground-truth `ru_maxrss`/RSS number, but `tracemalloc`'s
   own accounting doesn't show a top-grower line proportional to it, because
   Python-heap tracking isn't what the flow buffer's numpy arrays route
   through in every intermediate copy.

**Takeaway: trust `ru_maxrss` (the actual OS-reported resident set) for "how
much RAM did this cost," not the `tracemalloc` current/peak columns or the
per-checkpoint delta size — those are directional (which phase grew) but not
precisely attributable to one retained buffer.** The per-buffer formulas
above, cross-checked against `ru_maxrss` totals, are the reliable way to
predict cost for a new clip: compute `n_frames × W × H` for each of the three
buffers at their respective resolutions, sum, and that approximates the
`before_frame_loop` plateau RAM within the observed few-GB noise band.

## Whole-clip downscale tensor fix — chunked, shipped (2026-09-02)

### Root cause

`ru_maxrss` is a monotonic high-water mark since process start (confirmed via
`resource.getrusage(...).ru_maxrss`), and PyTorch's CPU allocator never
returns freed host memory to the OS once touched — confirmed directly by
adding an explicit `del` right after a large CPU tensor's last use and
observing **zero** RSS reclamation. So any single large CPU allocation
becomes a permanent floor for the rest of the process, regardless of whether
Python later frees the reference.

Fine-grained `ram_trace` checkpoints (added upstream of `WAFTWrapper.run()`,
not just inside it — the earlier "first CUDA inference call" hypothesis was
wrong because it only traced *inside* `.run()`, whose own start-of-call RSS
baseline was already past the real jump) isolated the jump to one line in
`spag4d/video.py`'s flow-setup block:

```python
viz_tensor = torch.from_numpy(viz_frames).permute(0, 3, 1, 2).float()
```

This converts the **entire clip** to a single float32 CPU tensor in one shot
— `frames × H × W × 3 × 4` bytes: ≈2.9 GB for `circulation_site_1_edit_coupe`
(39 frames, 3840×1920), ≈15 GB for `CIELE` (204 frames, same resolution).
Measured directly: this one line accounted for +2,866 MB of a +3,188 MB jump
(`after_waft_init` 3,131 MB → `before_flow_maskpass` 6,319 MB, circulation).

### Fix

Chunked downscale: process `viz_frames` in chunks of 8 frames (float32
convert → `F.interpolate` → cast back to uint8 → discard the float32
intermediate → concatenate at the end), bounding the largest
simultaneously-live float32 buffer to one chunk's worth instead of the whole
clip. Shipped as a fixed `_chunk = 8` in `spag4d/video.py` (no env var —
swept 1/2/4/8/16/39 on circulation, RAM was flat/good from 1 through 8,
noisy at 16, and time was flat across all sizes; 8 was picked as a safe,
unremarkable middle value rather than chasing sub-GB differences in a run
with several GB of measurement noise).

### Measured results

| Clip | Frames | Res | RAM before | RAM after | Δ | Time | VRAM |
|---|---|---|---|---|---|---|---|
| circulation_site_1_edit_coupe | 39 | 3840×1920 | 16,251 MB | 13,340 MB | **−17.9% / −2,911 MB** | 139.98s → 141.1s (noise) | unchanged (~6,217 MB) |
| CIELE | 204 | 3840×1920 | 64,053 MB | 45,692 MB | **−28.7% / −18,361 MB** | ~511-530s → 531.4s (noise) | 11,885 → 11,875 MB (noise) |

No time or VRAM cost at chunk=8 on either clip.

### Other levers tried this session, ruled out (measured, reverted)

- **`SPAG_SINGLE_PASS=1` + `SPAG_SP_FIX_SEAMGAP=1`** (merge the mask pass and
  the bidirectional pass into one seam-padded pass): no RAM improvement,
  slightly worse. Consistent with the root cause sitting entirely upstream of
  both WAFT passes — merging passes can't touch a buffer built before either
  one runs.
- **Skipping `flows.mp4`** (the debug/preview flow visualization video,
  confirmed via grep to be unconsumed downstream): no RAM improvement,
  slightly worse. Reverted.

### Open thread, not investigated

The chunk-size sweep's chunk=39 result (13,362 MB, circulation) did not match
the earlier-measured "baseline" for what should be the same effectively
one-shot code path (16,251 MB) — an unexplained ~3 GB / ~18% run-to-run
discrepancy on ostensibly identical computation. Not chased down; flagged
here in case it resurfaces.

## Two more whole-clip float buffers found and fixed (2026-09-02)

With the flow buffer, mask buffer, and `viz_tensor` fixes all shipped, CIELE
still peaked at 45,692 MB. Trying `SPAG_SAM3_MAXSIZE=1024` (matching WAFT's
1024px working resolution) to cap SAM3's own inference/mask resolution only
bought **−6.9%** (45,692 → 42,535 MB) — far short of what the "16.4 GiB mask
buffer" estimate implied. Re-tracing with `ram_trace` checkpoints on either
side of every remaining phase found the real cost was two more instances of
the same bug pattern as `viz_tensor`: a whole-clip float buffer materialized
in one shot, non-reclaimable once touched (PyTorch's CPU allocator never
returns freed host RSS to the OS).

### Bug 1: `compute_temporal_median` — two full-clip float32 buffers per channel

`compute_temporal_median()` runs unconditionally for the default
`get_background_method="temporal_median"` path (used by `freeze_bg`/`bglock`).
Per channel, it did:

```python
channel_tensor = torch.from_numpy(frames_array[..., c]).float().to(device)  # device forced "cpu"
median_calc_tensor = channel_tensor.clone()      # a SECOND full-clip float32 buffer
median_calc_tensor[masks_array] = float("nan")
median_frame[:, :, c] = torch.nanmedian(median_calc_tensor, dim=0).values...
std_frame += torch.std(channel_tensor, dim=0)...
```

On CIELE (204 frames, 3840×1920): `channel_tensor` alone is ≈6.0 GB; the
`.clone()` doubles that to ≈12 GB live per channel; and since none of it is
ever returned to the OS, three channels sequentially ratchet the RSS floor
up by roughly that amount. Measured directly via a `before`/`after`
checkpoint pair: this step alone accounted for **+17,184.6 MB** on one
CIELE trace before the fix (the single largest jump in the whole pipeline,
larger even than SAM3 segmentation's own +11.5 GB).

**Fix**: row-chunk the whole per-channel computation. Median/std are
per-pixel independent across the frame axis, so nothing requires holding
all `H` rows live at once — chunk over rows (~256 MB per chunk buffer,
chosen the same way as the `viz_tensor` fix) and drop the redundant
`.clone()` in favor of masking a per-chunk temporary. After the fix, the
same step costs **+642.8 → +944.0 MB** across reruns (down from the fixed
value's own noise band, but consistently under 1 GB, not 17 GB).

### Bug 2: post-loop `nanstd` over `aligned_depth_list_cpu`

After the per-frame loop, a diagnostics block computes cross-frame depth-std
statistics:

```python
stack_np = np.stack(aligned_depth_list_cpu, axis=0)   # second full-clip buffer
depth_std_all = np.nanstd(stack_np, axis=0)
```

`aligned_depth_list_cpu` (one native-res depth map per frame) is already held
for the entire run as part of the per-frame loop's own retained state — that
part isn't new. But `np.stack` materializes a *second* full-clip array just
to compute a per-pixel std that's independent across frames, momentarily
doubling that buffer's footprint. Measured: **+5,682.2 MB** for the stack
itself, plus a further **+8,662.4 MB** in the block immediately after it
(`cv2.imwrite`/`json.dump` calls that shouldn't cost anywhere near that) —
bisected with per-line checkpoints and found to be entirely attributable to
`nanstd`/`nanpercentile`/boolean-fancy-indexing on the *unchunked* stack, not
to any of the individually-instrumented lines themselves; the checkpoint
gap was mis-attributed to "the block after the stack" until the stack itself
was chunked.

**Fix**: same row-chunking pattern — compute `nanstd` per row-band directly
from the list of per-frame arrays, never materializing the full `(F, H, W)`
stack. After the fix, **both** the stack step and everything after it in the
diagnostics block are flat at +0 MB across five separate checkpoints
(`after_depth_std_jpg`, `after_unstable_png`, `after_inside_outside_stats`,
`after_activity_jpgs`, `after_depth_range`) — confirming the entire
+14.3 GB tail was this one bug, not several. Those fine-grained checkpoints
were collapsed back down to a single `after_aligned_depth_stack` marker once
confirmed, matching the cleanup pattern used for the flow phase.

### Combined result on CIELE (204 frames, 3840×1920)

| stage | peak RSS |
|---|---:|
| Original baseline | 64,053 MB |
| + `viz_tensor` chunking | 45,692 MB |
| + `compute_temporal_median` chunking | 42,535 MB |
| + `nanstd`/`aligned_depth_stack` chunking | **28,301 MB** |

**Total: −55.8% peak RAM, no time or VRAM cost** (565.3s, 11,616 MB VRAM —
consistent with every unfixed run). Both fixes are unconditional (no env
var), same as the `viz_tensor` fix.

### SAM3 mask-resolution lever, re-evaluated

`SPAG_SAM3_MAXSIZE=1024` was tested in isolation (bf16 off, per
CLAUDE.md's warning against stacking bf16 with a resolution cap) and gave
only ~6.9% RAM reduction before the two fixes above — most of what looked
like "mask buffer cost" in the original estimate was actually these two
bugs, now fixed. Whether capping SAM3's own inference resolution is still
worth doing on top of the fixes above is unresolved and would need its own
clean before/after pass.
