# Depth Stability Metrics Implementation

Added temporal depth-stability tracking (instead of raw Gaussian-count metrics) for comparing
generators, so a diff can be gated on flicker/spike behavior rather than splat count.

## Changes

**`spag4d/video.py`** — `run_video()` gained accumulators (`aligned_median_depth_bg`,
`background_depth_deltas`, `foreground_depth_deltas`, ~line 457/632): per-frame median depth of
static pixels (`fused_mask == 0`), frame-to-frame deltas (spikes), and per-frame accumulation.

**`spag4d/core.py`** — `ConversionResult` gained `depth_metrics: dict | None = None` (line 27),
optional, backward-compatible:
```python
{
    "raw_fg_median": [...], "aligned_fg_median": [...], "stabilized_fg_median": [...],
    "aligned_bg_median": [...], "bg_median_deltas": [...], "fg_median_deltas": [...],
}
```

**`batch_compare_generators.py`** — rewritten `calculate_temporal_stability()` to compute:

| Metric | Meaning |
|---|---|
| `bg_depth_cv` (PRIMARY) | `std/mean` of background median depth. Lower = more stable. Typical range 0.01–0.15. |
| `bg_delta_max` / `bg_delta_mean` | largest / average frame-to-frame background spike |
| `bg_spike_count` / `bg_spikes_per_frame` | spikes >0.1m (threshold configurable) |
| `fg_delta_std` / `fg_depth_cv` | foreground equivalents — should be **higher** than bg CV; if too close, objects likely aren't tracked properly |

Winner is the generator with the lowest `bg_depth_cv` (secondary display: `bg_spike_count`).

## Note

The `run_batch_compare_generators.sh` script referenced by early versions of this doc no longer
exists in the repo — see `CLAUDE.md`'s "Note" section. `batch_compare_generators.py` itself is
current.
