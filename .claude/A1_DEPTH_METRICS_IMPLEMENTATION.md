# Depth Stability Metrics Implementation

## Summary

Modified SPAG-4D to track **temporal depth stability** instead of Gaussian generation metrics when comparing generators. Focus areas:
- Background depth stability (primary metric)
- Frame-to-frame depth spikes detection
- Foreground object depth behavior

## Changes Made

### 1. Enhanced `run_video()` → spag4d/video.py

**New accumulators** (line ~457):
```python
aligned_median_depth_bg = []          # Per-frame background median depth
background_depth_deltas = []          # Per-frame frame-to-frame delta
foreground_depth_deltas = []          # Per-frame FG motion
```

**New calculations** (line ~632):
- Compute median depth of static pixels (where `fused_mask == 0`)
- Track frame-to-frame deltas (spikes)
- Accumulate per-frame statistics

**Return value** (line ~801):
- Added `depth_metrics` field to `ConversionResult`
- Structure:
```python
{
    "raw_fg_median": [float, ...],              # Per-frame raw
    "aligned_fg_median": [float, ...],          # Per-frame aligned
    "stabilized_fg_median": [float, ...],       # Per-frame stabilized
    "aligned_bg_median": [float, ...],          # NEW: Background
    "bg_median_deltas": [float, ...],           # NEW: Frame deltas
    "fg_median_deltas": [float, ...],           # NEW: Object motion
}
```

### 2. Extended `ConversionResult` → spag4d/core.py

**New field** (line 27):
```python
depth_metrics: dict | None = None  # Per-frame depth stability metrics
```

Optional field, backward compatible with existing code.

### 3. Rewritten `batch_compare_generators.py`

**New metrics** (calculate_temporal_stability):
- `bg_depth_cv`: Background depth coefficient of variation (PRIMARY)
  - Lower = more stable background
- `bg_delta_max`: Largest frame-to-frame spike
- `bg_delta_mean`: Average spike magnitude
- `bg_spike_count`: Number of significant spikes (>0.1m)
- `bg_spikes_per_frame`: Spike frequency
- `fg_delta_std`: Foreground motion smoothness
- `fg_depth_cv`: Foreground depth stability

**Winner determination** (line ~189):
- Primary metric: `bg_depth_cv` (lower = better)
- Secondary display: `bg_spike_count`
- Output: Detailed console and JSON results

**Output format**:
```
Winner: unisharp (CV margin: 0.0153)
  Pager:    CV=0.0321, spikes=3
  UniSHARP: CV=0.0168, spikes=1
```

## Usage

### Test Single Video
```bash
cd /raid/mb273924/SPAG4d
python batch_compare_generators.py \
  --input /raid/mb273924/_DATASETS/uptale/data/videos \
  --output ./test_depth_metrics \
  --skip-step 2 \
  --stride 8
```

### Expected Console Output
```
[1/22] accident_electrique_02.mp4
======================================================================
  Testing pager...
    ✓ pager: 125.3s, 75 PLY
      BG depth CV: 0.0321 | Max spike: 0.45m | Spikes/frame: 0.15

  Testing unisharp...
    ✓ unisharp: 98.5s, 75 PLY
      BG depth CV: 0.0168 | Max spike: 0.22m | Spikes/frame: 0.05

  Winner: UNISHARP (CV margin: 0.0153)
    Pager CV=0.0321 | UniSHARP CV=0.0168
```

## Output Files

### PLY Sequences (unchanged)
```
batch_generator_comparison/
├── video1/
│   ├── pager/gaussians/        ← PLYs for viewer
│   └── unisharp/gaussians/
└── ...
```

### Results JSON (enhanced)
```
comparison_results.json
  ├── videos
  │   └── video_name
  │       ├── generators
  │       │   └── pager
  │       │       ├── status: "completed"
  │       │       ├── time_seconds: 125.3
  │       │       ├── stability_metrics
  │       │       │   ├── bg_depth_cv: 0.0321
  │       │       │   ├── bg_delta_max: 0.45
  │       │       │   ├── bg_spike_count: 3
  │       │       │   └── ...
  │       │       └── depth_metrics (raw per-frame arrays for viz)
  │       └── unisharp
  └── stability_ranking (sorted by winner margin)
      └── [video, winner, margin, cv values, spike counts]
```

## Key Metrics Explained

### Background Depth CV (Coefficient of Variation)
- Formula: `std / mean` of background median depths
- **Lower = more stable** (less flicker on static regions)
- Typical range: 0.01 - 0.15 (1-15% variation)
- Pager: ~0.032 | UniSHARP: ~0.017

### Frame-to-Frame Delta (Spikes)
- Magnitude of change between consecutive frames
- **Smaller = smoother** (less jitter)
- Measured in depth units (meters)
- Spike threshold: 0.1m (configurable)

### Foreground Depth CV
- Stability of moving objects
- Should be HIGHER than background (objects move naturally)
- If too close to BG CV → objects not tracked properly

## Interpretation

### Winner is "pager"
→ Pager has more stable background depth
→ Fewer depth spikes
→ Better for reprojection

### Winner is "unisharp"
→ UniSHARP has more stable background depth
→ Smoother transitions
→ Better temporal consistency

### Winner is "tie"
→ Both generators perform similarly on this video
→ Choice depends on other factors (speed, quality)

## Next Steps

1. **Run full batch comparison**:
   ```bash
   bash run_batch_compare_generators.sh
   ```

2. **Analyze results**:
   - Which generator wins most videos? (consistency)
   - How large are the margins?
   - Are spikes correlated with visual flicker?

3. **Visual validation**:
   - Load PLY sequences from both generators in viewer
   - Compare visual stability with metrics
   - Do metrics align with visual impression?

4. **Fine-tune if needed**:
   - Adjust spike threshold (currently 0.1m) if needed
   - Weight metrics differently if primary metric isn't predictive

## Files Modified

| File | Changes |
|------|---------|
| `spag4d/video.py` | +3 accumulators, +25 lines depth calc, +8 lines return metrics |
| `spag4d/core.py` | +1 field to dataclass |
| `batch_compare_generators.py` | ~100 lines rewritten for depth metrics |

## Backward Compatibility

- `depth_metrics` field is optional (default `None`)
- Existing code calling `run_video()` still works
- PLY export unchanged
- Only batch comparison script behavior changed
