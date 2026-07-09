# Fix B.1: Temporal Consistency in Depth Estimation

## Problem
DA360/PaGeR models were rescaling depth **independently per frame** to achieve ~5m median depth. This caused **scale drift** between consecutive frames in videos, even on static regions.

**Root cause**: These models were designed for single-image inference where per-frame rescaling makes sense. For video, this creates unnecessary temporal inconsistency.

## Solution: Temporal Consistency Mode

### Changes Made

#### 1. **DA360Model** (`spag4d/da360_model.py`)
- Added parameter `temporal_consistency: bool = False` to `predict()`
- If `temporal_consistency=True`: returns raw depth (disparity inverse) without per-frame rescaling
- If `temporal_consistency=False` (default): maintains old behavior (per-frame rescaling to ~5m)
- Backward compatible: single-image workflows use default `temporal_consistency=False`

#### 2. **PaGeRModel** (`spag4d/pager_model.py`)
- Added parameter `temporal_consistency: bool = False` to `predict()`
- If `temporal_consistency=True`: disables CLIP-based per-frame scale routing
- Returns raw scale-invariant depth without per-frame adjustments
- Backward compatible with existing single-image code

#### 3. **video.py** (`run_video` function)
- **New workflow for video mode**:
  1. Predict reference depth with `temporal_consistency=True` (no per-frame rescale)
  2. Calculate `scale_factor_to_5m = 5.0 / median(depth_ref_raw)` **once**
  3. Apply this fixed scale factor to **all frames** and reference
  4. Affine alignment now works with temporally consistent scales

```python
# Reference
depth_raw, _ = depth_engine.predict(reference_bg, temporal_consistency=True)
scale_factor_to_5m = 5.0 / depth_raw.median()
depth_ref = depth_raw * scale_factor_to_5m * global_scale

# Per-frame (loop)
depth_raw, _ = depth_engine.predict(frame, temporal_consistency=True)
depth = depth_raw * scale_factor_to_5m * global_scale  # Same scale for all
```

## Expected Improvements

### Metrics (from benchmarking)
1. **Depth Stability**: Variance in static regions should drop significantly
   - Old: Each frame rescales independently → high variance
   - New: Fixed scale → low variance
   
2. **Affine Alignment**: Scale factor `s` should be closer to 1.0
   - Old: `s` varies widely per frame (drift correction needed)
   - New: `s` ≈ 1.0 (temporal consistency already applied)
   
3. **Frame-to-Frame Flicker**: Should reduce on static regions
   - Old: Depth changes between frames due to rescaling drift
   - New: Changes only from actual scene depth variation

## How to Benchmark

Run the benchmarking script:
```bash
python benchmark_temporal_consistency.py --video <path/to/video.mp4> --max-frames 100
```

This compares:
- Mode 1: `temporal_consistency=True` (new, optimized for video)
- Mode 2: `temporal_consistency=False` (old, per-frame rescaling)

Output: `./benchmark_results/benchmark_results.json` with detailed metrics.

## Backward Compatibility

- **Single-image code**: No changes needed. Default `temporal_consistency=False` preserves old behavior.
- **Video code**: Automatically uses new mode via `run_video()`.
- **Custom depth estimation**: Can opt-in by passing `temporal_consistency=True` to `predict()`.

## Notes

- The fix addresses **B.1** (scale drift source) directly
- **B.2** (mask quality) is decoupled: better masks still help, but temporal consistency no longer adds unnecessary drift
- Aligns with the original design intent: the models were made for images, not video sequences
