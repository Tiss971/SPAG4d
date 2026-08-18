"""§8 -- verify SPAG_HARD_DEPTH_CUTOVER removes the interpolated-depth smear.

Synthetic step: left half of the frame is a foreground object at 2m, right
half is background at 10m (both flat). Feed a mask covering the left half as
"dynamic", run composite_bg_locked with and without hard_depth_cutover, and
report how many pixels in the feather band take a depth value that never
appears in either source plate (the "phantom slab" this fix targets).
"""
import numpy as np

from spag4d.flow_depth_propagation import composite_bg_locked

H, W = 200, 400
FG, BG = 2.0, 10.0

object_depth = np.full((H, W), FG, dtype=np.float32)
depth_ref = np.full((H, W), BG, dtype=np.float32)
dynamic_mask = np.zeros((H, W), dtype=np.uint8)
dynamic_mask[:, : W // 2] = 1

for hard in (False, True):
    depth_final, alpha = composite_bg_locked(
        object_depth, depth_ref, dynamic_mask, dilate_px=12, feather_px=9,
        hard_depth_cutover=hard,
    )
    band = (alpha > 0.02) & (alpha < 0.98)
    n_band = int(band.sum())
    vals = depth_final[band]
    is_source = np.isclose(vals, FG, atol=1e-3) | np.isclose(vals, BG, atol=1e-3)
    n_phantom = int((~is_source).sum())
    lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (float("nan"), float("nan"))
    print(f"hard_depth_cutover={hard}: {n_band} feather-band px/row, "
          f"{n_phantom} phantom-depth px ({n_phantom / max(n_band,1):.1%}), "
          f"range=[{lo:.2f}, {hi:.2f}]m")
