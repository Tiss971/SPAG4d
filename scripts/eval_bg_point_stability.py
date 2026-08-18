"""3D flicker/jerk/drift decomposition for background depth stability.

`bg_depth_cv` (batch_compare_generators.py / benchmark_solutions.py) is a single
scalar per video: std/mean of the *frame-level median* background depth. That
can't tell smooth global drift apart from per-pixel high-frequency flicker
(the failure mode that actually pops in FreeTimeGS), and it can't catch a
locked/frozen background winning "stability" for free by construction rather
than by being accurate -- see .claude's metric-scale-confound-style trap
(ported from vipe's docs/metric_scale_confound.md, where consistency-only
ranking crowned a config that just reconstructed less scene).

This tracks individual static pixels' depth over time instead of one
per-frame aggregate, then decomposes each trajectory into:
  - drift   : low-frequency component (a moving median) -- harmless, a 4D
              trainer fits a smooth trajectory fine.
  - flicker : raw - drift, the high-frequency residual -- this is what pops.
  - jerk    : ||d[t-1] - 2*d[t] + d[t+1]|| -- second-derivative roughness.
All three are reported relative to local depth (divided by drift), not in
absolute metres, since bg-locked and free pixels sit at different depths and
absolute error isn't comparable between them (same normalization lesson as
vipe's world-scale confound).

No camera pose / reprojection needed: SPAG4D's camera is fixed, so "track a
point across frames" is just "read the depth map at the same pixel index
across frames" -- unlike vipe's SLAM-trajectory setting.

Input: a depth_npy_dir produced by run_video(..., depth_npy_dir=...), i.e.
depth_{idx}.npy (float32 metric depth) + mask_{idx}.npy (uint8 SAM mask,
0 = static/background) for every frame.

Usage:
    python scripts/eval_bg_point_stability.py _data/scene01/depth_maps \\
        --out scene01_bg_stability.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter


def load_sequence(depth_dir: Path):
    depth_files = sorted(
        depth_dir.glob("depth_*.npy"),
        key=lambda p: int(re.search(r"depth_(\d+)\.npy", p.name).group(1)),
    )
    if not depth_files:
        raise FileNotFoundError(f"No depth_*.npy files found in {depth_dir}")

    depths, masks = [], []
    for depth_path in depth_files:
        idx = re.search(r"depth_(\d+)\.npy", depth_path.name).group(1)
        mask_path = depth_dir / f"mask_{idx}.npy"
        depths.append(np.load(depth_path))
        masks.append(np.load(mask_path) if mask_path.exists() else np.zeros_like(depths[-1], dtype=np.uint8))
    return np.stack(depths, axis=0), np.stack(masks, axis=0)  # [T,H,W]


def pick_static_points(masks: np.ndarray, target_points: int = 4000):
    """Pixels that stay unmasked (background) for the whole clip, subsampled to a grid."""
    always_static = np.all(masks == 0, axis=0)  # [H,W]
    ys, xs = np.nonzero(always_static)
    n = len(ys)
    if n == 0:
        raise RuntimeError("No pixel is background in every frame -- can't build static trajectories")
    stride = max(1, int(np.sqrt(n / target_points)))
    # grid subsample (not pure stride-in-index) so kept points stay spatially spread
    grid_keep = (ys % stride == 0) & (xs % stride == 0)
    if grid_keep.sum() == 0:
        grid_keep = np.ones(n, dtype=bool)
    return ys[grid_keep], xs[grid_keep]


def decompose(trajectories: np.ndarray, drift_window: int = 15):
    """trajectories: [N_points, T]. Returns per-point relative flicker/jerk/drift-range."""
    drift = median_filter(trajectories, size=(1, drift_window), mode="nearest")
    flicker = trajectories - drift
    jerk = np.zeros_like(trajectories)
    jerk[:, 1:-1] = np.abs(trajectories[:, :-2] - 2 * trajectories[:, 1:-1] + trajectories[:, 2:])

    safe_drift = np.clip(drift, 1e-3, None)
    rel_flicker = np.abs(flicker) / safe_drift
    rel_jerk = jerk / safe_drift
    drift_range = (drift.max(axis=1) - drift.min(axis=1)) / np.median(safe_drift, axis=1)

    return {
        "rel_flicker_median": float(np.median(rel_flicker)),
        "rel_flicker_p90": float(np.percentile(rel_flicker, 90)),
        "rel_jerk_median": float(np.median(rel_jerk)),
        "rel_jerk_p90": float(np.percentile(rel_jerk, 90)),
        "drift_range_median": float(np.median(drift_range)),
        "n_points": int(trajectories.shape[0]),
        "n_frames": int(trajectories.shape[1]),
        "per_frame_rel_flicker": np.median(rel_flicker, axis=0).tolist(),
        "per_frame_rel_jerk": np.median(rel_jerk, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("depth_dir", help="Directory with depth_{idx}.npy / mask_{idx}.npy (run_video's depth_npy_dir)")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON results here")
    parser.add_argument("--target-points", type=int, default=4000, help="Approx. number of static points to track")
    parser.add_argument("--drift-window", type=int, default=15, help="Median-filter window (frames) for the drift/flicker split")
    args = parser.parse_args()

    depths, masks = load_sequence(Path(args.depth_dir))
    ys, xs = pick_static_points(masks, args.target_points)
    trajectories = depths[:, ys, xs].T  # [N_points, T]

    # A handful of "always background" pixels still go NaN in some frames (e.g.
    # bglock's flow-propagated region near the seam/poles) -- drop those points
    # entirely rather than let one NaN pixel poison every aggregate stat.
    valid = ~np.any(np.isnan(trajectories), axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(valid)} points with a NaN depth in at least one frame")
    trajectories = trajectories[valid]

    result = decompose(trajectories, args.drift_window)
    print(f"Tracked {result['n_points']} static points over {result['n_frames']} frames")
    print(f"  rel_flicker  median={result['rel_flicker_median']:.4f}  p90={result['rel_flicker_p90']:.4f}")
    print(f"  rel_jerk     median={result['rel_jerk_median']:.4f}  p90={result['rel_jerk_p90']:.4f}")
    print(f"  drift_range  median={result['drift_range_median']:.4f}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
