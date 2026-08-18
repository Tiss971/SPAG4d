"""3D flicker/jerk decomposition for FOREGROUND (dynamic) depth stability.

Sibling to eval_bg_point_stability.py, which only tracks pixels that are
background in every frame -- exactly the points this script must exclude, since
they don't move. Foreground points move in pixel space, so "read the same
pixel index across frames" doesn't work; each point is instead propagated
frame-to-frame with the dense forward WAFT flow (flow_{idx}.npy, dumped
alongside depth_{idx}.npy/mask_{idx}.npy by run_video's bglock path, or
backfilled for any run via scripts/export_flow_only.py) before its depth is
sampled. This answers "does a moving object's own depth stay smooth as it
moves," not "is it placed correctly relative to the background" (no ground
truth for that here).

A trajectory ends the first time its propagated position leaves the frame or
the mask no longer marks it foreground there (occlusion / object left frame /
flow drifted off the object) -- points shorter than --min-length are dropped
entirely since jerk needs at least 3 samples and drift needs a real window.

Input: a depth_npy_dir with depth_{idx}.npy, mask_{idx}.npy (1 = dynamic/FG),
and flow_{idx}.npy (HxWx2 float32, forward flow frame i-1 -> i, idx>=1).

Usage:
    python scripts/eval_fg_point_stability.py benchmarks/scene01/bglock/depth_maps \\
        --out benchmarks/scene01/bglock_fg_stability.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_sequence(depth_dir: Path):
    depth_files = sorted(
        depth_dir.glob("depth_*.npy"),
        key=lambda p: int(re.search(r"depth_(\d+)\.npy", p.name).group(1)),
    )
    if not depth_files:
        raise FileNotFoundError(f"No depth_*.npy files found in {depth_dir}")

    depths, masks, flows = [], [], [None]  # flows[0] unused (no flow into frame 0)
    for i, depth_path in enumerate(depth_files):
        idx = int(re.search(r"depth_(\d+)\.npy", depth_path.name).group(1))
        mask_path = depth_dir / f"mask_{idx}.npy"
        depths.append(np.load(depth_path))
        masks.append(np.load(mask_path) if mask_path.exists() else np.zeros_like(depths[-1], dtype=np.uint8))
        if i > 0:
            flow_path = depth_dir / f"flow_{idx}.npy"
            if not flow_path.exists():
                raise FileNotFoundError(
                    f"{flow_path} missing -- backfill with scripts/export_flow_only.py first"
                )
            flows.append(np.load(flow_path))
    return np.stack(depths, axis=0), np.stack(masks, axis=0), flows  # depths/masks [T,H,W], flows list of [H,W,2]


def pick_fg_seed_points(mask0: np.ndarray, target_points: int):
    ys, xs = np.nonzero(mask0 > 0)
    n = len(ys)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    stride = max(1, int(np.sqrt(n / target_points)))
    grid_keep = (ys % stride == 0) & (xs % stride == 0)
    if grid_keep.sum() == 0:
        grid_keep = np.ones(n, dtype=bool)
    return ys[grid_keep], xs[grid_keep]


def track_points(depths, masks, flows, ys0, xs0):
    """Propagate each seed point frame-to-frame via flow; sample depth along the way.

    Returns a list of (depth_trajectory, start_frame) for points that survive
    at least 2 frames -- trajectories end (are truncated) the first time the
    point leaves the frame or its current pixel is no longer foreground.
    """
    T, H, W = depths.shape
    n = len(ys0)
    cur_y = ys0.astype(np.float64)
    cur_x = xs0.astype(np.float64)
    alive = np.ones(n, dtype=bool)
    trajectories = [[] for _ in range(n)]

    for i in range(n):
        trajectories[i].append(float(depths[0, ys0[i], xs0[i]]))

    for t in range(1, T):
        flow = flows[t]  # [H,W,2], (dx,dy) frame t-1 -> t
        yi = np.clip(np.round(cur_y).astype(int), 0, H - 1)
        xi = np.clip(np.round(cur_x).astype(int), 0, W - 1)
        dx = flow[yi, xi, 0]
        dy = flow[yi, xi, 1]
        cur_x = cur_x + dx
        cur_y = cur_y + dy

        in_bounds = (cur_x >= 0) & (cur_x < W) & (cur_y >= 0) & (cur_y < H)
        yi2 = np.clip(np.round(cur_y).astype(int), 0, H - 1)
        xi2 = np.clip(np.round(cur_x).astype(int), 0, W - 1)
        still_fg = masks[t, yi2, xi2] > 0

        for i in range(n):
            if not alive[i]:
                continue
            if not (in_bounds[i] and still_fg[i]):
                alive[i] = False
                continue
            trajectories[i].append(float(depths[t, yi2[i], xi2[i]]))

    return trajectories


def decompose_variable(trajectories, drift_window: int, min_length: int):
    from scipy.ndimage import median_filter

    per_frame_flicker = {}  # global frame idx -> list of relative flicker values
    per_frame_jerk = {}
    all_flicker, all_jerk, all_drift_range = [], [], []
    n_kept = 0

    n_dropped_nan = 0
    for traj in trajectories:
        arr = np.array(traj)
        if len(arr) < min_length:
            continue
        if np.any(np.isnan(arr)):
            # bglock's flow-propagated depth goes NaN for a handful of pixels in
            # later frames (same issue eval_bg_point_stability.py hit) -- drop
            # the whole track rather than let one NaN sample poison its stats.
            n_dropped_nan += 1
            continue
        n_kept += 1
        drift = median_filter(arr, size=min(drift_window, len(arr)), mode="nearest")
        flicker = arr - drift
        jerk = np.zeros_like(arr)
        jerk[1:-1] = np.abs(arr[:-2] - 2 * arr[1:-1] + arr[2:])
        safe_drift = np.clip(drift, 1e-3, None)
        rel_flicker = np.abs(flicker) / safe_drift
        rel_jerk = jerk / safe_drift

        all_flicker.append(rel_flicker)
        all_jerk.append(rel_jerk)
        all_drift_range.append((drift.max() - drift.min()) / np.median(safe_drift))

        for local_i, (rf, rj) in enumerate(zip(rel_flicker, rel_jerk)):
            per_frame_flicker.setdefault(local_i, []).append(rf)
            per_frame_jerk.setdefault(local_i, []).append(rj)

    if n_dropped_nan:
        print(f"Dropping {n_dropped_nan} tracks with a NaN depth in at least one frame")

    if n_kept == 0:
        return None

    flat_flicker = np.concatenate(all_flicker)
    flat_jerk = np.concatenate(all_jerk)
    max_frame = max(per_frame_flicker.keys())
    per_frame_median_flicker = [
        float(np.median(per_frame_flicker[i])) if i in per_frame_flicker else None
        for i in range(max_frame + 1)
    ]
    per_frame_median_jerk = [
        float(np.median(per_frame_jerk[i])) if i in per_frame_jerk else None
        for i in range(max_frame + 1)
    ]

    return {
        "rel_flicker_median": float(np.median(flat_flicker)),
        "rel_flicker_p90": float(np.percentile(flat_flicker, 90)),
        "rel_jerk_median": float(np.median(flat_jerk)),
        "rel_jerk_p90": float(np.percentile(flat_jerk, 90)),
        "drift_range_median": float(np.median(all_drift_range)),
        "n_points_kept": n_kept,
        "n_points_seeded": len(trajectories),
        "median_track_length": float(np.median([len(t) for t in trajectories if len(t) >= min_length])),
        "per_frame_rel_flicker": per_frame_median_flicker,
        "per_frame_rel_jerk": per_frame_median_jerk,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("depth_dir", help="Directory with depth_{idx}.npy / mask_{idx}.npy / flow_{idx}.npy")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON results here")
    parser.add_argument("--target-points", type=int, default=1500, help="Approx. number of FG seed points (frame 0)")
    parser.add_argument("--drift-window", type=int, default=15, help="Median-filter window (frames) for drift/flicker split")
    parser.add_argument("--min-length", type=int, default=20, help="Drop tracks shorter than this many frames")
    args = parser.parse_args()

    depths, masks, flows = load_sequence(Path(args.depth_dir))
    ys0, xs0 = pick_fg_seed_points(masks[0], args.target_points)
    if len(ys0) == 0:
        raise SystemExit("No foreground pixels in frame 0 -- nothing to track")

    trajectories = track_points(depths, masks, flows, ys0, xs0)
    result = decompose_variable(trajectories, args.drift_window, args.min_length)
    if result is None:
        raise SystemExit(f"No track survived --min-length={args.min_length} frames")

    print(f"Seeded {result['n_points_seeded']} FG points, kept {result['n_points_kept']} "
          f"(>= {args.min_length} frames, median length {result['median_track_length']:.0f})")
    print(f"  rel_flicker  median={result['rel_flicker_median']:.4f}  p90={result['rel_flicker_p90']:.4f}")
    print(f"  rel_jerk     median={result['rel_jerk_median']:.4f}  p90={result['rel_jerk_p90']:.4f}")
    print(f"  drift_range  median={result['drift_range_median']:.4f}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
