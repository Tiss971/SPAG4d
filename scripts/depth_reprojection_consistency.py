#!/usr/bin/env python3
"""Flow-warped depth-reprojection consistency: a self-referential reconstruction
metric that needs no ground truth.

For each consecutive pair (depth_{i-1}, depth_i) with a saved flow_i (WAFT flow
i-1 -> i, same convention as flow_depth_propagation.warp_backward), warp
depth_{i-1} into frame i's grid and compare to the actual depth_i. A perfectly
consistent reconstruction would have warp(depth_{i-1}) == depth_i wherever the
surface didn't actually move/get occluded; large disagreement flags frames
where the monocular depth estimate is jumping around independent of real
scene motion -- exactly the kind of "depth flicker" bug report this is meant
to catch, without needing a renderer or 3D ground truth.

Usage:
  python scripts/depth_reprojection_consistency.py <run_dir> [--plot out.png]

  <run_dir> is a benchmark output folder containing depth_maps/depth_i.npy and
  depth_maps/flow_i.npy (e.g. .../accident_electrique_02).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spag4d.flow_depth_propagation import warp_backward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    depth_dir = args.run_dir / "depth_maps"
    depth_idx = sorted(
        int(m.group(1)) for f in depth_dir.glob("depth_*.npy")
        if (m := re.search(r"depth_(\d+)\.npy", f.name))
    )
    flow_idx = set(
        int(m.group(1)) for f in depth_dir.glob("flow_*.npy")
        if (m := re.search(r"flow_(\d+)\.npy", f.name))
    )

    rows = []
    prev_i, prev_depth = None, None
    for i in depth_idx:
        depth = np.load(depth_dir / f"depth_{i}.npy")
        if prev_depth is not None and i in flow_idx and i == prev_i + 1:
            flow = np.load(depth_dir / f"flow_{i}.npy")
            pred = warp_backward(prev_depth, flow)
            err = np.abs(pred - depth)
            rows.append({
                "frame": i,
                "mean_err": float(err.mean()),
                "p95_err": float(np.percentile(err, 95)),
                "max_err": float(err.max()),
                "frac_gt_1m": float((err > 1.0).mean()),
            })
        prev_i, prev_depth = i, depth

    print(f"{'frame':>6} {'mean_err':>9} {'p95_err':>9} {'max_err':>9} {'frac>1m':>9}")
    for r in rows:
        print(f"{r['frame']:>6} {r['mean_err']:>9.4f} {r['p95_err']:>9.4f} {r['max_err']:>9.3f} {r['frac_gt_1m']:>9.4f}")

    mean_errs = np.array([r["mean_err"] for r in rows])
    thresh = mean_errs.mean() + 2 * mean_errs.std()
    flagged = [r["frame"] for r in rows if r["mean_err"] > thresh]
    print(f"\nmean(mean_err)={mean_errs.mean():.4f} std={mean_errs.std():.4f} "
          f"flag-threshold(mean+2std)={thresh:.4f}")
    print(f"flagged frames (reconstruction-consistency outliers): {flagged}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frames = [r["frame"] for r in rows]
        plt.figure(figsize=(10, 4))
        plt.plot(frames, mean_errs, label="mean_err")
        plt.axhline(thresh, color="r", linestyle="--", label="flag threshold")
        plt.xlabel("frame")
        plt.ylabel("flow-warped depth reprojection error (m)")
        plt.title(f"Depth reprojection consistency: {args.run_dir.name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
