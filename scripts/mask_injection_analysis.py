#!/usr/bin/env python3
"""§6.3 mask injection -- proxy-GT accuracy for the bglock dynamic path.

Run the pipeline with SPAG_MASK_INJECT set and --depth-raw (depth_npy_dir), then
point this script at that directory. It scores every injected patch, every frame,
against depth_ref.npy -- the one place in the whole pipeline where the "dynamic"
path's answer is actually known, because the injected content is genuinely static.

Usage:
    python scripts/mask_injection_analysis.py <depth_npy_dir> \
        --patches '[[0.5,0.5,80],[0.15,0.5,80],[0.85,0.5,80]]' \
        --labels equator,north_pole,south_pole
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def patch_to_rect(lat_frac: float, lon_frac: float, half: int, n_rows: int, n_cols: int):
    cy = int(lat_frac * (n_rows - 1))
    cx = int(lon_frac * (n_cols - 1))
    y0, y1 = max(0, cy - half), min(n_rows, cy + half)
    x0, x1 = max(0, cx - half), min(n_cols, cx + half)
    return y0, y1, x0, x1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("depth_npy_dir")
    ap.add_argument("--patches", required=True, help="same JSON as SPAG_MASK_INJECT")
    ap.add_argument("--labels", default=None, help="comma-separated names, same order as --patches")
    args = ap.parse_args()

    d = Path(args.depth_npy_dir)
    depth_ref = np.load(d / "depth_ref.npy")
    n_rows, n_cols = depth_ref.shape

    specs = json.loads(args.patches)
    labels = args.labels.split(",") if args.labels else [f"patch{i}" for i in range(len(specs))]
    rects = [patch_to_rect(lat, lon, half, n_rows, n_cols) for lat, lon, half in specs]

    depth_files = [p for p in d.glob("depth_*.npy") if p.name != "depth_ref.npy"]
    depth_files.sort(key=lambda p: int(re.match(r"depth_(\d+)\.npy", p.name).group(1)))
    if not depth_files:
        raise SystemExit(f"No depth_{{idx}}.npy files found in {d}")

    per_patch_err = {label: [] for label in labels}
    for fpath in depth_files:
        depth = np.load(fpath)
        for label, (y0, y1, x0, x1) in zip(labels, rects):
            gt = depth_ref[y0:y1, x0:x1]
            pred = depth[y0:y1, x0:x1]
            valid = np.isfinite(gt) & np.isfinite(pred)
            if valid.sum() == 0:
                continue
            err = np.abs(pred[valid] - gt[valid])
            per_patch_err[label].append(err)

    print(f"{'patch':<14}{'lat_frac':>9}{'n_frames':>10}{'RMSE(m)':>10}{'mean|e|(m)':>12}{'p95|e|(m)':>11}{'max|e|(m)':>11}")
    print("-" * 77)
    rows = []
    for label, (lat, lon, half), errs in zip(labels, specs, per_patch_err.values()):
        if not errs:
            print(f"{label:<14}{lat:>9.2f}{'(no valid px)':>10}")
            continue
        all_err = np.concatenate(errs)
        rmse = float(np.sqrt(np.mean(all_err ** 2)))
        mean_e = float(np.mean(all_err))
        p95 = float(np.percentile(all_err, 95))
        max_e = float(np.max(all_err))
        print(f"{label:<14}{lat:>9.2f}{len(errs):>10}{rmse:>10.4f}{mean_e:>12.4f}{p95:>11.4f}{max_e:>11.4f}")
        rows.append({"label": label, "lat_frac": lat, "lon_frac": lon, "half_px": half,
                      "n_frames": len(errs), "rmse_m": rmse, "mean_abs_err_m": mean_e,
                      "p95_abs_err_m": p95, "max_abs_err_m": max_e})

    out_json = d / "mask_injection_results.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
