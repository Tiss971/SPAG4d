"""Per-frame SAM mask IoU between a matched baseline/test pair.

Implements BENCHMARK_RULES rule 2 verbatim so every packet reports the same
numbers: mean_iou, min_iou, and frames_iou_below_0.5 as a COUNT, not just a mean.
Aggregate depth-CV metrics cannot see a localized segmentation failure; this can.

Both runs must have been launched with depth_npy_dir set (mask_{idx}.npy) and be a
matched same-GPU, same-session pair (rule 1).

Usage:
    python scripts/compute_mask_iou.py <baseline_run_dir> <test_run_dir> [--out x.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_dir", type=Path)
    p.add_argument("test_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    bd = args.base_dir / "depth_maps"
    td = args.test_dir / "depth_maps"
    b_idx = {int(f.stem.split("_")[1]) for f in bd.glob("mask_*.npy")}
    t_idx = {int(f.stem.split("_")[1]) for f in td.glob("mask_*.npy")}
    shared = sorted(b_idx & t_idx)
    if not shared:
        raise SystemExit(f"no shared mask indices ({len(b_idx)} base, {len(t_idx)} test)")
    if b_idx != t_idx:
        print(f"WARNING: index sets differ - base {len(b_idx)}, test {len(t_idx)}, "
              f"shared {len(shared)}. Comparing shared only.")

    ious, areas_b, areas_t = [], [], []
    for i in shared:
        b = np.load(bd / f"mask_{i}.npy") > 0
        m = np.load(td / f"mask_{i}.npy") > 0
        if b.shape != m.shape:
            raise SystemExit(f"shape mismatch at {i}: {b.shape} vs {m.shape}")
        union = np.logical_or(b, m).sum()
        # Both masks empty = perfect agreement (nothing tracked either side), not nan.
        ious.append(1.0 if union == 0 else float(np.logical_and(b, m).sum() / union))
        areas_b.append(int(b.sum()))
        areas_t.append(int(m.sum()))

    ious = np.array(ious)
    ab, at = np.array(areas_b, float), np.array(areas_t, float)
    ratio = np.divide(at, ab, out=np.ones_like(at), where=ab > 0)
    below = int((ious < 0.5).sum())

    res = {
        "n_frames": len(shared),
        "mean_iou": float(ious.mean()),
        "min_iou": float(ious.min()),
        "frames_iou_below_0.5": below,
        "frames_iou_below_0.5_frac": float(below / len(ious)),
        "area_ratio_mean": float(ratio.mean()),
        "area_ratio_std": float(ratio.std()),
        "worst_frames": [int(shared[k]) for k in np.argsort(ious)[:5]],
    }
    print(json.dumps(res, indent=2))
    verdict = "PASS" if res["mean_iou"] >= 0.85 and below == 0 else (
        "FAIL - localized segmentation regression" if below else "MARGINAL")
    print(f"VERDICT: {verdict}  (mean_iou {res['mean_iou']:.3f}, "
          f"{below}/{len(ious)} frames below 0.5)")
    res["verdict"] = verdict
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
