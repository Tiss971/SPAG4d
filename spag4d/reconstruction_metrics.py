"""Reconstruction self-consistency metrics computed from already-saved
per-frame depth/flow/mask npy artifacts (depth_npy_dir). No ground truth needed.

See scripts/depth_reprojection_consistency.py for the original standalone CLI
this was lifted from.
"""
import re
from pathlib import Path

import cv2
import numpy as np

from spag4d.flow_depth_propagation import warp_backward

# Foreground silhouette edges are where flow-warp disagreement is EXPECTED
# (occlusion/dis-occlusion as a moving object's boundary reveals/hides
# background) even for a perfectly consistent reconstruction -- see
# atelier_1 frame 84 case study, 2026-09-02. Eroding the fg mask before
# computing fg_mean_err keeps only the object's interior, isolating "is this
# object's own depth internally coherent as it moves" from that edge noise.
_FG_ERODE_PX = 9


def compute_depth_reprojection_consistency(depth_npy_dir: Path | str) -> dict | None:
    """Flow-warp depth_{i-1} into frame i's grid (WAFT flow_i, i-1 -> i) and
    compare to the actual depth_i, split by static-background vs.
    moving-foreground (mask_{i}.npy, fused SAM mask):

    - bg_mean_err: background-only disagreement (mask==0). This is the
      primary flicker signal and what `flagged_frames` is based on -- bglock
      background should sit still, so any warp disagreement there is a real
      depth-instability bug, not scene motion.
    - fg_mean_err: foreground-only disagreement on the mask's ERODED
      interior (mask==1, boundary stripped). Informational, not used for
      flagging: it answers "does this moving object's depth stay internally
      coherent frame to frame" without being swamped by expected edge
      occlusion noise. High fg_mean_err with a low bg_mean_err means the
      background is solid but the tracked object's own shape/depth is
      unstable as it moves -- worth a look, but a different failure mode
      than background flicker.

    Returns None if depth_npy_dir doesn't exist or has fewer than 2 usable
    frames (nothing to compare consecutively).
    """
    depth_npy_dir = Path(depth_npy_dir)
    if not depth_npy_dir.is_dir():
        return None

    depth_idx = sorted(
        int(m.group(1)) for f in depth_npy_dir.glob("depth_*.npy")
        if (m := re.search(r"depth_(\d+)\.npy", f.name))
    )
    flow_idx = set(
        int(m.group(1)) for f in depth_npy_dir.glob("flow_*.npy")
        if (m := re.search(r"flow_(\d+)\.npy", f.name))
    )
    mask_idx = set(
        int(m.group(1)) for f in depth_npy_dir.glob("mask_*.npy")
        if (m := re.search(r"mask_(\d+)\.npy", f.name))
    )
    if len(depth_idx) < 2:
        return None

    erode_kernel = np.ones((_FG_ERODE_PX, _FG_ERODE_PX), np.uint8)

    per_frame = []
    prev_i, prev_depth = None, None
    for i in depth_idx:
        depth = np.load(depth_npy_dir / f"depth_{i}.npy")
        if prev_depth is not None and i in flow_idx and i == prev_i + 1:
            flow = np.load(depth_npy_dir / f"flow_{i}.npy")
            pred = warp_backward(prev_depth, flow, edge_aware=True)
            err = np.abs(pred - depth)

            row = {
                "frame": i,
                "mean_err": float(err.mean()),
                "p95_err": float(np.percentile(err, 95)),
                "max_err": float(err.max()),
                "frac_gt_1m": float((err > 1.0).mean()),
                "bg_mean_err": None,
                "fg_mean_err": None,
                "fg_frac_px": None,
            }
            if i in mask_idx:
                mask = np.load(depth_npy_dir / f"mask_{i}.npy").astype(np.uint8)
                bg = mask == 0
                fg_eroded = cv2.erode(mask, erode_kernel, iterations=1) > 0
                row["bg_mean_err"] = float(err[bg].mean()) if bg.any() else None
                row["fg_mean_err"] = float(err[fg_eroded].mean()) if fg_eroded.any() else None
                row["fg_frac_px"] = float(fg_eroded.mean())
            per_frame.append(row)
        prev_i, prev_depth = i, depth

    if not per_frame:
        return None

    mean_errs = np.array([r["mean_err"] for r in per_frame])
    thresh = float(mean_errs.mean() + 2 * mean_errs.std())
    flagged = [r["frame"] for r in per_frame if r["mean_err"] > thresh]

    bg_errs = np.array([r["bg_mean_err"] for r in per_frame if r["bg_mean_err"] is not None])
    bg_thresh = float(bg_errs.mean() + 2 * bg_errs.std()) if bg_errs.size else None
    bg_flagged = (
        [r["frame"] for r in per_frame if r["bg_mean_err"] is not None and r["bg_mean_err"] > bg_thresh]
        if bg_thresh is not None else []
    )
    fg_errs = np.array([r["fg_mean_err"] for r in per_frame if r["fg_mean_err"] is not None])

    return {
        "per_frame": per_frame,
        "mean_err": float(mean_errs.mean()),
        "std_err": float(mean_errs.std()),
        "max_mean_err": float(mean_errs.max()),
        "flag_threshold": thresh,
        "flagged_frames": flagged,
        "bg_mean_err": float(bg_errs.mean()) if bg_errs.size else None,
        "bg_max_mean_err": float(bg_errs.max()) if bg_errs.size else None,
        "bg_flag_threshold": bg_thresh,
        "bg_flagged_frames": bg_flagged,
        "fg_mean_err": float(fg_errs.mean()) if fg_errs.size else None,
        "fg_max_mean_err": float(fg_errs.max()) if fg_errs.size else None,
    }
