#!/usr/bin/env python3
"""
Empirical comparison: current per-frame affine alignment (align_depth_frame)
vs. flow-based depth propagation (spag4d.flow_depth_propagation), on a fixed
360 ERP video.

Standalone: does NOT run SAM3. A dynamic/foreground mask is approximated
from WAFT flow magnitude (same fallback signal the pipeline itself computes
before SAM3 runs) — good enough to compare temporal *stability* of the two
depth-correction strategies on the pixels that matter for flicker.

Usage:
  conda run -n spag4d python benchmark_flow_propagation.py \
      --video temp_fast_track_h264.mp4 --output ./benchmark_flow_prop --max-frames 60
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from spag4d.da360_model import DA360Model
from spag4d.detect_opticalflow import WAFTWrapper
from spag4d.flow_depth_propagation import (
    PropagationState,
    compute_bidirectional_flow,
    propagate_depth_via_flow,
)
from spag4d.video import align_depth_frame, extract_video_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="./benchmark_flow_prop")
    parser.add_argument("--skip-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--motion-threshold", type=float, default=1.0, help="px, WAFT flow magnitude for the foreground proxy mask")
    parser.add_argument("--seam-pad", type=int, default=64)
    parser.add_argument("--fb-err-threshold", type=float, default=1.5)
    parser.add_argument("--pole-margin-frac", type=float, default=0.08)
    parser.add_argument(
        "--da360-weights", type=str, default=str(Path.home() / ".cache/spag4d/DA360_large.pth")
    )
    parser.add_argument(
        "--waft-checkpoint", type=str, default="/raid/mb273924/_DATASETS/uptale/tar-c-t.pth"
    )
    parser.add_argument("--waft-config", type=str, default="/raid/mb273924/WAFT/config/a1/tar-c-t.json")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print("[1/5] Extracting frames...")
    frames, meta = extract_video_frames(args.video, skip_step=args.skip_step)
    frames = frames[: args.max_frames]
    n = len(frames)
    frames_rgb = frames[:, :, :, ::-1].copy()  # BGR -> RGB, for DA360
    print(f"  {n} frames | {meta['W']}x{meta['H']}")

    print("[2/5] Loading models...")
    da360 = DA360Model.load(model_path=args.da360_weights, device=device)
    waft = WAFTWrapper(checkpoint=args.waft_checkpoint, config=args.waft_config)

    print("[3/5] Dense flow + foreground proxy mask (flow magnitude)...")
    flows_fwd, flows_bwd = [], []
    fg_masks = []  # proxy for SAM3 dynamic mask: uint8 0/1, len == n (last frame duplicates n-2 flow)
    t0 = time.time()
    for i in range(n - 1):
        ff, fb = compute_bidirectional_flow(waft, frames[i], frames[i + 1], seam_pad=args.seam_pad)
        flows_fwd.append(ff)
        flows_bwd.append(fb)
        mag = np.sqrt(ff[..., 0] ** 2 + ff[..., 1] ** 2)
        fg_masks.append((mag > args.motion_threshold).astype(np.uint8))
    fg_masks.append(
        fg_masks[-1].copy() if fg_masks else np.zeros((meta["H"], meta["W"]), dtype=np.uint8)
    )
    print(f"  flow computed in {time.time() - t0:.1f}s for {n - 1} pairs")

    print("[4/5] Reference depth (temporal median background, masked)...")

    def masked_median(frames_arr, masks):
        F_, H, W, C = frames_arr.shape
        masks_arr = np.stack(masks, axis=0).astype(bool)
        median_frame = np.zeros((H, W, C), dtype=np.float32)
        for c in range(C):
            ch = frames_arr[..., c].astype(np.float32).copy()
            ch[masks_arr] = np.nan
            with np.errstate(all="ignore"):
                med = np.nanmedian(ch, axis=0)
            median_frame[..., c] = np.nan_to_num(med, nan=0.0)
        return median_frame

    master_bg = masked_median(frames_rgb, fg_masks).astype(np.uint8)
    cv2.imwrite(str(out / "_master_bg.jpg"), cv2.cvtColor(master_bg, cv2.COLOR_RGB2BGR))

    bg_tensor = torch.from_numpy(master_bg).to(device)
    with torch.inference_mode():
        depth_raw, _ = da360.predict(bg_tensor, temporal_consistency=True)
    depth_median = float(depth_raw.median())
    scale_to_5m = 5.0 / depth_median if depth_median > 1e-6 else 1.0
    depth_ref = (depth_raw * scale_to_5m).cpu().numpy()

    print("[5/5] Per-frame depth: affine-alignment baseline vs flow-propagation...")
    median_affine_fg, median_propagated_fg = [], []
    affine_stack, propagated_stack = [], []
    state = PropagationState(decay=0.85)
    depth_prev_final = None
    debug_saved = False

    for idx in range(n):
        frame_tensor = torch.from_numpy(frames_rgb[idx]).to(device)
        with torch.inference_mode():
            depth_raw_i, _ = da360.predict(frame_tensor, temporal_consistency=True)
        depth_curr = (depth_raw_i * scale_to_5m).cpu().numpy()

        fused_mask = fg_masks[idx]
        depth_affine = align_depth_frame(depth_curr.copy(), depth_ref, fused_mask, method="lstsq").copy()

        if idx == 0 or depth_prev_final is None:
            depth_final = depth_affine.copy()
            confidence = np.zeros_like(depth_affine)
        else:
            flow_fwd, flow_bwd = flows_fwd[idx - 1], flows_bwd[idx - 1]
            depth_final, confidence, debug = propagate_depth_via_flow(
                depth_prev_final,
                depth_affine,
                depth_ref,
                flow_fwd,
                flow_bwd,
                state,
                fb_err_threshold=args.fb_err_threshold,
                pole_margin_frac=args.pole_margin_frac,
                disocclusion_mask=fused_mask,
            )
            if not debug_saved and fused_mask.sum() > 0:
                cv2.imwrite(str(out / "_debug_fb_err.jpg"), (np.clip(debug["fb_err"], 0, 10) * 25.5).astype(np.uint8))
                cv2.imwrite(str(out / "_debug_confidence.jpg"), (confidence * 255).astype(np.uint8))
                cv2.imwrite(str(out / "_debug_fg_mask.jpg"), (fused_mask * 255).astype(np.uint8))
                debug_saved = True

        depth_prev_final = depth_final

        fg = fused_mask > 0
        if fg.sum() > 0:
            median_affine_fg.append(float(np.median(depth_affine[fg])))
            median_propagated_fg.append(float(np.median(depth_final[fg])))
        else:
            median_affine_fg.append(float("nan"))
            median_propagated_fg.append(float("nan"))

        affine_stack.append(depth_affine)
        propagated_stack.append(depth_final)

    affine_stack = np.stack(affine_stack, axis=0)
    propagated_stack = np.stack(propagated_stack, axis=0)
    fg_stack = np.stack(fg_masks[:n], axis=0).astype(bool)

    def fg_temporal_std(stack, fg_mask_stack):
        std_map = np.nanstd(stack, axis=0)
        any_fg = fg_mask_stack.any(axis=0)
        if any_fg.sum() == 0:
            return float("nan")
        return float(std_map[any_fg].mean())

    stats = {
        "n_frames": n,
        "affine_fg_temporal_std_mean": fg_temporal_std(affine_stack, fg_stack),
        "propagated_fg_temporal_std_mean": fg_temporal_std(propagated_stack, fg_stack),
        "affine_median_fg_series": median_affine_fg,
        "propagated_median_fg_series": median_propagated_fg,
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    plt.figure(figsize=(12, 5))
    plt.plot(median_affine_fg, marker="o", label="Affine alignment (current)", color="orange")
    plt.plot(median_propagated_fg, marker="o", label="Flow propagation (prototype)", color="green")
    plt.title("Profondeur médiane du foreground (masque flow) — comparaison")
    plt.xlabel("Frame index")
    plt.ylabel("Profondeur médiane (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "comparison.png")

    print("\n=== RESULTS ===")
    print(f"Affine alignment  — mean temporal std over FG pixels: {stats['affine_fg_temporal_std_mean']:.4f} m")
    print(f"Flow propagation  — mean temporal std over FG pixels: {stats['propagated_fg_temporal_std_mean']:.4f} m")
    print(f"Saved: {out}/stats.json, {out}/comparison.png")


if __name__ == "__main__":
    main()
