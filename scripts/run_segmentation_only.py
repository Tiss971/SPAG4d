"""Fast-iteration entry point: runs only frame extraction + WAFT flow +
segment_with_flows (SAM3 tracking), then exits -- skips depth estimation and
Gaussian generation entirely (the ~250s+ slow part of run_video).

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_segmentation_only.py <video_path> <output_dir> [--skip-step N]
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from spag4d.detect_opticalflow import WAFTWrapper
from spag4d.video import extract_video_frames, reencode_h264, segment_with_flows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("output_dir")
    parser.add_argument("--skip-step", type=int, default=4)
    parser.add_argument("--size-threshold", type=int, default=50)
    parser.add_argument("--prox-threshold", type=int, default=80)
    parser.add_argument("--min-times-seen", type=int, default=3)
    parser.add_argument("--merge-gap-px", type=int, default=30)
    args = parser.parse_args()

    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    viz_frames, meta = extract_video_frames(args.video_path, output_folder, skip_step=args.skip_step, export=True)
    n_total_frames = len(viz_frames)
    print(f"[seg-only] Extracted {n_total_frames} frames in {time.time() - t0:.1f}s")

    model = WAFTWrapper(
        checkpoint='/raid/mb273924/_DATASETS/uptale/tar-c-t.pth',
        config='/raid/mb273924/WAFT/config/a1/tar-c-t.json'
    )
    MAX_SIZE = 1024
    scale = min(MAX_SIZE / max(meta["W"], meta["H"]), 1.0)
    if scale < 1.0:
        new_W = int(meta["W"] * scale) & ~1
        new_H = int(meta["H"] * scale) & ~1
        viz_tensor = torch.from_numpy(viz_frames).permute(0, 3, 1, 2).float()
        downscaled = F.interpolate(viz_tensor, size=(new_H, new_W), mode="bilinear")
        waft_frames = downscaled.permute(0, 2, 3, 1).byte().numpy()
    else:
        new_W, new_H = meta["W"], meta["H"]
        waft_frames = viz_frames
    meta["new_W"] = new_W
    meta["new_H"] = new_H

    t1 = time.time()
    stats = model.run(waft_frames, meta, output_folder, 0.5)
    try:
        reencode_h264(str(output_folder / "flows.mp4"))
    except Exception:
        pass
    flow_masks = stats["masks"]
    print(f"[seg-only] Flow computed in {time.time() - t1:.1f}s")

    t2 = time.time()
    segment_with_flows(
        args.video_path, output_folder, waft_frames, flow_masks, meta, n_total_frames,
        args.skip_step,
        size_threshold=args.size_threshold,
        prox_threshold=args.prox_threshold,
        min_times_seen=args.min_times_seen,
        merge_gap_px=args.merge_gap_px,
    )
    print(f"[seg-only] Segmentation done in {time.time() - t2:.1f}s")
    print(f"[seg-only] TOTAL: {time.time() - t0:.1f}s -> {output_folder}")


if __name__ == "__main__":
    main()
