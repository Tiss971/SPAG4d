"""Export per-frame forward WAFT flow for a scene whose frames already exist.

run_video() writes flow_{idx}.npy only as a side effect of the bglock depth
path, so scenes converted before that export existed have depth_{idx}.npy and
mask_{idx}.npy but no flow -- and FreeTimeGS's velocity init silently falls back
to KNN for them. Re-running the whole pipeline just to recover the flow would
redo depth estimation and SAM3 tracking, which are by far the expensive parts
and whose outputs are already on disk and unchanged.

The flow is a pure function of the exported RGB frames, so this reproduces it
directly: same WAFT checkpoint, same <=1024px downscale, same seam padding, same
upscale back to native resolution as spag4d/video.py's bglock branch. Frames are
read from the images/ folder run_video() exported with cv2.imwrite, so cv2.imread
hands back the identical BGR arrays it fed WAFT.

Only the forward flow is computed (run_video also computes the backward flow, but
that feeds propagate_depth_via_flow's consistency check and is never written to
disk), which halves the WAFT calls per pair.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/export_flow_only.py \\
        _data/tissc/images _data/tissc/depth_maps
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from spag4d.detect_opticalflow import WAFTWrapper
from spag4d.flow_depth_propagation import (
    crop_horizontal,
    pad_circular_horizontal,
    upscale_flow,
)

# Same defaults run_video() hardcodes.
WAFT_CHECKPOINT = "/raid/mb273924/_DATASETS/uptale/tar-c-t.pth"
WAFT_CONFIG = "/raid/mb273924/WAFT/config/a1/tar-c-t.json"
MAX_SIZE = 1024


def downscale_like_pipeline(frame_bgr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Reproduce run_video()'s pre-WAFT downscale exactly.

    It uses torch bilinear interpolation followed by .byte() (truncation, not
    rounding); cv2.resize would land a LSB off on many pixels.
    """
    t = torch.from_numpy(frame_bgr).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear")
    return t[0].permute(1, 2, 0).byte().numpy()


def main():
    parser = argparse.ArgumentParser(description="Export forward WAFT flow from exported ERP frames")
    parser.add_argument("images_dir", help="Folder of frame_%04d.png as exported by run_video()")
    parser.add_argument("output_dir", help="Where to write flow_{idx}.npy (normally the scene's depth_maps/)")
    parser.add_argument("--name-pattern", default="frame_{idx:04d}.png")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None, help="Inclusive; default = last frame present")
    parser.add_argument("--seam-pad", type=int, default=64,
                        help="Circular horizontal padding, matching run_video()'s flow_seam_pad")
    parser.add_argument("--overwrite", action="store_true", help="Recompute flows that already exist")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_end = args.frame_end
    if frame_end is None:
        idx = args.frame_start
        while (images_dir / args.name_pattern.format(idx=idx + 1)).exists():
            idx += 1
        frame_end = idx
    n_frames = frame_end - args.frame_start + 1
    if n_frames < 2:
        raise SystemExit(f"Need at least 2 frames, got {n_frames}")

    def load(idx: int) -> np.ndarray:
        path = images_dir / args.name_pattern.format(idx=idx)
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read {path}")
        return frame

    first = load(args.frame_start)
    H, W = first.shape[:2]
    scale = min(MAX_SIZE / max(W, H), 1.0)  # never upscale
    if scale < 1.0:
        new_W = int(W * scale) & ~1
        new_H = int(H * scale) & ~1
    else:
        new_W, new_H = W, H
    print(f"[flow-only] {n_frames} frames at {W}x{H}, WAFT at {new_W}x{new_H}, seam_pad={args.seam_pad}")

    model = WAFTWrapper(checkpoint=WAFT_CHECKPOINT, config=WAFT_CONFIG)

    prev = downscale_like_pipeline(first, new_H, new_W) if scale < 1.0 else first
    n_written = n_skipped = 0
    t0 = time.time()
    for idx in tqdm(range(args.frame_start + 1, frame_end + 1), desc="[flow-only]", unit="pair"):
        out_path = output_dir / f"flow_{idx}.npy"
        if out_path.exists() and not args.overwrite:
            # Still need this frame as the next pair's left operand.
            cur = load(idx)
            prev = downscale_like_pipeline(cur, new_H, new_W) if scale < 1.0 else cur
            n_skipped += 1
            continue

        cur = load(idx)
        cur = downscale_like_pipeline(cur, new_H, new_W) if scale < 1.0 else cur

        flow = crop_horizontal(
            model.infer_pair(
                pad_circular_horizontal(prev, args.seam_pad),
                pad_circular_horizontal(cur, args.seam_pad),
            ),
            args.seam_pad,
            new_W,
        )
        if scale < 1.0:
            flow = upscale_flow(flow, H, W)
        np.save(out_path, flow.astype(np.float32))

        prev = cur
        n_written += 1

    dt = time.time() - t0
    rate = n_written / dt if n_written else 0.0
    print(f"[flow-only] Wrote {n_written} flows ({n_skipped} skipped) in {dt:.1f}s ({rate:.2f} pair/s)")


if __name__ == "__main__":
    main()
