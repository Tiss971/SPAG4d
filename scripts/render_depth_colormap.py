"""Render a colored-depthmap video from a SPAG4D depth_npy_dir dump.

Reads depth_{idx}.npy (float32 aligned depth, meters) + optionally
mask_{idx}.npy (fused SAM mask, drawn as a contour overlay) and writes an
mp4 with a fixed colormap + fixed depth range (so brightness is comparable
frame-to-frame and, if --depth-min/--depth-max are pinned, across separate
renders of the same clip under different configs).

Usage:
    python scripts/render_depth_colormap.py <npy_dir> <out.mp4> \\
        --fps 10 --depth-min 0.5 --depth-max 15 --colormap TURBO
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npy_dir", help="Directory with depth_{idx}.npy (+ optional mask_{idx}.npy)")
    parser.add_argument("out_mp4", help="Output video path")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--depth-min", type=float, default=None, help="Fixed depth (m) mapped to colormap 0; auto (1st pctile) if omitted")
    parser.add_argument("--depth-max", type=float, default=None, help="Fixed depth (m) mapped to colormap 255; auto (99th pctile) if omitted")
    parser.add_argument("--colormap", default="TURBO", help="cv2.COLORMAP_* name")
    parser.add_argument("--draw-mask-contour", action="store_true", default=True)
    parser.add_argument("--no-mask-contour", dest="draw_mask_contour", action="store_false")
    args = parser.parse_args()

    idxs = sorted(
        int(os.path.basename(p).split("_")[1].split(".")[0])
        for p in glob.glob(os.path.join(args.npy_dir, "depth_*.npy"))
    )
    if not idxs:
        raise SystemExit(f"No depth_*.npy found in {args.npy_dir}")

    depth_min, depth_max = args.depth_min, args.depth_max
    if depth_min is None or depth_max is None:
        sample = np.load(os.path.join(args.npy_dir, f"depth_{idxs[len(idxs)//2]}.npy"))
        finite = sample[np.isfinite(sample) & (sample > 0)]
        if depth_min is None:
            depth_min = float(np.percentile(finite, 1)) if finite.size else 0.1
        if depth_max is None:
            depth_max = float(np.percentile(finite, 99)) if finite.size else 20.0

    colormap = getattr(cv2, f"COLORMAP_{args.colormap}")

    first = np.load(os.path.join(args.npy_dir, f"depth_{idxs[0]}.npy"))
    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (W, H))

    for idx in idxs:
        depth = np.load(os.path.join(args.npy_dir, f"depth_{idx}.npy"))
        norm = np.clip((depth - depth_min) / max(depth_max - depth_min, 1e-6), 0, 1)
        norm_u8 = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, colormap)
        invalid = ~np.isfinite(depth) | (depth <= 0)
        colored[invalid] = (0, 0, 0)

        mask_path = os.path.join(args.npy_dir, f"mask_{idx}.npy")
        if args.draw_mask_contour and os.path.exists(mask_path):
            mask = (np.load(mask_path) > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(colored, contours, -1, (255, 255, 255), 2)

        writer.write(colored)

    writer.release()
    print(f"Wrote {len(idxs)} frames to {args.out_mp4} (depth range {depth_min:.2f}-{depth_max:.2f}m, {W}x{H})")


if __name__ == "__main__":
    main()
