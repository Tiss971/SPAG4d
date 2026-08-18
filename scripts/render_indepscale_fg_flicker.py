"""Render a side-by-side depth-colormap video of the dynamic-object crop,
each side colorized on its OWN 2nd-98th percentile range.

Unlike render_fg_depth_flicker.py's shared-scale rendering, this is for
comparing two configs that are NOT on the same absolute depth scale (e.g.
active_generator="da360" metric depth vs active_generator="pager" scale-
invariant depth) -- a joint colorscale there makes the smaller-scale side
look artificially flat/collapsed rather than showing its own real flicker.

Usage:
    python scripts/render_indepscale_fg_flicker.py \\
        --a-dir .sandbox/.../da360/depth_maps --a-label "da360 (bglock)" \\
        --b-dir .sandbox/.../pager/depth_maps --b-label "pager (bglock, spatial-only)" \\
        --out .sandbox/.../da360_vs_pager_indepscale_fg_flicker.mp4
"""
import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np


def load_sequence(depth_dir: Path):
    depth_files = sorted(
        depth_dir.glob("depth_*.npy"),
        key=lambda p: int(re.search(r"depth_(\d+)\.npy", p.name).group(1)),
    )
    depths, masks = [], []
    for depth_path in depth_files:
        idx = re.search(r"depth_(\d+)\.npy", depth_path.name).group(1)
        mask_path = depth_dir / f"mask_{idx}.npy"
        depths.append(np.load(depth_path))
        masks.append(np.load(mask_path))
    return np.stack(depths, axis=0), np.stack(masks, axis=0)


def union_bbox(masks: np.ndarray, margin: int = 40):
    any_fg = np.any(masks > 0, axis=0)
    ys, xs = np.nonzero(any_fg)
    H, W = masks.shape[1:]
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + margin)
    return y0, y1, x0, x1


def colorize(depth_crop: np.ndarray, vmin: float, vmax: float):
    nan_mask = ~np.isfinite(depth_crop)
    d = np.nan_to_num(depth_crop, nan=vmax)
    norm = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    colored[nan_mask] = (128, 128, 128)
    return colored


def label(img: np.ndarray, text: str):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=900)
    args = parser.parse_args()

    a_depths, a_masks = load_sequence(args.a_dir)
    b_depths, b_masks = load_sequence(args.b_dir)
    n = min(len(a_depths), len(b_depths))
    a_depths, a_masks = a_depths[:n], a_masks[:n]
    b_depths, b_masks = b_depths[:n], b_masks[:n]

    y0a, y1a, x0a, x1a = union_bbox(a_masks)
    y0b, y1b, x0b, x1b = union_bbox(b_masks)
    y0, y1 = min(y0a, y0b), max(y1a, y1b)
    x0, x1 = min(x0a, x0b), max(x1a, x1b)

    a_crop = a_depths[:, y0:y1, x0:x1]
    b_crop = b_depths[:, y0:y1, x0:x1]
    a_vmin, a_vmax = np.percentile(a_crop[np.isfinite(a_crop)], [2, 98])
    b_vmin, b_vmax = np.percentile(b_crop[np.isfinite(b_crop)], [2, 98])

    ch, cw = y1 - y0, x1 - x0
    out_w = args.panel_width
    out_h = int(round(ch * (out_w / cw)))
    out_h -= out_h % 2
    gap = 6
    frame_w = out_w * 2 + gap

    interp = cv2.INTER_AREA if out_w < cw else cv2.INTER_NEAREST
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{frame_w}x{out_h}", "-r", str(args.fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.out),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    for t in range(n):
        a_img = colorize(a_crop[t], a_vmin, a_vmax)
        b_img = colorize(b_crop[t], b_vmin, b_vmax)
        a_img = cv2.resize(a_img, (out_w, out_h), interpolation=interp)
        b_img = cv2.resize(b_img, (out_w, out_h), interpolation=interp)
        label(a_img, f"{args.label_a}  frame {t}")
        label(b_img, f"{args.label_b}  frame {t}")
        gap_col = np.full((out_h, gap, 3), 255, dtype=np.uint8)
        composite = np.concatenate([a_img, gap_col, b_img], axis=1)
        ffmpeg.stdin.write(composite.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {ffmpeg.returncode}")
    print(f"Wrote {args.out} ({n} frames @ {args.fps}fps, crop {cw}x{ch} -> {out_w}x{out_h} each panel, "
          f"a_range=[{a_vmin:.3f},{a_vmax:.3f}] b_range=[{b_vmin:.3f},{b_vmax:.3f}])")


if __name__ == "__main__":
    main()
