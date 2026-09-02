"""Render an RGB video from a per-frame gaussians/ PLY sequence, camera offset
ramping from 0 up to a small max translation over the clip.

Purpose: BENCHMARK_RULES rule 8 (visual sanity check) covers depth/mask via
render_depth_colormap.py, but neither depth-CV nor mask IoU can catch a splat that
looks flat/wrong under parallax (e.g. a foreground object reconstructed as a thin
shell rather than real geometry) — a translated RGB view exposes that at a glance
because static background parallaxes correctly while broken/flat geometry warps or
tears. The offset ramps from 0 (frame 0 matches the input photo exactly, so any
early-frame artifact is unambiguously a geometry defect, not a translation choice)
up to --tx/--ty/--tz by the last frame, growing the parallax as the clip plays
instead of jumping straight to the full offset.

Reuses spag4d/refine/geometric/render_utils.render_base_from_pose (gsplat cube-face
->ERP composition), the same renderer the geometric-refine pipeline uses to score
its own splats against novel poses.

Usage:
    python scripts/render_translated_rgb.py <gaussians_dir> <out.mp4> \\
        --tx 0.1 --ty 0 --tz 0 --fps 10 --resolution 960 1920
"""
import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _RawGaussians:
    """Minimal duck-typed stand-in for GSFix3D's GaussianModel.

    third_party/GSFix3D is an empty vendor directory in this environment (no `gs`
    module in any conda env), so spag4d.refine.format_compat.load_gaussians_from_ply
    can't be used. render_base_from_pose only ever reads these 5 properties
    (raw, pre-activation — it applies exp/sigmoid/etc itself), so a plain PLY
    reader matching spag4d/ply_writer.py's on-disk schema is enough to reuse it
    unmodified.
    """

    def __init__(self, xyz, rot, scale, opacity, features):
        self._xyz = xyz
        self._rot = rot
        self._scale = scale
        self._opacity = opacity
        self._features = features

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return self._rot

    @property
    def get_scaling(self):
        return self._scale

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_features(self):
        return self._features


def load_gaussians_from_ply_standalone(ply_path: str, device="cpu") -> _RawGaussians:
    """Read a SPAG4D PLY (spag4d/ply_writer.py's save_ply_gsplat schema) into raw tensors."""
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    opacity = np.asarray(v["opacity"], dtype=np.float32)[:, None]
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    features = sh_dc[:, None, :]  # (N, 1, 3) — SH degree-0 band, matches get_features[:, 0, :] usage

    return _RawGaussians(
        xyz=torch.from_numpy(xyz).to(device),
        rot=torch.from_numpy(rot).to(device),
        scale=torch.from_numpy(scale).to(device),
        opacity=torch.from_numpy(opacity).to(device),
        features=torch.from_numpy(features).to(device),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gaussians_dir", type=Path, help="Directory with frame_{idx}.ply")
    p.add_argument("out_mp4")
    p.add_argument("--tx", type=float, default=0.1, help="Camera x-translation reached by the last frame (m)")
    p.add_argument("--ty", type=float, default=0.0, help="Camera y-translation reached by the last frame (m)")
    p.add_argument("--tz", type=float, default=0.0, help="Camera z-translation reached by the last frame (m)")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--resolution", type=int, nargs=2, default=[960, 1920], metavar=("H", "W"))
    p.add_argument("--face-size", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=None, help="Cap for a fast preview run")
    p.add_argument("--hold-frac", type=float, default=0.0,
                   help="Fraction of the clip (from the start) to hold at the original pose "
                        "before ramping to --tx/--ty/--tz over the remaining frames")
    args = p.parse_args()

    from spag4d.refine.geometric.render_utils import render_base_from_pose

    idxs = sorted(
        (int(m.group(1)), p_) for p_ in args.gaussians_dir.glob("frame_*.ply")
        if (m := re.match(r"frame_(\d+)\.ply$", p_.name))
    )
    if not idxs:
        raise SystemExit(f"No frame_*.ply found in {args.gaussians_dir}")
    if args.max_frames:
        idxs = idxs[: args.max_frames]

    H, W = args.resolution
    writer = cv2.VideoWriter(args.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    n = len(idxs)
    for i, (idx, ply_path) in enumerate(idxs):
        raw_t = i / max(n - 1, 1)  # 0 at first frame, 1 at last frame
        t = 0.0 if raw_t <= args.hold_frac else (raw_t - args.hold_frac) / max(1.0 - args.hold_frac, 1e-8)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [args.tx * t, args.ty * t, args.tz * t]

        gaussians = load_gaussians_from_ply_standalone(str(ply_path))
        render = render_base_from_pose(gaussians, pose, (H, W), face_size=args.face_size)
        rgb_u8 = np.clip(render.rgb * 255.0, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        if (i + 1) % 20 == 0 or i + 1 == len(idxs):
            print(f"  [{i + 1}/{len(idxs)}] frame_{idx}", flush=True)

    writer.release()
    print(f"Wrote {len(idxs)} frames to {args.out_mp4} "
          f"(hold at original pose for {args.hold_frac:.0%} of clip, then ramping to "
          f"tx={args.tx} ty={args.ty} tz={args.tz}m, {W}x{H})")


if __name__ == "__main__":
    main()
