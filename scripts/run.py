"""Full-frame, full-stride bglock run (no skip_step/stride subsampling) on a
new dataset, for a presentation-quality (not benchmark-quality) render.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_scene03_fullres_bglock.py \\
        /raid/mb273924/_DATASETS/uptale/data/videos/scene_03.mp4 \\
        .sandbox/bglock_vs_affine/scene_03_fullres/bglock
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spag4d.pipeline.core import SPAG4D
from spag4d.pipeline.video import run_video

KWARGS = dict(
    skip_step=2,
    stride=4,
    active_generator="da360",
    depth_correction="bglock",
    temporal_consistency=False,
    freeze_bg=False,
    outlier_pruning=0.0,
    grazing_angle=85.0,
    sparse_pruning=0.05,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("out_dir", type=Path, help="Output dir")
    args = parser.parse_args()

    converter = SPAG4D(device="cuda")
    depth_dir = args.out_dir / "depth_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== bglock full-frame/full-stride on {args.video} ===", flush=True)
    t0 = time.time()
    res = run_video(
        converter=converter,
        video_path=args.video,
        output_path=str(args.out_dir),
        depth_npy_dir=str(depth_dir),
        **KWARGS,
    )
    elapsed = time.time() - t0
    max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    n_frames = len(res.splat_count) if res.splat_count else 0
    print(f"bglock: {elapsed:.1f}s, {n_frames} frames, vram={max_vram_mb:.0f}MB", flush=True)
    (args.out_dir / "run_info.json").write_text(json.dumps({
        "config": "bglock", "kwargs": {k: str(v) for k, v in KWARGS.items()},
        "time_seconds": elapsed, "vram_max_mb": max_vram_mb, "n_frames": n_frames,
    }, indent=2))


if __name__ == "__main__":
    main()
