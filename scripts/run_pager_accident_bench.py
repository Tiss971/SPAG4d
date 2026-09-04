"""Run accident_electrique_02 through active_generator="pager" (spatial-only
cubemap+DA3, no temporal batching) and dump per-frame depth+mask, so the
result is directly comparable to the existing da360 benchmarks/accident_electrique_02/{affine,bglock}
runs via eval_fg_point_stability.py / eval_bg_point_stability.py.

Same BASE_KWARGS (skip_step=4, stride=8) as run_bg_stability_bench.py /
benchmark_solutions.py, generator swapped to "pager". depth_correction stays
"bglock" (the shipped default) so only the depth-estimation backend changes,
not the stabilization stage.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/run_pager_accident_bench.py \\
        /raid/mb273924/_DATASETS/uptale/data/videos/accident_electrique_02.mp4 \\
        benchmarks/accident_electrique_02/pager
"""
import argparse
import json
import time
from pathlib import Path

import torch

from spag4d.pipeline.core import SPAG4D
from spag4d.pipeline.video import run_video

BASE_KWARGS = dict(
    skip_step=4,
    stride=8,
    active_generator="pager",
    depth_correction="bglock",
    temporal_consistency=False,
    freeze_bg=False,
    outlier_pruning=0.3,
    grazing_angle=85.0,
    sparse_pruning=0.1,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("out_dir", type=Path, help="Output dir (will contain depth_maps/)")
    args = parser.parse_args()

    converter = SPAG4D(device="cuda")
    depth_dir = args.out_dir / "depth_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== pager ===", flush=True)
    t0 = time.time()
    res = run_video(
        converter=converter,
        video_path=args.video,
        output_path=str(args.out_dir),
        depth_npy_dir=str(depth_dir),
        **BASE_KWARGS,
    )
    elapsed = time.time() - t0
    max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats()
    n_frames = len(res.splat_count) if res.splat_count else 0
    print(f"pager: {elapsed:.1f}s, {n_frames} frames, vram={max_vram_mb:.0f}MB", flush=True)
    (args.out_dir / "run_info.json").write_text(json.dumps({
        "config": "pager", "kwargs": {k: str(v) for k, v in BASE_KWARGS.items()},
        "time_seconds": elapsed, "vram_max_mb": max_vram_mb, "n_frames": n_frames,
    }, indent=2))


if __name__ == "__main__":
    main()
