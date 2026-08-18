"""Run a video through affine vs bglock and dump per-frame depth+mask for
scripts/eval_bg_point_stability.py / eval_fg_point_stability.py.

Generalized driver for the benchmarks/<scene>/{affine,bglock} comparison
(originally scene01-only in run_scene01_bg_stability_bench.py). Uses the same
BASE_KWARGS as benchmark_solutions.py (stride=8, skip_step=4) so it stays fast.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/run_bg_stability_bench.py \\
        /raid/mb273924/_DATASETS/uptale/data/videos/accident_electrique_02.mp4 \\
        benchmarks/accident_electrique_02
"""
import argparse
import json
import time
from pathlib import Path

import torch

from spag4d.core import SPAG4D
from spag4d.video import run_video

BASE_KWARGS = dict(
    skip_step=4,
    stride=8,
    active_generator="da360",
    temporal_consistency=False,
    freeze_bg=False,
    outlier_pruning=0.3,
    grazing_angle=85.0,
    sparse_pruning=0.1,
)

CONFIGS = {
    "affine": dict(depth_correction="affine"),
    "bglock": dict(depth_correction="bglock"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("out_root", type=Path, help="Output root dir (will contain affine/ and bglock/ subdirs)")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    args = parser.parse_args()

    converter = SPAG4D(device="cuda")
    for name in args.configs:
        extra = CONFIGS[name]
        out_dir = args.out_root / name
        depth_dir = out_dir / "depth_maps"
        depth_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        res = run_video(
            converter=converter,
            video_path=args.video,
            output_path=str(out_dir),
            depth_npy_dir=str(depth_dir),
            **BASE_KWARGS,
            **extra,
        )
        elapsed = time.time() - t0
        max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
        n_frames = len(res.splat_count) if res.splat_count else 0
        print(f"{name}: {elapsed:.1f}s, {n_frames} frames, vram={max_vram_mb:.0f}MB", flush=True)
        (out_dir / "run_info.json").write_text(json.dumps({
            "config": name, "extra_kwargs": {k: str(v) for k, v in extra.items()},
            "base_kwargs": {k: str(v) for k, v in BASE_KWARGS.items()},
            "time_seconds": elapsed, "vram_max_mb": max_vram_mb, "n_frames": n_frames,
        }, indent=2))


if __name__ == "__main__":
    main()
