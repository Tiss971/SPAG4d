#!/usr/bin/env python3
"""One-off benchmark run: bglock_sol1_median_w5 config, with freeze_bg + live
bg color, SAM3 bf16 + maxsize=1536 (per user request -- CLAUDE.md flags this
combo as a confirmed IoU regression, running anyway), skip_step=4 (matches
benchmark_solutions.py's BASE_KWARGS convention), and raw depth_maps/ npy dump.

Each video runs in its own subprocess (--single is the actual worker) so
ram_max_mb (resource.getrusage peak RSS) is per-video, not a cumulative peak
across the whole sweep -- ru_maxrss never resets within one process.

See benchmarks/BENCHMARK_RULES.md for the benchmarks/ output convention.

Usage: python scripts/run_atelier1_bglock_sol1.py <video_name> [<video_name> ...]
  (video_name is the .mp4 stem under /raid/mb273924/_DATASETS/uptale/data/videos/)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("SPAG_SAM3_BF16", "1")
os.environ.setdefault("SPAG_SAM3_MAXSIZE", "1536")

VIDEO_DIR = Path("/raid/mb273924/_DATASETS/uptale/data/videos")
OUTPUT_BASE = Path("/raid/mb273924/SPAG4d/benchmarks/new_bchmk_2026-08-31")
CONFIG_NAME = "bglock_sol1_median_w5_livebg_sam3bf16_maxsize1536"

BASE_KWARGS = dict(
    skip_step=4,
    stride=8,
    temporal_consistency=False,
    outlier_pruning=0.1,
    grazing_angle=85.0,
    sparse_pruning=0.1,
)

EXTRA_KWARGS = dict(
    depth_correction="bglock",
    depth_smoothing=True,
    depth_smoothing_window=5,
    depth_smoothing_method="median",
    freeze_bg=True,
    freeze_bg_live_color=True,
)


def run_single(video_name: str):
    """Actual worker: runs one video in THIS process. Invoked as a subprocess
    per video by main() so ram_max_mb reflects only this video."""
    import numpy as np
    import torch

    from spag4d.core import SPAG4D
    from spag4d.video import run_video

    video_path = VIDEO_DIR / f"{video_name}.mp4"
    out_dir = OUTPUT_BASE / CONFIG_NAME / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir = out_dir / "depth_maps"

    print(f"\nRunning {video_name} -> {out_dir}")
    print(f"env: SPAG_SAM3_BF16={os.environ['SPAG_SAM3_BF16']} SPAG_SAM3_MAXSIZE={os.environ['SPAG_SAM3_MAXSIZE']}")
    print(f"kwargs: {BASE_KWARGS | EXTRA_KWARGS}")

    converter = SPAG4D(device="cuda")
    t0 = time.time()
    res = run_video(
        converter=converter,
        video_path=str(video_path),
        output_path=str(out_dir),
        active_generator="da360",
        depth_npy_dir=depth_npy_dir,
        **BASE_KWARGS,
        **EXTRA_KWARGS,
    )
    elapsed = time.time() - t0
    max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    summary = {
        "config": CONFIG_NAME,
        "video": video_name,
        "env": {"SPAG_SAM3_BF16": os.environ["SPAG_SAM3_BF16"], "SPAG_SAM3_MAXSIZE": os.environ["SPAG_SAM3_MAXSIZE"]},
        "kwargs": {k: str(v) for k, v in (BASE_KWARGS | EXTRA_KWARGS).items()},
        "time_seconds": elapsed,
        "vram_max_mb": max_vram_mb,
        "ram_max_mb": res.ram_max_mb,
        "n_frames": len(res.splat_count) if res.splat_count else 0,
        "mean_splats": float(np.mean(res.splat_count)) if res.splat_count else None,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"done {elapsed:.1f}s | vram={max_vram_mb:.0f}MB | ram={res.ram_max_mb:.0f}MB | frames={summary['n_frames']}")
    print(f"results: {out_dir / 'results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="video stems under data/videos/")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        assert len(args.videos) == 1
        run_single(args.videos[0])
        return

    for video_name in args.videos:
        print(f"\n=== spawning subprocess for {video_name} ===", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, video_name, "--single"],
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            print(f"!! {video_name} subprocess exited with code {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
