#!/usr/bin/env python3
"""Full 15-clip benchmark of the RAM-fix pipeline (viz_tensor/flow/mask buffer
chunking + compute_temporal_median/nanstd row-chunking, all shipped
unconditionally in spag4d/video.py -- see docs/RAM_USAGE_INVESTIGATION_PLAN.md),
run at stride=4 (denser Gaussian sampling than the reference sweep's stride=8)
with outlier_pruning=0.0.

outlier_pruning is disabled here (not just lowered) because prune_outliers is
Statistical Outlier Removal (SOR): a GLOBAL mean/std of nearest-neighbor
distance across the whole point set. At stride=4's higher point density, far
background/sky geometry becomes relatively even sparser vs. dense nearby
geometry than it already was at stride=8, so SOR's single global threshold is
more likely to flag legitimate far/background points as "outliers" and delete
them permanently -- this is the same failure mode already flagged in memory
for bglock (SOR must stay OFF, deletes far/bg points). grazing_angle (relative
depth-gradient based, distance-independent) and sparse_pruning
(prune_sparse_regions -- per-splat scale-adaptive radius, explicitly designed
to preserve legitimately sparse distant backgrounds) are NOT distance-biased
the same way and are left on unchanged.

Otherwise same config/kwargs as the reference sweep at
benchmarks/new_bchmk_2026-08-31/bglock_sol1_median_w5_livebg_sam3bf16_maxsize1536
(skip_step=4, bglock+freeze_bg_live_color, SAM3 bf16+maxsize=1536,
depth_npy_dir set).

Each video runs in its own subprocess (--single is the actual worker) so
ram_max_mb (resource.getrusage peak RSS) is per-video, not a cumulative peak
across the whole sweep -- ru_maxrss never resets within one process.

See benchmarks/BENCHMARK_RULES.md for the benchmarks/ output convention.

Usage: python scripts/run_full_clip_postfix_sweep.py <video_name> [<video_name> ...]
  (video_name is the .mp4 stem under /raid/mb273924/_DATASETS/uptale/data/videos/)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("SPAG_SAM3_BF16", "1")
os.environ.setdefault("SPAG_SAM3_MAXSIZE", "1536")
os.environ.setdefault("SPAG_RAM_TRACE", "1")

VIDEO_DIR = Path("/raid/mb273924/_DATASETS/uptale/data/videos")
OUTPUT_BASE = Path("/raid/mb273924/SPAG4d/benchmarks/postfix_sweep_2026-09-02")
CONFIG_NAME = "bglock_sol1_median_w5_livebg_sam3bf16_maxsize1536_ramfix_stride4_nosor"

BASE_KWARGS = dict(
    skip_step=4,
    stride=4,
    temporal_consistency=False,
    outlier_pruning=0.0,  # SOR disabled -- see module docstring, keeps far/bg Gaussians
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

ALL_VIDEOS = [
    "vid360_bruit_operatrice", "CIELE", "trop_long_embouteillage",
    "productionPont_2_MAX", "risque_securite_machine_2", "Dispo_RDV",
    "boutique1_HQ", "tissc", "scene01", "scene_03", "atelier_1",
    "accident_electrique_02", "projections_yeux_2", "MattSwift",
    "circulation_site_1_edit_coupe",
]


def run_single(video_name: str):
    """Actual worker: runs one video in THIS process. Invoked as a subprocess
    per video by main() so ram_max_mb reflects only this video."""
    import numpy as np
    import torch

    from spag4d.pipeline.core import SPAG4D
    from spag4d.pipeline.video import run_video

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
        "n_tracked_objects": res.n_tracked_objects,
        "mean_objects_per_frame": res.mean_objects_per_frame,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"done {elapsed:.1f}s | vram={max_vram_mb:.0f}MB | ram={res.ram_max_mb:.0f}MB | "
          f"frames={summary['n_frames']} | tracked_objs={res.n_tracked_objects}")
    print(f"results: {out_dir / 'results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=ALL_VIDEOS, help="video stems under data/videos/ (default: all 15)")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        assert len(args.videos) == 1
        run_single(args.videos[0])
        return

    config_dir = OUTPUT_BASE / CONFIG_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps({
        "config": CONFIG_NAME,
        "env": {"SPAG_SAM3_BF16": os.environ["SPAG_SAM3_BF16"], "SPAG_SAM3_MAXSIZE": os.environ["SPAG_SAM3_MAXSIZE"],
                "SPAG_RAM_TRACE": os.environ["SPAG_RAM_TRACE"]},
        "kwargs": {k: str(v) for k, v in (BASE_KWARGS | EXTRA_KWARGS).items()},
        "active_generator": "da360",
        "depth_npy_dir": "set (per-clip depth_maps/ subfolder)",
        "note": "outlier_pruning=0.0 (SOR) intentionally disabled at stride=4 to avoid "
                "deleting far/background Gaussians -- see script module docstring.",
        "videos": ALL_VIDEOS,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"config written to {config_dir / 'config.json'}")
    shutil.copy2(__file__, config_dir / "run_full_clip_postfix_sweep.py")
    print(f"script copied to {config_dir / 'run_full_clip_postfix_sweep.py'}")

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
