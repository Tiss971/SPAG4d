#!/usr/bin/env python3
"""
One-off: run the bglock_sol1_median_w5 config (from benchmark_solutions.py) on
accident_electrique_fast2.mp4, writing straight into FreeTimeGsVanilla's _data/
convention (gaussians/, depth_maps/, images/, ...) with mask/freeze-bg export
enabled, so combine_frames_fast_keyframes.py can pick up is_background tagging.
"""
from pathlib import Path

from benchmark_solutions import BASE_KWARGS, CONFIGS

from spag4d.core import SPAG4D
from spag4d.video import run_video

VIDEO_PATH = "/raid/mb273924/_DATASETS/uptale/data/accident_electrique_fast2.mp4"
OUTPUT_PATH = Path("/raid/mb273924/FreeTimeGsVanilla/_data/accident_electrique_fast2")

kwargs = dict(BASE_KWARGS)
kwargs.update(CONFIGS["bglock_sol1_median_w5"])

converter = SPAG4D(device="cuda")

result = run_video(
    converter=converter,
    video_path=VIDEO_PATH,
    output_path=str(OUTPUT_PATH),
    active_generator="da360",
    depth_npy_dir=OUTPUT_PATH / "depth_maps",
    **kwargs,
)

n_frames = len(result.splat_count) if result.splat_count else 0
mean_splats = sum(result.splat_count) / n_frames if n_frames else 0
print(f"\nDone. frames={n_frames} mean_splats={mean_splats:.0f}")
print(f"Output: {OUTPUT_PATH}")
