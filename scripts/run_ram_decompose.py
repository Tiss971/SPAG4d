import os, sys, time, json
os.environ.setdefault("SPAG_SAM3_BF16", "1")
os.environ.setdefault("SPAG_SAM3_MAXSIZE", "1536")
os.environ["SPAG_RAM_TRACE"] = "1"
import numpy as np
import torch
from spag4d.core import SPAG4D
from spag4d.video import run_video

video_name = sys.argv[1] if len(sys.argv) > 1 else "circulation_site_1_edit_coupe"
video_path = f"/raid/mb273924/_DATASETS/uptale/data/videos/{video_name}.mp4"
out_dir = f"/raid/mb273924/SPAG4d/benchmarks/ram_trace_2026-09-02_decompose/{video_name}"

BASE_KWARGS = dict(skip_step=4, stride=8, temporal_consistency=False,
                    outlier_pruning=0.1, grazing_angle=85.0, sparse_pruning=0.1)
EXTRA_KWARGS = dict(depth_correction="bglock", depth_smoothing=True,
                     depth_smoothing_window=5, depth_smoothing_method="median",
                     freeze_bg=True, freeze_bg_live_color=True)

converter = SPAG4D(device="cuda")
t0 = time.time()
res = run_video(
    converter=converter, video_path=video_path, output_path=out_dir,
    active_generator="da360", depth_npy_dir=f"{out_dir}/depth_maps",
    **BASE_KWARGS, **EXTRA_KWARGS,
)
elapsed = time.time() - t0
max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
summary = {
    "video": video_name, "time_seconds": elapsed, "vram_max_mb": max_vram_mb,
    "ram_max_mb": res.ram_max_mb,
    "n_frames": len(res.splat_count) if res.splat_count else 0,
}
with open(f"{out_dir}/results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"done {elapsed:.1f}s | vram={max_vram_mb:.0f}MB | ram={res.ram_max_mb:.0f}MB | frames={summary['n_frames']}")
