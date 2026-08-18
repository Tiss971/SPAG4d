"""Fast-loop runner for the bglock_open_questions audit (T1/T3 instrumentation).

Config rationale (see benchmarks/BENCHMARK_RULES.md):
  - SPAG_SAM3_MAXSIZE=1536 alone, bf16 OFF: the only SAM3 VRAM lever validated
    non-destructive on thin-limb clips (mask IoU 0.87-0.93). Stacking it with
    SPAG_SAM3_BF16 collapses IoU to 0.35-0.40 deterministically.
  - SPAG_DETERMINISTIC=1 on every side of every comparison.
  - depth_npy_dir always set, so mask/depth/flow are dumped for post-hoc IoU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_bglock_audit.py <clip.mp4> <out_dir>
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spag4d.core import SPAG4D
from spag4d.video import run_video
from benchmark_solutions import calculate_temporal_stability

KWARGS = dict(
    skip_step=2,
    stride=4,
    active_generator="da360",
    depth_correction="bglock",
    depth_smoothing=True,
    depth_smoothing_window=5,
    depth_smoothing_method="median",
    temporal_consistency=False,
    freeze_bg=True,
    outlier_pruning=0.0,  # SOR deletes far/bg points permanently under bglock, never comes back
    grazing_angle=85.0,
    sparse_pruning=0.1,
)


class _Tee(io.TextIOBase):
    """Pass stdout through untouched while keeping a copy, so n_objects can be
    parsed from run_video's own log line (BENCHMARK_RULES rule 5: n_objects is
    not on ConversionResult, it only appears in stdout)."""

    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.real.write(s)

    def flush(self):
        self.real.flush()


_N_OBJ_RE = re.compile(r"Propagating (\d+) moving object")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.out_dir / "depth_maps"
    depth_dir.mkdir(exist_ok=True)
    os.environ["SPAG_CONF_HIST"] = str(args.out_dir / "confidence_trace.json")

    converter = SPAG4D(device="cuda")
    tee = _Tee(sys.stdout)
    t0 = time.time()
    with contextlib.redirect_stdout(tee):
        res = run_video(
            converter=converter,
            video_path=args.video,
            output_path=str(args.out_dir),
            depth_npy_dir=str(depth_dir),
            **KWARGS,
        )
    elapsed = time.time() - t0
    n_obj_hits = _N_OBJ_RE.findall(tee.buf.getvalue())
    n_objects = int(n_obj_hits[0]) if n_obj_hits else None

    n_frames = len(res.splat_count) if res.splat_count else 0
    info = {
        "config": {
            **{k: str(v) for k, v in KWARGS.items()},
            "env": {k: os.environ.get(k) for k in
                    ("SPAG_SAM3_MAXSIZE", "SPAG_SAM3_BF16", "SPAG_DETERMINISTIC",
                     "SPAG_OCCL_FBGATE", "SPAG_LOCK_ACTIVITY", "CUDA_VISIBLE_DEVICES",
                     "SPAG_CONF_LEGACY", "SPAG_CONF_DECAY", "SPAG_CONF_FLOOR")},
        },
        "time_seconds": elapsed,
        "vram_max_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "n_frames": n_frames,
        "n_objects": n_objects,
        "mean_splats": float(np.mean(res.splat_count)) if res.splat_count else None,
        "stability": calculate_temporal_stability(res.depth_metrics) if res.depth_metrics else None,
    }
    (args.out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"FINAL: {elapsed:.1f}s, {n_frames} frames, vram={info['vram_max_mb']:.0f}MB")
    print("EXIT 0")


if __name__ == "__main__":
    main()
