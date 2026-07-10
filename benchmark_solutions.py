#!/usr/bin/env python3
"""
Benchmark temporal-stability solutions against a baseline.

Each "config" is a full run of the video pipeline over the same set of videos,
with the same winning generator, differing only in the solution being tested.
Stability is measured from run_video()'s depth_metrics (no PLY reading needed):

  - bg_depth_cv         : coeff. of variation of aligned background median depth
                          (PRIMARY: lower = steadier background across frames)
  - bg_spikes_per_frame : fraction of frames with a >0.1m background depth jump
  - fg_depth_cv         : coeff. of variation of stabilized foreground depth
  - fg_delta_mean       : mean frame-to-frame foreground depth change
  - time_seconds        : wall time per video

Results are written incrementally so the run is resumable and monitorable.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from spag4d.core import SPAG4D
from spag4d.video import run_video
from batch_compare_generators import calculate_temporal_stability

DEFAULT_VIDEO_DIR = "/raid/mb273924/_DATASETS/uptale/data/videos"

# Fixed pipeline config shared by every run (the recommended baseline setup).
BASE_KWARGS = dict(
    skip_step=8,
    stride=8,
    temporal_consistency=False,
    freeze_bg=True,
    outlier_pruning=0.3,
    grazing_angle=85.0,
    sparse_pruning=0.1,
)

# Each config = extra kwargs layered on top of BASE_KWARGS.
# "baseline" uses run_video defaults (lstsq align, single-frame ref, fg buffer 21).
CONFIGS = {
    "baseline": {},
    # Solution 1: temporal depth smoothing
    "sol1_median_w5": dict(depth_smoothing=True, depth_smoothing_window=5,
                           depth_smoothing_method="median"),
    "sol1_gaussian_w5": dict(depth_smoothing=True, depth_smoothing_window=5,
                             depth_smoothing_method="gaussian"),
    # Solution 3: robust affine alignment methods (already supported in code)
    "sol3_ransac": dict(alignement_method="ransac"),
    "sol3_median": dict(alignement_method="median"),
    # Solution 4: multi-frame depth reference
    "sol4_ref7": dict(reference_frames_for_median=7),
    # Solution 6: FG stabilizer tuning (baseline is buffer=21, jump=0.5)
    "sol6_b5_j0.2": dict(fg_buffer_size=5, fg_jump_threshold=0.2),
    "sol6_b11_j0.5": dict(fg_buffer_size=11, fg_jump_threshold=0.5),
    "sol6_b15_j1.0": dict(fg_buffer_size=15, fg_jump_threshold=1.0),
}


def find_videos(video_dir: str):
    d = Path(video_dir)
    exts = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    vids = []
    for e in exts:
        vids.extend(d.glob(f"*{e}"))
    return sorted(set(vids))


def run_config(converter, config_name, extra_kwargs, videos, gen, output_base):
    out_dir = Path(output_base) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = {
            "config": config_name,
            "generator": gen,
            "extra_kwargs": {k: str(v) for k, v in extra_kwargs.items()},
            "base_kwargs": {k: str(v) for k, v in BASE_KWARGS.items()},
            "start_time": datetime.now().isoformat(),
            "videos": {},
        }

    for vid in videos:
        name = vid.stem
        if name in results["videos"] and results["videos"][name].get("status") == "completed":
            print(f"  [skip] {config_name}/{name} already done", flush=True)
            continue

        vid_out = out_dir / name
        vid_out.mkdir(parents=True, exist_ok=True)
        print(f"\n  >>> {config_name} :: {name}", flush=True)
        t0 = time.time()
        try:
            res = run_video(
                converter=converter,
                video_path=str(vid),
                output_path=str(vid_out),
                active_generator=gen,
                **BASE_KWARGS,
                **extra_kwargs,
            )
            elapsed = time.time() - t0
            stab = calculate_temporal_stability(res.depth_metrics) if res.depth_metrics else None
            n_frames = len(res.splat_count) if res.splat_count else 0
            mean_splats = float(np.mean(res.splat_count)) if res.splat_count else None
            results["videos"][name] = {
                "status": "completed",
                "time_seconds": elapsed,
                "n_frames": n_frames,
                "mean_splats": mean_splats,
                "stability": stab,
            }
            cv = stab.get("bg_depth_cv") if stab else None
            spk = stab.get("bg_spikes_per_frame") if stab else None
            print(f"      done {elapsed:.1f}s | frames={n_frames} | "
                  f"bg_cv={cv} | spikes/frame={spk}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["videos"][name] = {"status": "failed", "error": str(e)}
            print(f"      FAILED: {e}", flush=True)

        # Incremental save (resumable / monitorable)
        results_path.write_text(json.dumps(results, indent=2))
        # Free disk: the PLY sequence is not needed for metrics
        _cleanup_plys(vid_out)

    results["end_time"] = datetime.now().isoformat()
    results_path.write_text(json.dumps(results, indent=2))
    return results


def _cleanup_plys(vid_out: Path):
    gdir = vid_out / "gaussians"
    if gdir.exists():
        for p in gdir.glob("*.ply"):
            try:
                p.unlink()
            except OSError:
                pass


def aggregate(results):
    """Mean stability metrics across successfully processed videos."""
    keys = ["bg_depth_cv", "bg_spikes_per_frame", "bg_delta_mean",
            "fg_depth_cv", "fg_delta_mean"]
    acc = {k: [] for k in keys}
    times = []
    for name, v in results["videos"].items():
        if v.get("status") != "completed":
            continue
        times.append(v.get("time_seconds", np.nan))
        stab = v.get("stability") or {}
        for k in keys:
            if stab.get(k) is not None:
                acc[k].append(stab[k])
    agg = {k: (float(np.mean(vals)) if vals else None) for k, vals in acc.items()}
    agg["mean_time_seconds"] = float(np.nanmean(times)) if times else None
    agg["n_videos"] = sum(1 for v in results["videos"].values()
                          if v.get("status") == "completed")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="winning generator (pager|unisharp)")
    ap.add_argument("--configs", default="all",
                    help="comma-separated config names, or 'all'")
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output", default="./benchmark_solutions")
    ap.add_argument("--summary-name", default="SUMMARY.json",
                    help="per-process summary filename (avoids races when parallelizing)")
    args = ap.parse_args()

    videos = find_videos(args.video_dir)
    print(f"Videos ({len(videos)}): {[v.name for v in videos]}")

    if args.configs == "all":
        config_names = list(CONFIGS.keys())
    else:
        config_names = [c.strip() for c in args.configs.split(",")]

    converter = SPAG4D(device="cuda")
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cname in config_names:
        if cname not in CONFIGS:
            print(f"!! unknown config {cname}, skipping")
            continue
        print(f"\n{'=' * 70}\nCONFIG: {cname}  (gen={args.gen})\n{'=' * 70}")
        res = run_config(converter, cname, CONFIGS[cname], videos, args.gen, output_base)
        summary[cname] = aggregate(res)

    # Write / update global summary
    summary_path = output_base / args.summary_name
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing.update(summary)
    summary_path.write_text(json.dumps(existing, indent=2))

    # Pretty print
    print(f"\n{'=' * 90}\nSOLUTION BENCHMARK SUMMARY (gen={args.gen})\n{'=' * 90}")
    hdr = f"{'config':<20}{'bg_cv':>10}{'spikes/f':>10}{'fg_cv':>10}{'fg_dlt':>10}{'time(s)':>10}"
    print(hdr)
    print("-" * len(hdr))
    base = existing.get("baseline", {})

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "N/A"

    for cname in existing:
        a = existing[cname]
        print(f"{cname:<20}{fmt(a.get('bg_depth_cv')):>10}"
              f"{fmt(a.get('bg_spikes_per_frame')):>10}"
              f"{fmt(a.get('fg_depth_cv')):>10}"
              f"{fmt(a.get('fg_delta_mean')):>10}"
              f"{fmt(a.get('mean_time_seconds')):>10}")
    print(f"\nBaseline bg_cv = {fmt(base.get('bg_depth_cv'))} "
          f"(lower = steadier background)")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
