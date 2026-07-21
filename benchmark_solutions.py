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
import torch

from spag4d.core import SPAG4D
from spag4d.video import run_video

DEFAULT_VIDEO_DIR = "/raid/mb273924/_DATASETS/uptale/data/videos"

# Fixed pipeline config shared by every run (the recommended baseline setup).
BASE_KWARGS = dict(
    skip_step=2,
    stride=4,
    temporal_consistency=False,
    freeze_bg=True,
    outlier_pruning=0.3,
    grazing_angle=85.0,
    sparse_pruning=0.1,
)

# Each config = extra kwargs layered on top of BASE_KWARGS.
# "baseline" and every sol* config pin depth_correction="affine" explicitly:
# run_video's default flipped to "bglock" after the temporal-solutions merge,
# and these configs predate/test-against the legacy per-frame affine path —
# without the pin they'd silently start running bglock instead of what their
# name says.
CONFIGS = {
    "baseline": dict(depth_correction="affine"),
    # Background-locked compositing + flow propagation (see .claude/depth_stability_benchmark.md)
    "bglock": dict(depth_correction="bglock"),
    # Solution 1: temporal depth smoothing
    "sol1_median_w5": dict(depth_correction="affine", depth_smoothing=True,
                           depth_smoothing_window=5, depth_smoothing_method="median"),
    # "sol1_gaussian_w5": dict(depth_correction="affine", depth_smoothing=True,
    #                          depth_smoothing_window=5, depth_smoothing_method="gaussian"),
    "bglock_sol1_median_w5": dict(depth_correction="bglock", depth_smoothing=True,
                           depth_smoothing_window=5, depth_smoothing_method="median"),
    # "sol1_median_sol4": dict(depth_correction="affine", depth_smoothing=True,
    #                        depth_smoothing_window=5, depth_smoothing_method="median",
    #                        reference_frames_for_median=7),
    # # Solution 4: multi-frame depth reference
    # "sol4_ref7": dict(depth_correction="affine", reference_frames_for_median=7),
    # "bglock_sol4_ref7": dict(depth_correction="bglock", reference_frames_for_median=7),
    # # Solution 6: FG stabilizer tuning (baseline is buffer=21, jump=0.5)
    # "sol6_b5_j0.2": dict(depth_correction="affine", fg_buffer_size=5, fg_jump_threshold=0.2),
    # "sol6_b11_j0.5": dict(depth_correction="affine", fg_buffer_size=11, fg_jump_threshold=0.5),
    # "sol6_b15_j1.0": dict(depth_correction="affine", fg_buffer_size=15, fg_jump_threshold=1.0),
}


def calculate_temporal_stability(depth_metrics):
    """
    Calculate temporal stability metrics from depth data.
    Focuses on: background depth stability, spikes, and foreground motion.
    Returns dict with stability scores (lower = more stable).
    """
    if not depth_metrics:
        return None

    metrics = {}

    # Background median depth coefficient of variation (PRIMARY METRIC)
    bg_median = np.array(depth_metrics.get("aligned_bg_median", []))
    bg_median_valid = bg_median[~np.isnan(bg_median)]
    if len(bg_median_valid) >= 2:
        bg_mean = np.mean(bg_median_valid)
        bg_std = np.std(bg_median_valid)
        if bg_mean > 0:
            metrics["bg_depth_cv"] = float(bg_std / bg_mean)  # Lower = more stable
            metrics["bg_depth_mean"] = float(bg_mean)
            metrics["bg_depth_std"] = float(bg_std)

    # Background median depth deltas (frame-to-frame spikes)
    bg_deltas = np.array(depth_metrics.get("bg_median_deltas", []))
    bg_deltas_valid = bg_deltas[~np.isnan(bg_deltas)]
    if len(bg_deltas_valid) >= 1:
        metrics["bg_delta_max"] = float(np.max(bg_deltas_valid))  # Largest spike
        metrics["bg_delta_mean"] = float(np.mean(bg_deltas_valid))  # Avg spike
        metrics["bg_delta_std"] = float(np.std(bg_deltas_valid))
        # Count "significant spikes" (e.g., > 0.1m)
        spike_threshold = 0.1
        metrics["bg_spike_count"] = int(np.sum(bg_deltas_valid > spike_threshold))
        metrics["bg_spikes_per_frame"] = float(metrics["bg_spike_count"] / len(bg_deltas_valid)) if len(bg_deltas_valid) > 0 else 0.0

    # Foreground median depth deltas (object motion smoothness)
    fg_deltas = np.array(depth_metrics.get("fg_median_deltas", []))
    fg_deltas_valid = fg_deltas[~np.isnan(fg_deltas)]
    if len(fg_deltas_valid) >= 1:
        metrics["fg_delta_max"] = float(np.max(fg_deltas_valid))
        metrics["fg_delta_mean"] = float(np.mean(fg_deltas_valid))
        metrics["fg_delta_std"] = float(np.std(fg_deltas_valid))

    # Foreground median depth coefficient of variation (smoothness)
    fg_median = np.array(depth_metrics.get("stabilized_fg_median", []))
    fg_median_valid = fg_median[~np.isnan(fg_median)]
    if len(fg_median_valid) >= 2:
        fg_mean = np.mean(fg_median_valid)
        fg_std = np.std(fg_median_valid)
        if fg_mean > 0:
            metrics["fg_depth_cv"] = float(fg_std / fg_mean)  # Object depth stability

    return metrics if metrics else None


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

            max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats() # Réinitialise pour la prochaine vidéo

            stab = calculate_temporal_stability(res.depth_metrics) if res.depth_metrics else None
            n_frames = len(res.splat_count) if res.splat_count else 0
            mean_splats = float(np.mean(res.splat_count)) if res.splat_count else None
            results["videos"][name] = {
                "status": "completed",
                "time_seconds": elapsed,
                "vram_max_mb": max_vram_mb,
                "n_frames": n_frames,
                "mean_splats": mean_splats,
                "stability": stab,
            }
            cv = stab.get("bg_depth_cv") if stab else None
            spk = stab.get("bg_spikes_per_frame") if stab else None
            print(f"      done {elapsed:.1f}s | vram={max_vram_mb:.0f}MB | frames={n_frames} | "
                  f"bg_cv={cv} | spikes/frame={spk}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["videos"][name] = {"status": "failed", "error": str(e)}
            print(f"      FAILED: {e}", flush=True)

        # Incremental save (resumable / monitorable)
        results_path.write_text(json.dumps(results, indent=2))
        # Free disk: the PLY sequence is not needed for metrics
        # _cleanup_plys(vid_out)

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
    vrams = []
    for name, v in results["videos"].items():
        if v.get("status") != "completed":
            continue
        times.append(v.get("time_seconds", np.nan))
        vrams.append(v.get("vram_max_mb", np.nan))
        stab = v.get("stability") or {}
        for k in keys:
            if stab.get(k) is not None:
                acc[k].append(stab[k])
    agg = {k: (float(np.mean(vals)) if vals else None) for k, vals in acc.items()}
    agg["mean_time_seconds"] = float(np.nanmean(times)) if times else None
    agg["mean_vram_max_mb"] = float(np.nanmean(vrams)) if vrams else None
    agg["n_videos"] = sum(1 for v in results["videos"].values()
                          if v.get("status") == "completed")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="winning generator (da360|pager|unisharp)")
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
              f"{fmt(a.get('mean_time_seconds')):>10}"
              f"{fmt(a.get('mean_vram_max_mb')):>10}")
    print(f"\nBaseline bg_cv = {fmt(base.get('bg_depth_cv'))} "
          f"(lower = steadier background)")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
