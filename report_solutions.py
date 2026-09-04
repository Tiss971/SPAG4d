#!/usr/bin/env python3
"""
Combine per-config results.json produced by benchmark_solutions.py (possibly run
as several parallel processes) into one comparison table vs. the baseline.

Usage:
    python report_solutions.py --dir benchmark_solutions
"""

import argparse
import json
from pathlib import Path

import numpy as np

STAB_KEYS = ["bg_depth_cv", "bg_spikes_per_frame", "bg_delta_mean",
             "fg_depth_cv", "fg_delta_mean"]

# Which metric best reflects each solution's intended effect under freeze_bg
# (background is frozen -> foreground stability drives output flicker; background
#  CV still reports how steady the aligned intermediate depth is).
RELEVANT = {
    "sol1": "both (bg_depth_cv + fg_depth_cv)",
    "sol3": "fg_depth_cv / fg_delta_mean (alignment quality on moving regions)",
    "sol4": "bg_depth_cv (reference steadiness)",
    "sol6": "fg_depth_cv / fg_delta_mean (foreground stabilizer)",
}


def aggregate(results, allowed_videos=None):
    """Aggregate stability metrics. If allowed_videos is given, only those
    videos are averaged (used to restrict to the common set across configs)."""
    acc = {k: [] for k in STAB_KEYS}
    times, frames = [], []
    per_video = {}
    for name, v in results.get("videos", {}).items():
        if v.get("status") != "completed":
            continue
        if allowed_videos is not None and name not in allowed_videos:
            continue
        times.append(v.get("time_seconds", np.nan))
        frames.append(v.get("n_frames", 0))
        stab = v.get("stability") or {}
        per_video[name] = stab
        for k in STAB_KEYS:
            if stab.get(k) is not None:
                acc[k].append(stab[k])
    agg = {k: (float(np.mean(vals)) if vals else None) for k, vals in acc.items()}
    agg["mean_time_seconds"] = float(np.nanmean(times)) if times else None
    agg["n_videos"] = len(per_video)
    return agg, per_video


def completed_videos(results):
    return {n for n, v in results.get("videos", {}).items()
            if v.get("status") == "completed"}


def fmt(v, p=4):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  N/A"


def pct_delta(new, base):
    if not isinstance(new, (int, float)) or not isinstance(base, (int, float)) or base == 0:
        return "   -"
    d = (new - base) / abs(base) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmark_solutions")
    args = ap.parse_args()
    root = Path(args.dir)

    raw = {}
    for rj in sorted(root.glob("*/results.json")):
        cname = rj.parent.name
        raw[cname] = json.loads(rj.read_text())

    # Restrict to the intersection of videos completed by every config, so all
    # configs are averaged over exactly the same set (fair comparison).
    common = None
    for results in raw.values():
        cv = completed_videos(results)
        common = cv if common is None else (common & cv)
    common = common or set()
    print(f"Common videos across all configs ({len(common)}): {sorted(common)}")

    configs = {}
    for cname, results in raw.items():
        agg, per_video = aggregate(results, allowed_videos=common)
        configs[cname] = {"agg": agg, "per_video": per_video,
                          "gen": results.get("generator")}

    if "baseline" not in configs:
        print("!! no baseline config found in", root)
    base = configs.get("baseline", {}).get("agg", {})

    order = ["baseline"] + [c for c in sorted(configs) if c != "baseline"]

    print(f"\n{'=' * 100}")
    print(f"SOLUTION BENCHMARK REPORT  —  {root}")
    gen = next(iter(configs.values()), {}).get("gen", "?") if configs else "?"
    print(f"generator = {gen}   (lower bg_cv / fg_cv / spikes = more temporally stable)")
    print(f"{'=' * 100}")
    hdr = (f"{'config':<20}{'bg_cv':>10}{'spikes/f':>10}{'fg_cv':>10}"
           f"{'fg_delta':>10}{'time(s)':>10}{'nvid':>6}")
    print(hdr)
    print("-" * len(hdr))
    for c in order:
        a = configs[c]["agg"]
        print(f"{c:<20}{fmt(a.get('bg_depth_cv')):>10}{fmt(a.get('bg_spikes_per_frame')):>10}"
              f"{fmt(a.get('fg_depth_cv')):>10}{fmt(a.get('fg_delta_mean')):>10}"
              f"{fmt(a.get('mean_time_seconds'),1):>10}{a.get('n_videos',0):>6}")

    print(f"\n{'Δ vs baseline (negative = better for stability metrics)':<40}")
    print("-" * len(hdr))
    for c in order:
        if c == "baseline":
            continue
        a = configs[c]["agg"]
        print(f"{c:<20}"
              f"{pct_delta(a.get('bg_depth_cv'), base.get('bg_depth_cv')):>10}"
              f"{pct_delta(a.get('bg_spikes_per_frame'), base.get('bg_spikes_per_frame')):>10}"
              f"{pct_delta(a.get('fg_depth_cv'), base.get('fg_depth_cv')):>10}"
              f"{pct_delta(a.get('fg_delta_mean'), base.get('fg_delta_mean')):>10}"
              f"{pct_delta(a.get('mean_time_seconds'), base.get('mean_time_seconds')):>10}")

    print("\nRelevant metric per solution family (under freeze_bg):")
    for k, v in RELEVANT.items():
        print(f"  {k}: {v}")

    # Save combined json
    out = root / "REPORT.json"
    out.write_text(json.dumps({c: configs[c]["agg"] for c in order}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
