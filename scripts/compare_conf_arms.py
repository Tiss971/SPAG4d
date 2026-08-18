"""Paired stability/fidelity comparison across the confidence-decay arms.

BENCHMARK_RULES rule 11: no stability number may be reported without a paired
fidelity number, because `output = D_ref` aces every stability metric in the repo.
So this reports, per arm:

  stability  bg_depth_cv / fg_depth_cv / bg_spikes_per_frame  (from run_info.json)
  fidelity   mean per-frame |depth_t - depth_{t-1}| inside the dynamic mask,
             normalised by the frame's median depth, expressed as a RATIO against
             the `noprop` arm (SPAG_CONF_DECAY=0 SPAG_CONF_FLOOR=0 -> confidence is
             identically 0 -> the fresh per-frame monocular estimate, i.e. "raw").

A ratio near 1.0 means the arm preserves the raw motion signal; a ratio well below
1.0 means motion is being suppressed, which is the failure mode a stability-only
table cannot see.

Usage:
    python scripts/compare_conf_arms.py benchmarks/confdecay_2026-08-17
"""
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ["legacy", "decay", "noprop"]


def fg_derivative(depth_dir: Path) -> dict:
    """Mean |d_t - d_{t-1}| over pixels dynamic in BOTH frames, scale-normalised."""
    idxs = sorted(int(p.stem.split("_")[1]) for p in depth_dir.glob("depth_*.npy"))
    prev_d = prev_m = None
    vals, npx = [], []
    for i in idxs:
        d = np.load(depth_dir / f"depth_{i}.npy").astype(np.float32)
        mp = depth_dir / f"mask_{i}.npy"
        m = np.load(mp) > 0 if mp.exists() else np.zeros(d.shape, bool)
        if prev_d is not None:
            both = m & prev_m
            n = int(both.sum())
            if n > 50:
                scale = float(np.median(d[d > 0])) or 1.0
                vals.append(float(np.abs(d[both] - prev_d[both]).mean()) / scale)
                npx.append(n)
        prev_d, prev_m = d, m
    if not vals:
        return {"fg_abs_derivative": None, "n_pairs": 0}
    v, w = np.array(vals), np.array(npx, float)
    return {
        # weighted by in-mask pixel count: an unweighted mean over frames lets a
        # 60-pixel sliver count as much as a full-body mask (psnr_by_view lesson).
        "fg_abs_derivative": float((v * w).sum() / w.sum()),
        "fg_abs_derivative_unweighted": float(v.mean()),
        "n_pairs": len(vals),
    }


def main():
    root = Path(sys.argv[1])
    out = {}
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        clip = {}
        for arm in ARMS:
            d = clip_dir / arm
            if not (d / "run_info.json").exists():
                continue
            info = json.loads((d / "run_info.json").read_text())
            stab = info.get("stability") or {}
            rec = {
                "env": {k: v for k, v in info["config"]["env"].items()
                        if k.startswith("SPAG_CONF") and v is not None},
                "time_seconds": round(info["time_seconds"], 1),
                "vram_max_mb": round(info["vram_max_mb"]),
                "n_frames": info["n_frames"],
                "n_objects": info["n_objects"],
                "stability": {k: stab.get(k) for k in
                              ("bg_depth_cv", "fg_depth_cv", "bg_spikes_per_frame",
                               "fg_delta_mean")},
            }
            rec.update(fg_derivative(d / "depth_maps"))
            trace_p = d / "confidence_trace.json"
            if trace_p.exists():
                tr = json.loads(trace_p.read_text())
                rec["confidence"] = {
                    "n_frames": len(tr),
                    "first": tr[0],
                    "last": tr[-1],
                    "max_n_distinct": max(t["n_distinct"] for t in tr),
                    "frac_zero_first10": float(np.mean([t["frac_zero"] for t in tr[:10]])),
                    "frac_zero_last10": float(np.mean([t["frac_zero"] for t in tr[-10:]])),
                    "mean_conf_last10": float(np.mean([t["mean"] for t in tr[-10:]])),
                }
            clip[arm] = rec
        base = (clip.get("noprop") or {}).get("fg_abs_derivative")
        for arm, rec in clip.items():
            if base and rec.get("fg_abs_derivative"):
                rec["fidelity_vs_noprop"] = round(rec["fg_abs_derivative"] / base, 4)
        out[clip_dir.name] = clip

    (root / "comparison.json").write_text(json.dumps(out, indent=2))
    for clip, arms in out.items():
        print(f"\n=== {clip}")
        print(f"{'arm':8} {'bg_cv':>8} {'fg_cv':>8} {'fg_deriv':>10} {'fid_ratio':>10} "
              f"{'n_dist':>7} {'conf_last':>10} {'time_s':>8}")
        for arm, r in arms.items():
            s, c = r["stability"], r.get("confidence", {})
            f = lambda v, p=4: "-" if v is None else f"{v:.{p}f}"
            print(f"{arm:8} {f(s['bg_depth_cv']):>8} {f(s['fg_depth_cv']):>8} "
                  f"{f(r.get('fg_abs_derivative'), 5):>10} {f(r.get('fidelity_vs_noprop'), 3):>10} "
                  f"{str(c.get('max_n_distinct', '-')):>7} {f(c.get('mean_conf_last10'), 3):>10} "
                  f"{r['time_seconds']:>8.0f}")
    print(f"\nwrote {root / 'comparison.json'}")


if __name__ == "__main__":
    main()
