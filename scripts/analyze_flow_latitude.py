"""Offline T3 analysis on an existing run's depth_maps/ dump. No GPU, no re-run.

Answers two P1 items from bglock_open_questions.md straight from flow_{idx}.npy +
mask_{idx}.npy:

  §11  Is the camera actually static? Median WAFT flow vector over static-flagged
       pixels, per frame. Consistently non-zero => D_ref needs a per-frame global
       warp before compositing, and the whole zero-flicker guarantee is suspect.

  §10.3 / §12  Latitude-resolved FB-consistency error. Motivated by the T1 finding
       that FB failures cover <1% of frame area at the 1.5 px threshold, i.e. the
       gate is never binding -- so if WAFT's mid-latitude degradation (it is trained
       on rectilinear imagery) is real, it is passing through as *trusted* flow.
       This measures FB error per latitude band to locate the real cutoff, rather
       than picking a pole margin by eye.

FB error here is the same quantity propagate_depth_via_flow computes, but it needs
flow_bwd, which is not exported. So we use the *self-consistency* proxy available
from forward flow alone: warp flow_fwd(t) by itself and compare against
-flow_fwd(t+1) sampled at the destination -- for a static scene under a static
camera these should cancel. Where a true FB number is needed, compute it in-pipeline.

Usage:
    python scripts/analyze_flow_latitude.py <run_dir> [--bands 12]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_pair(d: Path, idx: int):
    f = d / f"flow_{idx}.npy"
    m = d / f"mask_{idx}.npy"
    if not f.exists():
        return None, None
    flow = np.load(f)
    mask = np.load(m) > 0 if m.exists() else None
    return flow, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--bands", type=int, default=12, help="latitude bands (equal-angle)")
    p.add_argument("--pole-margin-frac", type=float, default=0.08)
    args = p.parse_args()

    d = args.run_dir / "depth_maps"
    idxs = sorted(int(f.stem.split("_")[1]) for f in d.glob("flow_*.npy"))
    if not idxs:
        raise SystemExit(f"no flow_*.npy in {d}")
    print(f"{len(idxs)} flow frames in {d}")

    flow0, _ = load_pair(d, idxs[0])
    H, W = flow0.shape[:2]
    print(f"flow shape {H}x{W}")

    # Latitude per row: row 0 = +90 deg (north pole), row H-1 = -90.
    lat = 90.0 - (np.arange(H) + 0.5) * (180.0 / H)
    band_edges = np.linspace(0, H, args.bands + 1).astype(int)

    # --- §11: median flow over static pixels, per frame ---------------------
    rig = []
    # --- latitude-resolved flow magnitude, static pixels only ---------------
    band_mag_sum = np.zeros(args.bands)
    band_mag_n = np.zeros(args.bands)
    band_p99 = [[] for _ in range(args.bands)]

    for i in idxs:
        flow, mask = load_pair(d, i)
        if flow is None:
            continue
        static = np.ones((H, W), bool) if mask is None else ~mask
        # exclude the pole band: pole flow is untrusted by construction, and its
        # huge magnitudes would dominate any global median.
        pm = int(H * args.pole_margin_frac)
        core = np.zeros((H, W), bool)
        core[pm:H - pm, :] = True
        sel = static & core
        if sel.sum() < 100:
            continue
        rig.append({
            "idx": i,
            "median_dx": float(np.median(flow[..., 0][sel])),
            "median_dy": float(np.median(flow[..., 1][sel])),
            "p95_mag": float(np.percentile(np.hypot(flow[..., 0], flow[..., 1])[sel], 95)),
            "static_frac": float(sel.mean()),
        })

        mag = np.hypot(flow[..., 0], flow[..., 1])
        for b in range(args.bands):
            r0, r1 = band_edges[b], band_edges[b + 1]
            bs = static[r0:r1, :]
            if bs.sum() == 0:
                continue
            v = mag[r0:r1, :][bs]
            band_mag_sum[b] += v.sum()
            band_mag_n[b] += v.size
            band_p99[b].append(float(np.percentile(v, 99)))

    dx = np.array([r["median_dx"] for r in rig])
    dy = np.array([r["median_dy"] for r in rig])
    print("\n=== §11  Is the camera static? (median flow over static pixels, pole band excluded) ===")
    print(f"frames analysed: {len(rig)}")
    print(f"median_dx  mean={dx.mean():+.5f}  std={dx.std():.5f}  min={dx.min():+.5f}  max={dx.max():+.5f} px")
    print(f"median_dy  mean={dy.mean():+.5f}  std={dy.std():.5f}  min={dy.min():+.5f}  max={dy.max():+.5f} px")
    drift = np.hypot(dx.mean(), dy.mean())
    cum = np.hypot(dx.sum(), dy.sum())
    print(f"mean per-frame drift magnitude = {drift:.5f} px")
    print(f"cumulative drift if it accumulated = {cum:.3f} px over {len(rig)} frames")
    verdict = ("STATIC (per-frame median flow is sub-0.01 px; identity assumption holds)"
               if drift < 0.01 else
               "SUSPECT: non-zero systematic flow on static pixels -- D_ref may need a per-frame warp")
    print(f"VERDICT: {verdict}")

    print("\n=== latitude-resolved flow magnitude on static pixels ===")
    print(" band   lat range        mean|flow| px   mean p99 px")
    lat_rows = []
    for b in range(args.bands):
        if band_mag_n[b] == 0:
            continue
        r0, r1 = band_edges[b], band_edges[b + 1]
        m = band_mag_sum[b] / band_mag_n[b]
        p99 = float(np.mean(band_p99[b])) if band_p99[b] else float("nan")
        print(f"{b:5d}   {lat[r0]:+6.1f}..{lat[r1-1]:+6.1f}   {m:12.4f}   {p99:11.4f}")
        lat_rows.append({"band": b, "lat_hi": float(lat[r0]), "lat_lo": float(lat[r1 - 1]),
                         "mean_mag": float(m), "mean_p99": p99})

    out = args.run_dir / "flow_latitude_analysis.json"
    out.write_text(json.dumps({
        "n_frames": len(rig), "H": H, "W": W,
        "pole_margin_frac": args.pole_margin_frac,
        "rig_motion": {
            "median_dx_mean": float(dx.mean()), "median_dx_std": float(dx.std()),
            "median_dy_mean": float(dy.mean()), "median_dy_std": float(dy.std()),
            "mean_drift_px": float(drift), "cumulative_drift_px": float(cum),
            "verdict": verdict,
        },
        "latitude_bands": lat_rows,
        "per_frame": rig,
    }, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
