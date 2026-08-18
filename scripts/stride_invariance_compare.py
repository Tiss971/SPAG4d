"""§6.1 -- stride-invariance sweep comparison.

Compares depth at native frame indices shared by all of skip_step in {1,2,3,5}
(multiples of lcm(1,2,3,5)=30) across the four runs already produced by the
background sweep in scripts/../benchmarks/... (see run_sweep.sh in the
scratchpad). Answers: how much does accumulated propagation error grow with
stride, and is the effect concentrated in dynamic regions (flow-propagation
error) or spread evenly (would point to something else, e.g. the affine fit).

Usage: python scripts/stride_invariance_compare.py <sweep_dir>
  where <sweep_dir> contains skip1/, skip2/, skip3/, skip5/ each with
  out.ply/depth_maps/depth_<local_idx>.npy (from --depth-raw).
"""
import sys
from pathlib import Path

import numpy as np

sweep_dir = Path(sys.argv[1])
strides = [1, 2, 3, 5]
SHARED_STRIDE = 30  # lcm(1,2,3,5)

dirs = {s: sweep_dir / f"skip{s}" / "out.ply" / "depth_maps" for s in strides}
for s, d in dirs.items():
    n = len(list(d.glob("depth_*.npy")))
    print(f"skip_step={s}: {n} depth maps in {d}")

native_shared = list(range(0, 156, SHARED_STRIDE))  # 0,30,60,90,120,150
print(f"\nshared native frame indices: {native_shared}")

depths = {}  # (stride, native_idx) -> array
for s in strides:
    for nat in native_shared:
        local_idx = nat // s
        f = dirs[s] / f"depth_{local_idx}.npy"
        if f.exists():
            depths[(s, nat)] = np.load(f).astype(np.float64)
        else:
            print(f"  MISSING: skip{s} native={nat} local={local_idx} ({f})")

print(f"\n{'native_frame':>12} {'pair':>10} {'median|d1-d2|':>16} {'p95|d1-d2|':>12} {'rel_median':>12}")
ref_stride = 1
for nat in native_shared:
    if (ref_stride, nat) not in depths:
        continue
    d_ref = depths[(ref_stride, nat)]
    for s in strides[1:]:
        if (s, nat) not in depths:
            continue
        d = depths[(s, nat)]
        if d.shape != d_ref.shape:
            print(f"  shape mismatch stride{s} vs stride{ref_stride} at frame {nat}: {d.shape} vs {d_ref.shape}")
            continue
        diff = np.abs(d - d_ref)
        valid = np.isfinite(diff)
        med = float(np.median(diff[valid]))
        p95 = float(np.percentile(diff[valid], 95))
        rel = med / float(np.median(d_ref[valid])) if valid.any() else float("nan")
        print(f"{nat:>12} {f'{ref_stride}v{s}':>10} {med:>16.4f} {p95:>12.4f} {rel:>12.4f}")

print("\nGrowth trend (median abs diff vs stride, averaged over shared frames):")
for s in strides[1:]:
    vals = []
    for nat in native_shared:
        if (ref_stride, nat) in depths and (s, nat) in depths:
            d_ref = depths[(ref_stride, nat)]
            d = depths[(s, nat)]
            if d.shape == d_ref.shape:
                diff = np.abs(d - d_ref)
                valid = np.isfinite(diff)
                vals.append(float(np.median(diff[valid])))
    if vals:
        print(f"  stride {s}: mean-of-per-frame-median diff = {np.mean(vals):.4f} m  (n={len(vals)} shared frames)")
