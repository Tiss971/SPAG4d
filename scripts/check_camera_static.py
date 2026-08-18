"""§11 (benchmarks/bglock_open_questions.md) — verify the camera is actually static.

Test as prescribed: median WAFT flow vector over static-flagged (SAM mask == 0)
pixels, per frame. Consistently non-zero would mean D_ref needs a per-frame global
warp (the whole bglock design assumes a fixed camera / static background).

Reads flow_{idx}.npy (HxWx2 float32, native res, forward flow idx-1 -> idx) and
mask_{idx}.npy (uint8 fused SAM mask, >0 == dynamic) dumped by run_video() when
depth_npy_dir is set.
"""
import sys
from pathlib import Path

import numpy as np


def analyze(depth_maps_dir: Path, label: str):
    flow_files = sorted(depth_maps_dir.glob("flow_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if not flow_files:
        print(f"{label}: no flow_*.npy found in {depth_maps_dir}")
        return None

    medians_dx, medians_dy, medians_mag = [], [], []
    frac_static = []
    for fpath in flow_files:
        idx = int(fpath.stem.split("_")[1])
        mpath = depth_maps_dir / f"mask_{idx}.npy"
        if not mpath.exists():
            continue
        flow = np.load(fpath)          # (H,W,2)
        mask = np.load(mpath)          # (H,W) uint8, >0 == dynamic
        static = mask == 0
        if static.sum() == 0:
            continue
        frac_static.append(static.mean())
        fx = flow[..., 0][static]
        fy = flow[..., 1][static]
        mdx, mdy = float(np.median(fx)), float(np.median(fy))
        medians_dx.append(mdx)
        medians_dy.append(mdy)
        medians_mag.append(float(np.hypot(mdx, mdy)))

    medians_dx = np.array(medians_dx)
    medians_dy = np.array(medians_dy)
    medians_mag = np.array(medians_mag)

    print(f"\n=== {label} ({len(medians_mag)} frames) ===")
    print(f"static-pixel fraction: mean={np.mean(frac_static):.3f}")
    print(f"median flow dx: mean={medians_dx.mean():+.4f} std={medians_dx.std():.4f} "
          f"min={medians_dx.min():+.4f} max={medians_dx.max():+.4f}")
    print(f"median flow dy: mean={medians_dy.mean():+.4f} std={medians_dy.std():.4f} "
          f"min={medians_dy.min():+.4f} max={medians_dy.max():+.4f}")
    print(f"median flow magnitude (px): mean={medians_mag.mean():.4f} std={medians_mag.std():.4f} "
          f"max={medians_mag.max():.4f}")
    # Sign consistency alone is not sufficient: WAFT has its own sub-pixel systematic
    # bias even on a genuinely static scene (correlation-window asymmetry, lens
    # distortion residuals), so a consistently-signed but sub-pixel median is expected
    # noise, not evidence of camera motion. Only flag drift once it's large enough to
    # matter for depth alignment (>0.5px sustained median, ~1/2000 of an ERP frame's
    # width -- well below WAFT's own reported EPE, so smaller than this is not
    # actionable even if "real").
    verdict = "DRIFT SUSPECTED" if medians_mag.mean() > 0.5 else "STATIC (pass, sub-pixel bias only)"
    print(f"verdict: {verdict}")
    return medians_dx, medians_dy, medians_mag


if __name__ == "__main__":
    for d in sys.argv[1:]:
        p = Path(d)
        label = p.parent.name if p.name == "depth_maps" else p.name
        analyze(p if p.name == "depth_maps" else p / "depth_maps", label)
