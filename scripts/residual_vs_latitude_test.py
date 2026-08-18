"""§10.3 -- residual-vs-latitude test.

bglock_open_questions.md §10.3: "run this regardless" -- both backends (DA360, PaGeR)
on the SAME static plate, scale-only fit, residual plotted against latitude. A
convention mismatch (e.g. one backend silently planar where the fit assumes radial,
or vice versa) shows up as a smooth systematic latitude-dependent trend; genuine
per-pixel model disagreement (different networks, different training data) shows up
as noise with no latitude structure. Answers two questions in one plot: convention
correctness AND whether a single global affine is too coarse for ERP (in which case a
latitude-banded or spherical-harmonic fit would be the fix).

Reuses an existing D_ref plate (`_temporal_median_background_masked.jpg`, a per-pixel
temporal-median RGB image, dynamic content already excluded) from a prior benchmark run
instead of building a new one -- it's exactly the "static plate" the test calls for, no
video pipeline needed. Both backends run once, single forward pass each.

Usage: PYTHONPATH=. python scripts/residual_vs_latitude_test.py [path/to/plate.jpg]
"""
import sys

import numpy as np
import torch
from PIL import Image

from spag4d.core import SPAG4D

DEFAULT_PLATE = (
    "/raid/mb273924/SPAG4d/benchmarks/confdecay_2026-08-17/MattSwift/decay/"
    "_temporal_median_background_masked.jpg"
)

plate_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PLATE
img = np.array(Image.open(plate_path).convert("RGB"))
H, W, _ = img.shape
print(f"plate: {plate_path}  ({W}x{H})")

image_t = torch.from_numpy(img).to("cuda")

converter = SPAG4D(generator="da360")

da360 = converter._get_depth_model("da360")
with torch.inference_mode():
    depth_da360, _ = da360.predict(image_t)
depth_da360 = depth_da360.cpu().numpy().astype(np.float64)

pager = converter._get_depth_model("pager")
with torch.inference_mode():
    depth_pager, sky_pager = pager.predict(image_t)
depth_pager = depth_pager.cpu().numpy().astype(np.float64)
sky_pager = sky_pager.cpu().numpy() if sky_pager is not None else None

print(f"DA360 depth range: [{depth_da360.min():.3f}, {depth_da360.max():.3f}], median={np.median(depth_da360):.3f}")
print(f"PaGeR depth range: [{depth_pager.min():.3f}, {depth_pager.max():.3f}], median={np.median(depth_pager):.3f}")

# Scale-only fit: da360 ~= s * pager (both radial-ish, median-normalized already by
# da360.predict's default rescale-to-~5m; pager returns its own scale). Exclude sky /
# non-finite / non-positive on either side.
valid = np.isfinite(depth_da360) & np.isfinite(depth_pager) & (depth_da360 > 0) & (depth_pager > 0)
if sky_pager is not None:
    valid &= sky_pager == 0

x = depth_pager[valid]
y = depth_da360[valid]
s = float(np.sum(x * y) / np.sum(x * x))  # scale-only least squares, y ~= s*x
print(f"scale-only fit: da360 ~= {s:.4f} * pager  (n={valid.sum()} valid pixels)")

pred = s * depth_pager
resid = np.where(valid, np.abs(depth_da360 - pred), np.nan)

N_BINS = 16
edges = np.linspace(0, H, N_BINS + 1).astype(int)
lat_deg = 90.0 - 180.0 * (np.arange(N_BINS) + 0.5) / N_BINS  # row 0 = north pole (+90)
print(f"\n{'lat_center_deg':>16} {'row_band':>12} {'median_resid_m':>16} {'n_valid':>8}")
band_medians = []
for i in range(N_BINS):
    band = resid[edges[i]:edges[i + 1]]
    vals = band[np.isfinite(band)]
    med = float(np.median(vals)) if vals.size else float("nan")
    band_medians.append(med)
    print(f"{lat_deg[i]:>16.1f} {f'{edges[i]}:{edges[i+1]}':>12} {med:>16.4f} {vals.size:>8}")

finite_meds = np.array([m for m in band_medians if np.isfinite(m)])
overall_med = float(np.nanmedian(resid))
print(f"\noverall median residual: {overall_med:.4f} m")
if finite_meds.size >= 2:
    # crude trend check: correlation of band index (proxy for latitude) with band median
    idx = np.arange(len(band_medians))
    mask = np.isfinite(band_medians)
    corr = float(np.corrcoef(idx[mask], np.array(band_medians)[mask])[0, 1])
    print(f"band-index vs residual correlation: {corr:.3f}  (near 0 = noise-like; |corr|>0.6 = systematic trend)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(lat_deg, band_medians, "o-", color="tab:blue")
ax1.axhline(overall_med, color="gray", ls=":", label=f"overall median ({overall_med:.3f} m)")
ax1.set_xlabel("latitude (deg, +90=pole, 0=equator, -90=pole)")
ax1.set_ylabel("median |da360 - s*pager| (m)")
ax1.set_title("Residual vs latitude (scale-only cross-backend fit)")
ax1.legend()
ax1.grid(alpha=0.3)

im = ax2.imshow(resid, cmap="inferno", vmax=np.nanpercentile(resid, 95))
ax2.set_title("Residual map")
plt.colorbar(im, ax=ax2, fraction=0.025)

fig.tight_layout()
out_path = "/tmp/claude-1005/-raid-mb273924-SPAG4d/421b83c9-a42a-4b3a-bdc7-2b1075571b80/scratchpad/residual_vs_latitude.png"
fig.savefig(out_path, dpi=140)
print(f"\nplot saved: {out_path}")
