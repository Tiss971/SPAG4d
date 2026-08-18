"""§6.4 follow-up — sweep SPAG_CONF_FLOOR on the synthetic radial-motion test.

Reuses the setup from synth_radial_motion_test.py (same object, same ground-truth
trajectory, same noise seed) and varies conf_floor at fixed decay=0.85 to see how the
tracking-lag / stability tradeoff moves.

Steady-state prediction: once age saturates, the propagation blend is a first-order
IIR low-pass with weight `conf_floor` on the stale propagated value. Against a ramp of
slope v (m/frame) that settles to a lag of v * floor/(1-floor). Slope here is
6m/30frames = 0.2 m/frame, so:
    floor=0.3 -> lag ~0.09m   floor=0.7 -> lag ~0.47m
    floor=0.5 -> lag ~0.20m   floor=0.9 -> lag ~1.80m
conf_floor=0.5 (production) sits well inside the regime where lag is still small
relative to the noise floor; this sweep checks that empirically rather than trusting
the closed-form approximation (age-ramp-up transient + the direction reversal at
frame 30 aren't captured by the steady-state formula).
"""
import numpy as np

from synth_radial_motion_test import gt_depth, rmse, run_arm, t

FLOORS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DECAY = 0.85

results = {f: run_arm(decay=DECAY, conf_floor=f, legacy=False) for f in FLOORS}

print(f"{'conf_floor':>10} {'RMSE (m)':>10} {'max err (m)':>12} {'predicted steady lag (m)':>26}")
for f in FLOORS:
    series = results[f]
    err = series - gt_depth
    predicted_lag = 0.2 * f / (1.0 - f)  # slope=0.2 m/frame, see module docstring
    print(f"{f:>10.2f} {rmse(series, gt_depth):>10.3f} {np.max(np.abs(err)):>12.3f} {predicted_lag:>26.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11), sharex=False)

ax1.plot(t, gt_depth, "k--", lw=2, label="ground truth", zorder=5)
cmap = matplotlib.colormaps["viridis"].resampled(len(FLOORS))
for i, f in enumerate(FLOORS):
    ax1.plot(t, results[f], color=cmap(i), lw=1.3, alpha=0.9, label=f"floor={f}")
ax1.set_ylabel("object depth (m)")
ax1.set_xlabel("frame")
ax1.set_title("Radial motion tracking vs SPAG_CONF_FLOOR (decay=0.85 fixed)")
ax1.legend(loc="upper right", fontsize=7, ncol=2)
ax1.grid(alpha=0.3)

for i, f in enumerate(FLOORS):
    ax2.plot(t, results[f] - gt_depth, color=cmap(i), lw=1.3, alpha=0.9, label=f"floor={f}")
ax2.axhline(0, color="k", lw=1)
ax2.set_ylabel("error vs ground truth (m)")
ax2.set_xlabel("frame")
ax2.set_title("Tracking error vs SPAG_CONF_FLOOR")
ax2.grid(alpha=0.3)

rmses = [rmse(results[f], gt_depth) for f in FLOORS]
predicted = [0.2 * f / (1.0 - f) for f in FLOORS]
ax3.plot(FLOORS, rmses, "o-", color="tab:blue", label="measured RMSE")
ax3.plot(FLOORS, predicted, "x--", color="tab:red", label="predicted steady-state lag")
ax3.axvline(0.5, color="gray", ls=":", label="production (0.5)")
ax3.set_xlabel("conf_floor")
ax3.set_ylabel("error (m)")
ax3.set_title("RMSE vs conf_floor: measured vs first-order IIR prediction")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

fig.tight_layout()
out_path = "/tmp/claude-1005/-raid-mb273924-SPAG4d/421b83c9-a42a-4b3a-bdc7-2b1075571b80/scratchpad/synth_radial_motion_floor_sweep.png"
fig.savefig(out_path, dpi=140)
print(f"\nplot saved: {out_path}")
