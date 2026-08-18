"""§6.7 -- fp16 quantization floor + bg_depth_cv units.

Two questions from bglock_open_questions.md §6.7:
  1. "Round-trip a depth map through fp16 and measure induced pseudo-flicker" --
     is fp16 rounding (PaGeR's forward runs at torch.float16) large enough to explain
     observed bg_depth_cv, or is it negligible?
  2. "Establish the units of the 0.037 figure and whether the §4.3 outliers were
     excluded when it was computed."

bg_depth_cv is defined in benchmark_solutions.py as `float(bg_std / bg_mean)` -- the
coefficient of variation of the run's per-frame *aligned background median depth*
series. It is dimensionless (a fraction of mean depth), NOT a distance in meters.
That already answers half of question 2: a raw fp16 step in meters (e.g. "0.016 m at
20 m") is not directly comparable to a CV without dividing by the depth at which it's
measured -- which happens to be constant (fp16 has fixed *relative* precision), so the
comparison collapses to a single number regardless of depth magnitude. This script
computes that number and compares it against every recorded bg_depth_cv in
benchmarks/baseline_2026-07-27/baseline_2026-07-27.json (needs no GPU / model run).
"""
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BASELINE_JSON = REPO / "benchmarks/baseline_2026-07-27/baseline_2026-07-27.json"

# --- fp16 relative quantization step (ULP / value), the same at every depth since
# floating point has constant *relative* precision, not constant absolute precision.
depths = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
rel_steps = []
for d in depths:
    t = torch.tensor(float(d), dtype=torch.float16)
    ulp = float(torch.nextafter(t, torch.tensor(float("inf"), dtype=torch.float16)) - t)
    rel_steps.append(ulp / d)
rel_steps = np.array(rel_steps)

print("fp16 relative quantization step (single round-trip, deterministic for a fixed value):")
for d, r in zip(depths, rel_steps):
    print(f"  {d:>6.1f} m -> ulp/d = {r:.6f}")
fp16_floor_cv = float(rel_steps.mean())
print(f"\nmean relative step across 0.5-100m range: {fp16_floor_cv:.6f}  (~constant, as expected for fp fp16)")
print(
    "This is a HARD FLOOR from a single cast, not a stochastic noise source -- casting the same\n"
    "value twice gives the same rounded result. Real cross-frame flicker would come from *different*\n"
    "rounding paths through many fp16 ops per forward pass (varying attention weights etc. as other\n"
    "image content changes frame to frame), which can exceed one ULP but stays the same order of\n"
    "magnitude -- so this is a reasonable floor estimate, not an exact one."
)

# --- compare against every recorded bg_depth_cv
if BASELINE_JSON.exists():
    data = json.load(open(BASELINE_JSON))
    print(f"\n{'clip':<32} {'bg_depth_cv':>12} {'bg_depth_mean':>14} {'/ fp16_floor':>13}")
    for name, v in data.get("videos", {}).items():
        s = v.get("stability", {})
        cv = s.get("bg_depth_cv")
        mean = s.get("bg_depth_mean")
        if cv is None:
            continue
        ratio = cv / fp16_floor_cv
        flag = "  <-- at/below floor, treat as noise" if ratio < 2.0 else ""
        print(f"{name:<32} {cv:>12.5f} {mean:>14.3f} {ratio:>12.2f}x{flag}")
    print(
        f"\nbaseline_2026-07-27 predates the §4.3 outlier-exclusion fix (2026-08-17/18):\n"
        f"the .json's bg_depth_cv values were computed WITHOUT excluding the magnitude-capped\n"
        f"backend outliers from the median. Reproducing with the current code + SPAG_CONF_LEGACY=0\n"
        f"and the current _valid_metric_mask gate would answer whether any of these numbers were\n"
        f"outlier-inflated; not done here (needs a full pipeline re-run, out of scope for this script)."
    )
else:
    print(f"\n{BASELINE_JSON} not found -- skipping bg_depth_cv comparison table")
