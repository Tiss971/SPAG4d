"""§6.4 (benchmarks/bglock_open_questions.md) — synthetic radial-motion ground truth.

2D optical flow cannot observe motion purely along the camera ray (toward/away from a
fixed camera): the object's image-plane position doesn't change, only its depth. So a
naive "warp previous depth forward with flow" propagator would freeze such an object at
its initial depth forever. The confidence-decay mechanism (PropagationState / §4.1-4.2
fix, SPAG_CONF_DECAY / SPAG_CONF_FLOOR) is the *only* thing standing between that failure
mode and correct tracking — this script is the controlled test that isolates it.

No rendering/compositing pipeline needed: propagate_depth_via_flow only consumes depth
maps + flow fields, so we synthesize those directly.

Setup, one circular "object" patch on a flat background plate:
  - flow_fwd = flow_bwd = 0 everywhere (radial motion is invisible to 2D flow by
    construction — this is the entire point of the test).
  - depth_ref = the static background plate depth (object region irrelevant, never
    revealed as disocclusion here since the object never leaves frame_confidence>0).
  - depth_curr_affine = ground truth trajectory + iid noise, standing in for a fresh
    per-frame monocular estimate. Ground truth: object depth decreases linearly
    (approaches camera) then increases (recedes) over N frames.
  - depth_prev_final: fed back from the previous step's depth_final (t=0 seeded from
    the first frame's affine estimate, as core.py does for the first frame).

Compared arms:
  - "decay"  : current production PropagationState (decay=0.85, conf_floor=0.5).
  - "legacy" : SPAG_CONF_LEGACY behaviour (non-compounding, confidence in {0, 0.85}).
  - "frozen" : decay=1.0, conf_floor=1.0 -> running_confidence stays 1.0 forever once
               trusted, i.e. depth_propagated (== initial depth, since flow is zero)
               is used unconditionally. This is the "flow-prop with no decay" failure
               mode the mechanism exists to avoid; expected to fail badly.
  - "raw"    : no propagation at all, i.e. output == depth_curr_affine every frame
               (fidelity ceiling: as good as the noisy monocular estimate alone).

Metric: RMSE of depth_final's object-patch median against ground truth, per arm.
"""
import numpy as np

from spag4d.flow_depth_propagation import PropagationState, propagate_depth_via_flow

rng = np.random.default_rng(0)

H, W = 256, 512
N_FRAMES = 60
OBJ_CY, OBJ_CX, OBJ_R = 128, 256, 30

yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
obj_mask = (yy - OBJ_CY) ** 2 + (xx - OBJ_CX) ** 2 < OBJ_R**2

BG_DEPTH = 10.0
depth_ref = np.full((H, W), BG_DEPTH, dtype=np.float32)

# Ground truth radial trajectory: approach from 8m to 2m over first half, recede back
# to 8m over the second half. Straight toward/away from camera -> zero 2D flow.
t = np.arange(N_FRAMES)
half = N_FRAMES // 2
gt_depth = np.where(
    t < half,
    8.0 - 6.0 * (t / half),
    2.0 + 6.0 * ((t - half) / (N_FRAMES - half)),
).astype(np.float32)

MONO_NOISE_STD = 0.15  # simulated per-frame monocular depth-estimate noise

flow_zero = np.zeros((H, W, 2), dtype=np.float32)


def make_frame_affine(depth_val: float) -> np.ndarray:
    frame = np.full((H, W), BG_DEPTH, dtype=np.float32)
    noise = rng.normal(0.0, MONO_NOISE_STD, size=obj_mask.sum()).astype(np.float32)
    frame[obj_mask] = depth_val + noise
    return frame


def run_arm(decay: float, conf_floor: float, legacy: bool):
    state = PropagationState(decay=decay, conf_floor=conf_floor, legacy=legacy)
    depth_prev = make_frame_affine(float(gt_depth[0]))  # t=0 seed, as core.py does
    obj_medians = [float(np.median(depth_prev[obj_mask]))]
    for i in range(1, N_FRAMES):
        affine = make_frame_affine(float(gt_depth[i]))
        depth_final, _, _ = propagate_depth_via_flow(
            depth_prev_final=depth_prev,
            depth_curr_affine=affine,
            depth_ref=depth_ref,
            flow_fwd=flow_zero,
            flow_bwd=flow_zero,
            state=state,
        )
        obj_medians.append(float(np.median(depth_final[obj_mask])))
        depth_prev = depth_final
    return np.array(obj_medians)


def raw_arm():
    medians = []
    for i in range(N_FRAMES):
        affine = make_frame_affine(float(gt_depth[i]))
        medians.append(float(np.median(affine[obj_mask])))
    return np.array(medians)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    arms = {
        "raw (no propagation)": raw_arm(),
        "decay (production, 0.85/0.5)": run_arm(decay=0.85, conf_floor=0.5, legacy=False),
        "legacy (pre-2026-08-17)": run_arm(decay=0.85, conf_floor=0.5, legacy=True),
        "frozen (decay=1.0, floor=1.0)": run_arm(decay=1.0, conf_floor=1.0, legacy=False),
    }

    print(f"{'arm':<32} {'RMSE (m)':>10} {'max err (m)':>12} {'final err (m)':>14}")
    for name, series in arms.items():
        err = series - gt_depth
        print(f"{name:<32} {rmse(series, gt_depth):>10.3f} {np.max(np.abs(err)):>12.3f} {abs(err[-1]):>14.3f}")

    print()
    print("gt_depth head:", np.round(gt_depth[:8], 2))
    for name, series in arms.items():
        print(f"{name} head:", np.round(series[:8], 2))

    # --- visual: object-median depth vs frame, all arms vs ground truth ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(t, gt_depth, "k--", lw=2, label="ground truth", zorder=5)
    colors = {"raw (no propagation)": "tab:gray", "decay (production, 0.85/0.5)": "tab:green",
              "legacy (pre-2026-08-17)": "tab:orange", "frozen (decay=1.0, floor=1.0)": "tab:red"}
    for name, series in arms.items():
        ax1.plot(t, series, color=colors[name], lw=1.5, alpha=0.85, label=name)
    ax1.set_ylabel("object depth (m)")
    ax1.set_title("Radial motion (toward then away from camera): tracked depth vs ground truth")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    for name, series in arms.items():
        ax2.plot(t, series - gt_depth, color=colors[name], lw=1.5, alpha=0.85, label=name)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_ylabel("error vs ground truth (m)")
    ax2.set_xlabel("frame")
    ax2.set_title("Tracking error")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = "/tmp/claude-1005/-raid-mb273924-SPAG4d/421b83c9-a42a-4b3a-bdc7-2b1075571b80/scratchpad/synth_radial_motion.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nplot saved: {out_path}")
