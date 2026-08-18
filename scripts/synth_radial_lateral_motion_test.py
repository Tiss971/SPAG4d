"""§6.4 follow-up — combined radial + lateral motion.

Neither synth_radial_motion_test.py nor synth_lateral_motion_test.py alone tests
whether the two interact: does correctly warping a moving object's *position* via
flow interfere with correctly updating its *depth* via confidence decay when both
happen at once? A real moving subject (person walking toward the camera at an angle)
does both simultaneously.

Setup: same circular object as the other two tests, translating laterally at a
constant pixel velocity (as in synth_lateral_motion_test.py) AND simultaneously
following the radial approach/recede depth trajectory (as in
synth_radial_motion_test.py). Flow captures only the lateral component (uniform
(VX, 0) inside the object's current footprint, zero elsewhere) -- 2D flow still
cannot observe the radial component even though the object is also moving laterally.

Expected result: this should reproduce the radial test's finding (decay >> legacy >>
frozen on depth accuracy) close to unchanged, since flow correctly relocates the
*position* being sampled regardless of the depth-tracking mechanism -- if it doesn't,
that's a real interaction bug worth knowing about, not an expected outcome.
"""
import numpy as np

from spag4d.flow_depth_propagation import PropagationState, propagate_depth_via_flow

rng = np.random.default_rng(0)

H, W = 256, 512
N_FRAMES = 60
OBJ_CY, OBJ_R = 128, 30
OBJ_CX0 = 80.0
VX = 4.0  # px/frame lateral component
MONO_NOISE_STD = 0.15

BG_DEPTH = 10.0
depth_ref = np.full((H, W), BG_DEPTH, dtype=np.float32)

yy_full, xx_full = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

# Same radial trajectory as synth_radial_motion_test.py: approach 8m->2m, recede back.
t = np.arange(N_FRAMES)
half = N_FRAMES // 2
gt_depth = np.where(
    t < half,
    8.0 - 6.0 * (t / half),
    2.0 + 6.0 * ((t - half) / (N_FRAMES - half)),
).astype(np.float32)


def obj_mask_at(cx: float) -> np.ndarray:
    return (yy_full - OBJ_CY) ** 2 + (xx_full - cx) ** 2 < OBJ_R**2


def cx_at(i: int) -> float:
    return OBJ_CX0 + VX * i


def make_frame_affine(i: int) -> np.ndarray:
    frame = np.full((H, W), BG_DEPTH, dtype=np.float32)
    m = obj_mask_at(cx_at(i))
    noise = rng.normal(0.0, MONO_NOISE_STD, size=m.sum()).astype(np.float32)
    frame[m] = float(gt_depth[i]) + noise
    return frame


def make_flow(i: int):
    """Lateral component only -- flow cannot observe the radial depth change."""
    m = obj_mask_at(cx_at(i))
    flow_fwd = np.zeros((H, W, 2), dtype=np.float32)
    flow_bwd = np.zeros((H, W, 2), dtype=np.float32)
    flow_fwd[m, 0] = VX
    flow_bwd[m, 0] = -VX
    return flow_fwd, flow_bwd


def run_arm(decay: float, conf_floor: float, legacy: bool):
    state = PropagationState(decay=decay, conf_floor=conf_floor, legacy=legacy)
    depth_prev = make_frame_affine(0)
    obj_medians = [float(np.median(depth_prev[obj_mask_at(cx_at(0))]))]
    for i in range(1, N_FRAMES):
        affine = make_frame_affine(i)
        flow_fwd, flow_bwd = make_flow(i)
        depth_final, _, _ = propagate_depth_via_flow(
            depth_prev_final=depth_prev,
            depth_curr_affine=affine,
            depth_ref=depth_ref,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            state=state,
        )
        obj_medians.append(float(np.median(depth_final[obj_mask_at(cx_at(i))])))
        depth_prev = depth_final
    return np.array(obj_medians)


def raw_arm():
    medians = []
    for i in range(N_FRAMES):
        affine = make_frame_affine(i)
        medians.append(float(np.median(affine[obj_mask_at(cx_at(i))])))
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
    ax1.set_title("Combined radial + lateral motion: tracked depth vs ground truth")
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
    out_path = "/tmp/claude-1005/-raid-mb273924-SPAG4d/421b83c9-a42a-4b3a-bdc7-2b1075571b80/scratchpad/synth_radial_lateral_motion.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nplot saved: {out_path}")
