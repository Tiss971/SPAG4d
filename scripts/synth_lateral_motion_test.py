"""§6.4 follow-up — synthetic lateral-motion ground truth.

Companion to synth_radial_motion_test.py. Lateral motion (across the image plane, at
roughly constant depth) is the case radial motion was contrasted against: 2D flow CAN
observe it, so a correct implementation should track it primarily via the flow-warp
term, with the confidence-decay/floor mechanism playing little to no role (unlike
radial motion, where decay is the only thing that lets depth update at all).

Expected shape of the result: "frozen" (no decay, confidence stays 1.0 once trusted)
should do at least as well as "decay" here, possibly better — pure flow-warp
propagation is exact for a rigid object at constant depth, whereas the production
floor=0.5 still blends in 50% of noisy per-frame monocular depth once age saturates,
which is unnecessary reintroduced noise on a target flow tracks perfectly. That
contrast (frozen wins here, frozen catastrophically fails in the radial test) is the
point: it isolates that the floor's cost is genuinely paid only to buy back the radial
blind spot, not a general tracking improvement.

Setup: a circular "object" patch translates laterally (x direction) at constant pixel
velocity across a flat background plate, depth constant (no radial component). Flow
fields are constructed directly (uniform (vx, vy) inside the object's CURRENT-frame
footprint, zero on background) rather than run through WAFT -- this isolates the
propagation/confidence logic from flow-estimation error, exactly as the radial test
isolates it from flow *magnitude* (there, zero; here, geometrically exact).
"""
import numpy as np

from spag4d.flow_depth_propagation import PropagationState, propagate_depth_via_flow

rng = np.random.default_rng(0)

H, W = 256, 512
N_FRAMES = 60
OBJ_CY, OBJ_R = 128, 30
OBJ_CX0 = 80.0
VX = 4.0  # px/frame, pure lateral -- no vertical, no depth change
D0 = 5.0  # constant ground-truth depth (m)
MONO_NOISE_STD = 0.15

BG_DEPTH = 10.0
depth_ref = np.full((H, W), BG_DEPTH, dtype=np.float32)

yy_full, xx_full = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")


def obj_mask_at(cx: float) -> np.ndarray:
    return (yy_full - OBJ_CY) ** 2 + (xx_full - cx) ** 2 < OBJ_R**2


def cx_at(i: int) -> float:
    return OBJ_CX0 + VX * i


gt_depth = np.full(N_FRAMES, D0, dtype=np.float32)  # constant depth throughout
t = np.arange(N_FRAMES)


def make_frame_affine(i: int) -> np.ndarray:
    frame = np.full((H, W), BG_DEPTH, dtype=np.float32)
    m = obj_mask_at(cx_at(i))
    noise = rng.normal(0.0, MONO_NOISE_STD, size=m.sum()).astype(np.float32)
    frame[m] = D0 + noise
    return frame


def make_flow(i: int):
    """Flow for the step that lands ON frame i (i.e. describes motion from i-1 to i).
    Defined on frame i's grid, matching propagate_depth_via_flow's convention
    (both flow_fwd and flow_bwd sampled via p - flow(p), see flow_depth_propagation.py).
    Uniform (VX, 0) inside the object's CURRENT (frame i) footprint, zero on background
    -- an exact, noise-free flow field for a rigid laterally-translating disk."""
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

    ax1.plot(t, gt_depth, "k--", lw=2, label="ground truth (constant depth)", zorder=5)
    colors = {"raw (no propagation)": "tab:gray", "decay (production, 0.85/0.5)": "tab:green",
              "legacy (pre-2026-08-17)": "tab:orange", "frozen (decay=1.0, floor=1.0)": "tab:red"}
    for name, series in arms.items():
        ax1.plot(t, series, color=colors[name], lw=1.5, alpha=0.85, label=name)
    ax1.set_ylabel("object depth (m)")
    ax1.set_title("Lateral motion (constant depth, translating across frame): tracked depth vs ground truth")
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
    out_path = "/tmp/claude-1005/-raid-mb273924-SPAG4d/421b83c9-a42a-4b3a-bdc7-2b1075571b80/scratchpad/synth_lateral_motion.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nplot saved: {out_path}")
