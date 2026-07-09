# spag4d/flow_depth_propagation.py
"""
Flow-based depth propagation for a FIXED equirectangular (ERP) 360 camera.

Alternative/complement to the per-frame affine alignment in video.py
(`align_depth_frame`). Instead of re-estimating monocular depth from scratch
every frame and rescaling it to match a static reference, this warps the
*previous frame's already-stabilized depth* forward using dense optical flow
(WAFT), and only falls back to monocular depth where the warp cannot be
trusted (disocclusion, pole singularity, low flow confidence).

Why this needs ERP-specific handling (camera is FIXED, panorama is ERP):
  1. Seam wraparound: WAFT is a perspective-video flow model with a finite
     correlation search window; it has no notion that column 0 and column
     W-1 are adjacent on the sphere. An object crossing the seam looks like
     a teleport (huge spurious horizontal flow) unless we give the model
     wrapped context. We circularly pad the frame pair horizontally before
     inference, then crop back — this lets local correlation search "see
     across" the seam for objects that don't move more than `seam_pad`
     pixels per frame.
  2. Pole singularity: near the top/bottom rows the ERP mapping is
     degenerate (a full row of pixels can represent a near-point on the
     sphere). Small angular motion there produces huge/unstable pixel-space
     flow, and WAFT was not trained for this distortion. We simply refuse
     to trust flow propagation within `pole_margin_frac` of the top/bottom
     edges and always fall back to monocular depth there.
  3. No global camera motion: because the camera is fixed, static
     background pixels have ~zero flow everywhere. This means (a) flow is
     cheap to trust as "no propagation needed" over most of the frame, and
     (b) any pixel that becomes newly visible (disocclusion) is, by
     construction, revealed *static background* — so instead of trusting a
     fresh (possibly scale-drifted) monocular estimate there, we can just
     read the already-known, artifact-free reference depth D_ref at that
     pixel.
  4. Blind spot — pure radial motion: optical flow only captures apparent
     2D displacement. An object moving directly along the camera ray
     (towards/away from the fixed camera) has near-zero flow but a real
     depth change. A pure "replace with warped depth" strategy would freeze
     such objects at a stale depth. We therefore never fully replace the
     monocular signal — we blend, weighted by a per-pixel confidence that
     decays over consecutive propagation steps, so any radial drift gets
     continuously re-anchored to the (noisier but unbiased) monocular
     estimate rather than accumulating forever.

This module is intentionally standalone (no SAM3 dependency) so it can be
prototyped/benchmarked without the full run_video pipeline. Integration
point in video.py would replace the body of the per-frame loop that calls
`align_depth_frame` with a call to `propagate_depth_via_flow`, feeding it
the previous frame's fused mask/depth and this frame's WAFT flow.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


def pad_circular_horizontal(frame: np.ndarray, pad: int) -> np.ndarray:
    """Wrap-pad an ERP frame horizontally so flow inference can match content
    across the left/right seam (column 0 is adjacent to column W-1)."""
    if pad <= 0:
        return frame
    left = frame[:, -pad:]
    right = frame[:, :pad]
    return np.concatenate([left, frame, right], axis=1)


def crop_horizontal(arr: np.ndarray, pad: int, width: int) -> np.ndarray:
    if pad <= 0:
        return arr
    return arr[:, pad : pad + width]


def compute_bidirectional_flow(
    waft, frame_a: np.ndarray, frame_b: np.ndarray, seam_pad: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Dense flow a->b and b->a, computed with circular horizontal padding so
    displacements up to `seam_pad` px across the ERP seam are resolved
    correctly. Returned flow is in ORIGINAL (unpadded) pixel coordinates and
    values may point outside [0, W) — callers must wrap x modulo W (see
    `warp_backward`).
    """
    H, W = frame_a.shape[:2]
    a_pad = pad_circular_horizontal(frame_a, seam_pad)
    b_pad = pad_circular_horizontal(frame_b, seam_pad)
    flow_fwd = crop_horizontal(waft.infer_pair(a_pad, b_pad), seam_pad, W)
    flow_bwd = crop_horizontal(waft.infer_pair(b_pad, a_pad), seam_pad, W)
    return flow_fwd, flow_bwd


def _sample_grid_erp(H: int, W: int, flow: np.ndarray, device) -> torch.Tensor:
    """Build a grid_sample() grid for backward-warping: source coordinates
    `p - flow(p)`, wrapped modulo W on the horizontal (seam) axis and
    clamped on the vertical axis (no wraparound over the poles).
    """
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    src_x = (xs - flow[..., 0]) % W  # circular wrap across the seam
    src_y = ys - flow[..., 1]
    src_y = np.clip(src_y, 0, H - 1)  # no wraparound over poles: clamp

    grid_x = (src_x / (W - 1)) * 2.0 - 1.0
    grid_y = (src_y / (H - 1)) * 2.0 - 1.0
    grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    return torch.from_numpy(grid).unsqueeze(0).to(device)  # (1,H,W,2)


def warp_backward(src: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp a single-channel map (e.g. depth) from frame t-1 into
    frame t's pixel grid using flow (t-1 -> t): out(p) = src(p - flow(p)),
    with circular sampling on x (ERP seam) and clamped sampling on y.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = src.shape
    grid = _sample_grid_erp(H, W, flow, device)
    src_t = torch.from_numpy(src).float().to(device).view(1, 1, H, W)
    out = F.grid_sample(src_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.view(H, W).cpu().numpy()


def fb_consistency_error(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    """Forward-backward consistency error in pixels: warp flow_bwd back
    along flow_fwd and measure how far it is from cancelling flow_fwd.
    High error => occlusion / disocclusion / unreliable flow at that pixel.
    """
    H, W = flow_fwd.shape[:2]
    warped_bwd_x = warp_backward(flow_bwd[..., 0], flow_fwd)
    warped_bwd_y = warp_backward(flow_bwd[..., 1], flow_fwd)
    err_x = flow_fwd[..., 0] + warped_bwd_x
    err_y = flow_fwd[..., 1] + warped_bwd_y
    return np.sqrt(err_x**2 + err_y**2)


def pole_trust_mask(H: int, W: int, pole_margin_frac: float = 0.08) -> np.ndarray:
    """1.0 in the trustworthy latitude band, 0.0 within `pole_margin_frac` of
    top/bottom rows where ERP distortion makes flow unreliable."""
    margin = int(H * pole_margin_frac)
    mask = np.ones((H, W), dtype=np.float32)
    if margin > 0:
        mask[:margin, :] = 0.0
        mask[H - margin :, :] = 0.0
    return mask


@dataclass
class PropagationState:
    """Carries the per-pixel confidence decay across frames so radial motion
    (invisible to flow) gets continuously re-anchored to monocular depth."""

    confidence: np.ndarray | None = None  # (H, W) in [0, 1], None before first frame
    decay: float = 0.85  # confidence multiplier per consecutive propagation step


def propagate_depth_via_flow(
    depth_prev_final: np.ndarray,
    depth_curr_affine: np.ndarray,
    depth_ref: np.ndarray,
    flow_fwd: np.ndarray,
    flow_bwd: np.ndarray,
    state: PropagationState,
    fb_err_threshold: float = 1.5,
    pole_margin_frac: float = 0.08,
    disocclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Produce this frame's depth by blending:
      - depth_propagated = warp(depth_prev_final, flow_fwd)   [trusted where flow is good]
      - depth_ref                                              [trusted for newly-revealed
                                                                  static background, since the
                                                                  camera is fixed]
      - depth_curr_affine                                      [fallback: fresh monocular
                                                                  estimate, current pipeline]

    Returns:
        depth_final, confidence_map (H,W) in [0,1], debug dict with the raw
        FB-error / trust masks (useful for tuning thresholds).
    """
    H, W = depth_curr_affine.shape

    depth_propagated = warp_backward(depth_prev_final, flow_fwd)

    fb_err = fb_consistency_error(flow_fwd, flow_bwd)
    fb_trust = (fb_err < fb_err_threshold).astype(np.float32)
    pole_trust = pole_trust_mask(H, W, pole_margin_frac)

    frame_confidence = fb_trust * pole_trust

    if state.confidence is None:
        running_confidence = frame_confidence
    else:
        running_confidence = np.minimum(frame_confidence, state.confidence * state.decay)

    # Disocclusion: pixels flagged unreliable by FB-consistency that are NOT
    # inside the (previous) dynamic-object mask are revealed static
    # background -> use D_ref directly instead of noisy fresh monocular depth.
    depth_final = (
        running_confidence * depth_propagated + (1.0 - running_confidence) * depth_curr_affine
    )
    if disocclusion_mask is not None:
        reveal = (running_confidence < 0.5) & (disocclusion_mask == 0)
        depth_final = np.where(reveal, depth_ref, depth_final)

    depth_final = np.clip(depth_final, 0.0, None)

    # Confidence carried forward resets to 1.0 wherever we just trusted a
    # fresh propagation, decays otherwise -- re-anchors radial-motion drift.
    state.confidence = np.where(frame_confidence > 0, 1.0, running_confidence)

    debug = {
        "fb_err": fb_err,
        "fb_trust": fb_trust,
        "pole_trust": pole_trust,
        "confidence": running_confidence,
        "depth_propagated": depth_propagated,
    }
    return depth_final, running_confidence, debug
