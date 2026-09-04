# spag4d/flow_depth_propagation.py
"""
WAFT optical-flow computation and bg-locked depth compositing for a FIXED
equirectangular (ERP) 360 camera, used by the bglock depth-correction path
in video.py.

Seam handling: WAFT is a perspective-video flow model with a finite
correlation search window; it has no notion that column 0 and column W-1
are adjacent on the sphere. `compute_bidirectional_flow` circularly pads
the frame pair horizontally before inference (`pad_circular_horizontal`),
then crops back (`crop_horizontal`), so displacements up to `seam_pad`
pixels across the seam resolve correctly.

Flow-warp depth propagation (warping the previous frame's depth forward
along WAFT flow to feed the compositor's object-depth input) was removed
2026-09-04: benchmarked worse (slower, worse fg_depth_cv) than a plain
prev/current blend on real clips. See docs/SPAG_ENV_FLAGS.md and
video.py's bglock compositing loop for the current approach.
"""

import cv2
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


def upscale_flow(flow: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a (h,w,2) pixel-displacement flow field to (out_h,out_w,2),
    scaling the displacement magnitudes to the new pixel grid. Used when flow
    is computed at a reduced resolution (WAFT correlation cost) but depth/
    compositing runs at native resolution."""
    h, w = flow.shape[:2]
    if (h, w) == (out_h, out_w):
        return flow
    sx, sy = out_w / w, out_h / h
    resized = cv2.resize(flow, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= sx
    resized[..., 1] *= sy
    return resized


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


_BASE_GRID_CACHE: dict = {}  # (H, W) -> (ys, xs) base pixel meshgrid, reused across frames


def _base_meshgrid(H: int, W: int):
    key = (H, W)
    g = _BASE_GRID_CACHE.get(key)
    if g is None:
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        g = (ys.astype(np.float32), xs.astype(np.float32))
        _BASE_GRID_CACHE[key] = g
    return g


def _sample_grid_erp(H: int, W: int, flow: np.ndarray, device) -> torch.Tensor:
    """Build a grid_sample() grid for backward-warping: source coordinates
    `p - flow(p)`, wrapped modulo W on the horizontal (seam) axis and
    clamped on the vertical axis (no wraparound over the poles).
    """
    ys, xs = _base_meshgrid(H, W)
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


def feather_dynamic_mask(
    mask: np.ndarray, dilate_px: int = 12, feather_px: int = 9
) -> np.ndarray:
    """Turn a binary dynamic-object mask into a soft alpha map in [0, 1] for
    seamless compositing of per-frame object depth over a locked background.

    The mask is first dilated (to cover DA360's depth-bleed halo around moving
    edges and any 1-frame lag in the mask) then Gaussian-blurred so the
    background<->object depth transition has no hard seam. Both operations are
    done with circular horizontal padding so an object straddling the ERP seam
    is handled correctly.
    """
    H, W = mask.shape
    pad = max(dilate_px + feather_px, 1)
    m = (mask > 0).astype(np.uint8) * 255
    m = np.concatenate([m[:, -pad:], m, m[:, :pad]], axis=1)  # wrap-pad seam
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    alpha = m.astype(np.float32) / 255.0
    if feather_px > 0:
        ksz = 2 * feather_px + 1
        alpha = cv2.GaussianBlur(alpha, (ksz, ksz), 0)
    return alpha[:, pad : pad + W]


def composite_bg_locked(
    object_depth: np.ndarray,
    depth_ref: np.ndarray,
    dynamic_mask: np.ndarray,
    dilate_px: int = 12,
    feather_px: int = 9,
    hard_depth_cutover: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-camera depth compositing.

    The camera never moves, so every pixel outside a moving object is, by
    construction, static background whose true depth is *constant in time* and
    already known artifact-free from the reference depth `depth_ref` (DA360 on
    the temporal-median background). Re-estimating that depth every frame — and
    then affine- or flow-correcting it — only reintroduces monocular noise as
    temporal flicker. So we don't: static pixels are locked to `depth_ref`
    (exactly zero temporal variance), and the fresh per-frame `object_depth`
    (ideally already flow-propagated for temporal coherence) is used *only*
    inside the feathered dynamic mask, where the scene genuinely changes.

    Returns (depth_final, alpha) where alpha is the soft object weight in [0,1].

    `hard_depth_cutover` (§8, bglock_open_questions.md): a plain linear blend
    of `object_depth` and `depth_ref` inside the feather band produces *depth*
    values that are a linear interpolation between foreground and background
    surfaces — a physically-nonexistent "phantom slab" hovering in mid-air at
    a discontinuity (e.g. person in front of a wall), which splats into a
    visible smear of floating points. When True, `alpha` is still returned
    soft (for callers that want a soft opacity/weight), but the *depth value*
    itself is selected by nearest source (alpha>=0.5 -> object_depth, else
    depth_ref) rather than interpolated, so no invented mid-air depth is ever
    written. Opt-in / default off: changes production point-cloud geometry at
    every dynamic-object edge, not benchmarked against the shipped default yet.
    """
    alpha = feather_dynamic_mask(dynamic_mask, dilate_px, feather_px)
    # Where depth_ref is NaN ("ignore" mode), fall back entirely to object_depth.
    # This prevents NaN propagation (0.0 * NaN = NaN) from wiping out valid object depth
    # when alpha == 1.0, allowing the moving subject to cleanly fill the background hole.
    ref_undefined = np.isnan(depth_ref)
    safe_alpha = np.where(ref_undefined, 1.0, alpha)
    safe_ref = np.where(ref_undefined, 0.0, depth_ref)
    if hard_depth_cutover:
        depth_final = np.where(safe_alpha >= 0.5, object_depth, safe_ref)
    else:
        depth_final = safe_alpha * object_depth + (1.0 - safe_alpha) * safe_ref
    return np.clip(depth_final, 0.0, None), alpha


# --- GPU compositing (feather + bg-lock) -----------------------------------
# Torch equivalents of feather_dynamic_mask / composite_bg_locked so the final
# bg-locked composite stays on the GPU in the fast path. NOT byte-identical to
# the cv2/numpy versions: the dilation is a square max-pool (vs cv2's elliptical
# structuring element) and the Gaussian blur is a separable conv with cv2's
# sigma formula. After the feather blur these differ only by sub-pixel amounts,
# but downstream the composite is affected inside the feathered band -> measure.
_GAUSS_KERNEL_CACHE_T: dict = {}


def _gauss_kernel_torch(feather_px: int, device):
    key = (feather_px, str(device))
    k = _GAUSS_KERNEL_CACHE_T.get(key)
    if k is not None:
        return k
    ksz = 2 * feather_px + 1
    # cv2's sigma=0 formula: 0.3*((ksz-1)*0.5 - 1) + 0.8
    sigma = 0.3 * ((ksz - 1) * 0.5 - 1) + 0.8
    xs = torch.arange(ksz, device=device, dtype=torch.float32) - (ksz - 1) / 2.0
    g = torch.exp(-(xs**2) / (2 * sigma * sigma))
    g = g / g.sum()
    _GAUSS_KERNEL_CACHE_T[key] = (g, ksz)
    return g, ksz


def feather_dynamic_mask_torch(
    mask_t: torch.Tensor, dilate_px: int = 12, feather_px: int = 9
) -> torch.Tensor:
    """GPU feather: wrap-pad horizontally, square-dilate, separable Gaussian
    blur. mask_t (H,W) any numeric cuda tensor (>0 => foreground)."""
    H, W = mask_t.shape
    pad = max(dilate_px + feather_px, 1)
    m = (mask_t > 0).float()
    m = torch.cat([m[:, -pad:], m, m[:, :pad]], dim=1)  # wrap-pad seam
    x = m.unsqueeze(0).unsqueeze(0)  # (1,1,H,Wp)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        x = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=dilate_px)
    if feather_px > 0:
        g, ksz = _gauss_kernel_torch(feather_px, mask_t.device)
        f = feather_px
        # separable blur, reflect padding to mirror cv2's BORDER_REFLECT_101
        xh = torch.nn.functional.pad(x, (f, f, 0, 0), mode="reflect")
        xh = torch.nn.functional.conv2d(xh, g.view(1, 1, 1, ksz))
        xv = torch.nn.functional.pad(xh, (0, 0, f, f), mode="reflect")
        x = torch.nn.functional.conv2d(xv, g.view(1, 1, ksz, 1))
    alpha = x[0, 0]
    return alpha[:, pad : pad + W]


def composite_bg_locked_torch(
    object_depth: torch.Tensor,
    depth_ref: torch.Tensor,
    dynamic_mask: torch.Tensor,
    dilate_px: int = 12,
    feather_px: int = 9,
    hard_depth_cutover: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU equivalent of composite_bg_locked. depth_ref may contain NaN
    ("ignore" mode) -> fall back to object_depth there. See composite_bg_locked's
    docstring for what hard_depth_cutover does (§8, bglock_open_questions.md)."""
    alpha = feather_dynamic_mask_torch(dynamic_mask, dilate_px, feather_px)
    ref_undefined = torch.isnan(depth_ref)
    safe_alpha = torch.where(ref_undefined, torch.ones_like(alpha), alpha)
    safe_ref = torch.where(ref_undefined, torch.zeros_like(depth_ref), depth_ref)
    if hard_depth_cutover:
        depth_final = torch.where(safe_alpha >= 0.5, object_depth, safe_ref)
    else:
        depth_final = safe_alpha * object_depth + (1.0 - safe_alpha) * safe_ref
    return depth_final.clamp_min(0.0), alpha
