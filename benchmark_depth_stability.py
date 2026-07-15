#!/usr/bin/env python3
"""
Depth-stability benchmark for a FIXED 360 (ERP) camera, at native 4K-class
resolution and PLY stride 4.

Compares three per-frame depth strategies on the same footage:

  A. affine   — current pipeline: fresh DA360 monocular depth every frame,
                affine-aligned (s*D + t) to the reference over static pixels.
  B. flowprop — warp the previous frame's stabilized depth forward with WAFT
                optical flow, blend back to monocular by a confidence that
                decays over consecutive propagation steps (spag4d.flow_depth_propagation).
  C. bglock   — background-locked compositing: STATIC pixels are locked to the
                single reference depth D_ref (zero temporal variance by
                construction); the flow-propagated object depth is used only
                inside a feathered dynamic mask. Directly targets the goal
                "background fixed, moving object moves correctly in it."

Metrics (the reason this benchmark exists — depth *stability*, not accuracy):

  bg_temporal_std   mean over always-static pixels of the per-pixel temporal
                    std of depth. THE headline: a fixed camera's background
                    should not change depth. Lower = more fixed background.
  bg_flicker_p2p    mean over always-static pixels of mean_t |D_t - D_{t-1}|.
                    The frame-to-frame flicker a viewer actually sees.
  fg_temporal_std   same temporal std over ever-dynamic pixels. Not "lower is
                    strictly better" (objects genuinely move) — read together
                    with the median-FG-depth plot to check motion is tracked
                    smoothly rather than flickering.
  spatial_rough     mean over frames of mean |Laplacian(D)|/D over static
                    pixels — pixel-to-pixel roughness within a frame.

Flow is computed at a reduced resolution (--flow-max-size) then upscaled, since
WAFT correlation volumes are expensive at 2K+ width; depth/compositing/PLY all
run at native resolution.

Usage:
  conda run -n spag4d python benchmark_depth_stability.py \
      --video /raid/mb273924/SPAG4d/TestImage/9_MattSwift.mp4 \
      --output ./benchmark_depth_stability --max-frames 150 --ply-stride 4
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from spag4d.core import SPAG4D
from spag4d.da360_model import DA360Model
from spag4d.detect_opticalflow import WAFTWrapper
from spag4d.flow_depth_propagation import (
    PropagationState,
    compute_bidirectional_flow,
    composite_bg_locked,
    propagate_depth_via_flow,
)
from spag4d.ply_writer import save_ply_gsplat
from spag4d.video import align_depth_frame, extract_video_frames, to_gaussians


def resize_frame(frame: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    """Downscale a single (H,W,3) uint8 frame so the longest side <= max_size.
    Returns (resized, scale) where scale = new/old (<=1)."""
    H, W = frame.shape[:2]
    scale = min(max_size / max(H, W), 1.0)
    if scale >= 1.0:
        return frame, 1.0
    new_H, new_W = int(H * scale) & ~1, int(W * scale) & ~1
    out = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_AREA)
    return out, new_W / W


def upscale_flow(flow: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a (h,w,2) pixel-displacement flow field to (out_h,out_w,2),
    scaling the displacement magnitudes to the new pixel grid."""
    h, w = flow.shape[:2]
    if (h, w) == (out_h, out_w):
        return flow
    sx, sy = out_w / w, out_h / h
    resized = cv2.resize(flow, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= sx
    resized[..., 1] *= sy
    return resized


class StreamStats:
    """Per-pixel streaming temporal statistics (mean, std, frame-to-frame diff)
    so we never hold a full [N, H, W] depth stack for each method in RAM."""

    def __init__(self, shape):
        self.n = 0
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.prev = None
        self.p2p_sum = np.zeros(shape, dtype=np.float64)  # sum_t |D_t - D_{t-1}|
        self.p2p_n = 0

    def update(self, depth: np.ndarray):
        d = depth.astype(np.float64)
        self.sum += d
        self.sumsq += d * d
        self.n += 1
        if self.prev is not None:
            self.p2p_sum += np.abs(d - self.prev)
            self.p2p_n += 1
        self.prev = d

    def std_map(self) -> np.ndarray:
        mean = self.sum / max(self.n, 1)
        var = np.maximum(self.sumsq / max(self.n, 1) - mean * mean, 0.0)
        return np.sqrt(var)

    def p2p_map(self) -> np.ndarray:
        return self.p2p_sum / max(self.p2p_n, 1)


def relative_laplacian_roughness(depth: np.ndarray, region: np.ndarray) -> float:
    """Mean |Laplacian(depth)| / depth over `region` — a scale-invariant
    measure of within-frame pixel-to-pixel roughness (high-frequency noise)."""
    lap = cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F, ksize=3)
    rel = np.abs(lap) / np.maximum(depth, 1e-3)
    vals = rel[region]
    return float(vals.mean()) if vals.size else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--output", type=str, default="./benchmark_depth_stability")
    p.add_argument("--skip-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--work-max-size", type=int, default=0,
                   help="Downscale longest side to this before depth/compositing (0 = native).")
    p.add_argument("--flow-max-size", type=int, default=1024,
                   help="Resolution for WAFT flow (upscaled to working res for warping).")
    p.add_argument("--motion-threshold", type=float, default=1.0, help="px flow mag for dynamic mask")
    p.add_argument("--mask-dilate", type=int, default=9, help="px dilation of the raw dynamic mask (proxy for SAM)")
    p.add_argument("--seam-pad", type=int, default=64)
    p.add_argument("--fb-err-threshold", type=float, default=1.5)
    p.add_argument("--pole-margin-frac", type=float, default=0.08)
    p.add_argument("--ply-stride", type=int, default=4)
    p.add_argument("--ply-export-count", type=int, default=5)
    p.add_argument("--da360-weights", type=str, default=str(Path.home() / ".cache/spag4d/DA360_large.pth"))
    p.add_argument("--waft-checkpoint", type=str, default="/raid/mb273924/_DATASETS/uptale/tar-c-t.pth")
    p.add_argument("--waft-config", type=str, default="/raid/mb273924/WAFT/config/a1/tar-c-t.json")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print("[1/6] Extracting frames...")
    frames, meta = extract_video_frames(args.video, skip_step=args.skip_step)
    frames = frames[: args.max_frames]
    native_H, native_W = frames.shape[1:3]
    if args.work_max_size and max(native_H, native_W) > args.work_max_size:
        frames = np.stack([resize_frame(f, args.work_max_size)[0] for f in frames], axis=0)
    n = len(frames)
    H, W = frames.shape[1:3]
    frames_rgb = frames[:, :, :, ::-1].copy()  # BGR->RGB for DA360
    print(f"  {n} frames | native {native_W}x{native_H} -> working {W}x{H} | flow @ <= {args.flow_max_size}px")

    print("[2/6] Loading models (DA360 + WAFT)...")
    da360 = DA360Model.load(model_path=args.da360_weights, device=device)
    waft = WAFTWrapper(checkpoint=args.waft_checkpoint, config=args.waft_config)

    print("[3/6] Bidirectional WAFT flow + dynamic masks...")
    flows_fwd, flows_bwd, dyn_masks = [], [], []
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.mask_dilate + 1,) * 2) if args.mask_dilate else None
    t0 = time.time()
    for i in range(n - 1):
        fa, _ = resize_frame(frames[i], args.flow_max_size)
        fb, _ = resize_frame(frames[i + 1], args.flow_max_size)
        ff_lo, fb_lo = compute_bidirectional_flow(waft, fa, fb, seam_pad=args.seam_pad)
        ff = upscale_flow(ff_lo, H, W)
        fbk = upscale_flow(fb_lo, H, W)
        flows_fwd.append(ff)
        flows_bwd.append(fbk)
        mag = np.sqrt(ff[..., 0] ** 2 + ff[..., 1] ** 2)
        m = (mag > args.motion_threshold).astype(np.uint8)
        if dilate_k is not None:
            m = cv2.dilate(m, dilate_k)
        dyn_masks.append(m)
    dyn_masks.append(dyn_masks[-1].copy() if dyn_masks else np.zeros((H, W), np.uint8))
    print(f"  {n - 1} pairs in {time.time() - t0:.1f}s")

    print("[4/6] Reference depth from masked temporal-median background...")
    masks_arr = np.stack(dyn_masks, axis=0).astype(bool)
    master_bg = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        ch = frames_rgb[..., c].astype(np.float32).copy()
        ch[masks_arr] = np.nan
        with np.errstate(all="ignore"):
            master_bg[..., c] = np.nan_to_num(np.nanmedian(ch, axis=0), nan=0.0)
    master_bg = master_bg.astype(np.uint8)
    cv2.imwrite(str(out / "_master_bg.jpg"), cv2.cvtColor(master_bg, cv2.COLOR_RGB2BGR))

    with torch.inference_mode():
        depth_raw, _ = da360.predict(torch.from_numpy(master_bg).to(device), temporal_consistency=True)
    dmed = float(depth_raw.median())
    scale_to_5m = 5.0 / dmed if dmed > 1e-6 else 1.0
    depth_ref = (depth_raw * scale_to_5m).cpu().numpy().astype(np.float32)

    print("[5/6] Per-frame depth for all three methods + streaming stats...")
    methods = ["affine", "flowprop", "bglock"]
    stats = {m: StreamStats((H, W)) for m in methods}
    rough = {m: [] for m in methods}
    median_fg = {m: [] for m in methods}
    ever_dynamic = np.zeros((H, W), dtype=bool)

    state = PropagationState(decay=0.85)
    depth_prev_flowprop = None

    ply_frames = set()
    if args.ply_export_count > 0 and n > 0:
        ply_frames = set(np.linspace(0, n - 1, min(args.ply_export_count, n), dtype=int).tolist())
        for m in ("affine", "bglock"):
            (out / f"gaussians_{m}").mkdir(exist_ok=True)
        converter = SPAG4D(device="cuda", generator="pager")  # avoid eager DA360 reload

    # Static region for the spatial-roughness metric: pixels dynamic in NO frame.
    # (accumulated online; roughness uses the running static estimate, close to
    #  final since dynamic coverage stabilizes within a few frames)
    for idx in range(n):
        frame_tensor = torch.from_numpy(frames_rgb[idx]).to(device)
        with torch.inference_mode():
            draw_i, _ = da360.predict(frame_tensor, temporal_consistency=True)
        depth_curr = (draw_i * scale_to_5m).cpu().numpy().astype(np.float32)

        mask = dyn_masks[idx]
        ever_dynamic |= mask.astype(bool)

        depth_affine = align_depth_frame(depth_curr.copy(), depth_ref, mask, method="lstsq").copy()

        if idx == 0 or depth_prev_flowprop is None:
            depth_flow = depth_affine.copy()
        else:
            depth_flow, _, _ = propagate_depth_via_flow(
                depth_prev_flowprop, depth_affine, depth_ref,
                flows_fwd[idx - 1], flows_bwd[idx - 1], state,
                fb_err_threshold=args.fb_err_threshold,
                pole_margin_frac=args.pole_margin_frac,
                disocclusion_mask=mask,
            )
        depth_prev_flowprop = depth_flow

        depth_bglock, _ = composite_bg_locked(depth_flow, depth_ref, mask,
                                               dilate_px=max(args.mask_dilate, 6), feather_px=9)

        frame_depths = {"affine": depth_affine, "flowprop": depth_flow, "bglock": depth_bglock}
        static_now = ~ever_dynamic
        fg = mask > 0
        for m in methods:
            d = frame_depths[m]
            stats[m].update(d)
            rough[m].append(relative_laplacian_roughness(d, static_now))
            median_fg[m].append(float(np.median(d[fg])) if fg.sum() > 0 else float("nan"))

        if idx in ply_frames:
            for tag, dmap in (("affine", depth_affine), ("bglock", depth_bglock)):
                g = to_gaussians(converter, dmap, frame_tensor, args.ply_stride, H)
                save_ply_gsplat(g, str(out / f"gaussians_{tag}" / f"frame_{idx}.ply"),
                                sh_degree=0, colors_linear=False)
        if (idx + 1) % 20 == 0:
            print(f"  frame {idx + 1}/{n}")

    print("[6/6] Aggregating metrics...")
    always_static = ~ever_dynamic
    ns, nf = int(always_static.sum()), int(ever_dynamic.sum())

    def region_mean(arr, region):
        v = arr[region]
        return float(v.mean()) if v.size else float("nan")

    results = {
        "video": args.video,
        "n_frames": n,
        "native_res": [native_W, native_H],
        "working_res": [W, H],
        "flow_max_size": args.flow_max_size,
        "ply_stride": args.ply_stride,
        "ref_depth_median_m": 5.0,
        "n_static_px": ns, "n_dynamic_px": nf,
        "static_frac": ns / (H * W), "dynamic_frac": nf / (H * W),
        "methods": {},
    }
    for m in methods:
        std_map, p2p_map = stats[m].std_map(), stats[m].p2p_map()
        results["methods"][m] = {
            "bg_temporal_std_m": region_mean(std_map, always_static),
            "bg_flicker_p2p_m": region_mean(p2p_map, always_static),
            "fg_temporal_std_m": region_mean(std_map, ever_dynamic),
            "spatial_rough_static": float(np.nanmean(rough[m])),
            "median_fg_series": median_fg[m],
        }

    with open(out / "stats.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- plots ----
    colors = {"affine": "orange", "flowprop": "tab:blue", "bglock": "tab:green"}
    labels = {"affine": "A. affine (current)", "flowprop": "B. flow-prop", "bglock": "C. bg-locked (new)"}

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # (1) background stability bars — the headline
    bg_std = [results["methods"][m]["bg_temporal_std_m"] for m in methods]
    bg_p2p = [results["methods"][m]["bg_flicker_p2p_m"] for m in methods]
    x = np.arange(len(methods))
    ax[0].bar(x - 0.2, bg_std, 0.4, label="temporal std", color="tab:red", alpha=0.8)
    ax[0].bar(x + 0.2, bg_p2p, 0.4, label="frame-to-frame flicker", color="tab:purple", alpha=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([labels[m] for m in methods], rotation=15, ha="right")
    ax[0].set_ylabel("meters"); ax[0].set_title("Background depth stability (lower = more fixed)")
    ax[0].legend(); ax[0].grid(True, axis="y", alpha=0.3)
    # (2) spatial roughness bars
    rgh = [results["methods"][m]["spatial_rough_static"] for m in methods]
    ax[1].bar(x, rgh, 0.5, color=[colors[m] for m in methods])
    ax[1].set_xticks(x); ax[1].set_xticklabels([labels[m] for m in methods], rotation=15, ha="right")
    ax[1].set_ylabel("mean |lap D|/D"); ax[1].set_title("Within-frame spatial roughness (static)")
    ax[1].grid(True, axis="y", alpha=0.3)
    # (3) FG median depth series — objects should move smoothly, not flicker
    for m in methods:
        ax[2].plot(median_fg[m], marker=".", label=labels[m], color=colors[m], alpha=0.85)
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("median FG depth (m)")
    ax[2].set_title("Foreground (moving object) depth over time")
    ax[2].legend(); ax[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=110)

    print("\n=== DEPTH STABILITY RESULTS ===")
    print(f"  {n} frames @ {W}x{H} | static {results['static_frac']*100:.1f}% / dynamic {results['dynamic_frac']*100:.1f}% of frame")
    hdr = f"{'method':<20}{'bg_std(m)':>12}{'bg_flicker(m)':>15}{'fg_std(m)':>12}{'spatial_rough':>15}"
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        r = results["methods"][m]
        print(f"{labels[m]:<20}{r['bg_temporal_std_m']:>12.4f}{r['bg_flicker_p2p_m']:>15.4f}"
              f"{r['fg_temporal_std_m']:>12.4f}{r['spatial_rough_static']:>15.5f}")
    print(f"\nSaved: {out}/stats.json, {out}/comparison.png")
    if ply_frames:
        print(f"PLYs (stride {args.ply_stride}): {out}/gaussians_affine/, {out}/gaussians_bglock/")


if __name__ == "__main__":
    main()
