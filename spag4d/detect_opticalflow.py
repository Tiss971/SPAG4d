"""
benchmark_flow.py
-----------------
Benchmark motion/tracking models on your own videos.

Models:
  - WAFT       (princeton-vl/WAFT)            optical flow dense, frame-to-frame
  - CoTracker3 (facebookresearch/co-tracker)  point tracking long-term, grid-based
  - DELTA      (snap-research/DELTA_densetrack3d) dense 3D tracking, needs depth

Paradigm differences
--------------------
WAFT        : flow (H,W,2) between consecutive frames  → direct motion mask
CoTracker3  : trajectories (B,T,N,2) for grid points   → motion mask via displacement
DELTA       : 3D trajectories (T,H,W,3) every pixel    → motion mask via 3D displacement

Usage:
    python benchmark_flow.py \
        --video /path/to/video.mp4 \
        --models waft cotracker delta \
        --waft_checkpoint    /path/to/waft_a1.pth \
        --waft_config        /path/to/waft_a1.json \
        --delta_checkpoint   /path/to/densetrack3d.pth \
        --output_folder      /path/to/output \
        --max_frames 200 \
        --max_size 1280 \
        --motion_threshold 2.0 \
        --cotracker_grid_size 50

    # CoTracker loads from torch.hub automatically (no checkpoint arg needed)
    # DELTA requires UniDepth installed (see DELTA README)
    # WAFT requires WAFT_ROOT env var pointing to the cloned repo

Outputs per model (in output_folder/<model>/):
    flow_viz.mp4        HSV colorwheel (WAFT) or displacement colormap (CoTracker/DELTA)
    motion_mask.mp4     binary motion mask thresholded on magnitude
    benchmark_stats.csv timing + mean magnitude across all models
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .progress import log_tqdm as tqdm

# from spag4d.video import print_gpu_stats

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def flow_to_rgb(flow_xy: np.ndarray) -> np.ndarray:
    """(H,W,2) float32 → BGR HSV colorwheel visualization."""
    fx, fy = flow_xy[..., 0], flow_xy[..., 1]
    angle = np.arctan2(fy, fx) + np.pi
    mag = np.sqrt(fx**2 + fy**2)
    mag = np.clip(mag / (mag.max() + 1e-6), 0, 1)
    h, w = flow_xy.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def mag_to_mask(flow_xy: np.ndarray, threshold: float) -> np.ndarray:
    """(H,W,2) → binary mask uint8 0/255."""
    mag = np.sqrt(flow_xy[..., 0] ** 2 + flow_xy[..., 1] ** 2)
    return (mag > threshold).astype(np.uint8) * 255

def reencode_h264(src: str | Path) -> str:
    if isinstance(src, Path):
        src = str(src)

    dst = src.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst],
            check=True,
            capture_output=True,
        )
    except Exception as e:
        print(e)
        pass
    else:
        # delete path
        os.remove(src)
    return dst

def abs_to_rel_coords(coords, width, height, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / width, y / height] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / width, y / height, w / width, h / height]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

def read_video_frames(path: str, max_frames=None, max_size=1280):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"Cannot open: {path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute scale factor to fit longest side within max_length
    scale = min(max_size / max(W, H), 1.0)  # never upscale
    out_W = int(W * scale) & ~1  # force even dimensions (codec requirement)
    out_H = int(H * scale) & ~1

    meta = {
        "width":  out_W,
        "height": out_H,
        "fps":    int(cap.get(cv2.CAP_PROP_FPS)),
        "total":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "scale":  scale,
    }

    frames = []
    limit = min(meta["total"], max_frames) if max_frames else meta["total"]
    for _ in range(limit):
        ret, f = cap.read()
        if not ret:
            break
        if scale < 1.0:
            f = cv2.resize(f, (out_W, out_H), interpolation=cv2.INTER_AREA)
        frames.append(f)

    cap.release()
    return frames, meta


# ---------------------------------------------------------------------------
# WAFT wrapper
# Optical flow between consecutive frame pairs → (H,W,2) per pair
# ---------------------------------------------------------------------------
class WAFTWrapper:
    """
    Requires: WAFT repo cloned, WAFT_ROOT env var set (or edit path below).
    Checkpoint: waft_a1.pth recommended for downstream applications.
    Input:  two (H,W,3) uint8 BGR frames
    Output: flow (H,W,2) in pixels, for that frame pair
    """

    def __init__(self, checkpoint: str, config: str):
        waft_root = os.environ.get("WAFT_ROOT", "/raid/mb273924/WAFT")
        if waft_root not in sys.path:
            sys.path.insert(0, waft_root)

        # Forcer le cwd vers WAFT_ROOT pour que les chemins relatifs fonctionnent
        _saved_cwd = os.getcwd()
        os.chdir(waft_root)


        try:
            from config.parser import json_to_args
            from inference_tools import InferenceWrapper
            from model import fetch_model
            from utils.utils import load_ckpt

            args = json_to_args(config)
            args.ckpt = checkpoint

            model = fetch_model(args)
            load_ckpt(model, checkpoint)
            model = model.cuda().eval()

            self.model = InferenceWrapper(
                model,
                scale=0.0,
                train_size=args.image_size,
                pad_to_train_size=False,
                tiling=False,
            )
            print("[WAFT] Loaded.")
        finally:
            os.chdir(_saved_cwd)  # toujours restaurer, même si erreur

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0-255, pas /255
        return torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).cuda()

    @torch.no_grad()
    def infer_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        t1, t2 = self._to_tensor(frame1), self._to_tensor(frame2)
        output = self.model.calc_flow(t1, t2)
        flow = output['flow'][-1]               # (1,2,H,W)
        return flow[0].permute(1, 2, 0).cpu().numpy()  # (H,W,2)

    def run(self, frames: list, meta: dict, out_dir: str, threshold: float) -> dict:
        # Dimensions prises sur les frames elles-mêmes : meta["new_H"/"new_W"]
        # n'existe que si un downscale a eu lieu en amont.
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz_w = cv2.VideoWriter(f"{out_dir}/flows.mp4", fourcc, meta["fps"], (W, H*2))

        timings, mags = [], []
        pbar = tqdm(range(len(frames) - 1), desc="[WAFT]")
        # zeros et non empty : la boucle ne remplit que 0..N-2, la dernière
        # frame doit avoir un masque vide (pas de garbage mémoire).
        masks = np.zeros((len(frames), H, W), dtype=np.uint8)
        for i in pbar:
            t0 = time.perf_counter()
            flow = self.infer_pair(frames[i], frames[i + 1])
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mask = (mag > threshold).astype(np.uint8) * 255
            masks[i] = mask
            timings.append(time.perf_counter() - t0)
            mags.append(float(mag.mean()))
            viz_w.write(np.concatenate([flow_to_rgb(flow), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=0))

            pbar.set_postfix(fps=f"{1/np.mean(timings[-10:]):.1f}", mean_mag=f"{mags[-1]:.2f}")


        viz_w.release()
        return {"timings": timings, "mags": mags, "masks": masks}


# ---------------------------------------------------------------------------
# CoTracker3 wrapper
# Tracks a grid of N×N points across the whole video → displacement per frame
# Motion mask = points with large displacement from frame t-1
# ---------------------------------------------------------------------------
class CoTrackerWrapper:
    """
    Loads CoTracker3 via torch.hub (no manual checkpoint needed).
    Use --cotracker_online for long videos (sliding window, less VRAM).
    Use --cotracker_grid_size to control point density (default 50 = 2500 pts).

    Output per frame: pseudo-flow (H,W,2) built from point displacements via
    nearest-neighbour interpolation — not dense optical flow, but comparable
    for motion masking purposes.
    """

    def __init__(self, online: bool = False, grid_size: int = 50):
        self.online = online
        self.grid_size = grid_size
        model_name = "cotracker3_online" if online else "cotracker3_offline"
        print(f"[CoTracker] Loading {model_name} from torch.hub...")
        self.model = torch.hub.load("facebookresearch/co-tracker", model_name).to(device)
        self.model.eval()
        print(f"[CoTracker] Loaded. {'Online' if self.online else 'Offline'}")

    def _tracks_to_flow_field(
        self,
        tracks: np.ndarray,  # (T, N, 2)  pixel coords
        visibility: np.ndarray,  # (T, N)     bool
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Build a (T-1, H, W, 2) pseudo-flow from point trajectories.
        For each frame t, displacement = coords[t] - coords[t-1].
        Interpolated to full resolution via nearest-neighbour.
        """
        T, N, _ = tracks.shape
        flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)

        for t in range(1, T):
            disp = tracks[t] - tracks[t - 1]  # (N,2)
            vis = visibility[t] & visibility[t - 1]  # (N,)
            pts = tracks[t - 1][vis].astype(np.float32)
            dxy = disp[vis]

            if len(pts) == 0:
                continue

            # Scatter displacements onto a canvas, then dilate to fill
            canvas_x = np.zeros((H, W), dtype=np.float32)
            canvas_y = np.zeros((H, W), dtype=np.float32)
            xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
            ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
            canvas_x[ys, xs] = dxy[:, 0]
            canvas_y[ys, xs] = dxy[:, 1]

            # Simple dilation to propagate sparse values to neighbors
            k = max(H, W) // self.grid_size + 2
            kernel = np.ones((k, k), np.float32)
            canvas_x = cv2.dilate(canvas_x, kernel)
            canvas_y = cv2.dilate(canvas_y, kernel)
            flows[t - 1, ..., 0] = canvas_x
            flows[t - 1, ..., 1] = canvas_y

        return flows

    @torch.no_grad()
    def run(self, frames: list, meta: dict, out_dir: str, threshold: float) -> dict:
        H, W = meta["height"], meta["width"]
        T = len(frames)

        # Build video tensor (B=1, T, C, H, W) float [0,255]
        video = (
            torch.tensor(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], axis=0))
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        t0 = time.perf_counter()

        if self.online:
            self.model(video_chunk=video, is_first_step=True, grid_size=self.grid_size)
            pred_tracks_list, pred_vis_list = [], []
            for ind in range(0, T - self.model.step, self.model.step):
                chunk = video[:, ind : ind + self.model.step * 2]
                pred_tracks, pred_visibility = self.model(video_chunk=chunk)
                pred_tracks_list.append(pred_tracks)
                pred_vis_list.append(pred_visibility)
                # print_gpu_stats("co_tracker FRAME "+ str(ind) + " "  + str(video.shape) + " loaded")
            pred_tracks = torch.cat(pred_tracks_list, dim=1)
            pred_visibility = torch.cat(pred_vis_list, dim=1)
        else:
            pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size)

        total_time = time.perf_counter() - t0
        print(f"  [CoTracker] Inference done in {total_time:.1f}s for {T} frames")
        # print_gpu_stats("co_tracker AFTER " + str(video.shape) + " loaded")

        from detect_opticalflow_utils import CoTrackerVisualizer
        ctvis = CoTrackerVisualizer(save_dir=out_dir, pad_value=120, linewidth=3, fps=meta["fps"])
        ctvis.visualize(video, pred_tracks, pred_visibility, filename="viz")

        # pred_tracks: (1,T,N,2)  pred_visibility: (1,T,N)
        tracks = pred_tracks[0].cpu().numpy()  # (T,N,2)
        vis = pred_visibility[0].cpu().numpy()  # (T,N)

        print("  [CoTracker] Building flow fields...")
        flows = self._tracks_to_flow_field(tracks, vis, H, W)  # (T-1,H,W,2)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz_w = cv2.VideoWriter(f"{out_dir}/flow_viz.mp4", fourcc, meta["fps"], (W, H))
        mask_w = cv2.VideoWriter(f"{out_dir}/motion_mask.mp4", fourcc, meta["fps"], (W, H))
        mags = []

        for flow in flows:
            mags.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
            viz_w.write(flow_to_rgb(flow))
            mask = mag_to_mask(flow, threshold)
            mask_w.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        viz_w.release()
        mask_w.release()

        # Timings: distribute total time evenly across frames
        per_frame = total_time / max(T - 1, 1)
        timings = [per_frame] * (T - 1)
        return {"timings": timings, "mags": mags}


# ---------------------------------------------------------------------------
# FlowIt stub
# ---------------------------------------------------------------------------
# class FlowItWrapper:
#     def __init__(self, checkpoint): ...
#     def run(self, frames, meta, out_dir, threshold): ...


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Tracking Benchmark")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["waft", "cotracker", "delta"],
        choices=["waft", "cotracker", "delta", "flowit"],
    )
    parser.add_argument("--waft_checkpoint", type=str, default=None)
    parser.add_argument("--waft_config", type=str, default=None)
    parser.add_argument("--delta_checkpoint", type=str, default=None, help="Path to densetrack3d.pth")
    parser.add_argument(
        "--delta_use_fp16",
        action="store_true",
        help="Enable fp16 for DELTA (~20GB VRAM instead of ~40GB)",
    )
    parser.add_argument(
        "--delta_upsample_factor",
        type=int,
        default=4,
        help="DELTA upsample factor (4=default, 8=less VRAM)",
    )
    parser.add_argument(
        "--cotracker_grid_size", type=int, default=20, help="NxN grid of tracked points for CoTracker"
    )
    parser.add_argument(
        "--cotracker_online",
        action="store_true",
        help="Use CoTracker online mode (more memory efficient for long videos)",
    )
    parser.add_argument("--output_folder", type=str, default="./flow_benchmark")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_size", type=int, default=1280,
                        help="Downscale frames so longest side <= max_length")
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=1.0,
        help="Pixel displacement threshold for motion mask",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_folder, exist_ok=True)
    args.output_folder = args.output_folder  + "/" + os.path.basename(args.video).split('.')[0]
    os.makedirs(args.output_folder, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Main
    # ---------------------------------------------------------------------------
    print(f"\nReading video: {args.video}")
    frames, meta = read_video_frames(args.video, args.max_frames, args.max_size)
    print(f"  {len(frames)} frames | {meta['width']}×{meta['height']} | {meta['fps']} fps")

    csv_rows = []

    for model_name in args.models:
        print(f"\n{'=' * 55}")
        print(f"  {model_name.upper()}")
        print(f"{'=' * 55}")

        out_dir = os.path.join(args.output_folder, model_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            if model_name == "waft":
                assert args.waft_checkpoint, "--waft_checkpoint required for WAFT"
                model = WAFTWrapper(args.waft_checkpoint, args.waft_config)
            elif model_name == "cotracker":
                model = CoTrackerWrapper(online=args.cotracker_online, grid_size=args.cotracker_grid_size)
            elif model_name == "flowit":
                raise NotImplementedError("FlowIt not yet released.")
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        try:
            stats = model.run(frames, meta, out_dir, args.motion_threshold)
        except Exception as e:
            print(f"  [ERROR during inference] {e}")
            import traceback

            traceback.print_exc()
            continue

        # Re-encode for VSCode preview
        for fname in ["flow_viz.mp4", "motion_mask.mp4" , "viz.mp4"]:
            src = os.path.join(out_dir, fname)
            if os.path.exists(src):
                try:
                    reencode_h264(src)
                except Exception:
                    pass

        timings = stats["timings"]
        mags = stats["mags"]
        mean_t = float(np.mean(timings))
        std_t = float(np.std(timings))
        inf_fps = 1.0 / (mean_t + 1e-9)

        print(f"\n[{model_name.upper()}] Results:")
        print(f"  Inference FPS  : {inf_fps:.1f}")
        print(f"  Time/frame     : {mean_t * 1000:.1f} ± {std_t * 1000:.1f} ms")
        print(f"  Mean magnitude : {float(np.mean(mags)):.3f} px/frame")

        csv_rows.append({
            "model": model_name,
            "frames": len(timings),
            "mean_ms": round(mean_t * 1000, 2),
            "std_ms": round(std_t * 1000, 2),
            "fps": round(inf_fps, 1),
            "mean_magnitude": round(float(np.mean(mags)), 4),
        })

    if csv_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_stats.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nStats → {csv_path}")

    print(f"Done. Outputs in: {args.output_folder}/")


