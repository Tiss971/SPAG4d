"""Animated 3D point-cloud viewer for a SPAG4D depth_maps sequence.

Unlike the actual gaussians/frame_*.ply output (only produced by some runs),
depth_maps/ (depth_{idx}.npy + mask_{idx}.npy) + images/ (frame_{idx:04d}.png)
exist for every run, so this reads that pair directly -- no PLY required.
Reuses the exact ERP unprojection formula from spag4d/spherical_grid.py /
spag_converter.py (rhat = [sin(phi)cos(theta), cos(phi), -sin(phi)sin(theta)],
position = depth * rhat) so the point cloud matches where SPAG4D actually
places its Gaussians, not a reinvented projection.

Requires viser>=1.0 (PointCloudHandle.points/.colors are real mutable props
there -- on 0.2.7 they're inert dataclass fields and updates never reach the
client; see git history of this file). Ray geometry (theta/phi/rhat) depends
only on (H, W, stride), so it's computed once and reused across all frames --
only depth/color resampling happens per frame. Frames are prefetched off the
render/network thread so decode latency doesn't stall playback.

Usage:
    python scripts/view_depth_sequence.py benchmarks/scene01/bglock_fixed \\
        --stride 4 --port 8080
Then open the printed http://localhost:8080 URL and use the frame slider.
"""
import argparse
import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import viser
import viser.transforms as vtf

assert tuple(int(p) for p in viser.__version__.split(".")[:2]) >= (1, 0), (
    f"viser {viser.__version__} is too old -- PointCloudHandle.points/.colors "
    "are inert on <1.0 and updates silently never reach the client. "
    "pip install -U viser"
)


def load_frame(depth_dir: Path, images_dir: Path, idx: int):
    depth = np.load(depth_dir / f"depth_{idx}.npy")
    img = cv2.cvtColor(cv2.imread(str(images_dir / f"frame_{idx:04d}.png")), cv2.COLOR_BGR2RGB)
    return depth, img


class FramePrefetcher:
    """Decodes upcoming frames on a thread pool, ahead of when they're shown.

    Disk read + PNG decode (~ms per frame) was blocking the render loop right
    before every point-cloud update; a small look-ahead window hides that
    latency behind whatever the previous frame's display time already spent.
    """

    def __init__(self, depth_dir: Path, images_dir: Path, n_frames: int, window: int = 12, workers: int = 4):
        self.depth_dir = depth_dir
        self.images_dir = images_dir
        self.n_frames = n_frames
        self.window = window
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        self.futures: OrderedDict[int, object] = OrderedDict()

    def _ensure_submitted(self, idx: int):
        if idx not in self.futures:
            self.futures[idx] = self.pool.submit(load_frame, self.depth_dir, self.images_dir, idx)

    def get(self, idx: int):
        with self.lock:
            self._ensure_submitted(idx)
            for k in range(idx, min(idx + self.window, self.n_frames)):
                self._ensure_submitted(k)
            # Evict anything far behind the playhead so the cache doesn't grow unbounded.
            for k in [k for k in self.futures if k < idx - 2]:
                del self.futures[k]
            fut = self.futures[idx]
        return fut.result()


def unproject_geometry(H: int, W: int, stride: int, device: str):
    """Ray directions only depend on (H, W, stride) -- computed once, reused every frame."""
    v = np.arange(0, H, stride, dtype=np.float32) + stride / 2
    u = np.arange(0, W, stride, dtype=np.float32) + stride / 2
    vv, uu = np.meshgrid(v, u, indexing="ij")

    theta = (1 - uu / W) * 2 * math.pi
    phi = vv / H * math.pi
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    rhat = np.stack([sin_phi * cos_theta, cos_phi, -sin_phi * sin_theta], axis=-1)  # [h,w,3]

    rows = vv.astype(np.int64)
    cols = uu.astype(np.int64)
    return (
        torch.from_numpy(rhat).to(device),
        torch.from_numpy(rows).to(device),
        torch.from_numpy(cols).to(device),
    )


def unproject_frame(depth: np.ndarray, img: np.ndarray, rhat, rows, cols, depth_min: float, depth_max: float, device: str):
    # The per-frame cost here is dominated by fancy indexing + boolean-mask compaction
    # on ~1M-element arrays (measured ~55ms/frame on CPU at stride=2, vs ~4ms on GPU) --
    # gather/compact on the GPU, only transferring the already-compacted result back.
    depth_t = torch.from_numpy(depth).to(device, non_blocking=True)
    img_t = torch.from_numpy(img).to(device, non_blocking=True)
    d = depth_t[rows, cols]
    colors = img_t[rows, cols]
    valid = torch.isfinite(d) & (d > depth_min) & (d < depth_max)
    positions = (d.unsqueeze(-1) * rhat)[valid]
    colors = colors[valid]
    return positions.to(torch.float32).cpu().numpy(), colors.to(torch.uint8).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run output dir with depth_maps/ and images/ subfolders")
    parser.add_argument("--stride", type=int, default=4, help="Spatial subsample stride (matches pipeline's own stride concept)")
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=100.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    depth_dir = args.run_dir / "depth_maps"
    images_dir = args.run_dir / "images"
    n_frames = len(list(depth_dir.glob("depth_[0-9]*.npy")))  # exclude depth_ref.npy
    print(f"Found {n_frames} frames in {depth_dir}")

    prefetcher = FramePrefetcher(depth_dir, images_dir, n_frames)
    depth0, _ = prefetcher.get(0)
    H, W = depth0.shape
    rhat, rows, cols = unproject_geometry(H, W, args.stride, args.device)
    print(f"{rhat.shape[0] * rhat.shape[1]} rays/frame at stride {args.stride}, device={args.device}")

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = False

    gui_frame = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("Play", initial_value=False)
    gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=8)
    gui_up_from_view = server.gui.add_button("up = current view", icon=viser.Icon.ARROW_BIG_UP_LINES)

    point_cloud = None

    def show_frame(idx: int):
        nonlocal point_cloud
        depth, img = prefetcher.get(idx)
        positions, colors = unproject_frame(depth, img, rhat, rows, cols, args.depth_min, args.depth_max, args.device)
        if point_cloud is None:
            point_cloud = server.scene.add_point_cloud(
                "/depth_pointcloud", points=positions, colors=colors, point_size=0.01,
            )
        else:
            # Real prop setters in viser>=1.0 -- sends a binary diff instead of
            # recreating the scene node, unlike the 0.2.7 no-op this replaced.
            point_cloud.points = positions
            point_cloud.colors = colors

    show_frame(0)

    @gui_frame.on_update
    def _(_):
        show_frame(gui_frame.value)

    @gui_up_from_view.on_click
    def _(event: viser.GuiEvent) -> None:
        # Re-level the horizon to whatever the user is currently looking at: take the camera's own
        # "up" (-Y in its local frame) as the world up. The escape hatch when the pose convention
        # or the scene's real gravity direction doesn't match the first-camera guess.
        if event.client is None:
            return
        event.client.camera.up_direction = vtf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    print(f"Serving at http://localhost:{args.port} -- open in a browser, drag the Frame slider or hit Play.")
    t_start = None
    frame0 = 0

    @gui_fps.on_update
    def _(_):
        # Reset the timing anchor so a mid-playback fps change takes effect immediately
        # without the frame index jumping (elapsed-time-since-anchor would otherwise be
        # reinterpreted at the new period on the very next loop iteration).
        nonlocal t_start, frame0
        if t_start is not None:
            t_start = time.monotonic()
            frame0 = gui_frame.value

    while True:
        if gui_play.value:
            period = 1.0 / gui_fps.value
            if t_start is None:
                t_start = time.monotonic()
                frame0 = gui_frame.value
            # Drive the frame index off wall-clock elapsed time rather than a fixed +1 step,
            # so slow frames (large stride, cold prefetch cache) drop frames to stay at
            # real-time fps instead of silently playing back slower than requested.
            target = frame0 + int((time.monotonic() - t_start) / period)
            gui_frame.value = target % n_frames
            time.sleep(max(0.0, period - (time.monotonic() - t_start) % period))
        else:
            t_start = None
            time.sleep(0.05)


if __name__ == "__main__":
    main()
