import bisect
import contextlib
import csv
import gc
import os
import resource
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# Non-portable boundary: sam3 is the vendored third_party/sam3 fork, repo-specific.
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

from .core import SPAG4D, ConversionResult
from ..motion.detect_opticalflow import WAFTWrapper, abs_to_rel_coords
from ..depth.flow_depth_propagation import (
    composite_bg_locked,
    composite_bg_locked_torch,
    compute_bidirectional_flow,
    fb_consistency_error,
    feather_dynamic_mask,
    upscale_flow,
)
from ..geometry.ply_writer import save_ply_gsplat
from ..progress import log_tqdm as tqdm
from ..analysis.scene_analysis import compute_scene_defaults


def reencode_h264(src: str) -> str:
    import subprocess

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

def print_gpu_stats(label: str = "", file=None):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("=" * 50, file=file)
        print(
            f"[GPU{f' {label}' if label else ''}] "
            f"allocated: {allocated:.2f}GB (max: {max_allocated:.2f}GB) | "
            f"reserved: {reserved:.2f}GB (max: {max_reserved:.2f}GB) | "
            f"total: {total:.2f}GB",
            file=file
        )
        print("=" * 50, file=file)

def extract_video_frames(video_path: str, output_folder:str, skip_step: int = 10, export: bool = False) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_kept = (total_frames + skip_step - 1) // skip_step  # ceil division
    frames = np.empty((n_kept, H, W, 3), dtype=np.uint8)

    meta = {
        "W": W,
        "H": H,
        "fps":    int(cap.get(cv2.CAP_PROP_FPS)),
        "total":  total_frames,
    }

    if export:
        frames_folder = output_folder / "images"
        frames_folder.mkdir(parents=True, exist_ok=True)
        # USED TO CREATE SHORTER VIDEOS FOR DEBUG/TEST
        # video_writer = cv2.VideoWriter("temp_fast_track.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))

    idx = 0
    try:
        with tqdm(total=total_frames, desc="Extracting frames", unit=" frame") as pbar:
            frame_num = 0
            while True:
                # ── frames we KEEP: full decode ──────────────────────────────
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if export:
                    cv2.imwrite(str(frames_folder / f"frame_{idx:04d}.png"), frame_bgr)
                    # video_writer.write(frame_bgr)
                frames[idx] = frame_bgr


                idx += 1
                pbar.update(1)
                frame_num += 1

                # ── frames we SKIP: grab only (no decode) ───────────────────
                for _ in range(skip_step - 1):
                    if not cap.grab():
                        break
                    pbar.update(1)
                    frame_num += 1
    finally:
        cap.release()
        # video_writer.release()
        # reencode_h264('temp_fast_track.mp4')
    if idx == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frames[:idx], meta  # slice in case video ended early


def _activity_mask_from_std(std_frame: np.ndarray, abs_threshold: float, fallback_frac: float = 0.01) -> np.ndarray:
    """Absolute std threshold for "real" frame-to-frame activity.

    A quantile/rank-based cut always flags exactly `quantile` of the pixels
    as "active" no matter the content (empirically confirmed: same fraction
    on a near-static warehouse clip and a clip with real moving subjects —
    see .claude/TEMPORAL_STABILITY_SUMMARY.md). An absolute threshold lets a
    genuinely static scene end up with ~0% active. Falls back to the top
    `fallback_frac` of pixels (by std) if the absolute threshold clears
    fewer than that, so the mask is never completely empty.
    """
    mask = std_frame > abs_threshold
    if mask.mean() < fallback_frac:
        thr = np.quantile(std_frame, 1.0 - fallback_frac)
        mask = std_frame > thr
    return mask


def compute_temporal_median(
    frames_array, masks_dict: list[dict], activity_std_threshold: float = 10.0, skip_step: int = 1,
    diag_dir: "Path | None" = None, min_sample_frames: int = 0,
):
    print("Calcul de la médiane temporelle (cela peut prendre quelques secondes)...")
    start = time.time()
    # Stack once — if viz_frames is already an ndarray, this is a no-op
    if not isinstance(frames_array, np.ndarray):
        frames_array = np.stack(frames_array, axis=0)  # avoids extra copy vs np.array()

    # Calcul de la médiane sur l'axe 0 (l'axe du temps)
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    F, H, W, C = frames_array.shape
    masks_array = np.zeros((F, H, W), dtype=bool)
    for idx in range(F):
        native_idx = idx * skip_step  # masks_dict is keyed by native (non-decimated) frame index
        if native_idx in masks_dict:
            for _, mask in masks_dict[native_idx].items():
                # Guard against malformed SAM masks (some videos yield a
                # degenerate 1D array instead of an (H, W) mask -> broadcast error)
                mask_shape = getattr(mask, "shape", None)
                if mask_shape is None or len(mask_shape) != 2:
                    continue
                if mask_shape != (H, W):
                    # outputs_per_frame/masks_dict is kept at SAM3's working
                    # resolution (not eagerly upsampled -- see
                    # docs/RAM_USAGE_INVESTIGATION_PLAN.md); upsample lazily
                    # here too, same INTER_NEAREST as the old eager path.
                    mask_u8 = mask.astype(np.uint8) if mask.dtype == bool else mask
                    mask = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                masks_array[idx] = np.maximum(masks_array[idx], mask)

    median_frame = np.zeros((H, W, C), dtype=frames_array.dtype)
    median_frame[:, :, 0] = 1.0
    std_frame = np.zeros((H, W), dtype=np.float32)
    # Row-chunked: converting a whole channel (F,H,W) to float32 in one shot,
    # then .clone()-ing it for the NaN-median pass, holds ~2 full-clip float32
    # buffers live at once per channel (~6GB + ~6GB on a 204-frame 3840x1920
    # clip) -- and since PyTorch's CPU allocator never returns freed RSS to
    # the OS, each channel's peak becomes a permanent floor for the rest of
    # the process (same failure mode fixed for viz_tensor in the flow-setup
    # block above; see docs/RAM_USAGE_INVESTIGATION_PLAN.md). Median/std are
    # independent per-pixel across H, so chunk over rows instead of holding
    # the whole (F,H,W) channel live at once.
    _row_chunk = max(1, 256 * 1024 * 1024 // max(1, F * W * 4))  # ~256MB per chunk buffer
    for c in range(C):  # Channel by channel to reduce VRAM usage
        for r0 in range(0, H, _row_chunk):
            r1 = min(r0 + _row_chunk, H)
            channel_tensor = torch.from_numpy(frames_array[:, r0:r1, :, c]).float().to(device)
            median_calc_tensor = channel_tensor.clone()
            median_calc_tensor[masks_array[:, r0:r1, :]] = float("nan")
            median_frame[r0:r1, :, c] = torch.nanmedian(median_calc_tensor, dim=0).values.cpu().numpy()
            std_frame[r0:r1, :] += torch.std(channel_tensor, dim=0).cpu().numpy()
            del channel_tensor, median_calc_tensor

    sample_count = (F - masks_array.sum(axis=0)).astype(np.int32)  # (H, W): valid (unmasked) frames per pixel
    always_masked = masks_array.all(axis=0)  # (H, W): masked in every single frame
    # §5/bglock_open_questions.md: a pixel with a *few* unmasked samples isn't "always masked"
    # but its median is a near-arbitrary pick from 1-4 frames, not a stable background estimate --
    # same downstream failure mode as the zero-sample case (a bad-depth halo locked in for the
    # whole video), just less severe. min_sample_frames (SPAG_DREF_MIN_SAMPLES) extends the same
    # undefined/inpaint/NaN treatment to thin-coverage pixels. Default 0 preserves old behavior
    # (only zero-sample pixels are undefined).
    undefined = always_masked | (sample_count < min_sample_frames) if min_sample_frames > 0 else always_masked
    if undefined.any():
        # A pixel masked in every frame has no valid median sample; it lands here as
        # solid black (0,0,0) -- either via nan_to_num or the implicit float->uint8 cast
        # above. Depth models have a receptive field much larger than one pixel and
        # hallucinate wildly wrong depth in and around a hard black hole (visible as a
        # bad-depth halo bleeding onto real, non-masked neighbors under bglock, since
        # depth_ref is computed once from this image and locked for the whole video).
        # Inpaint instead so the depth model sees plausible texture, not a hole.
        median_frame = cv2.inpaint(median_frame, (undefined.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
    else:
        median_frame = np.nan_to_num(median_frame, nan=0.0)

    # On peut aussi enregistrer les pixels qui varient le plus pour visualiser les zones les plus dynamiques
    std_frame /= C
    activity_mask = (_activity_mask_from_std(std_frame, activity_std_threshold) * 255).astype(np.uint8)
    print(f"abs threshold: {activity_std_threshold} -> active frac {(activity_mask > 0).mean():.4f}")

    print(f"Median computed in {(time.time() - start):.1f}s")

    # §9 diagnostics pass (SPAG_DIAG_CSV): D_ref sample-count map, saved once per
    # run. The median RGB plate itself is already saved unconditionally by the
    # caller as _temporal_median_background_masked.jpg -- no need to duplicate it.
    if diag_dir is not None:
        np.save(Path(diag_dir) / "diag_dref_sample_count.npy", sample_count)

    if undefined.any():
        print(f"[D_ref] {int(undefined.sum())} px undefined (zero-sample: {int(always_masked.sum())}, "
              f"thin-coverage <{min_sample_frames}: {int((~always_masked & undefined).sum())})")

    return median_frame, activity_mask, undefined


def to_gaussians(
    converter: SPAG4D,
    aligned_depth_np,
    image_tensor,
    stride: int,
    H: int,
    depth_min: float | None = None,
    depth_max: float | None = None,
    sky_threshold: float | None = None,
    outlier_pruning: float = 0.1,
    grazing_angle: float = 85.0,  # 65.0,
    sparse_pruning: float = 0.1,  # 0.3,
):
    # C1 FIX: Use provided defaults; don't recalculate per-frame
    if depth_min is None or depth_max is None or sky_threshold is None:
        scene_defaults = compute_scene_defaults(aligned_depth_np, image_height=H)
        if depth_min is None:
            depth_min = scene_defaults["depth_min"]
        if depth_max is None:
            depth_max = scene_defaults["depth_max"]
        if sky_threshold is None:
            sky_threshold = scene_defaults["sky_threshold"]

    gaussians = converter._run_spag_pipeline(
        image_tensor,
        torch.from_numpy(aligned_depth_np),
        depth_min=depth_min,
        depth_max=depth_max,
        sky_threshold=sky_threshold,
        stride=stride,
    )

    # Post-generation filters
    if outlier_pruning > 0.0 and gaussians["means"].shape[0] > 0:
        try:
            from .scene_filter import prune_outliers

            gaussians = prune_outliers(gaussians, strength=outlier_pruning)
        except Exception as e:
            import warnings

            warnings.warn(f"Outlier pruning failed: {e}")

    if grazing_angle < 90.0 and gaussians["means"].shape[0] > 0:
        try:
            from .scene_filter import prune_grazing_angle

            depth_np = (
                aligned_depth_np.detach().cpu().numpy()
                if hasattr(aligned_depth_np, "detach")
                else aligned_depth_np
            )
            gaussians = prune_grazing_angle(
                gaussians,
                depth_np,
                stride=stride,
                max_angle_deg=grazing_angle,
            )
        except Exception as e:
            import warnings

            warnings.warn(f"Grazing angle filter failed: {e}")

    if sparse_pruning > 0.0 and gaussians["means"].shape[0] > 0:
        try:
            from .scene_filter import prune_sparse_regions

            # Map strength 0-1 to min_neighbors 1-6
            min_n = max(1, int(1 + sparse_pruning * 5))
            gaussians = prune_sparse_regions(
                gaussians,
                min_neighbors=min_n,
                radius_multiplier=3.0,
            )
        except Exception as e:
            import warnings

            warnings.warn(f"Sparse region pruning failed: {e}")

    return gaussians

def run_video(
    converter: SPAG4D,
    video_path: str | Path,
    output_path: str | Path,
    active_generator: str = "da360",
    get_background_method: str = "temporal_median",
    alignement_mask: str = "sam_and_activity",  # sam, sam_and_activity, nothing, all
    alignement_method: str = "lstsq",  # lstsq, median, ransac
    activity_std_threshold: float = 10.0,
    skip_step: int = 10,
    freeze_bg: bool = False,
    freeze_bg_live_color: bool = True,
    depth_min: float | None = None,
    depth_max: float | None = None,
    sky_threshold: float | None = None,
    stride: int = 8,
    outlier_pruning: float = 0.1,
    grazing_angle: float = 85.0,  # 65.0,
    global_scale: float = 1.0,
    sparse_pruning: float = 0.1,  # 0.3,
    depth_preview_path: Path | None = None,
    depth_npy_dir: Path | str | None = None,
    temporal_consistency: bool = False,
    # --- Solution 1: temporal depth smoothing ---
    depth_smoothing: bool = False,
    depth_smoothing_window: int = 5,
    depth_smoothing_method: str = "median",  # "median" | "gaussian"
    # --- Solution 6: foreground stabilizer tuning ---
    fg_buffer_size: int = 21,
    fg_jump_threshold: float = 0.5,
    # --- Solution 5: mask quality diagnostics (diagnostic only) ---
    mask_diagnostics: bool = False,
    # --- background-locked compositing + flow propagation (fixed-camera ERP) ---
    depth_correction: str = "bglock",  # "affine" (legacy per-frame align) | "bglock" (background-locked to depth_ref + flow-propagated dynamic-mask objects; see .claude/depth_stability_benchmark.md)
    bg_lock_dilate_px: int = 12,
    bg_lock_feather_px: int = 9,
    flow_seam_pad: int = 64,
    unisharp_repo: str | None = "/raid/mb273924/SPAG4d/third_party/UniSHARP",
    unisharp_python: str | None = None,
    unisharp_checkpoint: str | None = "/raid/mb273924/SPAG4d/third_party/UniSHARP/pretained_model.pt",
    unisharp_scale_align: str = "global",
    unisharp_format_mode: str = "convert",
    unisharp_max_gaussians: int | None = None,
):
    """

    Args :
        get_background_method : determine background images use as anchor. Options:
        'first': use first frame (default)
        'last' : use last frame
        'median_temporal': use average color of each frame as background color
        depth_correction : per-frame depth stabilization strategy for a FIXED
            camera. 'bglock' locks static (non-mask) pixels to depth_ref (the
            temporal-median-background depth, ~zero temporal variance) and
            uses the SAM3/activity dynamic mask (fused_mask) only to blend in
            flow-propagated object depth — validated to cut background
            temporal std ~1000x vs 'affine' with lower foreground flicker too.
            'affine' keeps the legacy per-frame affine-alignment behavior.
        freeze_bg_live_color : default True (only takes effect with
            freeze_bg=True, otherwise a no-op). The frozen background
            layer's GEOMETRY still comes from the one-time temporal-median
            build (gaussians_bg), but its COLOR is refreshed every frame
            from the live image wherever that background pixel is
            currently visible (sam_mask==0), falling back to the frozen
            median color where a dynamic object currently covers it.
            Fixes frozen_bg's dead appearance on shadows/screens without
            reintroducing bglock-alone's trailing-edge occlusion holes.
            Validated 2026-08-21 on circulation_site_1_edit_coupe (ground
            shadows) and boutique1_HQ (flat screens, mixed SAM-tracked/
            untracked): background block color drifts frame-to-frame as
            intended while frozen_bg alone stays byte-identical, at
            ~0-40s / ~250-300MB VRAM overhead.
        depth_npy_dir : optional directory to dump the raw per-frame metric
            depth map (the exact array passed to to_gaussians, i.e. after
            alignment/bglock compositing/freeze_bg masking) as float32 .npy,
            one file per frame ("depth_{idx}.npy"). Unlike depth_preview_path
            (which only writes a log1p + percentile-normalized colormap JPEG
            for visualization -- see SPAG4D._save_depth_preview -- and cannot
            be inverted back to metric depth), this preserves real depth
            values so the map can be used directly as an alternative Gaussian
            init source (e.g. isotropic per-pixel splats sized from depth,
            instead of the anisotropic SPAG-fitted scale/rotation which tends
            to produce view-aligned "needle" Gaussians for a fixed camera).
            Also dumps the SAM mask ("mask_{idx}.npy", uint8) and, on the
            bglock flow path, the dense forward WAFT flow ("flow_{idx}.npy",
            HxWx2 float32, native res, for idx>=1) for downstream velocity
            supervision -- see .claude/D1_FUTURE_WORK_PLAN.md.
    """
    if depth_correction not in ("affine", "bglock"):
        raise ValueError(f"depth_correction must be 'affine' or 'bglock', got {depth_correction!r}")
    if freeze_bg_live_color and not freeze_bg:
        raise ValueError("freeze_bg_live_color requires freeze_bg=True (it only refreshes the frozen background layer's color)")
    if depth_correction == "bglock" and alignement_mask not in ("sam", "sam_and_activity"):
        raise ValueError(
            "depth_correction='bglock' requires a real dynamic mask: "
            "alignement_mask must be 'sam' or 'sam_and_activity'"
        )
    if alignement_mask == "sam_and_activity":
        get_background_method = "temporal_median"

    # §9 diagnostics pass (bglock_open_questions.md): SPAG_DIAG_CSV=<path.csv> turns
    # on one CSV row per frame (affine residual, fitted scale, static-pixel flow,
    # running_confidence summary, per-region depth deltas, per-object flow
    # magnitude, residual-vs-latitude) plus two one-off files next to the CSV
    # (diag_dref_sample_count.npy, saved from compute_temporal_median). Rule 13:
    # entirely inert when unset -- no new work, no new allocations on the hot path.
    diag_csv_path = os.environ.get("SPAG_DIAG_CSV")
    diag_dir = Path(diag_csv_path).parent if diag_csv_path else None
    if diag_dir is not None:
        diag_dir.mkdir(parents=True, exist_ok=True)

    # Optional determinism (opt-in, default OFF). Production keeps cuDNN autotune
    # for throughput; enable for benchmarking/gating where reproducibility matters
    # more than speed. Motivation: fg_depth_cv swung 3.75x between identical
    # reruns (B1 MattSwift) since nothing seeds RNG/cuDNN and the marginal
    # found_object contour threshold flips which frames register a track. See D1
    # P2. Deliberately NOT torch.use_deterministic_algorithms(True): it raises on
    # WAFT/SAM3 ops lacking deterministic kernels; seeds + cudnn.deterministic is
    # the safe subset that stabilizes this pipeline's nondeterminism.
    if os.environ.get("SPAG_DETERMINISTIC", "0") == "1":
        seed = int(os.environ.get("SPAG_SEED", "42"))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[SPAG4D] Deterministic mode ON (seed={seed}, cudnn.benchmark=False)")

    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    step_times = {}  # phase_name -> wall seconds, printed as a summary table at the end

    ram_trace = os.environ.get("SPAG_RAM_TRACE", "0") == "1"
    ram_trace_path = output_folder / "ram_trace.csv"
    if ram_trace:
        with open(ram_trace_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["checkpoint", "ru_maxrss_mb", "tracemalloc_current_mb", "tracemalloc_peak_mb", "top_growth"]
            )

    def _ram_checkpoint(label: str, prev_snapshot=None):
        if not ram_trace:
            return prev_snapshot
        cur_mb, peak_mb = (v / (1024 * 1024) for v in tracemalloc.get_traced_memory())
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        snapshot = tracemalloc.take_snapshot()
        top_growth = ""
        if prev_snapshot is not None:
            diffs = snapshot.compare_to(prev_snapshot, "lineno")[:3]
            top_growth = " | ".join(str(d) for d in diffs)
        with open(ram_trace_path, "a", newline="") as f:
            csv.writer(f).writerow([label, f"{rss_mb:.1f}", f"{cur_mb:.1f}", f"{peak_mb:.1f}", top_growth])
        return snapshot

    if ram_trace:
        tracemalloc.start()
    _ram_trace_prev_snapshot = _ram_checkpoint("start")

    t0 = time.time()
    viz_frames, meta = extract_video_frames(str(video_path), output_folder, skip_step=skip_step, export=True)
    step_times["frame_extraction"] = time.time() - t0
    n_total_frames = len(viz_frames)
    _ram_trace_prev_snapshot = _ram_checkpoint("after_frame_extraction", _ram_trace_prev_snapshot)

    # Flow
    # Defined up-front so a WAFT failure falls back to SAM cleanly instead of
    # raising UnboundLocalError on `if flow_masks is None` below.
    flow_masks = None

    if active_generator == "unisharp360":
        # UniSHARP is a whole-frame external subprocess pipeline (no raw depth
        # map exposed in-process), so none of the SAM3/WAFT/depth-compositing
        # machinery below (incl. depth_correction/bglock) applies here — it
        # runs one full 3DGS reconstruction per frame instead of a depth map
        # to composite. Kept as a separate early branch rather than forcing it
        # through the per-frame depth loop it's architecturally incompatible with.
        import tempfile

        from .unisharp360 import convert_unisharp360

        gaussians_dir = output_folder / "gaussians"
        gaussians_dir.mkdir(parents=True, exist_ok=True)
        n_gaussians = 0
        with tempfile.TemporaryDirectory(prefix="spag4d_unisharp_frames_") as tmp_dir:
            for idx, frame in enumerate(tqdm(viz_frames, desc="[UniSHARP]")):
                frame_path = str(Path(tmp_dir) / f"frame_{idx}.jpg")
                cv2.imwrite(frame_path, frame)  # frame is already BGR (cv2 convention)
                result = convert_unisharp360(
                    input_path=frame_path,
                    output_path=str(gaussians_dir / f"frame_{idx}.ply"),
                    device=torch.device("cuda"),
                    unisharp_repo=unisharp_repo,
                    unisharp_python=unisharp_python,
                    checkpoint_path=unisharp_checkpoint,
                    scale_align=unisharp_scale_align,
                    format_mode=unisharp_format_mode,
                )
                n_gaussians = result["num_gaussians"]
                if unisharp_max_gaussians:
                    from .unisharp_format import subsample_ply

                    sub = subsample_ply(
                        str(gaussians_dir / f"frame_{idx}.ply"),
                        unisharp_max_gaussians,
                        seed=idx,
                    )
                    n_gaussians = sub["num_gaussians"]

        processing_time = time.time() - start_time
        file_size = (gaussians_dir / f"frame_{n_total_frames - 1}.ply").stat().st_size
        return ConversionResult(
            output_path=str(gaussians_dir / f"frame_{n_total_frames - 1}.ply"),
            splat_count=n_gaussians,
            file_size=file_size,
            processing_time=processing_time,
            depth_range=(0.0, 0.0),
            depth_npy_path=None,
            panorama_size=(meta["W"], meta["H"]),
        )

    # Flow
    flows_fwd, flows_bwd = [], []
    t_flow0 = time.time()
    try:
        model = WAFTWrapper(
            checkpoint = '/raid/mb273924/_DATASETS/uptale/tar-c-t.pth',
            config = '/raid/mb273924/WAFT/config/a1/tar-c-t.json'
        )
        _ram_trace_prev_snapshot = _ram_checkpoint("after_waft_init", _ram_trace_prev_snapshot)
        MAX_SIZE = 1024
        scale = min(MAX_SIZE / max(meta["W"], meta["H"]), 1.0)  # never upscale
        if scale < 1.0:
            new_W = int(meta["W"] * scale) & ~1  # force even dimensions (codec requirement)
            new_H = int(meta["H"] * scale) & ~1

            # Chunked downscale: converting the whole clip to float32 in one
            # torch.from_numpy(...).float() call materializes a CPU tensor
            # sized frames*H*W*3*4 bytes (~2.9GB for a 39-frame 3840x1920
            # clip, ~15GB for a 204-frame one) -- and PyTorch's CPU allocator
            # never returns that RSS to the OS once touched (confirmed: an
            # explicit del right after does not reclaim it), so it becomes a
            # permanent floor for the rest of the process. Chunking bounds
            # the largest simultaneously-live float32 buffer to one chunk's
            # worth instead of the whole clip's. Measured: -17.9% peak RAM on
            # a 39-frame clip, -28.7% on a 204-frame clip, no time/VRAM cost
            # at chunk=8 (see docs/RAM_USAGE_INVESTIGATION_PLAN.md).
            _chunk = 8
            waft_chunks = []
            for _start in range(0, len(viz_frames), _chunk):
                _batch = torch.from_numpy(viz_frames[_start:_start + _chunk]).permute(0, 3, 1, 2).float()
                _batch = F.interpolate(_batch, size=(new_H, new_W), mode="bilinear")
                waft_chunks.append(_batch.permute(0, 2, 3, 1).byte().numpy())
                del _batch
            waft_frames = np.concatenate(waft_chunks, axis=0)
            del waft_chunks
            _ram_trace_prev_snapshot = _ram_checkpoint("after_waft_frames_numpy", _ram_trace_prev_snapshot)
        else:
            new_W, new_H = meta["W"], meta["H"]
            waft_frames = viz_frames
        # Toujours défini : segment_with_flows en dépend même sans downscale
        meta["new_W"] = new_W
        meta["new_H"] = new_H

        # Tier 3: single WAFT pass (opt-in SPAG_SINGLE_PASS=1). Derives the SAM
        # mask from the same unified flow instead of running WAFT twice (~38%
        # time win). NOT lossless: reuses seam_pad=0 flow for fg depth propagation
        # too, dropping ERP seam-wraparound correction (root cause of scene01's
        # stabilized_fg_median divergence). Don't flip default; see B1 doc.
        single_pass = (
            depth_correction == "bglock"
            and os.environ.get("SPAG_SINGLE_PASS", "0") == "1"
        )
        if single_pass:
            # seam_pad=0 default keeps the SAM mask byte-identical to baseline
            # (same infer_pair); the cost is dropped seam-crossing propagation,
            # which bg-lock doesn't rely on.
            sp_seam = int(os.environ.get("SPAG_SP_SEAMPAD", "0"))
            # P4 conditional seam-padding -- still the only unexplored avenue
            # for single_pass's seam regression, see D1 Priority 4 / C1.
            cond_seam = (
                os.environ.get("SPAG_SP_COND_SEAMPAD", "0") == "1"
                and sp_seam != flow_seam_pad
            )
            seam_band = flow_seam_pad  # px from each vertical edge (at waft res)
            Hw, Ww = waft_frames.shape[1], waft_frames.shape[2]
            flow_masks = np.zeros((len(waft_frames), Hw, Ww), dtype=np.uint8)
            print(f"[SPAG4D] Single-pass WAFT (seam_pad={sp_seam}, "
                  f"cond_seampad={cond_seam}): bidirectional flow + derived SAM mask...")
            t0 = time.time()
            n_seam_recompute = 0
            for i in range(n_total_frames - 1):
                ff, fb = compute_bidirectional_flow(model, waft_frames[i], waft_frames[i + 1], seam_pad=sp_seam)
                mag = np.sqrt(ff[..., 0] ** 2 + ff[..., 1] ** 2)
                mask_i = (mag > 0.5).astype(np.uint8) * 255
                flow_masks[i] = mask_i
                need_pad = False
                if cond_seam and seam_band > 0:
                    # moving foreground within `seam_band` px of either vertical edge?
                    if mask_i[:, :seam_band].any() or mask_i[:, -seam_band:].any():
                        need_pad = True
                        n_seam_recompute += 1
                if need_pad:
                    ff, fb = compute_bidirectional_flow(model, waft_frames[i], waft_frames[i + 1], seam_pad=flow_seam_pad)
                # Kept at WAFT working resolution -- see comment on the
                # default-path population below.
                flows_fwd.append(ff)
                flows_bwd.append(fb)
            n_pairs = n_total_frames - 1
            extra = f", {n_seam_recompute}/{n_pairs} seam-adjacent recomputes" if cond_seam else ""
            print(f"  {n_pairs} pairs (single pass{extra}) in {time.time() - t0:.1f}s")
        else:
            # Forward-only WAFT pass to build the SAM magnitude masks.
            _ram_trace_prev_snapshot = _ram_checkpoint("before_flow_maskpass", _ram_trace_prev_snapshot)
            stats = model.run(waft_frames, meta, output_folder, 0.5)
            _ram_trace_prev_snapshot = _ram_checkpoint("after_flow_maskpass", _ram_trace_prev_snapshot)
            flow_masks = stats["masks"]
            try:
                reencode_h264(str(output_folder / "flows.mp4"))
            except Exception:
                pass
            _ram_trace_prev_snapshot = _ram_checkpoint("after_reencode", _ram_trace_prev_snapshot)

            if depth_correction == "bglock":
                # Bidirectional flow (seam-padded, forward-backward consistency)
                # for propagate_depth_via_flow. model.run() above only kept flow
                # *magnitude masks* for SAM prompting, so this is a second WAFT
                # pass over the same (already-downscaled) waft_frames.
                print("[SPAG4D] Computing bidirectional WAFT flow for background-locked depth compositing...")
                t0 = time.time()
                # Kept at WAFT working resolution (not eagerly upscaled to
                # native) -- flows_fwd/flows_bwd is a full-clip buffer held
                # for the whole run because the per-frame loop does idx-1
                # random-access lookback into it; upscale_flow's native-res
                # resize was the dominant RAM contributor on real-world clips
                # (see docs/RAM_USAGE_INVESTIGATION_PLAN.md). Callers upscale
                # lazily, once per read, via upscale_flow (no-op if already
                # native res).
                _bidir_mid = (n_total_frames - 1) // 2
                for i in range(n_total_frames - 1):
                    ff, fb = compute_bidirectional_flow(model, waft_frames[i], waft_frames[i + 1], seam_pad=flow_seam_pad)
                    flows_fwd.append(ff)
                    flows_bwd.append(fb)
                    if ram_trace and i == _bidir_mid:
                        _ram_trace_prev_snapshot = _ram_checkpoint(f"bidir_mid_pair{i}", _ram_trace_prev_snapshot)
                print(f"  {n_total_frames - 1} pairs in {time.time() - t0:.1f}s")
        step_times["flow_waft"] = time.time() - t_flow0
        if torch.cuda.is_available():
            print(f"[VRAM] cumulative peak through flow phase (WAFT freed): "
                  f"{torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    except Exception as e:
        step_times["flow_waft"] = time.time() - t_flow0
        print(f"  [ERROR during inference] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release WAFT model
        del model
        gc.collect()
        torch.cuda.empty_cache()
    _ram_trace_prev_snapshot = _ram_checkpoint("after_flow_waft", _ram_trace_prev_snapshot)
    # bgr to rgb
    viz_frames = viz_frames[:, :, :, ::-1].copy()
    # SAM
    n_tracked_objects = None
    mean_objects_per_frame = None
    if alignement_mask in ["sam", "sam_and_activity", "sam_and_flow", "nothing"]:
        startsam = time.time()
        if flow_masks is None:
            outputs_per_frame = segment_with_sam(str(video_path), viz_frames, n_total_frames, skip_step)
        else:
            outputs_per_frame = segment_with_flows(str(video_path), output_folder, waft_frames, flow_masks, meta, n_total_frames, skip_step, flows_fwd=flows_fwd, flows_bwd=flows_bwd)
        step_times["sam3_segmentation"] = time.time() - startsam
        _ram_trace_prev_snapshot = _ram_checkpoint("after_sam3_segmentation", _ram_trace_prev_snapshot)
        print(f"[SPAG4D] Background/Front segmentation of video completed in {(time.time() - startsam):.2f}s")
        # outputs_per_frame[frame_idx] is {obj_id: mask} post prepare_masks_for_visualization.
        _all_obj_ids = set()
        _objs_per_frame_counts = []
        for _frame_output in outputs_per_frame.values():
            if isinstance(_frame_output, dict):
                _all_obj_ids.update(_frame_output.keys())
                _objs_per_frame_counts.append(len(_frame_output))
        n_tracked_objects = len(_all_obj_ids)
        mean_objects_per_frame = float(np.mean(_objs_per_frame_counts)) if _objs_per_frame_counts else 0.0
        print(f"[SPAG4D] SAM3 tracked {n_tracked_objects} distinct object id(s), "
              f"mean {mean_objects_per_frame:.1f} concurrent objects/frame")
        # Return SAM3's cached GPU blocks before the depth loop's peak.
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"[VRAM] cumulative peak through SAM3 segmentation: "
                  f"{torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    # Solution 5: mask quality diagnostics (diagnostic only, no pipeline change)
    if mask_diagnostics:
        try:
            from .analyze_mask_quality import analyze_mask_quality
            _fh, _fw = viz_frames.shape[1], viz_frames.shape[2]
            analyze_mask_quality(
                flow_masks=flow_masks,
                outputs_per_frame=outputs_per_frame,
                n_frames=n_total_frames,
                skip_step=skip_step,
                target_hw=(_fh, _fw),
                output_dir=output_folder / "mask_diagnostics",
            )
        except Exception as e:
            print(f"[MaskDiag] failed: {e}")

    # On récup un background fixe
    t0 = time.time()
    always_masked_mask = None
    if get_background_method == "temporal_median":
        # Opt-in / default off (§5, bglock_open_questions.md): pixels seen unmasked in fewer than
        # this many frames get the same undefined/inpaint/NaN treatment as zero-sample pixels --
        # their median is a near-arbitrary pick from a handful of frames, not a stable D_ref
        # estimate. Measured on a real clip: <2% of pixels fall below the doc's K~5-10 threshold,
        # and they cluster into one contiguous region rather than scattering across the frame.
        dref_min_samples = int(os.environ.get("SPAG_DREF_MIN_SAMPLES", "0"))
        master_background, activity_mask, always_masked_mask = compute_temporal_median(
            viz_frames, outputs_per_frame, activity_std_threshold, skip_step, diag_dir=diag_dir,
            min_sample_frames=dref_min_samples,
        )
        _ram_trace_prev_snapshot = _ram_checkpoint("after_temporal_median", _ram_trace_prev_snapshot)
        cv2.imwrite(output_folder / f"_temporal_activity_mask_abs{activity_std_threshold:g}.jpg", activity_mask)
    elif get_background_method == "last":
        master_background = viz_frames[-1]
    else:
        master_background = viz_frames[0]
    step_times["background_extraction"] = time.time() - t0

    # Efficient conversion to uint8 for saving
    if master_background.dtype != np.uint8:
        median_u8 = (master_background * 255).astype(np.uint8)
    else:
        median_u8 = master_background.astype(np.uint8)
    cv2.imwrite(output_folder / "_temporal_median_background_masked.jpg", cv2.cvtColor(median_u8, cv2.COLOR_RGB2BGR))

    # Get reference depth
    if active_generator in ("da360", "pager"):
        dm_name = active_generator
    else:
        dm_name = converter.default_depth_model
        print(f"Can't use {active_generator}. Only 'da360' and 'pager' are supported for video. Fallback to da360")
    depth_engine = converter._get_depth_model(dm_name)
    print(f"[SPAG4D] Running {dm_name.upper()} depth estimation...", flush=True)

    start_depthref = time.time()
    reference_bg = torch.from_numpy(master_background).to("cuda")
    with torch.inference_mode():
        depth_raw, _ = depth_engine.predict(reference_bg, temporal_consistency=temporal_consistency)

    depth_ref = depth_raw *  global_scale
    if temporal_consistency:
        # Calculate scale factor to normalize median depth to ~5m (single time, for all frames)
        depth_median = float(depth_raw.median())
        scale_factor_to_5m = 5.0 / depth_median if depth_median > 1e-6 else 1.0
        depth_ref *= scale_factor_to_5m

    # Save depth preview if requested
    if depth_preview_path:
        converter._save_depth_preview(
            depth_ref, str(depth_preview_path / "_masterdepth.jpg")
        )
    depth_ref_np = depth_ref.cpu().numpy()
    if always_masked_mask.any():
        # Mark pixels never seen unmasked as NaN
        # Forces filter_gaussian_candidates to drop them,
        # creating an honest hole in the scene instead of a fabricated surface.
        depth_ref_np[always_masked_mask] = np.nan
        print(f"[SPAG4D] {int(always_masked_mask.sum())} px excluded from depth_ref (no Gaussians will be generated there)")
    step_times["depth_ref"] = time.time() - start_depthref
    print(f"[SPAG4D] Depth estimation for master bg completed in {1000 * (time.time() - start_depthref):.2f} ms")

    # Calculate scene defaults once from reference depth (not per-frame)
    from .scene_analysis import compute_scene_defaults
    ref_defaults = compute_scene_defaults(depth_ref_np, image_height=reference_bg.shape[0])
    if depth_min is None:
        depth_min = ref_defaults["depth_min"]
    if depth_max is None:
        depth_max = ref_defaults["depth_max"]
    if sky_threshold is None:
        sky_threshold = ref_defaults["sky_threshold"]

    W, H, _ = reference_bg.shape
    gaussians_bg = to_gaussians(
        converter,
        depth_ref_np,
        reference_bg,
        stride,
        H,
        depth_min,
        depth_max,
        sky_threshold,
        outlier_pruning,
        grazing_angle,
        sparse_pruning,
    )
    n_bg_points = int(gaussians_bg["means"].shape[0])

    output_path = str(output_path)
    depth_estimation_cumtime = 0
    depth_alignement_cumtime = 0
    gs_generation_cumtime = 0
    aligned_depth_list_cpu = []

    # §4.3 metrics-side exclusion (benchmarks/BENCHMARK_RULES.md rule-12 TODO): the
    # same magnitude gate used for the affine fit, applied to the per-frame medians
    # that feed bg_depth_cv/fg_depth_cv. depth_ref_np is the frozen plate, so this
    # cap is computed once and reused for every frame.
    metrics_outlier_cap = _fit_outlier_cap(depth_ref_np)

    median_depth_fg = []
    aligned_median_depth_fg = []
    stabilized_aligned_median_depth_fg = []

    # Background depth metrics (NEW)
    aligned_median_depth_bg = []
    background_depth_deltas = []
    foreground_depth_deltas = []

    from collections import deque
    class FGDepthStabilizer2:
        """
        Baseline = médiane glissante sur les N dernières valeurs FG brutes.

        ", "Robuste automatiquement aux pics < N/2 frames (la médiane les ignore)
        ", "S'adapte automatiquement aux changements >= N/2 frames soutenus
                (la médiane finit par "suivre" le buffer, sans logique de resync)
        ", "Pas d'EMA, pas de compteur, pas d'état "capturable"
        """

        def __init__(
            self,
            buffer_size: int = 11,  # >= 2x la durée max attendue des pics + 1
            jump_threshold: float = 0.5,
            scale_clip: tuple = (0.5, 2.0),
        ):
            self.buffer = deque(maxlen=buffer_size)
            self.jump_threshold = jump_threshold
            self.scale_clip = scale_clip

        def __call__(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
            fg = mask > 0
            if fg.sum() == 0:
                return depth

            current = float(np.median(depth[fg]))

            # Phase de warm-up : pas assez d'historique pour une médiane fiable
            if len(self.buffer) < self.buffer.maxlen:
                self.buffer.append(current)
                return depth

            baseline = float(np.median(self.buffer))

            # Toujours ajouter la valeur BRUTE (pas corrigée) au buffer
            self.buffer.append(current)

            delta = abs(current - baseline)
            if delta <= self.jump_threshold or current <= 0:
                return depth
            scale = np.clip(baseline / current, *self.scale_clip)
            out = depth.copy()
            out[fg] *= scale
            return out

    # Solution 6: FG stabilizer parameters are now configurable (tuning benchmark).
    fg_stabilizer = FGDepthStabilizer2(
        buffer_size=fg_buffer_size, jump_threshold=fg_jump_threshold
    )

    class TemporalDepthSmoother:
        """
        Solution 1: temporal depth smoothing.

        Causal per-pixel smoothing of aligned depth maps over a sliding window
        of the last `window` frames. Reduces frame-to-frame jitter on regions
        whose depth should be steady, at the cost of a small lag on real motion.

        method:
            "median"   -> per-pixel median over the window (robust to spikes)
            "gaussian" -> per-pixel Gaussian-weighted average (recent frames
                          weighted highest)
        """

        def __init__(self, window: int = 5, method: str = "median"):
            self.window = max(1, int(window))
            self.method = method
            self.buffer = deque(maxlen=self.window)
            self.buffer_t = deque(maxlen=self.window)  # GPU tensors, for call_torch
            if method == "gaussian":
                sigma = max(self.window / 6.0, 1e-3)
                # weights over the causal window; index 0 = oldest, -1 = current
                offsets = np.arange(self.window - 1, -1, -1, dtype=np.float64)
                w = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
                self._full_weights = w / w.sum()

        def __call__(self, depth: np.ndarray) -> np.ndarray:
            self.buffer.append(depth)
            if len(self.buffer) == 1:
                return depth
            stack = np.stack(self.buffer, axis=0)  # [k, H, W]
            # np.median / np.tensordot promote to float64; cast back to float32 so
            # the downstream GPU tensors (to_gaussians) stay fp32 (VRAM + fp64 math).
            if self.method == "median":
                # Exact per-pixel median via partition (kth-selection) instead of
                # np.median's full sort — identical result, much faster for the
                # small causal window. Odd k: the middle order statistic; even k:
                # mean of the two middle ones (matches np.median).
                k = stack.shape[0]
                if k % 2 == 1:
                    med = np.partition(stack, k // 2, axis=0)[k // 2]
                else:
                    lo = k // 2 - 1
                    part = np.partition(stack, [lo, lo + 1], axis=0)
                    med = (part[lo] + part[lo + 1]) / 2.0
                return med.astype(np.float32, copy=False)
            # gaussian: use the tail of the full weight vector matching buffer len
            k = stack.shape[0]
            w = self._full_weights[-k:]
            w = w / w.sum()
            return np.tensordot(w, stack, axes=([0], [0])).astype(np.float32, copy=False)

        def call_torch(self, depth: "torch.Tensor") -> "torch.Tensor":
            """GPU/torch equivalent of __call__ for method="median": keeps the
            causal window as resident cuda tensors (no np.stack host copy) and
            takes the per-pixel median via kthvalue selection — matches
            np.median (odd k: middle order statistic; even k: mean of the two
            middle) without torch.quantile's 2^24-element size cap."""
            self.buffer_t.append(depth)
            if len(self.buffer_t) == 1:
                return depth
            stack = torch.stack(list(self.buffer_t), dim=0)  # (k,H,W) cuda
            k = stack.shape[0]
            if k % 2 == 1:
                med = stack.kthvalue(k // 2 + 1, dim=0).values
            else:
                lo = stack.kthvalue(k // 2, dim=0).values
                hi = stack.kthvalue(k // 2 + 1, dim=0).values
                med = (lo + hi) / 2.0
            return med.to(depth.dtype)

    # Under bglock the smoother's output only survives where the composite alpha > 0
    # (inside the SAM mask + feather band); background pixels are overwritten by
    # depth_ref regardless. So it cannot stabilize the background here, and its only
    # remaining effect is smearing the moving silhouette: the per-pixel median over a
    # causal window resolves to the *background* depth for pixels the subject has just
    # moved onto, writing a ramp of background-depth splats inside the mask.
    _smoothing_active = depth_smoothing and depth_correction != "bglock"
    depth_smoother = (
        TemporalDepthSmoother(depth_smoothing_window, depth_smoothing_method)
        if _smoothing_active
        else None
    )
    if depth_smoothing and not _smoothing_active:
        print("[SPAG4D] depth_smoothing disabled under depth_correction=bglock "
              "(cannot affect the locked background; smears dynamic-object edges)")

    depth_prev_final = None  # bg-locked mode: previous frame's composited depth, warped forward each step
    # SPAG_BGLOCK_NOFLOW=1: skip flow-propagation for the dynamic-object depth entirely.
    # Instead of warping depth_prev_final along the WAFT flow field, just blend the
    # previous frame's composited depth with this frame's own affine-aligned depth,
    # pixel-for-pixel (no per-pixel correspondence). Benchmarked 2026-09-04 against
    # flow-warp propagation on 2 motion clips (atelier_1, accident_electrique_02):
    # this blend was faster (-14/-19%) and gave better fg_depth_cv (-20/-38%), no
    # bg cost -- flow-warp interpolation was making DA360's object-edge halo worse,
    # not helping it. Flow-warp propagation (propagate_depth_via_flow[_torch],
    # PropagationState, SPAG_CONF_*/SPAG_FLOW_EDGE_*/SPAG_COMPOSITE_LOWRES) removed
    # accordingly. See docs/SPAG_ENV_FLAGS.md.
    _bglock_noflow_blend = float(os.environ.get("SPAG_BGLOCK_NOFLOW_BLEND", "0.5"))

    # GPU-resident depth chain: for the winner hot path (bglock + lstsq align +
    # median/none smoother) run align and compositing on the GPU (depth is already
    # there from DA360), avoiding the per-frame numpy<->cuda round-trips.
    # Numerically within float rounding of the numpy path, not byte-identical.
    # Any other config falls back to the numpy path below.
    gpu_depth_chain = (
        depth_correction == "bglock"
        and alignement_mask not in ("nothing", "all")
        and alignement_method == "lstsq"
        and (depth_smoother is None or depth_smoother.method == "median")
        and torch.cuda.is_available()
    )
    depth_ref_t = torch.from_numpy(depth_ref_np).to("cuda") if gpu_depth_chain else None
    depth_prev_t = None  # previous frame's composited depth as a cuda tensor
    # Keep the final bg-locked composite (feather + blend) on the GPU too.
    # Default on: near-lossless (bg_cv/spikes identical; fg_cv delta ~3e-7; depth
    # differs only sub-cm in the feather band) for ~6% wall at zero VRAM cost.
    # Not byte-identical (square dilate + conv blur vs cv2 elliptical/GaussianBlur);
    # set SPAG_GPU_COMPOSITE=0 to fall back to the cv2/numpy composite.
    gpu_composite = gpu_depth_chain and os.environ.get("SPAG_GPU_COMPOSITE", "1") != "0"

    # Opt-in / default off (§8, bglock_open_questions.md): the default composite
    # linearly blends object_depth and depth_ref inside the feather band, which
    # invents physically-meaningless mid-air depth at discontinuities. Set
    # SPAG_HARD_DEPTH_CUTOVER=1 to select depth by nearest source instead of
    # interpolating it (alpha stays soft for anything else that wants it).
    hard_depth_cutover = os.environ.get("SPAG_HARD_DEPTH_CUTOVER", "0") == "1"

    # Default on (SPAG_LOCK_ACTIVITY=0 to opt out): lock activity-only pixels to
    # depth_ref -- activity is a whole-video std, so gating compositing on it
    # reintroduces the flicker bg-lock removes (211% p99 drift unlocked vs 0.000%
    # locked). Cost: SAM-missed moving background gets its geometry frozen; see D1.
    lock_activity = os.environ.get("SPAG_LOCK_ACTIVITY", "1") != "0"

    (output_folder / "gaussians").mkdir(parents=True, exist_ok=True)
    if freeze_bg:
        import json
        with open(output_folder / "gaussians" / "_freeze_bg_meta.json", "w") as f:
            json.dump({"freeze_bg": True, "n_bg_points": n_bg_points}, f, indent=2)
    if depth_npy_dir is not None:
        depth_npy_dir = Path(depth_npy_dir)
        depth_npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(depth_npy_dir / "depth_ref.npy", depth_ref_np.astype(np.float32))

    # §6.3 mask injection: opt-in proxy-GT test. Forces a genuinely static patch
    # into `lock_mask` so it runs the full dynamic-path machinery even though
    # depth_ref there IS ground truth -- gives an accuracy measurement, not just
    # stability. SPAG_MASK_INJECT='[[lat_frac, lon_frac, half_size_px], ...]',
    # default unset = no-op.
    _mask_inject_rects: list[tuple[int, int, int, int]] = []
    _mask_inject_env = os.environ.get("SPAG_MASK_INJECT")
    if _mask_inject_env:
        import json as _json
        _n_rows, _n_cols = depth_ref_np.shape
        for _lat_frac, _lon_frac, _half in _json.loads(_mask_inject_env):
            _cy = int(_lat_frac * (_n_rows - 1))
            _cx = int(_lon_frac * (_n_cols - 1))
            _y0, _y1 = max(0, _cy - _half), min(_n_rows, _cy + _half)
            _x0, _x1 = max(0, _cx - _half), min(_n_cols, _cx + _half)
            _mask_inject_rects.append((_y0, _y1, _x0, _x1))
        print(f"[SPAG4D] SPAG_MASK_INJECT active: {len(_mask_inject_rects)} patch(es) {_mask_inject_rects}")
    n_gaussians = []
    # Tier 4.5: async PLY writes. save_ply_gsplat's GPU->CPU sync, numpy
    # encoding and disk I/O are pure post-processing on already-computed
    # gaussians for frame i; they don't gate frame i+1's GPU work, so hand
    # them to a background thread and only join before anything downstream
    # reads the files (right after the loop). Lossless: same bytes, written
    # off the critical path. max_workers=2 keeps disk I/O bounded.
    ply_executor = ThreadPoolExecutor(max_workers=2)
    ply_futures = []

    # §9 diagnostics pass: CSV writer + per-region "previous frame" medians for
    # the frame-to-frame depth-delta columns (mirrors the existing bg/fg delta
    # pattern above, but split by the actual composite alpha bands instead of
    # the coarser fused_mask).
    _DIAG_LAT_BINS = 8
    diag_file = None
    diag_writer = None
    diag_prev_region_median = {"static": None, "dynamic": None, "feather": None}
    if diag_dir is not None:
        diag_file = open(diag_csv_path, "w", newline="")
        diag_writer = csv.writer(diag_file)
        diag_writer.writerow(
            ["frame", "scale", "shift", "n_static", "resid_median", "resid_p95"]
            + [f"resid_lat_{i}" for i in range(_DIAG_LAT_BINS)]
            + ["flow_static_med_dx", "flow_static_med_dy", "flow_static_med_mag"]
            + ["conf_mean", "conf_frac_zero", "conf_frac_lt_0.5", "age_mean", "age_max"]
            + ["depth_delta_static", "depth_delta_dynamic", "depth_delta_feather"]
            + ["per_object_flow_mag"]
        )

    def _diag_resid_by_latitude(resid_full: np.ndarray, static_mask: np.ndarray) -> list:
        h = static_mask.shape[0]
        edges = np.linspace(0, h, _DIAG_LAT_BINS + 1).astype(int)
        out = []
        for i in range(_DIAG_LAT_BINS):
            band_static = static_mask[edges[i]:edges[i + 1]]
            band_resid = resid_full[edges[i]:edges[i + 1]]
            vals = band_resid[band_static]
            out.append(float(np.median(vals)) if vals.size else float("nan"))
        return out

    def _diag_region_median(depth_np: np.ndarray, region_mask: np.ndarray):
        vals = depth_np[region_mask]
        return float(np.median(vals)) if vals.size else None

    t_depth_loop0 = time.time()
    # Tier 4.4: prefetch DA360 depth for frame i+1. depth_engine.predict is a
    # GPU op; issuing it before frame i's host-sync-heavy post-processing
    # (align/smoother/propagate/composite) lets its kernels run on the GPU
    # while the host is busy with frame i's bookkeeping. Purely a scheduling
    # reorder -- same predict() call/inputs, so output is lossless. Runs on
    # its own CUDA stream so it can execute concurrently with the main
    # stream's post-processing (same-stream ordering wouldn't overlap
    # anything -- the host-blocking sync points only drain the current stream).
    depth_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _predict_depth(frame_np):
        stream_ctx = torch.cuda.stream(depth_stream) if depth_stream is not None else contextlib.nullcontext()
        with stream_ctx:
            image_tensor = torch.from_numpy(frame_np).to("cuda", non_blocking=True)
            with torch.inference_mode():
                depth_raw, _ = depth_engine.predict(image_tensor, temporal_consistency=temporal_consistency)
        return image_tensor, depth_raw

    next_depth_result = _predict_depth(viz_frames[0])
    # Growth-curve instrumentation for docs/RAM_USAGE_INVESTIGATION_PLAN.md: periodic
    # ru_maxrss + tracemalloc snapshots every 20 frames, gated behind SPAG_RAM_TRACE so
    # it stays zero-cost by default. Snapshot deltas distinguish linear-with-frames
    # (leak/accumulation) from flat-with-spikes (peak driven by per-frame content).
    _ram_trace_prev_snapshot = _ram_checkpoint("before_frame_loop", _ram_trace_prev_snapshot)

    for idx, frame in enumerate(tqdm(viz_frames)):
        # Depth estimation
        start_depth = time.time()
        W, H, _ = frame.shape
        if depth_stream is not None:
            torch.cuda.current_stream().wait_stream(depth_stream)
        image_tensor, depth_raw = next_depth_result
        if depth_stream is not None:
            # Tell the caching allocator these were allocated on depth_stream
            # but are now live on the default stream, so it doesn't recycle
            # their blocks until this stream's consuming ops finish too.
            image_tensor.record_stream(torch.cuda.current_stream())
            depth_raw.record_stream(torch.cuda.current_stream())
        if idx + 1 < len(viz_frames):
            next_depth_result = _predict_depth(viz_frames[idx + 1])
        depth = depth_raw * global_scale
        if temporal_consistency:
            depth *= scale_factor_to_5m
        depth_np = depth.cpu().numpy()

        # Save depth preview if requested
        if depth_preview_path:
            converter._save_depth_preview(
                depth, str(depth_preview_path / f"depth_{idx}.jpg")
            )
        depth_estimation_cumtime += time.time() - start_depth

        start_depth = time.time()
        # Init with activity if needed
        fused_mask = (
            (activity_mask / 255).astype(np.uint8)
            if alignement_mask == "sam_and_activity"
            else np.zeros((W, H), dtype=np.uint8)
        )
        if alignement_mask == "all":
            fused_mask = np.zeros((W, H), dtype=np.uint8)  # Align on all pixels ; Nothing is considerer moving
            output = {}  # no SAM output in this mode; keeps §9 diagnostics' per-object loop safe
        else:  # Add SAM masks
            output = outputs_per_frame[idx * skip_step]
            sam_mask = np.zeros((W, H), dtype=np.uint8)
            for _, mask in output.items():
                # Lazy upsample: outputs_per_frame is kept at SAM3's native
                # working resolution (see segment_with_sam) instead of eagerly
                # upsampled to (W, H) for the whole clip up front -- avoids
                # holding a full-native-res mask array per object per frame
                # simultaneously (dominant RAM buffer, see
                # docs/RAM_USAGE_INVESTIGATION_PLAN.md). Resize once per read
                # instead, same INTER_NEAREST as the old eager path.
                if mask.shape[:2] != (W, H):
                    mask_u8 = mask.astype(np.uint8) if mask.dtype == bool else mask
                    mask = cv2.resize(mask_u8, (H, W), interpolation=cv2.INTER_NEAREST)
                sam_mask = np.maximum(sam_mask, mask)
            fused_mask = np.maximum(fused_mask, sam_mask)

        fused_t = torch.from_numpy(fused_mask).to("cuda") if gpu_depth_chain else None

        # Which pixels the bg-lock composite treats as moving (see SPAG_LOCK_ACTIVITY).
        # sam_mask is always defined here under bglock: run_video() rejects
        # depth_correction="bglock" unless alignement_mask is "sam"/"sam_and_activity",
        # both of which take the else-branch above. When alignement_mask="sam" the two
        # masks are already identical, so the flag is a no-op there.
        lock_mask = sam_mask if lock_activity else fused_mask
        lock_t = (
            fused_t if lock_mask is fused_mask
            else (torch.from_numpy(lock_mask).to("cuda") if gpu_depth_chain else None)
        )

        if _mask_inject_rects:
            # np.maximum always returns a new array, so this never mutates
            # fused_mask/sam_mask even where lock_mask aliases one of them.
            _inject_np = np.zeros_like(lock_mask)
            for _y0, _y1, _x0, _x1 in _mask_inject_rects:
                _inject_np[_y0:_y1, _x0:_x1] = 1
            lock_mask = np.maximum(lock_mask, _inject_np)
            if gpu_depth_chain:
                lock_t = torch.maximum(lock_t, torch.from_numpy(_inject_np).to("cuda"))

        # Align depth_ref_np
        median_depth_fg.append(float(np.median(
            depth_np[_valid_metric_mask(depth_np, fused_mask > 0, metrics_outlier_cap)]
        )))
        diag_out = {} if diag_writer is not None else None
        # Fix #2 (bglock silhouette-bleed): under bglock every static pixel is
        # overwritten by depth_ref in the composite regardless of what the
        # affine correction wrote there (alpha≈0 -> locked), so it's cheaper
        # and equally correct to apply (s, t) to the WHOLE frame rather than
        # try to compute the exact composite footprint (dilate+feather of
        # lock_mask, already computed once inside composite_bg_locked — no
        # need to pay for it twice). The one thing that still needs the ring:
        # the FIT must not be biased by DA360 edge-bleed on the pixels just
        # outside fused_mask that the dilated mask will later blend in, so we
        # widen the fit-exclusion (not the correction-mask param's original
        # apply role) to also exclude that ring. Measured real: ~1/3 of the
        # composite footprint sits outside fused_mask on this clip (frame 160:
        # 34.6k / 95.0k px). Confirmed via SPAG_DIAG_CSV that widening the
        # *application* alone doesn't move depth_delta_feather — that
        # instability is local per-pixel noise near object edges, not a
        # missing global correction, so apply_everywhere is strictly a
        # cost/simplicity win here, not a behavior regression.
        fit_exclude_ring = None
        fit_exclude_ring_t = None
        apply_everywhere = depth_correction == "bglock" and alignement_mask != "nothing"
        if apply_everywhere:
            feather_alpha = feather_dynamic_mask(lock_mask, bg_lock_dilate_px, bg_lock_feather_px)
            fit_exclude_ring = (feather_alpha > 0.02).astype(np.uint8)
            if gpu_depth_chain:
                fit_exclude_ring_t = torch.from_numpy(fit_exclude_ring).to("cuda")
        if gpu_depth_chain:
            # align + smoothing on the GPU (depth is already a cuda tensor)
            aligned_t = align_depth_frame_gpu(
                depth, depth_ref_t, fused_t, diag_out=diag_out,
                correction_mask=fit_exclude_ring_t, apply_everywhere=apply_everywhere,
            )
            if depth_smoother is not None:
                aligned_t = depth_smoother.call_torch(aligned_t)
            aligned_depth_np = aligned_t.cpu().numpy()
        else:
            aligned_t = None
            if alignement_mask == "nothing":
                aligned_depth_np = depth_np
            else:
                aligned_depth_np = align_depth_frame(
                    depth_np, depth_ref_np, fused_mask, alignement_method, verbose=True, diag_out=diag_out,
                    correction_mask=fit_exclude_ring, apply_everywhere=apply_everywhere,
                ).copy()
            # Solution 1: temporal depth smoothing (causal window) on the aligned
            # depth, applied before FG stabilization / bg-locked compositing / GS.
            if depth_smoother is not None:
                aligned_depth_np = depth_smoother(aligned_depth_np)
        aligned_median_depth_fg.append(float(np.median(
            aligned_depth_np[_valid_metric_mask(aligned_depth_np, fused_mask > 0, metrics_outlier_cap)]
        )))

        # Background depth metrics (NEW): median depth of static regions
        bg_pixels = fused_mask == 0
        if bg_pixels.sum() > 0:
            bg_valid = _valid_metric_mask(aligned_depth_np, bg_pixels, metrics_outlier_cap)
            bg_median = float(np.median(aligned_depth_np[bg_valid]))
            aligned_median_depth_bg.append(bg_median)
            # Frame-to-frame delta (spike detection)
            if len(aligned_median_depth_bg) > 1:
                delta = abs(aligned_median_depth_bg[-1] - aligned_median_depth_bg[-2])
                background_depth_deltas.append(delta)
        else:
            aligned_median_depth_bg.append(np.nan)
            if len(aligned_median_depth_bg) > 1:
                background_depth_deltas.append(np.nan)

        # Foreground depth deltas
        fg_pixels = fused_mask > 0
        if fg_pixels.sum() > 0:
            fg_median = float(np.median(aligned_depth_np[fg_pixels]))
            if len(aligned_median_depth_fg) > 1:
                fg_delta = abs(aligned_median_depth_fg[-1] - aligned_median_depth_fg[-2])
                foreground_depth_deltas.append(fg_delta)

        if depth_correction == "bglock" and alignement_mask != "nothing":
            # Blend the previous frame's composited depth with this frame's own
            # aligned depth for temporal coherence on dynamic-object pixels, then
            # lock every static pixel to depth_ref_np (fixed camera => zero-
            # variance background) and blend in the object depth only inside the
            # feathered mask. See docs/SPAG_ENV_FLAGS.md for why this replaced
            # flow-warp propagation.
            depth_object_t = None  # set in the gpu path, reused for gpu composite
            if idx == 0 or depth_prev_final is None:
                depth_object = aligned_depth_np.copy()
                if gpu_composite:
                    depth_object_t = aligned_t
            elif gpu_depth_chain:
                depth_object_t = (
                    (1.0 - _bglock_noflow_blend) * depth_prev_t
                    + _bglock_noflow_blend * aligned_t
                )
                depth_object = None if gpu_composite else depth_object_t.cpu().numpy()
            else:
                depth_object = (
                    (1.0 - _bglock_noflow_blend) * depth_prev_final
                    + _bglock_noflow_blend * aligned_depth_np
                )
            composite_alpha = None  # (H,W) soft object weight, only needed for §9 diagnostics
            if gpu_composite:
                aligned_depth_np2_t, alpha_t = composite_bg_locked_torch(
                    depth_object_t, depth_ref_t, lock_t,
                    dilate_px=bg_lock_dilate_px, feather_px=bg_lock_feather_px,
                    hard_depth_cutover=hard_depth_cutover,
                )
                depth_prev_t = aligned_depth_np2_t
                aligned_depth_np2 = aligned_depth_np2_t.cpu().numpy()
                if diag_writer is not None:
                    composite_alpha = alpha_t.float().cpu().numpy()
            else:
                aligned_depth_np2, alpha = composite_bg_locked(
                    depth_object, depth_ref_np, lock_mask,
                    dilate_px=bg_lock_dilate_px, feather_px=bg_lock_feather_px,
                    hard_depth_cutover=hard_depth_cutover,
                )
                if diag_writer is not None:
                    composite_alpha = alpha
                if gpu_depth_chain:
                    depth_prev_t = torch.from_numpy(aligned_depth_np2).to("cuda")
            depth_prev_final = aligned_depth_np2
        else:
            aligned_depth_np2 = fg_stabilizer(aligned_depth_np, fused_mask).copy()
            composite_alpha = None
        stabilized_aligned_median_depth_fg.append(
            float(np.median(
                aligned_depth_np2[_valid_metric_mask(aligned_depth_np2, fused_mask > 0, metrics_outlier_cap)]
            ))
        )

        aligned_depth_list_cpu.append(aligned_depth_np2)

        if diag_writer is not None:
            resid_median = diag_out.get("resid_median", float("nan"))
            resid_p95 = diag_out.get("resid_p95", float("nan"))
            scale = diag_out.get("scale", float("nan"))
            shift = diag_out.get("shift", float("nan"))
            n_static = diag_out.get("n_static", 0)
            if "resid_full" in diag_out:
                lat_resid = _diag_resid_by_latitude(diag_out["resid_full"], diag_out["static_mask"])
            else:
                lat_resid = [float("nan")] * _DIAG_LAT_BINS

            # (e) median flow vector over static pixels -- reuses §11's
            # check_camera_static.py logic (median over the composite's own
            # static definition, lock_mask==0).
            if idx >= 1 and (idx - 1) < len(flows_fwd):
                # flows_fwd is at WAFT working resolution; lock_mask is native
                # -- upscale lazily to match (no-op if already native res).
                flow_static = upscale_flow(flows_fwd[idx - 1], meta["H"], meta["W"])[lock_mask == 0]
                flow_med_dx = float(np.median(flow_static[:, 0])) if flow_static.size else float("nan")
                flow_med_dy = float(np.median(flow_static[:, 1])) if flow_static.size else float("nan")
                flow_med_mag = float(np.median(np.hypot(flow_static[:, 0], flow_static[:, 1]))) if flow_static.size else float("nan")
            else:
                flow_med_dx = flow_med_dy = flow_med_mag = float("nan")

            # (d) running_confidence histogram: dead now that flow-warp propagation
            # (and its confidence-decay tracking, flow_state) is gone -- columns
            # kept for CSV schema stability, always NaN.
            conf_mean = conf_frac_zero = conf_frac_lt_half = float("nan")
            age_mean = age_max = float("nan")

            # (h) depth variance split static/dynamic/feather, as frame-to-frame
            # median delta per region (same pattern as the existing bg/fg deltas
            # above, but keyed on the real composite alpha instead of fused_mask).
            deltas = {}
            if composite_alpha is not None:
                region_masks = {
                    "static": composite_alpha <= 0.02,
                    "dynamic": composite_alpha >= 0.98,
                    "feather": (composite_alpha > 0.02) & (composite_alpha < 0.98),
                }
                for name, rmask in region_masks.items():
                    med = _diag_region_median(aligned_depth_np2, rmask)
                    prev = diag_prev_region_median[name]
                    deltas[name] = abs(med - prev) if (med is not None and prev is not None) else float("nan")
                    diag_prev_region_median[name] = med
            else:
                deltas = {"static": float("nan"), "dynamic": float("nan"), "feather": float("nan")}

            # (f) per-object mean in-mask flow magnitude.
            obj_flow_strs = []
            if idx >= 1 and (idx - 1) < len(flows_fwd):
                # flows_fwd is at WAFT working resolution; output's masks are
                # SAM3's native working resolution too (see segment_with_sam),
                # which is not necessarily the same size as WAFT's -- resize
                # the (small) per-object mask to flow's shape rather than
                # upscaling the (large) flow field, to match resolutions
                # cheaply here.
                ff = flows_fwd[idx - 1]
                flow_mag_full = np.hypot(ff[..., 0], ff[..., 1])
                for obj_id, obj_mask in output.items():
                    m = obj_mask.astype(np.uint8) if obj_mask.dtype == bool else obj_mask
                    if m.shape[:2] != flow_mag_full.shape[:2]:
                        m = cv2.resize(m, (flow_mag_full.shape[1], flow_mag_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                    m = m.astype(bool)
                    if m.any():
                        obj_flow_strs.append(f"{obj_id}:{float(flow_mag_full[m].mean()):.3f}")
            per_object_flow_mag = ";".join(obj_flow_strs)

            diag_writer.writerow(
                [idx, scale, shift, n_static, resid_median, resid_p95]
                + lat_resid
                + [flow_med_dx, flow_med_dy, flow_med_mag]
                + [conf_mean, conf_frac_zero, conf_frac_lt_half, age_mean, age_max]
                + [deltas["static"], deltas["dynamic"], deltas["feather"]]
                + [per_object_flow_mag]
            )

        depth_alignement_cumtime += time.time() - start_depth

        # Run SPAG pipeline
        start_depth = time.time()
        if freeze_bg:
            assert alignement_mask in ["sam", "sam_and_activity", "nothing"], (
                "SAM must be computed to use freeze_bg, alignement_mask must be in ['sam', 'sam_and_activity']"
            )
            depth_for_gaussians = aligned_depth_np2.copy()
            depth_for_gaussians[sam_mask == 0] = np.nan  # set nan out of mask;
        else:
            depth_for_gaussians = aligned_depth_np2

        if depth_npy_dir is not None:
            np.save(depth_npy_dir / f"depth_{idx}.npy", aligned_depth_np2.astype(np.float32))
            np.save(depth_npy_dir / f"mask_{idx}.npy", sam_mask.astype(np.uint8))
            # Dense forward WAFT flow t-1 -> t at native (meta H x W) resolution,
            # H x W x 2 float32, original-pixel displacement (x wraps at the ERP
            # seam -- same convention propagate_depth_via_flow consumes). Consumed
            # downstream by FreeTimeGS as velocity supervision (its Phase 2, see
            # .claude/D1_FUTURE_WORK_PLAN.md). flows_fwd only exists on the bglock
            # flow path and has no entry for frame 0 (no t-1), so guard on both.
            if idx >= 1 and (idx - 1) < len(flows_fwd):
                # flows_fwd is stored at WAFT working resolution (not eagerly
                # upscaled -- see docs/RAM_USAGE_INVESTIGATION_PLAN.md);
                # upscale lazily here to preserve this export's native-res
                # contract with FreeTimeGS.
                flow_native = upscale_flow(flows_fwd[idx - 1], meta["H"], meta["W"])
                np.save(depth_npy_dir / f"flow_{idx}.npy", flow_native.astype(np.float32))

        # Foreground depth_min must NOT reuse the background-reference-derived
        # `depth_min` computed above: depth_ref_np is a static/background estimate
        # that, on a fixed camera, by construction never contains a person standing
        # close to the lens. Its 1st-percentile-derived depth_min (e.g. ~2.6m on a
        # scene whose empty background starts at ~3.3m) then silently zeroes out
        # every dynamic-subject pixel closer than that in filter_gaussian_candidates'
        # `depth_stride > depth_min` check, frame after frame -- close-range people
        # go fully invisible while the SAM3 mask around them looks correct. depth_min
        # exists to reject degenerate near-zero/behind-camera depth, not to gate real
        # foreground subjects, so floor it low and independently of the background
        # estimate here. See docs/DISPO_RDV_CLOSE_RANGE_PRUNING_FIX.md.
        fg_depth_min = min(depth_min, 0.15) if depth_min is not None else 0.15
        gaussians = to_gaussians(
            converter,
            depth_for_gaussians,
            image_tensor,
            stride,
            H,
            fg_depth_min,
            depth_max,
            sky_threshold,
            outlier_pruning,
            grazing_angle,
            sparse_pruning,
        )
        if freeze_bg:
            bg_frame = gaussians_bg
            if freeze_bg_live_color and gaussians_bg["means"].shape[0] > 0:
                # Geometry stays frozen (gaussians_bg["means"/"scales"/"quats"]
                # never change); only re-sample color at each bg Gaussian's
                # stored pixel_idx from the current frame's live image, where
                # that pixel is currently visible (sam_mask==0). Pixels
                # currently covered by a dynamic object keep the frozen
                # median color -- there's no live appearance to sample there.
                img_f = image_tensor.float()
                if image_tensor.dtype == torch.uint8:
                    img_f = img_f / 255.0
                flat_img = img_f.reshape(-1, 3)
                pixel_idx = gaussians_bg["pixel_idx"].to(flat_img.device)
                visible_flat = torch.from_numpy((sam_mask == 0).reshape(-1)).to(flat_img.device)
                visible_at_bg = visible_flat[pixel_idx]
                bg_colors = gaussians_bg["colors"].clone()
                bg_colors[visible_at_bg] = flat_img[pixel_idx[visible_at_bg]]
                bg_frame = dict(gaussians_bg)
                bg_frame["colors"] = bg_colors
            gaussians = {k: torch.cat([bg_frame[k], gaussians[k]], dim=0) for k in bg_frame}
        colors_linear = False
        gs_generation_cumtime += time.time() - start_depth

        # Save PLY
        n_gaussians.append(gaussians["means"].shape[0])
        ply_futures.append(ply_executor.submit(
            save_ply_gsplat,
            gaussians,
            str(output_folder / "gaussians" / f"frame_{idx}.ply"),
            sh_degree=0,
            colors_linear=colors_linear,
        ))

        if ram_trace and (idx % 20 == 0 or idx == len(viz_frames) - 1):
            _ram_trace_prev_snapshot = _ram_checkpoint(f"frame_{idx}", _ram_trace_prev_snapshot)

    for f in ply_futures:
        f.result()
    ply_executor.shutdown(wait=True)
    _ram_trace_prev_snapshot = _ram_checkpoint("after_ply_futures_drained", _ram_trace_prev_snapshot)
    if diag_file is not None:
        diag_file.close()
        print(f"[SPAG4D] §9 diagnostics CSV written: {diag_csv_path}")
    step_times["depth_loop_total"] = time.time() - t_depth_loop0
    step_times["depth_loop_estimation"] = depth_estimation_cumtime
    step_times["depth_loop_alignment"] = depth_alignement_cumtime
    step_times["depth_loop_gs_generation"] = gs_generation_cumtime

    plt.figure(figsize=(12, 5))
    plt.plot(median_depth_fg, marker="o", label="Original", color="orange")
    plt.plot(aligned_median_depth_fg, marker="o", label="Aligned", color="blue")
    plt.plot(stabilized_aligned_median_depth_fg, marker="o", label="Stabilized", color="green")
    plt.title("Profondeur médiane des pixels en mouvement")
    plt.xlabel("Frame index")
    plt.ylabel("Profondeur médiane")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "_motion_mask_stats.png")

    if torch.cuda.is_available():
        print(f"[VRAM] cumulative peak after depth loop (whole-run peak): "
              f"{torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    processing_time = time.time() - start_time
    file_size = (output_folder / "gaussians" / f"frame_{idx}.ply").stat().st_size

    print("="*30)
    step_order = [
        "frame_extraction", "flow_waft", "sam3_segmentation", "background_extraction",
        "depth_ref", "depth_loop_total",
        "depth_loop_estimation", "depth_loop_alignment", "depth_loop_gs_generation",
    ]
    indented = {"depth_loop_estimation", "depth_loop_alignment", "depth_loop_gs_generation"}
    label_w = max(len(k) for k in step_order) + 2
    print(f"\n[SPAG4D] === Wall time per step (total {processing_time:.1f}s) ===")
    for key in step_order:
        if key not in step_times:
            continue
        t = step_times[key]
        pct = 100 * t / processing_time if processing_time > 0 else 0.0
        prefix = "    └ " if key in indented else ""
        label = key.replace("depth_loop_", "") if key in indented else key
        print(f"  {prefix}{label:<{label_w}} {t:8.2f}s  ({pct:5.1f}%)")
    accounted = sum(v for k, v in step_times.items() if k not in indented)
    other = processing_time - accounted
    print(f"  {'other/unaccounted':<{label_w}} {other:8.2f}s  ({100 * other / processing_time if processing_time > 0 else 0:5.1f}%)")
    print("="*30)

    # Compute pixels depth std across frames for metadata
    import json

    def stats_it(depth_std_all):
        stats = {
            "std_mean": float(np.nanmean(depth_std_all)),
            "std_p50": float(np.nanpercentile(depth_std_all, 50)),
            "std_p90": float(np.nanpercentile(depth_std_all, 90)),
            "std_p95": float(np.nanpercentile(depth_std_all, 95)),
            "std_p99": float(np.nanpercentile(depth_std_all, 99)),
            "std_max": float(np.nanmax(depth_std_all)),
            "fraction_gt_1": float(np.nanmean(depth_std_all > 1.0)),
            "fraction_gt_2": float(np.nanmean(depth_std_all > 2.0)),
        }
        return stats

    if aligned_depth_list_cpu:
        # Row-chunked nanstd: np.stack(aligned_depth_list_cpu) materializes a
        # second full-clip float buffer on top of the list already held for
        # the whole run (~6GB on a 204-frame 3840x1920 clip, doubling
        # momentarily to ~12GB) -- nanstd is per-pixel independent across the
        # frame axis, so chunk over rows instead of stacking the whole clip
        # at once (same pattern as compute_temporal_median's fix above; see
        # docs/RAM_USAGE_INVESTIGATION_PLAN.md).
        _dh, _dw = aligned_depth_list_cpu[0].shape[:2]
        depth_std_all = np.zeros((_dh, _dw), dtype=np.float32)
        _drow_chunk = max(1, 256 * 1024 * 1024 // max(1, len(aligned_depth_list_cpu) * _dw * 4))
        for _r0 in range(0, _dh, _drow_chunk):
            _r1 = min(_r0 + _drow_chunk, _dh)
            _stack_chunk = np.stack([a[_r0:_r1] for a in aligned_depth_list_cpu], axis=0)
            depth_std_all[_r0:_r1] = np.nanstd(_stack_chunk, axis=0)
            del _stack_chunk
        _ram_trace_prev_snapshot = _ram_checkpoint("after_aligned_depth_stack", _ram_trace_prev_snapshot)
        stats_all = stats_it(depth_std_all)
        if np.nanmax(depth_std_all) > 0:
            depth_std_map = np.clip(
                depth_std_all / max(np.nanpercentile(depth_std_all, 99), 1e-6), 0, 1
            )
            depth_std_map = (depth_std_map * 255).astype(np.uint8)
        else:
            depth_std_map = np.zeros((W, H), dtype=np.uint8)
        cv2.imwrite(
            str(output_folder / "_temporal_depth_std.jpg"),
            cv2.applyColorMap(depth_std_map, cv2.COLORMAP_TURBO),
        )

        unstable = depth_std_all > 1.0
        cv2.imwrite(str(output_folder / "_unstable_1.png"), (unstable.astype(np.uint8) * 255))

        mask_inside = activity_mask.astype(bool)
        mask_outside = ~activity_mask.astype(bool)
        # 2. Initialize empty black images for both regions
        H, W = depth_std_all.shape
        activity_map_inside = np.zeros((H, W), dtype=np.uint8)
        activity_map_outside = np.zeros((H, W), dtype=np.uint8)
        # --- Process INSIDE the mask ---
        if mask_inside.any():
            depth_std_inmask = depth_std_all[mask_inside]
            stats_in = stats_it(depth_std_inmask)
            if np.nanmax(depth_std_inmask) > 0:
                inside_pixels = depth_std_inmask / np.nanmax(depth_std_inmask)
                activity_map_inside[mask_inside] = (inside_pixels * 255.0).astype(np.uint8)

        # --- Process OUTSIDE the mask ---
        if mask_outside.any():
            depth_std_outmask = depth_std_all[mask_outside]
            stats_out = stats_it(depth_std_outmask)
            if np.nanmax(depth_std_outmask) > 0:
                outside_pixels = depth_std_outmask / np.nanmax(depth_std_outmask)
                activity_map_outside[mask_outside] = (outside_pixels * 255.0).astype(np.uint8)

        with open(
            str(
                output_folder
                / f"_depth_stats_{alignement_mask}_{activity_std_threshold}_{'freeze' if freeze_bg else ''}.json"
            ),
            "w",
        ) as f:
            json.dump({"all": stats_all, "in": stats_in, "out": stats_out}, f, indent=2)

        # 3. Save both results
        cv2.imwrite(
            str(output_folder / "_aligned_depth_activity_inside.jpg"),
            cv2.applyColorMap(activity_map_inside, cv2.COLORMAP_TURBO),
        )
        cv2.imwrite(
            str(output_folder / "_aligned_depth_activity_outside.jpg"),
            cv2.applyColorMap(activity_map_outside, cv2.COLORMAP_TURBO),
        )
        print("|" * 100)

    # Compute actual depth range
    distances = gaussians["means"].norm(dim=-1)
    if distances.numel() > 0:
        depth_range = (distances.min().item(), distances.max().item())
    else:
        depth_range = (0.0, 0.0)

    # Compile depth metrics
    depth_metrics = {
        "raw_fg_median": median_depth_fg,
        "aligned_fg_median": aligned_median_depth_fg,
        "stabilized_fg_median": stabilized_aligned_median_depth_fg,
        "aligned_bg_median": aligned_median_depth_bg,
        "bg_median_deltas": background_depth_deltas,
        "fg_median_deltas": foreground_depth_deltas,
    }

    # ru_maxrss is peak RSS (KB on Linux) since process start, not just this call --
    # matches the semantics of torch.cuda.max_memory_allocated() callers already use
    # for VRAM (a running peak, reset by the caller between runs if isolation is needed).
    _ram_trace_prev_snapshot = _ram_checkpoint("before_return", _ram_trace_prev_snapshot)
    ram_max_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    depth_reproj_consistency = None
    if depth_npy_dir is not None:
        from spag4d.analysis.reconstruction_metrics import compute_depth_reprojection_consistency
        depth_reproj_consistency = compute_depth_reprojection_consistency(depth_npy_dir)
        if depth_reproj_consistency is not None:
            print(f"[SPAG4D] depth reprojection consistency: bg_mean_err="
                  f"{depth_reproj_consistency['bg_mean_err']} fg_mean_err="
                  f"{depth_reproj_consistency['fg_mean_err']} bg_flagged_frames="
                  f"{depth_reproj_consistency['bg_flagged_frames']}")

    return ConversionResult(
        output_path=output_path.replace(".ply", "_0.ply"),
        splat_count=n_gaussians,
        file_size=file_size,
        processing_time=processing_time,
        depth_range=depth_range,
        depth_npy_path=None,  # str(depth_npy_path) if depth_npy_path else None,
        panorama_size=(W, H),
        depth_metrics=depth_metrics,
        ram_max_mb=ram_max_mb,
        n_tracked_objects=n_tracked_objects,
        mean_objects_per_frame=mean_objects_per_frame,
        depth_reproj_consistency=depth_reproj_consistency,
    )


# §10.2 (benchmarks/bglock_open_questions.md): DA360 disparity near zero inverts to ~1e6
# (sky/poles), and PaGeR clamps to exactly 200.0 — both pass depth>0 & isfinite, so
# without this they silently enter the affine fit and pull scale/shift toward
# nonsense. Neither backend declares a proper validity mask (§10.1), so this is a
# generic magnitude gate: real scene depth is never 50x the plate's own median.
_FIT_OUTLIER_RATIO = 50.0


def _fit_outlier_cap(depth_ref: np.ndarray) -> float:
    finite_ref = depth_ref[np.isfinite(depth_ref) & (depth_ref > 0)]
    if finite_ref.size == 0:
        return float("inf")
    ref_median = float(np.median(finite_ref))
    return ref_median * _FIT_OUTLIER_RATIO if ref_median > 0 else float("inf")


def _valid_metric_mask(depth: np.ndarray, base_mask: np.ndarray, cap: float) -> np.ndarray:
    """`base_mask` narrowed to pixels a per-frame median metric can trust: finite,
    positive, under the §4.3 outlier cap. Falls back to `base_mask` unfiltered if the
    gate would empty the selection (e.g. a region that is genuinely all sky/outlier),
    since np.median on an empty array raises."""
    gated = base_mask & np.isfinite(depth) & (depth > 0) & (depth < cap)
    return gated if gated.any() else base_mask


def align_depth_frame(
    depth_frame: np.ndarray,
    depth_ref: np.ndarray,
    mask_moving: np.ndarray,
    method: str = "lstsq",  # "lstsq" | "ransac" | "median"
    min_static_pixels: int = 100,
    scale_clip: tuple = (0.5, 2.0),
    ransac_residual_threshold: float = 0.1,  # en unités de profondeur relative
    ransac_max_trials: int = 100,
    verbose: bool = False,
    diag_out: "dict | None" = None,
    correction_mask: "np.ndarray | None" = None,
    apply_everywhere: bool = False,
) -> np.ndarray:
    """
    Aligne depth_frame sur depth_ref en estimant scale s et shift t
    tels que : depth_aligned ≈ s * depth_frame + t

    N'utilise que les pixels statiques (mask_moving == 0) pour l'estimation.

    Args:
        depth_frame             : depth à aligner (H, W), float
        depth_ref               : depth de référence (H, W), float
        mask_moving             : masque binaire, 1 = pixel mobile (H, W), uint8
        method                  : algorithme d'estimation
            "lstsq"  → moindres carrés classiques (rapide, sensible aux outliers)
            "ransac" → RANSAC (robuste aux outliers, plus lent)
            "median" → estimateur médian de Siegel/Theil-Sen simplifié :
                       s = median(y/x) sur pixels statiques valides,
                       t recalculé après. Très robuste, très rapide.
        min_static_pixels       : nb minimum de pixels statiques requis
        scale_clip              : (min, max) pour clipper s et éviter les explosions
        ransac_residual_threshold: seuil résiduel RANSAC (en unités depth_ref).
                                   Mettre ~1-5% de la plage de depth_ref typiquement.
        ransac_max_trials       : nb max d'itérations RANSAC
        verbose                 : affiche les stats d'estimation

    Returns:
        aligned_depth : np.ndarray (H, W), depth alignée, clippée >= 0
    """

    # ── 1. Masque statique ───────────────────────────────────────────────────
    outlier_cap = _fit_outlier_cap(depth_ref)
    static_mask = (
        (mask_moving == 0)
        & (depth_frame > 0)
        & (depth_ref > 0)
        & np.isfinite(depth_frame)
        & np.isfinite(depth_ref)
        & (depth_frame < outlier_cap)  # §4.3: excludes backend outliers from the fit
        & (depth_ref < outlier_cap)
    )
    if correction_mask is not None:
        # Keep the composite-footprint ring out of the FIT — it's a
        # mask-transition band (DA360 edge-bleed near a moving object), not a
        # trustworthy static sample, even though under bglock the correction
        # itself gets applied everywhere (see apply_everywhere / step 6).
        static_mask &= correction_mask == 0

    n_static = np.sum(static_mask)
    if n_static < min_static_pixels:
        if verbose:
            print(
                f"[align_depth] ⚠ Seulement {n_static} pixels statiques "
                f"(< {min_static_pixels}), pas d'alignement appliqué."
            )
        return depth_frame.copy()

    x = depth_frame[static_mask].flatten()  # valeurs à corriger
    y = depth_ref[static_mask].flatten()  # cible

    # ── 2. Normalisation pour que le seuil RANSAC soit invariant à l'échelle ─
    # On normalise x et y par la médiane de y pour que residual_threshold
    # soit interprétable comme une fraction relative (ex: 0.1 = 10%)
    y_med = np.median(y)
    if y_med <= 0:
        y_med = 1.0
    x_norm = x / y_med
    y_norm = y / y_med

    # ── 3. Estimation selon la méthode choisie ───────────────────────────────
    s_raw, t_raw = _estimate_scale_shift(
        x_norm,
        y_norm,
        method=method,
        ransac_residual_threshold=ransac_residual_threshold,
        ransac_max_trials=ransac_max_trials,
        verbose=verbose,
    )

    # ── 4. Re-dénormalisation ────────────────────────────────────────────────
    # On a estimé : y_norm = s_raw * x_norm + t_raw
    # soit y/y_med = s_raw * (x/y_med) + t_raw
    # soit y       = s_raw * x + t_raw * y_med
    s = s_raw
    t = t_raw * y_med

    # ── 5. Clipping de s + recalcul de t pour rester cohérent ───────────────
    s_clipped = np.clip(s, scale_clip[0], scale_clip[1])
    if s_clipped != s:
        if verbose:
            print(f"[align_depth] ⚠ s={s:.3f} clippé à {s_clipped:.3f} — drift important détecté")
        # On recalcule t sur les médianes pour minimiser le biais introduit
        # par le clipping (plutôt que garder un t calculé avec un s erroné)
        t = np.median(y) - s_clipped * np.median(x)
    s = s_clipped

    if verbose:
        residuals = np.abs(y - (s * x + t))
        print(
            f"[align_depth] method={method} | s={s:.4f} t={t:.4f} | "
            f"n_static={n_static} | "
            f"résidus médian={np.median(residuals):.4f} "
            f"p95={np.percentile(residuals, 95):.4f}"
        )

    if diag_out is not None:
        # §9 diagnostics pass (SPAG_DIAG_CSV): fitted scale/shift + residual on
        # the static set the fit itself used, plus a full-frame residual map
        # (evaluated everywhere, not just on static_mask) so the caller can bin
        # it by latitude without redoing the fit.
        residuals = np.abs(y - (s * x + t))
        diag_out["scale"] = float(s)
        diag_out["shift"] = float(t)
        diag_out["n_static"] = int(n_static)
        diag_out["resid_median"] = float(np.median(residuals)) if residuals.size else float("nan")
        diag_out["resid_p95"] = float(np.percentile(residuals, 95)) if residuals.size else float("nan")
        diag_out["resid_full"] = np.abs(depth_ref - (s * depth_frame + t))
        diag_out["static_mask"] = static_mask

    # ── 6. Application hors mask statique (ou partout si apply_everywhere,
    # sûr sous bglock où la compositing verrouille de toute façon le fond au
    # depth_ref là où le mask est nul) ──────────────────────────────────────
    apply_mask = np.ones_like(static_mask) if apply_everywhere else ~static_mask
    depth_frame[apply_mask] = s * depth_frame[apply_mask] + t

    return np.clip(depth_frame, 0.0, None)


def align_depth_frame_gpu(
    depth_frame: torch.Tensor,
    depth_ref: torch.Tensor,
    mask_moving: torch.Tensor,
    min_static_pixels: int = 100,
    scale_clip: tuple = (0.5, 2.0),
    diag_out: "dict | None" = None,
    correction_mask: "torch.Tensor | None" = None,
    apply_everywhere: bool = False,
) -> torch.Tensor:
    """GPU/torch equivalent of align_depth_frame(method="lstsq").

    Fits y ~ s*x + t on static pixels via closed-form normal equations (float64
    accumulation), clips s, and applies (s, t) to the NON-static pixels only —
    exactly like the numpy path. Not byte-identical to it (GPU reduction order),
    but within float rounding. All tensors are (H,W) cuda; returns a new (H,W).

    `correction_mask`, when given, widens the FIT exclusion to also cover the
    composite's dilated+feathered footprint ring (DA360 edge-bleed near a
    moving object shouldn't feed the fit). `apply_everywhere`, when True,
    applies (s, t) to the whole frame instead of just the non-static pixels —
    safe under bglock, where the composite locks alpha≈0 pixels to depth_ref
    regardless of what this function wrote there, and cheaper than computing
    an exact composite-footprint mask for the application step too.
    """
    finite_ref = depth_ref[torch.isfinite(depth_ref) & (depth_ref > 0)]
    if finite_ref.numel() > 0:
        ref_median = float(torch.median(finite_ref))
        outlier_cap = ref_median * _FIT_OUTLIER_RATIO if ref_median > 0 else float("inf")
    else:
        outlier_cap = float("inf")
    static = (
        (mask_moving == 0)
        & (depth_frame > 0)
        & (depth_ref > 0)
        & torch.isfinite(depth_frame)
        & torch.isfinite(depth_ref)
        & (depth_frame < outlier_cap)  # §4.3: excludes backend outliers from the fit
        & (depth_ref < outlier_cap)
    )
    if correction_mask is not None:
        static &= correction_mask == 0
    n_static = int(static.sum().item())
    if n_static < min_static_pixels:
        return depth_frame.clamp_min(0.0)

    x = depth_frame[static]
    y = depth_ref[static]
    y_med = torch.median(y)
    if float(y_med) <= 0:
        y_med = torch.ones((), device=y.device)
    x_norm = (x / y_med).double()
    y_norm = (y / y_med).double()

    n = x_norm.numel()
    sx = x_norm.sum()
    sy = y_norm.sum()
    sxx = torch.dot(x_norm, x_norm)
    sxy = torch.dot(x_norm, y_norm)
    denom = n * sxx - sx * sx
    if float(denom) == 0.0:
        s_raw, t_raw = 0.0, float(sy / n)
    else:
        s_raw = float((n * sxy - sx * sy) / denom)
        t_raw = float((sy - s_raw * sx) / n)
    s = s_raw
    t = t_raw * float(y_med)

    s_clipped = float(np.clip(s, scale_clip[0], scale_clip[1]))
    if s_clipped != s:
        t = float(torch.median(y)) - s_clipped * float(torch.median(x))
    s = s_clipped

    if diag_out is not None:
        # §9 diagnostics pass (SPAG_DIAG_CSV): mirrors align_depth_frame's diag_out.
        resid = (y - (s * x + t)).abs()
        diag_out["scale"] = float(s)
        diag_out["shift"] = float(t)
        diag_out["n_static"] = n_static
        diag_out["resid_median"] = float(resid.median()) if resid.numel() else float("nan")
        diag_out["resid_p95"] = float(torch.quantile(resid, 0.95)) if resid.numel() else float("nan")
        diag_out["resid_full"] = (depth_ref - (s * depth_frame + t)).abs().cpu().numpy()
        diag_out["static_mask"] = static.cpu().numpy()

    out = depth_frame.clone()
    nm = torch.ones_like(static) if apply_everywhere else ~static
    out[nm] = s * out[nm] + t
    return out.clamp_min(0.0)


def _estimate_scale_shift(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    ransac_residual_threshold: float,
    ransac_max_trials: int,
    verbose: bool,
) -> tuple[float, float]:
    """
    Estime s, t tel que y ≈ s*x + t sur les vecteurs 1D x et y.
    Travaille sur des données déjà normalisées.
    """

    if method == "lstsq":
        # Closed-form normal equations for the 2-parameter fit y ~ s*x + t.
        # Mathematically identical to np.linalg.lstsq's solution but O(N) with a
        # few reductions instead of building an (N,2) matrix and running SVD over
        # millions of static pixels. Accumulate in float64 for stability.
        x64 = x.astype(np.float64, copy=False)
        y64 = y.astype(np.float64, copy=False)
        n = x64.size
        sx = x64.sum()
        sy = y64.sum()
        sxx = np.dot(x64, x64)
        sxy = np.dot(x64, y64)
        denom = n * sxx - sx * sx
        if denom == 0.0:
            s, t = 0.0, sy / n
        else:
            s = (n * sxy - sx * sy) / denom
            t = (sy - s * sx) / n
        return float(s), float(t)

    elif method == "ransac":
        try:
            from sklearn.linear_model import LinearRegression, RANSACRegressor
        except ImportError:
            raise ImportError(
                "scikit-learn est requis pour method='ransac'. "
                "Installe-le avec : pip install scikit-learn"
            )

        reg = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=True),
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
            random_state=42,
        )
        reg.fit(x.reshape(-1, 1), y)

        if verbose:
            n_inliers = reg.inlier_mask_.sum()
            inlier_ratio = n_inliers / len(x)
            print(f"[align_depth/RANSAC] inliers: {n_inliers}/{len(x)} ({inlier_ratio:.1%})")
            if inlier_ratio < 0.3:
                print(
                    f"[align_depth/RANSAC] ⚠ Ratio d'inliers bas ({inlier_ratio:.1%})"
                    f" — le masque moving est peut-être insuffisant"
                )

        s = float(reg.estimator_.coef_[0])
        t = float(reg.estimator_.intercept_)
        return s, t

    elif method == "median":
        # Estimateur de Siegel simplifié : s = median(y/x) sur les pixels
        # où x > epsilon (évite division par zéro).
        # Beaucoup plus rapide que Theil-Sen complet (O(n) vs O(n²)),
        # robuste à ~50% d'outliers, suffisant pour du depth monoculaire.
        eps = 1e-6
        valid = x > eps
        if valid.sum() < 10:
            # Fallback lstsq si trop peu de pixels valides
            A = np.vstack([x, np.ones(len(x))]).T
            s, t = np.linalg.lstsq(A, y, rcond=None)[0]
            return float(s), float(t)

        s = float(np.median(y[valid] / x[valid]))
        # t estimé par médiane des résidus (robuste)
        t = float(np.median(y - s * x))
        return s, t

    else:
        raise ValueError(f"method='{method}' inconnu. Choix : 'lstsq', 'ransac', 'median'.")

def segment_with_flows(
    video_path: str,
    output_path: Path,
    waft_frames: np.ndarray,
    flow_masks: np.ndarray,
    meta: dict,
    n_total_frames: int,
    skip_step: int = 10,
    enable_viz: bool = False,
    size_threshold: int = 50,
    prox_threshold: int = 80,
    min_times_seen: int = 3,
    merge_gap_px: int = 30,
    flows_fwd: list | None = None,
    flows_bwd: list | None = None,
):
    # On prépare SAM3 vidéo
    gpus_to_use = [torch.cuda.current_device()]
    video_predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # Tier 4.3 (opt-in, SPAG_SAM3_SCALE<1.0): pass a downscaled in-memory PIL
    # frame list instead of a file path (avoids an earlier temp-mp4-proxy
    # IndexError from frame-count desync). Not lossless -- coarser input changes
    # SAM3's detections. Built at NATIVE frame count, not waft_frames (which is
    # skip_step-decimated; add_prompt uses native indices -- confirmed IndexError).
    sam3_scale = float(os.environ.get("SPAG_SAM3_SCALE", "1.0"))
    sam3_maxsize = int(os.environ.get("SPAG_SAM3_MAXSIZE", "2048"))
    if sam3_maxsize > 0:
        # Absolute cap on the longer edge (not a relative scale): only clips above
        # it get downscaled. Must use meta['H']/meta['W'] (true native), not
        # `waft_frames` (~1024px cap) -- scaling off that anchored every clip to
        # the same ~512x256 regardless of native res (confirmed bug, see B1 §3).
        native_long_edge = max(meta['H'], meta['W'])
        sam3_scale = min(1.0, sam3_maxsize / native_long_edge)
    # Always build an in-memory PIL frame list, never pass `str(video_path)`
    # directly: SAM3's path-based cv2 loader normalizes without dividing by 255
    # first (~255x too-large pixel values); the list path does it correctly.
    # Confirmed via repro: same "person" prompt found both people via list, only
    # 1 via raw path.
    if sam3_scale < 1.0:
        # Apply to NATIVE dims (frames are decoded at native res below) --
        # scaling waft_frames instead would compose the two shrinks, e.g. a
        # 3840px clip asking for 1536 would actually get 408px.
        sW = max(int(meta['W'] * sam3_scale) & ~1, 2)
        sH = max(int(meta['H'] * sam3_scale) & ~1, 2)
    else:
        sW, sH = meta['W'], meta['H']
    cap = cv2.VideoCapture(str(video_path))
    sam3_resource_path = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if sam3_scale < 1.0:
            frame_rgb = cv2.resize(frame_rgb, (sW, sH), interpolation=cv2.INTER_AREA)
        sam3_resource_path.append(Image.fromarray(frame_rgb))
    cap.release()
    if sam3_scale < 1.0:
        _scale_src = f"SPAG_SAM3_MAXSIZE={sam3_maxsize}" if sam3_maxsize > 0 else f"SPAG_SAM3_SCALE={sam3_scale}"
        print(f"[SPAG4D] SAM3 segmentation pass downscaled to {sW}x{sH} ({_scale_src}, effective scale={sam3_scale:.3f}), {len(sam3_resource_path)} in-memory native-frame-count frames...")
    else:
        print(f"[SPAG4D] SAM3 segmentation pass at native resolution {sW}x{sH}, {len(sam3_resource_path)} in-memory frames...")

    # Start a session
    initial_response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=sam3_resource_path,
            offload_video_to_cpu=True,
            # Offload the per-frame inference state (memory bank) to CPU. It scales
            # with clip length x tracked objects and is the dominant SAM3 GPU peak
            # for long videos; offloading trades ~10-15% tracking fps for a large
            # VRAM saving. Masks/outputs are unchanged (lossless).
            offload_state_to_cpu=True,
        )
    )
    session_id = initial_response["session_id"]
    print("SAM3 video session started with session_id:", session_id)

    kernel_erode = np.ones((3, 3), np.uint8)
    kernel_dilate = np.ones((7, 7), np.uint8)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)) # Much wider to bridge the gaps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    boxes_preview = cv2.VideoWriter(output_path / 'boxes_input_sam.mp4', fourcc, meta['fps'], (meta['new_W'], meta['new_H']))

    active_tracks = []
    next_available_obj_id = 1
    start_frame_index = None

    for frame_idx in tqdm(range(flow_masks.shape[0]), desc="Analyzing Motion Splits"):
        fg_mask = flow_masks[frame_idx]
        # Apply morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        fg_mask = cv2.erode(fg_mask, kernel_erode, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)

        def _sample_extra_points(contour, x, y, w, h, n_bands: int = 4):
            # A single centroid tends to grow a mask around the most salient part
            # (e.g. torso) for an elongated object rather than the whole body.
            # Sample real interior points across height bands instead, all
            # verified inside the contour (never a synthetic box interpolation).
            contour_mask = np.zeros((y + h, x + w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            pts = []
            for frac in np.linspace(0.15, 0.85, n_bands):
                row = y + int(frac * h)
                row = min(max(row, y), y + h - 1)
                xs = np.nonzero(contour_mask[row, x:x + w])[0]
                if xs.size == 0:
                    continue
                px = x + int(np.median(xs))
                pts.append((px, row))
            return pts

        # Find and filter contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_frame_centers = []
        bounding_boxes = []
        scores = []
        for contour in contours:
            # Filter : minimum area threshold
            # Ignore small pixels (noise)
            area = cv2.contourArea(contour)
            if area < size_threshold:
                continue

            # Filter : Weird aspect ratio (Optional)
            # Prevents long, thin lines of light/shadows from triggering detection
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > 4.0 or aspect_ratio < 0.2:
                continue # Skip unrealistic shapes

            # Filter : Check the fill ratio of the bounding box in the mask
            # If filling ratio is low, it's likely a ghost from last frames
            roi = fg_mask[y:y+h, x:x+w]
            white_pixels = cv2.countNonZero(roi)
            # conf (0.0 à 1.0) est le ratio de remplissage
            conf = white_pixels / (w * h)
            if conf < 0.5:
                continue

            M = cv2.moments(contour) # Mass center (inmask while x+w/2 can be outside if L-shaped for example)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                extra_points = _sample_extra_points(contour, x, y, w, h)
                current_frame_centers.append((cX, cY, x, y, w, h, conf, area, extra_points))
                bounding_boxes.append((x, y, w, h))
                scores.append(conf)

            if len(current_frame_centers) <= 0:
                print("no contour satisfying")
                continue #no move found

        # Merge nearby/overlapping fragment boxes (e.g. hat + shirt + pants of one
        # person breaking apart in the flow mask) into ONE box per real object
        # before track matching -- otherwise each fragment spawns its own track and
        # its own SAM3 obj_id, splitting a single silhouette into several colors.
        # Two boxes merge if they overlap or the rect-to-rect gap is <= merge_gap_px.
        def _box_gap(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
            dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
            return max(dx, dy)

        n = len(current_frame_centers)
        parent = list(range(n))

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i, j):
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if _box_gap(current_frame_centers[i][2:6], current_frame_centers[j][2:6]) <= merge_gap_px:
                    _union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(_find(i), []).append(i)

        merged_centers = []
        merged_boxes = []
        for members in clusters.values():
            xs1, ys1, xs2, ys2, areas, confs, centroids = [], [], [], [], [], [], []
            # Real, mask-derived points across all merged members: centroid + height-
            # band samples. Used as SAM3 prompt points instead of geometric fractions
            # of the merged box, which can fall in empty space for an L-shaped merge.
            member_points = []
            for idx in members:
                mcx, mcy, x, y, w, h, conf, area, extra_points = current_frame_centers[idx]
                xs1.append(x)
                ys1.append(y)
                xs2.append(x + w)
                ys2.append(y + h)
                areas.append(area)
                confs.append(conf)
                centroids.append((mcx, mcy))
                member_points.append((mcx, mcy))
                member_points.extend(extra_points)
            mx, my = min(xs1), min(ys1)
            mw, mh = max(xs2) - mx, max(ys2) - my
            total_area = sum(areas)
            merged_conf = sum(c * a for c, a in zip(confs, areas)) / total_area if total_area > 0 else 0.0
            # Union-rect center can fall on background for an L-shaped merge; use the
            # centroid of the largest member -- guaranteed to sit on real foreground.
            mcX, mcY = centroids[max(range(len(members)), key=lambda k: areas[k])]
            merged_centers.append((mcX, mcY, mx, my, mw, mh, merged_conf, total_area, member_points))
            merged_boxes.append((mx, my, mw, mh))

        if len(bounding_boxes) > len(merged_boxes):
            print(f"Merged {len(bounding_boxes)} fragment boxes into {len(merged_boxes)} object(s) at frame {frame_idx}")

        current_frame_centers = merged_centers
        bounding_boxes = merged_boxes

        matched_this_frame = set()
        visual_frame = waft_frames[frame_idx].copy()
        # Match found motion regions to existing tracks, or spawn new IDs (e.g., flyaway hat)
        for (cX, cY, x, y, w, h, conf, area, member_points) in current_frame_centers:
            matched_obj_id: int | None = None
            best_track = None
            best_score = -1.0  # We want to MAXIMIZE IoU/Overlap instead of minimizing distance

            # Spatial distance check to see if this point belongs to an existing target
            for track in active_tracks:
                if track["obj_id"] in matched_this_frame:
                    continue # Skip si cette piste a déjà trouvé son centre dans cette frame

                # 1. Compute IoU between current box and track's last box
                tx, ty, tw, th = track["last_box"]

                # Determine intersection coordinates
                ix1 = max(x, tx)
                iy1 = max(y, ty)
                ix2 = min(x + w, tx + tw)
                iy2 = min(y + h, ty + th)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih

                # Union
                union = (w * h) + (tw * th) - intersection
                iou = intersection / union if union > 0 else 0.0

                # 2. Fallback: If no direct overlap (fast motion), check normalized center distance
                if iou == 0:
                    dist = np.sqrt((cX - track["last_center"][0])**2 + (cY - track["last_center"][1])**2)
                    # Normalize distance by the diagonal of the tracking box so threshold scales with object size
                    max_dim = max(tw, th, prox_threshold)
                    if dist < max_dim:
                        # Map distance to a pseudo-score between 0 and 1 (closer = higher score)
                        score = 1.0 - (dist / max_dim)
                    else:
                        score = -1.0
                else:
                    # Prioritize overlap over pure distance
                    score = iou + 1.0

                # Find the track with the highest match score
                if score > best_score:
                    best_score = score
                    matched_obj_id = track["obj_id"]
                    best_track = track

            occurence = {"conf": conf, "idx": frame_idx, "center": (cX, cY), "box": (x, y, w, h), "member_points": member_points}
            # Si match, on met à jour l'objet existant in-place
            if best_track is not None:  # Sécurité : on vérifie directement l'objet
                best_track["last_center"] = (cX, cY)
                best_track["last_box"] = (x, y, w, h)  # Update box dimensions
                best_track["seen"].append(occurence)
                matched_this_frame.add(matched_obj_id)
            # Si aucun track ne correspond, SEULEMENT ICI on spawn
            else:
                matched_obj_id = next_available_obj_id
                active_tracks.append({
                    "obj_id": matched_obj_id,
                    "last_center": (cX, cY),
                    "last_box": (x, y, w, h),  # Update box dimensions
                    "seen": [occurence]
                })
                next_available_obj_id += 1

            # UI visualization: assign colors based on unique object tracking IDs
            color = (int((matched_obj_id * 85) % 255), int((matched_obj_id * 130) % 255), int((matched_obj_id * 45) % 255)) # type: ignore
            cv2.rectangle(visual_frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(visual_frame, (cX, cY), 6, (0, 0, 255), -1)
            cv2.putText(visual_frame, f"ID: {matched_obj_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(output_path / f"frames/frame_{frame_idx:04d}.jpg"), np.vstack((visual_frame, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))))
        boxes_preview.write(visual_frame)

    boxes_preview.release()
    reencode_h264(str(output_path / 'boxes_input_sam.mp4'))
    print(f"-> Diagnostic preview saved: {output_path}")

    # Cross-track de-dup: merge_gap_px above only fuses fragments WITHIN one
    # frame; a hand/head blob that repeatedly detaches from the torso blob
    # becomes two independent tracks instead (confirmed: MattSwift 2 people ->
    # 4 obj_ids). Fold tracks whose boxes overlap heavily (by containment, not
    # IoU -- nested boxes have low IoU) at most shared frames into one.
    def _containment(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix1, iy1 = max(x1, x2), max(y1, y2)
        ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        smaller = min(w1 * h1, w2 * h2)
        return inter / smaller if smaller > 0 else 0.0

    def _gap(b1, b2):
        # Same edge-to-edge rect distance as _box_gap above (0 if overlapping).
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
        dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
        return max(dx, dy)

    def _size_ratio_ok(b1, b2, min_ratio):
        # _containment (inter/smaller-area) and _gap (0 whenever boxes touch)
        # are both blind to scale: a small box fully nested inside -- or merely
        # touching -- a much larger one reads as a "duplicate" either way. That's
        # correct for a real single-object fragment (a head box nested in its own
        # wholebody box, comparable overall scale over the track's lifetime) but
        # wrong for two genuinely distinct, differently-sized objects that happen
        # to overlap (e.g. a "person" text detection sitting on/in a much larger
        # "forklift" flow track) -- confirmed root cause of the forklift going
        # untracked on circulation_site_1_edit_coupe: the driver's small person
        # box "covered" (per containment/gap) the forklift's large flow box, so
        # the flow-detected forklift track was discarded as an already-covered
        # duplicate. Require comparable area before containment/gap can dedup.
        a1 = max(0.0, b1[2]) * max(0.0, b1[3])
        a2 = max(0.0, b2[2]) * max(0.0, b2[3])
        if a1 <= 0 or a2 <= 0:
            return False
        return min(a1, a2) / max(a1, a2) >= min_ratio

    # Fragments of ONE object are often spatially DISJOINT (why merge_gap_px
    # failed): high containment is one dup signal, a small persistent GAP is
    # the other (the one that fires on MattSwift). Flag on EITHER signal.
    # SPAG_TRACK_DEDUP_PROFILE picks all 5 thresholds below at once -- no
    # individual per-threshold override anymore (removed 2026-09-04 to cut
    # flag count; see docs/SPAG_ENV_FLAGS.md). Only "default" is
    # production-validated (it's the prior individual defaults, unchanged);
    # "strict"/"loose" are directional presets, not yet benchmarked.
    _dedup_profiles = {
        "strict": {"GAP": 20.0, "OVERLAP": 0.75, "SIZE_RATIO": 0.5, "MIN_COOCCUR": 3, "IDX_TOL": 2},
        "default": {"GAP": 40.0, "OVERLAP": 0.6, "SIZE_RATIO": 0.3, "MIN_COOCCUR": 2, "IDX_TOL": 3},
        "loose": {"GAP": 60.0, "OVERLAP": 0.45, "SIZE_RATIO": 0.2, "MIN_COOCCUR": 1, "IDX_TOL": 5},
    }
    _dedup_base = _dedup_profiles[os.environ.get("SPAG_TRACK_DEDUP_PROFILE", "default")]
    dup_gap_thresh = _dedup_base["GAP"]
    dup_overlap_thresh = _dedup_base["OVERLAP"]
    # Guards the text<->flow and text<->text coverage dedups below (NOT the
    # track-vs-track fragment-fusion dedup a few lines down, which must keep
    # fusing very differently-sized fragments of one real object, e.g. a head
    # box and its own wholebody box). See _size_ratio_ok for the forklift/driver
    # bug this closes.
    dup_size_ratio_thresh = _dedup_base["SIZE_RATIO"]
    min_cooccur = _dedup_base["MIN_COOCCUR"]
    # Ghost tracks rarely share an EXACT frame index (matched_this_frame blocks
    # both from claiming one frame, so they alternate). Match nearest-in-time
    # occurrence within this many decimated-frame steps instead of idx equality.
    idx_tolerance = _dedup_base["IDX_TOL"]

    n_tracks = len(active_tracks)
    dedup_parent = list(range(n_tracks))

    def _dedup_find(a):
        while dedup_parent[a] != a:
            dedup_parent[a] = dedup_parent[dedup_parent[a]]
            a = dedup_parent[a]
        return a

    def _dedup_union(a, b):
        ra, rb = _dedup_find(a), _dedup_find(b)
        if ra != rb:
            dedup_parent[ra] = rb

    def _nearest_matches(occ_a, occ_b, tol):
        # occ_a/occ_b: sorted list of (idx, box). For each occurrence in the
        # shorter list, pair it with the closest-in-time occurrence in the
        # other list if within `tol` decimated frames.
        pairs = []
        idxs_b = [o[0] for o in occ_b]
        for idx_a, box_a in occ_a:
            j = bisect.bisect_left(idxs_b, idx_a)
            best = None
            for cand in (j - 1, j):
                if 0 <= cand < len(occ_b):
                    d = abs(occ_b[cand][0] - idx_a)
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, occ_b[cand][1])
            if best is not None:
                pairs.append((box_a, best[1]))
        return pairs

    for a in range(n_tracks):
        occ_a = sorted((s['idx'], s['box']) for s in active_tracks[a]['seen'])
        for b in range(a + 1, n_tracks):
            occ_b = sorted((s['idx'], s['box']) for s in active_tracks[b]['seen'])
            shorter, longer = (occ_a, occ_b) if len(occ_a) <= len(occ_b) else (occ_b, occ_a)
            pairs = _nearest_matches(shorter, longer, idx_tolerance)
            if len(pairs) < min_cooccur:
                continue
            mean_ov = sum(_containment(ba, bb) for ba, bb in pairs) / len(pairs)
            mean_gap = sum(_gap(ba, bb) for ba, bb in pairs) / len(pairs)
            if mean_ov >= dup_overlap_thresh or mean_gap <= dup_gap_thresh:
                _dedup_union(a, b)

    # Second signal: re-identification via anchor proximity. A track that
    # drifts over its lifetime can miss the pass above even when its
    # REGISTRATION anchor is coincident with another's (confirmed on
    # MattSwift). Compare FIRST/LAST occurrences instead of whole trajectories,
    # gated by a frame-gap cap so unrelated objects aren't fused.
    reid_max_gap = 40
    for a in range(n_tracks):
        seen_a = sorted(active_tracks[a]['seen'], key=lambda s: s['idx'])
        if not seen_a:
            continue
        anchors_a = [seen_a[0], seen_a[-1]]
        for b in range(a + 1, n_tracks):
            if _dedup_find(a) == _dedup_find(b):
                continue
            seen_b = sorted(active_tracks[b]['seen'], key=lambda s: s['idx'])
            if not seen_b:
                continue
            anchors_b = [seen_b[0], seen_b[-1]]
            for sa in anchors_a:
                for sb in anchors_b:
                    time_gap = abs(sa['idx'] - sb['idx'])
                    if time_gap > reid_max_gap:
                        continue
                    spatial_gap = _gap(sa['box'], sb['box'])
                    spatial_ov = _containment(sa['box'], sb['box'])
                    if spatial_ov >= dup_overlap_thresh or spatial_gap <= dup_gap_thresh:
                        _dedup_union(a, b)
                        break
                else:
                    continue
                break

    dedup_groups: dict[int, list[int]] = {}
    for idx in range(n_tracks):
        dedup_groups.setdefault(_dedup_find(idx), []).append(idx)

    deduped_tracks = []
    for members in dedup_groups.values():
        if len(members) == 1:
            deduped_tracks.append(active_tracks[members[0]])
            continue
        members_sorted = sorted(members, key=lambda m: len(active_tracks[m]['seen']), reverse=True)
        senior = active_tracks[members_sorted[0]]
        seen_by_idx = {s['idx']: s for s in senior['seen']}
        for m in members_sorted[1:]:
            junior = active_tracks[m]
            print(f"[Track dedup] Folding object {junior['obj_id']} into {senior['obj_id']} "
                  f"({len(junior['seen'])} occurrences, mean overlap over shared frames)")
            for s in junior['seen']:
                fi = s['idx']
                # On a frame both tracks share, keep whichever box is larger --
                # the whole-body box is the more useful SAM3 prompt than the
                # fragment's.
                if fi not in seen_by_idx or (s['box'][2] * s['box'][3]) > (
                    seen_by_idx[fi]['box'][2] * seen_by_idx[fi]['box'][3]
                ):
                    seen_by_idx[fi] = s
        senior['seen'] = [seen_by_idx[k] for k in sorted(seen_by_idx)]
        deduped_tracks.append(senior)
    if len(deduped_tracks) < len(active_tracks):
        print(f"[Track dedup] {len(active_tracks)} tracks -> {len(deduped_tracks)} after de-duplication")
    active_tracks = deduped_tracks

    # Occlusion gate (opt-in, SPAG_OCCL_FBGATE=1): if a track's first-seen frame
    # is an occlusion event, flow is unreliable and SAM3 anchors badly -> drift.
    # When enabled, anchor at the first occurrence whose FB-error is below
    # `fb_gate_thresh`, falling back to the plain first occurrence otherwise.
    occl_fbgate = (
        os.environ.get("SPAG_OCCL_FBGATE", "0") == "1"
        and flows_fwd is not None and flows_bwd is not None
    )
    # Default 3.0px: on trop_long_embouteillage this is the win-only regime --
    # recovers the two genuinely-occluded tracks (+797/+746 present-frames) while
    # marginal 1.5px skips snap back to baseline. Best net +1198 (+5.4%). See D1.
    fb_gate_thresh = float(os.environ.get("SPAG_OCCL_FB_THRESH", "3.0"))
    # Max occurrences a re-anchor may skip (cost cap; resolves the obj-1 caveat -- see
    # the gate body below for the mechanism).
    max_reanchor_skip = int(os.environ.get("SPAG_OCCL_MAX_SKIP", "8"))
    _fb_err_cache: dict = {}

    def _occurrence_fb_err(s):
        fi = s['idx']
        if fi >= len(flows_fwd) or fi >= len(flows_bwd):
            return 0.0  # last frame has no forward pair -> treat as clean
        if fi not in _fb_err_cache:
            _fb_err_cache[fi] = fb_consistency_error(flows_fwd[fi], flows_bwd[fi])
        fb = _fb_err_cache[fi]
        Hf, Wf = fb.shape[:2]
        x, y, w, h = s['box']  # new_W/new_H space
        sx, sy = Wf / meta['new_W'], Hf / meta['new_H']
        x0, y0 = int(x * sx), int(y * sy)
        x1, y1 = int((x + w) * sx), int((y + h) * sy)
        x0, x1 = max(0, x0), min(Wf, max(x0 + 1, x1))
        y0, y1 = max(0, y0), min(Hf, max(y0 + 1, y1))
        roi = fb[y0:y1, x0:x1]
        return float(roi.mean()) if roi.size else 0.0

    # SAM3 text-prompt registration -- MUST run before flow-based point
    # registration: text add_prompt resets session state, wiping objects
    # already registered via points. Purpose: flow analysis misses standing
    # people and fragments a moving person into separate obj_ids; text
    # detection gives one box per person regardless of motion. Seeds points
    # into the same path the flow loop uses so both propagate together below.

    # Common subjects likely to appear and be worth tracking in an arbitrary
    # video; kept short since each entry costs one extra SAM3 forward pass
    # (see the two-phase note below) and a broad open-vocab prompt like
    # "vehicle"/"animal" already covers most unlisted subclasses.
    text_prompts = ["person", "dog", "cat", "car", "bicycle", "motorcycle", "animal", "vehicle", "forklift", "truck"]
    text_obj_boxes: list[tuple[int, list[float]]] = []  # (obj_id, abs xywh box in meta['new_W']/new_H space)

    def _points_from_mask(mask_np, n_bands: int = 4):
        # Centroid + height-band medians over the FULL y-range, in [0,1]
        # fractions of the mask's own shape. Bands must include y0/y1: a head
        # sits at y0, and without a point there Phase-2 points-reregistration
        # undersegments and drops it (repro on Dispo_RDV: mask top y=475 -> 601).
        ys, xs = np.where(mask_np)
        if len(xs) == 0:
            return []
        mh, mw = mask_np.shape[:2]
        pts = [[float(xs.mean()) / mw, float(ys.mean()) / mh]]
        y0, y1 = int(ys.min()), int(ys.max())
        for i in range(n_bands + 1):
            row_y = y0 + int((y1 - y0) * i / n_bands)
            row_xs = xs[ys == row_y]
            if len(row_xs) == 0:
                continue
            pts.append([float(np.median(row_xs)) / mw, float(row_y) / mh])
        return pts

    if text_prompts:
        # Phase 1: run every text prompt FIRST, before registering any via
        # points (each add_prompt resets state, so interleaving would wipe
        # earlier registrations). bf16 autocast makes borderline detections
        # flip run-to-run (process-level cudnn-autotune effect, confirmed on
        # MattSwift), so retrying the same frame is a dead end -- sample a
        # couple of DIFFERENT anchor frames and dedup by box position instead.
        anchor_frames = sorted({0, min((n_total_frames // 2) * skip_step, max(0, n_total_frames * skip_step - 1))})
        pending: list[tuple[str, list[float], list, int]] = []  # (prompt, box_abs, points, anchor_frame)

        def _register_text_detections(prompt, out, n_det, anchor_frame):
            # Dedup a NEW box only against entries pending BEFORE this call --
            # detections from ONE call are already distinct per SAM3's own NMS
            # and must not be folded together by dup_gap_thresh (tuned for
            # merging fragments of a SINGLE object across flow tracks).
            baseline = list(pending)
            for i in range(n_det):
                mask = out["out_binary_masks"][i]
                mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                ys, xs = np.where(mask_np)
                if len(xs) == 0:
                    continue
                mh, mw = mask_np.shape[:2]
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                # Scale the box up to meta['new_W']/new_H space (see
                # _points_from_mask note above) so it's directly comparable
                # to flow track boxes (s['box']) for the IoU dedup below.
                sx, sy = meta['new_W'] / mw, meta['new_H'] / mh
                box_abs = [x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy]
                points = _points_from_mask(mask_np)
                if not points:
                    continue
                # Dedup against everything pending BEFORE this call -- catches
                # both overlapping vocab (e.g. "dog" and "animal" firing on
                # the same instance) and repeat detections of the same
                # instance across retries of the same prompt.
                dup = next(
                    (p for _pr, p, _pts, _af in baseline
                     if _size_ratio_ok(box_abs, p, dup_size_ratio_thresh)
                     and (_containment(box_abs, p) >= dup_overlap_thresh or _gap(box_abs, p) <= dup_gap_thresh)),
                    None,
                )
                if dup is not None:
                    continue
                pending.append((prompt, box_abs, points, anchor_frame))

        for anchor_frame in anchor_frames:
            for prompt in text_prompts:
                resp = video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=anchor_frame,
                        text=prompt,
                    )
                )
                out = resp["outputs"]
                n_det = len(out.get("out_obj_ids", []))
                if n_det == 0:
                    print(f"[By text] no '{prompt}' detections at frame {anchor_frame}", flush=True)
                    continue
                _register_text_detections(prompt, out, n_det, anchor_frame)

        # Phase 2: register every pending detection via points now that all
        # text resets are done. None of these calls reset state.
        for prompt, box_abs, points, anchor_frame in pending:
            obj_id = next_available_obj_id
            next_available_obj_id += 1
            point_labels = [1] * len(points)
            tqdm.write(f"[By text] '{prompt}' object {obj_id} registered at frame {anchor_frame} (box={[round(v) for v in box_abs]})")
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=anchor_frame,
                    obj_id=obj_id,
                    points=points,
                    point_labels=point_labels,
                )
            )
            text_obj_boxes.append((obj_id, box_abs))
            if start_frame_index is None:
                start_frame_index = anchor_frame

    registered_anchor_frames: list = []
    found_object = len(active_tracks) + len(text_obj_boxes)
    text_covered_ids: set = set()
    for track in active_tracks:
        seens = track['seen']
        if len(seens) < min_times_seen:
            print(f"Skip object {track['obj_id']} because seen less than {min_times_seen} times", flush=True)
            found_object -= 1
            # active_tracks.remove(track)
            continue

        if text_obj_boxes:
            first_box = seens[0]['box']
            covered = next(
                (tid for tid, tbox in text_obj_boxes
                 if _size_ratio_ok(first_box, tbox, dup_size_ratio_thresh)
                 and (_containment(first_box, tbox) >= dup_overlap_thresh or _gap(first_box, tbox) <= dup_gap_thresh)),
                None,
            )
            if covered is not None:
                print(f"Skip object {track['obj_id']} because text-prompt object {covered} already covers it", flush=True)
                found_object -= 1
                text_covered_ids.add(track['obj_id'])
                continue

        if occl_fbgate:
            # Reorder so the first non-occluded occurrence is the registration
            # anchor; keep original order among the rest as a fallback.
            errs = [_occurrence_fb_err(s) for s in seens]
            if os.environ.get("SPAG_OCCL_DEBUG", "0") == "1":
                tqdm.write(f"[Occl gate][dbg] object {track['obj_id']} FB-err/occurrence "
                           f"(frame:err): " + ", ".join(
                               f"{s['idx']*skip_step}:{e:.2f}" for s, e in zip(seens, errs)))
            clean = [s for s, e in zip(seens, errs) if e < fb_gate_thresh]
            if clean and clean[0] is not seens[0]:
                skipped = seens.index(clean[0])
                # Second signal: re-anchoring DISCARDS every occurrence before the
                # clean one. A well-tracked-but-noisy object racks up many skipped
                # occurrences (real coverage lost); FB-error alone can't tell that
                # apart from genuine occlusion, so cap the skip (SPAG_OCCL_MAX_SKIP).
                if skipped > max_reanchor_skip:
                    if os.environ.get("SPAG_OCCL_DEBUG", "0") == "1":
                        tqdm.write(f"[Occl gate] object {track['obj_id']}: first occurrence "
                                   f"FB-err {errs[0]:.2f}>={fb_gate_thresh}px but clean anchor is "
                                   f"{skipped} occ away (>{max_reanchor_skip}); object is tracked-"
                                   f"but-noisy, keeping original anchor (no re-anchor)")
                else:
                    tqdm.write(f"[Occl gate] object {track['obj_id']}: first {skipped} occurrence(s) "
                               f"occluded (FB-err {errs[0]:.2f}>={fb_gate_thresh}px), anchoring at frame "
                               f"{clean[0]['idx'] * skip_step} (FB-err {_occurrence_fb_err(clean[0]):.2f}) instead")
                    seens = clean + [s for s in seens if s not in clean]

        for s in seens:
            matched_obj_id = track['obj_id']
            # s['idx'] is a DECIMATED flow-analysis index; the SAM3 session runs on the
            # NATIVE video, so convert to native frame space (== s['idx'] when skip_step==1).
            frame_idx = s['idx'] * skip_step
            box = abs_to_rel_coords([list(s['box'])], meta['new_W'], meta['new_H'], coord_type="box")
            # Prompt SAM3 with several real, mask-derived points spread across the
            # object (centroid + height-band samples from each merged fragment)
            # instead of a single centroid -- a lone point tends to grow a mask
            # around whichever part is most salient nearby (e.g. torso) rather
            # than the whole silhouette, even for an elongated standing/prone body.
            points = abs_to_rel_coords([list(p) for p in s["member_points"]], meta['new_W'], meta['new_H'])
            point_labels = [1] * len(s["member_points"])
            # Instantly register this new independent entity directly to SAM 3
            tqdm.write(f"[By flow] Moving object {matched_obj_id} registered at frame {frame_idx} (Coords: {str(s['center'])}) with conf={s['conf']}, seen {len(seens)} times, {len(points)} points")
            o = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    obj_id=matched_obj_id,
                    points=points,
                    point_labels=point_labels,
                    # bounding_boxes=box,
                    # bounding_box_labels=[1],
                    # text="visual",
                )
            )
            if start_frame_index is None:
                start_frame_index = frame_idx
            registered_anchor_frames.append(frame_idx)
            break #register first time seen only

    # (b) Occlusion-gate collateral damping: re-anchoring a track can move which
    # track is registered first and hence start_frame_index (the origin of the
    # bidirectional propagation), which perturbs *other*, non-re-anchored tracks'
    # schedules. Pin the propagation origin to the EARLIEST registered anchor so
    # it no longer depends on iteration order / which track got re-anchored.
    # Gated on the flag so the baseline (gate off) stays byte-identical.
    if occl_fbgate and registered_anchor_frames:
        start_frame_index = min(registered_anchor_frames)

    # STEP B: Propagate All Interleaved Multi-Object Tracks
    print(f"\n--- Propagating {found_object} moving object starting at {start_frame_index} ---")

    # Combined preview: motion-detection binary mask (top) and the colored ID
    # mask alpha-blended over the real source frame (bottom) -- previously only
    # produced post-hoc in build_medias.py's build_mask_vs_source() via an
    # ffmpeg colorkey overlay; now baked directly into the main run so every
    # clip gets it for free.
    mask_writer = cv2.VideoWriter(
        str(output_path / 'sam3_masks_preview.mp4'), fourcc, meta['fps'],
        (meta['new_W'], meta['new_H'] * 2),
    )
    np.random.seed(42)  # Pour garder les mêmes couleurs d'une frame à l'autre
    _colored_flow_tracks = [
        t for t in active_tracks
        if len(t['seen']) >= min_times_seen and t['obj_id'] not in text_covered_ids
    ]
    STATIC_COLORS = np.random.randint(0, 255, size=(max(1, len(_colored_flow_tracks) + len(text_obj_boxes)), 3), dtype=np.uint8)
    color_map = {
        track['obj_id']: STATIC_COLORS[i].tolist()
        for i, track in enumerate(_colored_flow_tracks)
    }
    color_map.update({
        obj_id: STATIC_COLORS[len(color_map) + i].tolist()
        for i, (obj_id, _box) in enumerate(text_obj_boxes)
    })

    outputs_per_frame = {}
    # propagate_in_video("both") streams responses in traversal order (forward
    # then backward from start_frame_index), not ascending native-frame order --
    # writing straight to mask_writer as responses arrive puts the preview mp4's
    # frames out of temporal order, so overlaying it against the (correctly
    # ordered) source video in build_medias.py shows a growing shift/offset.
    # Buffer by native frame_idx and flush to the writer in sorted order instead.
    mask_frames_by_idx = {}

    def _reseed_window(anchor_idx, target_idx):
        # Tier 4.2 chunking: a fresh propagate_in_video call on the same session
        # does NOT reliably continue tracking a registered object -- confirmed
        # via debug run, object silently lost at window boundaries. Re-prompt
        # each object with a centroid from its last non-empty mask instead.
        anchor_out = outputs_per_frame.get(anchor_idx)
        if anchor_out is None:
            return
        for i, obj_id in enumerate(anchor_out['out_obj_ids']):
            mask = anchor_out['out_binary_masks'][i]
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            mh, mw = mask.shape[:2]
            cx, cy = float(xs.mean()) / mw, float(ys.mean()) / mh
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=target_idx,
                    obj_id=int(obj_id),
                    points=[[cx, cy]],
                    point_labels=[1],
                )
            )

    def _consume_response(response):
        frame_idx = response["frame_index"]
        frame_output = response["outputs"]
        outputs_per_frame[frame_idx] = frame_output

        if frame_idx % 50 == 0:
            print_gpu_stats(f'in propagate : step {frame_idx}')

        # frame_output contains a mapping of {obj_id: mask_data}
        assert isinstance(frame_output, dict)

        # Blank canvas matching target dimensions
        accumulated_mask = np.zeros((meta['new_H'], meta['new_W']), dtype=np.uint8)
        separated_mask = np.zeros((meta['new_H'], meta['new_W'], 3), dtype=np.uint8)

        # Merge ALL independent object mask layers tracking across the frame session
        for idx, mask in enumerate(frame_output['out_binary_masks']):
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255

            if mask.shape[:2] != (meta['new_H'], meta['new_W']):
                mask = cv2.resize(mask, (meta['new_W'], meta['new_H']), interpolation=cv2.INTER_NEAREST)

            # Combine via logical OR so any moving item (person OR hat) gets added
            accumulated_mask = cv2.bitwise_or(accumulated_mask, mask)
            # Random color for each mask
            # Récupération de l'ID d'objet réel pour stabiliser la couleur
            obj_id = frame_output['out_obj_ids'][idx]
            color = color_map.get(obj_id, [255, 255, 255])
            separated_mask[mask == 255, :] = color
            boxes_xyhw = frame_output['out_boxes_xywh'][idx]
            x_pixel = int(boxes_xyhw[0] * meta['new_W'])
            y_pixel = int(boxes_xyhw[1] * meta['new_H'])
            w_pixel = int(boxes_xyhw[2] * meta['new_W'])
            h_pixel = int(boxes_xyhw[3] * meta['new_H'])
            cv2.rectangle(separated_mask, (x_pixel, y_pixel), (x_pixel + w_pixel, y_pixel + h_pixel), color, 2)
            cv2.putText(separated_mask, f"ID:{obj_id}", (x_pixel, y_pixel - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # Bottom panel: colored ID mask alpha-blended over the real source frame,
        # so coverage can be checked directly against the actual clip instead of
        # a bare mask. waft_frames is already (new_H, new_W)-sized and BGR, and
        # indexed by DECIMATED (skip_step) index; frame_idx here is native.
        src_frame = waft_frames[min(frame_idx // skip_step, len(waft_frames) - 1)]
        if src_frame.shape[:2] != (meta['new_H'], meta['new_W']):
            src_frame = cv2.resize(src_frame, (meta['new_W'], meta['new_H']))
        has_mask = np.any(separated_mask != 0, axis=-1)
        overlay_frame = src_frame.copy()
        alpha = 0.6
        overlay_frame[has_mask] = (
            (1 - alpha) * src_frame[has_mask].astype(np.float32)
            + alpha * separated_mask[has_mask].astype(np.float32)
        ).astype(np.uint8)

        # Save combined black-and-white mask + source overlay stacked
        combined_frame = np.concatenate(
            [np.stack([accumulated_mask] * 3, axis=-1), overlay_frame], axis=0
        )
        mask_frames_by_idx[frame_idx] = combined_frame
        return frame_idx

    # Tier 4.2 (opt-in, SPAG_SAM3_MAX_FRAMES=<int>): bound per-call VRAM/memory-bank
    # growth by chunking propagate_in_video into successive windows instead of one
    # unbounded "both" call. A single capped call leaves every frame outside the
    # cap's reach with NO SAM3 output (confirmed: 75-frame test clip, cap=20
    # covered only 23/38 sampled frames). Chunking re-issues propagate_in_video
    # repeatedly on the SAME session (memory bank / tracked-object state carries
    # over), each call bounded to max_frame_num_to_track, walking forward from
    # start_frame_index to the last needed frame and backward to 0 -- covering
    # every needed frame with real tracking while bounding per-call attention.
    _max_frames_env = os.environ.get("SPAG_SAM3_MAX_FRAMES")
    try:
        if _max_frames_env:
            cap = int(_max_frames_env)
            native_last_needed = (n_total_frames - 1) * skip_step
            print(f"[SPAG4D] SAM3 propagation chunked to {cap} frames/window (SPAG_SAM3_MAX_FRAMES), "
                  f"covering native frames 0-{native_last_needed}")

            cur = start_frame_index
            prev_last_idx = None
            while cur <= native_last_needed and cur not in outputs_per_frame:
                if prev_last_idx is not None:
                    _reseed_window(prev_last_idx, cur)
                req = dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    start_frame_index=cur,
                    propagation_direction="forward",
                    output_prob_thresh=0.5,
                    max_frame_num_to_track=cap,
                )
                last_idx = cur - 1
                window_start = cur
                for response in video_predictor.handle_stream_request(request=req):
                    last_idx = max(last_idx, _consume_response(response))
                first_out = outputs_per_frame.get(window_start)
                nonempty = first_out is not None and any(m.any() for m in first_out.get('out_binary_masks', []))
                print(f"[SPAG4D][DEBUG] fwd window start={window_start} end={last_idx} "
                      f"first-frame-nonempty={nonempty}")
                if last_idx < cur:
                    break  # no progress this window -- avoid an infinite loop
                prev_last_idx = last_idx
                cur = last_idx + 1

            cur = start_frame_index - 1
            prev_first_idx = None
            while cur >= 0 and cur not in outputs_per_frame:
                if prev_first_idx is not None:
                    _reseed_window(prev_first_idx, cur)
                req = dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    start_frame_index=cur,
                    propagation_direction="backward",
                    output_prob_thresh=0.5,
                    max_frame_num_to_track=cap,
                )
                first_idx = cur + 1
                for response in video_predictor.handle_stream_request(request=req):
                    first_idx = min(first_idx, _consume_response(response))
                if first_idx > cur:
                    break  # no progress this window
                prev_first_idx = first_idx
                cur = first_idx - 1

            covered = sorted(outputs_per_frame.keys())
            needed = [idx * skip_step for idx in range(n_total_frames)]
            n_missing = sum(1 for fi in needed if fi not in outputs_per_frame)
            print(f"[SPAG4D] SAM3 chunked coverage: {len(covered)} real frames tracked, "
                  f"{n_missing}/{len(needed)} needed frames still missing after chunking")
            # Any frame still missing (e.g. object left frame / occluded at a
            # window boundary and was never re-registered) falls back to the
            # nearest tracked frame's output rather than crashing downstream.
            if covered:
                for fi in needed:
                    if fi not in outputs_per_frame:
                        nearest = min(covered, key=lambda c: abs(c - fi))
                        outputs_per_frame[fi] = outputs_per_frame[nearest]
                        if nearest in mask_frames_by_idx:
                            mask_frames_by_idx[fi] = mask_frames_by_idx[nearest]
        else:
            propagate_request = dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=start_frame_index,
                propagation_direction="both", #both
                output_prob_thresh=0.5, #0.5
            )
            for response in video_predictor.handle_stream_request(
                request=propagate_request
            ):
                _consume_response(response)

        outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
        # Tier 4.3 (SPAG_SAM3_SCALE<1.0): masks come back at the downscaled
        # SAM3 input resolution, but downstream (run_video's per-frame depth
        # loop) fuses them against the OUTER-SCOPE, full-native-resolution
        # `viz_frames` array (meta['H'], meta['W']) -- not this function's own
        # `waft_frames` param (WAFT-scaled) and not meta['new_H']/new_W (the
        # WAFT/mask working resolution, a different knob). Confirmed by two
        # real crashes against each wrong target before landing on native
        # meta['H']/meta['W'].
        #
        # Upsampling used to happen here, eagerly, for every object in every
        # frame -- but that means holding a full-native-resolution mask array
        # per object per frame simultaneously for the whole clip (dominant RAM
        # buffer on several benchmark clips, see
        # docs/RAM_USAGE_INVESTIGATION_PLAN.md). Masks are now left at SAM3's
        # working resolution here; run_video's per-frame loop upsamples each
        # mask lazily, once, at the point it's actually consumed.
    except Exception as e:
        print(f"Error during propagation: {e}")
    else:
        print("\nSuccess! All multi-object segments flattened cleanly")
    finally:
        with open(str(output_path / "gpu_stats.txt"), "a") as f:
            print_gpu_stats('Finish propagate', file=f)

        # Close the session to free GPU memory
        _ = video_predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        video_predictor.shutdown()

        # Flush buffered preview frames in ascending native-frame order (see
        # mask_frames_by_idx comment above) before closing the writer.
        for _fi in sorted(mask_frames_by_idx):
            mask_writer.write(mask_frames_by_idx[_fi])
        mask_writer.release()
        reencode_h264(str(output_path / 'sam3_masks_preview.mp4'))

    return outputs_per_frame

def segment_with_sam(
    video_path: str,
    viz_frames: np.ndarray,
    n_total_frames: int,
    skip_step: int = 10,
    enable_viz: bool = False,
):
    # On prépare SAM3 vidéo
    from sam3.model_builder import build_sam3_video_predictor
    from sam3.visualization_utils import (
        prepare_masks_for_visualization,
    )
    # use all available GPUs on the machine
    # gpus_to_use = range(torch.cuda.device_count())
    # # use only a single GPU
    gpus_to_use = [torch.cuda.current_device()]
    video_predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # Start a session
    initial_response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(video_path),
            offload_video_to_cpu=True,
            # See segment_with_flows: offload the SAM3 inference-state memory bank
            # to CPU to cut the dominant GPU peak (~10-15% fps cost, lossless).
            offload_state_to_cpu=True,
        )
    )
    session_id = initial_response["session_id"]
    print("SAM3 video session started with session_id:", session_id)

    # Find first instance
    prompt = "human"
    frame_idx = 0
    n = 0
    print(f"SAM3 video is looking for : '{prompt}' ...")
    while (n == 0) and (frame_idx < n_total_frames):
        # We call this 'frame_response' so we don't overwrite our session variables
        frame_response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,  # Uses the permanent ID variable
                frame_index=frame_idx,
                text=prompt,
            )
        )
        out = frame_response.get("outputs", {})
        # Safely get length of probabilities array if it exists
        n = len(out["out_probs"]) if "out_probs" in out and out["out_probs"] is not None else 0
        if n <= 0:
            print(
                f"\r🔍 Searching... Nothing found in frame {frame_idx} with prompt '{prompt}'",
                end="",
                flush=True,
            )
            frame_idx += 1
        else:
            print(f"🎯 Success! Found {n} target(s) matching '{prompt}' at frame {frame_idx}!")

    if n == 0 and frame_idx == n_total_frames:
        print(f"\n\nCan't found any {prompt} in any frame of this video {str(video_path)}")
        exit()

        plt.close("all")
        visualize_formatted_frame_output(
            frame_idx,
            viz_frames,
            outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )

    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    # Tier 4.2 (opt-in, SPAG_SAM3_MAX_FRAMES=<int>): see segment_with_flows for
    # the full rationale. Unlike segment_with_flows, this function's default
    # propagate call is unidirectional forward from `frame_idx`, so chunking
    # here is forward-only: repeatedly call propagate_in_video on the SAME
    # session, each bounded to max_frame_num_to_track, walking forward until
    # every needed native frame index (idx * skip_step) is actually tracked,
    # instead of leaving everything past a single capped call with no output.
    _max_frames_env = os.environ.get("SPAG_SAM3_MAX_FRAMES")
    needed = [idx * skip_step for idx in range(n_total_frames)]
    native_last_needed = needed[-1] if needed else 0
    if _max_frames_env:
        cap = int(_max_frames_env)
        print(f"[SPAG4D] SAM3 propagation chunked to {cap} frames/window (SPAG_SAM3_MAX_FRAMES), "
              f"covering native frames {frame_idx}-{native_last_needed}")
        cur = frame_idx
        while cur <= native_last_needed and cur not in outputs_per_frame:
            req = dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=cur,
                propagation_direction="forward",
                max_frame_num_to_track=cap,
            )
            last_idx = cur - 1
            for response in video_predictor.handle_stream_request(request=req):
                fi = response["frame_index"]
                outputs_per_frame[fi] = response["outputs"]
                last_idx = max(last_idx, fi)
            if last_idx < cur:
                break  # no progress this window
            cur = last_idx + 1
    else:
        propagate_request = dict(
            type="propagate_in_video",
            session_id=session_id,
        )
        for response in video_predictor.handle_stream_request(
            request=propagate_request
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

    # Any frame still missing (e.g. object lost at a window boundary) falls
    # back to the nearest tracked frame's output rather than crashing
    # downstream. NOT lossless, only a safety net.
    if _max_frames_env and outputs_per_frame:
        covered = sorted(outputs_per_frame.keys())
        for fi in needed:
            if fi not in outputs_per_frame:
                nearest = min(covered, key=lambda c: abs(c - fi))
                outputs_per_frame[fi] = outputs_per_frame[nearest]

    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
    if enable_viz:
        plt.close("all")
        for frame_idx in tqdm(range(0, len(viz_frames))):
            visualize_formatted_frame_output(
                frame_idx,
                viz_frames,
                outputs_list=[outputs_per_frame],
                titles=["SAM 3 Dense Tracking outputs"],
                figsize=(6, 4),
                skip_step=skip_step,
            )

    # Close
    _ = video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    return outputs_per_frame
