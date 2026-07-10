import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
from tqdm import tqdm

from .core import SPAG4D, ConversionResult
from .detect_opticalflow import WAFTWrapper, abs_to_rel_coords, reencode_h264
from .ply_writer import save_ply_gsplat
from .scene_analysis import compute_scene_defaults


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


def extract_video_frames(video_path: str, skip_step: int = 10) -> tuple[np.ndarray, int]:
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

    # video_writer = cv2.VideoWriter(  # USED TO CREATE SHORTER VIDEOS FOR DEBUG/TEST
    #     "temp_fast_track.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H)
    # )

    idx = 0
    try:
        with tqdm(total=total_frames, desc="Extracting frames", unit=" frame") as pbar:
            frame_num = 0
            while True:
                # ── frames we KEEP: full decode ──────────────────────────────
                ok, frame_bgr = cap.read()
                if not ok:
                    break

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


def compute_temporal_median(frames_array, quantile: float = 0.33):
    print("Calcul de la médiane temporelle (cela peut prendre quelques secondes)...")
    start = time.time()

    # Stack once — if viz_frames is already an ndarray, this is a no-op
    if not isinstance(frames_array, np.ndarray):
        frames_array = np.stack(frames_array, axis=0)  # avoids extra copy vs np.array()

    # Calcul de la médiane sur l'axe 0 (l'axe du temps)
    start = time.time()
    F, H, W, C = frames_array.shape
    median_frame = np.empty((H, W, C), dtype=frames_array.dtype)
    std_frame = torch.zeros((H, W), dtype=torch.float32)
    for c in range(C):  # Channel by channel to reduce VRAM usage
        channel = torch.from_numpy(frames_array[:, :, :, c])  # (F, H, W)
        median_frame[:, :, c] = torch.median(channel.to("cuda"), dim=0).values.cpu().numpy()
        std_frame += torch.std(channel, dim=0).cpu()  # accumulate std per channel
    med = time.time()
    print(f"Median computed in {(med - start):.1f}s")

    # On peut aussi enregistrer les pixels qui varient le plus pour visualiser les zones les plus dynamiques
    std_frame /= C
    std_max = std_frame.max().item()
    if std_max > 0:
        activity_map = (std_frame * (255.0 / std_max)).byte().numpy()
    else:
        activity_map = np.zeros((H, W), dtype=np.uint8)
    thr = torch.quantile(std_frame, quantile).item()
    activity_mask = ((std_frame > thr).byte() * 255).numpy()
    print(f"min: {std_frame.min()}, {quantile} quantile: {thr:.4f}")
    cv2.imwrite("temporal_activity_map.jpg", activity_map)
    print(f"Temporal_activity_map computed in {(time.time() - med):.1f}s")

    return median_frame, activity_mask


def compute_temporal_median2(frames_array, masks_dict: list[dict], quantile: float = 0.33):
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
        if idx in masks_dict:
            for _, mask in masks_dict[idx].items():
                # Guard against malformed SAM masks (some videos yield a
                # degenerate 1D array instead of an (H, W) mask -> broadcast error)
                if getattr(mask, "shape", None) != (H, W):
                    continue
                masks_array[idx] = np.maximum(masks_array[idx], mask)

    median_frame = np.zeros((H, W, C), dtype=frames_array.dtype)
    median_frame[:, :, 0] = 1.0
    std_frame = np.zeros((H, W), dtype=np.float32)
    for c in range(C):  # Channel by channel to reduce VRAM usage
        channel_tensor = torch.from_numpy(frames_array[..., c]).float().to(device)
        # We temporarily set masked positions to NaN just for the median operation
        median_calc_tensor = channel_tensor.clone()
        median_calc_tensor[masks_array] = float("nan")
        median_frame[:, :, c] = torch.nanmedian(median_calc_tensor, dim=0).values.cpu().numpy()

        # 3. For the STD: Compute normally
        std_frame += torch.std(channel_tensor, dim=0).cpu().numpy()

    # 4. Handle any pixels that were masked out across ALL frames (avoiding NaN artifacts)
    # If a pixel was ALWAYS masked, nanmedian will return NaN. We replace them with 0.
    np.nan_to_num(median_frame, copy=False, nan=0.0)

    # On peut aussi enregistrer les pixels qui varient le plus pour visualiser les zones les plus dynamiques
    std_frame /= C
    thr = np.quantile(std_frame, quantile).item()
    activity_mask = ((std_frame > thr) * 255).astype(np.uint8)

    print(f"Median computed in {(time.time() - start):.1f}s")
    return median_frame, activity_mask


def to_gaussians(
    converter: SPAG4D,
    aligned_depth_np,
    image_tensor,
    stride: int,
    H: int,
    depth_min: float | None = None,
    depth_max: float | None = None,
    sky_threshold: float | None = None,
    outlier_pruning: float = 0.3,
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


def stabilize_fg_depth_scale(
    depth_current: np.ndarray,
    depth_history: list,  # liste de depth_np précédentes (alignées)
    mask: np.ndarray,
    window: int = 3,
    jump_threshold: float = 0.5,  # calibré sur ton graphique : écarts > 0.5 = outlier
) -> np.ndarray:
    """
    Détecte un saut ponctuel de depth médiane sur le masque foreground
    et le remplace par la médiane des frames voisines (sans toucher au fond).
    """
    if len(depth_history) < (window - 1):
        return depth_current

    fg = mask > 0
    if fg.sum() == 0:
        return depth_current

    current_median = np.median(depth_current[fg])
    recent_medians = [np.median(d[fg]) for d in depth_history[-window:]]
    neighbor_median = np.median(recent_medians)

    delta = abs(current_median - neighbor_median)

    if delta > jump_threshold:
        # Outlier ponctuel détecté → correction par ratio global
        scale_correction = neighbor_median / current_median if current_median > 0 else 1.0
        out = depth_current.copy()
        out[fg] *= scale_correction
        return out

    return depth_current


def run_video(
    converter: SPAG4D,
    video_path: str | Path,
    output_path: str | Path,
    active_generator: str = "da360",
    get_background_method: str = "temporal_median",
    alignement_mask: str = "sam_and_activity",  # sam, sam_and_activity, nothing, all
    alignement_method: str = "lstsq",  # lstsq, median, ransac
    quantile: float = 0.33,
    skip_step: int = 10,
    freeze_bg: bool = False,
    depth_min: float | None = None,
    depth_max: float | None = None,
    sky_threshold: float | None = None,
    stride: int = 8,
    outlier_pruning: float = 0.3,
    grazing_angle: float = 85.0,  # 65.0,
    global_scale: float = 1.0,
    sparse_pruning: float = 0.1,  # 0.3,
    depth_preview_path: Path | None = None,
    temporal_consistency: bool = False,
    # --- Solution 1: temporal depth smoothing ---
    depth_smoothing: bool = False,
    depth_smoothing_window: int = 5,
    depth_smoothing_method: str = "median",  # "median" | "gaussian"
    # --- Solution 4: multi-frame depth reference ---
    reference_frames_for_median: int = 1,
    # --- Solution 6: foreground stabilizer tuning ---
    fg_buffer_size: int = 21,
    fg_jump_threshold: float = 0.5,
    # --- Solution 5: mask quality diagnostics (diagnostic only) ---
    mask_diagnostics: bool = False,
):
    """

    Args :
        get_background_method : determine background images use as anchor. Options:
        'first': use first frame (default)
        'last' : use last frame
        'median_temporal': use average color of each frame as background color
    """
    if alignement_mask == "sam_and_activity":
        get_background_method = "temporal_median"

    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    viz_frames, meta = extract_video_frames(str(video_path), skip_step=skip_step)
    n_total_frames = len(viz_frames)

    # Flow
    # Defined up-front so a WAFT failure falls back to SAM cleanly instead of
    # raising UnboundLocalError on `if flow_masks is None` below.
    flow_masks = None
    try:
        model = WAFTWrapper(
            checkpoint = '/raid/mb273924/_DATASETS/uptale/tar-c-t.pth',
            config = '/raid/mb273924/WAFT/config/a1/tar-c-t.json'
        )
        MAX_SIZE = 1024
        scale = min(MAX_SIZE / max(meta["W"], meta["H"]), 1.0)  # never upscale
        if scale < 1.0:
            new_W = int(meta["W"] * scale) & ~1  # force even dimensions (codec requirement)
            new_H = int(meta["H"] * scale) & ~1

            viz_tensor = torch.from_numpy(viz_frames).permute(0, 3, 1, 2).float()
            downscaled = F.interpolate(viz_tensor, size=(new_H, new_W), mode="bilinear")
            # Retour en NHWC uint8 NumPy pour WAFT
            waft_frames = downscaled.permute(0, 2, 3, 1).byte().numpy()
        else:
            new_W, new_H = meta["W"], meta["H"]
            waft_frames = viz_frames
        # Toujours défini : segment_with_flows en dépend même sans downscale
        meta["new_W"] = new_W
        meta["new_H"] = new_H

        stats = model.run(waft_frames, meta, output_folder, 0.5)
        try:
            reencode_h264(str(output_folder / "flow_mask.mp4"))
            reencode_h264(str(output_folder / "flow_mag.mp4" ))
        except Exception:
            pass
        flow_masks = stats["masks"]
    except Exception as e:
        print(f"  [ERROR during inference] {e}")
        import traceback
        traceback.print_exc()
    # bgr to rgb
    viz_frames = viz_frames[:, :, :, ::-1].copy()
    # SAM
    if alignement_mask in ["sam", "sam_and_activity", "sam_and_flow", "nothing"]:
        startsam = time.time()
        if flow_masks is None:
            outputs_per_frame = segment_with_sam(str(video_path), viz_frames, n_total_frames, skip_step)
        else:
            outputs_per_frame = segment_with_flows(str(video_path), output_folder, waft_frames, flow_masks, meta, n_total_frames, skip_step)
        print(f"[SPAG4D] Background/Front segmentation of video completed in {(time.time() - startsam):.2f}s")

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
    if get_background_method == "temporal_median":
        master_background, activity_mask = compute_temporal_median2(viz_frames, outputs_per_frame, quantile)
        cv2.imwrite(output_folder / f"_temporal_activity_mask_thr_{f'{quantile:.2f}'.split('.')[-1]}.jpg", activity_mask)
    elif get_background_method == "last":
        master_background = viz_frames[-1]
    else:
        master_background = viz_frames[0]

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
        print("Only 'da360' and 'pager' are supported for video. Fallback to da360")
    depth_engine = converter._get_depth_model(dm_name)
    print(f"[SPAG4D] Running {dm_name.upper()} depth estimation...", flush=True)

    start_depthref = time.time()
    reference_bg = torch.from_numpy(master_background).to("cuda")
    with torch.inference_mode():
        # Use temporal_consistency mode (if requested) for consistent scaling across frames
        depth_raw, _ = depth_engine.predict(reference_bg, temporal_consistency=temporal_consistency)

        # Solution 4: multi-frame depth reference.
        # Instead of a single reference depth (from the temporal-median background),
        # use the temporal median of depth predictions over N evenly-spaced frames.
        # More robust to a noisy/atypical single reference.
        if reference_frames_for_median and reference_frames_for_median > 1:
            n_ref = min(reference_frames_for_median, n_total_frames)
            ref_indices = np.linspace(0, n_total_frames - 1, n_ref).astype(int)
            ref_depths = []
            for ridx in ref_indices:
                rframe = torch.from_numpy(viz_frames[ridx].copy()).to("cuda")
                rdepth, _ = depth_engine.predict(rframe, temporal_consistency=temporal_consistency)
                ref_depths.append(rdepth.cpu().numpy())
            depth_raw_np_median = np.median(np.stack(ref_depths, axis=0), axis=0)
            depth_raw = torch.from_numpy(depth_raw_np_median).to(depth_raw.device)
            print(f"[SPAG4D] Solution 4: reference depth = median of {n_ref} frames "
                  f"(indices {list(ref_indices)})")

    # Calculate scale factor to normalize median depth to ~5m (single time, for all frames)
    depth_median = float(depth_raw.median())
    scale_factor_to_5m = 5.0 / depth_median if depth_median > 1e-6 else 1.0

    depth_ref = depth_raw * scale_factor_to_5m * global_scale
    # Save depth preview if requested
    if depth_preview_path:
        converter._save_depth_preview(
            depth_ref, str(depth_preview_path / "_masterdepth.jpg")
        )
    depth_ref_np = depth_ref.cpu().numpy()
    print(f"[SPAG4D] Depth estimation for master bg completed in {1000 * (time.time() - start_depthref):.2f} ms")

    # C1 FIX: Calculate scene defaults once from reference depth (not per-frame)
    if depth_min is None or depth_max is None or sky_threshold is None:
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

    output_path = str(output_path)
    depth_estimation_cumtime = 0
    depth_alignement_cumtime = 0
    gs_generation_cumtime = 0
    aligned_depth_list_cpu = []

    median_depth_fg = []
    aligned_median_depth_fg = []
    stabilized_aligned_median_depth_fg = []

    # Background depth metrics (NEW)
    aligned_median_depth_bg = []
    background_depth_deltas = []
    foreground_depth_deltas = []

    class FGDepthStabilizer:
        """
        Stabilisation causale de la depth foreground.

        Maintient une baseline EMA qui n'est mise à jour que sur les frames
        jugées normales — évite l'effet de cliquet où chaque correction
        pollue la référence pour la frame suivante.

        Gère aussi le cas d'un vrai changement de profondeur soutenu :
        si trop de frames consécutives sont "outliers", on considère que
        c'est un changement réel et on resynchronise la baseline.
        """

        def __init__(
            self,
            jump_threshold: float = 0.5,
            ema_alpha: float = 0.3,  # vitesse d'adaptation de la baseline
            max_outlier_run: int = 4,  # au-delà, on accepte le changement comme réel
            scale_clip: tuple = (0.5, 2.0),
        ):
            self.jump_threshold = jump_threshold
            self.ema_alpha = ema_alpha
            self.max_outlier_run = max_outlier_run
            self.scale_clip = scale_clip

            self.baseline: float | None = None
            self.outlier_run: int = 0

        def __call__(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
            fg = mask > 0
            if fg.sum() == 0:
                return depth

            current_median = float(np.median(depth[fg]))

            # Première frame : initialise la baseline, rien à corriger
            if self.baseline is None:
                self.baseline = current_median
                return depth

            delta = abs(current_median - self.baseline)

            if delta <= self.jump_threshold:
                # Frame normale : update baseline (EMA) + reset compteur
                self.baseline = (
                    self.ema_alpha * current_median + (1 - self.ema_alpha) * self.baseline
                )
                self.outlier_run = 0
                return depth

            # --- Outlier détecté ---
            self.outlier_run += 1

            if self.outlier_run > self.max_outlier_run:
                # Changement soutenu sur plusieurs frames → probablement réel.
                # On accepte la nouvelle valeur comme baseline et on arrête de corriger.
                self.baseline = current_median
                self.outlier_run = 0
                return depth

            # Outlier ponctuel → correction vers la baseline (gelée, non polluée)
            if current_median <= 0:
                return depth

            scale = np.clip(self.baseline / current_median, *self.scale_clip)
            corrected = depth.copy()
            corrected[fg] *= scale
            return corrected

    fg_stabilizer = FGDepthStabilizer(jump_threshold=0.5, max_outlier_run=4)

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
            if self.method == "median":
                return np.median(stack, axis=0)
            # gaussian: use the tail of the full weight vector matching buffer len
            k = stack.shape[0]
            w = self._full_weights[-k:]
            w = w / w.sum()
            return np.tensordot(w, stack, axes=([0], [0]))

    depth_smoother = (
        TemporalDepthSmoother(depth_smoothing_window, depth_smoothing_method)
        if depth_smoothing
        else None
    )

    (output_folder / "gaussians").mkdir(parents=True, exist_ok=True)
    n_gaussians = []
    for idx, frame in enumerate(tqdm(viz_frames)):
        # Depth estimation
        start_depth = time.time()
        W, H, _ = frame.shape
        image_tensor = torch.from_numpy(frame).to("cuda")
        with torch.inference_mode():
            # Use temporal_consistency mode (if requested) for consistency across frames
            depth_raw, _ = depth_engine.predict(image_tensor, temporal_consistency=temporal_consistency)
        depth = depth_raw * scale_factor_to_5m * global_scale
        depth_np = depth.cpu().numpy()

        # Save depth preview if requested
        if depth_preview_path:
            converter._save_depth_preview(
                depth, str(depth_preview_path / f"depth_{idx}.jpg")
            )
        depth_estimation_cumtime += time.time() - start_depth

        start_depth = time.time()
        if alignement_mask in ["sam", "sam_and_activity", "nothing"]:
            output = outputs_per_frame[idx * skip_step]
        # fuse all mask into one single mask (we can also keep them separate if we want to track them separately, but for now let's just fuse them)
        fused_mask = (
            (activity_mask / 255).astype(np.uint8)
            if alignement_mask == "sam_and_activity"
            else np.zeros((W, H), dtype=np.uint8)
        )
        if alignement_mask == "all":
            fused_mask = np.zeros((W, H), dtype=np.uint8)  # Nothing is considerer moving;
        else:  # SAM, SAM_AND_ACTIVITY, NOTHING(just for freeze_bg)
            sam_mask = np.zeros((W, H), dtype=np.uint8)
            for _, mask in output.items():
                sam_mask = np.maximum(sam_mask, mask)
            fused_mask = np.maximum(fused_mask, sam_mask)

        # Align depth_ref_np
        median_depth_fg.append(float(np.median(depth_np[fused_mask > 0])))
        if alignement_mask == "nothing":
            aligned_depth_np = depth_np
        else:
            aligned_depth_np = align_depth_frame(
                depth_np, depth_ref_np, fused_mask, alignement_method, verbose=True
            ).copy()
        # Solution 1: temporal depth smoothing (causal window) on the aligned depth,
        # applied before FG stabilization / Gaussian generation.
        if depth_smoother is not None:
            aligned_depth_np = depth_smoother(aligned_depth_np)
        aligned_median_depth_fg.append(float(np.median(aligned_depth_np[fused_mask > 0])))

        # Background depth metrics (NEW): median depth of static regions
        bg_pixels = fused_mask == 0
        if bg_pixels.sum() > 0:
            bg_median = float(np.median(aligned_depth_np[bg_pixels]))
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

        aligned_depth_np2 = fg_stabilizer(aligned_depth_np, fused_mask).copy()
        stabilized_aligned_median_depth_fg.append(
            float(np.median(aligned_depth_np2[fused_mask > 0]))
        )

        aligned_depth_list_cpu.append(aligned_depth_np2)

        depth_alignement_cumtime += time.time() - start_depth

        # Run SPAG pipeline
        start_depth = time.time()
        if freeze_bg:
            assert alignement_mask in ["sam", "sam_and_activity", "nothing"], (
                "SAM must be computed to use freeze_bg, alignement_mask must be in ['sam', 'sam_and_activity']"
            )
            aligned_depth_np[sam_mask == 0] = np.nan  # set nan out of mask;
            aligned_depth_np2[sam_mask == 0] = np.nan  # set nan out of mask;
        gaussians = to_gaussians(
            converter,
            aligned_depth_np2,
            image_tensor,
            max(stride//2, 1),
            H,
            depth_min,
            depth_max,
            sky_threshold,
            outlier_pruning,
            grazing_angle,
            sparse_pruning,
        )
        if freeze_bg:
            gaussians = {k: torch.cat([gaussians_bg[k], gaussians[k]], dim=0) for k in gaussians_bg}
        colors_linear = False
        gs_generation_cumtime += time.time() - start_depth

        # Save PLY
        n_gaussians.append(gaussians["means"].shape[0])
        save_ply_gsplat(
            gaussians,
            str(output_folder / "gaussians" / f"frame_{idx}.ply"),
            sh_degree=0,
            colors_linear=colors_linear,
        )

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

    processing_time = time.time() - start_time
    file_size = (output_folder / "gaussians" / f"frame_{idx}.ply").stat().st_size
    print(f"[SPAG4D] Processing_time of all frames in {processing_time:.2f}s")
    print(f"depth_estimation_cumtime : {depth_estimation_cumtime:.2f}s")
    print(f"depth_alignement_cumtime : {depth_alignement_cumtime:.2f}s")
    print(f"gs_generation_cumtime : {gs_generation_cumtime:.2f}s")

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
        # stack_np = torch.stack(aligned_depth_list_cpu, dim=0).cpu().numpy()
        stack_np = np.stack(aligned_depth_list_cpu, axis=0)
        depth_std_all = np.nanstd(stack_np, axis=0)
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
                / f"_depth_stats_{alignement_mask}_{quantile}_{'freeze' if freeze_bg else ''}.json"
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

    return ConversionResult(
        output_path=output_path.replace(".ply", "_0.ply"),
        splat_count=n_gaussians,
        file_size=file_size,
        processing_time=processing_time,
        depth_range=depth_range,
        depth_npy_path=None,  # str(depth_npy_path) if depth_npy_path else None,
        panorama_size=(W, H),
        depth_metrics=depth_metrics,
    )


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
    static_mask = (
        (mask_moving == 0)
        & (depth_frame > 0)
        & (depth_ref > 0)
        & np.isfinite(depth_frame)
        & np.isfinite(depth_ref)
    )

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

    # ── 6. Application hors mask statique───────────────────────────────────
    depth_frame[~static_mask] = s * depth_frame[~static_mask] + t

    return np.clip(depth_frame, 0.0, None)


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
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        s, t = result[0]
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
    viz_frames: np.ndarray,
    flow_masks: np.ndarray,
    meta: dict,
    n_total_frames: int,
    skip_step: int = 10,
    enable_viz: bool = False,
    size_threshold: int = 50,
    prox_threshold: int = 80,
    min_times_seen: int = 3,

):
    # On prépare SAM3 vidéo
    gpus_to_use = [torch.cuda.current_device()]
    video_predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    # Start a session
    initial_response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(video_path),
            offload_video_to_cpu=True,
        )
    )
    session_id = initial_response["session_id"]
    print("SAM3 video session started with session_id:", session_id)

    kernel_erode = np.ones((3, 3), np.uint8)
    kernel_dilate = np.ones((7, 7), np.uint8)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)) # Much wider to bridge the gaps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_preview = cv2.VideoWriter(output_path / 'traitement_mask.mp4', fourcc, meta['fps'], (meta['new_W'], meta['new_H']))
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

        # Find and filter contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_frame_centers = []
        bounding_boxes = []
        scores = []
        mask_preview.write(np.stack([fg_mask] * 3, axis=-1))
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
                current_frame_centers.append((cX, cY, x, y, w, h, conf, area))
                bounding_boxes.append((x, y, w, h))
                scores.append(conf)

            if len(current_frame_centers) <= 0:
                print("no contour satisfying")
                continue #no move found

        if len(bounding_boxes) > 1:
            print(f"Found {len(bounding_boxes)} boxes at frame {frame_idx} ! Consider implementing dnn.NMSBoxes")

        matched_this_frame = set()
        visual_frame = viz_frames[frame_idx].copy()
        # Match found motion regions to existing tracks, or spawn new IDs (e.g., flyaway hat)
        for (cX, cY, x, y, w, h, conf, area) in current_frame_centers:
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

            occurence = {"conf": conf,"idx": frame_idx, "center": (cX, cY), "box": (x, y, w, h),}
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
    mask_preview.release()
    reencode_h264(str(output_path / 'boxes_input_sam.mp4'))
    reencode_h264(str(output_path / 'traitement_mask.mp4'))
    print(f"-> Diagnostic preview saved: {output_path}")

    found_object = len(active_tracks)
    for track in active_tracks:
        seens = track['seen']
        if len(seens) < min_times_seen:
            print(f"Skip object {track['obj_id']} because seen less than {min_times_seen} times", flush=True)
            found_object -= 1
            # active_tracks.remove(track)
            continue

        for s in seens:
            matched_obj_id = track['obj_id']
            frame_idx = s['idx']
            box = abs_to_rel_coords([list(s['box'])], meta['new_W'], meta['new_H'], coord_type="box")
            center = abs_to_rel_coords([list(s['center'])], meta['new_W'], meta['new_H'])
            # Instantly register this new independent entity directly to SAM 3
            tqdm.write(f"[By flow] Moving object {matched_obj_id} registered at frame {frame_idx} (Coords: {str(s['center'])}) with conf={s['conf']}, seen {len(seens)} times")
            o = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    obj_id=matched_obj_id,
                    points=center,
                    point_labels=[1],
                    # bounding_boxes=box,
                    # bounding_box_labels=[1],
                    # text="visual",
                )
            )
            if start_frame_index is None:
                start_frame_index = frame_idx
            break #register first time seen only

    # Text prompts also
    # prompts = ["person", "animal", "vehicle", "ball", "balloon", "gun", "pet", "car", "bus"]
    # n = 0
    # for p in prompts:
    #     frame_response = video_predictor.handle_request(
    #         request=dict(
    #             type="add_prompt",
    #             session_id=session_id,  # Uses the permanent ID variable
    #             frame_index=start_frame_index,
    #             text=p,
    #         )
    #     )
    #     # Safely get length of probabilities array if it exists
    #     out = frame_response.get("outputs", None)
    #     m = len(out.get("out_probs", [])) if out is not None else 0
    #     n += m
    #     if m > 0:
    #         print(f"[By prompt] registered [{m} {p}]")

    #     found_object += n

    # STEP B: Propagate All Interleaved Multi-Object Tracks
    print(f"\n--- Propagating {found_object} moving object starting at {start_frame_index} ---")

    mask_writer = cv2.VideoWriter(str(output_path / 'motion_detection_preview.mp4'), fourcc, meta['fps'], (meta['new_W'], meta['new_H']))
    mask_sep_writer = cv2.VideoWriter(str(output_path / 'sam3_output_masks.mp4'), fourcc, meta['fps'], (meta['new_W'], meta['new_H']))
    np.random.seed(42)  # Pour garder les mêmes couleurs d'une frame à l'autre
    STATIC_COLORS = np.random.randint(0, 255, size=(found_object, 3), dtype=np.uint8)
    color_map = {
        track['obj_id']: STATIC_COLORS[i].tolist()
        for i, track in enumerate(t for t in active_tracks if len(t['seen']) >= min_times_seen)
    }

    outputs_per_frame = {}
    try:
        for response in video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=start_frame_index,
                # max_frame_num_to_track=128,
                propagation_direction="both", #both
                output_prob_thresh=0.5, #0.5
            )
        ):
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
            # Save combined black-and-white mask frame
            mask_writer.write(np.stack([accumulated_mask] * 3, axis=-1))
            mask_sep_writer.write(separated_mask)
        outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
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

        # Close writers
        mask_writer.release()
        mask_sep_writer.release()
        reencode_h264(str(output_path / 'motion_detection_preview.mp4'))
        reencode_h264(str(output_path / 'sam3_output_masks.mp4'))

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
    for response in video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

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
