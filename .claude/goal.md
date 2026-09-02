# Technical Specification: 360° Video Pre-processing Pipeline for 3DGS

_Original design doc. Kept as historical reference — the shipped pipeline has since diverged in one
key way: **Step 3 below (pure per-frame affine alignment) is no longer the final stabilization
stage.** It's still the first pass (`align_depth_frame`), but the shipped default composites on top
of it with `bglock` (background locked exactly to `D_ref`, not just affine-corrected) — see
`docs/WORK_LOG_DYNAMIC_360_RECONSTRUCTION.md` §5 for why bglock replaced affine-alone as the
default, and `pipeline_overview.md` for the current step-by-step. Steps 1 and 2 below still
describe the shipped approach accurately._

## Context & Objective

**Goal**: Convert a monocular equirectangular 360° video from a fixed camera into a consistent 3D
Gaussian Splatting (3DGS) scene (6-DoF navigation).

**Core Problem**: Mitigate temporal depth flickering (arbitrary scale and shift breathing) inherent
to frame-by-frame monocular depth estimation models.

**Strategy**: Absolute stabilization of the static background (~90% of the geometry) and affine
alignment of individual frame depth maps over static regions.

## Tech Stack & Dependencies

- **Depth Model**: DA360 (native equirectangular monocular depth estimation).
- **Motion Estimation**: WAFT (Warping-Alone Field Transforms) for dense optical flow.
- **Segmentation**: Segment Anything Model 3.1 (SAM 3.1) with Object Multiplexing.
- **Core Logic**: OpenCV (Python), NumPy, Scikit-Learn (Least-Squares Regression).

## Pipeline Architecture & Data Flow

```
[Input Video 360] ──> Time Median ──> Master Background ──> DA360 ──> Master Depth (D_ref)
        │
        ├──> WAFT Flow ──> DIoU Tracker ──> SAM 3.1 ──> Dynamic Masks (M_i)
        │                                                     │
        └──> Frame i Depth (D_i) ── [Affine Alignment over 1-M_i] ──> Aligned Depth Map i
```

### Step 1: Geometry Initialization (Master Frame)

- Compute Temporal Median: sample the video (e.g. 2 fps) to compute a temporal median image,
  eliminating moving objects. Output: `master_background_360.jpg`.
- Compute Reference Depth: run DA360 on the master background. Output: Master Depth Map (`D_ref`),
  the absolute rigid geometry of the room.

### Step 2: Advanced Tracking & Segmentation (WAFT + SAM 3.1)

- **Dense Optical Flow**: compute high-resolution flow fields using WAFT between frame `t` and
  `t+1`.
- **Occlusion & Consistency Check**: bidirectional forward/backward flow error threshold. If the
  error spikes, flag an occlusion event and pause SAM 3's memory accumulation for that object ID
  to prevent track drift.
- **Hybrid Distance-IoU (DIoU) Tracker**: detect bounding boxes (`last_box`) from WAFT active
  motion areas; track object IDs across frames using IoU overlap first, falling back to Euclidean
  center distance normalized by object scale if the mask fragments (IoU drops to 0). Supports up
  to 16 simultaneous objects via SAM 3's internal Object Multiplex mechanism.
- **SAM 3.1 Propagation**: pass active boxes/points as spatial prompts into SAM 3.1. Optional:
  explicitly warp spatial prompts using sparse flow vectors from WAFT
  ($(x_{t+1}, y_{t+1}) = (x_t + \Delta x, y_t + \Delta y)$) to anchor SAM's attention in
  high-velocity maneuvers. Output: temporally stable binary sequence masks ($M_i$, `1 = motion,
  0 = static background`).

### Step 3: Affine Depth Alignment (Anti-Flickering)

For each frame $i$:
1. Isolate Static Region: invert the frame mask, $M_{\text{static}, i} = 1 - M_i$.
2. Morphological Expansion: dilate $M_i$ by 5–15px (`cv2.dilate`) before inversion — absorbs the
   boundary blur (depth bleeding) DA360 generates around moving foreground edges.
3. Compute Real-Time Affine Parameters: least-squares regression on pixels where
   $M_{\text{static}, i} == 1$: $D_{\text{ref}} \approx s_i \cdot D_i + t_i$.
4. Align Full Map: $D_{\text{aligned}, i} = s_i \cdot D_i + t_i$.

## Target Outputs for 3DGS Dataloader

Per video frame: `frame_xxxx.png` (original RGB), `mask_xxxx.png` (SAM 3.1 binary mask),
`depth_xxxx.npy` / `.png` (globally aligned, flicker-free depth map).
