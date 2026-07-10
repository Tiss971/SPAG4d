"""
Solution 5: Mask quality inspection & diagnostics.

Purely diagnostic — does not change the pipeline output. Quantifies whether the
WAFT optical-flow motion masks and the SAM refined masks are a bottleneck for
temporal stability, by measuring per-frame:

  - flow_coverage : fraction of pixels flagged moving by WAFT
  - sam_coverage  : fraction of pixels flagged foreground by SAM
  - agreement     : IoU between flow and SAM masks (do they agree on motion?)
  - stability     : |Δ foreground pixel count| / prev count (frame-to-frame)

Interpretation (from SOLUTIONS_TO_TEST.md decision tree):
  masks are "poor" if mean coverage < 40% is *not* the issue, but rather if
  flow/SAM agreement < 70% or stability is high (masks jitter a lot).
"""

import json
from pathlib import Path

import cv2
import numpy as np


def analyze_mask_quality(
    flow_masks: np.ndarray,          # [N, hf, wf] uint8 (WAFT resolution)
    outputs_per_frame: dict,         # {frame_idx: {obj_id: mask [H, W] uint8}}
    n_frames: int,
    skip_step: int,
    target_hw: tuple,                # (H, W) of the SAM masks (full res)
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    H, W = target_hw

    metrics = {
        "flow_coverage": [],
        "sam_coverage": [],
        "agreement": [],
        "stability": [],
    }

    prev_sam_count = None
    for idx in range(n_frames):
        # --- SAM mask for this frame ---
        output = outputs_per_frame.get(idx * skip_step, {})
        sam_m = np.zeros((H, W), dtype=np.uint8)
        for _, m in output.items():
            sam_m = np.maximum(sam_m, (m > 0).astype(np.uint8))

        # --- Flow mask for this frame, resized to SAM resolution ---
        if idx < len(flow_masks):
            fm = flow_masks[idx]
            if fm.shape[:2] != (H, W):
                fm = cv2.resize(fm, (W, H), interpolation=cv2.INTER_NEAREST)
            flow_m = (fm > 0).astype(np.uint8)
        else:
            flow_m = np.zeros((H, W), dtype=np.uint8)

        flow_cov = float(np.mean(flow_m))
        sam_cov = float(np.mean(sam_m))
        metrics["flow_coverage"].append(flow_cov)
        metrics["sam_coverage"].append(sam_cov)

        inter = int(np.sum((flow_m > 0) & (sam_m > 0)))
        union = int(np.sum((flow_m > 0) | (sam_m > 0)))
        metrics["agreement"].append(float(inter / (union + 1e-6)))

        sam_count = int(sam_m.sum())
        if prev_sam_count is not None:
            metrics["stability"].append(
                float(abs(sam_count - prev_sam_count) / (prev_sam_count + 1e-6))
            )
        prev_sam_count = sam_count

    summary = {
        "n_frames": n_frames,
        "flow_coverage_mean": float(np.mean(metrics["flow_coverage"])) if metrics["flow_coverage"] else None,
        "sam_coverage_mean": float(np.mean(metrics["sam_coverage"])) if metrics["sam_coverage"] else None,
        "agreement_mean": float(np.mean(metrics["agreement"])) if metrics["agreement"] else None,
        "stability_mean": float(np.mean(metrics["stability"])) if metrics["stability"] else None,
        "per_frame": metrics,
    }

    out_json = output_dir / "mask_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"[MaskDiag] Flow coverage: {summary['flow_coverage_mean']:.1%}" if summary["flow_coverage_mean"] is not None else "[MaskDiag] no flow")
    print(f"[MaskDiag] SAM coverage:  {summary['sam_coverage_mean']:.1%}" if summary["sam_coverage_mean"] is not None else "")
    print(f"[MaskDiag] Flow-SAM IoU:  {summary['agreement_mean']:.1%}" if summary["agreement_mean"] is not None else "")
    print(f"[MaskDiag] SAM stability (frame drift): {summary['stability_mean']:.3f}" if summary["stability_mean"] is not None else "")
    print(f"[MaskDiag] saved: {out_json}")
    return summary
