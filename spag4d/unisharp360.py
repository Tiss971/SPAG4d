"""High-level SPAG4d-facing wrapper for the UniSHARP backend."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)


def convert_unisharp360(
    input_path: str,
    output_path: str,
    device: torch.device,
    unisharp_repo: str | None,
    unisharp_python: str | None = None,
    checkpoint_path: str | None = None,
    scale_align: str = "global",
    format_mode: str = "copy",
    save_debug: bool = False,
    raw_output_dir: str | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    t0 = time.time()

    # ---- resolve env fallbacks -------------------------------------------
    unisharp_repo = unisharp_repo or os.environ.get("SPAG4D_UNISHARP_REPO")
    unisharp_python = unisharp_python or os.environ.get("SPAG4D_UNISHARP_PYTHON")
    checkpoint_path = checkpoint_path or os.environ.get("SPAG4D_UNISHARP_CHECKPOINT")

    # ---- validation -------------------------------------------------------
    if not unisharp_repo:
        raise ValueError(
            "UniSHARP backend requires --unisharp-repo (or SPAG4D_UNISHARP_REPO) "
            "pointing to a local clone of Insta360-Research-Team/UniSHARP."
        )
    repo = Path(unisharp_repo)
    if not repo.exists():
        raise FileNotFoundError(f"UniSHARP repo not found: {repo}")
    if not checkpoint_path:
        raise ValueError(
            "UniSHARP backend requires --unisharp-checkpoint "
            "(or SPAG4D_UNISHARP_CHECKPOINT)."
        )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"UniSHARP checkpoint not found: {checkpoint_path}")

    with Image.open(input_path) as im:
        w, h = im.size
    if abs((w / h) - 2.0) > 0.05:
        raise ValueError(
            f"UniSHARP panorama mode expects a 2:1 ERP image. Got {w}x{h}. "
            "Use --sharp-backend sharp for non-ERP inputs."
        )

    # ---- working directory ------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_output_dir:
        work_dir = Path(raw_output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="spag4d_unisharp_")
        work_dir = Path(tmp_ctx.name)

    try:
        if progress_callback:
            progress_callback("unisharp_inference", 0, 1)

        # SPAG4d only consumes the PLY. When SPAG4D_UNISHARP_NO_RENDER is set,
        # pass --no-render so UniSHARP skips its gsplat GIF/preview rendering.
        # This is required on platforms where gsplat's CUDA op cannot build
        # (e.g. native Windows) and is faster everywhere. Requires a UniSHARP
        # repo patched to accept --no-render; leave the env var unset otherwise.
        extra_args = ["--no-render", "True"] # if os.environ.get("SPAG4D_UNISHARP_NO_RENDER") else None

        from .unisharp_adapter import run_unisharp_inference
        run = run_unisharp_inference(
            image_path=input_path,
            out_dir=str(work_dir / "unisharp_out"),
            repo_dir=str(repo),
            checkpoint_path=str(checkpoint_path),
            python_exe=unisharp_python,
            camera="panorama",
            device=("cuda:0" if device.type == "cuda" else "cpu"),
            save_ply=True,
            max_images=1,
            extra_args=extra_args,
        )
        raw_ply = Path(run["ply_path"])

        if progress_callback:
            progress_callback("unisharp_inference", 1, 1)

        # ---- format handling ---------------------------------------------
        from .unisharp_format import (
            convert_unisharp_ply_to_spag,
            copy_unisharp_ply_to_output,
        )
        if format_mode == "copy":
            stats = copy_unisharp_ply_to_output(str(raw_ply), str(out_path))
        elif format_mode == "convert":
            stats = convert_unisharp_ply_to_spag(str(raw_ply), str(out_path))
        else:
            raise ValueError(f"Unknown format_mode {format_mode!r}")

        # ---- optional scale alignment ------------------------------------
        # scale_align in {"none","global","da360_grid"}. M1 ships "none"/"global"
        # as a no-op-safe passthrough; real alignment lands in a follow-up plan.

        # ---- debug artifacts ---------------------------------------------
        if save_debug:
            dbg = out_path.parent / (out_path.stem + "_unisharp_debug")
            dbg.mkdir(parents=True, exist_ok=True)
            for art in ("forward.gif", "rotate.gif", "metadata.json"):
                src = raw_ply.parent / art
                if src.exists():
                    shutil.copy2(src, dbg / art)
            shutil.copy2(raw_ply, dbg / "raw_gaussians.ply")

        return {
            "num_gaussians": stats["num_gaussians"],
            "num_faces": 0,  # native ERP: no faces
            "output_path": str(out_path),
            "processing_time": time.time() - t0,
            "backend": "unisharp",
        }
    finally:
        if tmp_ctx is not None and not save_debug:
            tmp_ctx.cleanup()
