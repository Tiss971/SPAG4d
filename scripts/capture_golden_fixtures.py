"""One-time (re-)generator for the golden regression fixtures.

Runs the real SPAG4D video pipeline over the trimmed fixture clips under
SPAG_DETERMINISTIC=1 and pins depth/mask/flow arrays + a PLY hash as
golden fixtures for tests/test_regression_golden.py.

Re-run ONLY when a change to run_video()'s output is intentional and has
been reviewed -- running this script overwrites the fixtures that the
regression test compares against.

NOTE on what's actually committed to git: the per-frame depth/mask/flow .npy
arrays under tests/fixtures/golden/<clip>/npy/ are LOCAL-ONLY and gitignored
(see tests/fixtures/golden/*/npy/ in .gitignore) -- native-resolution flow
arrays alone run to ~600MB-1.4GB per clip, ~3.2GB total across both fixture
clips, too large to put in git history without LFS. Only manifest.json
(splat_count + vertex_hash) is tracked. This means: after a fresh clone, or
whenever tests/fixtures/golden/<clip>/npy/ is missing/stale, Task 4's
per-array diff test needs this script re-run locally to regenerate the npy/
fixtures before it can pass -- manifest.json alone is not sufficient for that
test. This should also be documented in tests/README.md (Task 5); leaving
this note here as the trail since that file doesn't exist yet.

NOTE on run_video()'s actual signature (verified against spag4d/video.py:334-379
and spag4d/cli.py:151-270 before writing this script -- it differs from a naive
guess in two ways):
  - `output_path` is NOT a single .ply file. run_video() treats it as an output
    DIRECTORY: it mkdir's it directly and writes one PLY per frame under
    `<output_path>/gaussians/frame_{idx}.ply`. There is no single merged PLY.
  - `result.splat_count` is a list[int] (one entry per frame), not a single int
    (see spag4d/core.py ConversionResult.splat_count: int | list[int], and the
    CLI's `sum(result.splat_count) / len(result.splat_count)` mean computation).
    This script pins the LAST frame's splat count, matching the PLY it hashes
    (the same frame `file_size`/`idx` the pipeline itself reports internally).
  - `freeze_bg_live_color` defaults to True but run_video() raises ValueError
    unless `freeze_bg=True` is also passed (freeze_bg defaults False). Since
    this harness isn't exercising the freeze_bg feature, pass
    `freeze_bg_live_color=False` explicitly to satisfy the guard with the
    otherwise-default freeze_bg=False.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ["SPAG_DETERMINISTIC"] = "1"

from spag4d.pipeline.core import SPAG4D  # noqa: E402
from spag4d.pipeline.video import run_video  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CLIPS_DIR = REPO_ROOT / "tests" / "fixtures" / "clips"
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"

CLIPS = ["scene01_trim25.mp4", "circulation_trim25.mp4"]


def _vertex_hash(ply_path: Path) -> str:
    return hashlib.sha256(ply_path.read_bytes()).hexdigest()


def capture_one(clip_name: str) -> None:
    clip_path = FIXTURE_CLIPS_DIR / clip_name
    out_dir = GOLDEN_DIR / clip_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    depth_npy_dir = out_dir / "npy"
    depth_npy_dir.mkdir()

    # run_video's output_path is an output DIRECTORY (it mkdir's it and writes
    # gaussians/frame_{idx}.ply inside), not a single .ply file -- see module
    # docstring above.
    run_output_dir = out_dir / "run_output"

    converter = SPAG4D(device="cuda", depth_model="da360")
    result = run_video(
        converter=converter,
        video_path=str(clip_path),
        output_path=str(run_output_dir),
        skip_step=1,
        depth_npy_dir=depth_npy_dir,
        freeze_bg_live_color=False,
    )

    n_frames = len(result.splat_count)
    last_ply_path = run_output_dir / "gaussians" / f"frame_{n_frames - 1}.ply"
    if not last_ply_path.exists():
        raise FileNotFoundError(
            f"Expected last-frame PLY not found at {last_ply_path} "
            f"(n_frames={n_frames}); check run_video's per-frame naming."
        )

    manifest = {
        "splat_count": result.splat_count[-1],
        "vertex_hash": _vertex_hash(last_ply_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Only the manifest hash pins the PLY content; the run's scratch output
    # directory (per-frame PLYs, previews, diagnostic images) is not part of
    # the golden fixture set.
    shutil.rmtree(run_output_dir)

    print(f"Captured golden fixtures for {clip_name} -> {out_dir}")


def main() -> None:
    for clip_name in CLIPS:
        capture_one(clip_name)


if __name__ == "__main__":
    main()
