import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spag4d.core import SPAG4D
from spag4d.video import run_video

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CLIPS_DIR = FIXTURES_DIR / "clips"
GOLDEN_DIR = FIXTURES_DIR / "golden"

CLIPS = ["scene01_trim25.mp4", "circulation_trim25.mp4"]


def _vertex_hash(ply_path: Path) -> str:
    return hashlib.sha256(ply_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("clip_name", CLIPS)
def test_run_video_matches_golden_fixtures(clip_name, tmp_path):
    golden_dir = GOLDEN_DIR / clip_name
    golden_npy_dir = golden_dir / "npy"
    manifest = json.loads((golden_dir / "manifest.json").read_text())

    clip_path = CLIPS_DIR / clip_name
    out_npy_dir = tmp_path / "npy"
    out_npy_dir.mkdir()
    # run_video's output_path is an output DIRECTORY -- it mkdir's it and
    # writes one PLY per frame under <output_path>/gaussians/frame_{idx}.ply.
    run_output_dir = tmp_path / "run_output"

    converter = SPAG4D(device="cuda", depth_model="da360")
    result = run_video(
        converter=converter,
        video_path=str(clip_path),
        output_path=str(run_output_dir),
        skip_step=1,
        depth_npy_dir=out_npy_dir,
        freeze_bg_live_color=False,
    )

    n_frames = len(result.splat_count)
    last_ply_path = run_output_dir / "gaussians" / f"frame_{n_frames - 1}.ply"

    assert result.splat_count[-1] == manifest["splat_count"]
    assert _vertex_hash(last_ply_path) == manifest["vertex_hash"]

    golden_files = sorted(golden_npy_dir.glob("*.npy"))
    assert golden_files, f"no golden .npy files under {golden_npy_dir}"

    for golden_file in golden_files:
        produced_file = out_npy_dir / golden_file.name
        assert produced_file.exists(), f"missing output file {produced_file.name}"
        golden_arr = np.load(golden_file)
        produced_arr = np.load(produced_file)
        np.testing.assert_array_equal(
            golden_arr, produced_arr,
            err_msg=f"{golden_file.name} diverged from golden fixture",
        )
