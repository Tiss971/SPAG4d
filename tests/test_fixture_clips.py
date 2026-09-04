# tests/test_fixture_clips.py
from pathlib import Path

import cv2
import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "clips"

CLIPS = ["scene01_trim25.mp4", "circulation_trim25.mp4"]


@pytest.mark.parametrize("clip_name", CLIPS)
def test_fixture_clip_has_25_frames(clip_name):
    path = FIXTURE_DIR / clip_name
    assert path.exists(), f"missing fixture clip {path}; run scripts/make_fixture_clips.py"

    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    assert count == 25, f"{clip_name} has {count} frames, expected 25"
