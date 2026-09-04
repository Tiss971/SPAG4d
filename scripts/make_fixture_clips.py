"""One-time generator for the golden-regression-harness fixture clips.

Trims the first N frames of a source clip into a short mp4 under
tests/fixtures/clips/. Re-run only if the fixture clips need to change --
this is not part of the pytest suite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

SOURCE_DIR = Path("/raid/mb273924/_DATASETS/uptale/data/videos")
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "clips"

CLIPS = {
    "scene01_trim25.mp4": "scene01.mp4",
    "circulation_trim25.mp4": "circulation_site_1_edit_coupe.mp4",
}


def trim_clip(src: Path, dst: Path, n_frames: int = 25) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise OSError(f"Cannot open: {src}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (W, H))

    written = 0
    try:
        while written < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()

    if written < n_frames:
        raise RuntimeError(
            f"{src} only had {written} frames, wanted {n_frames}"
        )
    print(f"Wrote {dst} ({written} frames, {W}x{H})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-frames", type=int, default=25)
    args = ap.parse_args()

    for dst_name, src_name in CLIPS.items():
        trim_clip(SOURCE_DIR / src_name, FIXTURE_DIR / dst_name, args.n_frames)


if __name__ == "__main__":
    main()
