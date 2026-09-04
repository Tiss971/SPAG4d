# Golden-Output Regression Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `pytest` suite that lets a future change to `run_video()` or its `SPAG_*` flags be
verified byte-identical to current production output in seconds, instead of requiring a full-clip
benchmark run.

**Architecture:** Two short (~25-frame) fixture clips are trimmed from existing production videos
and committed under `tests/fixtures/`. A one-time generation script runs the real pipeline over them
under `SPAG_DETERMINISTIC=1` and pins the resulting `depth_*.npy`/`mask_*.npy`/`flow_*.npy` arrays
plus a PLY vertex-count/hash as golden fixtures. The `pytest` suite re-runs the same pipeline call
and asserts byte-identity against those pins. A second, independent test file covers pure-function
helpers already in `spag4d/video.py` directly, with no pipeline run.

**Tech Stack:** Python, `pytest` (already declared in `pyproject.toml`), `numpy`, `opencv-python`
(`cv2`, already a dependency, used to trim fixture clips and matches the codec pattern already
present in `extract_video_frames`/`reencode_h264`).

**Spec:** [docs/superpowers/specs/2026-09-03-spag4d-codebase-health-design.md](../specs/2026-09-03-spag4d-codebase-health-design.md) — section "1. Golden-output regression harness"

## Global Constraints

- No algorithmic changes to depth/mask/flow behavior anywhere in this plan — every step only adds
  test infrastructure.
- Fixture clips: `scene01.mp4` (seam-crossing foreground) and `circulation_site_1_edit_coupe.mp4`
  (static background, ground shadows), sourced from
  `/raid/mb273924/_DATASETS/uptale/data/videos/`.
- Fixture clips trimmed to the first ~25 frames each, committed under `tests/fixtures/clips/`.
- All pipeline runs in this plan use `SPAG_DETERMINISTIC=1` (seeds np/torch/CUDA/cuDNN, already
  shipped in `spag4d/video.py:464-465`).
- Total committed fixture size (trimmed clips + pinned `.npy` arrays) must stay in the tens-of-MB
  range — verify with `du -sh tests/fixtures/` before committing and flag it rather than committing
  blindly if it's larger.
- `ffmpeg`/`ffprobe` are **not installed** on this machine — use `cv2.VideoCapture`/`cv2.VideoWriter`
  for all video trimming, matching the existing pattern in `spag4d/video.py`'s
  `extract_video_frames`/`reencode_h264`.

---

### Task 1: Trimmed fixture clips + generation script

**Files:**
- Create: `tests/fixtures/clips/scene01_trim25.mp4`
- Create: `tests/fixtures/clips/circulation_trim25.mp4`
- Create: `scripts/make_fixture_clips.py`
- Test: `tests/test_fixture_clips.py`

**Interfaces:**
- Produces: two short mp4 files under `tests/fixtures/clips/` that later tasks read via
  `cv2.VideoCapture` and feed into `run_video()`.

- [ ] **Step 1: Write `scripts/make_fixture_clips.py`**

```python
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
```

- [ ] **Step 2: Run the generator**

Run: `cd /raid/mb273924/SPAG4d && python scripts/make_fixture_clips.py`
Expected: prints two `Wrote ...` lines; `tests/fixtures/clips/scene01_trim25.mp4` and
`tests/fixtures/clips/circulation_trim25.mp4` exist.

- [ ] **Step 3: Write the sanity test**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /raid/mb273924/SPAG4d && pytest tests/test_fixture_clips.py -v`
Expected: 2 passed

- [ ] **Step 5: Check fixture size, then commit**

```bash
du -sh tests/fixtures/clips/
```

If the total is more than a few tens of MB, stop and flag it before committing (reconsider frame
count or resolution) rather than committing blindly.

```bash
git add scripts/make_fixture_clips.py tests/fixtures/clips/ tests/test_fixture_clips.py
git commit -m "test: add trimmed fixture clips for the golden regression harness"
```

---

### Task 2: Unit tests for pure-function helpers

**Files:**
- Create: `tests/test_helpers.py`

**Interfaces:**
- Consumes: `spag4d.video._fit_outlier_cap`, `spag4d.video._valid_metric_mask`,
  `spag4d.video._estimate_scale_shift`, `spag4d.video._activity_mask_from_std` — all existing
  module-level functions in `spag4d/video.py`, unmodified.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_helpers.py
import numpy as np

from spag4d.video import (
    _activity_mask_from_std,
    _estimate_scale_shift,
    _fit_outlier_cap,
    _valid_metric_mask,
)


def test_fit_outlier_cap_scales_with_median():
    depth_ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cap = _fit_outlier_cap(depth_ref)
    # cap = median * _FIT_OUTLIER_RATIO; median here is 3.0
    assert cap > 3.0
    assert np.isfinite(cap)


def test_fit_outlier_cap_empty_input_returns_inf():
    depth_ref = np.array([-1.0, 0.0, np.nan])
    cap = _fit_outlier_cap(depth_ref)
    assert cap == float("inf")


def test_valid_metric_mask_filters_outliers():
    depth = np.array([1.0, 2.0, 100.0, np.nan, -1.0])
    base_mask = np.array([True, True, True, True, True])
    cap = 10.0
    result = _valid_metric_mask(depth, base_mask, cap)
    np.testing.assert_array_equal(result, [True, True, False, False, False])


def test_valid_metric_mask_falls_back_when_all_gated_out():
    depth = np.array([1000.0, 2000.0])
    base_mask = np.array([True, True])
    cap = 10.0
    result = _valid_metric_mask(depth, base_mask, cap)
    # gating would empty the selection -> falls back to base_mask unfiltered
    np.testing.assert_array_equal(result, base_mask)


def test_estimate_scale_shift_lstsq_recovers_exact_affine():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=1000).astype(np.float32)
    true_s, true_t = 2.5, -1.3
    y = (true_s * x + true_t).astype(np.float32)

    s, t = _estimate_scale_shift(
        x, y, method="lstsq",
        ransac_residual_threshold=0.0, ransac_max_trials=0, verbose=False,
    )

    assert s == pytest.approx(true_s, abs=1e-4)
    assert t == pytest.approx(true_t, abs=1e-4)


def test_estimate_scale_shift_lstsq_constant_x_falls_back_to_mean():
    x = np.zeros(10, dtype=np.float32)
    y = np.full(10, 7.0, dtype=np.float32)

    s, t = _estimate_scale_shift(
        x, y, method="lstsq",
        ransac_residual_threshold=0.0, ransac_max_trials=0, verbose=False,
    )

    assert s == 0.0
    assert t == pytest.approx(7.0)


def test_activity_mask_from_std_thresholds_above_abs_threshold():
    std_frame = np.array([[1.0, 20.0], [5.0, 30.0]])
    mask = _activity_mask_from_std(std_frame, abs_threshold=10.0)
    np.testing.assert_array_equal(mask, [[False, True], [False, True]])
```

Add `import pytest` at the top alongside `numpy` (needed for `pytest.approx`).

- [ ] **Step 2: Run tests to verify they fail or pass against real implementations**

Run: `cd /raid/mb273924/SPAG4d && pytest tests/test_helpers.py -v`
Expected: all tests PASS immediately (these are characterization tests of existing, unmodified
code — there is no "make it pass" implementation step). If any fails, that means the assumed
behavior in this task's docstring/comments doesn't match the real function — read the function at
its current location in `spag4d/video.py` and correct the test to match actual behavior, not the
other way around.

- [ ] **Step 3: Commit**

```bash
git add tests/test_helpers.py
git commit -m "test: characterize pure-function depth-alignment helpers"
```

---

### Task 3: Golden-fixture capture script

**Files:**
- Create: `scripts/capture_golden_fixtures.py`

**Interfaces:**
- Consumes: `spag4d.core.SPAG4D`, `spag4d.video.run_video` (existing, unmodified), the fixture
  clips from Task 1.
- Produces: `tests/fixtures/golden/<clip_name>/depth_XXXX.npy`,
  `tests/fixtures/golden/<clip_name>/mask_XXXX.npy`, `tests/fixtures/golden/<clip_name>/flow_XXXX.npy`
  (`idx>=1`), and `tests/fixtures/golden/<clip_name>/manifest.json` containing
  `{"splat_count": <int>, "vertex_hash": "<sha256 hex>"}` — the exact fixture shape Task 4's test
  reads.

- [ ] **Step 1: Inspect `run_video`'s call signature and `depth_npy_dir` behavior**

Read `spag4d/video.py:334-379` (the `run_video` signature) and `spag4d/cli.py:151-270` (how the CLI
invokes it, in particular `depth_npy_dir=depth_raw_path`) to confirm the exact keyword names before
writing the capture script — do not guess parameter names from memory.

- [ ] **Step 2: Write `scripts/capture_golden_fixtures.py`**

```python
"""One-time (re-)generator for the golden regression fixtures.

Runs the real SPAG4D video pipeline over the trimmed fixture clips under
SPAG_DETERMINISTIC=1 and pins depth/mask/flow arrays + a PLY summary as
golden fixtures for tests/test_regression_golden.py.

Re-run ONLY when a change to run_video()'s output is intentional and has
been reviewed -- running this script overwrites the fixtures that the
regression test compares against.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ["SPAG_DETERMINISTIC"] = "1"

from spag4d.core import SPAG4D  # noqa: E402
from spag4d.video import run_video  # noqa: E402

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
    ply_path = out_dir / "output.ply"

    converter = SPAG4D(device="cuda", depth_model="da360")
    result = run_video(
        converter=converter,
        video_path=str(clip_path),
        output_path=str(ply_path),
        skip_step=1,
        depth_npy_dir=depth_npy_dir,
    )

    manifest = {
        "splat_count": result.splat_count,
        "vertex_hash": _vertex_hash(ply_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ply_path.unlink()  # only the manifest hash is pinned, not the PLY itself
    print(f"Captured golden fixtures for {clip_name} -> {out_dir}")


def main() -> None:
    for clip_name in CLIPS:
        capture_one(clip_name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the capture script**

Run: `cd /raid/mb273924/SPAG4d && python scripts/capture_golden_fixtures.py`
Expected: prints two `Captured golden fixtures for ...` lines;
`tests/fixtures/golden/scene01_trim25.mp4/npy/` and
`tests/fixtures/golden/circulation_trim25.mp4/npy/` each contain `depth_*.npy`/`mask_*.npy`/
`flow_*.npy` files and a `manifest.json`.

If `run_video`'s actual keyword arguments differ from Step 2's draft (confirmed in Step 1), fix the
script to match the real signature before running it — this is expected, not a failure.

- [ ] **Step 4: Check fixture size**

```bash
du -sh tests/fixtures/golden/
```

If it's far larger than the trimmed clips (tens of MB), reconsider before committing — e.g. confirm
`depth_npy_dir` isn't also dumping full-resolution intermediate files unrelated to the golden
comparison.

- [ ] **Step 5: Commit**

```bash
git add scripts/capture_golden_fixtures.py tests/fixtures/golden/
git commit -m "test: capture golden depth/mask/flow/PLY fixtures for regression harness"
```

---

### Task 4: Golden regression test

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_regression_golden.py`

**Interfaces:**
- Consumes: `tests/fixtures/clips/*.mp4` (Task 1), `tests/fixtures/golden/<clip>/npy/*.npy` +
  `manifest.json` (Task 3), `spag4d.core.SPAG4D`, `spag4d.video.run_video` (unmodified).
- Produces: the `pytest` entry point (`pytest tests/test_regression_golden.py`) that every later
  extraction/flag-deletion task in the codebase-health plan must pass before and after each change.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
# tests/conftest.py
import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _deterministic_env():
    os.environ["SPAG_DETERMINISTIC"] = "1"
    yield
```

- [ ] **Step 2: Write the failing regression test**

```python
# tests/test_regression_golden.py
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
    ply_path = tmp_path / "output.ply"

    converter = SPAG4D(device="cuda", depth_model="da360")
    result = run_video(
        converter=converter,
        video_path=str(clip_path),
        output_path=str(ply_path),
        skip_step=1,
        depth_npy_dir=out_npy_dir,
    )

    assert result.splat_count == manifest["splat_count"]
    assert _vertex_hash(ply_path) == manifest["vertex_hash"]

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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `cd /raid/mb273924/SPAG4d && pytest tests/test_regression_golden.py -v`
Expected: 2 passed. This should pass immediately since Task 3 captured the fixtures from this exact
same code path — a failure here means either the capture script and this test build the output
directory differently (check keyword arguments match) or `run_video` is not actually deterministic
end-to-end under `SPAG_DETERMINISTIC=1` (a real finding worth reporting, not silently working
around).

- [ ] **Step 4: Verify the test actually catches a regression**

Temporarily edit `spag4d/video.py`'s `_fit_outlier_cap` to return a clearly wrong value (e.g. hard
`return 1.0`), re-run `pytest tests/test_regression_golden.py -v`, confirm it FAILS with an
array-mismatch error, then revert the edit (`git checkout -- spag4d/video.py`) and re-run to confirm
it passes again. This is a one-time sanity check on the harness itself, not a permanent test.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_regression_golden.py
git commit -m "test: add golden-output regression test for run_video()"
```

---

### Task 5: Document the harness for future extraction/flag-pruning work

**Files:**
- Create: `tests/README.md`

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Write `tests/README.md`**

```markdown
# SPAG4D tests

## Regression harness

`test_regression_golden.py` runs the real `run_video()` pipeline over two trimmed fixture clips
(`fixtures/clips/`) under `SPAG_DETERMINISTIC=1` and asserts byte-identical output against pinned
fixtures in `fixtures/golden/`. Requires a GPU. This is the safety net for any future change to
`run_video()` or its `SPAG_*` flags -- run it before and after each change:

    pytest tests/test_regression_golden.py -v

`test_helpers.py` covers pure-function helpers (`_fit_outlier_cap`, `_estimate_scale_shift`, etc.)
directly, no GPU/clip needed, runs in under a second.

## Updating the golden fixtures

Only do this when a change to depth/mask/flow output is intentional and has been reviewed --
regenerating the fixtures makes the regression test compare against the new behavior instead of
catching it as a diff.

    python scripts/capture_golden_fixtures.py
    git add tests/fixtures/golden/
    git commit -m "test: update golden fixtures for <reason>"

## Fixture clips

`fixtures/clips/*.mp4` are the first 25 frames of `scene01.mp4` (seam-crossing foreground) and
`circulation_site_1_edit_coupe.mp4` (static background), trimmed via `scripts/make_fixture_clips.py`
from `/raid/mb273924/_DATASETS/uptale/data/videos/`. Regenerate only if the fixture clips themselves
need to change (e.g. a different frame range).
```

- [ ] **Step 2: Commit**

```bash
git add tests/README.md
git commit -m "docs: document the regression harness in tests/README.md"
```
