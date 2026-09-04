# SPAG4D tests

## Regression harness

`test_regression_golden.py` runs the real `run_video()` pipeline over two trimmed fixture clips
(`fixtures/clips/`) under `SPAG_DETERMINISTIC=1` and asserts byte-identical output against pinned
fixtures in `fixtures/golden/`. Requires a GPU. This is the safety net for any future change to
`run_video()` or its `SPAG_*` flags -- run it before and after each change:

    pytest tests/test_regression_golden.py -v

`test_helpers.py` covers pure-function helpers (`_fit_outlier_cap`, `_estimate_scale_shift`, etc.)
directly, no GPU/clip needed, runs in under a second.

### Running tests for the first time

After a fresh clone, `tests/fixtures/golden/*/npy/` does not exist (gitignored; only `manifest.json`
travels with git). Populate the fixtures locally by running:

    PYTHONPATH=/raid/mb273924/SPAG4d /home/mb273924/miniforge3/envs/spag4d/bin/python scripts/capture_golden_fixtures.py

This takes ~2 minutes on GPU and regenerates the `.npy` reference outputs that the regression test
compares against. Use the correct Python interpreter (`/home/mb273924/miniforge3/envs/spag4d/bin/python`)
for all test commands and pipeline scripts in this repo — the base conda `python` cannot import SAM3 or `spag4d.video`.

## Updating the golden fixtures

Only do this when a change to depth/mask/flow output is intentional and has been reviewed --
regenerating the fixtures overwrites the pinned manifest.json values and makes the regression test
compare against the new behavior instead of catching it as a diff.

    python scripts/capture_golden_fixtures.py
    git add tests/fixtures/golden/
    git commit -m "test: update golden fixtures for <reason>"

## Fixture clips

`fixtures/clips/*.mp4` are the first 25 frames of `scene01.mp4` (seam-crossing foreground) and
`circulation_site_1_edit_coupe.mp4` (static background), trimmed via `scripts/make_fixture_clips.py`
from `/raid/mb273924/_DATASETS/uptale/data/videos/`. Regenerate only if the fixture clips themselves
need to change (e.g. a different frame range).
