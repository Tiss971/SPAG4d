import numpy as np
import pytest

from spag4d.pipeline.video import (
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
