"""Tests for post-hoc isotonic recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from face_occlusion.calibration import (
    IsotonicMapping,
    fit_weighted_isotonic,
    load_mapping,
    oof_recalibrate,
    save_mapping,
)
from face_occlusion.metrics.challenge_metric import weighted_mse


def _under_predicting(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.clip(rng.beta(1.5, 8.0, n), 0, 1)
    y_pred = np.clip(0.6 * y + rng.normal(0, 0.03, n), 0, 1)  # systematic under-prediction
    groups = rng.integers(0, 150, n)
    return y_pred, y, groups


def test_isotonic_is_monotone_and_corrects_bias():
    y_pred, y, _ = _under_predicting()
    m = fit_weighted_isotonic(y_pred, y)
    recal = m.apply(y_pred)
    assert weighted_mse(recal, y) < weighted_mse(y_pred, y)
    grid = np.linspace(0, 1, 100)
    assert np.all(np.diff(m.apply(grid)) >= -1e-9)  # non-decreasing


def test_challenge_weight_strengthens_tail_correction():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.uniform(0, 0.1, 500), rng.uniform(0.5, 0.9, 20)])
    y_pred = np.clip(0.6 * y, 0, 1)
    weighted = fit_weighted_isotonic(y_pred, y)  # default 1/30+y weights
    unweighted = fit_weighted_isotonic(y_pred, y, weights=np.ones_like(y))
    tail = y >= 0.5
    # The challenge-weighted fit should track the tail at least as closely.
    err_w = np.mean((weighted.apply(y_pred)[tail] - y[tail]) ** 2)
    err_u = np.mean((unweighted.apply(y_pred)[tail] - y[tail]) ** 2)
    assert err_w <= err_u + 1e-6


def test_slope_cap_limits_amplification():
    # Two tail points create a near-vertical raw step; slope_cap must bound it.
    y_pred = np.array([0.0, 0.1, 0.2, 0.50, 0.51])
    y = np.array([0.0, 0.1, 0.2, 0.90, 0.95])
    m = fit_weighted_isotonic(y_pred, y, slope_cap=2.0, min_samples=1)
    x = np.asarray(m.x_knots)
    yk = np.asarray(m.y_knots)
    slopes = np.diff(yk) / np.clip(np.diff(x), 1e-9, None)
    assert slopes.max() <= 2.0 + 1e-6


def test_min_samples_merges_thin_steps():
    rng = np.random.default_rng(2)
    y = np.clip(rng.beta(1.5, 8, 2000), 0, 1)
    y_pred = np.clip(0.7 * y + rng.normal(0, 0.02, 2000), 0, 1)
    m = fit_weighted_isotonic(y_pred, y, min_samples=50, slope_cap=None)
    # Every retained segment must cover >= min_samples raw predictions.
    xs = np.sort(y_pred)
    x = np.asarray(m.x_knots)
    for lo, hi in zip(x[:-1], x[1:]):
        n = np.searchsorted(xs, hi, "right") - np.searchsorted(xs, lo, "right")
        assert n >= 50 or hi == x[-1]


def test_oof_folds_are_group_disjoint():
    # The honest-OOF guarantee: an identity never appears in both fit and held-out sets.
    from sklearn.model_selection import StratifiedGroupKFold

    from face_occlusion.data.normalize import assign_occlusion_bin
    from face_occlusion.metrics.eval_lenses import DEFAULT_LENS_EDGES

    y_pred, y, groups = _under_predicting()
    ybin = assign_occlusion_bin(y, DEFAULT_LENS_EDGES)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(y_pred.reshape(-1, 1), ybin, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
    # And the function runs end-to-end, covering every row.
    oof = oof_recalibrate(y_pred, y, groups, n_folds=5)
    assert oof.shape == y_pred.shape and np.all(np.isfinite(oof))


def test_oof_is_not_optimistic():
    # OOF recalibrated error must be >= the (optimistic) in-sample recalibrated error.
    y_pred, y, groups = _under_predicting()
    in_sample = fit_weighted_isotonic(y_pred, y).apply(y_pred)
    oof = oof_recalibrate(y_pred, y, groups, n_folds=5)
    assert weighted_mse(oof, y) >= weighted_mse(in_sample, y) - 1e-9


def test_mapping_json_roundtrip(tmp_path):
    y_pred, y, _ = _under_predicting()
    m = fit_weighted_isotonic(y_pred, y)
    path = tmp_path / "m.json"
    save_mapping(m, path)
    m2 = load_mapping(path)
    grid = np.linspace(0, 1, 50)
    assert np.allclose(m.apply(grid), m2.apply(grid))


def test_apply_handles_out_of_range_and_clips():
    m = IsotonicMapping(x_knots=[0.0, 0.5], y_knots=[0.0, 1.0], y_min=0.0, y_max=1.0)
    out = m.apply([-0.3, 0.0, 0.25, 0.5, 2.0])
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[2] == pytest.approx(0.5)  # interpolated midpoint


def test_constant_predictions_are_identity():
    y = np.linspace(0, 1, 100)
    m = fit_weighted_isotonic(np.full(100, 0.3), y)
    assert np.allclose(m.apply([0.1, 0.5, 0.9]), np.clip([0.1, 0.5, 0.9], 0, 1))
