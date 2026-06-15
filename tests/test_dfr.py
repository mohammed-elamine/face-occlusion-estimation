"""Tests for the DFR (group-balanced last-layer refit) core helpers."""

from __future__ import annotations

import numpy as np

from face_occlusion.training.dfr import (
    apply_head,
    fit_ridge_head,
    group_balance_weights,
    occlusion_group_ids,
)


def test_occlusion_group_ids():
    gids = occlusion_group_ids(
        targets=[0.01, 0.5, 0.9, 0.3],
        genders=[0.0, 1.0, 1.0, 0.5],  # last has unknown gender
        edges=[0.0, 0.4, 1.01],  # 2 occ bins
    )
    # female@bin0 -> 0; male@bin1 -> 3; male@bin1 -> 3; unknown -> -1
    assert gids.tolist() == [0, 3, 3, -1]


def test_group_balance_weights_equalize_group_totals():
    gids = np.array([0, 0, 0, 1])  # 3 in group 0, 1 in group 1
    w = group_balance_weights(gids)
    assert np.isclose(w[:3].sum(), w[3])  # equal total influence per group
    assert np.isclose(w.mean(), 1.0)  # normalized


def test_group_balance_weights_excludes_unknown_gender():
    w = group_balance_weights(np.array([-1, 0, 0]))
    assert w[0] == 0.0
    assert w[1] > 0 and w[2] > 0


def test_ridge_head_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    w_true = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ w_true + 0.7
    w, b = fit_ridge_head(X, y, ridge=1e-8)
    assert np.allclose(w, w_true, atol=1e-3)
    assert abs(b - 0.7) < 1e-3
    assert np.allclose(apply_head(X, w, b), y, atol=1e-3)


def test_sample_weight_pulls_fit():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    # heavily weight the second point -> intercept pulled up toward it
    _, b_eq = fit_ridge_head(X, y, sample_weight=[1.0, 1.0], ridge=1e-8)
    _, b_hi = fit_ridge_head(X, y, sample_weight=[1.0, 100.0], ridge=1e-8)
    assert b_hi > b_eq


def test_group_balancing_recovers_minority_group():
    # DFR mechanism: a minority group with a different intercept is otherwise drowned out.
    rng = np.random.default_rng(1)
    xa = rng.uniform(0, 1, 900)
    xb = rng.uniform(0, 1, 100)
    X = np.concatenate([xa, xb]).reshape(-1, 1)
    y = np.concatenate([2 * xa + 0.0, 2 * xb + 1.0])  # same slope, +1.0 offset for the minority
    gids = np.array([0] * 900 + [1] * 100)

    _, b_unweighted = fit_ridge_head(X, y, ridge=1e-8)
    w = group_balance_weights(gids)
    _, b_balanced = fit_ridge_head(X, y, sample_weight=w, ridge=1e-8)

    assert b_unweighted < 0.25  # dominated by the majority (offset ~0)
    assert 0.4 < b_balanced < 0.6  # balanced -> halfway between the two group offsets


def test_ridge_shrinks_weights():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 5))
    y = X @ np.array([5.0, 0, 0, 0, 0])
    w_small, _ = fit_ridge_head(X, y, ridge=1e-6)
    w_big, _ = fit_ridge_head(X, y, ridge=100.0)
    assert np.linalg.norm(w_big) < np.linalg.norm(w_small)


def test_bias_is_not_regularized():
    # With heavy ridge, weights -> 0 but the bias should still capture the mean.
    X = np.zeros((50, 3))
    y = np.full(50, 0.42)
    w, b = fit_ridge_head(X, y, ridge=1e6)
    assert abs(b - 0.42) < 1e-6
