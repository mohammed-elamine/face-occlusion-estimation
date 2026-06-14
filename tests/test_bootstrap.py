"""Tests for bootstrap confidence intervals on the challenge metric."""

from __future__ import annotations

import numpy as np
import pytest

from face_occlusion.metrics.bootstrap import MetricCI, bootstrap_challenge_metrics
from face_occlusion.metrics.challenge_metric import challenge_score


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.beta(1.5, 8.0, size=n)  # right-skewed like the real data
    preds = np.clip(targets + rng.normal(0, 0.05, size=n), 0, 1)
    genders = rng.integers(0, 2, size=n).astype(float)
    gender_str = np.array([f"{g:.1f}" for g in genders])
    group_ids = rng.integers(0, 40, size=n)  # identity-like clusters
    return preds, targets, gender_str, group_ids


def test_point_estimate_matches_challenge_score():
    preds, targets, genders, _ = _data()
    out = bootstrap_challenge_metrics(preds, targets, genders, n_boot=50, seed=1)
    direct = challenge_score(preds, targets, genders)
    assert out["score"].point == pytest.approx(direct["score"])
    assert out["err_female"].point == pytest.approx(direct["err_female"])
    assert out["gender_gap"].point == pytest.approx(direct["gender_gap"])


def test_ci_brackets_point_and_is_ordered():
    preds, targets, genders, _ = _data()
    out = bootstrap_challenge_metrics(preds, targets, genders, n_boot=300, seed=2)
    for key in ("score", "err_female", "err_male", "gender_gap"):
        ci = out[key]
        assert ci.lo <= ci.hi
        # The point estimate should sit inside (or on) a 95% CI for a smooth stat.
        assert ci.lo - 1e-9 <= ci.point <= ci.hi + 1e-9
        assert ci.std >= 0.0


def test_is_deterministic_given_seed():
    preds, targets, genders, _ = _data()
    a = bootstrap_challenge_metrics(preds, targets, genders, n_boot=100, seed=7)
    b = bootstrap_challenge_metrics(preds, targets, genders, n_boot=100, seed=7)
    assert a["score"].lo == b["score"].lo
    assert a["score"].hi == b["score"].hi


def test_group_bootstrap_widens_or_matches_ci_under_clustering():
    # Build data where rows within a group are identical -> i.i.d. row bootstrap
    # understates variance vs cluster bootstrap.
    rng = np.random.default_rng(3)
    n_groups = 20
    group_means = rng.beta(1.5, 8.0, size=n_groups)
    rows_per_group = 10
    targets, preds, genders, groups = [], [], [], []
    for gi in range(n_groups):
        t = group_means[gi]
        for _ in range(rows_per_group):
            targets.append(t)
            preds.append(min(1.0, max(0.0, t + 0.1)))  # constant bias within group
            genders.append(f"{gi % 2:.1f}")
            groups.append(gi)
    targets = np.array(targets)
    preds = np.array(preds)
    genders = np.array(genders)
    groups = np.array(groups)

    row_ci = bootstrap_challenge_metrics(
        preds, targets, genders, group_ids=groups, unit="row", n_boot=300, seed=4
    )["score"]
    grp_ci = bootstrap_challenge_metrics(
        preds, targets, genders, group_ids=groups, unit="group", n_boot=300, seed=4
    )["score"]
    row_width = row_ci.hi - row_ci.lo
    grp_width = grp_ci.hi - grp_ci.lo
    # Cluster bootstrap must not understate variance: its CI is at least as wide.
    assert grp_width >= row_width - 1e-9


def test_group_unit_requires_group_ids():
    preds, targets, genders, _ = _data(n=20)
    with pytest.raises(ValueError, match="requires group_ids"):
        bootstrap_challenge_metrics(preds, targets, genders, unit="group")


def test_metric_ci_as_dict():
    ci = MetricCI(0.1, 0.05, 0.15, 0.02)
    assert ci.as_dict() == {"point": 0.1, "lo": 0.05, "hi": 0.15, "std": 0.02}
