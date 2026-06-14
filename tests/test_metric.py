"""Tests for the challenge metric helpers."""

import numpy as np
import pytest

from face_occlusion.metrics.challenge_metric import (
    challenge_score,
    error_by_occlusion_bin,
    weighted_mse,
    weighted_mse_by_group,
)


def test_weighted_mse_perfect():
    y = np.array([0.0, 0.2, 0.8])
    assert weighted_mse(y, y) == 0.0


def test_weighted_mse_known_value():
    y = np.array([0.0, 1.0])
    p = np.array([0.0, 0.0])
    # weights: 1/30, 1/30 + 1 = 31/30; numerator = (31/30)*1 = 31/30; denom = 32/30
    expected = (31 / 30) / (32 / 30)
    assert abs(weighted_mse(p, y, clip=False) - expected) < 1e-9


def test_weighted_mse_clips():
    y = np.array([0.0, 1.0])
    p_raw = np.array([-1.0, 2.0])
    p_clipped = np.array([0.0, 1.0])
    assert weighted_mse(p_raw, y, clip=True) == weighted_mse(p_clipped, y, clip=False)


def test_challenge_score_balanced_groups():
    y = np.array([0.1, 0.5, 0.9, 0.2])
    p = np.array([0.1, 0.5, 0.9, 0.2])
    genders = np.array(["0.0", "0.0", "1.0", "1.0"])
    result = challenge_score(p, y, genders)
    assert result["score"] == 0.0
    assert result["err_female"] == 0.0
    assert result["err_male"] == 0.0
    assert result["gender_gap"] == 0.0


def test_challenge_score_missing_group_falls_back():
    y = np.array([0.1, 0.5])
    p = np.array([0.1, 0.5])
    genders = np.array(["1.0", "1.0"])
    with pytest.warns(RuntimeWarning, match="missing one gender subgroup"):
        result = challenge_score(p, y, genders)
    assert np.isnan(result["err_female"])
    assert result["score"] == 0.0


def test_challenge_score_uses_female_zero_male_one_by_default():
    y = np.array([1.0, 1.0, 0.5, 0.5])
    p = np.array([0.0, 0.0, 0.5, 0.5])
    genders = np.array([0.0, 0.0, 1.0, 1.0])
    result = challenge_score(p, y, genders)
    assert result["err_female"] > 0.0
    assert result["err_male"] == 0.0


def test_challenge_score_combines_mean_and_gap_with_known_values():
    # Load-bearing: pin score == mean(Err_F, Err_M) + |Err_F - Err_M| with a
    # deliberately UNBALANCED case, so a regression to "mean only" or
    # "mean - gap" would fail (the balanced tests cannot catch that).
    #   female: pred 0.5, target 0.0 -> w=1/30, WMSE = 0.25
    #   male:   pred 1.0, target 0.0 -> w=1/30, WMSE = 1.0
    p = np.array([0.5, 1.0])
    y = np.array([0.0, 0.0])
    genders = np.array(["0.0", "1.0"])
    result = challenge_score(p, y, genders)
    assert result["err_female"] == pytest.approx(0.25)
    assert result["err_male"] == pytest.approx(1.0)
    assert result["err_mean"] == pytest.approx(0.625)
    assert result["gender_gap"] == pytest.approx(0.75)
    # The whole point: mean + |gap|, not mean alone (0.625) nor mean - gap.
    assert result["score"] == pytest.approx(1.375)


def test_by_group():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.0, 1.0, 0.5, 0.5])
    g = np.array(["a", "a", "b", "b"])
    res = weighted_mse_by_group(p, y, g)
    assert res["a"] == 0.0
    assert res["b"] > 0.0


def test_bins():
    y = np.array([0.01, 0.5, 0.95])
    p = np.array([0.0, 0.5, 1.0])
    bins = [0.0, 0.1, 0.6, 1.0]
    res = error_by_occlusion_bin(p, y, bins=bins)
    assert set(res.keys()) == {"0.00_0.10", "0.10_0.60", "0.60_1.00"}


# ── Optional sample_weight (evaluation lenses) ─────────────────────────────────


def test_sample_weight_none_is_unchanged():
    rng = np.random.default_rng(3)
    y = rng.beta(1.5, 8.0, size=300)
    p = np.clip(y + rng.normal(0, 0.05, size=300), 0, 1)
    g = rng.integers(0, 2, size=300).astype(float)
    assert weighted_mse(p, y, sample_weight=None) == weighted_mse(p, y)
    base = challenge_score(p, y, g)
    none = challenge_score(p, y, g, sample_weight=None)
    assert none["score"] == base["score"]


def test_all_ones_weight_equals_unweighted():
    rng = np.random.default_rng(4)
    y = rng.beta(1.5, 8.0, size=300)
    p = np.clip(y + rng.normal(0, 0.05, size=300), 0, 1)
    g = rng.integers(0, 2, size=300).astype(float)
    ones = np.ones_like(y)
    assert weighted_mse(p, y, sample_weight=ones) == pytest.approx(weighted_mse(p, y))
    assert challenge_score(p, y, g, sample_weight=ones)["score"] == pytest.approx(
        challenge_score(p, y, g)["score"]
    )


def test_sample_weight_upweights_targeted_rows():
    # Two rows; weighting the erroneous row more must raise the weighted error.
    y = np.array([0.5, 0.5])
    p = np.array([0.5, 0.0])  # second row is wrong
    low = weighted_mse(p, y, clip=False, sample_weight=np.array([1.0, 1.0]))
    high = weighted_mse(p, y, clip=False, sample_weight=np.array([1.0, 5.0]))
    assert high > low
