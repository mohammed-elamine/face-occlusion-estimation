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
