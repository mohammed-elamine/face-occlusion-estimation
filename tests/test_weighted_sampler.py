"""Tests for the teammate-style per-sample WeightedRandomSampler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import WeightedRandomSampler

from face_occlusion.data.samplers import (
    build_batch_sampler_from_config,
    build_weighted_sampler_from_config,
    compute_weighted_sample_weights,
)


class TestComputeWeights:
    def test_gender_occ_oversamples_high_occlusion(self):
        # Same gender -> the gender term is constant, so weight rises with occlusion.
        targets = np.array([0.0, 0.0, 0.5, 0.5])
        genders = np.array([0.0, 0.0, 0.0, 0.0])
        w = compute_weighted_sample_weights(targets, genders, mode="gender_occ", occ_power=0.5)
        assert w[2] > w[0] and w[3] > w[1]
        assert np.isclose(w.mean(), 1.0)

    def test_gender_occ_upweights_minority_gender(self):
        # Constant occlusion -> the inverse-frequency term dominates; rare gender wins.
        targets = np.full(4, 0.1)
        genders = np.array([0.0, 0.0, 0.0, 1.0])  # gender 1 is the minority
        w = compute_weighted_sample_weights(targets, genders, mode="gender_occ")
        assert w[3] > w[0]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown weighted sampler mode"):
            compute_weighted_sample_weights(np.array([0.1]), np.array([0.0]), mode="bogus")


def _df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "FaceOcclusion": rng.uniform(0.0, 0.6, n),
            "gender": rng.choice([0.0, 1.0], size=n),
        }
    )


def _cfg(**sampler):
    return {"sampler": sampler, "data": {"target_col": "FaceOcclusion", "gender_col": "gender"}}


class TestBuildWeightedSampler:
    def test_builds_weighted_random_sampler(self):
        df = _df(200)
        cfg = _cfg(enabled=True, strategy="gender_occ_weighted", mode="gender_occ", occ_power=0.5)
        sampler = build_weighted_sampler_from_config(df, cfg, seed=42)
        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == len(df)  # one epoch == dataset size
        assert sampler.replacement is True
        assert len(list(sampler)) == len(df)

    def test_disabled_returns_none(self):
        assert build_weighted_sampler_from_config(_df(), _cfg(enabled=False), seed=42) is None

    def test_batch_strategy_returns_none(self):
        # A batch strategy is handled elsewhere; the weighted builder defers.
        cfg = _cfg(enabled=True, strategy="gender_occlusion_balanced_batch")
        assert build_weighted_sampler_from_config(_df(), cfg, seed=42) is None

    def test_reproducible_with_seed(self):
        df = _df(120)
        cfg = _cfg(enabled=True, strategy="gender_occ_weighted")
        a = list(build_weighted_sampler_from_config(df, cfg, seed=7))
        b = list(build_weighted_sampler_from_config(df, cfg, seed=7))
        assert a == b


class TestBatchFactoryDefers:
    def test_batch_factory_returns_none_for_weighted_strategy(self):
        # Must defer (return None), not raise, so the datamodule can route it.
        cfg = _cfg(enabled=True, strategy="gender_occ_weighted")
        assert build_batch_sampler_from_config(_df(), cfg, batch_size=16) is None

    def test_batch_factory_raises_on_unknown_strategy(self):
        cfg = _cfg(enabled=True, strategy="totally_unknown")
        with pytest.raises(ValueError, match="Unknown sampler strategy"):
            build_batch_sampler_from_config(_df(), cfg, batch_size=16)
