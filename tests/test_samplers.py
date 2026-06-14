"""Tests for GenderOcclusionBalancedBatchSampler."""

from __future__ import annotations

import numpy as np
import pytest

from face_occlusion.data.samplers import GenderOcclusionBalancedBatchSampler

BINS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]
BIN_WEIGHTS = {
    "0.00_0.05": 1.0,
    "0.05_0.10": 1.2,
    "0.10_0.20": 1.5,
    "0.20_0.40": 2.0,
    "0.40_0.60": 3.0,
    "0.60_1.00": 4.0,
}


def _make_imbalanced_dataset(n: int = 1000, seed: int = 0):
    """Create a dataset where low occlusion dominates and high occlusion is rare."""
    rng = np.random.default_rng(seed)
    # 80% low occlusion [0, 0.1], 15% medium [0.1, 0.4], 5% high [0.4, 1.0]
    n_low = int(n * 0.80)
    n_med = int(n * 0.15)
    n_high = n - n_low - n_med
    targets = np.concatenate(
        [
            rng.uniform(0.0, 0.10, n_low),
            rng.uniform(0.10, 0.40, n_med),
            rng.uniform(0.40, 1.00, n_high),
        ]
    )
    genders = rng.choice([0.0, 1.0], size=n)
    return targets, genders


class TestSamplerBasic:
    """Test that the sampler creates valid batches."""

    def test_yields_correct_batch_sizes(self):
        targets, genders = _make_imbalanced_dataset(200)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=16,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            drop_last=True,
            seed=42,
        )
        batches = list(sampler)
        assert len(batches) > 0
        for batch in batches:
            assert len(batch) == 16

    def test_indices_within_range(self):
        n = 300
        targets, genders = _make_imbalanced_dataset(n)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=32,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            drop_last=True,
            seed=42,
        )
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < n

    def test_num_batches(self):
        n = 100
        targets, genders = _make_imbalanced_dataset(n)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=16,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            drop_last=True,
            seed=42,
        )
        # With drop_last=True: 100 // 16 = 6
        assert len(sampler) == 6
        assert len(list(sampler)) == 6

    def test_num_batches_no_drop_last(self):
        n = 100
        targets, genders = _make_imbalanced_dataset(n)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=16,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            drop_last=False,
            seed=42,
        )
        # ceil(100 / 16) = 7
        assert len(sampler) == 7
        batches = list(sampler)
        assert len(batches) == 7
        # Last batch should be smaller.
        assert len(batches[-1]) == 100 % 16

    def test_custom_num_samples(self):
        targets, genders = _make_imbalanced_dataset(200)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=32,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            num_samples=128,
            drop_last=True,
            seed=42,
        )
        assert len(sampler) == 4
        assert len(list(sampler)) == 4


class TestHighOcclusionExposure:
    """Test that high-occlusion samples get more exposure than under random sampling."""

    def test_high_occlusion_overrepresented(self):
        n = 2000
        targets, genders = _make_imbalanced_dataset(n, seed=1)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=64,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            balance_strength=0.8,
            seed=42,
        )

        # Collect sampled indices across all batches.
        sampled = []
        for batch in sampler:
            sampled.extend(batch)

        sampled_targets = targets[sampled]
        # In original data, high occlusion (>0.4) is ~5%.
        original_high_frac = (targets > 0.4).mean()
        sampled_high_frac = (sampled_targets > 0.4).mean()
        # Sampled fraction should be noticeably higher.
        assert sampled_high_frac > original_high_frac * 1.5


class TestGenderCorrection:
    """Test that gender correction prevents one gender from dominating high-occlusion batches."""

    def test_minority_gender_gets_exposure(self):
        rng = np.random.default_rng(99)
        n = 1000
        # High occlusion: mostly female (gender=0), very few male.
        n_high = 100
        n_low = n - n_high
        targets = np.concatenate(
            [
                rng.uniform(0.0, 0.10, n_low),
                rng.uniform(0.60, 1.00, n_high),
            ]
        )
        genders = np.concatenate(
            [
                rng.choice([0.0, 1.0], size=n_low),
                np.concatenate([np.zeros(90), np.ones(10)]),  # 90 female, 10 male high-occ
            ]
        )

        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=64,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            gender_balance_strength=1.0,
            balance_strength=1.0,
            seed=42,
        )

        sampled = []
        for batch in sampler:
            sampled.extend(batch)

        # Among sampled high-occlusion items, male should appear more than 10% (their original
        # proportion is 10/100 = 10%). With gender correction, they should be boosted.
        sampled_arr = np.array(sampled)
        high_mask = targets[sampled_arr] > 0.6
        if high_mask.sum() > 0:
            male_frac = genders[sampled_arr[high_mask]].mean()
            # Male should have >15% share (boosted from 10%).
            assert male_frac > 0.15, f"Male fraction in high-occ was only {male_frac:.2%}"


class TestEdgeCases:
    """Test edge cases: missing columns, empty strata, boundary targets."""

    def test_invalid_gender_raises(self):
        targets = np.array([0.1, 0.2, 0.3])
        genders = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="invalid values"):
            GenderOcclusionBalancedBatchSampler(
                targets=targets,
                genders=genders,
                batch_size=2,
                bins=BINS,
                bin_weights=BIN_WEIGHTS,
            )

    def test_empty_strata_skipped(self):
        # All samples are female, low occlusion -> male strata are empty.
        targets = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        genders = np.zeros(5)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=2,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
        )
        # Should not crash, should have only female strata.
        batches = list(sampler)
        assert len(batches) > 0

    def test_target_1_0_in_last_bin(self):
        targets = np.array([0.0, 0.5, 1.0])
        genders = np.array([0.0, 1.0, 0.0])
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=2,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
        )
        # target=1.0 should be in the last bin (0.60_1.00), index 2 in the dataset.
        last_bin_idx = len(BINS) - 2  # bin index 5
        assert (0, last_bin_idx) in sampler._strata
        assert 2 in sampler._strata[(0, last_bin_idx)]

    def test_invalid_bins_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            GenderOcclusionBalancedBatchSampler(
                targets=np.array([0.1]),
                genders=np.array([0.0]),
                batch_size=1,
                bins=[0.5, 0.3, 1.0],
                bin_weights={},
            )

    def test_too_few_bins_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            GenderOcclusionBalancedBatchSampler(
                targets=np.array([0.1]),
                genders=np.array([0.0]),
                batch_size=1,
                bins=[0.5],
                bin_weights={},
            )

    def test_summary_structure(self):
        targets, genders = _make_imbalanced_dataset(100)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=16,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            seed=42,
        )
        summary = sampler.summary
        assert summary["strategy"] == "gender_occlusion_balanced_batch"
        assert summary["batch_size"] == 16
        assert isinstance(summary["strata"], list)
        assert all("gender" in s and "final_prob" in s for s in summary["strata"])

    def test_reproducibility(self):
        targets, genders = _make_imbalanced_dataset(200)
        kwargs = dict(
            targets=targets,
            genders=genders,
            batch_size=16,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            seed=123,
        )
        batches_1 = [b for b in GenderOcclusionBalancedBatchSampler(**kwargs)]
        batches_2 = [b for b in GenderOcclusionBalancedBatchSampler(**kwargs)]
        assert batches_1 == batches_2


class TestExposureCap:
    """Verify the per-image repeat cap from `max_repeats_per_image`."""

    def _tiny_high_dataset(self):
        # 1000 samples, only 2 of them in the highest-occlusion bin.
        rng = np.random.default_rng(7)
        n = 1000
        n_low = n - 2
        targets = np.concatenate([rng.uniform(0.0, 0.1, n_low), np.array([0.95, 0.98])])
        genders = np.concatenate([rng.choice([0.0, 1.0], size=n_low), np.array([0.0, 1.0])])
        return targets, genders

    def test_no_image_exceeds_repeat_cap(self):
        targets, genders = self._tiny_high_dataset()
        max_reps = 5
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=32,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            balance_strength=1.0,
            max_repeats_per_image=max_reps,
            seed=0,
        )
        from collections import Counter

        counts = Counter(i for batch in sampler for i in batch)
        assert counts, "sampler produced no batches"
        assert max(counts.values()) <= max_reps

    def test_tiny_stratum_total_draws_bounded(self):
        targets, genders = self._tiny_high_dataset()
        max_reps = 4
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=32,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            balance_strength=1.0,
            max_repeats_per_image=max_reps,
            seed=1,
        )
        from collections import Counter

        counts = Counter(i for batch in sampler for i in batch)
        # The two high-occlusion samples are at indices 998, 999.
        for tiny_idx in (998, 999):
            assert counts[tiny_idx] <= max_reps

    def test_capped_strata_flagged_in_summary(self):
        targets, genders = self._tiny_high_dataset()
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=32,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            balance_strength=1.0,
            size_aware_weighting=False,
            max_repeats_per_image=10,
            seed=0,
        )
        summary = sampler.summary
        assert summary["num_capped_strata"] >= 1
        capped = [s for s in summary["strata"] if s["was_capped"]]
        assert capped, "expected at least one capped stratum"
        for s in capped:
            # Cap is n_s * max_repeats_per_image.
            assert s["draws_after_cap"] <= s["count"] * sampler.max_repeats_per_image


class TestBalanceStrength:
    """`balance_strength` interpolates between natural and balanced sampling."""

    def test_zero_strength_close_to_natural(self):
        targets, genders = _make_imbalanced_dataset(2000, seed=3)
        sampler = GenderOcclusionBalancedBatchSampler(
            targets=targets,
            genders=genders,
            batch_size=64,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            balance_strength=0.0,
            seed=42,
        )
        sampled = np.array([i for batch in sampler for i in batch])
        natural_high = (targets > 0.4).mean()
        sampled_high = (targets[sampled] > 0.4).mean()
        # Within sampling noise of the natural distribution.
        assert abs(sampled_high - natural_high) < 0.02

    def test_higher_strength_lifts_high_occlusion(self):
        targets, genders = _make_imbalanced_dataset(2000, seed=4)
        kwargs = dict(
            targets=targets,
            genders=genders,
            batch_size=64,
            bins=BINS,
            bin_weights=BIN_WEIGHTS,
            seed=42,
        )
        s_low = GenderOcclusionBalancedBatchSampler(balance_strength=0.1, **kwargs)
        s_high = GenderOcclusionBalancedBatchSampler(balance_strength=0.8, **kwargs)
        frac_low = (targets[np.array([i for b in s_low for i in b])] > 0.4).mean()
        frac_high = (targets[np.array([i for b in s_high for i in b])] > 0.4).mean()
        assert frac_high > frac_low


class TestBuildFromConfig:
    """Factory `build_batch_sampler_from_config` resolves column names from cfg.data."""

    @staticmethod
    def _cfg(target_col="FaceOcclusion", gender_col="gender", **sampler_overrides):
        import pandas as pd  # noqa: F401

        sampler_block = {
            "enabled": True,
            "strategy": "gender_occlusion_balanced_batch",
            "bins": BINS,
            "bin_weights": BIN_WEIGHTS,
            "balance_strength": 0.3,
            "gender_balance_strength": 0.5,
            "max_stratum_weight": 8.0,
            "min_stratum_size": 5,
            "size_aware_weighting": True,
            "reliable_stratum_size": 20,
            "max_repeats_per_image": 10,
            "tiny_stratum_policy": "cap",
            "drop_last": False,
            "num_samples": None,
            "seed": 42,
            **sampler_overrides,
        }
        return {
            "data": {"target_col": target_col, "gender_col": gender_col},
            "sampler": sampler_block,
        }

    def test_uses_configured_target_and_gender_columns(self):
        import pandas as pd

        from face_occlusion.data.samplers import build_batch_sampler_from_config

        targets, genders = _make_imbalanced_dataset(200)
        df = pd.DataFrame({"FaceOcclusion": targets, "gender": genders})
        sampler = build_batch_sampler_from_config(df, self._cfg(), batch_size=32)
        assert sampler is not None
        assert len(list(sampler)) > 0

    def test_missing_target_column_raises_clear_error(self):
        import pandas as pd

        from face_occlusion.data.samplers import build_batch_sampler_from_config

        targets, genders = _make_imbalanced_dataset(50)
        df = pd.DataFrame({"some_other_col": targets, "gender": genders})
        with pytest.raises(KeyError) as exc:
            build_batch_sampler_from_config(df, self._cfg(), batch_size=8)
        msg = str(exc.value)
        assert "FaceOcclusion" in msg
        assert "data.target_col" in msg
        assert "some_other_col" in msg

    def test_sampler_block_can_override_columns(self):
        import pandas as pd

        from face_occlusion.data.samplers import build_batch_sampler_from_config

        targets, genders = _make_imbalanced_dataset(120)
        df = pd.DataFrame({"y": targets, "sex": genders})
        cfg = self._cfg(
            target_col="FaceOcclusion",
            gender_col="gender",
            target_col_override="y",
            gender_col_override="sex",
        )
        # Sampler-level overrides win over data-level defaults.
        cfg["sampler"]["target_col"] = "y"
        cfg["sampler"]["gender_col"] = "sex"
        sampler = build_batch_sampler_from_config(df, cfg, batch_size=16)
        assert sampler is not None

    def test_disabled_sampler_returns_none(self):
        import pandas as pd

        from face_occlusion.data.samplers import build_batch_sampler_from_config

        df = pd.DataFrame({"FaceOcclusion": [0.1, 0.2], "gender": [0, 1]})
        cfg = self._cfg()
        cfg["sampler"]["enabled"] = False
        assert build_batch_sampler_from_config(df, cfg, batch_size=2) is None
