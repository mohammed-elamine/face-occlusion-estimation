"""Tests for the ordered-bin distribution head (DEX expectation + DLDL/LDS soft labels)."""

from __future__ import annotations

import pytest
import torch

from face_occlusion.models.distribution import (
    dldl_kl_loss,
    expectation,
    make_bin_centers,
    soft_label_distribution,
)
from face_occlusion.models.regressor import OcclusionRegressor

_TINY = "vit_tiny_patch16_224"


class TestDistributionMath:
    def test_make_bin_centers(self):
        c = make_bin_centers(21, 0.0, 1.0)
        assert c.shape == (21,)
        assert torch.isclose(c[0], torch.tensor(0.0)) and torch.isclose(c[-1], torch.tensor(1.0))
        with pytest.raises(ValueError):
            make_bin_centers(1)

    def test_soft_labels_sum_to_one_and_peak_at_target(self):
        c = make_bin_centers(21)
        y = torch.tensor([0.0, 0.5, 1.0])
        p = soft_label_distribution(y, c, sigma=0.05)
        assert p.shape == (3, 21)
        assert torch.allclose(p.sum(dim=1), torch.ones(3), atol=1e-5)
        # peak bin is the one nearest the target
        for i, yy in enumerate(y):
            assert torch.isclose(c[p[i].argmax()], yy, atol=0.05)

    def test_soft_label_expectation_recovers_target(self):
        c = make_bin_centers(41)
        y = torch.tensor([0.07, 0.33, 0.61])
        p = soft_label_distribution(y, c, sigma=0.04)
        assert torch.allclose(expectation(p, c), y, atol=0.02)

    def test_expectation_onehot_recovers_center(self):
        c = make_bin_centers(5)  # 0,.25,.5,.75,1
        onehot = torch.tensor([[0.0, 0, 1, 0, 0], [0, 0, 0, 0, 1.0]])
        assert torch.allclose(expectation(onehot, c), torch.tensor([0.5, 1.0]))

    def test_dldl_loss_nonneg_and_min_at_match(self):
        c = make_bin_centers(21)
        soft = soft_label_distribution(torch.tensor([0.2, 0.4]), c, sigma=0.05)
        # logits that reproduce the soft target -> ~0 KL; perturbed -> larger
        matched = torch.log(soft.clamp_min(1e-8))
        worse = torch.zeros_like(soft)  # uniform logits
        loss_matched = dldl_kl_loss(matched, soft)
        loss_worse = dldl_kl_loss(worse, soft)
        assert loss_matched >= -1e-6
        assert loss_worse > loss_matched + 1e-4

    def test_sigma_must_be_positive(self):
        with pytest.raises(ValueError):
            soft_label_distribution(torch.tensor([0.5]), make_bin_centers(5), sigma=0.0)


class TestDistributionHeadModel:
    def test_builds_and_forward(self):
        m = OcclusionRegressor(
            backbone=_TINY,
            pretrained=False,
            head={"type": "distribution", "n_bins": 21, "range": [0.0, 1.0]},
        ).eval()
        assert m.bin_centers.shape == (21,)
        out = m(torch.randn(4, 3, 224, 224))
        assert out.y_pred.shape == (4,)
        assert bool((out.y_pred >= 0).all() and (out.y_pred <= 1).all())  # bounded expectation
        assert out.bin_logits.shape == (4, 21)

    def test_param_groups_has_head_group(self):
        m = OcclusionRegressor(
            backbone=_TINY, pretrained=False, head={"type": "distribution", "n_bins": 11}
        )
        groups = m.param_groups(head_lr=1e-3, backbone_lr=1e-4, weight_decay=1e-4)
        assert {g["lr"] for g in groups} == {1e-3, 1e-4}

    def test_distribution_incompatible_with_ordinal_head(self):
        with pytest.raises(ValueError):
            OcclusionRegressor(
                backbone=_TINY,
                pretrained=False,
                use_ordinal_head=True,
                head={"type": "distribution"},
            )
