"""Tests for the gender-balanced loss and the LoRA-aware optimizer filter."""

from __future__ import annotations

from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import torch.nn as nn

from face_occlusion.metrics.challenge_metric import challenge_score
from face_occlusion.training.lit_module import (
    FaceOcclusionLitModule,
    gender_balanced_weighted_mse_loss,
    weighted_mse_loss,
)


def test_matches_challenge_metric_mean_term():
    # power=1, gap_lambda=0 must equal 0.5*(Err_F + Err_M) with the metric weight 1/30+y.
    torch.manual_seed(0)
    y = torch.rand(128)
    p = torch.rand(128)
    g = (torch.rand(128) > 0.5).float()
    loss = float(gender_balanced_weighted_mse_loss(p, y, g, female_value=0.0, male_value=1.0))
    cs = challenge_score(
        p.numpy(), y.numpy(), g.numpy(), female_value="0.0", male_value="1.0", clip=False
    )
    assert abs(loss - cs["err_mean"]) < 1e-5


def test_single_gender_batch_is_finite():
    y = torch.rand(10)
    p = torch.rand(10)
    g = torch.zeros(10)  # all female
    loss = gender_balanced_weighted_mse_loss(p, y, g)
    assert torch.isfinite(loss)


def test_gap_lambda_adds_gap_penalty():
    # Female perfect, male wrong -> nonzero gap; gap_lambda must raise the loss.
    y = torch.tensor([0.5, 0.5, 0.5, 0.5])
    p = torch.tensor([0.5, 0.5, 0.9, 0.9])
    g = torch.tensor([0.0, 0.0, 1.0, 1.0])
    base = float(gender_balanced_weighted_mse_loss(p, y, g, gap_lambda=0.0))
    with_gap = float(gender_balanced_weighted_mse_loss(p, y, g, gap_lambda=1.0))
    assert with_gap > base


def test_high_occ_power_changes_loss():
    torch.manual_seed(1)
    y = torch.rand(64)
    p = torch.rand(64)
    g = (torch.rand(64) > 0.5).float()
    l1 = float(gender_balanced_weighted_mse_loss(p, y, g, high_occ_power=1.0))
    l2 = float(gender_balanced_weighted_mse_loss(p, y, g, high_occ_power=2.0))
    assert l1 != l2


def test_gradients_flow():
    p = torch.rand(16, requires_grad=True)
    y = torch.rand(16)
    g = (torch.rand(16) > 0.5).float()
    loss = gender_balanced_weighted_mse_loss(p, y, g)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_weighted_mse_unchanged():
    # The default pooled loss is untouched.
    p = torch.tensor([0.1, 0.5, 0.9])
    y = torch.tensor([0.2, 0.4, 0.8])
    expected = float(weighted_mse_loss(p, y))
    assert expected == float(weighted_mse_loss(p, y, sample_weight=None))


def test_configure_optimizers_filters_frozen_params():
    module = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(module)
    trainable = nn.Linear(4, 1)
    frozen = nn.Linear(4, 4)
    for prm in frozen.parameters():
        prm.requires_grad = False
    module.add_module("_trainable", trainable)
    module.add_module("_frozen", frozen)
    module.cfg = SimpleNamespace(
        training=SimpleNamespace(learning_rate=1e-3, weight_decay=0.0, max_epochs=5)
    )
    opt = module.configure_optimizers()["optimizer"]
    n_opt = sum(p.numel() for grp in opt.param_groups for p in grp["params"])
    n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    assert n_opt == n_trainable == sum(p.numel() for p in trainable.parameters())
