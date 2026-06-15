"""Tests for the synthetic monotonic ranking loss and its training-step wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch

from face_occlusion.models.outputs import OcclusionModelOutput
from face_occlusion.models.ranking import (
    monotonic_ranking_loss,
    ordering_accuracy,
    ranknet_loss,
)
from face_occlusion.training.lit_module import FaceOcclusionLitModule

# ─── pure losses ──────────────────────────────────────────────────────────────


def test_ranknet_loss_lower_when_order_satisfied():
    higher = torch.tensor([2.0, 3.0])
    lower = torch.tensor([0.0, 1.0])
    good = ranknet_loss(higher, lower)
    bad = ranknet_loss(lower, higher)  # reversed -> order violated
    assert good < bad
    assert good > 0  # softplus is never exactly zero


def test_ranknet_loss_empty_is_zero():
    assert ranknet_loss(torch.empty(0), torch.empty(0)).item() == 0.0


def test_ranknet_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ranknet_loss(torch.zeros(3), torch.zeros(2))


def test_monotonic_ranking_loss_rewards_increasing_scores():
    clean = torch.tensor([0.0, 0.1])
    mild = torch.tensor([0.3, 0.4])
    strong = torch.tensor([0.7, 0.8])
    ordered = monotonic_ranking_loss(clean, mild, strong)
    scrambled = monotonic_ranking_loss(strong, mild, clean)
    assert ordered < scrambled


def test_ordering_accuracy_counts_strictly_increasing():
    clean = torch.tensor([0.0, 0.5, 0.0])
    mild = torch.tensor([0.1, 0.4, 0.2])  # 2nd triple violates (0.5>0.4)
    strong = torch.tensor([0.2, 0.9, 0.3])
    assert ordering_accuracy(clean, mild, strong).item() == pytest.approx(2.0 / 3.0)


def test_ordering_accuracy_empty_is_zero():
    e = torch.empty(0)
    assert ordering_accuracy(e, e, e).item() == 0.0


def test_monotonic_ranking_loss_is_differentiable():
    clean = torch.zeros(2, requires_grad=True)
    mild = torch.ones(2, requires_grad=True)
    strong = 2 * torch.ones(2, requires_grad=True)
    monotonic_ranking_loss(clean, mild, strong).backward()
    assert clean.grad is not None and torch.any(clean.grad != 0)


# ─── training-step integration ────────────────────────────────────────────────


class _ScoreModel(torch.nn.Module):
    """Maps a (N, 1) image to its scalar value, so scores are controllable."""

    use_ordinal_head = False

    def forward(self, x):
        return OcclusionModelOutput(y_pred=x.reshape(x.shape[0], -1).mean(dim=1))


def _ranking_module(weight=0.1):
    m = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(m)
    m.cfg = SimpleNamespace(split=SimpleNamespace(occlusion_bins=[0.0, 0.5, 1.0]))
    m._val_buffer = []
    m._female_value = "0.0"
    m._male_value = "1.0"
    m.model = _ScoreModel()
    for attr in ("_ord_loss_enabled", "_cons_loss_enabled", "_mono_loss_enabled"):
        setattr(m, attr, False)
    m._rank_loss_enabled = True
    m._rank_weight = weight
    m._rank_warmup_epochs = 0
    m._rank_warmup_start_weight = 0.0
    m._bgc_enabled = False
    m._shadow_loss_enabled = False
    m._reg_reweight = "none"
    m._reg_bin_weights = None
    m._reg_edges = None
    m._reg_loss_type = "weighted_mse"
    m._reg_high_occ_power = 1.0
    m._reg_gap_lambda = 0.0
    return m


def _ranking_batch(valid_flags):
    n = len(valid_flags)
    # clean<mild<strong by construction via constant-valued 1-pixel "images".
    clean = torch.full((n, 1), 0.1)
    mild = torch.full((n, 1), 0.3)
    strong = torch.full((n, 1), 0.6)
    return {
        "image": clean.clone(),
        "target": torch.full((n,), 0.3),
        "synthetic_clean_image": clean,
        "synthetic_mild_image": mild,
        "synthetic_strong_image": strong,
        "synthetic_valid": torch.tensor(valid_flags),
    }


def test_training_step_adds_ranking_term():
    module = _ranking_module(weight=0.5)
    logs = {}
    module.log = lambda name, value, *a, **k: logs.__setitem__(
        name, float(value.detach().item()) if torch.is_tensor(value) else float(value)
    )
    module.optimizers = lambda: None

    class _T:
        current_epoch = 0

    module._trainer = _T()
    total = module.training_step(_ranking_batch([True, True]), 0)
    assert torch.isfinite(total)
    assert "train/loss_rank" in logs
    assert logs["train/lambda_rank"] == 0.5
    assert logs["train/rank_ordering_acc"] == 1.0  # clean<mild<strong holds
    # total = loss_reg + 0.5 * loss_rank
    expected = logs["train/loss_reg"] + 0.5 * logs["train/loss_rank"]
    assert abs(logs["train/loss"] - expected) < 1e-5


def test_training_step_skips_ranking_when_no_valid_rows():
    module = _ranking_module()
    logs = {}
    module.log = lambda name, value, *a, **k: logs.__setitem__(
        name, float(value.detach().item()) if torch.is_tensor(value) else float(value)
    )
    module.optimizers = lambda: None

    class _T:
        current_epoch = 0

    module._trainer = _T()
    module.training_step(_ranking_batch([False, False]), 0)
    assert "train/loss_rank" not in logs
