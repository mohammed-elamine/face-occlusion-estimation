"""Unit tests for the regression–ordinal consistency loss (Stage 2)."""

from __future__ import annotations

import pytest
import torch

from face_occlusion.models.ordinal import (
    CONSISTENCY_MODES,
    regression_ordinal_consistency_loss,
)

THRESHOLDS = torch.tensor([0.05, 0.10, 0.20, 0.40, 0.60])


def _logits_for_targets(y: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Hard logits whose sigmoid equals 1[y > t_k] (saturated)."""
    targets = (y.unsqueeze(-1) > thresholds.unsqueeze(0)).float()
    return targets * 20.0 - 10.0  # ≈sigmoid(±10) ⇒ ~1 / ~0


# ─── shape / scalar ───────────────────────────────────────────────────────────


def test_loss_is_scalar_tensor():
    y = torch.tensor([0.1, 0.5, 0.9])
    logits = torch.zeros(3, THRESHOLDS.numel())
    loss = regression_ordinal_consistency_loss(y, logits, THRESHOLDS)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_loss_accepts_column_y_pred():
    y = torch.tensor([[0.3]])
    logits = torch.zeros(1, THRESHOLDS.numel())
    loss = regression_ordinal_consistency_loss(y, logits, THRESHOLDS)
    assert loss.ndim == 0


# ─── coherence: smaller when branches agree ──────────────────────────────────


def test_loss_smaller_when_branches_agree():
    y = torch.tensor([0.03, 0.35, 0.70])
    agree_logits = _logits_for_targets(y, THRESHOLDS)
    # Contradict every threshold.
    contradict_logits = -agree_logits

    loss_agree = regression_ordinal_consistency_loss(y, agree_logits, THRESHOLDS)
    loss_disagree = regression_ordinal_consistency_loss(y, contradict_logits, THRESHOLDS)
    assert loss_agree < loss_disagree
    assert loss_agree < 0.1


# ─── modes ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", CONSISTENCY_MODES)
def test_all_supported_modes_run(mode: str):
    y = torch.tensor([0.2, 0.5])
    logits = torch.zeros(2, THRESHOLDS.numel())
    loss = regression_ordinal_consistency_loss(y, logits, THRESHOLDS, mode=mode)
    assert torch.isfinite(loss).item()


def test_invalid_mode_raises():
    y = torch.tensor([0.2])
    logits = torch.zeros(1, THRESHOLDS.numel())
    with pytest.raises(ValueError):
        regression_ordinal_consistency_loss(y, logits, THRESHOLDS, mode="bogus")


def test_non_positive_temperature_raises():
    y = torch.tensor([0.2])
    logits = torch.zeros(1, THRESHOLDS.numel())
    with pytest.raises(ValueError):
        regression_ordinal_consistency_loss(y, logits, THRESHOLDS, temperature=0.0)


# ─── gradient detach behaviour ────────────────────────────────────────────────


def _grad_norm(t: torch.Tensor) -> float:
    return 0.0 if t.grad is None else float(t.grad.detach().abs().sum().item())


def test_symmetric_mode_propagates_to_both_branches():
    y = torch.tensor([0.3], requires_grad=True)
    logits = torch.zeros(1, THRESHOLDS.numel(), requires_grad=True)
    loss = regression_ordinal_consistency_loss(y, logits, THRESHOLDS, mode="symmetric")
    loss.backward()
    assert _grad_norm(y) > 0
    assert _grad_norm(logits) > 0


def test_ordinal_as_teacher_blocks_ordinal_grad():
    y = torch.tensor([0.3], requires_grad=True)
    logits = torch.zeros(1, THRESHOLDS.numel(), requires_grad=True)
    loss = regression_ordinal_consistency_loss(y, logits, THRESHOLDS, mode="ordinal_as_teacher")
    loss.backward()
    assert _grad_norm(y) > 0
    assert _grad_norm(logits) == 0.0


def test_regression_as_teacher_blocks_regression_grad():
    y = torch.tensor([0.3], requires_grad=True)
    logits = torch.zeros(1, THRESHOLDS.numel(), requires_grad=True)
    loss = regression_ordinal_consistency_loss(y, logits, THRESHOLDS, mode="regression_as_teacher")
    loss.backward()
    assert _grad_norm(y) == 0.0
    assert _grad_norm(logits) > 0


def test_thresholds_moved_to_y_dtype():
    y = torch.tensor([0.3], dtype=torch.float64)
    logits = torch.zeros(1, THRESHOLDS.numel(), dtype=torch.float64)
    loss = regression_ordinal_consistency_loss(y, logits, THRESHOLDS.float())
    assert loss.dtype == torch.float64
