"""Unit tests for the ordinal helpers (Stage 1)."""

from __future__ import annotations

import pytest
import torch

from face_occlusion.models.ordinal import (
    OrdinalHead,
    make_ordinal_targets,
    threshold_weighted_bce,
)

# ─── make_ordinal_targets ─────────────────────────────────────────────────────


def test_make_ordinal_targets_basic_pattern():
    y = torch.tensor([0.03, 0.35, 0.70])
    thresholds = torch.tensor([0.05, 0.10, 0.20, 0.40, 0.60])
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    out = make_ordinal_targets(y, thresholds)
    assert out.shape == (3, 5)
    assert torch.equal(out, expected)


def test_make_ordinal_targets_accepts_column_y():
    y = torch.tensor([[0.35]])
    thresholds = torch.tensor([0.05, 0.40])
    out = make_ordinal_targets(y, thresholds)
    assert out.shape == (1, 2)
    assert torch.equal(out, torch.tensor([[1.0, 0.0]]))


def test_make_ordinal_targets_preserves_device_and_dtype():
    y = torch.tensor([0.5], dtype=torch.float64)
    thresholds = torch.tensor([0.4], dtype=torch.float32)  # different dtype
    out = make_ordinal_targets(y, thresholds)
    assert out.dtype == y.dtype
    assert out.device == y.device


def test_make_ordinal_targets_strict_inequality_at_boundary():
    # y == t_k must NOT count as positive (formula is y > t_k).
    y = torch.tensor([0.40])
    thresholds = torch.tensor([0.40])
    out = make_ordinal_targets(y, thresholds)
    assert out.item() == 0.0


# ─── threshold_weighted_bce ───────────────────────────────────────────────────


def test_threshold_weighted_bce_matches_unweighted_when_weights_are_ones():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = (torch.rand(8, 5) > 0.5).float()
    unweighted = threshold_weighted_bce(logits, targets, None)
    weighted = threshold_weighted_bce(logits, targets, torch.ones(5))
    assert torch.allclose(unweighted, weighted, atol=1e-6)


def test_threshold_weighted_bce_rejects_wrong_weight_shape():
    logits = torch.randn(4, 5)
    targets = torch.zeros(4, 5)
    with pytest.raises(ValueError):
        threshold_weighted_bce(logits, targets, torch.ones(3))


def test_threshold_weighted_bce_is_differentiable():
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.zeros(4, 5)
    loss = threshold_weighted_bce(logits, targets, torch.tensor([1.0, 1.0, 1.2, 2.0, 3.0]))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


# ─── OrdinalHead ──────────────────────────────────────────────────────────────


def test_ordinal_head_output_shape():
    head = OrdinalHead(in_features=16, num_thresholds=5)
    out = head(torch.randn(7, 16))
    assert out.shape == (7, 5)
