"""Synthetic monotonic ranking loss (Stage 4).

Synthetic occluded views carry no exact label, only a reliable *ordering*:

    s(clean) < s(mild) < s(strong)

where ``s`` is the regression scalar. We enforce this with a RankNet-style
logistic loss on the regression score, applied only to MediaPipe-valid pairs.
This injects new high-occlusion signal instead of re-sampling the same rare real
rows. Because it lands on the calibrated head, it is kept at a small, warmed-up
weight and watched for calibration regressions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ranknet_loss(higher: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
    """Logistic ranking loss encouraging ``higher > lower``.

    ``-log sigmoid(higher - lower)`` averaged over the batch, computed via
    ``softplus`` for numerical stability. Zero when ``higher`` exceeds ``lower``
    by a wide margin, large when the order is violated.
    """
    if higher.shape != lower.shape:
        raise ValueError(f"shape mismatch: {tuple(higher.shape)} vs {tuple(lower.shape)}")
    if higher.numel() == 0:
        return higher.new_zeros(())
    return F.softplus(-(higher - lower)).mean()


def monotonic_ranking_loss(
    s_clean: torch.Tensor, s_mild: torch.Tensor, s_strong: torch.Tensor
) -> torch.Tensor:
    """RankNet loss for the chain ``clean < mild < strong``."""
    return ranknet_loss(s_mild, s_clean) + ranknet_loss(s_strong, s_mild)


def ordering_accuracy(
    s_clean: torch.Tensor, s_mild: torch.Tensor, s_strong: torch.Tensor
) -> torch.Tensor:
    """Fraction of triples with strictly increasing scores (diagnostic).

    Returns a zero scalar for an empty input.
    """
    if s_clean.numel() == 0:
        return s_clean.new_zeros(())
    ok = (s_clean < s_mild) & (s_mild < s_strong)
    return ok.to(torch.float32).mean()
