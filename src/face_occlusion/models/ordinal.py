"""Ordinal occlusion-bin head, target builder, and weighted BCE loss.

Stage 1 of the occlusion-aware contrastive learning roadmap
(see ``docs/occlusion_aware_auxiliary_learning.md``).

The ordinal head predicts, for each threshold ``t_k``, the logit of the event
``y > t_k``. It is **not** a softmax over bins: every threshold is independent
and trained with binary cross-entropy. This keeps gradients informative for
rare high-occlusion regimes via per-threshold loss weights.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_ORDINAL_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60)
DEFAULT_ORDINAL_THRESHOLD_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 1.2, 2.0, 3.0)

CONSISTENCY_MODES: tuple[str, ...] = (
    "symmetric",
    "ordinal_as_teacher",
    "regression_as_teacher",
)


class OrdinalHead(nn.Module):
    """One logit per threshold; logit_k predicts ``P(y > t_k)``."""

    def __init__(self, in_features: int, num_thresholds: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_thresholds)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def make_ordinal_targets(y: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Build cumulative-style ordinal targets ``c_ik = 1[y_i > t_k]``.

    Parameters
    ----------
    y:
        Continuous occlusion scores, shape ``(B,)`` or ``(B, 1)``.
    thresholds:
        1-D tensor of thresholds, shape ``(K,)``.

    Returns
    -------
    Tensor of shape ``(B, K)`` with values in ``{0., 1.}``, on the same device
    and dtype as ``y``.
    """
    if y.dim() == 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if y.dim() != 1:
        raise ValueError(f"`y` must be 1-D or (B, 1); got shape {tuple(y.shape)}")
    if thresholds.dim() != 1:
        raise ValueError(f"`thresholds` must be 1-D; got shape {tuple(thresholds.shape)}")
    thresholds = thresholds.to(device=y.device, dtype=y.dtype)
    return (y.unsqueeze(-1) > thresholds.unsqueeze(0)).to(y.dtype)


def threshold_weighted_bce(
    ordinal_logits: torch.Tensor,
    ordinal_targets: torch.Tensor,
    threshold_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Threshold-weighted BCE-with-logits, averaged over batch and thresholds.

    With per-threshold weights ``beta_k``:

        L = mean_{i, k} beta_k * BCEWithLogits(o_ik, c_ik)

    Weights are broadcast over the batch dimension only; they upweight rare
    high-occlusion thresholds without rescaling the total loss magnitude.
    """
    per_elem = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets, reduction="none")
    if threshold_weights is not None:
        w = threshold_weights.to(device=per_elem.device, dtype=per_elem.dtype)
        if w.shape != (per_elem.shape[-1],):
            raise ValueError(
                f"threshold_weights shape {tuple(w.shape)} does not match "
                f"num_thresholds={per_elem.shape[-1]}"
            )
        per_elem = per_elem * w.unsqueeze(0)
    return per_elem.mean()


def ordinal_monotonicity_loss(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """Hinge penalty for non-monotone ordinal probabilities (doc §7).

    Threshold probabilities must be non-increasing, because ``P(y > t_k)``
    can only drop as ``t_k`` grows::

        q_{i,1} >= q_{i,2} >= ... >= q_{i,K}

    where ``q_ik = sigmoid(ordinal_logits_ik)``. We penalize any increase::

        L = mean_{i,k} relu(q_{i,k+1} - q_{i,k})

    This is a soft regulariser kept at a small weight; it nudges the ordinal
    head toward valid (monotone) outputs without hard-constraining it. Returns a
    zero scalar when there are fewer than two thresholds.
    """
    if ordinal_logits.shape[-1] < 2:
        return ordinal_logits.new_zeros(())
    q = torch.sigmoid(ordinal_logits)
    # diffs <= 0 everywhere when perfectly monotone.
    diffs = q[..., 1:] - q[..., :-1]
    return torch.relu(diffs).mean()


def ordinal_monotonicity_violation_rate(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """Fraction of adjacent threshold pairs that are non-monotone (``q_{k+1} > q_k``).

    A pure diagnostic (no gradient role): 0.0 means every sample's thresholds
    are correctly ordered. Returns a zero scalar for fewer than two thresholds.
    """
    if ordinal_logits.shape[-1] < 2:
        return ordinal_logits.new_zeros(())
    q = torch.sigmoid(ordinal_logits)
    diffs = q[..., 1:] - q[..., :-1]
    return (diffs > 0).to(torch.float32).mean()


def as_tensor_1d(values: Iterable[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Small helper to normalise a list/tuple of floats into a 1-D tensor."""
    return torch.tensor(list(values), dtype=dtype)


def regression_ordinal_consistency_loss(
    y_pred: torch.Tensor,
    ordinal_logits: torch.Tensor,
    thresholds: torch.Tensor,
    temperature: float = 0.05,
    mode: str = "symmetric",
) -> torch.Tensor:
    """Soft consistency between the regression and ordinal heads.

    The regression prediction ``y_pred`` implies a soft threshold probability::

        r_ik = sigmoid((y_pred_i - t_k) / temperature)

    which we match against the ordinal head's threshold probability::

        q_ik = sigmoid(ordinal_logits_ik)

    via a per-element MSE averaged over batch and thresholds. This is a
    *soft* regulariser (not a hard bin constraint): it nudges the two heads
    to agree on the occlusion regime without forcing the regression head to
    snap to threshold boundaries.

    Parameters
    ----------
    y_pred:
        Regression scores, shape ``(B,)`` or ``(B, 1)``.
    ordinal_logits:
        Threshold logits from the ordinal head, shape ``(B, K)``.
    thresholds:
        1-D tensor of thresholds ``t_k``, shape ``(K,)``. Cast to ``y_pred``
        device/dtype internally.
    temperature:
        Sharpness of the regression-implied probability. Smaller values
        produce sharper transitions; defaults to ``0.05``.
    mode:
        One of ``"symmetric"`` (gradients to both branches),
        ``"ordinal_as_teacher"`` (regression follows detached ordinal), or
        ``"regression_as_teacher"`` (ordinal follows detached regression).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if mode not in CONSISTENCY_MODES:
        raise ValueError(f"consistency mode must be one of {CONSISTENCY_MODES}, got {mode!r}")

    y = y_pred.view(-1, 1)
    thr = thresholds.to(device=y.device, dtype=y.dtype).view(1, -1)

    q = torch.sigmoid(ordinal_logits)
    r = torch.sigmoid((y - thr) / temperature)

    if mode == "ordinal_as_teacher":
        q = q.detach()
    elif mode == "regression_as_teacher":
        r = r.detach()
    # symmetric → no detach

    return torch.mean((q - r) ** 2)
