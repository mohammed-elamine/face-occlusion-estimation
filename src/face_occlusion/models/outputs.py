"""Structured model output contract for occlusion models.

Stage 0 of the occlusion-aware contrastive learning roadmap (see
``docs/occlusion_aware_contrastive_learning_approach.md``).

The current baseline is a pure regression model and only populates ``y_pred``.
The remaining fields are kept as ``None`` placeholders so that future stages
(ordinal head, projection head, contrastive learning) can extend the contract
without forcing further changes to the training loop or downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OcclusionModelOutput:
    """Structured forward output of an occlusion model.

    Attributes
    ----------
    y_pred:
        Continuous occlusion score prediction with shape ``(B,)``.
        This is the only field populated by the Stage 0 baseline.
    ordinal_logits:
        Future ordinal-bin head logits with shape ``(B, K-1)`` (cumulative
        link / corn-style). ``None`` while the ordinal head is disabled.
    projection:
        Future L2-normalised embedding from the projection head used by the
        triplet / contrastive loss, shape ``(B, p)``. ``None`` while the
        projection head is disabled.
    features:
        Optional pooled encoder features ``(B, d)`` exposed for reuse by
        future heads or analyses. ``None`` unless explicitly computed.
    """

    y_pred: torch.Tensor
    ordinal_logits: torch.Tensor | None = None
    projection: torch.Tensor | None = None
    features: torch.Tensor | None = None
    # Per-bin logits from the distribution (DEX/DLDL) head, shape ``(B, K)``. ``None`` unless
    # ``model.head.type == "distribution"``. ``y_pred`` is then the bin expectation.
    bin_logits: torch.Tensor | None = None
    # Auxiliary face-shadow prediction in ``[0, 1]``, shape ``(B,)``. ``None`` unless
    # ``model.use_shadow_head``. A training-only multi-task signal (predict the within-face
    # deep-shadow fraction) that pushes the encoder to represent illumination — shadow is the
    # one image property that correlates with the occlusion label (see tmp/model_study). Dropped
    # at inference: downstream code never reads it for the occlusion prediction.
    shadow_pred: torch.Tensor | None = None
