"""Smoke tests for the Stage 0/1 model output contract.

These tests verify that:
* ``OcclusionRegressor.forward`` returns an ``OcclusionModelOutput``;
* ``y_pred`` has the expected shape and dtype;
* when the ordinal head is disabled, ``ordinal_logits`` stays ``None``;
* when the ordinal head is enabled, ``ordinal_logits`` has shape ``(B, K)``;
* the baseline ``sigmoid`` activation still constrains predictions to [0, 1].
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from face_occlusion.models import OcclusionModelOutput, OcclusionRegressor
from face_occlusion.models.ordinal import OrdinalHead


class _TinyBackbone(nn.Module):
    """Tiny timm-like backbone sufficient for Stage 0/1 forward paths."""

    num_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(self.num_features, 1)

    # Stage 0 fast path uses ``self.backbone(x)`` directly.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    # Stage 1 multi-head path uses these split entry points.
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return x  # treat input as raw features (B, C, H, W)

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        pooled = features.mean(dim=(-1, -2))  # (B, num_features)
        if pre_logits:
            return pooled
        return self.classifier(pooled)

    def get_classifier(self) -> nn.Linear:
        return self.classifier


def _make_regressor(
    activation: str = "identity",
    *,
    use_ordinal_head: bool = False,
    ordinal_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60),
) -> OcclusionRegressor:
    """Build a regressor without downloading any pretrained timm weights."""
    model = OcclusionRegressor.__new__(OcclusionRegressor)
    nn.Module.__init__(model)
    model.output_activation = activation
    model.backbone = _TinyBackbone()
    model.use_ordinal_head = bool(use_ordinal_head)
    if use_ordinal_head:
        thresholds = torch.tensor(list(ordinal_thresholds), dtype=torch.float32)
        model.register_buffer("ordinal_thresholds", thresholds, persistent=True)
        model.ordinal_head = OrdinalHead(
            in_features=_TinyBackbone.num_features,
            num_thresholds=int(thresholds.numel()),
        )
    else:
        model.ordinal_head = None
    return model


def test_forward_returns_structured_output():
    model = _make_regressor()
    x = torch.randn(4, 8, 2, 2)
    out = model(x)
    assert isinstance(out, OcclusionModelOutput)


def test_y_pred_shape_and_dtype():
    model = _make_regressor()
    out = model(torch.randn(3, 8, 2, 2))
    assert out.y_pred.shape == (3,)
    assert out.y_pred.dtype == torch.float32


@pytest.mark.parametrize("field", ["ordinal_logits", "projection", "features"])
def test_future_stage_fields_are_none(field: str):
    model = _make_regressor()
    out = model(torch.randn(2, 8, 2, 2))
    assert getattr(out, field) is None


def test_sigmoid_activation_bounds_predictions():
    model = _make_regressor(activation="sigmoid")
    out = model(torch.randn(5, 8, 2, 2))
    assert torch.all((out.y_pred >= 0.0) & (out.y_pred <= 1.0))


# ─── Stage 1: ordinal head ────────────────────────────────────────────────────


def test_ordinal_head_disabled_keeps_logits_none():
    model = _make_regressor(use_ordinal_head=False)
    out = model(torch.randn(2, 8, 2, 2))
    assert out.ordinal_logits is None
    assert out.features is None


def test_ordinal_head_enabled_populates_logits_shape():
    thresholds = (0.05, 0.10, 0.20, 0.40, 0.60)
    model = _make_regressor(use_ordinal_head=True, ordinal_thresholds=thresholds)
    out = model(torch.randn(3, 8, 2, 2))
    assert out.ordinal_logits is not None
    assert out.ordinal_logits.shape == (3, len(thresholds))
    # y_pred should still be the regression scalar.
    assert out.y_pred.shape == (3,)
    # Features are exposed for future heads (Stage 2+ consistency loss).
    assert out.features is not None
    assert out.features.shape == (3, _TinyBackbone.num_features)
