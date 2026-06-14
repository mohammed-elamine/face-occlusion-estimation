"""timm-backed regressor used by Face Occlusion model configs."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn


class OcclusionRegressor(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_small.fb_in22k_ft_in1k",
        pretrained: bool = True,
        output_activation: str = "identity",
        dropout: float = 0.0,
        mean_target: float | None = None,
    ) -> None:
        super().__init__()
        if output_activation not in {"identity", "sigmoid"}:
            raise ValueError(f"output_activation must be identity|sigmoid, got {output_activation}")
        self.output_activation = output_activation
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            # num_classes=1 replaces the classifier with a scalar regression head.
            num_classes=1,
            drop_rate=float(dropout),
        )
        self._init_head_bias(mean_target)

    def _init_head_bias(self, mean_target: float | None) -> None:
        # Warm-start the regression bias near the training mean so optimisation
        # does not waste epochs learning the global offset.
        if mean_target is None:
            return
        m = float(mean_target)
        if self.output_activation == "sigmoid":
            m = min(max(m, 1e-4), 1 - 1e-4)
            bias_value = math.log(m / (1 - m))  # logit
        else:
            bias_value = m
        head = self.backbone.get_classifier()
        if isinstance(head, nn.Linear) and head.bias is not None:
            with torch.no_grad():
                head.bias.fill_(bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x).squeeze(-1)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits


def build_model(cfg, mean_target: float | None = None) -> OcclusionRegressor:
    m = cfg.model
    return OcclusionRegressor(
        backbone=m.backbone,
        pretrained=bool(m.pretrained),
        output_activation=m.output_activation,
        dropout=float(m.get("dropout", 0.0) or 0.0),
        mean_target=mean_target,
    )
