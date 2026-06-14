"""timm-backed regressor used by Face Occlusion model configs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import timm
import torch
import torch.nn as nn

from .ordinal import DEFAULT_ORDINAL_THRESHOLDS, OrdinalHead
from .outputs import OcclusionModelOutput


class OcclusionRegressor(nn.Module):
    """Shared-encoder model with a regression head and an optional ordinal head.

    Stage 1 wiring:
      * regression head: the timm classifier with ``num_classes=1`` (unchanged).
      * ordinal head: a single linear layer on top of the pooled encoder
        features, producing one logit per threshold that predicts
        ``P(y > t_k)``. Enabled only when ``use_ordinal_head=True``; otherwise
        the forward pass is bit-identical to the Stage 0 baseline.
    """

    def __init__(
        self,
        backbone: str = "convnext_small.fb_in22k_ft_in1k",
        pretrained: bool = True,
        output_activation: str = "identity",
        dropout: float = 0.0,
        mean_target: float | None = None,
        use_ordinal_head: bool = False,
        ordinal_thresholds: Sequence[float] = DEFAULT_ORDINAL_THRESHOLDS,
        img_size: int | None = None,
        lora: dict | None = None,
    ) -> None:
        super().__init__()
        if output_activation not in {"identity", "sigmoid"}:
            raise ValueError(f"output_activation must be identity|sigmoid, got {output_activation}")
        self.output_activation = output_activation
        create_kwargs: dict = dict(
            pretrained=pretrained,
            # num_classes=1 replaces the classifier with a scalar regression head.
            num_classes=1,
            drop_rate=float(dropout),
        )
        if img_size is not None:
            # ViT backbones (e.g. DINOv2 patch-14) take an explicit input size and
            # interpolate position embeddings; CNN configs leave this unset (unchanged).
            create_kwargs["img_size"] = int(img_size)
            create_kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(backbone, **create_kwargs)
        self._init_head_bias(mean_target)

        self.use_ordinal_head = bool(use_ordinal_head)
        lora_enabled = bool(lora.get("enabled", False)) if lora else False
        if lora_enabled and self.use_ordinal_head:
            raise ValueError(
                "LoRA fine-tuning does not support the ordinal head yet "
                "(set model.use_ordinal_head=false or model.lora.enabled=false)."
            )
        if self.use_ordinal_head:
            thresholds = torch.tensor(list(ordinal_thresholds), dtype=torch.float32)
            if thresholds.numel() == 0:
                raise ValueError("`ordinal_thresholds` must contain at least one value")
            # Persist thresholds with the module so checkpoints stay self-contained.
            self.register_buffer("ordinal_thresholds", thresholds, persistent=True)
            self.ordinal_head = OrdinalHead(
                in_features=int(self.backbone.num_features),
                num_thresholds=int(thresholds.numel()),
            )
        else:
            self.ordinal_head = None

        if lora_enabled:
            self._wrap_lora(lora)

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

    def _wrap_lora(self, lora: dict) -> None:
        """Wrap the backbone with LoRA adapters; freeze the rest, keep the head trainable."""
        from peft import LoraConfig, get_peft_model

        target_modules = list(
            lora.get("target_modules", ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
        )
        modules_to_save = list(lora.get("modules_to_save", ["head"]))
        config = LoraConfig(
            r=int(lora.get("rank", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias="none",
            target_modules=target_modules,
            # Without modules_to_save peft freezes the regression head too.
            modules_to_save=modules_to_save,
        )
        self.backbone = get_peft_model(self.backbone, config)

    def _apply_activation(self, raw: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(raw)
        return raw

    def forward(self, x: torch.Tensor) -> OcclusionModelOutput:
        # Fast path: with no auxiliary head we keep the exact Stage 0 call
        # (``self.backbone(x)``) so baseline runs stay bit-identical.
        if self.ordinal_head is None:
            logits = self.backbone(x).squeeze(-1)
            return OcclusionModelOutput(y_pred=self._apply_activation(logits))

        # Multi-head path: share pooled encoder features between heads.
        feats = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        reg_head = self.backbone.get_classifier()
        raw = reg_head(pooled).squeeze(-1)
        ordinal_logits = self.ordinal_head(pooled)
        return OcclusionModelOutput(
            y_pred=self._apply_activation(raw),
            ordinal_logits=ordinal_logits,
            features=pooled,
        )


def build_model(cfg, mean_target: float | None = None) -> OcclusionRegressor:
    m = cfg.model
    use_ordinal_head = bool(m.get("use_ordinal_head", False))
    ordinal_thresholds = m.get("ordinal_thresholds", list(DEFAULT_ORDINAL_THRESHOLDS))
    img_size = m.get("img_size", None)
    lora = m.get("lora", None)
    # Config objects are dict-like; pass a plain dict to OcclusionRegressor.
    lora = dict(lora) if lora else None
    return OcclusionRegressor(
        backbone=m.backbone,
        pretrained=bool(m.pretrained),
        output_activation=m.output_activation,
        dropout=float(m.get("dropout", 0.0) or 0.0),
        mean_target=mean_target,
        use_ordinal_head=use_ordinal_head,
        ordinal_thresholds=list(ordinal_thresholds),
        img_size=int(img_size) if img_size is not None else None,
        lora=lora,
    )
