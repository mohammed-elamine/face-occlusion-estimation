"""timm-backed regressor used by Face Occlusion model configs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import timm
import torch
import torch.nn as nn

from .distribution import expectation, make_bin_centers
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
        head: dict | None = None,
        backbone_source: str = "timm",
        use_shadow_head: bool = False,
        shadow_head: dict | None = None,
    ) -> None:
        super().__init__()
        if output_activation not in {"identity", "sigmoid"}:
            raise ValueError(f"output_activation must be identity|sigmoid, got {output_activation}")
        self.output_activation = output_activation

        head_cfg = dict(head) if head else {}
        self.head_type = str(head_cfg.get("type", "linear"))
        if self.head_type not in {"linear", "mlp", "distribution"}:
            raise ValueError(
                f"model.head.type must be linear|mlp|distribution, got {self.head_type!r}"
            )
        mlp_head = self.head_type == "mlp"
        dist_head = self.head_type == "distribution"
        separate_head = mlp_head or dist_head  # a head module outside the backbone

        self.use_ordinal_head = bool(use_ordinal_head)
        lora_enabled = bool(lora.get("enabled", False)) if lora else False
        if self.use_ordinal_head and (separate_head or lora_enabled):
            raise ValueError(
                "The ordinal head is only supported with model.head.type=linear and LoRA off."
            )

        # Build the backbone via one of two loaders, and resolve the pooled-feature dim.
        #  - "timm" (default): timm.create_model. The linear-head path uses timm's own
        #    classifier (num_classes=1); the mlp path makes it a pure feature extractor.
        #  - "torchhub": the official DINOv2 (e.g. dinov2_vitb14_reg) via torch.hub. It has
        #    no built-in classifier and returns the CLS token, so it requires the separate
        #    MLP head and interpolates its position embeddings to the input size itself.
        self.backbone_source = str(backbone_source)
        if self.backbone_source == "torchhub":
            if not separate_head:
                raise ValueError(
                    "backbone_source='torchhub' requires a separate head "
                    "(model.head.type=mlp|distribution)"
                )
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", backbone, pretrained=pretrained
            )
            feat_dim = int(self.backbone.embed_dim)
        elif self.backbone_source == "timm":
            create_kwargs: dict = dict(
                pretrained=pretrained,
                # mlp head: backbone is a pure feature extractor (num_classes=0).
                # linear head: timm's own classifier with num_classes=1 (Stage 0 behaviour).
                num_classes=0 if separate_head else 1,
                drop_rate=float(dropout),
            )
            if img_size is not None:
                # ViT backbones (e.g. DINOv2 patch-14) take an explicit input size and
                # interpolate position embeddings; CNN configs leave this unset (unchanged).
                create_kwargs["img_size"] = int(img_size)
                create_kwargs["dynamic_img_size"] = True
            self.backbone = timm.create_model(backbone, **create_kwargs)
            feat_dim = int(self.backbone.num_features)
        else:
            raise ValueError(
                f"model.backbone_source must be 'timm' or 'torchhub', got {self.backbone_source!r}"
            )

        # Separate MLP regression head on the pooled features (mlp path). For the linear
        # path the head lives inside the timm backbone (num_classes=1) and self.head is None.
        if mlp_head:
            hidden = int(head_cfg.get("hidden_dim", 256))
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(head_cfg.get("dropout", 0.0))),
                nn.Linear(hidden, 1),
            )
        elif dist_head:
            # Ordered-bin distribution head (DEX/DLDL): K logits over occlusion bins; the
            # prediction is the expectation over the bin centers (see models/distribution.py).
            n_bins = int(head_cfg.get("n_bins", 21))
            rng = head_cfg.get("range", [0.0, 1.0])
            self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, n_bins))
            self.register_buffer(
                "bin_centers",
                make_bin_centers(n_bins, float(rng[0]), float(rng[1])),
                persistent=True,
            )
        else:
            self.head = None

        self._init_head_bias(mean_target)

        if self.use_ordinal_head:
            thresholds = torch.tensor(list(ordinal_thresholds), dtype=torch.float32)
            if thresholds.numel() == 0:
                raise ValueError("`ordinal_thresholds` must contain at least one value")
            # Persist thresholds with the module so checkpoints stay self-contained.
            self.register_buffer("ordinal_thresholds", thresholds, persistent=True)
            self.ordinal_head = OrdinalHead(
                in_features=feat_dim,
                num_thresholds=int(thresholds.numel()),
            )
        else:
            self.ordinal_head = None

        # Auxiliary shadow head (training-only multi-task): predicts the within-face deep-shadow
        # fraction from the pooled features, pushing the encoder to represent illumination. Built
        # before LoRA wrapping so it stays trainable and outside the peft model. Dropped at
        # inference (the occlusion prediction never reads it).
        self.use_shadow_head = bool(use_shadow_head)
        if self.use_shadow_head:
            sh_cfg = dict(shadow_head) if shadow_head else {}
            hidden = int(sh_cfg.get("hidden_dim", 64))
            self.shadow_head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(sh_cfg.get("dropout", 0.0))),
                nn.Linear(hidden, 1),
            )
        else:
            self.shadow_head = None

        if lora_enabled:
            self._wrap_lora(lora, has_separate_head=mlp_head)

    def _init_head_bias(self, mean_target: float | None) -> None:
        # Warm-start the regression bias near the training mean so optimisation
        # does not waste epochs learning the global offset.
        # The distribution head is a K-way classifier; a scalar mean-target bias does not apply.
        if mean_target is None or self.head_type == "distribution":
            return
        m = float(mean_target)
        if self.output_activation == "sigmoid":
            m = min(max(m, 1e-4), 1 - 1e-4)
            bias_value = math.log(m / (1 - m))  # logit
        else:
            bias_value = m
        # The final Linear is the MLP head's last layer, or timm's classifier (linear path).
        final = self.head[-1] if self.head is not None else self.backbone.get_classifier()
        if isinstance(final, nn.Linear) and final.bias is not None:
            with torch.no_grad():
                final.bias.fill_(bias_value)

    def _wrap_lora(self, lora: dict, *, has_separate_head: bool) -> None:
        """Wrap the backbone with LoRA adapters; freeze the rest.

        With a separate MLP head (``self.head``) the head is already trainable and outside the
        peft model, so no ``modules_to_save`` is needed. With the linear (in-backbone) head we
        keep the classifier trainable via ``modules_to_save``.
        """
        from peft import LoraConfig, get_peft_model

        target_modules = list(lora.get("target_modules", ["attn.qkv", "attn.proj"]))
        kwargs: dict = dict(
            r=int(lora.get("rank", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias="none",
            target_modules=target_modules,
        )
        if not has_separate_head:
            kwargs["modules_to_save"] = list(lora.get("modules_to_save", ["head"]))
        self.backbone = get_peft_model(self.backbone, LoraConfig(**kwargs))

    def param_groups(self, head_lr: float, backbone_lr: float, weight_decay: float = 0.0) -> list:
        """Discriminative AdamW param groups: head at ``head_lr``, backbone/LoRA at
        ``backbone_lr``; weight decay only on 2-D weights (none on LayerNorm/bias)."""

        def split(params):
            decay, no_decay = [], []
            for p in params:
                if p.requires_grad:
                    (no_decay if p.ndim <= 1 else decay).append(p)
            return decay, no_decay

        if self.head is not None:  # mlp path: head is a separate module
            head_params = list(self.head.parameters())
            backbone_params = list(self.backbone.parameters())
        else:  # linear path: the head is the timm classifier inside the backbone
            clf_ids = {id(p) for p in self.backbone.get_classifier().parameters()}
            head_params = [p for p in self.backbone.parameters() if id(p) in clf_ids]
            backbone_params = [p for p in self.backbone.parameters() if id(p) not in clf_ids]
        if self.ordinal_head is not None:
            head_params += list(self.ordinal_head.parameters())
        if self.shadow_head is not None:
            head_params += list(self.shadow_head.parameters())

        groups = []
        for params, lr in ((head_params, head_lr), (backbone_params, backbone_lr)):
            decay, no_decay = split(params)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    def _apply_activation(self, raw: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(raw)
        return raw

    @torch.no_grad()
    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-logit pooled encoder features ``(B, d)`` — the input the final head sees.

        Used by DFR (last-layer retraining): for the mlp/distribution paths the backbone is a
        pure feature extractor (``backbone(x)``); for the linear path we take
        ``forward_head(forward_features(x), pre_logits=True)`` so ``get_classifier()(features)``
        reproduces the model's output.
        """
        if self.head is not None:
            return self.backbone(x)
        return self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)

    def _shadow_from(self, feat: torch.Tensor) -> torch.Tensor | None:
        """Auxiliary shadow prediction in [0, 1] from pooled features (None if head disabled)."""
        if self.shadow_head is None:
            return None
        return torch.sigmoid(self.shadow_head(feat).squeeze(-1))

    def forward(self, x: torch.Tensor) -> OcclusionModelOutput:
        # Distribution (DEX/DLDL) head: pooled features -> K bin logits -> softmax -> the
        # prediction is the bin expectation (bounded to [c_1, c_K], so no activation needed).
        if self.head_type == "distribution":
            feat = self.backbone(x)
            logits = self.head(feat)
            probs = torch.softmax(logits, dim=-1)
            y_pred = expectation(probs, self.bin_centers)
            return OcclusionModelOutput(
                y_pred=y_pred, bin_logits=logits, features=feat, shadow_pred=self._shadow_from(feat)
            )

        # MLP-head path: backbone (num_classes=0) -> pooled features -> separate MLP head.
        if self.head is not None:
            feat = self.backbone(x)
            raw = self.head(feat).squeeze(-1)
            return OcclusionModelOutput(
                y_pred=self._apply_activation(raw),
                features=feat if self.shadow_head is not None else None,
                shadow_pred=self._shadow_from(feat),
            )

        # Fast path: with no auxiliary head we keep the exact Stage 0 call
        # (``self.backbone(x)``) so baseline runs stay bit-identical.
        if self.ordinal_head is None and self.shadow_head is None:
            logits = self.backbone(x).squeeze(-1)
            return OcclusionModelOutput(y_pred=self._apply_activation(logits))

        # Multi-head path: share pooled encoder features between heads (ordinal and/or shadow).
        feats = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        reg_head = self.backbone.get_classifier()
        raw = reg_head(pooled).squeeze(-1)
        ordinal_logits = self.ordinal_head(pooled) if self.ordinal_head is not None else None
        return OcclusionModelOutput(
            y_pred=self._apply_activation(raw),
            ordinal_logits=ordinal_logits,
            features=pooled,
            shadow_pred=self._shadow_from(pooled),
        )


def build_model(cfg, mean_target: float | None = None) -> OcclusionRegressor:
    m = cfg.model
    use_ordinal_head = bool(m.get("use_ordinal_head", False))
    ordinal_thresholds = m.get("ordinal_thresholds", list(DEFAULT_ORDINAL_THRESHOLDS))
    img_size = m.get("img_size", None)
    lora = m.get("lora", None)
    head = m.get("head", None)
    # Config objects are dict-like; pass plain dicts to OcclusionRegressor.
    lora = dict(lora) if lora else None
    head = dict(head) if head else None
    use_shadow_head = bool(m.get("use_shadow_head", False))
    shadow_head = m.get("shadow_head", None)
    shadow_head = dict(shadow_head) if shadow_head else None
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
        head=head,
        backbone_source=str(m.get("backbone_source", "timm")),
        use_shadow_head=use_shadow_head,
        shadow_head=shadow_head,
    )
