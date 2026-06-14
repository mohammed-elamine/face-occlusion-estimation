"""Smoke test for one Stage 1 training step with the ordinal loss active."""

from __future__ import annotations

import torch
import torch.nn as nn

from face_occlusion.models import OcclusionModelOutput, OrdinalHead
from face_occlusion.training.lit_module import FaceOcclusionLitModule


class _StubModel(nn.Module):
    """Minimal stand-in for OcclusionRegressor with both heads.

    Exposes the same attributes the LitModule reads from the real model
    (``use_ordinal_head``, ``ordinal_thresholds``) and returns the same
    structured output.
    """

    def __init__(self, num_features: int = 8, num_thresholds: int = 5) -> None:
        super().__init__()
        self.use_ordinal_head = True
        self.register_buffer(
            "ordinal_thresholds",
            torch.tensor([0.05, 0.10, 0.20, 0.40, 0.60], dtype=torch.float32),
        )
        self.encoder = nn.Linear(3 * 4 * 4, num_features)
        self.reg_head = nn.Linear(num_features, 1)
        self.ordinal_head = OrdinalHead(num_features, num_thresholds)

    def forward(self, x: torch.Tensor) -> OcclusionModelOutput:
        flat = x.flatten(1)
        feats = torch.relu(self.encoder(flat))
        y_pred = self.reg_head(feats).squeeze(-1)
        ordinal_logits = self.ordinal_head(feats)
        return OcclusionModelOutput(y_pred=y_pred, ordinal_logits=ordinal_logits, features=feats)


def _make_cfg():
    # Lightweight cfg-like object: just needs the attributes the LitModule reads.
    # The bypassed __init__ never touches it in this test.
    return None


def _make_module() -> FaceOcclusionLitModule:
    """Build a LitModule but swap in a stub model to avoid timm downloads."""
    module = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    # Bypass parent inits that would trigger build_model; set fields manually.
    import pytorch_lightning as pl

    pl.LightningModule.__init__(module)
    module.cfg = _make_cfg()
    module._val_buffer = []
    module._female_value = "0.0"
    module._male_value = "1.0"
    module.model = _StubModel()
    module._ord_loss_enabled = True
    module._ord_weight = 0.2
    module.register_buffer("_ord_thresholds", module.model.ordinal_thresholds.detach().clone())
    module.register_buffer(
        "_ord_threshold_weights",
        torch.tensor([1.0, 1.0, 1.2, 2.0, 3.0], dtype=torch.float32),
    )
    # Stage 2 consistency defaults: disabled.
    module._cons_loss_enabled = False
    module._cons_weight = 0.05
    module._cons_temperature = 0.05
    module._cons_mode = "symmetric"
    return module


def test_training_step_with_ordinal_runs_and_backpropagates():
    module = _make_module()
    images = torch.randn(4, 3, 4, 4)
    targets = torch.tensor([0.03, 0.12, 0.35, 0.70])
    from face_occlusion.training.lit_module import weighted_mse_loss

    outputs = module(images)
    loss_reg = weighted_mse_loss(outputs.y_pred, targets)
    loss_ord = module._compute_ordinal_loss(outputs, targets)
    assert loss_ord is not None
    total = loss_reg + module._ord_weight * loss_ord
    assert total.ndim == 0
    assert torch.isfinite(total).item()
    total.backward()
    # Gradients reach both heads (regression + ordinal).
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.reg_head.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in module.model.ordinal_head.parameters()
    )


def test_training_step_with_ordinal_disabled_matches_regression_only():
    """When the ordinal flag is off, ``_compute_ordinal_loss`` returns None."""
    module = _make_module()
    module._ord_loss_enabled = False
    images = torch.randn(4, 3, 4, 4)
    targets = torch.tensor([0.03, 0.12, 0.35, 0.70])
    outputs = module(images)
    assert module._compute_ordinal_loss(outputs, targets) is None


def test_training_step_with_ordinal_and_consistency_runs():
    """Stage 2 smoke: reg + ordinal + consistency all backpropagate together."""
    module = _make_module()
    module._cons_loss_enabled = True
    from face_occlusion.training.lit_module import weighted_mse_loss

    images = torch.randn(4, 3, 4, 4)
    targets = torch.tensor([0.03, 0.12, 0.35, 0.70])
    outputs = module(images)
    loss_reg = weighted_mse_loss(outputs.y_pred, targets)
    loss_ord = module._compute_ordinal_loss(outputs, targets)
    loss_cons = module._compute_consistency_loss(outputs)

    assert loss_cons is not None
    total = loss_reg + module._ord_weight * loss_ord + module._cons_weight * loss_cons
    assert torch.isfinite(total).item()
    total.backward()
    # Both heads should receive gradients from the consistency term too.
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.reg_head.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in module.model.ordinal_head.parameters()
    )


def test_consistency_disabled_returns_none():
    module = _make_module()
    module._cons_loss_enabled = False
    outputs = module(torch.randn(2, 3, 4, 4))
    assert module._compute_consistency_loss(outputs) is None
