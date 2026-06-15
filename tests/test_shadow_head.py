"""Tests for the auxiliary shadow head (multi-task dark_frac prediction)."""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch

from face_occlusion.models.outputs import OcclusionModelOutput
from face_occlusion.models.regressor import OcclusionRegressor
from face_occlusion.training.lit_module import FaceOcclusionLitModule
from face_occlusion.utils import Config

_TINY = "vit_tiny_patch16_224"


# ── Model: the head produces a bounded aux prediction; baseline stays clean ──
class TestShadowHeadModel:
    def test_linear_head_plus_shadow(self):
        m = OcclusionRegressor(backbone=_TINY, pretrained=False, use_shadow_head=True).eval()
        out = m(torch.randn(3, 3, 224, 224))
        assert out.y_pred.shape == (3,)
        assert out.shadow_pred is not None and out.shadow_pred.shape == (3,)
        assert bool((out.shadow_pred >= 0).all() and (out.shadow_pred <= 1).all())

    def test_mlp_head_plus_shadow(self):
        m = OcclusionRegressor(
            backbone=_TINY, pretrained=False, use_shadow_head=True, head={"type": "mlp"}
        ).eval()
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shadow_pred is not None and out.shadow_pred.shape == (2,)

    def test_baseline_has_no_shadow_pred(self):
        m = OcclusionRegressor(backbone=_TINY, pretrained=False, use_shadow_head=False).eval()
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shadow_pred is None

    def test_param_groups_include_shadow_head(self):
        m = OcclusionRegressor(backbone=_TINY, pretrained=False, use_shadow_head=True)
        groups = m.param_groups(head_lr=1e-3, backbone_lr=1e-4)
        all_ids = {id(p) for g in groups for p in g["params"]}
        shadow_ids = {id(p) for p in m.shadow_head.parameters()}
        assert shadow_ids <= all_ids


# ── Loss helper: masking, L1/L2, gating (lightweight, like test_bg_consistency) ──
def _module(enabled: bool = True, loss_type: str = "l2"):
    m = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(m)
    m._shadow_loss_enabled = enabled
    m._shadow_loss_type = loss_type
    return m


def _out(shadow_pred):
    return OcclusionModelOutput(y_pred=torch.zeros(shadow_pred.shape[0]), shadow_pred=shadow_pred)


def test_shadow_loss_l2():
    m = _module(loss_type="l2")
    out = _out(torch.tensor([0.1, 0.5]))
    batch = {"shadow_target": torch.tensor([0.2, 0.1])}
    loss = m._compute_shadow_loss(out, batch)
    # mean((0.1-0.2)^2, (0.5-0.1)^2) = mean(0.01, 0.16) = 0.085
    assert torch.isclose(loss, torch.tensor(0.085), atol=1e-6)


def test_shadow_loss_l1():
    m = _module(loss_type="l1")
    out = _out(torch.tensor([0.1, 0.5]))
    batch = {"shadow_target": torch.tensor([0.2, 0.1])}
    loss = m._compute_shadow_loss(out, batch)
    # mean(0.1, 0.4) = 0.25
    assert torch.isclose(loss, torch.tensor(0.25), atol=1e-6)


def test_shadow_loss_masks_nan_targets():
    m = _module()
    out = _out(torch.tensor([0.1, 0.9]))
    batch = {"shadow_target": torch.tensor([0.2, float("nan")])}
    loss = m._compute_shadow_loss(out, batch)
    # only the first (valid) row counts: (0.1-0.2)^2 = 0.01
    assert torch.isclose(loss, torch.tensor(0.01), atol=1e-6)


def test_shadow_loss_none_paths():
    out = _out(torch.tensor([0.1, 0.5]))
    batch = {"shadow_target": torch.tensor([0.2, 0.3])}
    assert _module(enabled=False)._compute_shadow_loss(out, batch) is None  # disabled
    assert _module()._compute_shadow_loss(out, {}) is None  # no target in batch
    assert (
        _module()._compute_shadow_loss(
            _out(torch.tensor([0.1])), {"shadow_target": torch.tensor([float("nan")])}
        )
        is None
    )  # all NaN
    # shadow_pred None (head off)
    assert (
        _module()._compute_shadow_loss(OcclusionModelOutput(y_pred=torch.zeros(2)), batch) is None
    )


# ── LightningModule init wiring + guard ──
def _lit_cfg(use_shadow_head: bool, shadow_enabled: bool) -> Config:
    return Config(
        {
            "data": {"female_value": 0.0, "male_value": 1.0},
            "model": {
                "backbone": _TINY,
                "pretrained": False,
                "output_activation": "identity",
                "use_shadow_head": use_shadow_head,
            },
            "losses": {"shadow": {"enabled": shadow_enabled, "weight": 0.2, "loss": "l2"}},
        }
    )


def test_loss_enabled_without_head_raises():
    with pytest.raises(ValueError, match="use_shadow_head"):
        FaceOcclusionLitModule(_lit_cfg(use_shadow_head=False, shadow_enabled=True))


def test_enabled_with_head_wires_on():
    m = FaceOcclusionLitModule(_lit_cfg(use_shadow_head=True, shadow_enabled=True))
    assert m._shadow_loss_enabled is True


def test_head_without_loss_is_silent():
    m = FaceOcclusionLitModule(_lit_cfg(use_shadow_head=True, shadow_enabled=False))
    assert m._shadow_loss_enabled is False
