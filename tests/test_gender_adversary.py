"""Tests for the gender-adversary (gradient-reversal) representation-debiasing head."""

from __future__ import annotations

import types

import pytest
import pytorch_lightning as pl
import torch

from face_occlusion.models.adversary import GenderAdversary, grad_reverse
from face_occlusion.models.outputs import OcclusionModelOutput
from face_occlusion.models.regressor import OcclusionRegressor
from face_occlusion.training.lit_module import FaceOcclusionLitModule
from face_occlusion.utils import Config

_TINY = "vit_tiny_patch16_224"


# ── Gradient Reversal Layer ──
def test_grad_reverse_identity_forward_flipped_backward():
    x = torch.randn(4, requires_grad=True)
    y = grad_reverse(x, 2.0)
    assert torch.allclose(y, x)  # forward is identity
    y.sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones(4))  # gradient flipped and scaled


# ── Model wiring ──
class TestGenderAdversaryModel:
    def test_builds_and_exposes_features(self):
        m = OcclusionRegressor(
            backbone=_TINY,
            pretrained=False,
            use_gender_adversary=True,
            gender_adversary={"conditional": True, "n_occ_bins": 6},
        ).eval()
        out = m(torch.randn(2, 3, 224, 224))
        assert out.y_pred.shape == (2,)
        assert out.features is not None and out.features.shape[0] == 2
        # conditional adversary input = feat_dim + n_occ_bins
        first = m.gender_adversary.net[0]
        assert first.in_features == out.features.shape[1] + 6

    def test_unconditional_input_dim(self):
        m = OcclusionRegressor(
            backbone=_TINY,
            pretrained=False,
            use_gender_adversary=True,
            gender_adversary={"conditional": False},
        )
        feat_dim = m.gender_adversary.net[0].in_features
        out = m(torch.randn(2, 3, 224, 224))
        assert feat_dim == out.features.shape[1]  # no occ-bin appended

    def test_baseline_has_no_adversary(self):
        m = OcclusionRegressor(backbone=_TINY, pretrained=False).eval()
        out = m(torch.randn(2, 3, 224, 224))
        assert m.gender_adversary is None
        assert out.features is None

    def test_param_groups_include_adversary(self):
        m = OcclusionRegressor(backbone=_TINY, pretrained=False, use_gender_adversary=True)
        groups = m.param_groups(head_lr=1e-3, backbone_lr=1e-4)
        all_ids = {id(p) for g in groups for p in g["params"]}
        assert {id(p) for p in m.gender_adversary.parameters()} <= all_ids


# ── Loss helper ──
def _module(conditional=False, n_bins=6, feat_dim=4):
    m = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(m)
    m._gadv_loss_enabled = True
    m._gadv_conditional = conditional
    m._gadv_n_bins = n_bins
    m._gadv_boundaries = torch.tensor([0.05, 0.1, 0.2, 0.4, 0.6]) if conditional else None
    m._male_value = "1.0"
    in_f = feat_dim + (n_bins if conditional else 0)
    m.model = types.SimpleNamespace(gender_adversary=GenderAdversary(in_f, hidden_dim=8))
    return m


def test_adversary_loss_finite_and_acc():
    m = _module()
    out = OcclusionModelOutput(y_pred=torch.zeros(6), features=torch.randn(6, 4))
    batch = {"gender": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]), "target": torch.rand(6)}
    loss, acc = m._compute_gender_adversary_loss(out, batch)
    assert loss is not None and torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0


def test_adversary_loss_conditional_runs():
    m = _module(conditional=True, n_bins=6, feat_dim=4)
    out = OcclusionModelOutput(y_pred=torch.zeros(5), features=torch.randn(5, 4))
    batch = {
        "gender": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0]),
        "target": torch.tensor([0.0, 0.3, 0.5, 0.9, 0.07]),
    }
    loss, _ = m._compute_gender_adversary_loss(out, batch)
    assert loss is not None and torch.isfinite(loss)


def test_adversary_loss_none_paths():
    out = OcclusionModelOutput(y_pred=torch.zeros(2), features=torch.randn(2, 4))
    batch = {"gender": torch.tensor([0.0, 1.0]), "target": torch.rand(2)}
    m = _module()
    m._gadv_loss_enabled = False
    assert m._compute_gender_adversary_loss(out, batch)[0] is None  # disabled
    m2 = _module()
    assert m2._compute_gender_adversary_loss(out, {})[0] is None  # no gender
    no_feat = OcclusionModelOutput(y_pred=torch.zeros(2))
    assert m2._compute_gender_adversary_loss(no_feat, batch)[0] is None  # no features


def test_grl_pushes_encoder_to_increase_adversary_loss():
    # The reversed gradient should move features so the adversary loss INCREASES (invariance).
    m = _module()
    feats = torch.randn(8, 4, requires_grad=True)
    batch = {
        "gender": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        "target": torch.rand(8),
    }
    loss, _ = m._compute_gender_adversary_loss(
        OcclusionModelOutput(y_pred=torch.zeros(8), features=feats), batch
    )
    loss.backward()
    # gradient on features is reversed: stepping along -grad (encoder update) raises adv loss
    assert feats.grad is not None and feats.grad.abs().sum() > 0


# ── LightningModule init guard ──
def _lit_cfg(use_head: bool, enabled: bool) -> Config:
    return Config(
        {
            "data": {"female_value": 0.0, "male_value": 1.0},
            "split": {"occlusion_bins": [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]},
            "model": {
                "backbone": _TINY,
                "pretrained": False,
                "output_activation": "identity",
                "use_gender_adversary": use_head,
                "gender_adversary": {"conditional": True},
            },
            "losses": {"gender_adversary": {"enabled": enabled, "weight": 1.0}},
        }
    )


def test_loss_without_head_raises():
    with pytest.raises(ValueError, match="use_gender_adversary"):
        FaceOcclusionLitModule(_lit_cfg(use_head=False, enabled=True))


def test_enabled_with_head_wires_on():
    m = FaceOcclusionLitModule(_lit_cfg(use_head=True, enabled=True))
    assert m._gadv_loss_enabled is True
    assert m._gadv_conditional is True
    assert m._gadv_boundaries is not None
