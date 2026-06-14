"""Tests for the background-invariance consistency loss term."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from face_occlusion.models.outputs import OcclusionModelOutput
from face_occlusion.training.lit_module import FaceOcclusionLitModule


class _MeanModel(torch.nn.Module):
    """Stub model: prediction = mean pixel value (so different views -> different preds)."""

    def forward(self, x):
        return OcclusionModelOutput(y_pred=x.mean(dim=[1, 2, 3]))


def _module(enabled: bool = True, loss_type: str = "l1"):
    m = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(m)
    m.model = _MeanModel()
    m._bgc_enabled = enabled
    m._bgc_weight = 0.1
    m._bgc_warmup_epochs = 0
    m._bgc_warmup_start_weight = 0.0
    m._bgc_loss_type = loss_type
    return m


def _batch():
    # bg_view image 0 has mean 0.4, image 1 has mean 0.1
    bg = torch.stack([torch.full((3, 4, 4), 0.4), torch.full((3, 4, 4), 0.1)])
    return {"bg_view_image": bg}


def test_bgc_loss_positive_when_views_differ():
    m = _module(loss_type="l1")
    preds = torch.tensor([0.2, 0.5])
    loss = m._compute_bg_consistency_loss(_batch(), preds)
    # l1 = mean(|0.2-0.4|, |0.5-0.1|) = mean(0.2, 0.4) = 0.3
    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.3), atol=1e-6)


def test_bgc_l2_differs_from_l1():
    preds = torch.tensor([0.2, 0.5])
    l2 = _module(loss_type="l2")._compute_bg_consistency_loss(_batch(), preds)
    # l2 = mean(0.2^2, 0.4^2) = mean(0.04, 0.16) = 0.10
    assert torch.isclose(l2, torch.tensor(0.10), atol=1e-6)


def test_bgc_none_when_disabled_or_view_missing():
    assert _module(enabled=False)._compute_bg_consistency_loss(_batch(), torch.zeros(2)) is None
    assert _module(enabled=True)._compute_bg_consistency_loss({}, torch.zeros(2)) is None


def test_bgc_zero_when_views_identical():
    m = _module(loss_type="l1")
    # preds equal the bg_view means -> zero disagreement
    preds = torch.tensor([0.4, 0.1])
    loss = m._compute_bg_consistency_loss(_batch(), preds)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
