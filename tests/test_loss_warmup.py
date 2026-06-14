"""Tests for the auxiliary loss warmup mechanism."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch

from face_occlusion.training.lit_module import (
    FaceOcclusionLitModule,
    _scheduled_loss_weight,
)

# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


def test_static_behavior_when_warmup_disabled():
    for ep in range(5):
        w = _scheduled_loss_weight(
            target_weight=0.2,
            warmup_epochs=0,
            warmup_start_weight=0.0,
            current_epoch=ep,
        )
        assert w == 0.2


def test_linear_warmup_from_zero():
    target, warmup, start = 0.1, 3, 0.0
    expected = {0: 1 / 30, 1: 2 / 30, 2: 0.1, 3: 0.1, 7: 0.1}
    for ep, want in expected.items():
        got = _scheduled_loss_weight(
            target_weight=target,
            warmup_epochs=warmup,
            warmup_start_weight=start,
            current_epoch=ep,
        )
        assert math.isclose(got, want, rel_tol=0, abs_tol=1e-9), (ep, got, want)


def test_linear_warmup_with_nonzero_start():
    target, warmup, start = 0.1, 4, 0.02
    # progress = (ep+1)/4, lambda = 0.02 + progress*(0.08)
    expected = {
        0: 0.02 + 0.25 * 0.08,
        1: 0.02 + 0.50 * 0.08,
        2: 0.02 + 0.75 * 0.08,
        3: 0.10,
        10: 0.10,
    }
    for ep, want in expected.items():
        got = _scheduled_loss_weight(
            target_weight=target,
            warmup_epochs=warmup,
            warmup_start_weight=start,
            current_epoch=ep,
        )
        assert math.isclose(got, want, abs_tol=1e-9), (ep, got, want)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target_weight=0.1, warmup_epochs=-1, warmup_start_weight=0.0),
        dict(target_weight=-0.1, warmup_epochs=0, warmup_start_weight=0.0),
        dict(target_weight=0.1, warmup_epochs=0, warmup_start_weight=-0.01),
        dict(target_weight=0.1, warmup_epochs=3, warmup_start_weight=0.2),
    ],
)
def test_invalid_warmup_configs_raise(kwargs):
    with pytest.raises(ValueError):
        _scheduled_loss_weight(current_epoch=0, **kwargs)


# ---------------------------------------------------------------------------
# Lightning-module integration: effective weight + train/lambda_ord log
# ---------------------------------------------------------------------------


THRESHOLDS = [0.05, 0.10, 0.20, 0.40, 0.60]


class _StubOrdinalModel(torch.nn.Module):
    """Tiny stand-in that produces both a regression and an ordinal output."""

    use_ordinal_head = True

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 1)
        self.ord_head = torch.nn.Linear(4, len(THRESHOLDS))
        self.register_buffer("ordinal_thresholds", torch.tensor(THRESHOLDS, dtype=torch.float32))

    def forward(self, x):
        # x: (N, 4)
        from face_occlusion.models.outputs import OcclusionModelOutput

        return OcclusionModelOutput(
            y_pred=self.head(x).squeeze(-1),
            ordinal_logits=self.ord_head(x),
            projection=None,
            features=None,
        )


def _make_module(*, ord_weight: float, warmup_epochs: int, warmup_start: float = 0.0):
    module = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(module)
    module.cfg = SimpleNamespace(
        split=SimpleNamespace(occlusion_bins=[0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0])
    )
    module._val_buffer = []
    module._female_value = "0.0"
    module._male_value = "1.0"
    module.model = _StubOrdinalModel()
    module._ord_loss_enabled = True
    module._ord_weight = ord_weight
    module._ord_warmup_epochs = warmup_epochs
    module._ord_warmup_start_weight = warmup_start
    module.register_buffer("_ord_thresholds", torch.tensor(THRESHOLDS, dtype=torch.float32))
    module.register_buffer(
        "_ord_threshold_weights",
        torch.tensor([1.0, 1.0, 1.2, 2.0, 3.0], dtype=torch.float32),
    )
    module._cons_loss_enabled = False
    module._cons_weight = 0.0
    module._cons_warmup_epochs = 0
    module._cons_warmup_start_weight = 0.0
    module._cons_temperature = 0.05
    module._cons_mode = "symmetric"
    return module


def test_effective_ordinal_weight_tracks_current_epoch():
    module = _make_module(ord_weight=0.1, warmup_epochs=3)

    class _FakeTrainer:
        current_epoch = 0

    module._trainer = _FakeTrainer()  # type: ignore[attr-defined]
    assert math.isclose(module._effective_ordinal_weight(), 1 / 30, abs_tol=1e-9)
    _FakeTrainer.current_epoch = 1
    assert math.isclose(module._effective_ordinal_weight(), 2 / 30, abs_tol=1e-9)
    _FakeTrainer.current_epoch = 5
    assert math.isclose(module._effective_ordinal_weight(), 0.1, abs_tol=1e-9)


def test_training_step_logs_lambda_ord_and_scales_loss():
    module = _make_module(ord_weight=0.1, warmup_epochs=3)

    logs: dict[str, float] = {}

    def fake_log(name, value, *args, **kwargs):
        v = float(value.detach().item()) if torch.is_tensor(value) else float(value)
        logs[name] = v

    module.log = fake_log  # type: ignore[assignment]
    module.optimizers = lambda: None  # type: ignore[assignment]

    class _FakeTrainer:
        current_epoch = 0

    module._trainer = _FakeTrainer()  # type: ignore[attr-defined]

    torch.manual_seed(0)
    batch = {
        "image": torch.randn(8, 4),
        "target": torch.rand(8).clamp(0.0, 1.0),
    }
    total = module.training_step(batch, 0)
    assert torch.isfinite(total)

    assert "train/lambda_ord" in logs
    assert math.isclose(logs["train/lambda_ord"], 1 / 30, abs_tol=1e-6)

    # Sanity: train/loss = train/loss_reg + lambda_ord * train/loss_ord.
    expected = logs["train/loss_reg"] + logs["train/lambda_ord"] * logs["train/loss_ord"]
    assert math.isclose(logs["train/loss"], expected, abs_tol=1e-5)


def test_lambda_ord_not_logged_when_ordinal_disabled():
    module = _make_module(ord_weight=0.1, warmup_epochs=0)
    module._ord_loss_enabled = False  # disable

    logs: dict[str, float] = {}
    module.log = lambda name, value, *a, **k: logs.__setitem__(  # type: ignore[assignment]
        name, float(value.detach().item()) if torch.is_tensor(value) else float(value)
    )
    module.optimizers = lambda: None  # type: ignore[assignment]

    class _FakeTrainer:
        current_epoch = 0

    module._trainer = _FakeTrainer()  # type: ignore[attr-defined]

    batch = {"image": torch.randn(4, 4), "target": torch.rand(4)}
    module.training_step(batch, 0)
    assert "train/lambda_ord" not in logs
    assert "train/lambda_cons" not in logs
