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
    module._mono_loss_enabled = False
    module._mono_weight = 0.0
    module._mono_warmup_epochs = 0
    module._mono_warmup_start_weight = 0.0
    module._rank_loss_enabled = False
    module._rank_weight = 0.0
    module._rank_warmup_epochs = 0
    module._rank_warmup_start_weight = 0.0
    module._bgc_enabled = False
    module._reg_reweight = "none"
    module._reg_bin_weights = None
    module._reg_edges = None
    module._reg_loss_type = "weighted_mse"
    module._reg_high_occ_power = 1.0
    module._reg_gap_lambda = 0.0
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


# ---------------------------------------------------------------------------
# Distribution-aware regression reweighting (Intervention B)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from face_occlusion.metrics.eval_lenses import (  # noqa: E402
    DEFAULT_LENS_EDGES,
    balanced_proportions,
    per_bin_importance_weights,
)
from face_occlusion.training.lit_module import weighted_mse_loss  # noqa: E402


def test_weighted_mse_loss_sample_weight_none_is_identity():
    p = torch.tensor([0.1, 0.5, 0.9])
    t = torch.tensor([0.2, 0.4, 0.8])
    assert float(weighted_mse_loss(p, t)) == float(weighted_mse_loss(p, t, sample_weight=None))
    ones = torch.ones(3)
    base = float(weighted_mse_loss(p, t))
    with_ones = float(weighted_mse_loss(p, t, sample_weight=ones))
    assert abs(base - with_ones) < 1e-9


def _make_reweight_module(reweight: str, *, warmup_epochs: int = 0):
    module = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(module)
    module._reg_reweight = reweight
    module._reg_weight = 1.0
    module._reg_warmup_epochs = warmup_epochs
    module._reg_warmup_start_weight = 0.0
    module._reg_edges = np.asarray(DEFAULT_LENS_EDGES, dtype=float)
    if reweight == "none":
        module._reg_bin_weights = None
    else:
        # Right-skewed train distribution -> tail bins get up-weighted.
        rng = np.random.default_rng(0)
        train = np.clip(rng.beta(1.5, 8.0, 5000), 0, 1)
        n_bins = len(DEFAULT_LENS_EDGES) - 1
        bw = per_bin_importance_weights(train, balanced_proportions(n_bins), DEFAULT_LENS_EDGES)
        module.register_buffer("_reg_bin_weights", torch.tensor(bw, dtype=torch.float32))

    class _FakeTrainer:
        current_epoch = 0

    module._trainer = _FakeTrainer()  # type: ignore[attr-defined]
    return module


def test_reweight_none_yields_no_sample_weight():
    module = _make_reweight_module("none")
    assert module._regression_sample_weight(torch.rand(16)) is None


def test_reweight_balanced_upweights_tail_rows():
    module = _make_reweight_module("balanced")
    targets = torch.tensor([0.01, 0.02, 0.03, 0.7, 0.8])  # 3 easy, 2 tail
    sw = module._regression_sample_weight(targets)
    assert sw is not None
    assert float(sw[3:].mean()) > float(sw[:3].mean())  # tail weighted more
    assert abs(float(sw.mean()) - 1.0) < 1e-5  # renormalised to mean 1


def test_reweight_warmup_blends_toward_official():
    # With warmup active at epoch 0, lambda<1 so weights are pulled toward all-ones.
    warm = _make_reweight_module("balanced", warmup_epochs=4)
    full = _make_reweight_module("balanced", warmup_epochs=0)
    targets = torch.tensor([0.01, 0.02, 0.7, 0.8])
    sw_warm = warm._regression_sample_weight(targets)
    sw_full = full._regression_sample_weight(targets)
    spread_warm = float(sw_warm.max() - sw_warm.min())
    spread_full = float(sw_full.max() - sw_full.min())
    assert spread_warm < spread_full  # warmup compresses the reweighting
