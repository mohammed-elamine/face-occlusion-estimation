"""Tests for the high-occlusion aggregate validation diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from face_occlusion.training.lit_module import FaceOcclusionLitModule


def _make_cfg() -> SimpleNamespace:
    split = SimpleNamespace(occlusion_bins=[0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0])
    return SimpleNamespace(split=split)


def _make_module() -> FaceOcclusionLitModule:
    """Build a LitModule without invoking the real __init__ (no timm download)."""
    module = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(module)
    module.cfg = _make_cfg()
    module._val_buffer = []
    module._female_value = "0.0"
    module._male_value = "1.0"
    module._ord_loss_enabled = False
    module._cons_loss_enabled = False
    return module


def _seed_val_buffer(
    module: FaceOcclusionLitModule,
    preds: np.ndarray,
    targets: np.ndarray,
    genders: np.ndarray,
    databases: list[str] | None = None,
) -> None:
    n = len(targets)
    databases = databases if databases is not None else ["db_a"] * n
    module._val_buffer.append(
        {
            "preds": torch.as_tensor(preds, dtype=torch.float32),
            "targets": torch.as_tensor(targets, dtype=torch.float32),
            "genders": torch.as_tensor(genders, dtype=torch.float32),
            "ordinal_logits": None,
            "image_ids": [f"id_{i}" for i in range(n)],
            "filenames": [f"f_{i}.png" for i in range(n)],
            "paths": [f"p_{i}" for i in range(n)],
            "databases": list(databases),
            "source_subfolders": ["s"] * n,
            "group_ids": [f"g_{i}" for i in range(n)],
            "face_ids": torch.arange(n, dtype=torch.int64),
        }
    )


def _capture_logs(module: FaceOcclusionLitModule) -> dict[str, float]:
    logs: dict[str, float] = {}

    def fake_log(name, value, *args, **kwargs):
        logs[name] = float(value)

    module.log = fake_log  # type: ignore[assignment]
    module.on_validation_epoch_end()
    return logs


def test_aggregate_metrics_present_and_count_matches():
    module = _make_module()
    rng = np.random.default_rng(0)
    targets = np.concatenate(
        [
            rng.uniform(0.0, 0.10, 30),
            rng.uniform(0.10, 0.40, 15),
            rng.uniform(0.40, 0.60, 4),  # [0.40, 0.60)
            rng.uniform(0.60, 1.00, 3),  # [0.60, 1.00]
        ]
    )
    preds = targets + rng.normal(0.0, 0.05, targets.shape)
    genders = rng.choice([0.0, 1.0], size=targets.shape)
    _seed_val_buffer(module, preds, targets, genders)

    logs = _capture_logs(module)

    label = "val/high_occ_0.40_1.00"
    for suffix in ("_count", "_err", "_mae", "_bias", "_weighted_mse"):
        assert label + suffix in logs, f"missing {label + suffix}"
    # Count covers both [0.40, 0.60) and [0.60, 1.00] -- i.e. all targets >= 0.40.
    assert logs[label + "_count"] == int((targets >= 0.40).sum()) == 7


def test_aggregate_err_matches_pooled_weighted_mse():
    from face_occlusion.metrics.challenge_metric import weighted_mse

    module = _make_module()
    targets = np.array([0.05, 0.30, 0.45, 0.55, 0.75, 0.90], dtype=np.float64)
    preds = np.array([0.10, 0.25, 0.50, 0.40, 0.60, 0.95], dtype=np.float64)
    genders = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    _seed_val_buffer(module, preds, targets, genders)

    logs = _capture_logs(module)

    mask = targets >= 0.40
    expected = float(weighted_mse(preds[mask], targets[mask], clip=True))
    assert logs["val/high_occ_0.40_1.00_err"] == pytest.approx(expected)
    assert logs["val/high_occ_0.40_1.00_weighted_mse"] == pytest.approx(expected)


def test_fine_grained_bin_metrics_still_logged():
    module = _make_module()
    rng = np.random.default_rng(1)
    targets = np.concatenate([rng.uniform(0.0, 0.6, 30), rng.uniform(0.4, 1.0, 10)])
    preds = targets + rng.normal(0.0, 0.02, targets.shape)
    genders = rng.choice([0.0, 1.0], size=targets.shape)
    _seed_val_buffer(module, preds, targets, genders)

    logs = _capture_logs(module)

    # Existing fine-grained bin metrics must remain.
    assert "val/bin_0.40_0.60_err" in logs
    assert "val/bin_0.60_1.00_err" in logs
    assert "val/bin_0.40_0.60_count" in logs
    assert "val/bin_0.60_1.00_count" in logs


def test_empty_high_occlusion_does_not_crash():
    module = _make_module()
    # All targets below 0.40 -> empty aggregate mask.
    rng = np.random.default_rng(2)
    targets = rng.uniform(0.0, 0.30, 20)
    preds = targets + rng.normal(0.0, 0.02, targets.shape)
    genders = rng.choice([0.0, 1.0], size=targets.shape)
    _seed_val_buffer(module, preds, targets, genders)

    logs = _capture_logs(module)

    assert logs["val/high_occ_0.40_1.00_count"] == 0
    # No error/mae/bias for an empty aggregate.
    for suffix in ("_err", "_mae", "_bias", "_weighted_mse"):
        assert "val/high_occ_0.40_1.00" + suffix not in logs


def test_gender_specific_aggregate_metrics_when_simple():
    module = _make_module()
    targets = np.array([0.05, 0.45, 0.55, 0.70, 0.95], dtype=np.float64)
    preds = np.array([0.10, 0.40, 0.50, 0.60, 0.80], dtype=np.float64)
    # high-occ: indices 1..4 -> two female (0.0), two male (1.0)
    genders = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    _seed_val_buffer(module, preds, targets, genders)

    logs = _capture_logs(module)

    assert logs["val/high_occ_0.40_1.00_count_female"] == 2
    assert logs["val/high_occ_0.40_1.00_count_male"] == 2
    assert "val/high_occ_0.40_1.00_err_female" in logs
    assert "val/high_occ_0.40_1.00_err_male" in logs
    assert "val/high_occ_0.40_1.00_gap" in logs
    assert logs["val/high_occ_0.40_1.00_gap"] >= 0.0
