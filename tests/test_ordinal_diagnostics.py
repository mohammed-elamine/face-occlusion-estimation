"""Tests for full-epoch ordinal-head and consistency validation diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from face_occlusion.models import OrdinalHead
from face_occlusion.training.lit_module import (
    FaceOcclusionLitModule,
    _per_threshold_prf,
    _safe_mean,
)

THRESHOLDS = [0.05, 0.10, 0.20, 0.40, 0.60]
BINS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]


def _make_module(*, ord_enabled: bool = True, cons_enabled: bool = False):
    module = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(module)
    module.cfg = SimpleNamespace(split=SimpleNamespace(occlusion_bins=BINS))
    module._val_buffer = []
    module._female_value = "0.0"
    module._male_value = "1.0"

    # Ordinal wiring.
    module._ord_loss_enabled = ord_enabled
    thresholds_t = torch.tensor(THRESHOLDS, dtype=torch.float32)
    module.register_buffer("_ord_thresholds", thresholds_t.clone())
    module.register_buffer(
        "_ord_threshold_weights",
        torch.ones(len(THRESHOLDS), dtype=torch.float32),
    )

    # Consistency wiring.
    module._cons_loss_enabled = cons_enabled
    module._cons_temperature = 0.05
    module._cons_mode = "symmetric"

    # The consistency path reads thresholds from the model; provide a stub.
    stub_model = SimpleNamespace(
        use_ordinal_head=True,
        ordinal_thresholds=thresholds_t.clone(),
        ordinal_head=OrdinalHead(4, len(THRESHOLDS)),
    )
    module.model = stub_model
    return module


def _seed_val_buffer(
    module,
    preds: np.ndarray,
    targets: np.ndarray,
    genders: np.ndarray,
    ordinal_logits: np.ndarray | None,
    databases: list[str] | None = None,
) -> None:
    n = len(targets)
    databases = databases if databases is not None else ["database1"] * n
    module._val_buffer.append(
        {
            "preds": torch.as_tensor(preds, dtype=torch.float32),
            "targets": torch.as_tensor(targets, dtype=torch.float32),
            "genders": torch.as_tensor(genders, dtype=torch.float32),
            "ordinal_logits": (
                torch.as_tensor(ordinal_logits, dtype=torch.float32)
                if ordinal_logits is not None
                else None
            ),
            "image_ids": [f"id_{i}" for i in range(n)],
            "filenames": [f"f_{i}.png" for i in range(n)],
            "paths": [f"p_{i}" for i in range(n)],
            "databases": list(databases),
            "source_subfolders": ["s"] * n,
            "group_ids": [f"g_{i}" for i in range(n)],
            "face_ids": torch.arange(n, dtype=torch.int64),
        }
    )


def _capture(module) -> dict[str, float]:
    logs: dict[str, float] = {}

    def fake_log(name, value, *args, **kwargs):
        logs[name] = float(value)

    module.log = fake_log  # type: ignore[assignment]
    module.on_validation_epoch_end()
    return logs


def _logits_matching_targets(targets: np.ndarray, thresholds=THRESHOLDS) -> np.ndarray:
    """Return logits whose argmax-threshold predictions perfectly match targets."""
    out = np.zeros((len(targets), len(thresholds)), dtype=np.float64)
    for k, t in enumerate(thresholds):
        out[:, k] = np.where(targets > t, 5.0, -5.0)
    return out


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_per_threshold_prf_safe_when_no_positives():
    preds = np.array([[False, False], [False, False]])
    y = np.array([[False, False], [False, False]])
    acc, prec, rec, f1 = _per_threshold_prf(preds, y)
    assert (acc == 1.0).all()
    assert rec == [None, None]
    assert prec == [None, None]
    assert f1 == [None, None]


def test_per_threshold_prf_matches_hand_calc():
    preds = np.array([[True, True], [True, False], [False, True], [False, False]])
    y = np.array([[True, False], [True, True], [False, True], [False, False]])
    acc, prec, rec, f1 = _per_threshold_prf(preds, y)
    assert acc[0] == 1.0
    assert prec[0] == pytest.approx(1.0)
    assert rec[0] == pytest.approx(1.0)
    assert f1[0] == pytest.approx(1.0)


def test_safe_mean_ignores_none_and_handles_empty():
    assert _safe_mean([0.5, None, 1.0]) == pytest.approx(0.75)
    assert _safe_mean([None, None]) == 0.0
    assert _safe_mean([]) == 0.0


# ---------------------------------------------------------------------------
# Epoch-end ordinal logging
# ---------------------------------------------------------------------------


def test_ordinal_metrics_absent_when_head_disabled():
    module = _make_module(ord_enabled=False)
    rng = np.random.default_rng(0)
    n = 20
    targets = rng.uniform(0.0, 0.6, n)
    preds = targets + rng.normal(0, 0.05, n)
    genders = rng.choice([0.0, 1.0], size=n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=None)
    logs = _capture(module)
    assert not any(k.startswith("val/ord") for k in logs)


def test_global_ordinal_metrics_present_and_finite():
    module = _make_module(ord_enabled=True)
    rng = np.random.default_rng(1)
    n = 80
    targets = np.concatenate(
        [
            rng.uniform(0.0, 0.10, 50),
            rng.uniform(0.10, 0.40, 20),
            rng.uniform(0.40, 0.60, 6),
            rng.uniform(0.60, 1.0, 4),
        ]
    )
    preds = targets + rng.normal(0, 0.05, n)
    logits = _logits_matching_targets(targets) + rng.normal(0, 0.5, (n, len(THRESHOLDS)))
    genders = rng.choice([0.0, 1.0], size=n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    for key in (
        "val/ord_loss",
        "val/ord_threshold_acc_mean",
        "val/ord_threshold_precision_mean",
        "val/ord_threshold_recall_mean",
        "val/ord_threshold_f1_mean",
    ):
        assert key in logs
        assert np.isfinite(logs[key])


def test_per_threshold_support_counts_match_targets():
    module = _make_module(ord_enabled=True)
    rng = np.random.default_rng(2)
    n = 60
    targets = np.concatenate(
        [rng.uniform(0.0, 0.10, 30), rng.uniform(0.20, 0.60, 25), rng.uniform(0.60, 1.0, 5)]
    )
    preds = targets.copy()
    logits = _logits_matching_targets(targets)
    genders = rng.choice([0.0, 1.0], size=n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    for t in THRESHOLDS:
        expected_pos = int((targets > t).sum())
        expected_neg = n - expected_pos
        assert logs[f"val/ord_t_{t:.2f}_support_pos"] == expected_pos
        assert logs[f"val/ord_t_{t:.2f}_support_neg"] == expected_neg


def test_high_threshold_recall_perfect_when_logits_match_targets():
    module = _make_module(ord_enabled=True)
    rng = np.random.default_rng(3)
    n = 50
    targets = np.concatenate([rng.uniform(0.0, 0.40, 30), rng.uniform(0.40, 1.0, 20)])
    preds = targets.copy()
    logits = _logits_matching_targets(targets)
    genders = rng.choice([0.0, 1.0], size=n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    assert logs["val/ord_high_threshold_recall_0.40"] == pytest.approx(1.0)
    assert logs["val/ord_high_threshold_recall_0.60"] == pytest.approx(1.0)
    assert logs["val/ord_threshold_recall_0.40"] == pytest.approx(1.0)


def test_high_threshold_recall_safe_when_no_positives():
    module = _make_module(ord_enabled=True)
    n = 20
    targets = np.full(n, 0.1)  # no positives above any threshold > 0.10
    preds = targets.copy()
    logits = _logits_matching_targets(targets)
    genders = np.zeros(n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    # No positives for t=0.40 or 0.60 -> those keys should be absent (skipped).
    assert "val/ord_high_threshold_recall_0.40" not in logs
    assert "val/ord_high_threshold_recall_0.60" not in logs
    # Support counts are still emitted.
    assert logs["val/ord_t_0.40_support_pos"] == 0
    assert logs["val/ord_t_0.60_support_pos"] == 0


def test_aggregated_high_occlusion_uses_threshold_0_40():
    module = _make_module(ord_enabled=True)
    rng = np.random.default_rng(4)
    n = 60
    targets = np.concatenate(
        [rng.uniform(0.0, 0.40, 40), rng.uniform(0.40, 0.60, 12), rng.uniform(0.60, 1.0, 8)]
    )
    preds = targets.copy()
    logits = _logits_matching_targets(targets)
    genders = rng.choice([0.0, 1.0], size=n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    assert logs["val/ord/high_occ_0.40_1.00_count"] == int((targets >= 0.40).sum()) == 20
    assert "val/ord/high_occ_0.40_1.00_threshold_acc_mean" in logs
    assert "val/ord/high_occ_0.40_1.00_threshold_f1_mean" in logs
    # Perfect logits => perfect recall at high thresholds inside the aggregate.
    assert logs["val/ord/high_occ_0.40_1.00_recall_t_0.40"] == pytest.approx(1.0)
    assert logs["val/ord/high_occ_0.40_1.00_recall_t_0.60"] == pytest.approx(1.0)


def test_per_gender_ordinal_metrics_present():
    module = _make_module(ord_enabled=True)
    rng = np.random.default_rng(5)
    n = 40
    targets = rng.uniform(0.0, 1.0, n)
    preds = targets.copy()
    logits = _logits_matching_targets(targets)
    genders = np.array([0.0, 1.0] * (n // 2))
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    for g in ("female", "male"):
        assert logs[f"val/ord/{g}_count"] == n // 2
        assert f"val/ord/{g}_threshold_acc_mean" in logs
        assert f"val/ord/{g}_threshold_f1_mean" in logs


def test_per_database_ordinal_metrics_present():
    module = _make_module(ord_enabled=True)
    rng = np.random.default_rng(6)
    n = 30
    targets = rng.uniform(0.0, 1.0, n)
    preds = targets.copy()
    logits = _logits_matching_targets(targets)
    genders = rng.choice([0.0, 1.0], size=n)
    dbs = ["database1"] * 10 + ["database2"] * 10 + ["database3"] * 10
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits, databases=dbs)
    logs = _capture(module)
    for db in ("database1", "database2", "database3"):
        assert logs[f"val/ord/database/{db}_count"] == 10
        assert f"val/ord/database/{db}_threshold_acc_mean" in logs
        assert f"val/ord/database/{db}_threshold_f1_mean" in logs


def test_per_bin_ordinal_metrics_present():
    module = _make_module(ord_enabled=True)
    rng = np.random.default_rng(7)
    n = 60
    targets = rng.uniform(0.0, 1.0, n)
    preds = targets.copy()
    logits = _logits_matching_targets(targets)
    genders = rng.choice([0.0, 1.0], size=n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        label = f"{lo:.2f}_{hi:.2f}"
        assert f"val/ord/bin_{label}_count" in logs
        n_bin = int(((targets >= lo) & (targets < hi if hi < 1.0 else targets <= hi)).sum())
        assert logs[f"val/ord/bin_{label}_count"] == n_bin


def test_consistency_metrics_include_per_threshold_gap():
    module = _make_module(ord_enabled=True, cons_enabled=True)
    rng = np.random.default_rng(8)
    n = 30
    targets = rng.uniform(0.0, 1.0, n)
    preds = targets + rng.normal(0, 0.05, n)
    logits = _logits_matching_targets(targets) + rng.normal(0, 0.2, (n, len(THRESHOLDS)))
    genders = rng.choice([0.0, 1.0], size=n)
    _seed_val_buffer(module, preds, targets, genders, ordinal_logits=logits)
    logs = _capture(module)
    assert "val/cons_loss" in logs
    assert "val/cons_gap_mean" in logs
    for t in THRESHOLDS:
        assert f"val/cons_gap_t_{t:.2f}" in logs
        assert np.isfinite(logs[f"val/cons_gap_t_{t:.2f}"])
