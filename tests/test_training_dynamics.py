"""Tests for training-dynamics analysis helpers in analyze_val_predictions.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the scripts directory importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import _bootstrap  # noqa: F401, E402
from analyze_val_predictions import (  # noqa: E402
    _find_metrics_csv,
    _read_training_metrics,
    _write_training_dynamics_plots,
)

# ─── _find_metrics_csv ────────────────────────────────────────────────────────


def test_find_metrics_csv_none_when_no_dir():
    assert _find_metrics_csv(None) is None


def test_find_metrics_csv_none_when_missing(tmp_path: Path):
    assert _find_metrics_csv(tmp_path) is None


def test_find_metrics_csv_finds_standard_location(tmp_path: Path):
    metrics = tmp_path / "logs" / "csv_logs" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("epoch,val/score\n0,0.001\n")
    assert _find_metrics_csv(tmp_path) == metrics


def test_find_metrics_csv_returns_latest_version(tmp_path: Path):
    for version in ("version_0", "version_1", "version_2"):
        p = tmp_path / "logs" / "csv_logs" / version / "metrics.csv"
        p.parent.mkdir(parents=True)
        p.write_text("epoch,val/score\n0,0.001\n")
    result = _find_metrics_csv(tmp_path)
    assert result is not None
    assert "version_2" in str(result)


# ─── _read_training_metrics ───────────────────────────────────────────────────


def _make_sparse_csv(tmp_path: Path) -> Path:
    """Create a Lightning-style sparse metrics.csv with 3 epochs."""
    rows = [
        # epoch NaN = lr-only step
        {"epoch": float("nan"), "step": 0, "train/loss": float("nan"), "val/score": float("nan")},
        # val row for epoch 0
        {"epoch": 0.0, "step": 99, "train/loss": float("nan"), "val/score": 0.010},
        # train row for epoch 0
        {"epoch": 0.0, "step": 99, "train/loss": 0.050, "val/score": float("nan")},
        # val row for epoch 1
        {"epoch": 1.0, "step": 199, "train/loss": float("nan"), "val/score": 0.008},
        # train row for epoch 1
        {"epoch": 1.0, "step": 199, "train/loss": 0.030, "val/score": float("nan")},
        # val row for epoch 2
        {"epoch": 2.0, "step": 299, "train/loss": float("nan"), "val/score": 0.006},
        # train row for epoch 2
        {"epoch": 2.0, "step": 299, "train/loss": 0.020, "val/score": float("nan")},
    ]
    p = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_read_training_metrics_returns_one_row_per_epoch(tmp_path: Path):
    p = _make_sparse_csv(tmp_path)
    result = _read_training_metrics(p)
    assert result is not None
    assert len(result) == 3
    assert list(result["epoch"]) == [0, 1, 2]


def test_read_training_metrics_coalesces_train_and_val(tmp_path: Path):
    p = _make_sparse_csv(tmp_path)
    result = _read_training_metrics(p)
    assert result is not None
    # val/score and train/loss should both be present per epoch
    assert not result["val/score"].isna().any()
    assert not result["train/loss"].isna().any()
    assert pytest.approx(result["val/score"].iloc[0]) == 0.010
    assert pytest.approx(result["train/loss"].iloc[0]) == 0.050


def test_read_training_metrics_returns_none_for_empty(tmp_path: Path):
    p = tmp_path / "metrics.csv"
    p.write_text("epoch,val/score\n")  # header only
    assert _read_training_metrics(p) is None


def test_read_training_metrics_handles_all_nan_epoch(tmp_path: Path):
    """A file where all epoch values are NaN should return None."""
    p = tmp_path / "metrics.csv"
    pd.DataFrame([{"epoch": float("nan"), "step": 0, "val/score": 0.01}]).to_csv(p, index=False)
    assert _read_training_metrics(p) is None


# ─── _write_training_dynamics_plots ──────────────────────────────────────────


def _make_metrics_df(n_epochs: int = 5, include_new_cols: bool = False) -> pd.DataFrame:
    """Build a synthetic per-epoch metrics DataFrame."""
    rng = np.random.default_rng(0)
    epochs = list(range(n_epochs))
    data: dict[str, list] = {
        "epoch": epochs,
        "train/loss": list(rng.uniform(0.01, 0.1, n_epochs)),
        "val/loss": list(rng.uniform(0.01, 0.05, n_epochs)),
        "val/score": list(rng.uniform(0.001, 0.01, n_epochs)),
        "val/err_female": list(rng.uniform(0.001, 0.005, n_epochs)),
        "val/err_male": list(rng.uniform(0.001, 0.005, n_epochs)),
        "val/bin_0.00_0.05_err": list(rng.uniform(0.0001, 0.001, n_epochs)),
        "val/bin_0.60_1.00_err": list(rng.uniform(0.05, 0.3, n_epochs)),
        "val/database/database1_err": list(rng.uniform(0.001, 0.01, n_epochs)),
        "val/database/database2_err": list(rng.uniform(0.001, 0.01, n_epochs)),
    }
    if include_new_cols:
        data["val/mae"] = list(rng.uniform(0.01, 0.05, n_epochs))
        data["val/female_bias"] = list(rng.uniform(-0.01, 0.01, n_epochs))
        data["val/male_bias"] = list(rng.uniform(-0.01, 0.01, n_epochs))
        data["val/bin_0.00_0.05_bias"] = list(rng.uniform(-0.001, 0.001, n_epochs))
        data["val/bin_0.60_1.00_bias"] = list(rng.uniform(-0.05, 0.05, n_epochs))
    return pd.DataFrame(data)


def test_dynamics_plots_old_metrics_creates_subset(tmp_path: Path):
    """Old-style metrics (no bias cols) should still produce partial plots."""
    metrics_df = _make_metrics_df(n_epochs=5, include_new_cols=False)
    created = _write_training_dynamics_plots(metrics_df, tmp_path)
    names = {p.name for p in created}
    # These always exist
    assert "20_training_global_metrics.png" in names
    assert "21_training_weighted_mse_by_occlusion_bin.png" in names
    assert "23_training_weighted_mse_by_gender.png" in names
    assert "25_training_weighted_mse_by_database.png" in names
    # Bias plots need new columns — should NOT be in created for old metrics
    assert "22_training_bias_by_occlusion_bin.png" not in names
    assert "24_training_bias_by_gender.png" not in names


def test_dynamics_plots_new_metrics_creates_all(tmp_path: Path):
    """New-style metrics (with bias cols) should produce all 6 plots."""
    metrics_df = _make_metrics_df(n_epochs=5, include_new_cols=True)
    created = _write_training_dynamics_plots(metrics_df, tmp_path)
    names = {p.name for p in created}
    for expected in [
        "20_training_global_metrics.png",
        "21_training_weighted_mse_by_occlusion_bin.png",
        "22_training_bias_by_occlusion_bin.png",
        "23_training_weighted_mse_by_gender.png",
        "24_training_bias_by_gender.png",
        "25_training_weighted_mse_by_database.png",
    ]:
        assert expected in names, f"Expected plot {expected} not generated"


def test_dynamics_plots_single_epoch(tmp_path: Path):
    """Single-epoch metrics should not crash."""
    metrics_df = _make_metrics_df(n_epochs=1, include_new_cols=True)
    created = _write_training_dynamics_plots(metrics_df, tmp_path)
    assert len(created) > 0


def test_dynamics_plots_all_nan_val_score(tmp_path: Path):
    """If val/score is all-NaN the plot should still be created via other metrics."""
    metrics_df = _make_metrics_df(n_epochs=4, include_new_cols=False)
    metrics_df["val/score"] = float("nan")
    created = _write_training_dynamics_plots(metrics_df, tmp_path)
    # Plot 20 should still be created from train/loss or val/loss
    names = {p.name for p in created}
    assert "20_training_global_metrics.png" in names
