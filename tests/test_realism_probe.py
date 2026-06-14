"""Tests for the realism-probe AUC core (no image model needed)."""

from __future__ import annotations

import numpy as np
from scripts.analysis.realism_probe import probe_auc


def test_separable_features_give_high_auc():
    rng = np.random.default_rng(0)
    n = 200
    real = rng.normal(3.0, 1.0, size=(n, 8))  # clearly shifted
    synth = rng.normal(-3.0, 1.0, size=(n, 8))
    feats = np.vstack([real, synth])
    labels = np.array([1] * n + [0] * n)
    assert probe_auc(feats, labels, seed=0) > 0.95


def test_indistinguishable_features_give_chance_auc():
    rng = np.random.default_rng(1)
    n = 300
    feats = rng.normal(0.0, 1.0, size=(2 * n, 8))  # both classes same distribution
    labels = np.array([1] * n + [0] * n)
    auc = probe_auc(feats, labels, seed=0)
    assert 0.35 < auc < 0.65  # near chance


def test_auc_is_deterministic():
    rng = np.random.default_rng(2)
    feats = rng.normal(0, 1, size=(120, 5))
    labels = np.array([1, 0] * 60)
    assert probe_auc(feats, labels, seed=3) == probe_auc(feats, labels, seed=3)
