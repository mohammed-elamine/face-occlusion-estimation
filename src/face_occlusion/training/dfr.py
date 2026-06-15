"""Deep Feature Reweighting (DFR) core — group-balanced last-layer refit.

DFR (Kirichenko, Izmailov, Wilson, ICLR 2023) removes a model's reliance on a spurious feature
by **retraining only the final linear layer** on a **group-balanced** version of the data, with
the encoder frozen. Here the spurious correlation is gender↔occlusion (see
``tmp/model_study/05_gender_gap.md``): the encoder under-predicts occluded males. We refit the
regression head on features balanced across gender×occlusion groups so the head stops using the
gender shortcut.

These are pure NumPy helpers (no torch) so they are fast and unit-testable; the driver that
extracts encoder features and wires them up is ``scripts.analysis.fit_dfr``.
"""

from __future__ import annotations

import numpy as np


def occlusion_group_ids(
    targets, genders, edges, *, female_value: float = 0.0, male_value: float = 1.0
) -> np.ndarray:
    """Assign each sample a gender×occlusion-bin group id (``gender_code * n_bins + occ_bin``).

    Samples whose gender is neither value get group id ``-1`` (excluded from balancing).
    """
    t = np.asarray(targets, dtype=float).reshape(-1)
    g = np.asarray(genders, dtype=float).reshape(-1)
    edges = np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    occ_bin = np.clip(np.digitize(t, edges[1:-1], right=False), 0, n_bins - 1)
    gcode = np.where(np.isclose(g, male_value), 1, np.where(np.isclose(g, female_value), 0, -1))
    gid = gcode * n_bins + occ_bin
    return np.where(gcode < 0, -1, gid)


def group_balance_weights(group_ids) -> np.ndarray:
    """Per-sample weights that equalize total influence across groups (mean ≈ 1).

    Each group gets weight ∝ 1/size, so all groups contribute equally regardless of count.
    Group id ``-1`` (unknown gender) gets weight 0.
    """
    gids = np.asarray(group_ids)
    w = np.zeros(len(gids), dtype=float)
    for gid in np.unique(gids):
        if gid < 0:
            continue
        mask = gids == gid
        w[mask] = 1.0 / int(mask.sum())
    pos = w > 0
    if pos.any():
        w[pos] = w[pos] / w[pos].mean()  # normalize to mean 1 over included rows
    return w


def fit_ridge_head(
    features, targets, sample_weight=None, *, ridge: float = 1.0
) -> tuple[np.ndarray, float]:
    """Closed-form weighted ridge regression of ``targets`` on ``features`` → (weight, bias).

    Minimizes ``Σ_i sw_i (xᵢ·w + b − yᵢ)² + ridge·‖w‖²`` (bias unregularized). For an
    identity-output head this is the exact metric-aligned (weighted-MSE) last layer.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1)
    n, d = X.shape
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    xb = np.concatenate([X, np.ones((n, 1))], axis=1)  # append bias column
    a = xb.T @ (sw[:, None] * xb)  # (d+1, d+1)
    reg = ridge * np.eye(d + 1)
    reg[-1, -1] = 0.0  # do not regularize the bias term
    beta = np.linalg.solve(a + reg, xb.T @ (sw * y))
    return beta[:-1], float(beta[-1])


def apply_head(features, weight, bias) -> np.ndarray:
    """Linear head output ``X·w + b`` (apply activation/clip downstream)."""
    return np.asarray(features, dtype=float) @ np.asarray(weight, dtype=float) + float(bias)
