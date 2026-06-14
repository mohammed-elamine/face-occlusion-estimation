"""Post-hoc monotonic recalibration of occlusion predictions.

The regressor under-predicts the mid-high occlusion band. A monotonic map ``g(ŷ)`` fit to
minimise the **challenge-weighted** squared error (``w = 1/30 + y``) is exactly the
MSE-optimal monotone calibrator for our metric, and it needs no retraining — only the
saved ``val_predictions.csv``.

Two entry points:

* :func:`fit_weighted_isotonic` — fit one mapping (with tail regularisation). Used to
  produce the **deploy** artifact (fit on all of validation; inference sees no labels, so
  there is no leak at deploy time).
* :func:`oof_recalibrate` — the honest **evaluation** path: identity-grouped
  out-of-fold recalibration, so the calibrator is never fit and scored on the same person.

The fitted map is stored as plain knot arrays (JSON), so inference depends only on
``numpy.interp`` — no sklearn version / pickle coupling at deploy time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

CHALLENGE_W0 = 1.0 / 30.0


@dataclass(frozen=True)
class IsotonicMapping:
    """A monotone piecewise-linear map ``ŷ -> y``, stored as interpolation knots."""

    x_knots: list[float]
    y_knots: list[float]
    y_min: float = 0.0
    y_max: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def apply(self, preds) -> np.ndarray:
        p = np.asarray(preds, dtype=float).reshape(-1)
        if len(self.x_knots) < 2:
            # Degenerate fit (e.g. constant predictions): identity, clipped.
            return np.clip(p, self.y_min, self.y_max)
        out = np.interp(p, self.x_knots, self.y_knots)
        return np.clip(out, self.y_min, self.y_max)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weighted_isotonic_knots(
    y_pred: np.ndarray, y_true: np.ndarray, weights: np.ndarray, y_min: float, y_max: float
) -> tuple[np.ndarray, np.ndarray]:
    iso = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=True, out_of_bounds="clip")
    iso.fit(y_pred, y_true, sample_weight=weights)
    return np.asarray(iso.X_thresholds_, dtype=float), np.asarray(iso.y_thresholds_, dtype=float)


def _apply_slope_cap(x: np.ndarray, y: np.ndarray, slope_cap: float) -> np.ndarray:
    """Clamp the local slope ``Δy/Δx`` so a couple of tail points can't define a near
    vertical jump. Stays monotone non-decreasing."""
    if slope_cap is None or len(x) < 2:
        return y
    out = y.copy()
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        max_y = out[i - 1] + slope_cap * dx
        if out[i] > max_y:
            out[i] = max_y
    return out


def _enforce_min_samples(
    x: np.ndarray, y: np.ndarray, y_pred_sorted: np.ndarray, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Thin knots so every retained segment is supported by >= ``min_samples`` raw rows."""
    if min_samples <= 1 or len(x) <= 2:
        return x, y
    keep = [0]
    last = x[0]
    for i in range(1, len(x) - 1):
        # rows with prediction in (last, x[i]]
        n = int(
            np.searchsorted(y_pred_sorted, x[i], side="right")
            - np.searchsorted(y_pred_sorted, last, side="right")
        )
        if n >= min_samples:
            keep.append(i)
            last = x[i]
    keep.append(len(x) - 1)
    return x[keep], y[keep]


def fit_weighted_isotonic(
    y_pred,
    y_true,
    *,
    weights=None,
    slope_cap: float | None = 3.0,
    min_samples: int = 10,
    y_min: float = 0.0,
    y_max: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> IsotonicMapping:
    """Fit a challenge-weighted monotone calibrator ``ŷ -> y`` with tail regularisation.

    ``weights`` defaults to the challenge weight ``1/30 + y_true`` so the fit minimises the
    exact metric. ``slope_cap`` and ``min_samples`` guard against the tiny high-occlusion
    tail (only tens of rows) defining a spurious steep step.
    """
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same length")
    w = (CHALLENGE_W0 + y_true) if weights is None else np.asarray(weights, dtype=float).reshape(-1)

    if np.unique(y_pred).size < 2:
        # Constant predictions: nothing to calibrate.
        return IsotonicMapping([], [], y_min, y_max, meta or {})

    x, y = _weighted_isotonic_knots(y_pred, y_true, w, y_min, y_max)
    x, y = _enforce_min_samples(x, y, np.sort(y_pred), min_samples)
    y = _apply_slope_cap(x, y, slope_cap)

    info = {
        "n_fit": int(y_pred.size),
        "challenge_weighted": weights is None,
        "slope_cap": slope_cap,
        "min_samples": int(min_samples),
        "n_knots": int(len(x)),
    }
    if meta:
        info.update(meta)
    return IsotonicMapping(x.tolist(), y.tolist(), float(y_min), float(y_max), info)


def oof_recalibrate(
    y_pred,
    y_true,
    group_ids,
    *,
    n_folds: int = 5,
    seed: int = 42,
    edges=None,
    **fit_kwargs,
) -> np.ndarray:
    """Identity-grouped out-of-fold recalibration → an unbiased recalibrated prediction set.

    No identity ever appears in both the isotonic-fit set and the held-out set, so the
    recalibrated predictions are honest (the same leakage the group-cluster bootstrap was
    built to expose). Folds are stratified by occlusion bin so each contains tail rows.
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    from ..data.normalize import assign_occlusion_bin
    from ..metrics.eval_lenses import DEFAULT_LENS_EDGES

    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).reshape(-1)
    edges = DEFAULT_LENS_EDGES if edges is None else edges

    n_groups = np.unique(groups).size
    n_splits = max(2, min(int(n_folds), n_groups))
    y_bin = assign_occlusion_bin(y_true, edges)
    x_dummy = y_pred.reshape(-1, 1)

    # Stratify by occlusion bin when possible; fall back to plain grouped folds if the rare
    # tail strata have too few groups for StratifiedGroupKFold (mirrors splits.py).
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(x_dummy, y=y_bin, groups=groups))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(x_dummy, y=y_true, groups=groups))

    oof = np.full_like(y_pred, np.nan, dtype=float)
    for train_idx, test_idx in folds:
        m = fit_weighted_isotonic(y_pred[train_idx], y_true[train_idx], **fit_kwargs)
        oof[test_idx] = m.apply(y_pred[test_idx])

    # Any row never held out (shouldn't happen) falls back to identity.
    missing = np.isnan(oof)
    if missing.any():
        oof[missing] = np.clip(y_pred[missing], 0.0, 1.0)
    return oof


def save_mapping(mapping: IsotonicMapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping.as_dict(), indent=2), encoding="utf-8")


def load_mapping(path: str | Path) -> IsotonicMapping:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return IsotonicMapping(
        x_knots=[float(v) for v in data["x_knots"]],
        y_knots=[float(v) for v in data["y_knots"]],
        y_min=float(data.get("y_min", 0.0)),
        y_max=float(data.get("y_max", 1.0)),
        meta=data.get("meta", {}),
    )
