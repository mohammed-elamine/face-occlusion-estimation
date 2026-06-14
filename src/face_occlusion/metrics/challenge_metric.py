"""Official challenge metric.

Err = sum_i w_i (y_hat_i - y_i)^2 / sum_i w_i   with   w_i = 1/30 + y_i
Score = (Err_F + Err_M) / 2 + |Err_F - Err_M|

Predictions are clipped to [0, 1] for validation metrics and submission.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

DEFAULT_BINS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0)


def _to_np(x) -> np.ndarray:
    # Metrics accept either NumPy arrays or torch tensors from Lightning.
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=float)


def weighted_mse(preds, targets, clip: bool = True, sample_weight=None) -> float:
    p = _to_np(preds).reshape(-1)
    t = _to_np(targets).reshape(-1).astype(float)
    if clip:
        p = np.clip(p, 0.0, 1.0)
    w = 1.0 / 30.0 + t
    if sample_weight is not None:
        # Optional per-row importance weight for evaluation "lenses" (e.g. matching the
        # test distribution). It multiplies the challenge weight; passing None or an
        # all-ones vector leaves the official metric unchanged.
        w = w * _to_np(sample_weight).reshape(-1)
    denom = w.sum()
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * (p - t) ** 2) / denom)


def weighted_mse_by_group(preds, targets, groups, clip: bool = True) -> dict[str, float]:
    p = _to_np(preds).reshape(-1)
    t = _to_np(targets).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    out: dict[str, float] = {}
    for key in np.unique(g):
        mask = g == key
        if mask.sum() == 0:
            continue
        out[str(key)] = weighted_mse(p[mask], t[mask], clip=clip)
    return out


def challenge_score(
    preds,
    targets,
    genders,
    female_value="0.0",
    male_value="1.0",
    clip: bool = True,
    sample_weight=None,
) -> dict[str, float]:
    p = _to_np(preds).reshape(-1)
    t = _to_np(targets).reshape(-1)
    g = np.asarray(genders).astype(str).reshape(-1)
    female_value = str(female_value)
    male_value = str(male_value)
    sw = _to_np(sample_weight).reshape(-1) if sample_weight is not None else None

    mask_f = g == female_value
    mask_m = g == male_value

    # Compute subgroup errors separately because the final metric penalizes imbalance.
    sw_f = sw[mask_f] if sw is not None else None
    sw_m = sw[mask_m] if sw is not None else None
    err_f = (
        weighted_mse(p[mask_f], t[mask_f], clip=clip, sample_weight=sw_f)
        if mask_f.any()
        else float("nan")
    )
    err_m = (
        weighted_mse(p[mask_m], t[mask_m], clip=clip, sample_weight=sw_m)
        if mask_m.any()
        else float("nan")
    )

    if np.isnan(err_f) or np.isnan(err_m):
        # If one subgroup is missing the score is ill-defined; fall back to overall WMSE.
        warnings.warn(
            "Challenge score is missing one gender subgroup; falling back to overall weighted MSE.",
            RuntimeWarning,
            stacklevel=2,
        )
        overall = weighted_mse(p, t, clip=clip, sample_weight=sw)
        return {
            "score": overall,
            "err_female": err_f,
            "err_male": err_m,
            "gender_gap": float("nan"),
            "err_mean": overall,
        }

    gap = abs(err_f - err_m)
    mean = 0.5 * (err_f + err_m)
    return {
        "score": float(mean + gap),
        "err_female": float(err_f),
        "err_male": float(err_m),
        "gender_gap": float(gap),
        "err_mean": float(mean),
    }


def error_by_occlusion_bin(
    preds,
    targets,
    genders=None,
    bins: Sequence[float] = DEFAULT_BINS,
    clip: bool = True,
) -> dict[str, float]:
    p = _to_np(preds).reshape(-1)
    t = _to_np(targets).reshape(-1)
    edges = np.asarray(bins, dtype=float)
    out: dict[str, float] = {}
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # The final bin includes its right edge so target=1.0 is not dropped.
        mask = (t >= lo) & (t < hi if i < len(edges) - 2 else t <= hi)
        if mask.sum() == 0:
            out[f"{lo:.2f}_{hi:.2f}"] = float("nan")
        else:
            out[f"{lo:.2f}_{hi:.2f}"] = weighted_mse(p[mask], t[mask], clip=clip)
    return out
