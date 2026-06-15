"""Prediction-ensemble helpers.

Ensembling diverse, individually-tied models by **averaging their per-image
predictions** is the one lever that significantly beat the single-model champion on
this challenge (see ``tmp/comparison_reports/06_ensemble.md``). The averaging itself
is pure ``pandas``/``numpy`` over each member's prediction CSV, so it is decoupled from
inference: per-model *test* predictions are produced once by ``scripts.inference.predict_test``
(which needs the checkpoints, i.e. the pod), and fused here anywhere the small CSVs land.
``pred_clipped`` is the column to average — it is the metric/submission-convention value
and a mean of values in ``[0, 1]`` stays in ``[0, 1]`` (no re-clip needed).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def ensemble_average(
    frames: Sequence[pd.DataFrame],
    weights: Sequence[float] | None = None,
    *,
    value_col: str = "pred_clipped",
    key: str = "image_id",
    keep_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Weighted-average ``value_col`` across per-member prediction frames.

    Frames are aligned on ``key`` (so member row order does not matter). Every frame must
    contain every key of the first frame; a missing or duplicated key is an error rather
    than a silently dropped/averaged row. Returns one row per key with the kept passthrough
    columns (from the first frame), a ``member_{i}_{value_col}`` column per member, and the
    ensemble ``value_col``. ``weights=None`` is a plain mean.
    """
    if not frames:
        raise ValueError("ensemble_average needs at least one frame")
    n = len(frames)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights length {w.shape} does not match {n} frames")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")

    base = frames[0]
    if key not in base.columns:
        raise ValueError(f"key column {key!r} missing from frame 0")
    index = base[key]
    if index.duplicated().any():
        raise ValueError(f"duplicate {key!r} values in frame 0")
    index = pd.Index(index)

    member_cols: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if key not in frame.columns or value_col not in frame.columns:
            raise ValueError(f"frame {i} missing {key!r} or {value_col!r}")
        indexed = frame.set_index(key)
        if indexed.index.duplicated().any():
            raise ValueError(f"duplicate {key!r} values in frame {i}")
        missing = index.difference(indexed.index)
        if len(missing):
            raise ValueError(
                f"frame {i} is missing {len(missing)} keys present in frame 0 "
                f"(e.g. {list(missing[:3])})"
            )
        member_cols.append(indexed.loc[index, value_col].to_numpy(dtype=float))

    stacked = np.vstack(member_cols)  # (n_members, n_rows)
    ensemble = (stacked * w[:, None]).sum(axis=0) / w.sum()

    if keep_cols is None:
        keep = [c for c in base.columns if c != value_col]
    else:
        keep = [c for c in keep_cols if c in base.columns]
    out = base[keep].copy().reset_index(drop=True)
    for i, col in enumerate(member_cols):
        out[f"member_{i}_{value_col}"] = col
    out[value_col] = ensemble
    return out


def score_val_ensemble(
    member_dirs: Sequence[str | Path],
    weights: Sequence[float] | None = None,
    *,
    value_col: str = "pred_clipped",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Average each member's ``predictions/val_predictions.csv`` and score it.

    Returns the ``challenge_score`` dict and the averaged frame. Lets us confirm the
    expected ensemble ``val/score`` from on-disk artifacts (no checkpoints needed) before
    trusting a test submission built from the same members/weights.
    """
    from face_occlusion.metrics.challenge_metric import challenge_score

    frames = [pd.read_csv(Path(d) / "predictions" / "val_predictions.csv") for d in member_dirs]
    ens = ensemble_average(
        frames,
        weights,
        value_col=value_col,
        keep_cols=["image_id", "filename", "target", "gender", "group_id"],
    )
    score = challenge_score(
        ens[value_col].to_numpy(),
        ens["target"].to_numpy(),
        ens["gender"].astype(str).to_numpy(),
    )
    return score, ens
