"""Shared target normalization and occlusion-bin assignment.

These two operations were previously re-implemented in three places:

* ``dataset._normalize_target_scale`` (honoured ``unit``/``percent``/``auto``),
* ``splits._normalize_targets`` / ``splits._occlusion_bin`` (``auto`` only),
* the warm-start heuristic in ``scripts/training/train.py`` (inline ``auto``).

Three copies can silently disagree. In particular the sampler used to re-bin the
*raw* training target instead of the split's normalized bins, which is only safe
because the labels happen to be in ``[0, 1]`` today. Unifying both operations
here means the model targets, the split bins and the sampler bins always share a
single definition.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Challenge labels may be stored as fractions in [0, 1] or as percentages in
# [0, 100]. Under ``auto`` scaling, a max above this cutoff is treated as
# percent-scaled and divided by 100.
PERCENT_SCALE_CUTOFF = 1.5

_VALID_SCALES = ("unit", "percent", "auto")


def normalize_target(values: pd.Series | np.ndarray | Sequence[float], scale: str = "auto"):
    """Normalize occlusion targets to the unit range ``[0, 1]``.

    Parameters
    ----------
    values:
        Target values. A ``pandas.Series`` is returned as a ``Series``; any
        other array-like is returned as a ``numpy.ndarray``.
    scale:
        ``"unit"`` leaves values unchanged, ``"percent"`` divides by 100, and
        ``"auto"`` divides by 100 only when ``max(values) > PERCENT_SCALE_CUTOFF``.
    """
    if scale not in _VALID_SCALES:
        raise ValueError(f"scale must be one of {_VALID_SCALES}, got {scale!r}")
    if isinstance(values, pd.Series):
        arr = values.astype(float)
    else:
        arr = np.asarray(values, dtype=float)
    if scale == "unit":
        return arr
    if scale == "percent":
        return arr / 100.0
    # auto
    if float(arr.max()) > PERCENT_SCALE_CUTOFF:
        return arr / 100.0
    return arr


def assign_occlusion_bin(
    values: np.ndarray | Sequence[float], edges: Sequence[float]
) -> np.ndarray:
    """Assign each value to an occlusion-bin index in ``[0, len(edges) - 2]``.

    ``np.digitize`` is applied on the *interior* edges (``edges[1:-1]``) with
    ``right=False`` so each bin is ``[lo, hi)``. The closing edge value
    (typically ``1.0``) would otherwise land one bin too high, so the result is
    clipped back into the last valid bin.
    """
    edges_arr = np.asarray(edges, dtype=float)
    if edges_arr.ndim != 1 or edges_arr.size < 2:
        raise ValueError("`edges` must be a 1-D sequence of at least 2 values")
    vals = np.asarray(values, dtype=float)
    idx = np.digitize(vals, edges_arr[1:-1], right=False)
    return np.clip(idx, 0, len(edges_arr) - 2).astype(int)
