"""Evaluation "lenses": per-row importance weights for the challenge metric.

The challenge metric already up-weights high occlusion (``w = 1/30 + y``), but our
validation set is drawn from the right-skewed *train* distribution. To understand how a
model behaves under a *different* occlusion distribution without retraining, we reweight
each validation row by ``p_target(bin) / p_train(bin)`` and feed that as ``sample_weight``
to :func:`~face_occlusion.metrics.challenge_metric.challenge_score`.

Three lenses are exposed:

* ``official``    — no reweighting (``None``); the metric exactly as scored.
* ``balanced``    — every occlusion bin contributes equally (robustness across the range).
* ``test_matched``— match the digitised test histogram (a *diagnostic* of leaderboard
  behaviour; never a selection target — the grade rewards generalisation, not test-fit).

All weights are normalised to mean 1 (so the official challenge weight scale is preserved)
and clipped, so the data-sparse high-occlusion tail cannot blow up the estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from ..data.normalize import assign_occlusion_bin

LENS_NAMES = ("official", "balanced", "test_matched")

# Default occlusion-bin edges for the lenses (finer than the metric's default so the
# 0.15-0.45 band where train and test differ most is resolved).
DEFAULT_LENS_EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 1.0)

# Repo-relative location of the digitised test distribution.
_DEFAULT_TEST_DIST_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "eval" / "test_distribution.yaml"
)


def _empirical_proportions(targets: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Fraction of ``targets`` falling in each bin defined by ``edges``."""
    n_bins = len(edges) - 1
    bins = assign_occlusion_bin(targets, edges)
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def load_test_distribution(
    path: str | Path | None = None,
) -> tuple[tuple[float, ...], np.ndarray]:
    """Load ``(edges, proportions)`` for the test distribution; proportions sum to 1."""
    path = Path(path) if path is not None else _DEFAULT_TEST_DIST_PATH
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    edges = tuple(float(e) for e in cfg["edges"])
    props = np.asarray(cfg["proportions"], dtype=float)
    if props.shape[0] != len(edges) - 1:
        raise ValueError(
            f"test_distribution: {props.shape[0]} proportions for {len(edges) - 1} bins"
        )
    s = props.sum()
    return edges, (props / s if s > 0 else props)


def importance_weights(
    targets: Sequence[float] | np.ndarray,
    target_proportions: Sequence[float] | np.ndarray,
    edges: Sequence[float] = DEFAULT_LENS_EDGES,
    *,
    clip_max: float = 10.0,
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Per-row weight ``p_target(bin)/p_train(bin)``, mean-normalised and clipped.

    ``p_train`` is estimated empirically from ``targets``. ``smoothing`` is added to both
    histograms so an empty target/train bin cannot create a 0 or infinite ratio.
    Weights are mean-normalised to 1, then clipped to ``[0, clip_max]`` and re-normalised
    to mean 1 so the official challenge-weight scale is preserved.
    """
    t = np.asarray(targets, dtype=float).reshape(-1)
    n_bins = len(edges) - 1
    p_target = np.asarray(target_proportions, dtype=float).reshape(-1)
    if p_target.shape[0] != n_bins:
        raise ValueError(f"target_proportions has {p_target.shape[0]} entries, need {n_bins}")
    p_train = _empirical_proportions(t, edges)

    p_target = p_target + smoothing
    p_train = p_train + smoothing
    ratio = p_target / p_train  # per-bin

    bins = assign_occlusion_bin(t, edges)
    w = ratio[bins]
    w = _mean_normalise(w)
    if clip_max is not None:
        w = np.clip(w, 0.0, clip_max)
        w = _mean_normalise(w)
    return w


def balanced_proportions(n_bins: int) -> np.ndarray:
    """Uniform target proportions — every occlusion bin contributes equally."""
    return np.full(n_bins, 1.0 / n_bins, dtype=float)


def rebin_proportions(
    src_edges: Sequence[float],
    src_props: Sequence[float],
    dst_edges: Sequence[float],
) -> np.ndarray:
    """Re-aggregate a bin distribution from ``src_edges`` onto ``dst_edges``.

    Each source bin's proportion is split across destination bins by overlap fraction, so a
    fine distribution (e.g. the digitised test histogram on ``DEFAULT_LENS_EDGES``) can be
    expressed on a coarser set of edges (e.g. the sampler/split bins). The result is
    renormalised to sum to 1.
    """
    src_edges = np.asarray(src_edges, dtype=float)
    src_props = np.asarray(src_props, dtype=float)
    dst_edges = np.asarray(dst_edges, dtype=float)
    n_dst = len(dst_edges) - 1
    out = np.zeros(n_dst, dtype=float)
    for i in range(len(src_edges) - 1):
        a, b = src_edges[i], src_edges[i + 1]
        width = b - a
        if width <= 0:
            continue
        for j in range(n_dst):
            lo = max(a, dst_edges[j])
            hi = min(b, dst_edges[j + 1])
            if hi > lo:
                out[j] += src_props[i] * (hi - lo) / width
    s = out.sum()
    return out / s if s > 0 else out


def per_bin_importance_weights(
    train_targets: Sequence[float] | np.ndarray,
    target_proportions: Sequence[float] | np.ndarray,
    edges: Sequence[float] = DEFAULT_LENS_EDGES,
    *,
    clip_max: float = 10.0,
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Per-**bin** importance weight vector (length ``n_bins``) for the training side.

    Same operator as :func:`importance_weights`, but it returns one weight per occlusion
    bin (estimating ``p_train`` once from the full ``train_targets``) so training can apply
    it to any batch by indexing with :func:`~face_occlusion.data.normalize.assign_occlusion_bin`
    — avoiding the noise of estimating ``p_train`` from a single mini-batch. Normalised so
    the mean weight over the training population is 1, then clipped, then re-normalised.
    """
    t = np.asarray(train_targets, dtype=float).reshape(-1)
    n_bins = len(edges) - 1
    p_target = np.asarray(target_proportions, dtype=float).reshape(-1)
    if p_target.shape[0] != n_bins:
        raise ValueError(f"target_proportions has {p_target.shape[0]} entries, need {n_bins}")
    p_train = _empirical_proportions(t, edges) + smoothing
    ratio = (p_target + smoothing) / p_train

    bins = assign_occlusion_bin(t, edges)
    pop_mean = float(ratio[bins].mean()) if t.size else 1.0
    if pop_mean > 0:
        ratio = ratio / pop_mean
    if clip_max is not None:
        ratio = np.clip(ratio, 0.0, clip_max)
        pop_mean = float(ratio[bins].mean()) if t.size else 1.0
        if pop_mean > 0:
            ratio = ratio / pop_mean
    return ratio


def lens_weights(
    name: str,
    targets: Sequence[float] | np.ndarray,
    *,
    edges: Sequence[float] = DEFAULT_LENS_EDGES,
    test_proportions: Sequence[float] | np.ndarray | None = None,
    test_dist_path: str | Path | None = None,
    clip_max: float = 10.0,
) -> np.ndarray | None:
    """Return the ``sample_weight`` vector for a lens (``None`` for ``official``).

    For ``test_matched`` either pass ``test_proportions`` directly or let the function
    load them (its own ``edges`` override ``edges``).
    """
    if name == "official":
        return None
    if name == "balanced":
        return importance_weights(
            targets, balanced_proportions(len(edges) - 1), edges, clip_max=clip_max
        )
    if name == "test_matched":
        if test_proportions is None:
            edges, test_proportions = load_test_distribution(test_dist_path)
        return importance_weights(targets, test_proportions, edges, clip_max=clip_max)
    raise ValueError(f"unknown lens {name!r}; expected one of {LENS_NAMES}")


def _mean_normalise(w: np.ndarray) -> np.ndarray:
    m = float(w.mean())
    return w / m if m > 0 else w
