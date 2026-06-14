"""Bootstrap confidence intervals for the challenge metric.

The validation tail is tiny (tens of rows with ``y >= 0.40``, a handful with
``y >= 0.60``) yet the metric weights exactly that tail, so a single point
estimate of ``val/score`` hides large sampling noise. These helpers resample the
validation rows and report percentile CIs so an ablation delta can be read as
signal vs noise.

Two resampling units are supported:

* ``"row"`` — resample rows i.i.d. (standard nonparametric bootstrap).
* ``"group"`` — resample *groups* (e.g. ``group_id`` identity clusters) with
  replacement, keeping each group's rows together. This is the honest choice
  when identities leak across train/val: i.i.d. row resampling understates the
  variance because rows from the same identity are correlated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .challenge_metric import challenge_score, weighted_mse


@dataclass(frozen=True)
class MetricCI:
    """Point estimate plus a percentile confidence interval."""

    point: float
    lo: float
    hi: float
    std: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "std": self.std}


def _compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    genders: np.ndarray,
    *,
    clip: bool,
    female_value: str,
    male_value: str,
    high_occ_threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """All scalar metrics for one (resampled) set of rows."""
    score = challenge_score(
        preds,
        targets,
        genders,
        female_value=female_value,
        male_value=male_value,
        clip=clip,
        sample_weight=sample_weight,
    )
    out = {
        "score": score["score"],
        "err_female": score["err_female"],
        "err_male": score["err_male"],
        "gender_gap": score["gender_gap"],
        "err_mean": score["err_mean"],
    }

    def _wmse(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        sw = sample_weight[mask] if sample_weight is not None else None
        return weighted_mse(preds[mask], targets[mask], clip=clip, sample_weight=sw)

    # High-occlusion aggregate (pools everything above the threshold).
    hi = targets >= high_occ_threshold
    out["high_occ_err"] = _wmse(hi)
    g = genders.astype(str)
    err_hf = _wmse(hi & (g == str(female_value)))
    err_hm = _wmse(hi & (g == str(male_value)))
    out["high_occ_gender_gap"] = (
        abs(err_hf - err_hm) if not (np.isnan(err_hf) or np.isnan(err_hm)) else float("nan")
    )
    return out


def _resample_indices(
    rng: np.random.Generator,
    n: int,
    unit: str,
    group_codes: np.ndarray | None,
    group_members: list[np.ndarray] | None,
) -> np.ndarray:
    if unit == "row":
        return rng.integers(0, n, size=n)
    # group: resample group ids with replacement, concatenate their members.
    assert group_members is not None
    n_groups = len(group_members)
    chosen = rng.integers(0, n_groups, size=n_groups)
    return np.concatenate([group_members[c] for c in chosen])


def bootstrap_challenge_metrics(
    preds: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    genders: Sequence | np.ndarray,
    *,
    group_ids: Sequence | np.ndarray | None = None,
    unit: str = "row",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    clip: bool = True,
    female_value: str = "0.0",
    male_value: str = "1.0",
    high_occ_threshold: float = 0.40,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> dict[str, MetricCI]:
    """Return point estimate + percentile CI for each challenge metric.

    Parameters mirror :func:`challenge_score`. ``unit="group"`` requires
    ``group_ids`` and resamples identity clusters rather than rows. ``sample_weight``
    is an optional per-row evaluation weight (an evaluation "lens"); see
    :mod:`face_occlusion.metrics.eval_lenses`.
    """
    if unit not in {"row", "group"}:
        raise ValueError(f"unit must be 'row' or 'group', got {unit!r}")
    preds = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    genders = np.asarray(genders).reshape(-1)
    n = preds.shape[0]
    if not (n == targets.shape[0] == genders.shape[0]):
        raise ValueError("preds, targets and genders must have the same length")
    sw = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else None

    group_members: list[np.ndarray] | None = None
    if unit == "group":
        if group_ids is None:
            raise ValueError("unit='group' requires group_ids")
        group_arr = np.asarray(group_ids).reshape(-1)
        group_members = [np.flatnonzero(group_arr == g) for g in np.unique(group_arr)]

    point = _compute_metrics(
        preds,
        targets,
        genders,
        clip=clip,
        female_value=female_value,
        male_value=male_value,
        high_occ_threshold=high_occ_threshold,
        sample_weight=sw,
    )

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(int(n_boot)):
        idx = _resample_indices(rng, n, unit, None, group_members)
        m = _compute_metrics(
            preds[idx],
            targets[idx],
            genders[idx],
            clip=clip,
            female_value=female_value,
            male_value=male_value,
            high_occ_threshold=high_occ_threshold,
            sample_weight=sw[idx] if sw is not None else None,
        )
        for k, v in m.items():
            samples[k].append(v)

    return {k: _ci_from_samples(point[k], samples[k], ci) for k in point}


def _ci_from_samples(point_value: float, vals: Sequence[float], ci: float) -> MetricCI:
    """Percentile CI for a bootstrap sample of one scalar."""
    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 * (1.0 - (1.0 - ci) / 2.0)
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return MetricCI(float(point_value), float("nan"), float("nan"), float("nan"))
    return MetricCI(
        point=float(point_value),
        lo=float(np.percentile(finite, lo_q)),
        hi=float(np.percentile(finite, hi_q)),
        std=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
    )


def _group_members(group_ids: Sequence | np.ndarray | None, unit: str) -> list[np.ndarray] | None:
    if unit == "row":
        return None
    if group_ids is None:
        raise ValueError("unit='group' requires group_ids")
    arr = np.asarray(group_ids).reshape(-1)
    return [np.flatnonzero(arr == g) for g in np.unique(arr)]


def _bin_weighted_error(
    preds: np.ndarray,
    targets: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
    *,
    clip: bool,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(weighted_error_sum, weight_sum)`` per bin (unnormalised)."""
    p = np.clip(preds, 0.0, 1.0) if clip else preds
    w = (1.0 / 30.0 + targets).astype(float)
    if sample_weight is not None:
        w = w * sample_weight
    contrib = w * (p - targets) ** 2
    werr = np.bincount(bin_idx, weights=contrib, minlength=n_bins)
    wsum = np.bincount(bin_idx, weights=w, minlength=n_bins)
    return werr, wsum


def bootstrap_per_bin(
    preds: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    edges: Sequence[float],
    group_ids: Sequence | np.ndarray | None = None,
    unit: str = "row",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    clip: bool = True,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> dict[str, dict[str, object]]:
    """Per occlusion-bin diagnostics with CIs.

    For each bin (defined by ``edges``) returns ``count`` plus :class:`MetricCI` for the
    bin's ``weighted_mse`` and its ``score_share`` — the fraction of the total weighted
    error mass the bin contributes (this is the "2 rows = 13.6% of the score" diagnostic).
    """
    from ..data.normalize import assign_occlusion_bin

    preds = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    sw = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else None
    n = preds.shape[0]
    n_bins = len(edges) - 1
    bin_idx = assign_occlusion_bin(targets, edges)
    labels = [f"{float(edges[i]):.2f}_{float(edges[i + 1]):.2f}" for i in range(n_bins)]

    def _stats(p, t, b, w):
        werr, wsum = _bin_weighted_error(p, t, b, n_bins, clip=clip, sample_weight=w)
        total = werr.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            wmse = np.where(wsum > 0, werr / wsum, np.nan)
            share = np.where(total > 0, werr / total, np.nan)
        return wmse, share

    wmse_pt, share_pt = _stats(preds, targets, bin_idx, sw)
    counts = np.bincount(bin_idx, minlength=n_bins)

    members = _group_members(group_ids, unit)
    rng = np.random.default_rng(seed)
    wmse_s = [[] for _ in range(n_bins)]
    share_s = [[] for _ in range(n_bins)]
    for _ in range(int(n_boot)):
        idx = _resample_indices(rng, n, unit, None, members)
        wmse_b, share_b = _stats(
            preds[idx], targets[idx], bin_idx[idx], sw[idx] if sw is not None else None
        )
        for j in range(n_bins):
            wmse_s[j].append(wmse_b[j])
            share_s[j].append(share_b[j])

    out: dict[str, dict[str, object]] = {}
    for j, label in enumerate(labels):
        out[label] = {
            "count": int(counts[j]),
            "weighted_mse": _ci_from_samples(wmse_pt[j], wmse_s[j], ci),
            "score_share": _ci_from_samples(share_pt[j], share_s[j], ci),
        }
    return out


def bootstrap_score_delta(
    preds_a: Sequence[float] | np.ndarray,
    preds_b: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    genders: Sequence | np.ndarray,
    *,
    group_ids: Sequence | np.ndarray | None = None,
    unit: str = "row",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    clip: bool = True,
    female_value: str = "0.0",
    male_value: str = "1.0",
    high_occ_threshold: float = 0.40,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> dict[str, MetricCI]:
    """Paired bootstrap of ``metric(A) - metric(B)`` on the *same* resampled rows.

    Both runs must be evaluated on identical rows in identical order (guaranteed by the
    shared split snapshot). Because their per-row errors are correlated, the CI of the
    *difference* is far tighter than comparing two marginal CIs — a delta CI that
    excludes 0 means the two runs genuinely differ.
    """
    preds_a = np.asarray(preds_a, dtype=float).reshape(-1)
    preds_b = np.asarray(preds_b, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    genders = np.asarray(genders).reshape(-1)
    n = preds_a.shape[0]
    if not (n == preds_b.shape[0] == targets.shape[0] == genders.shape[0]):
        raise ValueError("preds_a, preds_b, targets and genders must have the same length")
    sw = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else None

    def _delta(idx):
        ma = _compute_metrics(
            preds_a[idx],
            targets[idx],
            genders[idx],
            clip=clip,
            female_value=female_value,
            male_value=male_value,
            high_occ_threshold=high_occ_threshold,
            sample_weight=sw[idx] if sw is not None else None,
        )
        mb = _compute_metrics(
            preds_b[idx],
            targets[idx],
            genders[idx],
            clip=clip,
            female_value=female_value,
            male_value=male_value,
            high_occ_threshold=high_occ_threshold,
            sample_weight=sw[idx] if sw is not None else None,
        )
        return {k: ma[k] - mb[k] for k in ma}

    point = _delta(np.arange(n))
    members = _group_members(group_ids, unit)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(int(n_boot)):
        idx = _resample_indices(rng, n, unit, None, members)
        d = _delta(idx)
        for k, v in d.items():
            samples[k].append(v)
    return {k: _ci_from_samples(point[k], samples[k], ci) for k in point}
