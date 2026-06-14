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
) -> dict[str, float]:
    """All scalar metrics for one (resampled) set of rows."""
    score = challenge_score(
        preds, targets, genders, female_value=female_value, male_value=male_value, clip=clip
    )
    out = {
        "score": score["score"],
        "err_female": score["err_female"],
        "err_male": score["err_male"],
        "gender_gap": score["gender_gap"],
        "err_mean": score["err_mean"],
    }
    # High-occlusion aggregate (pools everything above the threshold).
    hi = targets >= high_occ_threshold
    out["high_occ_err"] = (
        weighted_mse(preds[hi], targets[hi], clip=clip) if hi.any() else float("nan")
    )
    g = genders.astype(str)
    hi_f = hi & (g == str(female_value))
    hi_m = hi & (g == str(male_value))
    err_hf = weighted_mse(preds[hi_f], targets[hi_f], clip=clip) if hi_f.any() else float("nan")
    err_hm = weighted_mse(preds[hi_m], targets[hi_m], clip=clip) if hi_m.any() else float("nan")
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
) -> dict[str, MetricCI]:
    """Return point estimate + percentile CI for each challenge metric.

    Parameters mirror :func:`challenge_score`. ``unit="group"`` requires
    ``group_ids`` and resamples identity clusters rather than rows.
    """
    if unit not in {"row", "group"}:
        raise ValueError(f"unit must be 'row' or 'group', got {unit!r}")
    preds = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    genders = np.asarray(genders).reshape(-1)
    n = preds.shape[0]
    if not (n == targets.shape[0] == genders.shape[0]):
        raise ValueError("preds, targets and genders must have the same length")

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
        )
        for k, v in m.items():
            samples[k].append(v)

    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 * (1.0 - (1.0 - ci) / 2.0)
    out: dict[str, MetricCI] = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out[k] = MetricCI(point[k], float("nan"), float("nan"), float("nan"))
            continue
        out[k] = MetricCI(
            point=float(point[k]),
            lo=float(np.percentile(finite, lo_q)),
            hi=float(np.percentile(finite, hi_q)),
            std=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        )
    return out
