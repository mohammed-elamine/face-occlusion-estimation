"""Soft Gender × Occlusion Balanced Batch Sampler.

Builds each training batch from multiple gender × occlusion_bin strata so that
high-occlusion images get more exposure while both genders remain represented
across all occlusion ranges.  This reduces the risk of the model learning a
shortcut between gender-correlated visual cues and high occlusion labels.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GENDER_LABELS = {0: "female", 1: "male"}


def _bin_label(edges: Sequence[float], bin_idx: int) -> str:
    """Human-readable label for a bin index, e.g. '0.00_0.05'."""
    lo = edges[bin_idx]
    hi = edges[bin_idx + 1] if bin_idx + 1 < len(edges) else edges[-1]
    return f"{lo:.2f}_{hi:.2f}"


def _assign_bins(targets: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Assign each target value to a bin index (0-based)."""
    edges_arr = np.asarray(edges, dtype=float)
    idx = np.digitize(targets, edges_arr[1:-1], right=False)
    return idx.astype(int)


def _validate_bins(bins: Sequence[float]) -> None:
    if len(bins) < 2:
        raise ValueError(f"bins must have at least 2 edges, got {len(bins)}.")
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            raise ValueError(f"bins must be strictly increasing, got {bins}.")


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class GenderOcclusionBalancedBatchSampler(Sampler[list[int]]):
    """Batch sampler that softly balances gender × occlusion_bin strata.

    Each batch is constructed by repeatedly sampling a stratum according to
    computed stratum probabilities, then drawing one random index from that
    stratum (with replacement when a stratum is small).

    Parameters
    ----------
    targets : array-like
        Continuous occlusion targets in [0, 1].
    genders : array-like
        Gender labels (0 = female, 1 = male).
    batch_size : int
        Number of samples per batch.
    bins : sequence of float
        Occlusion bin edges, e.g. [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0].
    bin_weights : dict[str, float]
        Relative importance of each occlusion bin, keyed by bin label.
    gender_balance_strength : float
        Controls gender rebalancing inside each bin (0 = none, 1 = full).
    max_stratum_weight : float
        Cap on any single stratum weight to prevent extreme oversampling.
    min_stratum_size : int
        Warn when a stratum has fewer than this many samples.
    num_samples : int | None
        Total samples per epoch.  ``None`` means ``len(targets)``.
    drop_last : bool
        Whether to drop the last incomplete batch.
    seed : int
        Seed for the local random generator.
    """

    def __init__(
        self,
        targets: np.ndarray,
        genders: np.ndarray,
        batch_size: int,
        bins: Sequence[float],
        bin_weights: dict[str, float],
        gender_balance_strength: float = 0.5,
        max_stratum_weight: float = 8.0,
        min_stratum_size: int = 5,
        num_samples: int | None = None,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        targets = np.asarray(targets, dtype=float)
        genders = np.asarray(genders, dtype=float)

        _validate_bins(bins)
        self.bins = list(bins)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._num_samples = num_samples if num_samples is not None else len(targets)
        self._rng = np.random.default_rng(seed)

        # Validate gender values.
        unique_genders = set(np.unique(genders).tolist())
        invalid = unique_genders - {0.0, 1.0}
        if invalid:
            raise ValueError(
                f"Gender column contains invalid values {invalid}. "
                "Expected only 0 (female) and 1 (male)."
            )

        # Assign bins.
        bin_indices = _assign_bins(targets, bins)

        # Build strata: (gender, bin_idx) -> array of dataset indices.
        self._strata: dict[tuple[int, int], np.ndarray] = {}
        n_bins = len(bins) - 1
        for g in (0, 1):
            for b in range(n_bins):
                mask = (genders == g) & (bin_indices == b)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue
                self._strata[(g, b)] = indices

        if not self._strata:
            raise ValueError("No non-empty strata found. Check targets and genders.")

        # Count samples per bin (both genders combined) for gender correction.
        bin_counts: dict[int, int] = {}
        for (g, b), idx in self._strata.items():
            bin_counts[b] = bin_counts.get(b, 0) + len(idx)

        # Compute stratum weights.
        self._stratum_weights: dict[tuple[int, int], float] = {}
        for (g, b), idx in self._strata.items():
            label = _bin_label(bins, b)
            bw = bin_weights.get(label, 1.0)

            n_bin = bin_counts[b]
            n_stratum = len(idx)

            # Soft gender correction: (n_bin / n_stratum) ** strength.
            # When strength=0 this is 1.0 (no correction).
            # When strength=0.5 this is a square-root inverse-frequency correction.
            if gender_balance_strength > 0 and n_stratum > 0:
                gender_correction = (n_bin / n_stratum) ** gender_balance_strength
            else:
                gender_correction = 1.0

            weight = min(bw * gender_correction, max_stratum_weight)
            self._stratum_weights[(g, b)] = weight

            if n_stratum < min_stratum_size:
                warnings.warn(
                    f"Stratum (gender={g}, bin={label}) has only {n_stratum} samples "
                    f"(< min_stratum_size={min_stratum_size}). It may be oversampled.",
                    stacklevel=2,
                )

        # Normalize weights to probabilities.
        keys = list(self._stratum_weights.keys())
        weights = np.array([self._stratum_weights[k] for k in keys])
        probs = weights / weights.sum()
        self._stratum_keys = keys
        self._stratum_probs = probs

        # Store summary info for logging.
        self._summary = self._build_summary(
            bins, bin_weights, gender_balance_strength, max_stratum_weight, min_stratum_size, seed
        )

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        bins: Sequence[float],
        bin_weights: dict[str, float],
        gender_balance_strength: float,
        max_stratum_weight: float,
        min_stratum_size: int,
        seed: int,
    ) -> dict[str, Any]:
        strata_info = []
        for key, prob in zip(self._stratum_keys, self._stratum_probs):
            g, b = key
            label = _bin_label(bins, b)
            strata_info.append(
                {
                    "gender": int(g),
                    "gender_label": _GENDER_LABELS.get(g, str(g)),
                    "occlusion_bin": label,
                    "count": int(len(self._strata[key])),
                    "weight": round(float(self._stratum_weights[key]), 4),
                    "probability": round(float(prob), 6),
                }
            )
        return {
            "strategy": "gender_occlusion_balanced_batch",
            "batch_size": self.batch_size,
            "num_samples": self._num_samples,
            "num_batches": len(self),
            "bins": list(bins),
            "bin_weights": {str(k): v for k, v in bin_weights.items()},
            "gender_balance_strength": gender_balance_strength,
            "max_stratum_weight": max_stratum_weight,
            "min_stratum_size": min_stratum_size,
            "seed": seed,
            "strata": strata_info,
        }

    @property
    def summary(self) -> dict[str, Any]:
        return self._summary

    def log_summary(self) -> None:
        """Print a concise summary of the sampler configuration."""
        s = self._summary
        lines = [
            "Using GenderOcclusionBalancedBatchSampler",
            f"  batch_size: {s['batch_size']}",
            f"  num_samples: {s['num_samples']}",
            f"  num_batches: {s['num_batches']}",
            f"  bins: {s['bins']}",
            "  strata:",
        ]
        for st in s["strata"]:
            lines.append(
                f"    gender={st['gender']} ({st['gender_label']}), "
                f"bin={st['occlusion_bin']}, "
                f"count={st['count']}, "
                f"weight={st['weight']}, "
                f"prob={st['probability']:.4f}"
            )
        msg = "\n".join(lines)
        logger.info(msg)
        print(f"[sampler] {msg}")

    def save_summary(self, path: str | Path) -> None:
        """Write the sampler summary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._summary, f, indent=2)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        n = self._num_samples / self.batch_size
        return int(n) if self.drop_last else math.ceil(n)

    def __iter__(self) -> Iterator[list[int]]:
        num_full_batches = self._num_samples // self.batch_size
        remainder = self._num_samples % self.batch_size

        for _ in range(num_full_batches):
            yield self._sample_batch(self.batch_size)

        if not self.drop_last and remainder > 0:
            yield self._sample_batch(remainder)

    def _sample_batch(self, size: int) -> list[int]:
        # For each slot, pick a stratum then pick one sample from it.
        chosen_strata = self._rng.choice(len(self._stratum_keys), size=size, p=self._stratum_probs)
        indices: list[int] = []
        for si in chosen_strata:
            key = self._stratum_keys[si]
            pool = self._strata[key]
            idx = self._rng.choice(pool)
            indices.append(int(idx))
        self._rng.shuffle(indices)
        return indices


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------

_DEFAULT_BIN_WEIGHTS: dict[str, float] = {
    "0.00_0.05": 1.0,
    "0.05_0.10": 1.2,
    "0.10_0.20": 1.5,
    "0.20_0.40": 2.0,
    "0.40_0.60": 3.0,
    "0.60_1.00": 4.0,
}


def build_batch_sampler_from_config(
    df: pd.DataFrame,
    cfg: Any,
    batch_size: int,
) -> GenderOcclusionBalancedBatchSampler | None:
    """Build a batch sampler from the project config, or return None if disabled.

    Parameters
    ----------
    df : pd.DataFrame
        Training metadata with target and gender columns.
    cfg : Config
        Full project config (expects ``cfg.sampler`` section).
    batch_size : int
        Batch size for the DataLoader.

    Returns
    -------
    GenderOcclusionBalancedBatchSampler or None
    """
    sampler_cfg = cfg.get("sampler", None)
    if sampler_cfg is None or not bool(sampler_cfg.get("enabled", False)):
        return None

    strategy = sampler_cfg.get("strategy", "gender_occlusion_balanced_batch")
    if strategy != "gender_occlusion_balanced_batch":
        raise ValueError(f"Unknown sampler strategy '{strategy}'.")

    target_col = cfg.data.target_col
    gender_col = cfg.data.gender_col

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from training dataframe.")
    if gender_col not in df.columns:
        raise ValueError(f"Gender column '{gender_col}' missing from training dataframe.")

    targets = df[target_col].astype(float).to_numpy()
    # Normalize if stored as percentages.
    if targets.max() > 1.5:
        targets = targets / 100.0

    genders = df[gender_col].astype(float).to_numpy()

    bins = list(sampler_cfg.get("bins", [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]))

    raw_bw = sampler_cfg.get("bin_weights", None)
    if raw_bw is not None:
        bin_weights = {str(k): float(v) for k, v in raw_bw.items()}
    else:
        bin_weights = _DEFAULT_BIN_WEIGHTS

    num_samples_raw = sampler_cfg.get("num_samples", None)
    num_samples = int(num_samples_raw) if num_samples_raw is not None else None

    sampler = GenderOcclusionBalancedBatchSampler(
        targets=targets,
        genders=genders,
        batch_size=batch_size,
        bins=bins,
        bin_weights=bin_weights,
        gender_balance_strength=float(sampler_cfg.get("gender_balance_strength", 0.5)),
        max_stratum_weight=float(sampler_cfg.get("max_stratum_weight", 8.0)),
        min_stratum_size=int(sampler_cfg.get("min_stratum_size", 5)),
        num_samples=num_samples,
        drop_last=True,
        seed=int(sampler_cfg.get("seed", cfg.get("project", {}).get("seed", 42))),
    )
    return sampler
