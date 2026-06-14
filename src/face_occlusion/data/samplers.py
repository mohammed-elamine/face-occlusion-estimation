"""Exposure-capped soft gender x occlusion balanced batch sampler.

The sampler builds each epoch from gender x occlusion-bin strata. For every
stratum ``s`` we compute a target number of draws that mixes the natural
distribution with a balanced one. A *hard* per-image repeat cap then bounds how
many times any single image can appear in an epoch, so that tiny strata cannot
be amplified into label noise. Strata that hit the cap have their leftover
budget redistributed to the remaining strata with capacity.

This replaces the older slot-by-slot sampler, which could repeat the same image
hundreds of times per epoch when a stratum was very small relative to the
configured weight.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


_GENDER_LABELS = {0: "female", 1: "male"}
_TINY_STRATUM_POLICIES = ("cap", "warn")

# Softer defaults than the historical schema. These are intentionally gentle: the
# sampler should not solve high-occlusion rarity alone -- later synthetic
# ranking and triplet learning provide the remaining diversity.
_DEFAULT_BIN_WEIGHTS: dict[str, float] = {
    "0.00_0.05": 1.0,
    "0.05_0.10": 1.1,
    "0.10_0.20": 1.2,
    "0.20_0.40": 1.5,
    "0.40_0.60": 2.0,
    "0.60_1.00": 2.5,
}


def _bin_label(edges: Sequence[float], idx: int) -> str:
    return f"{edges[idx]:.2f}_{edges[idx + 1]:.2f}"


def _assign_bins(targets: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Assign each target to a bin index in ``[0, len(edges) - 2]``."""
    edges_arr = np.asarray(edges, dtype=float)
    # np.digitize with right=False produces 1..len(edges); subtract 1 to get
    # 0..len(edges)-1. The boundary value ``edges[-1]`` (typically 1.0) lands in
    # bin len(edges)-1, which is invalid -- clip it back into the last bin.
    bins = np.digitize(targets, edges_arr[1:-1], right=False)
    return np.clip(bins, 0, len(edges_arr) - 2)


def _validate_bins(bins: Sequence[float]) -> None:
    if len(bins) < 2:
        raise ValueError("`bins` must contain at least 2 edges")
    arr = np.asarray(bins, dtype=float)
    if np.any(np.diff(arr) <= 0):
        raise ValueError("`bins` must be strictly increasing")


def _validate_genders(genders: np.ndarray) -> None:
    unique = np.unique(genders)
    invalid = [g for g in unique if int(g) not in _GENDER_LABELS]
    if invalid:
        raise ValueError(
            f"Genders contain invalid values {invalid}; expected only 0 (female) and 1 (male)"
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class GenderOcclusionBalancedBatchSampler(Sampler[list[int]]):
    """Soft gender x occlusion balanced batch sampler with a hard exposure cap.

    Each epoch:

    1. Group samples into ``(gender, occlusion_bin)`` strata.
    2. Compute per-stratum sampling weights using bin weights and an
       inverse-frequency gender correction; optionally damp the weights of
       tiny strata via ``size_aware_weighting``.
    3. Build a *balanced* probability distribution from clipped weights and
       mix it with the *natural* distribution via ``balance_strength``.
    4. Translate per-stratum probabilities into integer draw counts, applying
       ``max_repeats_per_image`` as a hard cap and redistributing any leftover
       budget to non-saturated strata.
    5. For each stratum draw the prescribed number of indices using a
       ``repeat-then-shuffle`` pool, concatenate across strata, shuffle, and
       yield contiguous batches.
    """

    def __init__(
        self,
        targets: np.ndarray,
        genders: np.ndarray,
        batch_size: int,
        bins: Sequence[float],
        bin_weights: dict[str, float],
        *,
        gender_balance_strength: float = 0.5,
        balance_strength: float = 0.3,
        max_stratum_weight: float = 8.0,
        min_stratum_size: int = 5,
        size_aware_weighting: bool = True,
        reliable_stratum_size: int = 20,
        max_repeats_per_image: int = 10,
        tiny_stratum_policy: str = "cap",
        num_samples: int | None = None,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(None)

        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive")
        if not 0.0 <= float(balance_strength) <= 1.0:
            raise ValueError("`balance_strength` must lie in [0, 1]")
        if not 0.0 <= float(gender_balance_strength) <= 1.0:
            raise ValueError("`gender_balance_strength` must lie in [0, 1]")
        if max_stratum_weight <= 0:
            raise ValueError("`max_stratum_weight` must be positive")
        if reliable_stratum_size <= 0:
            raise ValueError("`reliable_stratum_size` must be positive")
        if max_repeats_per_image <= 0:
            raise ValueError("`max_repeats_per_image` must be positive")
        if tiny_stratum_policy not in _TINY_STRATUM_POLICIES:
            raise ValueError(
                f"`tiny_stratum_policy` must be one of {_TINY_STRATUM_POLICIES}, "
                f"got {tiny_stratum_policy!r}"
            )

        _validate_bins(bins)
        targets = np.asarray(targets, dtype=float).reshape(-1)
        genders = np.asarray(genders, dtype=float).reshape(-1)
        if targets.shape != genders.shape:
            raise ValueError("`targets` and `genders` must have the same length")
        _validate_genders(genders)

        self.batch_size = int(batch_size)
        self.bins = [float(b) for b in bins]
        self.bin_weights = dict(bin_weights)
        self.gender_balance_strength = float(gender_balance_strength)
        self.balance_strength = float(balance_strength)
        self.max_stratum_weight = float(max_stratum_weight)
        self.min_stratum_size = int(min_stratum_size)
        self.size_aware_weighting = bool(size_aware_weighting)
        self.reliable_stratum_size = int(reliable_stratum_size)
        self.max_repeats_per_image = int(max_repeats_per_image)
        self.tiny_stratum_policy = str(tiny_stratum_policy)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        n_total = int(targets.shape[0])
        self.num_samples_requested = int(num_samples) if num_samples is not None else n_total
        if self.num_samples_requested <= 0:
            raise ValueError("`num_samples` must be positive")

        # Build strata.
        bin_idx = _assign_bins(targets, self.bins)
        self._strata: dict[tuple[int, int], np.ndarray] = {}
        for g in (0, 1):
            for b in range(len(self.bins) - 1):
                mask = (genders.astype(int) == g) & (bin_idx == b)
                if mask.any():
                    self._strata[(g, b)] = np.flatnonzero(mask)

        # Per-bin totals (used by gender correction).
        bin_totals = np.zeros(len(self.bins) - 1, dtype=int)
        for (_, b), idxs in self._strata.items():
            bin_totals[b] += len(idxs)

        self._stratum_keys: list[tuple[int, int]] = sorted(self._strata.keys())
        if not self._stratum_keys:
            raise ValueError("No non-empty strata found; cannot build sampler")

        # Compute weights and probabilities.
        natural = np.array([len(self._strata[k]) for k in self._stratum_keys], dtype=float)
        natural_prob = natural / natural.sum()

        raw_weights = np.zeros(len(self._stratum_keys), dtype=float)
        effective_weights = np.zeros_like(raw_weights)
        for i, (g, b) in enumerate(self._stratum_keys):
            n_s = float(len(self._strata[(g, b)]))
            n_b = float(bin_totals[b])
            label = _bin_label(self.bins, b)
            w_bin = float(self.bin_weights.get(label, 1.0))
            gender_correction = (n_b / max(n_s, 1.0)) ** self.gender_balance_strength
            a_raw = w_bin * gender_correction
            if self.size_aware_weighting:
                reliability = min(1.0, n_s / float(self.reliable_stratum_size))
                a_safe = 1.0 + reliability * (a_raw - 1.0)
            else:
                a_safe = a_raw
            a_clip = min(a_safe, self.max_stratum_weight)
            raw_weights[i] = a_raw
            effective_weights[i] = a_clip

        # Per-stratum (not per-image) balanced probability: tiny strata can be
        # boosted, and the repeat cap below prevents that boost from turning
        # into pathological over-exposure of a handful of images.
        balanced_prob = effective_weights / effective_weights.sum()

        alpha = self.balance_strength
        final_prob = (1.0 - alpha) * natural_prob + alpha * balanced_prob
        final_prob = final_prob / final_prob.sum()

        # Cap per-stratum draws by ``n_s * max_repeats_per_image``.
        caps = (natural * self.max_repeats_per_image).astype(int)
        if self.tiny_stratum_policy == "warn":
            effective_caps = np.full_like(caps, fill_value=10 * self.num_samples_requested)
        else:
            effective_caps = caps

        desired = final_prob * float(self.num_samples_requested)
        draws_int, leftover = _allocate_with_caps(
            final_prob, effective_caps, self.num_samples_requested
        )
        if leftover > 0:
            warnings.warn(
                f"Sampler could not place {leftover} samples this epoch: every "
                f"stratum already at its repeat cap. Effective epoch size is "
                f"{int(draws_int.sum())} instead of {self.num_samples_requested}.",
                UserWarning,
                stacklevel=2,
            )

        self._draws_per_stratum: dict[tuple[int, int], int] = dict(
            zip(self._stratum_keys, (int(x) for x in draws_int))
        )
        self.num_samples_actual = int(draws_int.sum())

        # Build the human-readable summary and emit warnings for small strata.
        self._summary = self._build_summary(
            natural=natural,
            natural_prob=natural_prob,
            raw_weights=raw_weights,
            effective_weights=effective_weights,
            balanced_prob=balanced_prob,
            final_prob=final_prob,
            desired=desired,
            draws_int=draws_int,
            caps=caps,
        )
        self._maybe_warn_small_strata()

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        n = self.num_samples_actual
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        pool: list[int] = []
        for key, count in self._draws_per_stratum.items():
            if count <= 0:
                continue
            idxs = self._strata[key]
            n_s = len(idxs)
            # Repeat each image at most ``max_repeats_per_image`` times, shuffle,
            # then take ``count``. ``count`` <= n_s * max_repeats_per_image by
            # construction so this slice always succeeds.
            reps = min(self.max_repeats_per_image, int(np.ceil(count / max(n_s, 1))))
            repeated = np.tile(idxs, reps)
            self._rng.shuffle(repeated)
            pool.extend(int(x) for x in repeated[:count])

        pool_arr = np.asarray(pool, dtype=int)
        self._rng.shuffle(pool_arr)

        n = len(pool_arr)
        bs = self.batch_size
        n_full = n // bs
        for i in range(n_full):
            yield pool_arr[i * bs : (i + 1) * bs].tolist()
        if not self.drop_last and n % bs:
            yield pool_arr[n_full * bs :].tolist()

    # ------------------------------------------------------------------
    # Summary & diagnostics
    # ------------------------------------------------------------------
    def _build_summary(
        self,
        *,
        natural: np.ndarray,
        natural_prob: np.ndarray,
        raw_weights: np.ndarray,
        effective_weights: np.ndarray,
        balanced_prob: np.ndarray,
        final_prob: np.ndarray,
        desired: np.ndarray,
        draws_int: np.ndarray,
        caps: np.ndarray,
    ) -> dict[str, Any]:
        strata_summary: list[dict[str, Any]] = []
        for i, (g, b) in enumerate(self._stratum_keys):
            n_s = int(natural[i])
            n_draws = int(draws_int[i])
            was_capped = bool(desired[i] > caps[i] + 1e-6)
            expected_reps = n_draws / max(n_s, 1)
            strata_summary.append(
                {
                    "gender": int(g),
                    "gender_label": _GENDER_LABELS[int(g)],
                    "occlusion_bin": _bin_label(self.bins, b),
                    "count": n_s,
                    "natural_prob": float(natural_prob[i]),
                    "balanced_prob": float(balanced_prob[i]),
                    "final_prob": float(final_prob[i]),
                    "raw_weight": float(raw_weights[i]),
                    "effective_weight": float(effective_weights[i]),
                    "expected_draws_before_cap": float(desired[i]),
                    "draws_after_cap": n_draws,
                    "expected_repeats_per_image": float(expected_reps),
                    "was_capped": was_capped,
                    # Probability sample appears at least once (under sampling
                    # with replacement) -- handy for spotting underexposed strata.
                    "p_seen_at_least_once": float(
                        1.0 - (1.0 - 1.0 / max(n_s, 1)) ** n_draws if n_s > 0 else 0.0
                    ),
                }
            )
        max_reps = max((s["expected_repeats_per_image"] for s in strata_summary), default=0.0)
        n_capped = sum(1 for s in strata_summary if s["was_capped"])
        return {
            "strategy": "gender_occlusion_balanced_batch",
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "balance_strength": self.balance_strength,
            "gender_balance_strength": self.gender_balance_strength,
            "max_stratum_weight": self.max_stratum_weight,
            "size_aware_weighting": self.size_aware_weighting,
            "reliable_stratum_size": self.reliable_stratum_size,
            "max_repeats_per_image": self.max_repeats_per_image,
            "tiny_stratum_policy": self.tiny_stratum_policy,
            "min_stratum_size": self.min_stratum_size,
            "bins": list(self.bins),
            "bin_weights": dict(self.bin_weights),
            "num_samples_requested": int(self.num_samples_requested),
            "num_samples_actual": int(self.num_samples_actual),
            "num_batches": int(self.__len__()),
            "max_expected_repeats_per_image": float(max_reps),
            "num_capped_strata": int(n_capped),
            "seed": self.seed,
            "strata": strata_summary,
        }

    def _maybe_warn_small_strata(self) -> None:
        for s in self._summary["strata"]:
            if s["was_capped"]:
                warnings.warn(
                    f"Stratum (gender={s['gender']} [{s['gender_label']}], "
                    f"bin={s['occlusion_bin']}) has only {s['count']} samples. "
                    f"Desired draws before cap: {s['expected_draws_before_cap']:.1f}. "
                    f"Capped to {s['draws_after_cap']} draws using "
                    f"max_repeats_per_image={self.max_repeats_per_image}.",
                    UserWarning,
                    stacklevel=3,
                )
            elif s["count"] < self.min_stratum_size and s["draws_after_cap"] > 0:
                warnings.warn(
                    f"Stratum (gender={s['gender']} [{s['gender_label']}], "
                    f"bin={s['occlusion_bin']}) has only {s['count']} samples "
                    f"(< min_stratum_size={self.min_stratum_size}). "
                    f"Drawing {s['draws_after_cap']} times this epoch.",
                    UserWarning,
                    stacklevel=3,
                )

    @property
    def summary(self) -> dict[str, Any]:
        return self._summary

    def log_summary(self) -> None:
        s = self._summary
        logger.info(
            "GenderOcclusionBalancedBatchSampler: "
            "%d strata, %d samples/epoch (requested %d), %d batches "
            "(balance=%.2f, gender=%.2f, max_reps/img=%d, capped_strata=%d, "
            "max_exp_reps=%.2f)",
            len(s["strata"]),
            s["num_samples_actual"],
            s["num_samples_requested"],
            s["num_batches"],
            s["balance_strength"],
            s["gender_balance_strength"],
            s["max_repeats_per_image"],
            s["num_capped_strata"],
            s["max_expected_repeats_per_image"],
        )
        for st in s["strata"]:
            logger.info(
                "  gender=%s bin=%s n=%4d nat=%.3f bal=%.3f final=%.3f "
                "raw_w=%.2f eff_w=%.2f draws=%5d reps/img=%.2f capped=%s",
                st["gender_label"],
                st["occlusion_bin"],
                st["count"],
                st["natural_prob"],
                st["balanced_prob"],
                st["final_prob"],
                st["raw_weight"],
                st["effective_weight"],
                st["draws_after_cap"],
                st["expected_repeats_per_image"],
                st["was_capped"],
            )

    def save_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._summary, f, indent=2)


# ---------------------------------------------------------------------------
# Cap-aware allocation
# ---------------------------------------------------------------------------


def _allocate_with_caps(
    probs: np.ndarray,
    caps: np.ndarray,
    total: int,
) -> tuple[np.ndarray, int]:
    """Allocate ``total`` integer draws across strata respecting per-stratum caps.

    Returns ``(draws, leftover)`` where ``leftover`` is the number of samples
    that could not be placed because every stratum reached its cap.
    """
    probs = np.asarray(probs, dtype=float)
    caps = np.asarray(caps, dtype=int)
    draws = np.zeros(len(probs), dtype=int)
    remaining = int(total)

    for _ in range(len(probs) + 2):  # bounded by number of saturation rounds
        if remaining <= 0:
            break
        active = draws < caps
        if not active.any():
            break
        p_active = np.where(active, probs, 0.0)
        s = p_active.sum()
        if s <= 0:
            break
        p_active = p_active / s

        desired_extra = remaining * p_active  # float, this round only
        room = (caps - draws).astype(float)
        applied = np.minimum(desired_extra, room)
        new_int = np.floor(applied).astype(int)
        frac = applied - new_int
        # Distribute the rounding slack to strata with largest fractional
        # remainder, deterministically and respecting room.
        slack = int(remaining - new_int.sum())
        if slack > 0:
            order = np.argsort(-frac, kind="stable")
            for k in order:
                if slack <= 0:
                    break
                if new_int[k] < int(room[k]):
                    new_int[k] += 1
                    slack -= 1
        if new_int.sum() == 0:
            break
        draws += new_int
        remaining -= int(new_int.sum())

    return draws, max(remaining, 0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_batch_sampler_from_config(
    df: pd.DataFrame,
    cfg: Any,
    batch_size: int,
) -> GenderOcclusionBalancedBatchSampler | None:
    """Build a sampler from a config object, or return ``None`` if disabled."""
    sampler_cfg = cfg.get("sampler", None) if hasattr(cfg, "get") else getattr(cfg, "sampler", None)
    if sampler_cfg is None:
        return None
    enabled = (
        bool(sampler_cfg.get("enabled", False))
        if hasattr(sampler_cfg, "get")
        else bool(getattr(sampler_cfg, "enabled", False))
    )
    if not enabled:
        return None

    def _get(key: str, default: Any = None) -> Any:
        if hasattr(sampler_cfg, "get"):
            return sampler_cfg.get(key, default)
        return getattr(sampler_cfg, key, default)

    strategy = _get("strategy", "gender_occlusion_balanced_batch")
    if strategy != "gender_occlusion_balanced_batch":
        raise ValueError(f"Unknown sampler strategy: {strategy!r}")

    bins = _get("bins", [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0])
    bin_weights = _get("bin_weights", _DEFAULT_BIN_WEIGHTS)

    # Column names come from the dataset config (cfg.data.*). The sampler block
    # may override them via ``sampler.target_col`` / ``sampler.gender_col`` for
    # ad-hoc experiments, but normally they mirror what the dataset uses.
    data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else getattr(cfg, "data", {})

    def _data_get(key: str, default: Any) -> Any:
        if hasattr(data_cfg, "get"):
            return data_cfg.get(key, default)
        return getattr(data_cfg, key, default)

    target_col = str(_get("target_col", _data_get("target_col", "face_occluded")))
    gender_col = str(_get("gender_col", _data_get("gender_col", "gender")))

    for col, key in ((target_col, "data.target_col"), (gender_col, "data.gender_col")):
        if col not in df.columns:
            raise KeyError(
                f"Sampler requires column {col!r} (from cfg.{key}) but it is not "
                f"present in the training dataframe. Available columns: "
                f"{sorted(df.columns)}"
            )

    targets = df[target_col].to_numpy(dtype=float)
    genders = df[gender_col].to_numpy(dtype=float)

    return GenderOcclusionBalancedBatchSampler(
        targets=targets,
        genders=genders,
        batch_size=int(batch_size),
        bins=bins,
        bin_weights=dict(bin_weights),
        gender_balance_strength=float(_get("gender_balance_strength", 0.5)),
        balance_strength=float(_get("balance_strength", 0.3)),
        max_stratum_weight=float(_get("max_stratum_weight", 8.0)),
        min_stratum_size=int(_get("min_stratum_size", 5)),
        size_aware_weighting=bool(_get("size_aware_weighting", True)),
        reliable_stratum_size=int(_get("reliable_stratum_size", 20)),
        max_repeats_per_image=int(_get("max_repeats_per_image", 10)),
        tiny_stratum_policy=str(_get("tiny_stratum_policy", "cap")),
        num_samples=_get("num_samples", None),
        drop_last=bool(_get("drop_last", False)),
        seed=int(_get("seed", 42)),
    )
