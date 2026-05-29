"""Stratified train/val split on gender x occlusion-bin.

We keep the official gender labels as-is (a few are known to be wrong, but
not enough to justify our own re-labelling), and stratify on a coarse
occlusion-bin x gender key so each fold sees the full difficulty range
for both subgroups.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _occlusion_bin(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    edges = np.asarray(bins, dtype=float)
    idx = np.digitize(values, edges[1:-1], right=False)
    return idx.astype(int)


def make_stratified_split(
    df: pd.DataFrame,
    target_col: str,
    gender_col: str,
    id_col: str,
    bins: Sequence[float],
    val_size: float = 0.2,
    seed: int = 42,
    min_per_stratum: int = 2,
) -> pd.DataFrame:
    """Return a DataFrame [id_col, split] with split in {train, val}."""

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe.")
    if gender_col not in df.columns:
        raise ValueError(f"Gender column '{gender_col}' not in dataframe.")

    targets = df[target_col].astype(float).to_numpy()
    if targets.max() > 1.5:  # auto normalize percent -> unit
        targets = targets / 100.0
    occ_bin = _occlusion_bin(targets, bins)
    gender = df[gender_col].astype(str).to_numpy()
    strat_key = np.array([f"{g}_{b}" for g, b in zip(gender, occ_bin)])

    # Merge rare strata into a fallback "rare" bucket to keep stratification valid.
    counts = pd.Series(strat_key).value_counts()
    rare = set(counts[counts < min_per_stratum].index.tolist())
    if rare:
        print(f"[split] Warning: merging {len(rare)} rare strata into '_rare_' fallback.")
        strat_key = np.array([k if k not in rare else "_rare_" for k in strat_key])

    indices = np.arange(len(df))
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=seed,
            stratify=strat_key,
        )
    except ValueError as exc:
        print(f"[split] Warning: stratified split failed ({exc}); falling back to random.")
        train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=seed)

    split_col = np.array(["train"] * len(df), dtype=object)
    split_col[val_idx] = "val"
    return pd.DataFrame({id_col: df[id_col].values, "split": split_col})


def save_split(split_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)


def load_split(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
