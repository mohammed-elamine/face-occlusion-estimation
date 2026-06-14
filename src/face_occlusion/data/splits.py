"""Train/validation split helpers.

The default split is row-level and leaderboard-oriented. A group-level split is
also available to evaluate robustness to unseen identity-like groups.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .metadata import add_path_metadata

SPLIT_METADATA_COLS = [
    "database",
    "source_subfolder",
    "group_id",
    "face_id",
    "occlusion_bin",
    "gender",
]


def _normalize_targets(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    if values.max() > 1.5:
        return values / 100.0
    return values


def _occlusion_bin(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    edges = np.asarray(bins, dtype=float)
    # np.digitize maps each target to a coarse difficulty bucket.
    idx = np.digitize(values, edges[1:-1], right=False)
    return idx.astype(int)


def _prepare_split_frame(
    df: pd.DataFrame,
    target_col: str,
    gender_col: str,
    id_col: str,
    bins: Sequence[float],
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe.")
    if gender_col not in df.columns:
        raise ValueError(f"Gender column '{gender_col}' not in dataframe.")
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not in dataframe.")

    out = add_path_metadata(df, filename_col=id_col)
    targets = _normalize_targets(out[target_col])
    out["occlusion_bin"] = _occlusion_bin(targets.to_numpy(), bins)
    out["gender"] = out[gender_col]
    return out


def _strat_key(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df[list(cols)].astype(str).agg("_".join, axis=1).to_numpy()


def _merge_rare_strata(strat_key: np.ndarray, min_per_stratum: int) -> np.ndarray:
    counts = pd.Series(strat_key).value_counts()
    rare = set(counts[counts < min_per_stratum].index.tolist())
    if not rare:
        return strat_key
    print(f"[split] Warning: merging {len(rare)} rare strata into '_rare_' fallback.")
    return np.array([key if key not in rare else "_rare_" for key in strat_key])


def _row_split(
    df: pd.DataFrame,
    stratify_by: Sequence[str],
    val_size: float,
    seed: int,
    min_per_stratum: int,
) -> np.ndarray:
    indices = np.arange(len(df))
    fallback_cols = [col for col in ("gender", "occlusion_bin") if col in df.columns]
    candidate_cols = [list(stratify_by), fallback_cols, []]

    for cols in candidate_cols:
        stratify = None
        if cols:
            stratify = _merge_rare_strata(_strat_key(df, cols), min_per_stratum)
        try:
            _, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=stratify,
            )
            if cols != list(stratify_by):
                print(
                    f"[split] Warning: using fallback row stratification columns: {cols or 'none'}"
                )
            split_col = np.array(["train"] * len(df), dtype=object)
            split_col[val_idx] = "val"
            return split_col
        except ValueError as exc:
            print(f"[split] Warning: row stratification with {cols or 'none'} failed ({exc}).")

    raise RuntimeError("Could not create row-level split.")


def _group_split(
    df: pd.DataFrame,
    stratify_by: Sequence[str],
    group_col: str,
    val_size: float,
    seed: int,
    min_per_stratum: int,
) -> np.ndarray:
    groups = df[group_col].astype(str).to_numpy()
    n_splits = max(2, int(round(1.0 / val_size)))
    fallback_cols = [col for col in ("gender", "occlusion_bin") if col in df.columns]
    candidate_cols = [list(stratify_by), fallback_cols, []]

    for cols in candidate_cols:
        y = np.zeros(len(df), dtype=int)
        if cols:
            y = pd.factorize(_merge_rare_strata(_strat_key(df, cols), min_per_stratum))[0]
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            _, val_idx = next(splitter.split(np.zeros(len(df)), y=y, groups=groups))
            if cols != list(stratify_by):
                print(
                    "[split] Warning: using fallback group stratification columns: "
                    f"{cols or 'none'}"
                )
            split_col = np.array(["train"] * len(df), dtype=object)
            split_col[val_idx] = "val"
            return split_col
        except ValueError as exc:
            print(f"[split] Warning: group stratification with {cols or 'none'} failed ({exc}).")

    # Last resort: split unique groups randomly while preserving group isolation.
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(pd.unique(groups)))
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(len(unique_groups) * val_size)))
    val_groups = set(unique_groups[:n_val])
    split_col = np.array(
        ["val" if group in val_groups else "train" for group in groups],
        dtype=object,
    )
    print("[split] Warning: using random group split without stratification.")
    return split_col


def make_stratified_split(
    df: pd.DataFrame,
    target_col: str,
    gender_col: str,
    id_col: str,
    bins: Sequence[float],
    val_size: float = 0.2,
    seed: int = 42,
    min_per_stratum: int = 2,
    strategy: str = "row_stratified",
    stratify_by: Sequence[str] | None = None,
    group_col: str = "group_id",
) -> pd.DataFrame:
    """Return a split DataFrame with ids, split labels and useful diagnostics metadata."""

    strategy_aliases = {"gender_occlusion_stratified": "row_stratified"}
    strategy = strategy_aliases.get(strategy, strategy)
    if strategy not in {"row_stratified", "group_stratified"}:
        raise ValueError(f"Unknown split strategy '{strategy}'.")

    split_df = _prepare_split_frame(df, target_col, gender_col, id_col, bins)
    stratify_by = list(stratify_by or ["gender", "occlusion_bin", "database"])

    if strategy == "row_stratified":
        split_df["split"] = _row_split(split_df, stratify_by, val_size, seed, min_per_stratum)
    else:
        split_df["split"] = _group_split(
            split_df,
            stratify_by=stratify_by,
            group_col=group_col,
            val_size=val_size,
            seed=seed,
            min_per_stratum=min_per_stratum,
        )

    cols = [id_col, "split", *SPLIT_METADATA_COLS]
    return split_df[cols]


def save_split(split_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)


def load_split(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
