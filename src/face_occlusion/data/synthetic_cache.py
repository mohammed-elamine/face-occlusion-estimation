"""Offline synthetic-occlusion cache: schema, anchor selection, manifest IO.

Synthetic ranking views are slow to generate (MediaPipe + acceptance-rejection),
so they are precomputed once into a cache and loaded like normal images at
training time. This module holds the pieces shared between the offline builder
(``scripts/data/build_synthetic_cache.py``) and the cache-backed dataset:

* the manifest schema,
* balanced ``occlusion_bin x gender`` anchor selection,
* deterministic per-anchor view filenames,
* a manifest loader.

The manifest maps a real training row (by its id column, e.g. ``filename``) to
three on-disk views — ``clean`` (the un-augmented resized original), ``mild``
and ``strong`` — plus the severity proxies and provenance. Only rows whose
MediaPipe pair was valid are written, so the dataset can attach views by a
simple manifest join.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Columns written to manifest.csv. ``id`` joins back to the training row.
MANIFEST_COLUMNS = (
    "id",
    "occlusion_bin",
    "gender",
    "clean_path",
    "mild_path",
    "strong_path",
    "mask_path",
    "mild_severity",
    "strong_severity",
    "mild_occluder_type",
    "strong_occluder_type",
)

MANIFEST_FILENAME = "manifest.csv"
VIEWS_DIRNAME = "views"
MASKS_DIRNAME = "masks"


def view_filenames(index: int) -> dict[str, str]:
    """Deterministic relative view paths for the ``index``-th cached anchor."""
    stem = f"{VIEWS_DIRNAME}/{index:06d}"
    return {
        "clean": f"{stem}_clean.webp",
        "mild": f"{stem}_mild.webp",
        "strong": f"{stem}_strong.webp",
    }


def mask_filename(index: int) -> str:
    """Deterministic relative face-mask path for the ``index``-th cached anchor."""
    return f"{MASKS_DIRNAME}/{index:06d}_facemask.png"


def select_balanced_anchors(
    df: pd.DataFrame,
    *,
    bin_col: str,
    gender_col: str,
    max_per_bin_gender: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample up to ``max_per_bin_gender`` rows from each occlusion_bin x gender cell.

    Caps over-represented cells (low occlusion) while keeping every row of rare
    cells (high occlusion). This balances the synthetic ranking signal across
    both occlusion severity and gender, which matters because MediaPipe success
    is gender-skewed on high-occlusion faces.
    """
    if max_per_bin_gender <= 0:
        raise ValueError("max_per_bin_gender must be positive")
    parts: list[pd.DataFrame] = []
    for _, cell in df.groupby([bin_col, gender_col], dropna=False):
        if len(cell) <= max_per_bin_gender:
            parts.append(cell)
        else:
            take = rng.choice(cell.index.to_numpy(), size=max_per_bin_gender, replace=False)
            parts.append(cell.loc[take])
    out = pd.concat(parts).sort_index()
    return out


def coverage_table(manifest: pd.DataFrame, bin_order: Sequence[str] | None = None) -> pd.DataFrame:
    """Count cached valid pairs per occlusion_bin x gender."""
    if manifest.empty:
        return manifest
    table = (
        manifest.groupby(["occlusion_bin", "gender"], dropna=False).size().reset_index(name="count")
    )
    if bin_order is not None:
        order = {b: i for i, b in enumerate(bin_order)}
        table["_o"] = table["occlusion_bin"].map(lambda b: order.get(str(b), len(order)))
        table = table.sort_values(["_o", "gender"]).drop(columns="_o").reset_index(drop=True)
    return table


def load_cache_manifest(cache_dir: str | Path) -> pd.DataFrame:
    """Load a cache manifest; raise if it is missing or malformed."""
    cache_dir = Path(cache_dir)
    path = cache_dir / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No synthetic cache manifest at {path}")
    manifest = pd.read_csv(path)
    required = ("id", "clean_path", "mild_path", "strong_path")
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(f"Cache manifest {path} is missing columns {missing}")
    return manifest


class SyntheticCache:
    """Lookup over a precomputed synthetic-view cache, keyed by training id.

    Holds only the manifest in memory; image bytes are read lazily by the
    dataset (which owns the view transform). ``lookup`` returns absolute view
    paths + severities for an id, or ``None`` when the id was not cached (e.g.
    its MediaPipe pair was invalid), so the dataset can fall back cleanly.
    """

    def __init__(self, cache_dir: str | Path, manifest: pd.DataFrame | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        manifest = manifest if manifest is not None else load_cache_manifest(self.cache_dir)
        self._rows: dict[str, dict] = {str(r["id"]): r for r in manifest.to_dict(orient="records")}

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, sample_id: object) -> bool:
        return str(sample_id) in self._rows

    def lookup(self, sample_id: object) -> dict | None:
        """Absolute paths + severities for ``sample_id`` (or ``None`` if absent)."""
        row = self._rows.get(str(sample_id))
        if row is None:
            return None
        return {
            "clean_path": self.cache_dir / row["clean_path"],
            "mild_path": self.cache_dir / row["mild_path"],
            "strong_path": self.cache_dir / row["strong_path"],
            "mild_severity": float(row.get("mild_severity", float("nan"))),
            "strong_severity": float(row.get("strong_severity", float("nan"))),
        }

    def load_mask(self, sample_id: object) -> np.ndarray | None:
        """Boolean face mask for ``sample_id`` (or ``None`` if absent/unsaved).

        Used by background augmentation. Returns ``None`` when the id is not
        cached or the cache was built without masks, so callers no-op safely.
        """
        row = self._rows.get(str(sample_id))
        if row is None:
            return None
        rel = row.get("mask_path")
        if rel is None or (isinstance(rel, float) and np.isnan(rel)):
            return None
        mask_path = self.cache_dir / rel
        if not mask_path.exists():
            return None
        with Image.open(mask_path) as m:
            return np.asarray(m.convert("L")) > 127
