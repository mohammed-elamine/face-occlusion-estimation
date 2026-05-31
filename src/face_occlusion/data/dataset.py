"""FaceOcclusionDataset.

Returns dictionaries with image, target and metadata. We *keep* metadata
(gender, image_id, path) in each item because the official metric is
subgroup-aware and we want to be able to do error analysis downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .metadata import add_path_metadata


def _normalize_target_scale(values: pd.Series, scale: str) -> pd.Series:
    if scale == "unit":
        return values
    if scale == "percent":
        return values / 100.0
    # auto: challenge labels may be stored either as [0, 1] or percentages.
    if values.max() > 1.5:
        return values / 100.0
    return values


class FaceOcclusionDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        image_root: str | Path,
        transform=None,
        mode: str = "train",
        image_col: str = "filename",
        target_col: str = "FaceOcclusion",
        gender_col: str = "gender",
        id_col: str | None = None,
        target_scale: str = "auto",
    ) -> None:
        assert mode in {"train", "val", "test"}, f"Unknown mode {mode}"
        self.mode = mode
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_col = image_col
        self.target_col = target_col
        self.gender_col = gender_col
        self.id_col = id_col or image_col

        df = add_path_metadata(metadata.reset_index(drop=True), filename_col=image_col)
        if mode != "test":
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' missing in metadata.")
            if gender_col not in df.columns:
                raise ValueError(f"Gender column '{gender_col}' missing in metadata.")
            df[target_col] = _normalize_target_scale(df[target_col].astype(float), target_scale)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = str(row[self.image_col])
        path = self.image_root / rel_path
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            # Failing fast keeps bad image paths visible during validation/training.
            raise RuntimeError(f"Failed to load image '{path}': {exc}") from exc

        if self.transform is not None:
            image = self.transform(image)

        # Keep metadata in the batch so metrics and error analysis can group predictions.
        item: dict[str, Any] = {
            "image": image,
            "image_id": str(row[self.id_col]),
            "path": str(path),
            "filename": rel_path,
            "database": str(row["database"]),
            "source_subfolder": str(row["source_subfolder"]),
            "group_id": str(row["group_id"]),
            "face_id": int(row["face_id"]),
        }
        if self.mode != "test":
            item["gender"] = torch.tensor(float(row[self.gender_col]), dtype=torch.float32)
            item["target"] = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return item
