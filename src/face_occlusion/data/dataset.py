"""FaceOcclusionDataset.

Returns dictionaries with image, target and metadata. We *keep* metadata
(gender, image_id, path) in each item because the official metric is
subgroup-aware and we want to be able to do error analysis downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .metadata import add_path_metadata
from .normalize import normalize_target
from .synthetic_cache import SyntheticCache
from .synthetic_occlusion import SyntheticOcclusionGenerator


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
        synthetic_generator: SyntheticOcclusionGenerator | None = None,
        synthetic_view_transform=None,
        synthetic_target_size: int | None = None,
        synthetic_seed: int = 42,
        synthetic_cache: SyntheticCache | None = None,
        background_augment=None,
    ) -> None:
        assert mode in {"train", "val", "test"}, f"Unknown mode {mode}"
        self.mode = mode
        self.image_root = Path(image_root)
        self.transform = transform
        # Label-preserving background augmentation (only on train images).
        self.background_augment = background_augment if mode == "train" else None
        self.image_col = image_col
        self.target_col = target_col
        self.gender_col = gender_col
        self.id_col = id_col or image_col
        # Synthetic occlusion is optional and only active for train splits;
        # disable it by construction in val/test to keep evaluation clean. The
        # precomputed cache is the default source; the on-the-fly generator is
        # kept mainly for audits. Cache takes precedence when both are set.
        self.synthetic_generator = synthetic_generator if mode == "train" else None
        self.synthetic_cache = synthetic_cache if mode == "train" else None
        self.synthetic_view_transform = synthetic_view_transform
        self.synthetic_target_size = synthetic_target_size
        self.synthetic_seed = int(synthetic_seed)

        df = add_path_metadata(metadata.reset_index(drop=True), filename_col=image_col)
        if mode != "test":
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' missing in metadata.")
            if gender_col not in df.columns:
                raise ValueError(f"Gender column '{gender_col}' missing in metadata.")
            df[target_col] = normalize_target(df[target_col], target_scale)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = str(row[self.image_col])
        # cfg.data.image_root points to the crop root; CSV paths are relative to it.
        path = self.image_root / rel_path
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            # Failing fast keeps bad image paths visible during validation/training.
            raise RuntimeError(f"Failed to load image '{path}': {exc}") from exc

        # Label-preserving background augmentation: perturb only non-face pixels,
        # so the occlusion label stays valid. No-op when the id has no cached mask.
        if self.background_augment is not None:
            image = self.background_augment(image, str(row[self.id_col]), idx)

        if self.transform is not None:
            image_t = self.transform(image)
        else:
            image_t = image

        # Keep metadata in the batch so metrics and error analysis can group predictions.
        item: dict[str, Any] = {
            "image": image_t,
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
        # Synthetic ranking views (clean < mild < strong). The precomputed cache
        # is the default source; the on-the-fly generator is a fallback used
        # mainly for audits. All views skip spatial augmentation so the ranking
        # head sees the exact pixels the generator accepted; ``clean`` is the
        # un-augmented original (NOT ``item["image"]``, which is augmented).
        if self.synthetic_view_transform is not None:
            if self.synthetic_cache is not None:
                self._attach_cached_synthetic_views(image, item)
            elif self.synthetic_generator is not None:
                self._attach_generated_synthetic_views(image, item, idx)
        return item

    # ------------------------------------------------------------------
    def _resize_for_synthetic(self, pil_image: Image.Image) -> Image.Image:
        size = self.synthetic_target_size
        if size is not None and pil_image.size != (size, size):
            return pil_image.resize((size, size), Image.BILINEAR)
        return pil_image

    def _attach_view_tensors(
        self,
        item: dict[str, Any],
        *,
        clean: Image.Image,
        mild: Image.Image,
        strong: Image.Image,
        mild_severity: float,
        strong_severity: float,
        valid: bool,
        failure_reason: str,
    ) -> None:
        view_tf = self.synthetic_view_transform
        item["synthetic_clean_image"] = view_tf(clean)
        item["synthetic_mild_image"] = view_tf(mild)
        item["synthetic_strong_image"] = view_tf(strong)
        item["synthetic_mild_severity"] = torch.tensor(mild_severity, dtype=torch.float32)
        item["synthetic_strong_severity"] = torch.tensor(strong_severity, dtype=torch.float32)
        item["synthetic_valid"] = torch.tensor(bool(valid))
        item["synthetic_failure_reason"] = failure_reason

    def _attach_generated_synthetic_views(
        self, pil_image: Image.Image, item: dict[str, Any], idx: int
    ) -> None:
        clean = self._resize_for_synthetic(pil_image)
        # Seed a per-sample Generator from (synthetic_seed, idx) so synthetic
        # views are reproducible and NOT correlated across forked DataLoader
        # workers (a single shared Generator would be duplicated on fork).
        rng = np.random.default_rng([self.synthetic_seed, int(idx)])
        pair = self.synthetic_generator.generate_pair(clean, rng=rng)
        if pair.valid and pair.mild is not None and pair.strong is not None:
            self._attach_view_tensors(
                item,
                clean=clean,
                mild=pair.mild.image,
                strong=pair.strong.image,
                mild_severity=pair.mild.severity,
                strong_severity=pair.strong.severity,
                valid=True,
                failure_reason="",
            )
        else:
            # Fallback keeps collation stable; Stage 4 ranking skips these rows.
            self._attach_view_tensors(
                item,
                clean=clean,
                mild=clean,
                strong=clean,
                mild_severity=0.0,
                strong_severity=0.0,
                valid=False,
                failure_reason=str(pair.metadata.get("failure_reason") or "unknown_error"),
            )

    def _attach_cached_synthetic_views(self, pil_image: Image.Image, item: dict[str, Any]) -> None:
        clean = self._resize_for_synthetic(pil_image)
        entry = self.synthetic_cache.lookup(item["image_id"])
        if entry is None:
            # Not in the cache (e.g. MediaPipe failed at build time) -> invalid.
            self._attach_view_tensors(
                item,
                clean=clean,
                mild=clean,
                strong=clean,
                mild_severity=0.0,
                strong_severity=0.0,
                valid=False,
                failure_reason="not_in_cache",
            )
            return
        with (
            Image.open(entry["clean_path"]) as c,
            Image.open(entry["mild_path"]) as m,
            Image.open(entry["strong_path"]) as s,
        ):
            self._attach_view_tensors(
                item,
                clean=c.convert("RGB"),
                mild=m.convert("RGB"),
                strong=s.convert("RGB"),
                mild_severity=entry["mild_severity"],
                strong_severity=entry["strong_severity"],
                valid=True,
                failure_reason="",
            )
