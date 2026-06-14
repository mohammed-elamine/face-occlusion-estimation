"""LightningDataModule shared by all Face Occlusion training configs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..utils.reproducibility import make_dataloader_generator, seed_worker
from .background_augment import BackgroundAugment
from .dataset import FaceOcclusionDataset
from .face_mask_store import FaceMaskStore
from .metadata import add_path_metadata
from .samplers import build_batch_sampler_from_config, build_weighted_sampler_from_config
from .splits import load_split, make_stratified_split, save_split
from .synthetic_cache import SyntheticCache
from .synthetic_occlusion import build_generator_from_config
from .transforms import (
    build_eval_transform,
    build_synthetic_view_transform,
    build_train_transform,
)


class FaceOcclusionDataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: FaceOcclusionDataset | None = None
        self.val_ds: FaceOcclusionDataset | None = None
        self.test_ds: FaceOcclusionDataset | None = None
        self.train_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        cfg = self.cfg
        split_path = Path(cfg.split.split_path)
        if split_path.exists():
            # Reuse the saved split so different experiment configs share the same validation set.
            return
        # Create the configured split lazily so train.py works after a fresh checkout.
        df = pd.read_csv(cfg.data.train_csv)
        split = make_stratified_split(
            df,
            target_col=cfg.data.target_col,
            gender_col=cfg.data.gender_col,
            id_col=cfg.data.id_col,
            bins=cfg.split.occlusion_bins,
            val_size=float(cfg.split.get("val_size", cfg.data.val_size)),
            seed=int(cfg.split.get("random_state", cfg.project.seed)),
            strategy=cfg.split.get("strategy", "row_stratified"),
            stratify_by=list(cfg.split.get("stratify_by", ["gender", "occlusion_bin"])),
            group_col=cfg.split.get("group_col", "group_id"),
        )
        save_split(split, split_path)
        print(f"[datamodule] Wrote split to {split_path}")

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        cfg = self.cfg
        train_tf = build_train_transform(cfg)
        eval_tf = build_eval_transform(cfg)
        # Synthetic occlusion is opt-in (default: no-op). The generator is only
        # attached to the *training* dataset and only when both
        # ``synthetic_occlusion.enabled`` and ``.return_in_batch`` are true.
        so_cfg = cfg.get("synthetic_occlusion", {}) if hasattr(cfg, "get") else {}
        aug_cfg = cfg.get("augmentation", {}) if hasattr(cfg, "get") else {}
        bg_cfg = aug_cfg.get("background", {}) if hasattr(aug_cfg, "get") else {}
        synthetic_generator = None
        synthetic_cache = None
        synthetic_view_tf = None
        synthetic_target_size = None
        background_augment = None
        synthetic_seed = int(so_cfg.get("seed", 42)) if so_cfg else 42
        cache_dir = so_cfg.get("cache_dir", None) if so_cfg else None
        use_cache = bool(so_cfg.get("use_cache", True)) if so_cfg else True
        want_views = (
            bool(so_cfg.get("enabled", False) and so_cfg.get("return_in_batch", False))
            if so_cfg
            else False
        )
        want_bg = bool(bg_cfg.get("enabled", False)) if bg_cfg else False

        # A single cache backs both synthetic views and background-aug masks.
        if cache_dir and use_cache and (want_views or want_bg):
            synthetic_cache = SyntheticCache(cache_dir)
            print(
                f"[datamodule] Loaded synthetic cache from {cache_dir} "
                f"({len(synthetic_cache)} pairs)"
            )

        if want_views:
            synthetic_view_tf = build_synthetic_view_transform(cfg)
            synthetic_target_size = int(cfg.augmentation.resize)
            # On-the-fly generation is the fallback when no cache is configured.
            if synthetic_cache is None:
                synthetic_generator = build_generator_from_config(cfg)

        if want_bg:
            # Prefer the dedicated full-coverage face-mask store; fall back to the synthetic
            # cache (limited coverage) only if the store dir is absent.
            mask_dir = bg_cfg.get("mask_dir", "data/face_masks")
            mask_lookup = None
            mask_source = None
            if mask_dir and Path(mask_dir).exists():
                mask_lookup = FaceMaskStore(mask_dir).load_mask
                mask_source = f"face-mask store '{mask_dir}'"
            elif synthetic_cache is not None:
                mask_lookup = synthetic_cache.load_mask
                mask_source = "synthetic cache (fallback; only covers ranking anchors)"
            if mask_lookup is not None:
                background_augment = BackgroundAugment(
                    mask_lookup=mask_lookup,
                    p=float(bg_cfg.get("p", 0.5)),
                    modes=tuple(bg_cfg.get("modes", ["replace", "brightness", "noise"])),
                    seed=int(bg_cfg.get("seed", synthetic_seed)),
                    noise_std=float(bg_cfg.get("noise_std", 25.0)),
                )
                print(f"[datamodule] Background augmentation: masks from {mask_source}")
            else:
                print(
                    "[datamodule] Warning: augmentation.background.enabled but no face-mask "
                    f"store at '{mask_dir}' (build with scripts.data.build_face_masks) and no "
                    "synthetic cache; skipping background augmentation."
                )

        if stage in (None, "fit", "validate"):
            df = add_path_metadata(pd.read_csv(cfg.data.train_csv), filename_col=cfg.data.id_col)
            split = load_split(cfg.split.split_path)[[cfg.data.id_col, "split"]]
            # The split stores ids only; metadata is reloaded from train.csv for each run.
            merged = df.merge(split, on=cfg.data.id_col, how="inner")
            if len(merged) != len(df):
                missing = len(df) - len(merged)
                msg = (
                    f"[datamodule] Split/train.csv mismatch on '{cfg.data.id_col}': "
                    f"{len(df)} train rows but {len(merged)} matched the split "
                    f"({missing} dropped). The split is likely stale relative to "
                    f"train.csv. Regenerate it with scripts.data.make_split, or set "
                    f"split.allow_missing_rows=true to proceed anyway."
                )
                # Loud by default: a silent row drop quietly shrinks the dataset and
                # makes val/score incomparable across runs (review R9).
                if not bool(cfg.split.get("allow_missing_rows", False)):
                    raise ValueError(msg)
                print(f"WARNING: {msg}")

            train_df = merged[merged["split"] == "train"].reset_index(drop=True)
            val_df = merged[merged["split"] == "val"].reset_index(drop=True)
            self.train_df = train_df

            common = dict(
                image_root=cfg.data.image_root,
                image_col=cfg.data.image_col,
                target_col=cfg.data.target_col,
                gender_col=cfg.data.gender_col,
                id_col=cfg.data.id_col,
                target_scale=cfg.data.target_scale,
            )
            self.train_ds = FaceOcclusionDataset(
                train_df,
                transform=train_tf,
                mode="train",
                synthetic_generator=synthetic_generator,
                synthetic_cache=synthetic_cache,
                synthetic_view_transform=synthetic_view_tf,
                synthetic_target_size=synthetic_target_size,
                synthetic_seed=synthetic_seed,
                background_augment=background_augment,
                **common,
            )
            self.val_ds = FaceOcclusionDataset(val_df, transform=eval_tf, mode="val", **common)

        if stage in (None, "test", "predict"):
            test_df = add_path_metadata(
                pd.read_csv(cfg.data.test_csv),
                filename_col=cfg.data.id_col,
            )
            self.test_ds = FaceOcclusionDataset(
                test_df,
                image_root=cfg.data.image_root,
                transform=eval_tf,
                mode="test",
                image_col=cfg.data.image_col,
                target_col=cfg.data.target_col,
                gender_col=cfg.data.gender_col,
                id_col=cfg.data.id_col,
                target_scale=cfg.data.target_scale,
            )

    # ------------------------------------------------------------------
    def _loader(self, ds, batch_size, shuffle, drop_last):
        num_workers = int(self.cfg.data.num_workers)
        # persistent_workers avoids worker restart overhead when num_workers > 0.
        # Pinned memory helps CUDA transfers, but PyTorch warns that MPS does not support it.
        pin_memory = torch.cuda.is_available()
        # Seed worker RNGs (NumPy/random) and the shuffle generator so runs are
        # reproducible regardless of worker count.
        generator = make_dataloader_generator(int(self.cfg.project.seed)) if shuffle else None
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        batch_size = int(self.cfg.training.batch_size)
        sampler = build_batch_sampler_from_config(self.train_df, self.cfg, batch_size=batch_size)
        if sampler is not None:
            sampler.log_summary()
            # Save summary next to experiment logs when a run directory is available.
            run_dir = self.cfg.get("experiment", {}).get("run_dir", None)
            if run_dir is not None:
                sampler.save_summary(Path(run_dir) / "reports" / "sampler_summary.json")
            # batch_sampler is mutually exclusive with batch_size/shuffle/drop_last.
            num_workers = int(self.cfg.data.num_workers)
            pin_memory = torch.cuda.is_available()
            return DataLoader(
                self.train_ds,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
                worker_init_fn=seed_worker,
            )
        # Teammate-style per-sample WeightedRandomSampler: plugs in via sampler=
        # (mutually exclusive with shuffle), keeping batch_size/drop_last.
        weighted = build_weighted_sampler_from_config(
            self.train_df, self.cfg, seed=int(self.cfg.project.seed)
        )
        if weighted is not None:
            sampler_cfg = self.cfg.get("sampler", {})
            drop_last = bool(sampler_cfg.get("drop_last", True))
            num_workers = int(self.cfg.data.num_workers)
            pin_memory = torch.cuda.is_available()
            return DataLoader(
                self.train_ds,
                sampler=weighted,
                batch_size=batch_size,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
                worker_init_fn=seed_worker,
            )
        return self._loader(self.train_ds, batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, 128, False, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, 128, False, False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
