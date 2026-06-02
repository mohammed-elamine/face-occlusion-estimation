"""LightningDataModule shared by all Face Occlusion training configs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import FaceOcclusionDataset
from .metadata import add_path_metadata
from .samplers import build_batch_sampler_from_config
from .splits import load_split, make_stratified_split, save_split
from .transforms import build_eval_transform, build_train_transform


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

        if stage in (None, "fit", "validate"):
            df = add_path_metadata(pd.read_csv(cfg.data.train_csv), filename_col=cfg.data.id_col)
            split = load_split(cfg.split.split_path)[[cfg.data.id_col, "split"]]
            # The split stores ids only; metadata is reloaded from train.csv for each run.
            merged = df.merge(split, on=cfg.data.id_col, how="inner")
            if len(merged) != len(df):
                missing = len(df) - len(merged)
                print(f"[datamodule] Warning: {missing} rows missing from split, will be dropped.")

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
                train_df, transform=train_tf, mode="train", **common
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
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
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
            )
        return self._loader(self.train_ds, batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, 128, False, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, 128, False, False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
