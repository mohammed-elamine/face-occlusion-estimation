"""Lightning module for Face Occlusion regression experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..metrics.challenge_metric import (
    challenge_score,
    error_by_occlusion_bin,
    weighted_mse,
)
from ..models import build_model


def weighted_mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Per-sample weight w_i = 1/30 + y_i: hard (highly occluded) samples count more.
    weights = (1.0 / 30.0) + targets
    return (weights * (preds - targets) ** 2).sum() / weights.sum().clamp_min(1e-12)


class FaceOcclusionLitModule(pl.LightningModule):
    def __init__(self, cfg, mean_target: float | None = None) -> None:
        super().__init__()
        self.save_hyperparameters(dict(cfg), ignore=[])
        self.cfg = cfg
        self.model = build_model(cfg, mean_target=mean_target)
        # Validation metrics need the whole epoch because the score is grouped by gender.
        self._val_buffer: list[dict[str, Any]] = []
        self._female_value = str(cfg.data.get("female_value", "0.0"))
        self._male_value = str(cfg.data.get("male_value", "1.0"))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds = self(batch["image"])
        targets = batch["target"]
        batch_size = int(targets.shape[0])
        loss = weighted_mse_loss(preds, targets)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        if opt is not None:
            self.log(
                "train/lr",
                opt.param_groups[0]["lr"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["image"])
        targets = batch["target"]
        batch_size = int(targets.shape[0])
        # The challenge metric clips predictions, so validation loss follows that convention.
        loss = weighted_mse_loss(preds.clamp(0.0, 1.0), targets)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self._val_buffer.append(
            {
                "preds": preds.detach().cpu(),
                "targets": targets.detach().cpu(),
                "genders": batch["gender"].detach().cpu(),
                "image_ids": list(batch["image_id"]),
                "filenames": list(batch["filename"]),
                "paths": list(batch["path"]),
                "databases": list(batch["database"]),
                "source_subfolders": list(batch["source_subfolder"]),
                "group_ids": list(batch["group_id"]),
                "face_ids": batch["face_id"].detach().cpu(),
            }
        )

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self) -> None:
        if not self._val_buffer:
            return
        preds = torch.cat([b["preds"] for b in self._val_buffer]).numpy()
        targets = torch.cat([b["targets"] for b in self._val_buffer]).numpy()
        genders = torch.cat([b["genders"] for b in self._val_buffer]).numpy()
        num_val = int(targets.shape[0])
        # The official score is computed per gender then combined; we keep the
        # raw predictions for analysis but report the clipped metric.
        # Formatting keeps float gender labels consistent with config values like "1.0".
        gender_str = np.array([f"{float(g):.1f}" for g in genders])
        score = challenge_score(
            preds,
            targets,
            gender_str,
            female_value=self._female_value,
            male_value=self._male_value,
        )
        for k, v in score.items():
            self.log(f"val/{k}", float(v), prog_bar=(k == "score"), batch_size=num_val)

        # Sanity stats on raw (un-clipped) predictions: useful for the identity head.
        self.log("val/pred_min_raw", float(preds.min()), batch_size=num_val)
        self.log("val/pred_max_raw", float(preds.max()), batch_size=num_val)
        self.log("val/pct_pred_below_0", float((preds < 0).mean()), batch_size=num_val)
        self.log("val/pct_pred_above_1", float((preds > 1).mean()), batch_size=num_val)

        bins = list(self.cfg.split.occlusion_bins)
        bin_errs = error_by_occlusion_bin(preds, targets, bins=bins)
        for name, value in bin_errs.items():
            self.log(
                f"val/bin_{name}_err",
                float(value) if not np.isnan(value) else 0.0,
                batch_size=num_val,
            )

        databases = np.array(sum([b["databases"] for b in self._val_buffer], []))
        for database in sorted(np.unique(databases)):
            mask = databases == database
            db_err = weighted_mse(preds[mask], targets[mask], clip=True)
            self.log(f"val/database/{database}_err", db_err, batch_size=int(mask.sum()))

        # train.py reads this after trainer.validate() to write val_predictions.csv.
        self._last_val_outputs = {
            "preds": preds,
            "targets": targets,
            "genders": gender_str,
            "image_ids": sum([b["image_ids"] for b in self._val_buffer], []),
            "filenames": sum([b["filenames"] for b in self._val_buffer], []),
            "paths": sum([b["paths"] for b in self._val_buffer], []),
            "databases": databases.tolist(),
            "source_subfolders": sum([b["source_subfolders"] for b in self._val_buffer], []),
            "group_ids": sum([b["group_ids"] for b in self._val_buffer], []),
            "face_ids": torch.cat([b["face_ids"] for b in self._val_buffer]).numpy(),
        }
        self._val_buffer.clear()

    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        preds = self(batch["image"])
        out = {
            "preds": preds.detach().cpu(),
            "image_ids": list(batch["image_id"]),
            "filenames": list(batch["filename"]),
            "paths": list(batch["path"]),
            "databases": list(batch["database"]),
            "source_subfolders": list(batch["source_subfolder"]),
            "group_ids": list(batch["group_id"]),
            "face_ids": batch["face_id"].detach().cpu(),
        }
        if "gender" in batch:
            out["genders"] = batch["gender"].detach().cpu()
        return out

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=float(self.cfg.training.learning_rate),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        scheduler = CosineAnnealingLR(opt, T_max=int(self.cfg.training.max_epochs))
        return {"optimizer": opt, "lr_scheduler": scheduler}
