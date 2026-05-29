"""Lightning module for the Face Occlusion baseline."""

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
        self._val_buffer: list[dict[str, Any]] = []
        self._female_value = str(cfg.data.get("female_value", "1.0"))
        self._male_value = str(cfg.data.get("male_value", "0.0"))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds = self(batch["image"])
        targets = batch["target"]
        loss = weighted_mse_loss(preds, targets)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        if opt is not None:
            self.log("train/lr", opt.param_groups[0]["lr"], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["image"])
        targets = batch["target"]
        loss = weighted_mse_loss(preds.clamp(0.0, 1.0), targets)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_buffer.append(
            {
                "preds": preds.detach().cpu(),
                "targets": targets.detach().cpu(),
                "genders": batch["gender"].detach().cpu(),
                "image_ids": list(batch["image_id"]),
                "paths": list(batch["path"]),
            }
        )

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self) -> None:
        if not self._val_buffer:
            return
        preds = torch.cat([b["preds"] for b in self._val_buffer]).numpy()
        targets = torch.cat([b["targets"] for b in self._val_buffer]).numpy()
        genders = torch.cat([b["genders"] for b in self._val_buffer]).numpy()
        # The official score is computed per gender then combined; we keep the
        # raw predictions for analysis but report the clipped metric.
        gender_str = np.array([f"{float(g):.1f}" for g in genders])
        score = challenge_score(
            preds,
            targets,
            gender_str,
            female_value=self._female_value,
            male_value=self._male_value,
        )
        for k, v in score.items():
            self.log(f"val/{k}", float(v), prog_bar=(k == "score"))

        # Sanity stats on raw (un-clipped) predictions: useful for the identity head.
        self.log("val/pred_min_raw", float(preds.min()))
        self.log("val/pred_max_raw", float(preds.max()))
        self.log("val/pct_pred_below_0", float((preds < 0).mean()))
        self.log("val/pct_pred_above_1", float((preds > 1).mean()))

        bins = list(self.cfg.split.occlusion_bins)
        bin_errs = error_by_occlusion_bin(preds, targets, bins=bins)
        for name, value in bin_errs.items():
            self.log(f"val/bin_{name}_err", float(value) if not np.isnan(value) else 0.0)

        self._last_val_outputs = {
            "preds": preds,
            "targets": targets,
            "genders": gender_str,
            "image_ids": sum([b["image_ids"] for b in self._val_buffer], []),
            "paths": sum([b["paths"] for b in self._val_buffer], []),
        }
        self._val_buffer.clear()

    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        preds = self(batch["image"])
        return {
            "preds": preds.detach().cpu(),
            "genders": batch["gender"].detach().cpu(),
            "image_ids": list(batch["image_id"]),
            "paths": list(batch["path"]),
        }

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=float(self.cfg.training.learning_rate),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        scheduler = CosineAnnealingLR(opt, T_max=int(self.cfg.training.max_epochs))
        return {"optimizer": opt, "lr_scheduler": scheduler}
