#!/usr/bin/env python
"""Train the Face Occlusion baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from face_occlusion.data.datamodule import FaceOcclusionDataModule
from face_occlusion.training import FaceOcclusionLitModule, build_callbacks
from face_occlusion.utils import load_config, seed_everything


def _build_logger(cfg):
    if bool(cfg.logging.get("use_wandb", False)):
        try:
            from pytorch_lightning.loggers import WandbLogger

            return WandbLogger(
                project=cfg.logging.wandb_project,
                name=cfg.logging.run_name,
                save_dir=str(Path(cfg.project.output_dir) / "wandb"),
                config=dict(cfg),
            )
        except Exception as exc:
            print(f"[train] W&B disabled (failed to init: {exc}); using CSVLogger.")
    return CSVLogger(save_dir=str(Path(cfg.project.output_dir) / "logs"), name=cfg.logging.run_name)


def _save_val_predictions(module: FaceOcclusionLitModule, out_dir: Path) -> Path:
    out = getattr(module, "_last_val_outputs", None)
    if out is None:
        print("[train] No validation outputs to save.")
        return Path()
    preds = np.asarray(out["preds"])
    clipped = np.clip(preds, 0.0, 1.0)
    df = pd.DataFrame(
        {
            "image_id": out["image_ids"],
            "path": out["paths"],
            "gender": out["genders"],
            "target": out["targets"],
            "pred_raw": preds,
            "pred_clipped": clipped,
            "abs_error": np.abs(clipped - out["targets"]),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "val_predictions.csv"
    df.to_csv(path, index=False)
    print(f"[train] Wrote validation predictions: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.project.seed))
    pl.seed_everything(int(cfg.project.seed), workers=True)

    dm = FaceOcclusionDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")

    # Use training-set mean target to warm-start the head bias.
    mean_target = None
    if dm.train_df is not None and cfg.data.target_col in dm.train_df.columns:
        vals = dm.train_df[cfg.data.target_col].astype(float)
        if vals.max() > 1.5:
            vals = vals / 100.0
        mean_target = float(vals.mean())

    module = FaceOcclusionLitModule(cfg, mean_target=mean_target)
    logger = _build_logger(cfg)
    callbacks = build_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        precision=cfg.training.get("precision", "32-true"),
        gradient_clip_val=float(cfg.training.get("gradient_clip_val", 0.0)),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(Path(cfg.project.output_dir)),
        log_every_n_steps=20,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(module, datamodule=dm)
    trainer.validate(module, datamodule=dm, ckpt_path="best")

    _save_val_predictions(module, Path(cfg.project.output_dir) / "predictions")


if __name__ == "__main__":
    main()
