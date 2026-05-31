#!/usr/bin/env python
"""Train a Face Occlusion model from a YAML config."""

from __future__ import annotations

import argparse
import logging
import shutil
import warnings
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from face_occlusion.data.datamodule import FaceOcclusionDataModule
from face_occlusion.training import FaceOcclusionLitModule, build_callbacks
from face_occlusion.utils import (
    create_run_dir,
    load_config,
    save_config_snapshot,
    save_git_info,
    save_metadata,
    seed_everything,
    to_plain_dict,
    write_latest_run_pointer,
)


class _LightningTipFilter(logging.Filter):
    """Drop optional Lightning cloud-service tips while keeping useful trainer logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return (
            "try installing [litlogger]" not in message
            and "try installing [litmodels]" not in message
        )


def _configure_runtime_noise_filters() -> None:
    """Silence known third-party messages that are not actionable for this project."""

    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        module=r"pytorch_lightning\.utilities\._pytree",
    )
    tip_filter = _LightningTipFilter()
    logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(tip_filter)
    logging.getLogger("lightning_fabric.utilities.rank_zero").addFilter(tip_filter)


def _build_logger(cfg, run_dir: Path):
    logs_dir = run_dir / "logs"
    run_id = run_dir.name
    if bool(cfg.logging.get("use_wandb", False)):
        try:
            from pytorch_lightning.loggers import WandbLogger

            return WandbLogger(
                project=cfg.logging.wandb_project,
                name=run_id,
                save_dir=str(logs_dir / "wandb"),
                config=to_plain_dict(cfg),
            )
        except Exception as exc:
            # W&B is optional; CSV logs keep cluster runs usable without credentials.
            print(f"[train] W&B disabled (failed to init: {exc}); using CSVLogger.")
    return CSVLogger(save_dir=str(logs_dir), name="csv_logs")


def _config_snapshot(cfg, run_dir: Path) -> dict:
    # The saved config records resolved run paths, not only the user-provided YAML.
    snapshot = to_plain_dict(cfg)
    experiment = snapshot.setdefault("experiment", {})
    experiment["run_id"] = run_dir.name
    experiment["run_dir"] = str(run_dir)
    snapshot.setdefault("checkpoint", {})["dirpath"] = str(run_dir / "checkpoints")
    return snapshot


def _copy_split_snapshot(cfg, run_dir: Path) -> Path:
    split_path = Path(cfg.split.split_path)
    if not split_path.exists():
        print(f"[train] Split file not found, nothing to snapshot: {split_path}")
        return Path()

    # Store the split with each run so validation metrics remain reproducible.
    dest = run_dir / "splits" / split_path.name
    if split_path.resolve() != dest.resolve():
        shutil.copy2(split_path, dest)
    print(f"[train] Copied split snapshot: {dest}")
    return dest


def _save_val_predictions(module: FaceOcclusionLitModule, out_dir: Path) -> Path:
    out = getattr(module, "_last_val_outputs", None)
    if out is None:
        print("[train] No validation outputs to save.")
        return Path()
    preds = np.asarray(out["preds"])
    targets = np.asarray(out["targets"], dtype=float)
    # Save both raw and clipped predictions for calibration and official-score analysis.
    clipped = np.clip(preds, 0.0, 1.0)
    df = pd.DataFrame(
        {
            "image_id": out["image_ids"],
            "filename": out["filenames"],
            "path": out["paths"],
            "gender": out["genders"],
            "target": targets,
            "pred_raw": preds,
            "pred_clipped": clipped,
            "abs_error": np.abs(clipped - targets),
            "database": out["databases"],
            "source_subfolder": out["source_subfolders"],
            "group_id": out["group_ids"],
            "face_id": out["face_ids"],
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

    _configure_runtime_noise_filters()

    cfg = load_config(args.config)
    run_dir = create_run_dir(cfg)
    checkpoint_dir = run_dir / "checkpoints"

    # The Python training entrypoint owns experiment organization; Slurm only launches it.
    cfg.setdefault("experiment", {})["run_id"] = run_dir.name
    cfg.setdefault("experiment", {})["run_dir"] = str(run_dir)
    cfg.setdefault("checkpoint", {})["dirpath"] = str(checkpoint_dir)

    snapshot = _config_snapshot(cfg, run_dir)
    experiment_cfg = snapshot.get("experiment", {})
    if bool(experiment_cfg.get("save_config", True)):
        save_config_snapshot(snapshot, run_dir)
    if bool(experiment_cfg.get("save_git_info", True)):
        save_git_info(run_dir)
    save_metadata(snapshot, run_dir, config_path=args.config)
    if bool(experiment_cfg.get("create_latest_pointer", False)):
        write_latest_run_pointer(run_dir, experiment_cfg.get("output_root", run_dir.parent))

    print(f"[train] Experiment directory: {run_dir}")

    seed_everything(int(cfg.project.seed))
    pl.seed_everything(int(cfg.project.seed), workers=True)

    dm = FaceOcclusionDataModule(cfg)
    dm.prepare_data()
    _copy_split_snapshot(cfg, run_dir)
    dm.setup("fit")

    # Use training-set mean target to warm-start the head bias.
    mean_target = None
    if dm.train_df is not None and cfg.data.target_col in dm.train_df.columns:
        vals = dm.train_df[cfg.data.target_col].astype(float)
        if vals.max() > 1.5:
            vals = vals / 100.0
        mean_target = float(vals.mean())

    module = FaceOcclusionLitModule(cfg, mean_target=mean_target)
    logger = _build_logger(cfg, run_dir)
    callbacks = build_callbacks(cfg, checkpoint_dir=checkpoint_dir)

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        precision=cfg.training.get("precision", "32-true"),
        gradient_clip_val=float(cfg.training.get("gradient_clip_val", 0.0)),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(run_dir),
        enable_model_summary=bool(cfg.logging.get("model_summary", False)),
        log_every_n_steps=20,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(module, datamodule=dm)
    trainer.validate(module, datamodule=dm, ckpt_path="best")

    pred_path = _save_val_predictions(module, run_dir / "predictions")
    best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", "")

    print(f"Experiment directory: {run_dir}")
    print(f"Best checkpoint: {best_ckpt or checkpoint_dir / 'best.ckpt'}")
    print(f"Validation predictions: {pred_path}")


if __name__ == "__main__":
    main()
