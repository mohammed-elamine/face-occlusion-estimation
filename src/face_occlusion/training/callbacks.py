"""Training callbacks."""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def build_callbacks(cfg) -> list:
    ckpt_dir = Path(cfg.checkpoint.get("dirpath", "outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=cfg.checkpoint.get("filename", "best"),
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=int(cfg.checkpoint.save_top_k),
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        patience=int(cfg.training.get("early_stopping_patience", 5)),
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint, early_stop, lr_monitor]
