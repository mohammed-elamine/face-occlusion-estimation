"""Training callbacks for model selection and learning-rate logging."""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def build_callbacks(cfg, checkpoint_dir: str | Path | None = None) -> list:
    # Training passes checkpoint_dir so every run writes inside its experiment folder.
    configured_dir = cfg.checkpoint.get("dirpath", None)
    ckpt_dir = Path(checkpoint_dir or configured_dir or "outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=cfg.checkpoint.get("filename", "best"),
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=int(cfg.checkpoint.save_top_k),
        save_last=True,
        # Keep checkpoint names simple: best.ckpt and last.ckpt.
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        patience=int(cfg.training.get("early_stopping_patience", 5)),
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint, early_stop, lr_monitor]
