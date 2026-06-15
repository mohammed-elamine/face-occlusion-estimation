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

    # train_on_all (final-submission refit) has no genuine held-out set — the monitored
    # val/score is a leaked subset of train. So we don't select on it: no best.ckpt and no
    # early stopping; only last.ckpt (the fixed-budget final weights) is the deliverable.
    split_cfg = cfg.get("split", {}) if hasattr(cfg, "get") else {}
    train_on_all = bool(split_cfg.get("train_on_all", False)) if split_cfg else False

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=cfg.checkpoint.get("filename", "best"),
        monitor=None if train_on_all else cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=0 if train_on_all else int(cfg.checkpoint.save_top_k),
        save_last=True,
        # Keep checkpoint names simple: best.ckpt and last.ckpt.
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint, lr_monitor]
    if not train_on_all:
        callbacks.insert(
            1,
            EarlyStopping(
                monitor=cfg.checkpoint.monitor,
                mode=cfg.checkpoint.mode,
                patience=int(cfg.training.get("early_stopping_patience", 5)),
            ),
        )

    # Optional EMA of weights (default off). When on, validation + checkpointing run on the
    # EMA weights, so val/score, best.ckpt, and inference all use the averaged model.
    ema_cfg = cfg.training.get("ema", {}) if hasattr(cfg.training, "get") else {}
    if ema_cfg and bool(ema_cfg.get("enabled", False)):
        from .ema import EMACallback

        callbacks.append(
            EMACallback(
                decay=float(ema_cfg.get("decay", 0.999)),
                warmup=bool(ema_cfg.get("warmup", True)),
                validate_on_ema=bool(ema_cfg.get("validate_on_ema", True)),
            )
        )
    return callbacks
