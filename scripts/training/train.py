#!/usr/bin/env python
"""Train a Face Occlusion model from a YAML config."""

from __future__ import annotations

import argparse
import json
import logging
import secrets
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
import torch
from pytorch_lightning.loggers import CSVLogger

from face_occlusion.data.datamodule import FaceOcclusionDataModule
from face_occlusion.data.normalize import normalize_target
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


def _configure_float32_matmul_precision(cfg) -> None:
    """Enable Tensor Core-friendly float32 matmul behavior on CUDA GPUs."""

    precision = str(cfg.training.get("float32_matmul_precision", "high")).lower()
    valid_values = {"highest", "high", "medium"}
    if precision not in valid_values:
        raise ValueError(
            "training.float32_matmul_precision must be one of "
            f"{sorted(valid_values)}, got '{precision}'."
        )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(precision)
        print(f"[train] Float32 matmul precision: {precision} (applied for CUDA).")
    else:
        print(f"[train] Float32 matmul precision: {precision} (CUDA unavailable, skipped).")


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


def _finalize(trainer, module, dm, run_dir: Path, *, interrupted: bool) -> None:
    """Best-effort: persist the best-so-far model's validation predictions, then a status
    marker. Runs on normal completion AND on interrupt/error so an interrupted run still
    leaves an analysable artifact. Never raises (so it can't mask the original error)."""
    status = "interrupted" if interrupted else "completed"
    try:
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        best = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
        source = "last-epoch validation outputs (in memory)"
        if best and Path(best).exists():
            # Refresh _last_val_outputs to the BEST checkpoint's predictions.
            try:
                trainer.validate(module, datamodule=dm, ckpt_path="best")
                source = f"best checkpoint ({Path(best).name})"
            except Exception as exc:  # noqa: BLE001 - finalize must be resilient
                print(
                    f"[train] Could not re-validate best checkpoint ({exc}); "
                    "falling back to the last in-memory validation outputs."
                )
        elif getattr(module, "_last_val_outputs", None) is None:
            # Interrupted before any validation completed: try current weights.
            try:
                trainer.validate(module, datamodule=dm)
                source = "current (unvalidated) weights"
            except Exception as exc:  # noqa: BLE001
                print(f"[train] No checkpoint and no validation outputs to save ({exc}).")

        pred_path = _save_val_predictions(module, run_dir / "predictions")
        (run_dir / "training_status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "predictions_from": source,
                    "best_checkpoint": best,
                    "predictions": str(pred_path) if pred_path != Path() else None,
                },
                indent=2,
            )
        )
        print(f"[train] Finalize ({status}): predictions from {source}")
        if best:
            print(f"[train] Best checkpoint: {best}")
    except Exception as exc:  # noqa: BLE001 - never let finalize crash the process
        print(f"[train] WARNING: finalize could not save predictions: {exc}")


def _resolve_seed(cfg) -> int:
    """Resolve ``project.seed`` to a concrete int, drawing a random one when unset.

    A fixed int reproduces a specific run (and is right for clean ablations). ``null`` /
    ``"random"`` / ``"auto"`` / absent draws a fresh random seed each run so we can explore
    different weight inits and data orderings. The resolved value is written back into ``cfg``
    so the run's ``config.yaml`` snapshot and ``metadata.json`` record it — re-running that
    saved config reproduces the run exactly.

    Note: only training randomness (model init, batch order, sampler) keys off this seed; the
    val split uses ``split.random_state`` (and is saved-then-reloaded), so a random seed does
    NOT change the split and paired comparisons stay valid.
    """
    raw = cfg.project.get("seed", None) if hasattr(cfg, "get") else None
    is_random = raw is None or (
        isinstance(raw, str) and raw.strip().lower() in {"random", "auto", "none", ""}
    )
    seed = secrets.randbelow(2**31 - 1) if is_random else int(raw)
    cfg.setdefault("project", {})["seed"] = seed
    if is_random:
        print(
            f"[train] project.seed not fixed -> randomly selected seed={seed} "
            "(recorded in config.yaml + metadata.json for reproduction)"
        )
    else:
        print(f"[train] Using fixed seed={seed}")
    return seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--train-on-all",
        action="store_true",
        help=(
            "Final-submission refit: fold the val rows back into training (no genuine "
            "held-out set). Disables early stopping and best.ckpt; train a fixed epoch "
            "budget and use last.ckpt. Keep the recipe frozen (hyperparameters were "
            "selected on the held-out val). Equivalent to setting split.train_on_all=true."
        ),
    )
    args = parser.parse_args()

    _configure_runtime_noise_filters()

    cfg = load_config(args.config)
    if args.train_on_all:
        # Record it in cfg so the run's config.yaml / metadata.json snapshot reflects it.
        cfg.setdefault("split", {})["train_on_all"] = True
        print(
            "[train] --train-on-all: folding val into train; no early stopping / best.ckpt, "
            "use last.ckpt for inference."
        )
    # Resolve the seed BEFORE the run dir / config snapshot so the chosen seed is recorded.
    _resolve_seed(cfg)
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

    deterministic = bool(cfg.project.get("deterministic", False))
    seed_everything(int(cfg.project.seed), deterministic=deterministic)
    pl.seed_everything(int(cfg.project.seed), workers=True)
    _configure_float32_matmul_precision(cfg)

    dm = FaceOcclusionDataModule(cfg)
    dm.prepare_data()
    _copy_split_snapshot(cfg, run_dir)
    dm.setup("fit")

    # Use training-set mean target to warm-start the head bias. Use the same
    # normalization the dataset applies so the bias matches the model targets.
    mean_target = None
    train_targets = None
    if dm.train_df is not None and cfg.data.target_col in dm.train_df.columns:
        vals = normalize_target(dm.train_df[cfg.data.target_col], cfg.data.target_scale)
        mean_target = float(vals.mean())
        # Full training occlusion distribution for distribution-aware reweighting
        # (losses.regression); a no-op unless that reweighting is enabled.
        train_targets = np.asarray(vals, dtype=float)

    module = FaceOcclusionLitModule(cfg, mean_target=mean_target, train_targets=train_targets)
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
        deterministic=deterministic,
    )

    # Wrap fit so an interrupt (Ctrl+C) or an intermediate-epoch error still saves the
    # best-so-far model's predictions before exiting. KeyboardInterrupt -> clean exit;
    # a real error is re-raised AFTER finalize so the failure stays visible.
    interrupted = False
    try:
        trainer.fit(module, datamodule=dm)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[train] KeyboardInterrupt — saving best-so-far predictions before exit...")
    finally:
        _finalize(trainer, module, dm, run_dir, interrupted=interrupted)

    print(f"Experiment directory: {run_dir}")


if __name__ == "__main__":
    main()
