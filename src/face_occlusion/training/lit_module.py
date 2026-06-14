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
from ..models import (
    CONSISTENCY_MODES,
    DEFAULT_ORDINAL_THRESHOLD_WEIGHTS,
    OcclusionModelOutput,
    build_model,
    make_ordinal_targets,
    regression_ordinal_consistency_loss,
    threshold_weighted_bce,
)


def weighted_mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Per-sample weight w_i = 1/30 + y_i: hard (highly occluded) samples count more.
    weights = (1.0 / 30.0) + targets
    return (weights * (preds - targets) ** 2).sum() / weights.sum().clamp_min(1e-12)


def _scheduled_loss_weight(
    *,
    target_weight: float,
    warmup_epochs: int,
    warmup_start_weight: float,
    current_epoch: int,
) -> float:
    """Epoch-based linear warmup for an auxiliary loss coefficient.

    Returns ``target_weight`` immediately when ``warmup_epochs <= 0`` so any
    existing static-weight config keeps its exact behaviour. During warmup the
    effective weight grows linearly from ``warmup_start_weight`` to
    ``target_weight``; ``current_epoch + 1`` is used so epoch 0 already gets a
    non-zero coefficient (e.g. ``weight=0.1`` with ``warmup_epochs=3`` yields
    ``0.0333, 0.0667, 0.1, 0.1, ...``).
    """
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if target_weight < 0:
        raise ValueError(f"target_weight must be >= 0, got {target_weight}")
    if warmup_start_weight < 0:
        raise ValueError(f"warmup_start_weight must be >= 0, got {warmup_start_weight}")
    if warmup_start_weight > target_weight:
        # We only support warmup, not cooldown; surface the misconfig loudly.
        raise ValueError(
            "warmup_start_weight must be <= target_weight "
            f"(got start={warmup_start_weight}, target={target_weight}); "
            "cooldown schedules are not supported."
        )
    if warmup_epochs == 0:
        return float(target_weight)
    progress = min(1.0, (max(int(current_epoch), 0) + 1) / float(warmup_epochs))
    return float(warmup_start_weight + progress * (target_weight - warmup_start_weight))


def _safe_mean(values: list[float | None]) -> float:
    """Mean of a list that may contain ``None`` placeholders; ``0.0`` if empty."""
    finite = [v for v in values if v is not None]
    if not finite:
        return 0.0
    return float(sum(finite) / len(finite))


def _per_threshold_prf(
    preds_bin: np.ndarray, y_bin: np.ndarray
) -> tuple[np.ndarray, list[float | None], list[float | None], list[float | None]]:
    """Per-threshold accuracy, precision, recall, F1 for an ordinal head.

    ``preds_bin`` and ``y_bin`` are bool arrays of shape ``(N, K)``.
    Precision/recall/F1 entries are ``None`` when the denominator is zero,
    so callers can drop the metric for empty subgroups instead of logging NaN.
    """
    if preds_bin.size == 0:
        k = preds_bin.shape[1] if preds_bin.ndim == 2 else 0
        return (
            np.zeros(k, dtype=float),
            [None] * k,
            [None] * k,
            [None] * k,
        )
    acc = (preds_bin == y_bin).mean(axis=0).astype(float)
    prec: list[float | None] = []
    rec: list[float | None] = []
    f1: list[float | None] = []
    for k in range(preds_bin.shape[1]):
        tp = int((preds_bin[:, k] & y_bin[:, k]).sum())
        fp = int((preds_bin[:, k] & ~y_bin[:, k]).sum())
        fn = int((~preds_bin[:, k] & y_bin[:, k]).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else None
        r = tp / (tp + fn) if (tp + fn) > 0 else None
        if p is not None and r is not None and (p + r) > 0:
            f = 2 * p * r / (p + r)
        else:
            f = None
        prec.append(p)
        rec.append(r)
        f1.append(f)
    return acc, prec, rec, f1


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

        # ── Ordinal-head wiring (Stage 1) ────────────────────────────────
        # The ordinal loss is only active when the head exists AND the loss
        # is enabled in config; otherwise the module behaves as in Stage 0.
        losses_cfg = cfg.get("losses", {}) if hasattr(cfg, "get") else {}
        ord_cfg = losses_cfg.get("ordinal", {}) if losses_cfg else {}
        self._ord_loss_enabled = bool(
            getattr(self.model, "use_ordinal_head", False)
            and (ord_cfg.get("enabled", False) if ord_cfg else False)
        )
        self._ord_weight = float(ord_cfg.get("weight", 0.2)) if ord_cfg else 0.2
        self._ord_warmup_epochs = int(ord_cfg.get("warmup_epochs", 0)) if ord_cfg else 0
        self._ord_warmup_start_weight = (
            float(ord_cfg.get("warmup_start_weight", 0.0)) if ord_cfg else 0.0
        )
        # Validate eagerly so a typo surfaces at build time, not after epoch 0.
        _scheduled_loss_weight(
            target_weight=self._ord_weight,
            warmup_epochs=self._ord_warmup_epochs,
            warmup_start_weight=self._ord_warmup_start_weight,
            current_epoch=0,
        )
        if self._ord_loss_enabled:
            thresholds = self.model.ordinal_thresholds.detach().clone()
            raw_w = ord_cfg.get("threshold_weights", None)
            if raw_w is None:
                raw_w = list(DEFAULT_ORDINAL_THRESHOLD_WEIGHTS)[: thresholds.numel()]
            w = torch.tensor(list(raw_w), dtype=torch.float32)
            if w.numel() != thresholds.numel():
                raise ValueError(
                    f"losses.ordinal.threshold_weights has {w.numel()} entries but "
                    f"model.ordinal_thresholds has {thresholds.numel()}"
                )
            self.register_buffer("_ord_threshold_weights", w, persistent=False)
            self.register_buffer("_ord_thresholds", thresholds, persistent=False)
        else:
            self._ord_threshold_weights = None
            self._ord_thresholds = None

        # ── Regression–ordinal consistency wiring (Stage 2) ───────────────
        # Only active when an ordinal head exists AND consistency is enabled.
        # Misconfiguration (consistency on, ordinal off) raises immediately so
        # silent no-ops do not hide a broken run.
        cons_cfg = losses_cfg.get("consistency", {}) if losses_cfg else {}
        cons_requested = bool(cons_cfg.get("enabled", False)) if cons_cfg else False
        if cons_requested and not getattr(self.model, "use_ordinal_head", False):
            raise ValueError("losses.consistency.enabled=true requires model.use_ordinal_head=true")
        self._cons_loss_enabled = bool(
            cons_requested and getattr(self.model, "use_ordinal_head", False)
        )
        self._cons_weight = float(cons_cfg.get("weight", 0.05)) if cons_cfg else 0.05
        self._cons_warmup_epochs = int(cons_cfg.get("warmup_epochs", 0)) if cons_cfg else 0
        self._cons_warmup_start_weight = (
            float(cons_cfg.get("warmup_start_weight", 0.0)) if cons_cfg else 0.0
        )
        _scheduled_loss_weight(
            target_weight=self._cons_weight,
            warmup_epochs=self._cons_warmup_epochs,
            warmup_start_weight=self._cons_warmup_start_weight,
            current_epoch=0,
        )
        self._cons_temperature = float(cons_cfg.get("temperature", 0.05)) if cons_cfg else 0.05
        self._cons_mode = str(cons_cfg.get("mode", "symmetric")) if cons_cfg else "symmetric"
        if self._cons_loss_enabled and self._cons_mode not in CONSISTENCY_MODES:
            raise ValueError(
                f"losses.consistency.mode must be one of {CONSISTENCY_MODES}, "
                f"got {self._cons_mode!r}"
            )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> OcclusionModelOutput:
        # Returns the structured output contract (see models.outputs).
        # Downstream steps read ``outputs.y_pred`` rather than assuming a tensor.
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        preds = outputs.y_pred
        targets = batch["target"]
        batch_size = int(targets.shape[0])
        loss_reg = weighted_mse_loss(preds, targets)

        loss_ord = self._compute_ordinal_loss(outputs, targets)
        loss_cons = self._compute_consistency_loss(outputs)
        loss = loss_reg
        lambda_ord = self._effective_ordinal_weight() if loss_ord is not None else 0.0
        lambda_cons = self._effective_consistency_weight() if loss_cons is not None else 0.0
        if loss_ord is not None:
            loss = loss + lambda_ord * loss_ord
        if loss_cons is not None:
            loss = loss + lambda_cons * loss_cons

        # Keep `train/loss` = total so existing dashboards continue to work.
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if loss_ord is not None or loss_cons is not None:
            self.log(
                "train/loss_reg", loss_reg, on_step=False, on_epoch=True, batch_size=batch_size
            )
        if loss_ord is not None:
            self.log(
                "train/loss_ord", loss_ord, on_step=False, on_epoch=True, batch_size=batch_size
            )
            # Log effective coefficient so warmup is verifiable from metrics.csv.
            self.log(
                "train/lambda_ord",
                float(lambda_ord),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        if loss_cons is not None:
            self.log(
                "train/loss_cons", loss_cons, on_step=False, on_epoch=True, batch_size=batch_size
            )
            self.log(
                "train/lambda_cons",
                float(lambda_cons),
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

    # ------------------------------------------------------------------
    def _effective_ordinal_weight(self) -> float:
        """Current-epoch ordinal coefficient after optional linear warmup."""
        return _scheduled_loss_weight(
            target_weight=self._ord_weight,
            warmup_epochs=self._ord_warmup_epochs,
            warmup_start_weight=self._ord_warmup_start_weight,
            current_epoch=int(self.current_epoch),
        )

    def _effective_consistency_weight(self) -> float:
        """Current-epoch consistency coefficient after optional linear warmup."""
        return _scheduled_loss_weight(
            target_weight=self._cons_weight,
            warmup_epochs=self._cons_warmup_epochs,
            warmup_start_weight=self._cons_warmup_start_weight,
            current_epoch=int(self.current_epoch),
        )

    def _compute_ordinal_loss(
        self, outputs: OcclusionModelOutput, targets: torch.Tensor
    ) -> torch.Tensor | None:
        """Threshold-weighted BCE on ``outputs.ordinal_logits`` when active."""
        if not self._ord_loss_enabled or outputs.ordinal_logits is None:
            return None
        ord_targets = make_ordinal_targets(targets, self._ord_thresholds)
        return threshold_weighted_bce(
            outputs.ordinal_logits, ord_targets, self._ord_threshold_weights
        )

    def _compute_consistency_loss(self, outputs: OcclusionModelOutput) -> torch.Tensor | None:
        """Soft MSE between sigmoid(ordinal_logits) and regression-implied probs."""
        if not self._cons_loss_enabled or outputs.ordinal_logits is None:
            return None
        return regression_ordinal_consistency_loss(
            y_pred=outputs.y_pred,
            ordinal_logits=outputs.ordinal_logits,
            thresholds=self.model.ordinal_thresholds,
            temperature=self._cons_temperature,
            mode=self._cons_mode,
        )

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        preds = outputs.y_pred
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
                "ordinal_logits": (
                    outputs.ordinal_logits.detach().cpu()
                    if outputs.ordinal_logits is not None
                    and (self._ord_loss_enabled or self._cons_loss_enabled)
                    else None
                ),
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

        # ── Additional subgroup diagnostics: bias, MAE, count ─────────────────────
        preds_clipped = np.clip(preds, 0.0, 1.0)
        errors = preds_clipped - targets

        # Global MAE / bias / RMSE
        self.log("val/mae", float(np.abs(errors).mean()), batch_size=num_val)
        self.log("val/bias", float(errors.mean()), batch_size=num_val)
        self.log("val/rmse", float(np.sqrt((errors**2).mean())), batch_size=num_val)

        # Per-gender MAE, bias, count
        for g_val, g_label in ((self._female_value, "female"), (self._male_value, "male")):
            g_mask = gender_str == g_val
            if not g_mask.any():
                continue
            n = int(g_mask.sum())
            self.log(f"val/{g_label}_mae", float(np.abs(errors[g_mask]).mean()), batch_size=n)
            self.log(f"val/{g_label}_bias", float(errors[g_mask].mean()), batch_size=n)
            self.log(f"val/{g_label}_count", float(n), batch_size=n)

        # Per-occlusion-bin weighted_mse, MAE, bias, count
        for i in range(len(bins) - 1):
            lo, hi = float(bins[i]), float(bins[i + 1])
            bin_label = f"{lo:.2f}_{hi:.2f}"
            last_bin = i == len(bins) - 2
            b_mask = (targets >= lo) & (targets <= hi if last_bin else targets < hi)
            n = int(b_mask.sum())
            if n == 0:
                continue
            self.log(f"val/bin_{bin_label}_count", float(n), batch_size=n)
            self.log(
                f"val/bin_{bin_label}_weighted_mse",
                weighted_mse(preds[b_mask], targets[b_mask], clip=True),
                batch_size=n,
            )
            self.log(
                f"val/bin_{bin_label}_mae",
                float(np.abs(errors[b_mask]).mean()),
                batch_size=n,
            )
            self.log(
                f"val/bin_{bin_label}_bias",
                float(errors[b_mask].mean()),
                batch_size=n,
            )

        # High-occlusion aggregate (target >= 0.40). The fine-grained
        # [0.60, 1.00] bin is often too small for stable diagnostics, so we
        # report an aggregate that pools [0.40, 0.60) and [0.60, 1.00].
        self._log_high_occlusion_aggregate(
            preds=preds,
            targets=targets,
            errors=errors,
            gender_str=gender_str,
            threshold=0.40,
        )

        # Per-database MAE, bias, count
        for database in sorted(np.unique(databases)):
            db_mask = databases == database
            n = int(db_mask.sum())
            self.log(
                f"val/database/{database}_mae",
                float(np.abs(errors[db_mask]).mean()),
                batch_size=n,
            )
            self.log(f"val/database/{database}_bias", float(errors[db_mask].mean()), batch_size=n)
            self.log(f"val/database/{database}_count", float(n), batch_size=n)

        # ── Ordinal-head diagnostics (Stage 1) ──────────────────────────
        # Only computed when the head is enabled; safe no-op otherwise.
        self._log_ordinal_val_metrics(
            num_val,
            targets_np=targets,
            gender_str=gender_str,
            databases=databases,
            bins=bins,
        )
        # ── Consistency diagnostics (Stage 2) ────────────────────────────
        self._log_consistency_val_metrics(num_val)

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
    def _log_high_occlusion_aggregate(
        self,
        *,
        preds: np.ndarray,
        targets: np.ndarray,
        errors: np.ndarray,
        gender_str: np.ndarray,
        threshold: float,
    ) -> None:
        """Log aggregate diagnostics for ``targets >= threshold``.

        Pools all bins above ``threshold`` (e.g. [0.40, 0.60) and [0.60, 1.00])
        so the high-occlusion error has enough samples to be stable across
        epochs, while the fine-grained per-bin metrics remain available.
        """
        label = f"high_occ_{threshold:.2f}_1.00"
        mask = targets >= threshold
        n = int(mask.sum())
        self.log(f"val/{label}_count", float(n), batch_size=max(n, 1))
        if n == 0:
            return
        self.log(
            f"val/{label}_err",
            weighted_mse(preds[mask], targets[mask], clip=True),
            batch_size=n,
        )
        self.log(
            f"val/{label}_weighted_mse",
            weighted_mse(preds[mask], targets[mask], clip=True),
            batch_size=n,
        )
        self.log(f"val/{label}_mae", float(np.abs(errors[mask]).mean()), batch_size=n)
        self.log(f"val/{label}_bias", float(errors[mask].mean()), batch_size=n)

        # Gender-specific aggregate: same definitions, restricted to one gender.
        gender_errs: dict[str, float] = {}
        for g_val, g_label in ((self._female_value, "female"), (self._male_value, "male")):
            g_mask = mask & (gender_str == g_val)
            n_g = int(g_mask.sum())
            self.log(f"val/{label}_count_{g_label}", float(n_g), batch_size=max(n_g, 1))
            if n_g == 0:
                continue
            err_g = weighted_mse(preds[g_mask], targets[g_mask], clip=True)
            gender_errs[g_label] = float(err_g)
            self.log(f"val/{label}_err_{g_label}", err_g, batch_size=n_g)
        if "female" in gender_errs and "male" in gender_errs:
            self.log(
                f"val/{label}_gap",
                abs(gender_errs["female"] - gender_errs["male"]),
                batch_size=n,
            )

    # ------------------------------------------------------------------
    def _log_ordinal_val_metrics(
        self,
        num_val: int,
        *,
        targets_np: np.ndarray | None = None,
        gender_str: np.ndarray | None = None,
        databases: np.ndarray | None = None,
        bins: list[float] | None = None,
    ) -> None:
        """Log full-epoch ordinal-head diagnostics when the head is active.

        Metrics are computed once over the concatenated validation epoch (not
        averaged across batches), so rare high-threshold positives are not
        misrepresented when most batches have zero positives.
        """
        if not self._ord_loss_enabled:
            return
        logits_list = [
            b["ordinal_logits"] for b in self._val_buffer if b["ordinal_logits"] is not None
        ]
        if not logits_list:
            return
        ord_logits = torch.cat(logits_list, dim=0)
        targets_t = torch.cat([b["targets"] for b in self._val_buffer], dim=0)
        thresholds = self._ord_thresholds.detach().cpu()
        ord_targets = make_ordinal_targets(targets_t, thresholds)

        val_ord_loss = threshold_weighted_bce(
            ord_logits, ord_targets, self._ord_threshold_weights.detach().cpu()
        )
        self.log("val/ord_loss", float(val_ord_loss), batch_size=num_val)

        # Compute everything in numpy for clarity and to share with subgroup splits.
        logits = ord_logits.numpy()
        y_bin = ord_targets.numpy().astype(bool)
        preds_bin = logits >= 0.0  # sigmoid >= 0.5

        # --- Global threshold metrics (full-epoch) --------------------
        acc_per_thr, prec_per_thr, rec_per_thr, f1_per_thr = _per_threshold_prf(preds_bin, y_bin)
        self.log(
            "val/ord_threshold_acc_mean",
            float(_safe_mean(acc_per_thr)),
            batch_size=num_val,
        )
        self.log(
            "val/ord_threshold_precision_mean",
            float(_safe_mean(prec_per_thr)),
            batch_size=num_val,
        )
        self.log(
            "val/ord_threshold_recall_mean",
            float(_safe_mean(rec_per_thr)),
            batch_size=num_val,
        )
        self.log(
            "val/ord_threshold_f1_mean",
            float(_safe_mean(f1_per_thr)),
            batch_size=num_val,
        )

        # --- Per-threshold metrics + legacy "high threshold" keys -----
        for k, t in enumerate(thresholds.tolist()):
            tkey = f"{t:.2f}"
            n_pos = int(y_bin[:, k].sum())
            n_neg = int((~y_bin[:, k]).sum())
            self.log(f"val/ord_t_{tkey}_acc", float(acc_per_thr[k]), batch_size=num_val)
            if prec_per_thr[k] is not None:
                self.log(
                    f"val/ord_t_{tkey}_precision",
                    float(prec_per_thr[k]),
                    batch_size=num_val,
                )
            if rec_per_thr[k] is not None:
                self.log(
                    f"val/ord_t_{tkey}_recall",
                    float(rec_per_thr[k]),
                    batch_size=max(n_pos, 1),
                )
                # Preserve the original per-threshold recall key for back-compat.
                self.log(
                    f"val/ord_threshold_recall_{tkey}",
                    float(rec_per_thr[k]),
                    batch_size=max(n_pos, 1),
                )
                if abs(t - 0.40) < 1e-6:
                    self.log(
                        "val/ord_high_threshold_recall_0.40",
                        float(rec_per_thr[k]),
                        batch_size=max(n_pos, 1),
                    )
                if abs(t - 0.60) < 1e-6:
                    self.log(
                        "val/ord_high_threshold_recall_0.60",
                        float(rec_per_thr[k]),
                        batch_size=max(n_pos, 1),
                    )
            if f1_per_thr[k] is not None:
                self.log(f"val/ord_t_{tkey}_f1", float(f1_per_thr[k]), batch_size=num_val)
            self.log(f"val/ord_t_{tkey}_support_pos", float(n_pos), batch_size=num_val)
            self.log(f"val/ord_t_{tkey}_support_neg", float(n_neg), batch_size=num_val)

        # --- Per-occlusion-bin ordinal subgroup metrics ---------------
        if bins is not None and targets_np is not None:
            for i in range(len(bins) - 1):
                lo, hi = float(bins[i]), float(bins[i + 1])
                last_bin = i == len(bins) - 2
                mask = (targets_np >= lo) & (targets_np <= hi if last_bin else targets_np < hi)
                label = f"{lo:.2f}_{hi:.2f}"
                self._log_ord_subgroup(
                    f"val/ord/bin_{label}", preds_bin, y_bin, thresholds.tolist(), mask
                )

        # --- Aggregated high-occlusion (target >= 0.40) ---------------
        if targets_np is not None:
            mask = targets_np >= 0.40
            self._log_ord_subgroup(
                "val/ord/high_occ_0.40_1.00",
                preds_bin,
                y_bin,
                thresholds.tolist(),
                mask,
                emit_high_threshold_recall=True,
            )

        # --- Per-gender ordinal subgroup metrics ---------------------
        if gender_str is not None:
            for g_val, g_label in (
                (self._female_value, "female"),
                (self._male_value, "male"),
            ):
                mask = gender_str == g_val
                self._log_ord_subgroup(
                    f"val/ord/{g_label}",
                    preds_bin,
                    y_bin,
                    thresholds.tolist(),
                    mask,
                    emit_high_threshold_recall=True,
                )

        # --- Per-database ordinal subgroup metrics --------------------
        if databases is not None:
            for database in sorted(np.unique(databases)):
                mask = databases == database
                self._log_ord_subgroup(
                    f"val/ord/database/{database}",
                    preds_bin,
                    y_bin,
                    thresholds.tolist(),
                    mask,
                )

    def _log_ord_subgroup(
        self,
        prefix: str,
        preds_bin: np.ndarray,
        y_bin: np.ndarray,
        thresholds: list[float],
        mask: np.ndarray,
        *,
        emit_high_threshold_recall: bool = False,
    ) -> None:
        """Log a compact ordinal stat block for one subgroup mask.

        Always logs ``{prefix}_count``. When the subgroup is non-empty we add
        averaged threshold accuracy and F1 (and optionally per-threshold
        recall at 0.40/0.60 for high-occlusion or per-gender slices).
        """
        n = int(mask.sum())
        self.log(f"{prefix}_count", float(n), batch_size=max(n, 1))
        if n == 0:
            return
        acc, _prec, rec, f1 = _per_threshold_prf(preds_bin[mask], y_bin[mask])
        self.log(
            f"{prefix}_threshold_acc_mean",
            float(_safe_mean(acc)),
            batch_size=n,
        )
        self.log(
            f"{prefix}_threshold_f1_mean",
            float(_safe_mean(f1)),
            batch_size=n,
        )
        if emit_high_threshold_recall:
            for k, t in enumerate(thresholds):
                if abs(t - 0.40) < 1e-6 and rec[k] is not None:
                    self.log(f"{prefix}_recall_t_0.40", float(rec[k]), batch_size=n)
                if abs(t - 0.60) < 1e-6 and rec[k] is not None:
                    self.log(f"{prefix}_recall_t_0.60", float(rec[k]), batch_size=n)

    def _log_consistency_val_metrics(self, num_val: int) -> None:
        """Log ``val/cons_loss``, the mean q-r gap, and per-threshold gaps."""
        if not self._cons_loss_enabled:
            return
        logits_list = [
            b["ordinal_logits"] for b in self._val_buffer if b["ordinal_logits"] is not None
        ]
        if not logits_list:
            return
        ord_logits = torch.cat(logits_list, dim=0)
        preds_t = torch.cat([b["preds"] for b in self._val_buffer], dim=0)
        thresholds = self.model.ordinal_thresholds.detach().cpu()

        # The evaluation loss always uses ``symmetric`` so the value is
        # comparable across modes; gradient direction only matters during train.
        cons_loss = regression_ordinal_consistency_loss(
            y_pred=preds_t,
            ordinal_logits=ord_logits,
            thresholds=thresholds,
            temperature=self._cons_temperature,
            mode="symmetric",
        )
        self.log("val/cons_loss", float(cons_loss), batch_size=num_val)

        q = torch.sigmoid(ord_logits)
        r = torch.sigmoid((preds_t.view(-1, 1) - thresholds.view(1, -1)) / self._cons_temperature)
        gap = (q - r).abs()
        self.log("val/cons_gap_mean", float(gap.mean()), batch_size=num_val)
        # Per-threshold gap: highlights whether disagreement concentrates at
        # rare high thresholds.
        gap_per_thr = gap.mean(dim=0)
        for k, t in enumerate(thresholds.tolist()):
            self.log(
                f"val/cons_gap_t_{t:.2f}",
                float(gap_per_thr[k]),
                batch_size=num_val,
            )

    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        outputs = self(batch["image"])
        preds = outputs.y_pred
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
