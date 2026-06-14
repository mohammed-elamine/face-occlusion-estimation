"""Exponential Moving Average (EMA) of model weights, as a Lightning callback.

EMA keeps a slow running average of the training weights -- a "free ensemble over the
optimization trajectory". It is the single-run stand-in for multi-seed averaging, which we
cannot afford (fixed split, no repeats): the averaged weights sit in a flatter region of the
loss surface and usually generalize slightly better, at the cost of one extra weight-copy per
step.

Integration with the rest of the pipeline:
  * The shadow weights are swapped into the model for **validation and checkpointing**, so
    ``val/score``, the best checkpoint, ``_finalize``'s re-validation, and inference
    (``predict_test``) all transparently use the EMA model. Without this swap, EMA would be
    computed but never actually used for selection or inference.
  * The swap is **in-place** (``param.data.copy_``), so the optimizer state stays bound to the
    live weights; the live weights are restored at the start of the next training epoch.
  * The shadow is saved into the checkpoint (``ema_shadow``) for resume-safety.

Enabled via ``training.ema.{enabled, decay, warmup}`` (default off -> no behaviour change).
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """Maintain an EMA of the model parameters and use it for validation/checkpointing.

    Parameters
    ----------
    decay:
        Target EMA decay; the shadow tracks ``decay * shadow + (1 - decay) * live`` per step.
        Higher = slower/smoother (e.g. 0.999 averages roughly the last ~1000 steps).
    warmup:
        If true, use a smaller decay early (``(1 + n) / (10 + n)``) so the average is not
        dominated by the noisy initial weights, ramping up to ``decay``.
    validate_on_ema:
        If true (default), swap the EMA weights into the model for validation + checkpointing.
    """

    def __init__(
        self, decay: float = 0.999, warmup: bool = True, validate_on_ema: bool = True
    ) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"ema.decay must lie in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.validate_on_ema = bool(validate_on_ema)
        self._shadow: dict[str, torch.Tensor] | None = None
        self._backup: dict[str, torch.Tensor] | None = None
        self._n_updates = 0

    # -- shadow init & update -------------------------------------------------
    def on_fit_start(self, trainer, pl_module) -> None:
        if self._shadow is None:
            self._shadow = {n: p.detach().clone() for n, p in pl_module.named_parameters()}

    def _effective_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self._n_updates) / (10.0 + self._n_updates))

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        if self._shadow is None:
            return
        d = self._effective_decay()
        for n, p in pl_module.named_parameters():
            s = self._shadow[n]
            if p.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:  # integer params (rare) are copied verbatim, not averaged
                s.copy_(p.detach())
        self._n_updates += 1

    # -- swap EMA in for validation/checkpoint, restore live for training -----
    @torch.no_grad()
    def _swap_in_ema(self, pl_module) -> None:
        if self._shadow is None or self._backup is not None:
            return
        self._backup = {}
        for n, p in pl_module.named_parameters():
            self._backup[n] = p.detach().clone()
            p.data.copy_(self._shadow[n].to(p.device))

    @torch.no_grad()
    def _restore_live(self, pl_module) -> None:
        if self._backup is None:
            return
        for n, p in pl_module.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = None

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        # Hold EMA weights through validation AND the checkpoint save that follows it.
        if self.validate_on_ema:
            self._swap_in_ema(pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # Restore the live (training) weights before the next epoch optimizes.
        self._restore_live(pl_module)

    # -- persistence (resume-safety) -----------------------------------------
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if self._shadow is not None:
            checkpoint["ema_shadow"] = {n: t.detach().cpu() for n, t in self._shadow.items()}
            checkpoint["ema_n_updates"] = self._n_updates

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        shadow = checkpoint.get("ema_shadow")
        if shadow is not None:
            self._shadow = {n: t.clone() for n, t in shadow.items()}
            self._n_updates = int(checkpoint.get("ema_n_updates", 0))
