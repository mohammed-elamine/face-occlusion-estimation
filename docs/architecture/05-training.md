# 05 — Training & Losses

`src/face_occlusion/training/lit_module.py` holds `FaceOcclusionLitModule`, the
LightningModule that orchestrates the loss stack, optimization, and validation metrics.
`training/callbacks.py` builds the Trainer callbacks. The CLI driver is
`scripts/training/train.py` ([07](07-pipeline-and-experiments.md)).

## Loss helpers (module-level)

- `weighted_mse_loss(preds, targets, sample_weight=None)` — the base objective with the
  metric weight `w = 1/30 + y`. With `sample_weight=None` it is byte-identical to the
  official metric's per-row error; an optional `sample_weight` applies distribution-aware
  reweighting.
- `gender_balanced_weighted_mse_loss(preds, targets, genders, *, female_value, male_value,
  high_occ_power=1.0, gap_lambda=0.0)` — mirrors the **challenge metric's structure**:
  computes `Err_g = Σ_g w(ŷ−y)² / Σ_g w` with `w = 1/30 + y^high_occ_power`, then returns
  `0.5(Err_F + Err_M) + gap_lambda·|Err_F − Err_M|`. A batch missing one gender falls back
  to the present gender's error (drops the gap term).
- `_scheduled_loss_weight(target_weight, warmup_epochs, warmup_start_weight, current_epoch)`
  — linear per-epoch warmup for any auxiliary coefficient (validated as warmup-only, no
  cooldown).

## The multi-task loss stack

`losses.regression.type` selects the base term (`weighted_mse` or `gender_balanced`).
Auxiliary terms are each gated, warmed independently, and summed:

```
loss = λ_reg · loss_reg
     + λ_ord  · loss_ord     (ordinal weighted BCE,            losses.ordinal)
     + λ_cons · loss_cons    (regression↔ordinal consistency,  losses.consistency)
     + λ_mono · loss_mono    (ordinal monotonicity hinge,      losses.monotonicity)
     + λ_rank · loss_rank    (synthetic monotonic ranking,     losses.ranking)
```

Each `λ` is the **effective** (post-warmup) weight from an `_effective_*_weight()` method
that calls `_scheduled_loss_weight` with `self.current_epoch`. Each `_compute_*_loss(...)`
returns `None` when its feature is disabled, so the baseline run computes only `loss_reg`.

Gating and key config (see [02](02-configuration.md), [04](04-models.md) for the heads):

| Term | Gate | Key knobs |
|------|------|-----------|
| regression | always | `type`, `weight`, `high_occ_power`, `gender_gap_lambda`, `reweight`, warmup |
| ordinal | `model.use_ordinal_head` + `losses.ordinal.enabled` | `weight`, `threshold_weights`, warmup |
| consistency | `use_ordinal_head` + `losses.consistency.enabled` | `weight`, `temperature`, `mode`, warmup |
| monotonicity | `use_ordinal_head` + `losses.monotonicity.enabled` | `weight`, warmup |
| ranking | `losses.ranking.enabled` (+ synthetic views in batch) | `weight`, warmup |

Misconfiguration (a coupled loss enabled without the ordinal head) raises `ValueError` at
`__init__`.

### Distribution-aware reweighting (`losses.regression.reweight`)

`reweight ∈ {none, balanced, test_matched}`. When not `none`, the module precomputes
**per-bin importance weights** from the training target histogram via `metrics/eval_lenses.py`
(`balanced_proportions` / `load_test_distribution` + `per_bin_importance_weights`), requiring
the full `train_targets` array passed to `__init__`. At train time `_regression_sample_weight`
maps each sample to its bin weight and blends with all-ones through a warmup
(`iw_eff = (1−λ) + λ·iw`), so the loss can ramp from the official metric toward the
reweighted objective. This shares its target distribution with the sampler's
`target=balanced|test_matched` and the eval lenses.

## `training_step` / `validation_step`

- `training_step` — one forward pass; computes the base regression loss
  (`weighted_mse_loss` or `gender_balanced_weighted_mse_loss`), then each enabled auxiliary
  via its `_compute_*` helper; logs every active term, its effective weight, `train/lr`, and
  ranking diagnostics. The **ranking** term runs the model on the stacked
  `synthetic_clean/mild/strong` images and applies `monotonic_ranking_loss` over the
  `synthetic_valid` subset.
- `validation_step` — forward, clip predictions to `[0,1]`, compute the metric-convention
  weighted MSE, and **buffer** predictions/targets/genders/(ordinal logits)/metadata. bf16
  outputs are cast with `.float()` before CPU/numpy transfer (bfloat16 has no numpy dtype).
- `on_validation_epoch_end` — concatenate the buffer and compute `val/score` via
  `challenge_score` ([06](06-metrics-and-evaluation.md)); log global + per-bin + per-gender +
  per-database errors, the high-occlusion aggregate (`target ≥ 0.40` with gender splits),
  and ordinal/consistency diagnostics when enabled. `val/score` is the checkpoint monitor.
- `predict_step` — returns predictions + image metadata (and gender if present), `.float()`
  for bf16.

## Optimization — `configure_optimizers`

- **Discriminative LRs:** if both `training.head_lr` and `training.backbone_lr` are set and
  the model exposes `param_groups`, build `AdamW(model.param_groups(head_lr, backbone_lr,
  weight_decay))` — head at `head_lr`, backbone/LoRA at `backbone_lr`, WD on ≥2-D weights
  only ([04](04-models.md)). Otherwise a single-LR `AdamW(filter(requires_grad), lr=learning_rate)`.
- **Schedule:** if `training.warmup_frac` is set, a **per-step** `LambdaLR` does linear
  warmup → cosine over `trainer.estimated_stepping_batches` (returned with
  `interval="step"`); otherwise a per-epoch `CosineAnnealingLR(T_max=max_epochs)`.

## Callbacks — `training/callbacks.py`

`build_callbacks(cfg, checkpoint_dir)` returns:

- `ModelCheckpoint` — monitors `checkpoint.monitor` (`val/score`), `mode`, `save_top_k`,
  fixed filenames (`best.ckpt`, `last.ckpt`).
- `EarlyStopping` — `training.early_stopping_patience` epochs on the same monitor.
- `LearningRateMonitor` — per-epoch LR logging.

## Interrupt-safe finalization

`scripts/training/train.py` wraps `trainer.fit(...)` so that a `KeyboardInterrupt` or an
intermediate-epoch error still runs `_finalize(...)`: it re-validates the best checkpoint,
writes `predictions/val_predictions.csv`, and a `training_status.json`
(`status: completed|interrupted`, which checkpoint the predictions came from). `_finalize`
never raises, so a real error is re-raised afterward and stays visible — but the best-so-far
predictions are always saved.
