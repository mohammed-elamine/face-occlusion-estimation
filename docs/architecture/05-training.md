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

`losses.regression.type` selects the base term: `weighted_mse`, `gender_balanced`, or `dldl`
(label-distribution learning for the distribution head — `dldl_kl_loss` on the Gaussian/LDS soft
labels + a metric-weighted MSE on the expectation; requires `model.head.type=distribution`, see
[09](09-imbalanced-regression-and-expectation-head.md)).
Auxiliary terms are each gated, warmed independently, and summed:

```
loss = λ_reg · loss_reg
     + λ_ord  · loss_ord     (ordinal weighted BCE,            losses.ordinal)
     + λ_cons · loss_cons    (regression↔ordinal consistency,  losses.consistency)
     + λ_mono · loss_mono    (ordinal monotonicity hinge,      losses.monotonicity)
     + λ_rank · loss_rank    (synthetic monotonic ranking,     losses.ranking)
     + λ_bgc  · loss_bgc     (background-invariance consistency, losses.bg_consistency)
     + λ_shd  · loss_shadow  (auxiliary shadow-fraction prediction, losses.shadow)
     + λ_adv  · loss_gadv    (gradient-reversal gender adversary,    losses.gender_adversary)
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
| bg-consistency | `losses.bg_consistency.enabled` (+ `bg_view_image` in batch) | `weight`, `loss` (l1/l2), warmup |
| shadow (aux) | `model.use_shadow_head` + `losses.shadow.enabled` (+ `shadow_target` in batch) | `weight`, `loss` (l1/l2), warmup |
| gender adversary | `model.use_gender_adversary` + `losses.gender_adversary.enabled` | `weight`, warmup; head `conditional` |

Misconfiguration (a coupled loss enabled without the ordinal head) raises `ValueError` at
`__init__`.

### Background-invariance consistency (`losses.bg_consistency`)

`_compute_bg_consistency_loss(batch, preds)` forwards a second, differently-background-
randomized view of the same face (`bg_view_image`, produced by `BackgroundAugment.make_variant`
— see [03 — Data](03-data.md)) and penalizes the prediction disagreement
`‖ŷ(view) − ŷ(bg_view)‖` (`l1` or `l2`). Both views receive gradient, pulling the
representation toward background-invariance — an explicit "ignore everything but the face"
signal that augmentation only encourages implicitly. The datamodule sets the dataset's
`return_bg_pair` when this loss is enabled **and** a face mask source exists; needs
`augmentation.background.enabled`. Logged as `train/loss_bgc` / `train/lambda_bgc`.

### Auxiliary shadow head (`model.use_shadow_head` + `losses.shadow`)

A training-only multi-task head that predicts the **within-face deep-shadow fraction**
(`dark_frac`) from the pooled encoder features. Shadow is the one image property found to
correlate with the occlusion label (ρ≈+0.18; see `tmp/model_study`), so this loss pushes the
encoder to represent illumination and predict the genuinely-shadowed tail better. `_compute_shadow_loss`
applies an `l1`/`l2` loss between `outputs.shadow_pred` and the batch's `shadow_target`, **masking
NaN rows** (images whose `dark_frac` was not precomputed). Targets are produced offline by
`scripts.data.build_shadow_targets` and merged onto the train rows by the datamodule
(`data.shadow_targets_csv`). The head is **dropped at inference** — the occlusion prediction never
reads `shadow_pred`. Logged as `train/loss_shadow` / `train/lambda_shadow`. Built on the shared
pooled features (the linear-head path switches to the same `forward_features` route as the ordinal
head); the baseline forward stays bit-identical when the head is off.

### Gender-adversary invariance (`model.use_gender_adversary` + `losses.gender_adversary`)

The proper representation-level fix for the gender gap (DFR and loss-only failed because the gender
shortcut is *entangled in the encoder features*; see `tmp/model_study/05_gender_gap.md`).
`_compute_gender_adversary_loss` runs the model's `gender_adversary` head on the **gradient-reversed**
pooled features (`grad_reverse`, `models/adversary.py`) and applies BCE against gender — so the
reversed gradient pushes the **encoder** toward gender-invariant occlusion features (DANN; Ganin
et al. 2016). When `gender_adversary.conditional` (default), the occlusion bin is one-hot-appended to
the adversary input, so only the gender info *not explained by occlusion* is removed (equalized-odds
style). The warmed `weight` ramps the reversal strength. Logged as `train/loss_gadv` /
`train/lambda_gadv` / `train/gadv_acc` (adversary accuracy should **fall toward the gender base rate**
as invariance improves). Best paired with the gender×occ balanced sampler (`sampler.enabled`) and the
metric-aligned `gender_balanced` loss (`gender_gap_lambda`) — the full recipe is
`configs/experiments/gender_invariant.yaml`. Training-only; dropped at inference.

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
  `synthetic_valid` subset. The **bg-consistency** term runs a second forward on
  `bg_view_image` and penalizes its disagreement with the main prediction.
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
- `EMACallback` (`training/ema.py`) — **optional, gated by `training.ema.{enabled, decay,
  warmup}`** (default off). Keeps an exponential moving average of the model weights — a
  "free ensemble over the optimization trajectory", our single-run stand-in for multi-seed
  averaging (which we can't afford). The shadow updates per training batch
  (`shadow = decay·shadow + (1−decay)·live`, with an optional warmup so noisy early weights
  don't dominate). Crucially it **swaps the EMA weights into the model for validation and the
  checkpoint save** (`on_validation_epoch_start`) and **restores the live weights at the next
  `on_train_epoch_start`** — so `val/score`, `best.ckpt`, `_finalize`, and `predict_test` all
  transparently use the averaged model. Swaps are in-place (optimizer stays bound to the live
  weights); the shadow is persisted in the checkpoint for resume.

## Interrupt-safe finalization

`scripts/training/train.py` wraps `trainer.fit(...)` so that a `KeyboardInterrupt` or an
intermediate-epoch error still runs `_finalize(...)`: it re-validates the best checkpoint,
writes `predictions/val_predictions.csv`, and a `training_status.json`
(`status: completed|interrupted`, which checkpoint the predictions came from). `_finalize`
never raises, so a real error is re-raised afterward and stays visible — but the best-so-far
predictions are always saved.
