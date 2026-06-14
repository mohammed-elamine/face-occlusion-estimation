# Occlusion-Aware Contrastive Learning Configs

These configs validate the effect of **Stage 1** (ordinal occlusion-bin head)
and **Stage 2** (regression–ordinal consistency loss) from the
occlusion-aware contrastive learning approach. They run on top of the current
strong ConvNeXt-Small baseline with the gender × occlusion balanced batch
sampler.

All five configs share the same backbone, optimizer, scheduler, data,
sampler, augmentation, and training schedule. They differ only in the
multi-task loss switches under `model.*` and `losses.*`, so any change in
validation metrics is attributable to those switches.

Each config uses:

```yaml
model:
  backbone: convnext_small.fb_in22k_ft_in1k
```

Experiment folders and logger runs are named with the
`oacl_stage12_convnext_small_*` prefix so the approach stage and backbone are
visible in saved outputs.

## Configs

| File | `use_ordinal_head` | `losses.ordinal.enabled` | `losses.consistency.enabled` | `consistency.mode` | `consistency.weight` |
| --- | --- | --- | --- | --- | --- |
| `00_baseline.yaml` | false | false | false | — | — |
| `01_ordinal_only.yaml` | true | true | false | — | — |
| `02_ordinal_consistency_symmetric.yaml` | true | true | true | symmetric | 0.05 |
| `03_ordinal_consistency_ordinal_teacher.yaml` | true | true | true | ordinal_as_teacher | 0.05 |
| `04_ordinal_consistency_symmetric_low_weight.yaml` | true | true | true | symmetric | 0.01 |

## What each config tests

- **`00_baseline`** — Stage 0 reference. Pure regression with the challenge-weighted MSE.
- **`01_ordinal_only`** — Adds the ordinal head + threshold-weighted BCE. Asks: does
  forcing the encoder to recognize coarse occlusion regimes improve the regression
  score, especially on high-occlusion bins?
- **`02_ordinal_consistency_symmetric`** — Adds Stage 2's soft MSE consistency between
  the sigmoid of the ordinal logits and the regression-implied threshold probabilities.
  Symmetric mode lets both branches receive gradient; safest default.
- **`03_ordinal_consistency_ordinal_teacher`** — Same loss in `ordinal_as_teacher` mode:
  the ordinal probabilities are detached, so the regression head is *pulled toward* the
  ordinal head. Tests whether the easier ordinal task can guide regression.
- **`04_ordinal_consistency_symmetric_low_weight`** — Same as `02` but with
  `consistency.weight: 0.01`. Useful if the default weight degrades calibration.

## Recommended metrics to compare

Reported by the Lightning module on each validation epoch:

```
val/score
val/loss
val/err_mean
val/err_male
val/err_female
val/gender_gap
val/bin_0.40_0.60_err
val/bin_0.60_1.00_err
val/high_occ_0.40_1.00_err
val/high_occ_0.40_1.00_mae
val/high_occ_0.40_1.00_bias
val/high_occ_0.40_1.00_count
val/ord_loss
val/ord_threshold_acc_mean
val/ord_threshold_precision_mean
val/ord_threshold_recall_mean
val/ord_threshold_f1_mean
val/ord_t_{t}_{acc,precision,recall,f1,support_pos,support_neg}   # per threshold
val/ord_high_threshold_recall_0.40
val/ord_high_threshold_recall_0.60
val/ord/bin_{lo}_{hi}_{count,threshold_acc_mean,threshold_f1_mean}
val/ord/high_occ_0.40_1.00_{count,threshold_acc_mean,threshold_f1_mean,recall_t_0.40,recall_t_0.60}
val/ord/{female,male}_{count,threshold_acc_mean,threshold_f1_mean,recall_t_0.40,recall_t_0.60}
val/ord/database/{db}_{count,threshold_acc_mean,threshold_f1_mean}
val/cons_loss
val/cons_gap_mean
val/cons_gap_t_{t}   # per threshold
```

Auxiliary losses also support optional epoch-based linear warmup via
`losses.{ordinal,consistency}.{warmup_epochs,warmup_start_weight}`
(defaults: `0, 0.0` → exact static behaviour). When active, the effective
coefficient is logged each epoch as `train/lambda_ord` / `train/lambda_cons`.
See [docs/occlusion_aware_contrastive_learning_approach.md §17.1](../../docs/occlusion_aware_contrastive_learning_approach.md)
and the dedicated [ordinal_warmup_ablation](../ordinal_warmup_ablation/)
experiment grid.

> **High-occlusion bins (`0.40_0.60`, `0.60_1.00`) often contain very few
> validation samples**, so single-epoch numbers there are noisy. The fine-grained
> bins are still reported, but `val/high_occ_0.40_1.00_*` aggregates all
> samples with `y >= 0.40` to reduce variance when the extreme high-occlusion
> bin contains very few samples. Compare trends across epochs (e.g.
> best-of-last-N) rather than single best epochs, and prefer the per-gender
> combined `val/score` when ranking configs.

## How to launch

Locally:

```bash
python -m scripts.training.train --config configs/occlusion_aware_contrastive/00_baseline.yaml
python -m scripts.training.train --config configs/occlusion_aware_contrastive/01_ordinal_only.yaml
python -m scripts.training.train --config configs/occlusion_aware_contrastive/02_ordinal_consistency_symmetric.yaml
python -m scripts.training.train --config configs/occlusion_aware_contrastive/03_ordinal_consistency_ordinal_teacher.yaml
python -m scripts.training.train --config configs/occlusion_aware_contrastive/04_ordinal_consistency_symmetric_low_weight.yaml
```

On the SLURM cluster:

```bash
CONFIG_PATH=configs/occlusion_aware_contrastive/00_baseline.yaml sbatch jobs/train.slurm
CONFIG_PATH=configs/occlusion_aware_contrastive/01_ordinal_only.yaml sbatch jobs/train.slurm
CONFIG_PATH=configs/occlusion_aware_contrastive/02_ordinal_consistency_symmetric.yaml sbatch jobs/train.slurm
CONFIG_PATH=configs/occlusion_aware_contrastive/03_ordinal_consistency_ordinal_teacher.yaml sbatch jobs/train.slurm
CONFIG_PATH=configs/occlusion_aware_contrastive/04_ordinal_consistency_symmetric_low_weight.yaml sbatch jobs/train.slurm
```

SLURM logs land in:

```
outputs/slurm_logs/<RUN_NAME>/face_occ_train_<JOB_ID>.{out,err}
```

where `<RUN_NAME>` defaults to the config file basename, e.g. `01_ordinal_only`. Pass
`RUN_NAME=oacl_stage12_convnext_small_01_ordinal_only` if you want Slurm log
folders to match the experiment name exactly.

Experiment folders themselves always use the config's `experiment.name`, e.g.
`oacl_stage12_convnext_small_01_ordinal_only`.
