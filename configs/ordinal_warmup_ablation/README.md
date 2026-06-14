# Ordinal warmup ablation

Small focused ablation that compares static vs. linearly warmed-up ordinal
auxiliary loss coefficients on top of the exposure-capped sampler.

Motivation: under the new sampler, the ordinal-only model marginally improves
the official validation score and yields cleaner raw predictions, but full-
strength ordinal supervision can disturb regression calibration and subgroup
weighted error early in training. Warmup lets training start close to the
pure regression objective and progressively introduce the ordinal
regularisation. See
[docs/occlusion_aware_contrastive_learning_approach.md](../../docs/occlusion_aware_contrastive_learning_approach.md)
section 18.X for the formula.

| Config | `losses.ordinal.enabled` | `weight` | `warmup_epochs` |
|---|:---:|:---:|:---:|
| [00_baseline.yaml](00_baseline.yaml) | false | – | – |
| [01_ordinal_w005.yaml](01_ordinal_w005.yaml) | true | 0.05 | 0 |
| [02_ordinal_w010.yaml](02_ordinal_w010.yaml) | true | 0.10 | 0 |
| [03_ordinal_w005_warmup3.yaml](03_ordinal_w005_warmup3.yaml) | true | 0.05 | 3 |
| [04_ordinal_w010_warmup3.yaml](04_ordinal_w010_warmup3.yaml) | true | 0.10 | 3 |

Consistency loss is intentionally disabled in all five variants — it remains
unstable at the weights tested so far and is out of scope for this ablation.
All configs reuse the backbone, optimizer, scheduler, sampler, augmentation,
and split of `configs/occlusion_aware_contrastive/01_ordinal_only.yaml`.

## Launch (SLURM)

```bash
bash jobs/submit_ordinal_warmup_ablation.sh
```

or, to submit a single variant:

```bash
CONFIG_PATH=configs/ordinal_warmup_ablation/03_ordinal_w005_warmup3.yaml \
  sbatch jobs/train.slurm
```

## Verifying warmup is active

Every training step logs `train/lambda_ord` (and `train/lambda_cons` when
consistency is enabled). Inspect `metrics.csv` in the run directory: with
`weight=0.05, warmup_epochs=3` the column should rise as
`0.01667, 0.03333, 0.05, 0.05, ...`.
