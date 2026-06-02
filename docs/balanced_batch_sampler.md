# Soft Gender × Occlusion Balanced Batch Sampler

## Motivation

The challenge metric is a **gender-aware weighted MSE**: high occlusion values receive larger weights, female and male errors are computed separately, and the final score penalizes the error gap between genders.

Our baseline model performs well on common low-occlusion images but strongly underpredicts rare high-occlusion examples. A naive fix would be to oversample high-occlusion images. However, high-occlusion samples may be disproportionately associated with one gender (e.g. hair covering the face increases occlusion scores for females). Sampling only by occlusion bin risks teaching the model a shortcut:

```
gender-like visual cues → high occlusion
```

We want the model to learn:

```
true visibility / occlusion / degradation cues → occlusion score
```

## Why not occlusion-only sampling?

If high-occlusion images are mostly female, an occlusion-only sampler would flood each batch with female high-occlusion examples. The model could learn to associate female-presenting features with high occlusion, hurting generalization and widening the gender error gap — the exact thing the challenge metric penalizes.

## Strategy

We build each training batch from multiple **gender × occlusion_bin** strata. Each stratum gets a sampling probability based on:

1. **Occlusion bin weight** — higher bins receive more exposure.
2. **Gender correction** — inside each bin, the minority gender is softly upweighted using an inverse-frequency correction:

   ```
   gender_correction = (n_bin / n_stratum) ^ gender_balance_strength
   ```

   - `gender_balance_strength = 0` → no correction
   - `gender_balance_strength = 0.5` → square-root correction (default, recommended)
   - `gender_balance_strength = 1.0` → full inverse-frequency correction

3. **Weight clipping** — `max_stratum_weight` prevents tiny strata from being sampled too aggressively.

This is **soft balancing**: we increase exposure to rare groups without forcing an artificial uniform distribution.

### Batch construction

For each slot in a batch, the sampler:
1. Picks a stratum according to its probability.
2. Draws one random sample from that stratum.

This gives stochastic batch-level diversity without rigid quotas.

## Configuration

Add a `sampler` section to your YAML config:

```yaml
sampler:
  enabled: true
  strategy: gender_occlusion_balanced_batch
  bins: [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]
  bin_weights:
    0.00_0.05: 1.0
    0.05_0.10: 1.2
    0.10_0.20: 1.5
    0.20_0.40: 2.0
    0.40_0.60: 3.0
    0.60_1.00: 4.0
  gender_balance_strength: 0.5
  max_stratum_weight: 8.0
  min_stratum_size: 5
  num_samples: null    # null = len(train_dataset)
  seed: 42
```

| Field | Description |
|---|---|
| `enabled` | Set to `true` to activate the sampler. Default `false` preserves baseline behavior. |
| `strategy` | Must be `gender_occlusion_balanced_batch`. |
| `bins` | Occlusion bin edges. Must be sorted and have ≥ 2 values. |
| `bin_weights` | Relative importance per bin. Keys are `"lo_hi"` labels matching consecutive edges. |
| `gender_balance_strength` | How strongly to rebalance gender inside each bin (0–1). |
| `max_stratum_weight` | Cap on any stratum weight to prevent extreme oversampling. |
| `min_stratum_size` | Warn if a stratum has fewer samples than this. |
| `num_samples` | Samples per epoch. `null` = dataset size. |
| `seed` | Local RNG seed for reproducibility. |

## How to run

1. Copy the baseline config and enable the sampler:

   ```bash
   cp configs/baseline.yaml configs/convnext_small_gender_occ_sampler.yaml
   ```

   Or use the provided `configs/convnext_small_gender_occ_sampler.yaml`.

2. Train:

   ```bash
   python scripts/train.py --config configs/convnext_small_gender_occ_sampler.yaml
   ```

   Or on a cluster:

   ```bash
   CONFIG_PATH=configs/convnext_small_gender_occ_sampler.yaml sbatch jobs/train.slurm
   ```

3. The sampler prints a summary at startup and saves `reports/sampler_summary.json` in the experiment directory.

## What to monitor

Compare against the baseline using the same validation split:

| Metric | What to look for |
|---|---|
| Official challenge score | Should improve or stay similar. |
| `err_female` / `err_male` | Both should improve. |
| Gender gap | Should decrease or stay stable. |
| Bias by occlusion bin | High-occlusion error should drop. |
| Bias by gender × occlusion bin | No single cell should dominate. |
| High-occlusion worst errors | Fewer severe underpredictions. |
| Low-occlusion overprediction | Should not increase significantly. |

## Risks

- **Oversampling rare noisy samples.** Tiny strata with label noise will be seen more often. Mitigated by `max_stratum_weight` and `min_stratum_size`.
- **Overfitting high-occlusion outliers.** With replacement sampling, the same few images may repeat. Monitor validation loss for divergence.
- **Hurting low-occlusion performance.** Less time on the majority class. If low-occlusion error rises, reduce `bin_weights` for high bins or lower `gender_balance_strength`.
- **Artificial batch distribution.** Batches no longer reflect the true data distribution. Batch norm statistics may shift. Monitor training stability.
