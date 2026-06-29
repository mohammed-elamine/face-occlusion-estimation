# Exposure-Capped Soft Gender × Occlusion Balanced Batch Sampler

## Overview

An optional training sampler that over-represents the rare high-occlusion strata
(per gender) so the model sees the hard regime more often. The key safety knob is
`max_repeats_per_image`, which caps how many times any one image appears per epoch
so boosting tiny strata can't become label-noise memorization. Enable it when the
high-occlusion tail is starved and the gender gap is wide; leave it off otherwise.

For the high-level "where samplers fit" picture, see the *Samplers* section of
[architecture/03-data.md](architecture/03-data.md); this doc is the deep dive.

## Motivation

The challenge metric is a **gender-aware weighted MSE**: high-occlusion samples
receive larger weights, female and male errors are computed separately, and the
final score penalizes the error gap between genders. Our baseline model is
strong on the common low-occlusion regime and weak on the rare
high-occlusion regime, especially when one gender dominates that regime.

Naïvely oversampling rare strata in a stratum-balanced sampler creates two new
problems:

1. **Label-noise amplification.** A `(gender × bin)` stratum with only 2 images
   could be sampled hundreds of times per epoch, causing the model to memorize
   noisy labels in tiny strata.
2. **Shortcut learning.** If high-occlusion images are mostly one gender, the
   model can learn the shortcut "gender-like cues → high occlusion", which
   widens the very gap the metric penalizes.

This sampler addresses both: it provides **moderate, capped** exposure to rare
real samples, combined with an inverse-frequency gender correction. It is not
meant to solve high-occlusion rarity alone — the synthetic-ranking feature
supplies the additional diversity and relative supervision.

This supersedes an earlier uncapped per-slot sampler whose tiny strata could be
drawn dozens of times per epoch (rare-real exposure silently becoming label-noise
overfitting). All shipped configs already use the current schema below.

## Design

For every `(gender, occlusion_bin)` stratum `s` with `n_s` images and bin
weight `w_b`:

1. **Per-stratum weight.** A raw weight combines the bin weight and an
   inverse-frequency gender correction:

   $$ a_s^{\text{raw}} = w_b \cdot \left(\frac{n_b}{n_s}\right)^{\gamma_{\text{gender}}} $$

2. **Size-aware damping** (`size_aware_weighting: true`). Tiny strata have noisy
   weight estimates, so we shrink their boost toward 1:

   $$ R_s = \min\left(1, \frac{n_s}{n_{\text{reliable}}}\right), \qquad
      a_s^{\text{safe}} = 1 + R_s \cdot (a_s^{\text{raw}} - 1) $$

   The boost is then clipped: `a_s = min(a_s_safe, max_stratum_weight)`.

3. **Mixed sampling distribution.** A balanced probability over strata
   (proportional to `a_s`, not `a_s · n_s`) is mixed with the natural
   distribution via `balance_strength` $\alpha$:

   $$ p_s^{\text{nat}} = \frac{n_s}{N}, \qquad
      p_s^{\text{bal}} = \frac{a_s}{\sum_{s'} a_{s'}}, \qquad
      p_s = (1-\alpha) p_s^{\text{nat}} + \alpha p_s^{\text{bal}} $$

4. **Per-image repeat cap** (`max_repeats_per_image: r_max`). The desired
   number of draws per stratum is `D_s = N_{\text{epoch}} \cdot p_s`, but a
   stratum can never be sampled more than `n_s \cdot r_max` times in an epoch.
   Any leftover budget is redistributed proportionally to non-saturated
   strata; if every stratum is saturated, the epoch is shorter than requested
   and the sampler emits a warning.

5. **Epoch construction.** For each stratum we build a pool by repeating its
   indices up to `r_max` times, shuffle, and take the prescribed count. All
   per-stratum samples are concatenated, globally shuffled, and yielded as
   contiguous batches of `batch_size`. This guarantees no image appears more
   than `r_max` times per epoch.

## Configuration schema

```yaml
sampler:
  enabled: true
  strategy: gender_occlusion_balanced_batch
  bins: [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]
  bin_weights:
    0.00_0.05: 1.0
    0.05_0.10: 1.1
    0.10_0.20: 1.2
    0.20_0.40: 1.5
    0.40_0.60: 2.0
    0.60_1.00: 2.5
  balance_strength: 0.3          # mix between natural (0) and balanced (1)
  gender_balance_strength: 0.5   # exponent γ of the inverse-frequency gender correction
  max_stratum_weight: 8.0        # hard clip on a_s before mixing
  min_stratum_size: 5            # warn-only threshold for very small strata
  size_aware_weighting: true     # damp boosts of small strata toward 1
  reliable_stratum_size: 20      # size at which a stratum is considered "reliable"
  max_repeats_per_image: 10      # hard per-image cap (key safety knob)
  target: bin_weights            # bin_weights | balanced | test_matched (lens-targeting)
  clip_max: 10.0                 # cap on lens importance weights (balanced/test_matched)
  drop_last: false               # drop the trailing partial batch? (shipped configs: false)
  num_samples: null              # null = len(train_dataset)
  seed: 42
```

| Field | Description |
|---|---|
| `enabled` | Activate the sampler (otherwise the standard shuffled loader is used). |
| `bins` | Occlusion bin edges. Strictly increasing, length ≥ 2. |
| `bin_weights` | Bin importance. Keys are `"lo_hi"` labels with two decimals. |
| `balance_strength` | 0 reproduces natural sampling, 1 is fully balanced; defaults to a gentle 0.3. |
| `gender_balance_strength` | Exponent γ ∈ [0, 1] of the gender correction; 0.5 is square-root (default), 1.0 is full inverse-frequency. |
| `max_stratum_weight` | Hard clip on `a_s`. |
| `min_stratum_size` | Emits a warning when a stratum is smaller than this but still drawn. |
| `size_aware_weighting` | Shrinks weights of small strata toward 1 to avoid noisy estimates. |
| `reliable_stratum_size` | Size at which a stratum gets its full boost. |
| `max_repeats_per_image` | Hard upper bound on how many times any single image can appear per epoch (the memorization guard; always enforced). |
| `target` | Per-bin weighting source: `bin_weights` (the configured dict) or `balanced` / `test_matched`, which reuse the eval-lens operator (`per_bin_importance_weights`) so the sampler targets the **same** distribution as the loss reweighting and the evaluation lenses. |
| `clip_max` | Cap on the lens importance weights when `target` is `balanced` / `test_matched`. |
| `drop_last` | Whether to drop the trailing partial batch. The shipped configs set `false` (keep every sampled image). |
| `num_samples` | Target epoch length before capping. `null` uses dataset size. |
| `seed` | RNG seed for reproducibility. |

## Worked example

Suppose `n = 1000` images and the `(female, 0.60_1.00)` stratum has only `n_s = 2`
samples while the high-bin total is `n_b = 4`. With `bin_weights["0.60_1.00"] = 2.5`,
`gender_balance_strength = 0.5`, `size_aware_weighting = false`, and
`balance_strength = 1.0`:

```
raw weight a_s     = 2.5 · (4 / 2) ** 0.5     ≈ 3.54
effective weight   = min(3.54, 8.0)           = 3.54
p_balanced(s)      ≈ 3.54 / Σ a_s             (large)
desired draws D_s  = 1000 · p_s               ≈ 250
hard cap n_s · 10  = 20
draws_after_cap    = 20   ← capped
```

The sampler logs:

```
WARNING: Stratum (gender=0 [female], bin=0.60_1.00) has only 2 samples.
Desired draws before cap: 250.0. Capped to 20 draws using max_repeats_per_image=10.
```

The leftover 230 draws are redistributed across non-saturated strata,
proportionally to their `p_s`.

## How to run

```bash
# Local
.venv/bin/python -m scripts.training.train --config configs/experiments/balanced_sampler.yaml

# Cluster
CONFIG_PATH=configs/experiments/balanced_sampler.yaml sbatch jobs/train.slurm
```

At startup the sampler logs a per-stratum table and writes
`reports/sampler_summary.json` inside the experiment run directory.

## What the summary records

The JSON `summary` exposes per-stratum diagnostics plus epoch-level fields:

- Per stratum: `count`, `natural_prob`, `balanced_prob`, `final_prob`,
  `raw_weight`, `effective_weight`, `expected_draws_before_cap`,
  `draws_after_cap`, `expected_repeats_per_image`, `was_capped`,
  `p_seen_at_least_once`.
- Global: `num_samples_requested`, `num_samples_actual`, `num_batches`,
  `max_expected_repeats_per_image`, `num_capped_strata`.

A non-zero `num_capped_strata` is a clear signal that the rare strata are
genuinely tiny and that the cap is doing useful work.

## What to monitor

| Metric | Expectation |
|---|---|
| Official challenge score | Improves or stays similar. |
| `err_female` / `err_male` | Both improve; gender gap narrows. |
| Bias by occlusion bin | High-bin error drops without low-bin regressions. |
| Bias by `gender × bin` | No single cell dominates. |
| `max_expected_repeats_per_image` in `sampler_summary.json` | Stays ≤ `max_repeats_per_image` and well below "memorization" territory. |
| `num_capped_strata` | Non-zero indicates the cap is protecting tiny strata as intended. |

## Risks and trade-offs

- **Reduced bias-correction strength.** Softer defaults mean the sampler alone
  will *not* close the high-occlusion gap. This is intentional — the
  synthetic-occlusion and ranking features supply additional signal.
- **Capped strata stay underexposed.** The cap is a safety net, not a magic
  fix. If `num_capped_strata` is large, consider increasing
  `max_repeats_per_image` cautiously, or expanding the data via synthetic
  occlusion before tuning the sampler further.
- **Batch statistics shift.** Batch-norm statistics still drift slightly from
  the natural distribution. Mitigated by `balance_strength = 0.3` and by group
  norm in modern backbones.

## How it fits with synthetic occlusion / ranking

This sampler is the data-side complement of the regression + ordinal heads: it
reshapes *which real images* the loss sees. When the synthetic-occlusion feature
is on, its generated views populate the currently-tiny strata, so the cap becomes
less binding and the sampled distribution softens back toward natural. The two are
orthogonal and can be enabled together.
