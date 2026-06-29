# Ordinal head

Exemplar config (`01_ordinal_w005.yaml`) for the **ordinal-threshold head**: a per-threshold
weighted BCE on `P(y > t_k)` trained alongside the regression head (with optional loss warmup), which
reframes regression as a set of monotone threshold classifiers. See
[`docs/architecture/04-models.md`](../../docs/architecture/04-models.md) (ordinal head) and
[`05-training.md`](../../docs/architecture/05-training.md) (loss + warmup).

```bash
python -m scripts.training.train --config configs/ordinal_warmup_ablation/01_ordinal_w005.yaml
```

Finding: `ns`-to-mild on the metric. The weight/warmup sweep was pruned after the study.
