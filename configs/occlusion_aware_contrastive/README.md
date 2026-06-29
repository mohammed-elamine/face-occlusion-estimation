# Ordinal + regression consistency

Exemplar config (`02_ordinal_consistency_symmetric.yaml`) for the **regression↔ordinal consistency**
loss: an ordinal head plus a term that matches the regression head's implied threshold
probabilities `σ((ŷ − t_k)/T)` to the ordinal head's, so the two heads agree. See
[`docs/occlusion_aware_contrastive_learning_approach.md`](../../docs/occlusion_aware_contrastive_learning_approach.md)
and [`docs/architecture/04-models.md`](../../docs/architecture/04-models.md).

```bash
python -m scripts.training.train --config configs/occlusion_aware_contrastive/02_ordinal_consistency_symmetric.yaml
```

Finding: `ns` on the metric. The stage / teacher-mode variants were pruned after the study.
