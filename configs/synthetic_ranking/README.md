# Synthetic monotonic ranking

Exemplar config (`04_ranking_masks_hands.yaml`) for the **synthetic-occlusion ranking** objective:
a RankNet loss over synthetic `clean < mild < strong` occlusion views (masks + hands) teaches the
model to *order* occlusion severity. The synthetic views feed only the ranking head — the regression
label is never changed. Prereq: build the synthetic cache (see
[`docs/synthetic_occlusion_generation.md`](../../docs/synthetic_occlusion_generation.md)).

```bash
python -m scripts.training.train --config configs/synthetic_ranking/04_ranking_masks_hands.yaml
```

Finding: it improved the rank-ordering diagnostics but **not** the metric within CI (the
high-occlusion tail is too rare to supervise). The full sweep was pruned after the study; the
mechanism is documented in `docs/architecture/`.
