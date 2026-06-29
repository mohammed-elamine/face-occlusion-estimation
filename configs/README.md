# Configs

Every experiment in this project is **one YAML file**, not new code: `config + data + split → one
self-contained run folder`. To try something, copy `baseline.yaml`, change `experiment.name` and the
fields you care about, and train:

```bash
python -m scripts.training.train --config configs/<your-config>.yaml
```

## Layout

| Path | What lives here |
|------|-----------------|
| `baseline.yaml`      | The canonical model — ConvNeXt-Small, full fine-tune. Start here. |
| `ensemble/`          | The members that, averaged with the baseline, give our best submission. |
| `experiments/`       | One runnable exemplar per method we explored (the findings, kept or rejected). |
| `eval/`              | Non-training helpers (e.g. the held-out test occlusion distribution). |

Only the essential configs are kept; the dozens of one-off ablation variants were pruned. Their
**findings** are summarised below and in `docs/architecture/`, and any exact past config is
recoverable from git history.

## `baseline.yaml`

ConvNeXt-Small (`convnext_small.fb_in22k_ft_in1k`), full fine-tune, linear head, pooled
weighted-MSE, mild augmentation. Validation score **0.00129**. This is the champion single model and
the paired-Δ reference every other config is compared against.

## `ensemble/` — reproduces the best submission

Averaging the clipped predictions of `baseline.yaml` + these four diverse-but-tied members gives
**val 0.00118 (leaderboard ≈ 0.00112)** — the one change that was significant beyond noise.

| Config | One change from baseline | Single-model score |
|--------|--------------------------|--------------------|
| `convnext_base.yaml`    | larger ConvNeXt-Base backbone        | 0.00133 (ns) |
| `gender_balanced.yaml`  | gender-balanced loss                 | 0.00128 (ns) |
| `sigmoid.yaml`          | sigmoid output activation            | 0.00133 (ns) |
| `expectation_dldl.yaml` | DLDL/DEX ordered-bin expectation head | ns single, but the **decorrelated** member that moved the ensemble 0.00121 → 0.00118 |

The members tie as single models; the **diversity** is what helps. Each is just `baseline.yaml` with
one field flipped.

## `experiments/` — methods we explored

One exemplar config per idea. Full write-ups in `docs/architecture/`.

| Config | Method | Finding |
|--------|--------|---------|
| `dinov2_lora.yaml`           | DINOv2 ViT-B backbone + LoRA fine-tune        | strong alternative backbone; adds ensemble diversity |
| `shadow_head.yaml`           | auxiliary "shadow" head                       | rejected — ns-to-worse |
| `gender_invariant.yaml`      | gender-adversarial invariance                 | rejected — closes the gap only by levelling down |
| `synthetic_ranking.yaml`     | RankNet over synthetic `clean < mild < strong` | improves ordering, not the metric |
| `distribution_reweight.yaml` | distribution-aware sample reweighting          | bulk-neutral; the sparse high-occ tail stays unfixed |
| `ordinal_head.yaml`          | ordinal threshold head (warmup)               | ns; kept as a method exemplar |
| `ordinal_consistency.yaml`   | regression ↔ ordinal consistency loss         | ns; kept as a method exemplar |
| `background_invariance.yaml` | perturb only non-face pixels + consistency    | rejected — Δ +0.000115 significantly worse |
| `balanced_sampler.yaml`      | gender × occlusion balanced-batch sampler     | exemplar (ConvNeXt-Tiny); see `docs/balanced_batch_sampler.md` |

## `eval/`

`test_distribution.yaml` — the held-out test occlusion distribution. Not a training run; consumed by
the distribution-reweighting config and the evaluation lenses.

> Compare any run against the baseline with `scripts.analysis.bootstrap_metrics` (CI-first — the
> high-occlusion tail is tiny and identity-leaked, so never trust a raw tail delta).
