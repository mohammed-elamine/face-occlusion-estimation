# Architecture Guide

A code-oriented walkthrough of the `face-occlusion-estimation` system, component by
component, for ML/DL engineers. Each chapter maps concepts to the exact files and
symbols that implement them, so you can read top-down (overview → detail) or jump to a
subsystem.

The task: supervised image regression — given one `224×224` face crop, predict a
continuous occlusion score `y ∈ [0, 1]`. The scoring metric is **gender-aware and
high-occlusion-weighted**, and that single fact drives most of the design (see
[01 — Overview](01-overview.md) and [06 — Metrics & Evaluation](06-metrics-and-evaluation.md)).

## Reading order

| # | Chapter | What it covers |
|---|---------|----------------|
| 01 | [Overview & design philosophy](01-overview.md) | The one core idea (`config + data + split → experiment folder`), repo layout, the metric, end-to-end data flow. |
| 02 | [Configuration system](02-configuration.md) | `Config` object, how YAML maps to components, the default-OFF gating philosophy, config groups. |
| 03 | [Data subsystem](03-data.md) | Splits, dataset (the per-item dict), datamodule, transforms, target normalization, samplers, synthetic occlusion, background augmentation, caches. |
| 04 | [Models](04-models.md) | `OcclusionRegressor` (linear vs MLP head), LoRA wrapping, discriminative param groups, ordinal head, ranking utilities, the output contract. |
| 05 | [Training & losses](05-training.md) | `FaceOcclusionLitModule`: the multi-task loss stack, optimizer/scheduler, validation metrics, interrupt-safe finalization, callbacks. |
| 06 | [Metrics & evaluation](06-metrics-and-evaluation.md) | The challenge metric, bootstrap CIs, evaluation lenses, leakage-free scoring — the CI-first gate. |
| 07 | [Pipeline & experiments](07-pipeline-and-experiments.md) | The CLI scripts (validate → split → train → predict → analyze → compare), the experiment-folder layout, inference & submission. |
| 08 | [Cluster & remote](08-cluster-and-remote.md) | SLURM job, RunPod setup/sync/run helpers, the persistent uv cache/venv, the CUDA wheel pinning. |

## The library at a glance

Reusable code lives in `src/face_occlusion/`; experiments are **YAML configs, not code**.

```
src/face_occlusion/
├── data/        # splits, dataset, datamodule, transforms, normalize, samplers,
│                # synthetic_occlusion + occluders + compositing, caches, mask store
├── models/      # regressor (+ build_model factory), ordinal head, ranking, outputs
├── metrics/     # challenge_metric, eval_lenses, bootstrap
├── training/    # lit_module (the LightningModule), callbacks
├── inference/   # predict_dataframe (TTA, recalibration)
└── utils/       # config, experiment dirs, reproducibility

scripts/         # CLI entry points (run as `python -m scripts.<group>.<name>`)
├── data/        # validate_data, make_split, build_synthetic_cache
├── training/    # train
├── inference/   # predict_test
├── analysis/    # analyze_val_predictions, bootstrap_metrics, compare_experiments,
│                # fit_recalibration, generate_synthetic_occlusion_audit
└── runpod/      # setup_pod, sync_repo_to_remote, run_experiment(_tmux)

configs/         # one YAML per experiment + ablation groups
jobs/            # train.slurm
```

> Conventions and gotchas that bite if ignored live in the repo-root `CLAUDE.md`; the
> prose project guide is `docs/PROJECT_GUIDE.md`. This guide focuses on the **code
> architecture**.
