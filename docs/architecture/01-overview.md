# 01 — Overview & Design Philosophy

## The one core idea

The whole project is a single pattern:

```
config (YAML)  +  data (CSV + image crops)  +  fixed split  →  one self-contained experiment folder
```

`src/face_occlusion/` is a reusable library; **an experiment is a YAML file**, not a new
script. To add a model you copy a config, change `experiment.name` and `model.backbone`,
and run the same trainer. One run produces one timestamped directory under
`outputs/experiments/` that carries everything needed to reproduce and analyze it (config
snapshot, git info, checkpoints, the split it used, validation predictions, and reports).

This has three consequences that show up everywhere in the code:

1. **The config is the single source of truth.** Every component is built from a `Config`
   object (`src/face_occlusion/utils/config.py`); see [02 — Configuration](02-configuration.md).
2. **Analysis is decoupled from training.** The primary analysis artifact is
   `predictions/val_predictions.csv` — it carries predictions *plus* all metadata the
   metric and diagnostics need (gender, target, raw+clipped preds, path-derived fields), so
   no checkpoint reload is required to evaluate. See [06](06-metrics-and-evaluation.md) and [07](07-pipeline-and-experiments.md).
3. **Optional capability is gated, default-OFF.** A multi-task stack (ordinal head,
   consistency, monotonicity, ranking, synthetic occlusion, samplers) exists but every
   piece is gated by a config flag that defaults to disabled, so the baseline path stays
   bit-identical when they are off.

## The metric drives the design

Scoring (`src/face_occlusion/metrics/challenge_metric.py`) is a **weighted MSE that is
gender-aware and up-weights high occlusion**:

- per-sample weight `wᵢ = 1/30 + yᵢ` (errors on heavily occluded faces count more);
- per-gender error `Err_g = Σ wᵢ(ŷᵢ − yᵢ)² / Σ wᵢ`;
- final `Score = (Err_F + Err_M)/2 + |Err_F − Err_M|`.

Two terms matter independently: the **average error** and the **female/male gap**. The
checkpoint monitor is `val/score` (`mode: min`). This metric is why:

- the dataset returns **gender** with every item (the loss and metric are per-gender);
- the loss has a **gender-balanced** variant and an optional gap penalty ([05](05-training.md));
- a **gender × occlusion** sampler exists ([03](03-data.md));
- evaluation is **CI-first** with distribution lenses, because the high-occlusion tail is
  tiny and identity-leaked ([06](06-metrics-and-evaluation.md)).

## End-to-end data flow

```
                 configs/<exp>.yaml
                        │  load_config()  → Config            (utils/config.py)
                        ▼
   train.csv ──► FaceOcclusionDataModule.setup()              (data/datamodule.py)
   (data/raw)        │  make/load split (keyed by id_col)     (data/splits.py)
                     │  build transforms                      (data/transforms.py)
                     │  (opt) synthetic cache + masks         (data/synthetic_cache.py, face_mask_store.py)
                     ▼
            FaceOcclusionDataset  → per-item dict             (data/dataset.py)
                     │  {image, target, gender, image_id, path, database,
                     │   source_subfolder, group_id, face_id, [synthetic_* views]}
                     │  (opt) GenderOcclusion sampler         (data/samplers.py)
                     ▼
            build_model(cfg)  → OcclusionRegressor            (models/regressor.py)
                     ▼
            FaceOcclusionLitModule                            (training/lit_module.py)
                     │  multi-task loss stack, discriminative LRs, warmup→cosine
                     │  monitors val/score (challenge_score)  (metrics/challenge_metric.py)
                     ▼
   pl.Trainer.fit() ──► outputs/experiments/<ts>_<name>/      (scripts/training/train.py)
                     │     checkpoints/ logs/ predictions/val_predictions.csv
                     │     config.yaml metadata.json splits/ training_status.json
                     ▼
   analyze_val_predictions.py → reports/ (tables, plots, CIs) (scripts/analysis/)
   predict_test.py            → submission CSV                (scripts/inference/)
```

## Where each layer lives (jump table)

| Layer | Code | Chapter |
|-------|------|---------|
| Config | `utils/config.py` | [02](02-configuration.md) |
| Split / dataset / loaders | `data/splits.py`, `data/dataset.py`, `data/datamodule.py` | [03](03-data.md) |
| Augmentation & sampling | `data/transforms.py`, `data/samplers.py`, `data/synthetic_occlusion.py` | [03](03-data.md) |
| Model & heads | `models/regressor.py`, `models/ordinal.py`, `models/ranking.py`, `models/outputs.py` | [04](04-models.md) |
| Loss & optimization | `training/lit_module.py`, `training/callbacks.py` | [05](05-training.md) |
| Metric & CI gate | `metrics/challenge_metric.py`, `metrics/eval_lenses.py`, `metrics/bootstrap.py` | [06](06-metrics-and-evaluation.md) |
| CLI pipeline | `scripts/**`, `utils/experiment.py`, `inference/predict.py` | [07](07-pipeline-and-experiments.md) |
| Cluster/remote | `jobs/train.slurm`, `scripts/runpod/*.sh` | [08](08-cluster-and-remote.md) |
