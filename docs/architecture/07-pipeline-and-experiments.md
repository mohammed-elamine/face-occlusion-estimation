# 07 — Pipeline & Experiments

The pipeline is a set of CLI scripts run as modules (`python -m scripts.<group>.<name>
--config <yaml>`). They share the `Config` object and the experiment-folder convention.
Each script is thin: it wires library components and writes artifacts.

## The flow

```
validate_data ─► make_split ─► train ─► (analyze_val_predictions | bootstrap_metrics | compare_experiments)
                                  │
                                  └─► predict_test ─► submission CSV
            (optional) build_synthetic_cache ─► train with ranking
            (optional) fit_recalibration (diagnostic gate)
```

## Data scripts — `scripts/data/`

- `validate_data.py --config [--max-image-check N]` — sanity-checks paths, columns, row
  counts, target range, gender encoding, image openability, and **train/test `group_id`
  overlap**; writes `outputs/reports/data_validation_report.json` and exits non-zero on
  errors.
- `make_split.py --config [--strategy ...] [--split-path ...]` — builds the fixed split via
  `data/splits.py` and writes it to `split.split_path`. Run once; training reloads it.
- `build_synthetic_cache.py --config --cache-dir [--max-per-bin-gender 200] [--target-min
  0.10] [--target-max 1.0] [--quality 95] [--limit N]` — precomputes the synthetic
  `clean/mild/strong` views + masks from **train-split** anchors only, balanced by
  `occlusion_bin × gender`, and writes `views/`, `masks/`, and `manifest.csv` consumed by
  `SyntheticCache` ([03](03-data.md)). Needs the `synthetic` extra.

## Training — `scripts/training/train.py`

`--config <yaml>` is the only argument. It:

1. loads the config, creates the run dir (`utils/experiment.py::create_run_dir`), snapshots
   config + git info + metadata;
2. seeds (`utils/reproducibility.py::seed_everything`, optional `deterministic`);
3. builds the datamodule (`setup("fit")`), optionally loads `train_targets` for reweighting;
4. builds `FaceOcclusionLitModule` + a `pl.Trainer` (`accelerator/devices="auto"`,
   `precision=training.precision`, `gradient_clip_val`, logger, callbacks);
5. runs `trainer.fit(...)` inside the interrupt-safe wrapper ([05](05-training.md)) that always
   writes `predictions/val_predictions.csv` and `training_status.json`.

The Trainer reads only a few keys directly (`max_epochs`, `precision`, `gradient_clip_val`,
`deterministic`); everything else flows through the module and datamodule.

## Inference & submission — `scripts/inference/predict_test.py` (+ `inference/predict.py`)

- `inference/predict.py::predict_dataframe(model, loader, device, recalibration=None,
  tta=False)` — `@torch.no_grad()` pass returning one row per image with
  `image_id, filename, path, pred_raw, pred_clipped, database, source_subfolder, group_id,
  face_id` (+ `pred_recal` if recalibrating). `tta=True` averages the image and its
  horizontal flip; recalibration is applied to the raw prediction **before** clipping.
- `predict_test.py --config --checkpoint [--output-dir] [--recalibration MAP.json] [--tta]`
  — loads the checkpoint, runs prediction over the test set, writes
  `test_predictions_extended.csv` (full metadata) and `test_predictions.csv` (the submission:
  `filename, prediction, dummy gender`). `test_students.csv` has only `filename`, so the
  writer adds a dummy `gender` column purely to satisfy the upload format. Validation uses
  **no** TTA so `val/score` stays comparable across runs.

## Analysis — `scripts/analysis/`

- `analyze_val_predictions.py --experiment-dir <run>` (or `--predictions/--output-dir`) —
  the main report generator: `reports/summary_metrics.json` (with the `robust` CI block),
  `reports/tables/*.csv`, `reports/plots/*.png`, `reports/samples/*grid.png`, and a
  standalone `report.html`. Auto-detects the split copy for seen/unseen scoring.
- `bootstrap_metrics.py --predictions <csv> [--unit row|group] [--n-boot] [--ci]` — prints
  (and optionally writes) CIs for the challenge metrics; the quick CI on any predictions CSV.
- `compare_experiments.py --runs <dir...> --baseline <label> [--out-dir]` — the **paired-Δ**
  comparison vs a baseline; writes `comparison.{md,csv}` and a `delta_forest.png`, and flags
  runs whose Δscore CI excludes 0.
- `fit_recalibration.py --experiment-dir <run>` — fits an out-of-fold, identity-disjoint
  isotonic recalibration map, gates it on the official metric + lenses, and (if it passes)
  saves `calibration/mapping.json` for inference. A diagnostic: if high-occ error does not
  recover after recalibration, the model lacks discrimination and the fix is training-side.
- `generate_synthetic_occlusion_audit.py --config [--coverage-only]` — visual/coverage audit
  of the synthetic generator; `--coverage-only` reports MediaPipe success per `bin × gender`
  and gates whether synthetic ranking can cover the hard cases at all.

## The experiment folder — `utils/experiment.py`

`create_run_dir` makes `outputs/experiments/<timestamp>_<name>/` with subdirs
`RUN_SUBDIRS = (checkpoints, logs, predictions, reports, splits)`. A completed run contains:

```
<run>/
├── config.yaml            # resolved config snapshot (incl. run_id, run_dir)
├── metadata.json          # run_id, created_at, git_commit, git_dirty, python, platform
├── git_commit.txt / git_status.txt
├── training_status.json   # completed|interrupted + which checkpoint fed predictions
├── checkpoints/           # best.ckpt, last.ckpt
├── logs/                  # csv_logs/version_0/metrics.csv (+ wandb/ if enabled)
├── predictions/
│   └── val_predictions.csv   # gender, target, pred_raw, pred_clipped, abs_error, group_id, ...
├── reports/               # (from analyze_val_predictions) summary_metrics.json, tables/, plots/, samples/, report.html
├── splits/                # copy of the split CSV used (reproducibility)
└── calibration/           # (from fit_recalibration) mapping.json, gate_report.json
```

The split is **copied in** so the run stays reproducible even if the global split file
changes later, and `val_predictions.csv` is self-describing so all analysis runs without a
checkpoint reload.
