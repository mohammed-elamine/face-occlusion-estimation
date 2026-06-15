# Inference Scripts

Entrypoints for test prediction and challenge submission generation.

```bash
python -m scripts.inference.predict_test \
  --config configs/baseline.yaml \
  --checkpoint outputs/experiments/<run_id>/checkpoints/best.ckpt
```

The submission writer adds the required dummy `gender` column for
`test_students.csv`.

## Ensemble submission

Averaging decorrelated, individually-tied models is the lever that significantly beat the
single-model champion (`val/score` 0.00129 → **0.00118** for `{champion, sigmoid, dldl}`; see
`tmp/comparison_reports/06_ensemble.md`). `predict_ensemble` fuses members given as **run
folders**:

```bash
python -m scripts.inference.predict_ensemble \
  --members outputs/experiments/<champion> \
            outputs/experiments/<sigmoid> \
            outputs/experiments/<dldl> \
  [--weights 1 1 1] [--tta] [--output-dir outputs/ensemble_submission]
```

Two decoupled steps, because checkpoints live on the training pod but the prediction CSVs are
small:

1. **Per-member test predictions (pod):** run `predict_test` once per member so each run folder
   has `predictions/test_predictions_extended.csv`. `predict_ensemble` will also generate these
   itself if a member's `checkpoints/best.ckpt` is present.
2. **Fuse (anywhere):** `predict_ensemble` averages each member's `pred_clipped` (aligned on
   `image_id`) and writes `test_predictions.csv` (submission) + `ensemble_test_predictions.csv`
   (per-member columns + the mean). It first prints the ensemble **`val/score`** from each
   member's on-disk `val_predictions.csv` — no checkpoints needed — so the expected number is
   confirmed before trusting the submission.

The averaging core is `face_occlusion.inference.ensemble_average` (pure, unit-tested); use TTA
(`--tta`) at generation time for an extra leaderboard-only boost (it does not show in `val/score`).
