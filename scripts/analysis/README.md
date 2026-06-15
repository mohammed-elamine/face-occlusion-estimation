# Analysis Scripts

Post-training diagnostics that work from saved experiment outputs.

```bash
python -m scripts.analysis.analyze_val_predictions \
  --experiment-dir outputs/experiments/<run_id>
```

Reports include summary metrics, grouped metrics, worst-error tables, plots,
and optional image grids.

## DFR — gender-shortcut debiasing (`fit_dfr.py`)

Deep Feature Reweighting: refit only the champion's **linear head** on **gender×occlusion-balanced**
features (encoder frozen) to remove the gender shortcut driving the bulk gap (see
`tmp/model_study/05_gender_gap.md`). Needs the checkpoint + data (run on the pod).

```bash
python -m scripts.analysis.fit_dfr \
  --config outputs/experiments/<champion>/config.yaml \
  --checkpoint outputs/experiments/<champion>/checkpoints/best.ckpt \
  [--ridge 1.0] [--balance gender_occ] [--predict-test]
```

Prints original-vs-DFR val `score / err_F / err_M / gap / bulk_gap`, and writes
`outputs/dfr/<champion>_dfr/predictions/val_predictions.csv` (training schema) so you can gate it
with `compare_experiments` / `bootstrap_metrics`. `--predict-test` also writes a submission. The
fit is closed-form weighted ridge (`face_occlusion.training.dfr`), so it runs in seconds once
features are extracted.
