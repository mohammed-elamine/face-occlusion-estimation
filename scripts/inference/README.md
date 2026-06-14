# Inference Scripts

Entrypoints for test prediction and challenge submission generation.

```bash
python -m scripts.inference.predict_test \
  --config configs/baseline.yaml \
  --checkpoint outputs/experiments/<run_id>/checkpoints/best.ckpt
```

The submission writer adds the required dummy `gender` column for
`test_students.csv`.
