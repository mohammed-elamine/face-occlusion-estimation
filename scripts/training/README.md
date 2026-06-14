# Training Scripts

Entrypoints for model training.

```bash
python -m scripts.training.train --config configs/baseline.yaml
```

The training script owns experiment-directory creation, checkpoint paths,
logging, validation, and validation-prediction export.
