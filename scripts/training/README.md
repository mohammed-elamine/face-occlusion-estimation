# Training Scripts

Entrypoints for model training.

```bash
python -m scripts.training.train --config configs/baseline.yaml
```

The training script owns experiment-directory creation, checkpoint paths,
logging, validation, and validation-prediction export.

## Final-submission refit: `--train-on-all`

```bash
python -m scripts.training.train --config <run>/config.yaml --train-on-all
```

Folds the held-out val rows back into training so the final model also learns from them —
worth it here because val holds ~26% of the scarce high-occlusion tail (`target ≥ 0.4`:
train 219 + val 57). Use it only for the **final refit**, after the recipe is locked:

- It has **no genuine held-out set**, so model selection is turned off — no early stopping,
  no `best.ckpt`. Only **`last.ckpt`** is written; train a **fixed epoch budget** (set
  `training.max_epochs` to where the split run converged, e.g. ~17–20) and infer from
  `last.ckpt`. The logged `val/score` is a *leaked* monitor (a subset of train), not a
  selection signal.
- **Keep the recipe frozen** — hyperparameters were selected on the held-out val; reusing
  them on train+val is valid, re-tuning them is not.
- Equivalent to setting `split.train_on_all: true` in the config (the flag just overrides it).

Then build the ensemble submission from each refit member's `last.ckpt`:

```bash
python -m scripts.inference.predict_ensemble --members <champ_refit> <sig_refit> <dldl_refit> \
  --checkpoint-name last.ckpt --tta
```
