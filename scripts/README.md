# Scripts

Project commands are grouped by purpose.

| Folder | Purpose |
| --- | --- |
| `training/` | Train models from YAML configs. |
| `inference/` | Generate test predictions and submissions. |
| `data/` | Validate data and create train/validation splits. |
| `analysis/` | Analyze saved experiment predictions. |
| `setup/` | Local and cluster environment setup checks. |
| `runpod/` | RunPod-specific setup and launch helpers. |

Run Python entrypoints as modules from the repository root, for example:

```bash
python -m scripts.training.train --config configs/baseline.yaml
```
