# Slurm Jobs

This folder contains Slurm job scripts used to run the project on the school
compute cluster. Submit jobs from the repository root so relative paths resolve
correctly.

For the full project workflow, see [`../docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md).

## Contents

```text
jobs/
|-- train.slurm     # Generic training launcher
`-- README.md       # This file
```

## Training

Baseline:

```bash
sbatch jobs/train.slurm
```

Custom model or config:

```bash
CONFIG_PATH=configs/efficientnet_b3.yaml sbatch jobs/train.slurm
```

`jobs/train.slurm` only prepares the cluster runtime and launches:

```bash
python scripts/train.py --config "$CONFIG_PATH"
```

The experiment directory is created by `scripts/train.py` under:

```text
outputs/experiments/<run_id>/
```

Slurm stdout and stderr logs are saved under:

```text
outputs/slurm_logs/
```

## Typical Workflow

```bash
git pull --ff-only
bash scripts/setup_cluster_env.sh
python scripts/validate_data.py --config configs/baseline.yaml
python scripts/make_split.py --config configs/baseline.yaml
sbatch jobs/train.slurm
CONFIG_PATH=configs/convnext_small.yaml sbatch jobs/train.slurm
```

After training, copy the full run folder or at least:

```text
outputs/experiments/<run_id>/predictions/val_predictions.csv
```

for local analysis.

## Security Rules

These job files can be committed to GitHub as long as they do not contain
secrets. Never commit W&B API keys, GitHub tokens, passwords, SSH private keys,
dataset files, checkpoints, predictions or Slurm logs.
