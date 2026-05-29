# Slurm Jobs

This folder contains Slurm job scripts used to run the project on the school compute cluster.

The goal is to keep cluster execution reproducible and easy to launch from the repository root.

---

## Contents

```text
jobs/
├── train_baseline.slurm     # Launch baseline model training
├── predict_test.slurm       # Generate test predictions
└── README.md                # This file
```

---

## Usage

Always submit jobs from the repository root:

```bash
cd ~/projects/face-occlusion-estimation
sbatch jobs/train_baseline.slurm
```

To monitor your jobs:

```bash
squeue -u $USER
```

To cancel a job:

```bash
scancel <job_id>
```

Logs are written to:

```text
outputs/logs/
```

---

## Security Rules

These job files can be committed to GitHub as long as they do not contain secrets.

Never commit:

* W&B API keys,
* GitHub tokens,
* passwords,
* SSH private keys,
* personal absolute paths,
* dataset files,
* model checkpoints.

Use relative paths whenever possible. The scripts should be launched from the repository root and rely on:

```bash
cd "$SLURM_SUBMIT_DIR"
```

---

## Typical Workflow

```bash
# Update code
git pull

# Validate data
python scripts/validate_data.py --config configs/baseline.yaml

# Create split
python scripts/make_split.py --config configs/baseline.yaml

# Submit training job
sbatch jobs/train_baseline.slurm

# Monitor logs
tail -f outputs/logs/<job_log_file>
```
