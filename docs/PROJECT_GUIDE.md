# Project Guide

This guide explains how the Face Occlusion Estimation project is organized and
how the main workflow fits together. It is meant to be the reference document
for teammates who want to train the current baseline, add future model configs,
or analyze experiment outputs.

## Project Goal

The task is supervised image regression. Each input is a cropped face image,
normally resized to `224 x 224`, and the model predicts a continuous occlusion
score in `[0, 1]`.

The target is affected by visible face coverage, hair, masks, sunglasses,
helmets, hands, scarves, blur, bad crops and other image quality issues. Because
the challenge metric is gender-aware, the pipeline keeps `gender`, `image_id`,
`target` and image paths through validation and prediction.

## Baseline vs Future Models

`configs/baseline.yaml` is the starter experiment configuration. It currently
uses a ConvNeXt-Tiny backbone.

The code under `src/face_occlusion/` is a reusable project package for all
future configs. New experiments should usually be added
by creating a new YAML file in `configs/`, for example:

```text
configs/convnext_small.yaml
configs/efficientnet_b3.yaml
configs/dinov2_vitl14.yaml
```

The same scripts should continue to work:

```bash
python scripts/train.py --config configs/your_config.yaml
python scripts/predict_test.py --config configs/your_config.yaml --checkpoint <checkpoint>
```

## Repository Structure

```text
face-occlusion-estimation/
|-- assets/                 # Logos and public README illustrations
|-- configs/                # YAML experiment configs
|-- data/                   # Local challenge data, ignored by git
|-- docs/                   # Project-level documentation
|-- jobs/                   # Slurm launchers and cluster notes
|-- notebooks/              # Optional exploratory notebooks
|-- outputs/                # Generated artifacts, ignored by git
|-- scripts/                # CLI entrypoints for setup, splits, train, predict
|-- src/face_occlusion/     # Main reusable Python package
|-- tests/                  # Lightweight unit tests
|-- Makefile                # Common development commands
|-- pyproject.toml          # Dependencies and tool configuration
`-- README.md               # Short project overview and quickstart
```

## Source Package Map

```text
src/face_occlusion/
|-- data/
|   |-- dataset.py          # Loads images and preserves per-sample metadata
|   |-- datamodule.py       # LightningDataModule for train/val/test loaders
|   |-- splits.py           # Gender x occlusion-bin train/val split logic
|   `-- transforms.py       # Conservative image transforms
|-- inference/
|   `-- predict.py          # Batch prediction helpers returning DataFrames
|-- metrics/
|   `-- challenge_metric.py # Weighted and gender-aware challenge metric
|-- models/
|   `-- regressor.py        # timm backbone wrapped as a scalar regressor
|-- training/
|   |-- callbacks.py        # Checkpoint, early stopping and LR callbacks
|   `-- lit_module.py       # LightningModule with loss and validation metrics
`-- utils/
    |-- config.py           # Small YAML loader with dotted access
    |-- experiment.py       # Experiment directory and metadata helpers
    `-- reproducibility.py  # Random seed helper
```

## Library Choices

The project uses common, well-supported libraries and avoids heavy frameworks
unless they clearly reduce complexity.

| Library | Used For | Why This Choice |
|---|---|---|
| PyTorch | Model definition, tensors, training backend | Flexible research framework with strong computer-vision support. Easier to customize than higher-level APIs for regression with a custom metric. |
| PyTorch Lightning | Training loop, validation loop, checkpointing, logging | Keeps training code compact and reproducible while still allowing custom losses, metrics and callbacks. Avoids writing a fragile manual training loop. |
| timm | Pretrained image backbones | Gives access to many modern CNN and transformer backbones through one interface. This makes future configs easy without rewriting model code. |
| torchvision | Basic image transforms | Enough for the conservative augmentations used here. We avoid heavier augmentation libraries because many occlusion-style augmentations would change the label semantics. |
| pandas | CSV loading and metadata joins | The challenge data is tabular metadata plus image paths. pandas makes split merging, validation reports and prediction CSVs simple. |
| NumPy | Metric computation and array handling | Lightweight and reliable for grouped metric calculations outside the GPU training graph. |
| scikit-learn | Train/validation split | Provides a tested `train_test_split` implementation with stratification support. This is safer than hand-rolling split logic. |
| Pillow | Image loading | Simple RGB image loading for dataset items. It is enough for this pipeline and keeps image I/O easy to reason about. |
| OpenCV headless | Optional image-processing support | Available for future image utilities without requiring GUI/system display libraries on the cluster. |
| PyYAML | Config files | YAML keeps experiment configs readable for a student/team workflow. We intentionally avoid Hydra/OmegaConf for now because the project does not need complex config composition. |
| Matplotlib / Seaborn | Local plots and reports | Standard plotting tools for validation diagnostics and post-analysis. |
| W&B, optional | Experiment tracking | Useful on the cluster when enabled, but never required. CSV logs remain the default fallback. |
| uv | Dependency management | Fast, lock-file based environment setup for local and cluster reproducibility. |
| Ruff | Linting and formatting | One fast tool replaces several separate tools such as flake8, isort and black. |
| pytest | Tests | Simple, standard test runner for dataset and metric checks. |

The main rule is: prefer libraries that make the core workflow clearer. If a new
dependency does not improve reproducibility, readability or experiment speed in
a concrete way, keep the project simple.

## Configuration Logic

Training is config-driven. The YAML config describes the data, model and training settings. The main training script (`scripts/train.py`) turns one config into one reproducible experiment folder with checkpoints, logs, predictions and reports.

Important sections:

| Section | Purpose |
|---|---|
| `project` | Project name, random seed, global output root |
| `experiment` | Run name, experiment root, metadata options |
| `data` | CSV paths, image root, column names, target scaling |
| `split` | Split strategy, occlusion bins, split CSV path |
| `model` | timm backbone, pretrained flag, output activation, dropout |
| `training` | Batch size, epochs, optimizer settings, precision |
| `augmentation` | Resize and conservative image augmentation settings |
| `logging` | CSV/W&B logging options |
| `checkpoint` | Monitored validation metric and checkpoint naming |

To create a new model experiment, copy `configs/baseline.yaml`, change
`experiment.name`, update `model.backbone` and adjust training settings. Keep
the `data`, `split` and metric-related fields unless there is a deliberate
reason to change them.

## Data Flow

The training CSV is read from `cfg.data.train_csv`. The dataset expects:

```text
filename, FaceOcclusion, gender
```

The exact column names come from the config:

```yaml
data:
  image_col: filename
  target_col: FaceOcclusion
  gender_col: gender
  id_col: filename
```

`FaceOcclusionDataset` returns dictionaries, not just image tensors. Each
validation item includes:

```text
image, target, gender, image_id, path
```

This is intentional. The challenge score depends on gender groups, and
post-analysis needs image ids and paths.

## Split Logic

The validation split is stratified by:

```text
gender x occlusion_bin
```

This keeps the validation set representative across both gender and occlusion
difficulty. The split file stores only ids and split labels:

```text
filename, split
```

The project saves the split first, then reloads it when building
`FaceOcclusionDataset`. This is deliberate: the split file fixes which samples
belong to train and validation, while `train.csv` remains the source of full
metadata such as target, gender and image path. During setup, the saved split is
merged back with `train.csv` by `cfg.data.id_col`.

This makes validation scores comparable across configs. A baseline, ConvNeXt
Small and EfficientNet experiment can all use the exact same validation images
instead of silently creating different random splits.

By default it is written to:

```text
outputs/splits/baseline_split.csv
```

Training copies the exact split file into the experiment folder so results stay
reproducible even if the global split file changes later.

## Training Lifecycle

The main training entrypoint is:

```bash
python scripts/train.py --config configs/baseline.yaml
```

At startup, `scripts/train.py` creates:

```text
outputs/experiments/<run_id>/
```

The run id is generated from a timestamp and `experiment.name`, for example:

```text
2026-05-29_031500_baseline-convnext-tiny
```

The training script then:

1. Loads the YAML config.
2. Creates the run directory.
3. Saves config, git info and metadata.
4. Creates or loads the train/val split.
5. Copies the split into the run directory.
6. Builds the DataModule, model and LightningModule.
7. Writes checkpoints and logs inside the run directory.
8. Validates with the best checkpoint.
9. Saves per-sample validation predictions.
10. Prints the experiment directory, best checkpoint and prediction CSV path.

## Experiment Directory

Each training run is self-contained:

```text
outputs/experiments/<run_id>/
|-- config.yaml
|-- metadata.json
|-- git_commit.txt
|-- git_status.txt
|-- checkpoints/
|   |-- best.ckpt
|   `-- last.ckpt
|-- logs/
|   `-- csv_logs/
|-- predictions/
|   `-- val_predictions.csv
|-- reports/
`-- splits/
    `-- baseline_split.csv
```

The most important local analysis artifact is:

```text
outputs/experiments/<run_id>/predictions/val_predictions.csv
```

It contains:

```text
image_id,path,gender,target,pred_raw,pred_clipped,abs_error
```

Use this CSV for error analysis without loading a checkpoint.

## Metric Logic

The challenge metric is implemented in:

```text
src/face_occlusion/metrics/challenge_metric.py
```

It uses a weighted MSE:

```text
w_i = 1/30 + y_i
```

High-occlusion samples therefore matter more. The final score combines female
and male errors:

```text
score = (Err_F + Err_M) / 2 + abs(Err_F - Err_M)
```

Predictions are clipped to `[0, 1]` for validation metrics and submissions, but
raw predictions are still saved because they are useful for diagnosing model
calibration.

## Prediction Workflow

Generate test predictions with:

```bash
python scripts/predict_test.py \
  --config configs/baseline.yaml \
  --checkpoint outputs/experiments/<run_id>/checkpoints/best.ckpt
```

If the checkpoint is inside an experiment directory, outputs are written back to:

```text
outputs/experiments/<run_id>/predictions/
```

Files produced:

```text
test_predictions.csv           # Submission-style file
test_predictions_extended.csv  # Metadata-rich file for analysis
```

## Cluster Workflow

Set up the cluster environment once:

```bash
bash scripts/setup_cluster_env.sh
```

Launch the baseline:

```bash
sbatch jobs/train.slurm
```

Launch a custom config:

```bash
CONFIG_PATH=configs/efficientnet_b3.yaml sbatch jobs/train.slurm
```

The Slurm script only prepares the runtime and launches training. Experiment
directory creation stays in `scripts/train.py`.

Slurm logs are separate from Lightning logs and are written to:

```text
outputs/slurm_logs/
```

## Adding a New Experiment

Recommended steps:

1. Copy `configs/baseline.yaml` to a new config file.
2. Change `experiment.name`.
3. Change `model.backbone` and any relevant training settings.
4. Keep `split.split_path` fixed if you want fair comparison against prior runs.
5. Run locally for syntax/config checks when possible.
6. Submit on the cluster with `CONFIG_PATH=... sbatch jobs/train.slurm`.
7. Compare runs using each run's `val_predictions.csv` and logged metrics.

Avoid changing several major ideas at once. For example, if you change the
backbone, keep augmentation and split logic stable unless the experiment is
specifically about those pieces.

## Development Commands

Common commands:

```bash
make install        # Install dependencies and pre-commit hooks
make check          # Run lint and format checks
make format         # Format code with ruff
make setup-cluster  # Run cluster environment setup script
```

Data utilities:

```bash
python scripts/validate_data.py --config configs/baseline.yaml
python scripts/make_split.py --config configs/baseline.yaml
```

## Git and Artifact Policy

Tracked:

```text
source code, configs, docs, tests, lightweight .gitkeep files
```

Ignored:

```text
data, checkpoints, predictions, experiment outputs, Slurm logs, W&B logs
```

This keeps the repository clean while allowing each experiment folder to be
copied or archived separately.

## Mental Model

The project has one central idea:

```text
config + data + split -> one reproducible experiment folder
```

The Python package provides reusable building blocks. YAML configs describe
specific experiments. `scripts/train.py` turns one config into one complete run
folder that can be inspected locally or copied from the cluster.
