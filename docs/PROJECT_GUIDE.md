# Project Guide

This is the **orientation doc** for someone new to the repository: what the project is, how it
is laid out, why the tools were chosen, and how to add an experiment. It is deliberately a map,
not a manual — for the component-by-component technical detail it points into
[`docs/architecture/`](architecture/README.md), which is the canonical reference.

## Project goal

Supervised image regression: given one cropped face image (resized to `224 × 224`), predict a
continuous **occlusion score** `y ∈ [0, 1]` — how much of the face is covered by masks, hands,
sunglasses, hair, helmets, scarves, blur, or bad crops.

The scoring metric is **gender-aware** and **up-weights heavily-occluded faces**, and that single
fact shapes the whole pipeline (the data keeps `gender` with every item; the loss has a per-gender
variant; evaluation is confidence-interval-first). The full formula and its consequences are in
[01 — Overview](architecture/01-overview.md) and [06 — Metrics](architecture/06-metrics-and-evaluation.md).

## The mental model

```text
config (YAML)  +  data (CSV + crops)  +  fixed split  →  one self-contained experiment folder
```

`src/face_occlusion/` is a reusable library; **an experiment is a YAML config, not a new script.**
`scripts/training/train.py` turns one config into one timestamped folder under
`outputs/experiments/` that carries everything needed to reproduce and analyse it (config snapshot,
git info, checkpoints, the split it used, per-sample validation predictions, reports). To try a new
model you copy `configs/baseline.yaml`, change a few keys, and run the same trainer — you never
write training code.

## Repository structure

```text
face-occlusion-estimation/
├── assets/          # logos and README illustrations
├── configs/         # YAML experiments: baseline.yaml + ensemble/ + experiments/ + eval/
├── data/            # local challenge data (git-ignored): raw/occlusion_datasets/ + raw/crops/
├── docs/            # this guide, architecture/ (the technical reference), topic notes
├── jobs/            # Slurm launcher (train.slurm)
├── notebooks/       # optional exploratory notebooks
├── outputs/         # generated runs, splits, reports (git-ignored)
├── scripts/         # CLI entry points: data/ training/ inference/ analysis/ setup/ runpod/
├── src/face_occlusion/  # the reusable library (data, models, training, metrics, inference, utils)
├── tests/           # unit tests (metric, data, heads, …)
├── Makefile         # common dev commands
└── pyproject.toml   # dependencies + tool config (uv, ruff, pytest)
```

The package layout inside `src/face_occlusion/` and the symbols in each module are documented in
[the architecture guide](architecture/README.md#the-library-at-a-glance) — not repeated here.

## Library choices

The project prefers common, well-supported libraries and avoids heavy frameworks unless they
clearly reduce complexity. The rule: a dependency must improve **reproducibility, readability, or
experiment speed** in a concrete way, or it stays out.

| Library | Used for | Why |
|---|---|---|
| **PyTorch** | model, tensors, training backend | Flexible CV framework; easy to customise for regression with a non-standard metric. |
| **PyTorch Lightning** | training/validation loop, checkpointing, logging | Keeps training code compact and reproducible while still allowing custom losses, metrics, callbacks — no fragile manual loop. |
| **timm** | pretrained backbones | One interface to many modern CNN/ViT backbones, so new configs need no model code. |
| **torchvision** | image transforms | Enough for the conservative augmentations used here (heavier occlusion-style augments would corrupt the label). |
| **pandas / NumPy** | metadata joins, grouped metric math | The data is tabular metadata + image paths; pandas makes splits/reports simple, NumPy handles the grouped metric off-GPU. |
| **scikit-learn** | stratified split | Tested `train_test_split` is safer than hand-rolled split logic. |
| **Pillow** | image loading | Simple, reliable RGB I/O. |
| **PyYAML** | configs | Readable experiment files; we avoid Hydra/OmegaConf because the project needs no config composition. |
| **Matplotlib / Seaborn** | diagnostics & reports | Standard plotting for validation analysis. |
| **uv** | dependency management | Fast, lock-file based env for local + cluster reproducibility. |
| **Ruff** | lint + format | One fast tool replacing flake8/isort/black. |
| **pytest** | tests | Standard runner for the dataset/metric/head checks. |
| **W&B** *(optional)* | experiment tracking | Useful on the cluster when enabled; CSV logs are the always-available fallback. |

## Where the details live

The deep "how it works" for each subsystem lives in the architecture guide. Start there for any
of the following instead of looking for it here:

| You want to understand… | Read |
|---|---|
| The config object and how YAML maps to components | [02 — Configuration](architecture/02-configuration.md) |
| Splits, the per-item dataset dict, transforms, samplers, synthetic occlusion | [03 — Data](architecture/03-data.md) |
| Backbones, heads (linear/MLP/distribution/ordinal), LoRA, auxiliary heads | [04 — Models](architecture/04-models.md) |
| The multi-task loss stack, optimizer, callbacks, EMA | [05 — Training](architecture/05-training.md) |
| The metric, bootstrap CIs, evaluation lenses, the CI-first gate | [06 — Metrics & Evaluation](architecture/06-metrics-and-evaluation.md) |
| The CLI scripts, the experiment-folder layout, single-model & ensemble submission | [07 — Pipeline](architecture/07-pipeline-and-experiments.md) |
| Cluster (Slurm) and RunPod workflows | [08 — Cluster & Remote](architecture/08-cluster-and-remote.md) |

Method write-ups (the *why* behind specific ideas) live in the topic notes:
[imbalanced regression & the expectation head](architecture/09-imbalanced-regression-and-expectation-head.md),
[occlusion-aware auxiliary learning](occlusion_aware_auxiliary_learning.md),
[the balanced-batch sampler](balanced_batch_sampler.md),
[synthetic occlusion generation](synthetic_occlusion_generation.md).

## Adding a new experiment

1. Copy `configs/baseline.yaml` to a new file (in `configs/experiments/` for a method probe).
2. Change `experiment.name`.
3. Change `model.backbone` and/or the one or two keys under study — **change one idea at a time**
   so a paired comparison is meaningful.
4. Keep `split.split_path` fixed to compare fairly against prior runs on the exact same val images.
5. Run locally to sanity-check the config, then submit on the cluster:
   `CONFIG_PATH=configs/experiments/your_config.yaml sbatch jobs/train.slurm`.
6. Compare against the baseline with `scripts.analysis.compare_experiments` (paired Δ with CIs) —
   not raw score deltas, because the high-occlusion tail is tiny ([06](architecture/06-metrics-and-evaluation.md)).

## Development commands

```bash
make install        # install deps + pre-commit hooks (uv sync --group dev)
make check          # ruff lint + format check (what CI runs)
make format         # ruff format .
uv run pytest       # full test suite (CI does NOT run this — run it locally)
```

Data utilities (one-time / occasional):

```bash
python -m scripts.data.validate_data --config configs/baseline.yaml   # sanity-check data + paths
python -m scripts.data.make_split    --config configs/baseline.yaml   # write the fixed split CSV
```

## Git & artifact policy

Tracked: source, configs, docs, tests. Git-ignored: `data/`, `outputs/` (checkpoints, predictions,
reports), Slurm/W&B logs. Each experiment folder is self-contained, so a run can be copied or
archived independently of the repo.
