<div align="center">
  <table>
    <tr>
      <td align="center" valign="middle">
        <a href="https://www.telecom-paris.fr/en/home">
          <img src="assets/logos/telecom-paris.png" alt="Telecom Paris logo" height="82"/>
        </a>
      </td>
      <td width="28"></td>
      <td align="center" valign="middle">
        <a href="https://www.idemia.com/">
          <img src="assets/logos/idemia.png" alt="IDEMIA logo" height="82"/>
        </a>
      </td>
    </tr>
  </table>

  <h1>Face Occlusion Estimation</h1>

  <p>
    <strong>Predict how much of a face is hidden from a single cropped image.</strong><br/>
    A computer vision data challenge by
    <a href="https://www.telecom-paris.fr/en/home">Telecom Paris</a> x
    <a href="https://www.idemia.com/">IDEMIA</a>.
  </p>

  <p>
    <img alt="task: regression" src="https://img.shields.io/badge/Task-Regression-1f7a8c"/>
    <img alt="domain: computer vision" src="https://img.shields.io/badge/Domain-Computer%20Vision-3a5a40"/>
    <img alt="target: occlusion score" src="https://img.shields.io/badge/Target-Occlusion%20Score-c1121f"/>
    <img alt="focus: robustness and fairness" src="https://img.shields.io/badge/Focus-Robustness%20%26%20Fairness-f28482"/>
  </p>

  <p><em>Serious metric. Tiny chaos. Challenge accepted.</em></p>
</div>


<p align="center">
  <img src="assets/illustrations/funny-faces.png" alt="A set of funny cartoon faces" width="560"/>
</p>

<p align="center">
  <em>
    Real face crops have a talent for being messy: masks, hair, sunglasses, blur,<br/>
    and one face that clearly knows it is ruining your validation loss.<br/>
    The job sounds simple: one <code>224 x 224</code> crop, one occlusion score. Then the images start having opinions.
  </em>
</p>


## At a Glance

| Item | Details |
|---|---|
| Input | Cropped face image, `224 x 224` |
| Output | Continuous occlusion percentage |
| Task type | Supervised regression |
| Main challenge | Accuracy on hard, highly occluded samples |
| Extra pressure | Balanced performance across female and male subsets |

```text
face crop -> model -> occlusion score
```

Simple to write down. Annoyingly hard to do well.


## What the Model Learns to Notice

The visual clues are often obvious to humans but slippery for machines:

- masks and sunglasses,
- hands, hair, scarves, and hats,
- objects passing in front of the face,
- blur, bad crops, and partial visibility.

The challenge is to learn useful visual cues without overreacting to noisy crops or incidental occlusions.


## Why It Matters

Occlusion is not just an annoying corner case. It shows up in real-world face pipelines where reliability matters:

- face image quality assessment,
- biometric robustness,
- occlusion-aware recognition systems,
- fairness-aware evaluation,
- trustworthy AI under messy visual conditions.

The goal is not only to win on average. A strong solution should handle difficult images gracefully and avoid trading one subgroup's performance for another's.


## Scoring

Highly occluded samples carry more weight, so the benchmark gives extra attention to hard cases.

$$
\mathrm{Err} =
\frac{\sum_{i=1}^{N} w_i (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} w_i},
\qquad
w_i = \frac{1}{30} + y_i
$$

Where:

- $y_i$ is the true occlusion score,
- $\hat{y}_i$ is the predicted occlusion score,
- $w_i$ is the sample weight, larger when occlusion is higher.

The final challenge score combines subgroup performance and subgroup balance:

$$
\mathrm{Score} =
\frac{\mathrm{Err}_{\mathrm{female}} + \mathrm{Err}_{\mathrm{male}}}{2}
+
\left|\mathrm{Err}_{\mathrm{female}} - \mathrm{Err}_{\mathrm{male}}\right|
$$

The metric rewards low overall error, strong performance on highly occluded samples, and balanced errors across female and male subsets.


## Getting Started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it.

2. Clone and set up:

   ```bash
   git clone https://github.com/mohammed-elamine/face-occlusion-estimation.git
   cd face-occlusion-estimation
   make install
   ```

3. Run `make help` to see all available commands.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and guidelines.
For the full project architecture and workflow reference, see
[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).


## Approach & Results

The whole project is one idea: **`config + data + split → one self-contained experiment
folder`**. Models are YAML configs, not new scripts — `src/face_occlusion/` is the reusable
library, and every run lands in `outputs/experiments/<id>/` with its config, checkpoints,
predictions and reports. The full method write-ups (one page per component) live in
[`docs/architecture/`](docs/architecture/README.md).

**What we explored** (all config-gated, default off):

- **Backbones** — a fully fine-tuned **ConvNeXt-Small** (our strongest single model) and
  **DINOv2 ViT-B + LoRA** (`configs/convnext_ablation/`, `configs/dinov2_lora/`).
- **Metric-aligned & imbalanced-regression losses** — gender-balanced weighted MSE, an
  ordered-bin **expectation head (DEX + DLDL/LDS)**, an ordinal-threshold head, distribution-aware
  reweighting, and a synthetic monotonic-ranking objective (`configs/imbalanced_regression/`,
  `configs/synthetic_ranking/`, `configs/ordinal_warmup_ablation/`,
  [`docs/architecture/09-imbalanced-regression-and-expectation-head.md`](docs/architecture/09-imbalanced-regression-and-expectation-head.md)).
- **Sampling & augmentation** — a gender×occlusion balanced-batch sampler, background-invariance
  augmentation, and a label-aware synthetic-occlusion pipeline
  ([`docs/balanced_batch_sampler.md`](docs/balanced_batch_sampler.md),
  [`docs/synthetic_occlusion_generation.md`](docs/synthetic_occlusion_generation.md)).
- **Fairness for the gender gap** — Deep Feature Reweighting (`scripts/analysis/fit_dfr.py`) and a
  gradient-reversal gender adversary (`configs/convnext_ablation/10_gender_invariant.yaml`).
- **Combining models** — EMA weight averaging and **prediction ensembling**
  (`scripts/inference/predict_ensemble.py`).

**What worked.** Under a bootstrap-CI gate, no single architectural or loss lever beat a
well-tuned ConvNeXt-Small; the one reliable win came from **ensembling decorrelated,
individually-tied models** (validation ≈ `0.00118`, leaderboard ≈ `0.00112`). The remaining error
is **data-bound**: the rare high-occlusion tail (only ~8 training faces above occlusion `0.6`) and
the gender gap resist every model-side fix, which our analyses trace to scarcity and label noise
rather than the model. Evaluation is CI-first — we gate every change with paired-Δ bootstrap
confidence intervals (`scripts/analysis/compare_experiments.py`, `bootstrap_metrics.py`).


## Config-Driven Pipeline

The reusable training code lives under `src/face_occlusion/`. The current
starter experiment is the ConvNeXt-Small baseline in
[`configs/baseline.yaml`](configs/baseline.yaml), and future models should be
added as new YAML configs.

```bash
# 1. Sanity-check the data (paths, columns, target range, image readability).
python -m scripts.data.validate_data --config configs/baseline.yaml

# Optional guided notebook:
# notebooks/database3_identity_overlap.ipynb

# 2. Create a fixed row-level gender x occlusion-bin x database split.
python -m scripts.data.make_split --config configs/baseline.yaml

# 3. Train a config. Each run gets its own folder under outputs/experiments/.
python -m scripts.training.train --config configs/baseline.yaml

# 4. Generate test predictions / submission file from the best checkpoint.
python -m scripts.inference.predict_test \
  --config configs/baseline.yaml \
  --checkpoint outputs/experiments/<run_id>/checkpoints/best.ckpt
```

When the checkpoint comes from an experiment folder, test predictions are saved
back into that same run's `predictions/` directory.

Our best submission is an **ensemble** of a few decorrelated runs. Once each member has
test predictions, average them into one submission:

```bash
python -m scripts.inference.predict_ensemble \
  --members outputs/experiments/<run_a> outputs/experiments/<run_b> outputs/experiments/<run_c>
# -> outputs/ensemble_submission/test_predictions.csv  (prints the ensemble val score first)
```

Design choices worth knowing:

- **Backbone — `convnext_small.fb_in22k_ft_in1k`.** Strong ImageNet-22k features
  with more capacity than Tiny for leaderboard-oriented experiments, while still
  being practical on a single GPU. Easy to swap with any `timm` model via
  `model.backbone` in the config.
- **Augmentation is conservative by default.** The standard path avoids RandomErasing,
  heavy blur, random crops and occlusion: they change the *true* face visibility while the
  original label stays the same, which silently corrupts supervision. Only horizontal flip,
  mild color jitter and a small rotation are used. (A separate, opt-in, **label-aware**
  synthetic-occlusion pipeline exists for the ranking objective — see the docs.)
- **Metric is gender-aware.** Validation reports the official score
  `(Err_F + Err_M)/2 + |Err_F - Err_M|`, so we keep `gender` in every batch
  and stratify the default validation split on `gender x occlusion_bin x database`.
  Dataset encoding is `female=0`, `male=1`.
- **Path metadata is preserved.** `database`, `source_subfolder`, `group_id`
  and `face_id` are parsed from filenames for splits and diagnostics, not fed
  into the image model.
- **Two split protocols are supported.** `row_stratified` is the default for
  leaderboard-oriented comparison; `group_stratified` is available for a
  stricter robustness check with unseen `group_id` values. We do not train
  every model twice by default.
- **Database3 identity overlap can be inspected directly.** Open
  `notebooks/database3_identity_overlap.ipynb` to quantify how many
  `database3` `m.<id>` folders appear in both challenge train and test,
  with tables and visuals.

Validation predictions are written to
`outputs/experiments/<run_id>/predictions/val_predictions.csv` for error
analysis (per-sample target, raw and clipped predictions, absolute error,
gender, path and parsed path metadata).

Analyze a completed experiment folder with:

```bash
python -m scripts.analysis.analyze_val_predictions \
  --experiment-dir outputs/experiments/<run_id>
```

This generates a complete post-analysis report under
`outputs/experiments/<run_id>/reports/`:

```text
reports/
├── report.html          — standalone HTML report (open in a browser)
├── summary_metrics.json — key metrics as JSON
├── tables/              — grouped metrics and error tables (CSV)
├── plots/               — 15+ ordered diagnostic plots (PNG)
│   ├── 01–15_*.png      — per-prediction analysis (error, bias, calibration…)
│   ├── 20_training_global_metrics.png
│   ├── 21_training_weighted_mse_by_occlusion_bin.png
│   ├── 22_training_bias_by_occlusion_bin.png   (requires updated LitModule)
│   ├── 23_training_weighted_mse_by_gender.png
│   ├── 24_training_bias_by_gender.png          (requires updated LitModule)
│   └── 25_training_weighted_mse_by_database.png
└── samples/             — image grids of difficult examples (PNG)
```

When `--experiment-dir` contains a Lightning CSV logger metrics file
(`logs/csv_logs/version_*/metrics.csv`), the script automatically generates
training-dynamics plots (20–25) showing how validation metrics evolve per
epoch broken down by occlusion bin, gender, and database.  Plots 22 and 24
(per-subgroup bias over epochs) require the updated `FaceOcclusionLitModule`
that logs `val/bin_*_bias` and `val/female_bias` / `val/male_bias`.

Image grids are generated by default using
`data/raw/crops/Crop_224_5fp_100K` as the image root. To disable them or
override the root:

```bash
# Disable image grids
python -m scripts.analysis.analyze_val_predictions \
  --experiment-dir outputs/experiments/<run_id> \
  --no-image-grids

# Custom image root
python -m scripts.analysis.analyze_val_predictions \
  --experiment-dir outputs/experiments/<run_id> \
  --image-root /custom/path/to/Crop_224_5fp_100K
```

For a group-level robustness split, generate a separate split file and use a
config that points to it:

```bash
python -m scripts.data.make_split \
  --config configs/baseline.yaml \
  --strategy group_stratified \
  --split-path outputs/splits/group_robustness_split.csv
```

Then copy the model config and set `split.strategy: group_stratified` plus
`split.split_path: outputs/splits/group_robustness_split.csv` before training.


## Experiment Outputs

Training is organized as:

```text
one run = one self-contained folder
```

At startup, `scripts/training/train.py` creates a unique directory such as:

```text
outputs/experiments/2026-05-29_031500_baseline-convnext-small/
```

Important artifacts live inside that folder:

```text
config.yaml
metadata.json
git_commit.txt
git_status.txt
checkpoints/best.ckpt
checkpoints/last.ckpt
logs/
predictions/val_predictions.csv
reports/
splits/
```

The validation CSV is the main file to copy locally for post-analysis because
it contains the metadata needed by the challenge metric: `image_id`, `path`,
`gender`, `target`, raw predictions, clipped predictions and absolute error.


## Cluster Training

Set up the environment once from the repository root:

```bash
bash scripts/setup/setup_cluster_env.sh
```

Launch the baseline config:

```bash
sbatch jobs/train.slurm
```

Launch a custom config:

```bash
CONFIG_PATH=configs/convnext_ablation/01_convnext_base.yaml sbatch jobs/train.slurm
```

Slurm logs are saved under `outputs/slurm_logs/`. Experiment directories are
created and managed by `scripts/training/train.py`, not by the Slurm script.


## Project Map

```text
face-occlusion-estimation/
├── configs/                # Experiments as YAML: baseline.yaml + ablation groups (each with a README)
├── src/face_occlusion/     # Reusable library: data/ models/ training/ metrics/ inference/ utils/
├── scripts/                # CLI entry points: data/ training/ inference/ analysis/ setup/ runpod/
├── docs/                   # PROJECT_GUIDE + architecture/ (component-by-component) + topic notes
├── tests/                  # pytest suite (uv run pytest)
├── jobs/                   # Slurm job scripts (train.slurm)
├── notebooks/              # EDA & split-diagnostics notebooks
├── data/                   # Local data, git-ignored (raw/occlusion_datasets + raw/crops/...)
├── outputs/                # Experiment folders, splits, logs — git-ignored
├── assets/                 # Challenge logos & illustrations
├── .github/workflows/      # CI (ruff + pre-commit)
├── Makefile                # Local dev commands (make help)
├── pyproject.toml          # Dependencies & tooling config
└── README.md
```


<p align="center">
  <img
    src="assets/illustrations/challenge-accepted-meme.png"
    alt="Challenge accepted meme face"
    width="240"
  />
</p>

<p align="center">
  <em>Occlusion? Fairness penalty? Weird crops?</em><br/>
  <strong>Challenge accepted.</strong><br/>
  <sub>
    <a href="https://openclipart.org/detail/319872/funny-faces">faces</a>
    /
    <a href="https://openclipart.org/detail/168636/challenge-accepted">challenge accepted</a>
  </sub>
</p>

## Authors

Built with caffeinated determination by:

- **Mohammed Elamine** · [elamine.mohammed.14@gmail.com](mailto:elamine.mohammed.14@gmail.com)

<sub>One student vs. occluded faces. What could go wrong?</sub>
