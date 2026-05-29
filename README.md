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


## Baseline Pipeline

A clean ConvNeXt-Tiny regression baseline lives under `src/face_occlusion/`
and is driven by [`configs/baseline.yaml`](configs/baseline.yaml).

```bash
# 1. Sanity-check the data (paths, columns, target range, image readability).
python scripts/validate_data.py --config configs/baseline.yaml

# 2. Create a fixed gender x occlusion-bin stratified train/val split.
python scripts/make_split.py --config configs/baseline.yaml

# 3. Train the baseline. Logs to CSV by default; set logging.use_wandb=true for W&B.
python scripts/train.py --config configs/baseline.yaml

# 4. Generate test predictions / submission file from the best checkpoint.
python scripts/predict_test.py \
  --config configs/baseline.yaml \
  --checkpoint outputs/checkpoints/best.ckpt
```

Design choices worth knowing:

- **Backbone — `convnext_tiny.fb_in22k_ft_in1k`.** Strong ImageNet-22k features
  with a small enough footprint to iterate quickly on a single GPU. Easy to
  swap with any `timm` model via `model.backbone` in the config.
- **Augmentation is conservative.** We avoid RandomErasing, heavy blur,
  random crops and synthetic occlusion: they change the *true* face
  visibility while the original label stays the same, which silently
  corrupts supervision. Only horizontal flip, mild color jitter and a
  small rotation are used.
- **Metric is gender-aware.** Validation reports the official score
  `(Err_F + Err_M)/2 + |Err_F - Err_M|`, so we keep `gender` in every batch
  and stratify the validation split on `gender x occlusion_bin`.

Validation predictions are written to `outputs/predictions/val_predictions.csv`
for error analysis (per-sample target, raw and clipped predictions, absolute
error, gender, path).


## Project Map

```text
face-occlusion-estimation/
├── .github/workflows/      # CI pipelines
├── assets/
│   ├── illustrations/      # Public-domain cartoon and meme fuel
│   └── logos/              # Challenge logos
├── data/                   # Local data folder, not tracked
├── scripts/                # Dev utility scripts
├── src/face_occlusion/     # Main Python package
├── Makefile                # Local dev commands (make help)
├── pyproject.toml          # Project config & dependencies
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
- **Sara El Mountassir** · [sara.elmountasser@telecom-paris.fr](mailto:sara.elmountasser@telecom-paris.fr)

<sub>Two students vs. occluded faces. What could go wrong?</sub>
