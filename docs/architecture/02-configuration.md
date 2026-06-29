# 02 — Configuration System

Every component is constructed from a single `Config` object. There is no argument
plumbing through the code: a script loads one YAML, and each factory reads the keys it
needs. This is what makes "an experiment is a config, not code" literally true.

## The `Config` object — `src/face_occlusion/utils/config.py`

- `Config(dict)` — a `dict` subclass with **dotted attribute access**. `cfg.model.backbone`
  is `cfg["model"]["backbone"]`; nested mappings are wrapped recursively (`_wrap`).
- `load_config(path) -> Config` — parse a YAML file into a `Config` (raises if the root is
  not a mapping).
- Both `cfg.x` and `cfg.get("x", default)` work, because it is still a `dict`. Factories
  rely on `.get(...)` for optional keys, which is why most features can be omitted from a
  config and fall back to a default.

Because `Config` is a live dict (not a frozen snapshot), the training script can inject
resolved runtime values into it (e.g. `experiment.run_dir`, `checkpoint.dirpath`) before
handing it to builders.

## How a config maps to components

A config is a flat set of top-level sections, each consumed by one subsystem:

| Section | Consumed by | Purpose |
|---------|-------------|---------|
| `project` | `seed_everything`, trainer | seed, output dir, determinism |
| `experiment` / `logging` | `utils/experiment.py`, loggers | run name, output root, W&B |
| `data` | `FaceOcclusionDataModule`, `FaceOcclusionDataset` | CSV paths, image root, column names, `target_scale`, gender encoding |
| `split` | `data/splits.py` | strategy, `occlusion_bins`, `stratify_by`, `val_size`, `split_path` |
| `augmentation` | `data/transforms.py` (+ `augmentation.background.*` for background aug) | resize, flip, jitter, rotation |
| `synthetic_occlusion` | `data/synthetic_occlusion.py`, cache | occluders, severity bands, cache dir |
| `sampler` | `data/samplers.py` | sampler strategy + parameters |
| `model` | `models/regressor.py` (`build_model`) | backbone, head, LoRA, ordinal head |
| `losses` | `training/lit_module.py` | the loss stack (regression + auxiliaries) |
| `training` | `lit_module.configure_optimizers`, trainer | LRs, schedule, epochs, precision |
| `checkpoint` | `training/callbacks.py` | monitor, mode, save_top_k |
| `inference` | `scripts/inference/predict_test.py` | TTA flag |

**Seeding (`project.seed`).** A fixed int reproduces a run (and is right for clean ablations).
Setting it to `null` / `"random"` / omitting it makes `scripts/training/train.py::_resolve_seed`
draw a fresh random seed per run and write it back into the saved `config.yaml` + `metadata.json`
— so you get exploration (different inits / data orderings) while staying reproducible (re-run
the saved config). Only training randomness keys off this; the val split uses
`split.random_state` and is saved-then-reloaded, so a random seed does **not** change the split,
and paired comparisons stay valid (pin the same seed across an ablation pair to remove seed
variance from the Δ).

The mapping is deliberately mechanical: to know what a key does, find the factory that
reads it. The per-section detail lives in the chapter for that subsystem (linked above and
in the [index](README.md)).

## The gating philosophy: default-OFF, baseline-preserving

The library carries a full multi-task / imbalanced-regression stack, but **every optional
mechanism is gated by a flag that defaults to disabled**, and the code is written so that
when a flag is off the computation is bit-identical to the baseline. Examples (all detailed
in later chapters):

- `model.use_ordinal_head` + `losses.ordinal.enabled` — ordinal threshold head + weighted BCE.
- `losses.consistency.enabled`, `losses.monotonicity.enabled` — ordinal-coupled regularizers.
- `losses.ranking.enabled` + `synthetic_occlusion.enabled` — synthetic monotonic ranking.
- `losses.regression.reweight` — distribution-aware sample reweighting (`none|balanced|test_matched`).
- `sampler.enabled` — gender × occlusion balanced sampling.
- `model.lora.enabled` — PEFT LoRA fine-tuning.
- `augmentation.background.enabled` — feathered, label-preserving background perturbation
  (modes: replace/brightness/noise/blur/shuffle/texture).
- `losses.bg_consistency.enabled` — background-invariance consistency loss (needs background
  augmentation + a face-mask source).
- `training.ema.enabled` — EMA of weights (validation/checkpointing use the averaged model).

Misconfigurations are caught early: e.g. enabling `losses.consistency`/`monotonicity`
without `model.use_ordinal_head` raises `ValueError` at module init
(`training/lit_module.py`), and `model.use_ordinal_head` together with an MLP head or LoRA
raises in `OcclusionRegressor.__init__`.

## Config groups (ablation sets)

`configs/` holds one YAML per experiment, organised by role (see `configs/README.md` for the
index):

- `configs/baseline.yaml` — the canonical ConvNeXt-Small champion; the paired-Δ reference.
- `configs/ensemble/` — the members that, averaged with the baseline, give the best submission
  (`convnext_base`, `gender_balanced`, `sigmoid`, `expectation_dldl`).
- `configs/experiments/` — one runnable exemplar per explored method (DINOv2+LoRA, synthetic
  ranking, distribution reweighting, ordinal head, ordinal consistency, background invariance,
  balanced sampler, shadow head, gender invariance).
- `configs/eval/` — non-training helpers (e.g. `test_distribution.yaml`).

The convention for a new run: copy the closest existing config, change `experiment.name`
and the one or two keys under study, and keep everything else identical so a paired
comparison ([06](06-metrics-and-evaluation.md), `compare_experiments`) isolates the change.
