# 03 — Data Subsystem

All data code lives in `src/face_occlusion/data/`. The subsystem turns `train.csv` + a
crop directory into reproducible train/val datasets whose items carry every field the
gender-aware metric and the diagnostics need.

## Splits — `data/splits.py`

Splits are **built once, saved, then reloaded** so a run is reproducible even if the global
split file changes later.

- `make_stratified_split(df, target_col, gender_col, id_col, bins, val_size, seed, min_per_stratum, strategy, stratify_by, group_col) -> DataFrame`
  - `strategy="row_stratified"` (default): `train_test_split` stratified on a concatenated
    key (default `gender × occlusion_bin × database`), via `_row_split`. Rare strata
    (`< min_per_stratum`) are merged into a `_rare_` catch-all so stratification never
    raises; it falls back through `[gender, occlusion_bin]` → unstratified if a column set
    fails.
  - `strategy="group_stratified"`: `StratifiedGroupKFold` keyed by `group_col` so **no
    identity appears in both train and val** (`_group_split`); a robustness check, not the
    default.
- Targets are normalized and binned (`assign_occlusion_bin`) **before** splitting, so both
  splits use `[0,1]` targets and consistent bins.
- `save_split` / `load_split` persist the keyed frame (`id_col`, `split`, plus
  `database`, `source_subfolder`, `group_id`, `occlusion_bin`, `gender` for diagnostics).
- Path-derived metadata comes from `data/metadata.py::add_path_metadata` (parses
  `database`, `source_subfolder`, `group_id`, and a regex `FaceId-(\d+)` → `face_id`).

## Target normalization — `data/normalize.py`

- `normalize_target(values, scale)` — `scale ∈ {unit, percent, auto}`; `auto` divides by
  100 only if `max > 1.5` (so a percent-scaled CSV is detected automatically). Used by
  splits, sampler, and dataset so they all agree on the `[0,1]` scale.
- `assign_occlusion_bin(values, edges)` — `np.digitize` on interior edges; bins are
  `[lo, hi)` except the last `[lo, hi]`.

## Transforms — `data/transforms.py`

Augmentation is **deliberately conservative**: anything that changes face visibility would
silently corrupt supervision (the label is fixed). The pipeline therefore allows only flip,
mild color jitter, and small rotation — no RandomErasing, crops, heavy blur, or synthetic
occlusion on the standard path.

- `build_train_transform(cfg)` — `Resize → RandomHorizontalFlip(p) → RandomApply(ColorJitter
  [brightness, contrast, saturation, hue], p) → RandomRotation(degrees) → ToTensor →
  Normalize(ImageNet mean/std)`.
- `build_eval_transform(cfg)` — `Resize → ToTensor → Normalize` only.
- `build_synthetic_view_transform(cfg)` — `ToTensor → Normalize` only; synthetic views skip
  spatial augmentation so the ranking head sees the exact pixels the generator approved.

## Dataset — `data/dataset.py`

`FaceOcclusionDataset(Dataset)` returns a **dict per item** (not a bare tensor) because the
metric and analysis group by gender/identity. `__getitem__(idx)` yields:

| key | type | notes |
|-----|------|-------|
| `image` | `Tensor (3,H,W)` | after transform |
| `target` | `Tensor` (float32) | normalized `[0,1]`; train/val only |
| `gender` | `Tensor` (float32) | `female=0.0`, `male=1.0`; train/val only |
| `image_id` | `str` | the `id_col` value (key for masks/cache) |
| `filename`, `path` | `str` | relative + absolute paths |
| `database`, `source_subfolder`, `group_id` | `str` | path-derived; for splits/diagnostics only |
| `face_id` | `int` | parsed from filename (`-1` if absent) |
| `synthetic_clean_image` / `_mild_image` / `_strong_image` | `Tensor` | only if synthetic views enabled |
| `synthetic_mild_severity` / `_strong_severity` | `Tensor` | severity proxies (not labels) |
| `synthetic_valid` | `Tensor[bool]` | True if the triple is usable |

Key behaviors: path metadata is **never fed to the model** (splits/diagnostics only);
synthetic views and background augmentation are **train-mode only** (val/test stay clean);
per-sample synthetic RNG is seeded from `(synthetic_seed, idx)` so views are reproducible
and uncorrelated across workers; a cached synthetic view takes precedence over on-the-fly
generation.

## DataModule — `data/datamodule.py`

`FaceOcclusionDataModule(pl.LightningDataModule)` wires everything:

- `prepare_data()` — create the split lazily (once) if `split.split_path` is missing.
- `setup(stage)` — build train/eval transforms; load the synthetic cache and/or face-mask
  store if configured; merge `train.csv` against the saved split (warns loudly on mismatch
  unless `split.allow_missing_rows`); construct the three datasets.
- `train_dataloader()` — three-way dispatch:
  1. `build_batch_sampler_from_config` → if not `None`, a `DataLoader(batch_sampler=...)`;
  2. else `build_weighted_sampler_from_config` → if not `None`, `DataLoader(sampler=...,
     batch_size, drop_last)`;
  3. else a plain shuffled loader.
- `val_dataloader` / `test_dataloader` / `predict_dataloader` — fixed `batch_size=128`, no
  shuffle, no drop_last.
- Worker reproducibility via `seed_worker` + `make_dataloader_generator`
  (`utils/reproducibility.py`); pinned memory only when CUDA is available.

## Samplers — `data/samplers.py`

Two opt-in strategies (`sampler.enabled: true`), both keyed to the gender × occlusion grid:

1. **`gender_occlusion_balanced_batch`** → `GenderOcclusionBalancedBatchSampler` (a
   *batch* sampler). It groups samples into `(gender, occlusion_bin)` strata, computes a
   per-stratum probability that mixes the natural and a balanced distribution
   (`balance_strength`), applies an inverse-frequency gender correction
   (`gender_balance_strength`) and size-aware damping for tiny strata, and enforces a **hard
   per-image repeat cap** (`max_repeats_per_image`) so the rare tail cannot be memorized.
   `target ∈ {bin_weights, balanced, test_matched}` selects the per-bin weighting (the
   latter two reuse the eval lenses so sampling, loss reweighting, and measurement share one
   target distribution). It writes a `reports/sampler_summary.json`.
2. **`gender_occ_weighted`** → `build_weighted_sampler_from_config` returns a torch
   `WeightedRandomSampler` with per-sample weights from `compute_weighted_sample_weights`
   (`mode ∈ {gender, occ, gender_occ, cell}`, default `gender_occ`:
   `w = inv_freq(gender) · (1/30 + occ)^occ_power`). This mirrors a simpler exposure-based
   recipe; it draws with replacement and plugs into `DataLoader(sampler=...)`.

`build_batch_sampler_from_config` returns `None` (defers) for the weighted strategy and
raises on a genuinely unknown strategy. Design notes are in `docs/balanced_batch_sampler.md`.

## Synthetic occlusion (label-aware, opt-in)

Distinct from the conservative standard augmentation: synthetic occlusion **manufactures
high-occlusion training signal** and is used only for a **ranking** objective ([05](05-training.md)),
never to relabel real images. Requires the `synthetic` extra (MediaPipe).

- `data/synthetic_occlusion.py` — `SyntheticOcclusionGenerator.generate_pair(image)` builds
  a `clean < mild < strong` triple by acceptance-rejection on a **severity proxy** `ρ =
  Σ_r w_r · |occluder ∩ region_r| / |face|` (region masks from MediaPipe via
  `MediaPipeFaceRegionProvider`). `build_generator_from_config(cfg)` constructs it; severity
  bands, region weights, and occluder types are config-driven. The proxy gates view
  acceptance — it is **not** a regression label.
- Occluders: geometric primitives (mask-like, sunglasses, rectangles, textured polygons,
  blurred patch) plus realistic warps — `data/synthetic_mask_occluder.py` (MaskTheFace-style
  perspective-warped masks) and `data/synthetic_hand_occluder.py` (landmark-placed hands).
- `data/synthetic_compositing.py` — `composite_occluder(...)` + `CompositingConfig`: the
  shared "no-seam" blend engine (feather, luminance harmonize, contact shadow, optional
  skin-tone transfer for hands, grain, optional Poisson clone).
- `data/synthetic_cache.py` — offline cache: `SyntheticCache` (manifest-keyed lookup of
  precomputed `clean/mild/strong` views + masks), `select_balanced_anchors`,
  `coverage_table`. The default path is to precompute a cache (see
  [07](07-pipeline-and-experiments.md), `build_synthetic_cache`) and load it via
  `synthetic_occlusion.use_cache`.

## Background augmentation & mask store

- `data/background_augment.py` — `BackgroundAugment` perturbs **only non-face pixels**
  (`replace`/`brightness`/`noise`) using a cached face mask, so the occlusion label is
  unchanged. Applied with probability `p`, per-sample-seeded; a missing mask is a safe
  no-op.
- `data/face_mask_store.py` — `FaceMaskStore`: a full-coverage precomputed face-mask store
  (deterministic path per `id`), decoupled from the synthetic cache so background
  augmentation can cover **all** training images, not just synthetic anchors. The datamodule
  prefers this store and falls back to the synthetic cache's masks.
