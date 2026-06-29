# Synthetic Occlusion Generation

This document covers the synthetic occlusion pipeline in
[src/face_occlusion/data/](../src/face_occlusion/data/). It generates occluded
views of real faces and feeds them to the **synthetic monotonic ranking loss**
(`losses.ranking`, implemented). Synthetic views never receive a regression
label — only a relative ordering.

## 1. Why synthetic views, and why no regression label

Real images carry the official `FaceOcclusion` score: a calibrated number we
supervise directly with weighted MSE. A synthetic view — a real face with an
overlaid mask, hand, … — has *no* trustworthy ground-truth score. We only know
the **ordering**: the original is less occluded than a mildly occluded version,
which is less occluded than a strongly occluded one.

```text
real images      → regression + ordinal losses (calibration)
synthetic images → ranking only, never regression
```

So the generator exposes only a *severity proxy* (§5) in metadata; it is never
written into `batch["target"]`. The ranking loss orders the scalar predictions
`s(clean) < s(mild) < s(strong)`.

## 2. Region provider (MediaPipe)

Region localisation is **MediaPipe-only** — there is no geometric fallback,
because a wrong mask would create a wrong ordering. Config: `region_provider: mediapipe`.

The provider returns a small contract (`FaceRegionResult`): `valid`, the region
`masks` (bool `(H, W)` arrays), the dense `landmarks` (pixel coordinates, used by
the realistic occluders), a `failure_reason`, and `metadata`. Required masks:
`face, left_eye, right_eye, eyes, nose, mouth, lower_face, cheeks, forehead_chin,
background`. On any detection/sanity failure the result is `valid=False` with a
compact reason (`no_face_detected`, `missing_landmarks`, `invalid_face_mask`, …).

## 3. Occluder types

The occluder appearance is pluggable via `synthetic_occlusion.occluder_types`.
There are two **realistic** types (the default going forward) and the original
**geometric** primitives (kept for ablation / backward compatibility).

### Realistic occluders (landmark-driven, recommended)

| Type | Source | How it's placed |
|---|---|---|
| `realistic_mask` | MaskTheFace templates (MIT, committed under `assets/occluders/masks/`) | `synthetic_mask_occluder.py` warps a mask template onto the face via a perspective transform on landmark anchors; `coverage_level` raises the top edge from the mouth (mild) to the nose bridge (strong) |
| `realistic_hand` | 11k Hands cut-outs (academic-fair-use, generated locally — see §4) | `synthetic_hand_occluder.py` places a hand cut-out with a *similarity transform* (no shear) at a plausible spot (chin/cheek/mouth/forehead) oriented so the **wrist exits the frame**, sized to the face, then **skin-tone matched** to the face |

Both render an RGBA occluder, then hand it to the shared compositor (§6).

### Geometric primitives (legacy, ablation only)

`mask_like_lower_face`, `sunglasses_like_eyes`, `random_face_rectangle`,
`random_textured_polygon`, `blurred_patch` — flat shapes with a small natural
palette. They look unnatural and a CNN separates them trivially from real
occlusion, so they are not used for the main runs; they remain available for the
geometric-vs-realistic comparison.

## 4. Hand assets (generated locally, not committed)

11k Hands images are released for "reasonable academic fair use", **not**
redistribution, so hand assets are **not** committed. Generate them once into a
git-ignored directory:

```bash
# download Hands.zip (~632 MB) from https://sites.google.com/view/11khands, then:
python -m scripts.data.prepare_hand_assets --hands-dir /path/to/Hands --num 50
# -> assets_private/occluders/hands/*.png + anchors.json + PROVENANCE.md
```

The script segments each hand off its white background and **auto-derives** the
`palm` and `wrist` anchors (no manual labels). Mask templates, by contrast, are
MIT-licensed and committed under `assets/occluders/masks/`.

## 5. Severity proxy and bands

For an occluder mask `M`, the weighted coverage proxy is:

```text
ρ(M) = Σ_r  weight_r · |M ∩ R_r| / |face|
```

with default region weights `eyes 1.0, mouth 0.85, nose 0.75, cheeks 0.45,
forehead_chin 0.35, background 0.0`. `ρ` is clamped to `[0, 1]` and is a coverage
proxy, **not** a label.

Each level (`mild`, `strong`) has an accept band; the generator uses
acceptance-rejection (up to `max_attempts` retries per level) and enforces
`mild.severity < strong.severity`. **Bands are tuned to the occluder's natural
coverage range** — the geometric defaults (`mild 0.05–0.15`, `strong 0.35–0.60`)
do not fit a real mask, so the mask config uses `mild 0.08–0.20, strong
0.28–0.55` (≈70 % pair yield instead of ≈10 %), and the hand config widens the
strong band to `0.60` because hands cover more.

## 6. The compositor (the "no-seam" engine)

`synthetic_compositing.py` blends an RGBA occluder into the face so it belongs in
the photo — pasting looks fake mostly because of the *seam*, not the shape.
Toggleable steps (`CompositingConfig`):

- **feather** the alpha edge (no hard cut),
- **harmonize** the occluder luminance toward the local face lighting,
- **color_match**: Reinhard mean/std transfer toward a reference region (used for
  hands → match the face skin tone; off for masks, which keep their own colour),
- soft **contact shadow** beneath the occluder,
- light **grain** so the occluder isn't suspiciously clean,
- optional Poisson **seamless** clone (off by default — it would wash out colour).

The same engine serves masks, hands, and any future occluder.

## 7. One occluder type per triple (and why)

`generate_pair` fixes the occluder **type once per anchor** and uses it for both
the mild and the strong view. So every `clean → mild → strong` triple is
single-type — all masks *or* all hands — never mixed within a triple. Across
anchors, both types are produced (≈40 % masks / 60 % hands in practice), so both
appear at both severity levels.

**Why same-type triples:**

1. **Keep each triple a clean coverage-only comparison.** The ranking loss orders
   by occlusion *amount*. If a triple mixed types (mask-mild → hand-strong), the
   model could "explain" the ordering by the *type change* (hand vs mask) instead
   of the coverage — a shortcut. With the type held constant, the only thing that
   differs between mild and strong is how much of the face is covered.

2. **Avoid re-creating a "hand = high occlusion" shortcut.** Occluder types have
   different natural coverage ranges (masks ≈ 0.14–0.35, hands ≈ 0.20–0.60). If
   the type were drawn freely per view, the *strong* level would be dominated by
   hands (they reach high coverage more easily), silently correlating "hand" with
   "high occlusion". Fixing one type per anchor forces masks to also produce
   strong views and hands to also produce mild views, so **occluder type is
   decorrelated from occlusion level** across the dataset. The model must order by
   coverage, not by "mask vs hand".

This is implemented by drawing one type per `generate_pair` and passing it as
`forced_type` to both `_sample_view` calls. (A geometric, composing config is
unaffected — it still mixes primitives within a strong view.)

## 8. Precompute into a cache (don't generate live)

Generation is slow (MediaPipe + acceptance-rejection + compositing), so views are
built **once, offline** into a cache and loaded like ordinary images at train
time. Live generation would bottleneck the GPU.

```bash
python -m scripts.data.build_synthetic_cache \
  --config configs/experiments/synthetic_ranking.yaml \
  --cache-dir data/synthetic_cache/masks_hands_v1 \
  --target-min 0.10 --max-per-bin-gender 200
```

The builder samples **train-split anchors only** (no validation faces), balanced
by occlusion-bin × gender, runs `generate_pair`, and writes per anchor:
`views/<i>_clean.webp | _mild.webp | _strong.webp`, a face `mask`, and a
`manifest.csv` row (paths, severities, `*_occluder_type`, bin, gender). Only valid
pairs are kept. It prints a bin×gender coverage table. Caches live under `data/`
(git-ignored).

## 9. Dataset / DataModule integration

With `synthetic_occlusion.enabled` **and** `return_in_batch` true, the **train**
dataset attaches (default source: the cache via `use_cache` + `cache_dir`; the
on-the-fly generator is the fallback):

| Key | Type | Notes |
|---|---|---|
| `synthetic_clean_image` | float `(3, H, W)` | un-augmented original — the bottom of the ranking ordering |
| `synthetic_mild_image` | float `(3, H, W)` | normalised, no spatial aug |
| `synthetic_strong_image` | float `(3, H, W)` | normalised, no spatial aug |
| `synthetic_mild_severity` | scalar | severity proxy, never a label |
| `synthetic_strong_severity` | scalar | severity proxy, never a label |
| `synthetic_valid` | bool scalar | False when the anchor has no valid views |
| `synthetic_failure_reason` | string | empty when valid |

Ranking compares `synthetic_clean_image` (not the augmented `image`) against
mild/strong, so the ordering isn't polluted by flip/rotate. Validation/test
datasets never get synthetic views — evaluation stays clean.

## 10. The ranking loss (Stage 4, implemented)

`FaceOcclusionLitModule.training_step` forwards the valid `clean/mild/strong`
views through the regression head and adds a RankNet term
`−logσ(s_mild − s_clean) − logσ(s_strong − s_mild)` over `synthetic_valid` rows,
at a small warmed-up weight (`losses.ranking`). It logs `train/loss_rank`,
`train/lambda_rank`, and `train/rank_ordering_acc`. See `models/ranking.py`.

Because ranking lands on the *calibrated* head, the weight is kept small and
warmed, and low-bin calibration is watched. The accept/reject rule for the whole
approach: keep ranking only if **real** high-occlusion error improves within CI
on both splits — not if synthetic ordering accuracy alone rises.

## 11. Config schema

```yaml
synthetic_occlusion:
  enabled: false                 # master switch (default off -> baseline unchanged)
  return_in_batch: false         # attach synthetic_* keys to train batches
  use_cache: true                # prefer the precomputed cache over on-the-fly
  cache_dir: data/synthetic_cache/masks_hands_v1
  region_provider: mediapipe
  occluder_types: [realistic_mask, realistic_hand]   # one is picked per anchor
  severity:
    mild:   { min: 0.08, max: 0.20 }
    strong: { min: 0.28, max: 0.60 }
  mask:                          # realistic_mask params (optional)
    templates: [surgical, cloth, KN95, N95]
    compositing: { feather_px: 2.0, harmonize_strength: 0.5, shadow_strength: 0.30 }
  hand:                          # realistic_hand params (optional)
    asset_dir: assets_private/occluders/hands
    compositing: { color_match: true, color_match_strength: 0.7 }
  max_attempts: 50
  seed: 42
```

The ready-made exemplar is `configs/experiments/synthetic_ranking.yaml` (masks + hands —
one occluder per view). MediaPipe is the optional
`synthetic` dependency group. When the installed MediaPipe lacks `mp.solutions`
(the Tasks backend), it needs a Face Landmarker `.task` asset; the provider
searches `$FACE_OCCLUSION_MEDIAPIPE_FACE_LANDMARKER`, then `models/mediapipe/`,
`assets/mediapipe/`, `data/mediapipe/`. Get it once:

```bash
make mediapipe-model        # downloads to models/mediapipe/
```

The cache/mask builders (`build_synthetic_cache`, `build_face_masks`) also
auto-download it to `models/mediapipe/` on demand.

## 12. Audits and the realism probe

**Visual / coverage audit** — `scripts/analysis/generate_synthetic_occlusion_audit.py`
writes an `audit_grid.png` (original | regions | mild | strong), per-sample PNGs,
row + grouped metric CSVs, and a **coverage-by-bin×gender** table (MediaPipe
success rate, the Stage 3→4 gate). Use `--coverage-only` for large stats runs and
`--target-min/--target-max/--database/--gender` to focus.

**Realism probe** — `scripts/analysis/realism_probe.py` trains a synthetic-vs-real
classifier and reports the AUC (≈0.5 indistinguishable, ≈1.0 a big gap). It
confirmed two things: (a) realistic masks/hands look far better than geometric
patches by eye, but (b) the AUC stays ≈1.0 for *any* cut-paste approach — the gap
is dominated by **synthetic-ness** (the compositing signature + clean-source-face
confound), not occluder type. So the probe is a "gap exists" indicator, not a
realism tracker; whether the gap *hurts* is decided by the ranking ablation.

## 13. Semantic overlap diagnostics

Each accepted view records diagnostics in `SyntheticOcclusionView.metadata` /
the audit CSV (diagnostic only — they do not reject samples):
`face_overlap_ratio`, `background_overlap_ratio`, `important_region_overlap`,
`eye/mouth/nose/lower_face_overlap_ratio`, `occluder_area_ratio`,
`occluder_face_area_ratio`, `weighted_severity`, plus warning flags
`mostly_background_occlusion` (`background_overlap > 0.35`),
`low_important_region_overlap` (`important_region_overlap < 0.05`), and
`high_attempt_count` (`≥ 0.8 · max_attempts`). These help spot "strong" views that
are heavy only because of large background/low-importance coverage.
