# Background-invariance ablation

Tests whether **background-invariance** (feathered label-preserving background augmentation +
a `bg_consistency` loss over two background-randomized views) improves a model by forcing it
to read occlusion from the face, not the background. One config per backbone, each = its
no-bg baseline + the bg-invariance stack:

| config | baseline to compare against (paired-Δ) |
|--------|----------------------------------------|
| `convnext_small_bg_invariance.yaml` | ConvNeXt champion `…_baseline_convnext_small_wd3e-2` (val 0.00129) |
| `dinov2_full_finetune_bg_invariance.yaml` | `…_dinov2_vitb_reg_full_finetune` (in `configs/dinov2_lora/`-family runs) |

Note the DINOv2 baseline lives in a different config family — comparison groups span config
directories (see `outputs/experiments/runs_organisation.md`).

**Requires a face-mask store** (`python -m scripts.data.build_face_masks`); without masks both
the augmentation and the consistency loss are silent no-ops.

**Result (2026-06-14):** rejected — bg-invariance was *significantly worse* on ConvNeXt
(Δ +0.000115) and *neutral* on DINOv2 (Δ −0.000033, ns); neither beat the ConvNeXt champion.
It improved the high-occ tail / closed the gap but raised bulk error and hurt unseen-identity
generalization. Kept for the record; do not build on it.
