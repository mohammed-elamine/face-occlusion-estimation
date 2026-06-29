# Background-invariance augmentation

Exemplar config (`convnext_small_bg_invariance.yaml`) for **background-invariance**: perturb only the
non-face pixels (face region from the MediaPipe mesh) and add a consistency loss between two
background-randomised views of the same crop, pushing the model to read occlusion from the face, not
the background. See [`docs/architecture/03-data.md`](../../docs/architecture/03-data.md) and
[`05-training.md`](../../docs/architecture/05-training.md).

```bash
python -m scripts.training.train --config configs/bg_invariance/convnext_small_bg_invariance.yaml
```

Finding: **rejected** — paired Δ `+0.000115`, significantly worse than the champion. The DINOv2
variant was pruned after the study.
