# ConvNeXt champion ablation

ConvNeXt-small full fine-tune is our best model (val/score **0.00129**), and it has
**plateaued** (flat from ~epoch 17). These configs each change **exactly one thing** from
`00_baseline.yaml` (the champion recipe re-run on current code) so a paired-Δ isolates the
effect. Gate every variant:

```bash
python -m scripts.training.train --config configs/convnext_ablation/0X_*.yaml
python -m scripts.analysis.compare_experiments \
  --runs <variant_run> <00_baseline_run> --baseline <00_baseline_run>
```

Run `00_baseline` once first (same-code reference), then the variants (2 GPUs → run two at a
time). A variant is only adopted if its Δscore CI is **below 0**.

## Order (by expected value; each = champion + one change)

| # | Config | Change | Rationale / odds |
|---|--------|--------|------------------|
| 01 | `01_convnext_base` | `convnext_small → convnext_base` | **Top pick.** A real capacity bump for a CNN that works and has plateaued (unlike DINOv2, where capacity didn't help — its frozen SSL features resist adaptation). More compute. |
| 02 | `02_gender_balanced_loss` | `weighted_mse → gender_balanced` | Metric-aligned: the score is `0.5(Err_F+Err_M)+\|gap\|`; this trains that structure and targets the gap (~43% of the score). Cheap. |
| 03 | `03_sigmoid` | `identity+clip → sigmoid` | Bounded output, possible boundary calibration gain. Cheap, modest odds. |
| 04 | `04_mlp_head` | linear head → MLP head + discriminative LRs | "Stronger head" (as for DINOv2). Lower odds — a fully fine-tuned CNN's linear head is usually enough — but cheap. |
| 05 | `05_champion_ema` | champion + `training.ema` | **EMA of weights** — a free ensemble over the optimization trajectory; our single-run stand-in for multi-seed averaging. Validation/checkpointing run on the EMA model. Highest-confidence small gain. |
| 06 | `06_gender_balanced_ema` | `02` + EMA | gender_balanced loss + EMA — a strong ensemble member. |
| 07 | `07_convnext_base_ema` | `01` + EMA | bigger backbone + EMA — a diverse, stronger ensemble member. |
| 08 | `08_ordinal_expectation_dldl` | linear head → distribution head + `dldl` loss | Ordered-bin **expectation** head (DEX) with **DLDL/LDS** soft labels — the literature's imbalanced-regression approach. A *decorrelated* (classification-trained) ensemble member. See [docs/architecture/09](../../docs/architecture/09-imbalanced-regression-and-expectation-head.md). |

## Results (2026-06-14, all seed 42, paired-Δ vs champion 0.00129)

01_convnext_base 0.00133, 02_gender_balanced 0.00128, 03_sigmoid 0.00133, 04_mlp_head 0.00137 —
**all `ns`** (no single lever significantly beats the champion; the "capacity top pick" did not
pan out). **But the ENSEMBLE of {champion, 01, 02, 03} (averaged predictions) = 0.00121, paired-Δ
−0.000074 [−0.00013, −0.00002] → significantly better** — the first real win. The 05–07 EMA runs
test EMA in isolation *and* are meant to strengthen each ensemble member (and thus the ensemble).

## Explicitly rejected (with evidence)

- **torch.hub backbone** — the torch.hub win was DINOv2-specific (its 224 position-embedding
  interpolation); ConvNeXt's timm `convnext_small.fb_in22k_ft_in1k` is the canonical strong
  supervised checkpoint, no transfer.
- **background augmentation** — already tested (`convnext_small_bg_invariance`): paired Δ
  **+0.000115, significantly worse**. Do not repeat.
- **more epochs** — champion is plateaued by epoch ~17.

## Not config-only (recommended separately)

- **EMA / weight averaging** — the most reliable near-free generalization gain; needs a small
  Lightning callback (not yet implemented). Highest-confidence small win.
- **TTA (hflip) + ensemble / multi-seed** — submission-time boosts. They do **not** show in
  val/score (val is computed without TTA), but help the actual leaderboard number — use them
  when submitting regardless of this ablation.

Reminder: all models still saturate (`pred_max ≈ 0.5`); these levers move the bulk, not the
high-occlusion tail, which stays unmeasurable on our 57-row val tail.
