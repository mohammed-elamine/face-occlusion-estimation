# 06 — Metrics & Evaluation

Evaluation is **CI-first**: the high-occlusion val tail is tiny and the split is
identity-leaked, so raw point deltas are not trusted. Code lives in
`src/face_occlusion/metrics/`; the analysis scripts that consume it are in
`scripts/analysis/` ([07](07-pipeline-and-experiments.md)).

## The challenge metric — `metrics/challenge_metric.py`

- `weighted_mse(preds, targets, clip=True, sample_weight=None)` — per-row weight
  `w = 1/30 + y`; predictions clipped to `[0,1]` when `clip=True`. `sample_weight` enables
  the evaluation lenses below.
- `challenge_score(preds, targets, genders, female_value, male_value, clip=True,
  sample_weight=None)` — returns `{score, err_female, err_male, gender_gap, err_mean}` where
  `score = (Err_F + Err_M)/2 + |Err_F − Err_M|`. **This is the optimization target** and the
  `val/score` the checkpoint monitors.
- `error_by_occlusion_bin(...)` — per-bin weighted MSE on `DEFAULT_BINS =
  (0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0)`.
- `weighted_mse_by_group(...)` — per-group error (used for group diagnostics).

The decomposition matters: improving the **mean** error and closing the **gender gap** are
separate levers, and the metric rewards both.

## Bootstrap confidence intervals — `metrics/bootstrap.py`

The core of the gate. Resampling is on validation rows only (no retraining).

- `MetricCI` — frozen `{point, lo, hi, std}`.
- `bootstrap_challenge_metrics(preds, targets, genders, *, group_ids, unit, n_boot, ci,
  seed, ...)` — percentile CIs for `score, err_female, err_male, gender_gap, err_mean,
  high_occ_err, high_occ_gender_gap`. `unit="group"` with `group_ids` resamples **identity
  clusters** (honest under leakage); `unit="row"` is i.i.d. row bootstrap.
- `bootstrap_per_bin(...)` — per occlusion-bin `count`, `weighted_mse` (CI), and
  `score_share` (CI, fraction of total weighted error).
- `bootstrap_score_delta(preds_a, preds_b, targets, genders, ...)` — **paired** bootstrap of
  `metric(A) − metric(B)` on identical resampled rows. This is the right test for "did B
  beat A": per-row errors are correlated, so the paired Δ CI is much tighter than comparing
  two marginal CIs, and a Δ CI excluding 0 means a real difference.

## Evaluation lenses — `metrics/eval_lenses.py`

Lenses reweight the *same* validation predictions to ask "what would the score be under a
different occlusion distribution" — **diagnostic only; selection is always the official,
unweighted metric.**

- `LENS_NAMES = ("official", "balanced", "test_matched")`:
  - `official` → no reweighting (`sample_weight=None`).
  - `balanced` → uniform occlusion bins (`balanced_proportions`).
  - `test_matched` → the digitized test-set distribution (`load_test_distribution`, from
    `configs/eval/test_distribution.yaml`).
- `importance_weights` / `per_bin_importance_weights` — per-row / per-bin weights
  `p_target(bin)/p_train(bin)`, mean-normalized, clipped, smoothed.
- `rebin_proportions` — re-aggregate a fine source histogram onto coarser edges.

The same operators feed three places, so they share one definition of "target
distribution": the **sampler** (`sampler.target`), the **loss reweighting**
(`losses.regression.reweight`), and the **measurement** lenses here.

## Leakage-free scoring

Because the default split is row-stratified (identities can appear in both train and val),
the gate also reports **seen vs unseen** identity subsets (rows whose `group_id` is/ isn't
present in train). The unseen subset is the leakage-honest estimate of generalization;
`bootstrap` with `unit="group"` complements it by resampling identity clusters.

## How the gate is used in practice

1. Train → `predictions/val_predictions.csv` (carries gender, target, raw+clipped preds,
   `group_id`, bins).
2. `scripts/analysis/analyze_val_predictions.py` → `reports/summary_metrics.json` with a
   `robust` block (row+group CIs, the three lens scores, per-bin contribution, leakage-free
   seen/unseen), plus tables and plots.
3. `scripts/analysis/bootstrap_metrics.py` → quick CIs on any predictions CSV.
4. `scripts/analysis/compare_experiments.py` → **paired-Δ** vs a baseline (the decisive
   test for an ablation), with per-lens scores and a forest plot.

Rule of thumb baked into the workflow: never compare raw tail deltas; require the paired-Δ
CI to exclude 0 (or a clear, consistent shift across lenses) before believing a change
helped.
