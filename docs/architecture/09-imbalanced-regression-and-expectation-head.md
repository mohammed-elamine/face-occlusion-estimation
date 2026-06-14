# 09 — Imbalanced Regression: the Ordered-Bin Expectation Head (DEX + DLDL/LDS)

## Why this exists

The occlusion target is extremely imbalanced: of 100k rows, ~40k sit in `[0, 0.05]` but only
**211 are above 0.4 and 8 above 0.6**. Plain MSE is dominated by the low-occlusion bulk, so the
encoder predicts near the conditional mean and **never commits to high values** — predictions
saturate around 0.5 and the high-occlusion tail is severely under-predicted. No backbone /
capacity / loss-weighting / head / EMA change fixed this, because they cannot create tail signal
that the data lacks.

This head reframes the problem the way the imbalanced-/ordinal-regression literature does, and
serves primarily as a **decorrelated member for the prediction ensemble** (our one proven lever).

## Problem formalization

Target `y ∈ [0, 1]`. Pick `K` ordered bin centers `c_1 < … < c_K` spanning `[0, 1]`.

- **Soft label (DLDL + LDS):** instead of a scalar/one-hot target, build a smoothed distribution
  over bins, `p*_k(y) ∝ exp(−(c_k − y)² / 2σ²)`, normalized over `k`. The width `σ` spreads each
  sample's supervision onto neighbouring bins, so the data-poor tail **borrows statistical
  strength** from its populated neighbours (the core idea of LDS / label-distribution learning).
- **Model:** pooled features `→ logits ∈ ℝ^K`; `p = softmax(logits)`.
- **Objective:** `L = D_KL(p* ‖ p) + λ_exp · w·(ŷ − y)²`, where `ŷ = Σ_k p_k c_k` is the
  expectation and `w = 1/30 + y` is the challenge weight. The KL term handles imbalance and
  ordinality (via the ordered, smoothed labels); the expectation-MSE term keeps the continuous
  output aligned to the actual challenge metric.
- **Prediction:** `ŷ = Σ_k p_k c_k`, clipped to `[0, 1]`. It is a convex combination of the
  centers, hence bounded by construction — **no output activation is needed.**

## Literature

- **DEX** (Rothe et al., *IJCV* 2018) — classify into bins, predict the **softmax expectation**;
  the canonical "classification-for-regression" approach, from facial age estimation (our
  closest analog: continuous, ordinal, imbalanced, from a face crop).
- **DLDL** (Gao et al., 2017) / **Label Distribution Learning** (Geng, *TKDE* 2016) — train
  against a soft label *distribution* so neighbouring labels share supervision.
- **Deep Imbalanced Regression / LDS** (Yang et al., *ICML* 2020) — label-distribution smoothing
  lets data-poor regions borrow strength from data-rich neighbours.

The field converged on this family rather than hard two-stage "classify-bin → bin-specialist
regressor" routers, which fail under our imbalance (the router can't detect the 0.27% tail, and a
"tail specialist" has 8 training examples).

## Architecture & where it lives in the code

- **`src/face_occlusion/models/distribution.py`** — pure functions: `make_bin_centers`,
  `soft_label_distribution` (DLDL/LDS Gaussian soft labels), `expectation`, `dldl_kl_loss`.
- **`models/regressor.py`** — `model.head.type: distribution` builds a `LayerNorm → Linear(d, K)`
  head on the pooled features (backbone as a `num_classes=0` feature extractor, like the MLP
  head), registers `bin_centers` as a buffer, and `forward` returns
  `OcclusionModelOutput(y_pred=Σ p_k c_k, bin_logits=logits, features=…)`. Incompatible with the
  ordinal head (both are bin heads); compatible with full fine-tune and LoRA; `param_groups`
  treats it as the head group (discriminative LRs work).
- **`models/outputs.py`** — `OcclusionModelOutput.bin_logits` carries the `(B, K)` logits.
- **`training/lit_module.py`** — `losses.regression.type: dldl` computes
  `dldl_kl_loss(bin_logits, soft) + expectation_weight · weighted_mse(y_pred, y)` with
  `soft = soft_label_distribution(targets, model.bin_centers, lds_sigma)`. Validation is
  unchanged — `val/score` already uses `y_pred` (the expectation).
- **Config:** `configs/convnext_ablation/08_ordinal_expectation_dldl.yaml`
  (`head: {type: distribution, n_bins: 21, range: [0,1]}`, `losses.regression: {type: dldl,
  lds_sigma: 0.05, expectation_weight: 1.0}`), champion backbone, seed 42.

## What to expect (honest limits)

- This **does not manufacture tail data** — with 8 examples above 0.6, LDS can only *borrow*
  from the 0.2–0.4 region, mildly softening saturation. The real tail fix is synthetic high-occ
  data (a separate, deferred, board-gated project).
- The realistic, **measurable** value is **ensemble diversity**: a classification-trained
  expectation predictor is decorrelated from the MSE regressors, so it should strengthen the
  ensemble (which already moved 0.00129 → 0.00121) even if it merely ties as a single model.
- The bulk effect is measurable on val (paired-Δ); the tail effect is leakage-/scarcity-bound
  and only visible on the leaderboard.
- Risk to gate against: like sigmoid/MLP earlier, pushing tail predictions up can trade away
  bulk accuracy and *worsen* the score — adopt only if it clears the bootstrap paired-Δ gate.
