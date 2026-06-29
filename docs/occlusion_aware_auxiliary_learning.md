# Occlusion-Aware Auxiliary Learning

*The "why" behind the auxiliary heads and losses in this project. For the code-level mechanics
(classes, config flags, gating) see [04 — Models](architecture/04-models.md) and
[05 — Training](architecture/05-training.md); this note is the conceptual story.*

## The idea in one paragraph

The single supervised signal — a continuous occlusion label `y ∈ [0, 1]` — is dominated by the
easy, low-occlusion bulk of the data, so a plain regressor learns the conditional mean and
under-commits on the rare, heavily-occluded faces that the metric weights most. The strategy here
is to **keep the real label as the calibration anchor and add auxiliary objectives that shape the
representation** for occlusion: ordinal structure, regression↔ordinal consistency, a monotonic
ranking signal from synthetic occlusions, and invariances (background, gender). None of them
invents tail data, so each is a *representation* lever, not a *data* fix — and (honestly) most tie
or lose under the confidence-interval gate. The value is diversity: several of these become useful
**ensemble members** even when they don't win as single models ([01](architecture/01-overview.md)).

## What is actually built

Everything below is gated by a config flag and **defaults OFF**, so the baseline path is
unchanged. The "considered" rows were designed but deliberately not implemented — see the
appendix.

| Component | Status | Where |
|---|---|---|
| Weighted-MSE regression (the anchor) | built | `training/lit_module.py` |
| Ordinal threshold head + weighted-BCE | built (opt-in) | `models/ordinal.py` |
| Regression↔ordinal consistency | built (opt-in) | `models/ordinal.py` |
| Ordinal monotonicity hinge | built (opt-in) | `models/ordinal.py` |
| Ordered-bin **expectation head** (DEX + DLDL/LDS) | built (opt-in) | `09 — Imbalanced regression` |
| Synthetic occlusion generation + cache | built | `data/synthetic_occlusion.py`, `data/synthetic_cache.py` |
| Synthetic **monotonic ranking** | built (opt-in) | `models/ranking.py` |
| Label-preserving **background-invariance** | built (opt-in) | `data/background_augment.py` |
| **Shadow** auxiliary head | built (opt-in) | `models/regressor.py` + `losses.shadow` |
| **Gender-adversary** (gradient reversal) | built (opt-in) | `models/adversary.py` |
| Distribution-aware reweighting | built (opt-in) | `metrics/eval_lenses.py` |
| Gender × occlusion balanced sampler | built (opt-in) | `data/samplers.py` |
| Triplet contrastive loss + projection head | considered, not built | — |
| Gender-aware triplet mining | considered, not built | — |
| Progressive encoder unfreezing | considered, not built | — |

The `losses.triplet`, `triplet_sampling`, and `model.use_projection_head` config blocks are inert
placeholders for the "considered" rows.

## Framing: bins and ordinal targets

Although `y` is continuous, the error analysis is naturally **bin-based**, because the metric
cares about the heavily-occluded tail. We use six occlusion bins and the five thresholds between
them:

| Bin | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Interval | `[0,.05)` | `[.05,.10)` | `[.10,.20)` | `[.20,.40)` | `[.40,.60)` | `[.60,1]` |

`T = {0.05, 0.10, 0.20, 0.40, 0.60}`.

Rather than multiclass bin classification, we use **ordinal** targets `c_k = 1[y > t_k]` — e.g.
`y = 0.35 → [1,1,1,0,0]`. This respects the natural ordering: predicting a neighbouring bin should
cost less than predicting a distant one, which plain softmax classification ignores. It is also why
a boundary case like `0.399` vs `0.401` is a soft disagreement, not a hard class flip.

## The implemented auxiliary stack (why each exists)

Each term below is one line of the gated loss sum in [05 — Training](architecture/05-training.md);
that chapter has the exact gating, warmup, and log keys. Here is only the rationale.

- **Weighted-MSE regression** — `w = 1/30 + y`, identical to the metric's per-row error. This is
  the anchor; everything else is auxiliary. A `gender_balanced` variant mirrors the metric's
  `0.5(Err_F+Err_M) + λ·gap` structure directly.
- **Ordinal head** (`L_ord`) — per-threshold weighted BCE on `P(y > t_k)`, up-weighting the rare
  high thresholds. Gives the encoder explicit "is it past this much occlusion?" supervision that a
  single scalar regressor never sees. Config: `configs/experiments/ordinal_head.yaml`.
- **Consistency** (`L_cons`) — ties the regression head's *implied* threshold probabilities
  `σ((ŷ − t_k)/T)` to the ordinal head's, so the two views of the same target agree. `mode` chooses
  which head receives gradient (symmetric / ordinal-as-teacher / regression-as-teacher). Config:
  `configs/experiments/ordinal_consistency.yaml`.
- **Monotonicity** (`L_mono`) — a hinge that keeps the ordinal probabilities non-increasing
  (`P(y>t_1) ≥ … ≥ P(y>t_K)`), enforcing the ordering the threshold labels assume.
- **Expectation head (DEX + DLDL/LDS)** — reframes regression as a softmax over ordered bins,
  trained against Gaussian soft labels so the data-poor tail borrows gradient from its neighbours,
  and predicts `E[y] = Σ p_k c_k`. A decorrelated ensemble member; full treatment in
  [09](architecture/09-imbalanced-regression-and-expectation-head.md).
- **Synthetic monotonic ranking** (`L_rank`) — RankNet over a synthetic triple
  `clean < mild < strong` of the *same* face, enforcing `score(clean) < score(mild) <
  score(strong)`. The synthetic views never get a regression label (their absolute occlusion is
  unknown) — they supply **ordering**, not calibration. Concept and pitfalls in
  [synthetic occlusion generation](synthetic_occlusion_generation.md); config:
  `configs/experiments/synthetic_ranking.yaml`.
- **Background-invariance** (`L_bgc`) — penalizes prediction disagreement between two
  differently-background-randomised views of the same crop (the face pixels untouched), an explicit
  "read the face, ignore the background" signal. Config:
  `configs/experiments/background_invariance.yaml`.
- **Shadow head** (`L_shadow`) — a training-only head predicting the within-face deep-shadow
  fraction, the one image property we found correlated with the label (ρ≈+0.18). Pushes the encoder
  to represent illumination; dropped at inference.

## Gender-gap handling

The metric penalizes the female/male error gap directly, so reducing it is a first-class goal.
What we actually use is **three coordinated, shipped levers**, not contrastive mining:

1. **Balanced sampler** — the encoder sees occluded males as often as occluded females, so it can't
   lean on a gender prior for occlusion ([balanced-batch sampler](balanced_batch_sampler.md)).
2. **Metric-aligned loss** — the `gender_balanced` regression term (optionally with a gap penalty).
3. **Gender-adversary** — a gradient-reversal head (DANN) that removes gender information *not
   explained by occlusion* from the encoder features.

The recipe lives in `configs/experiments/gender_invariant.yaml`. The honest finding: a head-refit
(DFR) baseline and loss-only fixes both failed, and the adversary mostly closes the gap by
"levelling down" rather than lifting the minority gender — the shortcut is entangled in the encoder
and is data-bound. See [04](architecture/04-models.md) / [05](architecture/05-training.md) for the
mechanics.

## Evaluation and what we learned

Evaluation is **confidence-interval-first** ([06](architecture/06-metrics-and-evaluation.md)): the
high-occlusion validation tail is tiny and identity-leaked, so a raw tail delta is noise. Every
auxiliary is gated on a paired-Δ comparison with bootstrap CIs (row- and group-clustered), and for
the ranking lever specifically: **keep it only if real high-occ error improves within CI**, not
just because `train/rank_ordering_acc` rose.

The pedagogical payoff — the things that did *not* work are as informative as the one that did:

- **Ensembling decorrelated, individually-tied models is the only change that beat the baseline
  beyond noise** (`val 0.00129 → 0.00118`, leaderboard `≈ 0.00112`).
- **Post-hoc recalibration was rejected** — it can't add discrimination the model lacks; the
  prediction saturates around ~0.46 and the fix has to be training-side.
- **The gender gap is sticky** — representation-level and in-processing fixes both fail to close it
  without levelling down.
- Ordinal / consistency / ranking / reweighting / background-invariance were **tie-to-mild** as
  single models; their worth is ensemble diversity, not a solo win.

## Appendix — considered but not built

Three ideas were fully designed and intentionally left unimplemented, because the cheaper levers
above already exhausted what the representation could give on this tiny tail:

- **Triplet contrastive loss + projection head** — pull same-occlusion / push different-occlusion
  embeddings via a projection head. Heavier to tune (mining, temperature, head sizing) with no
  evidence it would beat the ordinal+ranking signals already in place.
- **Gender-aware triplet mining** — construct triplets balanced across gender to attack the gap in
  embedding space. Superseded by the simpler sampler + adversary recipe.
- **Progressive encoder unfreezing** — a staged backbone thaw; full fine-tune was already strong
  and cheaper to reason about.

*The full original design exploration (≈1900 lines, including the triplet derivations) remains in
git history if the contrastive direction is ever revisited.*
