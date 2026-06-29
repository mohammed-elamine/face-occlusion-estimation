# 04 — Models

Model code lives in `src/face_occlusion/models/`. The design is a **shared encoder + a
regression head**, with optional auxiliary heads that all consume the same pooled features.
A structured output dataclass keeps the training loop stable as heads are added.

## The output contract — `models/outputs.py`

`OcclusionModelOutput` (dataclass) is what every forward pass returns:

| field | shape | when populated |
|-------|-------|----------------|
| `y_pred` | `(B,)` | always — the continuous occlusion score (after activation, or the bin expectation for the distribution head) |
| `ordinal_logits` | `(B, K)` | only with the ordinal head; per-threshold logits |
| `bin_logits` | `(B, K)` | only with the distribution head; per-bin logits (`y_pred` is then `Σ softmax·centers`) — see [09](09-imbalanced-regression-and-expectation-head.md) |
| `features` | `(B, d)` | when the multi-head path computes pooled features |
| `shadow_pred` | `(B,)` | only with `model.use_shadow_head`; aux within-face shadow-fraction prediction in `[0,1]` (training-only, dropped at inference) — see [05](05-training.md) |
| `projection` | `(B, p)` | reserved for a future contrastive projection head |

Downstream code reads `out.y_pred`; auxiliary fields are `None` on the baseline path.

## The regressor — `models/regressor.py`

`OcclusionRegressor(nn.Module)` wraps a `timm` backbone and a head. Built via
`build_model(cfg, mean_target)`, which reads the `model.*` section.

**Head modes (`model.head.type`):**

- `linear` (default): the backbone is created with `num_classes=1`, i.e. timm's own
  classifier is the head. `forward` is the bit-identical baseline fast path
  (`self.backbone(x)` → activation).
- `mlp`: the backbone is created with `num_classes=0` (pure feature extractor of
  `num_features`), and a separate head is added:
  `LayerNorm(d) → Linear(d, hidden_dim) → GELU → Dropout → Linear(hidden_dim, 1)`.
  `forward` is `feat = backbone(x); raw = head(feat); y = activation(raw)`.
- `distribution`: `num_classes=0` extractor + `LayerNorm(d) → Linear(d, K)` over `K` ordered
  occlusion bins; `forward` returns `y_pred = Σ softmax(logits)·bin_centers` (the expectation,
  bounded so no activation) and `bin_logits`. Trained with the `dldl` loss. Incompatible with
  the ordinal head. The full method is [09](09-imbalanced-regression-and-expectation-head.md).

**Output activation (`model.output_activation`):** `identity` or `sigmoid` (bounded `[0,1]`
regression). `_init_head_bias(mean_target)` warm-starts the final Linear bias toward the
training mean (logit-transformed under sigmoid) so optimization does not waste epochs
learning the global offset.

**ViT resolution:** when `model.img_size` is set, the backbone is created with that size and
`dynamic_img_size=True` so DINOv2/ViT position embeddings are interpolated to the crop
resolution (CNN configs leave it unset).

**LoRA (`model.lora.enabled`, via PEFT):** `_wrap_lora` wraps the backbone with
`LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=...)`. Default
`target_modules=["attn.qkv", "attn.proj"]` (attention only). With a separate MLP head the
head is already trainable and stays outside PEFT (no `modules_to_save`); with the linear
head, `modules_to_save=["head"]` keeps the classifier trainable. The backbone is otherwise
frozen, so only adapters + head train.

**Discriminative param groups:** `param_groups(head_lr, backbone_lr, weight_decay)` returns
AdamW groups — head (and ordinal head) params at `head_lr`, backbone/LoRA at `backbone_lr`,
with **weight decay applied only to ≥2-D weights** (none on LayerNorm/bias). This is what
`configure_optimizers` uses for the discriminative-LR recipe ([05](05-training.md)).

**Forward paths (three):**
1. **MLP** (`self.head is not None`) → features → MLP head → activation.
2. **Fast/linear** (`head is None`, no ordinal head) → `backbone(x)` → activation (baseline).
3. **Multi-head** (`ordinal_head is not None`) → shared pooled features feed the regression
   classifier *and* the ordinal head; returns `ordinal_logits` and `features` too.

**Guards:** `use_ordinal_head=True` is incompatible with `head.type=mlp` or
`lora.enabled=True` (raises `ValueError`), because the ordinal/multi-head path expects the
in-backbone classifier.

**Auxiliary shadow head (`model.use_shadow_head`):** a small
`LayerNorm → Linear(d, hidden) → GELU → Dropout → Linear(hidden, 1) → sigmoid` on the pooled
features, predicting the within-face shadow fraction as a **training-only multi-task** signal
(`shadow_pred`); the occlusion prediction is unchanged. With the linear head it routes through the
same `forward_features` path as the ordinal head; with mlp/distribution heads it reads their pooled
features. Built before LoRA wrapping (stays trainable) and grouped with the head params. Works
alongside any head; the loss + targets are in [05](05-training.md).

**Gender-adversary head (`model.use_gender_adversary`, `models/adversary.py`):** a small MLP that
predicts gender from the pooled features, used with a **gradient-reversal layer** to push the
encoder toward gender-invariant occlusion features — the representation-level fix for the gender
shortcut (a head-refit DFR baseline failed: the gap is entangled in the encoder, not just the head).
Enabling it makes `forward` expose
`features`; the GRL, optional occlusion-bin conditioning, and BCE loss live in the LightningModule
(which has the targets) — [05](05-training.md). Built before LoRA, grouped with the head params,
**training-only** (dropped at inference). `gender_adversary.conditional` one-hots the occ bin into
the adversary (input dim `d + n_occ_bins`).

## Ordinal head & coupled regularizers — `models/ordinal.py`

The ordinal branch reframes regression as a set of monotone threshold classifiers
`P(y > t_k)`.

- `DEFAULT_ORDINAL_THRESHOLDS` (e.g. `(0.05, 0.10, 0.20, 0.40, 0.60)`) and per-threshold BCE
  weights that up-weight rare high-occlusion thresholds.
- `OrdinalHead(in_features, num_thresholds)` — a single linear layer producing one logit per
  threshold (independent sigmoids, no softmax).
- `make_ordinal_targets(y, thresholds)` — builds cumulative targets `c_ik = 1[y_i > t_k]`.
- `threshold_weighted_bce(logits, targets, weights)` — per-threshold weighted
  `BCEWithLogits` (the ordinal loss term).
- `ordinal_monotonicity_loss(logits)` — hinge penalty keeping
  `P(y>t_1) ≥ … ≥ P(y>t_K)`; `ordinal_monotonicity_violation_rate` is the gradient-free
  diagnostic.
- `regression_ordinal_consistency_loss(y_pred, logits, thresholds, temperature, mode)` —
  matches the regression head's implied threshold probs `σ((y_pred − t_k)/T)` to the ordinal
  head's `σ(logit)`; `mode ∈ {symmetric, ordinal_as_teacher, regression_as_teacher}` controls
  which head receives gradient (`CONSISTENCY_MODES`).

These terms are gated by `losses.ordinal/consistency/monotonicity.enabled` and require
`model.use_ordinal_head`; how they are warmed and combined is in [05](05-training.md).

## Ranking utilities — `models/ranking.py`

For the synthetic monotonic-ranking objective (lands on the regression head):

- `ranknet_loss(higher, lower)` — logistic `−log σ(higher − lower)` (via `softplus`).
- `monotonic_ranking_loss(s_clean, s_mild, s_strong)` —
  `ranknet(s_mild, s_clean) + ranknet(s_strong, s_mild)`, enforcing `clean < mild < strong`
  on the model's scores for the synthetic triple.
- `ordering_accuracy(...)` — gradient-free fraction of correctly ordered triples
  (`train/rank_ordering_acc`).

This is the only consumer of the `synthetic_*` views from the batch ([03](03-data.md)); it is
gated by `losses.ranking.enabled` + `synthetic_occlusion.enabled`.
