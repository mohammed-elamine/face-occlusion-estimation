# Occlusion-Aware Metric Learning for Face Occlusion Regression

## 1. Motivation

We consider a face occlusion regression task where each image $x_i$ is associated with a continuous occlusion score:

$$
y_i \in [0,1].
$$

The final goal is to predict a calibrated scalar score:

$$
\hat{y}_i = f(x_i),
$$

where larger values correspond to stronger face occlusion.

A major difficulty in the dataset is the **rarity of high-occlusion examples**. Most images belong to low or medium occlusion regimes, while images with strong occlusion are underrepresented. Consequently, a model trained only with a standard regression objective may obtain a strong global validation score while performing poorly on rare high-occlusion images.

The central idea of this approach is therefore:

> Keep the real regression task as the calibration objective, but enrich training with auxiliary tasks that explicitly teach the model occlusion ordering, occlusion-regime recognition, and occlusion-aware embedding geometry.

The method combines six complementary components:

1. **Weighted real-label regression** for calibrated score prediction.
2. **Ordinal occlusion-bin classification** to make the encoder aware of occlusion regimes.
3. **Regression/classification consistency** to ensure the continuous prediction is coherent with the predicted occlusion regime.
4. **Synthetic monotonic occlusion ranking** to exploit generated examples without assigning unreliable exact labels.
5. **Triplet contrastive learning** to structure the embedding space according to occlusion similarity.
6. **Gender-aware triplet construction** to encourage the encoder to organize representations by occlusion severity rather than demographic shortcuts.

The full method is designed to address rare high-occlusion samples while preserving calibration to the original challenge target. Since the challenge metric also accounts for the error gap between male and female samples, the triplet construction and evaluation protocol explicitly consider the joint structure of **occlusion bin × gender**.

---

## Implementation status

This document describes the full method; not all of it is built. Current state:

| Component | Status | Where |
|---|---|---|
| Weighted regression `L_reg` | implemented | `training/lit_module.py` |
| Ordinal head + loss `L_ord` | implemented (opt-in) | `models/ordinal.py` |
| Regression–ordinal consistency `L_cons` | implemented (opt-in) | `models/ordinal.py` |
| Monotonicity `L_mono` | implemented (opt-in) | `models/ordinal.py` |
| Synthetic generation (Stage 3) | implemented | `data/synthetic_occlusion.py` |
| Synthetic cache + cache-backed views | implemented | `data/synthetic_cache.py`, `scripts/data/build_synthetic_cache.py` |
| Label-preserving background aug | implemented (opt-in) | `data/background_augment.py` |
| Synthetic monotonic ranking `L_rank` | implemented (opt-in) | `models/ranking.py` |
| Bootstrap CIs + dual-split evaluation | implemented | `scripts/analysis/bootstrap_metrics.py` |
| MediaPipe coverage gate (bin×gender) | implemented | `scripts/analysis/generate_synthetic_occlusion_audit.py` |
| Triplet contrastive `L_triplet` + projection head | roadmap (Stage 5, not built) | — |
| Gender-aware triplet mining | roadmap (Stage 5, not built) | — |
| Progressive encoder unfreezing | roadmap (not built) | — |

So the objective `L_reg + λ_ord L_ord + λ_cons L_cons + λ_mono L_mono + λ_rank L_rank`
is implemented (all auxiliaries default-off); the `L_triplet` term, the projection
head, and gender-aware mining (§11–§12) remain roadmap. The `losses.triplet`,
`triplet_sampling`, and `model.use_projection_head` config blocks are inert
placeholders until then.

---

## 2. Notation

Let the real labeled dataset be:

$$
\mathcal{D}_{real} = \{(x_i, y_i, g_i)\}_{i=1}^{N},
$$

where:

- $x_i$ is a face image,
- $y_i \in [0,1]$ is the real occlusion score,
- $g_i \in \{\text{male}, \text{female}\}$ is the demographic group used by the challenge metric.

The gender label is **not** used as an input feature to the model. It is used for sampling, diagnostics, and fairness-aware triplet construction. The goal is not to remove all gender-related visual information from the representation, but to reduce systematic error differences between gender groups.

We define an encoder:

$$
z_i = E_\theta(x_i),
$$

where $z_i \in \mathbb{R}^d$ is a learned visual representation.

The model has several heads:

- a regression/scoring head $h_{reg}$,
- an ordinal head $h_{ord}$,
- a projection head $h_{proj}$ for metric learning.

The outputs are:

$$
\hat{y}_i = h_{reg}(z_i),
$$

$$
o_i = h_{ord}(z_i) \in \mathbb{R}^{K-1},
$$

$$
u_i = h_{proj}(z_i) \in \mathbb{R}^{p}.
$$

Here:

- $\hat{y}_i$ is the predicted continuous occlusion score,
- $o_i$ are ordinal logits,
- $u_i$ is the contrastive embedding.

The regression head is used for the final prediction. The ordinal and projection heads are auxiliary training components.

---

## 3. Model Architecture

The proposed architecture is:

```text
image x
   │
   ▼
pretrained encoder Eθ
   │
   ▼
shared representation z
   │
   ├── regression head h_reg(z) ─────► continuous score ŷ ∈ [0,1]
   │
   ├── ordinal head h_ord(z) ────────► threshold logits o₁, ..., o_{K-1}
   │
   └── projection head h_proj(z) ────► contrastive embedding u
```

A pretrained visual encoder should be used, for example:

- ConvNeXt-Tiny or ConvNeXt-Small pretrained on ImageNet,
- DINOv2 ViT-S/14 or ViT-B/14 as a strong self-supervised alternative.

The encoder should be fine-tuned progressively:

1. freeze encoder and train heads,
2. unfreeze last encoder block,
3. optionally fine-tune the full encoder with a small learning rate.

This is preferable to training from scratch because the dataset is imbalanced and high-occlusion samples are rare.

---

## 4. Occlusion Bins and Ordinal Labels

Although the target is continuous, the error analysis is naturally bin-based. We define occlusion bins:

| Bin | Score interval |
|---:|---|
| 0 | $[0.00, 0.05)$ |
| 1 | $[0.05, 0.10)$ |
| 2 | $[0.10, 0.20)$ |
| 3 | $[0.20, 0.40)$ |
| 4 | $[0.40, 0.60)$ |
| 5 | $[0.60, 1.00]$ |

Let the thresholds be:

$$
\mathcal{T} = \{t_1, t_2, t_3, t_4, t_5\}
= \{0.05, 0.10, 0.20, 0.40, 0.60\}.
$$

Instead of using plain multiclass bin classification, we use **ordinal classification**. For each real label $y_i$, define ordinal targets:

$$
c_{ik} = \mathbb{1}[y_i > t_k], \quad k = 1, \dots, K-1.
$$

Example: if $y_i = 0.35$, then:

```text
thresholds: 0.05  0.10  0.20  0.40  0.60
labels:       1     1     1     0     0
```

This formulation respects the natural ordering of the target. It is better than standard bin classification because predicting a neighboring bin is less severe than predicting a very distant bin.

---

## 4.1 Joint Occlusion-Bin × Gender Groups

Because the challenge metric also penalizes demographic error gaps, we should not reason only about occlusion bins marginally. We should also monitor the joint groups:

$$
c_i = (b_i, g_i),
$$

where:

- $b_i = \operatorname{bin}(y_i)$ is the occlusion bin,
- $g_i$ is the gender group.

For example:

```text
bin 0, male
bin 0, female
bin 1, male
bin 1, female
...
bin 5, male
bin 5, female
```

This matters because a dataset can look balanced by occlusion bin while still being imbalanced inside a specific bin-gender intersection. For instance, high occlusion may be rare overall, but even rarer for one gender group.

The joint group definition is used for:

- anchor sampling,
- positive and negative triplet sampling,
- synthetic generation coverage,
- bin-wise gender-gap evaluation.

The objective is:

> Images with similar occlusion severity should be close in the embedding space even when they belong to different gender groups.

This encourages the encoder to organize the representation around occlusion severity rather than gender-specific shortcuts.

---

## 5. Real-Label Regression Loss

The main supervised objective remains the original continuous regression task defined by the data challenge.

For each real training image $x_i$, we know the true occlusion score:

$$
y_i \in [0,1].
$$

The model predicts a continuous occlusion score:

$$
\hat{y}_i \in [0,1].
$$

Since the official challenge objective gives more importance to high-occlusion examples, we use the challenge-weighted MSE as the primary regression loss:

$$
\mathcal{L}_{reg}
=
\frac{1}{B}
\sum_{i=1}^{B}
w(y_i)(\hat{y}_i - y_i)^2,
$$

with:

$$
w(y_i) = \frac{1}{30} + y_i.
$$

Therefore:

$$
\boxed{
\mathcal{L}_{reg}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(\frac{1}{30} + y_i\right)(\hat{y}_i - y_i)^2
}
$$

This weighting scheme is important because it directly reflects the evaluation objective of the challenge.

For very low occlusion:

$$
y_i \approx 0
\quad \Rightarrow \quad
w(y_i) \approx \frac{1}{30} \approx 0.033.
$$

For very high occlusion:

$$
y_i \approx 1
\quad \Rightarrow \quad
w(y_i) \approx 1.033.
$$

So high-occlusion samples receive a much larger penalty than near-zero occlusion samples. This is desirable because high-occlusion examples are rare and difficult, but they are also more important under the challenge metric.

The constant term $\frac{1}{30}$ ensures that low-occlusion examples still contribute to the loss. Without this term, samples with $y_i = 0$ would have zero weight and would not influence training.

This loss should remain the **dominant calibration loss** of the whole method because the final task is still to predict the correct continuous score. The auxiliary losses introduced later — ordinal classification, regression-ordinal consistency, synthetic ranking, and triplet contrastive learning — are used to improve the representation and robustness, but they should not replace the challenge-weighted regression objective.

In the full method, this loss is applied only to real labeled images:

$$
(x_i, y_i) \sim \mathcal{D}_{real}.
$$

Synthetic occluded images are not assigned exact regression labels by default, because their true challenge score is unknown. Instead, synthetic images are mainly used to provide reliable relative supervision, such as:

$$
x_i
<
x_i^{mild}
<
x_i^{strong},
$$

through ranking and triplet losses.

---

## 6. Ordinal Occlusion-Regime Loss

The ordinal head outputs logits:

$$
o_{ik}, \quad k=1,\dots,K-1.
$$

The corresponding probabilities are:

$$
q_{ik}=\sigma(o_{ik}),
$$

where:

$$
q_{ik} \approx P(y_i > t_k \mid x_i).
$$

The ordinal loss is a weighted binary cross-entropy over thresholds:

$$
\mathcal{L}_{ord}
=
\frac{1}{B(K-1)}
\sum_{i=1}^{B}
\sum_{k=1}^{K-1}
\beta_k
\operatorname{BCEWithLogits}(o_{ik}, c_{ik}).
$$

where:
$$
\operatorname{BCEWithLogits}(o, c) = -[c \log \sigma(o) + (1-c) \log (1-\sigma(o))].
$$

The weights $\beta_k$ can emphasize high thresholds:

| Threshold | Meaning | Suggested weight |
|---:|---|---:|
| 0.05 | $y > 0.05$ | 1.0 |
| 0.10 | $y > 0.10$ | 1.0 |
| 0.20 | $y > 0.20$ | 1.2 |
| 0.40 | $y > 0.40$ | 2.0 |
| 0.60 | $y > 0.60$ | 3.0–4.0 |

This auxiliary task encourages the encoder to learn coarse occlusion regimes, especially rare high-occlusion regimes.

---

## 7. Monotonicity Constraint for Ordinal Predictions

Ordinal probabilities should be monotonic:

$$
P(y > 0.05) \geq P(y > 0.10) \geq P(y > 0.20) \geq P(y > 0.40) \geq P(y > 0.60).
$$

Equivalently:

$$
q_{i1} \geq q_{i2} \geq \dots \geq q_{i,K-1}.
$$

A simple monotonicity penalty is:

$$
\mathcal{L}_{mono}
=
\frac{1}{B(K-2)}
\sum_{i=1}^{B}
\sum_{k=1}^{K-2}
\max(0, q_{i,k+1} - q_{ik}).
$$

This penalizes impossible ordinal outputs such as:

```text
P(y > 0.40) > P(y > 0.20)
```

The monotonicity loss should be small and act as a regularizer.

---

## 8. Regression-Ordinal Consistency

The ordinal task is easier than exact regression. Therefore, it can provide a useful structural constraint for the regression head.

The regression prediction $\hat{y}_i$ implies soft threshold probabilities:

$$
r_{ik} = \sigma\left(\frac{\hat{y}_i - t_k}{\tau}\right),
$$

where $\tau > 0$ is a temperature controlling the softness of the threshold. A good initial value is:

$$
\tau = 0.05.
$$

The ordinal head predicts:

$$
q_{ik}=\sigma(o_{ik}).
$$

We enforce consistency between the two:

$$
\mathcal{L}_{cons}
=
\frac{1}{B(K-1)}
\sum_{i=1}^{B}
\sum_{k=1}^{K-1}
(q_{ik} - r_{ik})^2.
$$

This prevents contradictory outputs such as:

```text
regression score = 0.12
ordinal prediction = high occlusion > 0.60
```

The constraint should be soft, not hard, because samples near bin boundaries are naturally ambiguous. For example, $y=0.399$ and $y=0.401$ should not be treated as fundamentally different.

There are three possible gradient strategies:

1. **Symmetric consistency**: both heads influence each other.
2. **Regression-as-teacher**: detach the regression-implied probabilities $r$.
3. **Ordinal-as-teacher**: detach the ordinal probabilities $q$.

Because the ordinal task is easier, a useful ablation is ordinal-as-teacher:

$$
\mathcal{L}_{cons}
=
\left\|r - \operatorname{stopgrad}(q)\right\|_2^2.
$$

However, the safest initial version is symmetric consistency with a small coefficient.

---

## 9. Synthetic Occlusion Generation

> **Implementation status (Stage 3):** the generator, MediaPipe-only face
> region provider, acceptance-rejection sampler, dataset hooks, and visual
> audit script are implemented. There is no runtime geometric fallback; if
> MediaPipe fails or mask sanity checks fail, the pair is marked
> `synthetic_valid=false`. See
> [docs/synthetic_occlusion_generation.md](synthetic_occlusion_generation.md)
> for the API, config schema, and audit procedure. Default configs ship
> with `synthetic_occlusion.enabled=false` so training behaviour is
> unchanged until Stage 4 opts in.

The dataset contains few real high-occlusion images. To create additional supervision, we generate synthetic occlusions.

The key principle is:

> Synthetic samples should be used to teach relative occlusion ordering, not exact regression labels.

For a real image $x$, generate two views:

$$
x^{mild}, \quad x^{strong},
$$

such that:

$$
x \prec x^{mild} \prec x^{strong},
$$

where $a \prec b$ means that $b$ should be considered more occluded than $a$.

We do **not** need to know the exact synthetic scores. We only need the ordering to be reliable.

### 9.1 Face-Aware Region Localization

Synthetic occlusions should be placed on the face, not randomly in the image background.

The current runtime implementation uses MediaPipe Face Mesh to estimate:

- face oval,
- left eye region,
- right eye region,
- mouth region,
- nose region,
- lower face region.

From landmarks, construct binary masks or polygons for each region.
If detection fails or the derived masks fail simple sanity checks, synthetic
generation for that image is invalid. We do not replace such failures with
geometric masks.

A more advanced implementation can use face parsing / semantic segmentation models to obtain masks for facial components. However, a landmark-based solution is usually simpler, faster, and sufficient for controlled occluder placement.

### 9.2 Region Weights

Different facial regions may have different importance for perceived occlusion.

Example region weights:

| Region | Weight | Rationale |
|---|---:|---|
| Eyes | 1.00 | Very important for face visibility |
| Mouth | 0.85 | Important, especially for lower-face occlusion |
| Nose | 0.75 | Central face structure |
| Cheeks | 0.45 | Visible face area but less semantically critical |
| Forehead / chin | 0.35 | Usually less critical |
| Hair / background | 0.00 | Should not count as face occlusion |

For an occluder mask $M$, define a synthetic severity proxy:

$$
\rho(M)
=
\frac{\sum_{r \in \mathcal{R}} w_r \cdot A(M \cap R_r)}{A(R_{face})},
$$

where:

- $\mathcal{R}$ is the set of facial regions,
- $w_r$ is the weight of region $r$,
- $R_r$ is the mask of region $r$,
- $A(\cdot)$ denotes area,
- $R_{face}$ is the full face region.

The value $\rho(M)$ is not treated as a true label. It is only used to control augmentation strength.

### 9.3 Synthetic Severity Levels

Define severity intervals:

| Synthetic level | Target weighted occluded ratio |
|---|---:|
| Mild | $0.05$–$0.15$ |
| Medium | $0.15$–$0.35$ |
| Strong | $0.35$–$0.60+$ |

To generate a valid synthetic view:

1. sample an occluder type,
2. sample position, size, opacity, and rotation,
3. paste it onto the image,
4. compute weighted face overlap $\rho$,
5. accept only if $\rho$ lies in the desired severity interval,
6. otherwise resample.

This acceptance-rejection step is important because it guarantees reliable ordering between mild and strong views.

### 9.4 Occluder Types

Use diverse occluders to avoid overfitting to artificial artifacts.

Examples:

- sunglasses-like patches,
- mask-like lower-face patches,
- scarf-like occlusions,
- hand-like patches,
- phone or microphone-like objects,
- hair-like strands,
- random textured polygons,
- random erasing with natural textures.

Avoid relying only on black rectangles, because the model may learn the artifact rather than the concept of face visibility.

> We use pretrained models only for face-region localization. Synthetic occlusions are then generated by our own controlled augmentation module. This module is inspired by synthetic occlusion augmentation methods from the literature, but adapted to our task through face-aware placement, semantic region weighting, and severity-controlled acceptance-rejection sampling.

---

## 10. Synthetic Monotonic Ranking Loss

Given a real image $x$ and two synthetic views:

$$
x^{mild}, \quad x^{strong},
$$

we know the monotonic ordering:

$$
s(x) < s(x^{mild}) < s(x^{strong}),
$$

where $s(\cdot)$ is the scalar score produced by the regression/scoring head.

Use a RankNet-style logistic ranking loss:

$$
\mathcal{L}_{rank}
=
-
\log \sigma(s(x^{mild}) - s(x))
-
\log \sigma(s(x^{strong}) - s(x^{mild})).
$$

This enforces that synthetic occlusion increases the predicted occlusion score.

A margin-ranking alternative is:

$$
\mathcal{L}_{rank}
=
\max(0, m_r - [s(x^{mild}) - s(x)])
+
\max(0, m_r - [s(x^{strong}) - s(x^{mild})]).
$$

The logistic version is generally smoother.

Synthetic images should not receive regression loss unless a very reliable pseudo-labeling strategy is developed. In the first robust implementation:

```text
real images      → regression + ordinal losses
synthetic images → ranking + triplet losses
```

---

## 11. Triplet Contrastive Learning

Triplet learning structures the embedding space according to occlusion similarity.

The key definition is:

> Positive means close to the anchor in occlusion-score space. Negative means farther from the anchor in occlusion-score space.

It does **not** mean:

```text
positive = more occluded
negative = less occluded
```

For an anchor image $(x_a, y_a)$, choose:

$$
x_p \quad \text{such that} \quad |y_p - y_a| \leq \epsilon_p,
$$

and:

$$
x_n \quad \text{such that} \quad |y_n - y_a| \geq \epsilon_n,
$$

with:

$$
\epsilon_p < \epsilon_n.
$$

The projection head gives:

$$
u_a = h_{proj}(E(x_a)),
$$

$$
u_p = h_{proj}(E(x_p)),
$$

$$
u_n = h_{proj}(E(x_n)).
$$

The triplet loss is:

$$
\mathcal{L}_{triplet}
=
\max(0, d(u_a,u_p) - d(u_a,u_n) + m_t),
$$

where $d$ is a distance function, for example cosine distance or Euclidean distance.

The projection head is separate from the regression head. This is useful because the contrastive embedding can learn metric geometry without over-constraining the scalar prediction head.

### 11.1 Why Use a Projection Head Instead of Applying Triplet Loss Directly on the Score?

The triplet loss is applied to a separate projection embedding:

$$
u = h_{proj}(E(x)),
$$

rather than directly to the scalar prediction:

$$
\hat{y} = h_{reg}(E(x)).
$$

This design separates two related but different objectives:

| Component | Objective |
|---|---|
| Regression head $h_{reg}$ | Predict the calibrated continuous occlusion score |
| Projection head $h_{proj}$ | Learn an embedding space where occlusion-similar images are close |

The regression head is responsible for the final task:

$$
\hat{y} \approx y.
$$

Therefore, it must remain well-calibrated with respect to the original challenge target and the challenge-weighted MSE loss.

The projection head, on the other hand, is used only to impose a metric-learning structure on the representation. It encourages images with similar occlusion levels to have nearby embeddings, and images with different occlusion levels to be farther apart.

For a triplet $(x_a, x_p, x_n)$, this means:

$$
d(u_a, u_p) < d(u_a, u_n),
$$

This indirectly improves regression because the encoder $E$ receives gradients from the triplet loss. As a result, the shared representation $z = E(x)$ becomes more organized around occlusion severity, making the downstream regression task easier.

In contrast, applying a triplet loss directly on scalar predictions would give:

$$
\max(0, |\hat{y}_a - \hat{y}_p| - |\hat{y}_a - \hat{y}_n| + m_t).
$$

This is possible, but it is less flexible. Since $\hat{y}$ is one-dimensional, the loss mostly acts as a score-level consistency or ranking constraint. It does not create a rich embedding geometry. It can also interfere with calibration if the triplets are noisy or if the synthetic severity is only approximate.

For this reason, we use the following separation:

```text
Regression head:
    challenge-weighted MSE
    ordinal consistency
    synthetic ranking loss

Projection head:
    triplet contrastive loss
```

The score-level losses teach the model how much occlusion to predict.
The projection-level triplet loss teaches the encoder **which images have similar occlusion severity**.

This separation makes the approach more robust because the regression head remains focused on calibrated prediction, while the projection head absorbs the metric-learning constraints.

> In this method, the triplet loss is not meant to directly replace the regression loss. Its role is to regularize the encoder by shaping the representation space according to occlusion similarity. The final scalar prediction is still learned through the regression head and calibrated using real labels.

---

## 12. Triplet Construction Strategies

Triplet learning requires constructing triplets:

$$
(x_a, x_p, x_n),
$$

where:

- $x_a$ is the anchor,
- $x_p$ is the positive sample, close to the anchor in occlusion-score space,
- $x_n$ is the negative sample, farther from the anchor in occlusion-score space.

The general constraint is:

$$
|y_a - y_p| < |y_a - y_n|.
$$

However, because the challenge also accounts for the demographic gap between male and female samples, triplets should also be constructed to make the embedding robust across gender groups.

The key principle becomes:

> Triplets should group images by occlusion severity, not by gender.

This means that, when possible, positives should include cross-gender samples with similar occlusion scores. This explicitly teaches the encoder that two images with similar occlusion severity should be close even if they come from different gender groups.

### 12.1 Probabilistic Triplet-Type Sampling

For each anchor image $(x_a, y_a, g_a)$, we first choose the triplet construction type:

$$
T \sim \operatorname{Categorical}(\pi_{RRR}, \pi_{SYN}, \pi_{HYB}, \pi_{FAIR}),
$$

where:

- $T = RRR$ means real-real-real triplet,
- $T = SYN$ means synthetic ordered triplet,
- $T = HYB$ means hybrid real/synthetic triplet,
- $T = FAIR$ means gender-aware real triplet.

The probabilities satisfy:

$$
\pi_{RRR} + \pi_{SYN} + \pi_{HYB} + \pi_{FAIR} = 1.
$$

A reasonable initial configuration is:

| Triplet type | Probability | Role |
|---|---:|---|
| Real-real-real | $\pi_{RRR}=0.40$ | Uses clean real labels |
| Synthetic ordered | $\pi_{SYN}=0.25$ | Creates controlled occlusion variation |
| Hybrid | $\pi_{HYB}=0.20$ | Compensates for rare high-occlusion samples |
| Gender-aware real | $\pi_{FAIR}=0.15$ | Promotes cross-gender occlusion consistency |

Example configuration:

```yaml
triplet_sampling:
  pi_real_real_real: 0.40
  pi_synthetic_ordered: 0.25
  pi_hybrid: 0.20
  pi_gender_aware: 0.15
```

A more conservative first implementation can use:

```yaml
triplet_sampling:
  pi_real_real_real: 0.60
  pi_synthetic_ordered: 0.20
  pi_hybrid: 0.10
  pi_gender_aware: 0.10
```

This keeps most triplets based on real labels while still introducing an explicit fairness-aware signal.

The triplet-type probabilities are hyperparameters. They should be tuned through ablation studies, especially by monitoring both high-occlusion error and the gender gap.

### 12.2 Anchor Sampling with Occlusion Bin × Gender Balancing

The anchor $x_a$ is sampled from the real labeled dataset:

$$
(x_a, y_a, g_a) \sim \mathcal{D}_{real}.
$$

Since both high-occlusion samples and some bin-gender intersections may be rare, anchors should not be sampled purely uniformly.

Let:

$$
p_{nat}(i)
$$

be the natural dataset sampling probability, and let:

$$
p_{bal}^{bin\times gender}(i)
$$

be a balanced sampling probability over joint groups:

$$
c_i = (\operatorname{bin}(y_i), g_i).
$$

A soft anchor sampling distribution can be defined as:

$$
p_{anchor}(i)
=
(1-\alpha)p_{nat}(i)
+
\alpha p_{bal}^{bin\times gender}(i),
$$

where:

$$
\alpha \in [0,1].
$$

For example:

```yaml
anchor_sampling:
  balance_by:
    - occlusion_bin
    - gender
  alpha_balanced: 0.30
```

This increases exposure to rare high-occlusion and underrepresented bin-gender intersections without completely distorting the natural data distribution.

If a joint group contains too few examples, we can smooth the sampler by mixing three distributions:

$$
p_{anchor}(i)
=
\alpha_0 p_{nat}(i)
+
\alpha_1 p_{bal}^{bin}(i)
+
\alpha_2 p_{bal}^{bin\times gender}(i),
$$

with:

$$
\alpha_0 + \alpha_1 + \alpha_2 = 1.
$$

Example:

```yaml
anchor_sampling:
  alpha_natural: 0.50
  alpha_bin_balanced: 0.30
  alpha_bin_gender_balanced: 0.20
```

This is safer when some bin-gender cells are extremely small.

### 12.3 Real-Real-Real Triplets

All images come from the real labeled dataset.

For anchor $(x_a,y_a,g_a)$:

- choose positive $x_p$ with $|y_p-y_a| \leq \epsilon_p$,
- choose negative $x_n$ with $|y_n-y_a| \geq \epsilon_n$.

Example:

```text
positive distance: |y_p - y_a| <= 0.05
negative distance: |y_n - y_a| >= 0.15
```

This is the cleanest source of metric supervision because all labels are real.

More formally, define the positive candidate set:

$$
\mathcal{P}_{all}(a)
=
\{i \neq a : |y_i-y_a| \leq \epsilon_p\},
$$

and the negative candidate set:

$$
\mathcal{N}_{all}(a)
=
\{j : |y_j-y_a| \geq \epsilon_n\}.
$$

Then sample:

$$
x_p \sim \mathcal{P}_{all}(a),
$$

$$
x_n \sim \mathcal{N}_{all}(a).
$$

If either candidate set is empty, the thresholds can be relaxed or the anchor can be resampled.

### 12.4 Gender-Aware Real Triplets

Gender-aware triplets are real-real-real triplets with additional constraints on gender composition.

For an anchor $(x_a, y_a, g_a)$, define cross-gender positive candidates:

$$
\mathcal{P}_{cross}(a)
=
\{i \neq a : |y_i-y_a| \leq \epsilon_p,\ g_i \neq g_a\}.
$$

These are images that have similar occlusion scores but belong to the opposite gender group.

With probability $q_{cross}^{pos}$, choose the positive from the cross-gender set:

$$
x_p \sim \mathcal{P}_{cross}(a).
$$

Otherwise, choose it from the general positive set:

$$
x_p \sim \mathcal{P}_{all}(a).
$$

A default value is:

```yaml
fairness_aware_triplets:
  q_cross_gender_positive: 0.50
  fallback_to_any_gender: true
```

This means that, when possible, half of the positives explicitly come from the opposite gender group while having similar occlusion severity.

The negative still satisfies:

$$
|y_n-y_a| \geq \epsilon_n.
$$

Negatives can also be gender-balanced. Define:

$$
\mathcal{N}_{cross}(a)
=
\{j : |y_j-y_a| \geq \epsilon_n,\ g_j \neq g_a\},
$$

and:

$$
\mathcal{N}_{same}(a)
=
\{j : |y_j-y_a| \geq \epsilon_n,\ g_j = g_a\}.
$$

With probability $q_{cross}^{neg}$, sample a cross-gender negative; otherwise sample a same-gender or arbitrary negative.

Example:

```yaml
fairness_aware_triplets:
  q_cross_gender_positive: 0.50
  q_cross_gender_negative: 0.50
  fallback_to_any_gender: true
```

The most important signal is the cross-gender positive:

$$
|y_p-y_a| \leq \epsilon_p
\quad \text{and} \quad
 g_p \neq g_a.
$$

This enforces:

$$
d(u_a,u_p) < d(u_a,u_n),
$$

even when the anchor and positive belong to different gender groups.

Conceptually, this teaches:

> Similar occlusion should be close in the embedding space, regardless of gender.

This does not require adding a new fairness loss. It changes the triplet sampling distribution so that the existing triplet loss promotes gender-robust occlusion representations.

### 12.5 Synthetic Ordered Triplets

Synthetic ordered triplets are generated from the same anchor image.

```text
anchor   = original image
positive = mildly occluded version
negative = strongly occluded version
```

So:

$$
x_a = x,
$$

$$
x_p = x^{mild},
$$

$$
x_n = x^{strong}.
$$

The intended ordering is:

$$
x \prec x^{mild} \prec x^{strong},
$$

where $a \prec b$ means that $b$ should be more occluded than $a$.

The triplet constraint becomes:

$$
d(u_{original}, u_{mild}) < d(u_{original}, u_{strong}).
$$

This is especially safe because identity, pose, background, lighting, and gender are fixed. Only occlusion changes.

Synthetic ordered triplets help occlusion robustness, but they do not directly enforce cross-gender invariance. Therefore, synthetic generation should be balanced across gender and occlusion bins to avoid creating many more synthetic examples for one group than the other.

Example:

```yaml
synthetic_generation:
  balance_generation_by:
    - occlusion_bin
    - gender
```

The synthetic severity levels should be generated through controlled face-aware occlusion:

```text
mild   → small weighted face overlap
strong → larger weighted face overlap
```

For example:

```text
mild severity:   0.05 <= ρ(M) <= 0.15
strong severity: 0.35 <= ρ(M) <= 0.60
```

Synthetic ordered triplets are useful because they provide reliable relative supervision without requiring exact synthetic regression labels.

### 12.6 Hybrid Triplets

Hybrid triplets mix real and synthetic samples.

For example:

```text
anchor   = real low-occlusion image
positive = real image with close occlusion score
negative = strong synthetic occlusion from the anchor
```

This is useful because real high-occlusion negatives are rare.

One possible hybrid construction is:

$$
x_a = x_i,
$$

$$
x_p \sim \mathcal{P}_{all}(a),
$$

$$
x_n = x_i^{strong}.
$$

A gender-aware hybrid variant uses a cross-gender real positive:

$$
x_p \sim \mathcal{P}_{cross}(a),
$$

and a strong same-anchor synthetic negative:

$$
x_n = x_i^{strong}.
$$

This gives a triplet such as:

```text
anchor   = male, y = 0.20
positive = female, y = 0.22
negative = strong synthetic occlusion from the anchor
```

This combines two signals:

- cross-gender positives promote gender-robust occlusion similarity,
- synthetic negatives provide strong occlusion contrast when real high-occlusion examples are rare.

Hybrid triplets should be used carefully because synthetic samples have approximate severity, not exact labels. They are most reliable when the synthetic sample is generated from the same anchor image.

A useful rule is:

```text
Prefer same-anchor synthetic samples for hybrid triplets.
```

This reduces the risk that the triplet loss learns identity, pose, lighting, background, or gender differences instead of occlusion severity.

### 12.7 Same-Anchor vs Other-Image Synthetic Generation

When a synthetic sample is needed, we can choose whether to generate it from the anchor image or from another real image.

Let:

$$
q_{same}
$$

be the probability of generating the synthetic sample from the anchor itself.

Then:

```text
with probability q_same:
    generate synthetic sample from the anchor image

with probability 1 - q_same:
    generate synthetic sample from another real image with known score
```

A robust default is:

```yaml
synthetic_generation:
  q_same_anchor: 0.80
```

This means most synthetic triplets preserve identity, pose, background, lighting, and gender.

Generating synthetic samples from other images can increase diversity, but it also increases the risk of introducing unrelated visual differences. Therefore, it should be used less frequently, especially at the beginning of training.

If other-image synthetic generation is used, the source image should be sampled with bin-gender balancing to avoid introducing demographic imbalance into synthetic supervision.

### 12.8 Hardness Curriculum

Triplets can be easy, semi-hard, or hard.

Let:

$$
\Delta_p = |y_p-y_a|,
$$

and:

$$
\Delta_n = |y_n-y_a|.
$$

A valid triplet satisfies:

$$
\Delta_p < \Delta_n.
$$

#### Easy triplet

```text
anchor y = 0.10
positive y = 0.12
negative y = 0.80
```

Here, the negative is very far from the anchor. The triplet is easy and provides a clean signal.

#### Semi-hard triplet

```text
anchor y = 0.30
positive y = 0.33
negative y = 0.45
```

The negative is farther than the positive, but not extremely far. Semi-hard triplets are often the most useful once the model starts learning.

#### Hard triplet

```text
anchor y = 0.30
positive y = 0.34
negative y = 0.38
```

This triplet is difficult because positive and negative are close in occlusion-score space.

Very hard triplets may be noisy, especially if labels are imperfect. A robust curriculum is:

1. start with easy and medium triplets,
2. move to semi-hard triplets,
3. introduce hard triplets only after the embedding becomes meaningful.

A possible curriculum is:

| Training stage | Positive distance | Negative distance | Purpose |
|---|---:|---:|---|
| Early | $\leq 0.05$ | $\geq 0.30$ | Stable learning |
| Middle | $\leq 0.05$ | $0.15$–$0.30$ | Semi-hard structure |
| Late | $\leq 0.05$–$0.10$ | $0.10$–$0.20$ | Harder discrimination |

The hardness schedule can also be applied to the triplet-type probabilities. For example, early training can use more real-real-real triplets, while later training can increase synthetic, hybrid, and gender-aware triplets:

```yaml
triplet_schedule:
  early:
    pi_real_real_real: 0.60
    pi_synthetic_ordered: 0.20
    pi_hybrid: 0.10
    pi_gender_aware: 0.10

  middle:
    pi_real_real_real: 0.40
    pi_synthetic_ordered: 0.25
    pi_hybrid: 0.20
    pi_gender_aware: 0.15

  late:
    pi_real_real_real: 0.35
    pi_synthetic_ordered: 0.25
    pi_hybrid: 0.20
    pi_gender_aware: 0.20
```

This curriculum keeps early training stable and gradually introduces more difficult synthetic and fairness-aware supervision.

### 12.9 Limits of Gender-Aware Triplets

Gender-aware triplet construction can reduce the risk that embeddings organize themselves primarily by gender. However, it has limits:

1. **It does not guarantee fairness by itself.** The final objective is still regression, so fairness must be evaluated through gender-specific error metrics.
2. **It depends on available candidates.** Some occlusion bins may contain very few samples from one gender, making cross-gender positives hard to find.
3. **It should not force complete gender blindness.** The challenge asks for similar error across groups, not necessarily for removing all gender-predictive information from the representation.
4. **It can over-constrain the embedding if too strong.** If $q_{cross}^{pos}$ or $\pi_{FAIR}$ is too high, the model may over-prioritize cross-gender alignment and lose useful visual detail for regression.

Therefore, gender-aware sampling should be introduced as a moderate regularization strategy and validated through ablations.

---

## 13. Soft Occlusion- and Gender-Balanced Sampling

The project already uses a soft occlusion-balanced sampler. This component should remain part of the method, but it can be extended to account for gender because the challenge metric penalizes demographic error gaps.

The goal is to increase exposure to rare high-occlusion examples and underrepresented bin-gender intersections without completely distorting the natural distribution.

Let $p_{natural}(i)$ be the natural sampling probability, $p_{bin}(i)$ be the probability under occlusion-bin-balanced sampling, and $p_{bin\times gender}(i)$ be the probability under joint occlusion-bin × gender balancing.

A robust soft sampling distribution is:

$$
p(i)
=
\alpha_0 p_{natural}(i)
+
\alpha_1 p_{bin}(i)
+
\alpha_2 p_{bin\times gender}(i),
$$

with:

$$
\alpha_0 + \alpha_1 + \alpha_2 = 1.
$$

A conservative default is:

```yaml
sampler:
  alpha_natural: 0.50
  alpha_occlusion_bin_balanced: 0.30
  alpha_bin_gender_balanced: 0.20
```

This keeps the training distribution close to the natural data while increasing the probability of seeing rare high-occlusion examples and underrepresented demographic intersections.

The same idea can be used in three places:

1. **Main supervised batches** for regression and ordinal losses.
2. **Anchor sampling** for triplet construction.
3. **Synthetic generation source selection** to avoid generating many more synthetic samples from one gender group than the other.

This does not aim to make the model gender-blind. It aims to give the model enough balanced evidence so that errors are not systematically higher for one gender group.

---

## 14. Full Training Objective

The full objective is:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{reg}^{real}
+
\lambda_{ord}\mathcal{L}_{ord}^{real}
+
\lambda_{cons}\mathcal{L}_{cons}^{real}
+
\lambda_{mono}\mathcal{L}_{mono}^{real}
+
\lambda_{rank}\mathcal{L}_{rank}^{synthetic}
+
\lambda_{triplet}\mathcal{L}_{triplet}^{real/synthetic}.
$$

Where:

| Loss | Data source | Purpose |
|---|---|---|
| $\mathcal{L}_{reg}$ | Real labeled images | Calibrated continuous prediction |
| $\mathcal{L}_{ord}$ | Real labeled images | Occlusion-regime awareness |
| $\mathcal{L}_{cons}$ | Real labeled images | Align regression and ordinal predictions |
| $\mathcal{L}_{mono}$ | Real labeled images | Ensure valid ordinal probabilities |
| $\mathcal{L}_{rank}$ | Synthetic ordered views | Enforce monotonic occlusion ordering |
| $\mathcal{L}_{triplet}$ | Real and synthetic triplets | Structure embedding space by occlusion similarity |

Gender-awareness is introduced primarily through the sampling distributions used for batches, anchors, positives, negatives, and synthetic sources. Therefore, it does not necessarily require adding a separate fairness loss. The same triplet loss is applied, but to triplets sampled in a way that emphasizes occlusion similarity across gender groups.

A conservative initial weighting is:

```yaml
loss_weights:
  regression: 1.0
  ordinal: 0.2
  consistency: 0.05
  monotonicity: 0.01
  ranking: 0.1
  triplet: 0.05
```

The auxiliary losses should not dominate the regression loss. They should regularize and enrich the representation while preserving calibration to the original target.


### 14.1 Inference-Time Behavior

All components are used during training, but not all of them are needed during inference.

At inference time, the model only needs to output the final continuous occlusion score:

$$
\hat{y} = h_{reg}(E(x)).
$$

Therefore, the default inference path is:

```text
image → encoder → regression head → occlusion score
```

The ordinal head and projection head are mainly training-time components:

* the ordinal head helps the encoder learn coarse occlusion regimes,
* the projection head supports triplet contrastive learning,
* the synthetic occlusion generator is used only to create training constraints.

During validation, the ordinal outputs can still be logged for diagnostics, such as bin accuracy or high-occlusion recall. However, the final prediction submitted to the challenge should come from the regression head.

In short:

> Train with all components; predict with the regression head.

---

## 15. Training Schedule

A staged training schedule is safer than activating all components aggressively from the beginning.

### Stage 1: Regression Baseline

Train:

```text
pretrained encoder + regression head
```

Use:

- weighted challenge loss,
- soft occlusion-balanced sampler,
- bin-wise metrics.

Goal: establish a calibrated baseline.

### Stage 2: Add Ordinal Head

Train:

```text
regression loss + ordinal loss + consistency loss + monotonicity loss
```

Goal: improve occlusion-regime awareness.

### Stage 3: Add Synthetic Ranking

Generate:

```text
original < mild synthetic < strong synthetic
```

Train with:

```text
regression/ordinal losses on real images
ranking loss on synthetic views
```

Goal: teach monotonic occlusion severity using reliable synthetic ordering.

### Stage 4: Add Triplet Contrastive Learning

Train with:

```text
real-real-real triplets
synthetic ordered triplets
hybrid triplets
gender-aware real triplets
```

Goal: organize embeddings according to occlusion-level similarity while encouraging similar-occlusion samples to remain close across gender groups.

### Stage 5: Fine-Tuning and Ablation

Tune:

- auxiliary loss weights,
- sampler strength,
- synthetic occlusion severity intervals,
- triplet hardness,
- encoder unfreezing strategy.

---

## 16. Evaluation Protocol

Because high-occlusion examples are rare, global validation metrics are insufficient.

Report at least:

### 16.1 Natural Validation Metrics

These measure performance under the dataset distribution:

- global MSE / MAE,
- challenge score,
- prediction range checks,
- calibration plots.

### 16.2 Bin-Wise Regression Metrics

Report error by occlusion bin:

```text
val/bin_0.00_0.05_err
val/bin_0.05_0.10_err
val/bin_0.10_0.20_err
val/bin_0.20_0.40_err
val/bin_0.40_0.60_err
val/bin_0.60_1.00_err
```

Also report the number of samples per bin. A high-bin metric based on only two samples is unstable and should not be overinterpreted.

### 16.3 High-Occlusion Stress Evaluation

Create a separate high-occlusion stress validation set containing more examples from:

```text
0.40–0.60
0.60–1.00
```

This set does not represent the natural distribution. It evaluates robustness under rare but important cases.

### 16.4 Gender and Bin × Gender Metrics

Because the challenge metric accounts for the demographic gap between male and female samples, evaluation should report errors by gender and by joint occlusion-bin × gender groups.

Report:

```text
val/err_male
val/err_female
val/gender_gap
val/bin_gender_err
val/high_occlusion_err_male
val/high_occlusion_err_female
```

The most important diagnostic is the intersection table:

| Occlusion bin | Male error | Female error | Gap | Count male | Count female |
|---|---:|---:|---:|---:|---:|
| 0.00–0.05 | ... | ... | ... | ... | ... |
| 0.05–0.10 | ... | ... | ... | ... | ... |
| 0.10–0.20 | ... | ... | ... | ... | ... |
| 0.20–0.40 | ... | ... | ... | ... | ... |
| 0.40–0.60 | ... | ... | ... | ... | ... |
| 0.60–1.00 | ... | ... | ... | ... | ... |

This reveals whether the demographic gap is global or concentrated in specific occlusion regimes.

### 16.5 Ordinal Metrics

For the ordinal head:

- balanced accuracy,
- macro F1,
- high-bin recall,
- high-bin precision,
- mean absolute bin distance.

Mean absolute bin distance is useful because it respects ordering:

```text
true bin = 5
predicted bin = 3
bin distance = 2
```

#### 16.5.1 Ordinal-head diagnostics logged at validation

`FaceOcclusionLitModule._log_ordinal_val_metrics` computes all ordinal stats
over the full concatenated validation epoch (not per-batch averages), so rare
high-threshold positives are represented faithfully. When the head is
disabled no `val/ord*` key is logged. Empty subgroups always emit `_count`
and skip the rest (instead of NaN). Keys grouped by family:

- Global per-threshold (one entry per threshold `t ∈ {0.05, 0.10, 0.20, 0.40, 0.60}`):
  - `val/ord_t_{t}_acc`, `val/ord_t_{t}_precision`, `val/ord_t_{t}_recall`, `val/ord_t_{t}_f1`
  - `val/ord_t_{t}_support_pos`, `val/ord_t_{t}_support_neg`
  - Legacy keys preserved: `val/ord_threshold_recall_{t}` and, for high
    thresholds, `val/ord_high_threshold_recall_{0.40,0.60}`.
- Global means across thresholds:
  - `val/ord_loss`, `val/ord_threshold_acc_mean`,
    `val/ord_threshold_precision_mean`, `val/ord_threshold_recall_mean`,
    `val/ord_threshold_f1_mean`.
- Per occlusion bin (uses `cfg.split.occlusion_bins`):
  `val/ord/bin_{lo}_{hi}_{count,threshold_acc_mean,threshold_f1_mean}`.
- High-occlusion aggregate (`target ≥ 0.40`, pooling `[0.40, 0.60)` and `[0.60, 1.00]`):
  `val/ord/high_occ_0.40_1.00_{count,threshold_acc_mean,threshold_f1_mean,recall_t_0.40,recall_t_0.60}`.
- Per gender:
  `val/ord/{female,male}_{count,threshold_acc_mean,threshold_f1_mean,recall_t_0.40,recall_t_0.60}`.
- Per database (sorted unique):
  `val/ord/database/{db}_{count,threshold_acc_mean,threshold_f1_mean}`.
- Consistency (when enabled): `val/cons_loss`, `val/cons_gap_mean`, plus
  per-threshold `val/cons_gap_t_{t}` to expose where regression and ordinal
  predictions disagree.

### 16.6 Ranking and Triplet Diagnostics

For synthetic views, measure:

- percentage of correctly ordered pairs,
- percentage satisfying original < mild < strong,
- triplet violation rate,
- embedding visualization colored by true occlusion score.

---

## 17. Ablation Plan

To understand what actually helps, compare the following models:

| Experiment | Components |
|---|---|
| A | Baseline regression only |
| B | Regression + soft occlusion-balanced sampler |
| C | B + weighted challenge loss |
| D | C + ordinal head |
| E | D + regression-ordinal consistency |
| F | E + synthetic monotonic ranking |
| G | F + triplet contrastive learning |
| H | G + gender-aware triplet sampling |
| I | H + hard/semi-hard triplet mining |

The key question is not only whether global validation loss improves. The main questions are:

> Does the method reduce medium/high-occlusion error without degrading low-occlusion calibration?

And:

> Does the method reduce the male/female error gap, especially inside medium- and high-occlusion bins?

### 17.1 Auxiliary loss warmup

The ordinal and consistency losses are auxiliaries: they should regularise
the encoder, not dominate early training while the regression head is still
finding its calibration. To make this explicit and tunable, every auxiliary
coefficient in `losses.{ordinal,consistency}` accepts two optional fields:

```yaml
losses:
  ordinal:
    weight: 0.10              # target lambda
    warmup_epochs: 3          # 0 => static (current default)
    warmup_start_weight: 0.0
```

Formula (implemented by `_scheduled_loss_weight` in
[src/face_occlusion/training/lit_module.py](../src/face_occlusion/training/lit_module.py)):

$$
\lambda(e) = \begin{cases}
\text{target} & \text{if } E_{\mathrm{warmup}} = 0 \\
\text{start} + \min\!\big(1, \tfrac{e+1}{E_{\mathrm{warmup}}}\big)\,
(\text{target} - \text{start}) & \text{otherwise}
\end{cases}
$$

Using $e+1$ ensures epoch 0 already receives a non-zero coefficient. With
`weight=0.10, warmup_epochs=3, warmup_start_weight=0.0` the schedule is
`{0.0333, 0.0667, 0.10, 0.10, ...}`.

**Defaults.** All existing configs keep `warmup_epochs: 0`, which is exactly
the previous static behaviour. The training step logs `train/lambda_ord` and
`train/lambda_cons` whenever the corresponding loss is active, so warmup is
easy to verify from `metrics.csv`. Validation diagnostics (`val/ord_loss`,
`val/cons_loss`) stay unweighted on purpose.

**Recommended ordinal experiments.** The exemplar lives at
[configs/experiments/ordinal_head.yaml](../configs/experiments/ordinal_head.yaml) — a
warmed-up ordinal head ($\lambda = 0.05$) on top of the exposure-capped sampler;
[configs/experiments/ordinal_consistency.yaml](../configs/experiments/ordinal_consistency.yaml)
adds the regression↔ordinal consistency term.

Warmup is currently intended mainly for the ordinal loss. Consistency
warmup is supported for future experiments, but consistency remains disabled
by default because previous experiments showed calibration instability.

---

## 18. Expected Benefits

The proposed method should help because each component addresses a specific weakness:

| Problem | Component |
|---|---|
| Rare high-occlusion samples | Soft balanced sampling + weighted loss |
| Regression ignores coarse regimes | Ordinal occlusion loss |
| Regression and classification may disagree | Consistency loss |
| Synthetic labels are not exact | Ranking loss instead of pseudo-regression |
| Embeddings may not encode occlusion geometry | Triplet contrastive loss |
| Embeddings may organize by gender-related shortcuts | Gender-aware triplet sampling |
| Synthetic artifacts may dominate | Face-aware region-controlled occlusion |

The most important conceptual principle is:

> Real labels provide calibration. Synthetic occlusions provide relative structure.

---

## 19. Main Risks and Safeguards

### Risk 1: Synthetic occlusions are unrealistic

Safeguard:

- use diverse occluders,
- avoid only black rectangles,
- inspect synthetic samples visually,
- use face-aware placement,
- reject invalid augmentations.

### Risk 2: Synthetic data corrupts calibration

Safeguard:

- do not apply regression loss to synthetic images initially,
- use synthetic images only for ranking and triplet losses.

### Risk 3: Auxiliary losses dominate regression

Safeguard:

- keep regression loss dominant,
- use small auxiliary coefficients,
- warm up auxiliary losses gradually.

### Risk 4: High-occlusion overfitting

Safeguard:

- monitor train vs validation high-bin error,
- use stress validation,
- avoid excessive class/loss weights,
- inspect high-error examples.

### Risk 5: Bin boundaries introduce discontinuities

Safeguard:

- use ordinal classification instead of hard multiclass bin classification,
- use soft regression-ordinal consistency,
- avoid hard constraints near thresholds.

### Risk 6: Gender-aware sampling over-constrains the representation

Safeguard:

- keep gender-aware triplet probabilities moderate,
- evaluate both regression error and gender gap,
- monitor bin × gender counts,
- treat adversarial gender removal as an optional ablation, not the default.

---

## 20. Recommended First Complete Implementation

The first complete version should include:

```text
1. Pretrained encoder, preferably ConvNeXt-Tiny/Small or DINOv2 ViT-S.
2. Regression head trained with existing weighted challenge loss.
3. Existing soft occlusion-balanced sampler.
4. Ordinal threshold head using thresholds [0.05, 0.10, 0.20, 0.40, 0.60].
5. Soft consistency between ordinal probabilities and regression-implied thresholds.
6. MediaPipe-based face landmark preprocessing.
7. Same-anchor synthetic mild/strong occlusion generation.
8. Synthetic monotonic ranking loss: original < mild < strong.
9. Projection head with triplet loss.
10. Gender-aware anchor and triplet sampling using occlusion bin × gender groups.
11. Natural validation + high-occlusion stress validation + bin × gender diagnostics.
```

The recommended initial total loss is:

$$
\boxed{
\mathcal{L}_{total}
=
\mathcal{L}_{reg}^{real}
+
0.2\mathcal{L}_{ord}^{real}
+
0.05\mathcal{L}_{cons}^{real}
+
0.01\mathcal{L}_{mono}^{real}
+
0.1\mathcal{L}_{rank}^{synthetic}
+
0.05\mathcal{L}_{triplet}^{real/synthetic}
}
$$

These coefficients should be treated as starting points, not fixed optimal values.

---

## 21. Final Summary

The proposed approach is an **occlusion-aware multi-task and metric-learning framework** for face occlusion regression.

It keeps the original regression task as the main objective but improves robustness through auxiliary supervision:

- ordinal classification teaches coarse occlusion regimes,
- consistency aligns regime prediction with continuous regression,
- synthetic monotonic ranking teaches that added occlusion should increase the predicted score,
- triplet learning structures embeddings by occlusion similarity,
- gender-aware triplet sampling encourages similar-occlusion samples to remain close across gender groups,
- soft balanced sampling and weighted loss compensate for rare high-occlusion examples and underrepresented bin-gender intersections.

The core formulation is:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{reg}^{real}
+
\lambda_{ord}\mathcal{L}_{ord}^{real}
+
\lambda_{cons}\mathcal{L}_{cons}^{real}
+
\lambda_{mono}\mathcal{L}_{mono}^{real}
+
\lambda_{rank}\mathcal{L}_{rank}^{synthetic}
+
\lambda_{triplet}\mathcal{L}_{triplet}^{real/synthetic}.
$$

The method is robust because it does not pretend that synthetic occlusions have exact labels. Instead, it uses them where they are reliable: as **relative monotonic supervision**.

The final prediction remains the calibrated regression score:

$$
\hat{y} = h_{reg}(E_\theta(x)).
$$
