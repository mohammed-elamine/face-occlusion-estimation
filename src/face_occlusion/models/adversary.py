"""Gender-adversarial invariance head (DANN-style) for debiasing the representation.

Our gender gap is driven by a gender shortcut **entangled in the encoder features** — DFR (a
last-layer refit) could not remove it (see ``tmp/model_study/05_gender_gap.md``). The proper fix
is representation-level: a **gradient-reversal** gender adversary (Ganin et al. 2016; Zhang et al.
2018) trained on the pooled features. The adversary tries to predict gender; the reversed gradient
pushes the **encoder** to make features gender-uninformative, dismantling the shortcut at the
source. A **conditional** variant feeds the occlusion bin to the adversary so the encoder only
removes the gender information *not explained by occlusion* (equalized-odds style; Zhao et al.
2020) — preserving the legitimate occlusion signal. Training-only; dropped at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Identity forward; gradient multiplied by ``-lambda_`` on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Gradient-reversal layer: forward is identity, backward flips (and scales) the gradient."""
    return GradientReversal.apply(x, lambda_)


class GenderAdversary(nn.Module):
    """Small MLP predicting gender (1 logit, male=1) from features → adversary for invariance."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
