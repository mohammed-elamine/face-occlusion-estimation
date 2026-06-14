"""Ordered-bin distribution head utilities (DEX expectation + DLDL/LDS soft labels).

Reframes the imbalanced occlusion regression as label-distribution learning over ``K`` ordered
bins, then recovers a continuous estimate via the expectation ``E[y] = sum_k p_k * c_k``.

Why this helps imbalanced regression (vs plain MSE):
  * Classification handles class imbalance far better than MSE, which is dominated by the huge
    low-occlusion bulk (so the model never commits to high values -> saturation).
  * Training against a smoothed *soft label distribution* (DLDL) instead of a one-hot/scalar
    target lets neighbouring bins share supervision (LDS, Yang et al. ICML 2020), so the
    data-poor high-occlusion tail borrows statistical strength from populated neighbours.
  * The expectation is bounded to ``[c_1, c_K]`` by construction (a convex combination of the
    centers), so no output activation is needed.

References: DEX (Rothe et al., IJCV 2018), DLDL (Gao et al., 2017), Label Distribution Learning
(Geng, TKDE 2016), Deep Imbalanced Regression / LDS (Yang et al., ICML 2020).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_bin_centers(n_bins: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """``n_bins`` evenly spaced ordered bin centers over ``[lo, hi]`` (inclusive)."""
    if int(n_bins) < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if not hi > lo:
        raise ValueError(f"require hi > lo, got lo={lo}, hi={hi}")
    return torch.linspace(float(lo), float(hi), int(n_bins))


def soft_label_distribution(
    targets: torch.Tensor, centers: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Gaussian (DLDL/LDS) soft labels over the bins: ``p*_k(y) ∝ exp(-(c_k - y)^2 / 2σ^2)``.

    ``targets`` is ``(B,)`` or ``(B, 1)``; ``centers`` is ``(K,)``; returns ``(B, K)`` rows that
    sum to 1. ``sigma`` is the smoothing width that lets neighbouring bins share supervision.
    """
    if not float(sigma) > 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    t = targets.reshape(-1, 1).to(centers.dtype)  # (B, 1)
    c = centers.reshape(1, -1)  # (1, K)
    log_weights = -((c - t) ** 2) / (2.0 * float(sigma) ** 2)
    # softmax of the Gaussian log-weights == normalized exp, numerically stable.
    return torch.softmax(log_weights, dim=1)


def expectation(probs: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Bin expectation ``E[y] = sum_k p_k * c_k`` -> ``(B,)``."""
    return probs @ centers.to(probs.dtype)


def dldl_kl_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """KL(soft_targets || softmax(logits)), averaged over the batch (DLDL objective)."""
    log_p = F.log_softmax(logits, dim=1)
    return F.kl_div(log_p, soft_targets, reduction="batchmean")
