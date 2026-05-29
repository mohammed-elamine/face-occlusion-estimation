"""Small visualization helpers for validation logging."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def prediction_histogram(preds: np.ndarray, title: str = "Predictions") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(preds, bins=40, range=(min(0.0, float(preds.min())), max(1.0, float(preds.max()))))
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def scatter_pred_vs_target(preds: np.ndarray, targets: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, s=4, alpha=0.4)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(min(-0.1, float(preds.min())), max(1.1, float(preds.max())))
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.set_title("Prediction vs target")
    fig.tight_layout()
    return fig


def error_by_bin_plot(bins: Sequence[float], errors: Sequence[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
    ax.bar(labels, errors)
    ax.set_ylabel("weighted MSE")
    ax.set_title("Error by occlusion bin")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig
