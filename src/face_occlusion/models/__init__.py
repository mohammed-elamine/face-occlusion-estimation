"""Public model builders."""

from .distribution import dldl_kl_loss, expectation, make_bin_centers, soft_label_distribution
from .ordinal import (
    CONSISTENCY_MODES,
    DEFAULT_ORDINAL_THRESHOLD_WEIGHTS,
    DEFAULT_ORDINAL_THRESHOLDS,
    OrdinalHead,
    make_ordinal_targets,
    ordinal_monotonicity_loss,
    ordinal_monotonicity_violation_rate,
    regression_ordinal_consistency_loss,
    threshold_weighted_bce,
)
from .outputs import OcclusionModelOutput
from .ranking import monotonic_ranking_loss, ordering_accuracy, ranknet_loss
from .regressor import OcclusionRegressor, build_model

__all__ = [
    "CONSISTENCY_MODES",
    "DEFAULT_ORDINAL_THRESHOLDS",
    "DEFAULT_ORDINAL_THRESHOLD_WEIGHTS",
    "OcclusionModelOutput",
    "OcclusionRegressor",
    "OrdinalHead",
    "build_model",
    "dldl_kl_loss",
    "expectation",
    "make_bin_centers",
    "make_ordinal_targets",
    "soft_label_distribution",
    "monotonic_ranking_loss",
    "ordering_accuracy",
    "ordinal_monotonicity_loss",
    "ordinal_monotonicity_violation_rate",
    "ranknet_loss",
    "regression_ordinal_consistency_loss",
    "threshold_weighted_bce",
]
