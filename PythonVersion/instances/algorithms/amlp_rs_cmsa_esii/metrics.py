"""AMLP-RS-CMSA-ESII environment trends: restarts, repelling archive and prediction use.

The restart lifecycle keeps the per-environment counts. When an environment
ends, ``prepare_environment_metrics`` copies them into ``metric_state``; the
trends only read that state, so selecting them never changes the search.
"""

from __future__ import annotations

from typing import Any, ClassVar

from core.evaluation.base import EnvironmentTrendMetric


class _AMLPEnvironmentMetric(EnvironmentTrendMetric):
    key: ClassVar[str]

    def compute_sample(self, problem: Any, algorithm: Any) -> float:
        return float(algorithm.metric_state[self.key])


class ArchiveSizeByEnvironment(_AMLPEnvironmentMetric):
    """Archived restart bests that repel later samples when the environment ends."""

    name: ClassVar[str] = "amlp_archive_size_by_environment"
    key: ClassVar[str] = "amlp_archive_size"
    required_state_keys: ClassVar[set[str]] = {"amlp_archive_size"}


class CompletedRestartsByEnvironment(_AMLPEnvironmentMetric):
    """Restarts that met a stop or merge criterion within the environment."""

    name: ClassVar[str] = "amlp_completed_restarts_by_environment"
    key: ClassVar[str] = "amlp_completed_restarts"
    required_state_keys: ClassVar[set[str]] = {"amlp_completed_restarts"}


class PredictedRestartsByEnvironment(_AMLPEnvironmentMetric):
    """Restarts within the environment that began at a predicted optimum position."""

    name: ClassVar[str] = "amlp_predicted_restarts_by_environment"
    key: ClassVar[str] = "amlp_predicted_restarts"
    required_state_keys: ClassVar[set[str]] = {"amlp_predicted_restarts"}


class MeanPredictionLevelByEnvironment(_AMLPEnvironmentMetric):
    """Mean prediction level of those restarts; NaN when none began at a prediction."""

    name: ClassVar[str] = "amlp_mean_prediction_level_by_environment"
    key: ClassVar[str] = "amlp_mean_prediction_level"
    required_state_keys: ClassVar[set[str]] = {"amlp_mean_prediction_level"}


METRICS = (
    ArchiveSizeByEnvironment,
    CompletedRestartsByEnvironment,
    PredictedRestartsByEnvironment,
    MeanPredictionLevelByEnvironment,
)
METRIC_NAMES = tuple(metric.name for metric in METRICS)

__all__ = [
    "ArchiveSizeByEnvironment",
    "CompletedRestartsByEnvironment",
    "MeanPredictionLevelByEnvironment",
    "METRICS",
    "METRIC_NAMES",
    "PredictedRestartsByEnvironment",
]
