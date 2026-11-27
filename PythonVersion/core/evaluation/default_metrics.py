"""Default problem-level metrics."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from core.evaluation.base import (
    EnvironmentTrendMetric,
    FETrendMetric,
    RuntimeMetric,
    SummaryMetric,
)
from core.registry import metric_registry
from core.evaluation.precision_metrics import METRIC_NAMES as OPTIONAL_PRECISION_METRIC_NAMES


KNOWN_OPTIMUM_DEFAULT_METRIC_NAMES = ("E_o", "E_bbc")
UNKNOWN_OPTIMUM_DEFAULT_METRIC_NAMES = ("P_o", "P_bbc")
DEFAULT_METRIC_NAMES = (
    *KNOWN_OPTIMUM_DEFAULT_METRIC_NAMES,
    *UNKNOWN_OPTIMUM_DEFAULT_METRIC_NAMES,
)
CORE_RUNTIME_METRIC_OUTPUT_NAMES = ("T_r",)


@metric_registry.register("E_o")
class OfflineErrorMetric(SummaryMetric):
    name: ClassVar[str] = "E_o"

    def compute(self, problem: Any, algorithm: Any | None = None) -> float:
        current_error = problem.state.current_error
        if current_error is None:
            return float("nan")
        values = np.asarray(current_error, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")


@metric_registry.register("E_bbc")
class BestBeforeChangeMetric(SummaryMetric):
    name: ClassVar[str] = "E_bbc"

    def compute(self, problem: Any, algorithm: Any | None = None) -> float:
        e_bbc = problem.state.e_bbc
        if e_bbc is None:
            return float("nan")
        values = np.asarray(e_bbc, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")


@metric_registry.register("P_o")
class OfflinePerformanceMetric(SummaryMetric):
    name: ClassVar[str] = "P_o"

    def compute(self, problem: Any, algorithm: Any | None = None) -> float:
        current_performance = problem.state.current_performance
        if current_performance is None:
            return float("nan")
        values = np.asarray(current_performance, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")


@metric_registry.register("P_bbc")
class BestPerformanceBeforeChangeMetric(SummaryMetric):
    name: ClassVar[str] = "P_bbc"

    def compute(self, problem: Any, algorithm: Any | None = None) -> float:
        p_bbc = problem.state.p_bbc
        if p_bbc is None:
            return float("nan")
        values = np.asarray(p_bbc, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")


class CurrentErrorTrend(FETrendMetric):
    name: ClassVar[str] = "current_error"

    def compute_sample(self, problem: Any, algorithm: Any) -> float:
        current_error = problem.state.current_error
        index = int(problem.state.fe) - 1
        if current_error is None or index < 0 or index >= current_error.size:
            return float("nan")
        return float(current_error[index])


class CurrentErrorGridTrend(RuntimeMetric):
    """Current Error at a frozen 1%-of-environment FE grid.

    Computed after the run from the problem-owned paid-FE clock, so collecting
    this curve adds no per-FE algorithm hook or search-time overhead.
    """

    name: ClassVar[str] = "current_error_grid"
    kind: ClassVar[str] = "run_trend"

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.values: list[float] = []

    def sample_run(self, problem: Any, algorithm: Any) -> None:
        frequency = int(problem.change_frequency)
        error = np.asarray(problem.state.current_error, dtype=float)
        if error.size != frequency * int(problem.environment_number):
            raise ValueError("Current Error grid requires a complete FE budget")
        stride = max(1, frequency // 100)
        local = np.unique(np.r_[0, np.arange(stride - 1, frequency, stride), frequency - 1])
        for environment in range(int(problem.environment_number)):
            indices = environment * frequency + local
            self.keys.extend((indices + 1).tolist())
            self.values.extend(error[indices].tolist())

    def trend(self) -> dict[str, Any] | None:
        if not self.keys:
            return None
        return {"axis": "fe", "key": self.keys, "value": self.values, "name": self.name}


class CurrentPerformanceTrend(FETrendMetric):
    name: ClassVar[str] = "current_performance"

    def compute_sample(self, problem: Any, algorithm: Any) -> float:
        current_performance = problem.state.current_performance
        index = int(problem.state.fe) - 1
        if (
            current_performance is None
            or index < 0
            or index >= current_performance.size
        ):
            return float("nan")
        return float(current_performance[index])


class BestBeforeChangeErrorTrend(EnvironmentTrendMetric):
    """Run-aggregatable best error for each completed environment."""

    name: ClassVar[str] = "E_bbc_by_environment"

    def compute_sample(self, problem: Any, algorithm: Any) -> float:
        values = problem.state.e_bbc
        index = int(problem.state.environment_index)
        if values is None or index < 0 or index >= values.size:
            return float("nan")
        return float(values[index])


class BestBeforeChangePerformanceTrend(EnvironmentTrendMetric):
    """Run-aggregatable best performance for each completed environment."""

    name: ClassVar[str] = "P_bbc_by_environment"

    def compute_sample(self, problem: Any, algorithm: Any) -> float:
        values = problem.state.p_bbc
        index = int(problem.state.environment_index)
        if values is None or index < 0 or index >= values.size:
            return float("nan")
        return float(values[index])


def default_summary_metric_names(problem: Any | None) -> tuple[str, ...]:
    if problem is not None and has_known_optimum(problem):
        return KNOWN_OPTIMUM_DEFAULT_METRIC_NAMES
    if problem is not None:
        return UNKNOWN_OPTIMUM_DEFAULT_METRIC_NAMES
    return DEFAULT_METRIC_NAMES


def default_trend_metric_classes(
    problem: Any | None,
) -> tuple[type[RuntimeMetric], ...]:
    if problem is not None and has_known_optimum(problem):
        return (CurrentErrorTrend,)
    if problem is not None:
        return (CurrentPerformanceTrend,)
    return (
        CurrentErrorTrend,
        CurrentPerformanceTrend,
    )


def optional_trend_metric_classes(
    problem: Any | None,
) -> tuple[type[RuntimeMetric], ...]:
    """Core trends available only when explicitly selected by a task."""

    if problem is not None and has_known_optimum(problem):
        return (BestBeforeChangeErrorTrend, CurrentErrorGridTrend)
    if problem is not None:
        return (BestBeforeChangePerformanceTrend,)
    return (BestBeforeChangeErrorTrend, BestBeforeChangePerformanceTrend, CurrentErrorGridTrend)


def has_known_optimum(problem: Any) -> bool:
    return current_optimum_value_or_none(problem) is not None


def record_default_metrics(problem: Any, value: float) -> None:
    """Update default problem-owned metric state after one FE."""

    index = int(problem.state.fe) - 1
    optimum_value = current_optimum_value_or_none(problem)
    error = np.nan if optimum_value is None else float(optimum_value - value)
    if problem.state.fe % problem.change_frequency == 1:
        problem.state.best_error = error if np.isfinite(error) else np.inf
        problem.state.best_performance = value
    else:
        if np.isfinite(error) and error < problem.state.best_error:
            problem.state.best_error = error
        if value > problem.state.best_performance:
            problem.state.best_performance = value

    assert problem.state.current_error is not None
    assert problem.state.current_performance is not None
    problem.state.current_error[index] = (
        problem.state.best_error if np.isfinite(problem.state.best_error) else np.nan
    )
    problem.state.current_performance[index] = problem.state.best_performance

    if problem.state.fe % problem.change_frequency == 0:
        record_best_before_change(problem)


def record_best_before_change(problem: Any) -> None:
    assert problem.state.e_bbc is not None
    assert problem.state.p_bbc is not None
    env = min(problem.state.environment_index, problem.state.e_bbc.size - 1)
    problem.state.e_bbc[env] = (
        problem.state.best_error if np.isfinite(problem.state.best_error) else np.nan
    )
    problem.state.p_bbc[env] = (
        problem.state.best_performance
        if np.isfinite(problem.state.best_performance)
        else np.nan
    )


def current_optimum_value_or_none(problem: Any) -> float | None:
    try:
        value = problem.current_optimum_value()
    except NotImplementedError:
        return None
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def default_trend_metric_class_by_name(
    metric_classes: tuple[type[RuntimeMetric], ...] = (),
) -> dict[str, type[RuntimeMetric]]:
    return {
        metric_cls.name: metric_cls
        for metric_cls in (
            CurrentErrorTrend,
            CurrentErrorGridTrend,
            CurrentPerformanceTrend,
            BestBeforeChangeErrorTrend,
            BestBeforeChangePerformanceTrend,
            *metric_classes,
        )
        if getattr(metric_cls, "kind", None) in {"fe_trend", "environment_trend", "run_trend"}
    }


def core_trend_metric_names() -> set[str]:
    return set(default_trend_metric_class_by_name())


def core_metric_output_names() -> list[str]:
    return sorted(
        set(metric_registry.names())
        | core_trend_metric_names()
        | set(CORE_RUNTIME_METRIC_OUTPUT_NAMES)
    )
