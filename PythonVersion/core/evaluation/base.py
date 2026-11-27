"""Metric contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from core.registry import metric_registry


MetricKind = Literal["summary", "fe_trend", "environment_trend", "run_trend"]


class RuntimeMetric(ABC):
    """Base class for metrics sampled or summarized during a run."""

    name: ClassVar[str]
    kind: ClassVar[MetricKind]
    required_state_keys: ClassVar[set[str]] = set()
    export: ClassVar[bool] = True

    def sample_fe(self, problem: Any, algorithm: Any) -> None:
        """Sample a metric after one function evaluation."""

    def sample_environment(self, problem: Any, algorithm: Any) -> None:
        """Sample a metric at the end of one environment."""

    def sample_run(self, problem: Any, algorithm: Any) -> None:
        """Build a passive trend after the FE budget is complete."""

    def compute(self, problem: Any, algorithm: Any) -> float:
        raise NotImplementedError(f"{self.name} is not a summary metric.")

    def trend(self) -> dict[str, Any] | None:
        return None


class SummaryMetric(RuntimeMetric):
    kind: ClassVar[MetricKind] = "summary"

    @abstractmethod
    def compute(self, problem: Any, algorithm: Any) -> float:
        ...


class FETrendMetric(RuntimeMetric):
    kind: ClassVar[MetricKind] = "fe_trend"
    axis: ClassVar[str] = "fe"

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.values: list[float] = []

    def sample_fe(self, problem: Any, algorithm: Any) -> None:
        self.keys.append(int(problem.state.fe))
        self.values.append(float(self.compute_sample(problem, algorithm)))

    @abstractmethod
    def compute_sample(self, problem: Any, algorithm: Any) -> float:
        ...

    def trend(self) -> dict[str, Any] | None:
        if not self.keys:
            return None
        return {
            "axis": self.axis,
            "key": self.keys,
            "value": self.values,
            "name": self.name,
        }


class EnvironmentTrendMetric(RuntimeMetric):
    kind: ClassVar[MetricKind] = "environment_trend"
    axis: ClassVar[str] = "environment"

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.values: list[float] = []

    def sample_environment(self, problem: Any, algorithm: Any) -> None:
        environment = int(problem.state.environment_index) + 1
        self.keys.append(environment)
        self.values.append(float(self.compute_sample(problem, algorithm)))

    @abstractmethod
    def compute_sample(self, problem: Any, algorithm: Any) -> float:
        ...

    def trend(self) -> dict[str, Any] | None:
        if not self.keys:
            return None
        return {
            "axis": self.axis,
            "key": self.keys,
            "value": self.values,
            "name": self.name,
        }


class MetricManager:
    """Sample core and algorithm-owned metrics on the common FE clock."""

    def __init__(
        self,
        metric_names: list[str] | tuple[str, ...] | None = None,
        metric_classes: tuple[type[RuntimeMetric], ...] = (),
        problem: Any | None = None,
    ):
        from core.evaluation.default_metrics import default_trend_metric_class_by_name

        custom_summary_names = {
            metric_cls.name
            for metric_cls in metric_classes
            if getattr(metric_cls, "kind", None) == "summary"
        }
        core_trend_class_by_name = default_trend_metric_class_by_name()
        available_trend_class_by_name = default_trend_metric_class_by_name(
            metric_classes
        )
        requested_names = tuple(metric_names) if metric_names is not None else ()
        summary_names = _unique_names(
            (
                *self.default_summary_metric_names(problem),
                *(
                    name
                    for name in requested_names
                    if name in metric_registry or name in custom_summary_names
                ),
            )
        )
        unknown = [
            name
            for name in requested_names
            if (
                name not in metric_registry
                and name not in custom_summary_names
                and name not in available_trend_class_by_name
            )
        ]
        if unknown:
            raise KeyError(f"Unknown metric(s): {', '.join(unknown)}.")
        registered_names = tuple(name for name in summary_names if name in metric_registry)
        # ``metric_names`` is also the collection contract for trends.  The
        # previous implementation always instantiated the problem's default
        # FE trend (for GMPB, a Python list with one value per FE), even when a
        # task explicitly requested only summary/custom metrics.  That made a
        # long run retain an unnecessary copy of the full error trajectory in
        # every worker.  Keep the historical defaults when no selection was
        # supplied, but honor an explicit selection at construction time.
        selected_default_trends = (
            self.default_trend_metric_classes(problem)
            if metric_names is None
            else tuple(
                metric_cls
                for metric_cls in self.default_trend_metric_classes(problem)
                if metric_cls.name in requested_names
            )
        )
        selected_custom_metrics = tuple(
            metric_cls
            for metric_cls in metric_classes
            if metric_names is None
            or getattr(metric_cls, "kind", None) == "summary"
            or metric_cls.name in requested_names
        )
        trend_metrics = _unique_metric_classes(
            (
                *selected_default_trends,
                *(
                    core_trend_class_by_name[name]
                    for name in requested_names
                    if name in core_trend_class_by_name
                ),
            )
        )
        self._metrics: list[RuntimeMetric] = [
            *(metric_registry.get(name)() for name in registered_names),
            *(metric_cls() for metric_cls in trend_metrics),
            *(metric_cls() for metric_cls in selected_custom_metrics),
        ]
        # Keep the hot-path metric sets separate.  ``BaseProblem.evaluate``
        # calls ``sample_fe`` once for every consumed FE, so scanning summary
        # metrics and invoking algorithm preparation hooks when a task only
        # requested scalar outputs adds measurable overhead to long runs.
        # Environment-trend metrics are included in the FE-preparation gate
        # because their accumulators may be updated by ``prepare_fe_metrics``.
        self._fe_metrics = [
            metric for metric in self._metrics if metric.kind == "fe_trend"
        ]
        self._environment_metrics = [
            metric for metric in self._metrics if metric.kind == "environment_trend"
        ]
        self._run_metrics = [
            metric for metric in self._metrics if metric.kind == "run_trend"
        ]

    @staticmethod
    def default_summary_metric_names(problem: Any | None) -> tuple[str, ...]:
        from core.evaluation.default_metrics import default_summary_metric_names

        return default_summary_metric_names(problem)

    @staticmethod
    def default_trend_metric_classes(
        problem: Any | None,
    ) -> tuple[type[RuntimeMetric], ...]:
        from core.evaluation.default_metrics import default_trend_metric_classes

        return default_trend_metric_classes(problem)

    @staticmethod
    def has_known_optimum(problem: Any) -> bool:
        from core.evaluation.default_metrics import has_known_optimum

        return has_known_optimum(problem)

    @staticmethod
    def record_default_metrics(problem: Any, value: float) -> None:
        from core.evaluation.default_metrics import record_default_metrics

        record_default_metrics(problem, value)

    @staticmethod
    def record_best_before_change(problem: Any) -> None:
        from core.evaluation.default_metrics import record_best_before_change

        record_best_before_change(problem)

    @staticmethod
    def _current_optimum_value_or_none(problem: Any) -> float | None:
        from core.evaluation.default_metrics import current_optimum_value_or_none

        return current_optimum_value_or_none(problem)

    def sample_fe(self, problem: Any, algorithm: Any) -> None:
        if not self._fe_metrics and not self._environment_metrics:
            return
        prepare = getattr(algorithm, "prepare_fe_metrics", None)
        if prepare is not None:
            prepare(problem)
        for metric in self._fe_metrics:
            self._ensure_required_state(metric, algorithm)
            metric.sample_fe(problem, algorithm)

    def sample_environment(self, problem: Any, algorithm: Any) -> None:
        if not self._environment_metrics:
            return
        prepare = getattr(algorithm, "prepare_environment_metrics", None)
        if prepare is not None:
            prepare(problem)
        for metric in self._environment_metrics:
            self._ensure_required_state(metric, algorithm)
            metric.sample_environment(problem, algorithm)
        finalize = getattr(algorithm, "finalize_environment_metrics", None)
        if finalize is not None:
            finalize(problem)

    def summary(self, problem: Any, algorithm: Any) -> dict[str, float]:
        values: dict[str, float] = {}
        for metric in self._metrics:
            if metric.kind != "summary":
                continue
            self._ensure_required_state(metric, algorithm)
            values[metric.name] = float(metric.compute(problem, algorithm))
        return values

    def sample_run(self, problem: Any, algorithm: Any) -> None:
        for metric in self._run_metrics:
            self._ensure_required_state(metric, algorithm)
            metric.sample_run(problem, algorithm)

    def trends(self) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for metric in self._metrics:
            if metric.kind not in {"fe_trend", "environment_trend", "run_trend"}:
                continue
            trend = metric.trend()
            if trend is not None:
                values[metric.name] = trend
        return values

    def _ensure_required_state(self, metric: RuntimeMetric, algorithm: Any) -> None:
        if not metric.required_state_keys:
            return
        state = getattr(algorithm, "metric_state", {})
        missing = sorted(key for key in metric.required_state_keys if key not in state)
        if missing:
            names = ", ".join(missing)
            raise KeyError(
                f"{algorithm.name} metric {metric.name!r} requires "
                f"metric_state key(s): {names}."
            )


def _unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def _unique_metric_classes(
    metric_classes: tuple[type[RuntimeMetric], ...],
) -> tuple[type[RuntimeMetric], ...]:
    return tuple(dict.fromkeys(metric_classes))
