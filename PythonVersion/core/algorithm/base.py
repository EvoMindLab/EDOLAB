"""Algorithm contracts."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from core.compute import compute_report, reset_compute_trace
from core.evaluation.base import MetricManager, RuntimeMetric
from core.evaluation.result import RunResult
from core.parameters import ParameterSpec, resolve_parameters
from core.registry import problem_registry


class BaseAlgorithm(ABC):
    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        ...

    def run_once(
        self,
        problem_name: str,
        problem_overrides: dict[str, Any],
        algorithm_overrides: dict[str, Any],
        problem_seed: int,
        algorithm_seed: int,
        run_index: int = 1,
        metric_names: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        reset_compute_trace()
        problem_cls = problem_registry.get(problem_name)
        problem = problem_cls.create(problem_overrides)
        problem.reset(problem_seed)
        self.parameters = resolve_parameters(self.parameter_specs(), algorithm_overrides)
        self.rng = np.random.default_rng(algorithm_seed)
        self.problem = problem
        self.iteration = 0
        self.metric_state: dict[str, Any] = {}
        # ``T_r`` is synthesized from ``runtime_seconds`` by the result store;
        # it is not a runtime metric sampled on the FE clock.
        selected_metric_names = (
            None
            if metric_names is None
            else [name for name in metric_names if name != "T_r"]
        )
        self.metric_manager = MetricManager(
            selected_metric_names,
            self.custom_metric_classes(),
            problem=problem,
        )
        problem.attach_metric_observer(self.metric_manager, self)

        start = time.perf_counter()
        self.initialize(problem)
        iteration = 0
        while not problem.state.terminated:
            iteration += 1
            self.iteration = iteration
            self.iterate(problem)
            if problem.state.recent_change:
                problem.state.acknowledge_change()
                self.on_environment_change(problem)
        runtime = time.perf_counter() - start

        current_error = []
        if problem.state.current_error is not None:
            current_error = problem.state.current_error.astype(float).tolist()
        metrics = self.metric_manager.summary(problem, self)
        self.metric_manager.sample_run(problem, self)
        trends = self.metric_manager.trends()
        return RunResult(
            algorithm_name=self.name,
            problem_name=problem_name,
            run_index=run_index,
            problem_seed=problem_seed,
            algorithm_seed=algorithm_seed,
            metrics=metrics,
            runtime_seconds=runtime,
            fe=problem.state.fe,
            current_error=current_error,
            raw={
                "iteration": iteration,
                "compute": compute_report(),
                "trends": trends,
            },
        )

    @classmethod
    def custom_metric_classes(cls) -> tuple[type[RuntimeMetric], ...]:
        return ()

    def prepare_fe_metrics(self, problem) -> None:
        pass

    def prepare_environment_metrics(self, problem) -> None:
        pass

    def finalize_environment_metrics(self, problem) -> None:
        pass

    @abstractmethod
    def initialize(self, problem) -> None:
        ...

    @abstractmethod
    def iterate(self, problem) -> None:
        ...

    def on_environment_change(self, problem) -> None:
        pass
