"""Experiment runner."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import instances  # noqa: F401
from core.evaluation.aggregator import aggregate_runs
from core.evaluation.base import MetricManager
from core.evaluation.default_metrics import OPTIONAL_PRECISION_METRIC_NAMES, optional_trend_metric_classes
from core.evaluation.result import ExperimentResult, RunResult
from core.experiment.config import BatchExperimentConfig, ExperimentConfig
from core.experiment.scheduler import (
    CancelCheck,
    GlobalRunScheduler,
    RunSchedulerCanceled,
    RunTask,
)
from core.experiment.seeds import resolve_seed_sequence
from core.registry import algorithm_registry, problem_registry
from core.result.store import ResultStore, RunsetTrendAccumulators


ProgressCallback = Callable[[RunResult, int, int], None]


class ExperimentRunner:
    def __init__(self) -> None:
        self._metric_output_name_cache: dict[tuple[str, str], set[str]] = {}

    def run(
        self,
        config: ExperimentConfig,
        progress: ProgressCallback | None = None,
    ) -> list[ExperimentResult]:
        tasks = self._build_tasks(config)
        scheduler = GlobalRunScheduler(
            config.max_parallel_runs,
            gpu_inference_concurrency=config.gpu_inference_concurrency,
        )
        runs = scheduler.run(tasks, progress=progress)
        return self._aggregate_config(config, runs)

    def run_batch(
        self,
        config: BatchExperimentConfig,
        progress: ProgressCallback | None = None,
    ) -> list[ExperimentResult]:
        self._prepare_batch_labels(config.tasks)
        tasks: list[RunTask] = []
        for task_config in config.tasks:
            tasks.extend(self._build_tasks(task_config))

        scheduler = GlobalRunScheduler(
            config.max_parallel_runs,
            gpu_inference_concurrency=config.gpu_inference_concurrency,
        )
        runs = scheduler.run(tasks, progress=progress)

        grouped: dict[tuple[str | None, str], list[RunResult]] = defaultdict(list)
        for result in runs:
            grouped[(result.label, result.algorithm_name)].append(result)

        results: list[ExperimentResult] = []
        for task_config in config.tasks:
            for algorithm_name in task_config.algorithm_names:
                results.append(
                    aggregate_runs(
                        algorithm_name,
                        task_config.problem_name,
                        grouped[(task_config.label, algorithm_name)],
                        label=task_config.label,
                        metric_output_names=task_config.metric_output_names,
                    )
                )
        return results

    def run_to_store(
        self,
        config: ExperimentConfig,
        output_dir: str | Path,
        progress: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Path:
        label = config.label or "task-1"
        tasks = self._build_tasks(config)
        store = ResultStore(output_dir)
        try:
            store.set_meta("kind", "raw_runs")
            store.set_meta("mode", "run")
            task_id = store.register_task(
                label=label,
                problem_name=config.problem_name,
                run_number=config.run_number,
                problem_seed=config.problem_seed,
                algorithm_seed=config.algorithm_seed,
                problem_params=dict(config.problem_overrides),
                metric_output_names=config.metric_output_names,
                source_task_file=config.source_task_file,
                source_line=config.source_line,
                task_config_hash=config.task_config_hash,
            )
            runset_by_key: dict[tuple[str | None, str], int] = {}
            expected_by_runset: dict[int, int] = {}
            completed_by_runset: dict[int, int] = defaultdict(int)
            trend_by_runset: dict[int, RunsetTrendAccumulators] = {}
            metric_outputs_by_runset: dict[int, list[str] | None] = {}
            for algorithm_name in config.algorithm_names:
                runset_id = store.register_runset(
                    task_id=task_id,
                    label=label,
                    algorithm_name=algorithm_name,
                    problem_name=config.problem_name,
                    algorithm_params=self._algorithm_parameter_overrides(
                        config, algorithm_name
                    ),
                    expected_runs=config.run_number,
                )
                runset_by_key[(label, algorithm_name)] = runset_id
                runset_by_key[(None, algorithm_name)] = runset_id
                expected_by_runset[runset_id] = config.run_number
                trend_by_runset[runset_id] = RunsetTrendAccumulators()
                metric_outputs_by_runset[runset_id] = config.metric_output_names

            self._execute_tasks_to_store(
                tasks,
                store,
                runset_by_key,
                expected_by_runset,
                completed_by_runset,
                trend_by_runset,
                metric_outputs_by_runset,
                config.max_parallel_runs,
                config.gpu_inference_concurrency,
                progress,
                cancel_check,
            )
            store.set_meta("status", "completed")
            return store.output_dir
        except RunSchedulerCanceled:
            store.set_meta("status", "canceled")
            store.mark_incomplete_canceled("Canceled by user.")
            raise
        finally:
            store.close()

    def run_batch_to_store(
        self,
        config: BatchExperimentConfig,
        output_dir: str | Path,
        progress: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Path:
        self._prepare_batch_labels(config.tasks)
        task_runs: list[tuple[ExperimentConfig, list[RunTask]]] = []
        all_tasks: list[RunTask] = []
        for task_config in config.tasks:
            built = self._build_tasks(task_config)
            task_runs.append((task_config, built))
            all_tasks.extend(built)

        store = ResultStore(output_dir, event_sidecar=getattr(config, 'event_sidecar', True))
        try:
            store.set_meta("kind", "raw_runs")
            store.set_meta("mode", "batch")
            runset_by_key: dict[tuple[str | None, str], int] = {}
            expected_by_runset: dict[int, int] = {}
            completed_by_runset: dict[int, int] = defaultdict(int)
            trend_by_runset: dict[int, RunsetTrendAccumulators] = {}
            metric_outputs_by_runset: dict[int, list[str] | None] = {}
            for task_config, _ in task_runs:
                assert task_config.label is not None
                task_id = store.register_task(
                    label=task_config.label,
                    problem_name=task_config.problem_name,
                    run_number=task_config.run_number,
                    problem_seed=task_config.problem_seed,
                    algorithm_seed=task_config.algorithm_seed,
                    problem_params=dict(task_config.problem_overrides),
                    metric_output_names=task_config.metric_output_names,
                    source_task_file=task_config.source_task_file,
                    source_line=task_config.source_line,
                    task_config_hash=task_config.task_config_hash,
                )
                for algorithm_name in task_config.algorithm_names:
                    runset_id = store.register_runset(
                        task_id=task_id,
                        label=task_config.label,
                        algorithm_name=algorithm_name,
                        problem_name=task_config.problem_name,
                        algorithm_params=self._algorithm_parameter_overrides(
                            task_config, algorithm_name
                        ),
                        expected_runs=task_config.run_number,
                    )
                    runset_by_key[(task_config.label, algorithm_name)] = runset_id
                    expected_by_runset[runset_id] = task_config.run_number
                    trend_by_runset[runset_id] = RunsetTrendAccumulators()
                    metric_outputs_by_runset[runset_id] = task_config.metric_output_names

            self._execute_tasks_to_store(
                all_tasks,
                store,
                runset_by_key,
                expected_by_runset,
                completed_by_runset,
                trend_by_runset,
                metric_outputs_by_runset,
                config.max_parallel_runs,
                config.gpu_inference_concurrency,
                progress,
                cancel_check,
            )
            store.set_meta("status", "completed")
            return store.output_dir
        except RunSchedulerCanceled:
            store.set_meta("status", "canceled")
            store.mark_incomplete_canceled("Canceled by user.")
            raise
        finally:
            store.close()

    def _build_tasks(self, config: ExperimentConfig) -> list[RunTask]:
        self._validate_algorithm_parameters(config)
        self._validate_metrics(config)
        problem_seeds = resolve_seed_sequence(config.problem_seed, config.run_number)
        algorithm_seeds = resolve_seed_sequence(
            config.algorithm_seed, config.run_number
        )
        tasks: list[RunTask] = []
        for algorithm_name in config.algorithm_names:
            overrides = dict(config.shared_algorithm_overrides)
            overrides.update(config.algorithm_overrides.get(algorithm_name, {}))
            for run_index in range(config.run_number):
                tasks.append(
                    RunTask(
                        algorithm_name=algorithm_name,
                        problem_name=config.problem_name,
                        run_index=run_index + 1,
                        problem_seed=problem_seeds[run_index],
                        algorithm_seed=algorithm_seeds[run_index],
                        problem_overrides=dict(config.problem_overrides),
                        algorithm_overrides=overrides,
                        metric_output_names=(
                            list(config.metric_output_names)
                            if config.metric_output_names is not None
                            else None
                        ),
                        label=config.label,
                    )
                )
        return tasks

    def _execute_tasks_to_store(
        self,
        tasks: list[RunTask],
        store: ResultStore,
        runset_by_key: dict[tuple[str | None, str], int],
        expected_by_runset: dict[int, int],
        completed_by_runset: dict[int, int],
        trend_by_runset: dict[int, RunsetTrendAccumulators],
        metric_outputs_by_runset: dict[int, list[str] | None],
        max_parallel_runs: int,
        gpu_inference_concurrency: int | None,
        progress: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> None:
        scheduler = GlobalRunScheduler(
            max_parallel_runs,
            gpu_inference_concurrency=gpu_inference_concurrency,
        )
        for result, completed, total_runs in scheduler.iter_run(
            tasks,
            cancel_check=cancel_check,
        ):
            runset_id = runset_by_key[(result.label, result.algorithm_name)]
            metric_output_names = metric_outputs_by_runset.get(runset_id)
            store.write_run_result(
                runset_id,
                result,
                metric_output_names=metric_output_names,
            )
            trend_by_runset[runset_id].add_run(
                result,
                metric_output_names,
            )
            completed_by_runset[runset_id] += 1
            if completed_by_runset[runset_id] == expected_by_runset[runset_id]:
                trend_accumulators = trend_by_runset.pop(runset_id)
                store.finalize_runset(
                    runset_id,
                    trend_accumulators.summaries(),
                )
            if progress is not None:
                progress(result, completed, total_runs)

    def _algorithm_parameter_overrides(
        self,
        config: ExperimentConfig,
        algorithm_name: str,
    ) -> dict:
        overrides = dict(config.shared_algorithm_overrides)
        overrides.update(config.algorithm_overrides.get(algorithm_name, {}))
        return overrides

    def _aggregate_config(
        self,
        config: ExperimentConfig,
        runs: list[RunResult],
    ) -> list[ExperimentResult]:
        grouped: dict[str, list[RunResult]] = defaultdict(list)
        for result in runs:
            grouped[result.algorithm_name].append(result)

        return [
            aggregate_runs(
                algorithm_name,
                config.problem_name,
                grouped[algorithm_name],
                label=config.label,
                metric_output_names=config.metric_output_names,
            )
            for algorithm_name in config.algorithm_names
        ]

    def _prepare_batch_labels(self, tasks: list[ExperimentConfig]) -> None:
        labels: set[str] = set()
        for index, task_config in enumerate(tasks, start=1):
            label = (
                str(task_config.label).strip() if task_config.label is not None else ""
            )
            task_config.label = label or f"task-{index}"
            if task_config.label in labels:
                raise ValueError(f"Duplicate task label {task_config.label!r}.")
            labels.add(task_config.label)

    def _validate_algorithm_parameters(self, config: ExperimentConfig) -> None:
        for algorithm_name in config.algorithm_names:
            algorithm_cls = algorithm_registry.get(algorithm_name)
            specs = algorithm_cls.parameter_specs()
            shared_unknown = sorted(
                name for name in config.shared_algorithm_overrides if name not in specs
            )
            specific_unknown = sorted(
                name
                for name in config.algorithm_overrides.get(algorithm_name, {})
                if name not in specs
            )
            if shared_unknown:
                names = ", ".join(shared_unknown)
                raise ValueError(
                    f"Shared algorithm parameter(s) are not valid for {algorithm_name}: {names}. "
                    "Use AlgorithmName.parameter=value for algorithm-specific overrides."
                )
            if specific_unknown:
                names = ", ".join(specific_unknown)
                raise ValueError(f"Unknown parameter(s) for {algorithm_name}: {names}.")

    def _validate_metrics(self, config: ExperimentConfig) -> None:
        if config.metric_output_names is None:
            return
        computed_names = self._computed_metric_output_names(config)
        for algorithm_name in config.algorithm_names:
            algorithm_cls = algorithm_registry.get(algorithm_name)
            available_names = set(computed_names)
            available_names.update(
                metric_cls.name
                for metric_cls in algorithm_cls.custom_metric_classes()
                if getattr(metric_cls, "kind", None) in {
                    "summary",
                    "fe_trend",
                    "environment_trend",
                    "run_trend",
                }
            )
            unknown = sorted(
                name
                for name in config.metric_output_names
                if name not in available_names
            )
            if unknown:
                names = ", ".join(unknown)
                available = ", ".join(sorted(available_names))
                raise ValueError(
                    "Metric output(s) are not available for "
                    f"algorithm {algorithm_name} in this configuration: {names}. "
                    f"Available metrics for {algorithm_name}: {available}."
                )

    def _computed_metric_output_names(self, config: ExperimentConfig) -> set[str]:
        cache_key = (
            config.problem_name,
            json.dumps(
                config.problem_overrides,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        cached = self._metric_output_name_cache.get(cache_key)
        if cached is not None:
            return set(cached)
        problem_cls = problem_registry.get(config.problem_name)
        problem = problem_cls.create(config.problem_overrides)
        problem.reset(None)
        names = set(MetricManager.default_summary_metric_names(problem))
        names.update(OPTIONAL_PRECISION_METRIC_NAMES)
        names.update(
            metric_cls.name
            for metric_cls in MetricManager.default_trend_metric_classes(problem)
        )
        names.update(metric_cls.name for metric_cls in optional_trend_metric_classes(problem))
        names.add("T_r")
        self._metric_output_name_cache[cache_key] = set(names)
        return names
