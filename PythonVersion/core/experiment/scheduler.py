"""Run-level scheduler."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from collections.abc import Iterator
from contextlib import ExitStack
import multiprocessing
from typing import Any, Callable

from core.evaluation.result import RunResult


@dataclass(frozen=True, slots=True)
class RunTask:
    algorithm_name: str
    problem_name: str
    run_index: int
    problem_seed: int
    algorithm_seed: int
    problem_overrides: dict[str, Any]
    algorithm_overrides: dict[str, Any]
    metric_output_names: list[str] | None
    label: str | None = None
    execution_device: int | None = None
    prediction_device_group: tuple[int, ...] | None = None


ProgressCallback = Callable[[RunResult, int, int], None]
CancelCheck = Callable[[], bool]


class RunSchedulerCanceled(RuntimeError):
    """Raised when a caller requests cancellation between run tasks."""


def execute_run_task(task: RunTask) -> RunResult:
    from core.compute import configure_worker_thread_limits

    configure_worker_thread_limits()
    import instances  # noqa: F401
    from core.registry import algorithm_registry

    algorithm_cls = algorithm_registry.get(task.algorithm_name)
    algorithm = algorithm_cls()
    result = algorithm.run_once(
        problem_name=task.problem_name,
        problem_overrides=task.problem_overrides,
        algorithm_overrides=task.algorithm_overrides,
        problem_seed=task.problem_seed,
        algorithm_seed=task.algorithm_seed,
        run_index=task.run_index,
        metric_names=task.metric_output_names,
    )
    result.label = task.label
    return result


class GlobalRunScheduler:
    def __init__(
        self,
        max_parallel_runs: int,
        *,
        gpu_inference_concurrency: int | None = None,
    ):
        if max_parallel_runs <= 0:
            raise ValueError("max_parallel_runs must be positive.")
        if gpu_inference_concurrency is not None and gpu_inference_concurrency <= 0:
            raise ValueError("gpu_inference_concurrency must be positive.")
        self.max_parallel_runs = max_parallel_runs
        self.gpu_inference_concurrency = gpu_inference_concurrency

    def run(
        self, tasks: list[RunTask], progress: ProgressCallback | None = None
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for result, completed, total in self.iter_run(tasks):
            results.append(result)
            if progress is not None:
                progress(result, completed, total)
        return results

    def iter_run(
        self,
        tasks: list[RunTask],
        cancel_check: CancelCheck | None = None,
    ) -> Iterator[tuple[RunResult, int, int]]:
        total = len(tasks)
        if total == 0:
            return
        has_cuda_tasks = any(_task_requests_cuda(task) for task in tasks)
        if self.max_parallel_runs == 1 and not has_cuda_tasks:
            for completed, task in enumerate(tasks, start=1):
                if cancel_check is not None and cancel_check():
                    raise RunSchedulerCanceled("Run scheduler canceled.")
                result = execute_run_task(task)
                yield result, completed, total
            return

        scheduled_tasks, mp_context = _prepare_process_tasks(tasks)
        with ExitStack() as stack:
            predictor_pool = None
            initializer = None
            initargs: tuple[Any, ...] = ()
            executor = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=self.max_parallel_runs,
                    mp_context=mp_context,
                    initializer=initializer,
                    initargs=initargs,
                )
            )
            task_iter = iter(scheduled_tasks)
            pending = set()
            completed = 0

            def submit_next() -> bool:
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                pending.add(executor.submit(execute_run_task, task))
                return True

            for _ in range(min(self.max_parallel_runs, total)):
                if cancel_check is not None and cancel_check():
                    raise RunSchedulerCanceled("Run scheduler canceled.")
                submit_next()

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    completed += 1
                    yield result, completed, total
                    if cancel_check is not None and cancel_check():
                        for item in pending:
                            item.cancel()
                        raise RunSchedulerCanceled("Run scheduler canceled.")
                    while len(pending) < self.max_parallel_runs:
                        if cancel_check is not None and cancel_check():
                            for item in pending:
                                item.cancel()
                            raise RunSchedulerCanceled("Run scheduler canceled.")
                        if not submit_next():
                            break


def _prepare_process_tasks(
    tasks: list[RunTask],
) -> tuple[list[RunTask], multiprocessing.context.BaseContext | None]:
    """Prepare CUDA tasks for a process pool without forking CUDA state.

    Linux defaults to ``fork``.  Forking after the parent has imported
    PyTorch and then asking every child to initialize CUDA can leave the
    children blocked in the CUDA/runtime loader before the first run.  CUDA
    runs therefore use a fresh ``spawn`` context and are assigned to the
    available devices round-robin.  CPU-only batches retain the historical
    fork path and its lower startup overhead.
    """

    cuda_indices = [
        index for index, task in enumerate(tasks) if _task_requests_cuda(task)
    ]
    if not cuda_indices:
        return tasks, None

    device_count = _cuda_device_count()
    if device_count <= 0:
        # Let the child-side device selector produce the normal, descriptive
        # ``PredictionUnavailable`` result when CUDA is requested but absent.
        return tasks, multiprocessing.get_context("spawn")

    prepared = list(tasks)
    for ordinal, task_index in enumerate(cuda_indices):
        task = prepared[task_index]
        requested_policy = task.algorithm_overrides.get("PredictionDevice", "")
        requested = str(requested_policy)
        # An explicit device is a queue selection, not a hint to be replaced
        # by round-robin assignment.  Only the generic ``cuda`` policy is
        # assigned by the scheduler's GPU queue rotation.
        if requested.strip().lower().startswith("cuda:"):
            try:
                explicit_index = int(requested.strip().split(":", 1)[1])
            except (TypeError, ValueError):
                explicit_index = 0
            if 0 <= explicit_index < device_count:
                prepared[task_index] = _with_execution_device(task, explicit_index, (explicit_index,))
                continue
            # Keep the task runnable when a queue id is stale after a device
            # inventory change; the child receives a valid concrete queue id.
            overrides = dict(task.algorithm_overrides)
            overrides["PredictionDevice"] = "cuda:0"
            prepared[task_index] = RunTask(
                algorithm_name=task.algorithm_name,
                problem_name=task.problem_name,
                run_index=task.run_index,
                problem_seed=task.problem_seed,
                algorithm_seed=task.algorithm_seed,
                problem_overrides=task.problem_overrides,
                algorithm_overrides=overrides,
                metric_output_names=task.metric_output_names,
                label=task.label,
                execution_device=0,
                prediction_device_group=(0,),
            )
            continue
        group = _requested_cuda_group(requested_policy, device_count)
        if group:
            selected = group[ordinal % len(group)]
            # Preserve the full allow-list for predictor fallback, but put
            # the queue-selected device first so this inference starts on the
            # current task queue rather than silently resetting to group[0].
            reordered = [selected, *[item for item in group if item != selected]]
            overrides = dict(task.algorithm_overrides)
            overrides["PredictionDevice"] = f"cuda:{selected}"
            prepared[task_index] = _with_execution_device(
                RunTask(
                    algorithm_name=task.algorithm_name,
                    problem_name=task.problem_name,
                    run_index=task.run_index,
                    problem_seed=task.problem_seed,
                    algorithm_seed=task.algorithm_seed,
                    problem_overrides=task.problem_overrides,
                    algorithm_overrides=overrides,
                    metric_output_names=task.metric_output_names,
                    label=task.label,
                    prediction_device_group=tuple(group),
                ),
                selected,
                tuple(group),
            )
            continue
        overrides = dict(task.algorithm_overrides)
        overrides["PredictionDevice"] = f"cuda:{ordinal % device_count}"
        prepared[task_index] = RunTask(
            algorithm_name=task.algorithm_name,
            problem_name=task.problem_name,
            run_index=task.run_index,
            problem_seed=task.problem_seed,
            algorithm_seed=task.algorithm_seed,
            problem_overrides=task.problem_overrides,
            algorithm_overrides=overrides,
            metric_output_names=task.metric_output_names,
            label=task.label,
            execution_device=ordinal % device_count,
            prediction_device_group=tuple(range(device_count)),
        )
    return prepared, multiprocessing.get_context("spawn")


def _build_run_executors(
    tasks: list[RunTask],
    *,
    max_parallel_runs: int,
    stack: ExitStack,
    default_context: multiprocessing.context.BaseContext | None,
) -> dict[int | str, ProcessPoolExecutor]:
    """Compatibility helper returning one run-level executor.

    CUDA predictor services are now owned by :meth:`GlobalRunScheduler.iter_run`;
    this helper intentionally never creates per-GPU run executors.
    """
    return {
        "cpu": stack.enter_context(
            ProcessPoolExecutor(
                max_workers=max_parallel_runs,
                mp_context=default_context,
            )
        )
    }


def _executor_key(task: RunTask) -> int | str:
    return "cpu"


def _task_device_index(task: RunTask, device_count: int) -> int | None:
    if task.execution_device is not None:
        return max(0, min(device_count - 1, int(task.execution_device)))
    value = str(task.algorithm_overrides.get("PredictionDevice", "")).strip().lower()
    if not value.startswith("cuda:"):
        return None
    try:
        index = int(value.split(":", 1)[1])
    except (TypeError, ValueError):
        return None
    if 0 <= index < device_count:
        return index
    return 0


def _requested_cuda_group(value: Any, device_count: int) -> list[int]:
    """Return valid GPU ids from an explicit task device group."""

    if not isinstance(value, (list, tuple)):
        return []
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        text = str(item).strip().lower()
        if text.startswith("cuda:"):
            text = text.split(":", 1)[1]
        if not text.isdigit():
            continue
        index = int(text)
        if 0 <= index < device_count and index not in result:
            result.append(index)
    return result


def _with_execution_device(
    task: RunTask,
    device: int,
    prediction_device_group: tuple[int, ...] | None = None,
) -> RunTask:
    return RunTask(
        algorithm_name=task.algorithm_name,
        problem_name=task.problem_name,
        run_index=task.run_index,
        problem_seed=task.problem_seed,
        algorithm_seed=task.algorithm_seed,
        problem_overrides=task.problem_overrides,
        algorithm_overrides=task.algorithm_overrides,
        metric_output_names=task.metric_output_names,
        label=task.label,
        execution_device=int(device),
        prediction_device_group=(
            task.prediction_device_group
            if prediction_device_group is None
            else tuple(prediction_device_group)
        ),
    )


def _task_requests_cuda(task: RunTask) -> bool:
    value = task.algorithm_overrides.get("PredictionDevice", "")
    if isinstance(value, (list, tuple)):
        return bool(_requested_cuda_group(value, max(1, _cuda_device_count())))
    normalized = str(value).strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def _task_requests_cuda_evaluation(task: RunTask) -> bool:
    """Whether the problem evaluation explicitly requests a CUDA device."""

    value = task.problem_overrides.get("EvaluationDevice", "")
    normalized = str(value).strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def _cuda_device_count() -> int:
    try:
        import torch

        return max(0, int(torch.cuda.device_count()))
    except Exception:
        return 0
