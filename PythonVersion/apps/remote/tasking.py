"""Read-only validation, budget analysis, and provenance for batch task files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

import instances  # noqa: F401 - registry population is part of task validation

from apps.remote.domain import stable_json_dumps, utc_now
from core.experiment.config import BatchExperimentConfig, ExperimentConfig, load_batch_task_file
from core.registry import algorithm_registry, problem_registry


DEFAULT_LARGE_TOTAL_EVALS = 10_000_000
DEFAULT_FORMAL_RUN_COUNT = 31
MAX_INLINE_TASK_BYTES = 2 * 1024 * 1024
_TASK_PROVENANCE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "generated_at",
        "local_task_file",
        "provenance_sha256",
        "task_file_name",
    }
)
_TASK_PROVENANCE_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "task_file_sha256",
        "effective_max_parallel_runs",
        "tasks",
    }
)


@dataclass(frozen=True, slots=True)
class TaskWarning:
    code: str
    message: str
    task_label: str | None = None
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskBudget:
    label: str
    algorithms: tuple[str, ...]
    problem: str
    runs_per_algorithm: int
    max_evals_per_run: int
    run_count: int
    total_evals: int
    source_line: int | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["algorithms"] = list(self.algorithms)
        return value


@dataclass(frozen=True, slots=True)
class BatchTaskInspection:
    task_file: Path
    task_file_sha256: str
    configured_max_parallel_runs: int
    total_runs: int
    total_evals: int
    gpu_requested: bool
    tasks: tuple[TaskBudget, ...]
    warnings: tuple[TaskWarning, ...]
    task_bytes: bytes = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_file": str(self.task_file),
            "task_file_sha256": self.task_file_sha256,
            "configured_max_parallel_runs": self.configured_max_parallel_runs,
            "total_runs": self.total_runs,
            "total_evals": self.total_evals,
            "gpu_requested": self.gpu_requested,
            "tasks": [item.to_dict() for item in self.tasks],
            "warnings": [item.to_dict() for item in self.warnings],
        }


def load_and_inspect_batch_task(
    task_file: str | Path,
    *,
    large_total_evals: int = DEFAULT_LARGE_TOTAL_EVALS,
    formal_run_count: int = DEFAULT_FORMAL_RUN_COUNT,
) -> tuple[BatchExperimentConfig, BatchTaskInspection]:
    """Parse with the canonical loader and compute budgets without running a FE."""

    supplied_path = Path(task_file).expanduser()
    if supplied_path.is_symlink():
        raise ValueError(f"batch task file must not be a symlink: {supplied_path}")
    path = supplied_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"batch task file does not exist: {path}")
    if large_total_evals <= 0:
        raise ValueError("large_total_evals must be positive")
    if formal_run_count <= 0:
        raise ValueError("formal_run_count must be positive")

    task_bytes = _read_stable_task_bytes(path)
    config = _load_captured_task_bytes(path, task_bytes)
    budgets: list[TaskBudget] = []
    warnings: list[TaskWarning] = []
    gpu_labels: list[str] = []
    total_runs = 0
    total_evals = 0

    for task in config.tasks:
        _validate_registry_names(task)
        problem_type = problem_registry.get(task.problem_name)
        problem = problem_type.create(dict(task.problem_overrides))
        max_evals = int(problem.max_evals)
        if max_evals <= 0:
            raise ValueError(
                f"task {task.label or '<unnamed>'} has non-positive max_evals: {max_evals}"
            )
        run_count = len(task.algorithm_names) * int(task.run_number)
        task_total_evals = run_count * max_evals
        label = task.label or "task"
        budgets.append(
            TaskBudget(
                label=label,
                algorithms=tuple(task.algorithm_names),
                problem=task.problem_name,
                runs_per_algorithm=int(task.run_number),
                max_evals_per_run=max_evals,
                run_count=run_count,
                total_evals=task_total_evals,
                source_line=task.source_line,
            )
        )
        total_runs += run_count
        total_evals += task_total_evals

        if task.run_number >= formal_run_count:
            warnings.append(
                TaskWarning(
                    code="formal_run_count",
                    task_label=label,
                    message=(
                        f"{label} requests {task.run_number} runs per algorithm; "
                        "treat this as a formal experiment, not a smoke test."
                    ),
                    details={
                        "runs_per_algorithm": int(task.run_number),
                        "threshold": int(formal_run_count),
                    },
                )
            )
        if _task_requests_gpu(task):
            gpu_labels.append(label)

    if gpu_labels:
        warnings.append(
            TaskWarning(
                code="gpu_requested",
                message="One or more tasks request a GPU-backed device.",
                details={"task_labels": gpu_labels},
            )
        )
    if total_evals >= large_total_evals:
        warnings.append(
            TaskWarning(
                code="large_fe_budget",
                message=(
                    f"The batch requests {total_evals:,} fitness evaluations; "
                    "explicitly confirm server capacity before dispatch."
                ),
                details={
                    "total_evals": total_evals,
                    "threshold": int(large_total_evals),
                },
            )
        )

    inspection = BatchTaskInspection(
        task_file=path,
        task_file_sha256=hashlib.sha256(task_bytes).hexdigest(),
        configured_max_parallel_runs=int(config.max_parallel_runs),
        total_runs=total_runs,
        total_evals=total_evals,
        gpu_requested=bool(gpu_labels),
        tasks=tuple(budgets),
        warnings=tuple(warnings),
        task_bytes=task_bytes,
    )
    return config, inspection


def inspect_batch_task(
    task_file: str | Path,
    *,
    large_total_evals: int = DEFAULT_LARGE_TOTAL_EVALS,
    formal_run_count: int = DEFAULT_FORMAL_RUN_COUNT,
) -> BatchTaskInspection:
    return load_and_inspect_batch_task(
        task_file,
        large_total_evals=large_total_evals,
        formal_run_count=formal_run_count,
    )[1]


def inspect_batch_task_bytes(
    payload: bytes,
    *,
    task_name: str | Path = "inline-task.jsonl",
    large_total_evals: int = DEFAULT_LARGE_TOTAL_EVALS,
    formal_run_count: int = DEFAULT_FORMAL_RUN_COUNT,
) -> BatchTaskInspection:
    """Inspect immutable JSONL bytes using the canonical batch parser.

    Relay uses this for task text pasted into the local UI.  The supplied
    bytes are parsed from a private temporary regular file because the core
    batch loader is intentionally path based, then the inspection is rebound
    to a stable virtual name for provenance.  No pasted task is written into
    the repository ``tasks/`` directory.
    """

    if not isinstance(payload, bytes):
        raise TypeError("inline batch task payload must be bytes")
    if not payload.strip():
        raise ValueError("Batch task text must contain at least one task.")
    if len(payload) > MAX_INLINE_TASK_BYTES:
        raise ValueError(
            f"Inline batch task exceeds the {MAX_INLINE_TASK_BYTES}-byte UTF-8 limit."
        )
    virtual_path = Path(task_name)
    if virtual_path.suffix.lower() != ".jsonl":
        raise ValueError("inline batch task name must use .jsonl")
    with tempfile.TemporaryDirectory(prefix="edolab-relay-inline-task-") as temporary:
        captured = Path(temporary) / "task.jsonl"
        captured.write_bytes(payload)
        inspection = load_and_inspect_batch_task(
            captured,
            large_total_evals=large_total_evals,
            formal_run_count=formal_run_count,
        )[1]
    return replace(
        inspection,
        task_file=virtual_path,
        task_bytes=payload,
    )


def materialize_inline_task(
    payload: bytes,
    destination: str | Path,
    *,
    expected_sha256: str,
) -> Path:
    """Atomically write SHA-pinned inline task bytes to a private run path."""

    if not isinstance(payload, bytes):
        raise TypeError("inline batch task payload must be bytes")
    if len(payload) > MAX_INLINE_TASK_BYTES:
        raise ValueError(
            f"Inline batch task exceeds the {MAX_INLINE_TASK_BYTES}-byte UTF-8 limit."
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Inline batch task bytes do not match the frozen SHA-256.")
    target = Path(destination)
    if target.suffix.lower() != ".jsonl":
        raise ValueError("inline batch task destination must use .jsonl")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("inline batch task destination parent is not a safe directory")
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise ValueError("inline batch task destination is not a regular file")
        existing = _read_stable_task_bytes(target)
        if hashlib.sha256(existing).hexdigest() != expected_sha256 or existing != payload:
            raise ValueError("existing inline batch task does not match frozen bytes")
        target.chmod(0o600)
        return target

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".inline-task-",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() or target.is_symlink():
            raise ValueError("inline batch task destination appeared during capture")
        os.replace(temporary, target)
        target.chmod(0o600)
        captured = _read_stable_task_bytes(target)
        if captured != payload or hashlib.sha256(captured).hexdigest() != expected_sha256:
            raise ValueError("materialized inline batch task failed SHA-256 verification")
        return target
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def build_task_provenance(
    task_file: str | Path,
    *,
    forced_max_parallel_runs: int | None = None,
    inspection: BatchTaskInspection | None = None,
) -> dict[str, Any]:
    """Return a canonical, hash-addressed record of the exact parsed task."""

    path = Path(task_file).expanduser().resolve()
    if inspection is None:
        config, current = load_and_inspect_batch_task(path)
    else:
        if inspection.task_file.resolve() != path:
            raise ValueError("task inspection belongs to a different task file")
        digest = hashlib.sha256(inspection.task_bytes).hexdigest()
        if digest != inspection.task_file_sha256:
            raise ValueError("captured task bytes do not match the inspection digest")
        config = _load_captured_task_bytes(path, inspection.task_bytes)
        current = _inspect_captured_task_bytes(inspection.task_bytes)
    if forced_max_parallel_runs is not None and forced_max_parallel_runs <= 0:
        raise ValueError("forced_max_parallel_runs must be positive")

    normalized_tasks = [_normalize_experiment_config(item) for item in config.tasks]
    canonical = {
        "schema_version": 1,
        "task_file_name": path.name,
        "task_file_sha256": current.task_file_sha256,
        "configured_max_parallel_runs": int(config.max_parallel_runs),
        "configured_gpu_inference_concurrency": (
            int(config.gpu_inference_concurrency)
            if config.gpu_inference_concurrency is not None
            else None
        ),
        "effective_max_parallel_runs": int(
            forced_max_parallel_runs
            if forced_max_parallel_runs is not None
            else config.max_parallel_runs
        ),
        "configured_output_dir": config.output_dir,
        "tasks": normalized_tasks,
        "budget": {
            "total_runs": current.total_runs,
            "total_evals": current.total_evals,
            "gpu_requested": current.gpu_requested,
            "items": [item.to_dict() for item in current.tasks],
        },
        "warnings": [item.to_dict() for item in current.warnings],
    }
    provenance_sha256 = task_provenance_identity(canonical)
    return {
        **canonical,
        "provenance_sha256": provenance_sha256,
        "generated_at": utc_now(),
        # Diagnostic only; excluded from the canonical hash so provenance is
        # stable across workstations with different absolute checkout paths.
        "local_task_file": str(path),
    }


def task_provenance_identity(provenance: Mapping[str, Any]) -> str:
    """Hash only the execution semantics of a task provenance record.

    A task is uploaded under a fixed remote name (``task.jsonl``), so the
    local display filename and absolute path cannot be part of its scientific
    identity.  The stored ``provenance_sha256`` is also excluded because it is
    a derived value.  Recomputing from the actual fields makes comparison work
    for both new records and legacy records whose old hashes included a
    different local/remote filename, while any real task/configuration change
    still changes this identity.
    """

    if not isinstance(provenance, Mapping):
        raise TypeError("task provenance must be a JSON object")
    missing = sorted(_TASK_PROVENANCE_REQUIRED_FIELDS - set(provenance))
    if missing:
        raise ValueError(
            "task provenance is missing required fields: " + ", ".join(missing)
        )
    semantic = {
        key: value
        for key, value in provenance.items()
        if key not in _TASK_PROVENANCE_DIAGNOSTIC_FIELDS
    }
    return hashlib.sha256(stable_json_dumps(semantic).encode("utf-8")).hexdigest()


def _normalize_experiment_config(task: ExperimentConfig) -> dict[str, Any]:
    return {
        "label": task.label,
        "algorithm_names": list(task.algorithm_names),
        "problem_name": task.problem_name,
        "run_number": int(task.run_number),
        "problem_seed": task.problem_seed,
        "algorithm_seed": task.algorithm_seed,
        "problem_overrides": dict(sorted(task.problem_overrides.items())),
        "shared_algorithm_overrides": dict(
            sorted(task.shared_algorithm_overrides.items())
        ),
        "algorithm_overrides": {
            name: dict(sorted(values.items()))
            for name, values in sorted(task.algorithm_overrides.items())
        },
        "metric_output_names": (
            list(task.metric_output_names) if task.metric_output_names is not None else None
        ),
        "gpu_inference_concurrency": (
            int(task.gpu_inference_concurrency)
            if task.gpu_inference_concurrency is not None
            else None
        ),
        "source_line": task.source_line,
        "task_config_hash": task.task_config_hash,
    }


def _validate_registry_names(task: ExperimentConfig) -> None:
    problem_registry.get(task.problem_name)
    for algorithm_name in task.algorithm_names:
        algorithm_type = algorithm_registry.get(algorithm_name)
        parameter_names = set(algorithm_type.parameter_specs())
        shared_unknown = sorted(
            key for key in task.shared_algorithm_overrides if key not in parameter_names
        )
        specific_unknown = sorted(
            key
            for key in task.algorithm_overrides.get(algorithm_name, {})
            if key not in parameter_names
        )
        if shared_unknown:
            raise ValueError(
                f"shared algorithm parameters are invalid for {algorithm_name}: "
                + ", ".join(shared_unknown)
            )
        if specific_unknown:
            raise ValueError(
                f"algorithm parameters are invalid for {algorithm_name}: "
                + ", ".join(specific_unknown)
            )


def _task_requests_gpu(task: ExperimentConfig) -> bool:
    values: list[tuple[str, Any]] = list(task.problem_overrides.items())
    values.extend(task.shared_algorithm_overrides.items())
    for algorithm_values in task.algorithm_overrides.values():
        values.extend(algorithm_values.items())
    return any(_value_requests_gpu(key, value) for key, value in values)


def _value_requests_gpu(key: str, value: Any) -> bool:
    key_text = str(key).lower()
    if isinstance(value, Mapping):
        return any(_value_requests_gpu(str(k), v) for k, v in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_value_requests_gpu(key, item) for item in value)
    text = str(value).strip().lower()
    if text in {"cuda", "gpu", "mps"} or text.startswith("cuda:"):
        return True
    return "gpu" in key_text and text not in {"", "0", "false", "none", "cpu", "numpy"}


def _read_stable_task_bytes(path: Path) -> bytes:
    """Capture one regular task file without mixing bytes from concurrent saves."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"could not safely open batch task file: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"batch task file is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        current = path.lstat()
    except OSError as exc:
        raise ValueError("batch task file changed while it was being captured") from exc
    if not stat.S_ISREG(current.st_mode) or _file_identity(opened) != _file_identity(
        finished
    ) or _file_identity(finished) != _file_identity(current):
        raise ValueError("batch task file changed while it was being captured")
    return payload


def _load_captured_task_bytes(path: Path, payload: bytes) -> BatchExperimentConfig:
    """Parse the exact bytes whose digest is recorded in run provenance."""

    with tempfile.TemporaryDirectory(prefix="edolab-relay-task-") as temporary:
        captured = Path(temporary) / "task.jsonl"
        captured.write_bytes(payload)
        config = load_batch_task_file(captured)
    source_path = str(path)
    for task in config.tasks:
        task.source_task_file = source_path
    return config


def _inspect_captured_task_bytes(payload: bytes) -> BatchTaskInspection:
    """Apply canonical warning thresholds to already captured immutable bytes."""

    with tempfile.TemporaryDirectory(prefix="edolab-relay-task-inspect-") as temporary:
        captured = Path(temporary) / "task.jsonl"
        captured.write_bytes(payload)
        return load_and_inspect_batch_task(captured)[1]


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


__all__ = [
    "BatchTaskInspection",
    "DEFAULT_FORMAL_RUN_COUNT",
    "DEFAULT_LARGE_TOTAL_EVALS",
    "MAX_INLINE_TASK_BYTES",
    "TaskBudget",
    "TaskWarning",
    "build_task_provenance",
    "inspect_batch_task",
    "inspect_batch_task_bytes",
    "load_and_inspect_batch_task",
    "materialize_inline_task",
    "task_provenance_identity",
]
