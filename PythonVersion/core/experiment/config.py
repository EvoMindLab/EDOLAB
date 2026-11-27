"""Experiment configuration objects."""

from __future__ import annotations

import json
import os
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.experiment.seeds import SeedInput, parse_seed_text


_SYSTEM_KEYS = {
    "max_parallel_runs",
    "default_gpu_inference_concurrency",
    "output_dir",
    "default_runs",
    "default_problem",
    "default_problem_seed",
    "default_algorithm_seed",
    "default_metric_outputs",
    "default_problem_params",
    "default_algorithm_params",
    "event_sidecar",
}

_TASK_KEYS = {
    "label",
    "algorithm",
    "problem",
    "runs",
    "problem_seed",
    "algorithm_seed",
    "metric_outputs",
    "problem_params",
    "algorithm_params",
}


@dataclass(slots=True)
class ExperimentConfig:
    algorithm_names: list[str]
    problem_name: str = "GMPB"
    run_number: int = 1
    problem_seed: SeedInput = None
    algorithm_seed: SeedInput = None
    problem_overrides: dict[str, Any] = field(default_factory=dict)
    shared_algorithm_overrides: dict[str, Any] = field(default_factory=dict)
    algorithm_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    metric_output_names: list[str] | None = None
    max_parallel_runs: int = field(default_factory=lambda: os.cpu_count() or 1)
    output_dir: str | None = None
    label: str | None = None
    source_task_file: str | None = None
    source_line: int | None = None
    task_config_hash: str | None = None
    # Optional per-batch CUDA service concurrency.  ``None`` preserves the
    # existing environment-variable/default resolution in CudaPredictorPool.
    gpu_inference_concurrency: int | None = None


@dataclass(slots=True)
class BatchExperimentConfig:
    tasks: list[ExperimentConfig]
    max_parallel_runs: int = field(default_factory=lambda: os.cpu_count() or 1)
    output_dir: str | None = None
    # Applied to every task in this JSONL batch.  Keep optional for backwards
    # compatibility with task files created before GPU service tuning existed.
    gpu_inference_concurrency: int | None = None
    # Write run_events.jsonl next to summary.sqlite. The SQLite event tables are canonical and the sidecar can be
    # regenerated from them (core.result.event_streams.export_run_events), so large batches may skip it.
    event_sidecar: bool = True


def load_batch_task_file(path: str | Path) -> BatchExperimentConfig:
    records = _read_json_lines(Path(path))
    task_file = str(Path(path).resolve())
    if not records:
        raise ValueError("Batch task file must contain at least one task.")

    system: dict[str, Any] = {}
    task_records = records
    if _looks_like_system_record(records[0][1]):
        line_number, system = records[0]
        _validate_keys(system, _SYSTEM_KEYS, "system config", line_number)
        task_records = records[1:]

    if not task_records:
        raise ValueError("Batch task file must contain at least one task.")

    default_problem_params = _normalize_mapping(
        system.get("default_problem_params", {}),
        "default_problem_params",
    )
    default_algorithm_params = _normalize_mapping(
        system.get("default_algorithm_params", {}),
        "default_algorithm_params",
    )
    default_metric_outputs = _normalize_metric_output_names(
        system.get("default_metric_outputs")
    )
    max_parallel_runs = _normalize_positive_int(
        system.get("max_parallel_runs", os.cpu_count() or 1),
        "max_parallel_runs",
    )
    gpu_inference_concurrency = _normalize_optional_positive_int(
        system.get("default_gpu_inference_concurrency"),
        "default_gpu_inference_concurrency",
    )

    tasks: list[ExperimentConfig] = []
    labels: set[str] = set()
    for task_index, (line_number, record) in enumerate(task_records, start=1):
        _validate_keys(record, _TASK_KEYS, "task", line_number)
        algorithms = _normalize_algorithm_names(record.get("algorithm"), line_number)
        label = _normalize_label(record.get("label"), task_index)
        if label in labels:
            raise ValueError(f"Duplicate task label {label!r}.")
        labels.add(label)

        problem_params = dict(default_problem_params)
        problem_params.update(
            _normalize_mapping(record.get("problem_params", {}), "problem_params")
        )
        algorithm_params = dict(default_algorithm_params)
        algorithm_params.update(
            _normalize_mapping(record.get("algorithm_params", {}), "algorithm_params")
        )
        shared_algorithm_overrides, algorithm_overrides = split_algorithm_overrides(
            algorithm_params,
            algorithms,
        )

        task_config = {
            "label": label,
            "algorithm": algorithms,
            "problem": str(record.get("problem", system.get("default_problem", "GMPB"))),
            "runs": _normalize_positive_int(
                record.get("runs", system.get("default_runs", 1)),
                "runs",
            ),
            "problem_seed": _normalize_seed(
                record.get("problem_seed", system.get("default_problem_seed"))
            ),
            "algorithm_seed": _normalize_seed(
                record.get("algorithm_seed", system.get("default_algorithm_seed"))
            ),
            "problem_params": problem_params,
            "algorithm_params": algorithm_params,
            "metric_outputs": _normalize_metric_output_names(
                record.get("metric_outputs", default_metric_outputs)
            ),
            "gpu_inference_concurrency": gpu_inference_concurrency,
        }
        tasks.append(
            ExperimentConfig(
                label=label,
                algorithm_names=algorithms,
                problem_name=task_config["problem"],
                run_number=task_config["runs"],
                problem_seed=task_config["problem_seed"],
                algorithm_seed=task_config["algorithm_seed"],
                problem_overrides=problem_params,
                shared_algorithm_overrides=shared_algorithm_overrides,
                algorithm_overrides=algorithm_overrides,
                metric_output_names=task_config["metric_outputs"],
                max_parallel_runs=max_parallel_runs,
                gpu_inference_concurrency=gpu_inference_concurrency,
                output_dir=system.get("output_dir"),
                source_task_file=task_file,
                source_line=line_number,
                task_config_hash=_stable_hash(task_config),
            )
        )

    event_sidecar = system.get("event_sidecar", True)
    if type(event_sidecar) is not bool:
        raise ValueError("event_sidecar must be true or false.")
    return BatchExperimentConfig(
        tasks=tasks,
        max_parallel_runs=max_parallel_runs,
        gpu_inference_concurrency=gpu_inference_concurrency,
        output_dir=system.get("output_dir"),
        event_sidecar=event_sidecar,
    )


def split_algorithm_overrides(
    overrides: Mapping[str, Any],
    algorithms: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    shared: dict[str, Any] = {}
    specific: dict[str, dict[str, Any]] = {name: {} for name in algorithms}
    algorithm_set = set(algorithms)
    for key, value in overrides.items():
        key_text = str(key)
        if key_text in algorithm_set and isinstance(value, Mapping):
            specific[key_text].update(dict(value))
            continue
        if "." in key_text:
            prefix, parameter = key_text.split(".", 1)
            if prefix in algorithm_set:
                specific[prefix][parameter] = value
                continue
        shared[key_text] = value
    return shared, specific


def _read_json_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON object on line {line_number}: {exc.msg}."
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}.")
            records.append((line_number, value))
    return records


def _looks_like_system_record(record: dict[str, Any]) -> bool:
    return "algorithm" not in record and bool(set(record) & _SYSTEM_KEYS)


def _validate_keys(
    record: dict[str, Any],
    allowed: set[str],
    record_name: str,
    line_number: int,
) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(
            f"Unknown {record_name} key(s) on line {line_number}: {names}."
        )


def _normalize_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object.")
    return dict(value)


def _normalize_algorithm_names(value: Any, line_number: int) -> list[str]:
    if isinstance(value, str):
        names = [value.strip()]
    elif isinstance(value, list):
        names = [str(item).strip() for item in value]
    else:
        raise ValueError(f"Task on line {line_number} must include an algorithm.")
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"Task on line {line_number} must include an algorithm.")
    return names


def _normalize_label(value: Any, task_index: int) -> str:
    if value is None:
        return f"task-{task_index}"
    label = str(value).strip()
    return label or f"task-{task_index}"


def _normalize_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return number


def _normalize_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _normalize_positive_int(value, field_name)


def _normalize_seed(value: Any) -> SeedInput:
    if value is None:
        return None
    if isinstance(value, str):
        return parse_seed_text(value)
    if isinstance(value, bool):
        raise ValueError(
            "Seed values must be integers, integer lists, strings, or null."
        )
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [int(item) for item in value]
    raise ValueError("Seed values must be integers, integer lists, strings, or null.")


def _normalize_metric_output_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("metric_outputs must be a string, a string list, or null.")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
