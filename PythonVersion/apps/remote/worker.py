"""Remote, non-interactive EDOLAB batch worker.

The controller invokes this module inside an extracted immutable snapshot::

    python -m apps.remote.worker --task-file task.jsonl --output-dir results/run-id

Every stdout protocol message is a single ``EDOLAB_EVENT <json>`` line and is
flushed immediately.  A run-completed event is emitted only after
``ExperimentRunner`` has committed that complete run to ``summary.sqlite``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence

from apps.remote.domain import stable_json_dumps, utc_now
from core.compute import configure_worker_thread_limits
from core.evaluation.result import RunResult


# Keep these names in sync with ``core.result.store`` without importing the
# numerical result-storage stack before the worker has configured its thread
# pools.  The remote worker is forked on Linux; importing NumPy/PyTorch first
# lets the parent process initialize large OpenMP pools that every child then
# inherits, defeating the one-thread-per-run safety limit.
SUMMARY_DB_NAME = "summary.sqlite"
TREND_DB_NAME = "trend.sqlite"


EVENT_PREFIX = "EDOLAB_EVENT "
WORKER_STATUS_NAME = "worker-status.json"
TASK_PROVENANCE_NAME = "task-provenance.json"


class WorkerExitCode(IntEnum):
    OK = 0
    USAGE = 2
    TASK_INVALID = 10
    OUTPUT_INVALID = 11
    EXECUTION_FAILED = 20
    INTEGRITY_FAILED = 21
    INTERRUPTED = 130


class WorkerError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: WorkerExitCode) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    exit_code: WorkerExitCode
    status: Mapping[str, Any]


EventSink = Callable[[Mapping[str, Any]], None]


def __getattr__(name: str):
    """Lazily expose compatibility imports used by tests and integrations.

    ``ExperimentRunner`` must not be imported while this module is imported:
    the remote launcher calls :func:`configure_worker_thread_limits` before it
    loads the registry/model stack so forked run workers inherit one-thread
    pools.  A module-level lazy attribute preserves the historical
    ``apps.remote.worker.ExperimentRunner`` patch point without paying that
    import cost on the production path.
    """

    if name == "ExperimentRunner":
        from core.experiment.runner import ExperimentRunner

        return ExperimentRunner
    raise AttributeError(name)


def emit_event(payload: Mapping[str, Any]) -> None:
    """Write one stable, line-delimited worker protocol event."""

    print(EVENT_PREFIX + stable_json_dumps(payload), flush=True)


def run_worker(
    *,
    task_file: str | Path,
    output_dir: str | Path,
    max_parallel_runs: int = 1,
    event_sink: EventSink = emit_event,
) -> WorkerOutcome:
    # This must happen before importing ``apps.remote.tasking`` or
    # ``core.experiment.runner``.  Both populate the registry and import model,
    # NumPy, and OpenMP-backed libraries.  On a fork-based worker pool, setting
    # the environment only inside ``execute_run_task`` is too late: the parent
    # has already initialized (for example) a 128-thread libgomp pool and each
    # child inherits it.  Configure the parent first so children inherit a
    # genuinely one-thread runtime, while the per-task guard remains in place.
    configure_worker_thread_limits()
    from apps.remote.tasking import build_task_provenance, load_and_inspect_batch_task

    # Respect a compatibility/test override installed at the historical
    # module-level patch point.  In normal execution this stays absent and the
    # runner is imported only after the thread limits have been configured.
    runner_class = globals().get("ExperimentRunner")
    if runner_class is None:
        from core.experiment.runner import ExperimentRunner as runner_class

    if max_parallel_runs <= 0:
        raise WorkerError(
            "invalid_parallelism",
            "max_parallel_runs must be positive",
            WorkerExitCode.USAGE,
        )
    task_path = Path(task_file).expanduser().resolve()
    requested_output = Path(output_dir).expanduser()
    if requested_output.is_symlink():
        error = WorkerError(
            "output_invalid",
            f"output directory must not be a symlink: {requested_output}",
            WorkerExitCode.OUTPUT_INVALID,
        )
        status = _failure_status(error, requested_output.absolute())
        event_sink(status)
        return WorkerOutcome(error.exit_code, status)
    output_path = requested_output.resolve()

    try:
        _prepare_output_directory(output_path)
    except Exception as exc:
        error = WorkerError("output_invalid", str(exc), WorkerExitCode.OUTPUT_INVALID)
        status = _failure_status(error, output_path)
        event_sink(status)
        return WorkerOutcome(error.exit_code, status)

    try:
        config, inspection = load_and_inspect_batch_task(task_path)
        config.max_parallel_runs = int(max_parallel_runs)
        for task in config.tasks:
            task.max_parallel_runs = int(max_parallel_runs)
        provenance = build_task_provenance(
            task_path,
            forced_max_parallel_runs=max_parallel_runs,
            inspection=inspection,
        )
        _atomic_write_json(output_path / TASK_PROVENANCE_NAME, provenance)
    except Exception as exc:
        error = WorkerError("task_invalid", str(exc), WorkerExitCode.TASK_INVALID)
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return WorkerOutcome(error.exit_code, status)

    event_sink(
        {
            "event": "worker_started",
            "status": "running",
            "pid": os.getpid(),
            "task_file": task_path.name,
            "task_file_sha256": inspection.task_file_sha256,
            "output_dir": str(output_path),
            "max_parallel_runs": int(max_parallel_runs),
            "total_runs": inspection.total_runs,
            "total_evals": inspection.total_evals,
            "warnings": [warning.to_dict() for warning in inspection.warnings],
            "at": utc_now(),
        }
    )

    def progress(result: RunResult, completed: int, total: int) -> None:
        event_sink(
            {
                "event": "run_completed",
                "status": "running",
                "completed_runs": int(completed),
                "total_runs": int(total),
                "label": result.label,
                "algorithm": result.algorithm_name,
                "problem": result.problem_name,
                "run_index": int(result.run_index),
                "problem_seed": int(result.problem_seed),
                "algorithm_seed": int(result.algorithm_seed),
                "fe": int(result.fe),
                "runtime_seconds": float(result.runtime_seconds),
                "metrics": {
                    key: _finite_number_or_none(value)
                    for key, value in result.metrics.items()
                },
                "at": utc_now(),
            }
        )

    try:
        runner_class().run_batch_to_store(
            config,
            output_path,
            progress=progress,
        )
        integrity = verify_result_bundle(output_path)
        status = {
            "event": "worker_completed",
            "status": "completed",
            "exit_code": int(WorkerExitCode.OK),
            "output_dir": str(output_path),
            "integrity": integrity,
            "completed_at": utc_now(),
        }
        _atomic_write_json(output_path / WORKER_STATUS_NAME, status)
        event_sink(status)
        return WorkerOutcome(WorkerExitCode.OK, status)
    except KeyboardInterrupt as exc:
        error = WorkerError(
            "interrupted", "worker interrupted", WorkerExitCode.INTERRUPTED
        )
        _mark_bundle_failed(output_path, str(error))
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return WorkerOutcome(error.exit_code, status)
    except BundleIntegrityError as exc:
        error = WorkerError(
            "integrity_failed", str(exc), WorkerExitCode.INTEGRITY_FAILED
        )
        _mark_bundle_failed(output_path, str(error))
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return WorkerOutcome(error.exit_code, status)
    except Exception as exc:
        error = WorkerError(
            "execution_failed", str(exc), WorkerExitCode.EXECUTION_FAILED
        )
        _mark_bundle_failed(output_path, str(error))
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return WorkerOutcome(error.exit_code, status)


class BundleIntegrityError(RuntimeError):
    """A completed raw-result bundle did not pass read-only SQLite checks."""


def verify_result_bundle(output_dir: str | Path) -> dict[str, Any]:
    """Apply the core raw-run validator and preserve the worker status schema."""

    from core.result.validation import (
        ResultBundleValidationError,
        validate_result_bundle as validate_raw_run_bundle,
    )

    root = Path(output_dir).resolve()
    summary_path = root / SUMMARY_DB_NAME
    trend_path = root / TREND_DB_NAME
    try:
        validation = validate_raw_run_bundle(root, require_completed=True)
    except (ResultBundleValidationError, sqlite3.DatabaseError, OSError) as exc:
        raise BundleIntegrityError(str(exc)) from exc

    return {
        "ok": True,
        **validation.as_dict(),
        # These status-count keys predate the shared core validator and remain
        # in the worker protocol for compatibility.  A completed validation
        # guarantees that every task and runset represented here is complete.
        "tasks": {"completed": validation.task_count},
        "runsets": {"completed": validation.runset_count},
        "summary": _database_descriptor(summary_path, "ok"),
        "trend": _database_descriptor(trend_path, "ok"),
    }


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"output path is not a directory: {path}")
    entries = list(path.iterdir())
    if entries:
        names = ", ".join(sorted(item.name for item in entries[:5]))
        raise ValueError(
            "exact output directory must be empty to avoid overwriting a previous run"
            + (f": {names}" if names else "")
        )


def _database_descriptor(path: Path, quick_check: str) -> dict[str, Any]:
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "quick_check": quick_check,
    }


def _mark_bundle_failed(output_dir: Path, message: str) -> None:
    """Best-effort recovery that never creates a missing result database."""

    summary_path = output_dir / SUMMARY_DB_NAME
    if summary_path.is_file():
        try:
            connection = sqlite3.connect(summary_path)
            try:
                now = utc_now()
                if _table_exists(connection, "tasks"):
                    connection.execute(
                        """
                        update tasks set status = 'failed', completed_at = ?, error_message = ?
                        where status not in ('completed', 'failed', 'canceled')
                        """,
                        (now, message[:4000]),
                    )
                if _table_exists(connection, "runsets"):
                    connection.execute(
                        """
                        update runsets set status = 'failed', completed_at = ?, error_message = ?
                        where status not in ('completed', 'failed', 'canceled')
                        """,
                        (now, message[:4000]),
                    )
                if _table_exists(connection, "meta"):
                    connection.execute(
                        "insert or replace into meta(key, value) values ('status', ?)",
                        (json.dumps("failed"),),
                    )
                connection.commit()
            finally:
                connection.close()
        except Exception:
            pass

    trend_path = output_dir / TREND_DB_NAME
    if trend_path.is_file():
        try:
            connection = sqlite3.connect(trend_path)
            try:
                if _table_exists(connection, "meta"):
                    connection.execute(
                        "insert or replace into meta(key, value) values ('status', ?)",
                        (json.dumps("failed"),),
                    )
                    connection.commit()
            finally:
                connection.close()
        except Exception:
            pass


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?", (table,)
        ).fetchone()
        is not None
    )


def _failure_status(
    error: WorkerError,
    output_dir: Path,
    *,
    exception: BaseException | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if exception is not None:
        details = {
            "exception_type": type(exception).__name__,
            "traceback": "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            ),
        }
    return {
        "event": "worker_failed",
        "status": "failed",
        "exit_code": int(error.exit_code),
        "error_code": error.code,
        "message": str(error),
        "details": details,
        "output_dir": str(output_dir),
        "failed_at": utc_now(),
    }


def _best_effort_write_status(output_dir: Path, status: Mapping[str, Any]) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_dir / WORKER_STATUS_NAME, status)
    except Exception:
        pass


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(stable_json_dumps(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


class _WorkerArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        emit_event(
            {
                "event": "worker_failed",
                "status": "failed",
                "exit_code": int(WorkerExitCode.USAGE),
                "error_code": "invalid_arguments",
                "message": message,
                "failed_at": utc_now(),
            }
        )
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _WorkerArgumentParser(description="Run one EDOLAB batch task remotely.")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-parallel-runs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outcome = run_worker(
            task_file=args.task_file,
            output_dir=args.output_dir,
            max_parallel_runs=args.max_parallel_runs,
        )
    except WorkerError as exc:
        status = _failure_status(exc, Path(args.output_dir).expanduser().resolve())
        emit_event(status)
        return int(exc.exit_code)
    return int(outcome.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BundleIntegrityError",
    "EVENT_PREFIX",
    "TASK_PROVENANCE_NAME",
    "WORKER_STATUS_NAME",
    "WorkerError",
    "WorkerExitCode",
    "WorkerOutcome",
    "emit_event",
    "main",
    "run_worker",
    "verify_result_bundle",
]
