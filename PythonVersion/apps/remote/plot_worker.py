"""Detached worker for one frozen result-plot workload.

The controller supplies a UTF-8 JSON specification and an exact empty output
directory.  This module has no database, SSH, or scheduler knowledge, so the
same command can run inside a local immutable snapshot or on an SSH target::

    python -m apps.remote.plot_worker \
        --spec-file plot-spec.json \
        --output-dir results/workload-id

Every protocol message is one flushed ``EDOLAB_EVENT <json>`` line.  A
``worker_completed`` event is emitted only after every plot has been resolved
inside ``artifacts/``, hashed, and described by atomically written manifest,
result, and status files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import traceback
from typing import Any, Callable, Mapping, Sequence

from apps.remote.domain import stable_json_dumps, utc_now
from apps.remote.plot_workspace import PLOT_FORMATS, PlotWorkspace


EVENT_PREFIX = "EDOLAB_EVENT "
WORKER_STATUS_NAME = "worker-status.json"
WORKLOAD_RESULT_NAME = "workload-result.json"
ARTIFACT_MANIFEST_NAME = "artifact-manifest.json"
WORKLOAD_KIND = "plot"
SPEC_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
# Plot specifications carry bounded result series rather than a small set of
# benchmark parameters.  This remains a hard limit while allowing the public
# PlotWorkspace point limits to be useful for realistic trends.
MAX_SPEC_BYTES = 32 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_OPERATIONS = {"trend", "boxplot"}
_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "png": "image/png",
    "svg": "image/svg+xml",
}


class PlotWorkerExitCode(IntEnum):
    OK = 0
    USAGE = 2
    SPEC_INVALID = 10
    OUTPUT_INVALID = 11
    EXECUTION_FAILED = 20
    INTEGRITY_FAILED = 21
    INTERRUPTED = 130


class PlotWorkerError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        exit_code: PlotWorkerExitCode,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class PlotWorkerOutcome:
    exit_code: PlotWorkerExitCode
    status: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FrozenPlotSpec:
    operation: str
    payload: dict[str, Any]
    sha256: str
    source_name: str


EventSink = Callable[[Mapping[str, Any]], None]


def emit_event(payload: Mapping[str, Any]) -> None:
    print(EVENT_PREFIX + stable_json_dumps(payload), flush=True)


def run_plot_worker(
    *,
    spec_file: str | Path,
    output_dir: str | Path,
    expected_spec_sha256: str | None = None,
    event_sink: EventSink = emit_event,
) -> PlotWorkerOutcome:
    """Execute one validated plot specification into an empty directory."""

    requested_output = Path(output_dir).expanduser()
    try:
        output_path = _prepare_output_directory(requested_output)
    except Exception as exc:
        error = PlotWorkerError(
            "output_invalid",
            str(exc),
            PlotWorkerExitCode.OUTPUT_INVALID,
        )
        status = _failure_status(error, requested_output.absolute(), exception=exc)
        event_sink(status)
        return PlotWorkerOutcome(error.exit_code, status)

    try:
        frozen = _load_frozen_spec(spec_file, expected_spec_sha256)
        workspace = PlotWorkspace(output_path / "artifacts")
        request = workspace.validate_request(
            frozen.payload,
            operation=frozen.operation,
        )
    except Exception as exc:
        error = PlotWorkerError(
            "spec_invalid",
            str(exc),
            PlotWorkerExitCode.SPEC_INVALID,
        )
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return PlotWorkerOutcome(error.exit_code, status)

    series_count = len(request.series)
    point_count = sum(
        len(item.key) if hasattr(item, "key") else len(item.values)
        for item in request.series
    )
    event_sink(
        {
            "event": "worker_started",
            "status": "running",
            "pid": os.getpid(),
            "workload_kind": WORKLOAD_KIND,
            "operation": frozen.operation,
            "spec_file": frozen.source_name,
            "spec_sha256": frozen.sha256,
            "output_dir": str(output_path),
            "total_steps": 3,
            # Retained for the Relay's existing durable-run progress parser.
            "total_runs": 1,
            "at": utc_now(),
        }
    )
    _emit_progress(
        event_sink,
        frozen,
        current=1,
        stage="validated",
        message=(
            f"Validated {series_count} plot series with {point_count} point(s)."
        ),
    )

    try:
        raw_result = workspace.export_plot(frozen.payload, frozen.operation)
    except KeyboardInterrupt as exc:
        error = PlotWorkerError(
            "interrupted",
            "plot worker interrupted",
            PlotWorkerExitCode.INTERRUPTED,
        )
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return PlotWorkerOutcome(error.exit_code, status)
    except Exception as exc:
        error = PlotWorkerError(
            "execution_failed",
            str(exc),
            PlotWorkerExitCode.EXECUTION_FAILED,
        )
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return PlotWorkerOutcome(error.exit_code, status)

    _emit_progress(
        event_sink,
        frozen,
        current=2,
        stage="calculated",
        message=f"Completed {frozen.operation} plot calculation.",
    )

    try:
        published_result, artifacts = _publish_result(
            workspace=workspace,
            output_dir=output_path,
            operation=frozen.operation,
            raw_result=raw_result,
        )
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "workload_kind": WORKLOAD_KIND,
            "operation": frozen.operation,
            "spec_sha256": frozen.sha256,
            "files": artifacts,
        }
        _atomic_write_json(output_path / ARTIFACT_MANIFEST_NAME, manifest)
        manifest_descriptor = _file_descriptor(
            output_path,
            output_path / ARTIFACT_MANIFEST_NAME,
            kind="artifact_manifest",
            media_type="application/json",
        )
        workload_result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "completed",
            "workload_kind": WORKLOAD_KIND,
            "operation": frozen.operation,
            "spec_sha256": frozen.sha256,
            "result": published_result,
            "artifacts": artifacts,
        }
        _atomic_write_json(output_path / WORKLOAD_RESULT_NAME, workload_result)
        result_descriptor = _file_descriptor(
            output_path,
            output_path / WORKLOAD_RESULT_NAME,
            kind="workload_result",
            media_type="application/json",
        )
        _emit_progress(
            event_sink,
            frozen,
            current=3,
            stage="artifacts_written",
            message=f"Verified {len(artifacts)} plot artifact(s).",
        )
        status = {
            "event": "worker_completed",
            "status": "completed",
            "exit_code": int(PlotWorkerExitCode.OK),
            "workload_kind": WORKLOAD_KIND,
            "operation": frozen.operation,
            "spec_sha256": frozen.sha256,
            "output_dir": str(output_path),
            "artifacts": artifacts,
            "artifact_manifest": manifest_descriptor,
            "workload_result": result_descriptor,
            "completed_at": utc_now(),
        }
        _atomic_write_json(output_path / WORKER_STATUS_NAME, status)
        event_sink(status)
        return PlotWorkerOutcome(PlotWorkerExitCode.OK, status)
    except KeyboardInterrupt as exc:
        error = PlotWorkerError(
            "interrupted",
            "plot worker interrupted while publishing artifacts",
            PlotWorkerExitCode.INTERRUPTED,
        )
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return PlotWorkerOutcome(error.exit_code, status)
    except Exception as exc:
        error = PlotWorkerError(
            "integrity_failed",
            str(exc),
            PlotWorkerExitCode.INTEGRITY_FAILED,
        )
        status = _failure_status(error, output_path, exception=exc)
        _best_effort_write_status(output_path, status)
        event_sink(status)
        return PlotWorkerOutcome(error.exit_code, status)


def _publish_result(
    *,
    workspace: PlotWorkspace,
    output_dir: Path,
    operation: str,
    raw_result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(raw_result, Mapping):
        raise ValueError("plot workspace returned a non-object result")
    expected_kind = "trend_plot" if operation == "trend" else "boxplot"
    if raw_result.get("kind") != expected_kind:
        raise ValueError(
            f"plot result kind must be {expected_kind!r} for {operation!r}"
        )
    raw_artifacts = raw_result.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or not raw_artifacts:
        raise ValueError("plot result is missing its managed artifacts")

    by_format: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for raw_format in sorted(raw_artifacts, key=str):
        if not isinstance(raw_format, str) or raw_format not in PLOT_FORMATS:
            raise ValueError(f"plot result contains unsupported format {raw_format!r}")
        raw_artifact = raw_artifacts[raw_format]
        if not isinstance(raw_artifact, Mapping):
            raise ValueError(f"{raw_format} plot artifact metadata must be an object")
        storage_key = raw_artifact.get("storage_key")
        if not isinstance(storage_key, str):
            raise ValueError(f"{raw_format} plot artifact is missing storage_key")
        artifact_path = workspace.resolve_artifact(storage_key)
        descriptor = _file_descriptor(
            output_dir,
            artifact_path,
            kind="plot_artifact",
            media_type=_MEDIA_TYPES[raw_format],
        )
        if not str(descriptor["path"]).startswith("artifacts/"):
            raise ValueError("scientific plot artifact is outside artifacts/")
        descriptor["format"] = raw_format
        _verify_workspace_descriptor(raw_artifact, descriptor)
        by_format[raw_format] = descriptor
        artifacts.append(descriptor)

    published: dict[str, Any] = {
        "kind": expected_kind,
        "metric_name": _required_result_text(raw_result, "metric_name"),
        "series_count": _required_non_negative_int(raw_result, "series_count"),
        "point_count": _required_non_negative_int(raw_result, "point_count"),
        "artifacts": by_format,
        "preview_artifact": by_format.get("png"),
    }
    if operation == "trend":
        published["axis"] = _required_result_text(raw_result, "axis")
    return _json_safe(published), artifacts


def _verify_workspace_descriptor(
    raw: Mapping[str, Any],
    published: Mapping[str, Any],
) -> None:
    comparisons = {
        "format": published["format"],
        "media_type": published["media_type"],
        "size_bytes": published["size_bytes"],
        "sha256": published["sha256"],
    }
    for field, expected in comparisons.items():
        if raw.get(field) != expected:
            raise ValueError(
                f"plot workspace {field} metadata does not match the exported artifact"
            )


def _required_result_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"plot result {field} must be a non-empty string")
    return value


def _required_non_negative_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"plot result {field} must be a non-negative integer")
    return value


def _load_frozen_spec(
    spec_file: str | Path,
    expected_spec_sha256: str | None,
) -> FrozenPlotSpec:
    source = Path(spec_file).expanduser()
    payload_bytes = _read_regular_file(source, maximum_bytes=MAX_SPEC_BYTES)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if expected_spec_sha256 is not None:
        expected = expected_spec_sha256.strip().lower()
        if not _SHA256_PATTERN.fullmatch(expected):
            raise ValueError(
                "expected_spec_sha256 must be exactly 64 hexadecimal characters"
            )
        if digest != expected:
            raise ValueError(
                "frozen plot spec SHA-256 mismatch: "
                f"expected {expected}, received {digest}"
            )
    try:
        text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("plot spec must be UTF-8 JSON") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"plot spec is invalid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("plot spec root must be an object")

    allowed = {"schema_version", "workload_kind", "operation", "payload"}
    unknown = sorted(set(decoded) - allowed, key=str)
    if unknown:
        raise ValueError(
            "unknown plot spec field(s): "
            + ", ".join(str(item) for item in unknown)
        )
    schema_version = decoded.get("schema_version", SPEC_SCHEMA_VERSION)
    if type(schema_version) is not int or schema_version != SPEC_SCHEMA_VERSION:
        raise ValueError(f"plot spec schema_version must be {SPEC_SCHEMA_VERSION}")
    workload_kind = decoded.get("workload_kind", WORKLOAD_KIND)
    if workload_kind != WORKLOAD_KIND:
        raise ValueError(f"workload_kind must be {WORKLOAD_KIND!r}")
    operation = decoded.get("operation")
    if not isinstance(operation, str) or operation not in _OPERATIONS:
        raise ValueError("operation must be one of trend, boxplot")
    payload = decoded.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("plot spec payload must be an object")
    if "target_id" in payload:
        raise ValueError(
            "plot spec payload must not contain target_id; the controller selects the target"
        )
    return FrozenPlotSpec(
        operation=operation,
        payload=payload,
        sha256=digest,
        source_name=source.name,
    )


def _prepare_output_directory(requested: Path) -> Path:
    if "\x00" in str(requested):
        raise ValueError("output directory contains a NUL byte")
    _reject_symlink_components(requested, label="output directory")
    if requested.exists() and requested.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {requested}")
    parent = requested.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ValueError(f"output directory parent is unsafe: {parent}")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"output directory parent is unsafe: {parent}")
    requested.mkdir(mode=0o700, exist_ok=True)
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError(f"output path is not a safe directory: {requested}")
    entries = list(requested.iterdir())
    if entries:
        names = ", ".join(sorted(item.name for item in entries[:5]))
        raise ValueError(
            "exact output directory must be empty to avoid overwriting a previous workload"
            + (f": {names}" if names else "")
        )
    return requested.resolve(strict=True)


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    if "\x00" in str(path):
        raise ValueError("spec file path contains a NUL byte")
    _reject_symlink_components(path, label="spec file")
    if path.is_symlink():
        raise ValueError(f"spec file must not be a symlink: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open plot spec: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"plot spec is not a regular file: {path}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"plot spec exceeds the {maximum_bytes}-byte limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(
                    f"plot spec exceeds the {maximum_bytes}-byte limit"
                )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("plot spec changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject symlinks in any existing lexical component of an input path."""

    absolute = path.absolute()
    components = [absolute, *absolute.parents]
    for component in components:
        if component.exists() and component.is_symlink():
            raise ValueError(
                f"{label} must not be a symlink or traverse a symlink: {component}"
            )


def _file_descriptor(
    root: Path,
    path: Path,
    *,
    kind: str,
    media_type: str,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"artifact must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"artifact escapes the workload output directory: {path}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "kind": kind,
        "media_type": media_type,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (stable_json_dumps(_json_safe(payload)) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _assert_safe_write_target(path)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_safe_write_target(path: Path) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"artifact parent is unsafe: {parent}")
    if path.is_symlink() or path.exists():
        raise ValueError(f"refusing to overwrite workload file: {path}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _emit_progress(
    event_sink: EventSink,
    frozen: FrozenPlotSpec,
    *,
    current: int,
    stage: str,
    message: str,
) -> None:
    event_sink(
        {
            "event": "progress",
            "status": "running",
            "workload_kind": WORKLOAD_KIND,
            "operation": frozen.operation,
            "spec_sha256": frozen.sha256,
            "current": current,
            "total": 3,
            "unit": "steps",
            "stage": stage,
            "message": message,
            "at": utc_now(),
        }
    )


def _failure_status(
    error: PlotWorkerError,
    output_dir: Path,
    *,
    exception: BaseException | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if exception is not None:
        details = {
            "exception_type": type(exception).__name__,
            "traceback": "".join(
                traceback.format_exception(
                    type(exception),
                    exception,
                    exception.__traceback__,
                )
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


def _best_effort_write_status(
    output_dir: Path,
    status: Mapping[str, Any],
) -> None:
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            return
        target = output_dir / WORKER_STATUS_NAME
        if target.exists() or target.is_symlink():
            return
        _atomic_write_json(target, status)
    except Exception:
        pass


class _PlotWorkerArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        emit_event(
            {
                "event": "worker_failed",
                "status": "failed",
                "exit_code": int(PlotWorkerExitCode.USAGE),
                "error_code": "invalid_arguments",
                "message": message,
                "failed_at": utc_now(),
            }
        )
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _PlotWorkerArgumentParser(
        description="Run one frozen EDOLAB result-plot workload."
    )
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-spec-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outcome = run_plot_worker(
        spec_file=args.spec_file,
        output_dir=args.output_dir,
        expected_spec_sha256=args.expected_spec_sha256,
    )
    return int(outcome.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_MANIFEST_NAME",
    "EVENT_PREFIX",
    "PlotWorkerError",
    "PlotWorkerExitCode",
    "PlotWorkerOutcome",
    "WORKER_STATUS_NAME",
    "WORKLOAD_RESULT_NAME",
    "build_parser",
    "emit_event",
    "main",
    "run_plot_worker",
]
