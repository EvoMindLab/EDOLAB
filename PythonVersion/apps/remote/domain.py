"""Domain objects shared by the remote execution service and its adapters.

The module contains immutable records, lifecycle constants, and deterministic JSON/time
helpers so the same model can be used by a CLI, a local web API, or background
workers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4


# Recursive type aliases require syntax only available after the project's
# Python 3.10 minimum.  Runtime JSON validation is performed by ``json.dumps``.
JsonValue = Any
_SAFE_SSH_USERNAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_CONDA_ENVIRONMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def new_id() -> str:
    """Return an opaque, URL-safe identifier without embedding machine state."""

    return uuid4().hex


def utc_now() -> str:
    """Return a sortable ISO-8601 UTC timestamp with millisecond precision."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def normalize_utc_iso(value: str | datetime) -> str:
    """Normalize a timestamp to the canonical UTC representation used in SQLite."""

    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _json_default(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return normalize_utc_iso(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)  # type: ignore[return-value]
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def stable_json_dumps(value: object) -> str:
    """Serialize JSON deterministically for manifests and provenance records."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def stable_json_loads(value: str | None, default: Any = None) -> Any:
    """Deserialize a nullable JSON column without sharing mutable defaults."""

    if value is None:
        return default
    return json.loads(value)


class _StringEnum(str, Enum):
    """Python 3.10-compatible equivalent of ``enum.StrEnum``."""

    def __str__(self) -> str:
        return self.value


class RunStatus(_StringEnum):
    QUEUED = "queued"
    CHECKING = "checking"
    SNAPSHOTTING = "snapshotting"
    SYNCING = "syncing"
    PREPARING = "preparing"
    RUNNING = "running"
    TRANSFERRING = "transferring"
    VERIFYING = "verifying"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELING = "canceling"
    CANCELED = "canceled"
    INTERRUPTED = "interrupted"


TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELED,
        RunStatus.INTERRUPTED,
    }
)
NON_TERMINAL_RUN_STATUSES = frozenset(set(RunStatus) - TERMINAL_RUN_STATUSES)


# The worker normally follows the first path.  Forward skips are intentionally
# permitted because a no-op sync/transfer phase should not require fake states.
_FORWARD_PHASES = (
    RunStatus.QUEUED,
    RunStatus.CHECKING,
    RunStatus.SNAPSHOTTING,
    RunStatus.SYNCING,
    RunStatus.PREPARING,
    RunStatus.RUNNING,
    RunStatus.TRANSFERRING,
    RunStatus.VERIFYING,
    RunStatus.CLEANING,
    RunStatus.COMPLETED,
)
ALLOWED_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {}
for _index, _status in enumerate(_FORWARD_PHASES):
    _following = set(_FORWARD_PHASES[_index + 1 :])
    if _status not in TERMINAL_RUN_STATUSES:
        _following.update(
            {RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.CANCELING}
        )
    ALLOWED_RUN_TRANSITIONS[_status] = frozenset(_following)
ALLOWED_RUN_TRANSITIONS[RunStatus.CANCELING] = frozenset(
    {RunStatus.CANCELED, RunStatus.FAILED, RunStatus.INTERRUPTED}
)
for _status in TERMINAL_RUN_STATUSES:
    ALLOWED_RUN_TRANSITIONS.setdefault(_status, frozenset())


def validate_run_transition(
    current: RunStatus | str, target: RunStatus | str
) -> tuple[RunStatus, RunStatus]:
    current_status = RunStatus(current)
    target_status = RunStatus(target)
    if current_status == target_status:
        return current_status, target_status
    if target_status not in ALLOWED_RUN_TRANSITIONS[current_status]:
        raise ValueError(
            f"invalid run status transition: {current_status.value} -> "
            f"{target_status.value}"
        )
    return current_status, target_status


class EventLevel(_StringEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ArtifactStatus(_StringEnum):
    EXPECTED = "expected"
    TRANSFERRING = "transferring"
    TRANSFERRED = "transferred"
    VERIFIED = "verified"
    FAILED = "failed"


class ExecutionTargetKind(_StringEnum):
    """How Relay reaches a compute target.

    ``LOCAL`` is an explicit local-process transport.  It is intentionally not
    modelled as SSH-to-localhost: local execution must not require a key and
    must remain distinguishable in provenance and resource history.
    """

    SSH = "ssh"
    LOCAL = "local"


@dataclass(frozen=True, slots=True, kw_only=True)
class Server:
    name: str
    host: str
    username: str
    ssh_key_path: str
    remote_root: str
    id: str = field(default_factory=new_id)
    target_kind: ExecutionTargetKind = ExecutionTargetKind.SSH
    port: int = 22
    conda_env: str = "EDOLAB"
    poll_interval_seconds: float = 5.0
    default_max_parallel_runs: int = 1
    min_free_bytes: int = 5 * 1024**3
    max_load_ratio: float = 0.90
    enabled: bool = True
    last_probe_at: str | None = None
    last_probe: Mapping[str, Any] | None = None
    fingerprint: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_kind", ExecutionTargetKind(self.target_kind))
        for field_name in (
            "id",
            "name",
            "host",
            "username",
            "ssh_key_path",
            "remote_root",
            "conda_env",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        host = self.host.strip()
        if host.startswith("-") or any(character.isspace() for character in host):
            raise ValueError("host must be a valid hostname or address")
        if any(character in host for character in ("\x00", "\r", "\n")):
            raise ValueError("host contains an invalid control character")
        if not _SAFE_SSH_USERNAME.fullmatch(self.username.strip()):
            raise ValueError("username contains unsupported characters")
        if not _SAFE_CONDA_ENVIRONMENT.fullmatch(self.conda_env.strip()):
            raise ValueError("conda_env contains unsupported characters")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "username", self.username.strip())
        object.__setattr__(self, "conda_env", self.conda_env.strip())
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.default_max_parallel_runs < 1:
            raise ValueError("default_max_parallel_runs must be at least 1")
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes must not be negative")
        if not math.isfinite(self.max_load_ratio) or self.max_load_ratio <= 0:
            raise ValueError("max_load_ratio must be finite and positive")

        # A private key is intentionally represented only by a local filesystem
        # path.  Rejecting PEM material prevents accidental secret persistence.
        key_path = self.ssh_key_path.strip()
        if "\n" in key_path or "\r" in key_path or "PRIVATE KEY" in key_path.upper():
            raise ValueError("ssh_key_path must be a path, not private-key content")
        if self.last_probe_at is not None:
            object.__setattr__(
                self, "last_probe_at", normalize_utc_iso(self.last_probe_at)
            )
        object.__setattr__(self, "created_at", normalize_utc_iso(self.created_at))
        object.__setattr__(self, "updated_at", normalize_utc_iso(self.updated_at))

    def with_updates(self, **changes: Any) -> "Server":
        changes.setdefault("updated_at", utc_now())
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, kw_only=True)
class Run:
    server_id: str
    task_name: str
    id: str = field(default_factory=new_id)
    status: RunStatus = RunStatus.QUEUED
    stage: Mapping[str, Any] = field(default_factory=dict)
    task_kind: str = "batch"
    task_path: str | None = None
    task_sha256: str | None = None
    task_payload: Mapping[str, Any] = field(default_factory=dict)
    command_argv: tuple[str, ...] = ()
    max_parallel_runs: int = 1
    poll_interval_seconds: float = 5.0
    admission_override: bool = False

    # Exact code/input provenance used by this run.
    source_root: str | None = None
    snapshot_id: str | None = None
    snapshot_sha256: str | None = None
    snapshot_manifest: Mapping[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    git_diff_sha256: str | None = None

    remote_pid: int | None = None
    remote_run_dir: str | None = None
    worker_id: str | None = None
    heartbeat_at: str | None = None

    progress_current: float = 0.0
    progress_total: float | None = None
    progress_unit: str | None = None
    progress_message: str | None = None

    exit_code: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: Mapping[str, Any] = field(default_factory=dict)

    local_artifact_dir: str | None = None
    transfer_manifest: Mapping[str, Any] = field(default_factory=dict)
    transfer_started_at: str | None = None
    transfer_completed_at: str | None = None
    transfer_verified: bool = False
    transfer_verified_at: str | None = None
    cleanup_requested: bool = True
    remote_cleanup_completed: bool = False
    remote_cleanup_at: str | None = None
    remote_cleanup_error: str | None = None

    queued_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    remote_finished_at: str | None = None
    finished_at: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.server_id.strip():
            raise ValueError("id and server_id must not be empty")
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty")
        object.__setattr__(self, "status", RunStatus(self.status))
        if not isinstance(self.stage, Mapping):
            raise ValueError("stage must be a JSON object")
        try:
            normalized_stage = stable_json_loads(stable_json_dumps(self.stage), {})
        except (TypeError, ValueError) as exc:
            raise ValueError("stage must be a JSON object") from exc
        if not isinstance(normalized_stage, dict):
            raise ValueError("stage must be a JSON object")
        object.__setattr__(self, "stage", normalized_stage)
        object.__setattr__(self, "command_argv", tuple(self.command_argv))
        if not math.isfinite(self.progress_current) or self.progress_current < 0:
            raise ValueError("progress_current must be finite and non-negative")
        if self.progress_total is not None:
            if not math.isfinite(self.progress_total) or self.progress_total < 0:
                raise ValueError("progress_total must be finite and non-negative")
        if self.remote_pid is not None and self.remote_pid <= 0:
            raise ValueError("remote_pid must be positive")
        if self.max_parallel_runs < 1:
            raise ValueError("max_parallel_runs must be at least 1")
        if (
            not math.isfinite(self.poll_interval_seconds)
            or self.poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be finite and positive")
        for name in (
            "heartbeat_at",
            "transfer_started_at",
            "transfer_completed_at",
            "transfer_verified_at",
            "remote_cleanup_at",
            "queued_at",
            "started_at",
            "remote_finished_at",
            "finished_at",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, normalize_utc_iso(value))

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    def with_updates(self, **changes: Any) -> "Run":
        changes.setdefault("updated_at", utc_now())
        return replace(self, **changes)


LOCAL_TRANSFER_VERIFICATION_ERROR = "LOCAL_TRANSFER_VERIFICATION_FAILED"
_LEGACY_PATH_DEPENDENT_PROVENANCE_ERROR = (
    "Local and remote task provenance identities differ."
)


def _can_reverify_successful_transfer(run: Run) -> bool:
    """Return whether a failed record can safely repeat only local gates."""

    if not (
        run.status is RunStatus.FAILED
        and run.transfer_verified
        and run.exit_code == 0
    ):
        return False
    return bool(
        run.error_code == LOCAL_TRANSFER_VERIFICATION_ERROR
        or run.error_message == _LEGACY_PATH_DEPENDENT_PROVENANCE_ERROR
    )


def run_recovery_mode(run: Run) -> str | None:
    """Return the safe terminal-run recovery action, if one exists.

    Recovery is deliberately derived only from durable evidence.  A recorded
    remote PID means Relay can reconcile the worker and retrieve its artifacts;
    a persisted launch intent means it can recover the idempotent process
    identity; and a verified local transfer permits checksum-gated cleanup.
    When no launch intent exists, the marker-owned workspace is known to be
    pre-launch and may be cleaned without risking an uncollected scientific
    result.
    """

    if not run.terminal or not run.remote_run_dir:
        return None
    if run.cleanup_requested and not run.remote_cleanup_completed:
        if run.transfer_verified:
            if _can_reverify_successful_transfer(run):
                return "verify"
            return "cleanup"
        if run.remote_pid is not None:
            return "transfer"
        execution = run.task_payload.get("execution")
        if isinstance(execution, Mapping) and execution.get("launch_intent_at"):
            return "launch"
        return "cleanup"
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunEvent:
    run_id: str
    kind: str
    message: str
    level: EventLevel = EventLevel.INFO
    data: Mapping[str, Any] = field(default_factory=dict)
    id: int | None = None
    sequence: int | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.kind.strip():
            raise ValueError("run_id and kind must not be empty")
        object.__setattr__(self, "level", EventLevel(self.level))
        object.__setattr__(self, "created_at", normalize_utc_iso(self.created_at))


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceSample:
    server_id: str
    run_id: str | None = None
    target_identity: str | None = None
    id: int | None = None
    sampled_at: str = field(default_factory=utc_now)
    reachable: bool = True
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    cpu_percent: float | None = None
    load_1m: float | None = None
    load_5m: float | None = None
    load_15m: float | None = None
    load_ratio: float | None = None
    memory_total_bytes: int | None = None
    memory_available_bytes: int | None = None
    disk_total_bytes: int | None = None
    disk_free_bytes: int | None = None
    gpu_usable: bool | None = None
    gpu_error: str | None = None
    gpu_devices: Sequence[Mapping[str, Any]] = ()
    error: str | None = None
    aggregation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.server_id.strip():
            raise ValueError("server_id must not be empty")
        if self.target_identity is not None and not self.target_identity.strip():
            raise ValueError("target_identity must not be empty when provided")
        object.__setattr__(self, "sampled_at", normalize_utc_iso(self.sampled_at))
        object.__setattr__(self, "gpu_devices", tuple(self.gpu_devices))
        if self.gpu_usable is not None:
            object.__setattr__(self, "gpu_usable", bool(self.gpu_usable))
        for name in ("cpu_physical_cores", "cpu_logical_cores"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        for name in (
            "memory_total_bytes",
            "memory_available_bytes",
            "disk_total_bytes",
            "disk_free_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class Artifact:
    run_id: str
    relative_path: str
    id: str = field(default_factory=new_id)
    status: ArtifactStatus = ArtifactStatus.EXPECTED
    remote_path: str | None = None
    local_path: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    transferred_at: str | None = None
    verified_at: str | None = None
    verification_error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.run_id.strip():
            raise ValueError("id and run_id must not be empty")
        path = self.relative_path.strip()
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("relative_path must be a safe relative path")
        object.__setattr__(self, "status", ArtifactStatus(self.status))
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        for name in ("transferred_at", "verified_at", "created_at", "updated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, normalize_utc_iso(value))

    def with_updates(self, **changes: Any) -> "Artifact":
        changes.setdefault("updated_at", utc_now())
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, kw_only=True)
class DashboardCounts:
    servers_total: int
    servers_enabled: int
    runs_total: int
    active_runs: int
    status_counts: Mapping[str, int]
