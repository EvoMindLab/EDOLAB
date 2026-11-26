"""Application service for the EDOLAB Relay local control plane.

The service is deliberately transport-neutral: FastAPI is one adapter, while a
future CLI or desktop shell can use the same methods.  Remote execution is owned
by an optional controller implementing :class:`Controller`; durable records are
useful even when that controller is not present (for example in API tests).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import shlex
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Protocol
from uuid import uuid4
import zipfile

from core.result.validation import ResultBundleValidationError, validate_result_bundle

from .config import RelayPaths
from .benchmark_workspace import (
    BenchmarkWorkspace,
    BenchmarkWorkspaceError,
)
from .database import RemoteDatabase, ServerTargetChangedError
from .durability import fsync_directory, fsync_file
from .domain import (
    NON_TERMINAL_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    Artifact,
    ArtifactStatus,
    ExecutionTargetKind,
    ResourceSample,
    Run,
    RunEvent,
    RunStatus,
    Server,
    normalize_utc_iso,
    run_recovery_mode,
    stable_json_dumps,
    utc_now,
)
from .local_transport import LocalProcessClient, local_machine_fingerprint
from .plot_workspace import (
    PlotPreview,
    PlotWorkspace,
    PlotWorkspaceError,
)
from .environment import (
    missing_runtime_dependency_messages,
    runtime_dependency_statuses,
)
from .resource_monitor import ResourceMonitor
from .result_import_workspace import (
    ResultImportWorkspace,
    ResultImportWorkspaceError,
)
from .result_workspace import ResultWorkspace, ResultWorkspaceError
from .ssh import SSHClient, SSHConnection


SETTINGS_KEY = "relay.settings"


def _default_ssh_key_path() -> str:
    configured = os.environ.get("EDOLAB_RELAY_SSH_KEY", "").strip()
    if configured:
        return configured
    # This repository is commonly used with a dedicated research key outside
    # ~/.ssh.  Prefer an existing dedicated key, then fall back to OpenSSH's
    # conventional Ed25519 location.  The selected value is only a local path;
    # key bytes are never stored in Relay or copied to an execution target.
    for candidate in ("~/ssh/codex_ed25519", "~/.ssh/id_ed25519"):
        if Path(candidate).expanduser().is_file():
            return candidate
    return "~/.ssh/id_ed25519"


DEFAULT_SETTINGS: dict[str, Any] = {
    # Docker bind-mounts the selected host key at the exact same absolute path
    # recorded by the native launcher.  Using the environment value as the
    # first-run default lets both launch modes share one database unchanged.
    "default_ssh_key_path": _default_ssh_key_path(),
    "default_remote_root": "~/edolab-remote",
    "default_poll_interval_seconds": 5,
    "resource_sampling_interval_seconds": 3,
    "resource_history_retention_hours": 72,
    "default_max_parallel_runs": 1,
    "probe_timeout_seconds": 30,
    "require_clean_git": False,
    "verify_artifact_checksums": True,
    "delete_remote_results_after_transfer": True,
}
RESOURCE_REFRESH_CACHE_SECONDS = 2.0
RESOURCE_REFRESH_TIMEOUT_SECONDS = 10.0
RESOURCE_HISTORY_MAX_SAMPLES_PER_SERVER = 100_000
RESOURCE_STALE_INTERVAL_MULTIPLIER = 3.0
LOCAL_TARGET_ID = "local-machine"
RESULT_ARCHIVE_MANIFEST_NAME = "manifest.json"
RESULT_ARCHIVE_SINGLE_PREFIX = "result"
RESULT_ARCHIVE_MAX_MANIFEST_BYTES = 1024 * 1024
RESULT_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024 * 1024
RESULT_ARCHIVE_MAX_UPLOAD_BYTES = 64 * 1024 * 1024 * 1024
RESULT_ARCHIVE_MAX_RESULTS = 100
RESULT_ARCHIVE_COPY_CHUNK_BYTES = 1024 * 1024


class Controller(Protocol):
    def start(self) -> None: ...

    def stop(self, timeout: float | None = None) -> None: ...

    def wake(self) -> None: ...

    def request_cancel(self, run_id: str) -> object: ...

    def request_recovery(self, run_id: str) -> object: ...


class ProbeClient(Protocol):
    def probe(
        self, remote_path: str, conda_env: str = "EDOLAB", *, timeout: float = 30.0
    ) -> dict[str, Any]: ...

    def sample_resources(
        self,
        *,
        timeout: float = 10.0,
        expected_fingerprint: str | None = None,
    ) -> dict[str, Any]: ...


class ServiceError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class RelayService:
    def __init__(
        self,
        paths: RelayPaths,
        *,
        database: RemoteDatabase | None = None,
        controller: Controller | None = None,
        ssh_client_factory: Callable[[SSHConnection], ProbeClient] = SSHClient,
        local_client_factory: Callable[[], ProbeClient] = LocalProcessClient,
        enable_local_target: bool = False,
    ) -> None:
        self.paths = paths
        self.paths.ensure()
        self.database = database or RemoteDatabase(paths.database)
        self.result_workspace = ResultWorkspace(paths, self.database)
        self.result_import_workspace = ResultImportWorkspace(paths.data_root)
        self._cleanup_demoted_legacy_result_imports()
        self._migrate_legacy_result_imports()
        self.benchmark_workspace = BenchmarkWorkspace(
            paths.analyses / "interactive" / "benchmarks"
        )
        self.plot_workspace = PlotWorkspace(
            paths.analyses / "interactive" / "plots"
        )
        self._result_import_lock = threading.RLock()
        # NBN construction is quadratic in the sample count.  Interactive
        # requests are never queued, so reject overlapping calculations instead
        # of silently over-subscribing the local Relay process.
        self._interactive_benchmark_lock = threading.Lock()
        self.controller = controller
        self.ssh_client_factory = ssh_client_factory
        self.local_client_factory = local_client_factory
        self.enable_local_target = bool(enable_local_target)
        if self.enable_local_target:
            self._ensure_local_target()
        self._controller_lock = threading.Lock()
        self._controller_started = False
        self._resource_monitor_started = False
        self._local_maintenance_completed = False
        self._resource_refresh_state_lock = threading.Lock()
        self._resource_refresh_locks: dict[str, threading.Lock] = {}
        self._resource_refresh_cache: dict[
            str, tuple[float, bool, dict[str, Any] | tuple[int, str, str]]
        ] = {}
        self.resource_monitor = ResourceMonitor(
            eligible_server_ids=self._resource_monitor_server_ids,
            sample_once=self._background_resource_sample,
            sample_interval_seconds=self._resource_sampling_interval,
            prune_history=self._prune_resource_history,
        )

    def _cleanup_demoted_legacy_result_imports(self) -> None:
        """Remove managed copies created only by the old SA auto-promotion.

        Schema v9 removes those records before the service starts.  The queue
        lets the filesystem cleanup finish safely without ever touching the
        original run-owned result folder, which remains a result candidate.
        """

        for record in self.database.list_result_import_cleanup_queue():
            result_id = str(record["result_id"])
            expected_relative = Path("analyses") / "results" / result_id
            configured = Path(str(record["managed_relative_dir"]))
            if configured != expected_relative:
                self.database.complete_result_import_cleanup(result_id)
                continue
            target = self.paths.imported_results / result_id
            staged = self.paths.trash / f"legacy-result-{result_id}.deleting"
            if target.is_symlink() or staged.is_symlink():
                continue
            try:
                if not target.exists():
                    if staged.exists():
                        _remove_staged_path(staged)
                    self.database.complete_result_import_cleanup(result_id)
                    continue
                if staged.exists():
                    continue
                os.replace(target, staged)
                fsync_directory(self.paths.imported_results)
                _remove_staged_path(staged)
                fsync_directory(self.paths.trash)
                self.database.complete_result_import_cleanup(result_id)
            except OSError:
                # Keep the queue row so the next clean startup can retry.
                continue

    def _migrate_legacy_result_imports(self) -> None:
        """Detach old, explicitly imported results from synthetic run records."""

        for record in self.database.list_result_imports():
            if record.get("source_kind") != "legacy_run":
                continue
            result_id = str(record["id"])
            expected_relative = Path("analyses") / "results" / result_id
            if Path(str(record["managed_relative_dir"])) == expected_relative:
                continue
            source_run_id = str(record.get("source_run_id") or result_id)
            source_run = self.database.get_run(source_run_id)
            if source_run is None or source_run.task_kind != "result_import":
                continue
            try:
                source = self.result_workspace.verified_run_bundle_path(source_run_id)
            except ResultWorkspaceError:
                # A pre-upgrade record may already have lost its old run-owned
                # files.  Leave it unavailable rather than inventing evidence.
                continue
            staging = self.paths.working / f"legacy-result-{result_id}"
            final = self.paths.imported_results / result_id
            if final.exists() and not final.is_symlink():
                continue
            if staging.exists() and not staging.is_symlink():
                _remove_staged_path(staging)
            staging.mkdir(mode=0o700)
            try:
                imported = self.result_import_workspace.import_bundle(
                    source,
                    staging,
                    allow_managed_source=True,
                )
                staged_bundle = staging / "remote" / "results" / "bundle"
                os.replace(staged_bundle, final)
                _remove_staged_path(staging)
                files = {item.filename: item for item in imported.files}
                metadata = dict(record.get("source_metadata") or {})
                metadata["migration"] = "synthetic_result_import_run"
                self.database.update_result_import(
                    result_id,
                    managed_relative_dir=expected_relative.as_posix(),
                    source_metadata=metadata,
                    validation=imported.as_dict()["validation"],
                    bundle_id=imported.validation.bundle_id,
                    summary_size_bytes=files["summary.sqlite"].size_bytes,
                    summary_sha256=files["summary.sqlite"].sha256,
                    trend_size_bytes=files["trend.sqlite"].size_bytes,
                    trend_sha256=files["trend.sqlite"].sha256,
                )
            except (OSError, ResultImportWorkspaceError, ValueError):
                if staging.exists() and not staging.is_symlink():
                    _remove_staged_path(staging)
                if final.exists() and not final.is_symlink():
                    _remove_staged_path(final)
                continue

    # -- Process lifecycle ----------------------------------------------

    def start(self) -> None:
        with self._controller_lock:
            if not self._local_maintenance_completed:
                self._reconcile_delete_tombstones()
                self._local_maintenance_completed = True
            if self.enable_local_target:
                local_target = self.database.get_server(LOCAL_TARGET_ID)
                if local_target is not None and local_target.last_probe_at is None:
                    # First-run convenience: discover the already-required
                    # EDOLAB environment so the built-in target is selectable
                    # without asking the user for fictitious SSH settings.
                    try:
                        self.probe_server(local_target.id)
                    except ServiceError:
                        # The failed check is persisted and visible per
                        # condition in the UI; an unavailable local environment
                        # must not prevent management of remote targets.
                        pass
            try:
                if self.controller is not None:
                    self.controller.start()
                    self._controller_started = True
                self.resource_monitor.start()
                self._resource_monitor_started = True
            except BaseException:
                # Startup is all-or-nothing.  In particular, a controller
                # recovery failure must not leave a background SSH sampler
                # running after the API lifespan has failed to start.
                if self._resource_monitor_started or self.resource_monitor.running:
                    self.resource_monitor.request_stop()
                    self.resource_monitor.join(timeout=10.0)
                self._resource_monitor_started = False
                if self.controller is not None:
                    try:
                        self.controller.stop(timeout=10.0)
                    except BaseException:
                        # Preserve the original startup failure; the next
                        # explicit start will still reject a controller that
                        # remains in its stopping state.
                        pass
                    finally:
                        self._controller_started = False
                raise

    def stop(self, timeout: float | None = 10.0) -> None:
        with self._controller_lock:
            budget = 10.0 if timeout is None else max(0.0, float(timeout))
            if self._resource_monitor_started:
                # Stop scheduling immediately.  Existing bounded SSH attempts
                # can wind down while the run controller persists recovery
                # state, rather than consuming its entire shutdown budget.
                self.resource_monitor.request_stop()
            started = time.monotonic()
            if self._controller_started and self.controller is not None:
                try:
                    self.controller.stop(timeout=budget)
                finally:
                    self._controller_started = False
            elapsed = time.monotonic() - started
            if self._resource_monitor_started:
                try:
                    self.resource_monitor.join(
                        timeout=max(budget * 0.3, budget - elapsed)
                    )
                finally:
                    self._resource_monitor_started = False

    # -- Servers --------------------------------------------------------

    def list_servers(self) -> list[dict[str, Any]]:
        servers = self.database.list_servers()
        settings = self.get_settings()
        sample_maps = self._resource_monitor_sample_maps(servers)
        return [
            self.serialize_server(
                server,
                monitor_samples=self._resource_monitor_samples_for(
                    server.id, sample_maps
                ),
                settings=settings,
            )
            for server in servers
        ]

    def get_server(self, server_id: str) -> dict[str, Any]:
        return self.serialize_server(self._require_server(server_id))

    def create_server(self, values: Mapping[str, Any]) -> dict[str, Any]:
        settings = self.get_settings()
        try:
            server = Server(
                name=str(values.get("name", "")),
                target_kind=ExecutionTargetKind.SSH,
                host=str(values.get("host", "")),
                port=int(values.get("port", 22)),
                username=str(values.get("username", "")),
                ssh_key_path=_normalize_ssh_key_path(
                    str(
                        values.get("ssh_key_path")
                        or settings["default_ssh_key_path"]
                    )
                ),
                remote_root=_normalize_remote_root(
                    str(
                        values.get("remote_root") or settings["default_remote_root"]
                    )
                ),
                conda_env=str(values.get("conda_env", "EDOLAB")),
                poll_interval_seconds=float(
                    values.get(
                        "poll_interval_seconds",
                        settings["default_poll_interval_seconds"],
                    )
                ),
                default_max_parallel_runs=int(
                    values.get(
                        "default_max_parallel_runs",
                        settings["default_max_parallel_runs"],
                    )
                ),
                min_free_bytes=int(values.get("min_free_bytes", 5 * 1024**3)),
                max_load_ratio=float(values.get("max_load_ratio", 0.90)),
                enabled=bool(values.get("enabled", True)),
            )
            self._ensure_unique_execution_target(server)
            self.database.create_server(server)
        except (TypeError, ValueError) as exc:
            raise ServiceError(400, "invalid_server", str(exc)) from exc
        self.resource_monitor.wake(reschedule=True)
        return self.serialize_server(server)

    def update_server(
        self, server_id: str, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        current = self._require_server(server_id)
        if current.target_kind is ExecutionTargetKind.LOCAL:
            immutable = {
                "host",
                "port",
                "username",
                "ssh_key_path",
                "remote_root",
                "conda_env",
            }
            changed_immutable = sorted(
                key
                for key in immutable
                if key in values and values[key] != getattr(current, key)
            )
            if changed_immutable:
                raise ServiceError(
                    409,
                    "local_target_identity_managed",
                    "The built-in local target manages its own execution identity; "
                    "only its name, scheduling switch, intervals, limits, and "
                    "warning thresholds can be changed.",
                )
        allowed = {
            "name",
            "host",
            "port",
            "username",
            "ssh_key_path",
            "remote_root",
            "conda_env",
            "poll_interval_seconds",
            "default_max_parallel_runs",
            "min_free_bytes",
            "max_load_ratio",
            "enabled",
        }
        changes = {key: value for key, value in values.items() if key in allowed}
        if "ssh_key_path" in changes:
            changes["ssh_key_path"] = _normalize_ssh_key_path(
                str(changes["ssh_key_path"])
            )
        if "remote_root" in changes:
            changes["remote_root"] = _normalize_remote_root(str(changes["remote_root"]))
        try:
            probe_invalidating = {
                "host",
                "port",
                "username",
                "ssh_key_path",
                "remote_root",
                "conda_env",
            }
            if any(
                name in changes and getattr(current, name) != changes[name]
                for name in probe_invalidating
            ):
                changes.update(
                    last_probe_at=None,
                    last_probe=None,
                    fingerprint=None,
                )
            updated = current.with_updates(**changes)
            protected = {
                "host",
                "port",
                "username",
                "ssh_key_path",
                "remote_root",
                "conda_env",
                "poll_interval_seconds",
                "default_max_parallel_runs",
                "min_free_bytes",
                "max_load_ratio",
            }
            changed_protected = sorted(
                name
                for name in protected
                if getattr(current, name) != getattr(updated, name)
            )
            if changed_protected and self.database.list_runs(
                server_id=server_id,
                statuses=tuple(NON_TERMINAL_RUN_STATUSES),
                limit=1,
            ):
                raise ServiceError(
                    409,
                    "server_configuration_in_use",
                    "Cancel or finish the server's queued and active runs before "
                    "changing connection or admission fields: "
                    + ", ".join(changed_protected),
                )
            self._ensure_unique_execution_target(updated, excluding_id=current.id)
            self.database.update_server(updated)
            if changes:
                self._clear_resource_refresh_state(server_id)
        except ServiceError:
            raise
        except (TypeError, ValueError) as exc:
            raise ServiceError(400, "invalid_server", str(exc)) from exc
        self.resource_monitor.wake()
        return self.serialize_server(updated)

    def delete_server(self, server_id: str) -> None:
        server = self._require_server(server_id)
        if server.target_kind is ExecutionTargetKind.LOCAL:
            raise ServiceError(
                409,
                "local_target_managed",
                "The built-in local compute target cannot be deleted; disable "
                "scheduling for it if it should not receive new work.",
            )
        active = self.database.list_runs(
            server_id=server_id,
            statuses=tuple(NON_TERMINAL_RUN_STATUSES),
            limit=1,
        )
        if active:
            raise ServiceError(
                409,
                "server_has_active_runs",
                "The server has a queued or active run and cannot be deleted.",
            )
        try:
            deleted = self.database.delete_server(server_id)
        except sqlite3.IntegrityError as exc:
            raise ServiceError(
                409,
                "server_has_run_history",
                "The server is referenced by run history and cannot be deleted.",
            ) from exc
        if not deleted:
            raise ServiceError(404, "server_not_found", "Server not found.")
        self._clear_resource_refresh_state(server_id, remove_lock=True)
        self.resource_monitor.wake()

    def probe_server(self, server_id: str) -> dict[str, Any]:
        """Run the transport's read-only probe and persist both detail and sample."""

        server = self._require_server(server_id)
        self._clear_resource_refresh_state(server_id)
        settings = self.get_settings()
        checked_at = utc_now()
        try:
            payload = self._probe_client(server).probe(
                server.remote_root,
                server.conda_env,
                timeout=float(settings["probe_timeout_seconds"]),
            )
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            record = {
                "ok": False,
                "checked_at": checked_at,
                "error": message,
                "error_type": type(exc).__name__,
            }
            try:
                persisted = self.database.record_server_probe(
                    server.id,
                    record,
                    probed_at=checked_at,
                    expected_target=server,
                )
            except (KeyError, ServerTargetChangedError) as changed:
                self.resource_monitor.wake()
                raise ServiceError(
                    409,
                    "probe_target_changed",
                    "Server connection settings changed while the environment "
                    "probe was running; the obsolete result was discarded.",
                ) from changed
            self.database.add_resource_sample(
                ResourceSample(
                    server_id=persisted.id,
                    target_identity=self._resource_target_identity(persisted),
                    sampled_at=checked_at,
                    reachable=False,
                    error=message,
                )
            )
            self.resource_monitor.wake()
            raise ServiceError(502, "probe_failed", message) from exc
        record = {"ok": True, "checked_at": checked_at, "report": payload}
        try:
            server = self.database.record_server_probe(
                server.id,
                record,
                fingerprint=_optional_text(payload.get("fingerprint")),
                probed_at=checked_at,
                expected_target=server,
            )
        except (KeyError, ServerTargetChangedError) as exc:
            self.resource_monitor.wake()
            raise ServiceError(
                409,
                "probe_target_changed",
                "Server connection settings changed while the environment probe "
                "was running; the obsolete result was discarded.",
            ) from exc
        self.database.add_resource_sample(
            self._resource_sample(server, payload, checked_at=checked_at)
        )
        self.resource_monitor.wake()
        return self.serialize_server(server)

    def refresh_server_resources(self, server_id: str) -> dict[str, Any]:
        """Explicitly sample utilization and persist the attempt.

        Browser polling uses the read-only GET endpoints.  This mutation is
        retained for an intentional "sample now" action and therefore performs
        SSH, records success/failure in SQLite, and never replaces the complete
        environment probe.
        """

        server = self._require_server(server_id)
        self._require_resource_monitor_identity(server)

        lock = self._resource_refresh_lock(server.id)
        with lock:
            cached = self._resource_refresh_cache_entry(server.id)
            if cached is not None:
                ok, value = cached
                if ok:
                    return dict(value)
                status_code, code, message = value
                raise ServiceError(status_code, code, message)

            try:
                sample = self._collect_resource_sample(server)
                result = self.serialize_resource_sample(sample)
            except ServiceError as exc:
                error = (exc.status_code, exc.code, exc.message)
                self._store_resource_refresh_cache(server.id, False, error)
                raise
            self._store_resource_refresh_cache(server.id, True, result)
            return dict(result)

    def list_server_resource_monitors(self) -> dict[str, Any]:
        settings = self.get_settings()
        servers = self.database.list_servers()
        sample_maps = self._resource_monitor_sample_maps(servers)
        return {
            "sampled_at": utc_now(),
            "resource_sampling_interval_seconds": settings[
                "resource_sampling_interval_seconds"
            ],
            "resource_history_retention_hours": settings[
                "resource_history_retention_hours"
            ],
            "servers": {
                server.id: self._serialize_resource_monitor(
                    server,
                    settings=settings,
                    samples=self._resource_monitor_samples_for(
                        server.id, sample_maps
                    ),
                )
                for server in servers
            },
        }

    def get_server_resource_history(
        self,
        server_id: str,
        *,
        since: str | None = None,
        until: str | None = None,
        after_id: int | None = None,
        limit: int = 5_000,
    ) -> dict[str, Any]:
        server = self._require_server(server_id)
        try:
            normalized_since = normalize_utc_iso(since) if since else None
            normalized_until = normalize_utc_iso(until) if until else None
        except (TypeError, ValueError) as exc:
            raise ServiceError(
                400,
                "invalid_resource_time_range",
                "Resource history timestamps must be timezone-aware ISO-8601 values.",
            ) from exc
        if (
            normalized_since is not None
            and normalized_until is not None
            and normalized_since > normalized_until
        ):
            raise ServiceError(
                400,
                "invalid_resource_time_range",
                "Resource history 'since' must not be later than 'until'.",
            )
        try:
            samples = self.database.list_resource_samples(
                server.id,
                since=normalized_since,
                until=normalized_until,
                after_id=after_id,
                background_only=True,
                target_identity=self._resource_target_identity(server),
                limit=limit,
            )
        except ValueError as exc:
            raise ServiceError(400, "invalid_resource_query", str(exc)) from exc
        if after_id is None:
            samples = list(reversed(samples))
        return {
            "server_id": server.id,
            "resource_monitor": self._serialize_resource_monitor(server),
            "samples": [self.serialize_resource_sample(sample) for sample in samples],
            "next_after_id": samples[-1].id if samples else None,
        }

    # -- Task discovery and inspection ---------------------------------

    def list_tasks(self) -> list[dict[str, Any]]:
        task_root = self._task_root()
        if not task_root.is_dir():
            return []
        result: list[dict[str, Any]] = []
        for path in sorted(task_root.glob("*.jsonl"), key=lambda item: item.name.lower()):
            if path.is_symlink() or not path.is_file():
                continue
            info = path.stat()
            result.append(
                {
                    "path": path.relative_to(self.paths.repository_root).as_posix(),
                    "name": path.stem.replace("_", " "),
                    "modified_at": _timestamp_from_epoch(info.st_mtime),
                    "size_bytes": info.st_size,
                }
            )
        return result

    def get_task_document(self, task_path: str) -> dict[str, Any]:
        path = self._resolve_task_path(task_path)
        try:
            payload = path.read_bytes()
            text = payload.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ServiceError(409, "task_read_failed", str(exc)) from exc
        inspection = self._inspect_task_model(path)
        return self._serialize_task_document(path, text, inspection)

    def create_task_document(self, values: Mapping[str, Any]) -> dict[str, Any]:
        filename = self._normalize_task_filename(values.get("filename"))
        target = self._new_task_path(filename)
        if target.exists() or target.is_symlink():
            raise ServiceError(409, "task_already_exists", "A task file with this name already exists.")
        return self._save_task_document(target, values.get("content"), create=True)

    def update_task_document(
        self,
        task_path: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        target = self._resolve_task_path(task_path)
        self._ensure_task_not_used_by_legacy_queue(
            target.relative_to(self.paths.repository_root).as_posix()
        )
        requested_filename = values.get("filename")
        if requested_filename is not None:
            filename = self._normalize_task_filename(requested_filename)
            if filename != target.name:
                raise ServiceError(
                    400,
                    "task_rename_not_supported",
                    "Save under a new task name, then delete the old file after queued legacy runs finish.",
                )
        return self._save_task_document(target, values.get("content"), create=False)

    def delete_task_document(self, task_path: str) -> dict[str, Any]:
        target = self._resolve_task_path(task_path)
        relative = target.relative_to(self.paths.repository_root).as_posix()
        self._ensure_task_not_used_by_legacy_queue(relative)
        tombstone = self.paths.trash / (
            f"task-{secrets.token_hex(8)}-{target.name}"
        )
        if tombstone.exists() or tombstone.is_symlink():
            raise ServiceError(409, "task_delete_conflict", "Task trash destination already exists.")
        try:
            os.replace(target, tombstone)
            fsync_directory(target.parent)
            fsync_directory(tombstone.parent)
        except OSError as exc:
            raise ServiceError(409, "task_delete_failed", str(exc)) from exc
        return {
            "deleted": True,
            "path": relative,
            "recoverable_path": str(tombstone),
        }

    def _save_task_document(
        self,
        target: Path,
        content: Any,
        *,
        create: bool,
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ServiceError(400, "invalid_task_content", "Task content must be UTF-8 text.")
        try:
            payload = content.encode("utf-8")
        except UnicodeError as exc:
            raise ServiceError(400, "invalid_task_content", "Task content must be valid UTF-8.") from exc
        from .tasking import MAX_INLINE_TASK_BYTES, inspect_batch_task_bytes

        if len(payload) > MAX_INLINE_TASK_BYTES:
            raise ServiceError(
                400,
                "task_too_large",
                f"Task content exceeds the {MAX_INLINE_TASK_BYTES}-byte limit.",
            )
        try:
            inspection = inspect_batch_task_bytes(payload, task_name=target.name)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ServiceError(400, "invalid_task", str(exc)) from exc
        root = self._task_root().resolve()
        if target.parent.resolve() != root or target.is_symlink():
            raise ServiceError(400, "invalid_task_path", "Task target path is unsafe.")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".relay-task-",
            suffix=".tmp",
            dir=root,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if create and (target.exists() or target.is_symlink()):
                raise ServiceError(409, "task_already_exists", "Task file appeared while saving.")
            if not create and (target.is_symlink() or not target.is_file()):
                raise ServiceError(409, "task_changed", "Task file changed while saving.")
            os.replace(temporary, target)
            target.chmod(0o600)
            fsync_directory(root)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        return self._serialize_task_document(target, content, inspection)

    def _serialize_task_document(
        self,
        path: Path,
        content: str,
        inspection: Any,
    ) -> dict[str, Any]:
        info = path.stat()
        return {
            "path": path.relative_to(self.paths.repository_root).as_posix(),
            "name": path.stem.replace("_", " "),
            "filename": path.name,
            "content": content,
            "sha256": inspection.task_file_sha256,
            "modified_at": _timestamp_from_epoch(info.st_mtime),
            "size_bytes": info.st_size,
            "total_tasks": len(inspection.tasks),
            "total_runs": inspection.total_runs,
            "total_evaluations": inspection.total_evals,
            "gpu_requested": inspection.gpu_requested,
            "warnings": [item.message for item in inspection.warnings],
            "entries": [
                {
                    "label": item.label,
                    "algorithms": list(item.algorithms),
                    "problem": item.problem,
                    "run_count": item.run_count,
                    "fe_per_run": item.max_evals_per_run,
                    "total_evaluations": item.total_evals,
                }
                for item in inspection.tasks
            ],
        }

    def _new_task_path(self, filename: str) -> Path:
        root = self._task_root()
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ServiceError(409, "task_root_unsafe", "Repository task directory is unsafe.")
        target = root / filename
        if target.parent.resolve() != root.resolve():
            raise ServiceError(400, "invalid_task_filename", "Task filename escapes tasks/.")
        return target

    @staticmethod
    def _normalize_task_filename(value: Any) -> str:
        if not isinstance(value, str):
            raise ServiceError(400, "invalid_task_filename", "Task filename is required.")
        filename = value.strip()
        if (
            not filename
            or filename in {".", ".."}
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or filename.startswith(".")
            or any(character in filename for character in ("\x00", "\r", "\n"))
            or Path(filename).suffix.lower() != ".jsonl"
        ):
            raise ServiceError(
                400,
                "invalid_task_filename",
                "Task filename must be one direct, visible tasks/*.jsonl filename.",
            )
        if len(filename.encode("utf-8")) > 240:
            raise ServiceError(400, "invalid_task_filename", "Task filename is too long.")
        return filename

    def _ensure_task_not_used_by_legacy_queue(self, task_path: str) -> None:
        offset = 0
        while True:
            runs = self.database.list_runs(
                statuses=tuple(NON_TERMINAL_RUN_STATUSES),
                limit=10_000,
                offset=offset,
            )
            for run in runs:
                if run.task_path != task_path:
                    continue
                inline = run.task_payload.get("inline_task")
                if not isinstance(inline, Mapping):
                    raise ServiceError(
                        409,
                        "task_used_by_unfrozen_run",
                        "An older queued run still depends on this task path; wait for it to freeze or cancel it before editing/deleting.",
                    )
            if len(runs) < 10_000:
                return
            offset += len(runs)

    def inspect_task(self, values: Mapping[str, Any]) -> dict[str, Any]:
        _, task_display_path, inspection, _ = self._load_task_source(values)
        server = self._require_server(str(values.get("server_id", "")))
        max_parallel = _positive_int(
            values.get("max_parallel_runs", 1), "max_parallel_runs", maximum=256
        )
        return self._serialize_task_inspection(
            task_display_path,
            inspection,
            server,
            max_parallel_runs=max_parallel,
        )

    # -- Runs -----------------------------------------------------------

    def list_runs(
        self, *, limit: int = 500, offset: int = 0, status: str = "all", search: str | None = None
    ) -> list[dict[str, Any]]:
        servers = {server.id: server for server in self.database.list_servers()}
        status_groups = {"all": None, "active": NON_TERMINAL_RUN_STATUSES,
                         "succeeded": [RunStatus.COMPLETED],
                         "failed": [RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.CANCELED]}
        if status not in status_groups:
            raise ServiceError(400, "invalid_run_filter", "Unknown run status filter.")
        runs = self.database.list_runs(limit=limit, offset=offset, statuses=status_groups[status], search=search,
                                       summary=True)
        metadata = self.database.run_metadata([run.id for run in runs])
        return [self.serialize_run(run, servers.get(run.server_id), metadata=metadata) for run in runs]

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        return self.serialize_run(run, self.database.get_server(run.server_id), detail=True)

    def rename_run(self, run_id: str, display_name: str) -> dict[str, Any]:
        """Rename only the user-facing label, preserving scientific provenance."""

        run = self._require_run(run_id)
        normalized = _normalize_run_display_name(display_name)
        updated = self.database.rename_run(run.id, normalized)
        return self.serialize_run(
            updated,
            self.database.get_server(updated.server_id),
            detail=True,
        )

    def create_run(self, values: Mapping[str, Any], *, run_id: str | None = None,
                   source_snapshot_id: str | None = None) -> dict[str, Any]:
        task_identity, task_display_path, inspection, inline_task = (
            self._load_task_source(values)
        )
        server = self._require_server(str(values.get("server_id", "")))
        if not server.enabled:
            raise ServiceError(409, "server_disabled", "The selected server is disabled.")
        max_parallel = _positive_int(
            values.get("max_parallel_runs", 1), "max_parallel_runs", maximum=256
        )
        poll_interval = _positive_float(
            values.get("refresh_interval_seconds", server.poll_interval_seconds),
            "refresh_interval_seconds",
            maximum=300,
        )
        admission_override = bool(values.get("admission_override", False))

        public_inspection = self._serialize_task_inspection(
            task_display_path,
            inspection,
            server,
            max_parallel_runs=max_parallel,
        )
        hard_blocked = list(public_inspection["hard_blocked_reasons"])
        overrideable = list(public_inspection["overrideable_reasons"])
        blocked = [*hard_blocked, *overrideable]
        if hard_blocked:
            raise ServiceError(
                409,
                "environment_admission_blocked",
                "Run admission cannot be overridden: " + "; ".join(hard_blocked),
            )
        if overrideable and not admission_override:
            raise ServiceError(
                409,
                "admission_blocked",
                "Run admission is blocked: " + "; ".join(overrideable),
            )

        # Imported lazily because registry population is intentionally a cold
        # path used only for task inspection/submission.
        from .tasking import build_task_provenance

        provenance = build_task_provenance(
            task_identity,
            forced_max_parallel_runs=max_parallel,
            inspection=inspection,
        )
        labels = [item.label for item in inspection.tasks]
        task_payload: dict[str, Any] = {
            "provenance": provenance,
            "inspection": public_inspection,
            "server_snapshot": self._server_snapshot(server),
            "admission": {
                "override": admission_override,
                "blocked_reasons": blocked,
                "inspected_at": utc_now(),
            },
        }
        if inline_task is not None:
            task_payload["inline_task"] = inline_task
        if source_snapshot_id is not None:
            task_payload["source_snapshot_id"] = source_snapshot_id
        task_name = labels[0] if len(labels) == 1 else Path(task_display_path).stem
        if len(labels) != 1 and inline_task is not None:
            task_name = "Pasted JSONL batch"
        requested_display_name = values.get("display_name")
        if requested_display_name is not None:
            task_name = _normalize_run_display_name(requested_display_name)
        run = Run(
            server_id=server.id,
            task_name=task_name,
            task_kind="batch",
            task_path=task_display_path,
            task_sha256=inspection.task_file_sha256,
            task_payload=task_payload,
            command_argv=(
                "python",
                "-m",
                "cli",
                "batch",
                "--task-file",
                task_display_path,
            ),
            max_parallel_runs=max_parallel,
            poll_interval_seconds=poll_interval,
            admission_override=admission_override,
            source_root=str(self.paths.repository_root),
            progress_current=0,
            progress_total=float(inspection.total_runs),
            progress_unit="run",
            progress_message="Waiting for a local source snapshot.",
        )
        if run_id is not None:
            run = replace(run, id=run_id)
        self.database.create_run(run)
        self.database.append_run_event(
            RunEvent(
                run_id=run.id,
                kind="audit",
                message="Run was added to the durable local queue.",
                data={
                    "stream": "audit",
                    "phase": "queued",
                    "actor": "local",
                    "admission_override": admission_override,
                    "blocked_reasons": blocked,
                },
            )
        )
        self._wake_controller(run.id)
        return self.serialize_run(run, server)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        if run.status in TERMINAL_RUN_STATUSES:
            return self.serialize_run(run, self.database.get_server(run.server_id))
        if self.controller is not None:
            try:
                self.controller.request_cancel(run.id)
                self.controller.wake()
            except Exception as exc:
                self.database.append_run_event(
                    RunEvent(
                        run_id=run.id,
                        kind="cancel",
                        level="error",
                        message=f"Controller cancellation request failed: {exc}",
                        data={"stream": "system", "phase": "cleanup"},
                    )
                )
            current = self.database.require_run(run.id)
            # Simple injected controllers may only acknowledge the request.
            # Persist the intent if the controller did not update the record.
            if current.status is run.status:
                current = self.database.set_run_status(
                    run.id,
                    RunStatus.CANCELING,
                    event_message="Cancellation was requested by the local user.",
                )
            run = current
        elif run.status is not RunStatus.CANCELING:
            run = self.database.set_run_status(
                run.id,
                RunStatus.CANCELING,
                event_message="Cancellation was requested by the local user.",
            )
        if self.controller is None and run.remote_pid is None:
            run = self.database.set_run_status(
                run.id,
                RunStatus.CANCELED,
                exit_code=130,
                event_message="Queued run was canceled before remote launch.",
            )
        return self.serialize_run(run, self.database.get_server(run.server_id))

    def recover_run(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        mode = run_recovery_mode(run)
        if mode is None:
            raise ServiceError(
                409,
                "recovery_not_available",
                "This run has no safely recoverable remote workspace.",
            )
        if self.controller is None or not hasattr(
            self.controller, "request_recovery"
        ):
            raise ServiceError(
                503,
                "controller_unavailable",
                "The local Relay controller is not available to resume this run.",
            )
        try:
            self.controller.request_recovery(run.id)
        except Exception as exc:
            code = str(getattr(exc, "code", "recovery_request_failed")).lower()
            status = 409 if code.startswith("recovery_") else 500
            raise ServiceError(status, code, str(exc)) from exc
        current = self.database.require_run(run.id)
        return self.serialize_run(
            current,
            self.database.get_server(current.server_id),
            detail=True,
        )

    def delete_run(self, run_id: str) -> None:
        """Move local results to trash, delete DB rows, then erase staged files."""

        run = self._require_run(run_id)
        if run.status not in TERMINAL_RUN_STATUSES:
            raise ServiceError(
                409,
                "run_not_terminal",
                "Only completed, failed, canceled, or interrupted runs can be deleted.",
            )
        if (
            run.cleanup_requested
            and run.remote_run_dir
            and not run.remote_cleanup_completed
        ):
            raise ServiceError(
                409,
                "remote_cleanup_pending",
                "Remote cleanup is still pending. Keep this record so Relay can "
                "safely retry deletion of the exact marker-owned workspace.",
            )

        linked_results = [
            item
            for item in self.database.list_result_imports()
            if item.get("source_run_id") == run.id
        ]
        for item in linked_results:
            configured = Path(str(item["managed_relative_dir"]))
            expected = Path("analyses") / "results" / str(item["id"])
            if configured == expected:
                # The result-library copy has its own lifecycle.  A missing or
                # damaged copy must not prevent deleting the independent run
                # record; the result entry remains visible as unavailable until
                # the user repairs or explicitly removes it.
                continue
            # Retry the startup migration once at the destructive boundary.
            self._migrate_legacy_result_imports()
            refreshed = self.database.get_result_import(str(item["id"]))
            if refreshed is None or Path(str(refreshed["managed_relative_dir"])) != expected:
                raise ServiceError(
                    409,
                    "legacy_result_not_detached",
                    "This run still owns a legacy imported result. Repair its independent result copy before deleting the run.",
                )

        source = self._local_artifact_path(run)
        staged: Path | None = None
        if source.exists() or source.is_symlink():
            if source.is_symlink():
                raise ServiceError(
                    409,
                    "unsafe_artifact_path",
                    "Refusing to delete a symlinked run artifact path.",
                )
            source = source.resolve()
            allowed_roots = (self.paths.runs.resolve(), self.paths.working.resolve())
            if source.parent not in allowed_roots or source.name != run.id:
                raise ServiceError(
                    409,
                    "unsafe_artifact_path",
                    "Run artifacts are not the exact runs/<run-id> or "
                    "working/<run-id> directory owned by this record.",
                )
            staged = self.paths.trash / f"{run.id}.deleting"
            if staged.exists() or staged.is_symlink():
                raise ServiceError(
                    409,
                    "delete_recovery_pending",
                    "A previous deletion tombstone still exists; restart Relay to reconcile it.",
                )
            os.replace(source, staged)

        try:
            deleted = self.database.delete_run(run.id)
            if not deleted:
                raise ServiceError(404, "run_not_found", "Run not found.")
        except BaseException:
            if staged is not None and staged.exists() and not source.exists():
                os.replace(staged, source)
            raise

        if staged is not None:
            _remove_staged_path(staged)

    def _reconcile_delete_tombstones(self) -> None:
        """Finish or roll back a run deletion interrupted by process termination."""

        for staged in sorted(self.paths.trash.glob("*.deleting")):
            if staged.is_symlink():
                continue
            run_id = staged.name.removesuffix(".deleting")
            if len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
                continue
            run = self.database.get_run(run_id)
            if run is None:
                _remove_staged_path(staged)
                continue
            destination = self._local_artifact_path(run)
            resolved_parent = destination.parent.resolve()
            allowed_parents = {self.paths.runs.resolve(), self.paths.working.resolve()}
            if resolved_parent not in allowed_parents or destination.name != run.id:
                continue
            if destination.exists() or destination.is_symlink():
                continue
            os.replace(staged, destination)

    def list_run_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 10_000
    ) -> list[dict[str, Any]]:
        run = self._require_run(run_id)
        return [
            self.serialize_event(event)
            for event in self.database.list_run_events(
                run.id, after_sequence=after_sequence, limit=limit
            )
        ]

    def list_run_samples(
        self, run_id: str, *, after_id: int | None = None, limit: int = 10_000
    ) -> list[dict[str, Any]]:
        run = self._require_run(run_id)
        samples = self.database.list_resource_samples(
            run.server_id, run_id=run.id, after_id=after_id, limit=limit
        )
        if after_id is None:
            samples = list(reversed(samples))
        return [self.serialize_sample(sample) for sample in samples]

    # -- Result previews and statistical analysis ---------------------

    def list_result_candidates(self) -> list[dict[str, Any]]:
        """List completed run-owned bundles without importing or opening them."""

        servers = {server.id: server for server in self.database.list_servers()}
        candidates: list[dict[str, Any]] = []
        offset = 0
        while True:
            runs = self.database.list_runs(
                statuses=(RunStatus.COMPLETED,),
                limit=10_000,
                offset=offset,
                summary=True,
            )
            for run in runs:
                if run.task_kind != "batch" or not run.transfer_verified:
                    continue
                imported = self.database.find_result_import_by_source_run(run.id)
                server = servers.get(run.server_id)
                bundle = self.paths.runs / run.id / "remote" / "results" / "bundle"
                available = (
                    not bundle.is_symlink()
                    and bundle.is_dir()
                    and all(
                        not (bundle / filename).is_symlink()
                        and (bundle / filename).is_file()
                        for filename in ("summary.sqlite", "trend.sqlite")
                    )
                )
                candidates.append(
                    {
                        "candidate_id": run.id,
                        "run_id": run.id,
                        "display_name": run.task_name,
                        "task_name": run.task_name,
                        "task_path": run.task_path or run.task_name,
                        "task_kind": run.task_kind,
                        "server_id": run.server_id,
                        "server_name": server.name if server is not None else "Unknown server",
                        "target_kind": server.target_kind.value if server is not None else "ssh",
                        "created_at": run.created_at,
                        "completed_at": run.finished_at,
                        "completed_units": _number_for_json(run.progress_current),
                        "total_units": _number_for_json(run.progress_total or run.progress_current),
                        "bundle_relative_dir": (
                            Path("runs") / run.id / "remote" / "results" / "bundle"
                        ).as_posix(),
                        "available": available,
                        "imported": imported is not None,
                        "imported_result_id": imported["id"] if imported is not None else None,
                    }
                )
            if len(runs) < 10_000:
                break
            offset += len(runs)
        return candidates

    def list_result_catalog(self) -> list[dict[str, Any]]:
        return self._call_result_workspace(self.result_workspace.list_result_catalog)

    def import_result_bundle(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Copy one external complete bundle or exported ZIP into the result library."""

        raw_source = values.get("source_path")
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise ServiceError(
                400,
                "invalid_result_import_path",
                "source_path must be a non-empty absolute local bundle path.",
            )
        source_text = raw_source.strip()
        if len(source_text) > 4_096 or any(
            character in source_text for character in ("\x00", "\r", "\n")
        ):
            raise ServiceError(
                400,
                "invalid_result_import_path",
                "source_path is too long or contains control characters.",
            )
        source = Path(source_text).expanduser()
        requested_name = values.get("display_name")
        display_name = _normalize_run_display_name(
            requested_name
            if requested_name is not None
            else source.name or "Imported result"
        )

        if source.suffix.lower() == ".zip":
            archive = _safe_external_result_archive(source, self.paths.data_root)
            extracted_root = Path(
                tempfile.mkdtemp(prefix="result-archive-", dir=self.paths.working)
            )
            extracted_root.chmod(0o700)
            try:
                extracted_bundle = _extract_single_result_archive(
                    archive,
                    extracted_root,
                )
                return self._import_result_copy(
                    extracted_bundle,
                    display_name=display_name,
                    source_kind="external",
                    source_run_id=None,
                    source_metadata={
                        "source_path": str(archive),
                        "archive_format": "edolab-results-v1",
                    },
                    allow_managed_source=True,
                )
            finally:
                if extracted_root.exists() and not extracted_root.is_symlink():
                    _remove_staged_path(extracted_root)
                    fsync_directory(self.paths.working)

        return self._import_result_copy(
            source,
            display_name=display_name,
            source_kind="external",
            source_run_id=None,
            source_metadata={"source_path": str(source.expanduser().resolve())},
            allow_managed_source=False,
        )

    def import_uploaded_result_archive(
        self,
        archive_path: Path,
        *,
        original_filename: str,
    ) -> dict[str, Any]:
        """Validate an uploaded ZIP and import only content-unique Runsets."""

        filename = Path(str(original_filename).replace("\\", "/")).name.strip()
        if (
            not filename
            or len(filename) > 240
            or not filename.lower().endswith(".zip")
            or any(character in filename for character in ("\x00", "\r", "\n"))
        ):
            raise ServiceError(
                400,
                "invalid_result_upload_filename",
                "Uploaded result must have a safe .zip filename.",
            )
        extracted_root = Path(
            tempfile.mkdtemp(prefix="result-upload-extract-", dir=self.paths.working)
        )
        extracted_root.chmod(0o700)
        imported_results: list[dict[str, Any]] = []
        imported_runsets: list[dict[str, Any]] = []
        skipped_results: list[dict[str, Any]] = []
        failed_results: list[dict[str, Any]] = []
        try:
            archive_items = _extract_result_archive(archive_path, extracted_root)
            with self._result_import_lock:
                existing_by_fingerprint = self._result_runset_fingerprint_index()

                for index, item in enumerate(archive_items, start=1):
                    manifest_item = item["manifest_item"]
                    display_name = _archive_result_display_name(
                        manifest_item,
                        filename=filename,
                        index=index,
                        total=len(archive_items),
                    )
                    try:
                        source_runsets = _result_bundle_runset_records(item["bundle"])
                    except ResultBundleValidationError as exc:
                        failed_results.append(
                            {
                                "display_name": display_name,
                                "runset_uid": None,
                                "code": "invalid_result_bundle",
                                "message": str(exc),
                            }
                        )
                        continue

                    unique_runsets: list[dict[str, Any]] = []
                    selected_in_bundle: dict[str, dict[str, Any]] = {}
                    for runset in source_runsets:
                        duplicate = existing_by_fingerprint.get(runset["fingerprint"])
                        if duplicate is None:
                            duplicate = selected_in_bundle.get(runset["fingerprint"])
                        if duplicate is not None:
                            skipped_results.append(
                                {
                                    "display_name": runset["display_name"],
                                    "runset_uid": runset["runset_uid"],
                                    "reason": "duplicate",
                                    "duplicate_of_result_id": duplicate.get("result_id"),
                                    "duplicate_of_display_name": duplicate.get(
                                        "result_display_name", display_name
                                    ),
                                    "duplicate_of_runset_uid": duplicate["runset_uid"],
                                }
                            )
                            continue
                        unique_runsets.append(runset)
                        selected_in_bundle[runset["fingerprint"]] = runset

                    if not unique_runsets:
                        continue

                    import_source = item["bundle"]
                    if len(unique_runsets) != len(source_runsets):
                        import_source = extracted_root / f"subset-{index}"
                        _create_result_runset_subset(
                            item["bundle"],
                            import_source,
                            {str(runset["runset_uid"]) for runset in unique_runsets},
                        )
                    try:
                        imported = self._import_result_copy(
                            import_source,
                            display_name=display_name,
                            source_kind="external",
                            source_run_id=None,
                            source_metadata={
                                "upload_filename": filename,
                                "archive_format": "edolab-results-v1",
                                "archive_result_id": manifest_item.get("result_id"),
                                "archive_prefix": item["prefix"],
                                "source_runset_count": len(source_runsets),
                                "imported_runset_count": len(unique_runsets),
                                "skipped_duplicate_runset_count": (
                                    len(source_runsets) - len(unique_runsets)
                                ),
                            },
                            allow_managed_source=True,
                        )
                    except ServiceError as exc:
                        failed_results.extend(
                            {
                                "display_name": runset["display_name"],
                                "runset_uid": runset["runset_uid"],
                                "code": exc.code,
                                "message": exc.message,
                            }
                            for runset in unique_runsets
                        )
                        continue
                    imported_results.append(imported)
                    imported_result_id = str(imported["result"]["result_id"])
                    for runset in unique_runsets:
                        imported_item = {
                            "display_name": runset["display_name"],
                            "runset_uid": runset["runset_uid"],
                            "result_id": imported_result_id,
                        }
                        imported_runsets.append(imported_item)
                        existing_by_fingerprint[runset["fingerprint"]] = {
                            **runset,
                            "result_id": imported_result_id,
                            "result_display_name": display_name,
                        }
        finally:
            if extracted_root.exists() and not extracted_root.is_symlink():
                _remove_staged_path(extracted_root)
                fsync_directory(self.paths.working)

        return {
            "count_unit": "runset",
            "imported_count": len(imported_runsets),
            "skipped_count": len(skipped_results),
            "failed_count": len(failed_results),
            "total_count": (
                len(imported_runsets) + len(skipped_results) + len(failed_results)
            ),
            "imported": imported_results,
            "imported_runsets": imported_runsets,
            "skipped": skipped_results,
            "failed": failed_results,
        }

    def _result_runset_fingerprint_index(self) -> dict[str, dict[str, Any]]:
        """Index every imported Runset by its complete content fingerprint."""

        existing_by_fingerprint: dict[str, dict[str, Any]] = {}
        for existing in self.database.list_result_imports():
            existing_result_id = str(existing["result_id"])
            existing_bundle = self._call_result_workspace(
                self.result_workspace.result_bundle_path,
                existing_result_id,
            )
            try:
                existing_runsets = _result_bundle_runset_records(existing_bundle)
            except ResultBundleValidationError as exc:
                raise ServiceError(
                    409,
                    "stored_result_runset_scan_failed",
                    f"Imported result {existing_result_id} could not be checked by Runset: {exc}",
                ) from exc
            for runset in existing_runsets:
                existing_by_fingerprint.setdefault(
                    runset["fingerprint"],
                    {
                        **runset,
                        "result_id": existing_result_id,
                        "result_display_name": existing["display_name"],
                    },
                )
        return existing_by_fingerprint

    def import_result_candidate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        candidate_id = values.get("candidate_id", values.get("run_id"))
        run_id = _required_bounded_text(candidate_id, "candidate_id", maximum=200)
        run = self._require_run(run_id)
        if (
            run.task_kind != "batch"
            or run.status is not RunStatus.COMPLETED
            or not run.transfer_verified
        ):
            raise ServiceError(
                409,
                "run_result_not_importable",
                "Only completed batch candidates with checksum-verified local results can be imported.",
            )
        source = self.result_workspace.verified_run_bundle_path(run.id)
        display_name = _normalize_run_display_name(
            values.get("display_name") if values.get("display_name") is not None else run.task_name
        )
        subset_root: Path | None = None
        try:
            with self._result_import_lock:
                existing = self.database.find_result_import_by_source_run(run.id)
                if existing is not None:
                    raise ServiceError(
                        409,
                        "run_result_already_imported",
                        f"This run is already present in the result library as {existing['display_name']}.",
                    )
                try:
                    source_runsets = _result_bundle_runset_records(source)
                except ResultBundleValidationError as exc:
                    raise ServiceError(
                        400,
                        "invalid_result_bundle",
                        str(exc),
                    ) from exc
                existing_by_fingerprint = self._result_runset_fingerprint_index()
                unique_runsets: list[dict[str, Any]] = []
                skipped_runsets: list[dict[str, Any]] = []
                selected_in_candidate: dict[str, dict[str, Any]] = {}
                for runset in source_runsets:
                    duplicate = existing_by_fingerprint.get(runset["fingerprint"])
                    if duplicate is None:
                        duplicate = selected_in_candidate.get(runset["fingerprint"])
                    if duplicate is not None:
                        skipped_runsets.append(
                            {
                                "display_name": runset["display_name"],
                                "runset_uid": runset["runset_uid"],
                                "reason": "duplicate",
                                "duplicate_of_result_id": duplicate.get("result_id"),
                                "duplicate_of_display_name": duplicate.get(
                                    "result_display_name", display_name
                                ),
                                "duplicate_of_runset_uid": duplicate["runset_uid"],
                            }
                        )
                        continue
                    unique_runsets.append(runset)
                    selected_in_candidate[runset["fingerprint"]] = runset

                imported_results: list[dict[str, Any]] = []
                imported_runsets: list[dict[str, Any]] = []
                if unique_runsets:
                    import_source = source
                    if len(unique_runsets) != len(source_runsets):
                        subset_root = Path(
                            tempfile.mkdtemp(
                                prefix="candidate-runset-subset-",
                                dir=self.paths.working,
                            )
                        )
                        subset_root.chmod(0o700)
                        import_source = subset_root / "bundle"
                        _create_result_runset_subset(
                            source,
                            import_source,
                            {str(runset["runset_uid"]) for runset in unique_runsets},
                        )
                    imported = self._import_result_copy(
                        import_source,
                        display_name=display_name,
                        source_kind="run",
                        source_run_id=run.id,
                        source_metadata={
                            "task_path": run.task_path,
                            "task_sha256": run.task_sha256,
                            "snapshot_sha256": run.snapshot_sha256,
                            "run_created_at": run.created_at,
                            "run_finished_at": run.finished_at,
                            "source_runset_count": len(source_runsets),
                            "imported_runset_count": len(unique_runsets),
                            "skipped_duplicate_runset_count": len(skipped_runsets),
                        },
                        allow_managed_source=True,
                    )
                    imported_results.append(imported)
                    imported_result_id = str(imported["result"]["result_id"])
                    imported_runsets.extend(
                        {
                            "display_name": runset["display_name"],
                            "runset_uid": runset["runset_uid"],
                            "result_id": imported_result_id,
                        }
                        for runset in unique_runsets
                    )
        finally:
            if subset_root is not None and subset_root.exists() and not subset_root.is_symlink():
                _remove_staged_path(subset_root)
                fsync_directory(self.paths.working)

        return {
            "count_unit": "runset",
            "imported_count": len(imported_runsets),
            "skipped_count": len(skipped_runsets),
            "failed_count": 0,
            "total_count": len(source_runsets),
            "imported": imported_results,
            "imported_runsets": imported_runsets,
            "skipped": skipped_runsets,
            "failed": [],
        }

    def import_result_from_run(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Compatibility alias; new clients import from the candidate list."""

        outcome = self.import_result_candidate(values)
        if outcome["imported"]:
            return outcome["imported"][0]
        raise ServiceError(
            409,
            "run_result_runsets_already_imported",
            "Every Runset in this run already exists in the result library.",
        )

    def _import_result_copy(
        self,
        source: Path,
        *,
        display_name: str,
        source_kind: str,
        source_run_id: str | None,
        source_metadata: Mapping[str, Any],
        allow_managed_source: bool,
    ) -> dict[str, Any]:
        result_id = uuid4().hex
        now = utc_now()
        staging = self.paths.working / f"result-{result_id}"
        final = self.paths.imported_results / result_id
        if staging.exists() or staging.is_symlink() or final.exists() or final.is_symlink():
            raise ServiceError(409, "result_import_conflict", "A managed result directory already exists.")
        staging.mkdir(mode=0o700)
        fsync_directory(staging.parent)
        try:
            imported = self.result_import_workspace.import_bundle(
                source,
                staging,
                allow_managed_source=allow_managed_source,
            )
        except ResultImportWorkspaceError as exc:
            if staging.exists() and not staging.is_symlink():
                _remove_staged_path(staging)
                fsync_directory(self.paths.working)
            raise ServiceError(400, exc.code, exc.message) from exc
        staged_bundle = staging / "remote" / "results" / "bundle"
        try:
            os.replace(staged_bundle, final)
            _remove_staged_path(staging)
            fsync_directory(final)
            fsync_directory(self.paths.imported_results)
            fsync_directory(self.paths.working)
        except OSError as exc:
            if staging.exists() and not staging.is_symlink():
                _remove_staged_path(staging)
            if final.exists() and not final.is_symlink():
                _remove_staged_path(final)
            raise ServiceError(409, "result_import_publish_failed", str(exc)) from exc
        files = {item.filename: item for item in imported.files}
        validation = imported.as_dict()["validation"]
        record = {
            "id": result_id,
            "display_name": display_name,
            "source_kind": source_kind,
            "source_run_id": source_run_id,
            "source_metadata": dict(source_metadata),
            "bundle_id": imported.validation.bundle_id,
            "managed_relative_dir": f"analyses/results/{result_id}",
            "validation": validation,
            "summary_size_bytes": files["summary.sqlite"].size_bytes,
            "summary_sha256": files["summary.sqlite"].sha256,
            "trend_size_bytes": files["trend.sqlite"].size_bytes,
            "trend_sha256": files["trend.sqlite"].sha256,
            "imported_at": now,
        }
        try:
            stored = self.database.create_result_import(record)
            preview = self.preview_run_results(result_id)
        except sqlite3.IntegrityError as exc:
            self.database.delete_result_import(result_id)
            if final.exists() and not final.is_symlink():
                _remove_staged_path(final)
            if source_run_id and self.database.find_result_import_by_source_run(source_run_id):
                raise ServiceError(
                    409,
                    "run_result_already_imported",
                    "This run was imported concurrently by another request.",
                ) from exc
            raise ServiceError(
                409,
                "result_import_conflict",
                "The imported result conflicts with an existing library record.",
            ) from exc
        except BaseException:
            self.database.delete_result_import(result_id)
            if final.exists() and not final.is_symlink():
                _remove_staged_path(final)
            raise
        catalog_item = next(
            (item for item in self.list_result_catalog() if item["result_id"] == result_id),
            None,
        )
        return {"result": stored, "catalog_item": catalog_item, "preview": preview}

    def preview_run_results(self, run_id: str) -> dict[str, Any]:
        return self._call_result_workspace(self.result_workspace.preview_run, run_id)

    def preview_runset_samples(
        self,
        run_id: str,
        runset_uid: str,
        *,
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.preview_runset_samples,
            run_id,
            runset_uid,
            offset=offset,
            limit=limit,
        )

    def preview_runset_trend(
        self,
        run_id: str,
        runset_uid: str,
        metric_name: str,
        *,
        offset: int = 0,
        limit: int = 1_000,
    ) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.preview_trend,
            run_id,
            runset_uid,
            metric_name,
            offset=offset,
            limit=limit,
        )

    def delete_result(self, result_id: str) -> None:
        item = self.database.get_result_import(str(result_id))
        if item is None:
            raise ServiceError(404, "result_not_found", "Imported result not found.")
        expected = self.paths.imported_results / str(result_id)
        configured = Path(str(item["managed_relative_dir"]))
        legacy_owner_id = str(item.get("source_run_id") or result_id)
        legacy_run_path = (
            Path("runs")
            / legacy_owner_id
            / "remote"
            / "results"
            / "bundle"
        )
        if configured == legacy_run_path:
            # A migrated synthetic run may still be the physical owner.  Drop
            # only the result-library record; deleting that run remains a
            # separate explicit action until startup migration copies it.
            if not self.database.delete_result_import(str(result_id)):
                raise ServiceError(404, "result_not_found", "Imported result not found.")
            return
        if configured != Path("analyses") / "results" / str(result_id):
            raise ServiceError(409, "unsafe_result_path", "Imported result path is unsafe.")
        tombstone: Path | None = None
        if expected.exists() or expected.is_symlink():
            if expected.is_symlink() or not expected.is_dir():
                raise ServiceError(409, "unsafe_result_path", "Imported result path is unsafe.")
            tombstone = self.paths.trash / f"result-{result_id}-{uuid4().hex}.trash"
            os.replace(expected, tombstone)
        try:
            deleted = self.database.delete_result_import(str(result_id))
            if not deleted:
                raise ServiceError(404, "result_not_found", "Imported result not found.")
        except BaseException:
            if tombstone is not None and tombstone.exists() and not expected.exists():
                os.replace(tombstone, expected)
            raise
        if tombstone is not None:
            _remove_staged_path(tombstone)

    def export_results(self, result_ids: list[str]) -> tuple[Path, str]:
        normalized = list(dict.fromkeys(str(item) for item in result_ids if str(item)))
        if not normalized:
            raise ServiceError(400, "result_export_empty", "Select at least one result.")
        if len(normalized) > 100:
            raise ServiceError(400, "result_export_too_large", "At most 100 results may be exported together.")
        descriptor, raw_path = tempfile.mkstemp(
            prefix="result-export-",
            suffix=".zip",
            dir=self.paths.working,
        )
        os.close(descriptor)
        archive_path = Path(raw_path)
        archive_path.chmod(0o600)
        manifest: dict[str, Any] = {"schema_version": 1, "results": []}
        try:
            with zipfile.ZipFile(
                archive_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as archive:
                for result_id in normalized:
                    record = self.database.get_result_import(result_id)
                    if record is None:
                        raise ServiceError(404, "result_not_found", f"Imported result not found: {result_id}")
                    bundle = self.result_workspace.result_bundle_path(result_id, force_verify=True)
                    prefix = result_id if len(normalized) > 1 else "result"
                    for filename in ("summary.sqlite", "trend.sqlite"):
                        archive.write(bundle / filename, f"{prefix}/{filename}")
                    manifest["results"].append(
                        {
                            "result_id": result_id,
                            "display_name": record["display_name"],
                            "bundle_id": record["bundle_id"],
                            "source_kind": record["source_kind"],
                            "imported_at": record["imported_at"],
                            "files": {
                                "summary.sqlite": {
                                    "size_bytes": record["summary_size_bytes"],
                                    "sha256": record["summary_sha256"],
                                },
                                "trend.sqlite": {
                                    "size_bytes": record["trend_size_bytes"],
                                    "sha256": record["trend_sha256"],
                                },
                            },
                        }
                    )
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                )
            fsync_directory(self.paths.working)
        except BaseException:
            archive_path.unlink(missing_ok=True)
            fsync_directory(self.paths.working)
            raise
        return archive_path, f"edolab-results-{utc_now()[:10]}.zip"

    def export_runsets(
        self,
        selections: list[Mapping[str, Any]],
    ) -> tuple[Path, str]:
        """Export selected Runsets, preserving a valid standard result ZIP."""

        grouped: dict[str, list[str]] = {}
        for selection in selections:
            result_id = _required_bounded_text(
                selection.get("result_id"),
                "result_id",
                maximum=200,
            )
            raw_uids = selection.get("runset_uids")
            if not isinstance(raw_uids, list) or not raw_uids:
                raise ServiceError(
                    400,
                    "result_export_empty",
                    "Select at least one Runset to export.",
                )
            selected = grouped.setdefault(result_id, [])
            for raw_uid in raw_uids:
                runset_uid = _required_bounded_text(
                    raw_uid,
                    "runset_uid",
                    maximum=200,
                )
                if runset_uid not in selected:
                    selected.append(runset_uid)
        runset_count = sum(len(items) for items in grouped.values())
        if not runset_count:
            raise ServiceError(
                400,
                "result_export_empty",
                "Select at least one Runset to export.",
            )
        if len(grouped) > 100 or runset_count > 10_000:
            raise ServiceError(
                400,
                "result_export_too_large",
                "At most 100 results and 10,000 Runsets may be exported together.",
            )

        descriptor, raw_path = tempfile.mkstemp(
            prefix="runset-export-",
            suffix=".zip",
            dir=self.paths.working,
        )
        os.close(descriptor)
        archive_path = Path(raw_path)
        archive_path.chmod(0o600)
        subset_root = Path(
            tempfile.mkdtemp(prefix="runset-export-subsets-", dir=self.paths.working)
        )
        subset_root.chmod(0o700)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "export_scope": "runsets",
            "runset_count": runset_count,
            "results": [],
        }
        try:
            with zipfile.ZipFile(
                archive_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as archive:
                for index, (result_id, runset_uids) in enumerate(grouped.items(), start=1):
                    record = self.database.get_result_import(result_id)
                    if record is None:
                        raise ServiceError(
                            404,
                            "result_not_found",
                            f"Imported result not found: {result_id}",
                        )
                    source_bundle = self._call_result_workspace(
                        self.result_workspace.result_bundle_path,
                        result_id,
                        force_verify=True,
                    )
                    available = {
                        str(item["runset_uid"])
                        for item in _result_bundle_runset_records(source_bundle)
                    }
                    unknown = [uid for uid in runset_uids if uid not in available]
                    if unknown:
                        raise ServiceError(
                            400,
                            "result_export_unknown_runset",
                            f"Result {result_id} does not contain Runset {unknown[0]}.",
                        )
                    if len(runset_uids) == len(available):
                        export_bundle = source_bundle
                    else:
                        export_bundle = subset_root / f"bundle-{index}"
                        _create_result_runset_subset(
                            source_bundle,
                            export_bundle,
                            set(runset_uids),
                        )
                    prefix = result_id if len(grouped) > 1 else RESULT_ARCHIVE_SINGLE_PREFIX
                    files: dict[str, dict[str, Any]] = {}
                    for filename in ("summary.sqlite", "trend.sqlite"):
                        path = export_bundle / filename
                        archive.write(path, f"{prefix}/{filename}")
                        files[filename] = {
                            "size_bytes": path.stat().st_size,
                            "sha256": _sha256_path(path),
                        }
                    manifest["results"].append(
                        {
                            "result_id": result_id,
                            "display_name": record["display_name"],
                            "bundle_id": record["bundle_id"],
                            "source_kind": record["source_kind"],
                            "imported_at": record["imported_at"],
                            "runset_count": len(runset_uids),
                            "runset_uids": runset_uids,
                            "files": files,
                        }
                    )
                archive.writestr(
                    RESULT_ARCHIVE_MANIFEST_NAME,
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                )
            fsync_file(archive_path)
            fsync_directory(self.paths.working)
        except BaseException:
            archive_path.unlink(missing_ok=True)
            fsync_directory(self.paths.working)
            raise
        finally:
            if subset_root.exists() and not subset_root.is_symlink():
                _remove_staged_path(subset_root)
                fsync_directory(self.paths.working)
        return archive_path, f"edolab-runsets-{utc_now()[:10]}.zip"

    # -- Result plots --------------------------------------------------

    def list_plot_sources(self) -> list[dict[str, Any]]:
        """Return verified runsets as lightweight, immutable plot sources."""

        sources: list[dict[str, Any]] = []
        for catalog_item in self.list_result_catalog():
            result_id = str(catalog_item["result_id"])
            preview = self._call_result_workspace(
                self.result_workspace.preview_run,
                result_id,
            )
            for runset in preview.get("runsets", []):
                if not isinstance(runset, Mapping):
                    continue
                runset_uid = str(runset.get("runset_uid") or "")
                if not runset_uid:
                    continue
                scalar_metrics = [
                    str(item.get("name"))
                    for item in runset.get("scalar_metrics", [])
                    if isinstance(item, Mapping) and item.get("name")
                ]
                trend_axes = {
                    str(item.get("name")): str(item.get("axis") or "fe")
                    for item in runset.get("trend_metrics", [])
                    if isinstance(item, Mapping) and item.get("name")
                }
                expected = int(runset.get("expected_runs") or 0)
                completed = int(runset.get("completed_runs") or 0)
                status = str(runset.get("status") or "unknown")
                available = (
                    expected > 0
                    and completed == expected
                    and status.lower() in {"completed", "complete", "succeeded", "finished"}
                )
                sources.append(
                    {
                        "id": self._plot_source_id(result_id, runset_uid),
                        "result_id": result_id,
                        "run_id": result_id,
                        "runset_uid": runset_uid,
                        "label": str(runset.get("label") or runset_uid),
                        "algorithm_name": str(runset.get("algorithm_name") or ""),
                        "problem_name": str(runset.get("problem_name") or ""),
                        "algorithm_params": dict(runset.get("algorithm_params") or {}),
                        "task_name": str(preview.get("task_name") or ""),
                        "completed_runs": completed,
                        "expected_runs": expected,
                        "status": status,
                        "available": available,
                        "detail": (
                            f"result {result_id[:12]} · bundle "
                            f"{str(preview.get('validation', {}).get('bundle_id') or '')[:12]}"
                        ),
                        "scalar_metrics": sorted(set(scalar_metrics)),
                        "trend_metrics": sorted(trend_axes),
                        "trend_metric_axes": trend_axes,
                    }
                )
        return sources

    def load_plot_trend_series(
        self,
        source_id: str,
        metric_name: str,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        max_points: int | None = None,
    ) -> dict[str, Any]:
        run_id, runset_uid = self._parse_plot_source_id(source_id)
        metric = _required_bounded_text(metric_name, "metric_name", maximum=256)
        point_limit = (
            self.plot_workspace.max_points_per_series
            if max_points is None
            else max_points
        )
        if (
            isinstance(point_limit, bool)
            or not isinstance(point_limit, int)
            or point_limit < 1
            or point_limit > self.plot_workspace.max_points_per_series
        ):
            raise ServiceError(
                400,
                "invalid_plot_point_limit",
                "Trend max_points must be a positive integer no greater than "
                f"{self.plot_workspace.max_points_per_series}.",
            )
        return self._call_result_workspace(
            self.result_workspace.load_plot_trend_series,
            run_id,
            runset_uid,
            metric,
            max_points=point_limit,
            x_min=x_min,
            x_max=x_max,
        )

    def load_plot_boxplot_series(
        self,
        source_id: str,
        metric_name: str,
    ) -> dict[str, Any]:
        run_id, runset_uid = self._parse_plot_source_id(source_id)
        metric = _required_bounded_text(metric_name, "metric_name", maximum=256)
        offset = 0
        total: int | None = None
        values: list[float] = []
        while total is None or offset < total:
            page = self._call_result_workspace(
                self.result_workspace.preview_runset_samples,
                run_id,
                runset_uid,
                offset=offset,
                limit=1_000,
            )
            if total is None:
                total = int(page.get("total") or 0)
                expected = int(page.get("expected_runs") or 0)
                completed = int(page.get("completed_runs") or 0)
                if (
                    total <= 0
                    or total > self.plot_workspace.max_points_per_series
                    or completed != expected
                    or total != completed
                ):
                    raise ServiceError(
                        409,
                        "plot_runset_incomplete",
                        "Boxplot sources must be complete and remain within the plot point limit.",
                    )
            runs = page.get("runs")
            if not isinstance(runs, list) or not runs:
                raise ServiceError(
                    409,
                    "plot_samples_incomplete",
                    "Stored scalar samples ended before the declared run count.",
                )
            for item in runs:
                raw = _mapping_get(_mapping_get(item, "metrics"), metric)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                    raise ServiceError(
                        409,
                        "plot_metric_missing",
                        f"Metric {metric!r} is missing or non-finite for a stored run.",
                    )
                values.append(float(raw))
            offset += len(runs)
        return {"values": values}

    def render_plot(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Render one interactive result plot in the local Relay process.

        Result plots are read-only views over an already verified result
        bundle.  They are deliberately not scientific experiment runs: no
        execution target, source snapshot, admission check, queue record, or
        controller wake-up is involved.
        """

        operation = str(values.get("kind") or "").strip().lower()
        if operation not in {"trend", "boxplot"}:
            raise ServiceError(400, "invalid_plot_kind", "Choose trend or boxplot.")
        request_values = {
            key: values[key]
            for key in (
                "metric_name",
                "series",
                "formats",
                "title",
                "x_min",
                "x_max",
                "legend_position",
                "y_scale",
            )
            if key in values
        }
        try:
            return self.plot_workspace.export_plot(request_values, operation)
        except (PlotWorkspaceError, TypeError, ValueError) as exc:
            raise ServiceError(400, "invalid_plot_request", str(exc)) from exc

    def render_plot_preview(self, values: Mapping[str, Any]) -> PlotPreview:
        """Render a transient PNG preview without publishing an artifact."""

        operation = str(values.get("kind") or "").strip().lower()
        if operation not in {"trend", "boxplot"}:
            raise ServiceError(400, "invalid_plot_kind", "Choose trend or boxplot.")
        request_values = {
            key: values[key]
            for key in (
                "metric_name",
                "series",
                "title",
                "x_min",
                "x_max",
                "legend_position",
                "y_scale",
            )
            if key in values
        }
        try:
            return self.plot_workspace.preview_plot(request_values, operation)
        except (PlotWorkspaceError, TypeError, ValueError) as exc:
            raise ServiceError(400, "invalid_plot_request", str(exc)) from exc

    def resolve_interactive_plot_artifact(
        self,
        storage_key: str,
    ) -> tuple[Path, str]:
        """Resolve a locally rendered interactive plot artifact."""

        try:
            path = self.plot_workspace.resolve_artifact(storage_key)
        except FileNotFoundError as exc:
            raise ServiceError(404, "plot_artifact_not_found", str(exc)) from exc
        except PlotWorkspaceError as exc:
            raise ServiceError(400, "invalid_plot_artifact_path", str(exc)) from exc
        media_types = {
            ".png": "image/png",
            ".pdf": "application/pdf",
            ".svg": "image/svg+xml",
        }
        media_type = media_types.get(path.suffix.lower())
        if media_type is None:
            raise ServiceError(404, "plot_artifact_not_found", "Unsupported plot artifact.")
        return path, media_type

    def get_plot_run_result(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        if run.task_kind != "plot":
            raise ServiceError(404, "plot_run_not_found", "This run is not a result plot.")
        serialized = self.serialize_run(run, self.database.get_server(run.server_id))
        if run.status is not RunStatus.COMPLETED:
            return {
                "execution": "failed" if run.status in TERMINAL_RUN_STATUSES else "queued",
                "run_id": run.id,
                "status": serialized["status"],
                "message": run.error_message or run.progress_message,
                "run": serialized,
            }
        bundle = self._plot_bundle_path(run)
        result_path = bundle / "workload-result.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise ServiceError(409, "plot_result_missing", "Verified plot result is missing.")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServiceError(409, "plot_result_invalid", str(exc)) from exc
        if (
            not isinstance(result, Mapping)
            or result.get("workload_kind") != "plot"
            or result.get("spec_sha256") != run.task_sha256
        ):
            raise ServiceError(409, "plot_result_identity_mismatch", "Plot result does not match the frozen run.")
        scientific = result.get("result")
        if not isinstance(scientific, Mapping):
            raise ServiceError(409, "plot_result_invalid", "Plot result has no scientific payload.")
        enriched = dict(scientific)
        artifacts = scientific.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ServiceError(409, "plot_result_invalid", "Plot result declares no export artifacts.")
        enriched_artifacts: dict[str, Any] = {}
        descriptors_by_path: dict[str, dict[str, Any]] = {}
        for fmt, descriptor in artifacts.items():
            if not isinstance(descriptor, Mapping):
                raise ServiceError(409, "plot_result_invalid", "Plot artifact descriptor is invalid.")
            enriched_descriptor = self._plot_artifact_descriptor(run, bundle, descriptor)
            enriched_artifacts[str(fmt)] = enriched_descriptor
            descriptors_by_path[str(descriptor.get("path") or "")] = enriched_descriptor
        enriched["artifacts"] = enriched_artifacts
        preview = scientific.get("preview_artifact")
        if isinstance(preview, Mapping):
            preview_path = str(preview.get("path") or "")
            enriched_preview = descriptors_by_path.get(preview_path)
            if enriched_preview is None:
                raise ServiceError(
                    409,
                    "plot_preview_not_declared",
                    "Plot preview is not one of the run's declared export artifacts.",
                )
            enriched["preview_artifact"] = enriched_preview
        return {
            "execution": "completed",
            "run_id": run.id,
            "result": enriched,
            "run": serialized,
        }

    def resolve_plot_run_artifact(
        self,
        run_id: str,
        storage_key: str,
    ) -> tuple[Path, str]:
        run = self._require_run(run_id)
        if run.task_kind != "plot" or run.status is not RunStatus.COMPLETED:
            raise ServiceError(404, "plot_artifact_not_found", "Plot artifact is not available.")
        result_payload = self.get_plot_run_result(run.id).get("result")
        artifacts = result_payload.get("artifacts") if isinstance(result_payload, Mapping) else None
        declared = list(artifacts.values()) if isinstance(artifacts, Mapping) else []
        descriptor = next(
            (item for item in declared if isinstance(item, Mapping) and item.get("storage_key") == storage_key),
            None,
        )
        if descriptor is None:
            raise ServiceError(404, "plot_artifact_not_found", "Plot artifact is not declared by this run.")
        bundle = self._plot_bundle_path(run)
        relative = self._safe_plot_artifact_key(storage_key)
        path = bundle.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ServiceError(404, "plot_artifact_not_found", "Plot artifact file is unavailable.")
        if path.stat().st_size != int(descriptor.get("size_bytes", -1)) or _sha256_path(path) != descriptor.get("sha256"):
            raise ServiceError(409, "plot_artifact_changed", "Plot artifact checksum no longer matches its verified record.")
        return path, str(descriptor.get("media_type") or "application/octet-stream")

    @staticmethod
    def _plot_source_id(run_id: str, runset_uid: str) -> str:
        return f"{run_id}::{runset_uid}"

    def _parse_plot_source_id(self, source_id: str) -> tuple[str, str]:
        if not isinstance(source_id, str) or "::" not in source_id:
            raise ServiceError(400, "invalid_plot_source", "Plot source identity is invalid.")
        run_id, runset_uid = source_id.split("::", 1)
        if not run_id or not runset_uid or len(source_id.encode("utf-8")) > 4_096:
            raise ServiceError(400, "invalid_plot_source", "Plot source identity is invalid.")
        if self.database.get_result_import(run_id) is None:
            raise ServiceError(404, "result_not_found", "Imported result not found.")
        return run_id, runset_uid

    def _plot_bundle_path(self, run: Run) -> Path:
        root = self._local_artifact_path(run)
        expected_root = self.paths.runs.resolve() / run.id
        if root.is_symlink() or not root.is_dir() or root.resolve() != expected_root:
            raise ServiceError(409, "plot_result_path_invalid", "Managed run result path is unsafe.")
        bundle = root / "remote" / "results" / "bundle"
        if bundle.is_symlink() or not bundle.is_dir() or not bundle.resolve().is_relative_to(root.resolve()):
            raise ServiceError(409, "plot_result_path_invalid", "Managed plot bundle is unsafe.")
        return bundle

    def _plot_artifact_descriptor(
        self,
        run: Run,
        bundle: Path,
        descriptor: Mapping[str, Any],
    ) -> dict[str, Any]:
        relative = self._safe_plot_artifact_key(str(descriptor.get("path") or ""))
        path = bundle.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ServiceError(409, "plot_artifact_missing", "Declared plot artifact is missing.")
        digest = str(descriptor.get("sha256") or "")
        if path.stat().st_size != int(descriptor.get("size_bytes", -1)) or _sha256_path(path) != digest:
            raise ServiceError(409, "plot_artifact_changed", "Declared plot artifact checksum changed.")
        storage_key = relative.as_posix()
        return {
            **dict(descriptor),
            "id": digest[:24],
            "storage_key": storage_key,
            "filename": relative.name,
            "format": relative.suffix.lower().removeprefix("."),
            "url": f"/api/plots/runs/{run.id}/artifacts/{storage_key}",
        }

    @staticmethod
    def _safe_plot_artifact_key(value: str) -> PurePosixPath:
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffix.lower() not in {".png", ".pdf", ".svg"}
        ):
            raise ServiceError(400, "invalid_plot_artifact_path", "Plot artifact path is unsafe.")
        return relative

    # -- Problem visualization and NBN ---------------------------------

    def benchmark_catalog(self) -> dict[str, Any]:
        """Return registry-backed problem specs and public workload limits."""

        return {
            "problems": self.benchmark_workspace.list_problems(),
            "limits": self.benchmark_workspace.limits(),
        }

    def render_benchmark(
        self,
        operation: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Calculate one Benchmark/NBN view immediately on this machine."""

        operations = {
            "contour": self.benchmark_workspace.render_contour,
            "surface": self.benchmark_workspace.build_surface,
            "nbn_views": self.benchmark_workspace.render_nbn_views,
            "nbn_model": self.benchmark_workspace.build_nbn_model,
        }
        renderer = operations.get(operation)
        if renderer is None:
            raise ServiceError(
                404,
                "benchmark_operation_not_found",
                "Unknown problem-visualization operation.",
            )
        if not self._interactive_benchmark_lock.acquire(blocking=False):
            raise ServiceError(
                409,
                "interactive_benchmark_busy",
                "Another local Benchmark/NBN preview is still being calculated. "
                "Wait for it to finish, then try again.",
            )
        request_values = {
            key: values[key]
            for key in (
                "problem_name",
                "problem_params",
                "environment_number",
                "seed",
                "resolution",
                "sample_count",
            )
            if key in values
        }
        try:
            try:
                return renderer(request_values)
            except (BenchmarkWorkspaceError, TypeError, ValueError) as exc:
                raise ServiceError(400, "invalid_benchmark_request", str(exc)) from exc
        finally:
            self._interactive_benchmark_lock.release()

    def resolve_interactive_benchmark_artifact(
        self,
        storage_key: str,
    ) -> tuple[Path, str]:
        """Resolve one locally rendered Benchmark/NBN PNG."""

        try:
            path = self.benchmark_workspace.resolve_artifact(storage_key)
        except FileNotFoundError as exc:
            raise ServiceError(404, "benchmark_artifact_not_found", str(exc)) from exc
        except BenchmarkWorkspaceError as exc:
            raise ServiceError(
                400,
                "invalid_benchmark_artifact_path",
                str(exc),
            ) from exc
        return path, "image/png"

    def get_benchmark_run_result(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        if run.task_kind != "benchmark":
            raise ServiceError(404, "benchmark_run_not_found", "This run is not a problem visualization.")
        serialized = self.serialize_run(
            run,
            self.database.get_server(run.server_id),
        )
        if run.status is not RunStatus.COMPLETED:
            execution = "failed" if run.status in TERMINAL_RUN_STATUSES else "queued"
            return {
                "execution": execution,
                "run_id": run.id,
                "status": serialized["status"],
                "message": run.error_message or run.progress_message,
                "run": serialized,
            }
        bundle = self._benchmark_bundle_path(run)
        result_path = bundle / "workload-result.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise ServiceError(409, "benchmark_result_missing", "Verified workload result is missing.")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServiceError(409, "benchmark_result_invalid", str(exc)) from exc
        if not isinstance(result, Mapping) or result.get("spec_sha256") != run.task_sha256:
            raise ServiceError(409, "benchmark_result_identity_mismatch", "Workload result does not match the frozen run.")
        scientific = result.get("result")
        if not isinstance(scientific, Mapping):
            raise ServiceError(409, "benchmark_result_invalid", "Workload result has no scientific payload.")
        enriched = dict(scientific)
        for field in ("artifact", "model_artifact"):
            descriptor = enriched.get(field)
            if isinstance(descriptor, Mapping):
                enriched[field] = self._benchmark_artifact_descriptor(
                    run,
                    bundle,
                    descriptor,
                )
        return {
            "execution": "completed",
            "run_id": run.id,
            "result": enriched,
            "run": serialized,
        }

    def resolve_benchmark_run_artifact(
        self,
        run_id: str,
        storage_key: str,
    ) -> tuple[Path, str, str | None]:
        run = self._require_run(run_id)
        if run.task_kind != "benchmark" or run.status is not RunStatus.COMPLETED:
            raise ServiceError(404, "benchmark_artifact_not_found", "Benchmark artifact is not available.")
        bundle = self._benchmark_bundle_path(run)
        result_payload = self.get_benchmark_run_result(run.id).get("result")
        descriptors = []
        if isinstance(result_payload, Mapping):
            descriptors = [
                value
                for key in ("artifact", "model_artifact")
                if isinstance((value := result_payload.get(key)), Mapping)
            ]
        descriptor = next(
            (item for item in descriptors if item.get("storage_key") == storage_key),
            None,
        )
        if descriptor is None:
            raise ServiceError(404, "benchmark_artifact_not_found", "Benchmark artifact is not declared by this run.")
        relative = self._safe_benchmark_artifact_key(storage_key)
        path = bundle.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ServiceError(404, "benchmark_artifact_not_found", "Benchmark artifact file is unavailable.")
        if path.stat().st_size != int(descriptor.get("size_bytes", -1)) or _sha256_path(path) != descriptor.get("sha256"):
            raise ServiceError(409, "benchmark_artifact_changed", "Benchmark artifact checksum no longer matches its verified record.")
        return path, str(descriptor.get("media_type") or "application/octet-stream"), _optional_text(descriptor.get("encoding"))

    def _benchmark_bundle_path(self, run: Run) -> Path:
        root = self._local_artifact_path(run)
        expected_root = self.paths.runs.resolve() / run.id
        if root.is_symlink() or not root.is_dir() or root.resolve() != expected_root:
            raise ServiceError(409, "benchmark_result_path_invalid", "Managed run result path is unsafe.")
        bundle = root / "remote" / "results" / "bundle"
        if bundle.is_symlink() or not bundle.is_dir() or not bundle.resolve().is_relative_to(root.resolve()):
            raise ServiceError(409, "benchmark_result_path_invalid", "Managed benchmark bundle is unsafe.")
        return bundle

    def _benchmark_artifact_descriptor(
        self,
        run: Run,
        bundle: Path,
        descriptor: Mapping[str, Any],
    ) -> dict[str, Any]:
        relative = self._safe_benchmark_artifact_key(str(descriptor.get("path") or ""))
        path = bundle.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ServiceError(409, "benchmark_artifact_missing", "Declared benchmark artifact is missing.")
        digest = str(descriptor.get("sha256") or "")
        if path.stat().st_size != int(descriptor.get("size_bytes", -1)) or _sha256_path(path) != digest:
            raise ServiceError(409, "benchmark_artifact_changed", "Declared benchmark artifact checksum changed.")
        storage_key = relative.as_posix()
        return {
            **dict(descriptor),
            "id": digest[:24],
            "storage_key": storage_key,
            "filename": relative.name,
            "url": f"/api/benchmarks/runs/{run.id}/artifacts/{storage_key}",
        }

    @staticmethod
    def _safe_benchmark_artifact_key(value: str) -> PurePosixPath:
        relative = PurePosixPath(value)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ServiceError(400, "invalid_benchmark_artifact_path", "Benchmark artifact path is unsafe.")
        return relative

    def list_sa_imports(self) -> list[dict[str, Any]]:
        return self._call_result_workspace(self.result_workspace.list_imports)

    def import_sa_runsets(
        self,
        result_id: str,
        runset_uids: list[str] | tuple[str, ...] | None = None,
        *,
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.import_runsets,
            result_id,
            runset_uids,
            allow_duplicates=allow_duplicates,
        )

    def import_sa_runset_selections(
        self,
        selections: Sequence[Mapping[str, Any]],
        *,
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.import_runset_selections,
            selections,
            allow_duplicates=allow_duplicates,
        )

    def delete_sa_import(self, import_id: str) -> None:
        deleted = self._call_result_workspace(
            self.result_workspace.delete_import, import_id
        )
        if not deleted:
            raise ServiceError(404, "sa_import_not_found", "SA import not found.")

    def clear_sa_imports(self) -> int:
        return self._call_result_workspace(self.result_workspace.clear_imports)

    def match_sa_imports(
        self,
        *,
        base_import_id: str | None = None,
        method: str = "friedman",
    ) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.match_compatible_imports,
            base_import_id=base_import_id,
            method=method,
        )

    def quick_filter_sa_imports(
        self, *, allow_duplicates: bool = False
    ) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.quick_filter_imports,
            allow_duplicates=allow_duplicates,
        )

    def list_sa_analyses(self) -> list[dict[str, Any]]:
        return self._call_result_workspace(self.result_workspace.list_analyses)

    def create_sa_analysis(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.run_analysis,
            name=str(values.get("name", "")),
            method=str(values.get("method", "friedman")),
            import_ids=values.get("import_ids") or (),
            metrics=values.get("metrics"),
            alpha=values.get("alpha", 0.05),
            formats=values.get("formats"),
            label_parameter=values.get("label_parameter"),
        )

    def get_sa_analysis(self, analysis_id: str) -> dict[str, Any]:
        return self._call_result_workspace(
            self.result_workspace.get_analysis, analysis_id
        )

    def delete_sa_analysis(self, analysis_id: str) -> None:
        deleted = self._call_result_workspace(
            self.result_workspace.delete_analysis, analysis_id
        )
        if not deleted:
            raise ServiceError(404, "analysis_not_found", "SA record not found.")

    @staticmethod
    def _call_result_workspace(
        operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        try:
            return operation(*args, **kwargs)
        except ResultWorkspaceError as exc:
            raise ServiceError(exc.status_code, exc.code, exc.message) from exc

    # -- Settings and dashboard ----------------------------------------

    def get_settings(self) -> dict[str, Any]:
        stored = self.database.get_setting(SETTINGS_KEY, {})
        if not isinstance(stored, Mapping):
            stored = {}
        settings = {
            "workspace_path": str(self.paths.repository_root),
            **DEFAULT_SETTINGS,
            **{key: stored[key] for key in DEFAULT_SETTINGS if key in stored},
        }
        # These are scientific integrity invariants, not user preferences.
        settings["verify_artifact_checksums"] = True
        settings["delete_remote_results_after_transfer"] = True
        return settings

    def save_settings(self, values: Mapping[str, Any]) -> dict[str, Any]:
        current = self.get_settings()
        if values.get("verify_artifact_checksums", True) is not True or values.get(
            "delete_remote_results_after_transfer", True
        ) is not True:
            raise ServiceError(
                400,
                "immutable_safety_policy",
                "Artifact checksum verification and verified remote-result cleanup are mandatory Relay safety policies.",
            )
        try:
            ssh_key_path = _normalize_ssh_key_path(
                str(
                    values.get(
                        "default_ssh_key_path", current["default_ssh_key_path"]
                    )
                )
            )
        except ValueError as exc:
            raise ServiceError(400, "invalid_settings", str(exc)) from exc
        try:
            remote_root = _normalize_remote_root(
                str(values.get("default_remote_root", current["default_remote_root"]))
            )
        except ValueError as exc:
            raise ServiceError(400, "invalid_settings", str(exc)) from exc
        saved = {
            "default_ssh_key_path": ssh_key_path,
            "default_remote_root": remote_root,
            "default_poll_interval_seconds": _positive_float(
                values.get(
                    "default_poll_interval_seconds",
                    current["default_poll_interval_seconds"],
                ),
                "default_poll_interval_seconds",
                maximum=300,
            ),
            "resource_sampling_interval_seconds": _positive_float(
                values.get(
                    "resource_sampling_interval_seconds",
                    current["resource_sampling_interval_seconds"],
                ),
                "resource_sampling_interval_seconds",
                maximum=300,
            ),
            "resource_history_retention_hours": _positive_float(
                values.get(
                    "resource_history_retention_hours",
                    current["resource_history_retention_hours"],
                ),
                "resource_history_retention_hours",
                maximum=720,
            ),
            "default_max_parallel_runs": _positive_int(
                values.get(
                    "default_max_parallel_runs",
                    current["default_max_parallel_runs"],
                ),
                "default_max_parallel_runs",
                maximum=256,
            ),
            "probe_timeout_seconds": _positive_float(
                values.get("probe_timeout_seconds", current["probe_timeout_seconds"]),
                "probe_timeout_seconds",
                minimum=3,
                maximum=300,
            ),
            "require_clean_git": bool(
                values.get("require_clean_git", current["require_clean_git"])
            ),
            "verify_artifact_checksums": True,
            "delete_remote_results_after_transfer": True,
        }
        self.database.set_setting(SETTINGS_KEY, saved)
        self.resource_monitor.wake(reschedule=True)
        return {"workspace_path": str(self.paths.repository_root), **saved}

    def dashboard(self) -> dict[str, Any]:
        servers = self.database.list_servers()
        settings = self.get_settings()
        sample_maps = self._resource_monitor_sample_maps(servers)
        serialized_servers = [
            self.serialize_server(
                server,
                monitor_samples=self._resource_monitor_samples_for(
                    server.id, sample_maps
                ),
                settings=settings,
            )
            for server in servers
        ]
        server_counts = {
            "online": 0,
            "probing": 0,
            "offline": 0,
            "error": 0,
            "unknown": 0,
        }
        for server in serialized_servers:
            server_counts[server["status"]] += 1

        server_map = {server.id: server for server in servers}
        runs = self.database.list_runs(limit=50, summary=True)
        active = self.database.list_runs_requiring_recovery()
        metadata = self.database.run_metadata(list(dict.fromkeys(run.id for run in [*active, *runs])))
        recent_runs = [
            self.serialize_run(run, server_map.get(run.server_id), metadata=metadata) for run in runs
        ]
        counts = self.database.dashboard_counts()
        audit = [self._serialize_audit_event(event, event.run_id) for event in self.database.latest_audit_events()]
        return {
            "server_counts": server_counts,
            "active_run_count": counts.active_runs,
            "completed_run_count": counts.status_counts[RunStatus.COMPLETED.value],
            "failed_run_count": counts.status_counts[RunStatus.FAILED.value]
            + counts.status_counts[RunStatus.INTERRUPTED.value],
            "transferred_bytes": self.database.transferred_bytes(),
            "servers": serialized_servers,
            "recent_runs": recent_runs,
            "active_runs": [self.serialize_run(run, server_map.get(run.server_id), metadata=metadata) for run in active],
            "audit_events": audit[:50],
        }

    # -- Serialization --------------------------------------------------

    def serialize_server(
        self,
        server: Server,
        *,
        monitor_samples: tuple[
            ResourceSample | None,
            ResourceSample | None,
            ResourceSample | None,
        ]
        | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ok, report, probe_error = _probe_parts(server.last_probe)
        if not server.enabled:
            status = "offline"
        elif server.last_probe_at is None:
            status = "unknown"
        elif ok is True:
            status = "online"
        else:
            status = "error"
        return {
            "id": server.id,
            "target_kind": server.target_kind.value,
            "system_managed": server.target_kind is ExecutionTargetKind.LOCAL,
            "name": server.name,
            "host": server.host,
            "port": server.port,
            "username": server.username,
            "ssh_key_path": server.ssh_key_path,
            "remote_root": server.remote_root,
            "conda_env": server.conda_env,
            "enabled": server.enabled,
            "status": status,
            "last_probe_at": server.last_probe_at,
            "probe_error": probe_error if ok is False else None,
            "fingerprint": server.fingerprint,
            "system": self._serialize_system_report(report, server) if report else None,
            "resource_monitor": self._serialize_resource_monitor(
                server,
                settings=settings,
                samples=monitor_samples,
            ),
        }

    def serialize_run(
        self, run: Run, server: Server | None, *, detail: bool = False, metadata: tuple[dict, dict] | None = None
    ) -> dict[str, Any]:
        total = float(run.progress_total or 0)
        current = float(run.progress_current)
        if run.status is RunStatus.COMPLETED:
            progress_percent = 100.0
            if total and current < total:
                current = total
        elif total > 0:
            progress_percent = max(0.0, min(100.0, 100.0 * current / total))
        else:
            progress_percent = 0.0
        artifacts = self.database.list_artifacts(run.id) if metadata is None else metadata[0].get(run.id, [])
        payload = dict(run.task_payload or {})
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        imported_result = self.database.find_result_import_by_source_run(run.id) if metadata is None else metadata[1].get(run.id)
        result_candidate_ready = (
            run.status is RunStatus.COMPLETED
            and run.transfer_verified
            and run.task_kind == "batch"
            and imported_result is None
        )
        result: dict[str, Any] = {
            "id": run.id,
            "status": _frontend_status(run.status),
            "phase": _frontend_phase(run.status, run.error_details, run.stage),
            "stage": dict(run.stage) if run.stage else None,
            "task_path": run.task_path or run.task_name,
            "task_label": run.task_name,
            "task_kind": run.task_kind,
            "server_id": run.server_id,
            "target_kind": str(
                _mapping_get(payload.get("server_snapshot"), "target_kind")
                or (server.target_kind.value if server else "ssh")
            ),
            "server_name": str(
                _mapping_get(payload.get("server_snapshot"), "name")
                or (server.name if server else "Unknown server")
            ),
            "max_parallel_runs": run.max_parallel_runs,
            "refresh_interval_seconds": run.poll_interval_seconds,
            "progress_percent": round(progress_percent, 2),
            "completed_units": _number_for_json(current),
            "total_units": _number_for_json(total),
            "current_message": run.progress_message,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "exit_code": run.exit_code,
            "error_code": run.error_code,
            "error_message": run.error_message,
            "git": self._serialize_git(run, provenance),
            "artifacts": [self.serialize_artifact(item) for item in artifacts],
            "remote_cleanup": self._serialize_cleanup(run),
            "recovery": self._serialize_recovery(run),
            "admission_override": run.admission_override,
            "result_candidate_ready": result_candidate_ready,
            # Compatibility alias for older frontends.  Import is now exposed
            # only from the result-analysis candidate list.
            "result_importable": result_candidate_ready,
            "result_imported": imported_result is not None,
            "result_id": imported_result["id"] if imported_result is not None else None,
        }
        if detail:
            result.update(
                {
                    "remote_workspace": run.remote_run_dir,
                    "local_result_path": str(self._local_artifact_path(run)),
                    "command": shlex.join(run.command_argv)
                    if run.command_argv
                    else None,
                    "pid": run.remote_pid,
                    "environment_report": self._environment_report(run, server),
                    "server_snapshot": payload.get("server_snapshot"),
                }
            )
        return result

    @staticmethod
    def serialize_artifact(artifact: Artifact) -> dict[str, Any]:
        return {
            "path": artifact.relative_path,
            "kind": _artifact_kind(artifact.relative_path),
            "size_bytes": artifact.size_bytes or 0,
            "sha256": artifact.sha256,
            "transferred": artifact.status
            in {ArtifactStatus.TRANSFERRED, ArtifactStatus.VERIFIED},
            "verified": artifact.status is ArtifactStatus.VERIFIED,
        }

    @staticmethod
    def _server_snapshot(server: Server) -> dict[str, Any]:
        """Freeze a run's target without ever persisting private-key content."""

        return {
            "schema_version": 2,
            "id": server.id,
            "target_kind": server.target_kind.value,
            "name": server.name,
            "host": server.host,
            "port": server.port,
            "username": server.username,
            "ssh_key_path": server.ssh_key_path,
            "remote_root": server.remote_root,
            "conda_env": server.conda_env,
            "poll_interval_seconds": server.poll_interval_seconds,
            "default_max_parallel_runs": server.default_max_parallel_runs,
            "min_free_bytes": server.min_free_bytes,
            "max_load_ratio": server.max_load_ratio,
            "fingerprint": server.fingerprint,
            "captured_at": utc_now(),
        }

    @staticmethod
    def serialize_event(event: RunEvent) -> dict[str, Any]:
        data = dict(event.data or {})
        stream = str(data.get("stream", "system"))
        if stream not in {"stdout", "stderr", "system", "audit"}:
            stream = "system"
        phase = data.get("phase")
        if isinstance(phase, RunStatus):
            phase = _frontend_phase(phase, {})
        return {
            "id": str(event.id),
            "sequence": event.sequence,
            "timestamp": event.created_at,
            "stream": stream,
            "level": event.level.value,
            "kind": event.kind,
            "phase": phase,
            "stage": dict(data["stage"])
            if isinstance(data.get("stage"), Mapping)
            else None,
            "message": event.message,
        }

    @staticmethod
    def serialize_sample(sample: ResourceSample) -> dict[str, Any]:
        memory_percent = 0.0
        if sample.memory_total_bytes and sample.memory_available_bytes is not None:
            used = sample.memory_total_bytes - sample.memory_available_bytes
            memory_percent = max(0.0, min(100.0, 100.0 * used / sample.memory_total_bytes))
        return {
            "id": sample.id,
            "timestamp": sample.sampled_at,
            "cpu_percent": sample.cpu_percent or 0.0,
            "memory_percent": round(memory_percent, 2),
            "load_1m": sample.load_1m,
            "gpus": [_serialize_gpu_sample(item) for item in sample.gpu_devices],
        }

    @staticmethod
    def serialize_resource_sample(sample: ResourceSample) -> dict[str, Any]:
        memory_total_gb: float | None = None
        memory_used_gb: float | None = None
        memory_percent: float | None = None
        if sample.memory_total_bytes is not None and sample.memory_total_bytes > 0:
            memory_total_gb = sample.memory_total_bytes / 1024**3
            if sample.memory_available_bytes is not None:
                available = max(
                    0,
                    min(sample.memory_total_bytes, sample.memory_available_bytes),
                )
                used = sample.memory_total_bytes - available
                memory_used_gb = used / 1024**3
                memory_percent = round(100.0 * used / sample.memory_total_bytes, 2)
        return {
            "id": sample.id,
            "server_id": sample.server_id,
            "sampled_at": sample.sampled_at,
            "aggregation": dict(sample.aggregation),
            "reachable": sample.reachable,
            "error": sample.error,
            "cpu_physical_cores": sample.cpu_physical_cores,
            "cpu_logical_cores": sample.cpu_logical_cores,
            "cpu_percent": sample.cpu_percent,
            "load_1m": sample.load_1m,
            "load_5m": sample.load_5m,
            "load_15m": sample.load_15m,
            "load_ratio": sample.load_ratio,
            "memory_total_gb": memory_total_gb,
            "memory_used_gb": memory_used_gb,
            "memory_percent": memory_percent,
            "gpu_usable": sample.gpu_usable,
            "gpu_error": sample.gpu_error,
            "gpus": [_serialize_gpu_resource(item) for item in sample.gpu_devices],
        }

    # -- Internal helpers ----------------------------------------------

    def _require_server(self, server_id: str) -> Server:
        server = self.database.get_server(server_id)
        if server is None:
            raise ServiceError(404, "server_not_found", "Server not found.")
        return server

    def _resource_monitor_server_ids(self) -> list[str]:
        result: list[str] = []
        for server in self.database.list_servers():
            try:
                self._require_resource_monitor_identity(server)
            except ServiceError:
                continue
            result.append(server.id)
        return result

    def _resource_sampling_interval(self) -> float:
        return float(self.get_settings()["resource_sampling_interval_seconds"])

    def _background_resource_sample(self, server_id: str) -> bool:
        try:
            server = self._require_server(server_id)
            self._require_resource_monitor_identity(server)
        except ServiceError:
            # The scheduler will remove newly ineligible/deleted servers on its
            # next inventory pass; this is not a transport failure to back off.
            return True
        lock = self._resource_refresh_lock(server.id)
        with lock:
            try:
                self._collect_resource_sample(server)
            except ServiceError:
                return False
        return True

    def _collect_resource_sample(self, server: Server) -> ResourceSample:
        self._require_resource_monitor_identity(server)
        settings = self.get_settings()
        timeout = min(
            float(settings["probe_timeout_seconds"]),
            RESOURCE_REFRESH_TIMEOUT_SECONDS,
        )
        sampled_at = utc_now()
        try:
            payload = self._probe_client(
                server,
                connect_timeout_seconds=max(1, min(5, int(math.ceil(timeout)))),
            ).sample_resources(
                timeout=timeout,
                expected_fingerprint=server.fingerprint,
            )
            current = self.database.get_server(server.id)
            if current is None or not self._same_resource_identity(server, current):
                raise ServiceError(
                    409,
                    "resource_target_changed",
                    "Server connection identity changed while resource sampling was in progress; the old-target sample was discarded.",
                )
            sample = self._resource_sample(
                current,
                payload,
                checked_at=sampled_at,
            )
            return self.database.add_resource_sample(sample)
        except ServiceError:
            raise
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            current = self.database.get_server(server.id)
            if current is not None and self._same_resource_identity(server, current):
                try:
                    self.database.add_resource_sample(
                        ResourceSample(
                            server_id=server.id,
                            target_identity=self._resource_target_identity(server),
                            sampled_at=sampled_at,
                            reachable=False,
                            error=message,
                        )
                    )
                except (KeyError, sqlite3.IntegrityError):
                    pass
            else:
                raise ServiceError(
                    409,
                    "resource_target_changed",
                    "Server connection identity changed while resource sampling was in progress; the old-target failure was discarded.",
                ) from exc
            raise ServiceError(502, "resource_refresh_failed", message) from exc

    @staticmethod
    def _require_resource_monitor_identity(server: Server) -> None:
        probe_ok, report, _ = _probe_parts(server.last_probe)
        if (
            probe_ok is not True
            or report is None
            or server.last_probe_at is None
            or not server.fingerprint
        ):
            raise ServiceError(
                409,
                "resource_refresh_requires_probe",
                "Run a successful full environment probe and verify its SSH "
                "host-key fingerprint before resource monitoring.",
            )

    @staticmethod
    def _same_resource_identity(before: Server, after: Server) -> bool:
        return RelayService._resource_identity_values(
            before
        ) == RelayService._resource_identity_values(after)

    @staticmethod
    def _resource_identity_values(server: Server) -> tuple[object, ...]:
        return (
            server.target_kind.value,
            server.host,
            server.port,
            server.username,
            server.ssh_key_path,
            server.remote_root,
            server.fingerprint,
        )

    @staticmethod
    def _resource_target_identity(server: Server) -> str:
        payload = stable_json_dumps(
            RelayService._resource_identity_values(server)
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _prune_resource_history(self) -> int:
        retention_hours = float(
            self.get_settings()["resource_history_retention_hours"]
        )
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=retention_hours)
        removed = self.database.prune_background_resource_samples(
            before=normalize_utc_iso(cutoff),
            max_samples_per_server=RESOURCE_HISTORY_MAX_SAMPLES_PER_SERVER,
        )
        removed += self.database.aggregate_background_resource_samples(
            before=normalize_utc_iso((now - timedelta(hours=1)).replace(second=0, microsecond=0)),
        )
        return removed

    def _serialize_resource_monitor(
        self,
        server: Server,
        *,
        settings: Mapping[str, Any] | None = None,
        samples: tuple[
            ResourceSample | None,
            ResourceSample | None,
            ResourceSample | None,
        ]
        | None = None,
    ) -> dict[str, Any]:
        current_settings = settings or self.get_settings()
        interval = float(current_settings["resource_sampling_interval_seconds"])
        stale_after = max(15.0, interval * RESOURCE_STALE_INTERVAL_MULTIPLIER)
        if samples is None:
            latest_attempt = self.database.latest_background_resource_sample(
                server.id,
                target_identity=self._resource_target_identity(server),
            )
            latest_success = self.database.latest_background_resource_sample(
                server.id,
                reachable=True,
                target_identity=self._resource_target_identity(server),
            )
            last_error = self.database.latest_background_resource_sample(
                server.id,
                reachable=False,
                target_identity=self._resource_target_identity(server),
            )
        else:
            latest_attempt, latest_success, last_error = samples
        freshness: float | None = None
        if latest_success is not None:
            sampled = datetime.fromisoformat(
                latest_success.sampled_at.replace("Z", "+00:00")
            )
            freshness = max(
                0.0,
                (datetime.now(timezone.utc) - sampled).total_seconds(),
            )

        probe_ok, report, _ = _probe_parts(server.last_probe)
        if (
            probe_ok is not True
            or report is None
            or server.last_probe_at is None
            or not server.fingerprint
        ):
            state = "paused"
            pause_reason = (
                "pinned_fingerprint_required"
                if probe_ok is True and not server.fingerprint
                else "successful_probe_required"
            )
        elif latest_attempt is None:
            state = "never"
            pause_reason = None
        elif not latest_attempt.reachable:
            state = "failed"
            pause_reason = None
        elif freshness is None or freshness > stale_after:
            state = "stale"
            pause_reason = None
        else:
            state = "fresh"
            pause_reason = None

        return {
            "state": state,
            "sample_interval_seconds": interval,
            "stale_after_seconds": stale_after,
            "freshness_seconds": round(freshness, 3)
            if freshness is not None
            else None,
            "pause_reason": pause_reason,
            "latest_attempt": self._serialize_resource_attempt(latest_attempt),
            "last_success": self._serialize_resource_attempt(latest_success),
            "last_error": self._serialize_resource_attempt(last_error),
            "latest_success": self.serialize_resource_sample(latest_success)
            if latest_success is not None
            else None,
        }

    def _resource_monitor_sample_maps(
        self,
        servers: list[Server],
    ) -> tuple[
        dict[str, ResourceSample],
        dict[str, ResourceSample],
        dict[str, ResourceSample],
    ]:
        identities = {
            server.id: self._resource_target_identity(server) for server in servers
        }
        return self.database.latest_background_resource_sample_sets(
            identities
        )

    @staticmethod
    def _resource_monitor_samples_for(
        server_id: str,
        sample_maps: tuple[
            dict[str, ResourceSample],
            dict[str, ResourceSample],
            dict[str, ResourceSample],
        ],
    ) -> tuple[
        ResourceSample | None,
        ResourceSample | None,
        ResourceSample | None,
    ]:
        attempts, successes, errors = sample_maps
        return (
            attempts.get(server_id),
            successes.get(server_id),
            errors.get(server_id),
        )

    @staticmethod
    def _serialize_resource_attempt(
        sample: ResourceSample | None,
    ) -> dict[str, Any] | None:
        if sample is None:
            return None
        return {
            "id": sample.id,
            "sampled_at": sample.sampled_at,
            "reachable": sample.reachable,
            "error": sample.error,
        }

    def _ensure_unique_execution_target(
        self, server: Server, *, excluding_id: str | None = None
    ) -> None:
        """Prevent aliases from concurrently mutating one editable Conda env."""

        target = _execution_target_key(server)
        for existing in self.database.list_servers():
            if existing.id == excluding_id:
                continue
            if _execution_target_key(existing) == target:
                raise ServiceError(
                    409,
                    "duplicate_execution_target",
                    "Another compute profile already targets the same execution "
                    f"identity and Conda environment ({existing.name!r}). Use that "
                    "profile or a different isolated environment.",
                )

    def _ensure_local_target(self) -> Server:
        """Idempotently install the non-deletable local compute target."""

        existing = self.database.get_server(LOCAL_TARGET_ID)
        username = getpass.getuser().strip() or "local"
        if existing is not None:
            if existing.target_kind is not ExecutionTargetKind.LOCAL:
                raise RuntimeError(
                    f"reserved target id {LOCAL_TARGET_ID!r} is not a local target"
                )
            desired = existing.with_updates(
                host="localhost",
                port=22,
                username=username,
                ssh_key_path="local://no-ssh-key",
                remote_root=str(self.paths.local_execution),
                conda_env="EDOLAB",
            )
            identity_changed = self._resource_identity_values(existing) != self._resource_identity_values(desired)
            stale_fingerprint = existing.fingerprint not in (None, local_machine_fingerprint())
            if identity_changed or stale_fingerprint:
                desired = desired.with_updates(
                    last_probe_at=None,
                    last_probe=None,
                    fingerprint=None,
                )
                self.database.update_server(desired)
            return desired
        server = Server(
            id=LOCAL_TARGET_ID,
            target_kind=ExecutionTargetKind.LOCAL,
            name="This computer",
            host="localhost",
            port=22,
            username=username,
            ssh_key_path="local://no-ssh-key",
            remote_root=str(self.paths.local_execution),
            conda_env="EDOLAB",
            poll_interval_seconds=float(
                self.get_settings()["default_poll_interval_seconds"]
            ),
            default_max_parallel_runs=int(
                self.get_settings()["default_max_parallel_runs"]
            ),
        )
        self.database.create_server(server)
        return server

    def _probe_client(
        self,
        server: Server,
        *,
        connect_timeout_seconds: int = 10,
    ) -> ProbeClient:
        if server.target_kind is ExecutionTargetKind.LOCAL:
            return self.local_client_factory()
        connection = SSHConnection(
            host=server.host,
            username=server.username,
            key_path=server.ssh_key_path,
            port=server.port,
            known_hosts_path=self.paths.known_hosts,
            control_dir=self.paths.ssh_mux / server.id,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        return self.ssh_client_factory(connection)

    def _require_run(self, run_id: str) -> Run:
        run = self.database.get_run(run_id)
        if run is None:
            raise ServiceError(404, "run_not_found", "Run not found.")
        return run

    def _task_root(self) -> Path:
        return self.paths.repository_root / "tasks"

    def _load_task_source(
        self,
        values: Mapping[str, Any],
    ) -> tuple[Path, str, Any, dict[str, Any] | None]:
        task_path_value = values.get("task_path")
        task_jsonl_value = values.get("task_jsonl")
        if task_path_value is not None and not isinstance(task_path_value, str):
            raise ServiceError(400, "invalid_task_source", "task_path must be text.")
        if task_jsonl_value is not None and not isinstance(task_jsonl_value, str):
            raise ServiceError(400, "invalid_task_source", "task_jsonl must be text.")
        has_path = bool(task_path_value and task_path_value.strip())
        has_jsonl = bool(task_jsonl_value and task_jsonl_value.strip())
        if has_path == has_jsonl:
            raise ServiceError(
                400,
                "invalid_task_source",
                "Exactly one of task_path or task_jsonl is required.",
            )
        if has_path:
            task_path = self._resolve_task_path(str(task_path_value))
            task_display_path = task_path.relative_to(
                self.paths.repository_root
            ).as_posix()
            inspection = self._inspect_task_model(task_path)
            try:
                task_text = inspection.task_bytes.decode("utf-8")
            except UnicodeError as exc:
                raise ServiceError(
                    400,
                    "invalid_task",
                    "Repository task must be valid UTF-8 text.",
                ) from exc
            return (
                task_path,
                task_display_path,
                inspection,
                {
                    "schema_version": 1,
                    "display_name": task_display_path,
                    "sha256": inspection.task_file_sha256,
                    "utf8_text": task_text,
                    "source": "repository_task",
                },
            )

        assert isinstance(task_jsonl_value, str)
        try:
            task_bytes = task_jsonl_value.encode("utf-8")
        except UnicodeError as exc:
            raise ServiceError(
                400,
                "invalid_task",
                "task_jsonl must be valid UTF-8 text.",
            ) from exc
        from .tasking import MAX_INLINE_TASK_BYTES, inspect_batch_task_bytes

        if len(task_bytes) > MAX_INLINE_TASK_BYTES:
            raise ServiceError(
                400,
                "inline_task_too_large",
                f"task_jsonl exceeds the {MAX_INLINE_TASK_BYTES}-byte UTF-8 limit.",
            )
        digest = hashlib.sha256(task_bytes).hexdigest()
        task_display_path = f"inline/{digest[:16]}.jsonl"
        task_identity = Path(task_display_path)
        try:
            inspection = inspect_batch_task_bytes(
                task_bytes,
                task_name=task_identity,
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ServiceError(400, "invalid_task", str(exc)) from exc
        return (
            task_identity,
            task_display_path,
            inspection,
            {
                "schema_version": 1,
                "display_name": task_display_path,
                "sha256": digest,
                "utf8_text": task_jsonl_value,
            },
        )

    def _resolve_task_path(self, task_path: str) -> Path:
        if not task_path.strip():
            raise ServiceError(400, "invalid_task_path", "task_path is required.")
        supplied = Path(task_path)
        if supplied.is_absolute() or ".." in supplied.parts:
            raise ServiceError(
                400, "invalid_task_path", "Only repository tasks/*.jsonl paths are allowed."
            )
        if supplied.suffix.lower() != ".jsonl":
            raise ServiceError(400, "invalid_task_path", "Task file must use .jsonl.")
        candidate = self.paths.repository_root / supplied
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ServiceError(404, "task_not_found", "Task file not found.") from exc
        task_root = self._task_root().resolve()
        if candidate.is_symlink() or resolved.parent != task_root:
            raise ServiceError(
                400, "invalid_task_path", "Only direct, non-symlink tasks/*.jsonl files are allowed."
            )
        return resolved

    def _local_artifact_path(self, run: Run) -> Path:
        if not run.local_artifact_dir:
            return self.paths.runs / run.id
        configured = Path(run.local_artifact_dir)
        if configured.is_absolute():
            return configured
        return self.paths.data_root / configured

    @staticmethod
    def _inspect_task_model(task_path: Path):
        from .tasking import inspect_batch_task

        try:
            return inspect_batch_task(task_path)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ServiceError(400, "invalid_task", str(exc)) from exc

    def _serialize_task_inspection(
        self,
        task_display_path: str,
        inspection: Any,
        server: Server,
        *,
        max_parallel_runs: int,
    ) -> dict[str, Any]:
        warnings = [item.message for item in inspection.warnings]
        hard_blocked, overrideable, admission_warnings = self._admission_assessment(
            server, inspection, max_parallel_runs
        )
        warnings.extend(admission_warnings)
        if server.last_probe_at is None:
            warnings.append(
                "The compute target has not been probed yet; the controller will perform a fresh preflight before execution."
            )
        if self.database.list_runs(
            server_id=server.id,
            statuses=tuple(NON_TERMINAL_RUN_STATUSES),
            limit=1,
        ):
            warnings.append(
                "This compute target already has queued or active work; the new run will wait in its durable queue."
            )
        settings = self.get_settings()
        if settings["require_clean_git"] and self._git_is_dirty():
            hard_blocked.append(
                "The local Git worktree is dirty and clean-Git policy is enabled."
            )
        blocked = [*hard_blocked, *overrideable]
        algorithms = sorted(
            {algorithm for item in inspection.tasks for algorithm in item.algorithms}
        )
        problems = sorted({item.problem for item in inspection.tasks})
        temporary: dict[str, str] = {}  # no retired algorithm names in this release
        return {
            "task_path": task_display_path,
            "total_tasks": len(inspection.tasks),
            "total_runs": inspection.total_runs,
            "total_evaluations": inspection.total_evals,
            "gpu_requested": bool(inspection.gpu_requested),
            "estimated_worker_slots": min(max_parallel_runs, inspection.total_runs),
            "algorithms": algorithms,
            "temporary_algorithms": temporary,
            "problems": problems,
            "entries": [
                {
                    "label": item.label,
                    "algorithms": list(item.algorithms),
                    "problem": item.problem,
                    "run_count": item.run_count,
                    "fe_per_run": item.max_evals_per_run,
                    "total_evaluations": item.total_evals,
                }
                for item in inspection.tasks
            ],
            "warnings": warnings,
            "blocked_reasons": blocked,
            "hard_blocked_reasons": hard_blocked,
            "overrideable_reasons": overrideable,
            "override_allowed": not hard_blocked and bool(overrideable),
            "admissible": not blocked,
        }

    def _admission_assessment(
        self, server: Server, inspection: Any, max_parallel_runs: int
    ) -> tuple[list[str], list[str], list[str]]:
        hard: list[str] = []
        overrideable: list[str] = []
        warnings: list[str] = []
        local_target = server.target_kind is ExecutionTargetKind.LOCAL
        target_label = "local compute target" if local_target else "SSH compute target"
        if not server.enabled:
            hard.append("The selected compute target is disabled.")
        if max_parallel_runs > server.default_max_parallel_runs:
            overrideable.append(
                f"Requested parallelism {max_parallel_runs} exceeds this compute target's configured default {server.default_max_parallel_runs}."
            )
        probe_ok, report, probe_error = _probe_parts(server.last_probe)
        if probe_ok is False:
            hard.append(
                f"The latest {target_label} probe failed: {probe_error or 'unknown error'}"
            )
            return hard, overrideable, warnings
        if probe_ok is None or report is None:
            if local_target:
                hard.append(
                    "Run a successful local environment probe before scheduling; "
                    "the probe records the machine identity and EDOLAB environment."
                )
            else:
                hard.append(
                    "Run a successful SSH environment probe before scheduling; the "
                    "probe pins the host key fingerprint and records the target environment."
                )
            return hard, overrideable, warnings
        if not server.fingerprint:
            hard.append(
                (
                    "The successful probe did not produce a local machine fingerprint; "
                    "scheduling is blocked because the execution identity cannot be frozen."
                )
                if local_target
                else (
                    "The successful probe did not produce an SSH host-key fingerprint; "
                    "scheduling is blocked because the remote identity cannot be frozen."
                )
            )
        conda = report.get("conda", {})
        environment_available = isinstance(conda, Mapping) and bool(
            conda.get("environment_available")
        )
        if not environment_available:
            hard.append(
                f"The configured Conda environment {server.conda_env!r} is unavailable on the compute target."
            )
        else:
            conda_executable = _optional_text(conda.get("path"))
            if (
                conda_executable is None
                or not PurePosixPath(conda_executable).is_absolute()
            ):
                hard.append(
                    "The successful probe did not report an absolute Conda executable path."
                )
        environment_python = (
            conda.get("environment_python") if isinstance(conda, Mapping) else None
        )
        if not isinstance(environment_python, Mapping):
            detail = (
                _optional_text(conda.get("environment_python_error"))
                if isinstance(conda, Mapping)
                else None
            )
            message = (
                f"The Python version inside Conda environment {server.conda_env!r} "
                "could not be verified; Python 3.11 is required."
            )
            if detail:
                message += f" Probe error: {detail}"
            hard.append(message)
        else:
            version = _optional_text(environment_python.get("version"))
            if not _is_python_311(version):
                hard.append(
                    f"Conda environment {server.conda_env!r} must use Python 3.11; "
                    f"the probe reported {version or 'an unknown version'}."
                )
            if environment_python.get("pip_available") is not True:
                hard.append(
                    f"Conda environment {server.conda_env!r} did not confirm "
                    "that pip is available for installing the immutable source snapshot."
                )
            hard.extend(missing_runtime_dependency_messages(environment_python))
        disk = report.get("disk", {})
        if isinstance(disk, Mapping):
            if not disk.get("writable"):
                hard.append("The configured execution workspace is not writable.")
            free = _optional_int(disk.get("free_bytes"))
            if free is None:
                hard.append("Compute-target free disk space could not be verified.")
            elif free < server.min_free_bytes:
                hard.append(
                    f"Compute-target free space ({free} bytes) is below the configured minimum ({server.min_free_bytes} bytes)."
                )
            inodes = _optional_int(disk.get("inodes_free"))
            if inodes is not None and inodes < 1000:
                hard.append("The compute-target filesystem has too few free inodes.")
        cpu = report.get("cpu", {})
        if isinstance(cpu, Mapping):
            load_ratio = _optional_float(cpu.get("load_ratio"))
            if load_ratio is not None and load_ratio > server.max_load_ratio:
                warnings.append(
                    f"Compute-target load ratio {load_ratio:.2f} exceeds the configured warning threshold {server.max_load_ratio:.2f}; load does not block scheduling."
                )
        if inspection.gpu_requested:
            gpu = report.get("gpu", {})
            nvidia = gpu.get("nvidia_smi", {}) if isinstance(gpu, Mapping) else {}
            if not isinstance(nvidia, Mapping) or not nvidia.get("usable"):
                hard.append("The task requests GPU execution but no usable NVIDIA GPU was detected.")
        return hard, overrideable, warnings

    def _git_is_dirty(self) -> bool:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(self.paths.repository_root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
        return completed.returncode != 0 or bool(completed.stdout.strip())

    def _wake_controller(self, run_id: str) -> None:
        if self.controller is None:
            return
        try:
            self.controller.wake()
        except Exception as exc:
            self.database.append_run_event(
                RunEvent(
                    run_id=run_id,
                    kind="controller",
                    level="error",
                    message=f"Controller wake failed; the run remains queued: {exc}",
                    data={"stream": "system", "phase": "queued"},
                )
            )

    def _resource_refresh_lock(self, server_id: str) -> threading.Lock:
        with self._resource_refresh_state_lock:
            return self._resource_refresh_locks.setdefault(
                server_id, threading.Lock()
            )

    def _resource_refresh_cache_entry(
        self, server_id: str
    ) -> tuple[bool, dict[str, Any] | tuple[int, str, str]] | None:
        with self._resource_refresh_state_lock:
            cached = self._resource_refresh_cache.get(server_id)
            if cached is None:
                return None
            cached_at, ok, value = cached
            if time.monotonic() - cached_at >= RESOURCE_REFRESH_CACHE_SECONDS:
                self._resource_refresh_cache.pop(server_id, None)
                return None
            return ok, value

    def _store_resource_refresh_cache(
        self,
        server_id: str,
        ok: bool,
        value: dict[str, Any] | tuple[int, str, str],
    ) -> None:
        with self._resource_refresh_state_lock:
            self._resource_refresh_cache[server_id] = (
                time.monotonic(),
                ok,
                value,
            )

    def _clear_resource_refresh_state(
        self, server_id: str, *, remove_lock: bool = False
    ) -> None:
        with self._resource_refresh_state_lock:
            self._resource_refresh_cache.pop(server_id, None)
            if remove_lock:
                self._resource_refresh_locks.pop(server_id, None)

    @staticmethod
    def _serialize_live_resources(
        payload: Mapping[str, Any], *, sampled_at: str
    ) -> dict[str, Any]:
        if payload.get("schema_version") != 1:
            raise ValueError("remote resource sample has an unsupported schema")
        cpu = payload.get("cpu", {})
        memory = payload.get("memory", {})
        gpu = payload.get("gpu", {})
        nvidia = gpu.get("nvidia_smi", {}) if isinstance(gpu, Mapping) else {}
        devices = nvidia.get("devices", []) if isinstance(nvidia, Mapping) else []

        logical = _optional_int(_mapping_get(cpu, "logical_cores"))
        load_1m = _optional_float(_mapping_get(cpu, "load_1m"))
        load_ratio = _optional_float(_mapping_get(cpu, "load_ratio"))
        if load_ratio is None and load_1m is not None and logical:
            load_ratio = load_1m / logical

        total_bytes = _optional_int(_mapping_get(memory, "total_bytes"))
        available_bytes = _optional_int(_mapping_get(memory, "available_bytes"))
        if (
            total_bytes is not None
            and total_bytes > 0
            and available_bytes is not None
        ):
            available = max(0, min(total_bytes, available_bytes))
            used_bytes = total_bytes - available
            memory_total_gb: float | None = total_bytes / 1024**3
            memory_used_gb: float | None = used_bytes / 1024**3
            memory_percent: float | None = round(
                100.0 * used_bytes / total_bytes, 2
            )
        else:
            memory_total_gb = (
                total_bytes / 1024**3
                if total_bytes is not None and total_bytes > 0
                else None
            )
            memory_used_gb = None
            memory_percent = None

        return {
            "sampled_at": sampled_at,
            "cpu_percent": _optional_float(_mapping_get(cpu, "percent")),
            "load_1m": load_1m,
            "load_ratio": load_ratio,
            "memory_total_gb": memory_total_gb,
            "memory_used_gb": memory_used_gb,
            "memory_percent": memory_percent,
            "gpu_usable": isinstance(nvidia, Mapping) and bool(nvidia.get("usable")),
            "gpu_error": _optional_text(_mapping_get(nvidia, "error")),
            "gpus": [
                _serialize_gpu_system(item)
                for item in devices
                if isinstance(item, Mapping)
            ],
        }

    def _resource_sample(
        self, server: Server, payload: Mapping[str, Any], *, checked_at: str
    ) -> ResourceSample:
        cpu = payload.get("cpu", {})
        memory = payload.get("memory", {})
        disk = payload.get("disk", {})
        gpu = payload.get("gpu", {})
        nvidia = gpu.get("nvidia_smi", {}) if isinstance(gpu, Mapping) else {}
        devices = nvidia.get("devices", []) if isinstance(nvidia, Mapping) else []
        return ResourceSample(
            server_id=server.id,
            target_identity=self._resource_target_identity(server),
            sampled_at=checked_at,
            reachable=True,
            cpu_physical_cores=_optional_int(_mapping_get(cpu, "physical_cores")),
            cpu_logical_cores=_optional_int(_mapping_get(cpu, "logical_cores")),
            cpu_percent=_optional_float(_mapping_get(cpu, "percent")),
            load_1m=_optional_float(_mapping_get(cpu, "load_1m")),
            load_5m=_optional_float(_mapping_get(cpu, "load_5m")),
            load_15m=_optional_float(_mapping_get(cpu, "load_15m")),
            load_ratio=_optional_float(_mapping_get(cpu, "load_ratio")),
            memory_total_bytes=_optional_int(_mapping_get(memory, "total_bytes")),
            memory_available_bytes=_optional_int(
                _mapping_get(memory, "available_bytes")
            ),
            disk_total_bytes=_optional_int(_mapping_get(disk, "total_bytes")),
            disk_free_bytes=_optional_int(_mapping_get(disk, "free_bytes")),
            gpu_usable=(
                bool(nvidia.get("usable")) if isinstance(nvidia, Mapping) else None
            ),
            gpu_error=_optional_text(_mapping_get(nvidia, "error")),
            gpu_devices=tuple(item for item in devices if isinstance(item, Mapping)),
        )

    @staticmethod
    def _serialize_system_report(
        report: Mapping[str, Any], server: Server
    ) -> dict[str, Any]:
        platform = report.get("platform", {})
        cpu = report.get("cpu", {})
        memory = report.get("memory", {})
        disk = report.get("disk", {})
        python = report.get("python", {})
        conda = report.get("conda", {})
        environment_python = (
            conda.get("environment_python") if isinstance(conda, Mapping) else None
        )
        gpu = report.get("gpu", {})
        nvidia = gpu.get("nvidia_smi", {}) if isinstance(gpu, Mapping) else {}
        devices = nvidia.get("devices", []) if isinstance(nvidia, Mapping) else []
        environment_declared_available = isinstance(conda, Mapping) and bool(
            conda.get("environment_available")
        )
        conda_executable = (
            _optional_text(conda.get("path")) if isinstance(conda, Mapping) else None
        )
        conda_path_available = bool(
            conda_executable
            and PurePosixPath(conda_executable).is_absolute()
        )
        conda_available = environment_declared_available and conda_path_available
        environment_version = _optional_text(
            _mapping_get(environment_python, "version")
        )
        python_compatible = _is_python_311(environment_version)
        pip_available = (
            isinstance(environment_python, Mapping)
            and environment_python.get("pip_available") is True
        )
        dependency_statuses = runtime_dependency_statuses(environment_python)
        disk_writable = isinstance(disk, Mapping) and bool(disk.get("writable"))
        disk_free_bytes = _optional_int(_mapping_get(disk, "free_bytes"))
        disk_inodes_free = _optional_int(_mapping_get(disk, "inodes_free"))
        load_ratio = _optional_float(_mapping_get(cpu, "load_ratio"))
        gpu_usable = isinstance(nvidia, Mapping) and bool(nvidia.get("usable"))
        local_target = server.target_kind is ExecutionTargetKind.LOCAL
        workspace_label = "Local execution folder" if local_target else "Remote execution folder"
        target_location = "local compute target" if local_target else "SSH compute target"
        checks: list[dict[str, Any]] = []

        def add_check(
            key: str,
            label: str,
            status: str,
            detail: str,
            *,
            blocking: bool,
        ) -> None:
            checks.append(
                {
                    "key": key,
                    "label": label,
                    "status": status,
                    "detail": detail,
                    "blocking": blocking,
                }
            )

        conda_installed = isinstance(conda, Mapping) and bool(conda.get("installed"))
        add_check(
            "conda_environment",
            "Conda environment",
            "passed" if conda_available else "failed",
            (
                f"Conda environment {server.conda_env!r} is available."
                if conda_available
                else (
                    f"Conda is not installed or visible on the {target_location}."
                    if not conda_installed
                    else (
                        "The probe found the configured Conda environment but did not "
                        "report an absolute Conda executable path."
                        if environment_declared_available
                        else f"Conda environment {server.conda_env!r} is unavailable."
                    )
                )
            ),
            blocking=True,
        )
        add_check(
            "python_311",
            "Python 3.11",
            "passed" if python_compatible else "failed",
            (
                f"The configured environment reports Python {environment_version}."
                if environment_version
                else "The configured environment's Python version could not be verified."
            ),
            blocking=True,
        )
        add_check(
            "pip",
            "pip",
            "passed" if pip_available else "failed",
            (
                "pip is available for installing the immutable snapshot."
                if pip_available
                else "pip availability was not confirmed in the configured environment."
            ),
            blocking=True,
        )
        for dependency in dependency_statuses:
            add_check(
                f"runtime_dependency_{dependency.spec.key}",
                dependency.spec.label,
                "passed" if dependency.available else "failed",
                dependency.detail,
                blocking=True,
            )
        add_check(
            "remote_root_writable",
            workspace_label,
            "passed" if disk_writable else "failed",
            (
                f"Configured execution workspace {server.remote_root!r} is writable."
                if disk_writable
                else f"Configured execution workspace {server.remote_root!r} is not writable."
            ),
            blocking=True,
        )
        disk_space_ok = (
            disk_free_bytes is not None
            and disk_free_bytes >= server.min_free_bytes
        )
        add_check(
            "disk_space",
            "Disk space",
            "passed" if disk_space_ok else "failed",
            (
                f"{disk_free_bytes} bytes free; configured minimum is {server.min_free_bytes} bytes."
                if disk_free_bytes is not None
                else "Compute-target free disk space could not be verified."
            ),
            blocking=True,
        )
        add_check(
            "disk_inodes",
            "File system inodes",
            (
                "unknown"
                if disk_inodes_free is None
                else "passed"
                if disk_inodes_free >= 1000
                else "failed"
            ),
            (
                "Free inode count was not reported."
                if disk_inodes_free is None
                else f"{disk_inodes_free} free inodes; at least 1000 are required."
            ),
            blocking=True,
        )
        add_check(
            "server_load",
            "Machine load",
            (
                "unknown"
                if load_ratio is None
                else "warning"
                if load_ratio > server.max_load_ratio
                else "passed"
            ),
            (
                "Load ratio could not be sampled."
                if load_ratio is None
                else (
                    f"Load ratio {load_ratio:.2f} exceeds the warning threshold "
                    f"{server.max_load_ratio:.2f}; scheduling remains allowed."
                    if load_ratio > server.max_load_ratio
                    else f"Load ratio {load_ratio:.2f}; warning threshold is {server.max_load_ratio:.2f}."
                )
            ),
            blocking=False,
        )
        if isinstance(nvidia, Mapping) and nvidia:
            gpu_status = "passed" if gpu_usable else "warning"
            gpu_detail = (
                f"{len(devices)} usable NVIDIA GPU(s) detected."
                if gpu_usable
                else _optional_text(nvidia.get("error"))
                or "No usable NVIDIA GPU was detected; CPU-only tasks remain allowed."
            )
        else:
            gpu_status = "unknown"
            gpu_detail = "GPU availability was not reported."
        add_check(
            "gpu",
            "GPU availability",
            gpu_status,
            gpu_detail,
            blocking=False,
        )

        environment_issues = [
            f"{item['label']}: {item['detail']}"
            for item in checks
            if item["blocking"] and item["status"] == "failed"
        ]
        total_memory = _optional_int(_mapping_get(memory, "total_bytes")) or 0
        available_memory = _optional_int(_mapping_get(memory, "available_bytes")) or 0
        used_memory = max(0, total_memory - available_memory)
        memory_percent = 100.0 * used_memory / total_memory if total_memory else 0.0
        os_parts = [
            str(_mapping_get(platform, key) or "").strip()
            for key in ("system", "release", "machine")
        ]
        return {
            "hostname": str(report.get("hostname") or server.host),
            "os": " ".join(part for part in os_parts if part),
            "python_version": environment_version
            or _mapping_get(python, "version"),
            "conda_env": _mapping_get(conda, "environment"),
            "conda_environment_available": conda_available,
            "environment_eligible": not any(
                item["blocking"] and item["status"] == "failed"
                for item in checks
            ),
            "environment_issues": environment_issues,
            "checks": checks,
            "cpu_model": _mapping_get(cpu, "model"),
            "cpu_physical": _optional_int(_mapping_get(cpu, "physical_cores")) or 0,
            "cpu_logical": _optional_int(_mapping_get(cpu, "logical_cores")) or 0,
            "cpu_percent": _optional_float(_mapping_get(cpu, "percent")) or 0.0,
            "load_1m": _optional_float(_mapping_get(cpu, "load_1m")),
            "load_ratio": load_ratio,
            "memory_total_gb": total_memory / 1024**3,
            "memory_used_gb": used_memory / 1024**3,
            "memory_percent": round(memory_percent, 2),
            "disk_free_gb": (disk_free_bytes or 0) / 1024**3,
            "disk_writable": disk_writable,
            "gpu_usable": gpu_usable,
            "gpu_error": _optional_text(_mapping_get(nvidia, "error")),
            "gpus": [
                _serialize_gpu_system(item)
                for item in devices
                if isinstance(item, Mapping)
            ],
        }

    @staticmethod
    def _serialize_git(run: Run, provenance: Mapping[str, Any]) -> dict[str, Any] | None:
        commit = run.git_commit or provenance.get("git_commit")
        snapshot = run.snapshot_sha256
        if not any((commit, snapshot, run.task_sha256)):
            return None
        return {
            "commit": str(commit or "unavailable"),
            "branch": run.git_branch,
            "dirty": bool(run.git_dirty),
            "snapshot_sha256": snapshot or "pending",
            "task_sha256": run.task_sha256 or "unavailable",
        }

    @staticmethod
    def _serialize_cleanup(run: Run) -> dict[str, Any]:
        if not run.cleanup_requested or not run.remote_run_dir:
            status = "skipped"
        elif run.remote_cleanup_completed:
            status = "completed"
        elif run.remote_cleanup_error:
            status = "failed"
        elif run.status is RunStatus.CLEANING:
            status = "running"
        else:
            status = "pending"
        return {
            "status": status,
            "attempted_at": run.remote_cleanup_at,
            "completed_at": run.remote_cleanup_at
            if run.remote_cleanup_completed
            else None,
            "error": run.remote_cleanup_error,
        }

    @staticmethod
    def _serialize_recovery(run: Run) -> dict[str, Any]:
        mode = run_recovery_mode(run)
        messages = {
            "transfer": "Resume remote monitoring, download, verification, and checksum-gated cleanup using the frozen run evidence.",
            "launch": "Recover the idempotent remote process identity, then continue monitoring and result retrieval.",
            "verify": "Re-run the complete local checksum, provenance, result-bundle, and durability gates for the already transferred exit-zero result. The scientific task will not run again; remote cleanup and completion happen only if every gate passes.",
            "cleanup": "Retry deletion of the exact marker-owned remote workspace.",
        }
        return {
            "available": mode is not None,
            "mode": mode,
            "message": messages.get(mode),
        }

    def _environment_report(
        self, run: Run, server: Server | None
    ) -> dict[str, Any] | None:
        stored = run.error_details.get("environment_report")
        if isinstance(stored, Mapping):
            return dict(stored)
        failed_probe = run.error_details.get("probe")
        if isinstance(failed_probe, Mapping):
            report = failed_probe
        else:
            execution_probe = run.task_payload.get("execution_probe")
            if isinstance(execution_probe, Mapping):
                report = execution_probe
            else:
                report = None
        if report is None:
            if server is None or not isinstance(server.last_probe, Mapping):
                return None
            _, report, _ = _probe_parts(server.last_probe)
            if report is None:
                return None
        snapshot = run.task_payload.get("server_snapshot")
        if not isinstance(snapshot, Mapping):
            snapshot = {}
        conda_env = str(
            snapshot.get("conda_env")
            or (server.conda_env if server is not None else "EDOLAB")
        )
        remote_root = str(
            snapshot.get("remote_root")
            or (server.remote_root if server is not None else "<unknown>")
        )
        min_free_bytes = _optional_int(snapshot.get("min_free_bytes"))
        if min_free_bytes is None:
            min_free_bytes = server.min_free_bytes if server is not None else 0
        max_load_ratio = _optional_float(snapshot.get("max_load_ratio"))
        if max_load_ratio is None:
            max_load_ratio = server.max_load_ratio if server is not None else 0.90
        checks: list[dict[str, str]] = []
        conda = report.get("conda", {})
        conda_ok = isinstance(conda, Mapping) and bool(
            conda.get("environment_available")
        )
        checks.append(
            {
                "name": "Conda environment",
                "status": "passed" if conda_ok else "failed",
                "detail": f"{conda_env}: {'available' if conda_ok else 'unavailable'}",
            }
        )
        environment_python = (
            conda.get("environment_python") if isinstance(conda, Mapping) else None
        )
        python_version = (
            _optional_text(environment_python.get("version"))
            if isinstance(environment_python, Mapping)
            else None
        )
        python_ok = _is_python_311(python_version)
        checks.append(
            {
                "name": "Environment Python",
                "status": "passed" if python_ok else "failed",
                "detail": python_version or "version probe unavailable; Python 3.11 required",
            }
        )
        pip_ok = (
            isinstance(environment_python, Mapping)
            and environment_python.get("pip_available") is True
        )
        checks.append(
            {
                "name": "Environment pip",
                "status": "passed" if pip_ok else "failed",
                "detail": "available" if pip_ok else "pip availability was not confirmed",
            }
        )
        for dependency in runtime_dependency_statuses(environment_python):
            checks.append(
                {
                    "name": dependency.spec.label,
                    "status": "passed" if dependency.available else "failed",
                    "detail": dependency.detail,
                }
            )
        disk = report.get("disk", {})
        writable = isinstance(disk, Mapping) and bool(disk.get("writable"))
        checks.append(
            {
                "name": "Remote workspace",
                "status": "passed" if writable else "failed",
                "detail": f"{remote_root}: {'writable' if writable else 'not writable'}",
            }
        )
        free_bytes = _optional_int(_mapping_get(disk, "free_bytes"))
        disk_space_ok = free_bytes is not None and free_bytes >= min_free_bytes
        checks.append(
            {
                "name": "Remote disk space",
                "status": "passed" if disk_space_ok else "failed",
                "detail": (
                    f"{free_bytes} bytes free; minimum {min_free_bytes}"
                    if free_bytes is not None
                    else "free-space probe unavailable"
                ),
            }
        )
        cpu = report.get("cpu", {})
        load_ratio = _optional_float(_mapping_get(cpu, "load_ratio"))
        load_ok = load_ratio is None or load_ratio <= max_load_ratio
        checks.append(
            {
                "name": "Server load",
                "status": "passed" if load_ok else "warning",
                "detail": (
                    f"load ratio {load_ratio:.2f}; threshold {max_load_ratio:.2f}"
                    if load_ratio is not None
                    else "load ratio unavailable"
                ),
            }
        )
        inspection = run.task_payload.get("inspection")
        gpu_requested = isinstance(inspection, Mapping) and bool(
            inspection.get("gpu_requested")
        )
        if gpu_requested:
            gpu = report.get("gpu", {})
            nvidia = gpu.get("nvidia_smi", {}) if isinstance(gpu, Mapping) else {}
            gpu_ok = isinstance(nvidia, Mapping) and bool(nvidia.get("usable"))
            checks.append(
                {
                    "name": "GPU availability",
                    "status": "passed" if gpu_ok else "failed",
                    "detail": (
                        "usable NVIDIA GPU detected"
                        if gpu_ok
                        else _optional_text(_mapping_get(nvidia, "error"))
                        or "no usable NVIDIA GPU detected"
                    ),
                }
            )
        return {
            "eligible": all(item["status"] != "failed" for item in checks),
            "checked_at": run.started_at
            or (server.last_probe_at if server is not None else None)
            or run.updated_at,
            "checks": checks,
        }

    @staticmethod
    def _serialize_audit_event(event: RunEvent, run_id: str) -> dict[str, Any]:
        data = dict(event.data or {})
        if event.level.value == "error":
            outcome = "failure"
        elif event.level.value == "warning" or data.get("admission_override"):
            outcome = "warning"
        else:
            outcome = "success"
        return {
            "id": str(event.id),
            "timestamp": event.created_at,
            "action": event.kind,
            "actor": str(data.get("actor", "controller")),
            "target": run_id,
            "detail": event.message,
            "outcome": outcome,
        }


def _frontend_status(status: RunStatus) -> str:
    return {
        RunStatus.QUEUED: "queued",
        RunStatus.CHECKING: "probing",
        RunStatus.SNAPSHOTTING: "preparing",
        RunStatus.SYNCING: "syncing",
        RunStatus.PREPARING: "preparing",
        RunStatus.RUNNING: "running",
        RunStatus.TRANSFERRING: "retrieving",
        RunStatus.VERIFYING: "retrieving",
        RunStatus.CLEANING: "cleaning",
        RunStatus.COMPLETED: "succeeded",
        RunStatus.FAILED: "failed",
        RunStatus.CANCELING: "cancelling",
        RunStatus.CANCELED: "cancelled",
        RunStatus.INTERRUPTED: "failed",
    }[status]


def _frontend_phase(
    status: RunStatus,
    details: Mapping[str, Any],
    stage: Mapping[str, Any] | None = None,
) -> str:
    staged = stage.get("phase") if isinstance(stage, Mapping) else None
    stage_state = stage.get("state") if isinstance(stage, Mapping) else None
    supplied = details.get("phase") if isinstance(details, Mapping) else None
    allowed = {
        "queued",
        "snapshot",
        "preflight",
        "sync",
        "remote_setup",
        "execute",
        "retrieve",
        "verify",
        "cleanup",
        "complete",
    }
    # A completed synchronization stage remains attached as provenance after
    # launch.  It must not relabel a later worker/transfer failure as a sync
    # failure; only an active sync or the stage that actually failed owns the
    # frontend phase.
    if staged in allowed and (
        status is RunStatus.SYNCING
        or (
            stage_state == "failed"
            and status in {RunStatus.FAILED, RunStatus.INTERRUPTED}
        )
    ):
        return str(staged)
    if supplied in allowed:
        return str(supplied)
    return {
        RunStatus.QUEUED: "queued",
        RunStatus.CHECKING: "preflight",
        RunStatus.SNAPSHOTTING: "snapshot",
        RunStatus.SYNCING: "sync",
        RunStatus.PREPARING: "remote_setup",
        RunStatus.RUNNING: "execute",
        RunStatus.TRANSFERRING: "retrieve",
        RunStatus.VERIFYING: "verify",
        RunStatus.CLEANING: "cleanup",
        RunStatus.COMPLETED: "complete",
        RunStatus.FAILED: "complete",
        RunStatus.CANCELING: "cleanup",
        RunStatus.CANCELED: "complete",
        RunStatus.INTERRUPTED: "complete",
    }[status]


def _artifact_kind(path: str) -> str:
    name = Path(path).name.lower()
    if name == "summary.sqlite":
        return "summary"
    if name == "trend.sqlite":
        return "trend"
    if name.endswith(".log") or "stdout" in name or "stderr" in name:
        return "log"
    if "manifest" in name or "provenance" in name:
        return "manifest"
    return "other"


def _mapping_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _probe_parts(
    value: Mapping[str, Any] | None,
) -> tuple[bool | None, Mapping[str, Any] | None, str | None]:
    """Accept service-wrapped probes and raw probes written by the controller."""

    if not isinstance(value, Mapping) or not value:
        return None, None, None
    if "ok" in value:
        ok = bool(value.get("ok"))
        report = value.get("report")
        return (
            ok,
            report if isinstance(report, Mapping) else None,
            _optional_text(value.get("error")),
        )
    if value.get("schema_version") == 1:
        return True, value, None
    return False, None, _optional_text(value.get("error")) or "Unsupported probe data"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_bounded_text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ServiceError(400, f"invalid_{name}", f"{name} must be text.")
    text = value.strip()
    if not text or len(text) > maximum or "\x00" in text:
        raise ServiceError(
            400,
            f"invalid_{name}",
            f"{name} must contain 1 to {maximum} safe characters.",
        )
    return text


def _normalize_run_display_name(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ServiceError(
            400,
            "invalid_display_name",
            "Run display name must not be empty or whitespace only.",
        )
    if len(text) > 200:
        raise ServiceError(
            400,
            "invalid_display_name",
            "Run display name must be at most 200 characters.",
        )
    if any(character in text for character in ("\x00", "\r", "\n")):
        raise ServiceError(
            400,
            "invalid_display_name",
            "Run display name must be a single line without control characters.",
        )
    return text


def _normalize_ssh_key_path(value: str) -> str:
    """Persist a local absolute path so Python and Docker share one DB value."""

    text = value.strip()
    if not text or "\n" in text or "\r" in text or "PRIVATE KEY" in text.upper():
        raise ValueError("ssh_key_path must contain a local path, not key material")
    return str(Path(text).expanduser().resolve(strict=False))


def _normalize_remote_root(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("remote_root is required")
    if text in {"/", "~", ".", ".."}:
        raise ValueError("remote_root must be a dedicated subdirectory, not a broad path")
    if any(token in text for token in ("\x00", "\n", "\r", "$", "*", "?", "[", "]")):
        raise ValueError("remote_root contains unsupported shell or glob characters")
    return text


def _execution_target_key(server: Server) -> tuple[str, str, int, str, str]:
    return (
        server.target_kind.value,
        server.host.strip().casefold(),
        int(server.port),
        server.username.strip(),
        server.conda_env.strip(),
    )


def _is_python_311(version: str | None) -> bool:
    if version is None:
        return False
    parts = version.split(".")
    return len(parts) >= 2 and parts[0] == "3" and parts[1] == "11"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive_int(
    value: Any, name: str, *, minimum: int = 1, maximum: int | None = None
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ServiceError(400, "invalid_input", f"{name} must be an integer.") from exc
    if result < minimum or (maximum is not None and result > maximum):
        detail = f" between {minimum} and {maximum}" if maximum else f" at least {minimum}"
        raise ServiceError(400, "invalid_input", f"{name} must be{detail}.")
    return result


def _positive_float(
    value: Any,
    name: str,
    *,
    minimum: float = 1,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ServiceError(400, "invalid_input", f"{name} must be numeric.") from exc
    if not math.isfinite(result) or result < minimum or (
        maximum is not None and result > maximum
    ):
        detail = f" between {minimum} and {maximum}" if maximum else f" at least {minimum}"
        raise ServiceError(400, "invalid_input", f"{name} must be{detail}.")
    return result


def _timestamp_from_epoch(value: float) -> str:
    return (
        datetime.fromtimestamp(value, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_external_result_archive(value: Path, data_root: Path) -> Path:
    raw = value.expanduser()
    if not raw.is_absolute():
        raise ServiceError(
            400,
            "unsafe_result_archive",
            "Result archive path must be absolute.",
        )
    if ".." in raw.parts:
        raise ServiceError(
            400,
            "unsafe_result_archive",
            "Result archive path must not contain '..'.",
        )
    current = Path(raw.anchor)
    try:
        for part in raw.parts[1:]:
            current /= part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ServiceError(
                    400,
                    "unsafe_result_archive",
                    "Result archive path must not contain symlinked components.",
                )
        resolved = raw.resolve(strict=True)
        info = resolved.lstat()
    except FileNotFoundError as exc:
        raise ServiceError(
            400,
            "unsafe_result_archive",
            "Result archive does not exist.",
        ) from exc
    except OSError as exc:
        raise ServiceError(400, "unsafe_result_archive", str(exc)) from exc
    if not stat.S_ISREG(info.st_mode):
        raise ServiceError(
            400,
            "unsafe_result_archive",
            "Result archive must be a regular file.",
        )
    if _is_within(resolved, data_root.resolve()):
        raise ServiceError(
            400,
            "source_inside_data_root",
            "External result archive must be outside the Relay data root.",
        )
    return resolved


def _extract_single_result_archive(archive_path: Path, target_root: Path) -> Path:
    items = _extract_result_archive(
        archive_path,
        target_root,
        allow_multiple=False,
    )
    return items[0]["bundle"]


def _result_bundle_runset_records(bundle: Path) -> list[dict[str, Any]]:
    """Return content fingerprints that are independent of bundle-local IDs."""

    summary_path = bundle / "summary.sqlite"
    trend_path = bundle / "trend.sqlite"
    try:
        validate_result_bundle(bundle, require_completed=True)
        with sqlite3.connect(
            f"{summary_path.resolve().as_uri()}?mode=ro",
            uri=True,
        ) as summary, sqlite3.connect(
            f"{trend_path.resolve().as_uri()}?mode=ro",
            uri=True,
        ) as trend:
            summary.row_factory = sqlite3.Row
            trend.row_factory = sqlite3.Row
            summary.execute("pragma query_only = on")
            trend.execute("pragma query_only = on")
            runsets = summary.execute(
                """
                select rs.id, rs.runset_uid, rs.label, rs.algorithm_name,
                       rs.problem_name, rs.algorithm_params_json,
                       rs.expected_runs, rs.completed_runs, rs.status,
                       t.problem_name as task_problem_name,
                       t.run_number, t.problem_params_json,
                       t.metric_output_names_json
                from runsets as rs
                join tasks as t on t.id = rs.task_id
                order by t.id, rs.id
                """
            ).fetchall()
            records: list[dict[str, Any]] = []
            for row in runsets:
                runset_id = int(row["id"])
                runset_uid = str(row["runset_uid"])
                runs = [
                    {
                        "run_index": int(item["run_index"]),
                        "problem_seed": int(item["problem_seed"]),
                        "algorithm_seed": int(item["algorithm_seed"]),
                        "fe": int(item["fe"]),
                        "runtime_seconds": float(item["runtime_seconds"]),
                        "status": str(item["status"]),
                        "compute": json.loads(item["compute_json"]),
                        "iteration": int(item["iteration"]),
                    }
                    for item in summary.execute(
                        """
                        select run_index, problem_seed, algorithm_seed, fe,
                               runtime_seconds, status, compute_json, iteration
                        from runs
                        where runset_id = ?
                        order by run_index
                        """,
                        (runset_id,),
                    ).fetchall()
                ]
                scalar_values = [
                    [
                        int(item["run_index"]),
                        str(item["metric_name"]),
                        float(item["value"]),
                        str(item["direction"]),
                    ]
                    for item in summary.execute(
                        """
                        select r.run_index, rmv.metric_name, rmv.value, rmv.direction
                        from run_metric_values as rmv
                        join runs as r on r.id = rmv.run_id
                        where rmv.runset_id = ?
                        order by r.run_index, rmv.metric_name
                        """,
                        (runset_id,),
                    ).fetchall()
                ]
                summary_metrics = [
                    [
                        str(item["metric_name"]),
                        float(item["mean"]),
                        float(item["median"]),
                        float(item["stderr"]),
                        int(item["count"]),
                        str(item["direction"]),
                    ]
                    for item in summary.execute(
                        """
                        select metric_name, mean, median, stderr, count, direction
                        from summary_metrics
                        where runset_id = ?
                        order by metric_name
                        """,
                        (runset_id,),
                    ).fetchall()
                ]
                trends = [
                    [
                        str(item["metric_name"]),
                        str(item["axis"]),
                        int(item["sample_count"]),
                        hashlib.sha256(bytes(item["payload"])).hexdigest(),
                    ]
                    for item in trend.execute(
                        """
                        select tm.metric_name, tm.axis, tm.sample_count, tb.payload
                        from trend_metrics as tm
                        join trend_blobs as tb on tb.trend_metric_id = tm.id
                        where tm.runset_uid = ?
                        order by tm.metric_name
                        """,
                        (runset_uid,),
                    ).fetchall()
                ]
                content = {
                    "problem_name": str(row["problem_name"]),
                    "task_problem_name": str(row["task_problem_name"]),
                    "problem_params": json.loads(row["problem_params_json"]),
                    "algorithm_name": str(row["algorithm_name"]),
                    "algorithm_params": json.loads(row["algorithm_params_json"]),
                    "metric_output_names": json.loads(row["metric_output_names_json"]),
                    "run_number": int(row["run_number"]),
                    "expected_runs": int(row["expected_runs"]),
                    "completed_runs": int(row["completed_runs"]),
                    "status": str(row["status"]),
                    "runs": runs,
                    "scalar_values": scalar_values,
                    "summary_metrics": summary_metrics,
                    "trends": trends,
                }
                records.append(
                    {
                        "runset_uid": runset_uid,
                        "display_name": (
                            f"{str(row['label'])} · {str(row['algorithm_name'])}"
                        ),
                        "fingerprint": hashlib.sha256(
                            stable_json_dumps(content).encode("utf-8")
                        ).hexdigest(),
                    }
                )
    except ResultBundleValidationError:
        raise
    except (sqlite3.Error, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResultBundleValidationError(
            f"Runset content could not be read safely: {exc}"
        ) from exc
    if not records:
        raise ResultBundleValidationError("Result bundle contains no Runsets.")
    return records


def _create_result_runset_subset(
    source_bundle: Path,
    target_bundle: Path,
    runset_uids: set[str],
) -> None:
    """Create a valid completed bundle containing exactly the selected Runsets."""

    if not runset_uids:
        raise ServiceError(
            400,
            "result_subset_empty",
            "At least one Runset is required for a partial result import.",
        )
    target_bundle.mkdir(mode=0o700)
    for filename in ("summary.sqlite", "trend.sqlite"):
        shutil.copyfile(source_bundle / filename, target_bundle / filename)
        (target_bundle / filename).chmod(0o600)

    summary_path = target_bundle / "summary.sqlite"
    trend_path = target_bundle / "trend.sqlite"
    try:
        with sqlite3.connect(summary_path) as summary:
            summary.execute("pragma foreign_keys = off")
            available = {
                str(row[0])
                for row in summary.execute("select runset_uid from runsets").fetchall()
            }
            if not runset_uids.issubset(available):
                raise ServiceError(
                    400,
                    "result_subset_unknown_runset",
                    "The selected Runset no longer exists in the uploaded result.",
                )
            summary.execute("create temp table selected_runsets(uid text primary key)")
            summary.executemany(
                "insert into selected_runsets(uid) values (?)",
                [(uid,) for uid in sorted(runset_uids)],
            )
            selected_ids = "select id from runsets where runset_uid in (select uid from selected_runsets)"
            summary.execute(
                f"delete from run_metric_values where runset_id not in ({selected_ids})"
            )
            # Versioned audit records travel inside summary.sqlite. Keep only
            # selected run identities, including streams with zero events.
            from core.result.event_streams import has_event_streams
            if has_event_streams(summary):
                retained_runs = f"select run_uid from runs where runset_id in ({selected_ids})"
                summary.execute(f"delete from run_event_records where run_uid not in ({retained_runs})")
                summary.execute(f"delete from run_event_streams where run_uid not in ({retained_runs})")
            summary.execute(f"delete from runs where runset_id not in ({selected_ids})")
            summary.execute(
                f"delete from summary_metrics where runset_id not in ({selected_ids})"
            )
            summary.execute(
                f"delete from runset_algorithm_params where runset_id not in ({selected_ids})"
            )
            summary.execute(
                "delete from runsets where runset_uid not in (select uid from selected_runsets)"
            )
            summary.execute(
                "delete from task_problem_params where task_id not in (select distinct task_id from runsets)"
            )
            summary.execute(
                "delete from tasks where id not in (select distinct task_id from runsets)"
            )
            summary.execute("delete from result_events")
            summary.commit()
            summary.execute("vacuum")

        with sqlite3.connect(trend_path) as trend:
            trend.execute("pragma foreign_keys = off")
            trend.execute("create temp table selected_runsets(uid text primary key)")
            trend.executemany(
                "insert into selected_runsets(uid) values (?)",
                [(uid,) for uid in sorted(runset_uids)],
            )
            trend.execute(
                """
                delete from trend_blobs
                where trend_metric_id in (
                    select id from trend_metrics
                    where runset_uid not in (select uid from selected_runsets)
                )
                """
            )
            trend.execute(
                "delete from trend_metrics where runset_uid not in (select uid from selected_runsets)"
            )
            trend.commit()
            trend.execute("vacuum")
    except ServiceError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise ServiceError(
            400,
            "result_subset_failed",
            f"Uploaded Runsets could not be isolated safely: {exc}",
        ) from exc

    fsync_file(summary_path)
    fsync_file(trend_path)
    fsync_directory(target_bundle)
    try:
        validate_result_bundle(target_bundle, require_completed=True)
    except ResultBundleValidationError as exc:
        raise ServiceError(
            400,
            "invalid_result_subset",
            f"The filtered Runset bundle is invalid: {exc}",
        ) from exc


def _extract_result_archive(
    archive_path: Path,
    target_root: Path,
    *,
    allow_multiple: bool = True,
) -> list[dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(archive_path, flags)
    except OSError as exc:
        raise ServiceError(
            400,
            "unsafe_result_archive",
            f"Result archive could not be opened safely: {exc}",
        ) from exc
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ServiceError(
                400,
                "unsafe_result_archive",
                "Result archive must remain a regular file.",
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            with zipfile.ZipFile(handle, "r", allowZip64=True) as archive:
                infos = archive.infolist()
                prefixes = _validate_result_archive_members(
                    infos,
                    allow_multiple=allow_multiple,
                )
                by_name = {item.filename: item for item in infos}
                manifest = _read_result_archive_member(
                    archive,
                    by_name[RESULT_ARCHIVE_MANIFEST_NAME],
                    maximum=RESULT_ARCHIVE_MAX_MANIFEST_BYTES,
                )
                manifest_items = _validate_result_archive_manifest(
                    manifest,
                    prefixes,
                )
                database_infos = [
                    by_name[f"{prefix}/{filename}"]
                    for prefix, _ in manifest_items
                    for filename in ("summary.sqlite", "trend.sqlite")
                ]
                database_bytes = sum(item.file_size for item in database_infos)
                if database_bytes > RESULT_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                    raise ServiceError(
                        400,
                        "result_archive_too_large",
                        "Result archive exceeds the supported uncompressed size.",
                    )
                free_bytes = shutil.disk_usage(target_root).free
                required_bytes = database_bytes * 2 + 16 * 1024 * 1024
                if required_bytes > free_bytes:
                    raise ServiceError(
                        409,
                        "result_archive_insufficient_space",
                        "Not enough Relay disk space to safely import this result archive.",
                    )
                extracted: list[dict[str, Any]] = []
                for index, (prefix, manifest_item) in enumerate(
                    manifest_items,
                    start=1,
                ):
                    bundle = target_root / f"bundle-{index}"
                    bundle.mkdir(mode=0o700)
                    hashes: dict[str, str] = {}
                    for filename in ("summary.sqlite", "trend.sqlite"):
                        info = by_name[f"{prefix}/{filename}"]
                        expected_file = manifest_item["files"][filename]
                        if info.file_size != expected_file["size_bytes"]:
                            raise ServiceError(
                                400,
                                "result_archive_manifest_mismatch",
                                f"{prefix}/{filename} size does not match manifest.json.",
                            )
                        destination = bundle / filename
                        hashes[filename] = _copy_result_archive_member(
                            archive,
                            info,
                            destination,
                        )
                        if hashes[filename] != expected_file["sha256"]:
                            raise ServiceError(
                                400,
                                "result_archive_manifest_mismatch",
                                f"{prefix}/{filename} checksum does not match manifest.json.",
                            )
                    extracted.append(
                        {
                            "prefix": prefix,
                            "manifest_item": manifest_item,
                            "bundle": bundle,
                            "summary_sha256": hashes["summary.sqlite"],
                            "trend_sha256": hashes["trend.sqlite"],
                        }
                    )
            current = os.fstat(handle.fileno())
            if _archive_file_identity(current) != _archive_file_identity(initial):
                raise ServiceError(
                    409,
                    "result_archive_changed",
                    "Result archive changed while it was imported.",
                )
    except ServiceError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        NotImplementedError,
        RuntimeError,
        OSError,
    ) as exc:
        raise ServiceError(
            400,
            "invalid_result_archive",
            f"Result archive is invalid or unsupported: {exc}",
        ) from exc
    finally:
        os.close(descriptor)
    return extracted


def _validate_result_archive_members(
    infos: list[zipfile.ZipInfo],
    *,
    allow_multiple: bool,
) -> list[str]:
    if not infos:
        raise ServiceError(400, "invalid_result_archive", "Result archive is empty.")
    if len(infos) > RESULT_ARCHIVE_MAX_RESULTS * 2 + 1:
        raise ServiceError(
            400,
            "result_archive_too_many_results",
            f"Result archive may contain at most {RESULT_ARCHIVE_MAX_RESULTS} results.",
        )
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        raise ServiceError(
            400,
            "invalid_result_archive",
            "Result archive contains duplicate paths.",
        )
    database_prefixes: set[str] = set()
    for info in infos:
        name = info.filename
        parts = name.split("/")
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or info.is_dir()
            or bool(info.flag_bits & 0x1)
        ):
            raise ServiceError(
                400,
                "unsafe_result_archive",
                "Result archive contains an unsafe or encrypted member.",
            )
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode) if mode else 0
        if file_type not in {0, stat.S_IFREG}:
            raise ServiceError(
                400,
                "unsafe_result_archive",
                "Result archive members must be regular files.",
            )
        if len(path.parts) == 2 and path.name in {"summary.sqlite", "trend.sqlite"}:
            if len(path.parts[0]) > 200:
                raise ServiceError(
                    400,
                    "invalid_result_archive",
                    "Result archive prefix is too long.",
                )
            database_prefixes.add(path.parts[0])
    if not database_prefixes:
        raise ServiceError(
            400,
            "invalid_result_archive",
            "Result archive contains no standard EDOLAB result bundle.",
        )
    if len(database_prefixes) > RESULT_ARCHIVE_MAX_RESULTS:
        raise ServiceError(
            400,
            "result_archive_too_many_results",
            f"Result archive may contain at most {RESULT_ARCHIVE_MAX_RESULTS} results.",
        )
    if not allow_multiple and len(database_prefixes) > 1:
        raise ServiceError(
            400,
            "result_archive_multiple_results",
            "This ZIP contains multiple results. Extract it and import each result directory separately.",
        )
    expected = {RESULT_ARCHIVE_MANIFEST_NAME}
    for prefix in database_prefixes:
        expected.update(
            {
                f"{prefix}/summary.sqlite",
                f"{prefix}/trend.sqlite",
            }
        )
    if set(names) != expected:
        raise ServiceError(
            400,
            "invalid_result_archive",
            "Result archive must use the standard EDOLAB export ZIP layout.",
        )
    manifest = next(item for item in infos if item.filename == RESULT_ARCHIVE_MANIFEST_NAME)
    if manifest.file_size > RESULT_ARCHIVE_MAX_MANIFEST_BYTES:
        raise ServiceError(
            400,
            "invalid_result_archive",
            "Result archive manifest is too large.",
        )
    return sorted(database_prefixes)


def _read_result_archive_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    maximum: int,
) -> bytes:
    if info.file_size > maximum:
        raise ServiceError(
            400,
            "invalid_result_archive",
            "Result archive member exceeds its permitted size.",
        )
    total = 0
    chunks: list[bytes] = []
    with archive.open(info, "r") as source:
        while True:
            chunk = source.read(RESULT_ARCHIVE_COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ServiceError(
                    400,
                    "invalid_result_archive",
                    "Result archive member exceeds its permitted size.",
                )
            chunks.append(chunk)
    if total != info.file_size:
        raise ServiceError(
            400,
            "invalid_result_archive",
            "Result archive member size changed during extraction.",
        )
    return b"".join(chunks)


def _validate_result_archive_manifest(
    raw_manifest: bytes,
    prefixes: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceError(
            400,
            "invalid_result_archive_manifest",
            "manifest.json must be valid UTF-8 JSON.",
        ) from exc
    schema_version = manifest.get("schema_version") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ServiceError(
            400,
            "invalid_result_archive_manifest",
            "manifest.json must declare EDOLAB result export schema_version 1.",
        )
    raw_results = manifest.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(prefixes):
        raise ServiceError(
            400,
            "invalid_result_archive_manifest",
            "manifest.json result count does not match the ZIP contents.",
        )
    results: list[dict[str, Any]] = []
    for raw_item in raw_results:
        if not isinstance(raw_item, Mapping):
            raise ServiceError(
                400,
                "invalid_result_archive_manifest",
                "Each manifest result must be an object.",
            )
        result_id = raw_item.get("result_id")
        files = raw_item.get("files")
        if not isinstance(result_id, str) or not result_id or not isinstance(files, Mapping):
            raise ServiceError(
                400,
                "invalid_result_archive_manifest",
                "Each manifest result must declare result_id and files.",
            )
        normalized_files: dict[str, dict[str, Any]] = {}
        for filename in ("summary.sqlite", "trend.sqlite"):
            raw_file = files.get(filename)
            if not isinstance(raw_file, Mapping):
                raise ServiceError(
                    400,
                    "invalid_result_archive_manifest",
                    f"Manifest result {result_id} is missing {filename} evidence.",
                )
            size_bytes = raw_file.get("size_bytes")
            sha256 = raw_file.get("sha256")
            if (
                not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256.lower())
            ):
                raise ServiceError(
                    400,
                    "invalid_result_archive_manifest",
                    f"Manifest evidence for {result_id}/{filename} is invalid.",
                )
            normalized_files[filename] = {
                "size_bytes": size_bytes,
                "sha256": sha256.lower(),
            }
        results.append({**dict(raw_item), "files": normalized_files})

    if len(prefixes) == 1 and prefixes[0] == RESULT_ARCHIVE_SINGLE_PREFIX:
        return [(prefixes[0], results[0])]
    by_result_id = {str(item["result_id"]): item for item in results}
    if len(by_result_id) != len(results) or set(by_result_id) != set(prefixes):
        raise ServiceError(
            400,
            "invalid_result_archive_manifest",
            "Multi-result ZIP prefixes must match unique manifest result_id values.",
        )
    return [(str(item["result_id"]), item) for item in results]


def _archive_result_display_name(
    manifest_item: Mapping[str, Any],
    *,
    filename: str,
    index: int,
    total: int,
) -> str:
    raw_name = manifest_item.get("display_name")
    candidates = [raw_name, manifest_item.get("result_id")]
    archive_stem = Path(filename).stem.strip() or "Uploaded result"
    candidates.append(archive_stem if total == 1 else f"{archive_stem} · {index}")
    for candidate in candidates:
        try:
            return _normalize_run_display_name(candidate)
        except ServiceError:
            continue
    return f"Uploaded result · {index}"


def _copy_result_archive_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    total = 0
    digest = hashlib.sha256()
    try:
        with archive.open(info, "r") as source:
            while True:
                chunk = source.read(RESULT_ARCHIVE_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
                if total > info.file_size:
                    raise ServiceError(
                        400,
                        "invalid_result_archive",
                        "Result archive member expanded beyond its declared size.",
                    )
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        if total != info.file_size:
            raise ServiceError(
                400,
                "invalid_result_archive",
                "Result archive member ended before its declared size.",
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _archive_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        stat.S_IFMT(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _remove_staged_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_gpu_system(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": _optional_int(item.get("index")) or 0,
        "name": str(item.get("name") or "NVIDIA GPU"),
        "memory_total_mb": _optional_int(
            item.get("memory.total", item.get("memory_total_mb"))
        )
        or 0,
        "memory_used_mb": _optional_int(
            item.get("memory.used", item.get("memory_used_mb"))
        )
        or 0,
        "utilization_percent": _optional_float(
            item.get("utilization.gpu", item.get("utilization_percent"))
        )
        or 0.0,
        "temperature_c": _optional_float(
            item.get("temperature.gpu", item.get("temperature_c"))
        ),
    }


def _serialize_gpu_sample(item: Mapping[str, Any]) -> dict[str, Any]:
    system = _serialize_gpu_system(item)
    return {
        "index": system["index"],
        "utilization_percent": system["utilization_percent"],
        "memory_used_mb": system["memory_used_mb"],
        "memory_total_mb": system["memory_total_mb"],
    }


def _serialize_gpu_resource(item: Mapping[str, Any]) -> dict[str, Any]:
    system = _serialize_gpu_system(item)
    return {
        "index": system["index"],
        "uuid": _optional_text(item.get("uuid")),
        "name": system["name"],
        "memory_total_mb": _optional_int(
            item.get("memory.total", item.get("memory_total_mb"))
        ),
        "memory_used_mb": _optional_int(
            item.get("memory.used", item.get("memory_used_mb"))
        ),
        "utilization_percent": _optional_float(
            item.get("utilization.gpu", item.get("utilization_percent"))
        ),
        "temperature_celsius": _optional_float(
            item.get(
                "temperature.gpu",
                item.get("temperature_celsius", item.get("temperature_c")),
            )
        ),
    }


def _number_for_json(value: float) -> int | float:
    return int(value) if value.is_integer() else value


__all__ = [
    "Controller",
    "DEFAULT_SETTINGS",
    "ProbeClient",
    "RelayService",
    "ServiceError",
]
