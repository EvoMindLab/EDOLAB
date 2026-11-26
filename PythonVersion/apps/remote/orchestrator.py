"""Durable one-job-per-server scheduler for EDOLAB Relay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from apps.remote.config import RelayPaths
from apps.remote.database import RemoteDatabase, ServerTargetChangedError
from apps.remote.durability import fsync_directory, fsync_directory_tree, fsync_file
from apps.remote.domain import (
    Artifact,
    ArtifactStatus,
    EventLevel,
    ExecutionTargetKind,
    LOCAL_TRANSFER_VERIFICATION_ERROR,
    ResourceSample,
    Run,
    RunEvent,
    RunStatus,
    Server,
    new_id,
    run_recovery_mode,
    stable_json_dumps,
    utc_now,
)
from apps.remote.local_transport import LocalProcessClient
from apps.remote.environment import missing_runtime_dependency_messages
from apps.remote.remote_runtime import (
    RemoteArtifact,
    RemotePoll,
    RemoteWorkspace,
    build_remote_artifact,
    cancel_remote_worker,
    cleanup_remote_workspace,
    discover_remote_worker,
    extract_verified_artifact,
    launch_remote_worker,
    poll_remote_worker,
    prepare_remote_workspace,
    upload_and_verify_source,
)
from apps.remote.snapshot import (
    SnapshotBuilder,
    canonical_manifest_hash,
    diff_manifests,
    write_snapshot_delta_archive,
)
from apps.remote.ssh import SSHClient, SSHCommandError, SSHConnection, SSHError
from apps.remote.tasking import (
    build_task_provenance,
    inspect_batch_task,
    materialize_inline_task,
    task_provenance_identity,
)
from core.result.validation import validate_result_bundle


_DURABLE_PROVENANCE_FILENAMES = (
    "source.tar.gz",
    "source-manifest.json",
    "task.jsonl",
    "task-provenance.json",
    "server-probe.json",
    "source-diff.json",
)


class ControllerStopping(RuntimeError):
    """Internal sentinel used to leave a remote process running on shutdown."""


class PreRemoteCanceled(RuntimeError):
    """Internal sentinel for cancellation before a worker process was launched."""


class AdmissionError(RuntimeError):
    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class ExecutionBackend(Protocol):
    def client(self, server: Server, paths: RelayPaths) -> Any: ...

    def probe(self, client: Any, server: Server) -> Mapping[str, Any]: ...

    def prepare(
        self,
        client: Any,
        server: Server,
        installation_id: str,
        run_id: str,
        snapshot_id: str,
    ) -> RemoteWorkspace: ...

    def sync(
        self,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        payload_path: Path | None,
        manifest_path: Path,
        task_path: Path,
        snapshot_id: str,
        sync_plan: Mapping[str, Any],
        stage_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> Mapping[str, Any]: ...

    def launch(
        self,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        conda_env: str,
        conda_executable: str,
        max_parallel_runs: int,
        worker_kind: str = "batch",
    ) -> int: ...

    def recover_process(
        self,
        client: Any,
        workspace: RemoteWorkspace,
    ) -> int: ...

    def poll(
        self,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        remote_pid: int,
        stdout_offset: int,
        stderr_offset: int,
    ) -> RemotePoll: ...

    def cancel(
        self,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        remote_pid: int,
        force: bool,
    ) -> Mapping[str, Any]: ...

    def artifact(
        self,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        worker_kind: str = "batch",
    ) -> RemoteArtifact: ...

    def download(
        self,
        client: Any,
        remote_path: str,
        local_path: Path,
        *,
        expected_size: int,
    ) -> Any: ...

    def cleanup(
        self,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        installation_id: str,
        run_id: str,
    ) -> Mapping[str, Any]: ...


class OpenSSHExecutionBackend:
    def client(self, server: Server, paths: RelayPaths) -> SSHClient:
        return SSHClient(
            SSHConnection(
                host=server.host,
                username=server.username,
                key_path=server.ssh_key_path,
                port=server.port,
                known_hosts_path=paths.known_hosts,
                control_dir=paths.ssh_mux / server.id,
            )
        )

    def probe(self, client: SSHClient, server: Server) -> Mapping[str, Any]:
        return client.probe(server.remote_root, server.conda_env)

    def prepare(
        self,
        client: SSHClient,
        server: Server,
        installation_id: str,
        run_id: str,
        snapshot_id: str,
    ) -> RemoteWorkspace:
        return prepare_remote_workspace(
            client,
            remote_root=server.remote_root,
            installation_id=installation_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
        )

    def sync(
        self,
        client: SSHClient,
        workspace: RemoteWorkspace,
        *,
        payload_path: Path | None,
        manifest_path: Path,
        task_path: Path,
        snapshot_id: str,
        sync_plan: Mapping[str, Any],
        stage_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> Mapping[str, Any]:
        return upload_and_verify_source(
            client,
            workspace=workspace,
            payload_path=payload_path,
            manifest_path=manifest_path,
            task_path=task_path,
            snapshot_id=snapshot_id,
            sync_plan=sync_plan,
            stage_callback=stage_callback,
        )

    def launch(
        self,
        client: SSHClient,
        workspace: RemoteWorkspace,
        *,
        conda_env: str,
        conda_executable: str,
        max_parallel_runs: int,
        worker_kind: str = "batch",
    ) -> int:
        return launch_remote_worker(
            client,
            workspace=workspace,
            conda_env=conda_env,
            conda_executable=conda_executable,
            max_parallel_runs=max_parallel_runs,
            worker_kind=worker_kind,
        )

    def poll(
        self,
        client: SSHClient,
        workspace: RemoteWorkspace,
        *,
        remote_pid: int,
        stdout_offset: int,
        stderr_offset: int,
    ) -> RemotePoll:
        return poll_remote_worker(
            client,
            workspace=workspace,
            remote_pid=remote_pid,
            stdout_offset=stdout_offset,
            stderr_offset=stderr_offset,
        )

    def recover_process(
        self,
        client: SSHClient,
        workspace: RemoteWorkspace,
    ) -> int:
        return discover_remote_worker(client, workspace=workspace)

    def cancel(
        self,
        client: SSHClient,
        workspace: RemoteWorkspace,
        *,
        remote_pid: int,
        force: bool,
    ) -> Mapping[str, Any]:
        return cancel_remote_worker(
            client,
            workspace=workspace,
            remote_pid=remote_pid,
            force=force,
        )

    def artifact(
        self,
        client: SSHClient,
        workspace: RemoteWorkspace,
        *,
        worker_kind: str = "batch",
    ) -> RemoteArtifact:
        return build_remote_artifact(
            client,
            workspace=workspace,
            worker_kind=worker_kind,
        )

    def download(
        self,
        client: SSHClient,
        remote_path: str,
        local_path: Path,
        *,
        expected_size: int,
    ) -> Any:
        return client.download_file(
            remote_path,
            local_path,
            timeout=_artifact_download_timeout(expected_size),
        )

    def cleanup(
        self,
        client: SSHClient,
        workspace: RemoteWorkspace,
        *,
        installation_id: str,
        run_id: str,
    ) -> Mapping[str, Any]:
        return cleanup_remote_workspace(
            client,
            workspace=workspace,
            installation_id=installation_id,
            run_id=run_id,
        )


class UnifiedExecutionBackend(OpenSSHExecutionBackend):
    """Dispatch the shared durable protocol to SSH or an explicit local client."""

    def client(self, server: Server, paths: RelayPaths) -> Any:
        if server.target_kind is ExecutionTargetKind.LOCAL:
            return LocalProcessClient()
        return super().client(server, paths)

    def probe(self, client: Any, server: Server) -> Mapping[str, Any]:
        return client.probe(server.remote_root, server.conda_env)

    def launch(
        self,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        conda_env: str,
        conda_executable: str,
        max_parallel_runs: int,
        worker_kind: str = "batch",
    ) -> int:
        return launch_remote_worker(
            client,
            workspace=workspace,
            conda_env=conda_env,
            conda_executable=conda_executable,
            max_parallel_runs=max_parallel_runs,
            worker_kind=worker_kind,
            # Installing a local snapshot editable would repoint the user's
            # shared EDOLAB environment at a temporary directory.  Local jobs
            # instead execute from the frozen source through PYTHONPATH.
            install_editable=not isinstance(client, LocalProcessClient),
        )


@dataclass(slots=True)
class _MonitorState:
    stdout_offset: int = 0
    stderr_offset: int = 0
    stdout_pending: bytes = b""
    cancel_attempted_at: float | None = None
    cancel_sent_at: float | None = None
    force_attempted_at: float | None = None
    force_sent: bool = False
    consecutive_errors: int = 0
    dead_without_exit_count: int = 0


@dataclass(frozen=True, slots=True)
class _FrozenWorkloadInspection:
    task_bytes: bytes
    task_file_sha256: str
    workload_kind: str
    operation: str
    gpu_requested: bool
    total_runs: int = 1
    total_evals: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workload_kind": self.workload_kind,
            "operation": self.operation,
            "task_file_sha256": self.task_file_sha256,
            "gpu_requested": self.gpu_requested,
            "total_runs": self.total_runs,
            "total_evals": self.total_evals,
        }


class RelayController:
    """Persisted queue with one active immutable source snapshot per server."""

    def __init__(
        self,
        database: RemoteDatabase,
        paths: RelayPaths,
        *,
        backend: ExecutionBackend | None = None,
    ) -> None:
        self.database = database
        self.paths = paths
        self.backend = backend or UnifiedExecutionBackend()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._scheduler: threading.Thread | None = None
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._thread_targets: dict[str, str] = {}
        self._active_targets: set[str] = set()
        self._next_cleanup_retry = 0.0

    def start(self) -> None:
        with self._lock:
            if self._scheduler is not None and self._scheduler.is_alive():
                if self._stop.is_set():
                    raise RuntimeError(
                        "relay controller is still stopping; wait for shutdown "
                        "to finish before restarting"
                    )
                return
            self.paths.ensure()
            self.database.initialize()
            self._stop.clear()
            self._recover_runs()
            self._scheduler = threading.Thread(
                target=self._scheduler_loop,
                name="edolab-relay-scheduler",
                daemon=True,
            )
            self._scheduler.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        scheduler = self._scheduler
        if scheduler is not None:
            scheduler.join(timeout=max(0.0, timeout))
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def wake(self) -> None:
        self._wake.set()

    @property
    def running(self) -> bool:
        scheduler = self._scheduler
        return scheduler is not None and scheduler.is_alive()

    def request_cancel(self, run_id: str) -> Run:
        with self._lock:
            run = self.database.require_run(run_id)
            if run.terminal:
                return run
            # Retrieval and cleanup protect an already completed scientific
            # result.  A late cancel must not interrupt checksum verification
            # or turn a successful run into a canceled one.
            if run.status in {
                RunStatus.TRANSFERRING,
                RunStatus.VERIFYING,
                RunStatus.CLEANING,
            }:
                self._event(
                    run.id,
                    "cancel",
                    "Cancellation arrived after remote execution ended; verified result retrieval continues.",
                    EventLevel.WARNING,
                )
                return run
            thread = self._threads.get(run.id)
            active = thread is not None and thread.is_alive()
            try:
                if run.status is RunStatus.QUEUED and not active:
                    run = self.database.set_run_status(
                        run.id,
                        RunStatus.CANCELING,
                        error_code="CANCEL_REQUESTED",
                    )
                    return self.database.set_run_status(
                        run.id,
                        RunStatus.CANCELED,
                        exit_code=130,
                        error_code="CANCELED_BEFORE_START",
                        error_message="Canceled before remote execution started.",
                    )
                if run.status is not RunStatus.CANCELING:
                    run = self.database.set_run_status(
                        run.id,
                        RunStatus.CANCELING,
                        error_code="CANCEL_REQUESTED",
                        event_message="Cancellation requested; terminating the remote process group.",
                    )
            except ValueError:
                # A worker may have reached a terminal state between the read
                # and transition.  Cancellation remains idempotent in that race.
                current = self.database.require_run(run.id)
                if current.terminal:
                    return current
                raise
        self._wake.set()
        return run

    def request_recovery(self, run_id: str) -> Run:
        """Resume a terminal run without discarding its frozen evidence.

        This is intentionally explicit for deterministic checksum, protocol,
        and local-filesystem failures.  Retrying those forever would hide real
        corruption; leaving no recovery entry point would strand a valid remote
        result.  The immutable snapshot, task, server target, and remote marker
        identity are all reused.
        """

        with self._lock:
            run = self.database.require_run(run_id)
            mode = run_recovery_mode(run)
            if mode is None:
                raise AdmissionError(
                    "RECOVERY_NOT_AVAILABLE",
                    "This run has no safely recoverable remote workspace.",
                )
            thread = self._threads.get(run.id)
            if thread is not None and thread.is_alive():
                raise AdmissionError(
                    "RECOVERY_ALREADY_RUNNING",
                    "A recovery attempt is already active for this run.",
                )
            target_key = self._execution_target_key(run)
            if target_key in self._active_targets:
                raise AdmissionError(
                    "RECOVERY_TARGET_BUSY",
                    "The frozen remote execution target is busy; retry recovery after its active run finishes.",
                )

            previous_error = run.error_code
            if mode in {"transfer", "launch"}:
                details = dict(run.error_details)
                details.pop("phase", None)
                prior_recovery = details.get("recovery")
                recovery = (
                    dict(prior_recovery)
                    if isinstance(prior_recovery, Mapping)
                    else {}
                )
                recovery.update(
                    {
                        "attempts": int(recovery.get("attempts") or 0) + 1,
                        "last_requested_at": utc_now(),
                        "mode": mode,
                        "previous_error_code": previous_error,
                    }
                )
                details["recovery"] = recovery
                self.database.update_run_fields(
                    run.id,
                    finished_at=None,
                    error_code=None,
                    error_message=None,
                    error_details=details,
                    remote_cleanup_error=None,
                )
                target_status = (
                    RunStatus.RUNNING
                    if mode == "transfer"
                    else RunStatus.PREPARING
                )
                self.database.set_run_status(
                    run.id,
                    target_status,
                    validate_transition=False,
                    event_message=(
                        "User requested recovery of the remote result transfer."
                        if mode == "transfer"
                        else "User requested recovery of the idempotent remote launch identity."
                    ),
                )
                run = self.database.require_run(run.id)
            elif mode == "verify":
                self._event(
                    run.id,
                    "recovery",
                    "User requested a complete local re-verification of the already transferred successful result.",
                    EventLevel.WARNING,
                    data={"previous_error_code": previous_error},
                )
            else:
                self._event(
                    run.id,
                    "recovery",
                    "User requested an immediate retry of marker-owned remote cleanup.",
                    EventLevel.WARNING,
                    data={"previous_error_code": previous_error},
                )

            self._start_thread(run, resume=True)
            return self.database.require_run(run.id)

    def _recover_runs(self) -> None:
        for run in self.database.startup_recovery_runs():
            if run.status is RunStatus.QUEUED:
                continue
            if (
                run.remote_run_dir
                and run.transfer_verified
                and run.status in {
                    RunStatus.TRANSFERRING,
                    RunStatus.VERIFYING,
                    RunStatus.CLEANING,
                }
            ):
                self._start_thread(run, resume=True)
                continue
            if run.remote_run_dir and run.remote_pid and run.status in {
                RunStatus.PREPARING,
                RunStatus.RUNNING,
                RunStatus.CANCELING,
                RunStatus.TRANSFERRING,
                RunStatus.VERIFYING,
                RunStatus.CLEANING,
            }:
                self._start_thread(run, resume=True)
                continue
            if run.remote_run_dir and run.status in {
                RunStatus.SYNCING,
                RunStatus.PREPARING,
                RunStatus.CANCELING,
            }:
                # The remote launch script is idempotent and records
                # meta/process.json before returning its PID.  Re-entering the
                # pre-launch path closes the unavoidable crash/SSH-response
                # gap between starting the detached process and persisting its
                # PID locally.
                self._start_thread(run, resume=True)
                continue
            self.database.set_run_status(
                run.id,
                RunStatus.INTERRUPTED,
                error_code="CONTROLLER_INTERRUPTED",
                error_message=(
                    "The controller stopped before a recoverable remote process was recorded."
                ),
                validate_transition=False,
            )
        self._start_cleanup_retries()

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            self._reap_threads()
            if time.monotonic() >= self._next_cleanup_retry:
                self._next_cleanup_retry = time.monotonic() + 60.0
                self._start_cleanup_retries()
            for run in reversed(self.database.list_runs(statuses=[RunStatus.QUEUED], limit=1000)):
                if self._stop.is_set():
                    break
                server = self.database.get_server(run.server_id)
                if server is None or not server.enabled:
                    continue
                self._start_thread(run, resume=False)
            self._wake.wait(1.0)
            self._wake.clear()

    def _start_thread(self, run: Run, *, resume: bool) -> None:
        with self._lock:
            current = self.database.get_run(run.id)
            if current is None:
                return
            if current.terminal and not _needs_terminal_retry(current):
                return
            if not resume and current.status is not RunStatus.QUEUED:
                return
            run = current
            if run.id in self._threads and self._threads[run.id].is_alive():
                return
            target_key = self._execution_target_key(run)
            if target_key in self._active_targets:
                return
            self._active_targets.add(target_key)
            self._thread_targets[run.id] = target_key
            thread = threading.Thread(
                target=self._thread_main,
                args=(run.id, resume),
                name=f"edolab-relay-{run.id[:8]}",
                daemon=True,
            )
            self._threads[run.id] = thread
            thread.start()

    def _start_cleanup_retries(self) -> None:
        statuses = [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED]
        # Runs every minute over the whole ledger: summary records carry every field the retry checks read, and
        # _start_thread re-reads the full record before acting.
        for run in self.database.list_runs(statuses=statuses, limit=10_000, summary=True):
            # A successful-result re-verification may expose deterministic
            # provenance corruption.  Keep that action explicit instead of
            # retrying it every minute; ordinary marker-owned cleanup remains
            # automatic and idempotent.
            if _needs_terminal_retry(run) and run_recovery_mode(run) != "verify":
                self._start_thread(run, resume=True)

    def _reap_threads(self) -> None:
        with self._lock:
            finished = [run_id for run_id, thread in self._threads.items() if not thread.is_alive()]
            for run_id in finished:
                target_key = self._thread_targets.pop(run_id, None)
                if target_key is not None:
                    self._active_targets.discard(target_key)
                self._threads.pop(run_id, None)

    def _thread_main(self, run_id: str, resume: bool) -> None:
        try:
            if resume:
                self._resume_run(run_id)
            else:
                self._execute_run(run_id)
        except ControllerStopping:
            self._event(
                run_id,
                "controller",
                "Local controller stopped; the detached remote process is left running for recovery.",
                EventLevel.WARNING,
            )
        except BaseException as exc:
            self._fail_run(run_id, exc)
        finally:
            with self._lock:
                target_key = self._thread_targets.pop(run_id, None)
                if target_key is not None:
                    self._active_targets.discard(target_key)
                self._threads.pop(run_id, None)
            self._wake.set()

    def _execute_run(self, run_id: str) -> None:
        run = self.database.require_run(run_id)
        server = self._server_for_run(run)
        self._require_clean_git_if_configured()
        local_dir = self._working_dir(run.id)
        provenance_dir = local_dir / "provenance"
        controller_dir = local_dir / "controller"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        controller_dir.mkdir(parents=True, exist_ok=True)
        task_path = (
            self._resolve_task_path(run)
            if run.task_kind == "batch"
            else self._materialize_workload_input(run, controller_dir)
        )
        self.database.update_run_fields(
            run.id,
            local_artifact_dir=f"working/{run.id}",
            source_root=str(self.paths.repository_root),
        )
        workspace: RemoteWorkspace | None = None
        client: Any | None = None
        installation_id = self._installation_id()
        try:
            self._set_pre_remote_status(run.id, RunStatus.CHECKING)
            # Leaving the queue freezes the worker count (set_queued_run_parallelism only edits queued runs);
            # re-read so a change made while this thread was starting is the one the run uses.
            run = self.database.require_run(run.id)
            target_description = (
                "this machine"
                if server.target_kind is ExecutionTargetKind.LOCAL
                else f"{server.username}@{server.host}:{server.port}"
            )
            self._event(run.id, "preflight", f"Checking {target_description}.")
            client = self.backend.client(server, self.paths)
            probe = dict(self.backend.probe(client, server))
            (provenance_dir / "server-probe.json").write_text(
                stable_json_dumps(probe) + "\n", encoding="utf-8"
            )
            fresh_fingerprint = _optional_text(probe.get("fingerprint"))
            probe_record = {
                "ok": True,
                "checked_at": utc_now(),
                "report": probe,
            }
            fingerprint_mismatch = bool(
                server.fingerprint
                and fresh_fingerprint
                and fresh_fingerprint != server.fingerprint
            )
            try:
                self.database.record_server_probe(
                    server.id,
                    probe_record,
                    # Preserve the previously trusted profile fingerprint when a
                    # frozen run detects conflicting fresh evidence.
                    fingerprint=server.fingerprint
                    if fingerprint_mismatch
                    else fresh_fingerprint,
                    expected_target=server,
                )
            except ServerTargetChangedError:
                self._event(
                    run.id,
                    "server_profile_changed",
                    "The saved server profile changed after this run froze its "
                    "execution target; the fresh run probe remains attached only "
                    "to this run and did not overwrite the edited profile.",
                    EventLevel.WARNING,
                )
            self.database.add_resource_sample(_sample_from_probe(server.id, run.id, probe))
            current = self.database.require_run(run.id)
            self.database.update_run_fields(
                run.id,
                task_payload={
                    **dict(current.task_payload),
                    "execution_probe": probe,
                    "execution_fingerprint": fresh_fingerprint,
                },
            )
            if server.fingerprint:
                if not fresh_fingerprint:
                    raise AdmissionError(
                        "SSH_FINGERPRINT_UNAVAILABLE",
                        "The frozen server target has a pinned SSH host fingerprint, "
                        "but the fresh execution probe could not report one.",
                    )
                if fingerprint_mismatch:
                    raise AdmissionError(
                        "SSH_FINGERPRINT_MISMATCH",
                        "The fresh SSH host fingerprint differs from the fingerprint "
                        "frozen when this run was submitted.",
                        {
                            "expected_fingerprint": server.fingerprint,
                            "actual_fingerprint": fresh_fingerprint,
                        },
                    )
            inspection = (
                inspect_batch_task(task_path)
                if run.task_kind == "batch"
                else self._inspect_frozen_workload(run, task_path)
            )
            self._check_admission(server, run, probe, inspection.gpu_requested)

            if self._stop.is_set():
                raise ControllerStopping()
            self._set_pre_remote_status(run.id, RunStatus.SNAPSHOTTING)
            self._event(run.id, "snapshot", "Building an immutable snapshot of the current working tree.")
            if run.task_payload.get("source_snapshot_id"):
                from .snapshot import load_source_snapshot
                snapshot = load_source_snapshot(self.paths.snapshots, run.task_payload["source_snapshot_id"])
            else:
                snapshot = SnapshotBuilder(
                    self.paths.repository_root,
                    self.paths.snapshots,
                ).build()
            task_provenance = (
                build_task_provenance(
                    task_path,
                    forced_max_parallel_runs=run.max_parallel_runs,
                    inspection=inspection,
                )
                if run.task_kind == "batch"
                else {
                    "schema_version": 1,
                    "input_kind": "workload_spec",
                    "workload_kind": inspection.workload_kind,
                    "operation": inspection.operation,
                    "task_file_sha256": inspection.task_file_sha256,
                    "effective_execution": {
                        "max_parallel_runs": run.max_parallel_runs,
                    },
                }
            )
            shutil.copy2(snapshot.archive_path, provenance_dir / "source.tar.gz")
            shutil.copy2(snapshot.manifest_path, provenance_dir / "source-manifest.json")
            frozen_task_path = provenance_dir / "task.jsonl"
            if frozen_task_path.exists() or frozen_task_path.is_symlink():
                if (
                    frozen_task_path.is_symlink()
                    or frozen_task_path.resolve() != task_path.resolve()
                ):
                    raise AdmissionError(
                        "TASK_CAPTURE_PATH_CONFLICT",
                        "The immutable task capture path already exists unexpectedly.",
                    )
            else:
                with frozen_task_path.open("xb") as handle:
                    handle.write(inspection.task_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                frozen_task_path.chmod(0o600)
            if _sha256_file(frozen_task_path) != inspection.task_file_sha256:
                raise AdmissionError(
                    "TASK_CHANGED_DURING_SNAPSHOT",
                    "Task file changed while its immutable run input was being captured.",
                )
            (provenance_dir / "task-provenance.json").write_text(
                stable_json_dumps(task_provenance) + "\n", encoding="utf-8"
            )
            source = dict(snapshot.manifest.get("source") or {})
            command = (
                (
                    "python",
                    "-m",
                    "apps.remote.worker",
                    "--task-file",
                    "meta/task.jsonl",
                    "--output-dir",
                    "results/bundle",
                    "--max-parallel-runs",
                    str(run.max_parallel_runs),
                )
                if run.task_kind == "batch"
                else (
                    "python",
                    "-m",
                    (
                        "apps.remote.benchmark_worker"
                        if run.task_kind == "benchmark"
                        else "apps.remote.plot_worker"
                    ),
                    "--spec-file",
                    "meta/task.jsonl",
                    "--output-dir",
                    "results/bundle",
                )
            )
            current = self.database.require_run(run.id)
            self.database.update_run_fields(
                run.id,
                snapshot_id=snapshot.snapshot_id,
                snapshot_sha256=snapshot.manifest_sha256,
                snapshot_manifest=snapshot.manifest,
                git_commit=source.get("git_commit"),
                git_branch=source.get("git_branch"),
                git_dirty=bool(source.get("git_dirty")),
                git_diff_sha256=source.get("git_diff_sha256"),
                task_sha256=inspection.task_file_sha256,
                task_payload={
                    **dict(current.task_payload),
                    "inspection": inspection.to_dict(),
                    "provenance": task_provenance,
                },
                command_argv=command,
                progress_total=(
                    float(inspection.total_runs)
                    if run.task_kind == "batch"
                    else 3.0
                ),
                progress_unit=("runs" if run.task_kind == "batch" else "steps"),
            )

            self._set_pre_remote_status(run.id, RunStatus.SYNCING)
            self._set_sync_stage(
                run.id,
                code="sync_prepare_copy",
                ordinal=1,
                message="Preparing an isolated remote run copy and locating a reusable snapshot cache.",
            )
            workspace = self._retry_remote_operation(
                run.id,
                "prepare the marker-owned remote workspace",
                lambda: self.backend.prepare(
                    client,
                    server,
                    installation_id,
                    run.id,
                    snapshot.snapshot_id,
                ),
            )
            # Persist the marker-owned directory immediately.  A subsequent
            # sync/launch failure can then be audited and cleaned instead of
            # leaving an undiscoverable remote directory.
            self.database.update_run_fields(run.id, remote_run_dir=workspace.run_dir)
            self._raise_if_pre_remote_canceled(run.id)
            self._set_sync_stage(
                run.id,
                code="sync_diff",
                ordinal=2,
                message="Comparing the frozen local snapshot with the reusable remote baseline.",
                mode="cache" if workspace.cache_hit else None,
            )
            sync_plan, manifest_diff = _build_source_sync_plan(workspace, snapshot.manifest)
            transfer_paths = tuple(str(path) for path in sync_plan["transfer_paths"])
            payload_path: Path | None = None
            payload_details: dict[str, Any] = {
                "file_count": 0,
                "logical_size_bytes": 0,
                "archive_size_bytes": 0,
            }
            if transfer_paths:
                target_paths = tuple(
                    str(item["path"]) for item in snapshot.manifest.get("files") or []
                )
                if not workspace.base_snapshot_id and transfer_paths == target_paths:
                    payload_path = snapshot.archive_path
                    payload_details = {
                        "file_count": len(transfer_paths),
                        "logical_size_bytes": int(sync_plan["transfer_size_bytes"]),
                        "archive_size_bytes": snapshot.archive_size_bytes,
                        "archive_sha256": snapshot.archive_sha256,
                    }
                else:
                    payload_path = provenance_dir / "source-delta.tar.gz"
                    payload_details = write_snapshot_delta_archive(
                        snapshot.archive_path,
                        snapshot.manifest,
                        transfer_paths,
                        payload_path,
                    )
            sync_plan = {**sync_plan, "payload": payload_details}
            (provenance_dir / "source-diff.json").write_text(
                stable_json_dumps({**manifest_diff, "sync_plan": sync_plan}) + "\n",
                encoding="utf-8",
            )
            sync_plan_path = provenance_dir / "source-sync-plan.json"
            sync_plan_path.write_text(
                stable_json_dumps(sync_plan) + "\n",
                encoding="utf-8",
            )
            fsync_file(provenance_dir / "source-diff.json")
            fsync_file(sync_plan_path)
            if payload_path is not None and payload_path.parent == provenance_dir:
                fsync_file(payload_path)
            fsync_directory(provenance_dir)
            self._event(
                run.id,
                "sync",
                (
                    f"Remote comparison: {len(manifest_diff['added'])} added, "
                    f"{len(manifest_diff['changed'])} changed, "
                    f"{len(manifest_diff['removed'])} removed."
                ),
                data=manifest_diff,
            )

            def sync_stage_callback(stage: str, details: Mapping[str, Any]) -> None:
                self._record_sync_backend_stage(run.id, stage, details)

            try:
                sync_result = self._retry_remote_operation(
                    run.id,
                    "transfer only changed source files and verify the isolated remote copy",
                    lambda: self.backend.sync(
                        client,
                        workspace,
                        payload_path=payload_path,
                        manifest_path=snapshot.manifest_path,
                        task_path=frozen_task_path,
                        snapshot_id=snapshot.snapshot_id,
                        sync_plan=sync_plan,
                        stage_callback=sync_stage_callback,
                    ),
                )
            except Exception:
                self._cleanup_unlaunched_workspace(
                    run.id, client, workspace, installation_id
                )
                raise
            self._set_sync_stage(
                run.id,
                code="sync_verify",
                ordinal=4,
                state="completed",
                message="The independent remote source copy matches the frozen local snapshot.",
                mode=str(sync_result.get("strategy") or sync_plan["strategy"]),
                progress={
                    "current": int(sync_result.get("file_count") or sync_plan["target_file_count"]),
                    "total": int(sync_plan["target_file_count"]),
                    "unit": "files",
                },
                data=sync_result,
            )
            self._event(
                run.id,
                "sync",
                "The independent remote source copy matches the frozen local snapshot.",
                data=sync_result,
            )

            self._set_pre_remote_status(run.id, RunStatus.PREPARING)
            try:
                remote_pid = self._launch_and_record(
                    run.id,
                    server,
                    client,
                    workspace,
                    worker_id=snapshot.snapshot_id,
                    allow_prelaunch_cancel=True,
                )
            except (PreRemoteCanceled, ControllerStopping):
                raise
            except Exception:
                # The cleanup helper first proves that no marker-owned process
                # group is live.  If launch actually happened but its response
                # was malformed, cleanup safely refuses and the terminal record
                # remains explicitly recoverable through its launch intent.
                self._cleanup_unlaunched_workspace(
                    run.id, client, workspace, installation_id
                )
                raise
            self._event(run.id, "remote", f"Detached remote process group started with PID {remote_pid}.")
            self._monitor_and_collect(run.id, server, client, workspace, installation_id)
        except PreRemoteCanceled:
            if workspace is not None and client is not None:
                self._cleanup_unlaunched_workspace(
                    run.id, client, workspace, installation_id
                )
            self._finalize_pre_remote_cancellation(run.id)

    def _set_pre_remote_status(self, run_id: str, status: RunStatus) -> Run:
        with self._lock:
            self._raise_if_pre_remote_canceled(run_id)
            return self.database.set_run_status(run_id, status)

    def _set_sync_stage(
        self,
        run_id: str,
        *,
        code: str,
        ordinal: int,
        message: str,
        state: str = "running",
        mode: str | None = None,
        progress: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> Run:
        stage: dict[str, Any] = {
            "code": code,
            "phase": "sync",
            "state": state,
            "ordinal": int(ordinal),
            "count": 4,
            "message": message,
        }
        if mode:
            stage["mode"] = mode
        if progress is not None:
            stage["progress"] = dict(progress)
        if data is not None:
            stage["data"] = _compact_sync_details(data)
        return self.database.set_run_stage(
            run_id,
            stage,
            message=message,
            level=EventLevel.ERROR if state == "failed" else EventLevel.INFO,
        )

    def _record_sync_backend_stage(
        self,
        run_id: str,
        stage: str,
        details: Mapping[str, Any],
    ) -> None:
        if stage == "transfer":
            strategy = str(details.get("strategy") or "delta")
            file_count = int(details.get("transfer_file_count") or 0)
            payload_size = int(details.get("payload_size_bytes") or 0)
            if strategy == "cache":
                message = (
                    "Remote snapshot cache hit; copying it into this run's "
                    "independent source directory with zero source upload."
                )
            elif strategy == "full":
                message = (
                    f"No reusable remote baseline is available; uploading the "
                    f"initial {file_count}-file source payload."
                )
            else:
                message = (
                    f"Uploading only {file_count} added or content-changed files "
                    f"({payload_size} compressed bytes)."
                )
            self._set_sync_stage(
                run_id,
                code="sync_upload",
                ordinal=3,
                message=message,
                mode=strategy,
                progress={"current": 0, "total": payload_size, "unit": "bytes"},
                data=details,
            )
        elif stage == "verify":
            self._set_sync_stage(
                run_id,
                code="sync_verify",
                ordinal=4,
                message=(
                    "Materializing the task's independent source copy and "
                    "verifying it against the frozen local snapshot."
                ),
                mode=str(details.get("strategy") or "delta"),
                progress={
                    "current": 0,
                    "total": int(details.get("target_file_count") or 0),
                    "unit": "files",
                },
                data=details,
            )

    def _raise_if_pre_remote_canceled(self, run_id: str) -> None:
        current = self.database.require_run(run_id)
        if current.status is RunStatus.CANCELING or current.status is RunStatus.CANCELED:
            raise PreRemoteCanceled()
        if current.terminal:
            raise PreRemoteCanceled()

    def _cleanup_unlaunched_workspace(
        self,
        run_id: str,
        client: Any,
        workspace: RemoteWorkspace,
        installation_id: str,
    ) -> None:
        try:
            cleanup = self.backend.cleanup(
                client,
                workspace,
                installation_id=installation_id,
                run_id=run_id,
            )
            if not cleanup.get("deleted"):
                raise RuntimeError("Remote cleanup did not confirm deletion.")
            self.database.update_run_fields(
                run_id,
                remote_cleanup_completed=True,
                remote_cleanup_at=utc_now(),
                remote_cleanup_error=None,
            )
            self._event(
                run_id,
                "cleanup",
                "Removed marker-owned remote workspace before worker launch.",
            )
        except Exception as cleanup_exc:
            self.database.update_run_fields(
                run_id,
                remote_cleanup_completed=False,
                remote_cleanup_at=utc_now(),
                remote_cleanup_error=str(cleanup_exc),
            )
            self._event(
                run_id,
                "cleanup",
                "Could not remove the pre-launch remote workspace: "
                + str(cleanup_exc),
                EventLevel.WARNING,
            )

    def _finalize_pre_remote_cancellation(self, run_id: str) -> None:
        run = self.database.require_run(run_id)
        if run.terminal:
            return
        try:
            final_relative = self._promote_local_dir(run.id)
            self.database.update_run_fields(
                run.id,
                local_artifact_dir=final_relative,
                remote_finished_at=utc_now(),
            )
        except Exception:
            pass
        self.database.set_run_status(
            run.id,
            RunStatus.CANCELED,
            exit_code=130,
            error_code="CANCELED_BEFORE_REMOTE_START",
            error_message="Canceled before a remote worker process was launched.",
            validate_transition=False,
        )

    def _resume_run(self, run_id: str) -> None:
        run = self.database.require_run(run_id)
        server = self._server_for_run(run)
        client = self.backend.client(server, self.paths)
        if run.remote_run_dir is None:
            raise AdmissionError("RECOVERY_STATE_MISSING", "Remote recovery metadata is incomplete.")
        remote_root = str(Path(run.remote_run_dir).parent.parent)
        workspace = RemoteWorkspace(remote_root, run.remote_run_dir, None)
        if run_recovery_mode(run) == "verify":
            self._reverify_successful_transfer(
                run,
                client,
                workspace,
                self._installation_id(),
            )
            return
        if _needs_unlaunched_cleanup_retry(run):
            self._retry_unlaunched_cleanup_only(
                run,
                client,
                workspace,
                self._installation_id(),
            )
            return
        if _needs_cleanup_retry(run):
            self._retry_cleanup_only(
                run,
                client,
                workspace,
                self._installation_id(),
            )
            return
        if run.transfer_verified and run.status in {
            RunStatus.TRANSFERRING,
            RunStatus.VERIFYING,
            RunStatus.CLEANING,
        }:
            self._event(
                run.id,
                "recovery",
                "Local transfer was already verified; resuming checksum-gated remote cleanup.",
            )
            self._cleanup_and_finalize(
                run.id,
                client,
                workspace,
                self._installation_id(),
            )
            return
        if run.remote_pid is None and run.status is RunStatus.SYNCING:
            local_dir = self._existing_local_dir(run.id)
            provenance_dir = local_dir / "provenance"
            archive_path = provenance_dir / "source.tar.gz"
            manifest_path = provenance_dir / "source-manifest.json"
            task_path = provenance_dir / "task.jsonl"
            if not run.snapshot_id or not all(
                path.is_file() and not path.is_symlink()
                for path in (archive_path, manifest_path, task_path)
            ):
                raise AdmissionError(
                    "RECOVERY_STATE_MISSING",
                    "The immutable local snapshot needed to resume remote synchronization is incomplete.",
                )
            sync_plan_path = provenance_dir / "source-sync-plan.json"
            if sync_plan_path.exists() or sync_plan_path.is_symlink():
                if sync_plan_path.is_symlink() or not sync_plan_path.is_file():
                    raise AdmissionError(
                        "RECOVERY_STATE_MISSING",
                        "The frozen source synchronization plan is unsafe.",
                    )
                sync_plan = json.loads(sync_plan_path.read_text(encoding="utf-8"))
                if not isinstance(sync_plan, Mapping):
                    raise AdmissionError(
                        "RECOVERY_STATE_MISSING",
                        "The frozen source synchronization plan is invalid.",
                    )
                sync_plan = dict(sync_plan)
            else:
                # A controller may stop after the full immutable snapshot was
                # captured but before the remote baseline plan reached disk.
                # Fall back to a safe full payload from that frozen archive;
                # never rebuild from the mutable worktree during recovery.
                target_files = [
                    dict(item) for item in run.snapshot_manifest.get("files") or []
                ]
                sync_plan = {
                    "schema_version": 1,
                    "strategy": "full",
                    "base_snapshot_id": None,
                    "target_snapshot_id": run.snapshot_id,
                    "cache_hit": False,
                    "added": [str(item["path"]) for item in target_files],
                    "changed": [],
                    "metadata_only": [],
                    "removed": [],
                    "source_changed": True,
                    "transfer_paths": [str(item["path"]) for item in target_files],
                    "transfer_file_count": len(target_files),
                    "transfer_size_bytes": sum(
                        int(item["size_bytes"]) for item in target_files
                    ),
                    "reused_file_count": 0,
                    "target_file_count": len(target_files),
                    "payload": {
                        "file_count": len(target_files),
                        "logical_size_bytes": sum(
                            int(item["size_bytes"]) for item in target_files
                        ),
                        "archive_size_bytes": archive_path.stat().st_size,
                    },
                }
            if sync_plan.get("target_snapshot_id") != run.snapshot_id:
                raise AdmissionError(
                    "RECOVERY_STATE_MISSING",
                    "The frozen synchronization plan targets a different snapshot.",
                )
            transfer_paths = tuple(
                str(path) for path in sync_plan.get("transfer_paths") or ()
            )
            payload_path: Path | None = None
            if transfer_paths:
                if not sync_plan.get("base_snapshot_id"):
                    payload_path = archive_path
                else:
                    payload_path = provenance_dir / "source-delta.tar.gz"
                    if payload_path.is_symlink() or not payload_path.is_file():
                        raise AdmissionError(
                            "RECOVERY_STATE_MISSING",
                            "The frozen incremental source payload is missing.",
                        )
            workspace = RemoteWorkspace(
                remote_root,
                run.remote_run_dir,
                None,
                base_snapshot_id=(
                    str(sync_plan["base_snapshot_id"])
                    if sync_plan.get("base_snapshot_id")
                    else None
                ),
                cache_hit=bool(sync_plan.get("cache_hit", False)),
            )
            self._event(
                run.id,
                "recovery",
                "Resuming idempotent source and task synchronization after controller interruption.",
                EventLevel.WARNING,
            )
            self._retry_remote_operation(
                run.id,
                "resume incremental source synchronization from frozen local inputs",
                lambda: self.backend.sync(
                    client,
                    workspace,
                    payload_path=payload_path,
                    manifest_path=manifest_path,
                    task_path=task_path,
                    snapshot_id=run.snapshot_id,
                    sync_plan=sync_plan,
                    stage_callback=lambda stage, details: self._record_sync_backend_stage(
                        run.id, stage, details
                    ),
                ),
            )
            self._set_sync_stage(
                run.id,
                code="sync_verify",
                ordinal=4,
                state="completed",
                message="Recovered synchronization completed; the independent remote source copy is verified.",
                mode=str(sync_plan.get("strategy") or "delta"),
                progress={
                    "current": int(sync_plan.get("target_file_count") or 0),
                    "total": int(sync_plan.get("target_file_count") or 0),
                    "unit": "files",
                },
                data=sync_plan,
            )
            self.database.set_run_status(
                run.id,
                RunStatus.PREPARING,
                validate_transition=False,
            )
            run = self.database.require_run(run.id)
        if run.remote_pid is None and run.status in {
            RunStatus.PREPARING,
            RunStatus.CANCELING,
        }:
            self._event(
                run.id,
                "recovery",
                "Recovering the idempotent remote launch identity before monitoring or cancellation.",
                EventLevel.WARNING,
            )
            try:
                remote_pid = self.backend.recover_process(client, workspace)
            except Exception as recovery_exc:
                self._event(
                    run.id,
                    "recovery",
                    "A persisted remote process identity could not be read. Because "
                    "transport loss and a never-started worker are indistinguishable, "
                    "Relay is entering the idempotent launch path: " + str(recovery_exc),
                    EventLevel.WARNING,
                )
                self._launch_and_record(
                    run.id,
                    server,
                    client,
                    workspace,
                    worker_id=run.worker_id or run.snapshot_id,
                    allow_prelaunch_cancel=False,
                )
            else:
                self.database.set_run_remote_process(
                    run.id,
                    remote_pid=remote_pid,
                    remote_run_dir=workspace.run_dir,
                    worker_id=run.worker_id or run.snapshot_id,
                )
                self._event(
                    run.id,
                    "recovery",
                    f"Recovered remote process identity PID {remote_pid} without launching a second worker.",
                )
            run = self.database.require_run(run.id)
        if run.remote_pid is None:
            raise AdmissionError("RECOVERY_STATE_MISSING", "Remote process ID is missing.")
        self._event(run.id, "recovery", "Resuming monitoring of the detached remote process.")
        self._monitor_and_collect(run.id, server, client, workspace, self._installation_id())

    def _launch_and_record(
        self,
        run_id: str,
        server: Server,
        client: Any,
        workspace: RemoteWorkspace,
        *,
        worker_id: str | None,
        allow_prelaunch_cancel: bool,
    ) -> int:
        """Close the remote-launch/PID-persistence gap with an idempotent retry."""

        attempt = 0
        launch_intent_persisted = False
        while True:
            if self._stop.is_set():
                raise ControllerStopping()
            attempt += 1
            launch_acknowledged = False
            try:
                # Holding the lock across launch and local PID persistence
                # keeps cancellation atomic in-process.  The durable launch
                # intent and remote process.json cover process/transport
                # crashes that the local lock cannot cover.
                with self._lock:
                    current = self.database.require_run(run_id)
                    conda_executable = _execution_conda_executable(
                        current.task_payload
                    )
                    if conda_executable is None:
                        raise AdmissionError(
                            "CONDA_EXECUTABLE_MISSING",
                            "The frozen execution probe did not record an absolute "
                            "Conda executable path; launch cannot safely rediscover "
                            "a different installation.",
                        )
                    execution = dict(current.task_payload.get("execution") or {})
                    launch_intent_persisted = bool(execution.get("launch_intent_at"))
                    if not launch_intent_persisted:
                        if allow_prelaunch_cancel:
                            self._raise_if_pre_remote_canceled(run_id)
                        execution["launch_intent_at"] = utc_now()
                        execution["launch_state"] = "attempting"
                        self.database.update_run_fields(
                            run_id,
                            task_payload={
                                **dict(current.task_payload),
                                "execution": execution,
                            },
                        )
                        launch_intent_persisted = True
                    launch_arguments = {
                        "conda_env": server.conda_env,
                        "conda_executable": conda_executable,
                        "max_parallel_runs": current.max_parallel_runs,
                    }
                    if current.task_kind == "batch":
                        remote_pid = self.backend.launch(
                            client,
                            workspace,
                            **launch_arguments,
                        )
                    else:
                        remote_pid = self.backend.launch(
                            client,
                            workspace,
                            worker_kind=current.task_kind,
                            **launch_arguments,
                        )
                    launch_acknowledged = True
                    self.database.set_run_remote_process(
                        run_id,
                        remote_pid=remote_pid,
                        remote_run_dir=workspace.run_dir,
                        worker_id=worker_id,
                    )
                    persisted = self.database.require_run(run_id)
                    execution = dict(persisted.task_payload.get("execution") or {})
                    execution.update(
                        {
                            "launch_state": "recorded",
                            "remote_pid": remote_pid,
                            "conda_executable": conda_executable,
                            "recorded_at": utc_now(),
                        }
                    )
                    self.database.update_run_fields(
                        run_id,
                        task_payload={
                            **dict(persisted.task_payload),
                            "execution": execution,
                        },
                    )
                return remote_pid
            except PreRemoteCanceled:
                raise
            except ControllerStopping:
                raise
            except Exception as exc:
                if not launch_acknowledged and not _is_transport_failure(exc):
                    raise
                self._event(
                    run_id,
                    "remote_launch",
                    "Remote launch acknowledgement was not safely recorded; "
                    f"retrying the idempotent launch identity (attempt {attempt}): {exc}",
                    EventLevel.WARNING,
                    data={
                        "attempt": attempt,
                        "launch_intent_persisted": launch_intent_persisted,
                    },
                )
                current = self.database.require_run(run_id)
                self.database.update_run_progress(
                    run_id,
                    current=current.progress_current,
                    total=current.progress_total,
                    unit=current.progress_unit or "runs",
                    message="Remote launch state is uncertain; reconnecting without starting a second worker.",
                )
                if self._stop.wait(max(0.1, current.poll_interval_seconds)):
                    raise ControllerStopping()

    def _retry_remote_operation(
        self,
        run_id: str,
        description: str,
        operation: Any,
    ) -> Any:
        """Retry transport failures without terminalizing a recoverable run."""

        attempt = 0
        while True:
            if self._stop.is_set():
                raise ControllerStopping()
            attempt += 1
            try:
                return operation()
            except Exception as exc:
                if not _is_transport_failure(exc):
                    raise
                if attempt in {1, 3, 12} or attempt % 12 == 0:
                    self._event(
                        run_id,
                        "transport",
                        f"Could not {description}; local state remains recoverable "
                        f"and Relay will retry (attempt {attempt}): {exc}",
                        EventLevel.WARNING,
                    )
                current = self.database.require_run(run_id)
                self.database.update_run_progress(
                    run_id,
                    current=current.progress_current,
                    total=current.progress_total,
                    unit=current.progress_unit or "runs",
                    message=f"Transport unavailable while trying to {description}; retrying.",
                )
                if self._stop.wait(max(0.1, current.poll_interval_seconds)):
                    raise ControllerStopping()

    def _monitor_and_collect(
        self,
        run_id: str,
        server: Server,
        client: Any,
        workspace: RemoteWorkspace,
        installation_id: str,
    ) -> None:
        run = self.database.require_run(run_id)
        if run.remote_pid is None:
            raise AdmissionError("REMOTE_PID_MISSING", "Remote process ID is missing.")
        local_dir = self._existing_local_dir(run.id)
        controller_dir = local_dir / "controller"
        controller_dir.mkdir(parents=True, exist_ok=True)
        state = _monitor_state_from_logs(controller_dir)
        self._reconcile_protocol_log(run.id, controller_dir / "stdout.log")
        while True:
            if self._stop.is_set():
                raise ControllerStopping()
            current = self.database.require_run(run.id)
            if current.status is RunStatus.CANCELING:
                now = time.monotonic()
                if state.cancel_sent_at is None:
                    try:
                        cancellation = self.backend.cancel(
                            client,
                            workspace,
                            remote_pid=run.remote_pid,
                            force=False,
                        )
                    except Exception as exc:
                        self._event(
                            run.id,
                            "cancel",
                            "TERM was not sent because remote process identity could not be confirmed: "
                            + str(exc),
                            EventLevel.WARNING,
                        )
                    else:
                        state.cancel_attempted_at = now
                        if bool(cancellation.get("sent")):
                            state.cancel_sent_at = now
                            self._event(
                                run.id,
                                "cancel",
                                "Sent TERM to the remote process group.",
                                EventLevel.WARNING,
                            )
                        else:
                            self._event(
                                run.id,
                                "cancel",
                                "TERM helper completed but did not signal a matching live process group; Relay will verify remote state before retrying.",
                                EventLevel.WARNING,
                            )
                elif not state.force_sent and now - state.cancel_sent_at >= 10:
                    try:
                        cancellation = self.backend.cancel(
                            client,
                            workspace,
                            remote_pid=run.remote_pid,
                            force=True,
                        )
                    except Exception as exc:
                        self._event(
                            run.id,
                            "cancel",
                            "KILL was not sent because remote process identity could not be confirmed: "
                            + str(exc),
                            EventLevel.WARNING,
                        )
                    else:
                        state.force_attempted_at = now
                        if bool(cancellation.get("sent")):
                            state.force_sent = True
                            self._event(
                                run.id,
                                "cancel",
                                "Sent KILL after the cancellation grace period.",
                                EventLevel.WARNING,
                            )
                        else:
                            self._event(
                                run.id,
                                "cancel",
                                "KILL helper completed but did not signal a matching live process group; Relay will continue remote-state reconciliation.",
                                EventLevel.WARNING,
                            )
            try:
                poll = self.backend.poll(
                    client,
                    workspace,
                    remote_pid=run.remote_pid,
                    stdout_offset=state.stdout_offset,
                    stderr_offset=state.stderr_offset,
                )
                state.consecutive_errors = 0
            except Exception as exc:
                if not _is_transport_failure(exc):
                    raise
                state.consecutive_errors += 1
                if state.consecutive_errors in {1, 3, 12} or state.consecutive_errors % 12 == 0:
                    self._event(
                        run.id,
                        "monitor",
                        "Remote polling is unavailable; the detached job remains "
                        f"recoverable and will not be marked failed (attempt {state.consecutive_errors}): {exc}",
                        EventLevel.WARNING,
                    )
                current = self.database.require_run(run.id)
                self.database.update_run_progress(
                    run.id,
                    current=current.progress_current,
                    total=current.progress_total,
                    unit=current.progress_unit or "runs",
                    message="SSH monitoring unavailable; retrying while the remote job remains detached.",
                )
                if state.consecutive_errors % 12 == 0:
                    try:
                        client = self.backend.client(server, self.paths)
                    except Exception as reconnect_exc:
                        self._event(
                            run.id,
                            "monitor",
                            "Could not refresh the SSH transport: " + str(reconnect_exc),
                            EventLevel.WARNING,
                        )
                if self._stop.wait(run.poll_interval_seconds):
                    raise ControllerStopping()
                continue
            self._consume_poll(run.id, server, poll, state, controller_dir)
            observed = self.database.require_run(run.id)
            if poll.alive and observed.status in {
                RunStatus.PREPARING,
                RunStatus.TRANSFERRING,
                RunStatus.VERIFYING,
                RunStatus.CLEANING,
            }:
                self.database.set_run_status(
                    run.id,
                    RunStatus.RUNNING,
                    validate_transition=False,
                )
            if not poll.alive and not poll.stdout_more and not poll.stderr_more:
                if poll.exit_code is not None:
                    break
                state.dead_without_exit_count += 1
                current = self.database.require_run(run.id)
                if (
                    current.status is RunStatus.CANCELING
                    or not poll.process_identity_valid
                    or state.dead_without_exit_count >= 3
                ):
                    inferred = (
                        -9
                        if current.status is RunStatus.CANCELING and state.force_sent
                        else 143
                        if current.status is RunStatus.CANCELING
                        else 255
                    )
                    code = (
                        "REMOTE_CANCELED_NO_EXIT_STATUS"
                        if current.status is RunStatus.CANCELING
                        else "REMOTE_EXIT_STATUS_MISSING"
                    )
                    self.database.update_run_fields(
                        run.id,
                        exit_code=inferred,
                        error_code=code,
                        error_message=(
                            "Remote process ended without writing meta/exit_code; "
                            "the controller inferred a terminal status after logs drained."
                        ),
                    )
                    self._event(
                        run.id,
                        "monitor",
                        "Remote process is gone and logs are drained; continuing artifact recovery without an exit-code file.",
                        EventLevel.WARNING,
                        data={"inferred_exit_code": inferred, "identity_valid": poll.process_identity_valid},
                    )
                    break
            else:
                state.dead_without_exit_count = 0
            if self._stop.wait(run.poll_interval_seconds):
                raise ControllerStopping()
        self._collect(run.id, server, client, workspace, installation_id, state)

    def _collect(
        self,
        run_id: str,
        server: Server,
        client: Any,
        workspace: RemoteWorkspace,
        installation_id: str,
        state: _MonitorState,
    ) -> None:
        run = self.database.require_run(run_id)
        was_canceling = run.status is RunStatus.CANCELING
        self.database.set_run_status(run.id, RunStatus.TRANSFERRING, validate_transition=not was_canceling)
        self.database.update_run_fields(run.id, transfer_started_at=utc_now())
        def build_artifact() -> RemoteArtifact:
            if run.task_kind == "batch":
                # Preserve compatibility with injected test/third-party batch
                # backends implementing the original two-argument protocol.
                return self.backend.artifact(client, workspace)
            return self.backend.artifact(
                client,
                workspace,
                worker_kind=run.task_kind,
            )

        remote_artifact = self._retry_remote_operation(
            run.id,
            "build the execution result artifact",
            build_artifact,
        )
        local_dir = self._existing_local_dir(run.id)
        controller_dir = local_dir / "controller"
        archive_path = controller_dir / "remote-artifact.tar.gz"

        def download_and_check() -> Any:
            transfer = self.backend.download(
                client,
                remote_artifact.remote_path,
                archive_path,
                expected_size=remote_artifact.size_bytes,
            )
            if (
                int(getattr(transfer, "size_bytes", remote_artifact.size_bytes))
                != remote_artifact.size_bytes
            ):
                raise SSHError("Downloaded artifact size differs from remote manifest.")
            if (
                str(getattr(transfer, "sha256", remote_artifact.sha256))
                != remote_artifact.sha256
            ):
                raise SSHError("Downloaded artifact hash differs from remote manifest.")
            return transfer

        self._retry_remote_operation(
            run.id,
            "download and checksum the complete result artifact",
            download_and_check,
        )
        transfer_fields: dict[str, Any] = {
            "transfer_completed_at": utc_now(),
            "transfer_manifest": remote_artifact.manifest,
        }
        if remote_artifact.exit_code is not None:
            transfer_fields["exit_code"] = remote_artifact.exit_code
        self.database.update_run_fields(run.id, **transfer_fields)
        self.database.set_run_status(run.id, RunStatus.VERIFYING, validate_transition=False)
        extracted = local_dir / "remote"
        if extracted.exists():
            if extracted.is_symlink() or not extracted.is_dir():
                raise RuntimeError("Unsafe partial extraction path in local working directory.")
            # transfer_verified is still false here.  A previous controller may
            # have crashed midway through extraction; the verified remote tar
            # remains authoritative and can be extracted again from scratch.
            shutil.rmtree(extracted)
        verified_files = extract_verified_artifact(
            archive_path,
            extracted,
            remote_artifact.manifest,
        )
        bundle_dir = extracted / "results" / "bundle"
        if remote_artifact.exit_code == 0 and run.task_kind == "batch":
            expected_runs = int((run.task_payload.get("inspection") or {}).get("total_runs") or run.progress_total or 0)
            validate_result_bundle(
                bundle_dir,
                require_completed=True,
                expected_runs=expected_runs or None,
            )
        elif remote_artifact.exit_code == 0:
            self._verify_benchmark_bundle(run, bundle_dir)
        elif run.task_kind == "batch" and (bundle_dir / "summary.sqlite").is_file() and (bundle_dir / "trend.sqlite").is_file():
            try:
                validate_result_bundle(bundle_dir, require_completed=False)
            except Exception as exc:
                self._event(
                    run.id,
                    "verify",
                    f"Partial failed-run bundle is not complete: {exc}",
                    EventLevel.WARNING,
                )
        for item in verified_files:
            relative = f"remote/{item['path']}"
            self.database.upsert_artifact(
                Artifact(
                    run_id=run.id,
                    relative_path=relative,
                    status=ArtifactStatus.VERIFIED,
                    local_path=f"runs/{run.id}/{relative}",
                    size_bytes=int(item["size_bytes"]),
                    sha256=str(item["sha256"]),
                    transferred_at=utc_now(),
                    verified_at=utc_now(),
                )
            )
        for relative in (
            "provenance/source.tar.gz",
            "provenance/source-manifest.json",
            "provenance/task.jsonl",
            "provenance/task-provenance.json",
            "provenance/server-probe.json",
            "provenance/source-diff.json",
            "controller/remote-artifact.tar.gz",
        ):
            path = local_dir / relative
            if path.is_file():
                self.database.upsert_artifact(
                    Artifact(
                        run_id=run.id,
                        relative_path=relative,
                        status=ArtifactStatus.VERIFIED,
                        local_path=f"runs/{run.id}/{relative}",
                        size_bytes=path.stat().st_size,
                        sha256=_sha256_file(path),
                        transferred_at=utc_now(),
                        verified_at=utc_now(),
                    )
                )
        self.database.update_run_fields(
            run.id,
            transfer_verified=True,
            transfer_verified_at=utc_now(),
            # Preserve an inferred cancellation/termination code when the
            # remote EXIT trap was killed before writing its own status file.
            exit_code=(
                remote_artifact.exit_code
                if remote_artifact.exit_code is not None
                else self.database.require_run(run.id).exit_code
            ),
        )
        self._cleanup_and_finalize(
            run.id,
            client,
            workspace,
            installation_id,
            was_canceling=was_canceling,
            bundle_validation=remote_artifact.bundle_validation,
        )

    def _verify_local_transfer(self, run: Run) -> Path:
        if not run.transfer_verified:
            raise RuntimeError("Local transfer is not marked verified; remote cleanup is forbidden.")
        local_dir = self._existing_local_dir(run.id)
        archive_path = local_dir / "controller" / "remote-artifact.tar.gz"
        if archive_path.is_symlink() or not archive_path.is_file():
            raise RuntimeError("Downloaded remote artifact archive is missing.")
        provenance_dir = local_dir / "provenance"
        if provenance_dir.is_symlink() or not provenance_dir.is_dir():
            raise RuntimeError("Local run provenance directory is missing or unsafe.")
        provenance_paths: list[Path] = []
        for filename in _DURABLE_PROVENANCE_FILENAMES:
            path = provenance_dir / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"Required local provenance file is missing or unsafe: {filename}"
                )
            provenance_paths.append(path)
        manifest_files = list((run.transfer_manifest or {}).get("files") or [])
        if not manifest_files:
            raise RuntimeError("Verified transfer manifest contains no files.")
        verified_paths: list[Path] = []
        for item in manifest_files:
            relative = PurePosixPath(str(item.get("path") or ""))
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise RuntimeError("Unsafe path in persisted transfer manifest.")
            path = local_dir / "remote" / Path(*relative.parts)
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Verified local artifact is missing: {relative.as_posix()}")
            size = path.stat().st_size
            digest = _sha256_file(path)
            if size != int(item.get("size_bytes", -1)) or digest != str(item.get("sha256")):
                raise RuntimeError(f"Verified local artifact checksum changed: {relative.as_posix()}")
            verified_paths.append(path)
            stored_relative = f"remote/{relative.as_posix()}"
            self.database.upsert_artifact(
                Artifact(
                    run_id=run.id,
                    relative_path=stored_relative,
                    status=ArtifactStatus.VERIFIED,
                    local_path=f"runs/{run.id}/{stored_relative}",
                    size_bytes=size,
                    sha256=digest,
                    transferred_at=run.transfer_completed_at or utc_now(),
                    verified_at=run.transfer_verified_at or utc_now(),
                )
            )

        if run.task_sha256:
            for task_copy in (
                local_dir / "provenance" / "task.jsonl",
                local_dir / "remote" / "meta" / "task.jsonl",
            ):
                if task_copy.is_symlink() or not task_copy.is_file():
                    raise RuntimeError(f"Frozen task copy is missing: {task_copy}")
                if _sha256_file(task_copy) != run.task_sha256:
                    raise RuntimeError(f"Frozen task checksum mismatch: {task_copy}")

        if run.exit_code == 0 and run.task_kind == "batch":
            local_provenance_path = local_dir / "provenance" / "task-provenance.json"
            remote_provenance_path = (
                local_dir
                / "remote"
                / "results"
                / "bundle"
                / "task-provenance.json"
            )
            if not local_provenance_path.is_file() or not remote_provenance_path.is_file():
                raise RuntimeError("Successful result bundle is missing task provenance.")
            local_provenance = json.loads(
                local_provenance_path.read_text(encoding="utf-8")
            )
            remote_provenance = json.loads(
                remote_provenance_path.read_text(encoding="utf-8")
            )
            if local_provenance.get("task_file_sha256") != run.task_sha256:
                raise RuntimeError("Local task provenance does not match the frozen task.")
            if remote_provenance.get("task_file_sha256") != run.task_sha256:
                raise RuntimeError("Remote task provenance does not match the frozen task.")
            if task_provenance_identity(remote_provenance) != task_provenance_identity(
                local_provenance
            ):
                raise RuntimeError("Local and remote task provenance identities differ.")
            expected_runs = int(
                (run.task_payload.get("inspection") or {}).get("total_runs")
                or run.progress_total
                or 0
            )
            validate_result_bundle(
                local_dir / "remote" / "results" / "bundle",
                require_completed=True,
                expected_runs=expected_runs or None,
            )
        elif run.exit_code == 0:
            self._verify_benchmark_bundle(
                run,
                local_dir / "remote" / "results" / "bundle",
            )
        # This is the durability gate immediately before remote deletion.  It
        # deliberately runs after the complete checksum/result-bundle pass;
        # any failure propagates and leaves the remote workspace untouched.
        fsync_file(archive_path)
        for path in verified_paths:
            fsync_file(path)
        for path in provenance_paths:
            fsync_file(path)
        fsync_directory_tree(local_dir / "remote", include_parent=False)
        fsync_directory(local_dir / "controller")
        fsync_directory(provenance_dir)
        fsync_directory(local_dir)
        fsync_directory(local_dir.parent)
        return local_dir

    @staticmethod
    def _verify_benchmark_bundle(run: Run, bundle_dir: Path) -> dict[str, Any]:
        """Verify a successful non-batch workload independently of its worker."""

        if run.task_kind not in {"benchmark", "plot"}:
            raise RuntimeError(f"Unsupported successful workload kind: {run.task_kind!r}")
        result_path = bundle_dir / "workload-result.json"
        manifest_path = bundle_dir / "artifact-manifest.json"
        worker_status_path = bundle_dir / "worker-status.json"
        for path in (result_path, manifest_path, worker_status_path):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Workload result file is missing or unsafe: {path.name}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        scientific_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        worker_status = json.loads(worker_status_path.read_text(encoding="utf-8"))
        if not isinstance(result, Mapping) or result.get("schema_version") != 1:
            raise RuntimeError("Workload result has an unsupported schema.")
        if result.get("status") != "completed" or result.get("workload_kind") != run.task_kind:
            raise RuntimeError("Workload result did not record successful completion.")
        workload = run.task_payload.get("workload")
        expected_operation = (
            workload.get("operation") if isinstance(workload, Mapping) else None
        )
        if result.get("operation") != expected_operation:
            raise RuntimeError("Workload result operation differs from the frozen workload.")
        if not run.task_sha256 or result.get("spec_sha256") != run.task_sha256:
            raise RuntimeError("Workload result does not match the frozen workload SHA-256.")
        if worker_status.get("event") != "worker_completed" or worker_status.get("spec_sha256") != run.task_sha256:
            raise RuntimeError("Worker completion identity is missing or inconsistent.")
        declared = result.get("artifacts")
        if not isinstance(declared, list) or not declared:
            raise RuntimeError("Workload result declares no scientific artifacts.")
        if scientific_manifest.get("files") != declared or scientific_manifest.get("spec_sha256") != run.task_sha256:
            raise RuntimeError("Scientific artifact manifest differs from the workload result.")
        for item in declared:
            if not isinstance(item, Mapping):
                raise RuntimeError("Scientific artifact descriptor is invalid.")
            relative = PurePosixPath(str(item.get("path") or ""))
            if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise RuntimeError("Scientific artifact path is unsafe.")
            path = bundle_dir.joinpath(*relative.parts)
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle_dir.resolve()):
                raise RuntimeError(f"Scientific artifact is missing or unsafe: {relative.as_posix()}")
            if path.stat().st_size != int(item.get("size_bytes", -1)) or _sha256_file(path) != str(item.get("sha256") or ""):
                raise RuntimeError(f"Scientific artifact checksum mismatch: {relative.as_posix()}")
        return dict(result)

    def _cleanup_and_finalize(
        self,
        run_id: str,
        client: Any,
        workspace: RemoteWorkspace,
        installation_id: str,
        *,
        was_canceling: bool = False,
        bundle_validation: Mapping[str, Any] | None = None,
        preserve_terminal: bool = False,
    ) -> None:
        run = self.database.require_run(run_id)
        # This complete, repeated checksum pass is deliberately immediately
        # before the only destructive remote operation.
        self._verify_local_transfer(run)
        if not run.terminal and run.status is not RunStatus.CLEANING:
            self.database.set_run_status(
                run.id, RunStatus.CLEANING, validate_transition=False
            )

        cleanup_error: str | None = None
        run = self.database.require_run(run.id)
        if run.cleanup_requested and not run.remote_cleanup_completed:
            try:
                cleanup = self.backend.cleanup(
                    client,
                    workspace,
                    installation_id=installation_id,
                    run_id=run.id,
                )
                if not cleanup.get("deleted"):
                    raise RuntimeError("Remote cleanup did not confirm deletion.")
                self.database.update_run_fields(
                    run.id,
                    remote_cleanup_completed=True,
                    remote_cleanup_at=utc_now(),
                    remote_cleanup_error=None,
                )
                self._event(
                    run.id,
                    "cleanup",
                    "Verified remote run directory deletion.",
                    data={"already_absent": bool(cleanup.get("already_absent"))},
                )
            except Exception as exc:
                cleanup_error = str(exc)
                self.database.update_run_fields(
                    run.id,
                    remote_cleanup_completed=False,
                    remote_cleanup_at=utc_now(),
                    remote_cleanup_error=cleanup_error,
                )
                self._event(
                    run.id,
                    "cleanup",
                    "Results are safe locally, but remote cleanup is pending: "
                    + cleanup_error,
                    EventLevel.WARNING,
                )

        final_relative = self._promote_local_dir(run.id)
        self.database.update_run_fields(
            run.id,
            local_artifact_dir=final_relative,
            remote_finished_at=run.remote_finished_at or utc_now(),
        )
        if preserve_terminal or run.terminal:
            return

        current = self.database.require_run(run.id)
        exit_code = current.exit_code
        remote_cancel_requested = bool(
            (current.transfer_manifest or {}).get("cancel_requested")
        )
        canceled = (
            was_canceling
            or remote_cancel_requested
            or current.error_code == "CANCEL_REQUESTED"
            or exit_code in {130, 137, 143, -15, -9}
        )
        if canceled:
            self.database.set_run_status(
                run.id,
                RunStatus.CANCELED,
                exit_code=exit_code if exit_code is not None else 130,
                error_code="REMOTE_CANCELED",
                error_message="Remote execution was canceled.",
                validate_transition=False,
            )
        elif exit_code == 0:
            message = "Run completed; all artifacts were verified locally."
            if cleanup_error:
                message += " Remote cleanup remains pending and will be retried."
            self.database.set_run_status(
                run.id,
                RunStatus.COMPLETED,
                exit_code=0,
                event_message=message,
                validate_transition=False,
            )
        else:
            if current.error_code and current.error_code != "CANCEL_REQUESTED":
                error_code = current.error_code
                error_message = (
                    current.error_message
                    or f"Remote worker exited with code {exit_code}."
                )
            else:
                error_code, error_message = _remote_error_from_log(
                    self._existing_local_dir(run.id) / "controller" / "stderr.log",
                    exit_code,
                )
            self.database.set_run_status(
                run.id,
                RunStatus.FAILED,
                exit_code=exit_code,
                error_code=error_code,
                error_message=error_message,
                error_details={
                    **dict(current.error_details),
                    "bundle_validation": dict(bundle_validation or {}),
                    "remote_phase": "artifact_collected",
                },
                validate_transition=False,
            )

    def _retry_cleanup_only(
        self,
        run: Run,
        client: Any,
        workspace: RemoteWorkspace,
        installation_id: str,
    ) -> None:
        try:
            self._cleanup_and_finalize(
                run.id,
                client,
                workspace,
                installation_id,
                preserve_terminal=run.terminal,
            )
        except Exception as exc:
            self.database.update_run_fields(
                run.id,
                remote_cleanup_completed=False,
                remote_cleanup_at=utc_now(),
                remote_cleanup_error=str(exc),
            )
            self._event(
                run.id,
                "cleanup",
                "Cleanup retry was blocked before deletion: " + str(exc),
                EventLevel.WARNING,
            )

    def _reverify_successful_transfer(
        self,
        run: Run,
        client: Any,
        workspace: RemoteWorkspace,
        installation_id: str,
    ) -> None:
        """Repeat local gates for an exit-zero transfer, never the remote task."""

        if run_recovery_mode(run) != "verify":
            raise RuntimeError(
                "Successful-result re-verification is not safe for this run state."
            )
        previous_error = run.error_code
        details = dict(run.error_details)
        details.pop("phase", None)
        prior_recovery = details.get("recovery")
        recovery = (
            dict(prior_recovery) if isinstance(prior_recovery, Mapping) else {}
        )
        recovery.update(
            {
                "attempts": int(recovery.get("attempts") or 0) + 1,
                "last_requested_at": utc_now(),
                "mode": "verify",
                "previous_error_code": previous_error,
            }
        )
        details["recovery"] = recovery
        self.database.update_run_fields(
            run.id,
            finished_at=None,
            error_code=None,
            error_message=None,
            error_details=details,
            remote_cleanup_error=None,
        )
        self.database.set_run_status(
            run.id,
            RunStatus.VERIFYING,
            validate_transition=False,
            event_message=(
                "Re-verifying the already transferred exit-zero result; the "
                "remote scientific task will not be launched again."
            ),
        )
        # Do not use _retry_cleanup_only here: it deliberately swallows an
        # exception to preserve an existing terminal status.  A failed
        # re-verification must propagate through _thread_main so _fail_run
        # restores FAILED and leaves the remote workspace untouched.
        self._cleanup_and_finalize(
            run.id,
            client,
            workspace,
            installation_id,
        )

    def _retry_unlaunched_cleanup_only(
        self,
        run: Run,
        client: Any,
        workspace: RemoteWorkspace,
        installation_id: str,
    ) -> None:
        """Retry deletion only when durable state proves launch never began."""

        if not _needs_unlaunched_cleanup_retry(run):
            raise RuntimeError(
                "Pre-launch cleanup retry is not safe for this run state."
            )
        self._cleanup_unlaunched_workspace(
            run.id,
            client,
            workspace,
            installation_id,
        )

    def _consume_poll(
        self,
        run_id: str,
        server: Server,
        poll: RemotePoll,
        state: _MonitorState,
        controller_dir: Path,
    ) -> None:
        state.stdout_offset = poll.stdout_offset
        state.stderr_offset = poll.stderr_offset
        if poll.stdout:
            _append_bytes(controller_dir / "stdout.log", poll.stdout)
            self._event(run_id, "stdout", poll.stdout.decode("utf-8", errors="replace"), data={"stream": "stdout"})
            combined = state.stdout_pending + poll.stdout
            lines = combined.split(b"\n")
            state.stdout_pending = lines.pop() if lines else b""
            for raw in lines:
                self._consume_protocol_line(run_id, raw.decode("utf-8", errors="replace"))
        if poll.stderr:
            _append_bytes(controller_dir / "stderr.log", poll.stderr)
            self._event(
                run_id,
                "stderr",
                poll.stderr.decode("utf-8", errors="replace"),
                EventLevel.ERROR,
                data={"stream": "stderr"},
            )
        self.database.add_resource_sample(_sample_from_poll(server.id, run_id, poll.resource))
        current_run = self.database.require_run(run_id)
        self.database.update_run_progress(
            run_id,
            current=current_run.progress_current,
            total=current_run.progress_total,
            unit=current_run.progress_unit or "runs",
            message=current_run.progress_message or f"Execution phase: {poll.phase}",
        )

    def _reconcile_protocol_log(self, run_id: str, path: Path) -> None:
        """Idempotently restore progress/error state from durable worker lines.

        A controller can crash after appending a remote stdout chunk but before
        committing the corresponding SQLite updates.  The byte cursor must
        still resume from the durable log length, so startup replays only the
        narrow, monotonic protocol state without duplicating audit events.
        """

        if not path.exists():
            return
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Unsafe local controller stdout log.")
        run = self.database.require_run(run_id)
        completed = float(run.progress_current)
        total = float(run.progress_total) if run.progress_total is not None else None
        message = run.progress_message
        worker_failure: Mapping[str, Any] | None = None
        prefix = "EDOLAB_EVENT "
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if not line.startswith(prefix):
                    continue
                try:
                    payload = json.loads(line[len(prefix) :])
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, Mapping):
                    continue
                event = str(payload.get("event") or "worker")
                if event == "worker_started":
                    candidate = _optional_float(payload.get("total_runs"))
                    if candidate is not None and candidate > 0:
                        total = max(total or 0.0, candidate)
                elif event == "run_completed":
                    candidate = _optional_float(payload.get("completed_runs"))
                    candidate_total = _optional_float(payload.get("total_runs"))
                    if candidate is not None and candidate >= completed:
                        completed = candidate
                        message = (
                            f"{payload.get('algorithm')} run "
                            f"{payload.get('run_index')} completed"
                        )
                    if candidate_total is not None and candidate_total > 0:
                        total = max(total or 0.0, candidate_total)
                elif event == "progress":
                    candidate = _optional_float(payload.get("current"))
                    candidate_total = _optional_float(payload.get("total"))
                    if candidate is not None and candidate >= completed:
                        completed = candidate
                    if candidate_total is not None and candidate_total > 0:
                        total = max(total or 0.0, candidate_total)
                    message = _optional_text(payload.get("message")) or message
                elif event == "worker_completed":
                    candidate_total = _optional_float(payload.get("total")) or total
                    if candidate_total is not None:
                        total = candidate_total
                        completed = candidate_total
                    message = _optional_text(payload.get("message")) or "Workload completed."
                elif event == "worker_failed":
                    worker_failure = payload
        self.database.update_run_progress(
            run_id,
            current=completed,
            total=total,
            unit=run.progress_unit or "runs",
            message=message,
        )
        if worker_failure is not None:
            current = self.database.require_run(run_id)
            self.database.update_run_fields(
                run_id,
                error_code=(
                    _optional_text(worker_failure.get("error_code"))
                    or current.error_code
                    or "worker_failed"
                ),
                error_message=(
                    _optional_text(worker_failure.get("message"))
                    or current.error_message
                    or "Remote worker failed."
                ),
                error_details={
                    **dict(current.error_details),
                    "worker": dict(worker_failure.get("details") or {}),
                    "worker_exit_code": _optional_int(
                        worker_failure.get("exit_code")
                    ),
                },
            )

    def _consume_protocol_line(self, run_id: str, line: str) -> None:
        prefix = "EDOLAB_EVENT "
        if not line.startswith(prefix):
            return
        try:
            payload = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            self._event(run_id, "protocol", "Worker emitted malformed JSON.", EventLevel.WARNING)
            return
        event = str(payload.get("event") or "worker")
        if event == "run_completed":
            completed = float(payload.get("completed_runs") or 0)
            total = float(payload.get("total_runs") or 0) or None
            self.database.update_run_progress(
                run_id,
                current=completed,
                total=total,
                unit="runs",
                message=(
                    f"{payload.get('algorithm')} run {payload.get('run_index')} completed"
                ),
            )
        elif event == "worker_started":
            current = self.database.require_run(run_id)
            self.database.update_run_progress(
                run_id,
                current=0,
                total=(
                    current.progress_total
                    or float(payload.get("total_runs") or 0)
                    or None
                ),
                unit=current.progress_unit or "runs",
                message="Execution worker started.",
            )
        elif event == "progress":
            self.database.update_run_progress(
                run_id,
                current=float(payload.get("current") or 0),
                total=float(payload.get("total") or 0) or None,
                unit=str(payload.get("unit") or "steps"),
                message=_optional_text(payload.get("message")) or "Workload in progress.",
            )
        elif event == "worker_completed":
            current = self.database.require_run(run_id)
            total = current.progress_total or 1.0
            self.database.update_run_progress(
                run_id,
                current=total,
                total=total,
                unit=current.progress_unit or "runs",
                message="Execution worker completed; collecting verified artifacts.",
            )
        elif event == "worker_failed":
            error_code = _optional_text(payload.get("error_code")) or "worker_failed"
            error_message = _optional_text(payload.get("message")) or "Remote worker failed."
            current = self.database.require_run(run_id)
            self.database.update_run_fields(
                run_id,
                error_code=error_code,
                error_message=error_message,
                error_details={
                    **dict(current.error_details),
                    "worker": dict(payload.get("details") or {}),
                    "worker_exit_code": _optional_int(payload.get("exit_code")),
                },
            )
        self._event(run_id, event, _worker_event_message(payload), data=payload)

    def _check_admission(
        self,
        server: Server,
        run: Run,
        probe: Mapping[str, Any],
        gpu_requested: bool,
    ) -> None:
        disk = dict(probe.get("disk") or {})
        conda = dict(probe.get("conda") or {})
        cpu = dict(probe.get("cpu") or {})
        gpu = dict((probe.get("gpu") or {}).get("nvidia_smi") or {})
        hard: list[str] = []
        resource: list[str] = []
        advisory: list[str] = []
        if not disk.get("writable"):
            hard.append("Configured remote root or its nearest parent is not writable.")
        free = disk.get("free_bytes")
        if free is None or int(free) < server.min_free_bytes:
            hard.append(
                f"Remote filesystem has less than the required {server.min_free_bytes} bytes free."
            )
        if int(disk.get("inodes_free") or 0) < 1000:
            hard.append("Remote filesystem has too few free inodes.")
        if not conda.get("installed"):
            hard.append("Conda is not installed or visible in the remote login environment.")
        elif not conda.get("environment_available"):
            hard.append(f"Conda environment {server.conda_env!r} does not exist.")
        else:
            conda_executable = _optional_text(conda.get("path"))
            if (
                conda_executable is None
                or not PurePosixPath(conda_executable).is_absolute()
            ):
                hard.append(
                    "The fresh probe did not report an absolute Conda executable path."
                )
            environment_python = conda.get("environment_python")
            if not isinstance(environment_python, Mapping):
                detail = str(conda.get("environment_python_error") or "").strip()
                message = (
                    f"Python inside Conda environment {server.conda_env!r} could not "
                    "be verified; Python 3.11 is required."
                )
                if detail:
                    message += f" Probe error: {detail}"
                hard.append(message)
            else:
                environment_version = str(
                    environment_python.get("version") or ""
                ).strip()
                if not _is_python_311(environment_version):
                    hard.append(
                        f"Conda environment {server.conda_env!r} must use Python 3.11; "
                        f"the fresh probe reported {environment_version or 'an unknown version'}."
                    )
                if environment_python.get("pip_available") is not True:
                    hard.append(
                        f"Conda environment {server.conda_env!r} did not confirm an "
                        "available pip installation, which is required to install the "
                        "exact uploaded snapshot."
                    )
                hard.extend(missing_runtime_dependency_messages(environment_python))
        if gpu_requested and not gpu.get("usable"):
            hard.append("The task requests GPU execution but no usable NVIDIA GPU is available.")
        ratio = cpu.get("load_ratio")
        if ratio is not None and float(ratio) > server.max_load_ratio:
            advisory.append(
                f"Load ratio {float(ratio):.2f} exceeds the configured warning threshold {server.max_load_ratio:.2f}; execution remains allowed."
            )
        logical = int(cpu.get("logical_cores") or 0)
        if logical and run.max_parallel_runs > logical:
            resource.append(
                f"Requested {run.max_parallel_runs} parallel runs but the server reports {logical} logical CPUs."
            )
        if hard:
            raise AdmissionError("ENVIRONMENT_NOT_ELIGIBLE", " ".join(hard), {"probe": probe})
        if resource and not run.admission_override:
            raise AdmissionError("RESOURCE_ADMISSION_BLOCKED", " ".join(resource), {"probe": probe})
        if advisory:
            self._event(
                run.id,
                "resource_warning",
                " ".join(advisory),
                EventLevel.WARNING,
            )
        if resource:
            self._event(
                run.id,
                "admission_override",
                "Resource admission warnings were explicitly overridden: " + " ".join(resource),
                EventLevel.WARNING,
            )

    def _resolve_task_path(self, run: Run) -> Path:
        if not run.task_path:
            raise AdmissionError("TASK_PATH_MISSING", "Run has no task path.")
        inline = run.task_payload.get("inline_task")
        if inline is not None:
            if not isinstance(inline, Mapping):
                raise AdmissionError(
                    "INLINE_TASK_PAYLOAD_INVALID",
                    "The frozen inline task payload is not a mapping.",
                )
            display_name = str(inline.get("display_name") or "")
            expected_sha256 = str(inline.get("sha256") or "")
            utf8_text = inline.get("utf8_text")
            source = inline.get("source")
            if inline.get("schema_version") != 1:
                raise AdmissionError(
                    "INLINE_TASK_PAYLOAD_INVALID",
                    "The frozen inline task payload has an unsupported schema.",
                )
            repository_task = source == "repository_task"
            repository_identity = PurePosixPath(display_name)
            repository_identity_valid = (
                repository_identity.parts == ("tasks", repository_identity.name)
                and bool(repository_identity.name)
                and not repository_identity.name.startswith(".")
                and "\\" not in repository_identity.name
                and repository_identity.suffix.lower() == ".jsonl"
            )
            if source is not None and source != "repository_task":
                raise AdmissionError(
                    "INLINE_TASK_PAYLOAD_INVALID",
                    "The frozen task payload has an unsupported source.",
                )
            if (
                display_name != run.task_path
                or (repository_task and not repository_identity_valid)
                or (not repository_task and not display_name.startswith("inline/"))
            ):
                raise AdmissionError(
                    "INLINE_TASK_IDENTITY_MISMATCH",
                    "The frozen task display identity does not match the run record.",
                )
            if run.task_sha256 != expected_sha256:
                raise AdmissionError(
                    "INLINE_TASK_IDENTITY_MISMATCH",
                    "The inline task SHA-256 does not match the run record.",
                )
            if not isinstance(utf8_text, str):
                raise AdmissionError(
                    "INLINE_TASK_PAYLOAD_INVALID",
                    "The frozen inline task is missing its UTF-8 text.",
                )
            run_dir = self.paths.working / run.id
            provenance_dir = run_dir / "provenance"
            if (
                run_dir.is_symlink()
                or not run_dir.is_dir()
                or provenance_dir.is_symlink()
                or not provenance_dir.is_dir()
            ):
                raise AdmissionError(
                    "INLINE_TASK_CAPTURE_PATH_INVALID",
                    "The private local run directory is not available for inline task capture.",
                )
            try:
                payload = utf8_text.encode("utf-8")
                return materialize_inline_task(
                    payload,
                    provenance_dir / "task.jsonl",
                    expected_sha256=expected_sha256,
                )
            except (OSError, TypeError, UnicodeError, ValueError) as exc:
                raise AdmissionError(
                    "INLINE_TASK_CAPTURE_FAILED",
                    str(exc),
                ) from exc
        if str(run.task_path).startswith("inline/"):
            raise AdmissionError(
                "INLINE_TASK_PAYLOAD_MISSING",
                "The inline task text is missing from the durable run record.",
            )
        path = Path(run.task_path)
        if not path.is_absolute():
            path = self.paths.repository_root / path
        path = path.expanduser().resolve()
        tasks_root = (self.paths.repository_root / "tasks").resolve()
        try:
            path.relative_to(tasks_root)
        except ValueError as exc:
            raise AdmissionError(
                "TASK_PATH_OUTSIDE_WORKSPACE",
                f"Task file must be inside {tasks_root}.",
            ) from exc
        if not path.is_file() or path.suffix.lower() != ".jsonl":
            raise AdmissionError("TASK_FILE_MISSING", f"Task file does not exist: {path}")
        return path

    def _materialize_workload_input(self, run: Run, controller_dir: Path) -> Path:
        del controller_dir  # Reserved for future external workload inputs.
        workload = run.task_payload.get("workload")
        if not isinstance(workload, Mapping):
            raise AdmissionError(
                "WORKLOAD_PAYLOAD_MISSING",
                "The durable workload specification is missing or invalid.",
            )
        if run.task_kind not in {"benchmark", "plot"}:
            raise AdmissionError(
                "WORKLOAD_KIND_UNSUPPORTED",
                f"Unsupported durable workload kind: {run.task_kind!r}.",
            )
        payload = (stable_json_dumps(workload) + "\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if not run.task_sha256 or digest != run.task_sha256:
            raise AdmissionError(
                "WORKLOAD_IDENTITY_MISMATCH",
                "The durable workload specification no longer matches its frozen SHA-256.",
            )
        try:
            return materialize_inline_task(
                payload,
                self.paths.working / run.id / "provenance" / "task.jsonl",
                expected_sha256=digest,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise AdmissionError("WORKLOAD_CAPTURE_FAILED", str(exc)) from exc

    @staticmethod
    def _inspect_frozen_workload(
        run: Run,
        task_path: Path,
    ) -> _FrozenWorkloadInspection:
        task_bytes = task_path.read_bytes()
        digest = hashlib.sha256(task_bytes).hexdigest()
        if digest != run.task_sha256:
            raise AdmissionError(
                "WORKLOAD_IDENTITY_MISMATCH",
                "The materialized workload input does not match the run record.",
            )
        try:
            payload = json.loads(task_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdmissionError(
                "WORKLOAD_PAYLOAD_INVALID",
                "The frozen workload input is not valid UTF-8 JSON.",
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise AdmissionError(
                "WORKLOAD_PAYLOAD_INVALID",
                "The frozen workload input has an unsupported schema.",
            )
        if payload.get("workload_kind") != run.task_kind:
            raise AdmissionError(
                "WORKLOAD_PAYLOAD_INVALID",
                "The frozen workload kind does not match the run record.",
            )
        operation = str(payload.get("operation") or "")
        allowed_operations = (
            {"contour", "surface", "nbn_views", "nbn_model"}
            if run.task_kind == "benchmark"
            else {"trend", "boxplot"}
        )
        if operation not in allowed_operations:
            raise AdmissionError(
                "WORKLOAD_PAYLOAD_INVALID",
                "The frozen workload operation is unsupported.",
            )
        request = payload.get("payload")
        if not isinstance(request, Mapping):
            raise AdmissionError(
                "WORKLOAD_PAYLOAD_INVALID",
                "The frozen workload request is missing.",
            )
        if run.task_kind == "benchmark":
            device = str(request.get("evaluation_device") or "auto").lower()
            sample_count = int(request.get("sample_count") or 0)
            resolution = int(request.get("resolution") or 0)
            estimated = (
                sample_count
                if operation in {"nbn_views", "nbn_model"}
                else resolution * resolution
            )
            gpu_requested = device == "cuda" or device.startswith("cuda:")
        else:
            series = request.get("series")
            if not isinstance(series, list):
                raise AdmissionError(
                    "WORKLOAD_PAYLOAD_INVALID",
                    "The frozen plot workload has no series array.",
                )
            value_key = "key" if operation == "trend" else "values"
            estimated = sum(
                len(item.get(value_key) or [])
                for item in series
                if isinstance(item, Mapping)
            )
            gpu_requested = False
        return _FrozenWorkloadInspection(
            task_bytes=task_bytes,
            task_file_sha256=digest,
            workload_kind=run.task_kind,
            operation=operation,
            gpu_requested=gpu_requested,
            total_evals=max(0, estimated),
        )

    def _require_clean_git_if_configured(self) -> None:
        settings = self.database.get_setting("relay.settings", {})
        if not isinstance(settings, Mapping) or not bool(
            settings.get("require_clean_git", False)
        ):
            return
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(self.paths.repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AdmissionError(
                "GIT_STATUS_FAILED",
                "Clean-Git policy is enabled, but the controller could not "
                "verify the local worktree state"
                + (f": {detail}" if detail else "."),
            )
        if completed.stdout.strip():
            raise AdmissionError(
                "GIT_WORKTREE_DIRTY",
                "Clean-Git policy is enabled and the worktree changed after "
                "submission; refusing to dispatch a different source state.",
            )

    def _server_for_run(self, run: Run) -> Server:
        """Rebuild the immutable execution target captured at submission time."""

        snapshot = run.task_payload.get("server_snapshot")
        if not isinstance(snapshot, Mapping):
            # Compatibility for records created before server snapshots were
            # introduced and for low-level unit tests that construct Run
            # directly.  New API-created runs always take the frozen branch.
            return self.database.require_server(run.server_id)
        try:
            if int(snapshot.get("schema_version", 0)) not in {1, 2}:
                raise ValueError("unsupported server snapshot schema")
            snapshot_id = str(snapshot.get("id") or "")
            if snapshot_id != run.server_id:
                raise ValueError("server snapshot identity does not match the run")
            return Server(
                id=snapshot_id,
                target_kind=str(snapshot.get("target_kind") or "ssh"),
                name=str(snapshot["name"]),
                host=str(snapshot["host"]),
                port=int(snapshot["port"]),
                username=str(snapshot["username"]),
                ssh_key_path=str(snapshot["ssh_key_path"]),
                remote_root=str(snapshot["remote_root"]),
                conda_env=str(snapshot["conda_env"]),
                poll_interval_seconds=float(snapshot["poll_interval_seconds"]),
                default_max_parallel_runs=int(
                    snapshot["default_max_parallel_runs"]
                ),
                min_free_bytes=int(snapshot["min_free_bytes"]),
                max_load_ratio=float(snapshot["max_load_ratio"]),
                fingerprint=_optional_text(snapshot.get("fingerprint")),
                enabled=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdmissionError(
                "SERVER_SNAPSHOT_INVALID",
                "The run's frozen server target is missing or invalid; refusing "
                "to fall back to a potentially edited server profile.",
            ) from exc

    def _execution_target_key(self, run: Run) -> str:
        """Serialize jobs that could mutate the same remote Conda environment."""

        server = self._server_for_run(run)
        identity = (
            {"host_fingerprint": server.fingerprint.strip()}
            if server.fingerprint and server.fingerprint.strip()
            else {
                "host": server.host.rstrip(".").lower(),
                "port": server.port,
            }
        )
        return stable_json_dumps(
            {
                "target_kind": server.target_kind.value,
                **identity,
                "username": server.username,
                "conda_env": server.conda_env,
            }
        )

    def _installation_id(self) -> str:
        return str(
            self.database.get_or_create_setting("installation_id", new_id())
        )

    def _working_dir(self, run_id: str) -> Path:
        path = self.paths.working / run_id
        if path.exists():
            raise RuntimeError(f"Local working directory already exists: {path}")
        path.mkdir(mode=0o700, parents=True)
        return path

    def _existing_local_dir(self, run_id: str) -> Path:
        final = self.paths.runs / run_id
        if final.is_dir():
            return final
        working = self.paths.working / run_id
        if working.is_dir():
            return working
        working.mkdir(mode=0o700, parents=True, exist_ok=True)
        return working

    def _promote_local_dir(self, run_id: str) -> str:
        working = self.paths.working / run_id
        final = self.paths.runs / run_id
        if final.is_dir():
            fsync_directory(final)
            fsync_directory(self.paths.runs)
            fsync_directory(self.paths.working)
            return f"runs/{run_id}"
        if not working.is_dir():
            raise RuntimeError("Local artifact working directory is missing.")
        os.replace(working, final)
        # A cross-directory rename changes both directory entry sets.  Sync
        # the moved directory and both parents so recovery cannot lose the
        # promoted run or resurrect its old working entry after a crash.
        fsync_directory(final)
        fsync_directory(self.paths.runs)
        fsync_directory(self.paths.working)
        return f"runs/{run_id}"

    def _fail_run(self, run_id: str, exc: BaseException) -> None:
        run = self.database.get_run(run_id)
        if run is None or run.terminal:
            return
        code = getattr(exc, "code", None) or type(exc).__name__.upper()
        details = {
            **dict(run.error_details),
            **dict(getattr(exc, "details", None) or {}),
        }
        if (
            run.transfer_verified
            and run.exit_code == 0
            and run.status in {RunStatus.VERIFYING, RunStatus.CLEANING}
        ):
            details.setdefault("cause_code", str(code))
            details["phase"] = "verify"
            code = LOCAL_TRANSFER_VERIFICATION_ERROR
        if run.stage:
            failed_stage = {
                **dict(run.stage),
                "state": "failed",
                "message": str(exc),
            }
            try:
                run = self.database.set_run_stage(
                    run.id,
                    failed_stage,
                    message=str(exc),
                    level=EventLevel.ERROR,
                )
            except Exception:
                pass
            details.setdefault("phase", failed_stage.get("phase"))
            details.setdefault("stage", failed_stage)
        self._event(run.id, "failure", str(exc), EventLevel.ERROR, data={"code": code, **dict(details)})
        try:
            local_dir = self._existing_local_dir(run.id)
            (local_dir / "controller").mkdir(parents=True, exist_ok=True)
            (local_dir / "controller" / "failure.json").write_text(
                stable_json_dumps({"code": code, "message": str(exc), "at": utc_now()}) + "\n",
                encoding="utf-8",
            )
            final_relative = self._promote_local_dir(run.id)
            self.database.update_run_fields(run.id, local_artifact_dir=final_relative)
        except Exception:
            pass
        self.database.set_run_status(
            run.id,
            RunStatus.FAILED,
            error_code=str(code),
            error_message=str(exc),
            error_details=details,
            validate_transition=False,
        )

    def _event(
        self,
        run_id: str,
        kind: str,
        message: str,
        level: EventLevel = EventLevel.INFO,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self.database.append_run_event(
            RunEvent(
                run_id=run_id,
                kind=kind,
                message=message,
                level=level,
                data=dict(data or {}),
            )
        )


def _build_source_sync_plan(
    workspace: RemoteWorkspace,
    target_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_snapshot_id = str(target_manifest.get("snapshot_id") or "")
    if (
        not target_snapshot_id
        or target_manifest.get("manifest_sha256") != target_snapshot_id
        or canonical_manifest_hash(target_manifest) != target_snapshot_id
    ):
        raise RuntimeError("The frozen local source manifest identity is invalid.")

    if workspace.base_snapshot_id:
        if workspace.previous_manifest is None:
            raise RuntimeError("The remote source cache baseline manifest is missing.")
        baseline = dict(workspace.previous_manifest)
        if (
            baseline.get("snapshot_id") != workspace.base_snapshot_id
            or baseline.get("manifest_sha256") != workspace.base_snapshot_id
            or canonical_manifest_hash(baseline) != workspace.base_snapshot_id
        ):
            raise RuntimeError("The remote source cache baseline identity is invalid.")
    else:
        baseline = {"schema_version": 1, "source": None, "files": []}

    difference = diff_manifests(baseline, target_manifest)
    target_files = {
        str(item["path"]): dict(item) for item in target_manifest.get("files") or []
    }
    transfer_paths = set(difference.added)
    metadata_only_paths: list[str] = []
    for change in difference.changed:
        if (
            change.before.get("sha256") != change.after.get("sha256")
            or change.before.get("size_bytes") != change.after.get("size_bytes")
        ):
            transfer_paths.add(change.path)
        else:
            metadata_only_paths.append(change.path)
    ordered_transfer_paths = tuple(sorted(transfer_paths))
    strategy = (
        "cache"
        if workspace.cache_hit
        and workspace.base_snapshot_id == target_snapshot_id
        else "delta"
        if workspace.base_snapshot_id
        else "full"
    )
    if strategy == "cache" and not difference.is_empty:
        raise RuntimeError("The exact remote cache hit differs from the target manifest.")
    plan = {
        "schema_version": 1,
        "strategy": strategy,
        "base_snapshot_id": workspace.base_snapshot_id,
        "target_snapshot_id": target_snapshot_id,
        "cache_hit": workspace.cache_hit,
        "added": list(difference.added),
        "changed": [item.path for item in difference.changed],
        "metadata_only": sorted(metadata_only_paths),
        "removed": list(difference.removed),
        "source_changed": difference.source_changed,
        "transfer_paths": list(ordered_transfer_paths),
        "transfer_file_count": len(ordered_transfer_paths),
        "transfer_size_bytes": sum(
            int(target_files[path]["size_bytes"]) for path in ordered_transfer_paths
        ),
        "reused_file_count": len(target_files) - len(ordered_transfer_paths),
        "target_file_count": len(target_files),
    }
    manifest_diff = {
        **difference.to_dict(),
        "strategy": strategy,
        "base_snapshot_id": workspace.base_snapshot_id,
        "target_snapshot_id": target_snapshot_id,
        "metadata_only": sorted(metadata_only_paths),
        "transfer_paths": list(ordered_transfer_paths),
        "transfer_file_count": len(ordered_transfer_paths),
        "transfer_size_bytes": plan["transfer_size_bytes"],
        "reused_file_count": plan["reused_file_count"],
        "target_file_count": len(target_files),
    }
    return plan, manifest_diff


def _compact_sync_details(details: Mapping[str, Any]) -> dict[str, Any]:
    source = details.get("plan")
    if not isinstance(source, Mapping):
        source = details
    compact: dict[str, Any] = {}
    for key in (
        "strategy",
        "base_snapshot_id",
        "target_snapshot_id",
        "cache_hit",
        "transfer_file_count",
        "transfer_size_bytes",
        "payload_size_bytes",
        "reused_file_count",
        "target_file_count",
        "file_count",
        "transferred_files",
        "reused_files",
        "cache_storage",
        "cache_materialization",
        "run_materialization",
        "legacy_promoted",
        "cache_race",
        "archive_reads",
        "archive_writes",
        "archive_hashed_bytes",
        "content_hashed_bytes",
        "applied_transfer_files",
    ):
        if key in source:
            compact[key] = source[key]
        elif key in details:
            compact[key] = details[key]
    for key in ("added", "changed", "metadata_only", "removed"):
        value = source.get(key)
        if isinstance(value, (list, tuple)):
            compact[key + "_count"] = len(value)
    return compact


def _sample_from_probe(server_id: str, run_id: str | None, probe: Mapping[str, Any]) -> ResourceSample:
    cpu = dict(probe.get("cpu") or {})
    memory = dict(probe.get("memory") or {})
    disk = dict(probe.get("disk") or {})
    gpu = dict((probe.get("gpu") or {}).get("nvidia_smi") or {})
    return ResourceSample(
        server_id=server_id,
        run_id=run_id,
        cpu_physical_cores=_optional_int(cpu.get("physical_cores")),
        cpu_logical_cores=_optional_int(cpu.get("logical_cores")),
        cpu_percent=_optional_float(cpu.get("percent")),
        load_1m=_optional_float(cpu.get("load_1m")),
        load_5m=_optional_float(cpu.get("load_5m")),
        load_15m=_optional_float(cpu.get("load_15m")),
        load_ratio=_optional_float(cpu.get("load_ratio")),
        memory_total_bytes=_optional_int(memory.get("total_bytes")),
        memory_available_bytes=_optional_int(memory.get("available_bytes")),
        disk_total_bytes=_optional_int(disk.get("total_bytes")),
        disk_free_bytes=_optional_int(disk.get("free_bytes")),
        gpu_devices=tuple(gpu.get("devices") or ()),
    )


def _sample_from_poll(server_id: str, run_id: str, resource: Mapping[str, Any]) -> ResourceSample:
    memory = dict(resource.get("memory") or {})
    return ResourceSample(
        server_id=server_id,
        run_id=run_id,
        cpu_logical_cores=_optional_int(resource.get("cpu_logical_cores")),
        cpu_percent=_optional_float(resource.get("cpu_percent")),
        load_1m=_optional_float(resource.get("load_1m")),
        load_5m=_optional_float(resource.get("load_5m")),
        load_15m=_optional_float(resource.get("load_15m")),
        load_ratio=_optional_float(resource.get("load_ratio")),
        memory_total_bytes=_optional_int(memory.get("total_bytes")),
        memory_available_bytes=_optional_int(memory.get("available_bytes")),
        gpu_devices=tuple(resource.get("gpus") or ()),
    )


def _worker_event_message(payload: Mapping[str, Any]) -> str:
    event = str(payload.get("event") or "worker")
    if event == "run_completed":
        return (
            f"[{payload.get('completed_runs')}/{payload.get('total_runs')}] "
            f"{payload.get('algorithm')} run {payload.get('run_index')} completed "
            f"at FE={payload.get('fe')}."
        )
    if event == "worker_started":
        return f"Worker started {payload.get('total_runs')} run(s)."
    if event == "worker_completed":
        return "Remote worker completed and closed the result bundle."
    return str(payload.get("message") or event)


def _remote_error_from_log(path: Path, exit_code: int | None) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    for line in reversed(text.splitlines()):
        marker = "EDOLAB_ERROR "
        if marker in line:
            payload = line.split(marker, 1)[1]
            code, _, message = payload.partition(":")
            return code.strip() or "REMOTE_FAILED", message.strip() or line.strip()
    return "REMOTE_EXIT_NONZERO", f"Remote worker exited with code {exit_code}."


def _append_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)


def _monitor_state_from_logs(controller_dir: Path) -> _MonitorState:
    """Resume remote log cursors from durable local controller logs."""

    stdout_path = controller_dir / "stdout.log"
    stderr_path = controller_dir / "stderr.log"
    stdout_offset = stdout_path.stat().st_size if stdout_path.is_file() else 0
    stderr_offset = stderr_path.stat().st_size if stderr_path.is_file() else 0
    pending = b""
    if stdout_offset:
        with stdout_path.open("rb") as handle:
            handle.seek(max(0, stdout_offset - 64 * 1024))
            tail = handle.read()
        if b"\n" not in tail:
            pending = tail
        elif not tail.endswith(b"\n"):
            pending = tail.rsplit(b"\n", 1)[-1]
    return _MonitorState(
        stdout_offset=stdout_offset,
        stderr_offset=stderr_offset,
        stdout_pending=pending,
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _execution_conda_executable(payload: Mapping[str, Any]) -> str | None:
    probe = payload.get("execution_probe")
    if not isinstance(probe, Mapping):
        return None
    conda = probe.get("conda")
    if not isinstance(conda, Mapping):
        return None
    executable = _optional_text(conda.get("path"))
    if executable is None or not PurePosixPath(executable).is_absolute():
        return None
    return executable


def _is_transport_failure(exc: BaseException) -> bool:
    if isinstance(exc, SSHCommandError):
        # OpenSSH uses 255 for connection/protocol loss.  A negative return
        # code means the *local* ssh subprocess was terminated by a signal
        # (for example ``-SIGINT`` when Relay receives Ctrl+C); it is not
        # evidence that the detached remote worker failed.  Treat both as a
        # retryable transport interruption so controller shutdown can leave
        # the durable RUNNING/PREPARING record available for startup recovery.
        return exc.returncode == 255 or exc.returncode < 0
    # SSHError represents local validation, malformed remote protocol, unsafe
    # paths, or checksum mismatches.  Retrying those forever hides deterministic
    # corruption.  OpenSSH transport loss is reported as return code 255;
    # bounded timeouts and explicit connection errors remain retryable.
    return isinstance(
        exc,
        (subprocess.TimeoutExpired, TimeoutError, ConnectionError),
    )


def _artifact_download_timeout(size_bytes: int) -> float:
    """Allow large bundles to transfer at a conservative 128 KiB/s floor.

    The lab link has dropped to about 240 KB/s at times; a 300 MB bundle then needs about 22 minutes, which the
    earlier 512 KiB/s floor (about 15 minutes) cut off on every retry."""

    size = max(0, int(size_bytes))
    return max(600.0, min(28_800.0, 300.0 + size / (128 * 1024)))


def _is_python_311(value: str) -> bool:
    parts = value.strip().split(".")
    return len(parts) >= 2 and parts[0] == "3" and parts[1] == "11"


def _needs_cleanup_retry(run: Run) -> bool:
    return bool(
        run.transfer_verified
        and run.cleanup_requested
        and not run.remote_cleanup_completed
        and run.remote_run_dir
    )


def _needs_unlaunched_cleanup_retry(run: Run) -> bool:
    if run_recovery_mode(run) != "cleanup" or run.transfer_verified:
        return False
    execution = run.task_payload.get("execution")
    return not (
        isinstance(execution, Mapping) and execution.get("launch_intent_at")
    )


def _needs_terminal_retry(run: Run) -> bool:
    return _needs_cleanup_retry(run) or _needs_unlaunched_cleanup_retry(run)


__all__ = [
    "AdmissionError",
    "ExecutionBackend",
    "OpenSSHExecutionBackend",
    "UnifiedExecutionBackend",
    "RelayController",
]
