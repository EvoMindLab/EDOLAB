"""SQLite persistence for the local-first remote execution subsystem.

Every operation opens and closes its own connection.  Connections are therefore
never shared between worker, API, or monitor threads; WAL mode and a busy timeout
provide bounded coordination between those short-lived transactions.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Any
from .performance import metrics
from .resource_rollup import FIELDS as ROLLUP_FIELDS, merge_minute

from .domain import (
    NON_TERMINAL_RUN_STATUSES,
    Artifact,
    ArtifactStatus,
    DashboardCounts,
    EventLevel,
    ResourceSample,
    Run,
    RunEvent,
    RunStatus,
    Server,
    stable_json_dumps,
    stable_json_loads,
    utc_now,
    validate_run_transition,
)


SCHEMA_VERSION = 11

_RUN_STATUS_SQL = ",".join(f"'{status.value}'" for status in RunStatus)
_NON_TERMINAL_STATUS_SQL = ",".join(
    f"'{status.value}'" for status in sorted(NON_TERMINAL_RUN_STATUSES, key=str)
)
_ARTIFACT_STATUS_SQL = ",".join(f"'{status.value}'" for status in ArtifactStatus)
_EVENT_LEVEL_SQL = ",".join(f"'{level.value}'" for level in EventLevel)


class ServerTargetChangedError(RuntimeError):
    """Raised when a probe result no longer belongs to the saved SSH target."""


_MIGRATION_1 = (
    """
    CREATE TABLE IF NOT EXISTS servers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        host TEXT NOT NULL,
        port INTEGER NOT NULL CHECK(port BETWEEN 1 AND 65535),
        username TEXT NOT NULL,
        ssh_key_path TEXT NOT NULL
            CHECK(instr(upper(ssh_key_path), 'PRIVATE KEY') = 0)
            CHECK(instr(ssh_key_path, char(10)) = 0)
            CHECK(instr(ssh_key_path, char(13)) = 0),
        remote_root TEXT NOT NULL,
        conda_env TEXT NOT NULL,
        poll_interval_seconds REAL NOT NULL CHECK(poll_interval_seconds > 0),
        default_max_parallel_runs INTEGER NOT NULL
            CHECK(default_max_parallel_runs >= 1),
        min_free_bytes INTEGER NOT NULL CHECK(min_free_bytes >= 0),
        max_load_ratio REAL NOT NULL CHECK(max_load_ratio > 0),
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
        last_probe_at TEXT,
        last_probe_json TEXT,
        fingerprint TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        server_id TEXT NOT NULL REFERENCES servers(id) ON DELETE RESTRICT,
        status TEXT NOT NULL CHECK(status IN ("""
    + _RUN_STATUS_SQL
    + """)),
        task_name TEXT NOT NULL,
        task_kind TEXT NOT NULL,
        task_path TEXT,
        task_sha256 TEXT,
        task_payload_json TEXT NOT NULL DEFAULT '{}',
        command_argv_json TEXT NOT NULL DEFAULT '[]',
        max_parallel_runs INTEGER NOT NULL DEFAULT 1
            CHECK(max_parallel_runs >= 1),
        poll_interval_seconds REAL NOT NULL DEFAULT 5
            CHECK(poll_interval_seconds > 0),
        admission_override INTEGER NOT NULL DEFAULT 0
            CHECK(admission_override IN (0, 1)),

        source_root TEXT,
        snapshot_id TEXT,
        snapshot_sha256 TEXT,
        snapshot_manifest_json TEXT NOT NULL DEFAULT '{}',
        git_commit TEXT,
        git_branch TEXT,
        git_dirty INTEGER CHECK(git_dirty IN (0, 1) OR git_dirty IS NULL),
        git_diff_sha256 TEXT,

        remote_pid INTEGER CHECK(remote_pid > 0 OR remote_pid IS NULL),
        remote_run_dir TEXT,
        worker_id TEXT,
        heartbeat_at TEXT,

        progress_current REAL NOT NULL DEFAULT 0 CHECK(progress_current >= 0),
        progress_total REAL CHECK(progress_total >= 0 OR progress_total IS NULL),
        progress_unit TEXT,
        progress_message TEXT,

        exit_code INTEGER,
        error_code TEXT,
        error_message TEXT,
        error_details_json TEXT NOT NULL DEFAULT '{}',

        local_artifact_dir TEXT,
        transfer_manifest_json TEXT NOT NULL DEFAULT '{}',
        transfer_started_at TEXT,
        transfer_completed_at TEXT,
        transfer_verified INTEGER NOT NULL DEFAULT 0
            CHECK(transfer_verified IN (0, 1)),
        transfer_verified_at TEXT,
        cleanup_requested INTEGER NOT NULL DEFAULT 1
            CHECK(cleanup_requested IN (0, 1)),
        remote_cleanup_completed INTEGER NOT NULL DEFAULT 0
            CHECK(remote_cleanup_completed IN (0, 1)),
        remote_cleanup_at TEXT,
        remote_cleanup_error TEXT,

        queued_at TEXT NOT NULL,
        started_at TEXT,
        remote_finished_at TEXT,
        finished_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL CHECK(sequence >= 1),
        level TEXT NOT NULL CHECK(level IN ("""
    + _EVENT_LEVEL_SQL
    + """)),
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(run_id, sequence)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS resource_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        server_id TEXT NOT NULL REFERENCES servers(id) ON DELETE CASCADE,
        run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
        sampled_at TEXT NOT NULL,
        reachable INTEGER NOT NULL CHECK(reachable IN (0, 1)),
        cpu_physical_cores INTEGER CHECK(
            cpu_physical_cores >= 0 OR cpu_physical_cores IS NULL
        ),
        cpu_logical_cores INTEGER CHECK(
            cpu_logical_cores >= 0 OR cpu_logical_cores IS NULL
        ),
        cpu_percent REAL,
        load_1m REAL,
        load_5m REAL,
        load_15m REAL,
        load_ratio REAL,
        memory_total_bytes INTEGER CHECK(
            memory_total_bytes >= 0 OR memory_total_bytes IS NULL
        ),
        memory_available_bytes INTEGER CHECK(
            memory_available_bytes >= 0 OR memory_available_bytes IS NULL
        ),
        disk_total_bytes INTEGER CHECK(
            disk_total_bytes >= 0 OR disk_total_bytes IS NULL
        ),
        disk_free_bytes INTEGER CHECK(
            disk_free_bytes >= 0 OR disk_free_bytes IS NULL
        ),
        gpu_devices_json TEXT NOT NULL DEFAULT '[]',
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        relative_path TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ("""
    + _ARTIFACT_STATUS_SQL
    + """)),
        remote_path TEXT,
        local_path TEXT,
        size_bytes INTEGER CHECK(size_bytes >= 0 OR size_bytes IS NULL),
        sha256 TEXT,
        transferred_at TEXT,
        verified_at TEXT,
        verification_error TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(run_id, relative_path)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_servers_enabled_name "
    "ON servers(enabled, name COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_runs_created "
    "ON runs(created_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_server_created "
    "ON runs(server_id, created_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_server_status_created "
    "ON runs(server_id, status, created_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_status_updated "
    "ON runs(status, updated_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_nonterminal_updated "
    "ON runs(updated_at, id) WHERE status IN ("
    + _NON_TERMINAL_STATUS_SQL
    + ")",
    "CREATE INDEX IF NOT EXISTS idx_run_events_run_sequence "
    "ON run_events(run_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_resource_samples_server_time "
    "ON resource_samples(server_id, sampled_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_resource_samples_run_time "
    "ON resource_samples(run_id, sampled_at DESC, id DESC) WHERE run_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run_status "
    "ON artifacts(run_id, status, relative_path)",
)

_MIGRATION_2 = (
    "ALTER TABLE resource_samples ADD COLUMN gpu_usable INTEGER "
    "CHECK(gpu_usable IN (0, 1) OR gpu_usable IS NULL)",
    "ALTER TABLE resource_samples ADD COLUMN gpu_error TEXT",
)

_MIGRATION_3 = (
    "ALTER TABLE resource_samples ADD COLUMN target_identity TEXT",
    "CREATE INDEX IF NOT EXISTS idx_resource_samples_server_identity_time "
    "ON resource_samples(server_id, target_identity, sampled_at DESC, id DESC) "
    "WHERE run_id IS NULL",
)

_MIGRATION_4 = (
    "ALTER TABLE runs ADD COLUMN stage_json TEXT NOT NULL DEFAULT '{}'",
)

_MIGRATION_5 = (
    """
    CREATE TABLE IF NOT EXISTS sa_imports (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        runset_uid TEXT NOT NULL,
        bundle_id TEXT NOT NULL,
        task_name TEXT NOT NULL,
        label TEXT NOT NULL,
        algorithm_name TEXT NOT NULL,
        problem_name TEXT NOT NULL,
        problem_params_json TEXT NOT NULL,
        algorithm_params_json TEXT NOT NULL,
        scalar_metrics_json TEXT NOT NULL,
        trend_metrics_json TEXT NOT NULL,
        expected_runs INTEGER NOT NULL CHECK(expected_runs >= 0),
        completed_runs INTEGER NOT NULL CHECK(completed_runs >= 0),
        source_fingerprint TEXT NOT NULL,
        pairing_signature TEXT NOT NULL,
        snapshot_sha256 TEXT NOT NULL,
        task_sha256 TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        UNIQUE(run_id, runset_uid)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sa_analyses (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
        method TEXT NOT NULL,
        alpha REAL NOT NULL CHECK(alpha > 0 AND alpha < 1),
        metric_names_json TEXT NOT NULL,
        source_imports_json TEXT NOT NULL,
        result_json TEXT NOT NULL DEFAULT '{}',
        output_dir TEXT,
        output_files_json TEXT NOT NULL DEFAULT '{}',
        error_message TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sa_imports_run ON sa_imports(run_id, imported_at, id)",
    "CREATE INDEX IF NOT EXISTS idx_sa_imports_problem ON sa_imports(problem_name, algorithm_name, imported_at)",
    "CREATE INDEX IF NOT EXISTS idx_sa_analyses_created ON sa_analyses(created_at DESC, id DESC)",
)

_MIGRATION_6 = (
    "ALTER TABLE servers ADD COLUMN target_kind TEXT NOT NULL DEFAULT 'ssh' "
    "CHECK(target_kind IN ('ssh', 'local'))",
    "CREATE INDEX IF NOT EXISTS idx_servers_kind_enabled_name "
    "ON servers(target_kind, enabled, name COLLATE NOCASE)",
)

_MIGRATION_7 = (
    """
    CREATE TABLE IF NOT EXISTS result_imports (
        id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        source_kind TEXT NOT NULL
            CHECK(source_kind IN ('external', 'run', 'legacy_run')),
        source_run_id TEXT,
        source_metadata_json TEXT NOT NULL DEFAULT '{}',
        bundle_id TEXT NOT NULL,
        managed_relative_dir TEXT NOT NULL UNIQUE,
        validation_json TEXT NOT NULL,
        summary_size_bytes INTEGER NOT NULL CHECK(summary_size_bytes >= 0),
        summary_sha256 TEXT NOT NULL,
        trend_size_bytes INTEGER NOT NULL CHECK(trend_size_bytes >= 0),
        trend_sha256 TEXT NOT NULL,
        imported_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_result_imports_imported "
    "ON result_imports(imported_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_result_imports_source_run "
    "ON result_imports(source_run_id) WHERE source_run_id IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_result_imports_unique_run_source "
    "ON result_imports(source_run_id) "
    "WHERE source_run_id IS NOT NULL AND source_kind = 'run'",
    """
    INSERT OR IGNORE INTO result_imports (
        id, display_name, source_kind, source_run_id, source_metadata_json,
        bundle_id, managed_relative_dir, validation_json,
        summary_size_bytes, summary_sha256, trend_size_bytes, trend_sha256,
        imported_at
    )
    SELECT
        r.id,
        r.task_name,
        'legacy_run',
        r.id,
        json_object('migration', 'synthetic_result_import_run'),
        coalesce(json_extract(r.task_payload_json, '$.validation.bundle_id'), r.id),
        coalesce(r.local_artifact_dir, 'runs/' || r.id) || '/remote/results/bundle',
        coalesce(json_extract(r.task_payload_json, '$.validation'), '{}'),
        coalesce((
            SELECT a.size_bytes FROM artifacts AS a
            WHERE a.run_id = r.id
              AND a.relative_path = 'remote/results/bundle/summary.sqlite'
        ), 0),
        coalesce((
            SELECT a.sha256 FROM artifacts AS a
            WHERE a.run_id = r.id
              AND a.relative_path = 'remote/results/bundle/summary.sqlite'
        ), ''),
        coalesce((
            SELECT a.size_bytes FROM artifacts AS a
            WHERE a.run_id = r.id
              AND a.relative_path = 'remote/results/bundle/trend.sqlite'
        ), 0),
        coalesce((
            SELECT a.sha256 FROM artifacts AS a
            WHERE a.run_id = r.id
              AND a.relative_path = 'remote/results/bundle/trend.sqlite'
        ), ''),
        coalesce(r.finished_at, r.created_at)
    FROM runs AS r
    WHERE r.task_kind = 'result_import' AND r.transfer_verified = 1
    """,
)

_MIGRATION_8 = (
    "ALTER TABLE sa_imports RENAME TO sa_imports_legacy",
    """
    CREATE TABLE sa_imports (
        id TEXT PRIMARY KEY,
        result_id TEXT NOT NULL REFERENCES result_imports(id) ON DELETE CASCADE,
        runset_uid TEXT NOT NULL,
        bundle_id TEXT NOT NULL,
        task_name TEXT NOT NULL,
        label TEXT NOT NULL,
        algorithm_name TEXT NOT NULL,
        problem_name TEXT NOT NULL,
        problem_params_json TEXT NOT NULL,
        algorithm_params_json TEXT NOT NULL,
        scalar_metrics_json TEXT NOT NULL,
        trend_metrics_json TEXT NOT NULL,
        expected_runs INTEGER NOT NULL CHECK(expected_runs >= 0),
        completed_runs INTEGER NOT NULL CHECK(completed_runs >= 0),
        source_fingerprint TEXT NOT NULL,
        pairing_signature TEXT NOT NULL,
        snapshot_sha256 TEXT NOT NULL,
        task_sha256 TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        UNIQUE(result_id, runset_uid)
    )
    """,
    """
    INSERT INTO sa_imports (
        id, result_id, runset_uid, bundle_id, task_name, label,
        algorithm_name, problem_name, problem_params_json,
        algorithm_params_json, scalar_metrics_json, trend_metrics_json,
        expected_runs, completed_runs, source_fingerprint,
        pairing_signature, snapshot_sha256, task_sha256, imported_at
    )
    SELECT
        legacy.id,
        (
            SELECT imported.id
            FROM result_imports AS imported
            WHERE imported.source_run_id = legacy.run_id
               OR imported.managed_relative_dir =
                  coalesce(r.local_artifact_dir, 'runs/' || r.id) || '/remote/results/bundle'
            ORDER BY
                CASE imported.source_kind
                    WHEN 'run' THEN 0
                    WHEN 'external' THEN 1
                    ELSE 2
                END,
                imported.imported_at,
                imported.id
            LIMIT 1
        ),
        legacy.runset_uid, legacy.bundle_id,
        legacy.task_name, legacy.label, legacy.algorithm_name,
        legacy.problem_name, legacy.problem_params_json,
        legacy.algorithm_params_json, legacy.scalar_metrics_json,
        legacy.trend_metrics_json, legacy.expected_runs,
        legacy.completed_runs, legacy.source_fingerprint,
        legacy.pairing_signature, legacy.snapshot_sha256,
        legacy.task_sha256, legacy.imported_at
    FROM sa_imports_legacy AS legacy
    JOIN runs AS r ON r.id = legacy.run_id
    WHERE EXISTS (
        SELECT 1
        FROM result_imports AS imported
        WHERE imported.source_run_id = legacy.run_id
           OR imported.managed_relative_dir =
              coalesce(r.local_artifact_dir, 'runs/' || r.id) || '/remote/results/bundle'
    )
    """,
    "DROP TABLE sa_imports_legacy",
    "CREATE INDEX idx_sa_imports_result ON sa_imports(result_id, imported_at, id)",
    "CREATE INDEX idx_sa_imports_problem ON sa_imports(problem_name, algorithm_name, imported_at)",
)

_MIGRATION_9 = (
    """
    CREATE TABLE IF NOT EXISTS result_import_cleanup_queue (
        result_id TEXT PRIMARY KEY,
        managed_relative_dir TEXT NOT NULL,
        reason TEXT NOT NULL,
        queued_at TEXT NOT NULL
    )
    """,
    """
    INSERT OR IGNORE INTO result_import_cleanup_queue (
        result_id, managed_relative_dir, reason, queued_at
    )
    SELECT
        imported.id,
        imported.managed_relative_dir,
        'legacy_sa_candidate_auto_promotion',
        imported.imported_at
    FROM result_imports AS imported
    JOIN runs AS source_run ON source_run.id = imported.source_run_id
    WHERE imported.source_kind = 'legacy_run'
      AND source_run.task_kind <> 'result_import'
    """,
    """
    DELETE FROM result_imports
    WHERE id IN (
        SELECT result_id
        FROM result_import_cleanup_queue
        WHERE reason = 'legacy_sa_candidate_auto_promotion'
    )
    """,
)


_MIGRATION_10 = (
    "CREATE INDEX IF NOT EXISTS idx_resource_samples_identity_reachable_time "
    "ON resource_samples(server_id, target_identity, reachable, sampled_at DESC, id DESC) "
    "WHERE run_id IS NULL",
)

_MIGRATION_11 = (
    "ALTER TABLE resource_samples ADD COLUMN aggregation_json TEXT NOT NULL DEFAULT '{}'",
    "CREATE INDEX idx_resource_background_age ON resource_samples(sampled_at, id) WHERE run_id IS NULL",
    "CREATE INDEX idx_resource_background_raw_age ON resource_samples(sampled_at, id) WHERE run_id IS NULL AND aggregation_json = '{}'",
    "CREATE INDEX idx_resource_background_server_id ON resource_samples(server_id, id DESC) WHERE run_id IS NULL",
    "CREATE INDEX idx_run_recent_audit ON run_events(created_at DESC, id DESC) WHERE kind IN ('audit','status','cancel','cleanup','probe')",
    "CREATE INDEX idx_runs_created_id ON runs(created_at DESC, id DESC)",
)


class RemoteDatabase:
    """Thread-safe-by-construction SQLite repository.

    The object stores only a path and configuration.  ``sqlite3.Connection``
    instances live inside individual method calls and are always closed in the
    same thread that opened them.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 10_000,
    ) -> None:
        configured_path = Path(path).expanduser()
        if configured_path.is_symlink():
            raise RuntimeError(
                f"Relay database path must not be a symlink: {configured_path}"
            )
        self.path = configured_path.resolve()
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must not be negative")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._summary_run_columns: str | None = None
        self.initialize()
        self._secure_file_modes()

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        started = time.perf_counter()
        connection = self._open()
        statements = 0
        failed = False
        def count_statement(_statement):
            nonlocal statements
            statements += 1
        connection.set_trace_callback(count_statement)
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            failed = True
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_file_modes()
            metrics.increment("database.statements", statements)
            metrics.record("database.write" if write else "database.read", time.perf_counter() - started, failed=failed)

    def initialize(self) -> None:
        """Apply missing migrations; safe to call repeatedly or concurrently."""

        with self._schema_lock:
            connection = self._open()
            try:
                # journal_mode persists in the database file.  It must be set
                # outside the explicit migration transaction.
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    )
                    """
                )
                applied = {
                    int(row["version"])
                    for row in connection.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                }
                unsupported = sorted(
                    version for version in applied if version > SCHEMA_VERSION
                )
                if unsupported:
                    raise RuntimeError(
                        "remote database was created by a newer application "
                        f"(schema versions: {unsupported})"
                    )
                if 1 not in applied:
                    for statement in _MIGRATION_1:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (1, utc_now()),
                    )
                    applied.add(1)
                if 2 not in applied:
                    for statement in _MIGRATION_2:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (2, utc_now()),
                    )
                    applied.add(2)
                if 3 not in applied:
                    for statement in _MIGRATION_3:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (3, utc_now()),
                    )
                    applied.add(3)
                if 4 not in applied:
                    for statement in _MIGRATION_4:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (4, utc_now()),
                    )
                    applied.add(4)
                if 5 not in applied:
                    for statement in _MIGRATION_5:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (5, utc_now()),
                    )
                    applied.add(5)
                if 6 not in applied:
                    for statement in _MIGRATION_6:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (6, utc_now()),
                    )
                    applied.add(6)
                if 7 not in applied:
                    for statement in _MIGRATION_7:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (7, utc_now()),
                    )
                    applied.add(7)
                if 8 not in applied:
                    for statement in _MIGRATION_8:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (8, utc_now()),
                    )
                    applied.add(8)
                if 9 not in applied:
                    for statement in _MIGRATION_9:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (9, utc_now()),
                    )
                    applied.add(9)
                if 10 not in applied:
                    for statement in _MIGRATION_10:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (10, utc_now()),
                    )
                    applied.add(10)
                if 11 not in applied:
                    for statement in _MIGRATION_11:
                        connection.execute(statement)
                    connection.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (11, utc_now()))
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
                self._secure_file_modes()

    def _secure_file_modes(self) -> None:
        """Keep durable controller state private even under a permissive umask."""

        for candidate in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            if hasattr(os, "O_NONBLOCK"):
                flags |= os.O_NONBLOCK
            try:
                descriptor = os.open(candidate, flags)
            except FileNotFoundError:
                # SQLite removes WAL/SHM after the final connection closes.  A
                # concurrent close between directory lookup and open is normal.
                continue
            except OSError as exc:
                raise RuntimeError(
                    f"Relay database state path is unsafe: {candidate}"
                ) from exc
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise RuntimeError(
                        f"Relay database state path is unsafe: {candidate}"
                    )
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)

    @property
    def schema_version(self) -> int:
        with self._connection() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
            return int(row[0])

    # -- Servers ---------------------------------------------------------

    def create_server(self, server: Server) -> Server:
        with self._connection(write=True) as connection:
            self._insert_server(connection, server)
        return server

    def upsert_server(self, server: Server) -> Server:
        values = self._server_values(server)
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO servers (
                    id, target_kind, name, host, port, username, ssh_key_path, remote_root,
                    conda_env, poll_interval_seconds, default_max_parallel_runs,
                    min_free_bytes, max_load_ratio, enabled, last_probe_at,
                    last_probe_json, fingerprint, created_at, updated_at
                ) VALUES (
                    :id, :target_kind, :name, :host, :port, :username, :ssh_key_path,
                    :remote_root, :conda_env, :poll_interval_seconds,
                    :default_max_parallel_runs, :min_free_bytes, :max_load_ratio,
                    :enabled, :last_probe_at, :last_probe_json, :fingerprint,
                    :created_at, :updated_at
                )
                ON CONFLICT(id) DO UPDATE SET
                    target_kind = excluded.target_kind,
                    name = excluded.name,
                    host = excluded.host,
                    port = excluded.port,
                    username = excluded.username,
                    ssh_key_path = excluded.ssh_key_path,
                    remote_root = excluded.remote_root,
                    conda_env = excluded.conda_env,
                    poll_interval_seconds = excluded.poll_interval_seconds,
                    default_max_parallel_runs = excluded.default_max_parallel_runs,
                    min_free_bytes = excluded.min_free_bytes,
                    max_load_ratio = excluded.max_load_ratio,
                    enabled = excluded.enabled,
                    last_probe_at = excluded.last_probe_at,
                    last_probe_json = excluded.last_probe_json,
                    fingerprint = excluded.fingerprint,
                    updated_at = excluded.updated_at
                """,
                values,
            )
        return server

    def update_server(self, server: Server) -> Server:
        values = self._server_values(server)
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE servers SET
                    target_kind=:target_kind,
                    name=:name, host=:host, port=:port, username=:username,
                    ssh_key_path=:ssh_key_path, remote_root=:remote_root,
                    conda_env=:conda_env,
                    poll_interval_seconds=:poll_interval_seconds,
                    default_max_parallel_runs=:default_max_parallel_runs,
                    min_free_bytes=:min_free_bytes,
                    max_load_ratio=:max_load_ratio, enabled=:enabled,
                    last_probe_at=:last_probe_at,
                    last_probe_json=:last_probe_json, fingerprint=:fingerprint,
                    updated_at=:updated_at
                WHERE id=:id
                """,
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(f"server not found: {server.id}")
        return server

    def get_server(self, server_id: str) -> Server | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM servers WHERE id = ?", (server_id,)
            ).fetchone()
        return self._server_from_row(row) if row is not None else None

    def require_server(self, server_id: str) -> Server:
        server = self.get_server(server_id)
        if server is None:
            raise KeyError(f"server not found: {server_id}")
        return server

    def list_servers(self, *, enabled_only: bool = False) -> list[Server]:
        query = "SELECT * FROM servers"
        params: tuple[object, ...] = ()
        if enabled_only:
            query += " WHERE enabled = ?"
            params = (1,)
        query += " ORDER BY name COLLATE NOCASE, id"
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._server_from_row(row) for row in rows]

    def delete_server(self, server_id: str) -> bool:
        """Delete an unused server; SQLite rejects deletion if runs reference it."""

        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM servers WHERE id = ?", (server_id,)
            )
            return cursor.rowcount == 1

    def record_server_probe(
        self,
        server_id: str,
        probe: Mapping[str, Any],
        *,
        fingerprint: str | None = None,
        probed_at: str | None = None,
        expected_target: Server | None = None,
    ) -> Server:
        timestamp = probed_at or utc_now()
        effective_fingerprint = fingerprint
        if effective_fingerprint is None:
            candidate = probe.get("fingerprint")
            if candidate is None and isinstance(probe.get("report"), Mapping):
                candidate = probe["report"].get("fingerprint")
            if candidate is not None and str(candidate).strip():
                effective_fingerprint = str(candidate).strip()
        with self._connection(write=True) as connection:
            target_clause = ""
            target_params: tuple[object, ...] = ()
            if expected_target is not None:
                target_clause = (
                    " AND host = ? AND port = ? AND username = ?"
                    " AND ssh_key_path = ? AND remote_root = ? AND conda_env = ?"
                )
                target_params = (
                    expected_target.host,
                    expected_target.port,
                    expected_target.username,
                    expected_target.ssh_key_path,
                    expected_target.remote_root,
                    expected_target.conda_env,
                )
            cursor = connection.execute(
                """
                UPDATE servers SET
                    last_probe_at = ?, last_probe_json = ?,
                    fingerprint = COALESCE(?, fingerprint), updated_at = ?
                WHERE id = ?
                """
                + target_clause,
                (
                    timestamp,
                    stable_json_dumps(probe),
                    effective_fingerprint,
                    timestamp,
                    server_id,
                    *target_params,
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM servers WHERE id = ?", (server_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(f"server not found: {server_id}")
                raise ServerTargetChangedError(
                    f"server target changed while probe was running: {server_id}"
                )
            row = connection.execute(
                "SELECT * FROM servers WHERE id = ?", (server_id,)
            ).fetchone()
        return self._server_from_row(row)

    # -- Runs ------------------------------------------------------------

    def create_run(self, run: Run) -> Run:
        with self._connection(write=True) as connection:
            self._insert_run(connection, run)
        return run

    def update_run(self, run: Run) -> Run:
        values = self._run_values(run)
        with self._connection(write=True) as connection:
            self._update_run_record(connection, values)
        return run

    # ``replace_run`` is explicit for callers that prefer repository naming.
    replace_run = update_run

    def update_run_fields(self, run_id: str, **changes: Any) -> Run:
        """Atomically update selected Run fields through domain validation."""

        forbidden = {"id", "server_id", "created_at"}.intersection(changes)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"immutable run fields cannot be updated: {names}")
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"run not found: {run_id}")
            updated = self._run_from_row(row).with_updates(**changes)
            self._update_run_record(connection, self._run_values(updated))
        return updated

    def rename_run(self, run_id: str, task_name: str) -> Run:
        """Atomically update a display label and append its audit event."""

        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"run not found: {run_id}")
            current = self._run_from_row(row)
            if current.task_name == task_name:
                return current
            updated = current.with_updates(task_name=task_name)
            self._update_run_record(connection, self._run_values(updated))
            self._insert_event(
                connection,
                RunEvent(
                    run_id=run_id,
                    kind="audit",
                    message=(
                        f"Run display name changed from {current.task_name!r} "
                        f"to {task_name!r}."
                    ),
                    data={
                        "stream": "audit",
                        "phase": "metadata",
                        "actor": "local",
                        "field": "display_name",
                        "previous": current.task_name,
                        "current": task_name,
                    },
                ),
            )
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row)

    def set_queued_run_parallelism(self, run_id: str, max_parallel_runs: int) -> bool:
        """Change the worker count of a run that has not started; return whether it changed.

        The controller reads ``max_parallel_runs`` from the run record when it starts the run, so a queued run picks
        up the new value. A run that already left the queue keeps its workers: the check and the write share one
        transaction, so a run cannot start between them.
        """

        if type(max_parallel_runs) is not int or not 1 <= max_parallel_runs <= 256:
            raise ValueError("max_parallel_runs must be an integer from 1 to 256")
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"run not found: {run_id}")
            current = self._run_from_row(row)
            if current.status is not RunStatus.QUEUED or current.max_parallel_runs == max_parallel_runs:
                return False
            updated = current.with_updates(max_parallel_runs=max_parallel_runs)
            self._update_run_record(connection, self._run_values(updated))
            self._insert_event(
                connection,
                RunEvent(
                    run_id=run_id,
                    kind="audit",
                    message=(
                        f"Parallel runs of the queued run changed from {current.max_parallel_runs} "
                        f"to {max_parallel_runs}."
                    ),
                    data={
                        "stream": "audit",
                        "phase": "metadata",
                        "actor": "local",
                        "field": "max_parallel_runs",
                        "previous": current.max_parallel_runs,
                        "current": max_parallel_runs,
                    },
                ),
            )
        return True

    def get_run(self, run_id: str) -> Run | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row) if row is not None else None

    def require_run(self, run_id: str) -> Run:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"run not found: {run_id}")
        return run

    def list_runs(
        self,
        *,
        server_id: str | None = None,
        statuses: Sequence[RunStatus | str] | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
        summary: bool = False,
    ) -> list[Run]:
        """Runs newest first.

        ``summary=True`` is for list views that only serialize a run: its task payload keeps only the git
        commit, the server snapshot and the execution record, and its snapshot manifest is empty. Such records
        must never be written back.
        """
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must not be negative")
        clauses: list[str] = []
        params: list[object] = []
        if server_id is not None:
            clauses.append("server_id = ?")
            params.append(server_id)
        if statuses is not None:
            normalized = tuple(RunStatus(status).value for status in statuses)
            if not normalized:
                return []
            clauses.append("status IN (" + ",".join("?" for _ in normalized) + ")")
            params.extend(normalized)
        query = " FROM runs"
        if search:
            # instr treats percent/underscore literally; search covers the full
            # ledger before pagination, including saved and current device names.
            clauses.append("instr(lower(id || ' ' || task_name || ' ' || coalesce(task_path,'') || ' ' || coalesce(error_code,'') || ' ' || coalesce(json_extract(task_payload_json,'$.server_snapshot.name'),'') || ' ' || coalesce((SELECT name FROM servers WHERE servers.id=runs.server_id),'')), lower(?)) > 0")
            params.append(search.strip())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend((limit, offset))
        with self._connection() as connection:
            columns = self._summary_columns(connection) if summary else "*"
            rows = connection.execute("SELECT " + columns + query, params).fetchall()
        return [self._run_from_row(row) for row in rows]

    # A run's task payload holds the inline task and the source provenance, hundreds of kB per run; decoding
    # them in Python dominated every ledger and dashboard poll. SQLite extracts the few parts a list needs.
    _SUMMARY_TASK_PAYLOAD = (
        "json_object("
        "'provenance', json_object('git_commit', json_extract(task_payload_json, '$.provenance.git_commit')), "
        "'server_snapshot', json_extract(task_payload_json, '$.server_snapshot'), "
        "'execution', json_extract(task_payload_json, '$.execution'))"
    )

    def _summary_columns(self, connection: sqlite3.Connection) -> str:
        if self._summary_run_columns is None:
            replaced = {
                "task_payload_json": f"{self._SUMMARY_TASK_PAYLOAD} AS task_payload_json",
                "snapshot_manifest_json": "'{}' AS snapshot_manifest_json",
            }
            names = [row[1] for row in connection.execute("PRAGMA table_info(runs)")]
            self._summary_run_columns = ", ".join(replaced.get(name, name) for name in names)
        return self._summary_run_columns

    def list_runs_requiring_recovery(self) -> list[Run]:
        """Return every non-terminal record left behind at process startup."""

        statuses = tuple(
            status.value
            for status in sorted(NON_TERMINAL_RUN_STATUSES, key=str)
        )
        placeholders = ",".join("?" for _ in statuses)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM runs
                WHERE status IN ({placeholders})
                ORDER BY updated_at, id
                """,
                statuses,
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    # A short alias reads naturally in startup code.
    startup_recovery_runs = list_runs_requiring_recovery

    def set_run_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        exit_code: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error_details: Mapping[str, Any] | None = None,
        event_message: str | None = None,
        validate_transition: bool = True,
    ) -> Run:
        target = RunStatus(status)
        now = utc_now()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"run not found: {run_id}")
            current = RunStatus(row["status"])
            if validate_transition:
                validate_run_transition(current, target)
            started_at = row["started_at"]
            if started_at is None and target not in {
                RunStatus.QUEUED,
                RunStatus.CANCELED,
            }:
                started_at = now
            finished_at = row["finished_at"]
            if target in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELED,
                RunStatus.INTERRUPTED,
            }:
                finished_at = finished_at or now
            connection.execute(
                """
                UPDATE runs SET
                    status=?, started_at=?, finished_at=?, exit_code=?,
                    error_code=?, error_message=?, error_details_json=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    target.value,
                    started_at,
                    finished_at,
                    exit_code if exit_code is not None else row["exit_code"],
                    error_code if error_code is not None else row["error_code"],
                    error_message
                    if error_message is not None
                    else row["error_message"],
                    stable_json_dumps(error_details)
                    if error_details is not None
                    else row["error_details_json"],
                    now,
                    run_id,
                ),
            )
            if current != target or event_message is not None:
                self._insert_event(
                    connection,
                    RunEvent(
                        run_id=run_id,
                        level=EventLevel.ERROR
                        if target is RunStatus.FAILED
                        else EventLevel.INFO,
                        kind="status",
                        message=event_message
                        or f"Run status changed: {current.value} -> {target.value}",
                        data={"from": current.value, "to": target.value},
                    ),
                )
            updated = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(updated)

    def set_run_stage(
        self,
        run_id: str,
        stage: Mapping[str, Any],
        message: str | None = None,
        level: EventLevel | str = EventLevel.INFO,
    ) -> Run:
        """Persist one structured execution stage and its audit event atomically."""

        if not isinstance(stage, Mapping):
            raise ValueError("stage must be a JSON object")
        try:
            normalized_stage = stable_json_loads(stable_json_dumps(stage), {})
        except (TypeError, ValueError) as exc:
            raise ValueError("stage must be a JSON object") from exc
        if not isinstance(normalized_stage, dict):
            raise ValueError("stage must be a JSON object")

        now = utc_now()
        normalized_stage["schema_version"] = 1
        normalized_stage["updated_at"] = now
        phase = normalized_stage.get("phase")
        event_message = (
            message
            if message is not None
            else (
                f"Run stage updated: {phase}."
                if phase is not None
                else "Run stage updated."
            )
        )
        encoded_stage = stable_json_dumps(normalized_stage)

        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET stage_json=?, progress_message=?,
                    heartbeat_at=?, updated_at=? WHERE id=?
                """,
                (encoded_stage, event_message, now, now, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"run not found: {run_id}")
            self._insert_event(
                connection,
                RunEvent(
                    run_id=run_id,
                    level=EventLevel(level),
                    kind="stage",
                    message=event_message,
                    data={
                        "stream": "system",
                        "phase": phase,
                        "stage": normalized_stage,
                    },
                    created_at=now,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(updated)

    def update_run_progress(
        self,
        run_id: str,
        *,
        current: float,
        total: float | None = None,
        unit: str | None = None,
        message: str | None = None,
        heartbeat_at: str | None = None,
    ) -> Run:
        if current < 0:
            raise ValueError("current must not be negative")
        if total is not None and total < 0:
            raise ValueError("total must not be negative")
        now = utc_now()
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET
                    progress_current=?, progress_total=?, progress_unit=?,
                    progress_message=?, heartbeat_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    current,
                    total,
                    unit,
                    message,
                    heartbeat_at or now,
                    now,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"run not found: {run_id}")
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row)

    def set_run_remote_process(
        self,
        run_id: str,
        *,
        remote_pid: int,
        remote_run_dir: str,
        worker_id: str | None = None,
    ) -> Run:
        if remote_pid <= 0:
            raise ValueError("remote_pid must be positive")
        now = utc_now()
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET remote_pid=?, remote_run_dir=?, worker_id=?,
                    heartbeat_at=?, updated_at=? WHERE id=?
                """,
                (remote_pid, remote_run_dir, worker_id, now, now, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"run not found: {run_id}")
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row)

    def delete_run(self, run_id: str) -> bool:
        """Delete the DB record and child rows; filesystem deletion is a service job."""

        with self._connection(write=True) as connection:
            cursor = connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            return cursor.rowcount == 1

    # -- Events ----------------------------------------------------------

    def append_run_event(self, event: RunEvent) -> RunEvent:
        with self._connection(write=True) as connection:
            return self._insert_event(connection, event)

    def list_run_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 1_000,
    ) -> list[RunEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        if limit < 1 or limit > 100_000:
            raise ValueError("limit must be between 1 and 100000")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM run_events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (run_id, after_sequence, limit),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def latest_audit_events(self, *, limit: int = 50) -> list[RunEvent]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM run_events WHERE kind IN ('audit','status','cancel','cleanup','probe') ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [self._event_from_row(row) for row in rows]

    def run_metadata(self, run_ids: Sequence[str]) -> tuple[dict, dict]:
        artifacts: dict[str, list[Artifact]] = {}
        imports: dict[str, dict] = {}
        with self._connection() as connection:
            for start in range(0, len(run_ids), 500):
                ids = tuple(run_ids[start:start + 500])
                placeholders = ",".join("?" for _ in ids)
                for row in connection.execute(f"SELECT * FROM artifacts WHERE run_id IN ({placeholders}) ORDER BY relative_path", ids):
                    artifact = self._artifact_from_row(row)
                    artifacts.setdefault(artifact.run_id, []).append(artifact)
                for row in connection.execute(f"SELECT * FROM result_imports WHERE source_run_id IN ({placeholders}) ORDER BY imported_at DESC, id DESC", ids):
                    imports.setdefault(row["source_run_id"], self._result_import_from_row(row))
        return artifacts, imports

    def transferred_bytes(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT coalesce(sum(size_bytes),0) FROM artifacts WHERE status IN ('transferred','verified')").fetchone()[0])

    # -- Resource samples ------------------------------------------------

    def add_resource_sample(self, sample: ResourceSample) -> ResourceSample:
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                INSERT INTO resource_samples (
                    server_id, run_id, target_identity, sampled_at, reachable,
                    cpu_physical_cores, cpu_logical_cores, cpu_percent,
                    load_1m, load_5m, load_15m, load_ratio,
                    memory_total_bytes, memory_available_bytes,
                    disk_total_bytes, disk_free_bytes, gpu_usable, gpu_error,
                    gpu_devices_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.server_id,
                    sample.run_id,
                    sample.target_identity,
                    sample.sampled_at,
                    int(sample.reachable),
                    sample.cpu_physical_cores,
                    sample.cpu_logical_cores,
                    sample.cpu_percent,
                    sample.load_1m,
                    sample.load_5m,
                    sample.load_15m,
                    sample.load_ratio,
                    sample.memory_total_bytes,
                    sample.memory_available_bytes,
                    sample.disk_total_bytes,
                    sample.disk_free_bytes,
                    None if sample.gpu_usable is None else int(sample.gpu_usable),
                    sample.gpu_error,
                    stable_json_dumps(sample.gpu_devices),
                    sample.error,
                ),
            )
            sample_id = int(cursor.lastrowid)
        return replace(sample, id=sample_id)

    def list_resource_samples(
        self,
        server_id: str,
        *,
        run_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        after_id: int | None = None,
        background_only: bool = False,
        target_identity: str | None = None,
        limit: int = 300,
    ) -> list[ResourceSample]:
        if limit < 1 or limit > 100_000:
            raise ValueError("limit must be between 1 and 100000")
        clauses = ["server_id = ?"]
        params: list[object] = [server_id]
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        elif background_only:
            clauses.append("run_id IS NULL")
        if target_identity is not None:
            clauses.append("target_identity = ?")
            params.append(target_identity)
        if since is not None:
            clauses.append("sampled_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("sampled_at <= ?")
            params.append(until)
        if after_id is not None:
            if after_id < 0:
                raise ValueError("after_id must not be negative")
            clauses.append("id > ?")
            params.append(after_id)
        params.append(limit)
        order = "id ASC" if after_id is not None else "sampled_at DESC, id DESC"
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM resource_samples WHERE "
                + " AND ".join(clauses)
                + f" ORDER BY {order} LIMIT ?",
                params,
            ).fetchall()
        return [self._sample_from_row(row) for row in rows]

    def latest_resource_sample(self, server_id: str) -> ResourceSample | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM resource_samples WHERE server_id = ?
                ORDER BY sampled_at DESC, id DESC LIMIT 1
                """,
                (server_id,),
            ).fetchone()
        return self._sample_from_row(row) if row is not None else None

    def latest_background_resource_sample(
        self,
        server_id: str,
        *,
        reachable: bool | None = None,
        target_identity: str | None = None,
    ) -> ResourceSample | None:
        clauses = ["server_id = ?", "run_id IS NULL"]
        params: list[object] = [server_id]
        if reachable is not None:
            clauses.append("reachable = ?")
            params.append(int(reachable))
        if target_identity is not None:
            clauses.append("target_identity = ?")
            params.append(target_identity)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM resource_samples WHERE "
                + " AND ".join(clauses)
                + " ORDER BY sampled_at DESC, id DESC LIMIT 1",
                params,
            ).fetchone()
        return self._sample_from_row(row) if row is not None else None

    def latest_background_resource_samples(
        self,
        *,
        reachable: bool | None = None,
        target_identities: Mapping[str, str] | None = None,
    ) -> dict[str, ResourceSample]:
        clauses = ["run_id IS NULL"]
        params: list[object] = []
        if reachable is not None:
            clauses.append("reachable = ?")
            params.append(int(reachable))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT resource_samples.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY server_id, target_identity
                               ORDER BY sampled_at DESC, id DESC
                           ) AS resource_rank
                    FROM resource_samples
                    WHERE """
                + " AND ".join(clauses)
                + """
                ) WHERE resource_rank = 1
                ORDER BY server_id
                """,
                params,
            ).fetchall()
        result: dict[str, ResourceSample] = {}
        for row in rows:
            server_id = str(row["server_id"])
            if target_identities is not None and row["target_identity"] != target_identities.get(server_id):
                continue
            sample = self._sample_from_row(row)
            current = result.get(server_id)
            if current is None or (sample.sampled_at, sample.id or 0) > (
                current.sampled_at,
                current.id or 0,
            ):
                result[server_id] = sample
        return result

    def latest_background_resource_sample_sets(
        self,
        target_identities: Mapping[str, str],
    ) -> tuple[
        dict[str, ResourceSample],
        dict[str, ResourceSample],
        dict[str, ResourceSample],
    ]:
        """Read at most two samples per target, regardless of history length.

        The latest attempt is the newer of the latest success and failure.
        Seek those two IDs using the reachability index before loading sample
        payloads. Ranking every historical row also sorts its GPU JSON and can
        spill hundreds of MB to disk on each dashboard poll.
        """

        if not target_identities:
            return {}, {}, {}
        params: list[object] = []
        for server_id, target_identity in target_identities.items():
            params.extend((server_id, target_identity))
        targets = ", ".join("(?, ?)" for _ in target_identities)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                WITH targets(server_id, target_identity) AS (VALUES {targets}),
                     outcomes(reachable) AS (VALUES (0), (1))
                SELECT * FROM resource_samples
                WHERE id IN (
                    SELECT (
                        SELECT id FROM resource_samples AS history
                        WHERE history.server_id = targets.server_id
                          AND history.target_identity = targets.target_identity
                          AND history.run_id IS NULL
                          AND history.reachable = outcomes.reachable
                        ORDER BY history.sampled_at DESC, history.id DESC
                        LIMIT 1
                    )
                    FROM targets CROSS JOIN outcomes
                )
                """,
                params,
            ).fetchall()
        attempts: dict[str, ResourceSample] = {}
        successes: dict[str, ResourceSample] = {}
        errors: dict[str, ResourceSample] = {}
        for row in rows:
            sample = self._sample_from_row(row)
            server_id = sample.server_id
            current = attempts.get(server_id)
            if current is None or (sample.sampled_at, sample.id or 0) > (
                current.sampled_at, current.id or 0
            ):
                attempts[server_id] = sample
            (successes if sample.reachable else errors)[server_id] = sample
        return attempts, successes, errors

    def prune_background_resource_samples(
        self,
        *,
        before: str,
        max_samples_per_server: int = 100_000,
        batch_limit: int = 10_000,
    ) -> int:
        """Delete only general monitor history, never run-bound evidence."""

        if max_samples_per_server < 1:
            raise ValueError("max_samples_per_server must be at least 1")
        removed = 0
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                DELETE FROM resource_samples WHERE id IN (
                    SELECT id FROM resource_samples
                    WHERE run_id IS NULL AND sampled_at < ? ORDER BY sampled_at, id LIMIT ?
                )
                """,
                (before, batch_limit),
            )
            removed += max(0, int(cursor.rowcount))
            server_ids = [row[0] for row in connection.execute("SELECT id FROM servers")]
            for server_id in server_ids:
                if removed >= batch_limit:
                    break
                cursor = connection.execute("DELETE FROM resource_samples WHERE id IN (SELECT id FROM resource_samples WHERE server_id=? AND run_id IS NULL ORDER BY id DESC LIMIT ? OFFSET ?)", (server_id, batch_limit - removed, max_samples_per_server))
                removed += max(0, cursor.rowcount)
        return removed

    def aggregate_background_resource_samples(self, *, before: str, batch_limit: int = 10_000) -> int:
        """Compact a bounded batch of old raw samples; IDs only disappear.

        Retaining the latest timestamp/ID per identity, status and minute keeps
        latest-status queries correct. No new IDs are inserted, so incremental
        readers never receive historical rollups masquerading as new samples.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        removed = 0
        with self._connection(write=True) as connection:
            rows = connection.execute("SELECT * FROM resource_samples WHERE run_id IS NULL AND aggregation_json = '{}' AND sampled_at < ? ORDER BY sampled_at, id LIMIT ?", (before, max(1, min(batch_limit, 10_000)))).fetchall()
            for row in rows:
                groups[(row["server_id"], row["target_identity"], row["reachable"], row["sampled_at"][:16])].append(row)
            for (server, identity, reachable, minute), group in groups.items():
                previous = connection.execute("SELECT * FROM resource_samples WHERE server_id=? AND target_identity IS ? AND reachable=? AND run_id IS NULL AND sampled_at>=? AND sampled_at<? AND aggregation_json <> '{}'", (server, identity, reachable, minute + ':00', minute + ':60')).fetchall()
                group.extend(previous)
                representative, aggregate = merge_minute(group)
                assignments = ",".join(f"{field}=?" for field in ROLLUP_FIELDS)
                connection.execute(f"UPDATE resource_samples SET {assignments},aggregation_json=? WHERE id=?", (*[representative[field] for field in ROLLUP_FIELDS], stable_json_dumps(aggregate), representative["id"]))
                obsolete = [row["id"] for row in group if row["id"] != representative["id"]]
                for start in range(0, len(obsolete), 500):
                    ids = obsolete[start:start + 500]
                    connection.execute(f"DELETE FROM resource_samples WHERE id IN ({','.join('?' for _ in ids)})", ids)
                removed += len(obsolete)
        metrics.increment("resource_rollup.removed", removed)
        return removed

    # -- Artifacts -------------------------------------------------------

    def upsert_artifact(self, artifact: Artifact) -> Artifact:
        values = self._artifact_values(artifact)
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    id, run_id, relative_path, status, remote_path, local_path,
                    size_bytes, sha256, transferred_at, verified_at,
                    verification_error, metadata_json, created_at, updated_at
                ) VALUES (
                    :id, :run_id, :relative_path, :status, :remote_path,
                    :local_path, :size_bytes, :sha256, :transferred_at,
                    :verified_at, :verification_error, :metadata_json,
                    :created_at, :updated_at
                )
                ON CONFLICT(run_id, relative_path) DO UPDATE SET
                    status=excluded.status, remote_path=excluded.remote_path,
                    local_path=excluded.local_path, size_bytes=excluded.size_bytes,
                    sha256=excluded.sha256,
                    transferred_at=excluded.transferred_at,
                    verified_at=excluded.verified_at,
                    verification_error=excluded.verification_error,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND relative_path = ?",
                (artifact.run_id, artifact.relative_path),
            ).fetchone()
        return self._artifact_from_row(row)

    create_artifact = upsert_artifact
    update_artifact = upsert_artifact

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        return self._artifact_from_row(row) if row is not None else None

    def list_artifacts(self, run_id: str) -> list[Artifact]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM artifacts WHERE run_id = ?
                ORDER BY relative_path
                """,
                (run_id,),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def delete_artifact(self, artifact_id: str) -> bool:
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM artifacts WHERE id = ?", (artifact_id,)
            )
            return cursor.rowcount == 1

    # -- Imported result library ---------------------------------------

    def create_result_import(self, item: Mapping[str, Any]) -> dict[str, Any]:
        values = self._result_import_values(item)
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO result_imports (
                    id, display_name, source_kind, source_run_id,
                    source_metadata_json, bundle_id, managed_relative_dir,
                    validation_json, summary_size_bytes, summary_sha256,
                    trend_size_bytes, trend_sha256, imported_at
                ) VALUES (
                    :id, :display_name, :source_kind, :source_run_id,
                    :source_metadata_json, :bundle_id, :managed_relative_dir,
                    :validation_json, :summary_size_bytes, :summary_sha256,
                    :trend_size_bytes, :trend_sha256, :imported_at
                )
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM result_imports WHERE id = ?", (values["id"],)
            ).fetchone()
        return self._result_import_from_row(row)

    def get_result_import(self, result_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM result_imports WHERE id = ?", (str(result_id),)
            ).fetchone()
        return self._result_import_from_row(row) if row is not None else None

    def list_result_imports(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM result_imports
                ORDER BY imported_at DESC, display_name COLLATE NOCASE, id DESC
                """
            ).fetchall()
        return [self._result_import_from_row(row) for row in rows]

    def find_result_import_by_source_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM result_imports
                WHERE source_run_id = ?
                ORDER BY imported_at DESC, id DESC
                LIMIT 1
                """,
                (str(run_id),),
            ).fetchone()
        return self._result_import_from_row(row) if row is not None else None

    def find_result_import_by_content(
        self,
        summary_sha256: str,
        trend_sha256: str,
    ) -> dict[str, Any] | None:
        """Return the newest result whose two immutable databases are identical."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM result_imports
                WHERE summary_sha256 = ? AND trend_sha256 = ?
                ORDER BY imported_at DESC, id DESC
                LIMIT 1
                """,
                (str(summary_sha256), str(trend_sha256)),
            ).fetchone()
        return self._result_import_from_row(row) if row is not None else None

    def update_result_import(self, result_id: str, **changes: Any) -> dict[str, Any]:
        current = self.get_result_import(result_id)
        if current is None:
            raise KeyError(f"result import not found: {result_id}")
        updated = {**current, **changes, "id": str(result_id)}
        values = self._result_import_values(updated)
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE result_imports SET
                    display_name=:display_name, source_kind=:source_kind,
                    source_run_id=:source_run_id,
                    source_metadata_json=:source_metadata_json,
                    bundle_id=:bundle_id,
                    managed_relative_dir=:managed_relative_dir,
                    validation_json=:validation_json,
                    summary_size_bytes=:summary_size_bytes,
                    summary_sha256=:summary_sha256,
                    trend_size_bytes=:trend_size_bytes,
                    trend_sha256=:trend_sha256,
                    imported_at=:imported_at
                WHERE id=:id
                """,
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(f"result import not found: {result_id}")
            row = connection.execute(
                "SELECT * FROM result_imports WHERE id = ?", (str(result_id),)
            ).fetchone()
        return self._result_import_from_row(row)

    def delete_result_import(self, result_id: str) -> bool:
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM result_imports WHERE id = ?", (str(result_id),)
            )
            return cursor.rowcount == 1

    def list_result_import_cleanup_queue(self) -> list[dict[str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT result_id, managed_relative_dir, reason, queued_at
                FROM result_import_cleanup_queue
                ORDER BY queued_at, result_id
                """
            ).fetchall()
        return [
            {
                "result_id": str(row["result_id"]),
                "managed_relative_dir": str(row["managed_relative_dir"]),
                "reason": str(row["reason"]),
                "queued_at": str(row["queued_at"]),
            }
            for row in rows
        ]

    def complete_result_import_cleanup(self, result_id: str) -> bool:
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM result_import_cleanup_queue WHERE result_id = ?",
                (str(result_id),),
            )
            return cursor.rowcount == 1

    # -- Statistical-analysis workspace --------------------------------

    def upsert_sa_import(self, item: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "id",
            "result_id",
            "runset_uid",
            "bundle_id",
            "task_name",
            "label",
            "algorithm_name",
            "problem_name",
            "problem_params",
            "algorithm_params",
            "scalar_metrics",
            "trend_metrics",
            "expected_runs",
            "completed_runs",
            "source_fingerprint",
            "pairing_signature",
            "snapshot_sha256",
            "task_sha256",
            "imported_at",
        }
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"missing SA import field(s): {', '.join(missing)}")
        values = {
            "id": str(item["id"]),
            "result_id": str(item["result_id"]),
            "runset_uid": str(item["runset_uid"]),
            "bundle_id": str(item["bundle_id"]),
            "task_name": str(item["task_name"]),
            "label": str(item["label"]),
            "algorithm_name": str(item["algorithm_name"]),
            "problem_name": str(item["problem_name"]),
            "problem_params_json": stable_json_dumps(item["problem_params"]),
            "algorithm_params_json": stable_json_dumps(item["algorithm_params"]),
            "scalar_metrics_json": stable_json_dumps(item["scalar_metrics"]),
            "trend_metrics_json": stable_json_dumps(item["trend_metrics"]),
            "expected_runs": int(item["expected_runs"]),
            "completed_runs": int(item["completed_runs"]),
            "source_fingerprint": str(item["source_fingerprint"]),
            "pairing_signature": str(item["pairing_signature"]),
            "snapshot_sha256": str(item["snapshot_sha256"]),
            "task_sha256": str(item["task_sha256"]),
            "imported_at": str(item["imported_at"]),
        }
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO sa_imports (
                    id, result_id, runset_uid, bundle_id, task_name, label,
                    algorithm_name, problem_name, problem_params_json,
                    algorithm_params_json, scalar_metrics_json,
                    trend_metrics_json, expected_runs, completed_runs,
                    source_fingerprint, pairing_signature, snapshot_sha256,
                    task_sha256, imported_at
                ) VALUES (
                    :id, :result_id, :runset_uid, :bundle_id, :task_name, :label,
                    :algorithm_name, :problem_name, :problem_params_json,
                    :algorithm_params_json, :scalar_metrics_json,
                    :trend_metrics_json, :expected_runs, :completed_runs,
                    :source_fingerprint, :pairing_signature, :snapshot_sha256,
                    :task_sha256, :imported_at
                )
                ON CONFLICT(result_id, runset_uid) DO UPDATE SET
                    bundle_id=excluded.bundle_id,
                    task_name=excluded.task_name,
                    label=excluded.label,
                    algorithm_name=excluded.algorithm_name,
                    problem_name=excluded.problem_name,
                    problem_params_json=excluded.problem_params_json,
                    algorithm_params_json=excluded.algorithm_params_json,
                    scalar_metrics_json=excluded.scalar_metrics_json,
                    trend_metrics_json=excluded.trend_metrics_json,
                    expected_runs=excluded.expected_runs,
                    completed_runs=excluded.completed_runs,
                    source_fingerprint=excluded.source_fingerprint,
                    pairing_signature=excluded.pairing_signature,
                    snapshot_sha256=excluded.snapshot_sha256,
                    task_sha256=excluded.task_sha256
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM sa_imports WHERE result_id = ? AND runset_uid = ?",
                (values["result_id"], values["runset_uid"]),
            ).fetchone()
        return self._sa_import_from_row(row)

    def list_sa_imports(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sa_imports
                ORDER BY imported_at ASC, rowid ASC
                """
            ).fetchall()
        return [self._sa_import_from_row(row) for row in rows]

    def get_sa_imports(self, import_ids: Sequence[str]) -> list[dict[str, Any]]:
        normalized = [str(item) for item in import_ids]
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM sa_imports WHERE id IN ({placeholders})",
                tuple(normalized),
            ).fetchall()
        by_id = {str(row["id"]): self._sa_import_from_row(row) for row in rows}
        missing = [item for item in normalized if item not in by_id]
        if missing:
            raise KeyError(f"SA import not found: {', '.join(missing)}")
        return [by_id[item] for item in normalized]

    def delete_sa_import(self, import_id: str) -> bool:
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM sa_imports WHERE id = ?", (str(import_id),)
            )
            return cursor.rowcount == 1

    def clear_sa_imports(self) -> int:
        with self._connection(write=True) as connection:
            cursor = connection.execute("DELETE FROM sa_imports")
            return max(0, int(cursor.rowcount))

    def create_sa_analysis(self, item: Mapping[str, Any]) -> dict[str, Any]:
        values = self._sa_analysis_values(item)
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO sa_analyses (
                    id, name, status, method, alpha, metric_names_json,
                    source_imports_json, result_json, output_dir,
                    output_files_json, error_message, created_at, completed_at
                ) VALUES (
                    :id, :name, :status, :method, :alpha,
                    :metric_names_json, :source_imports_json, :result_json,
                    :output_dir, :output_files_json, :error_message,
                    :created_at, :completed_at
                )
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM sa_analyses WHERE id = ?", (values["id"],)
            ).fetchone()
        return self._sa_analysis_from_row(row)

    def update_sa_analysis(self, analysis_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "name",
            "status",
            "method",
            "alpha",
            "metric_names",
            "source_imports",
            "result",
            "output_dir",
            "output_files",
            "error_message",
            "completed_at",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"unknown SA analysis field(s): {', '.join(unknown)}")
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM sa_analyses WHERE id = ?", (str(analysis_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"SA analysis not found: {analysis_id}")
            current = self._sa_analysis_from_row(row)
            current.update(changes)
            values = self._sa_analysis_values(current)
            connection.execute(
                """
                UPDATE sa_analyses SET
                    name=:name, status=:status, method=:method, alpha=:alpha,
                    metric_names_json=:metric_names_json,
                    source_imports_json=:source_imports_json,
                    result_json=:result_json, output_dir=:output_dir,
                    output_files_json=:output_files_json,
                    error_message=:error_message, completed_at=:completed_at
                WHERE id=:id
                """,
                values,
            )
            updated = connection.execute(
                "SELECT * FROM sa_analyses WHERE id = ?", (str(analysis_id),)
            ).fetchone()
        return self._sa_analysis_from_row(updated)

    def get_sa_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sa_analyses WHERE id = ?", (str(analysis_id),)
            ).fetchone()
        return self._sa_analysis_from_row(row) if row is not None else None

    def list_sa_analyses(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM sa_analyses ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._sa_analysis_from_row(row) for row in rows]

    def delete_sa_analysis(self, analysis_id: str) -> bool:
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "DELETE FROM sa_analyses WHERE id = ?", (str(analysis_id),)
            )
            return cursor.rowcount == 1

    # -- Settings/dashboard ---------------------------------------------

    def set_setting(self, key: str, value: Any) -> None:
        if not key.strip():
            raise ValueError("setting key must not be empty")
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, stable_json_dumps(value), utc_now()),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        return stable_json_loads(row["value_json"])

    def get_or_create_setting(self, key: str, value: Any) -> Any:
        """Atomically return an existing setting or persist the supplied value."""

        if not key.strip():
            raise ValueError("setting key must not be empty")
        encoded = stable_json_dumps(value)
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (key, encoded, utc_now()),
            )
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"failed to initialize setting: {key}")
        return stable_json_loads(row["value_json"])

    def list_settings(self) -> dict[str, Any]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT key, value_json FROM settings ORDER BY key"
            ).fetchall()
        return {
            str(row["key"]): stable_json_loads(row["value_json"]) for row in rows
        }

    def delete_setting(self, key: str) -> bool:
        with self._connection(write=True) as connection:
            cursor = connection.execute("DELETE FROM settings WHERE key = ?", (key,))
            return cursor.rowcount == 1

    def dashboard_counts(self) -> DashboardCounts:
        with self._connection() as connection:
            server_row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END), 0)
                           AS enabled
                FROM servers
                """
            ).fetchone()
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM runs GROUP BY status"
            ).fetchall()
        status_counts = {status.value: 0 for status in RunStatus}
        status_counts.update(
            {str(row["status"]): int(row["count"]) for row in status_rows}
        )
        active = sum(
            status_counts[status.value] for status in NON_TERMINAL_RUN_STATUSES
        )
        return DashboardCounts(
            servers_total=int(server_row["total"]),
            servers_enabled=int(server_row["enabled"]),
            runs_total=sum(status_counts.values()),
            active_runs=active,
            status_counts=status_counts,
        )

    # -- Internal row/record helpers ------------------------------------

    @staticmethod
    def _result_import_values(item: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "id",
            "display_name",
            "source_kind",
            "bundle_id",
            "managed_relative_dir",
            "validation",
            "summary_size_bytes",
            "summary_sha256",
            "trend_size_bytes",
            "trend_sha256",
            "imported_at",
        }
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(
                f"missing result import field(s): {', '.join(missing)}"
            )
        source_kind = str(item["source_kind"])
        if source_kind not in {"external", "run", "legacy_run"}:
            raise ValueError(f"invalid result import source kind: {source_kind}")
        return {
            "id": str(item["id"]),
            "display_name": str(item["display_name"]),
            "source_kind": source_kind,
            "source_run_id": (
                str(item["source_run_id"])
                if item.get("source_run_id") is not None
                else None
            ),
            "source_metadata_json": stable_json_dumps(
                item.get("source_metadata", {})
            ),
            "bundle_id": str(item["bundle_id"]),
            "managed_relative_dir": str(item["managed_relative_dir"]),
            "validation_json": stable_json_dumps(item["validation"]),
            "summary_size_bytes": int(item["summary_size_bytes"]),
            "summary_sha256": str(item["summary_sha256"]),
            "trend_size_bytes": int(item["trend_size_bytes"]),
            "trend_sha256": str(item["trend_sha256"]),
            "imported_at": str(item["imported_at"]),
        }

    @staticmethod
    def _result_import_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "result_id": str(row["id"]),
            "run_id": str(row["id"]),
            "display_name": str(row["display_name"]),
            "source_kind": str(row["source_kind"]),
            "source_run_id": (
                str(row["source_run_id"])
                if row["source_run_id"] is not None
                else None
            ),
            "source_metadata": stable_json_loads(row["source_metadata_json"], {}),
            "bundle_id": str(row["bundle_id"]),
            "managed_relative_dir": str(row["managed_relative_dir"]),
            "validation": stable_json_loads(row["validation_json"], {}),
            "summary_size_bytes": int(row["summary_size_bytes"]),
            "summary_sha256": str(row["summary_sha256"]),
            "trend_size_bytes": int(row["trend_size_bytes"]),
            "trend_sha256": str(row["trend_sha256"]),
            "imported_at": str(row["imported_at"]),
        }

    @staticmethod
    def _sa_import_from_row(row: sqlite3.Row) -> dict[str, Any]:
        result_id = str(row["result_id"])
        return {
            "id": str(row["id"]),
            "result_id": result_id,
            # Compatibility alias for clients migrating from the old synthetic-run
            # result identity.  New writes and relationships use result_id only.
            "run_id": result_id,
            "runset_uid": str(row["runset_uid"]),
            "bundle_id": str(row["bundle_id"]),
            "task_name": str(row["task_name"]),
            "label": str(row["label"]),
            "algorithm_name": str(row["algorithm_name"]),
            "problem_name": str(row["problem_name"]),
            "problem_params": stable_json_loads(row["problem_params_json"]),
            "algorithm_params": stable_json_loads(row["algorithm_params_json"]),
            "scalar_metrics": stable_json_loads(row["scalar_metrics_json"]),
            "trend_metrics": stable_json_loads(row["trend_metrics_json"]),
            "expected_runs": int(row["expected_runs"]),
            "completed_runs": int(row["completed_runs"]),
            "source_fingerprint": str(row["source_fingerprint"]),
            "pairing_signature": str(row["pairing_signature"]),
            "snapshot_sha256": str(row["snapshot_sha256"]),
            "task_sha256": str(row["task_sha256"]),
            "imported_at": str(row["imported_at"]),
        }

    @staticmethod
    def _sa_analysis_values(item: Mapping[str, Any]) -> dict[str, object]:
        required = {
            "id",
            "name",
            "status",
            "method",
            "alpha",
            "metric_names",
            "source_imports",
            "created_at",
        }
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"missing SA analysis field(s): {', '.join(missing)}")
        return {
            "id": str(item["id"]),
            "name": str(item["name"]),
            "status": str(item["status"]),
            "method": str(item["method"]),
            "alpha": float(item["alpha"]),
            "metric_names_json": stable_json_dumps(item["metric_names"]),
            "source_imports_json": stable_json_dumps(item["source_imports"]),
            "result_json": stable_json_dumps(item.get("result") or {}),
            "output_dir": (
                str(item["output_dir"]) if item.get("output_dir") is not None else None
            ),
            "output_files_json": stable_json_dumps(item.get("output_files") or {}),
            "error_message": (
                str(item["error_message"])
                if item.get("error_message") is not None
                else None
            ),
            "created_at": str(item["created_at"]),
            "completed_at": (
                str(item["completed_at"])
                if item.get("completed_at") is not None
                else None
            ),
        }

    @staticmethod
    def _sa_analysis_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "status": str(row["status"]),
            "method": str(row["method"]),
            "alpha": float(row["alpha"]),
            "metric_names": stable_json_loads(row["metric_names_json"]),
            "source_imports": stable_json_loads(row["source_imports_json"]),
            "result": stable_json_loads(row["result_json"]),
            "output_dir": row["output_dir"],
            "output_files": stable_json_loads(row["output_files_json"]),
            "error_message": row["error_message"],
            "created_at": str(row["created_at"]),
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _server_values(server: Server) -> dict[str, object]:
        return {
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
            "enabled": int(server.enabled),
            "last_probe_at": server.last_probe_at,
            "last_probe_json": stable_json_dumps(server.last_probe)
            if server.last_probe is not None
            else None,
            "fingerprint": server.fingerprint,
            "created_at": server.created_at,
            "updated_at": server.updated_at,
        }

    @classmethod
    def _insert_server(cls, connection: sqlite3.Connection, server: Server) -> None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(servers)").fetchall()
        }
        if "target_kind" not in columns:
            # Migration fixtures and recovery utilities may need to populate a
            # historical schema before ``initialize`` applies v6.
            connection.execute(
                """
                INSERT INTO servers (
                    id, name, host, port, username, ssh_key_path, remote_root,
                    conda_env, poll_interval_seconds, default_max_parallel_runs,
                    min_free_bytes, max_load_ratio, enabled, last_probe_at,
                    last_probe_json, fingerprint, created_at, updated_at
                ) VALUES (
                    :id, :name, :host, :port, :username, :ssh_key_path,
                    :remote_root, :conda_env, :poll_interval_seconds,
                    :default_max_parallel_runs, :min_free_bytes, :max_load_ratio,
                    :enabled, :last_probe_at, :last_probe_json, :fingerprint,
                    :created_at, :updated_at
                )
                """,
                cls._server_values(server),
            )
            return
        connection.execute(
            """
            INSERT INTO servers (
                id, target_kind, name, host, port, username, ssh_key_path, remote_root,
                conda_env, poll_interval_seconds, default_max_parallel_runs,
                min_free_bytes, max_load_ratio, enabled, last_probe_at,
                last_probe_json, fingerprint, created_at, updated_at
            ) VALUES (
                :id, :target_kind, :name, :host, :port, :username, :ssh_key_path,
                :remote_root, :conda_env, :poll_interval_seconds,
                :default_max_parallel_runs, :min_free_bytes, :max_load_ratio,
                :enabled, :last_probe_at, :last_probe_json, :fingerprint,
                :created_at, :updated_at
            )
            """,
            cls._server_values(server),
        )

    @staticmethod
    def _server_from_row(row: sqlite3.Row) -> Server:
        return Server(
            id=row["id"],
            target_kind=row["target_kind"],
            name=row["name"],
            host=row["host"],
            port=int(row["port"]),
            username=row["username"],
            ssh_key_path=row["ssh_key_path"],
            remote_root=row["remote_root"],
            conda_env=row["conda_env"],
            poll_interval_seconds=float(row["poll_interval_seconds"]),
            default_max_parallel_runs=int(row["default_max_parallel_runs"]),
            min_free_bytes=int(row["min_free_bytes"]),
            max_load_ratio=float(row["max_load_ratio"]),
            enabled=bool(row["enabled"]),
            last_probe_at=row["last_probe_at"],
            last_probe=stable_json_loads(row["last_probe_json"]),
            fingerprint=row["fingerprint"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _run_values(run: Run) -> dict[str, object]:
        return {
            "id": run.id,
            "server_id": run.server_id,
            "status": run.status.value,
            "stage_json": stable_json_dumps(run.stage),
            "task_name": run.task_name,
            "task_kind": run.task_kind,
            "task_path": run.task_path,
            "task_sha256": run.task_sha256,
            "task_payload_json": stable_json_dumps(run.task_payload),
            "command_argv_json": stable_json_dumps(run.command_argv),
            "max_parallel_runs": run.max_parallel_runs,
            "poll_interval_seconds": run.poll_interval_seconds,
            "admission_override": int(run.admission_override),
            "source_root": run.source_root,
            "snapshot_id": run.snapshot_id,
            "snapshot_sha256": run.snapshot_sha256,
            "snapshot_manifest_json": stable_json_dumps(run.snapshot_manifest),
            "git_commit": run.git_commit,
            "git_branch": run.git_branch,
            "git_dirty": int(run.git_dirty) if run.git_dirty is not None else None,
            "git_diff_sha256": run.git_diff_sha256,
            "remote_pid": run.remote_pid,
            "remote_run_dir": run.remote_run_dir,
            "worker_id": run.worker_id,
            "heartbeat_at": run.heartbeat_at,
            "progress_current": run.progress_current,
            "progress_total": run.progress_total,
            "progress_unit": run.progress_unit,
            "progress_message": run.progress_message,
            "exit_code": run.exit_code,
            "error_code": run.error_code,
            "error_message": run.error_message,
            "error_details_json": stable_json_dumps(run.error_details),
            "local_artifact_dir": run.local_artifact_dir,
            "transfer_manifest_json": stable_json_dumps(run.transfer_manifest),
            "transfer_started_at": run.transfer_started_at,
            "transfer_completed_at": run.transfer_completed_at,
            "transfer_verified": int(run.transfer_verified),
            "transfer_verified_at": run.transfer_verified_at,
            "cleanup_requested": int(run.cleanup_requested),
            "remote_cleanup_completed": int(run.remote_cleanup_completed),
            "remote_cleanup_at": run.remote_cleanup_at,
            "remote_cleanup_error": run.remote_cleanup_error,
            "queued_at": run.queued_at,
            "started_at": run.started_at,
            "remote_finished_at": run.remote_finished_at,
            "finished_at": run.finished_at,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        }

    @classmethod
    def _insert_run(cls, connection: sqlite3.Connection, run: Run) -> None:
        values = cls._run_values(run)
        columns = ", ".join(values)
        placeholders = ", ".join(f":{name}" for name in values)
        connection.execute(
            f"INSERT INTO runs ({columns}) VALUES ({placeholders})", values
        )

    @staticmethod
    def _update_run_record(
        connection: sqlite3.Connection, values: Mapping[str, object]
    ) -> None:
        assignments = ", ".join(
            f"{column}=:{column}"
            for column in values
            if column not in {"id", "server_id", "created_at"}
        )
        cursor = connection.execute(
            f"UPDATE runs SET {assignments} WHERE id=:id", values
        )
        if cursor.rowcount != 1:
            raise KeyError(f"run not found: {values['id']}")

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> Run:
        return Run(
            id=row["id"],
            server_id=row["server_id"],
            status=RunStatus(row["status"]),
            stage=stable_json_loads(row["stage_json"], {}),
            task_name=row["task_name"],
            task_kind=row["task_kind"],
            task_path=row["task_path"],
            task_sha256=row["task_sha256"],
            task_payload=stable_json_loads(row["task_payload_json"], {}),
            command_argv=tuple(stable_json_loads(row["command_argv_json"], [])),
            max_parallel_runs=int(row["max_parallel_runs"]),
            poll_interval_seconds=float(row["poll_interval_seconds"]),
            admission_override=bool(row["admission_override"]),
            source_root=row["source_root"],
            snapshot_id=row["snapshot_id"],
            snapshot_sha256=row["snapshot_sha256"],
            snapshot_manifest=stable_json_loads(row["snapshot_manifest_json"], {}),
            git_commit=row["git_commit"],
            git_branch=row["git_branch"],
            git_dirty=bool(row["git_dirty"])
            if row["git_dirty"] is not None
            else None,
            git_diff_sha256=row["git_diff_sha256"],
            remote_pid=row["remote_pid"],
            remote_run_dir=row["remote_run_dir"],
            worker_id=row["worker_id"],
            heartbeat_at=row["heartbeat_at"],
            progress_current=float(row["progress_current"]),
            progress_total=float(row["progress_total"])
            if row["progress_total"] is not None
            else None,
            progress_unit=row["progress_unit"],
            progress_message=row["progress_message"],
            exit_code=row["exit_code"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            error_details=stable_json_loads(row["error_details_json"], {}),
            local_artifact_dir=row["local_artifact_dir"],
            transfer_manifest=stable_json_loads(
                row["transfer_manifest_json"], {}
            ),
            transfer_started_at=row["transfer_started_at"],
            transfer_completed_at=row["transfer_completed_at"],
            transfer_verified=bool(row["transfer_verified"]),
            transfer_verified_at=row["transfer_verified_at"],
            cleanup_requested=bool(row["cleanup_requested"]),
            remote_cleanup_completed=bool(row["remote_cleanup_completed"]),
            remote_cleanup_at=row["remote_cleanup_at"],
            remote_cleanup_error=row["remote_cleanup_error"],
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            remote_finished_at=row["remote_finished_at"],
            finished_at=row["finished_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: RunEvent
    ) -> RunEvent:
        if event.sequence is None:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM run_events "
                "WHERE run_id = ?",
                (event.run_id,),
            ).fetchone()
            sequence = int(row["next"])
        else:
            sequence = event.sequence
        cursor = connection.execute(
            """
            INSERT INTO run_events (
                run_id, sequence, level, kind, message, data_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.run_id,
                sequence,
                event.level.value,
                event.kind,
                event.message,
                stable_json_dumps(event.data),
                event.created_at,
            ),
        )
        return replace(event, id=int(cursor.lastrowid), sequence=sequence)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> RunEvent:
        return RunEvent(
            id=int(row["id"]),
            run_id=row["run_id"],
            sequence=int(row["sequence"]),
            level=EventLevel(row["level"]),
            kind=row["kind"],
            message=row["message"],
            data=stable_json_loads(row["data_json"], {}),
            created_at=row["created_at"],
        )

    @staticmethod
    def _sample_from_row(row: sqlite3.Row) -> ResourceSample:
        return ResourceSample(
            id=int(row["id"]),
            server_id=row["server_id"],
            run_id=row["run_id"],
            target_identity=row["target_identity"],
            sampled_at=row["sampled_at"],
            reachable=bool(row["reachable"]),
            cpu_physical_cores=row["cpu_physical_cores"],
            cpu_logical_cores=row["cpu_logical_cores"],
            cpu_percent=row["cpu_percent"],
            load_1m=row["load_1m"],
            load_5m=row["load_5m"],
            load_15m=row["load_15m"],
            load_ratio=row["load_ratio"],
            memory_total_bytes=row["memory_total_bytes"],
            memory_available_bytes=row["memory_available_bytes"],
            disk_total_bytes=row["disk_total_bytes"],
            disk_free_bytes=row["disk_free_bytes"],
            gpu_usable=(
                bool(row["gpu_usable"])
                if row["gpu_usable"] is not None
                else None
            ),
            gpu_error=row["gpu_error"],
            gpu_devices=tuple(stable_json_loads(row["gpu_devices_json"], [])),
            error=row["error"],
            aggregation=stable_json_loads(row["aggregation_json"], {}),
        )

    @staticmethod
    def _artifact_values(artifact: Artifact) -> dict[str, object]:
        return {
            "id": artifact.id,
            "run_id": artifact.run_id,
            "relative_path": artifact.relative_path,
            "status": artifact.status.value,
            "remote_path": artifact.remote_path,
            "local_path": artifact.local_path,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "transferred_at": artifact.transferred_at,
            "verified_at": artifact.verified_at,
            "verification_error": artifact.verification_error,
            "metadata_json": stable_json_dumps(artifact.metadata),
            "created_at": artifact.created_at,
            "updated_at": artifact.updated_at,
        }

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            run_id=row["run_id"],
            relative_path=row["relative_path"],
            status=ArtifactStatus(row["status"]),
            remote_path=row["remote_path"],
            local_path=row["local_path"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            transferred_at=row["transferred_at"],
            verified_at=row["verified_at"],
            verification_error=row["verification_error"],
            metadata=stable_json_loads(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
