"""Read-only validation for completed EDOLAB result bundles."""

from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.result.store import SUMMARY_DB_NAME, TREND_DB_NAME
from core.result.event_streams import validate_event_streams


class ResultBundleValidationError(ValueError):
    """Raised when a stored result bundle is missing, corrupt, or incomplete."""


@dataclass(frozen=True, slots=True)
class ResultBundleValidation:
    output_dir: Path
    bundle_id: str
    format_version: str
    mode: str
    status: str
    task_count: int
    runset_count: int
    expected_runs: int
    completed_runs: int
    stored_runs: int
    trend_metric_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "bundle_id": self.bundle_id,
            "format_version": self.format_version,
            "mode": self.mode,
            "status": self.status,
            "task_count": self.task_count,
            "runset_count": self.runset_count,
            "expected_runs": self.expected_runs,
            "completed_runs": self.completed_runs,
            "stored_runs": self.stored_runs,
            "trend_metric_count": self.trend_metric_count,
        }


def validate_result_bundle(
    output_dir: str | Path,
    *,
    require_completed: bool = True,
    expected_runs: int | None = None,
) -> ResultBundleValidation:
    """Validate both SQLite files without creating or modifying either one.

    The normal :class:`~core.result.store.ResultStore` constructor initializes
    schema and metadata, so it must not be used to inspect an immutable remote
    artifact.  This validator opens both databases with SQLite ``mode=ro``.
    """

    root = Path(output_dir).expanduser().resolve()
    summary_path = root / SUMMARY_DB_NAME
    trend_path = root / TREND_DB_NAME
    for path in (summary_path, trend_path):
        if not path.is_file():
            raise ResultBundleValidationError(f"Missing result database: {path}")

    with _open_read_only(summary_path) as summary, _open_read_only(trend_path) as trend:
        _quick_check(summary, summary_path)
        _quick_check(trend, trend_path)
        _foreign_key_check(summary, summary_path)
        _foreign_key_check(trend, trend_path)
        try:
            validate_event_streams(summary)
        except (ValueError, sqlite3.Error) as exc:
            raise ResultBundleValidationError(f'Invalid run event audit: {exc}') from exc
        summary_meta = _read_meta(summary, summary_path)
        trend_meta = _read_meta(trend, trend_path)

        summary_bundle_id = str(summary_meta.get("bundle_id") or "")
        trend_bundle_id = str(trend_meta.get("bundle_id") or "")
        if not summary_bundle_id or summary_bundle_id != trend_bundle_id:
            raise ResultBundleValidationError(
                "summary.sqlite and trend.sqlite do not identify the same bundle."
            )

        summary_version = str(summary_meta.get("format_version") or "")
        trend_version = str(trend_meta.get("format_version") or "")
        if not summary_version or summary_version != trend_version:
            raise ResultBundleValidationError(
                "summary.sqlite and trend.sqlite use different format versions."
            )

        status = str(summary_meta.get("status") or "")
        trend_status = str(trend_meta.get("status") or "")
        if status != trend_status:
            raise ResultBundleValidationError(
                "summary.sqlite and trend.sqlite report different bundle statuses."
            )
        if require_completed and status != "completed":
            raise ResultBundleValidationError(
                f"Result bundle status is {status or '<missing>'}, not completed."
            )

        task_count = _scalar(summary, "select count(*) from tasks")
        runset_count = _scalar(summary, "select count(*) from runsets")
        runset_totals = summary.execute(
            """
            select
                coalesce(sum(expected_runs), 0),
                coalesce(sum(completed_runs), 0),
                coalesce(sum(case when status = 'completed' then 0 else 1 end), 0)
            from runsets
            """
        ).fetchone()
        total_expected = int(runset_totals[0])
        total_completed = int(runset_totals[1])
        incomplete_runsets = int(runset_totals[2])
        stored_runs = _scalar(summary, "select count(*) from runs")
        incomplete_tasks = _scalar(
            summary, "select count(*) from tasks where status != 'completed'"
        )
        incomplete_runs = _scalar(
            summary, "select count(*) from runs where status != 'completed'"
        )

        inconsistent = _scalar(
            summary,
            """
            select count(*)
            from runsets as rs
            where rs.completed_runs != (
                select count(*) from runs as r where r.runset_id = rs.id
            )
            """,
        )
        if inconsistent:
            raise ResultBundleValidationError(
                f"{inconsistent} runset(s) disagree with their stored run rows."
            )
        if total_completed != stored_runs:
            raise ResultBundleValidationError(
                "Completed-run counters do not match the number of stored runs."
            )
        if require_completed and (
            incomplete_tasks
            or incomplete_runsets
            or incomplete_runs
            or total_expected != total_completed
        ):
            raise ResultBundleValidationError(
                "One or more tasks, runsets, or runs are incomplete despite completed bundle status."
            )
        if expected_runs is not None and stored_runs != int(expected_runs):
            raise ResultBundleValidationError(
                f"Expected {int(expected_runs)} stored runs, found {stored_runs}."
            )

        summary_runsets = {
            int(row[0]): str(row[1])
            for row in summary.execute("select id, runset_uid from runsets").fetchall()
        }
        orphan_trends = _scalar(
            trend,
            """
            select count(*)
            from trend_metrics as tm
            left join trend_blobs as tb on tb.trend_metric_id = tm.id
            where tb.trend_metric_id is null
            """,
        )
        if orphan_trends:
            raise ResultBundleValidationError(
                f"{orphan_trends} trend metric(s) have no payload."
            )
        trend_rows = trend.execute(
            """
            select tm.runset_id, tm.runset_uid, tm.metric_name, tm.sample_count,
                   tb.payload
            from trend_metrics as tm
            join trend_blobs as tb on tb.trend_metric_id = tm.id
            order by tm.id
            """
        ).fetchall()
        for runset_id, runset_uid, metric_name, sample_count, payload in trend_rows:
            expected_uid = summary_runsets.get(int(runset_id))
            if expected_uid is None or expected_uid != str(runset_uid):
                raise ResultBundleValidationError(
                    f"Trend metric {metric_name!r} does not match a summary runset."
                )
            _validate_trend_payload(
                payload,
                metric_name=str(metric_name),
                sample_count=int(sample_count),
            )
        trend_metric_count = len(trend_rows)

        return ResultBundleValidation(
            output_dir=root,
            bundle_id=summary_bundle_id,
            format_version=summary_version,
            mode=str(summary_meta.get("mode") or ""),
            status=status,
            task_count=task_count,
            runset_count=runset_count,
            expected_runs=total_expected,
            completed_runs=total_completed,
            stored_runs=stored_runs,
            trend_metric_count=trend_metric_count,
        )


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("pragma query_only = on")
    return connection


def _quick_check(connection: sqlite3.Connection, path: Path) -> None:
    rows = connection.execute("pragma quick_check").fetchall()
    messages = [str(row[0]) for row in rows]
    if messages != ["ok"]:
        raise ResultBundleValidationError(
            f"SQLite quick_check failed for {path}: {'; '.join(messages)}"
        )


def _foreign_key_check(connection: sqlite3.Connection, path: Path) -> None:
    rows = connection.execute("pragma foreign_key_check").fetchall()
    if rows:
        raise ResultBundleValidationError(
            f"SQLite foreign_key_check failed for {path}: {len(rows)} violation(s)."
        )


def _read_meta(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    try:
        rows = connection.execute("select key, value from meta").fetchall()
    except sqlite3.DatabaseError as exc:
        raise ResultBundleValidationError(f"Invalid metadata table in {path}.") from exc
    payload: dict[str, Any] = {}
    for key, value in rows:
        try:
            payload[str(key)] = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ResultBundleValidationError(
                f"Invalid JSON metadata value {key!r} in {path}."
            ) from exc
    return payload


def _scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def _validate_trend_payload(
    payload: bytes, *, metric_name: str, sample_count: int
) -> None:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            required = ("key", "mean", "stderr", "count")
            arrays = {name: np.asarray(archive[name]) for name in required}
    except Exception as exc:
        raise ResultBundleValidationError(
            f"Trend metric {metric_name!r} has an unreadable payload."
        ) from exc
    lengths = {name: int(value.size) for name, value in arrays.items()}
    if any(value.ndim != 1 for value in arrays.values()) or set(lengths.values()) != {
        sample_count
    }:
        raise ResultBundleValidationError(
            f"Trend metric {metric_name!r} payload shape disagrees with sample_count."
        )
