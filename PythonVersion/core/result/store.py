"""SQLite-backed raw run result storage."""

from __future__ import annotations

import io
import hashlib
import json
import math
import sqlite3
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from core.evaluation.result import RunResult
from core.parameters import resolve_parameters
from core.registry import algorithm_registry, problem_registry
from core.result.event_streams import export_run_events, prepare_event_streams, write_event_streams


FORMAT_VERSION = "2"
SUMMARY_DB_NAME = "summary.sqlite"
TREND_DB_NAME = "trend.sqlite"


@dataclass(slots=True)
class TrendSummary:
    metric_name: str
    axis: str
    key: list[Any]
    mean: list[float]
    stderr: list[float]
    count: list[int]


@dataclass(slots=True)
class RunsetMetricSamples:
    runset_id: int
    runset_uid: str
    task_uid: str
    task_label: str
    algorithm_name: str
    problem_name: str
    problem_params: dict[str, Any]
    metric_name: str
    direction: str
    values: list[float]


class TrendAccumulator:
    """Online aggregate for run-level trend samples without storing raw trends."""

    def __init__(self, metric_name: str, axis: str):
        self.metric_name = metric_name
        self.axis = axis
        self._sum: dict[Any, float] = defaultdict(float)
        self._sum_sq: dict[Any, float] = defaultdict(float)
        self._count: dict[Any, int] = defaultdict(int)

    def add(self, series: dict[str, Any]) -> None:
        keys = list(series.get("key", []))
        values = list(series.get("value", []))
        for key, value in zip(keys, values):
            number = float(value)
            if not math.isfinite(number):
                continue
            self._sum[key] += number
            self._sum_sq[key] += number * number
            self._count[key] += 1

    def summary(self) -> TrendSummary:
        keys = list(self._count)
        try:
            keys = sorted(keys)
        except TypeError:
            pass
        means: list[float] = []
        stderrs: list[float] = []
        counts: list[int] = []
        for key in keys:
            count = self._count[key]
            mean_value = self._sum[key] / count
            variance = max(self._sum_sq[key] / count - mean_value * mean_value, 0.0)
            means.append(float(mean_value))
            stderrs.append(float(math.sqrt(variance) / math.sqrt(count)))
            counts.append(int(count))
        return TrendSummary(
            metric_name=self.metric_name,
            axis=self.axis,
            key=keys,
            mean=means,
            stderr=stderrs,
            count=counts,
        )


class RunsetTrendAccumulators:
    """Metric-indexed trend accumulators for one runset."""

    def __init__(self) -> None:
        self._items: dict[str, TrendAccumulator] = {}

    def add_run(
        self,
        result: RunResult,
        metric_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        allowed = set(metric_names) if metric_names is not None else None
        for metric_name, series in result.raw.get("trends", {}).items():
            if allowed is not None and metric_name not in allowed:
                continue
            axis = str(series.get("axis", "fe"))
            accumulator = self._items.get(metric_name)
            if accumulator is None:
                accumulator = TrendAccumulator(metric_name, axis)
                self._items[metric_name] = accumulator
            accumulator.add(series)

    def summaries(self) -> list[TrendSummary]:
        return [item.summary() for item in self._items.values()]


class ResultStore:
    """Store raw scalar summaries and aggregate trends for one experiment."""

    def __init__(self, output_dir: str | Path, *, event_sidecar: bool = True):
        self.output_dir = Path(output_dir)
        self.event_sidecar = event_sidecar
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / SUMMARY_DB_NAME
        self.trend_path = self.output_dir / TREND_DB_NAME
        self.summary = sqlite3.connect(self.summary_path)
        self.trend = sqlite3.connect(self.trend_path)
        self.summary.row_factory = sqlite3.Row
        self.trend.row_factory = sqlite3.Row
        self._initialize()
        self.bundle_id = self.get_meta("bundle_id")
        if not self.bundle_id:
            self.bundle_id = str(uuid.uuid4())
            self.set_meta("bundle_id", self.bundle_id)
        self.set_meta("format_version", FORMAT_VERSION)

    def close(self) -> None:
        self.summary.close()
        self.trend.close()
        if self.event_sidecar:
            export_run_events(self.output_dir)

    def get_meta(self, key: str) -> Any:
        row = self.summary.execute(
            "select value from meta where key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        self.summary.execute(
            "insert or replace into meta(key, value) values (?, ?)",
            (key, payload),
        )
        self.trend.execute(
            "insert or replace into meta(key, value) values (?, ?)",
            (key, payload),
        )
        self.summary.commit()
        self.trend.commit()

    def register_task(
        self,
        *,
        label: str,
        problem_name: str,
        run_number: int,
        problem_seed: Any,
        algorithm_seed: Any,
        problem_params: dict[str, Any],
        metric_output_names: list[str] | None,
        source_task_file: str | None = None,
        source_line: int | None = None,
        task_config_hash: str | None = None,
        task_uid: str | None = None,
    ) -> int:
        configured_problem_params = _configured_problem_params(problem_name, problem_params)
        params_json = _canonical_json(configured_problem_params)
        config_hash = task_config_hash or stable_hash(
            {
                "label": label,
                "problem_name": problem_name,
                "run_number": run_number,
                "problem_seed": problem_seed,
                "algorithm_seed": algorithm_seed,
                "problem_params": problem_params,
                "metric_output_names": metric_output_names,
                "source_task_file": source_task_file,
                "source_line": source_line,
            }
        )
        uid = task_uid or _stable_uid(
            "task",
            self.bundle_id,
            source_task_file,
            source_line,
            label,
            config_hash,
        )
        cursor = self.summary.execute(
            """
            insert into tasks(
                task_uid, label, source_task_file, source_line, task_config_hash,
                problem_name, run_number, problem_seed_json, algorithm_seed_json,
                problem_params_json, problem_params_key, metric_output_names_json,
                status, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                uid,
                label,
                source_task_file,
                int(source_line) if source_line is not None else None,
                config_hash,
                problem_name,
                int(run_number),
                _canonical_json(problem_seed),
                _canonical_json(algorithm_seed),
                params_json,
                params_json,
                _canonical_json(metric_output_names),
                _now_text(),
            ),
        )
        self.summary.commit()
        task_id = int(cursor.lastrowid)
        self._write_task_problem_params(task_id, uid, configured_problem_params)
        self.write_event("task", uid, "registered", f"Task {label} registered.")
        return task_id

    def register_runset(
        self,
        *,
        task_id: int,
        label: str,
        algorithm_name: str,
        problem_name: str,
        algorithm_params: dict[str, Any],
        expected_runs: int,
        runset_uid: str | None = None,
    ) -> int:
        task_row = self.summary.execute(
            "select task_uid from tasks where id = ?",
            (int(task_id),),
        ).fetchone()
        if task_row is None:
            raise ValueError(f"Unknown task id {task_id}.")
        configured_algorithm_params = _configured_algorithm_params(
            algorithm_name,
            algorithm_params,
        )
        params_json = _canonical_json(configured_algorithm_params)
        uid = runset_uid or _stable_uid(
            "runset",
            self.bundle_id,
            task_row["task_uid"],
            algorithm_name,
            params_json,
        )
        cursor = self.summary.execute(
            """
            insert into runsets(
                runset_uid, task_id, label, algorithm_name, problem_name,
                algorithm_params_json, algorithm_params_key, expected_runs, completed_runs,
                status, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, 0, 'running', ?)
            """,
            (
                uid,
                int(task_id),
                label,
                algorithm_name,
                problem_name,
                params_json,
                params_json,
                int(expected_runs),
                _now_text(),
            ),
        )
        self.summary.commit()
        runset_id = int(cursor.lastrowid)
        self._write_runset_algorithm_params(
            runset_id,
            uid,
            configured_algorithm_params,
        )
        self.write_event("runset", uid, "registered", f"Runset {label}:{algorithm_name} registered.")
        return runset_id

    def write_run_result(
        self,
        runset_id: int,
        result: RunResult,
        *,
        metric_output_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> int:
        runset_uid = self.runset_uid(runset_id)
        run_uid = _stable_uid("run", self.bundle_id, runset_uid, result.run_index)
        streams = prepare_event_streams(result.raw.get('event_streams', {}))
        cursor = self.summary.execute(
            """
            insert into runs(
                run_uid, runset_id, run_index, problem_seed, algorithm_seed,
                fe, runtime_seconds, status, compute_json, iteration,
                created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?)
            """,
            (
                run_uid,
                int(runset_id),
                int(result.run_index),
                int(result.problem_seed),
                int(result.algorithm_seed),
                int(result.fe),
                float(result.runtime_seconds),
                _canonical_json(result.raw.get("compute")),
                int(result.raw.get("iteration", 0)),
                _now_text(),
            ),
        )
        run_id = int(cursor.lastrowid)
        write_event_streams(self.summary, run_uid, streams)
        values = _filter_metric_values(result.metrics, metric_output_names)
        if metric_output_names is None or "T_r" in set(metric_output_names):
            values["T_r"] = float(result.runtime_seconds)
        for metric_name, value in sorted(values.items()):
            scalar = float(value)
            # Optional summaries use NaN for an undefined value (for example,
            # the failed-environment tail when every environment reaches the
            # target). SQLite converts NaN to NULL, which violates this table's
            # schema. Keep the run and omit only that undefined metric sample.
            if math.isnan(scalar):
                continue
            if not math.isfinite(scalar):
                raise ValueError(f"Non-finite summary metric {metric_name}: {scalar}")
            self.summary.execute(
                """
                insert into run_metric_values(
                    run_id, runset_id, metric_name, value, direction
                )
                values (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(runset_id),
                    metric_name,
                    scalar,
                    metric_direction(metric_name),
                ),
            )
        completed_runs = self.summary.execute(
            "select count(*) from runs where runset_id = ?",
            (int(runset_id),),
        ).fetchone()[0]
        self.summary.execute(
            "update runsets set completed_runs = ? where id = ?",
            (int(completed_runs), int(runset_id)),
        )
        self.summary.commit()
        self.write_event(
            "run",
            run_uid,
            "completed",
            f"Run {result.run_index} completed.",
            {"runset_uid": runset_uid, "fe": result.fe, "runtime_seconds": result.runtime_seconds},
        )
        return run_id

    def finalize_runset(
        self,
        runset_id: int,
        trend_summaries: list[TrendSummary],
    ) -> None:
        runset_uid = self.runset_uid(runset_id)
        self._write_summary_metrics(runset_id)
        for summary in trend_summaries:
            self.write_trend_summary(runset_id, summary)
        self.summary.execute(
            "update runsets set status = 'completed', completed_at = ? where id = ?",
            (_now_text(), int(runset_id)),
        )
        task_id = self.summary.execute(
            "select task_id from runsets where id = ?",
            (int(runset_id),),
        ).fetchone()[0]
        remaining = self.summary.execute(
            """
            select count(*) from runsets
            where task_id = ? and status != 'completed'
            """,
            (int(task_id),),
        ).fetchone()[0]
        if int(remaining) == 0:
            self.summary.execute(
                "update tasks set status = 'completed', completed_at = ? where id = ?",
                (_now_text(), int(task_id)),
            )
        self.summary.commit()
        self.write_event("runset", runset_uid, "completed", "Runset completed.")

    def mark_failed(
        self,
        *,
        scope: str,
        scope_uid: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if scope == "task":
            self.summary.execute(
                "update tasks set status = 'failed', completed_at = ?, error_message = ? where task_uid = ?",
                (_now_text(), message, scope_uid),
            )
        elif scope == "runset":
            self.summary.execute(
                "update runsets set status = 'failed', completed_at = ?, error_message = ? where runset_uid = ?",
                (_now_text(), message, scope_uid),
            )
        self.summary.commit()
        self.write_event(scope, scope_uid, "failed", message, payload)

    def mark_incomplete_canceled(self, message: str = "Canceled.") -> None:
        now = _now_text()
        runsets = self.summary.execute(
            "select runset_uid from runsets where status not in ('completed', 'failed', 'canceled')"
        ).fetchall()
        tasks = self.summary.execute(
            "select task_uid from tasks where status not in ('completed', 'failed', 'canceled')"
        ).fetchall()
        self.summary.execute(
            """
            update runsets
            set status = 'canceled', completed_at = ?, error_message = ?
            where status not in ('completed', 'failed', 'canceled')
            """,
            (now, message),
        )
        self.summary.execute(
            """
            update tasks
            set status = 'canceled', completed_at = ?, error_message = ?
            where status not in ('completed', 'failed', 'canceled')
            """,
            (now, message),
        )
        self.summary.commit()
        for row in runsets:
            self.write_event("runset", row["runset_uid"], "canceled", message)
        for row in tasks:
            self.write_event("task", row["task_uid"], "canceled", message)

    def write_event(
        self,
        scope: str,
        scope_uid: str,
        event_type: str,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.summary.execute(
            """
            insert into result_events(scope, scope_uid, event_type, message, payload_json, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                scope,
                scope_uid,
                event_type,
                message,
                _canonical_json(payload or {}),
                _now_text(),
            ),
        )
        self.summary.commit()

    def write_trend_summary(self, runset_id: int, summary: TrendSummary) -> None:
        payload = _trend_payload(summary)
        runset_uid = self.runset_uid(runset_id)
        cursor = self.trend.execute(
            """
            insert or replace into trend_metrics(
                runset_id, runset_uid, metric_name, axis, sample_count, created_at
            )
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                int(runset_id),
                runset_uid,
                summary.metric_name,
                summary.axis,
                len(summary.key),
                _now_text(),
            ),
        )
        trend_metric_id = int(cursor.lastrowid)
        self.trend.execute(
            """
            insert or replace into trend_blobs(trend_metric_id, payload)
            values (?, ?)
            """,
            (trend_metric_id, payload),
        )
        self.trend.commit()

    def list_runsets(self) -> list[dict[str, Any]]:
        rows = self.summary.execute(
            """
            select
                runsets.id as runset_id,
                runsets.runset_uid,
                runsets.task_id,
                tasks.task_uid,
                runsets.label,
                runsets.algorithm_name,
                runsets.problem_name,
                runsets.algorithm_params_json,
                runsets.expected_runs,
                runsets.completed_runs,
                runsets.status,
                tasks.problem_params_json,
                tasks.source_task_file,
                tasks.source_line,
                tasks.task_config_hash
            from runsets
            join tasks on tasks.id = runsets.task_id
            order by tasks.id, runsets.id
            """
        ).fetchall()
        return [
            {
                "runset_id": int(row["runset_id"]),
                "runset_uid": row["runset_uid"],
                "task_id": int(row["task_id"]),
                "task_uid": row["task_uid"],
                "label": row["label"],
                "algorithm_name": row["algorithm_name"],
                "problem_name": row["problem_name"],
                "algorithm_params": json.loads(row["algorithm_params_json"]),
                "expected_runs": int(row["expected_runs"]),
                "completed_runs": int(row["completed_runs"]),
                "status": row["status"],
                "problem_params": json.loads(row["problem_params_json"]),
                "source_task_file": row["source_task_file"],
                "source_line": row["source_line"],
                "task_config_hash": row["task_config_hash"],
            }
            for row in rows
        ]

    def list_summary_metrics(
        self,
        runset_uids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if runset_uids:
            placeholders = ",".join("?" for _ in runset_uids)
            rows = self.summary.execute(
                f"""
                select runset_uid, metric_name, mean, median, stderr, count, direction
                from summary_metrics
                where runset_uid in ({placeholders})
                order by runset_uid, metric_name
                """,
                tuple(runset_uids),
            ).fetchall()
        else:
            rows = self.summary.execute(
                """
                select runset_uid, metric_name, mean, median, stderr, count, direction
                from summary_metrics
                order by runset_uid, metric_name
                """
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["runset_uid"]].append(
                {
                    "metric_name": row["metric_name"],
                    "mean": float(row["mean"]),
                    "median": float(row["median"]),
                    "stderr": float(row["stderr"]),
                    "count": int(row["count"]),
                    "direction": row["direction"],
                }
            )
        return dict(grouped)

    def list_tasks(self) -> list[dict[str, Any]]:
        rows = self.summary.execute(
            """
            select id, task_uid, label, source_task_file, source_line, task_config_hash,
                   problem_name, run_number, problem_params_json, status,
                   created_at, completed_at, error_message
            from tasks
            order by id
            """
        ).fetchall()
        return [
            {
                "task_id": int(row["id"]),
                "task_uid": row["task_uid"],
                "label": row["label"],
                "source_task_file": row["source_task_file"],
                "source_line": row["source_line"],
                "task_config_hash": row["task_config_hash"],
                "problem_name": row["problem_name"],
                "run_number": int(row["run_number"]),
                "problem_params": json.loads(row["problem_params_json"]),
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "error_message": row["error_message"],
            }
            for row in rows
        ]

    def list_scalar_metrics(self, runset_uids: list[str] | None = None) -> list[str]:
        if runset_uids:
            placeholders = ",".join("?" for _ in runset_uids)
            rows = self.summary.execute(
                f"""
                select distinct run_metric_values.metric_name
                from run_metric_values
                join runsets on runsets.id = run_metric_values.runset_id
                where runsets.runset_uid in ({placeholders})
                order by run_metric_values.metric_name
                """,
                tuple(runset_uids),
            ).fetchall()
        else:
            rows = self.summary.execute(
                "select distinct metric_name from run_metric_values order by metric_name"
            ).fetchall()
        return [row["metric_name"] for row in rows]

    def list_trend_metrics(self, runset_uid: str | None = None) -> list[str]:
        if runset_uid:
            rows = self.trend.execute(
                """
                select metric_name
                from trend_metrics
                where runset_uid = ?
                order by metric_name
                """,
                (runset_uid,),
            ).fetchall()
        else:
            rows = self.trend.execute(
                "select distinct metric_name from trend_metrics order by metric_name"
            ).fetchall()
        return [row["metric_name"] for row in rows]

    def list_trend_metric_axes(self, runset_uid: str) -> dict[str, str]:
        rows = self.trend.execute(
            """
            select metric_name, axis
            from trend_metrics
            where runset_uid = ?
            order by metric_name
            """,
            (runset_uid,),
        ).fetchall()
        return {str(row["metric_name"]): str(row["axis"]) for row in rows}

    def load_metric_samples(
        self,
        runset_ids: list[int] | tuple[int, ...],
        metric_name: str,
    ) -> list[RunsetMetricSamples]:
        if not runset_ids:
            return []
        placeholders = ",".join("?" for _ in runset_ids)
        rows = self.summary.execute(
            f"""
            select
                runsets.id as runset_id,
                runsets.runset_uid,
                tasks.task_uid,
                runsets.label,
                runsets.algorithm_name,
                runsets.problem_name,
                tasks.problem_params_json,
                runs.run_index,
                run_metric_values.value,
                run_metric_values.direction
            from run_metric_values
            join runs on runs.id = run_metric_values.run_id
            join runsets on runsets.id = run_metric_values.runset_id
            join tasks on tasks.id = runsets.task_id
            where run_metric_values.metric_name = ?
              and runsets.id in ({placeholders})
            order by runsets.id, runs.run_index
            """,
            (metric_name, *[int(item) for item in runset_ids]),
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            grouped[int(row["runset_id"])].append(row)
        samples: list[RunsetMetricSamples] = []
        for runset_id in runset_ids:
            group = grouped.get(int(runset_id), [])
            if not group:
                continue
            first = group[0]
            samples.append(
                RunsetMetricSamples(
                    runset_id=int(runset_id),
                    runset_uid=first["runset_uid"],
                    task_uid=first["task_uid"],
                    task_label=first["label"],
                    algorithm_name=first["algorithm_name"],
                    problem_name=first["problem_name"],
                    problem_params=json.loads(first["problem_params_json"]),
                    metric_name=metric_name,
                    direction=first["direction"],
                    values=[float(row["value"]) for row in group],
                )
            )
        return samples

    def load_metric_samples_by_uid(
        self,
        runset_uids: list[str] | tuple[str, ...],
        metric_name: str,
    ) -> list[RunsetMetricSamples]:
        runset_ids = self.resolve_runset_ids(runset_uids)
        return self.load_metric_samples(runset_ids, metric_name)

    def resolve_runset_ids(self, runset_uids: list[str] | tuple[str, ...]) -> list[int]:
        if not runset_uids:
            return []
        placeholders = ",".join("?" for _ in runset_uids)
        rows = self.summary.execute(
            f"""
            select id, runset_uid
            from runsets
            where runset_uid in ({placeholders})
            """,
            tuple(runset_uids),
        ).fetchall()
        by_uid = {row["runset_uid"]: int(row["id"]) for row in rows}
        missing = [uid for uid in runset_uids if uid not in by_uid]
        if missing:
            raise ValueError(f"Unknown runset uid(s): {', '.join(missing)}.")
        return [by_uid[uid] for uid in runset_uids]

    def load_trend(self, runset_id: int, metric_name: str) -> TrendSummary | None:
        row = self.trend.execute(
            """
            select trend_metrics.axis, trend_blobs.payload
            from trend_metrics
            join trend_blobs on trend_blobs.trend_metric_id = trend_metrics.id
            where trend_metrics.runset_id = ? and trend_metrics.metric_name = ?
            """,
            (int(runset_id), metric_name),
        ).fetchone()
        if row is None:
            return None
        payload = _load_trend_payload(row["payload"])
        return TrendSummary(
            metric_name=metric_name,
            axis=row["axis"],
            key=payload["key"],
            mean=payload["mean"],
            stderr=payload["stderr"],
            count=payload["count"],
        )

    def load_trend_by_uid(self, runset_uid: str, metric_name: str) -> TrendSummary | None:
        row = self.trend.execute(
            """
            select trend_metrics.axis, trend_blobs.payload
            from trend_metrics
            join trend_blobs on trend_blobs.trend_metric_id = trend_metrics.id
            where trend_metrics.runset_uid = ? and trend_metrics.metric_name = ?
            """,
            (runset_uid, metric_name),
        ).fetchone()
        if row is None:
            return None
        payload = _load_trend_payload(row["payload"])
        return TrendSummary(
            metric_name=metric_name,
            axis=row["axis"],
            key=payload["key"],
            mean=payload["mean"],
            stderr=payload["stderr"],
            count=payload["count"],
        )

    def load_trend_window_by_uid(
        self,
        runset_uid: str,
        metric_name: str,
        *,
        offset: int = 0,
        limit: int = 500,
    ) -> TrendSummary | None:
        trend = self.load_trend_by_uid(runset_uid, metric_name)
        if trend is None:
            return None
        start = max(0, int(offset))
        end = start + max(1, int(limit))
        return TrendSummary(
            metric_name=trend.metric_name,
            axis=trend.axis,
            key=trend.key[start:end],
            mean=trend.mean[start:end],
            stderr=trend.stderr[start:end],
            count=trend.count[start:end],
        )

    def runset_uid(self, runset_id: int) -> str:
        row = self.summary.execute(
            "select runset_uid from runsets where id = ?",
            (int(runset_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown runset id {runset_id}.")
        return str(row["runset_uid"])

    def _initialize(self) -> None:
        self._initialize_summary()
        self._initialize_trend()

    def _initialize_summary(self) -> None:
        self.summary.executescript(
            """
            pragma journal_mode = delete;
            create table if not exists meta(
                key text primary key,
                value text not null
            );
            create table if not exists tasks(
                id integer primary key autoincrement,
                task_uid text not null unique,
                label text not null,
                source_task_file text,
                source_line integer,
                task_config_hash text not null,
                problem_name text not null,
                run_number integer not null,
                problem_seed_json text not null,
                algorithm_seed_json text not null,
                problem_params_json text not null,
                problem_params_key text not null,
                metric_output_names_json text not null,
                status text not null,
                created_at text not null,
                completed_at text,
                error_message text
            );
            create table if not exists runsets(
                id integer primary key autoincrement,
                runset_uid text not null unique,
                task_id integer not null references tasks(id),
                label text not null,
                algorithm_name text not null,
                problem_name text not null,
                algorithm_params_json text not null,
                algorithm_params_key text not null,
                expected_runs integer not null,
                completed_runs integer not null,
                status text not null,
                created_at text not null,
                completed_at text,
                error_message text,
                unique(task_id, algorithm_name, algorithm_params_key)
            );
            create table if not exists runs(
                id integer primary key autoincrement,
                run_uid text not null unique,
                runset_id integer not null references runsets(id),
                run_index integer not null,
                problem_seed integer not null,
                algorithm_seed integer not null,
                fe integer not null,
                runtime_seconds real not null,
                status text not null,
                compute_json text not null,
                iteration integer not null,
                created_at text not null,
                unique(runset_id, run_index)
            );
            create table if not exists run_metric_values(
                run_id integer not null references runs(id),
                runset_id integer not null references runsets(id),
                metric_name text not null,
                value real not null,
                direction text not null,
                primary key(run_id, metric_name)
            );
            create index if not exists idx_run_metric_values_runset_metric
                on run_metric_values(runset_id, metric_name);
            create index if not exists idx_runsets_uid
                on runsets(runset_uid);
            create table if not exists summary_metrics(
                runset_id integer not null references runsets(id),
                runset_uid text not null,
                metric_name text not null,
                mean real not null,
                median real not null,
                stderr real not null,
                count integer not null,
                direction text not null,
                primary key(runset_id, metric_name)
            );
            create table if not exists task_problem_params(
                task_id integer not null references tasks(id),
                task_uid text not null,
                key text not null,
                value_json text not null,
                value_text text not null,
                primary key(task_id, key)
            );
            create index if not exists idx_task_problem_params_search
                on task_problem_params(key, value_text);
            create table if not exists runset_algorithm_params(
                runset_id integer not null references runsets(id),
                runset_uid text not null,
                key text not null,
                value_json text not null,
                value_text text not null,
                primary key(runset_id, key)
            );
            create index if not exists idx_runset_algorithm_params_search
                on runset_algorithm_params(key, value_text);
            create table if not exists result_events(
                id integer primary key autoincrement,
                scope text not null,
                scope_uid text not null,
                event_type text not null,
                message text not null,
                payload_json text not null,
                created_at text not null
            );
            create index if not exists idx_result_events_scope
                on result_events(scope, scope_uid, id);
            """
        )
        self.summary.commit()

    def _write_task_problem_params(
        self,
        task_id: int,
        task_uid: str,
        params: dict[str, Any],
    ) -> None:
        self.summary.executemany(
            """
            insert or replace into task_problem_params(
                task_id, task_uid, key, value_json, value_text
            )
            values (?, ?, ?, ?, ?)
            """,
            [
                (
                    int(task_id),
                    task_uid,
                    str(key),
                    _canonical_json(value),
                    _param_value_text(value),
                )
                for key, value in sorted(params.items())
            ],
        )
        self.summary.commit()

    def _write_runset_algorithm_params(
        self,
        runset_id: int,
        runset_uid: str,
        params: dict[str, Any],
    ) -> None:
        self.summary.executemany(
            """
            insert or replace into runset_algorithm_params(
                runset_id, runset_uid, key, value_json, value_text
            )
            values (?, ?, ?, ?, ?)
            """,
            [
                (
                    int(runset_id),
                    runset_uid,
                    str(key),
                    _canonical_json(value),
                    _param_value_text(value),
                )
                for key, value in sorted(params.items())
            ],
        )
        self.summary.commit()

    def _initialize_trend(self) -> None:
        self.trend.executescript(
            """
            pragma journal_mode = delete;
            create table if not exists meta(
                key text primary key,
                value text not null
            );
            create table if not exists trend_metrics(
                id integer primary key autoincrement,
                runset_id integer not null,
                runset_uid text not null,
                metric_name text not null,
                axis text not null,
                sample_count integer not null,
                created_at text not null,
                unique(runset_uid, metric_name)
            );
            create table if not exists trend_blobs(
                trend_metric_id integer primary key references trend_metrics(id),
                payload blob not null
            );
            """
        )
        self.trend.commit()

    def _write_summary_metrics(self, runset_id: int) -> None:
        runset_uid = self.runset_uid(runset_id)
        metric_rows = self.summary.execute(
            """
            select metric_name, value, direction
            from run_metric_values
            where runset_id = ?
            order by metric_name, run_id
            """,
            (int(runset_id),),
        ).fetchall()
        grouped: dict[str, list[float]] = defaultdict(list)
        directions: dict[str, str] = {}
        for row in metric_rows:
            grouped[row["metric_name"]].append(float(row["value"]))
            directions[row["metric_name"]] = row["direction"]
        for metric_name, values in grouped.items():
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                mean_value = float(sum(finite) / len(finite))
                median_value = float(median(finite))
                stderr_value = (
                    float(np.std(np.asarray(finite, dtype=float), ddof=0) / math.sqrt(len(finite)))
                    if finite
                    else float("nan")
                )
            else:
                mean_value = median_value = stderr_value = float("nan")
            self.summary.execute(
                """
                insert or replace into summary_metrics(
                    runset_id, runset_uid, metric_name, mean, median, stderr, count, direction
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(runset_id),
                    runset_uid,
                    metric_name,
                    mean_value,
                    median_value,
                    stderr_value,
                    len(finite),
                    directions.get(metric_name, metric_direction(metric_name)),
                ),
            )


def metric_direction(metric_name: str) -> str:
    """Return comparison direction for scalar metrics."""

    lower_names = {"E_o", "E_bbc", "current_error", "T_r"}
    higher_names = {"P_o", "P_bbc", "current_performance"}
    if metric_name in lower_names:
        return "lower"
    if metric_name in higher_names:
        return "higher"
    if metric_name.startswith("E_"):
        return "lower"
    if metric_name.startswith("P_"):
        return "higher"
    return "neutral"


def _filter_metric_values(
    values: dict[str, float],
    metric_names: list[str] | tuple[str, ...] | set[str] | None,
) -> dict[str, float]:
    if metric_names is None:
        return dict(values)
    allowed = set(metric_names)
    return {name: value for name, value in values.items() if name in allowed}


def _trend_payload(summary: TrendSummary) -> bytes:
    key_array = np.asarray(summary.key)
    if key_array.dtype == object:
        key_array = np.asarray([str(item) for item in summary.key])
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        key=key_array,
        mean=np.asarray(summary.mean, dtype=float),
        stderr=np.asarray(summary.stderr, dtype=float),
        count=np.asarray(summary.count, dtype=int),
    )
    return buffer.getvalue()


def _load_trend_payload(payload: bytes) -> dict[str, list[Any]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {
            "key": data["key"].tolist(),
            "mean": data["mean"].astype(float).tolist(),
            "stderr": data["stderr"].astype(float).tolist(),
            "count": data["count"].astype(int).tolist(),
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _configured_problem_params(problem_name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    try:
        problem_cls = problem_registry.get(problem_name)
        return resolve_parameters(problem_cls.parameter_specs(), dict(overrides))
    except Exception:
        return dict(overrides)


def _configured_algorithm_params(algorithm_name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    try:
        algorithm_cls = algorithm_registry.get(algorithm_name)
        return resolve_parameters(algorithm_cls.parameter_specs(), dict(overrides))
    except Exception:
        return dict(overrides)


def _param_value_text(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "" if value is None else str(value)
    return _canonical_json(value)


def _stable_uid(*parts: Any) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"edolab-python-oo:{text}"))


def _now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
