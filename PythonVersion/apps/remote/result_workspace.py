"""Read-only result previews and the Relay statistical-analysis workspace."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import threading
from typing import Any
from uuid import uuid4

import numpy as np

from core.analysis.service import (
    run_statistical_analysis_from_samples,
    write_analysis_outputs,
)
from core.analysis.statistics import ScalarRunSet
from core.result.plotting import select_trend_sample_indices
from core.result.validation import ResultBundleValidationError, validate_result_bundle

from .config import RelayPaths
from .database import RemoteDatabase
from .domain import RunStatus, stable_json_dumps, utc_now
from .performance import metrics


SA_METHODS = frozenset({"friedman", "signed-rank", "rank-sum"})
SA_OUTPUT_FORMATS = frozenset({"sqlite", "csv", "xlsx", "latex"})


class ResultWorkspaceError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class _ImportedResult:
    id: str
    task_name: str
    task_path: str | None
    status: RunStatus
    exit_code: int
    created_at: str
    finished_at: str
    snapshot_sha256: str | None
    task_sha256: str | None
    local_artifact_dir: str
    source_kind: str
    source_run_id: str | None
    source_metadata: Mapping[str, Any]
    bundle_id: str
    validation: Mapping[str, Any]
    summary_size_bytes: int
    summary_sha256: str
    trend_size_bytes: int
    trend_sha256: str


class ResultWorkspace:
    """Expose immutable result bundles without coupling them to the old Web UI."""

    def __init__(self, paths: RelayPaths, database: RemoteDatabase) -> None:
        self.paths = paths
        self.database = database
        self._verification_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._verification_lock = threading.RLock()
        self._verification_cache_limit = 64

    # -- Result catalog and preview ------------------------------------

    def list_result_catalog(self) -> list[dict[str, Any]]:
        imported_by_result: dict[str, int] = defaultdict(int)
        for item in self.database.list_sa_imports():
            imported_by_result[str(item.get("result_id") or item.get("run_id"))] += 1
        catalog: list[dict[str, Any]] = []
        for stored in self.database.list_result_imports():
            result = self._result_from_record(stored)
            try:
                bundle = self._bundle_path(result)
                validation = self._verify_bundle(result, bundle)
            except (OSError, ResultBundleValidationError, ResultWorkspaceError):
                continue
            catalog.append(
                {
                    "result_id": result.id,
                    "run_id": result.id,
                    "display_name": result.task_name,
                    "task_name": result.task_name,
                    "task_path": result.task_path,
                    "source_kind": result.source_kind,
                    "source_run_id": result.source_run_id,
                    "source_metadata": dict(result.source_metadata),
                    "status": "completed",
                    "exit_code": 0,
                    "created_at": result.created_at,
                    "finished_at": result.finished_at,
                    "imported_at": result.created_at,
                    "bundle_id": validation.bundle_id,
                    "bundle_status": validation.status,
                    "task_count": validation.task_count,
                    "runset_count": validation.runset_count,
                    "expected_runs": validation.expected_runs,
                    "completed_runs": validation.completed_runs,
                    "trend_metric_count": validation.trend_metric_count,
                    "imported_runset_count": imported_by_result.get(result.id, 0),
                    "preview_available": True,
                }
            )
        return catalog

    def verified_run_bundle_path(self, run_id: str) -> Path:
        """Resolve the exact checksum-verified bundle owned by a run record.

        This is intentionally separate from the imported-result read path and
        exists only for the explicit ``import from run`` copy operation.
        """

        run = self.database.get_run(str(run_id))
        if run is None:
            raise ResultWorkspaceError(
                "run_not_found", f"Run was not found: {run_id}", status_code=404
            )
        if not run.transfer_verified:
            raise ResultWorkspaceError(
                "result_not_verified",
                "The run has no checksum-verified local result bundle yet.",
                status_code=409,
            )
        configured = (
            Path(run.local_artifact_dir)
            if run.local_artifact_dir
            else Path("runs") / run.id
        )
        expected_relative = Path("runs") / run.id
        expected_absolute = _absolute_without_symlink_resolution(self.paths.runs) / run.id
        if configured not in {expected_relative, expected_absolute}:
            raise ResultWorkspaceError(
                "unsafe_result_path",
                "The run result path is outside the exact Relay-owned run directory.",
                status_code=409,
            )
        root = expected_absolute
        bundle = root / "remote" / "results" / "bundle"
        if root.is_symlink() or bundle.is_symlink() or not bundle.is_dir():
            raise ResultWorkspaceError(
                "result_not_found",
                "The verified local result bundle is missing or unsafe.",
                status_code=404,
            )
        verified = {
            artifact.relative_path: artifact
            for artifact in self.database.list_artifacts(run.id)
            if artifact.status.value == "verified"
        }
        for filename in ("summary.sqlite", "trend.sqlite"):
            relative = f"remote/results/bundle/{filename}"
            path = bundle / filename
            artifact = verified.get(relative)
            if path.is_symlink() or not path.is_file() or artifact is None:
                raise ResultWorkspaceError(
                    "result_not_verified",
                    f"Required run result artifact is not verified: {relative}",
                    status_code=409,
                )
            if artifact.size_bytes != path.stat().st_size or artifact.sha256 != _sha256_file(path):
                raise ResultWorkspaceError(
                    "result_checksum_changed",
                    f"Verified run result artifact changed on disk: {relative}",
                    status_code=409,
                )
        validate_result_bundle(bundle, require_completed=True)
        return bundle

    def preview_run(self, result_id: str, *, force_verify: bool = False) -> dict[str, Any]:
        result = self._require_run(result_id)
        bundle = self._bundle_path(result, force_verify=force_verify)
        validation = self._verify_bundle(result, bundle)
        summary_path = bundle / "summary.sqlite"
        trend_path = bundle / "trend.sqlite"
        with _open_read_only(summary_path) as summary, _open_read_only(trend_path) as trend:
            tasks = _read_tasks(summary)
            runsets = _read_runsets(summary, trend)
        scalar_metrics = sorted(
            {
                metric["name"]
                for runset in runsets
                for metric in runset["scalar_metrics"]
            }
        )
        trend_metrics = sorted(
            {
                metric["name"]
                for runset in runsets
                for metric in runset["trend_metrics"]
            }
        )
        algorithms = sorted({item["algorithm_name"] for item in runsets})
        problems = sorted({item["problem_name"] for item in runsets})
        imported = {
            (str(item.get("result_id") or item.get("run_id")), item["runset_uid"]): item["id"]
            for item in self.database.list_sa_imports()
        }
        for item in runsets:
            item["sa_import_id"] = imported.get((result.id, item["runset_uid"]))
        return {
            "result_id": result.id,
            "run_id": result.id,
            "display_name": result.task_name,
            "task_name": result.task_name,
            "task_path": result.task_path,
            "source_kind": result.source_kind,
            "source_run_id": result.source_run_id,
            "source_metadata": dict(result.source_metadata),
            "run_status": "completed",
            "exit_code": 0,
            "snapshot_sha256": result.snapshot_sha256,
            "task_sha256": result.task_sha256,
            "source_fingerprint": self._source_fingerprint(result, bundle),
            "validation": validation.as_dict(),
            "algorithms": algorithms,
            "problems": problems,
            "scalar_metrics": scalar_metrics,
            "trend_metrics": trend_metrics,
            "tasks": tasks,
            "runsets": runsets,
        }

    def result_bundle_path(self, result_id: str, *, force_verify: bool = False) -> Path:
        """Return an imported bundle after exact-path and checksum verification."""

        result = self._require_run(result_id)
        return self._bundle_path(result, force_verify=force_verify)

    def preview_runset_samples(
        self,
        run_id: str,
        runset_uid: str,
        *,
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        result = self._require_run(run_id)
        bundle = self._bundle_path(result)
        offset = max(0, int(offset))
        limit = max(1, min(1_000, int(limit)))
        with _open_read_only(bundle / "summary.sqlite") as summary:
            runset = summary.execute(
                """
                SELECT rs.id, rs.runset_uid, rs.label, rs.algorithm_name,
                       rs.problem_name, rs.expected_runs, rs.completed_runs,
                       rs.status, rs.algorithm_params_json,
                       t.task_uid, t.problem_params_json
                FROM runsets AS rs
                JOIN tasks AS t ON t.id = rs.task_id
                WHERE rs.runset_uid = ?
                """,
                (runset_uid,),
            ).fetchone()
            if runset is None:
                raise ResultWorkspaceError(
                    "runset_not_found",
                    f"Result runset was not found: {runset_uid}",
                    status_code=404,
                )
            total = int(
                summary.execute(
                    "SELECT count(*) FROM runs WHERE runset_id = ?",
                    (int(runset["id"]),),
                ).fetchone()[0]
            )
            rows = summary.execute(
                """
                SELECT id, run_uid, run_index, problem_seed, algorithm_seed,
                       fe, runtime_seconds, status, compute_json, iteration,
                       created_at
                FROM runs
                WHERE runset_id = ?
                ORDER BY run_index
                LIMIT ? OFFSET ?
                """,
                (int(runset["id"]), limit, offset),
            ).fetchall()
            run_ids = [int(row["id"]) for row in rows]
            metrics: dict[int, dict[str, float]] = defaultdict(dict)
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                metric_rows = summary.execute(
                    f"""
                    SELECT run_id, metric_name, value
                    FROM run_metric_values
                    WHERE run_id IN ({placeholders})
                    ORDER BY run_id, metric_name
                    """,
                    tuple(run_ids),
                ).fetchall()
                for metric in metric_rows:
                    metrics[int(metric["run_id"])][str(metric["metric_name"])] = float(
                        metric["value"]
                    )
        return {
            "result_id": result.id,
            "run_id": result.id,
            "runset_uid": str(runset["runset_uid"]),
            "label": str(runset["label"]),
            "algorithm_name": str(runset["algorithm_name"]),
            "problem_name": str(runset["problem_name"]),
            "problem_params": json.loads(runset["problem_params_json"]),
            "algorithm_params": json.loads(runset["algorithm_params_json"]),
            "expected_runs": int(runset["expected_runs"]),
            "completed_runs": int(runset["completed_runs"]),
            "status": str(runset["status"]),
            "offset": offset,
            "limit": limit,
            "total": total,
            "runs": [
                {
                    "run_uid": str(row["run_uid"]),
                    "run_index": int(row["run_index"]),
                    "problem_seed": int(row["problem_seed"]),
                    "algorithm_seed": int(row["algorithm_seed"]),
                    "fe": int(row["fe"]),
                    "runtime_seconds": _finite_or_none(row["runtime_seconds"]),
                    "status": str(row["status"]),
                    "iteration": int(row["iteration"]),
                    "compute": json.loads(row["compute_json"]),
                    "metrics": {
                        key: _finite_or_none(value)
                        for key, value in metrics[int(row["id"])].items()
                    },
                    "created_at": str(row["created_at"]),
                }
                for row in rows
            ],
        }

    def preview_trend(
        self,
        run_id: str,
        runset_uid: str,
        metric_name: str,
        *,
        offset: int = 0,
        limit: int = 1_000,
    ) -> dict[str, Any]:
        result = self._require_run(run_id)
        bundle = self._bundle_path(result)
        offset = max(0, int(offset))
        limit = max(1, min(10_000, int(limit)))
        with _open_read_only(bundle / "trend.sqlite") as trend:
            row = trend.execute(
                """
                SELECT tm.axis, tm.sample_count, tb.payload
                FROM trend_metrics AS tm
                JOIN trend_blobs AS tb ON tb.trend_metric_id = tm.id
                WHERE tm.runset_uid = ? AND tm.metric_name = ?
                """,
                (runset_uid, metric_name),
            ).fetchone()
        if row is None:
            raise ResultWorkspaceError(
                "trend_not_found",
                f"Trend metric {metric_name!r} was not found for {runset_uid}.",
                status_code=404,
            )
        payload = _load_trend_payload(row["payload"])
        end = offset + limit
        return {
            "result_id": result.id,
            "run_id": result.id,
            "runset_uid": runset_uid,
            "metric_name": metric_name,
            "axis": str(row["axis"]),
            "total": int(row["sample_count"]),
            "offset": offset,
            "limit": limit,
            "key": payload["key"][offset:end],
            "mean": [_finite_or_none(value) for value in payload["mean"][offset:end]],
            "stderr": [
                _finite_or_none(value) for value in payload["stderr"][offset:end]
            ],
            "count": payload["count"][offset:end],
        }

    def load_plot_trend_series(
        self,
        run_id: str,
        runset_uid: str,
        metric_name: str,
        *,
        max_points: int,
        x_min: float | int | None = None,
        x_max: float | int | None = None,
    ) -> dict[str, Any]:
        """Load, range-filter, and downsample one trend with a single blob decode.

        Plot loading intentionally does not page through :meth:`preview_trend`:
        every preview page would otherwise reopen and decompress the same NPZ
        payload.  Numeric range filtering is applied to the original aligned
        arrays before sampling, all in memory, and never modifies the source
        ``trend.sqlite`` bundle.
        """

        if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
            raise ResultWorkspaceError(
                "invalid_plot_point_limit",
                "Plot max_points must be a positive integer.",
            )
        lower = _plot_x_bound(x_min, "x_min")
        upper = _plot_x_bound(x_max, "x_max")
        if lower is not None and upper is not None and lower > upper:
            raise ResultWorkspaceError(
                "invalid_plot_x_range",
                "Plot x_min must be less than or equal to x_max.",
            )
        result = self._require_run(run_id)
        bundle = self._bundle_path(result)
        with _open_read_only(bundle / "trend.sqlite") as trend:
            row = trend.execute(
                """
                SELECT tm.axis, tm.sample_count, tb.payload
                FROM trend_metrics AS tm
                JOIN trend_blobs AS tb ON tb.trend_metric_id = tm.id
                WHERE tm.runset_uid = ? AND tm.metric_name = ?
                """,
                (runset_uid, metric_name),
            ).fetchone()
        if row is None:
            raise ResultWorkspaceError(
                "trend_not_found",
                f"Trend metric {metric_name!r} was not found for {runset_uid}.",
                status_code=404,
            )
        try:
            payload = _load_trend_arrays(row["payload"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ResultWorkspaceError(
                "plot_trend_invalid",
                "Stored trend data could not be decoded for plotting.",
                status_code=409,
            ) from exc
        declared_count = int(row["sample_count"])
        lengths = {len(payload[name]) for name in ("key", "mean", "stderr", "count")}
        if declared_count < 1 or lengths != {declared_count}:
            raise ResultWorkspaceError(
                "plot_trend_incomplete",
                "Stored trend data is incomplete or internally inconsistent.",
                status_code=409,
            )
        source_indices = np.arange(declared_count, dtype=np.int64)
        if lower is not None or upper is not None:
            try:
                numeric_keys = np.asarray(payload["key"], dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ResultWorkspaceError(
                    "plot_trend_non_numeric_keys",
                    "Plot x_min and x_max require finite numeric trend keys.",
                ) from exc
            if (
                numeric_keys.ndim != 1
                or numeric_keys.size != declared_count
                or not bool(np.all(np.isfinite(numeric_keys)))
            ):
                raise ResultWorkspaceError(
                    "plot_trend_non_numeric_keys",
                    "Plot x_min and x_max require finite numeric trend keys.",
                )
            in_range = np.ones(declared_count, dtype=bool)
            if lower is not None:
                in_range &= numeric_keys >= lower
            if upper is not None:
                in_range &= numeric_keys <= upper
            source_indices = source_indices[in_range]
            if source_indices.size == 0:
                raise ResultWorkspaceError(
                    "plot_trend_range_empty",
                    "Stored trend data has no points inside the requested x range.",
                )
        try:
            relative_indices = select_trend_sample_indices(
                payload["key"][source_indices],
                payload["mean"][source_indices],
                max_points,
            )
        except ValueError as exc:
            raise ResultWorkspaceError(
                "plot_trend_invalid",
                str(exc),
                status_code=409,
            ) from exc
        selected = source_indices[np.asarray(relative_indices, dtype=np.int64)]
        return {
            "axis": str(row["axis"]),
            "key": payload["key"][selected].tolist(),
            "mean": payload["mean"][selected].astype(float).tolist(),
            "stderr": payload["stderr"][selected].astype(float).tolist(),
            "count": payload["count"][selected].astype(int).tolist(),
        }

    # -- SA imports -----------------------------------------------------

    def import_runsets(
        self,
        result_id: str,
        runset_uids: Sequence[str] | None = None,
        *,
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        return self.import_runset_selections(
            [{"result_id": result_id, "runset_uids": runset_uids}],
            allow_duplicates=allow_duplicates,
        )

    def import_runset_selections(
        self,
        selections: Sequence[Mapping[str, Any]],
        *,
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        """Validate and import Runsets selected across multiple result bundles."""

        if not selections:
            raise ResultWorkspaceError(
                "runset_selection_empty",
                "Select at least one Runset to add to the SA list.",
            )
        current_imports = self.list_imports()
        existing = {
            (item["result_id"], item["runset_uid"]): item
            for item in current_imports
        }
        planned: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
        ordered_items: list[tuple[str, Mapping[str, Any] | None]] = []
        skipped: list[dict[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        previews: dict[str, Any] = {}
        for selection in selections:
            result_id = str(selection.get("result_id") or "").strip()
            if not result_id:
                raise ResultWorkspaceError(
                    "result_id_required",
                    "Each Runset selection requires a result_id.",
                )
            if result_id not in previews:
                previews[result_id] = self.preview_run(result_id, force_verify=True)
            preview = previews[result_id]
            available = {item["runset_uid"]: item for item in preview["runsets"]}
            raw_uids = selection.get("runset_uids")
            selected = list(
                dict.fromkeys(
                    str(item) for item in (raw_uids or tuple(available))
                )
            )
            missing = [item for item in selected if item not in available]
            if missing:
                raise ResultWorkspaceError(
                    "runset_not_found",
                    f"Unknown result runset(s): {', '.join(missing)}",
                    status_code=404,
                )
            for uid in selected:
                pair = (result_id, uid)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                runset = available[uid]
                if (
                    runset["status"] != "completed"
                    or runset["completed_runs"] != runset["expected_runs"]
                ):
                    skipped.append(
                        {
                            "runset_uid": uid,
                            "label": runset["label"],
                            "reason": (
                                f"status={runset['status']}, runs="
                                f"{runset['completed_runs']}/{runset['expected_runs']}"
                            ),
                        }
                    )
                    continue
                if pair in existing:
                    ordered_items.append(("existing", existing[pair]))
                    continue
                planned.append((result_id, preview, runset))
                ordered_items.append(("planned", None))

        duplicate_warning = self._validate_candidate_addition(
            [item[2] for item in planned],
            current_imports,
            allow_duplicates=allow_duplicates,
        )
        now = utc_now()
        created: list[dict[str, Any]] = []
        for result_id, preview, runset in planned:
            uid = str(runset["runset_uid"])
            item = self.database.upsert_sa_import(
                {
                    "id": uuid4().hex,
                    "result_id": result_id,
                    "runset_uid": uid,
                    "bundle_id": preview["validation"]["bundle_id"],
                    "task_name": preview["task_name"],
                    "label": runset["label"],
                    "algorithm_name": runset["algorithm_name"],
                    "problem_name": runset["problem_name"],
                    "problem_params": runset["problem_params"],
                    "algorithm_params": runset["algorithm_params"],
                    "scalar_metrics": runset["scalar_metrics"],
                    "trend_metrics": runset["trend_metrics"],
                    "expected_runs": runset["expected_runs"],
                    "completed_runs": runset["completed_runs"],
                    "source_fingerprint": _runset_source_fingerprint(
                        preview["source_fingerprint"], uid
                    ),
                    "pairing_signature": runset["pairing_signature"],
                    "snapshot_sha256": preview["snapshot_sha256"] or "",
                    "task_sha256": preview["task_sha256"] or "",
                    "imported_at": now,
                }
            )
            created.append(item)
        created_iterator = iter(created)
        imported = [
            item if kind == "existing" else next(created_iterator)
            for kind, item in ordered_items
        ]
        already = len(ordered_items) - len(created)
        return {
            "imported": len(created),
            "already_imported": already,
            "skipped": len(skipped),
            "skipped_details": skipped,
            "items": imported,
            "duplicate_warning": duplicate_warning,
        }

    def list_imports(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in self.database.list_sa_imports():
            stored = self.database.get_result_import(item["result_id"])
            available = stored is not None
            if available:
                try:
                    self._bundle_path(self._result_from_record(stored))
                except ResultWorkspaceError:
                    available = False
            result.append(
                {
                    **item,
                    "task_name": (
                        str(stored["display_name"])
                        if stored is not None
                        else item["task_name"]
                    ),
                    "source_available": available,
                    "source_status": "completed" if available else "deleted",
                }
            )
        return result

    def delete_import(self, import_id: str) -> bool:
        return self.database.delete_sa_import(import_id)

    def clear_imports(self) -> int:
        return self.database.clear_sa_imports()

    def _validate_candidate_addition(
        self,
        selected: Sequence[Mapping[str, Any]],
        current: Sequence[Mapping[str, Any]],
        *,
        allow_duplicates: bool,
    ) -> bool:
        if not selected:
            return False
        template = current[0] if current else selected[0]
        incompatible = [
            item for item in selected if not _sa_base_compatible(template, item)
        ]
        if incompatible:
            names = ", ".join(str(item.get("label") or item["runset_uid"]) for item in incompatible)
            raise ResultWorkspaceError(
                "sa_candidate_incompatible",
                (
                    "SA runsets must share the same problem, scientific problem "
                    f"parameters, and completed run count. Incompatible: {names}."
                ),
                status_code=409,
            )
        duplicate_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        seen = list(current)
        for item in selected:
            duplicate = next(
                (candidate for candidate in seen if _same_algorithm_configuration(candidate, item)),
                None,
            )
            if duplicate is not None:
                duplicate_pairs.append((duplicate, item))
            seen.append(item)
        if duplicate_pairs and not allow_duplicates:
            raise ResultWorkspaceError(
                "sa_duplicate_confirmation_required",
                (
                    "One or more runsets use an algorithm configuration already "
                    "present in the SA list. Retry with allow_duplicates=true to add them."
                ),
                status_code=409,
            )
        return bool(duplicate_pairs)

    def match_compatible_imports(
        self,
        *,
        base_import_id: str | None = None,
        method: str = "friedman",
    ) -> dict[str, Any]:
        normalized_method = _normalize_method(method)
        imports = [item for item in self.list_imports() if item["source_available"]]
        minimum = 3 if normalized_method == "friedman" else 2
        paired = normalized_method in {"friedman", "signed-rank"}
        if base_import_id:
            base = next((item for item in imports if item["id"] == base_import_id), None)
            if base is None:
                raise ResultWorkspaceError(
                    "sa_import_not_found",
                    f"SA import was not found or its source is unavailable: {base_import_id}",
                    status_code=404,
                )
            selected = [
                item
                for item in imports
                if _sa_base_compatible(base, item)
                and (not paired or item["pairing_signature"] == base["pairing_signature"])
            ]
        else:
            groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
            for item in imports:
                groups[_sa_method_group_key(item, paired=paired)].append(item)
            candidates = list(groups.values())
            # Prefer the largest group that actually shares a usable metric.  This
            # avoids selecting a larger but scientifically unusable group and then
            # reporting no common metric while a valid group exists.
            usable = [
                group
                for group in candidates
                if len(group) >= minimum and _common_scalar_metrics(group)
            ]
            selected = max(usable, key=len) if usable else []
        metrics = _common_scalar_metrics(selected)
        return {
            "method": normalized_method,
            "minimum": minimum,
            "imports": selected,
            "import_ids": [item["id"] for item in selected],
            "metrics": metrics,
            "compatible": len(selected) >= minimum and bool(metrics),
            "message": (
                f"Matched {len(selected)} compatible runset(s)."
                if selected
                else f"No compatible group has at least {minimum} runsets."
            ),
            "warnings": _source_version_warnings(selected),
        }

    def quick_filter_imports(
        self, *, allow_duplicates: bool = False
    ) -> dict[str, Any]:
        """Populate SA from every compatible runset in the imported result library.

        GUIMode requires one manually chosen SA task first; its benchmark,
        complete benchmark parameter set, and run number become the template.
        """

        current = self.list_imports()
        if not current:
            raise ResultWorkspaceError(
                "sa_template_required",
                "Add at least one runset to the SA list before using quick filter.",
                status_code=409,
            )
        template = min(
            current,
            key=lambda item: (str(item.get("imported_at") or ""), str(item["id"])),
        )
        existing = {(item["result_id"], item["runset_uid"]): item for item in current}
        already: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for catalog_item in self.list_result_catalog():
            result_id = str(catalog_item["result_id"])
            preview = self.preview_run(result_id)
            for runset in preview["runsets"]:
                descriptor = {**runset, "result_id": result_id}
                key = (result_id, str(runset["runset_uid"]))
                if key in existing:
                    already.append(existing[key])
                    continue
                if (
                    runset["status"] != "completed"
                    or runset["completed_runs"] != runset["expected_runs"]
                ):
                    skipped.append({**descriptor, "reason": "incomplete"})
                    continue
                if not _sa_base_compatible(template, runset):
                    skipped.append({**descriptor, "reason": "incompatible"})
                    continue
                pending.append((result_id, preview, runset))

        # Scan and validate the complete candidate set before the first write.
        # This keeps the initial duplicate-confirmation response atomic: a
        # compatible non-duplicate encountered earlier in the catalog is not
        # partially added before a later duplicate is discovered.
        duplicate_warning = self._validate_candidate_addition(
            [runset for _, _, runset in pending],
            current,
            allow_duplicates=allow_duplicates,
        )

        added: list[dict[str, Any]] = []
        for result_id, _, runset in pending:
            response = self.import_runsets(
                result_id,
                [str(runset["runset_uid"])],
                allow_duplicates=True,
            )
            for item in response["items"]:
                key = (item["result_id"], item["runset_uid"])
                if key not in existing:
                    existing[key] = item
                    added.append(item)
        return {
            "template_import_id": template["id"],
            "added": len(added),
            "already": len(already),
            "skipped": len(skipped),
            "items": added,
            "already_items": already,
            "skipped_items": skipped,
            "duplicate_warning": duplicate_warning,
        }

    # -- Statistical analyses -----------------------------------------

    def run_analysis(
        self,
        *,
        name: str,
        method: str,
        import_ids: Sequence[str],
        metrics: Sequence[str] | None = None,
        alpha: float = 0.05,
        formats: Sequence[str] | None = None,
        label_parameter: str | None = None,
    ) -> dict[str, Any]:
        analysis_name = str(name).strip()
        if not analysis_name:
            raise ResultWorkspaceError("invalid_analysis_name", "SA name is required.")
        if len(analysis_name) > 200:
            raise ResultWorkspaceError(
                "invalid_analysis_name", "SA name must be at most 200 characters."
            )
        normalized_method = _normalize_method(method)
        alpha_value = float(alpha)
        if not 0.0 < alpha_value < 1.0:
            raise ResultWorkspaceError("invalid_alpha", "Alpha must be between 0 and 1.")
        normalized_ids = list(dict.fromkeys(str(item) for item in import_ids if str(item)))
        minimum = 3 if normalized_method == "friedman" else 2
        if len(normalized_ids) < minimum:
            raise ResultWorkspaceError(
                "insufficient_runsets",
                f"{normalized_method} requires at least {minimum} imported runsets.",
            )
        try:
            imports = self.database.get_sa_imports(normalized_ids)
        except KeyError as exc:
            raise ResultWorkspaceError(
                "sa_import_not_found", str(exc), status_code=404
            ) from exc
        self._validate_import_group(imports, normalized_method)
        common_metrics = _common_scalar_metrics(imports)
        metric_names = list(dict.fromkeys(str(item) for item in (metrics or common_metrics)))
        if not metric_names:
            raise ResultWorkspaceError(
                "no_common_metrics", "Selected runsets have no common directional scalar metric."
            )
        unavailable = [item for item in metric_names if item not in common_metrics]
        if unavailable:
            raise ResultWorkspaceError(
                "metric_not_available",
                f"Metric(s) are not available with one consistent direction: {', '.join(unavailable)}",
            )
        output_formats = list(dict.fromkeys(str(item) for item in (formats or ("sqlite", "csv"))))
        unknown_formats = sorted(set(output_formats) - SA_OUTPUT_FORMATS)
        if unknown_formats:
            raise ResultWorkspaceError(
                "invalid_output_format",
                f"Unknown SA output format(s): {', '.join(unknown_formats)}",
            )

        label_dimension, display_labels = _analysis_display_labels(
            imports,
            label_parameter=label_parameter,
        )

        analysis_root = self._analysis_root()
        analysis_id = uuid4().hex
        created_at = utc_now()
        source_snapshot = self._analysis_source_snapshot(
            imports,
            label_dimension=label_dimension,
            display_labels=display_labels,
        )
        record = self.database.create_sa_analysis(
            {
                "id": analysis_id,
                "name": analysis_name,
                "status": "running",
                "method": normalized_method,
                "alpha": alpha_value,
                "metric_names": metric_names,
                "source_imports": source_snapshot,
                "created_at": created_at,
            }
        )
        output_dir = analysis_root / analysis_id
        try:
            if output_dir.exists() or output_dir.is_symlink():
                raise ResultWorkspaceError(
                    "analysis_output_conflict",
                    f"SA output directory already exists: {output_dir}",
                    status_code=409,
                )
            output_dir.mkdir(mode=0o700, parents=False)
            samples_by_metric: dict[str, list[ScalarRunSet]] = {}
            paired = normalized_method in {"friedman", "signed-rank"}
            for metric_name in metric_names:
                samples_by_metric[metric_name] = [
                    self._load_scalar_runset(
                        item,
                        metric_name,
                        require_pairing_signature=paired,
                    )
                    for item in imports
                ]
            payload = run_statistical_analysis_from_samples(
                method=normalized_method,
                samples_by_metric=samples_by_metric,
                alpha=alpha_value,
                runset_uids=[item["runset_uid"] for item in imports],
                comparison_labels=display_labels,
            )
            payload.update(
                {
                    "analysis_id": analysis_id,
                    "name": analysis_name,
                    "label_dimension": label_dimension,
                    "display_labels": [
                        display_labels[str(item["id"])] for item in imports
                    ],
                    "selected_runsets": source_snapshot,
                }
            )
            written = write_analysis_outputs(output_dir, payload, output_formats)
            relative_written = {
                key: self._relative_analysis_path(Path(value))
                for key, value in written.items()
            }
            record = self.database.update_sa_analysis(
                analysis_id,
                status="completed",
                result=payload,
                output_dir=f"analyses/{analysis_id}",
                output_files=relative_written,
                completed_at=utc_now(),
                error_message=None,
            )
        except Exception as exc:
            self.database.update_sa_analysis(
                analysis_id,
                status="failed",
                output_dir=f"analyses/{analysis_id}" if output_dir.exists() else None,
                error_message=str(exc),
                completed_at=utc_now(),
            )
            if isinstance(exc, ResultWorkspaceError):
                raise
            raise ResultWorkspaceError("analysis_failed", str(exc)) from exc
        return record

    def _analysis_source_snapshot(
        self,
        imports: Sequence[Mapping[str, Any]],
        *,
        label_dimension: Mapping[str, Any],
        display_labels: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        previews: dict[str, dict[str, Any]] = {}
        snapshots: list[dict[str, Any]] = []
        for item in imports:
            result_id = str(item["result_id"])
            preview = previews.get(result_id)
            if preview is None:
                preview = self.preview_run(result_id)
                previews[result_id] = preview
            runset_uid = str(item["runset_uid"])
            runset = next(
                (
                    candidate
                    for candidate in preview["runsets"]
                    if str(candidate["runset_uid"]) == runset_uid
                ),
                None,
            )
            if runset is None:
                raise ResultWorkspaceError(
                    "runset_not_found",
                    f"Result runset was not found while freezing SA source: {runset_uid}",
                    status_code=404,
                )
            snapshots.append(
                {
                    **runset,
                    "import_id": str(item["id"]),
                    "result_id": result_id,
                    "run_id": result_id,
                    "bundle_id": str(item["bundle_id"]),
                    "task_name": str(item["task_name"]),
                    "source_fingerprint": str(item["source_fingerprint"]),
                    "snapshot_sha256": str(item["snapshot_sha256"]),
                    "task_sha256": str(item["task_sha256"]),
                    "imported_at": str(item["imported_at"]),
                    "label_dimension": dict(label_dimension),
                    "display_label": display_labels[str(item["id"])],
                }
            )
        return snapshots

    def list_analyses(self) -> list[dict[str, Any]]:
        return self.database.list_sa_analyses()

    def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        record = self.database.get_sa_analysis(analysis_id)
        if record is None:
            raise ResultWorkspaceError(
                "analysis_not_found",
                f"SA record was not found: {analysis_id}",
                status_code=404,
            )
        return record

    def delete_analysis(self, analysis_id: str) -> bool:
        record = self.get_analysis(analysis_id)
        if not _is_safe_path_component(analysis_id):
            raise ResultWorkspaceError(
                "unsafe_analysis_path",
                "Refusing to delete an SA output with an unsafe analysis id.",
                status_code=409,
            )
        configured = record.get("output_dir")
        tombstone: Path | None = None
        candidate: Path | None = None
        if configured:
            analysis_root = self._analysis_root()
            expected_relative = Path("analyses") / analysis_id
            if Path(str(configured)) != expected_relative:
                raise ResultWorkspaceError(
                    "unsafe_analysis_path",
                    "Refusing to delete an SA output outside the exact analysis directory.",
                    status_code=409,
                )
            candidate = analysis_root / analysis_id
            if candidate.is_symlink():
                raise ResultWorkspaceError(
                    "unsafe_analysis_path",
                    "Refusing to delete a symlinked SA output directory.",
                    status_code=409,
                )
            if candidate.exists():
                if not candidate.is_dir():
                    raise ResultWorkspaceError(
                        "unsafe_analysis_path",
                        "Refusing to delete an SA output that is not a directory.",
                        status_code=409,
                    )
                trash_root = self._trash_root()
                tombstone = trash_root / f"analysis-{analysis_id}-{uuid4().hex}.trash"
                candidate.rename(tombstone)
        try:
            deleted = self.database.delete_sa_analysis(analysis_id)
        except BaseException:
            self._restore_analysis_tombstone(candidate, tombstone)
            raise
        if not deleted:
            self._restore_analysis_tombstone(candidate, tombstone)
            return False
        if tombstone is not None:
            trash_root = self._trash_root()
            if (
                tombstone.parent != trash_root
                or tombstone.is_symlink()
                or not tombstone.is_dir()
            ):
                raise ResultWorkspaceError(
                    "unsafe_analysis_path",
                    "Refusing to remove an unsafe SA deletion tombstone.",
                    status_code=409,
                )
            shutil.rmtree(tombstone)
        return True

    # -- Internal helpers ----------------------------------------------

    def _result_from_record(self, item: Mapping[str, Any]) -> _ImportedResult:
        metadata = item.get("source_metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        return _ImportedResult(
            id=str(item["id"]),
            task_name=str(item["display_name"]),
            task_path=(
                str(metadata["task_path"])
                if metadata.get("task_path") is not None
                else None
            ),
            status=RunStatus.COMPLETED,
            exit_code=0,
            created_at=str(item["imported_at"]),
            finished_at=str(item["imported_at"]),
            snapshot_sha256=(
                str(metadata["snapshot_sha256"])
                if metadata.get("snapshot_sha256") is not None
                else None
            ),
            task_sha256=(
                str(metadata["task_sha256"])
                if metadata.get("task_sha256") is not None
                else None
            ),
            local_artifact_dir=str(item["managed_relative_dir"]),
            source_kind=str(item["source_kind"]),
            source_run_id=(
                str(item["source_run_id"])
                if item.get("source_run_id") is not None
                else None
            ),
            source_metadata=metadata,
            bundle_id=str(item["bundle_id"]),
            validation=(
                item["validation"]
                if isinstance(item.get("validation"), Mapping)
                else {}
            ),
            summary_size_bytes=int(item["summary_size_bytes"]),
            summary_sha256=str(item["summary_sha256"]),
            trend_size_bytes=int(item["trend_size_bytes"]),
            trend_sha256=str(item["trend_sha256"]),
        )

    def _require_run(self, result_id: str) -> _ImportedResult:
        item = self.database.get_result_import(str(result_id))
        if item is None:
            raise ResultWorkspaceError(
                "result_not_found",
                f"Imported result was not found: {result_id}",
                status_code=404,
            )
        return self._result_from_record(item)

    def _bundle_path(self, run: _ImportedResult, *, force_verify: bool = False) -> Path:
        if not _is_safe_path_component(run.id):
            raise ResultWorkspaceError(
                "unsafe_result_path",
                "The imported result identity is unsafe.",
                status_code=409,
            )

        data_root = _absolute_without_symlink_resolution(self.paths.data_root)
        if data_root.is_symlink() or not data_root.is_dir():
            raise ResultWorkspaceError(
                "unsafe_result_path",
                "The Relay data directory is missing or symlinked.",
                status_code=409,
            )
        configured = Path(run.local_artifact_dir)
        imported_root = _absolute_without_symlink_resolution(
            self.paths.imported_results
        )
        analyses_root = _absolute_without_symlink_resolution(self.paths.analyses)
        runs_root = _absolute_without_symlink_resolution(self.paths.runs)
        expected_relative = Path("analyses") / "results" / run.id
        expected_absolute = imported_root / run.id
        legacy_owner_id = run.source_run_id or run.id
        legacy_relative = Path("runs") / legacy_owner_id / "remote" / "results" / "bundle"
        legacy_absolute = (
            _absolute_without_symlink_resolution(self.paths.runs)
            / legacy_owner_id
            / "remote"
            / "results"
            / "bundle"
        )
        if configured in {expected_relative, expected_absolute}:
            if (
                analyses_root != data_root / "analyses"
                or analyses_root.is_symlink()
                or not analyses_root.is_dir()
                or imported_root != analyses_root / "results"
                or imported_root.is_symlink()
                or not imported_root.is_dir()
            ):
                raise ResultWorkspaceError(
                    "unsafe_result_path",
                    "The imported-result library contains a missing or symlinked directory.",
                    status_code=409,
                )
            bundle = expected_absolute
        elif run.source_kind == "legacy_run" and configured in {
            legacy_relative,
            legacy_absolute,
        }:
            legacy_root = runs_root / legacy_owner_id
            legacy_chain = (
                runs_root,
                legacy_root,
                legacy_root / "remote",
                legacy_root / "remote" / "results",
            )
            if runs_root != data_root / "runs" or any(
                directory.is_symlink() or not directory.is_dir()
                for directory in legacy_chain
            ):
                raise ResultWorkspaceError(
                    "unsafe_result_path",
                    "The legacy result path contains a missing or symlinked directory.",
                    status_code=409,
                )
            bundle = legacy_absolute
        else:
            raise ResultWorkspaceError(
                "unsafe_result_path",
                "The local result path is outside the exact imported-result directory.",
                status_code=409,
            )
        try:
            bundle.relative_to(data_root)
        except ValueError as exc:
            raise ResultWorkspaceError(
                "unsafe_result_path",
                "The imported result path escaped the Relay data directory.",
                status_code=409,
            ) from exc
        if bundle.is_symlink():
            raise ResultWorkspaceError(
                "unsafe_result_path",
                "The verified imported result bundle is symlinked.",
                status_code=409,
            )
        if not bundle.is_dir():
            raise ResultWorkspaceError(
                "result_not_found",
                "The verified local result bundle is missing or unsafe.",
                status_code=404,
            )
        for filename in ("summary.sqlite", "trend.sqlite"):
            path = bundle / filename
            if path.is_symlink():
                raise ResultWorkspaceError(
                    "unsafe_result_path",
                    f"The verified result file is symlinked: {filename}",
                    status_code=409,
                )
            if not path.is_file():
                raise ResultWorkspaceError(
                    "result_not_found",
                    f"The verified result file is missing or unsafe: {filename}",
                    status_code=404,
                )
        self._verify_bundle(run, bundle, force=force_verify)
        return bundle

    @staticmethod
    def _bundle_signature(bundle: Path) -> tuple:
        # ctime catches same-size writes even when mtime is restored. Sidecars
        # must not be present: imported bundles are immutable, checkpointed DBs.
        signatures = []
        for filename in ("summary.sqlite", "trend.sqlite"):
            path = bundle / filename
            if path.is_symlink() or any((bundle / (filename + suffix)).exists() for suffix in ("-wal", "-journal")):
                raise ResultWorkspaceError("unsafe_result_path", "Imported databases must be immutable regular files without journals.", status_code=409)
            stat = path.stat()
            signatures.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return tuple(signatures)

    def _verify_bundle(self, run: _ImportedResult, bundle: Path, *, force: bool = False):
        required_artifacts = {
            "summary.sqlite": (
                bundle / "summary.sqlite",
                run.summary_size_bytes,
                run.summary_sha256,
            ),
            "trend.sqlite": (
                bundle / "trend.sqlite",
                run.trend_size_bytes,
                run.trend_sha256,
            ),
        }
        # Serialize cold verification so polling requests share the completed
        # result instead of concurrently scanning multi-GB trend databases.
        with self._verification_lock:
            before = self._bundle_signature(bundle)
            key = (run.id, str(bundle), run.task_sha256, run.snapshot_sha256,
                   run.summary_sha256, run.summary_size_bytes,
                   run.trend_sha256, run.trend_size_bytes, before)
            if not force and key in self._verification_cache:
                metrics.increment("result_verification.cache_hit")
                self._verification_cache.move_to_end(key)
                return self._verification_cache[key]
            metrics.increment("result_verification.forced" if force else "result_verification.cache_miss")
            with metrics.measure("result_verification.full"):
                for filename, (local_path, expected_size, expected_digest) in required_artifacts.items():
                    if expected_size != local_path.stat().st_size or expected_digest != _sha256_file(local_path):
                        raise ResultWorkspaceError("result_checksum_changed", f"Verified result artifact changed on disk: {filename}", status_code=409)
                validation = validate_result_bundle(bundle, require_completed=False)
            if self._bundle_signature(bundle) != before:
                raise ResultWorkspaceError(
                    "result_checksum_changed",
                    "Result files changed during verification.",
                    status_code=409,
                )
            self._verification_cache[key] = validation
            self._verification_cache.move_to_end(key)
            while len(self._verification_cache) > self._verification_cache_limit:
                self._verification_cache.popitem(last=False)
            return validation

    def _source_fingerprint(self, run: _ImportedResult, bundle: Path) -> str:
        self._verify_bundle(run, bundle)
        summary_digest = run.summary_sha256
        trend_digest = run.trend_sha256
        return hashlib.sha256(
            stable_json_dumps(
                {
                    "result_id": run.id,
                    "task_sha256": run.task_sha256,
                    "snapshot_sha256": run.snapshot_sha256,
                    "summary_sha256": summary_digest,
                    "trend_sha256": trend_digest,
                }
            ).encode("utf-8")
        ).hexdigest()

    def _validate_import_group(
        self, imports: Sequence[Mapping[str, Any]], method: str
    ) -> None:
        problem_keys = {
            (
                str(item["problem_name"]),
                _sa_problem_params_key(item["problem_params"]),
                int(item["completed_runs"]),
            )
            for item in imports
        }
        if len(problem_keys) != 1:
            raise ResultWorkspaceError(
                "incompatible_runsets",
                (
                    "Selected runsets must share the same problem, scientific problem "
                    "parameters, and completed run count."
                ),
            )
        if method in {"friedman", "signed-rank"}:
            signatures = {str(item["pairing_signature"]) for item in imports}
            if len(signatures) != 1:
                raise ResultWorkspaceError(
                    "paired_seed_mismatch",
                    "Paired SA requires the same ordered run indices and problem seeds.",
                )
        verified_ids: set[str] = set()
        for item in imports:
            run = self._require_run(str(item["result_id"]))
            bundle = self._bundle_path(run, force_verify=run.id not in verified_ids)
            verified_ids.add(run.id)
            current = _runset_source_fingerprint(
                self._source_fingerprint(run, bundle), str(item["runset_uid"])
            )
            if current != item["source_fingerprint"]:
                raise ResultWorkspaceError(
                    "result_source_changed",
                    f"Imported result source changed for {item['task_name']}:{item['label']}.",
                    status_code=409,
                )

    def _load_scalar_runset(
        self,
        item: Mapping[str, Any],
        metric_name: str,
        *,
        require_pairing_signature: bool,
    ) -> ScalarRunSet:
        run = self._require_run(str(item["result_id"]))
        bundle = self._bundle_path(run)
        with _open_read_only(bundle / "summary.sqlite") as summary:
            rows = summary.execute(
                """
                SELECT r.run_index, r.problem_seed, rmv.value, rmv.direction
                FROM run_metric_values AS rmv
                JOIN runs AS r ON r.id = rmv.run_id
                JOIN runsets AS rs ON rs.id = rmv.runset_id
                WHERE rs.runset_uid = ? AND rmv.metric_name = ?
                ORDER BY r.run_index
                """,
                (str(item["runset_uid"]), metric_name),
            ).fetchall()
        if not rows:
            raise ResultWorkspaceError(
                "metric_not_available",
                f"Metric {metric_name!r} is missing from {item['task_name']}:{item['label']}.",
            )
        if require_pairing_signature:
            metric_pairing_signature = _ordered_pairing_signature(rows)
            if metric_pairing_signature != str(item["pairing_signature"]):
                raise ResultWorkspaceError(
                    "paired_metric_samples_mismatch",
                    (
                        f"Metric {metric_name!r} does not contain one value for every "
                        "ordered paired (run_index, problem_seed) sample in "
                        f"{item['task_name']}:{item['label']}."
                    ),
                )
        directions = {str(row["direction"]) for row in rows}
        if len(directions) != 1 or next(iter(directions)) not in {"lower", "higher"}:
            raise ResultWorkspaceError(
                "metric_direction_invalid",
                f"Metric {metric_name!r} has no single lower/higher direction.",
            )
        values = [float(row["value"]) for row in rows]
        return ScalarRunSet(
            id=str(item["id"]),
            label=(
                f"{item['task_name']} · {item['label']} · {item['algorithm_name']}"
            ),
            algorithm=str(item["algorithm_name"]),
            metric_name=metric_name,
            values=values,
            direction=next(iter(directions)),
        )

    def _relative_analysis_path(self, path: Path) -> str:
        analysis_root = self._analysis_root()
        candidate = _absolute_without_symlink_resolution(path)
        try:
            relative_to_analysis = candidate.relative_to(analysis_root)
        except ValueError as exc:
            raise ResultWorkspaceError(
                "unsafe_analysis_path",
                f"SA output escaped the Relay analysis directory: {candidate}",
            ) from exc
        current = analysis_root
        for index, part in enumerate(relative_to_analysis.parts):
            current = current / part
            if current.is_symlink():
                raise ResultWorkspaceError(
                    "unsafe_analysis_path",
                    f"SA output path contains a symlink: {current}",
                )
            if index < len(relative_to_analysis.parts) - 1 and not current.is_dir():
                raise ResultWorkspaceError(
                    "unsafe_analysis_path",
                    f"SA output parent is not a directory: {current}",
                )
        if not candidate.is_file():
            raise ResultWorkspaceError(
                "unsafe_analysis_path",
                f"SA output is not a regular file: {candidate}",
            )
        data_root = _absolute_without_symlink_resolution(self.paths.data_root)
        return candidate.relative_to(data_root).as_posix()

    def _analysis_root(self) -> Path:
        data_root = _absolute_without_symlink_resolution(self.paths.data_root)
        analysis_root = _absolute_without_symlink_resolution(self.paths.analyses)
        if data_root.is_symlink() or not data_root.is_dir():
            raise ResultWorkspaceError(
                "unsafe_analysis_path",
                "The Relay data directory is missing or symlinked.",
                status_code=409,
            )
        if (
            analysis_root != data_root / "analyses"
            or analysis_root.is_symlink()
            or not analysis_root.is_dir()
        ):
            raise ResultWorkspaceError(
                "unsafe_analysis_path",
                "The Relay analysis directory is missing or unsafe.",
                status_code=409,
            )
        return analysis_root

    def _trash_root(self) -> Path:
        data_root = _absolute_without_symlink_resolution(self.paths.data_root)
        trash_root = _absolute_without_symlink_resolution(self.paths.trash)
        if data_root.is_symlink() or not data_root.is_dir():
            raise ResultWorkspaceError(
                "unsafe_analysis_path",
                "The Relay data directory is missing or symlinked.",
                status_code=409,
            )
        if (
            trash_root != data_root / "trash"
            or trash_root.is_symlink()
            or not trash_root.is_dir()
        ):
            raise ResultWorkspaceError(
                "unsafe_analysis_path",
                "The Relay trash directory is missing or unsafe.",
                status_code=409,
            )
        return trash_root

    def _restore_analysis_tombstone(
        self,
        candidate: Path | None,
        tombstone: Path | None,
    ) -> None:
        if candidate is None or tombstone is None:
            return
        try:
            analysis_root = self._analysis_root()
            trash_root = self._trash_root()
        except ResultWorkspaceError as exc:
            raise ResultWorkspaceError(
                "analysis_delete_restore_failed",
                "SA deletion failed and its output directory could not be restored safely.",
                status_code=500,
            ) from exc
        if candidate.exists() and not candidate.is_symlink():
            if tombstone.exists() or tombstone.is_symlink():
                raise ResultWorkspaceError(
                    "analysis_delete_restore_failed",
                    "SA deletion failed and both live and tombstone outputs exist.",
                    status_code=500,
                )
            return
        if (
            candidate.parent != analysis_root
            or tombstone.parent != trash_root
            or tombstone.is_symlink()
            or not tombstone.is_dir()
            or candidate.is_symlink()
        ):
            raise ResultWorkspaceError(
                "analysis_delete_restore_failed",
                "SA deletion failed and its output directory could not be restored safely.",
                status_code=500,
            )
        tombstone.rename(candidate)


def _open_read_only(path: Path) -> sqlite3.Connection:
    absolute = _absolute_without_symlink_resolution(path)
    connection = sqlite3.connect(f"{absolute.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _read_tasks(summary: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = summary.execute(
        """
        SELECT id, task_uid, label, source_task_file, source_line,
               task_config_hash, problem_name, run_number,
               problem_params_json, status, created_at, completed_at,
               error_message
        FROM tasks
        ORDER BY id
        """
    ).fetchall()
    return [
        {
            "task_id": int(row["id"]),
            "task_uid": str(row["task_uid"]),
            "label": str(row["label"]),
            "source_task_file": row["source_task_file"],
            "source_line": row["source_line"],
            "task_config_hash": str(row["task_config_hash"]),
            "problem_name": str(row["problem_name"]),
            "run_number": int(row["run_number"]),
            "problem_params": json.loads(row["problem_params_json"]),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
        }
        for row in rows
    ]


def _read_runsets(
    summary: sqlite3.Connection, trend: sqlite3.Connection
) -> list[dict[str, Any]]:
    rows = summary.execute(
        """
        SELECT rs.id, rs.runset_uid, rs.task_id, t.task_uid,
               rs.label, rs.algorithm_name, rs.problem_name,
               rs.algorithm_params_json, rs.expected_runs,
               rs.completed_runs, rs.status, t.problem_params_json,
               t.source_task_file, t.source_line, t.task_config_hash
        FROM runsets AS rs
        JOIN tasks AS t ON t.id = rs.task_id
        ORDER BY t.id, rs.id
        """
    ).fetchall()
    summary_rows = summary.execute(
        """
        SELECT runset_uid, metric_name, mean, median, stderr, count, direction
        FROM summary_metrics
        ORDER BY runset_uid, metric_name
        """
    ).fetchall()
    scalar_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        scalar_by_uid[str(row["runset_uid"])].append(
            {
                "name": str(row["metric_name"]),
                "mean": _finite_or_none(row["mean"]),
                "median": _finite_or_none(row["median"]),
                "stderr": _finite_or_none(row["stderr"]),
                "count": int(row["count"]),
                "direction": str(row["direction"]),
            }
        )
    trend_rows = trend.execute(
        """
        SELECT runset_uid, metric_name, axis, sample_count
        FROM trend_metrics
        ORDER BY runset_uid, metric_name
        """
    ).fetchall()
    trend_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trend_rows:
        trend_by_uid[str(row["runset_uid"])].append(
            {
                "name": str(row["metric_name"]),
                "axis": str(row["axis"]),
                "sample_count": int(row["sample_count"]),
            }
        )
    pairing_by_id: dict[int, str] = {}
    for row in rows:
        pairs = summary.execute(
            """
            SELECT run_index, problem_seed
            FROM runs
            WHERE runset_id = ?
            ORDER BY run_index
            """,
            (int(row["id"]),),
        ).fetchall()
        pairing_by_id[int(row["id"])] = _ordered_pairing_signature(pairs)
    return [
        {
            "runset_id": int(row["id"]),
            "runset_uid": str(row["runset_uid"]),
            "task_id": int(row["task_id"]),
            "task_uid": str(row["task_uid"]),
            "label": str(row["label"]),
            "algorithm_name": str(row["algorithm_name"]),
            "problem_name": str(row["problem_name"]),
            "algorithm_params": json.loads(row["algorithm_params_json"]),
            "problem_params": json.loads(row["problem_params_json"]),
            "expected_runs": int(row["expected_runs"]),
            "completed_runs": int(row["completed_runs"]),
            "status": str(row["status"]),
            "source_task_file": row["source_task_file"],
            "source_line": row["source_line"],
            "task_config_hash": str(row["task_config_hash"]),
            "scalar_metrics": scalar_by_uid[str(row["runset_uid"])],
            "trend_metrics": trend_by_uid[str(row["runset_uid"])],
            "pairing_signature": pairing_by_id[int(row["id"])],
        }
        for row in rows
    ]


def _analysis_display_labels(
    items: Sequence[Mapping[str, Any]],
    *,
    label_parameter: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    requested_parameter = (
        str(label_parameter).strip() if label_parameter is not None else ""
    )
    if len(requested_parameter) > 256:
        raise ResultWorkspaceError(
            "invalid_label_parameter",
            "SA label parameter must be at most 256 characters.",
        )

    algorithms = {str(item["algorithm_name"]) for item in items}
    parameter_maps = [
        item["algorithm_params"]
        if isinstance(item.get("algorithm_params"), Mapping)
        else {}
        for item in items
    ]
    parameter_name: str | None = None
    if len(algorithms) == 1:
        common_parameters = set(parameter_maps[0]) if parameter_maps else set()
        for parameters in parameter_maps[1:]:
            common_parameters.intersection_update(parameters)
        variable_parameters = sorted(
            str(key)
            for key in common_parameters
            if len(
                {
                    stable_json_dumps(parameters[key])
                    for parameters in parameter_maps
                }
            )
            > 1
        )
        if requested_parameter:
            if requested_parameter not in common_parameters:
                raise ResultWorkspaceError(
                    "label_parameter_not_common",
                    (
                        f"Algorithm parameter {requested_parameter!r} is not present "
                        "in every selected runset."
                    ),
                )
            if requested_parameter not in variable_parameters:
                raise ResultWorkspaceError(
                    "label_parameter_not_variable",
                    (
                        f"Algorithm parameter {requested_parameter!r} does not vary "
                        "across the selected runsets."
                    ),
                )
            parameter_name = requested_parameter
        elif variable_parameters:
            parameter_name = variable_parameters[0]

    if parameter_name is None:
        label_dimension: dict[str, Any] = {
            "kind": "algorithm",
            "parameter": None,
            "title": "Algorithm",
        }
        base_labels = [str(item["algorithm_name"]) for item in items]
    else:
        label_dimension = {
            "kind": "algorithm_parameter",
            "parameter": parameter_name,
            "title": parameter_name,
        }
        base_labels = [
            _display_parameter_value(parameters[parameter_name])
            for parameters in parameter_maps
        ]

    counts: dict[str, int] = defaultdict(int)
    for label in base_labels:
        counts[label] += 1
    qualified_labels: list[str] = []
    for item, label in zip(items, base_labels, strict=True):
        if counts[label] > 1:
            label = f"{label} · {str(item['runset_uid'])[:10]}"
        qualified_labels.append(label)

    qualified_counts: dict[str, int] = defaultdict(int)
    for label in qualified_labels:
        qualified_counts[label] += 1
    source_qualified_labels = [
        (
            f"{label} · import {str(item['id'])[:10]}"
            if qualified_counts[label] > 1
            else label
        )
        for item, label in zip(items, qualified_labels, strict=True)
    ]

    used_labels: set[str] = set()
    display_labels: dict[str, str] = {}
    for item, label in zip(items, source_qualified_labels, strict=True):
        unique_label = label
        suffix = 2
        while unique_label in used_labels:
            unique_label = f"{label} · {suffix}"
            suffix += 1
        used_labels.add(unique_label)
        display_labels[str(item["id"])] = unique_label
    return label_dimension, display_labels


def _display_parameter_value(value: Any) -> str:
    if isinstance(value, str):
        return value if value else '\"\"'
    return stable_json_dumps(value)


def _common_scalar_metrics(items: Sequence[Mapping[str, Any]]) -> list[str]:
    if not items:
        return []
    metric_maps: list[dict[str, str]] = []
    for item in items:
        mapping = {
            str(metric["name"]): str(metric["direction"])
            for metric in item.get("scalar_metrics", [])
            if isinstance(metric, Mapping)
            and str(metric.get("direction")) in {"lower", "higher"}
        }
        metric_maps.append(mapping)
    common = set(metric_maps[0])
    for mapping in metric_maps[1:]:
        common.intersection_update(mapping)
    return sorted(
        name
        for name in common
        if len({mapping[name] for mapping in metric_maps}) == 1
    )


# These values select the execution backend rather than changing the benchmark
# landscape.  They are retained in each runset's frozen parameters and source
# snapshot, but must not make otherwise identical scientific problem
# configurations ineligible for SA.
_SA_EXECUTION_PROBLEM_PARAMS = frozenset({"EvaluationDevice"})


def _sa_scientific_problem_params(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if str(key) not in _SA_EXECUTION_PROBLEM_PARAMS
    }


def _sa_problem_params_key(value: Mapping[str, Any]) -> str:
    return stable_json_dumps(_sa_scientific_problem_params(value))


def _sa_base_compatible(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """GUIMode benchmark + scientific parameters + run-number compatibility.

    Execution-backend parameters (currently ``EvaluationDevice``) are not part
    of the scientific benchmark definition.  A CPU and a GPU/auto run of the
    same problem therefore remain comparable; their original parameters stay
    in the source snapshot for provenance.
    """

    return (
        str(left["problem_name"]) == str(right["problem_name"])
        and _sa_problem_params_key(left["problem_params"])
        == _sa_problem_params_key(right["problem_params"])
        and int(left["completed_runs"]) == int(right["completed_runs"])
    )


def _same_algorithm_configuration(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return (
        str(left["algorithm_name"]) == str(right["algorithm_name"])
        and stable_json_dumps(left["algorithm_params"])
        == stable_json_dumps(right["algorithm_params"])
    )


def _sa_method_group_key(
    item: Mapping[str, Any], *, paired: bool
) -> tuple[str, str, int, str]:
    return (
        str(item["problem_name"]),
        _sa_problem_params_key(item["problem_params"]),
        int(item["completed_runs"]),
        str(item["pairing_signature"]) if paired else "unpaired",
    )


def _source_version_warnings(items: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    execution_values = {
        stable_json_dumps(
            {
                key: item.get("problem_params", {}).get(key)
                for key in _SA_EXECUTION_PROBLEM_PARAMS
                if key in item.get("problem_params", {})
            }
        )
        for item in items
    }
    if len(execution_values) > 1:
        warnings.append(
            "Selected runsets use different evaluation devices; this does not change "
            "scientific problem compatibility and is retained in the SA provenance."
        )
    snapshots = {str(item.get("snapshot_sha256") or "") for item in items}
    snapshots.discard("")
    if len(snapshots) > 1:
        warnings.append(
            "Selected runsets were produced by different code snapshots; this is allowed "
            "for version comparisons and will be preserved in the SA provenance."
        )
    task_hashes = {str(item.get("task_sha256") or "") for item in items}
    task_hashes.discard("")
    if len(task_hashes) > 1:
        warnings.append(
            "Selected runsets came from different frozen task files; compatible problem "
            "and metric checks still passed."
        )
    return warnings


def _normalize_method(value: str) -> str:
    method = str(value).strip().lower().replace("_", "-")
    aliases = {
        "friedman-nemenyi": "friedman",
        "wilcoxon": "signed-rank",
        "wilcoxon-signed-rank": "signed-rank",
        "wilcoxon-rank-sum": "rank-sum",
        "ranksum": "rank-sum",
    }
    method = aliases.get(method, method)
    if method not in SA_METHODS:
        raise ResultWorkspaceError(
            "invalid_sa_method",
            f"Unknown SA method {value!r}; choose friedman, signed-rank, or rank-sum.",
        )
    return method


def _load_trend_payload(payload: bytes) -> dict[str, list[Any]]:
    arrays = _load_trend_arrays(payload)
    return {
        "key": arrays["key"].tolist(),
        "mean": arrays["mean"].astype(float).tolist(),
        "stderr": arrays["stderr"].astype(float).tolist(),
        "count": arrays["count"].astype(int).tolist(),
    }


def _load_trend_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {
            "key": np.asarray(data["key"]).copy(),
            "mean": np.asarray(data["mean"], dtype=float).copy(),
            "stderr": np.asarray(data["stderr"], dtype=float).copy(),
            "count": np.asarray(data["count"], dtype=int).copy(),
        }


def _plot_x_bound(value: float | int | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ResultWorkspaceError(
            "invalid_plot_x_range",
            f"Plot {name} must be a finite number.",
        )
    number = float(value)
    if not math.isfinite(number):
        raise ResultWorkspaceError(
            "invalid_plot_x_range",
            f"Plot {name} must be a finite number.",
        )
    return number


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runset_source_fingerprint(bundle_fingerprint: str, runset_uid: str) -> str:
    return hashlib.sha256(
        stable_json_dumps(
            {
                "bundle_fingerprint": bundle_fingerprint,
                "runset_uid": runset_uid,
            }
        ).encode("utf-8")
    ).hexdigest()


def _ordered_pairing_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        stable_json_dumps(
            [[int(item["run_index"]), int(item["problem_seed"])] for item in rows]
        ).encode("utf-8")
    ).hexdigest()


def _absolute_without_symlink_resolution(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _is_safe_path_component(value: str) -> bool:
    candidate = str(value)
    return bool(candidate) and candidate not in {".", ".."} and Path(candidate).name == candidate


__all__ = ["ResultWorkspace", "ResultWorkspaceError"]
