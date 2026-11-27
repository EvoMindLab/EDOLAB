"""Shared statistical-analysis service and paper table exporters."""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

from core.analysis.statistics import (
    PAIRWISE_COLUMNS,
    ScalarRunSet,
    friedman_nemenyi,
    holm_adjust_pairwise_rows,
    wilcoxon_rank_sum,
    wilcoxon_signed_rank,
)
from core.result.store import ResultStore


SAMethod = Literal["friedman", "signed-rank", "rank-sum"]
SA_FORMATS = {"sqlite", "csv", "xlsx", "latex"}


def compatible_runset_groups(store: ResultStore) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in store.list_runsets():
        key = (
            item["problem_name"],
            json.dumps(item["problem_params"], sort_keys=True, separators=(",", ":")),
        )
        groups.setdefault(key, []).append(item)
    return [group for group in groups.values() if len(group) >= 2]


def common_scalar_metrics(store: ResultStore, runset_ids: list[int]) -> list[str]:
    if not runset_ids:
        return []
    metric_sets: list[set[str]] = []
    for runset_id in runset_ids:
        rows = store.summary.execute(
            """
            select distinct metric_name
            from run_metric_values
            where runset_id = ?
            order by metric_name
            """,
            (int(runset_id),),
        ).fetchall()
        metric_sets.append({row["metric_name"] for row in rows})
    return sorted(set.intersection(*metric_sets)) if metric_sets else []


def selected_sa_runset_ids(
    store: ResultStore,
    metric_names: list[str],
    method: SAMethod,
    requested_ids: list[int] | None = None,
    requested_uids: list[str] | None = None,
) -> list[int]:
    minimum = 3 if method == "friedman" else 2
    if requested_uids:
        requested_ids = store.resolve_runset_ids(requested_uids)
    if requested_ids:
        validate_runset_group(store, requested_ids, metric_names, minimum)
        return requested_ids

    for group in compatible_runset_groups(store):
        runset_ids = [item["runset_id"] for item in group]
        if len(runset_ids) < minimum:
            continue
        available = set(common_scalar_metrics(store, runset_ids))
        if set(metric_names) <= available:
            return runset_ids
    names = ", ".join(metric_names)
    raise ValueError(
        f"No compatible group with at least {minimum} runsets and metrics: {names}."
    )


def validate_runset_group(
    store: ResultStore,
    runset_ids: list[int],
    metric_names: list[str],
    minimum: int,
) -> None:
    if len(runset_ids) < minimum:
        raise ValueError(f"At least {minimum} runsets are required.")
    known = {item["runset_id"]: item for item in store.list_runsets()}
    missing = [runset_id for runset_id in runset_ids if runset_id not in known]
    if missing:
        raise ValueError(f"Unknown runset id(s): {', '.join(map(str, missing))}.")
    selected = [known[runset_id] for runset_id in runset_ids]
    problem_keys = {
        (
            item["problem_name"],
            json.dumps(item["problem_params"], sort_keys=True, separators=(",", ":")),
        )
        for item in selected
    }
    if len(problem_keys) != 1:
        raise ValueError("Selected runsets must share the same problem and parameters.")
    available = set(common_scalar_metrics(store, runset_ids))
    missing_metrics = sorted(set(metric_names) - available)
    if missing_metrics:
        raise ValueError(f"Metric(s) unavailable for all runsets: {', '.join(missing_metrics)}.")


def run_statistical_analysis(
    store: ResultStore,
    *,
    method: SAMethod,
    metrics: list[str],
    runset_ids: list[int] | None = None,
    runset_uids: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    metric_names = [name for name in metrics if name]
    if not metric_names:
        selected_ids, metric_names = _default_runsets_and_metrics(
            store,
            method,
            runset_ids=runset_ids,
            runset_uids=runset_uids,
        )
    else:
        selected_ids = selected_sa_runset_ids(
            store,
            metric_names,
            method,
            requested_ids=runset_ids,
            requested_uids=runset_uids,
        )
    analyses: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    for metric_name in metric_names:
        samples = store.load_metric_samples(selected_ids, metric_name)
        runsets = [
            ScalarRunSet(
                id=str(sample.runset_id),
                label=f"{sample.task_label}:{sample.algorithm_name}",
                algorithm=sample.algorithm_name,
                metric_name=sample.metric_name,
                values=sample.values,
                direction=_analysis_direction(sample.direction),
            )
            for sample in samples
        ]
        payload = _run_single_metric(method, runsets, alpha=alpha)
        analyses.append(payload)
        tables.append(_paper_table(method, metric_name, runsets, payload, alpha))
    return {
        "method": method,
        "metrics": metric_names,
        "alpha": float(alpha),
        "runset_ids": selected_ids,
        "runset_uids": [
            item["runset_uid"]
            for item in store.list_runsets()
            if item["runset_id"] in set(selected_ids)
        ],
        "analyses": analyses,
        "tables": tables,
        "created_at": _now_text(),
    }


def run_statistical_analysis_from_samples(
    *,
    method: SAMethod,
    samples_by_metric: dict[str, list[ScalarRunSet]],
    alpha: float = 0.05,
    runset_uids: list[str] | None = None,
    comparison_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    metric_names = [name for name in samples_by_metric if name]
    analyses: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    for metric_name in metric_names:
        runsets = samples_by_metric[metric_name]
        if method == "friedman" and len(runsets) < 3:
            raise ValueError("Friedman-Nemenyi requires at least three runsets.")
        if method != "friedman" and len(runsets) < 2:
            raise ValueError(f"{method} requires at least two runsets.")
        compared_runsets = (
            [
                replace(
                    runset,
                    label=comparison_labels.get(runset.id, runset.label),
                )
                for runset in runsets
            ]
            if comparison_labels
            else runsets
        )
        payload = _run_single_metric(method, compared_runsets, alpha=alpha)
        analyses.append(payload)
        tables.append(
            _paper_table(
                method,
                metric_name,
                compared_runsets,
                payload,
                alpha,
                comparison_labels=comparison_labels,
            )
        )
    return {
        "method": method,
        "metrics": metric_names,
        "alpha": float(alpha),
        "runset_ids": [],
        "runset_uids": runset_uids or [],
        "analyses": analyses,
        "tables": tables,
        "created_at": _now_text(),
    }


def _default_runsets_and_metrics(
    store: ResultStore,
    method: SAMethod,
    *,
    runset_ids: list[int] | None,
    runset_uids: list[str] | None,
) -> tuple[list[int], list[str]]:
    if runset_uids:
        selected_ids = store.resolve_runset_ids(runset_uids)
    elif runset_ids:
        selected_ids = runset_ids
    else:
        minimum = 3 if method == "friedman" else 2
        for group in compatible_runset_groups(store):
            ids = [item["runset_id"] for item in group]
            metrics = _supported_common_metrics(store, ids)
            if len(ids) >= minimum and metrics:
                return ids, metrics
        raise ValueError("No compatible scalar metrics found for SA.")
    minimum = 3 if method == "friedman" else 2
    validate_runset_group(store, selected_ids, [], minimum)
    metrics = _supported_common_metrics(store, selected_ids)
    if not metrics:
        raise ValueError("No compatible scalar metrics found for SA.")
    return selected_ids, metrics


def _supported_common_metrics(store: ResultStore, runset_ids: list[int]) -> list[str]:
    metrics = common_scalar_metrics(store, runset_ids)
    supported: list[str] = []
    for metric_name in metrics:
        samples = store.load_metric_samples(runset_ids, metric_name)
        if samples and all(sample.direction in {"lower", "higher"} for sample in samples):
            supported.append(metric_name)
    return supported


def write_analysis_outputs(
    output_dir: str | Path,
    payload: dict[str, Any],
    formats: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, str]:
    requested = set(formats or {"sqlite"})
    unknown = requested - SA_FORMATS
    if unknown:
        raise ValueError(f"Unknown SA output format(s): {', '.join(sorted(unknown))}.")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    if "sqlite" in requested:
        written["sqlite"] = str(_write_analysis_sqlite(path, payload))
    if requested & {"csv", "xlsx", "latex"}:
        table_dir = path / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        for table in payload.get("tables", []):
            stem = _safe_name(f"{payload['method']}_{table['metric_name']}")
            if "csv" in requested:
                target = table_dir / f"{stem}.csv"
                _write_table_csv(target, table)
                written[f"csv:{table['metric_name']}"] = str(target)
            if "xlsx" in requested:
                target = table_dir / f"{stem}.xlsx"
                _write_table_xlsx(target, table)
                written[f"xlsx:{table['metric_name']}"] = str(target)
            if "latex" in requested:
                target = table_dir / f"{stem}.tex"
                _write_table_latex(target, table)
                written[f"latex:{table['metric_name']}"] = str(target)
    return written


def _run_single_metric(
    method: SAMethod,
    runsets: list[ScalarRunSet],
    *,
    alpha: float,
) -> dict[str, Any]:
    if method == "friedman":
        return friedman_nemenyi(runsets, alpha=alpha).to_dict()
    pairwise_rows = []
    for left_index in range(len(runsets) - 1):
        for right_index in range(left_index + 1, len(runsets)):
            if method == "signed-rank":
                result = wilcoxon_signed_rank(runsets[left_index], runsets[right_index])
            elif method == "rank-sum":
                result = wilcoxon_rank_sum(runsets[left_index], runsets[right_index])
            else:
                raise ValueError(f"Unknown SA method {method!r}.")
            pairwise_rows.extend(result.rows)
    pairwise_rows = holm_adjust_pairwise_rows(pairwise_rows)
    rows = [asdict(row) for row in pairwise_rows]
    return {
        "method": method,
        "metric_name": runsets[0].metric_name,
        "direction": runsets[0].direction,
        "columns": PAIRWISE_COLUMNS,
        "rows": rows,
        "details": {
            "pair_count": len(rows),
            "p_adjustment": "holm",
            "p_adjustment_family_size": len(rows),
            "raw_p_value_field": "p_value",
            "significance_p_value_field": "adjusted_p_value",
            "effect_size_direction": "positive_favors_left",
        },
    }


def _paper_table(
    method: SAMethod,
    metric_name: str,
    runsets: list[ScalarRunSet],
    payload: dict[str, Any],
    alpha: float,
    comparison_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    columns = [
        runset.label if comparison_labels else runset.algorithm
        for runset in runsets
    ]
    rows: list[list[str]] = []
    rows.append([
        f"{_metric_label(metric_name)} Mean±SE (Median)",
        *[_mean_se_median(values) for values in runsets],
    ])
    if method == "friedman":
        rank_rows = {
            str(row["id"]): row
            for row in payload.get("rank_rows", [])
        }
        rows.append([
            f"{_metric_label(metric_name)} Rank",
            *[
                _rank_with_group(rank_rows.get(runset.id))
                for runset in runsets
            ],
        ])
    else:
        wtl = {runset.id: [0, 0, 0] for runset in runsets}
        for row in payload.get("rows", []):
            left = str(row["left_id"])
            right = str(row["right_id"])
            winner = row.get("winner_id") if _decision_p_value(row) < alpha else None
            if winner == left:
                wtl[left][0] += 1
                wtl[right][2] += 1
            elif winner == right:
                wtl[left][2] += 1
                wtl[right][0] += 1
            else:
                wtl[left][1] += 1
                wtl[right][1] += 1
        rows.append([
            f"{_metric_label(metric_name)} w/t/l",
            *[f"{win}/{tie}/{loss}" for win, tie, loss in (wtl[runset.id] for runset in runsets)],
        ])
        rows.append([
            f"{_metric_label(metric_name)} w-l",
            *[str(wtl[runset.id][0] - wtl[runset.id][2]) for runset in runsets],
        ])
    return {
        "metric_name": metric_name,
        "method": method,
        "columns": ["Statistic", *columns],
        "rows": rows,
        "caption": f"{method} results for {metric_name}.",
        "label": f"tab:{_safe_name(method)}_{_safe_name(metric_name)}",
    }


def _decision_p_value(row: dict[str, Any]) -> float:
    adjusted = row.get("adjusted_p_value")
    if adjusted is not None:
        return float(adjusted)
    return float(row.get("p_value", 1.0))


def _write_analysis_sqlite(output_dir: Path, payload: dict[str, Any]) -> Path:
    target = output_dir / "analysis.sqlite"
    with sqlite3.connect(target) as connection:
        connection.executescript(
            """
            create table if not exists meta(
                key text primary key,
                value text not null
            );
            create table if not exists analysis_results(
                id integer primary key autoincrement,
                method text not null,
                metric_name text not null,
                result_json text not null,
                created_at text not null
            );
            """
        )
        connection.execute(
            "insert or replace into meta(key, value) values (?, ?)",
            ("format_version", "2"),
        )
        for analysis in payload.get("analyses", []):
            connection.execute(
                """
                insert into analysis_results(method, metric_name, result_json, created_at)
                values (?, ?, ?, ?)
                """,
                (
                    payload["method"],
                    analysis["metric_name"],
                    json.dumps(
                        {**payload, "analyses": [analysis]},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    payload["created_at"],
                ),
            )
    return target


def _write_table_csv(path: Path, table: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(table["columns"])
        writer.writerows(table["rows"])


def _write_table_xlsx(path: Path, table: dict[str, Any]) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "SA"
    sheet.append(table["columns"])
    for row in table["rows"]:
        sheet.append(row)
    workbook.save(path)


def _write_table_latex(path: Path, table: dict[str, Any]) -> None:
    align = "l" + "c" * (len(table["columns"]) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_latex_escape(table['caption'])}}}",
        f"\\label{{{_latex_escape(table['label'])}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(_latex_escape(str(item)) for item in table["columns"]) + " \\\\",
        "\\midrule",
    ]
    for row in table["rows"]:
        lines.append(" & ".join(_latex_escape(str(item)) for item in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _rank_with_group(row: dict[str, Any] | None) -> str:
    if row is None:
        return "nan"
    rank_text = f"{float(row.get('average_rank', float('nan'))):.2f}"
    group = str(row.get("posthoc_group", "")).strip()
    if group:
        return f"{rank_text} ({group})"
    return rank_text


def _analysis_direction(direction: str):
    if direction not in {"lower", "higher"}:
        raise ValueError(
            f"Metric direction {direction!r} is not supported for SA. "
            "Use a lower- or higher-is-better scalar metric."
        )
    return direction


def _mean_se(runset: ScalarRunSet) -> str:
    import numpy as np

    values = runset.cleaned_values()
    mean = float(np.mean(values))
    stderr = float(np.std(values, ddof=0) / np.sqrt(values.size))
    return f"{mean:.4f}±{stderr:.4f}"


def _mean_se_median(runset: ScalarRunSet) -> str:
    return f"{_mean_se(runset)} ({_median_text(runset)})"


def _median_text(runset: ScalarRunSet) -> str:
    import numpy as np

    return f"{float(np.median(runset.cleaned_values())):.4f}"


def _metric_label(name: str) -> str:
    return name


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return text.strip("_") or "item"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
