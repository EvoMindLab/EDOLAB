"""Result exporters."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from core.evaluation.result import ExperimentResult


EXPORT_DECIMALS = 2


def export_json(
    results: list[ExperimentResult],
    output_dir: str | Path,
    metric_output_names: list[str] | tuple[str, ...] | None = None,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "summary.json"
    data = [
        _export_value(
            _result_to_dict(
                result,
                metric_output_names=(
                    list(metric_output_names)
                    if metric_output_names is not None
                    else result.metric_output_names
                ),
            )
        )
        for result in results
    ]
    target.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return target


def export_csv(
    results: list[ExperimentResult],
    output_dir: str | Path,
    metric_output_names: list[str] | tuple[str, ...] | None = None,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "summary.csv"
    include_label = any(result.label is not None for result in results)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "algorithm",
            "problem",
            "metric",
            "mean",
            "median",
            "stderr",
            "all_results",
        ]
        if include_label:
            header.insert(0, "label")
        writer.writerow(header)
        for result in results:
            output_names = (
                list(metric_output_names)
                if metric_output_names is not None
                else result.metric_output_names
            )
            summary = _filter_named_mapping(result.summary, output_names)
            for metric, stat in summary.items():
                row = [
                    result.algorithm_name,
                    result.problem_name,
                    metric,
                    _format_export_value(stat.mean),
                    _format_export_value(stat.median),
                    _format_export_value(stat.stderr),
                    " ".join(_format_export_value(value) for value in stat.all_results),
                ]
                if include_label:
                    row.insert(0, result.label or "")
                writer.writerow(row)
    return target


def _result_to_dict(
    result: ExperimentResult,
    metric_output_names: list[str] | tuple[str, ...] | None = None,
) -> dict:
    data = asdict(result)
    if data.get("label") is None:
        data.pop("label", None)
    data.pop("metric_output_names", None)
    data["summary"] = _filter_named_mapping(data["summary"], metric_output_names)
    data["trend_summary"] = _filter_named_mapping(
        data.get("trend_summary", {}),
        metric_output_names,
    )
    for run in data["runs"]:
        if run.get("label") is None:
            run.pop("label", None)
        run["metrics"] = _filter_named_mapping(run.get("metrics", {}), metric_output_names)
        if (
            metric_output_names is not None
            and "current_error" not in set(metric_output_names)
        ):
            run.pop("current_error", None)
        if isinstance(run.get("raw"), dict):
            run["raw"]["trends"] = _filter_named_mapping(
                run["raw"].get("trends", {}),
                metric_output_names,
            )
    return data


def _filter_named_mapping(
    values: dict[str, Any],
    names: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    if names is None:
        return values
    allowed = set(names)
    return {name: value for name, value in values.items() if name in allowed}


def _export_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _export_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_export_value(item) for item in value]
    if isinstance(value, tuple):
        return [_export_value(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return number
        return round(number, EXPORT_DECIMALS)
    return value


def _format_export_value(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return str(number)
        return f"{number:.{EXPORT_DECIMALS}f}"
    return str(value)
