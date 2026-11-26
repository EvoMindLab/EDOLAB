"""Aggregate run-level results."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from core.evaluation.result import (
    ExperimentResult,
    RunResult,
    SummaryStatistic,
    TrendStatistic,
)


def aggregate_runs(
    algorithm_name: str,
    problem_name: str,
    runs: list[RunResult],
    label: str | None = None,
    metric_output_names: list[str] | None = None,
) -> ExperimentResult:
    if not runs:
        raise ValueError("Cannot aggregate an empty run list.")
    metric_names = sorted({name for run in runs for name in run.metrics})
    summary: dict[str, SummaryStatistic] = {}
    for name in metric_names:
        values = [float(run.metrics[name]) for run in runs if name in run.metrics]
        stderr = pstdev(values) / math.sqrt(len(values)) if values else math.nan
        summary[name] = SummaryStatistic(
            mean=float(mean(values)),
            median=float(median(values)),
            stderr=float(stderr),
            all_results=values,
        )
    runtimes = [float(run.runtime_seconds) for run in runs]
    summary["T_r"] = SummaryStatistic(
        mean=float(mean(runtimes)),
        median=float(median(runtimes)),
        stderr=float(pstdev(runtimes) / math.sqrt(len(runtimes))),
        all_results=runtimes,
    )
    return ExperimentResult(
        algorithm_name=algorithm_name,
        problem_name=problem_name,
        runs=sorted(runs, key=lambda run: run.run_index),
        summary=summary,
        trend_summary=aggregate_trends(runs),
        label=label,
        metric_output_names=metric_output_names,
    )


def aggregate_trends(runs: list[RunResult]) -> dict[str, TrendStatistic]:
    trend_names = sorted(
        {
            name
            for run in runs
            for name in run.raw.get("trends", {})
        }
    )
    summaries: dict[str, TrendStatistic] = {}
    for name in trend_names:
        series = [
            run.raw.get("trends", {}).get(name)
            for run in runs
            if name in run.raw.get("trends", {})
        ]
        if not series:
            continue
        summaries[name] = _aggregate_trend_series(series)
    return summaries


def _aggregate_trend_series(series: list[dict[str, Any]]) -> TrendStatistic:
    axis = str(series[0]["axis"])
    keys = _ordered_union_keys(series)
    matrix = np.full((len(series), len(keys)), np.nan, dtype=float)
    key_to_column = {key: index for index, key in enumerate(keys)}
    for row_index, item in enumerate(series):
        item_keys = list(item.get("key", []))
        item_values = list(item.get("value", []))
        for key, value in zip(item_keys, item_values):
            column = key_to_column.get(key)
            if column is not None:
                matrix[row_index, column] = float(value)

    finite = np.isfinite(matrix)
    count = np.sum(finite, axis=0)
    mean_values = np.full(len(keys), np.nan, dtype=float)
    stderr_values = np.full(len(keys), np.nan, dtype=float)
    for column in range(len(keys)):
        values = matrix[finite[:, column], column]
        if values.size == 0:
            continue
        mean_values[column] = float(np.mean(values))
        stderr_values[column] = (
            float(np.std(values, ddof=0) / math.sqrt(values.size))
            if values.size > 0
            else float("nan")
        )
    return TrendStatistic(
        axis=axis,
        key=keys,
        mean=mean_values.astype(float).tolist(),
        stderr=stderr_values.astype(float).tolist(),
        count=count.astype(int).tolist(),
    )


def _ordered_union_keys(series: list[dict[str, Any]]) -> list[Any]:
    keys: list[Any] = []
    seen = set()
    for item in series:
        for key in item.get("key", []):
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    try:
        return sorted(keys)
    except TypeError:
        return keys
