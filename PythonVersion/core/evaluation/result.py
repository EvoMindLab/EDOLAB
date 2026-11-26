"""Result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunResult:
    algorithm_name: str
    problem_name: str
    run_index: int
    problem_seed: int
    algorithm_seed: int
    metrics: dict[str, float]
    runtime_seconds: float
    fe: int
    current_error: list[float] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


@dataclass(slots=True)
class SummaryStatistic:
    mean: float
    median: float
    stderr: float
    all_results: list[float]


@dataclass(slots=True)
class TrendStatistic:
    axis: str
    key: list[Any]
    mean: list[float]
    stderr: list[float]
    count: list[int]


@dataclass(slots=True)
class ExperimentResult:
    algorithm_name: str
    problem_name: str
    runs: list[RunResult]
    summary: dict[str, SummaryStatistic]
    trend_summary: dict[str, TrendStatistic] = field(default_factory=dict)
    label: str | None = None
    metric_output_names: list[str] | None = None
