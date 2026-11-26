"""Plot exporters for stored EDOLAB result bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.result.store import ResultStore

PLOT_FORMATS = {"pdf", "svg", "png"}


@dataclass(slots=True, frozen=True)
class TrendPlotSeries:
    label: str
    axis: str
    key: list[Any]
    mean: list[float]
    stderr: list[float] | None = None
    count: list[int] | None = None


@dataclass(slots=True, frozen=True)
class BoxplotSeries:
    label: str
    values: list[float]


def select_trend_sample_indices(
    key: Sequence[Any],
    mean: Sequence[float],
    max_points: int,
) -> list[int]:
    """Select a deterministic, shape-preserving subset of one trend series.

    The first and last observations are retained whenever the budget permits.
    The remaining budget is split into stable contiguous buckets whose local
    minima and maxima preserve the visually important peaks and valleys.  The
    helper only returns indices; callers can therefore apply the exact same
    selection to ``stderr`` and ``count`` without changing stored trend data.
    """

    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be a positive integer.")
    target = int(max_points)
    if target < 1:
        raise ValueError("max_points must be a positive integer.")
    point_count = len(key)
    if len(mean) != point_count:
        raise ValueError("Trend key and mean must have the same length.")
    if point_count == 0:
        return []
    if point_count <= target:
        return list(range(point_count))
    if target == 1:
        return [0]
    if target == 2:
        return [0, point_count - 1]

    values = np.asarray(mean, dtype=float)
    if values.ndim != 1 or values.size != point_count:
        raise ValueError("Trend mean must be one-dimensional.")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("Trend mean must contain only finite values.")

    interior_slots = target - 2
    pair_buckets = interior_slots // 2
    selected: set[int] = {0, point_count - 1}
    if pair_buckets:
        boundaries = np.linspace(
            1,
            point_count - 1,
            pair_buckets + 1,
            dtype=np.int64,
        )
        for bucket_index in range(pair_buckets):
            start = int(boundaries[bucket_index])
            stop = int(boundaries[bucket_index + 1])
            if stop <= start:
                continue
            bucket = values[start:stop]
            minimum = start + int(np.argmin(bucket))
            maximum = start + int(np.argmax(bucket))
            if minimum == maximum and stop - start > 1:
                # A flat bucket has no extrema preference.  Retaining both
                # bucket edges preserves its horizontal extent instead.
                minimum, maximum = start, stop - 1
            selected.add(minimum)
            selected.add(maximum)

    if interior_slots % 2:
        positions = np.linspace(0.0, 1.0, point_count)
        baseline = values[0] + (values[-1] - values[0]) * positions
        deviation = np.abs(values - baseline)
        deviation[np.fromiter(selected, dtype=np.int64)] = -np.inf
        index = int(np.argmax(deviation))
        if 0 < index < point_count - 1:
            selected.add(index)

    # Numerical bucket boundaries can occasionally collapse when the target is
    # close to the source size.  Fill any unused budget with evenly spread
    # observations while retaining deterministic ordering.
    if len(selected) < target:
        for raw_index in np.linspace(1, point_count - 2, target, dtype=np.int64):
            selected.add(int(raw_index))
            if len(selected) == target:
                break
    if len(selected) < target:
        for index in range(1, point_count - 1):
            selected.add(index)
            if len(selected) == target:
                break
    return sorted(selected)[:target]


def downsample_trend_series(
    series: TrendPlotSeries,
    max_points: int,
) -> TrendPlotSeries:
    """Return one trend series sampled to ``max_points`` without mutation."""

    point_count = len(series.key)
    if len(series.mean) != point_count:
        raise ValueError("Trend key and mean must have the same length.")
    if series.stderr is not None and len(series.stderr) != point_count:
        raise ValueError("Trend stderr must match the key length.")
    if series.count is not None and len(series.count) != point_count:
        raise ValueError("Trend count must match the key length.")
    indices = select_trend_sample_indices(series.key, series.mean, max_points)
    if len(indices) == point_count:
        return series
    return TrendPlotSeries(
        label=series.label,
        axis=series.axis,
        key=[series.key[index] for index in indices],
        mean=[float(series.mean[index]) for index in indices],
        stderr=(
            [float(series.stderr[index]) for index in indices]
            if series.stderr is not None
            else None
        ),
        count=(
            [int(series.count[index]) for index in indices]
            if series.count is not None
            else None
        ),
    )


def downsample_trend_series_collection(
    series: Sequence[TrendPlotSeries],
    *,
    max_points_per_series: int,
    max_total_points: int,
) -> list[TrendPlotSeries]:
    """Apply a fair per-series share of one plot's total point budget."""

    items = list(series)
    if not items:
        return []
    if max_points_per_series < 1 or max_total_points < 1:
        raise ValueError("Trend point budgets must be positive integers.")
    capacities = [min(len(item.key), int(max_points_per_series)) for item in items]
    if any(capacity < 1 for capacity in capacities):
        raise ValueError("Trend series must contain at least one point.")
    budgets = _fair_point_budgets(capacities, int(max_total_points))
    return [
        downsample_trend_series(item, budget)
        for item, budget in zip(items, budgets, strict=True)
    ]


def export_trend_plot(
    result_dir: str | Path,
    *,
    runset_uids: list[str],
    metric_name: str,
    output_dir: str | Path,
    formats: list[str] | tuple[str, ...] | set[str] | None = None,
    title: str | None = None,
) -> dict[str, str]:
    requested = _validate_formats(formats)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    store = ResultStore(result_dir)
    try:
        runsets = {item["runset_uid"]: item for item in store.list_runsets()}
        series: list[TrendPlotSeries] = []
        for runset_uid in runset_uids:
            trend = store.load_trend_by_uid(runset_uid, metric_name)
            if trend is None:
                continue
            item = runsets.get(runset_uid, {})
            label = f"{item.get('label', '')}:{item.get('algorithm_name', runset_uid)}"
            series.append(
                TrendPlotSeries(
                    label=label.strip(":"),
                    axis=trend.axis,
                    key=trend.key,
                    mean=trend.mean,
                    stderr=trend.stderr,
                    count=trend.count,
                )
            )
        if not series:
            raise ValueError(f"No trend data found for metric {metric_name!r}.")
        return export_trend_series(
            series,
            metric_name=metric_name,
            output_dir=output_path,
            formats=requested,
            title=title,
        )
    finally:
        store.close()


def export_trend_series(
    series: Sequence[TrendPlotSeries],
    *,
    metric_name: str,
    output_dir: str | Path,
    formats: list[str] | tuple[str, ...] | set[str] | None = None,
    title: str | None = None,
    x_min: float | int | None = None,
    x_max: float | int | None = None,
    legend_position: str = "outside_right",
    y_scale: str = "auto",
) -> dict[str, str]:
    requested = _validate_formats(formats)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cleaned = _clean_trend_series(series, x_min=x_min, x_max=x_max)
    if not cleaned:
        raise ValueError(f"No trend data found for metric {metric_name!r}.")
    axes = {item.axis for item in cleaned}
    if len(axes) != 1:
        raise ValueError(f"Trend metric {metric_name!r} uses mixed axes: {', '.join(sorted(axes))}.")

    axis = next(iter(axes))
    fig, ax = _new_figure()
    positive_values = True
    for index, item in enumerate(cleaned):
        values = np.asarray(item.mean, dtype=float)
        positive_values = positive_values and bool(values.size) and bool(np.nanmin(values) > 0)
        ax.plot(
            item.key,
            item.mean,
            label=item.label,
            linewidth=2.0,
            linestyle=_line_style(index),
            color=_line_color(index),
        )
    scale = y_scale.lower()
    if scale == "log" or (scale == "auto" and positive_values):
        ax.set_yscale("log")
    ax.set_xlabel(_axis_label(axis))
    ax.set_ylabel(_metric_ylabel(metric_name))
    ax.set_title(title or f"Trend of {metric_name} Over {_axis_label(axis)}")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    _apply_legend(ax, legend_position, len(cleaned))
    fig.tight_layout()
    return _save_figure(fig, output_path, f"trend_{_safe_name(metric_name)}", requested)


def export_boxplot(
    result_dir: str | Path,
    *,
    runset_uids: list[str],
    metric_name: str,
    output_dir: str | Path,
    formats: list[str] | tuple[str, ...] | set[str] | None = None,
    title: str | None = None,
) -> dict[str, str]:
    requested = _validate_formats(formats)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    store = ResultStore(result_dir)
    try:
        samples = store.load_metric_samples_by_uid(runset_uids, metric_name)
        if not samples:
            raise ValueError(f"No scalar samples found for metric {metric_name!r}.")
        return export_boxplot_series(
            [
                BoxplotSeries(
                    label=f"{sample.task_label}:{sample.algorithm_name}",
                    values=sample.values,
                )
                for sample in samples
            ],
            metric_name=metric_name,
            output_dir=output_path,
            formats=requested,
            title=title,
        )
    finally:
        store.close()


def export_boxplot_series(
    series: Sequence[BoxplotSeries],
    *,
    metric_name: str,
    output_dir: str | Path,
    formats: list[str] | tuple[str, ...] | set[str] | None = None,
    title: str | None = None,
) -> dict[str, str]:
    requested = _validate_formats(formats)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cleaned = [
        BoxplotSeries(
            label=item.label,
            values=[float(value) for value in item.values if np.isfinite(float(value))],
        )
        for item in series
    ]
    cleaned = [item for item in cleaned if item.values]
    if not cleaned:
        raise ValueError(f"No scalar samples found for metric {metric_name!r}.")
    labels = [item.label for item in cleaned]
    values = [item.values for item in cleaned]
    fig, ax = _new_figure()
    ax.boxplot(values, notch=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Boxplot of {metric_name}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    if len(labels) > 5:
        ax.tick_params(axis="x", rotation=45)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.tight_layout()
    return _save_figure(fig, output_path, f"boxplot_{_safe_name(metric_name)}", requested)


def _new_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    return fig, ax


def _save_figure(fig, output_dir: Path, stem: str, formats: set[str]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    written: dict[str, str] = {}
    for fmt in sorted(formats):
        target = output_dir / f"{stem}.{fmt}"
        fig.savefig(target, format=fmt, dpi=300, bbox_inches="tight")
        written[fmt] = str(target)
    plt.close(fig)
    return written


def _validate_formats(
    formats: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    requested = set(formats or {"pdf"})
    unknown = requested - PLOT_FORMATS
    if unknown:
        raise ValueError(f"Unknown plot format(s): {', '.join(sorted(unknown))}.")
    return requested


def _clean_trend_series(
    series: Sequence[TrendPlotSeries],
    *,
    x_min: float | int | None,
    x_max: float | int | None,
) -> list[TrendPlotSeries]:
    cleaned: list[TrendPlotSeries] = []
    for item in series:
        keys: list[Any] = []
        means: list[float] = []
        stderr: list[float] = []
        count: list[int] = []
        for index, (key, value) in enumerate(zip(item.key, item.mean)):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_value):
                continue
            if not _key_in_range(key, x_min=x_min, x_max=x_max):
                continue
            keys.append(key)
            means.append(numeric_value)
            if item.stderr is not None and index < len(item.stderr):
                stderr.append(float(item.stderr[index]))
            if item.count is not None and index < len(item.count):
                count.append(int(item.count[index]))
        if keys:
            cleaned.append(
                TrendPlotSeries(
                    label=item.label,
                    axis=item.axis,
                    key=keys,
                    mean=means,
                    stderr=stderr or None,
                    count=count or None,
                )
            )
    return cleaned


def _key_in_range(
    key: Any,
    *,
    x_min: float | int | None,
    x_max: float | int | None,
) -> bool:
    if x_min is None and x_max is None:
        return True
    try:
        numeric_key = float(key)
    except (TypeError, ValueError):
        return True
    if x_min is not None and numeric_key < float(x_min):
        return False
    if x_max is not None and numeric_key > float(x_max):
        return False
    return True


def _axis_label(axis: str) -> str:
    lowered = axis.lower()
    if lowered in {"fe", "fes", "fitness_evaluations", "function_evaluations"}:
        return "Function Evaluations"
    if lowered in {"environment", "environment_trend", "env"}:
        return "Environment Number"
    return axis.replace("_", " ").title()


def _metric_ylabel(metric_name: str) -> str:
    if metric_name == "current_error":
        return "Mean Current Error"
    if metric_name == "current_performance":
        return "Mean Current Performance"
    return metric_name


def _apply_legend(ax, legend_position: str, series_count: int) -> None:
    position = (legend_position or "outside_right").lower()
    num_columns = 2 if position in {"upper_left", "upper_right", "lower_left", "lower_right"} else max(1, min(5, (series_count + 19) // 20))
    if position in {"outside_right", "bestoutside"}:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=11,
            ncol=num_columns,
        )
        return
    locations = {
        "best": "best",
        "upper_left": "upper left",
        "upper_right": "upper right",
        "lower_left": "lower left",
        "lower_right": "lower right",
        "northwest": "upper left",
        "northeast": "upper right",
        "southwest": "lower left",
        "southeast": "lower right",
    }
    ax.legend(loc=locations.get(position, "best"), frameon=False, fontsize=11, ncol=num_columns)


def _line_style(index: int) -> str:
    return ["-", "--", ":", "-."][index % 4]


def _line_color(index: int):
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab10")
    return palette(index % 10)


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return text.strip("_") or "metric"


def _fair_point_budgets(capacities: Sequence[int], total_budget: int) -> list[int]:
    if total_budget < len(capacities):
        raise ValueError(
            "The total trend point budget must allow at least one point per series."
        )
    if sum(capacities) <= total_budget:
        return list(capacities)

    endpoint_minimums = [min(capacity, 2) for capacity in capacities]
    if sum(endpoint_minimums) <= total_budget:
        budgets = endpoint_minimums
    else:
        budgets = [1 for _ in capacities]
    remaining = total_budget - sum(budgets)
    while remaining:
        active = [
            index
            for index, (budget, capacity) in enumerate(zip(budgets, capacities, strict=True))
            if budget < capacity
        ]
        if not active:
            break
        share = max(1, remaining // len(active))
        for index in active:
            increment = min(share, capacities[index] - budgets[index], remaining)
            budgets[index] += increment
            remaining -= increment
            if remaining == 0:
                break
    return budgets
