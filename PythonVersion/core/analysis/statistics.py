"""Statistical tests for in-memory scalar run samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from scipy import stats

MetricDirection = Literal["lower", "higher"]


@dataclass(frozen=True, slots=True)
class ScalarRunSet:
    """Scalar values for one compared runset."""

    id: str
    label: str
    algorithm: str
    metric_name: str
    values: list[float]
    direction: MetricDirection

    def cleaned_values(self) -> np.ndarray:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"Runset {self.id!r} values must be one-dimensional.")
        if values.size == 0:
            raise ValueError(f"Runset {self.id!r} must contain at least one value.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Runset {self.id!r} contains non-finite values.")
        return values


@dataclass(frozen=True, slots=True)
class RankRow:
    id: str
    label: str
    algorithm: str
    median: float
    mean: float
    average_rank: float
    sample_size: int
    posthoc_group: str = ""


@dataclass(frozen=True, slots=True)
class PairwiseRow:
    left_id: str
    right_id: str
    left_label: str
    right_label: str
    left_median: float
    right_median: float
    median_delta: float
    statistic: float
    p_value: float
    winner_id: str | None
    method: str
    details: dict[str, Any]
    adjusted_p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None


@dataclass(frozen=True, slots=True)
class FriedmanNemenyiResult:
    method: str
    metric_name: str
    direction: MetricDirection
    statistic: float
    p_value: float
    sample_size: int
    rank_rows: list[RankRow]
    pairwise_rows: list[PairwiseRow]
    columns: list[str]
    details: dict[str, Any]
    kendalls_w: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairwiseComparisonResult:
    method: str
    metric_name: str
    direction: MetricDirection
    rows: list[PairwiseRow]
    columns: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def row(self) -> PairwiseRow:
        return self.rows[0]


PAIRWISE_COLUMNS = [
    "left_id",
    "right_id",
    "left_label",
    "right_label",
    "left_median",
    "right_median",
    "median_delta",
    "statistic",
    "p_value",
    "adjusted_p_value",
    "effect_size",
    "effect_size_name",
    "winner_id",
]

FRIEDMAN_COLUMNS = [
    "id",
    "label",
    "algorithm",
    "median",
    "mean",
    "average_rank",
    "sample_size",
    "posthoc_group",
]


def friedman_nemenyi(
    runsets: list[ScalarRunSet],
    *,
    alpha: float = 0.05,
) -> FriedmanNemenyiResult:
    """Run Friedman omnibus test and Nemenyi post-hoc comparisons."""

    if len(runsets) < 3:
        raise ValueError("Friedman-Nemenyi requires at least three runsets.")
    metric_name, direction = _common_metric(runsets)
    values = _paired_values(runsets)
    degenerate_all_ties = bool(np.all(values == values[:, :1]))
    if degenerate_all_ties:
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = stats.friedmanchisquare(
            *[values[:, i] for i in range(values.shape[1])]
        )
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            raise ValueError(
                "Friedman test returned a non-finite statistic for non-degenerate samples."
            )
    ranks = _rank_values(values, direction)
    average_ranks = np.mean(ranks, axis=0)
    kendalls_w = _kendalls_w(ranks)
    rank_rows = [
        RankRow(
            id=runset.id,
            label=runset.label,
            algorithm=runset.algorithm,
            median=float(np.median(values[:, index])),
            mean=float(np.mean(values[:, index])),
            average_rank=float(average_ranks[index]),
            sample_size=int(values.shape[0]),
        )
        for index, runset in enumerate(runsets)
    ]
    pairwise_rows = _nemenyi_rows(runsets, values, average_ranks, direction)
    groups = compact_letter_groups(rank_rows, pairwise_rows, alpha=alpha)
    rank_rows = [
        replace(row, posthoc_group=groups.get(row.id, ""))
        for row in rank_rows
    ]
    return FriedmanNemenyiResult(
        method="friedman_nemenyi",
        metric_name=metric_name,
        direction=direction,
        statistic=float(statistic),
        p_value=float(p_value),
        sample_size=int(values.shape[0]),
        rank_rows=rank_rows,
        pairwise_rows=pairwise_rows,
        columns=FRIEDMAN_COLUMNS,
        details={
            "effect_size_name": "kendalls_w",
            "kendalls_w": kendalls_w,
            "degenerate_all_ties": degenerate_all_ties,
            "kendalls_w_note": (
                "Kendall's W ranges from 0 (no agreement) to 1 (complete agreement)."
            ),
            "omnibus_note": (
                "All paired values are tied; Friedman is defined here as statistic 0 "
                "with p-value 1."
                if degenerate_all_ties
                else "Friedman omnibus statistic and p-value were computed by SciPy."
            ),
            "posthoc_method": "nemenyi",
            "posthoc_group_method": "compact_letters",
            "posthoc_group_alpha": float(alpha),
            "pairwise_columns": PAIRWISE_COLUMNS,
            "rank_note": "Average rank 1 is best for the metric direction.",
            "group_note": (
                "Compact letters are derived from Nemenyi pairwise p-values; "
                "runsets sharing at least one letter are not significantly different."
            ),
        },
        kendalls_w=kendalls_w,
    )


def compact_letter_groups(
    rank_rows: list[RankRow],
    pairwise_rows: list[PairwiseRow],
    *,
    alpha: float = 0.05,
) -> dict[str, str]:
    """Return compact Nemenyi post-hoc letters keyed by runset id."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("Alpha must be between 0 and 1.")
    if not rank_rows:
        return {}

    ids = [str(row.id) for row in rank_rows]
    id_to_index = {runset_id: index for index, runset_id in enumerate(ids)}
    neighbors: list[set[int]] = [set() for _ in ids]
    for row in pairwise_rows:
        left_index = id_to_index.get(str(row.left_id))
        right_index = id_to_index.get(str(row.right_id))
        if left_index is None or right_index is None:
            continue
        if _decision_p_value(row) >= float(alpha):
            neighbors[left_index].add(right_index)
            neighbors[right_index].add(left_index)

    cliques = _maximal_cliques(neighbors)
    rank_order = sorted(
        range(len(rank_rows)),
        key=lambda index: (float(rank_rows[index].average_rank), index),
    )
    rank_position = {index: position for position, index in enumerate(rank_order)}
    cliques = sorted(
        cliques,
        key=lambda clique: (
            min(rank_position[index] for index in clique),
            tuple(sorted(rank_position[index] for index in clique)),
        ),
    )

    groups = {runset_id: "" for runset_id in ids}
    for letter_index, clique in enumerate(cliques):
        label = _compact_letter_label(letter_index)
        for index in sorted(clique, key=lambda item: rank_position[item]):
            groups[ids[index]] += label
    return groups


def wilcoxon_signed_rank(
    left: ScalarRunSet,
    right: ScalarRunSet,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
) -> PairwiseComparisonResult:
    """Run paired Wilcoxon signed-rank test on two equal-length samples."""

    metric_name, direction = _common_metric([left, right])
    left_values, right_values = _paired_values([left, right]).T
    result = stats.wilcoxon(
        left_values,
        right_values,
        alternative=alternative,
        zero_method=zero_method,
    )
    row = _comparison_row(
        left,
        right,
        left_values,
        right_values,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method="wilcoxon_signed_rank",
        details={
            "alternative": alternative,
            "zero_method": zero_method,
            "paired": True,
            "effect_size_direction": "positive_favors_left",
        },
        effect_size=_matched_rank_biserial(left_values, right_values, direction),
        effect_size_name="matched_rank_biserial",
    )
    return PairwiseComparisonResult(
        method="wilcoxon_signed_rank",
        metric_name=metric_name,
        direction=direction,
        rows=[row],
        columns=PAIRWISE_COLUMNS,
        details={
            "alternative": alternative,
            "zero_method": zero_method,
            "paired": True,
            "effect_size_name": "matched_rank_biserial",
            "effect_size_direction": "positive_favors_left",
        },
    )


def wilcoxon_rank_sum(
    left: ScalarRunSet,
    right: ScalarRunSet,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> PairwiseComparisonResult:
    """Run unpaired Wilcoxon rank-sum test on two scalar samples."""

    metric_name, direction = _common_metric([left, right])
    left_values = left.cleaned_values()
    right_values = right.cleaned_values()
    result = stats.ranksums(left_values, right_values, alternative=alternative)
    row = _comparison_row(
        left,
        right,
        left_values,
        right_values,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method="wilcoxon_rank_sum",
        details={
            "alternative": alternative,
            "paired": False,
            "effect_size_direction": "positive_favors_left",
        },
        effect_size=_cliffs_delta(left_values, right_values, direction),
        effect_size_name="cliffs_delta",
    )
    return PairwiseComparisonResult(
        method="wilcoxon_rank_sum",
        metric_name=metric_name,
        direction=direction,
        rows=[row],
        columns=PAIRWISE_COLUMNS,
        details={
            "alternative": alternative,
            "paired": False,
            "effect_size_name": "cliffs_delta",
            "effect_size_direction": "positive_favors_left",
        },
    )


def holm_adjust_pairwise_rows(rows: list[PairwiseRow]) -> list[PairwiseRow]:
    """Return pairwise rows with Holm family-wise adjusted p-values.

    ``p_value`` remains the raw test p-value for compatibility and auditability.
    The adjusted value is stored separately in ``adjusted_p_value`` and should be
    used for family-level significance decisions.
    """

    if not rows:
        return []
    adjusted_values = _holm_adjusted_p_values([row.p_value for row in rows])
    family_size = len(rows)
    return [
        replace(
            row,
            adjusted_p_value=adjusted_p_value,
            details={
                **row.details,
                "p_adjustment": "holm",
                "p_adjustment_family_size": family_size,
                "raw_p_value": float(row.p_value),
                "adjusted_p_value": adjusted_p_value,
                "significance_p_value_field": "adjusted_p_value",
            },
        )
        for row, adjusted_p_value in zip(rows, adjusted_values, strict=True)
    ]


def _common_metric(runsets: list[ScalarRunSet]) -> tuple[str, MetricDirection]:
    metric_names = {runset.metric_name for runset in runsets}
    directions = {runset.direction for runset in runsets}
    if len(metric_names) != 1:
        raise ValueError("Compared runsets must use the same metric name.")
    if not directions <= {"lower", "higher"}:
        raise ValueError("Metric direction must be 'lower' or 'higher'.")
    if len(directions) != 1:
        raise ValueError("Compared runsets must use the same metric direction.")
    return next(iter(metric_names)), cast(MetricDirection, next(iter(directions)))


def _paired_values(runsets: list[ScalarRunSet]) -> np.ndarray:
    columns = [runset.cleaned_values() for runset in runsets]
    sample_lengths = {column.size for column in columns}
    if len(sample_lengths) != 1:
        raise ValueError("Paired statistical methods require equal sample lengths.")
    return np.column_stack(columns)


def _rank_values(values: np.ndarray, direction: MetricDirection) -> np.ndarray:
    rank_input = values if direction == "lower" else -values
    return np.apply_along_axis(stats.rankdata, 1, rank_input, method="average")


def _kendalls_w(ranks: np.ndarray) -> float:
    """Calculate tie-corrected Kendall's coefficient of concordance."""

    row_count, column_count = ranks.shape
    rank_sums = np.sum(ranks, axis=0)
    expected_rank_sum = row_count * (column_count + 1) / 2.0
    squared_deviation = float(np.sum((rank_sums - expected_rank_sum) ** 2))
    tie_correction = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = (
        row_count**2 * (column_count**3 - column_count)
        - row_count * tie_correction
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(12.0 * squared_deviation / denominator, 0.0, 1.0))


def _matched_rank_biserial(
    left_values: np.ndarray,
    right_values: np.ndarray,
    direction: MetricDirection,
) -> float:
    """Return matched-pairs rank-biserial correlation, oriented to the metric."""

    beneficial_differences = (
        right_values - left_values
        if direction == "lower"
        else left_values - right_values
    )
    nonzero = beneficial_differences[beneficial_differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive_rank_sum = float(np.sum(ranks[nonzero > 0.0]))
    negative_rank_sum = float(np.sum(ranks[nonzero < 0.0]))
    total_rank_sum = positive_rank_sum + negative_rank_sum
    if total_rank_sum == 0.0:
        return 0.0
    return float((positive_rank_sum - negative_rank_sum) / total_rank_sum)


def _cliffs_delta(
    left_values: np.ndarray,
    right_values: np.ndarray,
    direction: MetricDirection,
) -> float:
    """Return Cliff's delta without constructing the full comparison matrix."""

    sorted_right = np.sort(right_values)
    left_greater = np.searchsorted(sorted_right, left_values, side="left")
    left_less = sorted_right.size - np.searchsorted(
        sorted_right,
        left_values,
        side="right",
    )
    raw_delta = float(np.sum(left_greater - left_less)) / float(
        left_values.size * right_values.size
    )
    return raw_delta if direction == "higher" else -raw_delta


def _holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("P-values must be one-dimensional.")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("P-values must be finite values between 0 and 1.")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size, dtype=float)
    running_maximum = 0.0
    family_size = values.size
    for sorted_index, original_index in enumerate(order):
        candidate = (family_size - sorted_index) * float(values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = min(1.0, running_maximum)
    return adjusted.tolist()


def _nemenyi_rows(
    runsets: list[ScalarRunSet],
    values: np.ndarray,
    average_ranks: np.ndarray,
    direction: MetricDirection,
) -> list[PairwiseRow]:
    row_count, column_count = values.shape
    standard_error = np.sqrt(column_count * (column_count + 1) / (6.0 * row_count))
    rows: list[PairwiseRow] = []
    for left_index in range(column_count - 1):
        for right_index in range(left_index + 1, column_count):
            q_value = abs(average_ranks[left_index] - average_ranks[right_index]) / standard_error
            p_value = stats.studentized_range.sf(q_value * np.sqrt(2.0), column_count, np.inf)
            left_values = values[:, left_index]
            right_values = values[:, right_index]
            rows.append(
                _comparison_row(
                    runsets[left_index],
                    runsets[right_index],
                    left_values,
                    right_values,
                    statistic=float(q_value),
                    p_value=float(p_value),
                    method="nemenyi",
                    details={
                        "left_average_rank": float(average_ranks[left_index]),
                        "right_average_rank": float(average_ranks[right_index]),
                        "standard_error": float(standard_error),
                    },
                    direction=direction,
                )
            )
    return rows


def _maximal_cliques(neighbors: list[set[int]]) -> list[frozenset[int]]:
    cliques: list[frozenset[int]] = []

    def bron_kerbosch(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            cliques.append(frozenset(r))
            return
        pivot_candidates = p | x
        pivot = max(
            pivot_candidates,
            key=lambda item: len(p & neighbors[item]),
            default=None,
        )
        candidates = set(p)
        if pivot is not None:
            candidates -= neighbors[pivot]
        for vertex in sorted(candidates):
            bron_kerbosch(
                r | {vertex},
                p & neighbors[vertex],
                x & neighbors[vertex],
            )
            p.remove(vertex)
            x.add(vertex)

    bron_kerbosch(set(), set(range(len(neighbors))), set())
    return cliques


def _compact_letter_label(index: int) -> str:
    if index < 0:
        raise ValueError("Letter index must be non-negative.")
    label = ""
    value = int(index)
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("a") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _comparison_row(
    left: ScalarRunSet,
    right: ScalarRunSet,
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    statistic: float,
    p_value: float,
    method: str,
    details: dict[str, Any],
    direction: MetricDirection | None = None,
    adjusted_p_value: float | None = None,
    effect_size: float | None = None,
    effect_size_name: str | None = None,
) -> PairwiseRow:
    metric_direction = direction if direction is not None else left.direction
    left_median = float(np.median(left_values))
    right_median = float(np.median(right_values))
    return PairwiseRow(
        left_id=left.id,
        right_id=right.id,
        left_label=left.label,
        right_label=right.label,
        left_median=left_median,
        right_median=right_median,
        median_delta=_median_delta(left_median, right_median, metric_direction),
        statistic=statistic,
        p_value=p_value,
        winner_id=_median_winner(left.id, right.id, left_median, right_median, metric_direction),
        method=method,
        details=details,
        adjusted_p_value=adjusted_p_value,
        effect_size=effect_size,
        effect_size_name=effect_size_name,
    )


def _decision_p_value(row: PairwiseRow) -> float:
    if row.adjusted_p_value is not None:
        return float(row.adjusted_p_value)
    return float(row.p_value)


def _median_delta(left_median: float, right_median: float, direction: MetricDirection) -> float:
    if direction == "lower":
        return float(right_median - left_median)
    return float(left_median - right_median)


def _median_winner(
    left_id: str,
    right_id: str,
    left_median: float,
    right_median: float,
    direction: MetricDirection,
) -> str | None:
    if left_median == right_median:
        return None
    if direction == "lower":
        return left_id if left_median < right_median else right_id
    return left_id if left_median > right_median else right_id
