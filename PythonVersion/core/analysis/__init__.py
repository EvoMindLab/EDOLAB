"""Statistical analysis helpers for scalar experiment summaries."""

from __future__ import annotations

from core.analysis.statistics import (
    FriedmanNemenyiResult,
    MetricDirection,
    PairwiseComparisonResult,
    PairwiseRow,
    RankRow,
    ScalarRunSet,
    compact_letter_groups,
    friedman_nemenyi,
    holm_adjust_pairwise_rows,
    wilcoxon_rank_sum,
    wilcoxon_signed_rank,
)
from core.analysis.service import (
    compatible_runset_groups,
    common_scalar_metrics,
    run_statistical_analysis,
    selected_sa_runset_ids,
    write_analysis_outputs,
)

__all__ = [
    "FriedmanNemenyiResult",
    "MetricDirection",
    "PairwiseComparisonResult",
    "PairwiseRow",
    "RankRow",
    "ScalarRunSet",
    "compact_letter_groups",
    "friedman_nemenyi",
    "holm_adjust_pairwise_rows",
    "wilcoxon_rank_sum",
    "wilcoxon_signed_rank",
    "compatible_runset_groups",
    "common_scalar_metrics",
    "run_statistical_analysis",
    "selected_sa_runset_ids",
    "write_analysis_outputs",
]
