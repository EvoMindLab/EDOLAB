"""Optional, read-only precision and phase summaries for dynamic problems.

The threshold and censoring rule are part of each metric's name/definition so
batch configurations can freeze them before a run. Environments are summarized
within a run; downstream inference must use runs, not environments, as units.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from core.evaluation.base import SummaryMetric
from core.registry import metric_registry


TARGET_ERROR = 1.0
METRIC_NAMES = (
    "precision1_hit_rate",
    "precision1_hit_rate_post_change",
    "precision1_fe_capped",
    "precision1_fe_capped_post_change",
    "error_first_10pct_post_change",
    "error_last_50pct",
    "error_last_50pct_post_change",
    "error_final_q90",
    "error_failed_final_q90",
    "precision1_pre_target_stall_rate",
)


def _error_matrix(problem: Any) -> np.ndarray:
    frequency = int(problem.change_frequency)
    environments = int(problem.environment_number)
    error = np.asarray(problem.state.current_error, dtype=float)
    if error.size != frequency * environments:
        raise ValueError("Precision metrics require a complete FE budget")
    if not np.isfinite(error).all():
        raise ValueError("Precision metrics require finite known-optimum error")
    return error.reshape(environments, frequency)


def _stats(problem: Any) -> dict[str, float]:
    error = _error_matrix(problem)
    frequency = error.shape[1]
    reached = error <= TARGET_ERROR
    hit = np.any(reached, axis=1)
    first_fe = np.where(hit, np.argmax(reached, axis=1) + 1, frequency + 1)
    post = slice(1, None)
    first_tenth = max(1, int(np.ceil(frequency * 0.1)))
    last_half = frequency // 2
    final = error[:, -1]
    failures = final[~hit]
    window = max(1, int(np.ceil(frequency * 0.2)))
    if window < frequency:
        start = error[:, :-window]
        end = error[:, window:]
        # A 20%-of-environment window counts only while still above target;
        # ordinary slow convergence with measurable progress is not a stall.
        ineffective = start - end <= np.maximum(1e-6, 1e-4 * start)
        pre_target_stall = np.any((end > TARGET_ERROR) & ineffective, axis=1)
    else:
        pre_target_stall = np.zeros(error.shape[0], dtype=bool)
    return {
        "precision1_hit_rate": float(np.mean(hit)),
        "precision1_hit_rate_post_change": float(np.mean(hit[post])) if len(hit) > 1 else float("nan"),
        "precision1_fe_capped": float(np.mean(first_fe)),
        "precision1_fe_capped_post_change": float(np.mean(first_fe[post])) if len(hit) > 1 else float("nan"),
        "error_first_10pct_post_change": float(np.mean(error[post, :first_tenth])) if len(hit) > 1 else float("nan"),
        "error_last_50pct": float(np.mean(error[:, last_half:])),
        "error_last_50pct_post_change": float(np.mean(error[post, last_half:])) if len(hit) > 1 else float("nan"),
        "error_final_q90": float(np.quantile(final, 0.9)),
        "error_failed_final_q90": float(np.quantile(failures, 0.9)) if failures.size else float("nan"),
        "precision1_pre_target_stall_rate": float(np.mean(pre_target_stall)),
    }


class _PrecisionSummary(SummaryMetric):
    name: ClassVar[str]

    def compute(self, problem: Any, algorithm: Any | None = None) -> float:
        return _stats(problem)[self.name]


for _name in METRIC_NAMES:
    _class_name = "PrecisionMetric_" + _name
    _metric_class = type(_class_name, (_PrecisionSummary,), {
        "name": _name,
        "__module__": __name__,
    })
    globals()[_class_name] = metric_registry.register(_name)(_metric_class)


__all__ = ["TARGET_ERROR", "METRIC_NAMES"]
