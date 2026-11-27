"""SPMP trends, built after the run from the passive monitor (components/monitor.py).

They are run trends: nothing is sampled per evaluation, so requesting them adds no search-time work. FE-axis trends
hold the state at 20 points of every environment; environment-axis trends hold one value per environment.
"""

from __future__ import annotations

from typing import Any, ClassVar

from core.evaluation.base import RuntimeMetric


class _MonitorTrend(RuntimeMetric):
    kind: ClassVar[str] = "run_trend"
    axis: ClassVar[str] = "fe"

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.values: list[float] = []

    def sample_run(self, problem: Any, algorithm: Any) -> None:
        monitor = getattr(algorithm, "monitor", None)
        if monitor is None:
            return
        monitor.finish(problem)
        for key, value in self.series(monitor):
            if value is not None:
                self.keys.append(int(key))
                self.values.append(float(value))

    def series(self, monitor):
        raise NotImplementedError

    def trend(self) -> dict[str, Any] | None:
        if not self.keys:
            return None
        return {"axis": self.axis, "key": self.keys, "value": self.values, "name": self.name}


class _SnapshotTrend(_MonitorTrend):
    field: ClassVar[str]

    def series(self, monitor):
        return [(key, state[self.field]) for key, state in monitor.snapshots]


class _EnvironmentTrend(_MonitorTrend):
    axis: ClassVar[str] = "environment"

    def series(self, monitor):
        return [(record["e"], self.value(record)) for record in monitor.records]

    def value(self, record):
        raise NotImplementedError


def _share(part: int, record: dict) -> float | None:
    return part / record["fe_total"] if record["fe_total"] else None


class SubspaceTrend(_SnapshotTrend):
    name: ClassVar[str] = "spmp_subspaces"
    field: ClassVar[str] = "subspaces"


class BasinTrend(_SnapshotTrend):
    name: ClassVar[str] = "spmp_basins"
    field: ClassVar[str] = "basins"


class PopulationTrend(_SnapshotTrend):
    name: ClassVar[str] = "spmp_populations"
    field: ClassVar[str] = "populations"


class ActivePopulationTrend(_SnapshotTrend):
    name: ClassVar[str] = "spmp_active_populations"
    field: ClassVar[str] = "active"


class InactivePopulationTrend(_SnapshotTrend):
    name: ClassVar[str] = "spmp_inactive_populations"
    field: ClassVar[str] = "inactive"


class CloudRadiusTrend(_SnapshotTrend):
    name: ClassVar[str] = "spmp_cloud_radius"
    field: ClassVar[str] = "cloud_radius"


class TopShareTrend(_EnvironmentTrend):
    name: ClassVar[str] = "spmp_fe_top1_share"

    def value(self, record):
        return record["top1"]


class FundedPopulationTrend(_EnvironmentTrend):
    name: ClassVar[str] = "spmp_funded_populations"

    def value(self, record):
        return record["funded"]


class ChangeShareTrend(_EnvironmentTrend):
    name: ClassVar[str] = "spmp_fe_change_share"

    def value(self, record):
        return _share(record["fe_by"]["change"], record)


class CreationShareTrend(_EnvironmentTrend):
    name: ClassVar[str] = "spmp_fe_creation_share"

    def value(self, record):
        fe_by = record["fe_by"]
        return _share(fe_by["init"] + fe_by["replenish"] + fe_by["ghost"], record)


class RemovedShareTrend(_EnvironmentTrend):
    name: ClassVar[str] = "spmp_fe_removed_share"

    def value(self, record):
        return _share(record["fe_removed"], record)


class ShiftEstimateTrend(_EnvironmentTrend):
    name: ClassVar[str] = "spmp_shift_estimate"

    def value(self, record):
        return record.get("shift_estimate")


METRICS = (SubspaceTrend, BasinTrend, PopulationTrend, ActivePopulationTrend, InactivePopulationTrend,
           CloudRadiusTrend, TopShareTrend, FundedPopulationTrend, ChangeShareTrend, CreationShareTrend,
           RemovedShareTrend, ShiftEstimateTrend)
METRIC_NAMES = tuple(metric.name for metric in METRICS)

__all__ = ["METRICS", "METRIC_NAMES"]
