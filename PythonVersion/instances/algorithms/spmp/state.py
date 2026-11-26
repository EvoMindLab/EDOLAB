"""SPMP state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.algorithm.population import Swarm

@dataclass(slots=True, kw_only=True)
class SPMPMapRecord:
    mean: float = 0.0
    std_dev: float = 0.0
    num: int = 0
    indivs: list[float] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class SPMPHiberArea:
    optima: np.ndarray
    bounds: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class SPMPBoA:
    subspaces: np.ndarray
    allSubspaces: np.ndarray
    vol: float
    allVol: float
    peakSubspace: int
    robustPeakScore: float
    peakFitness: float
    isGhost: bool
    hasGenerated: bool = False
    persistence: float = 0.0
    confidence: float = 0.0
    sampleCount: int = 0
    coreSubspaceCount: int = 0
    allSubspaceCount: int = 0
    # Added: best History point in the mode cell, a partition-independent handle on the basin.
    peakPoint: np.ndarray | None = None


@dataclass(slots=True)
class SPMPSpatialModelState:
    IsReliable: bool = False
    Mode: str = "conservative"
    Score: float = 0.0
    ReferenceScore: float = 0.0
    GuidanceWeight: float = 0.0
    GuidanceShare: float = 0.0
    SampledRatio: float = 0.0
    OverlapRatio: float = 1.0
    ReliableBoACount: int = 0
    ConfidenceMean: float = 0.0
    Reason: str = "not_initialized"
    HistoryWindowFE: int = 0


@dataclass(slots=True, kw_only=True)
class SPMPSwarm(Swarm):
    gbest_past_environment: np.ndarray
    shifts: list[float]
    is_hibernated: int
    stagnating_count: int
    is_stagnated: int
    best_id: int
    center: np.ndarray
    init_radius: float
    current_radius: float
    swarm_type: str = "Exploratory"
    stagnation_count: int = 0
    last_radius: float = 0.0
    # Added: identity for the monitor (-1 until the algorithm adopts the swarm) and the tracker's own
    # step size for its local cloud (per coordinate) with the smoothed success rate that adapts it.
    uid: int = -1
    cloud_radius: float = 0.0
    success_rate: float = 0.0

    @property
    def X(self) -> np.ndarray:
        return self.positions

    @X.setter
    def X(self, value: np.ndarray) -> None:
        self.positions = np.asarray(value, dtype=float)

    @property
    def Velocity(self) -> np.ndarray:
        return self.velocities

    @Velocity.setter
    def Velocity(self, value: np.ndarray) -> None:
        self.velocities = np.asarray(value, dtype=float)

    @property
    def FitnessValue(self) -> np.ndarray:
        return self.fitness

    @FitnessValue.setter
    def FitnessValue(self, value: np.ndarray) -> None:
        self.fitness = np.asarray(value, dtype=float)

    @property
    def PbestPosition(self) -> np.ndarray:
        return self.pbest_positions

    @PbestPosition.setter
    def PbestPosition(self, value: np.ndarray) -> None:
        self.pbest_positions = np.asarray(value, dtype=float)

    @property
    def PbestValue(self) -> np.ndarray:
        return self.pbest_fitness

    @PbestValue.setter
    def PbestValue(self, value: np.ndarray) -> None:
        self.pbest_fitness = np.asarray(value, dtype=float)

    @property
    def GbestValue(self) -> float:
        return float(self.best_fitness)

    @GbestValue.setter
    def GbestValue(self, value: float) -> None:
        self.best_fitness = float(value)

    @property
    def GbestPosition(self) -> np.ndarray:
        return self.best_position

    @GbestPosition.setter
    def GbestPosition(self, value: np.ndarray) -> None:
        self.best_position = np.asarray(value, dtype=float)

    @property
    def GbestID(self) -> int:
        return int(self.best_id)

    @GbestID.setter
    def GbestID(self, value: int) -> None:
        self.best_id = int(value)

    @property
    def Gbest_past_environment(self) -> np.ndarray:
        return self.gbest_past_environment

    @Gbest_past_environment.setter
    def Gbest_past_environment(self, value: np.ndarray) -> None:
        self.gbest_past_environment = np.asarray(value, dtype=float)

    @property
    def Shifts(self) -> list[float]:
        return self.shifts

    @Shifts.setter
    def Shifts(self, value: list[float]) -> None:
        self.shifts = list(value)

    @property
    def Center(self) -> np.ndarray:
        return self.center

    @Center.setter
    def Center(self, value: np.ndarray) -> None:
        self.center = np.asarray(value, dtype=float)

    @property
    def InitRadius(self) -> float:
        return float(self.init_radius)

    @InitRadius.setter
    def InitRadius(self, value: float) -> None:
        self.init_radius = float(value)

    @property
    def CurrentRadius(self) -> float:
        return float(self.current_radius)

    @CurrentRadius.setter
    def CurrentRadius(self, value: float) -> None:
        self.current_radius = float(value)

    @property
    def IsHibernated(self) -> int:
        return int(self.is_hibernated)

    @IsHibernated.setter
    def IsHibernated(self, value: int) -> None:
        self.is_hibernated = int(value)

    @property
    def IsStagnated(self) -> int:
        return int(self.is_stagnated)

    @IsStagnated.setter
    def IsStagnated(self, value: int) -> None:
        self.is_stagnated = int(value)

    @property
    def StagnatingCount(self) -> int:
        return int(self.stagnating_count)

    @StagnatingCount.setter
    def StagnatingCount(self, value: int) -> None:
        self.stagnating_count = int(value)

    @property
    def SwarmType(self) -> str:
        return self.swarm_type

    @SwarmType.setter
    def SwarmType(self, value: str) -> None:
        self.swarm_type = str(value)

    @property
    def StagnationCount(self) -> int:
        return int(self.stagnation_count)

    @StagnationCount.setter
    def StagnationCount(self, value: int) -> None:
        self.stagnation_count = int(value)

    @property
    def LastRadius(self) -> float:
        return float(self.last_radius)

    @LastRadius.setter
    def LastRadius(self, value: float) -> None:
        self.last_radius = float(value)

    def clone(self) -> "SPMPSwarm":
        return SPMPSwarm(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
            velocities=np.asarray(self.velocities, dtype=float).copy(),
            pbest_positions=np.asarray(self.pbest_positions, dtype=float).copy(),
            pbest_fitness=np.asarray(self.pbest_fitness, dtype=float).copy(),
            best_position=np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=float(self.best_fitness),
            gbest_past_environment=np.asarray(
                self.gbest_past_environment,
                dtype=float,
            ).copy(),
            shifts=list(self.shifts),
            is_hibernated=int(self.is_hibernated),
            stagnating_count=int(self.stagnating_count),
            is_stagnated=int(self.is_stagnated),
            best_id=int(self.best_id),
            center=np.asarray(self.center, dtype=float).copy(),
            init_radius=float(self.init_radius),
            current_radius=float(self.current_radius),
            swarm_type=str(self.swarm_type),
            stagnation_count=int(self.stagnation_count),
            last_radius=float(self.last_radius),
            uid=int(self.uid),
            cloud_radius=float(self.cloud_radius),
            success_rate=float(self.success_rate),
        )


__all__ = [
    "SPMPMapRecord",
    "SPMPHiberArea",
    "SPMPBoA",
    "SPMPSpatialModelState",
    "SPMPSwarm",
]
