"""AMPDE state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.algorithm.population import Population


@dataclass(slots=True)
class AMPDEMapRecord:
    mean: float = 0.0
    std_dev: float = 0.0
    num: int = 0
    indivs: list[float] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class AMPDESubPopulation(Population):
    gbest_past_environment: np.ndarray
    donor_positions: np.ndarray
    trial_positions: np.ndarray
    is_hibernated: int
    stagnating_count: int
    is_stagnated: int
    best_id: int
    best_position: np.ndarray
    best_fitness: float
    center: np.ndarray
    init_radius: float
    current_radius: float

    def clone(self) -> "AMPDESubPopulation":
        return AMPDESubPopulation(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None
            if self.fitness is None
            else np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
            gbest_past_environment=np.asarray(
                self.gbest_past_environment,
                dtype=float,
            ).copy(),
            donor_positions=np.asarray(self.donor_positions, dtype=float).copy(),
            trial_positions=np.asarray(self.trial_positions, dtype=float).copy(),
            is_hibernated=int(self.is_hibernated),
            stagnating_count=int(self.stagnating_count),
            is_stagnated=int(self.is_stagnated),
            best_id=int(self.best_id),
            best_position=np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=float(self.best_fitness),
            center=np.asarray(self.center, dtype=float).copy(),
            init_radius=float(self.init_radius),
            current_radius=float(self.current_radius),
        )
