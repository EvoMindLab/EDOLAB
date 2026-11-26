"""mjDE state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.algorithm.population import Population


@dataclass(slots=True, kw_only=True)
class MjDESubPopulation(Population):
    gbest_past_environment: np.ndarray
    fgbest_past_environment: np.ndarray
    offspring_positions: np.ndarray
    offspring_fitness: np.ndarray
    donor: np.ndarray
    cr: np.ndarray
    f: np.ndarray
    f_lb: float
    f_ub: float
    r1: float
    r2: float
    shifts: list[float] = field(default_factory=list)
    best_id: int
    best_position: np.ndarray
    best_fitness: float

    def clone(self) -> "MjDESubPopulation":
        return MjDESubPopulation(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None
            if self.fitness is None
            else np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
            gbest_past_environment=np.asarray(
                self.gbest_past_environment,
                dtype=float,
            ).copy(),
            fgbest_past_environment=np.asarray(
                self.fgbest_past_environment,
                dtype=float,
            ).copy(),
            offspring_positions=np.asarray(
                self.offspring_positions,
                dtype=float,
            ).copy(),
            offspring_fitness=np.asarray(self.offspring_fitness, dtype=float).copy(),
            donor=np.asarray(self.donor, dtype=float).copy(),
            cr=np.asarray(self.cr, dtype=float).copy(),
            f=np.asarray(self.f, dtype=float).copy(),
            f_lb=float(self.f_lb),
            f_ub=float(self.f_ub),
            r1=float(self.r1),
            r2=float(self.r2),
            shifts=list(self.shifts),
            best_id=int(self.best_id),
            best_position=np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=float(self.best_fitness),
        )
