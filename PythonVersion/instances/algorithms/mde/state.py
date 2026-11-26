"""mDE state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.algorithm.population import Population


@dataclass(slots=True, kw_only=True)
class MDESubPopulation(Population):
    """Subpopulation state used by mDE's multi-swarm DE search."""

    gbest_past_environment: np.ndarray
    offspring_positions: np.ndarray
    offspring_fitness: np.ndarray
    donor_positions: np.ndarray
    crossover_rates: np.ndarray
    scale_factors: np.ndarray
    scale_factor_lower_bound: float
    scale_factor_upper_bound: float
    r1: float
    r2: float
    shifts: list[float] = field(default_factory=list)
    best_id: int = 0
    best_position: np.ndarray | None = None
    best_fitness: float = -np.inf

    def clone(self) -> "MDESubPopulation":
        return MDESubPopulation(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None
            if self.fitness is None
            else np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
            gbest_past_environment=np.asarray(
                self.gbest_past_environment,
                dtype=float,
            ).copy(),
            offspring_positions=np.asarray(self.offspring_positions, dtype=float).copy(),
            offspring_fitness=np.asarray(self.offspring_fitness, dtype=float).copy(),
            donor_positions=np.asarray(self.donor_positions, dtype=float).copy(),
            crossover_rates=np.asarray(self.crossover_rates, dtype=float).copy(),
            scale_factors=np.asarray(self.scale_factors, dtype=float).copy(),
            scale_factor_lower_bound=float(self.scale_factor_lower_bound),
            scale_factor_upper_bound=float(self.scale_factor_upper_bound),
            r1=float(self.r1),
            r2=float(self.r2),
            shifts=list(self.shifts),
            best_id=int(self.best_id),
            best_position=None
            if self.best_position is None
            else np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=float(self.best_fitness),
        )
