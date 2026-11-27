"""DynPopDE state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Population


@dataclass(slots=True, kw_only=True)
class DynPopDESubPopulation(Population):
    donor_positions: np.ndarray
    trial_positions: np.ndarray
    crossover_rates: np.ndarray
    scale_factors: np.ndarray
    strategies: np.ndarray
    individual_types: np.ndarray
    gbest_past_environment: np.ndarray
    best_id: int
    best_position: np.ndarray
    best_fitness: float
    center: np.ndarray
    current_radius: float
    reinit_state: int
    delete_state: int
    pen: int

    def clone(self) -> "DynPopDESubPopulation":
        return DynPopDESubPopulation(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None
            if self.fitness is None
            else np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
            donor_positions=np.asarray(self.donor_positions, dtype=float).copy(),
            trial_positions=np.asarray(self.trial_positions, dtype=float).copy(),
            crossover_rates=np.asarray(self.crossover_rates, dtype=float).copy(),
            scale_factors=np.asarray(self.scale_factors, dtype=float).copy(),
            strategies=np.asarray(self.strategies, dtype=int).copy(),
            individual_types=np.asarray(self.individual_types, dtype=int).copy(),
            gbest_past_environment=np.asarray(
                self.gbest_past_environment,
                dtype=float,
            ).copy(),
            best_id=int(self.best_id),
            best_position=np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=float(self.best_fitness),
            center=np.asarray(self.center, dtype=float).copy(),
            current_radius=float(self.current_radius),
            reinit_state=int(self.reinit_state),
            delete_state=int(self.delete_state),
            pen=int(self.pen),
        )
