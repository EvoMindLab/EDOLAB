"""CPSOR state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True, kw_only=True)
class CPSORSwarm(Swarm):
    gbest_past_environment: np.ndarray
    shifts: list[float]
    is_converged: int
    best_id: int
    center: np.ndarray
    init_radius: float
    current_radius: float

    def clone(self) -> "CPSORSwarm":
        return CPSORSwarm(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None
            if self.fitness is None
            else np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
            velocities=None
            if self.velocities is None
            else np.asarray(self.velocities, dtype=float).copy(),
            pbest_positions=None
            if self.pbest_positions is None
            else np.asarray(self.pbest_positions, dtype=float).copy(),
            pbest_fitness=None
            if self.pbest_fitness is None
            else np.asarray(self.pbest_fitness, dtype=float).copy(),
            best_position=None
            if self.best_position is None
            else np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=None if self.best_fitness is None else float(self.best_fitness),
            gbest_past_environment=np.asarray(
                self.gbest_past_environment,
                dtype=float,
            ).copy(),
            shifts=list(self.shifts),
            is_converged=int(self.is_converged),
            best_id=int(self.best_id),
            center=np.asarray(self.center, dtype=float).copy(),
            init_radius=float(self.init_radius),
            current_radius=float(self.current_radius),
        )
