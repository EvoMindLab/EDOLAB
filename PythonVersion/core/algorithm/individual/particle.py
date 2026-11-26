"""PSO particle data structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.individual.individual import Individual


@dataclass(slots=True)
class Particle(Individual):
    velocity: np.ndarray | None = None
    pbest_position: np.ndarray | None = None
    pbest_fitness: float | None = None

    def clone(self) -> "Particle":
        return Particle(
            position=np.asarray(self.position, dtype=float).copy(),
            fitness=self.fitness,
            metadata=dict(self.metadata),
            velocity=None if self.velocity is None else np.asarray(self.velocity, dtype=float).copy(),
            pbest_position=None
            if self.pbest_position is None
            else np.asarray(self.pbest_position, dtype=float).copy(),
            pbest_fitness=self.pbest_fitness,
        )
