"""PSO swarm data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from core.algorithm.population.population import Population


@runtime_checkable
class PSOSwarmState(Protocol):
    """Structural contract consumed by reusable PSO operators."""

    positions: np.ndarray
    velocities: np.ndarray | None
    pbest_positions: np.ndarray | None
    pbest_fitness: np.ndarray | None
    best_position: np.ndarray | None
    best_fitness: float | None


@dataclass(slots=True)
class Swarm(Population):
    velocities: np.ndarray | None = None
    pbest_positions: np.ndarray | None = None
    pbest_fitness: np.ndarray | None = None
    best_position: np.ndarray | None = None
    best_fitness: float | None = None

    def clone(self) -> "Swarm":
        return Swarm(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None if self.fitness is None else np.asarray(self.fitness, dtype=float).copy(),
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
            best_fitness=self.best_fitness,
        )
