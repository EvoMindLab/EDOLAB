"""Generic continuous optimization population contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class PopulationBase(ABC):
    """Base contract for population-like optimization state."""

    positions: np.ndarray
    fitness: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return int(np.asarray(self.positions).shape[0])

    @property
    def dimension(self) -> int:
        positions = np.asarray(self.positions)
        if positions.ndim < 2:
            return 0
        return int(positions.shape[1])

    @abstractmethod
    def clone(self) -> "PopulationBase":
        """Return an independent copy of the population."""


@dataclass(slots=True)
class Population(PopulationBase):
    positions: np.ndarray
    fitness: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "Population":
        return Population(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None if self.fitness is None else np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
        )
