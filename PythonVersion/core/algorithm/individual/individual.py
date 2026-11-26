"""Generic continuous optimization individual contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class IndividualBase(ABC):
    """Base contract for optimization individuals."""

    position: np.ndarray
    fitness: float | None
    metadata: dict[str, Any]

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.position).reshape(-1).size)

    @abstractmethod
    def clone(self) -> "IndividualBase":
        """Return an independent copy of the individual."""


@dataclass(slots=True)
class Individual(IndividualBase):
    position: np.ndarray
    fitness: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "Individual":
        return Individual(
            position=np.asarray(self.position, dtype=float).copy(),
            fitness=self.fitness,
            metadata=dict(self.metadata),
        )
