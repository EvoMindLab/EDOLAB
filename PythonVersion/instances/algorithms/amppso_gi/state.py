"""AMPPSO_GI state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True)
class AMPMapRecord:
    mean: float = 0.0
    std_dev: float = 0.0
    num: int = 0
    indivs: list[float] = field(default_factory=list)


@dataclass(slots=True)
class AMPHiberArea:
    optima: np.ndarray
    bounds: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class AMPPSOSwarm(Swarm):
    gbest_past_environment: np.ndarray
    shifts: list[float]
    is_hibernated: int
    stagnating_count: int
    is_stagnated: int
    best_id: int
    center: np.ndarray
    init_radius: float
    current_radius: float
