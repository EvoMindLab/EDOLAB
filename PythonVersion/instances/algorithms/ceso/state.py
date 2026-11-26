"""CESO state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Population, Swarm


@dataclass(slots=True, kw_only=True)
class CESOSwarm(Swarm):
    """PSO-based CESO population."""


@dataclass(slots=True, kw_only=True)
class CESOCRDEPopulation(Population):
    best_position: np.ndarray
    best_fitness: float
