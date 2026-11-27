"""HIA_DRI_AN state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True, kw_only=True)
class HiaDriAnPopulation(Swarm):
    """Single flat HIA_DRI_AN population with NBNC guide metadata."""

    slbest_positions: np.ndarray
    sgbest_positions: np.ndarray
    species_id: np.ndarray
    best_id: int

    def clone(self) -> "HiaDriAnPopulation":
        return HiaDriAnPopulation(
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
            slbest_positions=np.asarray(self.slbest_positions, dtype=float).copy(),
            sgbest_positions=np.asarray(self.sgbest_positions, dtype=float).copy(),
            species_id=np.asarray(self.species_id, dtype=int).copy(),
            best_id=int(self.best_id),
        )


@dataclass(slots=True, kw_only=True)
class HiaDriAnCMAState:
    """Minimal per-species CMA-ES state used in HIA_DRI_AN phase 2."""

    lambda_: int
    mu: int
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    b_matrix: np.ndarray
    d_vector: np.ndarray
    pc: np.ndarray
    ps: np.ndarray
    gen_count: int
    converged: bool
    best_x: np.ndarray
    best_fit: float
    weights: np.ndarray
    mueff: float
    cs_param: float
    damps: float
    cc: float
    c1: float
    cmu: float
    chi_n: float
