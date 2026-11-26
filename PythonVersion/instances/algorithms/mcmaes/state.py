"""mCMAES state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.algorithm.population import Population


@dataclass(slots=True, kw_only=True)
class MCMAESSubPopulation(Population):
    """Single CMA-ES subpopulation represented by its current mean."""

    sigma: float
    pc: np.ndarray
    ps: np.ndarray
    b_matrix: np.ndarray
    d_vector: np.ndarray
    covariance: np.ndarray
    invsqrt_covariance: np.ndarray
    eigen_eval: int
    eval_count: int
    gbest_past_environment: np.ndarray
    fgbest_past_environment: float
    shifts: list[float] = field(default_factory=list)
    shift_severity: float = 1.0
    last_offspring_positions: np.ndarray | None = None

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.positions[0], dtype=float)

    @mean.setter
    def mean(self, value: np.ndarray) -> None:
        self.positions[0, :] = np.asarray(value, dtype=float)

    @property
    def mean_fitness(self) -> float:
        if self.fitness is None or self.fitness.size == 0:
            return float("nan")
        return float(self.fitness[0])

    @mean_fitness.setter
    def mean_fitness(self, value: float) -> None:
        if self.fitness is None or self.fitness.size == 0:
            self.fitness = np.asarray([float(value)], dtype=float)
        else:
            self.fitness[0] = float(value)

    def clone(self) -> "MCMAESSubPopulation":
        return MCMAESSubPopulation(
            positions=np.asarray(self.positions, dtype=float).copy(),
            fitness=None if self.fitness is None else np.asarray(self.fitness, dtype=float).copy(),
            metadata=dict(self.metadata),
            sigma=float(self.sigma),
            pc=np.asarray(self.pc, dtype=float).copy(),
            ps=np.asarray(self.ps, dtype=float).copy(),
            b_matrix=np.asarray(self.b_matrix, dtype=float).copy(),
            d_vector=np.asarray(self.d_vector, dtype=float).copy(),
            covariance=np.asarray(self.covariance, dtype=float).copy(),
            invsqrt_covariance=np.asarray(self.invsqrt_covariance, dtype=float).copy(),
            eigen_eval=int(self.eigen_eval),
            eval_count=int(self.eval_count),
            gbest_past_environment=np.asarray(self.gbest_past_environment, dtype=float).copy(),
            fgbest_past_environment=float(self.fgbest_past_environment),
            shifts=list(self.shifts),
            shift_severity=float(self.shift_severity),
            last_offspring_positions=None
            if self.last_offspring_positions is None
            else np.asarray(self.last_offspring_positions, dtype=float).copy(),
        )
