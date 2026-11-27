"""State of the single active CMSA subpopulation and filtered archive."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ArchivedSolution:
    position: np.ndarray
    fitness: float
    taboo_distance: float


@dataclass(slots=True)
class RestartState:
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    basis: np.ndarray
    scales: np.ndarray
    invsqrt: np.ndarray
    initial_radius: float
    generation: int = 0
    best_position: np.ndarray | None = None
    # The author's bestVal is the CURRENT generation's best, not best-so-far.
    best_fitness: float = -np.inf
    best_history: list[float] = field(default_factory=list)
    median_history: list[float] = field(default_factory=list)
    criticality_history: list[float] = field(default_factory=list)
    merge_candidate_history: list[int] = field(default_factory=list)
    merge_count_history: list[int] = field(default_factory=list)
    last_criticality: float = 0.0
    check_merge_after: float = 0.0

