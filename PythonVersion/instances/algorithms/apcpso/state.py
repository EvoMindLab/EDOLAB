"""APCPSO state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True, kw_only=True)
class APCPSOSwarm(Swarm):
    shifts: list[float]
    is_converged: int
    is_excluded: int
    best_id: int
    center: np.ndarray
    init_radius: float
    current_radius: float
    local_iter: int
