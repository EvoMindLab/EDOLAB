"""HmSO state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True, kw_only=True)
class HmSOSwarm(Swarm):
    gbest_past_environment: np.ndarray
    best_id: int
    center: np.ndarray
    current_radius: float
    is_converged: int
    is_remove: int
