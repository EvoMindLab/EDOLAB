"""ACFPSO state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True, kw_only=True)
class ACFPSOSwarm(Swarm):
    gbest_past_environment: np.ndarray
    previous_best_position: np.ndarray
    shifts: list[float]
    sleep: int
    iteration_counter: int
    phase: int
    center: np.ndarray
    diversity: np.ndarray
