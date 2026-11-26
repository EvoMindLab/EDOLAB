"""TMIPSO state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True, kw_only=True)
class TMIPSOSwarm(Swarm):
    fitness_history: np.ndarray
