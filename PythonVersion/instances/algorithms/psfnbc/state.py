"""psfNBC state models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.population import Swarm


@dataclass(slots=True, kw_only=True)
class PsfNBCSwarm(Swarm):
    gbest_past_environment: np.ndarray
    shifts: list[float]
