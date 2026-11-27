"""Search-space bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Bounds:
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def uniform(cls, lower: float, upper: float, dimension: int) -> "Bounds":
        return cls(
            lower=np.full(dimension, lower, dtype=float),
            upper=np.full(dimension, upper, dtype=float),
        )

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return self.lower + (self.upper - self.lower) * rng.random((count, self.lower.size))

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def clip_in_place(self, x: np.ndarray) -> None:
        np.clip(x, self.lower, self.upper, out=x)

    def reflect(self, x: np.ndarray) -> np.ndarray:
        reflected = np.asarray(x, dtype=float).copy()
        high = reflected > self.upper
        reflected[high] = (2 * np.broadcast_to(self.upper, reflected.shape)[high]) - reflected[high]
        low = reflected < self.lower
        reflected[low] = (2 * np.broadcast_to(self.lower, reflected.shape)[low]) - reflected[low]
        return self.clip(reflected)

