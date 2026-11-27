"""Reusable PSO operators."""

from __future__ import annotations

import numpy as np

from core.algorithm.operators.base import Operator
from core.algorithm.population.swarm import PSOSwarmState


class PSOOperator(Operator):
    name = "pso"

    def update_velocity(
        self,
        swarm: PSOSwarmState,
        inertia: float,
        cognitive: float,
        social: float,
        rng: np.random.Generator,
    ) -> None:
        if swarm.velocities is None or swarm.pbest_positions is None or swarm.best_position is None:
            raise ValueError("Swarm is missing PSO state.")
        r1 = rng.random(swarm.positions.shape)
        r2 = rng.random(swarm.positions.shape)
        swarm.velocities = (
            inertia * swarm.velocities
            + cognitive * r1 * (swarm.pbest_positions - swarm.positions)
            + social * r2 * (swarm.best_position - swarm.positions)
        )

    def move(self, swarm: PSOSwarmState) -> None:
        if swarm.velocities is None:
            raise ValueError("Swarm is missing velocities.")
        swarm.positions = swarm.positions + swarm.velocities

    def update_and_move(
        self,
        swarm: PSOSwarmState,
        inertia: float,
        cognitive: float,
        social: float,
        rng: np.random.Generator,
    ) -> None:
        self.update_velocity(swarm, inertia, cognitive, social, rng)
        self.move(swarm)

    def clip_to_bounds(
        self,
        swarm: PSOSwarmState,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
        *,
        zero_velocity: bool = True,
    ) -> np.ndarray:
        """Clip positions and optionally zero velocities for clipped entries."""

        positions = np.asarray(swarm.positions, dtype=float)
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        high = positions > upper_arr
        low = positions < lower_arr
        clipped = high | low
        if np.any(high):
            positions[high] = np.broadcast_to(upper_arr, positions.shape)[high]
        if np.any(low):
            positions[low] = np.broadcast_to(lower_arr, positions.shape)[low]
        swarm.positions = positions
        if zero_velocity and swarm.velocities is not None and np.any(clipped):
            velocities = np.asarray(swarm.velocities, dtype=float)
            velocities[clipped] = 0.0
            swarm.velocities = velocities
        return clipped


def update_velocity(
    swarm: PSOSwarmState,
    inertia: float,
    cognitive: float,
    social: float,
    rng: np.random.Generator,
) -> None:
    PSOOperator().update_velocity(swarm, inertia, cognitive, social, rng)


def move(swarm: PSOSwarmState) -> None:
    PSOOperator().move(swarm)
