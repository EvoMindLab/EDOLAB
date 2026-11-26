"""RPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.rpso.state import RPSOSwarm


@algorithm_registry.register("RPSO")
class RPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "RPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                5,
                "integer",
                (0, 1000),
                "Number of particles in each sub-swarm.",
            ),
            "x": ParameterSpec(
                "x",
                0.729843788,
                "numeric",
                (0, 2),
                "Inertia weight used in particle velocity update.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Cognitive coefficient that scales influence of personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Social coefficient that scales influence of global best.",
            ),
            "RandomizingPercentage": ParameterSpec(
                "RandomizingPercentage",
                0.5,
                "numeric",
                (0, 1),
                "Percentage of particles randomly reinitialized when the environment changes.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.diversity_plus = 1
        self.swarm_number = 1
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        randomizing_percentage = float(self.parameters["RandomizingPercentage"])
        self.number_of_randomizing_particles = int(
            np.ceil(self.population_size * randomizing_percentage)
        )
        self.number_of_randomizing_particles = min(
            self.population_size, max(0, self.number_of_randomizing_particles)
        )
        self.pop = self._generate_subpopulation(problem)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        swarm = self.pop
        inertia = float(self.parameters["x"])
        cognitive = float(self.parameters["c1"])
        social = float(self.parameters["c2"])
        r1 = self.rng.random((self.population_size, self.dimension))
        r2 = self.rng.random((self.population_size, self.dimension))
        swarm.velocities = inertia * (
            swarm.velocities
            + cognitive * r1 * (swarm.pbest_positions - swarm.positions)
            + social * r2 * (swarm.best_position - swarm.positions)
        )
        swarm.positions = swarm.positions + swarm.velocities
        self._clip_to_bounds(swarm)

        values = evaluate_(problem, swarm.positions, self.population_size)
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm.fitness = self._nan_to_negative_infinity(values)

        improved = swarm.fitness > np.nan_to_num(swarm.pbest_fitness, nan=-np.inf)
        swarm.pbest_fitness[improved] = swarm.fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]

        best_id = self._best_index(swarm.pbest_fitness)
        best_value = float(swarm.pbest_fitness[best_id])
        if best_value > float(swarm.best_fitness):
            swarm.best_fitness = best_value
            swarm.best_position = swarm.pbest_positions[best_id, :].copy()
            swarm.best_id = best_id

    def on_environment_change(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        swarm = self.pop
        randomizing_count = min(
            self.population_size, max(0, int(self.number_of_randomizing_particles))
        )
        sorted_list = np.argsort(np.nan_to_num(swarm.pbest_fitness, nan=-np.inf))
        if randomizing_count > 0:
            randomizing_ids = sorted_list[:randomizing_count]
            new_positions = self._random_positions(randomizing_count)
            swarm.positions[randomizing_ids, :] = new_positions
            swarm.velocities[randomizing_ids, :] = 0.0
            swarm.pbest_positions[randomizing_ids, :] = new_positions

        values = evaluate_(problem, swarm.pbest_positions, self.population_size)
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm.pbest_fitness = self._nan_to_negative_infinity(values)
        self._reset_best(swarm)

    def _generate_subpopulation(self, problem) -> RPSOSwarm:
        positions = self._random_positions(self.population_size)
        fitness = self._nan_to_negative_infinity(
            evaluate_(problem, positions, self.population_size)
        )
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        best_id = self._best_index(pbest_fitness)
        return RPSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            velocities=np.zeros((self.population_size, self.dimension), dtype=float),
            shifts=[],
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
            best_id=best_id,
        )

    def _clip_to_bounds(self, swarm: RPSOSwarm) -> None:
        high = swarm.positions > self.max_coordinate
        low = swarm.positions < self.min_coordinate
        clipped = high | low
        if np.any(high):
            swarm.positions[high] = self.max_coordinate
        if np.any(low):
            swarm.positions[low] = self.min_coordinate
        if np.any(clipped):
            swarm.velocities[clipped] = 0.0

    def _reset_best(self, swarm: RPSOSwarm) -> None:
        best_id = self._best_index(swarm.pbest_fitness)
        swarm.best_id = best_id
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _random_positions(self, rows: int) -> np.ndarray:
        return self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((rows, self.dimension))
        )

    @staticmethod
    def _nan_to_negative_infinity(values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf).copy()

    @staticmethod
    def _best_index(values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        comparable = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)
        return int(np.argmax(comparable))
