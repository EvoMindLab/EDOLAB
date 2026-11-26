"""TMIPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.tmipso.state import TMIPSOSwarm


@algorithm_registry.register("TMIPSO")
class TMIPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "TMIPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                50,
                "integer",
                (1, 1000),
                "Total number of particles in both exploit and explore populations.",
            ),
            "x": ParameterSpec(
                "x",
                0.729843788,
                "numeric",
                (0, 2),
                "Inertia weight controlling influence of previous velocity.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Cognitive acceleration coefficient scaling attraction to personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Social acceleration coefficient scaling attraction to swarm best.",
            ),
            "ExploitPopSize": ParameterSpec(
                "ExploitPopSize",
                20,
                "integer",
                (1, 1000),
                "Size of the exploit population used for local refinement.",
            ),
            "ExplorePopSize": ParameterSpec(
                "ExplorePopSize",
                20,
                "integer",
                (1, 1000),
                "Size of the explore population used for global search.",
            ),
            "MemorySize": ParameterSpec(
                "MemorySize",
                10,
                "integer",
                (1, 1000),
                "Maximum number of past best solutions retained for triggered memory.",
            ),
            "b1": ParameterSpec(
                "b1",
                0.001,
                "numeric",
                (0, 1),
                "Fitness-change threshold to activate triggered memory.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.exploit_pop_size = max(1, int(self.parameters["ExploitPopSize"]))
        self.explore_pop_size = max(1, int(self.parameters["ExplorePopSize"]))
        self.memory_size = max(1, int(self.parameters["MemorySize"]))
        self.b1 = float(self.parameters["b1"])
        self.diversity_plus = 1
        self.shift_severity = 1.0
        self.swarm_number = 1

        # MATLAB reads x/c1/c2 from the config and then immediately overwrites them.
        self.x = 0.729843788
        self.c1 = 2.05
        self.c2 = 2.05

        self.exploit_pop = self._generate_subpopulation(problem, self.exploit_pop_size)
        self.explore_pop = self._generate_subpopulation(problem, self.explore_pop_size)
        self.memory_position = np.empty((0, self.dimension), dtype=float)
        self.memory_fitness = np.empty(0, dtype=float)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        self._move_swarm(self.exploit_pop, problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._move_swarm(self.explore_pop, problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self.explore_pop.fitness_history = np.append(
            self.explore_pop.fitness_history,
            float(self.explore_pop.best_fitness),
        )
        self._trigger_memory_if_needed(problem)

    def on_environment_change(self, problem) -> None:
        values = evaluate_(
            problem,
            self.exploit_pop.pbest_positions,
            self.exploit_pop.pbest_positions.shape[0],
        )
        if problem.state.recent_change or problem.state.terminated:
            return
        self.exploit_pop.pbest_fitness = self._nan_to_negative_infinity(values)
        self._reset_best(self.exploit_pop)

        values = evaluate_(
            problem,
            self.explore_pop.pbest_positions,
            self.explore_pop.pbest_positions.shape[0],
        )
        if problem.state.recent_change or problem.state.terminated:
            return
        self.explore_pop.pbest_fitness = self._nan_to_negative_infinity(values)
        self._reset_best(self.explore_pop)

        if self.memory_position.shape[0] == 0:
            self.memory_fitness = np.empty(0, dtype=float)
            return
        values = evaluate_(problem, self.memory_position, self.memory_position.shape[0])
        if problem.state.recent_change or problem.state.terminated:
            return
        self.memory_fitness = self._nan_to_negative_infinity(values)

    def _generate_subpopulation(self, problem, population_size: int) -> TMIPSOSwarm:
        return sub_population_generator_tmipso(
            self.dimension,
            self.min_coordinate,
            self.max_coordinate,
            population_size,
            problem,
            self.rng,
        )

    def _move_swarm(self, swarm: TMIPSOSwarm, problem) -> None:
        rows = int(swarm.positions.shape[0])
        if rows <= 0:
            return

        r1 = self.rng.random((rows, self.dimension))
        r2 = self.rng.random((rows, self.dimension))
        swarm.velocities = self.x * (
            swarm.velocities
            + self.c1 * r1 * (swarm.pbest_positions - swarm.positions)
            + self.c2 * r2 * (swarm.best_position - swarm.positions)
        )
        swarm.positions = swarm.positions + swarm.velocities
        self._clip_to_bounds(swarm)

        values = evaluate_(problem, swarm.positions, rows)
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

    def _trigger_memory_if_needed(self, problem) -> None:
        history = np.asarray(self.explore_pop.fitness_history, dtype=float).reshape(-1)
        if history.size != 6:
            return

        tmp1 = float(np.sum(history[:5]))
        tmp2 = float(np.sum(history[1:6]))
        self.explore_pop.fitness_history = history[1:].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            averfit = float(np.divide(tmp2 - tmp1, tmp2))
        if not averfit < self.b1:
            return

        memorysize = int(self.memory_fitness.size)
        if memorysize >= self.memory_size:
            self._replace_closest_memory_pair()

        self.memory_position = np.vstack(
            [self.memory_position, self.explore_pop.best_position.copy()]
        )
        self.memory_fitness = np.append(
            self.memory_fitness,
            float(self.explore_pop.best_fitness),
        )

        self._reset_exploit_from_memory()
        self._apply_memory_immigrants(memorysize)

        self.explore_pop = self._generate_subpopulation(problem, self.explore_pop_size)

    def _replace_closest_memory_pair(self) -> None:
        memorysize = int(self.memory_fitness.size)
        if memorysize == 0:
            return
        if memorysize == 1:
            self.memory_position = np.empty((0, self.dimension), dtype=float)
            self.memory_fitness = np.empty(0, dtype=float)
            return

        closest_distance = np.inf
        first = 0
        second = 1
        for ii in range(memorysize - 1):
            for jj in range(ii + 1, memorysize):
                distance = float(
                    np.linalg.norm(self.memory_position[ii, :] - self.memory_position[jj, :])
                )
                if distance < closest_distance:
                    closest_distance = distance
                    first = ii
                    second = jj

        if self.memory_fitness[first] < self.memory_fitness[second]:
            delete_id = first
        else:
            delete_id = second
        self.memory_position = np.delete(self.memory_position, delete_id, axis=0)
        self.memory_fitness = np.delete(self.memory_fitness, delete_id)

    def _reset_exploit_from_memory(self) -> None:
        if self.memory_fitness.size == 0:
            return
        best_id = self._best_index(self.memory_fitness)
        best_value = float(self.memory_fitness[best_id])
        if best_value > float(self.exploit_pop.best_fitness):
            self.exploit_pop.best_fitness = best_value
            self.exploit_pop.best_position = self.memory_position[best_id, :].copy()

    def _apply_memory_immigrants(self, memorysize: int) -> None:
        if memorysize <= 0 or self.memory_fitness.size == 0:
            return
        worst_indices = np.argsort(
            np.nan_to_num(self.exploit_pop.pbest_fitness, nan=-np.inf)
        )
        count = min(
            int(memorysize),
            self.memory_fitness.size,
            worst_indices.size,
        )
        for ii in range(count):
            target = int(worst_indices[ii])
            self.exploit_pop.pbest_positions[target, :] = self.memory_position[ii, :]
            self.exploit_pop.pbest_fitness[target] = self.memory_fitness[ii]

    def _clip_to_bounds(self, swarm: TMIPSOSwarm) -> None:
        high = swarm.positions > self.max_coordinate
        low = swarm.positions < self.min_coordinate
        clipped = high | low
        if np.any(high):
            swarm.positions[high] = self.max_coordinate
        if np.any(low):
            swarm.positions[low] = self.min_coordinate
        if np.any(clipped):
            swarm.velocities[clipped] = 0.0

    def _reset_best(self, swarm: TMIPSOSwarm) -> None:
        best_id = self._best_index(swarm.pbest_fitness)
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    @staticmethod
    def _nan_to_negative_infinity(values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf).copy()

    @staticmethod
    def _best_index(values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        comparable = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)
        return int(np.argmax(comparable))


def sub_population_generator_tmipso(
    dimension: int,
    min_coordinate: float,
    max_coordinate: float,
    population_size: int,
    problem,
    rng: np.random.Generator | None = None,
) -> TMIPSOSwarm:
    rows = max(1, int(population_size))
    rng = np.random.default_rng() if rng is None else rng
    positions = float(min_coordinate) + (
        (float(max_coordinate) - float(min_coordinate))
        * rng.random((rows, int(dimension)))
    )
    fitness = evaluate_(problem, positions, rows)
    pbest_positions = positions.copy()
    if problem.state.recent_change or problem.state.terminated:
        fitness = np.full(rows, -np.inf, dtype=float)
        pbest_fitness = fitness.copy()
    else:
        fitness = np.nan_to_num(np.asarray(fitness, dtype=float), nan=-np.inf).copy()
        pbest_fitness = fitness.copy()
    best_id = int(np.argmax(pbest_fitness))
    best_value = float(pbest_fitness[best_id])
    return TMIPSOSwarm(
        positions=positions,
        velocities=np.zeros((rows, int(dimension)), dtype=float),
        fitness=fitness,
        pbest_positions=pbest_positions,
        pbest_fitness=pbest_fitness,
        best_fitness=best_value,
        best_position=pbest_positions[best_id, :].copy(),
        fitness_history=np.array([best_value], dtype=float),
    )


SubPopulationGenerator_TMIPSO = sub_population_generator_tmipso
