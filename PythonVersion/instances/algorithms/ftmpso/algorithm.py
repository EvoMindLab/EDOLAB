"""FTMPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.ftmpso.state import FTMPSOSwarm


@algorithm_registry.register("FTMPSO")
class FTMPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "FTMPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec("PopulationSize", 5, "integer", (0, 100)),
            "x": ParameterSpec("x", 0.729843788, "numeric", (0, 2)),
            "c1": ParameterSpec("c1", 2.05, "numeric", (0, 5)),
            "c2": ParameterSpec("c2", 2.05, "numeric", (0, 5)),
            "P": ParameterSpec("P", 0.5, "numeric", (0, 1)),
            "Q": ParameterSpec("Q", 0.5, "numeric", (0, 1)),
            "SleepingLimit": ParameterSpec("SleepingLimit", 0.4, "numeric", (0, 1)),
            "CFmin": ParameterSpec("CFmin", 0.8, "numeric", (0, 10)),
            "TryNumber": ParameterSpec("TryNumber", 20, "integer", (0, 100)),
            "FinderPopulationSize": ParameterSpec(
                "FinderPopulationSize", 10, "integer", (0, 100)
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.finder_population_size = max(1, int(self.parameters["FinderPopulationSize"]))
        self.try_number = max(0, int(self.parameters["TryNumber"]))
        self.free_swarm_id = 0
        self.exclusion_limit = self._threshold_for_swarm_count(10)
        self.convergence_limit = self.exclusion_limit
        self.diversity_plus = 1
        self.shift_severity = 1.0
        self.cloud = 0.2 * self.shift_severity
        self.swarm_number = 0
        self.pop: list[FTMPSOSwarm] = []
        self.pso_operator = PSOOperator()
        self.finder = self._generate_subpopulation(problem, self.finder_population_size)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        previous_swarm_number = self.swarm_number
        self._move_swarm(self.finder, problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._apply_finder_exclusion(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        converged = (
            float(
                np.linalg.norm(
                    self.finder.gbest_position_history[0, :] - self.finder.best_position
                )
            )
            < self.convergence_limit
        )
        self._append_finder_history()
        if converged:
            self._create_tracker_from_finder()
            self.finder = self._generate_subpopulation(problem, self.finder_population_size)
            if problem.state.recent_change or problem.state.terminated:
                return

        if previous_swarm_number != self.swarm_number:
            self.exclusion_limit = self._threshold_for_swarm_count(self.swarm_number)

        if self.swarm_number <= 0:
            return

        previous_swarm_number = self.swarm_number
        for swarm in list(self.pop):
            if swarm.sleeping == 0:
                self._move_swarm(swarm, problem)
                if problem.state.recent_change or problem.state.terminated:
                    return

        self._apply_tracker_exclusion()
        self._update_sleeping()
        self._local_search(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        if previous_swarm_number != self.swarm_number:
            self.exclusion_limit = self._threshold_for_swarm_count(self.swarm_number)

    def on_environment_change(self, problem) -> None:
        shift_samples: list[float] = []
        for index, swarm in enumerate(self.pop):
            if index == self.free_swarm_id:
                continue
            if not np.isnan(swarm.gbest_past_environment).any():
                shift = float(np.linalg.norm(swarm.gbest_past_environment - swarm.best_position))
                swarm.shifts.append(shift)
            shift_samples.extend(swarm.shifts)
        if shift_samples:
            self.shift_severity = float(np.mean(shift_samples))
        self.cloud = 0.2 * self.shift_severity

        p_scale = float(self.parameters["P"]) * self.shift_severity
        q_scale = float(self.parameters["Q"]) * self.shift_severity
        for swarm in self.pop:
            swarm.positions = (
                np.tile(swarm.best_position, (self.population_size, 1))
                + (2.0 * self.rng.random((self.population_size, self.dimension)) - 1.0)
                * p_scale
            )
            swarm.positions[0, :] = swarm.best_position
            swarm.velocities = (
                2.0 * self.rng.random((self.population_size, self.dimension)) - 1.0
            ) * q_scale
            swarm.sleeping = 0

        for swarm in self.pop:
            values = evaluate_(problem, swarm.positions, self.population_size)
            if problem.state.recent_change or problem.state.terminated:
                return
            swarm.fitness = np.nan_to_num(values, nan=-np.inf)
            swarm.pbest_fitness = swarm.fitness.copy()
            swarm.pbest_positions = swarm.positions.copy()
            swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)

    def _generate_subpopulation(self, problem, population_size: int) -> FTMPSOSwarm:
        rows = max(1, int(population_size))
        positions = self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((rows, self.dimension))
        )
        fitness = evaluate_(problem, positions, rows)
        pbest_positions = positions.copy()
        if problem.state.recent_change or problem.state.terminated:
            fitness = np.full(rows, -np.inf)
            pbest_fitness = fitness.copy()
        else:
            fitness = np.nan_to_num(fitness, nan=-np.inf)
            pbest_fitness = fitness.copy()
        best_id = int(np.argmax(pbest_fitness))
        return FTMPSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            shifts=[],
            sleeping=0,
            k=2,
            convergence_limit=1.0,
            gbest_position_history=np.full((2, self.dimension), np.inf),
            velocities=np.zeros((rows, self.dimension)),
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
        )

    def _move_swarm(self, swarm: FTMPSOSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        constriction = float(self.parameters["x"])
        self.pso_operator.update_and_move(
            swarm,
            inertia=constriction,
            cognitive=constriction * float(self.parameters["c1"]),
            social=constriction * float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(swarm, self.min_coordinate, self.max_coordinate)
        values = evaluate_(problem, swarm.positions, rows)
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm.fitness = np.nan_to_num(values, nan=-np.inf)
        improved = swarm.fitness > swarm.pbest_fitness
        swarm.pbest_fitness[improved] = swarm.fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        best_id = int(np.argmax(swarm.pbest_fitness))
        best_value = float(swarm.pbest_fitness[best_id])
        if best_value > float(swarm.best_fitness):
            swarm.best_fitness = best_value
            swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _apply_finder_exclusion(self, problem) -> None:
        for swarm in self.pop:
            distance = float(np.linalg.norm(self.finder.best_position - swarm.best_position))
            if distance < self.exclusion_limit:
                self.finder = self._generate_subpopulation(problem, self.finder_population_size)
                if problem.state.recent_change or problem.state.terminated:
                    return

    def _append_finder_history(self) -> None:
        self.finder.gbest_position_history = np.vstack(
            [self.finder.gbest_position_history, self.finder.best_position.copy()]
        )
        if self.finder.gbest_position_history.shape[0] > self.finder.k:
            self.finder.gbest_position_history = self.finder.gbest_position_history[1:, :]

    def _create_tracker_from_finder(self) -> None:
        order = np.argsort(self.finder.pbest_fitness)[::-1]
        if order.size < self.population_size:
            order = np.resize(order, self.population_size)
        selected = order[: self.population_size]
        tracker = FTMPSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            shifts=[],
            sleeping=0,
            k=2,
            convergence_limit=1.0,
            gbest_position_history=np.full((2, self.dimension), np.inf),
            velocities=self.finder.velocities[selected, :].copy(),
            positions=self.finder.pbest_positions[selected, :].copy(),
            fitness=self.finder.fitness[selected].copy(),
            pbest_positions=self.finder.pbest_positions[selected, :].copy(),
            pbest_fitness=self.finder.fitness[selected].copy(),
            best_fitness=float(self.finder.best_fitness),
            best_position=self.finder.best_position.copy(),
        )
        self.pop.append(tracker)
        self.swarm_number += 1

    def _apply_tracker_exclusion(self) -> None:
        if self.swarm_number < 2:
            return
        for ii in range(self.swarm_number - 1):
            for jj in range(ii + 1, self.swarm_number):
                distance = float(np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position))
                if distance >= self.exclusion_limit:
                    continue
                loser = ii if self.pop[ii].best_fitness < self.pop[jj].best_fitness else jj
                del self.pop[loser]
                self.swarm_number -= 1
                self.free_swarm_id -= 1
                return

    def _update_sleeping(self) -> None:
        if self.swarm_number <= 1:
            return
        limit = float(self.parameters["SleepingLimit"])
        best_index = int(np.argmax([swarm.best_fitness for swarm in self.pop]))
        for swarm in self.pop:
            velocities = np.asarray(swarm.velocities, dtype=float).copy()
            velocities[(velocities > -limit) & (velocities < limit)] = 0.0
            if np.sum(velocities) == 0.0:
                swarm.sleeping = 1
        self.pop[best_index].sleeping = 0

    def _local_search(self, problem) -> None:
        if self.swarm_number <= 0:
            return
        best_index = int(np.argmax([swarm.best_fitness for swarm in self.pop]))
        best_swarm = self.pop[best_index]
        for _ in range(self.try_number):
            position = best_swarm.best_position + (
                2.0 * self.rng.random(self.dimension) - 1.0
            ) * self.cloud
            values = evaluate_(problem, position, 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if values.size and float(values[0]) > float(best_swarm.best_fitness):
                best_swarm.best_fitness = float(values[0])
                best_swarm.best_position = position.copy()
        self.cloud *= float(self.parameters["CFmin"]) + self.rng.random() * (
            1.0 - float(self.parameters["CFmin"])
        )

    def _reset_best(self, swarm: FTMPSOSwarm) -> None:
        best_id = int(np.argmax(swarm.pbest_fitness))
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _threshold_for_swarm_count(self, swarm_count: int) -> float:
        return float(
            0.5
            * (
                (self.max_coordinate - self.min_coordinate)
                / (max(1, int(swarm_count)) ** (1.0 / self.dimension))
            )
        )
