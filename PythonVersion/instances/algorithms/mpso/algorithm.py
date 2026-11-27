"""mPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.mpso.state import MPSOSwarm


@algorithm_registry.register("MPSO")
class MPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "MPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                5,
                "integer",
                (1, 100),
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
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.population_size = int(self.parameters["PopulationSize"])
        if self.population_size < 1:
            raise ValueError("MPSO requires PopulationSize >= 1.")
        self.free_swarm_id = 0
        self.exclusion_limit = self._threshold_for_swarm_count(10)
        self.convergence_limit = self.exclusion_limit
        self.shift_severity = 1.0
        self.swarm_number = 1
        self.pop: list[MPSOSwarm] = [self._generate_subpopulation(problem)]
        self.pso_operator = PSOOperator()

    def iterate(self, problem) -> None:
        previous_swarm_number = self.swarm_number

        for swarm in list(self.pop):
            self._move_swarm(swarm, problem)
            if problem.state.recent_change:
                return

        self._apply_exclusion(problem)
        if problem.state.recent_change:
            return

        if self.pop and self._free_swarm_has_converged():
            self.swarm_number += 1
            self.free_swarm_id = self.swarm_number - 1
            self.pop.append(self._generate_subpopulation(problem))
            if problem.state.recent_change:
                return

        if previous_swarm_number != self.swarm_number:
            self.exclusion_limit = self._threshold_for_swarm_count(self.swarm_number)
            self.convergence_limit = self.exclusion_limit

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

        for index, swarm in enumerate(self.pop):
            if index == self.free_swarm_id:
                continue
            swarm.positions = (
                np.tile(swarm.best_position, (self.population_size, 1))
                + (2.0 * self.rng.random((self.population_size, self.dimension)) - 1.0)
                * self.shift_severity
            )
            swarm.positions[0, :] = swarm.best_position

        for swarm in self.pop:
            values = evaluate_(problem, swarm.positions, self.population_size)
            if problem.state.recent_change:
                return
            swarm.fitness = values
            swarm.pbest_fitness = values.copy()
            swarm.pbest_positions = swarm.positions.copy()
            swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)

    def _generate_subpopulation(self, problem) -> MPSOSwarm:
        positions = self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((self.population_size, self.dimension))
        )
        fitness = evaluate_(problem, positions, self.population_size)
        pbest_positions = positions.copy()
        if problem.state.recent_change:
            fitness = np.full(self.population_size, -np.inf)
            pbest_fitness = fitness.copy()
        else:
            pbest_fitness = fitness.copy()
        best_id = int(np.argmax(pbest_fitness))
        return MPSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            velocities=np.zeros((self.population_size, self.dimension)),
            shifts=[],
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
        )

    def _move_swarm(self, swarm: MPSOSwarm, problem) -> None:
        constriction = float(self.parameters["x"])
        self.pso_operator.update_and_move(
            swarm,
            inertia=constriction,
            cognitive=constriction * float(self.parameters["c1"]),
            social=constriction * float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(swarm, self.min_coordinate, self.max_coordinate)
        values = evaluate_(problem, swarm.positions, self.population_size)
        if problem.state.recent_change:
            return
        swarm.fitness = values
        improved = swarm.fitness > swarm.pbest_fitness
        swarm.pbest_fitness[improved] = swarm.fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        best_id = int(np.argmax(swarm.pbest_fitness))
        best_value = float(swarm.pbest_fitness[best_id])
        if best_value > float(swarm.best_fitness):
            swarm.best_fitness = best_value
            swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _apply_exclusion(self, problem) -> None:
        exclusion_applied = False
        ii = 0
        while ii < self.swarm_number and not exclusion_applied:
            jj = 0
            while jj < self.swarm_number and not exclusion_applied:
                if ii == jj:
                    jj += 1
                    continue
                distance = float(
                    np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position)
                )
                if distance < self.exclusion_limit:
                    loser = ii if self.pop[ii].best_fitness < self.pop[jj].best_fitness else jj
                    if loser == self.free_swarm_id:
                        self.pop[loser] = self._generate_subpopulation(problem)
                        if problem.state.recent_change:
                            return
                    else:
                        self._delete_swarm(loser)
                    exclusion_applied = True
                    break
                jj += 1
            ii += 1

    def _free_swarm_has_converged(self) -> bool:
        if not self.pop or self.free_swarm_id >= self.swarm_number:
            return False
        positions = self.pop[self.free_swarm_id].positions
        if positions.shape[0] <= 1:
            return True
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        return bool(np.all(distances <= self.convergence_limit))

    def _delete_swarm(self, index: int) -> None:
        del self.pop[index]
        self.swarm_number -= 1
        if index < self.free_swarm_id:
            self.free_swarm_id -= 1

    def _reset_best(self, swarm: MPSOSwarm) -> None:
        best_id = int(np.argmax(swarm.pbest_fitness))
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _threshold_for_swarm_count(self, swarm_count: int) -> float:
        return MPSOHelper.threshold_for_swarm_count(
            self.min_coordinate,
            self.max_coordinate,
            self.dimension,
            swarm_count,
        )


class MPSOHelper:
    """Stateless helper methods used by the MPSO flow."""

    @staticmethod
    def threshold_for_swarm_count(
        min_coordinate: float,
        max_coordinate: float,
        dimension: int,
        swarm_count: int,
    ) -> float:
        """Return the MPSO exclusion/convergence threshold."""

        return float(
            0.5
            * (
                (max_coordinate - min_coordinate)
                / (max(1, swarm_count) ** (1.0 / dimension))
            )
        )
