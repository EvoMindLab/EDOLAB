"""mjDE algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.mjde.state import MjDESubPopulation


@algorithm_registry.register("mjDE")
class MjDEAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "mjDE"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                10,
                "integer",
                (1, 100),
                "Number of individuals per subpopulation.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.bounds = problem.bounds
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.min_coordinate_scalar = float(np.ravel(self.min_coordinate)[0])
        self.max_coordinate_scalar = float(np.ravel(self.max_coordinate)[0])

        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.diversity_plus = 1
        self.shift_severity = 1.0
        self.swarm_number = 1
        self.free_swarm_id = 0
        self.exclusion_limit = self._threshold(self.swarm_number)
        self.convergence_limit = self.exclusion_limit

        self.pop: list[MjDESubPopulation] = []
        swarm = self._generate_subpopulation(problem)
        if swarm is not None:
            self.pop.append(swarm)
        self.swarm_number = len(self.pop)
        if self.swarm_number == 0:
            self.free_swarm_id = -1

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        previous_swarm_number = self.swarm_number

        for index in range(self.swarm_number):
            swarm = self.pop[index]
            old_f = swarm.f.copy()
            old_cr = swarm.cr.copy()

            f_mask = self.rng.random(self.population_size) < swarm.r1
            cr_mask = self.rng.random(self.population_size) < swarm.r2
            if np.any(f_mask):
                swarm.f[f_mask] = (
                    self.min_coordinate_scalar
                    + self.rng.random(int(np.sum(f_mask))) * self.max_coordinate_scalar
                )
            if np.any(cr_mask):
                swarm.cr[cr_mask] = self.rng.random(int(np.sum(cr_mask)))

            r1, r2, r3 = self._rand_1_indices(self.population_size)
            swarm.donor = (
                swarm.positions[r1, :]
                + swarm.f[:, None]
                * (swarm.positions[r2, :] - swarm.positions[r3, :])
            )

            swarm.offspring_positions = swarm.positions.copy()
            forced_dimensions = self.rng.integers(
                0,
                self.dimension,
                size=self.population_size,
            )
            rows = np.arange(self.population_size)
            swarm.offspring_positions[rows, forced_dimensions] = swarm.donor[
                rows,
                forced_dimensions,
            ]
            crossover = self.rng.random((self.population_size, self.dimension)) < swarm.cr[
                :,
                None,
            ]
            swarm.offspring_positions[crossover] = swarm.donor[crossover]
            self._midpoint_boundary_check(swarm)

            values = evaluate_(problem, swarm.offspring_positions, self.population_size)
            if problem.state.recent_change or problem.state.terminated:
                return
            swarm.offspring_fitness = values

            better = swarm.offspring_fitness > swarm.fitness
            worse = ~better
            swarm.positions[better, :] = swarm.offspring_positions[better, :]
            swarm.fitness[better] = swarm.offspring_fitness[better]
            swarm.f[worse] = old_f[worse]
            swarm.cr[worse] = old_cr[worse]
            self._refresh_best(swarm)

        self._apply_exclusion(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        self._check_free_swarm_convergence(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        if previous_swarm_number != self.swarm_number:
            self._update_thresholds()

    def on_environment_change(self, problem) -> None:
        shift_samples: list[float] = []
        for index, swarm in enumerate(self.pop):
            if index == self.free_swarm_id:
                continue
            if not np.any(np.isnan(swarm.gbest_past_environment)):
                shift = float(
                    np.linalg.norm(swarm.gbest_past_environment - swarm.best_position)
                )
                swarm.shifts.append(shift)
            shift_samples.extend(swarm.shifts)
        if shift_samples:
            self.shift_severity = float(np.mean(shift_samples))

        for index, swarm in enumerate(self.pop):
            if index == self.free_swarm_id:
                continue
            swarm.positions = swarm.best_position + (
                (2.0 * self.rng.random((self.population_size, self.dimension)) - 1.0)
                * self.shift_severity
            )
            swarm.positions[0, :] = swarm.best_position.copy()

        for swarm in self.pop:
            swarm.fitness = evaluate_(problem, swarm.positions, self.population_size)
            swarm.gbest_past_environment = swarm.best_position.copy()
            if problem.state.recent_change or problem.state.terminated:
                return

    def _generate_subpopulation(self, problem) -> MjDESubPopulation | None:
        rows = max(1, int(self.population_size))
        positions = self.bounds.sample(rows, self.rng)
        fitness = evaluate_(problem, positions, rows)
        if problem.state.recent_change:
            fitness = np.full(rows, -np.inf, dtype=float)
            best_value = float("-inf")
            best_id = 0
        else:
            best_value, best_id = self._best_value_id(fitness)

        return MjDESubPopulation(
            positions=positions,
            fitness=fitness,
            gbest_past_environment=np.full(self.dimension, np.nan),
            fgbest_past_environment=np.full(1, np.nan),
            offspring_positions=np.full((rows, self.dimension), np.nan),
            offspring_fitness=np.full(rows, np.nan),
            donor=np.full((rows, self.dimension), np.nan),
            cr=np.full(rows, 0.9, dtype=float),
            f=np.full(rows, 0.5, dtype=float),
            f_lb=0.1,
            f_ub=0.9,
            r1=0.1,
            r2=0.1,
            shifts=[],
            best_id=best_id,
            best_position=positions[best_id, :].copy(),
            best_fitness=best_value,
        )

    def _rand_1_indices(self, rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r1 = np.empty(rows, dtype=int)
        r2 = np.empty(rows, dtype=int)
        r3 = np.empty(rows, dtype=int)
        for row in range(rows):
            r1[row] = self._random_index(rows, {row})
            r2[row] = self._random_index(rows, {row, int(r1[row])})
            r3[row] = self._random_index(rows, {row, int(r1[row]), int(r2[row])})
        return r1, r2, r3

    def _random_index(self, rows: int, excluded: set[int]) -> int:
        if rows <= 1:
            return 0
        candidates = np.asarray(
            [index for index in range(rows) if index not in excluded],
            dtype=int,
        )
        if candidates.size == 0:
            candidates = np.arange(rows, dtype=int)
        return int(candidates[int(self.rng.integers(0, candidates.size))])

    def _midpoint_boundary_check(self, swarm: MjDESubPopulation) -> None:
        lower = np.broadcast_to(self.min_coordinate, swarm.offspring_positions.shape)
        upper = np.broadcast_to(self.max_coordinate, swarm.offspring_positions.shape)
        lower_mask = swarm.offspring_positions < lower
        if np.any(lower_mask):
            replacement = (lower + swarm.positions) / 2.0
            swarm.offspring_positions[lower_mask] = replacement[lower_mask]
        upper_mask = swarm.offspring_positions > upper
        if np.any(upper_mask):
            replacement = (upper + swarm.positions) / 2.0
            swarm.offspring_positions[upper_mask] = replacement[upper_mask]

    def _apply_exclusion(self, problem) -> None:
        if self.swarm_number <= 1:
            return
        remove = np.zeros(self.swarm_number, dtype=bool)
        for left in range(self.swarm_number - 1):
            for right in range(left + 1, self.swarm_number):
                distance = float(
                    np.linalg.norm(
                        self.pop[left].best_position - self.pop[right].best_position
                    )
                )
                if distance >= self.exclusion_limit:
                    continue
                loser = (
                    left
                    if self.pop[left].best_fitness < self.pop[right].best_fitness
                    else right
                )
                if loser == self.free_swarm_id:
                    replacement = self._generate_subpopulation(problem)
                    if replacement is not None:
                        self.pop[loser] = replacement
                    if problem.state.recent_change or problem.state.terminated:
                        return
                else:
                    remove[loser] = True

        for index in range(self.swarm_number - 1, -1, -1):
            if remove[index]:
                del self.pop[index]
                self.swarm_number -= 1
                self.free_swarm_id -= 1
        if self.swarm_number > 0:
            self.free_swarm_id = min(max(self.free_swarm_id, 0), self.swarm_number - 1)
        else:
            self.free_swarm_id = -1

    def _check_free_swarm_convergence(self, problem) -> None:
        if self.free_swarm_id < 0 or not self.pop:
            return
        free_swarm = self.pop[self.free_swarm_id]
        distances = np.linalg.norm(
            free_swarm.positions[:, None, :] - free_swarm.positions[None, :, :],
            axis=2,
        )
        if np.sum(distances > self.convergence_limit) != 0:
            return
        self.swarm_number += 1
        self.free_swarm_id = self.swarm_number - 1
        swarm = self._generate_subpopulation(problem)
        if swarm is not None:
            self.pop.append(swarm)
        else:
            self.swarm_number -= 1
            self.free_swarm_id = self.swarm_number - 1

    def _refresh_best(self, swarm: MjDESubPopulation) -> None:
        best_value, best_id = self._best_value_id(swarm.fitness)
        swarm.best_fitness = best_value
        swarm.best_id = best_id
        swarm.best_position = swarm.positions[best_id, :].copy()

    def _best_value_id(self, values: np.ndarray) -> tuple[float, int]:
        if values.size == 0:
            return float("-inf"), 0
        comparable = np.nan_to_num(values, nan=-np.inf)
        best_id = int(np.argmax(comparable))
        if np.isneginf(comparable[best_id]) and np.isnan(values[best_id]):
            return float("-inf"), best_id
        return float(values[best_id]), best_id

    def _threshold(self, swarm_number: int) -> float:
        swarm_number = max(1, int(swarm_number))
        span = self.max_coordinate_scalar - self.min_coordinate_scalar
        return float(0.5 * (span / (swarm_number ** (1.0 / self.dimension))))

    def _update_thresholds(self) -> None:
        self.exclusion_limit = self._threshold(self.swarm_number)
        self.convergence_limit = self.exclusion_limit
