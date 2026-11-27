"""mDE algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import DEOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.mde.state import MDESubPopulation


@algorithm_registry.register("mDE")
class MDEAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "mDE"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                5,
                "integer",
                (5, 100),
                "Number of individuals per subpopulation; DE/best/2 requires at least 5.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.population_size = int(self.parameters["PopulationSize"])
        if self.population_size < 5:
            raise ValueError("mDE requires PopulationSize >= 5 for DE/best/2 mutation.")

        self.diversity_plus = 1
        self.shift_severity = 1.0
        self.swarm_number = 1
        self.free_swarm_id = 0
        self.exclusion_limit = self._threshold_for_swarm_count(self.swarm_number)
        self.convergence_limit = self.exclusion_limit
        self.de_operator = DEOperator()
        self.pop: list[MDESubPopulation] = []
        for _ in range(self.swarm_number):
            self.pop.append(self._generate_subpopulation(problem))

    def iterate(self, problem) -> None:
        previous_swarm_number = self.swarm_number

        for swarm in list(self.pop[: self.swarm_number]):
            self._reset_best(swarm)
            swarm.scale_factors = self.rng.random(self.population_size)
            swarm.crossover_rates = self.rng.random(self.population_size)
            swarm.donor_positions = self.de_operator.mutation_best_2(
                swarm.positions,
                swarm.best_position,
                swarm.scale_factors,
                self.rng,
            )
            swarm.offspring_positions = self.de_operator.binomial_crossover_population(
                swarm.positions,
                swarm.donor_positions,
                swarm.crossover_rates,
                self.rng,
            )
            swarm.offspring_positions = self.de_operator.midpoint_repair_bounds(
                swarm.offspring_positions,
                swarm.positions,
                self.min_coordinate,
                self.max_coordinate,
            )

            values = self._evaluate(problem, swarm.offspring_positions, self.population_size)
            if problem.state.recent_change:
                return
            swarm.offspring_fitness = values
            swarm.positions, swarm.fitness, _ = self.de_operator.greedy_select(
                swarm.positions,
                swarm.fitness,
                swarm.offspring_positions,
                swarm.offspring_fitness,
            )
            self._reset_best(swarm)

        if self._apply_exclusion(problem):
            return
        if self._free_swarm_has_converged():
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
        for jj, swarm in enumerate(self.pop):
            if jj == self.free_swarm_id:
                continue
            if not np.isnan(swarm.gbest_past_environment).any():
                shift = float(np.linalg.norm(swarm.gbest_past_environment - swarm.best_position))
                swarm.shifts.append(shift)
            shift_samples.extend(swarm.shifts)
        if shift_samples:
            self.shift_severity = float(np.mean(shift_samples))

        for jj, swarm in enumerate(self.pop):
            if jj == self.free_swarm_id:
                continue
            offsets = (
                2 * self.rng.random((self.population_size, self.dimension)) - 1
            ) * self.shift_severity
            swarm.positions = np.tile(swarm.best_position, (self.population_size, 1)) + offsets
            swarm.positions[0, :] = swarm.best_position

        for swarm in self.pop:
            previous_best = np.asarray(swarm.best_position, dtype=float).copy()
            values = self._evaluate(problem, swarm.positions, self.population_size)
            if problem.state.recent_change:
                return
            swarm.fitness = values
            swarm.gbest_past_environment = previous_best
            self._reset_best(swarm)

    def _generate_subpopulation(self, problem) -> MDESubPopulation:
        positions = self.bounds.sample(self.population_size, self.rng)
        fitness = self._evaluate(problem, positions, self.population_size)
        if problem.state.recent_change:
            fitness = np.full(self.population_size, -np.inf)
        best_id = int(np.argmax(np.nan_to_num(fitness, nan=-np.inf)))
        return MDESubPopulation(
            positions=positions,
            fitness=fitness,
            gbest_past_environment=np.full(self.dimension, np.nan),
            offspring_positions=np.full((self.population_size, self.dimension), np.nan),
            offspring_fitness=np.full(self.population_size, np.nan),
            donor_positions=np.full((self.population_size, self.dimension), np.nan),
            crossover_rates=np.full(self.population_size, 0.9),
            scale_factors=np.full(self.population_size, 0.5),
            scale_factor_lower_bound=0.1,
            scale_factor_upper_bound=0.9,
            r1=0.1,
            r2=0.1,
            shifts=[],
            best_id=best_id,
            best_position=positions[best_id, :].copy(),
            best_fitness=float(fitness[best_id]),
        )

    def _evaluate(self, problem, positions: np.ndarray, expected: int) -> np.ndarray:
        samples = np.asarray(positions, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        values = problem.evaluate(samples)
        if values.size == expected:
            return values
        output = np.full(expected, np.nan)
        output[: values.size] = values
        return output

    def _reset_best(self, swarm: MDESubPopulation) -> None:
        fitness = np.asarray(swarm.fitness, dtype=float)
        best_id = int(np.argmax(np.nan_to_num(fitness, nan=-np.inf)))
        swarm.best_id = best_id
        swarm.best_position = swarm.positions[best_id, :].copy()
        swarm.best_fitness = float(fitness[best_id])

    def _apply_exclusion(self, problem) -> bool:
        if self.swarm_number <= 1:
            return False

        remove = np.zeros(self.swarm_number, dtype=bool)
        for ii in range(self.swarm_number - 1):
            for jj in range(ii + 1, self.swarm_number):
                distance = float(np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position))
                if distance >= self.exclusion_limit:
                    continue
                if self.pop[ii].best_fitness < self.pop[jj].best_fitness:
                    weaker = ii
                else:
                    weaker = jj
                if weaker == self.free_swarm_id:
                    self.pop[weaker] = self._generate_subpopulation(problem)
                    if problem.state.recent_change:
                        return True
                else:
                    remove[weaker] = True

        for index in range(self.swarm_number - 1, -1, -1):
            if not remove[index]:
                continue
            del self.pop[index]
            self.swarm_number -= 1
            if index < self.free_swarm_id:
                self.free_swarm_id -= 1
        return False

    def _free_swarm_has_converged(self) -> bool:
        if self.free_swarm_id < 0 or self.free_swarm_id >= len(self.pop):
            return False
        positions = self.pop[self.free_swarm_id].positions
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        return not bool(np.any(distances > self.convergence_limit))

    def _threshold_for_swarm_count(self, swarm_count: int) -> float:
        return MDEHelper.threshold_for_swarm_count(
            self.min_coordinate,
            self.max_coordinate,
            self.dimension,
            swarm_count,
        )


class MDEHelper:
    """Stateless helper methods used by the mDE flow."""

    @staticmethod
    def threshold_for_swarm_count(
        min_coordinate: np.ndarray,
        max_coordinate: np.ndarray,
        dimension: int,
        swarm_count: int,
    ) -> float:
        """Return the mDE exclusion/convergence threshold."""

        span = float(np.mean(max_coordinate - min_coordinate))
        return 0.5 * (span / (max(1, swarm_count) ** (1.0 / dimension)))
