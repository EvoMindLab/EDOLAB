"""DynPopDE algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.dynpopde.state import DynPopDESubPopulation


@algorithm_registry.register("DynPopDE")
class DynPopDEAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "DynPopDE"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "initPopulationSize": ParameterSpec(
                "initPopulationSize",
                100,
                "integer",
                (0, 500),
                "Total number of individuals at initialization.",
            ),
            "CR": ParameterSpec(
                "CR",
                0.6,
                "numeric",
                (0, 2),
                "Crossover probability for generating trial vectors in DE.",
            ),
            "F": ParameterSpec(
                "F",
                0.5,
                "numeric",
                (0, 2),
                "Mutation scaling factor in donor vector computation.",
            ),
            "IndivSize": ParameterSpec(
                "IndivSize",
                10,
                "integer",
                (0, 100),
                "Number of individuals maintained in each sub-population.",
            ),
            "BrownNumber": ParameterSpec(
                "BrownNumber",
                5,
                "integer",
                (0, 100),
                "Count of particles receiving Brownian perturbation.",
            ),
            "ConvergenceLimit": ParameterSpec(
                "ConvergenceLimit",
                0.01,
                "numeric",
                (0, 0.1),
                "Average radius below which a sub-population is considered converged.",
            ),
            "RandomFlag": ParameterSpec(
                "RandomFlag",
                "Fixed CR/F",
                "option",
                ("Random CR/F", "Fixed CR/F"),
                "Method for selecting CR and F parameters per individual.",
            ),
            "StrategyFlag": ParameterSpec(
                "StrategyFlag",
                "DE/BEST/2",
                "option",
                (
                    "Random Strategy",
                    "DE/RAND/1",
                    "DE/RAND/2",
                    "DE/BEST/1",
                    "DE/BEST/2",
                    "DE/RAND-TO-BEST/1",
                    "DE/CURRENT-TO-RAND/1",
                    "DE/CURRENT-TO-BEST/1",
                ),
                "Mutation scheme for creating donor vectors in DE.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.bounds = problem.bounds
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)

        self.init_population_size = int(self.parameters["initPopulationSize"])
        self.indiv_size = max(1, int(self.parameters["IndivSize"]))
        self.brown_number = max(0, int(self.parameters["BrownNumber"]))
        self.convergence_limit = float(self.parameters["ConvergenceLimit"])
        self.swarm_number = 1
        self.exclusion_limit = self._exclusion_limit(self.swarm_number)
        self.worst_pop_id = -1
        self.evolve_id = -1
        self.diversity_flag = 1
        self.dynpop_iteration = 0
        self.random_flag = self._map_option(
            self.parameters["RandomFlag"],
            {"Random CR/F": 1, "Fixed CR/F": 2},
            default=2,
        )
        self.strategy_flag = self._map_option(
            self.parameters["StrategyFlag"],
            {
                "Random Strategy": 0,
                "DE/RAND/1": 1,
                "DE/RAND/2": 2,
                "DE/BEST/1": 3,
                "DE/BEST/2": 4,
                "DE/RAND-TO-BEST/1": 5,
                "DE/CURRENT-TO-RAND/1": 6,
                "DE/CURRENT-TO-BEST/1": 7,
            },
            default=4,
        )

        self.pop: list[DynPopDESubPopulation] = []
        for _ in range(self.swarm_number):
            swarm = self._generate_subpopulation(problem)
            if swarm is not None:
                self.pop.append(swarm)
            if problem.state.recent_change or problem.state.terminated:
                self.swarm_number = len(self.pop)
                self._reset_tracking_arrays()
                return
        self.swarm_number = len(self.pop)
        self._reset_tracking_arrays()

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        self.dynpop_iteration += 1
        if self.dynpop_iteration <= 2:
            self._initial_iterations(problem)
        else:
            self._adaptive_iteration(problem)

    def on_environment_change(self, problem) -> None:
        for swarm in self.pop:
            values = evaluate_(problem, swarm.positions, swarm.positions.shape[0])
            if values.size:
                swarm.gbest_past_environment = swarm.best_position.copy()
                count = min(values.size, swarm.positions.shape[0])
                valid = ~np.isnan(values[:count])
                if np.any(valid):
                    swarm.fitness[:count][valid] = values[:count][valid]
                    self._refresh_best(swarm)
            if problem.state.recent_change or problem.state.terminated:
                return
            if values.size != swarm.positions.shape[0]:
                return
        self.dynpop_iteration = 0
        self.evolve_id = 0

    def _initial_iterations(self, problem) -> None:
        for index in range(self.swarm_number):
            self._move_swarm(problem, index)
            if problem.state.recent_change or problem.state.terminated:
                return
            self._refresh_geometry(self.pop[index])
            self._brownian_best(problem, self.pop[index], tries=3, sigma=0.4)
            if problem.state.recent_change or problem.state.terminated:
                return
        self._check_initial_overlapping()
        self._reinitialize_marked(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        for index, swarm in enumerate(self.pop):
            self.pre_fitness[index] = self.cur_fitness[index]
            self.cur_fitness[index] = swarm.best_fitness

    def _adaptive_iteration(self, problem) -> None:
        evolved_id = (
            np.arange(self.swarm_number, dtype=int)
            if self.dynpop_iteration == 3
            else np.asarray([self.evolve_id], dtype=int)
        )
        if np.sum(np.abs(self.cur_fitness - self.pre_fitness)) == 0:
            swarm = self._generate_subpopulation(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            if swarm is not None:
                self.pop.append(swarm)
                self.swarm_number += 1
                self.cur_fitness = np.append(self.cur_fitness, swarm.best_fitness)
                self.pre_fitness = np.append(self.pre_fitness, 0.0)
                self.performance = np.append(self.performance, 0.0)

        self.worst_pop_id = self._find_worst_pop()
        self.performance = self._update_performance(evolved_id)
        self.evolve_id = int(np.argmax(self.performance))

        self._move_swarm(problem, self.evolve_id)
        if problem.state.recent_change or problem.state.terminated:
            return
        self._refresh_geometry(self.pop[self.evolve_id])

        self.exclusion_limit = self._exclusion_limit(self.swarm_number)
        self._check_adaptive_overlapping(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        self._reinitialize_marked(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        self._delete_marked()
        self.exclusion_limit = self._exclusion_limit(self.swarm_number)

        self._brownian_worst_individuals(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        for index, swarm in enumerate(self.pop):
            self.pre_fitness[index] = self.cur_fitness[index]
            self.cur_fitness[index] = swarm.best_fitness
            if abs(self.cur_fitness[index] - self.pre_fitness[index]) > 0:
                swarm.pen = 0
            else:
                swarm.pen += 1

    def _generate_subpopulation(self, problem) -> DynPopDESubPopulation | None:
        rows = max(1, int(self.indiv_size))
        positions = self.bounds.sample(rows, self.rng)
        if self.strategy_flag == 0:
            strategies = self.rng.integers(1, 8, size=rows)
        else:
            strategies = np.full(rows, self.strategy_flag, dtype=int)
        if self.random_flag == 2:
            crossover_rates = np.full(rows, 0.6, dtype=float)
            scale_factors = np.full(rows, 0.5, dtype=float)
        else:
            crossover_rates = self.rng.random(rows)
            scale_factors = self.rng.random(rows)

        individual_types = np.ones(rows, dtype=int)
        fitness = evaluate_(problem, positions, rows)
        if problem.state.recent_change or problem.state.terminated:
            return None
        if fitness.size != rows:
            return None

        swarm = DynPopDESubPopulation(
            positions=positions,
            fitness=fitness,
            donor_positions=np.zeros_like(positions),
            trial_positions=np.zeros_like(positions),
            crossover_rates=crossover_rates,
            scale_factors=scale_factors,
            strategies=strategies,
            individual_types=individual_types,
            gbest_past_environment=np.full(self.dimension, np.nan),
            best_id=0,
            best_position=positions[0, :].copy(),
            best_fitness=float("-inf"),
            center=np.zeros(self.dimension, dtype=float),
            current_radius=0.0,
            reinit_state=0,
            delete_state=0,
            pen=0,
        )
        self._refresh_best(swarm)
        self._source_brownian_initialization(problem, swarm)
        if problem.state.recent_change or problem.state.terminated:
            return None
        if problem.state.recent_change:
            swarm.fitness = np.full(rows, -np.inf, dtype=float)
        self._refresh_best(swarm)
        self._refresh_geometry(swarm)
        swarm.reinit_state = 0
        swarm.delete_state = 0
        swarm.pen = 0
        return swarm

    def _source_brownian_initialization(
        self,
        problem,
        swarm: DynPopDESubPopulation,
    ) -> None:
        if self.diversity_flag != 1:
            return
        count = min(5, swarm.positions.shape[0])
        if count <= 0:
            return
        order = np.argsort(np.nan_to_num(swarm.fitness, nan=-np.inf))[:count]
        for row in order:
            self._refresh_best(swarm)
            previous_best = swarm.best_fitness
            candidate = swarm.positions[swarm.best_id, :].copy()
            candidate += 0.2 * self.rng.standard_normal(self.dimension)
            candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
            swarm.positions[row, :] = candidate
            values = evaluate_(problem, candidate, 1)
            if values.size and not np.isnan(values[0]):
                swarm.fitness[swarm.best_id] = float(values[0])
                best_value, best_id = self._best_value_id(swarm.fitness)
                if best_value > previous_best:
                    swarm.best_fitness = best_value
                    swarm.best_id = best_id
                    swarm.best_position = swarm.positions[best_id, :].copy()
            if problem.state.recent_change or problem.state.terminated:
                return

    def _move_swarm(self, problem, index: int) -> None:
        swarm = self.pop[index]
        swarm.donor_positions = self._mutation(swarm)
        swarm.trial_positions = np.zeros_like(swarm.positions)
        for row in range(swarm.positions.shape[0]):
            if swarm.individual_types[row] == 2:
                continue
            for dim in range(self.dimension):
                randj = int(self.rng.integers(0, self.dimension))
                if self.rng.random() < swarm.crossover_rates[row] or dim == randj:
                    swarm.trial_positions[row, dim] = swarm.donor_positions[row, dim]
                else:
                    swarm.trial_positions[row, dim] = swarm.positions[row, dim]
                swarm.trial_positions[row, dim] = np.clip(
                    swarm.trial_positions[row, dim],
                    self.min_coordinate[dim],
                    self.max_coordinate[dim],
                )
            values = evaluate_(problem, swarm.trial_positions[row, :], 1)
            if values.size and not np.isnan(values[0]):
                value = float(values[0])
                if value > swarm.fitness[row]:
                    swarm.positions[row, :] = swarm.trial_positions[row, :]
                    swarm.fitness[row] = value
            if problem.state.recent_change or problem.state.terminated:
                return
        best_value, best_id = self._best_value_id(swarm.fitness)
        if best_value > swarm.best_fitness:
            swarm.best_fitness = best_value
            swarm.best_id = best_id
            swarm.best_position = swarm.positions[best_id, :].copy()

    def _mutation(self, swarm: DynPopDESubPopulation) -> np.ndarray:
        donors = np.zeros_like(swarm.positions)
        for row in range(swarm.positions.shape[0]):
            if swarm.individual_types[row] == 2:
                continue
            donors[row, :] = self._mutation_one(
                swarm,
                row,
                int(swarm.strategies[row]),
                float(swarm.scale_factors[row]),
            )
        return donors

    def _mutation_one(
        self,
        swarm: DynPopDESubPopulation,
        row: int,
        strategy: int,
        factor: float,
    ) -> np.ndarray:
        positions = swarm.positions
        rows = positions.shape[0]
        indices = np.arange(rows)

        if strategy == 4 and rows >= 5:
            choices = indices[indices != swarm.best_id]
            selected = self.rng.choice(choices, size=4, replace=False)
            return swarm.best_position + factor * (
                positions[selected[0], :]
                + positions[selected[1], :]
                - positions[selected[2], :]
                - positions[selected[3], :]
            )
        if strategy == 3 and rows >= 3:
            choices = indices[indices != swarm.best_id]
            selected = self.rng.choice(choices, size=2, replace=False)
            return swarm.best_position + factor * (
                positions[selected[0], :] - positions[selected[1], :]
            )
        if strategy == 2 and rows >= 5:
            selected = self.rng.choice(indices, size=5, replace=False)
            return positions[selected[0], :] + factor * (
                positions[selected[1], :]
                + positions[selected[2], :]
                - positions[selected[3], :]
                - positions[selected[4], :]
            )
        if strategy == 1 and rows >= 3:
            selected = self.rng.choice(indices, size=3, replace=False)
            return positions[selected[0], :] + factor * (
                positions[selected[1], :] - positions[selected[2], :]
            )
        if strategy == 5 and rows >= 4:
            choices = indices[indices != swarm.best_id]
            selected = self.rng.choice(choices, size=3, replace=False)
            base = positions[selected[0], :]
            return base + factor * (swarm.best_position - base) + factor * (
                positions[selected[1], :] - positions[selected[2], :]
            )
        if strategy == 6 and rows >= 4:
            choices = indices[indices != row]
            selected = self.rng.choice(choices, size=3, replace=False)
            current = positions[row, :]
            return current + factor * (positions[selected[0], :] - current) + factor * (
                positions[selected[1], :] - positions[selected[2], :]
            )
        if strategy == 7 and rows >= 4:
            if row == swarm.best_id:
                choices = indices[indices != row]
            else:
                choices = indices[(indices != row) & (indices != swarm.best_id)]
            if choices.size >= 2:
                selected = self.rng.choice(choices, size=2, replace=False)
                current = positions[row, :]
                return current + factor * (swarm.best_position - current) + factor * (
                    positions[selected[0], :] - positions[selected[1], :]
                )
        return positions[row, :].copy()

    def _brownian_best(
        self,
        problem,
        swarm: DynPopDESubPopulation,
        tries: int,
        sigma: float,
    ) -> None:
        if self.diversity_flag != 1:
            return
        for _ in range(tries):
            candidate = swarm.positions[swarm.best_id, :].copy()
            candidate += sigma * self.rng.standard_normal(self.dimension)
            candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
            values = evaluate_(problem, candidate, 1)
            if values.size and not np.isnan(values[0]):
                value = float(values[0])
                if value > swarm.fitness[swarm.best_id]:
                    swarm.fitness[swarm.best_id] = value
                    swarm.positions[swarm.best_id, :] = candidate
                    swarm.best_position = candidate.copy()
                    swarm.best_fitness = value
            if problem.state.recent_change or problem.state.terminated:
                return

    def _brownian_worst_individuals(self, problem) -> None:
        if self.diversity_flag != 1:
            return
        for swarm in self.pop:
            count = min(self.brown_number, swarm.positions.shape[0])
            if count <= 0:
                continue
            order = np.argsort(np.nan_to_num(swarm.fitness, nan=-np.inf))[:count]
            for row in order:
                candidate = swarm.positions[swarm.best_id, :].copy()
                candidate += 0.2 * self.rng.standard_normal(self.dimension)
                candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
                values = evaluate_(problem, candidate, 1)
                if values.size and not np.isnan(values[0]):
                    value = float(values[0])
                    if value > swarm.fitness[row]:
                        swarm.positions[row, :] = candidate
                        swarm.fitness[row] = value
                        best_value, best_id = self._best_value_id(swarm.fitness)
                        if best_value > swarm.best_fitness:
                            swarm.best_fitness = best_value
                            swarm.best_id = best_id
                            swarm.best_position = swarm.positions[best_id, :].copy()
                if problem.state.recent_change or problem.state.terminated:
                    return

    def _check_initial_overlapping(self) -> None:
        for left in range(self.swarm_number - 1):
            for right in range(left + 1, self.swarm_number):
                left_swarm = self.pop[left]
                right_swarm = self.pop[right]
                distance = float(
                    np.linalg.norm(
                        left_swarm.positions[left_swarm.best_id, :]
                        - right_swarm.positions[right_swarm.best_id, :]
                    )
                )
                if distance < self.exclusion_limit:
                    right_index = min(
                        left_swarm.best_id,
                        right_swarm.fitness.shape[0] - 1,
                    )
                    if (
                        left_swarm.fitness[left_swarm.best_id]
                        > right_swarm.fitness[right_index]
                    ):
                        right_swarm.reinit_state = 1
                    else:
                        left_swarm.reinit_state = 1

    def _check_adaptive_overlapping(self, problem) -> None:
        for left in range(self.swarm_number - 1):
            for right in range(left + 1, self.swarm_number):
                left_swarm = self.pop[left]
                right_swarm = self.pop[right]
                distance = float(
                    np.linalg.norm(left_swarm.best_position - right_swarm.best_position)
                )
                if (
                    left_swarm.reinit_state != 0
                    or right_swarm.reinit_state != 0
                    or distance >= self.exclusion_limit
                ):
                    continue
                midpoint = (left_swarm.best_position + right_swarm.best_position) / 2.0
                values = evaluate_(problem, midpoint, 1)
                if values.size and not np.isnan(values[0]):
                    mid_fitness = float(values[0])
                    if (
                        mid_fitness > left_swarm.best_fitness
                        and mid_fitness > right_swarm.best_fitness
                    ):
                        loser = (
                            right
                            if left_swarm.best_fitness > right_swarm.best_fitness
                            else left
                        )
                        self.pre_fitness[loser] = self.cur_fitness[loser]
                        self.cur_fitness[loser] = self.pop[loser].best_fitness
                        if (self.cur_fitness[loser] - self.pre_fitness[loser]) == 0:
                            self.pop[loser].reinit_state = 1
                        else:
                            self.pop[loser].delete_state = 1
                if problem.state.recent_change or problem.state.terminated:
                    return

    def _reinitialize_marked(self, problem) -> None:
        for index, swarm in enumerate(list(self.pop)):
            if swarm.reinit_state != 1:
                continue
            replacement = self._generate_subpopulation(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            if replacement is not None:
                self.pop[index] = replacement

    def _delete_marked(self) -> None:
        while True:
            deleted_index = -1
            for index in range(self.swarm_number):
                if self.pop[index].delete_state == 1:
                    del self.pop[index]
                    self.pre_fitness = np.delete(self.pre_fitness, index)
                    self.cur_fitness = np.delete(self.cur_fitness, index)
                    self.performance = np.delete(self.performance, index)
                    self.swarm_number -= 1
                    deleted_index = index
                    break
            if deleted_index < 0:
                break
            if deleted_index + 1 == self.swarm_number:
                break
        if self.swarm_number == 0:
            self.evolve_id = -1
            self.worst_pop_id = -1
        else:
            evolve_id = getattr(self, "evolve_id", 0)
            worst_pop_id = getattr(self, "worst_pop_id", 0)
            self.evolve_id = min(max(evolve_id, 0), self.swarm_number - 1)
            self.worst_pop_id = min(max(worst_pop_id, 0), self.swarm_number - 1)

    def _find_worst_pop(self) -> int:
        if not self.pop:
            return -1
        return int(np.argmin([swarm.best_fitness for swarm in self.pop]))

    def _update_performance(self, evolved_id: np.ndarray) -> np.ndarray:
        del evolved_id
        performance = self.performance.copy()
        if self.worst_pop_id < 0:
            return performance
        worst = self.pop[self.worst_pop_id].best_fitness
        for index, swarm in enumerate(self.pop):
            value = (1 + abs(swarm.best_fitness - worst)) * (
                1 + abs(self.cur_fitness[index] - self.pre_fitness[index])
            )
            if swarm.pen > 0:
                value /= swarm.pen
            performance[index] = value
        return performance

    def _refresh_geometry(self, swarm: DynPopDESubPopulation) -> None:
        if swarm.positions.shape[0] == 0:
            swarm.center = np.zeros(self.dimension, dtype=float)
            swarm.current_radius = 0.0
            return
        swarm.center = np.mean(swarm.positions, axis=0)
        swarm.current_radius = float(
            np.mean(np.linalg.norm(swarm.positions - swarm.center, axis=1))
        )

    def _refresh_best(self, swarm: DynPopDESubPopulation) -> None:
        best_value, best_id = self._best_value_id(swarm.fitness)
        swarm.best_fitness = best_value
        swarm.best_id = best_id
        swarm.best_position = swarm.positions[best_id, :].copy()

    def _best_value_id(self, values: np.ndarray) -> tuple[float, int]:
        if values.size == 0:
            return float("-inf"), 0
        comparable = np.nan_to_num(values, nan=-np.inf)
        best_id = int(np.argmax(comparable))
        return float(values[best_id]), best_id

    def _reset_tracking_arrays(self) -> None:
        self.pre_fitness = np.zeros(self.swarm_number, dtype=float)
        self.cur_fitness = np.zeros(self.swarm_number, dtype=float)
        self.performance = np.zeros(self.swarm_number, dtype=float)

    def _exclusion_limit(self, swarm_number: int) -> float:
        count = max(1, int(swarm_number))
        width = float(self.max_coordinate[0] - self.min_coordinate[0])
        return width / (2 * count ** (1 / self.dimension))

    def _map_option(self, value: object, mapping: dict[str, int], default: int) -> int:
        if isinstance(value, str):
            return mapping.get(value, default)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        return numeric if numeric in mapping.values() else default
