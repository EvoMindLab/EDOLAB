"""CDE algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import DEOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.cde.state import CDESubPopulation
from instances.algorithms.common import evaluate_


@algorithm_registry.register("CDE")
class CDEAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "CDE"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "initPopulationSize": ParameterSpec(
                "initPopulationSize",
                100,
                "integer",
                (0, 1000),
                "Total number of individuals at initialization.",
            ),
            "CR": ParameterSpec(
                "CR",
                0.6,
                "numeric",
                (0, 2),
                "Probability of inheriting donor vector components during crossover.",
            ),
            "F": ParameterSpec(
                "F",
                0.5,
                "numeric",
                (0, 2),
                "Scaling factor controlling mutation step size in differential evolution.",
            ),
            "SwarmNumber": ParameterSpec(
                "SwarmNumber",
                10,
                "integer",
                (0, 100),
                "Number of independent sub-swarms evolving in parallel.",
            ),
            "BrownNumber": ParameterSpec(
                "BrownNumber",
                5,
                "integer",
                (0, 100),
                "Number of worst-performing particles undergoing Brownian motion for diversity.",
            ),
            "IndivSize": ParameterSpec(
                "IndivSize",
                10,
                "integer",
                (0, 100),
                "Number of individuals maintained in each sub-population.",
            ),
            "ConvergenceLimit": ParameterSpec(
                "ConvergenceLimit",
                0.01,
                "numeric",
                (0, 0.1),
                "Radius threshold to declare sub-swarm convergence.",
            ),
            "OverlapDegree": ParameterSpec(
                "OverlapDegree",
                0.1,
                "numeric",
                (0, 1),
                "Minimum distance between sub-swarms before overlap reinitialization.",
            ),
            "RandomFlag": ParameterSpec(
                "RandomFlag",
                "Fixed CR/F",
                "option",
                ("Random CR/F", "Fixed CR/F"),
                "Parameter generation method for crossover rate and scaling factor.",
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
                "Mutation strategy for generating donor vectors in differential evolution.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.de_operator = DEOperator()

        self.init_population_size = max(0, int(self.parameters["initPopulationSize"]))
        self.swarm_number = max(1, int(self.parameters["SwarmNumber"]))
        self.indiv_size = max(5, int(self.parameters["IndivSize"]))
        self.brown_number = max(0, int(self.parameters["BrownNumber"]))
        self.diversity_flag = 1
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

        peak_number = max(1, int(getattr(problem, "parameters", {}).get("PeakNumber", 1)))
        span = np.asarray(self.max_coordinate - self.min_coordinate, dtype=float)
        self.exclusion_limit = span / (2.0 * (peak_number ** (1.0 / self.dimension)))
        self.convergence_limit = float(self.parameters["ConvergenceLimit"])
        self.overlap_degree = float(self.parameters["OverlapDegree"])
        self.worst_pop_id = -1
        self.evolve_id = -1
        self.optimizer_iteration = 0
        self.pre_fitness = np.zeros(self.swarm_number, dtype=float)
        self.cur_fitness = np.zeros(self.swarm_number, dtype=float)
        self.performance = np.zeros(self.swarm_number, dtype=float)
        self.pop: list[CDESubPopulation] = []

        for _ in range(self.swarm_number):
            swarm = self._generate_subpopulation(problem)
            if swarm is None:
                return
            self.pop.append(swarm)
            if problem.state.recent_change or problem.state.terminated:
                return
        self.swarm_number = len(self.pop)
        self.pre_fitness = np.zeros(self.swarm_number, dtype=float)
        self.cur_fitness = np.zeros(self.swarm_number, dtype=float)
        self.performance = np.zeros(self.swarm_number, dtype=float)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        if not self.pop:
            swarm = self._generate_subpopulation(problem)
            if swarm is not None:
                self.pop.append(swarm)
                self._resize_tracking()
            return

        self.optimizer_iteration += 1
        if self.optimizer_iteration <= 2:
            indices = list(range(self.swarm_number))
            brownian_best = True
        else:
            evolved = (
                list(range(self.swarm_number))
                if self.optimizer_iteration == 3 or self.evolve_id < 0
                else [self.evolve_id]
            )
            self.worst_pop_id = self._find_worst_pop()
            self._update_performance(evolved)
            self.evolve_id = int(np.argmax(self.performance))
            indices = [self.evolve_id]
            brownian_best = False

        for index in indices:
            self._evolve_swarm(self.pop[index], problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            self._refresh_geometry(self.pop[index])
            if brownian_best and self.diversity_flag == 1:
                self._brownian_best_move(self.pop[index], problem)
                if problem.state.recent_change or problem.state.terminated:
                    return

        if self.optimizer_iteration > 2:
            self._check_overlapping(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            self._reinitialize_marked(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            self._brownian_worst_move(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
        else:
            self._reinitialize_marked(problem)
            if problem.state.recent_change or problem.state.terminated:
                return

        for index, swarm in enumerate(self.pop):
            self.pre_fitness[index] = self.cur_fitness[index]
            self.cur_fitness[index] = swarm.best_fitness

    def on_environment_change(self, problem) -> None:
        for swarm in self.pop:
            values = evaluate_(problem, swarm.positions, swarm.positions.shape[0])
            if problem.state.recent_change or problem.state.terminated:
                return
            if values.size != swarm.positions.shape[0] or np.all(np.isnan(values)):
                return
            swarm.fitness = values
            swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)
        self.optimizer_iteration = 0
        self.evolve_id = -1

    def _generate_subpopulation(self, problem) -> CDESubPopulation | None:
        rows = self.indiv_size
        positions = self.bounds.sample(rows, self.rng)
        strategies = (
            self.rng.integers(1, 8, size=rows)
            if self.strategy_flag == 0
            else np.full(rows, self.strategy_flag, dtype=int)
        )
        if self.random_flag == 2:
            crossover_rates = np.full(rows, 0.6)
            scale_factors = np.full(rows, 0.5)
        else:
            crossover_rates = self.rng.random(rows)
            scale_factors = self.rng.random(rows)

        individual_types = self._initial_individual_types(rows)
        fitness = evaluate_(problem, positions, rows)
        if problem.state.recent_change or problem.state.terminated:
            return None
        if fitness.size != rows:
            return None

        swarm = CDESubPopulation(
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
            center=np.zeros(self.dimension),
            current_radius=0.0,
            reinit_state=0,
        )
        self._reset_best(swarm)
        if self.diversity_flag == 1:
            self._initial_brownian_replacements(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return None
        elif self.diversity_flag == 2:
            self._initialize_quantum_individuals(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return None
        self._refresh_geometry(swarm)
        swarm.reinit_state = 0
        return swarm

    def _initial_individual_types(self, rows: int) -> np.ndarray:
        if self.diversity_flag == 1:
            return np.ones(rows, dtype=int)
        if self.diversity_flag == 2:
            normal_count = min(5, rows)
            types = np.full(rows, 2, dtype=int)
            types[:normal_count] = 0
            return types
        return np.zeros(rows, dtype=int)

    def _initial_brownian_replacements(self, swarm: CDESubPopulation, problem) -> None:
        count = min(self.brown_number, swarm.positions.shape[0])
        if count <= 0:
            return
        for index in np.argsort(np.nan_to_num(swarm.fitness, nan=-np.inf))[:count]:
            candidate = self._best_gaussian_candidate(swarm, sigma=0.2)
            result = evaluate_(problem, candidate, 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if not result.size or np.isnan(result[0]):
                continue
            swarm.positions[index, :] = candidate
            swarm.fitness[index] = float(result[0])
            self._reset_best(swarm)

    def _initialize_quantum_individuals(self, swarm: CDESubPopulation, problem) -> None:
        quantum_indices = np.flatnonzero(swarm.individual_types == 2)
        if quantum_indices.size == 0:
            return
        self._reset_best(swarm)
        for index in quantum_indices:
            direction = self.rng.standard_normal(self.dimension)
            norm = float(np.linalg.norm(direction))
            if norm <= 0.0:
                continue
            candidate = swarm.best_position + direction * self.rng.random() / norm
            candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
            result = evaluate_(problem, candidate, 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if not result.size or np.isnan(result[0]):
                continue
            swarm.positions[index, :] = candidate
            swarm.fitness[index] = float(result[0])
        self._reset_best(swarm)

    def _evolve_swarm(self, swarm: CDESubPopulation, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows < 2:
            return
        self._reset_best(swarm)
        swarm.donor_positions = self._mutation(swarm)
        swarm.trial_positions = self.de_operator.binomial_crossover_population(
            swarm.positions,
            swarm.donor_positions,
            swarm.crossover_rates,
            self.rng,
        )
        swarm.trial_positions = np.clip(
            swarm.trial_positions,
            self.min_coordinate,
            self.max_coordinate,
        )

        for index in range(rows):
            if swarm.individual_types[index] == 2:
                continue
            result = evaluate_(problem, swarm.trial_positions[index, :], 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if not result.size or np.isnan(result[0]):
                continue
            value = float(result[0])
            if value > swarm.fitness[index]:
                swarm.positions[index, :] = swarm.trial_positions[index, :]
                swarm.fitness[index] = value
        self._reset_best(swarm)

        if self.diversity_flag == 2:
            self._quantum_move(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            self._reset_best(swarm)

    def _mutation(self, swarm: CDESubPopulation) -> np.ndarray:
        rows = swarm.positions.shape[0]
        donors = swarm.positions.copy()
        for index in range(rows):
            if swarm.individual_types[index] == 2:
                continue
            strategy = int(swarm.strategies[index])
            factor = float(swarm.scale_factors[index])
            donors[index, :] = self._mutation_one(swarm, index, strategy, factor)
        return donors

    def _mutation_one(
        self,
        swarm: CDESubPopulation,
        index: int,
        strategy: int,
        factor: float,
    ) -> np.ndarray:
        positions = swarm.positions
        rows = positions.shape[0]
        all_indices = np.arange(rows)

        if strategy == 4 and rows >= 5:
            choices = all_indices[all_indices != swarm.best_id]
            selected = self.rng.choice(choices, size=4, replace=False)
            return swarm.best_position + factor * (
                positions[selected[0], :]
                + positions[selected[1], :]
                - positions[selected[2], :]
                - positions[selected[3], :]
            )
        if strategy == 3 and rows >= 3:
            choices = all_indices[all_indices != swarm.best_id]
            selected = self.rng.choice(choices, size=2, replace=False)
            return swarm.best_position + factor * (
                positions[selected[0], :] - positions[selected[1], :]
            )
        if strategy == 2 and rows >= 5:
            selected = self.rng.choice(all_indices, size=5, replace=False)
            return positions[selected[0], :] + factor * (
                positions[selected[1], :]
                + positions[selected[2], :]
                - positions[selected[3], :]
                - positions[selected[4], :]
            )
        if strategy == 1 and rows >= 3:
            selected = self.rng.choice(all_indices, size=3, replace=False)
            return positions[selected[0], :] + factor * (
                positions[selected[1], :] - positions[selected[2], :]
            )
        if strategy == 5 and rows >= 4:
            choices = all_indices[all_indices != swarm.best_id]
            selected = self.rng.choice(choices, size=3, replace=False)
            base = positions[selected[0], :]
            return base + factor * (swarm.best_position - base) + factor * (
                positions[selected[1], :] - positions[selected[2], :]
            )
        if strategy == 6 and rows >= 4:
            choices = all_indices[all_indices != index]
            selected = self.rng.choice(choices, size=3, replace=False)
            current = positions[index, :]
            return current + factor * (positions[selected[0], :] - current) + factor * (
                positions[selected[1], :] - positions[selected[2], :]
            )
        if strategy == 7 and rows >= 4:
            mask = (all_indices != index) & (all_indices != swarm.best_id)
            choices = all_indices[mask]
            if choices.size >= 2:
                selected = self.rng.choice(choices, size=2, replace=False)
                current = positions[index, :]
                return current + factor * (swarm.best_position - current) + factor * (
                    positions[selected[0], :] - positions[selected[1], :]
                )
        return positions[index, :].copy()

    def _quantum_move(self, swarm: CDESubPopulation, problem) -> None:
        self._reset_best(swarm)
        for index in np.flatnonzero(swarm.individual_types == 2):
            if index == swarm.best_id:
                continue
            direction = self.rng.standard_normal(self.dimension)
            norm = float(np.linalg.norm(direction))
            if norm <= 0.0:
                continue
            candidate = (
                swarm.positions[swarm.best_id, :]
                + direction * self.rng.random() / norm
            )
            candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
            result = evaluate_(problem, candidate, 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if not result.size or np.isnan(result[0]):
                continue
            swarm.positions[index, :] = candidate
            swarm.fitness[index] = float(result[0])

    def _brownian_best_move(self, swarm: CDESubPopulation, problem) -> None:
        for _ in range(3):
            candidate = self._best_gaussian_candidate(swarm, sigma=0.4)
            result = evaluate_(problem, candidate, 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if not result.size or np.isnan(result[0]):
                continue
            value = float(result[0])
            if value > swarm.fitness[swarm.best_id]:
                swarm.positions[swarm.best_id, :] = candidate
                swarm.fitness[swarm.best_id] = value
                self._reset_best(swarm)

    def _brownian_worst_move(self, problem) -> None:
        if self.diversity_flag != 1:
            return
        for swarm in self.pop:
            count = min(self.brown_number, swarm.positions.shape[0])
            if count <= 0:
                continue
            for index in np.argsort(np.nan_to_num(swarm.fitness, nan=-np.inf))[:count]:
                candidate = self._best_gaussian_candidate(swarm, sigma=0.2)
                result = evaluate_(problem, candidate, 1)
                if problem.state.recent_change or problem.state.terminated:
                    return
                if not result.size or np.isnan(result[0]):
                    continue
                value = float(result[0])
                if value > swarm.fitness[index]:
                    swarm.positions[index, :] = candidate
                    swarm.fitness[index] = value
                    self._reset_best(swarm)

    def _best_gaussian_candidate(self, swarm: CDESubPopulation, sigma: float) -> np.ndarray:
        candidate = swarm.positions[swarm.best_id, :].copy()
        candidate += sigma * self.rng.standard_normal(self.dimension)
        return np.clip(candidate, self.min_coordinate, self.max_coordinate)

    def _check_overlapping(self, problem) -> None:
        if self.swarm_number <= 1:
            return
        limit = float(np.mean(np.asarray(self.exclusion_limit, dtype=float)))
        for left in range(self.swarm_number - 1):
            for right in range(left + 1, self.swarm_number):
                left_swarm = self.pop[left]
                right_swarm = self.pop[right]
                if left_swarm.reinit_state or right_swarm.reinit_state:
                    continue
                distance = float(
                    np.linalg.norm(left_swarm.best_position - right_swarm.best_position)
                )
                if distance >= limit:
                    continue
                mid_position = (left_swarm.best_position + right_swarm.best_position) / 2.0
                mid_fitness = evaluate_(problem, mid_position, 1)
                if problem.state.recent_change or problem.state.terminated:
                    return
                if (
                    mid_fitness.size
                    and not np.isnan(mid_fitness[0])
                    and mid_fitness[0] < left_swarm.best_fitness
                    and mid_fitness[0] < right_swarm.best_fitness
                ):
                    continue
                if left_swarm.best_fitness > right_swarm.best_fitness:
                    right_swarm.reinit_state = 1
                else:
                    left_swarm.reinit_state = 1

    def _reinitialize_marked(self, problem) -> None:
        for index, swarm in enumerate(list(self.pop)):
            if swarm.reinit_state != 1:
                continue
            replacement = self._generate_subpopulation(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            if replacement is not None:
                self.pop[index] = replacement

    def _find_worst_pop(self) -> int:
        if not self.pop:
            return -1
        values = np.asarray([swarm.best_fitness for swarm in self.pop], dtype=float)
        return int(np.argmin(np.nan_to_num(values, nan=np.inf)))

    def _update_performance(self, evolved_indices: list[int]) -> None:
        del evolved_indices
        if not self.pop or self.worst_pop_id < 0:
            return
        worst = self.pop[self.worst_pop_id].best_fitness
        for index, swarm in enumerate(self.pop):
            self.performance[index] = (1.0 + abs(swarm.best_fitness - worst)) * (
                1.0 + abs(self.cur_fitness[index] - self.pre_fitness[index])
            )

    def _refresh_geometry(self, swarm: CDESubPopulation) -> None:
        if swarm.positions.shape[0] == 0:
            swarm.center = np.zeros(self.dimension)
            swarm.current_radius = 0.0
            return
        swarm.center = np.mean(swarm.positions, axis=0)
        swarm.current_radius = float(
            np.mean(np.linalg.norm(swarm.positions - swarm.center, axis=1))
        )

    def _reset_best(self, swarm: CDESubPopulation) -> None:
        if swarm.positions.shape[0] == 0:
            return
        fitness = np.asarray(swarm.fitness, dtype=float)
        comparable = np.nan_to_num(fitness, nan=-np.inf)
        best_id = int(np.argmax(comparable))
        swarm.best_id = best_id
        swarm.best_position = swarm.positions[best_id, :].copy()
        swarm.best_fitness = float(fitness[best_id])

    def _resize_tracking(self) -> None:
        self.swarm_number = len(self.pop)
        self.pre_fitness = np.resize(self.pre_fitness, self.swarm_number)
        self.cur_fitness = np.resize(self.cur_fitness, self.swarm_number)
        self.performance = np.resize(self.performance, self.swarm_number)

    def _map_option(self, value: object, mapping: dict[str, int], default: int) -> int:
        if isinstance(value, str):
            return mapping.get(value, default)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        return numeric if numeric in mapping.values() else default
