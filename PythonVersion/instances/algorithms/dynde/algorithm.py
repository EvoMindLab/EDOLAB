"""DynDE algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.dynde.state import DynDESubPopulation


@algorithm_registry.register("DynDE")
class DynDEAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "DynDE"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "initPopulationSize": ParameterSpec(
                "initPopulationSize",
                100,
                "integer",
                (0, 500),
                "Total number of individuals across all sub-populations at start.",
            ),
            "CR": ParameterSpec(
                "CR",
                0.6,
                "numeric",
                (0, 2),
                "Crossover probability for generating trial vectors.",
            ),
            "F": ParameterSpec(
                "F",
                0.5,
                "numeric",
                (0, 2),
                "Mutation scaling factor in donor vector computation.",
            ),
            "SwarmNumber": ParameterSpec(
                "SwarmNumber",
                10,
                "integer",
                (0, 100),
                "Number of independent sub-populations.",
            ),
            "IndivSize": ParameterSpec(
                "IndivSize",
                10,
                "integer",
                (0, 100),
                "Number of individuals in each sub-population.",
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
                "Maximum average distance from center for convergence.",
            ),
            "OverlapDegree": ParameterSpec(
                "OverlapDegree",
                0.1,
                "numeric",
                (0, 1),
                "Fraction of overlap used by the original configuration.",
            ),
            "RandomFlag": ParameterSpec(
                "RandomFlag",
                "Fixed CR/F",
                "option",
                ("Random CR/F", "Fixed CR/F"),
                "Selection method for CR and F parameters per individual.",
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
                "Mutation scheme used to construct donor vectors in DE.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)

        self.init_population_size = int(self.parameters["initPopulationSize"])
        self.configured_swarm_number = int(self.parameters["SwarmNumber"])
        self.configured_indiv_size = int(self.parameters["IndivSize"])
        self.configured_brown_number = int(self.parameters["BrownNumber"])
        self.configured_convergence_limit = float(self.parameters["ConvergenceLimit"])
        self.configured_overlap_degree = float(self.parameters["OverlapDegree"])

        self.swarm_number = 10
        self.indiv_size = 10
        self.brown_number = 5
        self.convergence_limit = 0.01
        self.exclusion_limit = 15.5
        self.hiber_step = 0.5
        self.overlap_degree = 0.1
        self.first_update = 1
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

        self.pop: list[DynDESubPopulation] = []
        for _ in range(self.swarm_number):
            swarm = self._generate_subpopulation(problem)
            if swarm is not None:
                self.pop.append(swarm)
            if problem.state.recent_change or problem.state.terminated:
                self.swarm_number = len(self.pop)
                return
        self.swarm_number = len(self.pop)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        for swarm in self.pop:
            swarm.donor_positions = self._mutation(swarm)
            swarm.trial_positions = np.zeros_like(swarm.positions)
            rows = swarm.positions.shape[0]
            for row in range(rows):
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
                if problem.state.recent_change or problem.state.terminated:
                    return
                if not values.size or np.isnan(values[0]):
                    continue
                value = float(values[0])
                if value > swarm.fitness[row]:
                    swarm.positions[row, :] = swarm.trial_positions[row, :]
                    swarm.fitness[row] = value
            best_value, best_id = self._best_value_id(swarm.fitness)
            if best_value > swarm.best_fitness:
                swarm.best_fitness = best_value
                swarm.best_id = best_id
                swarm.best_position = swarm.positions[best_id, :].copy()

        if self.diversity_flag == 1:
            self._brownian_individuals(problem)
            if problem.state.recent_change or problem.state.terminated:
                return

        self._check_overlapping()
        self._reinitialize_marked(problem)

    def on_environment_change(self, problem) -> None:
        for swarm in self.pop:
            values = evaluate_(problem, swarm.positions, swarm.positions.shape[0])
            if problem.state.recent_change or problem.state.terminated:
                return
            if values.size != swarm.positions.shape[0]:
                return
            swarm.fitness = values
            swarm.gbest_past_environment = swarm.best_position.copy()
            best_value, best_id = self._best_value_id(swarm.fitness)
            swarm.best_fitness = best_value
            swarm.best_id = best_id
            swarm.best_position = swarm.positions[best_id, :].copy()

    def _generate_subpopulation(self, problem) -> DynDESubPopulation | None:
        rows = max(1, int(self.indiv_size))
        positions = self.bounds.sample(rows, self.rng)
        if self.strategy_flag == 0:
            strategies = self.rng.integers(1, 8, size=rows)
        else:
            strategies = np.full(rows, self.strategy_flag, dtype=int)
        if self.random_flag == 0:
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

        swarm = DynDESubPopulation(
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
        )
        self._refresh_best(swarm)
        if problem.state.recent_change:
            swarm.fitness = np.full(rows, -np.inf, dtype=float)
            self._refresh_best(swarm)
        self._refresh_geometry(swarm)
        swarm.reinit_state = 0
        return swarm

    def _mutation(self, swarm: DynDESubPopulation) -> np.ndarray:
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
        swarm: DynDESubPopulation,
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

    def _brownian_individuals(self, problem) -> None:
        last_swarm = None
        for swarm in self.pop:
            last_swarm = swarm
            count = min(max(0, int(self.brown_number)), swarm.positions.shape[0])
            if count <= 0:
                continue
            order = np.argsort(np.nan_to_num(swarm.fitness, nan=-np.inf))[:count]
            for row in order:
                candidate = swarm.positions[swarm.best_id, :].copy()
                candidate += 0.2 * self.rng.standard_normal(self.dimension)
                candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
                values = evaluate_(problem, candidate, 1)
                if problem.state.recent_change or problem.state.terminated:
                    return
                if not values.size or np.isnan(values[0]):
                    continue
                value = float(values[0])
                if value > swarm.fitness[row]:
                    swarm.positions[row, :] = candidate
                    swarm.fitness[row] = value
                    best_value, best_id = self._best_value_id(swarm.fitness)
                    if best_value > swarm.best_fitness:
                        swarm.best_fitness = best_value
                        swarm.best_id = best_id
                        swarm.best_position = swarm.positions[best_id, :].copy()
        if last_swarm is not None:
            self._refresh_geometry(last_swarm)

    def _check_overlapping(self) -> None:
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
                if (
                    left_swarm.reinit_state == 0
                    and right_swarm.reinit_state == 0
                    and distance < self.exclusion_limit
                ):
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

    def _reinitialize_marked(self, problem) -> None:
        for index, swarm in enumerate(list(self.pop)):
            if swarm.reinit_state != 1:
                continue
            replacement = self._generate_subpopulation(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            if replacement is not None:
                self.pop[index] = replacement

    def _refresh_geometry(self, swarm: DynDESubPopulation) -> None:
        if swarm.positions.shape[0] == 0:
            swarm.center = np.zeros(self.dimension, dtype=float)
            swarm.current_radius = 0.0
            return
        swarm.center = np.mean(swarm.positions, axis=0)
        swarm.current_radius = float(
            np.mean(np.linalg.norm(swarm.positions - swarm.center, axis=1))
        )

    def _refresh_best(self, swarm: DynDESubPopulation) -> None:
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

    def _map_option(self, value: object, mapping: dict[str, int], default: int) -> int:
        if isinstance(value, str):
            return mapping.get(value, default)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        return numeric if numeric in mapping.values() else default
