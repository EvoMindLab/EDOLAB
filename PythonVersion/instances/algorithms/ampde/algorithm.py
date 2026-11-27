"""AMPDE algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import DEOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.ampde.components.clustering import AMPDEClusterer
from instances.algorithms.ampde.state import AMPDEMapRecord, AMPDESubPopulation
from instances.algorithms.common import evaluate_, stack_rows


class AMPDESubPopulationFactory:
    """Build AMPDE subpopulations from sampled positions."""

    def __init__(self, algorithm) -> None:
        self.algorithm = algorithm

    def __getattr__(self, name: str):
        return getattr(self.algorithm, name)

    def generate_subpopulations(
        self,
        positions: np.ndarray,
        problem,
    ) -> tuple[list[AMPDESubPopulation], np.ndarray]:
        positions = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        groups, remain = AMPDEClusterer().groups_and_remain(positions)
        valid_groups: list[np.ndarray] = []
        remain_rows = [row for row in remain]
        for group in groups:
            if group.shape[0] < 5:
                remain_rows.extend(group)
            else:
                valid_groups.append(group)

        swarms: list[AMPDESubPopulation] = []
        for group in valid_groups:
            values = evaluate_(problem, group, group.shape[0])
            if problem.state.recent_change:
                return swarms, stack_rows(remain_rows, self.dimension)
            center = np.mean(group, axis=0)
            init_radius = float(np.mean(np.linalg.norm(group - center, axis=1)))
            best_id = int(np.argmax(np.nan_to_num(values, nan=-np.inf)))
            swarms.append(
                AMPDESubPopulation(
                    positions=group.copy(),
                    fitness=values,
                    gbest_past_environment=np.full(self.dimension, np.nan),
                    donor_positions=np.zeros_like(group),
                    trial_positions=np.zeros_like(group),
                    is_hibernated=0,
                    stagnating_count=0,
                    is_stagnated=0,
                    best_id=best_id,
                    best_position=group[best_id, :].copy(),
                    best_fitness=float(values[best_id]),
                    center=center,
                    init_radius=init_radius,
                    current_radius=init_radius,
                )
            )
        return swarms, stack_rows(remain_rows, self.dimension)


@algorithm_registry.register("AMPDE")
class AMPDEAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "AMPDE"

    def __init__(self) -> None:
        self.swarm_factory = AMPDESubPopulationFactory(self)

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
                "Crossover probability used to mix donor and target vectors.",
            ),
            "F": ParameterSpec(
                "F",
                0.5,
                "numeric",
                (0, 2),
                "Scaling factor controlling the influence of difference vectors in mutation.",
            ),
            "ConvergenceLimit": ParameterSpec(
                "ConvergenceLimit",
                0.00001,
                "numeric",
                (0, 0.1),
                "Sub-swarm radius below which it is considered converged (hibernated).",
            ),
            "OverlapDegree": ParameterSpec(
                "OverlapDegree",
                0.1,
                "numeric",
                (0, 1),
                "Proportion of individuals that must overlap to trigger sub-swarm removal.",
            ),
            "diversityFactor": ParameterSpec(
                "diversityFactor",
                0.005,
                "numeric",
                (0, 1),
                "Factor used to scale domain size for diversity-based activation of sub-swarms.",
            ),
            "StepIndivs": ParameterSpec(
                "StepIndivs",
                10,
                "integer",
                (0, 100),
                "Base number of individuals used as a unit when increasing diversity.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_indivs = 1000
        self.min_indivs = 10
        self.shift_severity = 1.0
        self.swarm_number = 0
        self.convergence_limit = float(self.parameters["ConvergenceLimit"])
        self.hiber_step = 0.5
        self.first_update = 1
        self.off_peak = 3
        self.current_pop_size = 0
        self.previous_pop_size = 0
        self.iter_individuals_begin: list[float] = []
        self.iter_pop_size_end: list[int] = []
        self.adjust_iter = 0
        self.distance_accuracy = 0.2
        self.avoidance_flag = 0
        self.domain_size = float(np.linalg.norm(self.max_coordinate - self.min_coordinate))
        self.de_operator = DEOperator()

        init_size = max(0, int(self.parameters["initPopulationSize"]))
        init_x = self.bounds.sample(init_size, self.rng)
        self.pop, self.remain_indivs = self.swarm_factory.generate_subpopulations(
            init_x,
            problem,
        )
        self.swarm_number = len(self.pop)
        self.map_pop_inds = [AMPDEMapRecord() for _ in range(max(self.swarm_number, 1))]

    def iterate(self, problem) -> None:
        if problem.state.recent_change:
            return
        self.domain_size = float(np.linalg.norm(self.max_coordinate - self.min_coordinate))
        for swarm in list(self.pop):
            before_radius = swarm.current_radius
            if swarm.is_hibernated == 0:
                self._de_move(swarm, problem)
                if problem.state.recent_change:
                    return
            self._refresh_radius(swarm)
            if swarm.current_radius / (before_radius + 0.000001) < 0.9:
                swarm.stagnating_count = 0
            else:
                swarm.stagnating_count += 1
            self._brownian_or_cauchy_move(swarm, problem)
            if problem.state.recent_change:
                return
            if swarm.current_radius < self.convergence_limit:
                swarm.is_hibernated = 1
                swarm.is_stagnated = 0
            if swarm.is_hibernated == 1:
                swarm.init_radius = self.distance_accuracy

        self._check_stagnant()
        self._check_overlapping()
        self._check_diversity(problem)

    def on_environment_change(self, problem) -> None:
        for swarm in self.pop:
            values = evaluate_(problem, swarm.positions, swarm.positions.shape[0])
            if problem.state.recent_change:
                return
            swarm.fitness = values
            swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)

    def _de_move(self, swarm: AMPDESubPopulation, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows < 5:
            return
        self._reset_best(swarm)
        factors = np.full(rows, float(self.parameters["F"]))
        crossover_rates = np.full(rows, float(self.parameters["CR"]))
        swarm.donor_positions = self._mutation_best_2_excluding_best(swarm, factors)
        swarm.trial_positions = self.de_operator.binomial_crossover_population(
            swarm.positions,
            swarm.donor_positions,
            crossover_rates,
            self.rng,
        )
        swarm.trial_positions = np.clip(
            swarm.trial_positions,
            self.min_coordinate,
            self.max_coordinate,
        )
        trial_fitness = evaluate_(problem, swarm.trial_positions, rows)
        valid = ~np.isnan(trial_fitness)
        if np.any(valid):
            selected_positions, selected_fitness, _ = self.de_operator.greedy_select(
                swarm.positions[valid, :],
                swarm.fitness[valid],
                swarm.trial_positions[valid, :],
                trial_fitness[valid],
            )
            swarm.positions[valid, :] = selected_positions
            swarm.fitness[valid] = selected_fitness
            self._reset_best(swarm)
        if problem.state.recent_change:
            return

    def _mutation_best_2_excluding_best(
        self,
        swarm: AMPDESubPopulation,
        factors: np.ndarray,
    ) -> np.ndarray:
        rows = swarm.positions.shape[0]
        choices = np.delete(np.arange(rows), swarm.best_id)
        donors = np.empty_like(swarm.positions)
        for row in range(rows):
            selected = self.rng.choice(choices, size=4, replace=False)
            donors[row, :] = swarm.best_position + factors[row] * (
                swarm.positions[selected[0], :]
                + swarm.positions[selected[1], :]
                - swarm.positions[selected[2], :]
                - swarm.positions[selected[3], :]
            )
        return donors

    def _brownian_or_cauchy_move(self, swarm: AMPDESubPopulation, problem) -> None:
        if swarm.positions.shape[0] == 0:
            return
        if swarm.is_stagnated == 0:
            for _ in range(5):
                candidate = swarm.positions[swarm.best_id, :].copy()
                candidate += swarm.current_radius * self.rng.standard_normal(self.dimension)
                candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
                result = evaluate_(problem, candidate, 1)
                if result.size and not np.isnan(result[0]):
                    best_result = float(result[0])
                    if best_result > swarm.fitness[swarm.best_id]:
                        swarm.fitness[swarm.best_id] = best_result
                        swarm.positions[swarm.best_id, :] = candidate
                        swarm.best_position = candidate.copy()
                        swarm.best_fitness = best_result
                if problem.state.recent_change:
                    return
        else:
            candidate = swarm.positions[swarm.best_id, :].copy()
            scale = (self.max_coordinate - self.min_coordinate) / 10.0
            candidate += scale * self.rng.standard_cauchy(self.dimension)
            candidate = np.clip(candidate, self.min_coordinate, self.max_coordinate)
            result = evaluate_(problem, candidate, 1)
            if result.size and not np.isnan(result[0]):
                swarm.positions[swarm.best_id, :] = candidate
                swarm.fitness[swarm.best_id] = float(result[0])
                self._reset_best(swarm)
                swarm.stagnating_count = 0
            if problem.state.recent_change:
                return

    def _check_stagnant(self) -> None:
        active = [swarm.current_radius for swarm in self.pop if swarm.is_hibernated == 0]
        ave_radius = float(np.mean(active)) if active else 0.0
        diversity_limit = float(self.parameters["diversityFactor"]) * self.domain_size
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                continue
            size = max(1, swarm.positions.shape[0])
            if (
                (
                    swarm.stagnating_count > size
                    and swarm.current_radius >= ave_radius
                    and swarm.current_radius >= diversity_limit
                )
                or swarm.stagnating_count > 10 * size
            ):
                swarm.is_stagnated = 1
            else:
                swarm.is_stagnated = 0

    def _check_overlapping(self) -> None:
        threshold = float(self.parameters["OverlapDegree"])
        idx = None
        while idx != -1:
            idx = -1
            for ii in range(self.swarm_number):
                if self.pop[ii].positions.shape[0] == 0:
                    continue
                for jj in range(ii + 1, self.swarm_number):
                    if self.pop[jj].positions.shape[0] == 0:
                        continue
                    distance = float(np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position))
                    if distance >= self.pop[ii].init_radius or distance >= self.pop[jj].init_radius:
                        continue
                    c1 = sum(
                        np.linalg.norm(self.pop[ii].center - row) < self.pop[ii].init_radius
                        for row in self.pop[jj].positions
                    )
                    c2 = sum(
                        np.linalg.norm(self.pop[jj].center - row) < self.pop[jj].init_radius
                        for row in self.pop[ii].positions
                    )
                    if (
                        c1 > threshold * self.pop[jj].positions.shape[0]
                        and c2 > threshold * self.pop[ii].positions.shape[0]
                    ):
                        if self.pop[ii].best_fitness > self.pop[jj].best_fitness:
                            self._activate_swarm(ii)
                            self._delete_swarm(jj)
                            idx = jj
                        else:
                            self._activate_swarm(jj)
                            self._delete_swarm(ii)
                            idx = ii
                        break
                if idx != -1:
                    break

    def _check_diversity(self, problem) -> None:
        active = [
            swarm.current_radius
            for swarm in self.pop
            if swarm.is_stagnated == 0 and swarm.is_hibernated == 0
        ]
        non_stagnated_radius = float(np.mean(active)) if active else 0.0
        diversity_limit = float(self.parameters["diversityFactor"]) * self.domain_size
        if self.swarm_number > 0 and non_stagnated_radius >= diversity_limit:
            return

        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                swarm.is_hibernated = 0

        self.iter_pop_size_end.append(self.swarm_number)
        if len(self.iter_pop_size_end) > 2:
            self.iter_pop_size_end.pop(0)
        if self.adjust_iter == 0:
            total = sum(swarm.positions.shape[0] for swarm in self.pop)
            self.iter_individuals_begin = [float(total)]

        self._update_memory()
        if self.first_update == 1:
            self.current_pop_size = self.iter_pop_size_end[-1]
            self.previous_pop_size = self.current_pop_size
            self.first_update = 0
        else:
            self.current_pop_size = self.iter_pop_size_end[-1]
            self.previous_pop_size = self.iter_pop_size_end[-2]

        record = self._map_record_for_count(self.current_pop_size)
        ratio = abs(self.current_pop_size - self.previous_pop_size) / self.off_peak
        base = record.mean + record.std_dev * self.rng.standard_normal()
        if ratio >= 1 or (0 < ratio and self.rng.random() <= ratio):
            base += int(self.parameters["StepIndivs"]) * (
                self.current_pop_size - self.previous_pop_size
            )

        total_survive = sum(swarm.positions.shape[0] for swarm in self.pop)
        if base <= total_survive:
            base = total_survive + int(self.parameters["StepIndivs"]) * 2
        next_individuals = int(np.ceil(min(self.max_indivs, max(self.min_indivs, base))))
        self.iter_individuals_begin = [float(next_individuals)]

        num_remain = self.remain_indivs.shape[0]
        num_add = next_individuals - total_survive - num_remain
        if num_add > 0:
            add = self.bounds.sample(num_add, self.rng)
            add = np.vstack([add, self.remain_indivs]) if num_remain else add
            self._add_generated_swarms(add, problem)
        elif num_remain > next_individuals - total_survive:
            add = np.vstack([self.bounds.sample(20, self.rng), self.remain_indivs])
            self._add_generated_swarms(add, problem)

        self.adjust_iter += 1

    def _add_generated_swarms(self, positions: np.ndarray, problem) -> None:
        self.remain_indivs = np.empty((0, self.dimension))
        swarms, remain = self.swarm_factory.generate_subpopulations(positions, problem)
        self.remain_indivs = remain
        if problem.state.recent_change:
            return
        self.pop.extend(swarms)
        self.swarm_number = len(self.pop)

    def _update_memory(self) -> None:
        record = self._map_record_for_count(self.swarm_number)
        record.num += 1
        value = float(self.iter_individuals_begin[-1]) if self.iter_individuals_begin else 0.0
        record.indivs.append(value)
        record.mean = ((record.mean * (record.num - 1)) + value) / record.num
        samples = np.asarray(record.indivs, dtype=float)
        record.std_dev = float(np.sqrt(np.mean((samples - record.mean) ** 2)))

    def _map_record_for_count(self, swarm_count: int) -> AMPDEMapRecord:
        index = max(1, int(swarm_count)) - 1
        while index >= len(self.map_pop_inds):
            self.map_pop_inds.append(AMPDEMapRecord())
        return self.map_pop_inds[index]

    def _activate_swarm(self, index: int) -> None:
        self.pop[index].is_hibernated = 0
        self.pop[index].stagnating_count = 0
        self.pop[index].is_stagnated = 0

    def _delete_swarm(self, index: int) -> None:
        del self.pop[index]
        self.swarm_number = len(self.pop)

    def _refresh_radius(self, swarm: AMPDESubPopulation) -> None:
        if swarm.positions.shape[0] == 0:
            swarm.center = np.zeros(self.dimension)
            swarm.current_radius = 0.0
            return
        swarm.center = np.mean(swarm.positions, axis=0)
        swarm.current_radius = float(np.mean(np.linalg.norm(swarm.positions - swarm.center, axis=1)))

    def _reset_best(self, swarm: AMPDESubPopulation) -> None:
        fitness = np.asarray(swarm.fitness, dtype=float)
        best_id = int(np.argmax(np.nan_to_num(fitness, nan=-np.inf)))
        swarm.best_id = best_id
        swarm.best_position = swarm.positions[best_id, :].copy()
        swarm.best_fitness = float(fitness[best_id])
