"""AMPPSO_GI algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.amppso_gi.components.clustering import AMPPSOGIClusterer
from instances.algorithms.amppso_gi.state import AMPHiberArea, AMPMapRecord, AMPPSOSwarm
from instances.algorithms.common import (
    evaluate_,
    update_center,
    update_current_radius,
)


class AMPPSOGISwarmFactory:
    """Build AMPPSO_GI sub-swarms from sampled positions."""

    def __init__(self, algorithm) -> None:
        self.algorithm = algorithm

    def __getattr__(self, name: str):
        return getattr(self.algorithm, name)

    def generate_subpopulations(
        self,
        positions: np.ndarray,
        problem,
    ) -> tuple[list[AMPPSOSwarm], np.ndarray]:
        positions = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        groups, remain = AMPPSOGIClusterer().groups_and_remain(positions)
        swarms: list[AMPPSOSwarm] = []
        for group in groups:
            values = evaluate_(problem, group, group.shape[0])
            center = np.mean(group, axis=0)
            init_radius = float(np.mean(np.linalg.norm(group - center, axis=1)))
            velocity = np.zeros_like(group)
            if not problem.state.recent_change:
                pbest_value = values.copy()
            else:
                values = np.full(group.shape[0], -np.inf)
                pbest_value = values.copy()
            best_id = int(np.argmax(pbest_value))
            swarms.append(
                AMPPSOSwarm(
                    positions=group.copy(),
                    gbest_past_environment=np.full(self.dimension, np.nan),
                    velocities=velocity,
                    shifts=[],
                    fitness=values,
                    pbest_positions=group.copy(),
                    is_hibernated=0,
                    stagnating_count=0,
                    is_stagnated=0,
                    pbest_fitness=pbest_value,
                    best_fitness=float(pbest_value[best_id]),
                    best_position=group[best_id, :].copy(),
                    best_id=best_id,
                    center=center,
                    init_radius=init_radius,
                    current_radius=init_radius,
                )
            )
            if problem.state.recent_change:
                break
        return swarms, remain


@algorithm_registry.register("AMPPSO_GI")
class AMPPSOGIAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "AMPPSO_GI"

    def __init__(self) -> None:
        self.swarm_factory = AMPPSOGISwarmFactory(self)

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "initPopulationSize": ParameterSpec("initPopulationSize", 100, "integer", (0, 1000)),
            "w": ParameterSpec("w", 0.7298, "numeric", (0, 2)),
            "c1": ParameterSpec("c1", 1.496, "numeric", (0, 5)),
            "c2": ParameterSpec("c2", 1.496, "numeric", (0, 5)),
            "ConvergenceLimit": ParameterSpec("ConvergenceLimit", 0.00001, "numeric", (0, 0.1)),
            "OverlapDegree": ParameterSpec("OverlapDegree", 0.1, "numeric", (0, 1)),
            "diversityFactor": ParameterSpec("diversityFactor", 0.005, "numeric", (0, 1)),
            "StepIndivs": ParameterSpec("StepIndivs", 10, "integer", (0, 100)),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.max_indivs = 1000
        self.min_indivs = 10
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.shift_severity = 1.0
        self.swarm_number = 0
        self.hiber_step = 0.5
        self.first_update = 1
        self.off_peak = 3
        self.current_pop_size = 0
        self.previous_pop_size = 0
        self.iter_individuals_begin: list[float] = []
        self.iter_pop_size_end: list[int] = []
        self.adjust_iter = 0
        self.distance_accuracy = 0.2
        self.record_pop_info: list[int] = []
        self.avoidance_flag = 0
        self.pso_operator = PSOOperator()
        init_x = self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((int(self.parameters["initPopulationSize"]), self.dimension))
        )
        self.pop, self.remain_indivs = self.swarm_factory.generate_subpopulations(
            init_x,
            problem,
        )
        self.swarm_number = len(self.pop)
        self.map_pop_inds = [AMPMapRecord() for _ in range(max(self.swarm_number, 1))]
        self.hiber_area: list[AMPHiberArea] = []

    def iterate(self, problem) -> None:
        self.domain_size = float(np.sqrt(problem.dimension * (self.max_coordinate - self.min_coordinate) ** 2))
        for ii, swarm in list(enumerate(self.pop)):
            before_radius = swarm.current_radius
            if swarm.is_hibernated == 0:
                self._pso_move(swarm, problem)
                if problem.state.recent_change:
                    return
            swarm.center = update_center(swarm)
            swarm.current_radius = update_current_radius(swarm)
            if swarm.current_radius / (before_radius + 0.000001) < 0.9:
                swarm.stagnating_count = 0
            else:
                swarm.stagnating_count += 1
            self._brownian_or_cauchy_move(swarm, problem)
            if problem.state.recent_change:
                return
            if swarm.current_radius < float(self.parameters["ConvergenceLimit"]):
                swarm.is_hibernated = 1
                swarm.is_stagnated = 0
                if not self.hiber_area:
                    self.hiber_area.append(AMPHiberArea(optima=swarm.best_position.copy()))
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
            swarm.pbest_fitness = values.copy()
            swarm.pbest_positions = swarm.positions.copy()
            swarm.gbest_past_environment = swarm.best_position.copy()
            best_id = int(np.argmax(swarm.pbest_fitness))
            swarm.best_fitness = float(swarm.pbest_fitness[best_id])
            swarm.best_id = best_id
            swarm.best_position = swarm.pbest_positions[best_id, :].copy()
            swarm.velocities = -1 + 2 * self.rng.random((swarm.positions.shape[0], self.dimension))

    def _pso_move(self, swarm: AMPPSOSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        inertia = float(self.parameters["w"])
        self.pso_operator.update_and_move(
            swarm,
            inertia=inertia,
            cognitive=inertia * float(self.parameters["c1"]),
            social=inertia * float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(swarm, self.min_coordinate, self.max_coordinate)
        swarm.fitness = evaluate_(problem, swarm.positions, rows)
        if problem.state.recent_change:
            return
        for jj in range(rows):
            if swarm.fitness[jj] > swarm.pbest_fitness[jj]:
                swarm.pbest_fitness[jj] = swarm.fitness[jj]
                swarm.pbest_positions[jj, :] = swarm.positions[jj, :]
                best_id = int(np.argmax(swarm.pbest_fitness))
                best_value = float(swarm.pbest_fitness[best_id])
                if best_value > swarm.best_fitness:
                    swarm.best_fitness = best_value
                    swarm.best_position = swarm.pbest_positions[best_id, :].copy()
                    swarm.best_id = best_id

    def _brownian_or_cauchy_move(self, swarm: AMPPSOSwarm, problem) -> None:
        if swarm.positions.shape[0] < 2:
            return
        if swarm.is_stagnated == 0:
            tries = 2
            for _ in range(tries):
                randval = self.rng.standard_normal()
                order = np.argsort(-swarm.pbest_fitness)
                for gg in range(self.dimension):
                    newpos1 = swarm.pbest_positions[order[0], gg] + self.rng.standard_normal() * swarm.current_radius
                    swarm.positions[swarm.best_id, gg] = newpos1 + randval * (
                        swarm.pbest_positions[order[1], gg] - newpos1
                    )
                    if swarm.positions[swarm.best_id, gg] > self.max_coordinate:
                        swarm.positions[swarm.best_id, gg] = self.max_coordinate
                        swarm.velocities[swarm.best_id, gg] = 0
                    elif swarm.positions[swarm.best_id, gg] < self.min_coordinate:
                        swarm.positions[swarm.best_id, gg] = self.min_coordinate
                        swarm.velocities[swarm.best_id, gg] = 0
                result = problem.evaluate(swarm.positions[swarm.best_id, :])
                if result.size:
                    best_result = float(result[0])
                    swarm.fitness[swarm.best_id] = best_result
                    if best_result > swarm.pbest_fitness[swarm.best_id]:
                        swarm.pbest_fitness[swarm.best_id] = best_result
                        swarm.pbest_positions[swarm.best_id, :] = swarm.positions[swarm.best_id, :]
                        swarm.best_position = swarm.pbest_positions[swarm.best_id, :].copy()
                        swarm.best_fitness = best_result
                if problem.state.recent_change:
                    return
        else:
            randval = self.rng.standard_normal()
            order = np.argsort(-swarm.pbest_fitness)
            if order.size < 2:
                return
            candidate = swarm.pbest_positions[swarm.best_id, :].copy()
            for gg in range(self.dimension):
                newpos1 = swarm.pbest_positions[order[0], gg] + self.rng.standard_normal() * swarm.current_radius
                newpos2 = newpos1 + randval * (swarm.pbest_positions[order[1], gg] - newpos1)
                candidate[gg] = newpos2
                if candidate[gg] > self.max_coordinate:
                    candidate[gg] = self.max_coordinate
                    swarm.velocities[swarm.best_id, gg] = 0
                elif candidate[gg] < self.min_coordinate:
                    candidate[gg] = self.min_coordinate
                    swarm.velocities[swarm.best_id, gg] = 0
            result = evaluate_(problem, candidate, 1)
            if problem.state.recent_change:
                return
            if result.size and not np.isnan(result[0]):
                value = float(result[0])
                swarm.pbest_fitness[swarm.best_id] = value
                swarm.fitness[swarm.best_id] = value
                swarm.pbest_positions[swarm.best_id, :] = candidate
                swarm.positions[swarm.best_id, :] = candidate
                best_id = int(np.argmax(swarm.pbest_fitness))
                swarm.best_fitness = float(swarm.pbest_fitness[best_id])
                swarm.best_position = swarm.pbest_positions[best_id, :].copy()
                swarm.best_id = best_id
                swarm.stagnating_count = 0

    def _check_stagnant(self) -> None:
        active = [swarm.current_radius for swarm in self.pop if swarm.is_hibernated == 0]
        ave_radius = float(np.mean(active)) if active else 0.0
        diversity_limit = float(self.parameters["diversityFactor"]) * self.domain_size
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                continue
            if (
                (
                    swarm.stagnating_count > swarm.positions.shape[0]
                    and swarm.current_radius >= ave_radius
                    and swarm.current_radius >= diversity_limit
                )
                or swarm.stagnating_count > 10 * swarm.positions.shape[0]
            ):
                swarm.is_stagnated = 1
            else:
                swarm.is_stagnated = 0

    def _check_overlapping(self) -> None:
        idx = None
        while idx != -1:
            idx = -1
            for ii in range(self.swarm_number):
                if self.pop[ii].positions.shape[0] == 0:
                    continue
                for jj in range(ii + 1, self.swarm_number):
                    if self.pop[jj].positions.shape[0] == 0:
                        continue
                    dist = np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position)
                    if dist >= self.pop[ii].init_radius or dist >= self.pop[jj].init_radius:
                        continue
                    c1 = sum(
                        np.linalg.norm(self.pop[ii].center - row) < self.pop[ii].init_radius
                        for row in self.pop[jj].positions
                    )
                    c2 = sum(
                        np.linalg.norm(self.pop[jj].center - row) < self.pop[jj].init_radius
                        for row in self.pop[ii].positions
                    )
                    overlap_degree = float(self.parameters["OverlapDegree"])
                    required_jj = max(
                        1,
                        int(np.ceil(overlap_degree * self.pop[jj].positions.shape[0])),
                    )
                    required_ii = max(
                        1,
                        int(np.ceil(overlap_degree * self.pop[ii].positions.shape[0])),
                    )
                    if c1 >= required_jj and c2 >= required_ii:
                        if self.pop[ii].best_fitness > self.pop[jj].best_fitness:
                            self.pop[ii].is_hibernated = 0
                            self.pop[ii].stagnating_count = 0
                            self.pop[ii].is_stagnated = 0
                            self._delete_swarm(jj)
                            idx = jj
                        else:
                            self.pop[jj].is_hibernated = 0
                            self.pop[jj].stagnating_count = 0
                            self.pop[jj].is_stagnated = 0
                            self._delete_swarm(ii)
                            idx = ii
                        break
                if idx != -1:
                    break

    def _delete_swarm(self, index: int) -> None:
        del self.pop[index]
        self.swarm_number -= 1

    def _check_diversity(self, problem) -> None:
        active = [
            swarm.current_radius
            for swarm in self.pop
            if swarm.is_stagnated == 0 and swarm.is_hibernated == 0
        ]
        non_stagnated_radius = float(np.mean(active)) if active else 0.0
        if non_stagnated_radius >= float(self.parameters["diversityFactor"]) * self.domain_size:
            return
        self.record_pop_info.append(self.swarm_number)
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                swarm.is_hibernated = 0
        self.hiber_area = []
        self.iter_pop_size_end.append(self.swarm_number)
        if len(self.iter_pop_size_end) > 2:
            self.iter_pop_size_end.pop(0)
        if self.adjust_iter == 0:
            self.iter_individuals_begin = [sum(swarm.positions.shape[0] for swarm in self.pop)]
        self._update_memory()
        if self.first_update == 1:
            self.current_pop_size = self.iter_pop_size_end[-1]
            self.previous_pop_size = self.current_pop_size
            self.first_update = 0
        else:
            self.current_pop_size = self.iter_pop_size_end[-1]
            self.previous_pop_size = self.iter_pop_size_end[-2]
        ratio = abs(self.current_pop_size - self.previous_pop_size) / self.off_peak
        mean = self.map_pop_inds[self.current_pop_size - 1].mean
        std = self.map_pop_inds[self.current_pop_size - 1].std_dev
        if ratio >= 1:
            next_individuals = mean + std * self.rng.standard_normal() + int(self.parameters["StepIndivs"]) * (self.current_pop_size - self.previous_pop_size)
        elif ratio == 0:
            next_individuals = mean + std * self.rng.standard_normal()
        elif self.rng.random() <= ratio:
            next_individuals = mean + std * self.rng.standard_normal() + int(self.parameters["StepIndivs"]) * (self.current_pop_size - self.previous_pop_size)
        else:
            next_individuals = mean + std * self.rng.standard_normal()
        total_survive = sum(swarm.positions.shape[0] for swarm in self.pop)
        if next_individuals <= total_survive:
            next_individuals = total_survive + int(self.parameters["StepIndivs"]) * 2
        next_individuals = min(self.max_indivs, max(self.min_indivs, next_individuals))
        next_individuals = int(np.ceil(next_individuals))
        self.iter_individuals_begin = [next_individuals]
        num_remain = self.remain_indivs.shape[0]
        num_add = next_individuals - total_survive - num_remain
        if num_add > 0:
            add = self.min_coordinate + (
                (self.max_coordinate - self.min_coordinate) * self.rng.random((num_add, self.dimension))
            )
            add = np.vstack([add, self.remain_indivs]) if num_remain else add
            self.remain_indivs = np.empty((0, self.dimension))
            swarms, remain = self.swarm_factory.generate_subpopulations(add, problem)
            self.remain_indivs = remain
            if problem.state.recent_change:
                return
            self.pop.extend(swarms)
            self.swarm_number = len(self.pop)
        self.adjust_iter += 1

    def _update_memory(self) -> None:
        while self.swarm_number > len(self.map_pop_inds):
            self.map_pop_inds.append(AMPMapRecord())
        record = self.map_pop_inds[self.swarm_number - 1]
        record.num += 1
        value = float(self.iter_individuals_begin[-1])
        record.indivs.append(value)
        record.mean = ((record.mean * (record.num - 1)) + value) / record.num
        record.std_dev = float(np.sqrt(np.mean((np.asarray(record.indivs) - record.mean) ** 2)))
