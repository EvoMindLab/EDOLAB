"""AMPPSO_BC algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.amppso_bc.components.clustering import AMPPSOBCClusterer
from instances.algorithms.amppso_bc.state import (
    AMPBCHiberArea,
    AMPBCMapRecord,
    AMPPSOBCSwarm,
)
from instances.algorithms.common import (
    evaluate_,
    stack_rows,
    update_center,
    update_current_radius,
)


class AMPPSOBCSwarmFactory:
    """Build AMPPSO_BC sub-swarms from sampled positions."""

    def __init__(self, algorithm: "AMPPSOBCAlgorithm") -> None:
        self.algorithm = algorithm

    def __getattr__(self, name: str):
        return getattr(self.algorithm, name)

    def generate_subpopulations(
        self,
        positions: np.ndarray,
        problem,
    ) -> tuple[list[AMPPSOBCSwarm], np.ndarray]:
        positions = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        groups, remain = AMPPSOBCClusterer().groups_and_remain(positions)
        swarms: list[AMPPSOBCSwarm] = []
        remain_rows = [row for row in remain]

        for group in groups:
            if group.shape[0] == 0:
                continue
            values = evaluate_(problem, group, group.shape[0])
            if problem.state.recent_change:
                values = np.full(group.shape[0], -np.inf, dtype=float)
            elif problem.state.terminated and (
                values.size != group.shape[0] or np.all(np.isnan(values))
            ):
                return swarms, stack_rows(remain_rows, self.dimension)

            if not problem.state.recent_change and (
                values.size != group.shape[0] or np.all(np.isnan(values))
            ):
                remain_rows.extend(group)
                continue

            center = np.mean(group, axis=0)
            init_radius = float(np.mean(np.linalg.norm(group - center, axis=1)))
            pbest_value = values.copy()
            best_id = int(np.argmax(np.nan_to_num(pbest_value, nan=-np.inf)))
            swarms.append(
                AMPPSOBCSwarm(
                    positions=group.copy(),
                    gbest_past_environment=np.full(self.dimension, np.nan),
                    velocities=np.zeros_like(group),
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
            if problem.state.recent_change or problem.state.terminated:
                return swarms, stack_rows(remain_rows, self.dimension)
        return swarms, stack_rows(remain_rows, self.dimension)


@algorithm_registry.register("AMPPSO_BC")
class AMPPSOBCAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "AMPPSO_BC"

    def __init__(self) -> None:
        self.swarm_factory = AMPPSOBCSwarmFactory(self)

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
            "w": ParameterSpec(
                "w",
                0.7298,
                "numeric",
                (0, 2),
                "Inertia weight balancing exploration and exploitation.",
            ),
            "c1": ParameterSpec(
                "c1",
                1.496,
                "numeric",
                (0, 5),
                "Cognitive acceleration coefficient scaling influence of personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                1.496,
                "numeric",
                (0, 5),
                "Social acceleration coefficient scaling influence of global best.",
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
        self.shift_severity = float(problem.parameters.get("ShiftSeverity", 1.0))
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
        self.domain_size = self._domain_size()
        self.pso_operator = PSOOperator()

        init_size = max(0, int(self.parameters["initPopulationSize"]))
        init_x = self.bounds.sample(init_size, self.rng)
        self.pop, self.remain_indivs = self.swarm_factory.generate_subpopulations(
            init_x,
            problem,
        )
        self.swarm_number = len(self.pop)
        self.map_pop_inds = [AMPBCMapRecord() for _ in range(max(self.swarm_number, 1))]
        self.hiber_area: list[AMPBCHiberArea] = []

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        self.domain_size = self._domain_size()
        for swarm in list(self.pop):
            before_radius = swarm.current_radius
            if swarm.is_hibernated == 0:
                self._pso_move(swarm, problem)
                if problem.state.recent_change or problem.state.terminated:
                    return
            self._refresh_radius(swarm)
            if swarm.current_radius / (before_radius + 0.000001) < 0.9:
                swarm.stagnating_count = 0
            else:
                swarm.stagnating_count += 1

            self._brownian_or_cauchy_move(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return

            if swarm.current_radius < float(self.parameters["ConvergenceLimit"]):
                swarm.is_hibernated = 1
                swarm.is_stagnated = 0
                if not self.hiber_area:
                    self.hiber_area.append(AMPBCHiberArea(optima=swarm.best_position.copy()))
            if swarm.is_hibernated == 1:
                swarm.init_radius = self.distance_accuracy

        self._check_stagnant()
        self._check_overlapping()
        self._check_diversity(problem)

    def on_environment_change(self, problem) -> None:
        velocity_range = self._environment_velocity_range(problem)
        for swarm in self.pop:
            swarm.gbest_past_environment = swarm.best_position.copy()
            values = evaluate_(problem, swarm.positions, swarm.positions.shape[0])
            if problem.state.recent_change:
                return
            if values.size != swarm.positions.shape[0] or np.all(np.isnan(values)):
                return
            swarm.fitness = values
            swarm.pbest_fitness = values.copy()
            swarm.pbest_positions = swarm.positions.copy()
            self._reset_best(swarm)
            swarm.velocities = -velocity_range + (
                2.0 * velocity_range * self.rng.random((swarm.positions.shape[0], self.dimension))
            )

    def _pso_move(self, swarm: AMPPSOBCSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        inertia = float(self.parameters["w"])
        self.pso_operator.update_and_move(
            swarm,
            inertia=inertia,
            cognitive=inertia * float(self.parameters["c1"]),
            social=inertia * float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(swarm, self.min_coordinate, self.max_coordinate)
        values = evaluate_(problem, swarm.positions, rows)
        if problem.state.recent_change:
            return
        if values.size != rows:
            swarm.fitness[: values.size] = values
            return
        swarm.fitness = values
        self._update_personal_bests(swarm)

    def _brownian_or_cauchy_move(self, swarm: AMPPSOBCSwarm, problem) -> None:
        if swarm.positions.shape[0] == 0:
            return
        best_id = int(swarm.best_id)
        if swarm.is_stagnated == 0:
            original_position = swarm.positions[best_id, :].copy()
            original_fitness = float(swarm.fitness[best_id])
            for attempt in range(1, 3):
                search_radius = swarm.current_radius * (1.0 + 0.1 * attempt)
                candidate = swarm.pbest_positions[best_id, :].copy()
                candidate += self.rng.standard_normal(self.dimension) * search_radius
                candidate = self._clip_candidate_and_velocity(candidate, swarm, best_id)
                value = evaluate_(problem, candidate, 1)
                if problem.state.recent_change:
                    return
                if not value.size or np.isnan(value[0]):
                    if attempt < 2:
                        swarm.positions[best_id, :] = original_position
                        swarm.fitness[best_id] = original_fitness
                    continue
                best_result = float(value[0])
                swarm.positions[best_id, :] = candidate
                swarm.fitness[best_id] = best_result
                if best_result > swarm.pbest_fitness[best_id]:
                    swarm.pbest_fitness[best_id] = best_result
                    swarm.pbest_positions[best_id, :] = candidate
                    swarm.best_position = candidate.copy()
                    swarm.best_fitness = best_result
                    break
                if attempt < 2:
                    swarm.positions[best_id, :] = original_position
                    swarm.fitness[best_id] = original_fitness
        else:
            scale = self._cauchy_scale(problem)
            candidate = swarm.pbest_positions[best_id, :].copy()
            candidate += scale * self.rng.standard_cauchy(self.dimension)
            candidate = self._clip_candidate_and_velocity(candidate, swarm, best_id)
            value = evaluate_(problem, candidate, 1)
            if problem.state.recent_change:
                return
            if not value.size or np.isnan(value[0]):
                return
            swarm.pbest_positions[best_id, :] = candidate
            swarm.positions[best_id, :] = candidate
            swarm.pbest_fitness[best_id] = float(value[0])
            swarm.fitness[best_id] = float(value[0])
            self._reset_best(swarm)
            swarm.stagnating_count = 0

    def _clip_candidate_and_velocity(
        self,
        candidate: np.ndarray,
        swarm: AMPPSOBCSwarm,
        index: int,
    ) -> np.ndarray:
        clipped = np.clip(candidate, self.min_coordinate, self.max_coordinate)
        if swarm.velocities is not None:
            mask = clipped != candidate
            if np.any(mask):
                swarm.velocities[index, mask] = 0.0
        return clipped

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
                    distance = float(
                        np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position)
                    )
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

        self.record_pop_info.append(self.swarm_number)
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                swarm.is_hibernated = 0
        self.hiber_area = []

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
        next_individuals = record.mean + record.std_dev * self.rng.standard_normal()
        if ratio >= 1 or (0 < ratio and self.rng.random() <= ratio):
            next_individuals += int(self.parameters["StepIndivs"]) * (
                self.current_pop_size - self.previous_pop_size
            )

        total_survive = sum(swarm.positions.shape[0] for swarm in self.pop)
        if next_individuals <= total_survive:
            next_individuals = total_survive + int(self.parameters["StepIndivs"]) * 2
        next_individuals = int(
            np.ceil(min(self.max_indivs, max(self.min_indivs, next_individuals)))
        )
        self.iter_individuals_begin = [float(next_individuals)]

        num_remain = self.remain_indivs.shape[0]
        num_add = next_individuals - total_survive - num_remain
        if num_add > 0:
            add = self.bounds.sample(num_add, self.rng)
            add = np.vstack([add, self.remain_indivs]) if num_remain else add
            self._add_generated_swarms(add, problem)
        elif self.swarm_number == 0 and num_remain > 0:
            add = np.vstack(
                [self.bounds.sample(self.min_indivs, self.rng), self.remain_indivs]
            )
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

    def _map_record_for_count(self, swarm_count: int) -> AMPBCMapRecord:
        index = max(1, int(swarm_count)) - 1
        while index >= len(self.map_pop_inds):
            self.map_pop_inds.append(AMPBCMapRecord())
        return self.map_pop_inds[index]

    def _activate_swarm(self, index: int) -> None:
        self.pop[index].is_hibernated = 0
        self.pop[index].stagnating_count = 0
        self.pop[index].is_stagnated = 0

    def _delete_swarm(self, index: int) -> None:
        del self.pop[index]
        self.swarm_number = len(self.pop)

    def _refresh_radius(self, swarm: AMPPSOBCSwarm) -> None:
        if swarm.positions.shape[0] == 0:
            swarm.center = np.zeros(self.dimension)
            swarm.current_radius = 0.0
            return
        swarm.center = update_center(swarm)
        swarm.current_radius = update_current_radius(swarm)

    def _update_personal_bests(self, swarm: AMPPSOBCSwarm) -> None:
        improved = swarm.fitness > swarm.pbest_fitness
        if np.any(improved):
            swarm.pbest_fitness[improved] = swarm.fitness[improved]
            swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        self._reset_best(swarm)

    def _reset_best(self, swarm: AMPPSOBCSwarm) -> None:
        fitness = np.asarray(swarm.pbest_fitness, dtype=float)
        if fitness.size == 0:
            return
        best_id = int(np.argmax(np.nan_to_num(fitness, nan=-np.inf)))
        swarm.best_id = best_id
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()
        swarm.best_fitness = float(fitness[best_id])

    def _domain_size(self) -> float:
        return float(np.linalg.norm(self.max_coordinate - self.min_coordinate))

    def _public_environment_number(self, problem) -> int:
        return int(problem.state.environment_index) + 1

    def _cauchy_scale(self, problem) -> np.ndarray:
        ratio = self._public_environment_number(problem) / 20.0
        scale_factor = max(0.1, min(0.5, ratio))
        return scale_factor * (self.max_coordinate - self.min_coordinate)

    def _environment_velocity_range(self, problem) -> float:
        ratio = self._public_environment_number(problem) / 10.0
        return float(max(0.5, min(2.0, ratio)))
