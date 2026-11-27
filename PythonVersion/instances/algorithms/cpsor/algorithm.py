"""CPSOR algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import (
    evaluate_,
    has_complete_values,
    update_center,
    update_current_radius,
)
from instances.algorithms.cpsor.state import CPSORSwarm
from utility.clustering import CompleteLinkageClusterer


class CPSORSwarmFactory:
    """Build CPSOR sub-swarms from sampled positions."""

    def __init__(self, algorithm: "CPSORAlgorithm") -> None:
        self.algorithm = algorithm

    def __getattr__(self, name: str):
        return getattr(self.algorithm, name)

    def generate_subpopulations(
        self,
        positions: np.ndarray,
        problem,
    ) -> list[CPSORSwarm]:
        positions = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        groups, _remain = CompleteLinkageClusterer(
            max_size=int(self.parameters["maxSubsize"]),
        ).groups_and_remain(positions)
        swarms: list[CPSORSwarm] = []

        for group in groups:
            if group.shape[0] == 0:
                continue
            values = evaluate_(problem, group, group.shape[0])
            if problem.state.recent_change:
                values = np.full(group.shape[0], -np.inf, dtype=float)
            elif problem.state.terminated and not has_complete_values(
                values,
                group.shape[0],
            ):
                return swarms
            if (
                not problem.state.recent_change
                and not has_complete_values(values, group.shape[0])
            ):
                continue

            center = np.mean(group, axis=0)
            init_radius = float(np.mean(np.linalg.norm(group - center, axis=1)))
            span = self.max_coordinate - self.min_coordinate
            velocities = -(span / 4.0) + (2.0 * span / 4.0) * self.rng.random(
                group.shape
            )
            pbest_fitness = self._finite_fitness(values)
            best_id = self._best_index(pbest_fitness)
            swarms.append(
                CPSORSwarm(
                    positions=group.copy(),
                    gbest_past_environment=np.full(self.dimension, np.nan),
                    velocities=velocities,
                    shifts=[],
                    fitness=values.copy(),
                    pbest_positions=group.copy(),
                    is_converged=0,
                    pbest_fitness=pbest_fitness,
                    best_fitness=float(pbest_fitness[best_id]),
                    best_position=group[best_id, :].copy(),
                    best_id=best_id,
                    center=center,
                    init_radius=init_radius,
                    current_radius=init_radius,
                )
            )
            if problem.state.recent_change or problem.state.terminated:
                return swarms
        return swarms


@algorithm_registry.register("CPSOR")
class CPSORAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "CPSOR"

    def __init__(self) -> None:
        self.swarm_factory = CPSORSwarmFactory(self)

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "initPopulationSize": ParameterSpec(
                "initPopulationSize",
                100,
                "integer",
                (0, 1000),
                "Total number of particles at initialization.",
            ),
            "maxSubsize": ParameterSpec(
                "maxSubsize",
                5,
                "integer",
                (0, 100),
                "Maximum number of particles retained in a sub-swarm.",
            ),
            "w": ParameterSpec(
                "w",
                0.6,
                "numeric",
                (0, 2),
                "Inertia weight applied to the PSO update.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.83,
                "numeric",
                (0, 5),
                "Cognitive coefficient scaling attraction to personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.83,
                "numeric",
                (0, 5),
                "Social coefficient scaling attraction to global best.",
            ),
            "ConvergenceLimit": ParameterSpec(
                "ConvergenceLimit",
                0.01,
                "numeric",
                (0, 1),
                "Sub-swarm radius below which it is considered converged.",
            ),
            "OverlapDegree": ParameterSpec(
                "OverlapDegree",
                0.7,
                "numeric",
                (0, 1),
                "Fraction of overlapping particles required to merge sub-swarms.",
            ),
            "DiversityDegree": ParameterSpec(
                "DiversityDegree",
                0.3,
                "numeric",
                (0, 1),
                "Minimum survivor ratio before diversity reinitialization.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.swarm_number = 0
        self.pso_operator = PSOOperator()

        init_size = max(1, int(self.parameters["initPopulationSize"]))
        init_x = self.bounds.sample(init_size, self.rng)
        self.pop = self.swarm_factory.generate_subpopulations(init_x, problem)
        self.swarm_number = len(self.pop)
        if problem.state.recent_change or problem.state.terminated:
            return

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        for swarm in list(self.pop):
            if swarm.is_converged == 0:
                self._pso_move(swarm, problem)
                if problem.state.recent_change or problem.state.terminated:
                    return

        self._refresh_centers()
        self._check_overlapping()
        self._check_overcrowding()
        self._refresh_radii()
        self._check_diversity(problem)

    def on_environment_change(self, problem) -> None:
        for swarm in self.pop:
            rows = swarm.pbest_positions.shape[0]
            values = evaluate_(problem, swarm.pbest_positions, rows)
            if problem.state.recent_change:
                return
            if problem.state.terminated and not has_complete_values(values, rows):
                return
            if not has_complete_values(values, rows):
                return
            swarm.pbest_fitness = self._finite_fitness(values)
            swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)
            swarm.velocities = -2.0 + 4.0 * self.rng.random(swarm.positions.shape)
            if problem.state.terminated:
                return

    def _pso_move(self, swarm: CPSORSwarm, problem) -> None:
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
        if problem.state.terminated and not has_complete_values(values, rows):
            return
        if not has_complete_values(values, rows):
            return

        swarm.fitness = values.copy()
        comparable = self._finite_fitness(values)
        for index in range(rows):
            if comparable[index] <= swarm.pbest_fitness[index]:
                continue
            swarm.pbest_fitness[index] = comparable[index]
            swarm.pbest_positions[index, :] = swarm.positions[index, :]
            self._reset_best_if_improved(swarm, index)
            self._local_search(swarm, index, problem)
            if problem.state.recent_change or problem.state.terminated:
                return

    def _local_search(self, swarm: CPSORSwarm, index: int, problem) -> None:
        for dim in range(self.dimension):
            candidate = swarm.best_position.copy()
            candidate[dim] = swarm.positions[index, dim]
            value = evaluate_(problem, candidate, 1)
            if problem.state.recent_change:
                return
            if problem.state.terminated and not has_complete_values(value, 1):
                return
            if not has_complete_values(value, 1):
                continue
            result = float(value[0])
            if result <= float(swarm.best_fitness):
                if problem.state.terminated:
                    return
                continue
            best_id = int(swarm.best_id)
            swarm.positions[best_id, :] = candidate
            swarm.pbest_positions[best_id, :] = candidate
            swarm.fitness[best_id] = result
            swarm.pbest_fitness[best_id] = result
            swarm.best_position = candidate.copy()
            swarm.best_fitness = result
            if problem.state.terminated:
                return

    def _refresh_centers(self) -> None:
        for swarm in self.pop:
            if swarm.is_converged == 1:
                continue
            if swarm.positions.shape[0] == 0:
                swarm.center = np.zeros(self.dimension)
            else:
                swarm.center = update_center(swarm)

    def _check_overlapping(self) -> None:
        idx = None
        while idx != -1:
            idx = -1
            for left in range(self.swarm_number):
                if (
                    self.pop[left].positions.shape[0] == 0
                    or self.pop[left].is_converged == 1
                ):
                    continue
                for right in range(left + 1, self.swarm_number):
                    if (
                        self.pop[right].positions.shape[0] == 0
                        or self.pop[right].is_converged == 1
                    ):
                        continue
                    distance = float(
                        np.linalg.norm(
                            self.pop[left].best_position - self.pop[right].best_position
                        )
                    )
                    if (
                        distance >= self.pop[left].init_radius
                        or distance >= self.pop[right].init_radius
                    ):
                        continue
                    overlap_left = sum(
                        np.linalg.norm(self.pop[left].center - row)
                        < self.pop[left].init_radius
                        for row in self.pop[right].positions
                    )
                    overlap_right = sum(
                        np.linalg.norm(self.pop[right].center - row)
                        < self.pop[right].init_radius
                        for row in self.pop[left].positions
                    )
                    left_threshold = float(self.parameters["OverlapDegree"]) * self.pop[
                        right
                    ].positions.shape[0]
                    right_threshold = float(self.parameters["OverlapDegree"]) * self.pop[
                        left
                    ].positions.shape[0]
                    if overlap_left <= left_threshold or overlap_right <= right_threshold:
                        continue
                    if self.pop[left].best_fitness > self.pop[right].best_fitness:
                        self._merge_swarms(left, right)
                        idx = right
                    else:
                        self._merge_swarms(right, left)
                        idx = left
                    break
                if idx != -1:
                    break

    def _merge_swarms(self, keep: int, remove: int) -> None:
        kept = self.pop[keep]
        removed = self.pop[remove]
        kept.positions = np.vstack([kept.positions, removed.positions])
        kept.pbest_positions = np.vstack([kept.pbest_positions, removed.pbest_positions])
        kept.velocities = np.vstack([kept.velocities, removed.velocities])
        kept.fitness = np.concatenate([kept.fitness, removed.fitness])
        kept.pbest_fitness = np.concatenate([kept.pbest_fitness, removed.pbest_fitness])
        kept.init_radius = (kept.init_radius + removed.init_radius) / 2.0
        kept.is_converged = 0
        self._reset_best(kept)
        del self.pop[remove]
        self.swarm_number = len(self.pop)

    def _check_overcrowding(self) -> None:
        max_subsize = int(self.parameters["maxSubsize"])
        if max_subsize < 1:
            return
        for swarm in self.pop:
            if swarm.positions.shape[0] <= max_subsize:
                continue
            order = np.argsort(np.nan_to_num(swarm.pbest_fitness, nan=-np.inf))[::-1]
            keep = order[:max_subsize]
            swarm.positions = swarm.positions[keep, :]
            swarm.velocities = swarm.velocities[keep, :]
            swarm.fitness = swarm.fitness[keep]
            swarm.pbest_positions = swarm.pbest_positions[keep, :]
            swarm.pbest_fitness = swarm.pbest_fitness[keep]
            swarm.center = np.mean(swarm.pbest_positions, axis=0)
            self._reset_best(swarm)

    def _refresh_radii(self) -> None:
        for swarm in self.pop:
            if swarm.is_converged == 1:
                continue
            if swarm.positions.shape[0] == 0:
                swarm.current_radius = 0.0
                continue
            swarm.current_radius = update_current_radius(swarm)

    def _check_diversity(self, problem) -> None:
        if not self.pop:
            self.swarm_number = 0
            return
        best_id = self._best_swarm_index()
        any_converged = 0
        for index, swarm in enumerate(self.pop):
            if (
                swarm.current_radius < float(self.parameters["ConvergenceLimit"])
                and index != best_id
            ):
                swarm.is_converged = 1
                any_converged += 1

        survived_particles = sum(
            swarm.positions.shape[0] for swarm in self.pop if not swarm.is_converged
        )
        init_size = max(1, int(self.parameters["initPopulationSize"]))
        diversity_limit = init_size * float(self.parameters["DiversityDegree"])
        if survived_particles >= diversity_limit:
            return

        save_best_positions = [self.pop[best_id].best_position.copy()]
        del self.pop[best_id]
        self.swarm_number = len(self.pop)

        while any_converged:
            for index, swarm in enumerate(self.pop):
                if swarm.is_converged:
                    save_best_positions.append(swarm.best_position.copy())
                    del self.pop[index]
                    self.swarm_number = len(self.pop)
                    break
            any_converged -= 1

        count = len(save_best_positions)
        add_count = max(0, init_size - survived_particles - count)
        random_positions = self.bounds.sample(add_count, self.rng)
        saved = np.vstack(save_best_positions).reshape(-1, self.dimension)
        add_positions = (
            np.vstack([random_positions, saved]) if random_positions.size else saved
        )
        swarms = self.swarm_factory.generate_subpopulations(add_positions, problem)
        if problem.state.recent_change:
            return

        for swarm in self.pop:
            if swarm.current_radius < 10.0 * float(self.parameters["ConvergenceLimit"]):
                swarm.velocities = (
                    -swarm.init_radius
                    + 2.0 * swarm.init_radius * self.rng.random(swarm.positions.shape)
                )
        self.pop = [*self.pop, *swarms]
        self.swarm_number = len(self.pop)

    def _best_swarm_index(self) -> int:
        values = [float(swarm.best_fitness) for swarm in self.pop]
        return int(np.argmax(np.asarray(values, dtype=float)))

    def _reset_best_if_improved(self, swarm: CPSORSwarm, index: int) -> None:
        value = float(swarm.pbest_fitness[index])
        if value > float(swarm.best_fitness):
            swarm.best_id = index
            swarm.best_fitness = value
            swarm.best_position = swarm.pbest_positions[index, :].copy()

    def _reset_best(self, swarm: CPSORSwarm) -> None:
        if swarm.pbest_fitness.size == 0:
            return
        best_id = self._best_index(swarm.pbest_fitness)
        swarm.best_id = best_id
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    @staticmethod
    def _finite_fitness(values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=float)
        return np.where(np.isnan(raw), -np.inf, raw).copy()

    @staticmethod
    def _best_index(values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        raw = np.asarray(values, dtype=float)
        comparable = np.where(np.isnan(raw), -np.inf, raw)
        return int(np.argmax(comparable))
