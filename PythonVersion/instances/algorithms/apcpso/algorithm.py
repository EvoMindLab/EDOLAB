"""APCPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.apcpso.state import APCPSOSwarm
from instances.algorithms.common import evaluate_
from utility.clustering import AffinityPropagationClusterer


@algorithm_registry.register("APCPSO")
class APCPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "APCPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                100,
                "integer",
                (0, 1000),
                "Total number of individuals at initialization.",
            ),
            "omega_max": ParameterSpec(
                "omega_max",
                0.6,
                "numeric",
                (0, 2),
                "Maximum inertia weight in velocity update.",
            ),
            "omega_min": ParameterSpec(
                "omega_min",
                0.3,
                "numeric",
                (0, 2),
                "Minimum inertia weight in velocity update.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Acceleration coefficient for cognitive component.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Acceleration coefficient for social component.",
            ),
            "MaxSubPopIterations": ParameterSpec(
                "MaxSubPopIterations",
                50,
                "integer",
                (1, 1000),
                "Maximum local iterations per sub-population.",
            ),
            "ConvergenceThreshold": ParameterSpec(
                "ConvergenceThreshold",
                0.1,
                "numeric",
                (0, 1),
                "Threshold for sub-swarm convergence radius.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.swarm_number = 0
        self.pop: list[APCPSOSwarm] = []
        self.pso_operator = PSOOperator()
        positions = self._random_positions(self.population_size)
        self.pop = self._build_swarms(positions, problem)
        self.swarm_number = len(self.pop)
        self.exclusion_radius = self._exclusion_radius()

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        for swarm in self.pop:
            self._move_swarm(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return

        self._apply_exclusion(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._update_convergence()
        if not self.pop:
            return
        if all(swarm.is_converged == 1 for swarm in self.pop):
            worst_idx = int(np.argmin([float(swarm.best_fitness) for swarm in self.pop]))
            replacement = self._reset_swarm_shape(self.pop[worst_idx], problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            self.pop[worst_idx] = replacement

    def on_environment_change(self, problem) -> None:
        if not self.pop:
            return
        self.pop.sort(key=lambda swarm: float(swarm.best_fitness), reverse=True)

        gbest_values = np.full(len(self.pop), -np.inf, dtype=float)
        for idx, swarm in enumerate(self.pop):
            values = evaluate_(problem, swarm.best_position, 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if values.size and not np.isnan(values[0]):
                swarm.best_fitness = float(values[0])
                gbest_values[idx] = float(values[0])

        order = np.argsort(gbest_values)[::-1]
        self.pop = [self.pop[int(idx)] for idx in order]
        self.swarm_number = len(self.pop)

        new_positions = np.zeros((self.population_size, self.dimension), dtype=float)
        keep_count = min(self.swarm_number, self.population_size)
        for idx in range(keep_count):
            best_position, best_value = self._calibrate_best(self.pop[idx], problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            new_positions[idx, :] = best_position
            self.pop[idx].best_position = best_position.copy()
            self.pop[idx].best_fitness = best_value

        if keep_count < self.population_size:
            new_positions[keep_count:, :] = self._random_positions(
                self.population_size - keep_count
            )

        new_pop = self._build_swarms(new_positions, problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        self.pop = new_pop
        self.swarm_number = len(self.pop)
        self.exclusion_radius = self._exclusion_radius()

    def _move_swarm(self, swarm: APCPSOSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        swarm.local_iter += 1
        max_iter = max(1, int(self.parameters["MaxSubPopIterations"]))
        omega = float(self.parameters["omega_max"]) - (
            float(self.parameters["omega_max"]) - float(self.parameters["omega_min"])
        ) * (swarm.local_iter / max_iter)
        self.pso_operator.update_and_move(
            swarm,
            inertia=omega,
            cognitive=float(self.parameters["c1"]),
            social=float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(swarm, self.min_coordinate, self.max_coordinate)
        values = evaluate_(problem, swarm.positions, rows)
        if problem.state.recent_change or problem.state.terminated:
            return
        if values.size != rows or np.all(np.isnan(values)):
            return
        swarm.fitness = values
        improved = swarm.fitness > swarm.pbest_fitness
        swarm.pbest_fitness[improved] = swarm.fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        self._reset_best(swarm)

    def _apply_exclusion(self, problem) -> None:
        rex = self._exclusion_radius()
        for ii in range(max(0, self.swarm_number - 1)):
            for jj in range(ii + 1, self.swarm_number):
                if self.pop[ii].is_excluded == 1 or self.pop[jj].is_excluded == 1:
                    continue
                distance = float(
                    np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position)
                )
                if distance >= rex:
                    continue
                if self.pop[ii].best_fitness > self.pop[jj].best_fitness:
                    self.pop[ii] = self._merge_subpopulations(self.pop[ii], self.pop[jj])
                    self.pop[jj].is_excluded = 1
                else:
                    self.pop[jj] = self._merge_subpopulations(self.pop[ii], self.pop[jj])
                    self.pop[ii].is_excluded = 1

        for idx, swarm in enumerate(list(self.pop)):
            if swarm.is_excluded == 1:
                replacement = self._reset_swarm_shape(swarm, problem)
                if problem.state.recent_change or problem.state.terminated:
                    return
                self.pop[idx] = replacement

    def _update_convergence(self) -> None:
        for swarm in self.pop:
            if swarm.positions.shape[0] == 0:
                swarm.is_converged = 1
                swarm.center = np.zeros(self.dimension)
                swarm.current_radius = 0.0
                continue
            swarm.center = np.mean(swarm.pbest_positions, axis=0)
            swarm.current_radius = float(
                np.mean(np.linalg.norm(swarm.pbest_positions - swarm.center, axis=1))
            )
            swarm.is_converged = int(
                swarm.current_radius < float(self.parameters["ConvergenceThreshold"])
            )

    def _calibrate_best(
        self,
        swarm: APCPSOSwarm,
        problem,
    ) -> tuple[np.ndarray, float]:
        try_position = swarm.best_position.copy()
        try_value = float(swarm.best_fitness)
        for dim in range(self.dimension):
            candidate = swarm.best_position.copy()
            candidate[dim] = min(self.max_coordinate, candidate[dim] + self.rng.random())
            values = evaluate_(problem, candidate, 1)
            if problem.state.recent_change or problem.state.terminated:
                return try_position, try_value
            if values.size and not np.isnan(values[0]) and float(values[0]) > try_value:
                try_position = candidate
                try_value = float(values[0])
                continue

            candidate = swarm.best_position.copy()
            candidate[dim] = max(self.min_coordinate, candidate[dim] - self.rng.random())
            values = evaluate_(problem, candidate, 1)
            if problem.state.recent_change or problem.state.terminated:
                return try_position, try_value
            if values.size and not np.isnan(values[0]) and float(values[0]) > try_value:
                try_position = candidate
                try_value = float(values[0])
        return try_position, try_value

    def _build_swarms(self, positions: np.ndarray, problem) -> list[APCPSOSwarm]:
        population = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        groups, _remain = AffinityPropagationClusterer().groups_and_remain(population)
        swarms: list[APCPSOSwarm] = []
        for group in groups:
            if group.shape[0] == 0:
                continue
            swarm = self._new_swarm_from_positions(group, problem)
            swarms.append(swarm)
            if problem.state.recent_change or problem.state.terminated:
                return swarms
        if not swarms and not problem.state.recent_change and not problem.state.terminated:
            fallback = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
            if fallback.shape[0] == 0:
                fallback = self._random_positions(1)
            swarms.append(self._new_swarm_from_positions(fallback, problem))
        return swarms

    def _new_swarm_from_positions(
        self,
        positions: np.ndarray,
        problem,
        *,
        local_iter: int = 0,
    ) -> APCPSOSwarm:
        rows = positions.shape[0]
        fitness = evaluate_(problem, positions, rows)
        if problem.state.recent_change:
            fitness = np.full(rows, -np.inf, dtype=float)
        else:
            fitness = self._padded_fitness(fitness, rows)
        pbest_fitness = fitness.copy()
        best_id = self._safe_best_id(pbest_fitness)
        center = np.mean(positions, axis=0)
        current_radius = float(np.mean(np.linalg.norm(positions - center, axis=1)))
        return APCPSOSwarm(
            positions=positions.copy(),
            velocities=self._random_velocities(positions.shape),
            shifts=[],
            fitness=fitness,
            pbest_positions=positions.copy(),
            is_converged=0,
            is_excluded=0,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=positions[best_id, :].copy(),
            best_id=best_id,
            center=center,
            init_radius=current_radius,
            current_radius=current_radius,
            local_iter=local_iter,
        )

    def _reset_swarm_shape(self, swarm: APCPSOSwarm, problem) -> APCPSOSwarm:
        rows = max(1, swarm.positions.shape[0])
        return self._new_swarm_from_positions(
            self._random_positions(rows),
            problem,
            local_iter=swarm.local_iter,
        )

    def _merge_subpopulations(
        self,
        pop1: APCPSOSwarm,
        pop2: APCPSOSwarm,
    ) -> APCPSOSwarm:
        positions = np.vstack([pop1.positions, pop2.positions])
        fitness = np.concatenate([pop1.fitness, pop2.fitness])
        pbest_positions = np.vstack([pop1.pbest_positions, pop2.pbest_positions])
        pbest_fitness = np.concatenate([pop1.pbest_fitness, pop2.pbest_fitness])
        velocities = np.vstack([pop1.velocities, pop2.velocities])
        keep = max(1, positions.shape[0] - pop2.positions.shape[0])
        order = np.argsort(np.nan_to_num(fitness, nan=-np.inf))[::-1][:keep]
        positions = positions[order, :]
        fitness = fitness[order]
        pbest_positions = pbest_positions[order, :]
        pbest_fitness = pbest_fitness[order]
        velocities = velocities[order, :]
        best_id = self._safe_best_id(pbest_fitness)
        center = np.mean(pbest_positions, axis=0)
        current_radius = float(np.mean(np.linalg.norm(pbest_positions - center, axis=1)))
        return APCPSOSwarm(
            positions=positions.copy(),
            velocities=velocities.copy(),
            shifts=[],
            fitness=fitness.copy(),
            pbest_positions=pbest_positions.copy(),
            is_converged=0,
            is_excluded=0,
            pbest_fitness=pbest_fitness.copy(),
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
            best_id=best_id,
            center=center,
            init_radius=current_radius,
            current_radius=current_radius,
            local_iter=max(pop1.local_iter, pop2.local_iter),
        )

    def _reset_best(self, swarm: APCPSOSwarm) -> None:
        best_id = self._safe_best_id(swarm.pbest_fitness)
        swarm.best_id = best_id
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _safe_best_id(self, values: np.ndarray) -> int:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 0
        return int(np.argmax(np.nan_to_num(values, nan=-np.inf)))

    def _padded_fitness(self, values: np.ndarray, expected: int) -> np.ndarray:
        fitness = np.full(expected, -np.inf, dtype=float)
        count = min(expected, values.size)
        if count:
            fitness[:count] = np.nan_to_num(values[:count], nan=-np.inf)
        return fitness

    def _random_positions(self, rows: int) -> np.ndarray:
        return self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((rows, self.dimension))
        )

    def _random_velocities(self, shape: tuple[int, int]) -> np.ndarray:
        return -5.0 + 10.0 * self.rng.random(shape)

    def _exclusion_radius(self) -> float:
        return float(
            (self.max_coordinate - self.min_coordinate)
            / (2.0 * (max(1, self.swarm_number) ** (1.0 / self.dimension)))
        )
