"""DPCPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.dpcpso.state import DPCPSOSwarm
from utility.clustering import DensityPeakClusterer


@algorithm_registry.register("DPCPSO")
class DPCPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "DPCPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                100,
                "integer",
                (1, 1000),
                "Total number of individuals at initialization.",
            ),
            "omega_max": ParameterSpec(
                "omega_max",
                0.5,
                "numeric",
                (0, 2),
                "Upper bound on inertia weight for velocity update.",
            ),
            "omega_min": ParameterSpec(
                "omega_min",
                0.2,
                "numeric",
                (0, 2),
                "Lower bound on inertia weight for velocity update.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Scaling factor for attraction toward each particle's personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Scaling factor for attraction toward the sub-swarm's global best.",
            ),
            "MaxSubPopIterations": ParameterSpec(
                "MaxSubPopIterations",
                50,
                "integer",
                (0, 100),
                "Local iteration count used to adapt inertia.",
            ),
            "StagnationThreshold": ParameterSpec(
                "StagnationThreshold",
                15,
                "integer",
                (0, 100),
                "Maximum no-improvement steps before a particle is reinitialized.",
            ),
            "ConvergenceThreshold": ParameterSpec(
                "ConvergenceThreshold",
                0.1,
                "numeric",
                (0, 1),
                "Average distance threshold at which a sub-swarm is converged.",
            ),
            "QuantumNumber": ParameterSpec(
                "QuantumNumber",
                5,
                "integer",
                (0, 100),
                "Number of quantum particles generated per change.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.swarm_number = 0
        self.pop: list[DPCPSOSwarm] = []
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
        if self.pop and all(swarm.is_converged == 1 for swarm in self.pop):
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

        self.pop = self._build_swarms(new_positions, problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        self.swarm_number = len(self.pop)
        self.exclusion_radius = self._exclusion_radius()

    def _move_swarm(self, swarm: DPCPSOSwarm, problem) -> None:
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
        swarm.fitness = np.nan_to_num(values, nan=-np.inf)

        improved = swarm.fitness > swarm.pbest_fitness
        swarm.pbest_fitness[improved] = swarm.fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        swarm.stagnation_counter[improved] = 0
        swarm.stagnation_counter[~improved] += 1
        self._reset_best(swarm)

        threshold = int(self.parameters["StagnationThreshold"])
        stagnant_ids = np.flatnonzero(swarm.stagnation_counter >= threshold)
        for row in stagnant_ids:
            swarm.positions[row, :] = self._random_positions(1)[0]
            swarm.velocities[row, :] = self._random_velocities((1, self.dimension))[0]
            swarm.stagnation_counter[row] = 0
            value = evaluate_(problem, swarm.positions[row, :], 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            if value.size and not np.isnan(value[0]):
                swarm.fitness[row] = float(value[0])
                swarm.pbest_fitness[row] = float(value[0])
                swarm.pbest_positions[row, :] = swarm.positions[row, :]

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
        swarm: DPCPSOSwarm,
        problem,
    ) -> tuple[np.ndarray, float]:
        original_position = swarm.best_position.copy()
        original_value = float(swarm.best_fitness)
        try_position = original_position.copy()
        try_value = original_value
        for dim in range(self.dimension):
            candidate = original_position.copy()
            candidate[dim] = min(self.max_coordinate, candidate[dim] + 0.0001)
            values = evaluate_(problem, candidate, 1)
            if problem.state.recent_change or problem.state.terminated:
                return try_position, try_value
            direction = 1.0
            if not values.size or np.isnan(values[0]) or float(values[0]) <= original_value:
                direction = -1.0

            candidates = np.tile(original_position, (10, 1))
            for step in range(1, 11):
                candidates[step - 1, dim] = original_position[dim] + direction * 0.1 * step
            candidates = np.clip(candidates, self.min_coordinate, self.max_coordinate)
            all_values = evaluate_(problem, candidates, 10)
            if problem.state.recent_change or problem.state.terminated:
                return try_position, try_value
            if all_values.size == 0 or np.all(np.isnan(all_values)):
                continue
            best_idx = int(np.argmax(np.nan_to_num(all_values, nan=-np.inf)))
            best_value = float(all_values[best_idx])
            if best_value > try_value:
                try_position = candidates[best_idx, :].copy()
                try_value = best_value
        return try_position, try_value

    def _build_swarms(self, positions: np.ndarray, problem) -> list[DPCPSOSwarm]:
        groups, _remain = DensityPeakClusterer(
            peak_number=self._peak_number(problem),
        ).groups_and_remain(np.asarray(positions, dtype=float).reshape(-1, self.dimension))
        swarms: list[DPCPSOSwarm] = []
        for group in groups:
            if group.shape[0] == 0:
                continue
            trimmed, cached_fitness = self._trim_top5_if_needed(group, problem)
            if problem.state.recent_change or problem.state.terminated:
                return swarms
            swarm = self._new_swarm_from_positions(
                trimmed,
                problem,
                fitness=cached_fitness,
            )
            if problem.state.recent_change or problem.state.terminated:
                return swarms
            swarms.append(swarm)
        if not swarms and not problem.state.recent_change and not problem.state.terminated:
            fallback = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
            if fallback.shape[0] == 0:
                fallback = self._random_positions(1)
            swarms.append(self._new_swarm_from_positions(fallback, problem))
        return swarms

    def _trim_top5_if_needed(
        self,
        positions: np.ndarray,
        problem,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        group = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        if group.shape[0] <= 5:
            return group, None
        fitness = evaluate_(problem, group, group.shape[0])
        if problem.state.recent_change or problem.state.terminated:
            return group, None
        fitness = self._padded_fitness(fitness, group.shape[0])
        order = np.argsort(fitness)[::-1][:5]
        trimmed = group[order, :].copy()
        fitness = evaluate_(problem, trimmed, trimmed.shape[0])
        if problem.state.recent_change or problem.state.terminated:
            return trimmed, None
        return trimmed, self._padded_fitness(fitness, trimmed.shape[0])

    def _new_swarm_from_positions(
        self,
        positions: np.ndarray,
        problem,
        *,
        local_iter: int = 0,
        fitness: np.ndarray | None = None,
    ) -> DPCPSOSwarm:
        rows = positions.shape[0]
        initial_fitness = (
            evaluate_(problem, positions, rows)
            if fitness is None
            else np.asarray(fitness, dtype=float)
        )
        initial_fitness = self._padded_fitness(initial_fitness, rows)
        pbest_fitness = initial_fitness.copy()
        best_id = self._safe_best_id(pbest_fitness)
        center = np.mean(positions, axis=0)
        current_radius = float(np.mean(np.linalg.norm(positions - center, axis=1)))
        return DPCPSOSwarm(
            positions=positions.copy(),
            velocities=self._random_velocities(positions.shape),
            shifts=[],
            fitness=initial_fitness,
            pbest_positions=positions.copy(),
            is_converged=0,
            stagnation_counter=np.zeros(rows, dtype=int),
            is_stagnated=0,
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

    def _reset_swarm_shape(self, swarm: DPCPSOSwarm, problem) -> DPCPSOSwarm:
        rows = max(1, swarm.positions.shape[0])
        return self._new_swarm_from_positions(self._random_positions(rows), problem)

    def _merge_subpopulations(
        self,
        pop1: DPCPSOSwarm,
        pop2: DPCPSOSwarm,
    ) -> DPCPSOSwarm:
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
        return DPCPSOSwarm(
            positions=positions.copy(),
            velocities=velocities.copy(),
            shifts=[],
            fitness=fitness.copy(),
            pbest_positions=pbest_positions.copy(),
            is_converged=0,
            stagnation_counter=np.zeros(positions.shape[0], dtype=int),
            is_stagnated=0,
            is_excluded=0,
            pbest_fitness=pbest_fitness.copy(),
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
            best_id=best_id,
            center=center,
            init_radius=current_radius,
            current_radius=current_radius,
            local_iter=min(pop1.local_iter, pop2.local_iter),
        )

    def _reset_best(self, swarm: DPCPSOSwarm) -> None:
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
        return -4.0 + 8.0 * self.rng.random(shape)

    def _exclusion_radius(self) -> float:
        swarm_count = max(1, int(self.swarm_number))
        dimension = max(1, int(self.dimension))
        return float(
            0.5
            * (
                (self.max_coordinate - self.min_coordinate)
                / (swarm_count ** (1.0 / dimension))
            )
        )

    def _peak_number(self, problem) -> int:
        peak_number = int(getattr(problem, "parameters", {}).get("PeakNumber", 1))
        if peak_number <= 0:
            raise ValueError("DPCPSO requires problem PeakNumber >= 1.")
        return peak_number
