"""HmSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.hmso.state import HmSOSwarm


@algorithm_registry.register("HmSO")
class HmSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "HmSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "ParentSize": ParameterSpec("ParentSize", 5, "integer", (0, 100)),
            "ChildSize": ParameterSpec("ChildSize", 5, "integer", (0, 100)),
            "w": ParameterSpec("w", 0.729844, "numeric", (0, 2)),
            "c1": ParameterSpec("c1", 2.05, "numeric", (0, 5)),
            "c2": ParameterSpec("c2", 2.05, "numeric", (0, 5)),
            "Rpc": ParameterSpec("Rpc", 10, "numeric", (0, 100)),
            "Rexcl": ParameterSpec("Rexcl", 10, "numeric", (0, 100)),
            "Rconv": ParameterSpec("Rconv", 0.01, "numeric", (0, 10)),
            "Rsearch": ParameterSpec("Rsearch", 0.5, "numeric", (0, 10)),
            "Margin": ParameterSpec("Margin", 5, "numeric", (0, 10)),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.shift_severity = 1.0
        self.swarm_number = 0
        self.parent_size = max(1, int(self.parameters["ParentSize"]))
        self.child_size = max(1, int(self.parameters["ChildSize"]))
        self.parent_pop = self._generate_parent(problem)
        self.child_pop: list[HmSOSwarm] = []
        self.pop: list[HmSOSwarm] = [self.parent_pop]
        self.pre_gbest_value = float(self.parent_pop.best_fitness)
        self.cur_gbest_value = float(self.parent_pop.best_fitness)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        self._move_parent(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._migrate_parent_to_children(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self.cur_gbest_value = float(self.parent_pop.best_fitness)
        if self.cur_gbest_value > self.pre_gbest_value:
            self._generate_child(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
        self.pre_gbest_value = self.cur_gbest_value

        self._move_children(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._remove_overlapping_children()
        self.pop = [self.parent_pop, *self.child_pop]

    def on_environment_change(self, problem) -> None:
        parent = self.parent_pop
        values = evaluate_(problem, parent.pbest_positions, parent.pbest_positions.shape[0])
        if problem.state.recent_change or problem.state.terminated:
            return
        parent.pbest_fitness = self._finite(values, parent.pbest_positions.shape[0])
        parent.gbest_past_environment = parent.best_position.copy()
        self._reset_best(parent)
        self.pre_gbest_value = float(parent.best_fitness)
        self.cur_gbest_value = float(parent.best_fitness)

        for child in self.child_pop:
            values = evaluate_(problem, child.pbest_positions, child.pbest_positions.shape[0])
            if problem.state.recent_change or problem.state.terminated:
                return
            child.pbest_fitness = self._finite(values, child.pbest_positions.shape[0])
            child.gbest_past_environment = child.best_position.copy()
            self._reset_best(child)

        for child in self.child_pop:
            if child.is_converged == 1:
                child.is_converged = 0

        rsearch = float(self.parameters["Rsearch"])
        for child in self.child_pop:
            for row in range(child.positions.shape[0]):
                if row == child.best_id:
                    continue
                child.positions[row, :] = child.best_position + (
                    2.0 * self.rng.random(self.dimension) - 1.0
                ) * rsearch
                child.velocities[row, :] = (
                    2.0 * self.rng.random(self.dimension) - 1.0
                ) * rsearch

            values = evaluate_(problem, child.positions, child.positions.shape[0])
            if problem.state.recent_change or problem.state.terminated:
                return
            child.fitness = self._finite(values, child.positions.shape[0])
            child.pbest_fitness = child.fitness.copy()
            child.pbest_positions = child.positions.copy()
            child.gbest_past_environment = child.best_position.copy()
            self._reset_best(child)
            self._update_shape(child)

        self.pop = [self.parent_pop, *self.child_pop]

    def _generate_parent(self, problem) -> HmSOSwarm:
        rows = self.parent_size
        positions = self._random_positions(rows)
        velocities = -50.0 + 100.0 * self.rng.random((rows, self.dimension))
        values = evaluate_(problem, positions, rows)
        fitness = self._finite(values, rows)
        if problem.state.recent_change:
            fitness = np.full(rows, -np.inf, dtype=float)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        best_id = int(np.argmax(pbest_fitness))
        center = self._center(pbest_positions)
        return HmSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            best_id=best_id,
            center=center,
            current_radius=self._radius(pbest_positions, center),
            is_converged=0,
            is_remove=0,
            velocities=velocities,
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
        )

    def _generate_child(self, problem) -> None:
        child, self.parent_pop = self._build_child_from_parent(problem)
        self.child_pop.append(child)
        self.pop = [self.parent_pop, *self.child_pop]

    def _build_child_from_parent(self, problem) -> tuple[HmSOSwarm, HmSOSwarm]:
        parent = self.parent_pop
        rows = self.child_size
        positions = np.zeros((rows, self.dimension), dtype=float)
        velocities = -10.0 + 20.0 * self.rng.random((rows, self.dimension))
        pbest_positions = np.zeros_like(positions)
        fitness = np.full(rows, -np.inf, dtype=float)
        pbest_fitness = np.full(rows, -np.inf, dtype=float)

        particle_count = 0
        positions[0, :] = parent.best_position.copy()
        pbest_positions[0, :] = parent.best_position.copy()
        fitness[0] = parent.fitness[parent.best_id]
        pbest_fitness[0] = fitness[0]
        particle_count += 1

        rpc = float(self.parameters["Rpc"])
        for row in range(parent.positions.shape[0]):
            distance = float(np.linalg.norm(parent.pbest_positions[row, :] - parent.best_position))
            if distance < rpc and row != parent.best_id and particle_count < rows:
                positions[particle_count, :] = parent.pbest_positions[row, :]
                pbest_positions[particle_count, :] = parent.pbest_positions[row, :]
                fitness[particle_count] = parent.fitness[row]
                pbest_fitness[particle_count] = fitness[particle_count]
                particle_count += 1

        while particle_count < rows:
            positions[particle_count, :] = positions[0, :] + (
                2.0 * self.rng.random(self.dimension) - 1.0
            ) * rpc / 3.0
            pbest_positions[particle_count, :] = positions[particle_count, :]
            values = evaluate_(problem, positions[particle_count, :], 1)
            if problem.state.recent_change or problem.state.terminated:
                fitness[:] = -np.inf
                pbest_fitness[:] = -np.inf
                break
            fitness[particle_count] = self._first(values)
            pbest_fitness[particle_count] = fitness[particle_count]
            particle_count += 1

        if problem.state.recent_change:
            fitness[:] = -np.inf
            pbest_fitness[:] = -np.inf

        best_id = int(np.argmax(pbest_fitness))
        center = self._center(pbest_positions)
        child = HmSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            best_id=best_id,
            center=center,
            current_radius=self._radius(pbest_positions, center),
            is_converged=0,
            is_remove=0,
            velocities=velocities,
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
        )
        return child, self._generate_parent(problem)

    def _move_parent(self, problem) -> None:
        self._pso_move(self.parent_pop, problem)

    def _move_children(self, problem) -> None:
        for child in self.child_pop:
            if child.is_converged == 1:
                continue
            self._pso_move(child, problem, update_best_id=False)
            if problem.state.recent_change or problem.state.terminated:
                return
            self._update_shape(child)
            best_child_id = self._best_child_id()
            if (
                child.current_radius < float(self.parameters["Rconv"])
                and child.best_fitness
                < self.child_pop[best_child_id].best_fitness - float(self.parameters["Margin"])
            ):
                child.is_converged = 1

    def _pso_move(
        self,
        swarm: HmSOSwarm,
        problem,
        *,
        update_best_id: bool = True,
    ) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        r1 = self.rng.random((rows, self.dimension))
        r2 = self.rng.random((rows, self.dimension))
        inertia = float(self.parameters["w"])
        swarm.velocities = inertia * (
            swarm.velocities
            + float(self.parameters["c1"]) * r1 * (swarm.pbest_positions - swarm.positions)
            + float(self.parameters["c2"]) * r2 * (swarm.best_position - swarm.positions)
        )
        swarm.positions = swarm.positions + swarm.velocities
        self._clip(swarm)

        values = evaluate_(problem, swarm.positions, rows)
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm.fitness = self._finite(values, rows)
        improved = swarm.fitness > swarm.pbest_fitness
        swarm.pbest_fitness[improved] = swarm.fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        best_id = int(np.argmax(swarm.pbest_fitness))
        best_value = float(swarm.pbest_fitness[best_id])
        if best_value > float(swarm.best_fitness):
            swarm.best_fitness = best_value
            swarm.best_position = swarm.pbest_positions[best_id, :].copy()
            if update_best_id:
                swarm.best_id = best_id

    def _migrate_parent_to_children(self, problem) -> None:
        rpc = float(self.parameters["Rpc"])
        for parent_row in range(self.parent_pop.positions.shape[0]):
            for child in self.child_pop:
                distance = float(
                    np.linalg.norm(
                        self.parent_pop.pbest_positions[parent_row, :] - child.best_position
                    )
                )
                if distance >= rpc:
                    continue
                if self.parent_pop.pbest_fitness[parent_row] > child.best_fitness:
                    old_gbest_id = child.best_id
                    child.positions[old_gbest_id, :] = self.parent_pop.positions[parent_row, :]
                    child.pbest_positions[old_gbest_id, :] = self.parent_pop.pbest_positions[
                        parent_row, :
                    ]
                    child.pbest_fitness[old_gbest_id] = self.parent_pop.pbest_fitness[
                        parent_row
                    ]
                    child.velocities[old_gbest_id, :] = self.parent_pop.velocities[
                        parent_row, :
                    ]
                    child.fitness[old_gbest_id] = self.parent_pop.fitness[parent_row]
                    child.best_fitness = float(np.max(child.pbest_fitness))
                    child.best_id = int(np.argmax(child.pbest_fitness))
                    child.best_position = child.pbest_positions[old_gbest_id, :].copy()

                self.parent_pop.positions[parent_row, :] = self._random_positions(1)[0, :]
                values = evaluate_(problem, self.parent_pop.positions[parent_row, :], 1)
                if problem.state.recent_change or problem.state.terminated:
                    return
                self.parent_pop.fitness[parent_row] = self._first(values)
                self.parent_pop.pbest_positions[parent_row, :] = self.parent_pop.positions[
                    parent_row, :
                ]
                self.parent_pop.pbest_fitness[parent_row] = self.parent_pop.fitness[parent_row]
                self.parent_pop.velocities[parent_row, :] = (
                    2.0 * self.rng.random(self.dimension) - 1.0
                ) * rpc / 3.0
                self._reset_best(self.parent_pop)

    def _remove_overlapping_children(self) -> None:
        for left in range(len(self.child_pop) - 1):
            for right in range(left + 1, len(self.child_pop)):
                distance = float(
                    np.sqrt(
                        np.sum(
                            self.child_pop[left].best_position
                            - self.child_pop[right].best_position
                        )
                        ** 2
                    )
                )
                if distance < float(self.parameters["Rexcl"]):
                    if self.child_pop[left].best_fitness > self.child_pop[right].best_fitness:
                        self.child_pop[right].is_remove = 1
                    else:
                        self.child_pop[left].is_remove = 1

        while True:
            for index, child in enumerate(self.child_pop):
                if child.is_remove == 1:
                    del self.child_pop[index]
                    break
            else:
                break

    def _reset_best(self, swarm: HmSOSwarm) -> None:
        swarm.best_id = int(np.argmax(swarm.pbest_fitness))
        swarm.best_fitness = float(swarm.pbest_fitness[swarm.best_id])
        swarm.best_position = swarm.pbest_positions[swarm.best_id, :].copy()

    def _update_shape(self, swarm: HmSOSwarm) -> None:
        swarm.center = self._center(swarm.pbest_positions)
        swarm.current_radius = self._radius(swarm.pbest_positions, swarm.center)

    def _best_child_id(self) -> int:
        return int(np.argmax([child.best_fitness for child in self.child_pop]))

    def _clip(self, swarm: HmSOSwarm) -> None:
        high = swarm.positions > self.max_coordinate
        low = swarm.positions < self.min_coordinate
        clipped = high | low
        swarm.positions[high] = self.max_coordinate
        swarm.positions[low] = self.min_coordinate
        swarm.velocities[clipped] = 0.0

    def _random_positions(self, rows: int) -> np.ndarray:
        return self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((rows, self.dimension))
        )

    def _finite(self, values: np.ndarray, expected: int) -> np.ndarray:
        fitness = np.full(expected, -np.inf, dtype=float)
        count = min(expected, values.size)
        if count:
            fitness[:count] = np.nan_to_num(values[:count], nan=-np.inf)
        return fitness

    def _first(self, values: np.ndarray) -> float:
        if values.size == 0 or np.isnan(values[0]):
            return -np.inf
        return float(values[0])

    def _center(self, positions: np.ndarray) -> np.ndarray:
        return np.mean(positions, axis=0)

    def _radius(self, positions: np.ndarray, center: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(positions - center, axis=1)))
