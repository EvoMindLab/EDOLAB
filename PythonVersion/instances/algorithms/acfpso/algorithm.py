"""ACFPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.acfpso.state import ACFPSOSwarm
from instances.algorithms.common import evaluate_


@algorithm_registry.register("ACFPSO")
class ACFPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "ACFPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec("PopulationSize", 5, "integer", (0, 100)),
            "x": ParameterSpec("x", 0.729843788, "numeric", (0, 2)),
            "c1": ParameterSpec("c1", 2.05, "numeric", (0, 5)),
            "c2": ParameterSpec("c2", 2.05, "numeric", (0, 5)),
            "ConvergenceSleepLayer": ParameterSpec("ConvergenceSleepLayer", 0.05, "numeric", (0, 2)),
            "ConvergenceOuterLayer": ParameterSpec("ConvergenceOuterLayer", 0.8, "numeric", (0, 2)),
            "ConvergenceInnerLayer": ParameterSpec("ConvergenceInnerLayer", 0.3, "numeric", (0, 2)),
            "ExclusionOuterLayer": ParameterSpec("ExclusionOuterLayer", 1.0, "numeric", (0, 2)),
            "ExclusionInnerLayer": ParameterSpec("ExclusionInnerLayer", 0.3, "numeric", (0, 2)),
        }

    def initialize(self, problem) -> None:
        self.swarm_number = 1
        self.dimension = problem.dimension
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.free_swarm_id = 0
        self.radius = np.ones(self.dimension) * (
            self.max_coordinate - self.min_coordinate
        ) / self.swarm_number
        self.diversity_plus = 1
        self.shift_severity = np.ones(self.dimension)
        self.relocations = np.full((problem.environment_number, self.dimension), np.nan)
        self.relocations[0, :] = self.shift_severity
        self.pop: list[ACFPSOSwarm] = []
        self.pso_operator = PSOOperator()
        for _ in range(self.swarm_number):
            self.pop.append(self._generate_subpopulation(problem))

    def iterate(self, problem) -> None:
        best_index = self._best_swarm_index()
        ticket_winners = self._ticket_winners(best_index)
        while ticket_winners:
            ii = ticket_winners.pop(0)
            if ii >= self.swarm_number:
                continue
            swarm = self.pop[ii]
            self._move_swarm(swarm, problem)
            if problem.state.recent_change:
                return
            self._update_phase_and_sleep(swarm)
            if swarm.sleep == 1:
                ticket_winners = [winner for winner in ticket_winners if winner != ii]
            if self._apply_exclusion_for_swarm(ii, ticket_winners, problem):
                return

        if self.pop and self.pop[self.free_swarm_id].phase > 1:
            if self.swarm_number > 30:
                self._remove_worst_non_free_swarm()
            self.swarm_number += 1
            self.free_swarm_id = self.swarm_number - 1
            self.pop.append(self._generate_subpopulation(problem))
            if problem.state.recent_change:
                return

        tracker_number = sum(1 for swarm in self.pop if swarm.phase == 3)
        self.radius = np.ones(self.dimension) * (
            (self.max_coordinate - self.min_coordinate) / max(1, tracker_number)
        )

    def on_environment_change(self, problem) -> None:
        dummy = np.full((self.swarm_number, self.dimension), np.nan)
        for jj, swarm in enumerate(self.pop):
            if swarm.phase == 3 and not np.isnan(swarm.gbest_past_environment).any():
                dummy[jj, :] = np.abs(swarm.gbest_past_environment - swarm.best_position)
        dummy = dummy[~np.isnan(dummy).any(axis=1)]
        if dummy.size:
            env = min(problem.state.environment_index, self.relocations.shape[0] - 1)
            self.relocations[env, :] = np.mean(dummy, axis=0)
        relocation_samples = self.relocations[~np.isnan(self.relocations).any(axis=1)]
        if relocation_samples.size:
            self.shift_severity = np.mean(relocation_samples, axis=0)

        for jj, swarm in enumerate(self.pop):
            if jj != self.free_swarm_id and swarm.phase == 3:
                signs = self.rng.integers(1, 3, size=(self.population_size, self.dimension)) - 1
                signs[signs == 0] = -1
                swarm.positions = (
                    np.tile(swarm.best_position, (self.population_size, 1))
                    + np.tile(self.shift_severity, (self.population_size, 1)) * signs
                )
                swarm.positions[0, :] = swarm.best_position

        for swarm in self.pop:
            swarm.sleep = 0
            values = evaluate_(problem, swarm.positions, self.population_size)
            if problem.state.recent_change:
                return
            swarm.fitness = values
            swarm.pbest_fitness = values.copy()
            swarm.pbest_positions = swarm.positions.copy()
            if swarm.phase == 3:
                swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)

    def _generate_subpopulation(self, problem) -> ACFPSOSwarm:
        x = self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((self.population_size, self.dimension))
        )
        fitness = evaluate_(problem, x, self.population_size)
        pbest_position = x.copy()
        center = np.mean(pbest_position, axis=0)
        diversity = np.maximum(
            np.max(pbest_position, axis=0) - center,
            center - np.min(pbest_position, axis=0),
        )
        if not problem.state.recent_change:
            pbest_value = fitness.copy()
        else:
            fitness = np.full(self.population_size, -np.inf)
            pbest_value = fitness.copy()
        best_id = int(np.argmax(pbest_value))
        return ACFPSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            previous_best_position=np.full(self.dimension, np.nan),
            velocities=np.zeros((self.population_size, self.dimension)),
            shifts=[],
            sleep=0,
            iteration_counter=0,
            phase=1,
            positions=x,
            fitness=fitness,
            pbest_positions=pbest_position,
            center=center,
            diversity=diversity,
            pbest_fitness=pbest_value,
            best_fitness=float(pbest_value[best_id]),
            best_position=pbest_position[best_id, :].copy(),
        )

    def _best_swarm_index(self) -> int:
        return int(np.argmax([swarm.best_fitness for swarm in self.pop]))

    def _ticket_winners(self, best_index: int) -> list[int]:
        active = [
            (idx, swarm.best_fitness)
            for idx, swarm in enumerate(self.pop)
            if idx != self.free_swarm_id and swarm.sleep == 0
        ]
        if not active:
            return [self.free_swarm_id]
        active, starts, ends, weights = ACFPSOHelper.build_ticket_ranges(active)
        random_tickets = self.rng.integers(1, int(np.sum(weights)) + 1, size=len(active))
        winners: list[tuple[int, float]] = []
        for ticket in random_tickets:
            for column, (start, end) in enumerate(zip(starts, ends)):
                if start <= ticket <= end:
                    winners.append(active[column])
                    break
        winners.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in winners] + [best_index, self.free_swarm_id]

    def _move_swarm(self, swarm: ACFPSOSwarm, problem) -> None:
        constriction = float(self.parameters["x"])
        self.pso_operator.update_and_move(
            swarm,
            inertia=constriction,
            cognitive=constriction * float(self.parameters["c1"]),
            social=constriction * float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(swarm, self.min_coordinate, self.max_coordinate)
        values = evaluate_(problem, swarm.positions, self.population_size)
        if problem.state.recent_change:
            return
        swarm.fitness = values
        improved = swarm.fitness > swarm.pbest_fitness
        swarm.pbest_fitness[improved] = swarm.fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        best_id = int(np.argmax(swarm.pbest_fitness))
        best_value = float(swarm.pbest_fitness[best_id])
        if best_value > swarm.best_fitness:
            swarm.best_fitness = best_value
            swarm.previous_best_position = swarm.best_position.copy()
            swarm.best_position = swarm.pbest_positions[best_id, :].copy()
        swarm.center = np.mean(swarm.pbest_positions, axis=0)
        swarm.diversity = np.maximum(
            np.max(swarm.pbest_positions, axis=0) - swarm.center,
            swarm.center - np.min(swarm.pbest_positions, axis=0),
        )

    def _update_phase_and_sleep(self, swarm: ACFPSOSwarm) -> None:
        if not np.any(swarm.diversity > float(self.parameters["ConvergenceInnerLayer"]) * self.radius):
            swarm.phase = 3
        elif not np.any(
            swarm.diversity > float(self.parameters["ConvergenceOuterLayer"]) * self.radius
        ):
            swarm.phase = 2
        else:
            swarm.phase = 1
        if not np.any(swarm.diversity > float(self.parameters["ConvergenceSleepLayer"])):
            swarm.sleep = 1

    def _apply_exclusion_for_swarm(self, ii: int, ticket_winners: list[int], problem) -> bool:
        jj = 0
        while True:
            if jj == ii:
                jj += 1
                continue
            if jj >= self.swarm_number or ii >= self.swarm_number:
                break
            left = self.pop[ii]
            right = self.pop[jj]
            distance = np.abs(left.best_position - right.best_position)
            if not np.any(float(self.parameters["ExclusionInnerLayer"]) * self.radius < distance):
                if ii == self.free_swarm_id or jj == self.free_swarm_id:
                    self.pop[self.free_swarm_id] = self._generate_subpopulation(problem)
                    if problem.state.recent_change:
                        return True
                else:
                    target = ii if left.best_fitness < right.best_fitness else jj
                    self._delete_swarm(target, ticket_winners)
                    if target == ii:
                        break
                    if target < ii:
                        ii -= 1
                        jj -= 1
            elif not np.any(float(self.parameters["ExclusionOuterLayer"]) * self.radius < distance):
                if left.phase == 3 and right.phase == 3:
                    pass
                elif left.phase == 3 or right.phase == 3:
                    target = self._warning_area_target(ii, jj)
                    if target is not None:
                        if target == self.free_swarm_id:
                            self.pop[self.free_swarm_id] = self._generate_subpopulation(problem)
                            if problem.state.recent_change:
                                return True
                        else:
                            self._delete_swarm(target, ticket_winners)
                            if target == ii:
                                break
                            if target < ii:
                                ii -= 1
                                jj -= 1
                else:
                    if ii == self.free_swarm_id or jj == self.free_swarm_id:
                        self.pop[self.free_swarm_id] = self._generate_subpopulation(problem)
                        if problem.state.recent_change:
                            return True
                    else:
                        target = ii if left.best_fitness < right.best_fitness else jj
                        self._delete_swarm(target, ticket_winners)
                        if target == ii:
                            break
                        if target < ii:
                            ii -= 1
                            jj -= 1
            jj += 1
        return False

    def _warning_area_target(self, ii: int, jj: int) -> int | None:
        left = self.pop[ii]
        right = self.pop[jj]
        if left.phase == 3:
            if left.best_fitness < right.best_fitness:
                return None
            return jj
        if right.phase == 3:
            if right.best_fitness < left.best_fitness:
                return None
            return ii
        return None

    def _delete_swarm(self, index: int, ticket_winners: list[int]) -> None:
        del self.pop[index]
        self.swarm_number -= 1
        if index <= self.free_swarm_id:
            self.free_swarm_id -= 1
        ticket_winners[:] = [
            winner - 1 if winner > index else winner
            for winner in ticket_winners
            if winner != index
        ]

    def _remove_worst_non_free_swarm(self) -> None:
        candidates = [
            (idx, swarm.best_fitness)
            for idx, swarm in enumerate(self.pop)
            if idx != self.free_swarm_id
        ]
        if not candidates:
            return
        target = min(candidates, key=lambda item: item[1])[0]
        del self.pop[target]
        self.swarm_number -= 1
        if target <= self.free_swarm_id:
            self.free_swarm_id -= 1

    def _reset_best(self, swarm: ACFPSOSwarm) -> None:
        best_id = int(np.argmax(swarm.pbest_fitness))
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()


class ACFPSOHelper:
    """Stateless helper methods used by the ACFPSO flow."""

    @staticmethod
    def build_ticket_ranges(
        active: list[tuple[int, float]],
    ) -> tuple[list[tuple[int, float]], np.ndarray, np.ndarray, np.ndarray]:
        """Sort active swarms and build rank-based ticket intervals."""

        ordered = sorted(active, key=lambda item: item[1], reverse=True)
        weights = np.arange(len(ordered), 0, -1)
        starts = np.empty_like(weights)
        ends = np.empty_like(weights)
        starts[0] = 1
        ends[0] = weights[0]
        for idx in range(1, len(weights)):
            starts[idx] = ends[idx - 1] + 1
            ends[idx] = starts[idx] + weights[idx] - 1
        return ordered, starts, ends, weights
