"""ImQSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.imqso.state import ImQSOSwarm


@algorithm_registry.register("ImQSO")
class ImQSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "ImQSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                5,
                "integer",
                (0, 100),
                "Number of particles in each sub-swarm.",
            ),
            "x": ParameterSpec(
                "x",
                0.729843788,
                "numeric",
                (0, 2),
                "Inertia weight used in particle velocity update.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Cognitive coefficient that scales influence of personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Social coefficient that scales influence of global best.",
            ),
            "QuantumNumber": ParameterSpec(
                "QuantumNumber",
                5,
                "integer",
                (0, 100),
                "Number of quantum particles generated to explore space around best solution.",
            ),
            "SwarmNumber": ParameterSpec(
                "SwarmNumber",
                10,
                "integer",
                (0, 100),
                "Number of independent sub-swarms in the population.",
            ),
            "alpha": ParameterSpec(
                "alpha",
                1,
                "integer",
                (0, 10),
                "Exponent that controls how sharply exclusion probability changes with distance.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.quantum_number = max(0, int(self.parameters["QuantumNumber"]))
        self.swarm_number = max(1, int(self.parameters["SwarmNumber"]))
        self.alpha = max(0, int(self.parameters["alpha"]))

        self.diversity_plus = 1
        self.shift_severity = 1.0
        self.quantum_radius = self.shift_severity
        self.exclusion_limit = self._threshold_for_swarm_count(self.swarm_number)
        self.convergence_limit = self.exclusion_limit
        self.pso_operator = PSOOperator()
        self.pop: list[ImQSOSwarm] = [
            self._generate_subpopulation(problem) for _ in range(self.swarm_number)
        ]

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        for swarm in self.pop:
            self._move_swarm(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            self._sample_quantum_particles(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return

        self._apply_exclusion(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._apply_anti_convergence(problem)

    def on_environment_change(self, problem) -> None:
        shift_samples: list[float] = []
        for swarm in self.pop:
            if not np.isnan(swarm.gbest_past_environment).any():
                shift = float(
                    np.linalg.norm(swarm.gbest_past_environment - swarm.best_position)
                )
                swarm.shifts.append(shift)
            shift_samples.extend(swarm.shifts)

        if shift_samples:
            self.shift_severity = float(np.mean(shift_samples))
        self.quantum_radius = self.shift_severity

        for swarm in self.pop:
            values = evaluate_(problem, swarm.pbest_positions, self.population_size)
            if problem.state.recent_change or problem.state.terminated:
                return
            swarm.pbest_fitness = self._nan_to_negative_infinity(values)
            swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)
            swarm.is_converged = 0

    def _generate_subpopulation(self, problem) -> ImQSOSwarm:
        positions = self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((self.population_size, self.dimension))
        )
        fitness = evaluate_(problem, positions, self.population_size)
        pbest_positions = positions.copy()
        if problem.state.recent_change:
            fitness = np.full(self.population_size, -np.inf)
            pbest_fitness = fitness.copy()
        else:
            pbest_fitness = self._nan_to_negative_infinity(fitness)
        best_id = self._best_index(pbest_fitness)
        return ImQSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            velocities=np.zeros((self.population_size, self.dimension)),
            shifts=[],
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
            is_converged=0,
        )

    def _move_swarm(self, swarm: ImQSOSwarm, problem) -> None:
        constriction = float(self.parameters["x"])
        self.pso_operator.update_and_move(
            swarm,
            inertia=constriction,
            cognitive=constriction * float(self.parameters["c1"]),
            social=constriction * float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(
            swarm, self.min_coordinate, self.max_coordinate
        )
        values = evaluate_(problem, swarm.positions, self.population_size)
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm.fitness = values
        comparable = self._nan_to_negative_infinity(values)
        improved = comparable > swarm.pbest_fitness
        swarm.pbest_fitness[improved] = comparable[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        best_id = self._best_index(swarm.pbest_fitness)
        best_value = float(swarm.pbest_fitness[best_id])
        if best_value > float(swarm.best_fitness):
            swarm.best_fitness = best_value
            swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _sample_quantum_particles(self, swarm: ImQSOSwarm, problem) -> None:
        for _ in range(self.quantum_number):
            quantum_position = swarm.best_position + (
                (2.0 * self.rng.random(self.dimension) - 1.0) * self.quantum_radius
            )
            values = evaluate_(problem, quantum_position, 1)
            if problem.state.recent_change or problem.state.terminated:
                return
            quantum_fitness = float(values[0])
            if quantum_fitness > float(swarm.best_fitness):
                swarm.best_fitness = quantum_fitness
                swarm.best_position = quantum_position.copy()

    def _apply_exclusion(self, problem) -> None:
        if self.swarm_number <= 1 or self.exclusion_limit <= 0:
            return
        for ii in range(self.swarm_number - 1):
            for jj in range(ii + 1, self.swarm_number):
                distance = float(
                    np.linalg.norm(
                        self.pop[ii].best_position - self.pop[jj].best_position
                    )
                )
                if distance >= self.exclusion_limit:
                    continue
                probability = (
                    (self.exclusion_limit - distance) / self.exclusion_limit
                ) ** self.alpha
                if self.rng.random() >= probability:
                    continue
                if self.pop[ii].best_fitness < self.pop[jj].best_fitness:
                    loser = ii
                else:
                    loser = jj
                self.pop[loser] = self._generate_subpopulation(problem)
                if problem.state.recent_change or problem.state.terminated:
                    return

    def _apply_anti_convergence(self, problem) -> None:
        worst_value = np.inf
        worst_index = 0
        converged_count = 0
        for index, swarm in enumerate(self.pop):
            if swarm.positions.shape[0] <= 1:
                radius = 0.0
            else:
                radius = float(
                    np.max(
                        np.abs(
                            swarm.positions[:, None, :] - swarm.positions[None, :, :]
                        )
                    )
                )
            swarm.is_converged = 1 if radius < self.convergence_limit else 0
            converged_count += swarm.is_converged
            if float(swarm.best_fitness) < worst_value:
                worst_value = float(swarm.best_fitness)
                worst_index = index

        if converged_count == self.swarm_number:
            self.pop[worst_index] = self._generate_subpopulation(problem)

    def _reset_best(self, swarm: ImQSOSwarm) -> None:
        best_id = self._best_index(swarm.pbest_fitness)
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _threshold_for_swarm_count(self, swarm_count: int) -> float:
        return float(
            0.5
            * (
                (self.max_coordinate - self.min_coordinate)
                / (max(1, swarm_count) ** (1.0 / self.dimension))
            )
        )

    @staticmethod
    def _nan_to_negative_infinity(values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf).copy()

    @staticmethod
    def _best_index(values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        comparable = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)
        return int(np.argmax(comparable))
