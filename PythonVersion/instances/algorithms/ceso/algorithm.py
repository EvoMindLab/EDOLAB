"""CESO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import DEOperator, PSOOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.ceso.state import CESOCRDEPopulation, CESOSwarm
from instances.algorithms.common import evaluate_


@algorithm_registry.register("CESO")
class CESOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "CESO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSizeSwarm": ParameterSpec(
                "PopulationSizeSwarm",
                10,
                "integer",
                (0, 100),
                "Number of particles in the PSO-based swarm. Must not exceed the CRDE population size.",
            ),
            "PopulationSizeCRDE": ParameterSpec(
                "PopulationSizeCRDE",
                10,
                "integer",
                (0, 100),
                "Number of individuals in the DE-based population (CRDE). Should be >= swarm size.",
            ),
            "x": ParameterSpec(
                "x",
                0.729843788,
                "numeric",
                (0, 2),
                "Inertia weight used to balance exploration and exploitation in PSO.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Coefficient for particle personal best guidance in PSO.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Coefficient for global best guidance in PSO.",
            ),
            "theta": ParameterSpec(
                "theta",
                1,
                "numeric",
                (0, 2),
                "Threshold for transferring best solution between CRDE and swarm populations.",
            ),
            "CR": ParameterSpec(
                "CR",
                0.9,
                "numeric",
                (0, 2),
                "Probability of applying crossover in CRDE differential evolution process.",
            ),
            "F": ParameterSpec(
                "F",
                0.5,
                "numeric",
                (0, 2),
                "Differential mutation scaling factor in CRDE.",
            ),
            "Vmax": ParameterSpec(
                "Vmax",
                0.1,
                "numeric",
                (-10, 10),
                "Maximum allowed velocity value for particles in PSO.",
            ),
            "Vmin": ParameterSpec(
                "Vmin",
                -0.1,
                "numeric",
                (-10, 10),
                "Minimum allowed velocity value for particles in PSO.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.population_size_swarm = max(1, int(self.parameters["PopulationSizeSwarm"]))
        self.population_size_crde = max(
            3,
            int(self.parameters["PopulationSizeCRDE"]),
            self.population_size_swarm,
        )
        self.diversity_plus = 1
        self.shift_severity = 1.0
        self.pso_operator = PSOOperator()
        self.de_operator = DEOperator()

        self.pop = self._generate_swarm(problem)
        self.crde = self._empty_crde()
        if problem.state.recent_change or problem.state.terminated:
            return
        self.crde = self._generate_crde(problem)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        self._transmit_crde_to_swarm()
        self._update_swarm(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._evolve_crde(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._transmit_swarm_to_crde()

    def on_environment_change(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        self._transmit_crde_to_swarm()

        values = evaluate_(problem, self.pop.pbest_positions, self.population_size_swarm)
        if problem.state.recent_change or problem.state.terminated:
            return
        self.pop.pbest_fitness = self._finite_fitness(values)
        self._reset_swarm_best()

        values = evaluate_(problem, self.crde.positions, self.population_size_crde)
        if problem.state.recent_change or problem.state.terminated:
            return
        self.crde.fitness = self._finite_fitness(values)
        self.crde.best_position = self.pop.best_position.copy()
        self.crde.best_fitness = float(self.pop.best_fitness)

    def _generate_swarm(self, problem) -> CESOSwarm:
        positions = self.bounds.sample(self.population_size_swarm, self.rng)
        fitness = evaluate_(problem, positions, self.population_size_swarm)
        pbest_positions = positions.copy()
        pbest_fitness = (
            np.full(self.population_size_swarm, -np.inf)
            if problem.state.recent_change
            else self._finite_fitness(fitness)
        )
        best_id = self._best_index(pbest_fitness)
        return CESOSwarm(
            positions=positions,
            fitness=self._finite_fitness(fitness),
            velocities=np.zeros((self.population_size_swarm, self.dimension)),
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_position=pbest_positions[best_id, :].copy(),
            best_fitness=float(pbest_fitness[best_id]),
        )

    def _generate_crde(self, problem) -> CESOCRDEPopulation:
        positions = self.bounds.sample(self.population_size_crde, self.rng)
        fitness = evaluate_(problem, positions, self.population_size_crde)
        crde_fitness = (
            np.full(self.population_size_crde, -np.inf)
            if problem.state.recent_change
            else self._finite_fitness(fitness)
        )
        best_id = self._best_index(crde_fitness)
        return CESOCRDEPopulation(
            positions=positions,
            fitness=crde_fitness,
            best_position=positions[best_id, :].copy(),
            best_fitness=float(crde_fitness[best_id]),
        )

    def _empty_crde(self) -> CESOCRDEPopulation:
        midpoint = (self.min_coordinate + self.max_coordinate) / 2.0
        positions = np.tile(midpoint, (self.population_size_crde, 1))
        return CESOCRDEPopulation(
            positions=positions,
            fitness=np.full(self.population_size_crde, -np.inf),
            best_position=positions[0, :].copy(),
            best_fitness=-np.inf,
        )

    def _transmit_crde_to_swarm(self) -> None:
        if not np.isfinite(self.crde.best_fitness) or not np.isfinite(self.pop.best_fitness):
            return
        distance = float(np.linalg.norm(self.crde.best_position - self.pop.best_position))
        if distance >= float(self.parameters["theta"]):
            return

        if self.population_size_crde > self.population_size_swarm:
            order = np.argsort(self._finite_fitness(self.crde.fitness))[::-1]
            self.pop.positions = self.crde.positions[
                order[: self.population_size_swarm],
                :,
            ].copy()
        else:
            self.pop.positions = self.crde.positions.copy()
        self.pop.pbest_positions = self.pop.positions.copy()
        self.pop.velocities = self._random_velocities()

    def _update_swarm(self, problem) -> None:
        constriction = float(self.parameters["x"])
        self.pso_operator.update_and_move(
            self.pop,
            inertia=constriction,
            cognitive=constriction * float(self.parameters["c1"]),
            social=constriction * float(self.parameters["c2"]),
            rng=self.rng,
        )
        self.pso_operator.clip_to_bounds(self.pop, self.min_coordinate, self.max_coordinate)
        values = evaluate_(problem, self.pop.positions, self.population_size_swarm)
        if problem.state.recent_change or problem.state.terminated:
            return

        self.pop.fitness = self._finite_fitness(values)
        improved = self.pop.fitness > self.pop.pbest_fitness
        self.pop.pbest_fitness[improved] = self.pop.fitness[improved]
        self.pop.pbest_positions[improved, :] = self.pop.positions[improved, :]
        best_id = self._best_index(self.pop.pbest_fitness)
        best_value = float(self.pop.pbest_fitness[best_id])
        if best_value > float(self.pop.best_fitness):
            self.pop.best_fitness = best_value
            self.pop.best_position = self.pop.pbest_positions[best_id, :].copy()

    def _evolve_crde(self, problem) -> None:
        trials = np.empty_like(self.crde.positions)
        for row in range(self.population_size_crde):
            indices = self.rng.permutation(self.population_size_crde)
            r1, r2, r3 = indices[:3]
            donor = self.crde.positions[r1, :] + float(self.parameters["F"]) * (
                self.crde.positions[r2, :] - self.crde.positions[r3, :]
            )
            mix = float(self.rng.random())
            donor = (1.0 - mix) * self.crde.best_position + mix * donor
            donor = np.clip(donor, self.min_coordinate, self.max_coordinate)
            trial = self.crde.positions[row, :].copy()
            for col in range(self.dimension):
                if float(self.rng.random()) < float(self.parameters["CR"]):
                    trial[col] = donor[col]
                else:
                    break
            trials[row, :] = trial

        values = evaluate_(problem, trials, self.population_size_crde)
        if problem.state.recent_change or problem.state.terminated:
            return
        trial_fitness = self._finite_fitness(values)

        for row in range(self.population_size_crde):
            distances = np.linalg.norm(self.crde.positions - trials[row, :], axis=1)
            nearest_id = int(np.argmin(distances))
            _, selected_fitness, better = self.de_operator.greedy_select(
                self.crde.positions[nearest_id : nearest_id + 1, :],
                self.crde.fitness[nearest_id : nearest_id + 1],
                trials[row : row + 1, :],
                trial_fitness[row : row + 1],
            )
            if bool(better[0]):
                self.crde.positions[nearest_id, :] = trials[row, :]
                self.crde.fitness[nearest_id] = selected_fitness[0]

        best_id = self._best_index(self.crde.fitness)
        best_value = float(self.crde.fitness[best_id])
        if best_value > float(self.crde.best_fitness):
            self.crde.best_fitness = best_value
            self.crde.best_position = self.crde.positions[best_id, :].copy()

    def _transmit_swarm_to_crde(self) -> None:
        if float(self.crde.best_fitness) < float(self.pop.best_fitness):
            self.crde.best_fitness = float(self.pop.best_fitness)
            self.crde.best_position = self.pop.best_position.copy()

    def _reset_swarm_best(self) -> None:
        best_id = self._best_index(self.pop.pbest_fitness)
        self.pop.best_fitness = float(self.pop.pbest_fitness[best_id])
        self.pop.best_position = self.pop.pbest_positions[best_id, :].copy()

    def _random_velocities(self) -> np.ndarray:
        vmin = float(self.parameters["Vmin"])
        vmax = float(self.parameters["Vmax"])
        lower = min(vmin, vmax)
        upper = max(vmin, vmax)
        return lower + (upper - lower) * self.rng.random(
            (self.population_size_swarm, self.dimension)
        )

    @staticmethod
    def _finite_fitness(values: np.ndarray | None) -> np.ndarray:
        return np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)

    @staticmethod
    def _best_index(values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        return int(np.argmax(np.nan_to_num(values, nan=-np.inf)))
