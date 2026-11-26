"""IDSPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.idspso.state import IDSPSOSwarm


@algorithm_registry.register("IDSPSO")
class IDSPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "IDSPSO"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                100,
                "integer",
                (0, 1000),
                "Total number of individuals in the population.",
            ),
            "x": ParameterSpec(
                "x",
                0.729843788,
                "numeric",
                (0, 2),
                "Inertia weight used in velocity updates.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Scaling factor for movement toward each particle's personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Scaling factor for movement toward the best of its species.",
            ),
            "Radius": ParameterSpec(
                "Radius",
                30,
                "integer",
                (0, 100),
                "Distance threshold for forming species clusters.",
            ),
            "Pmax": ParameterSpec(
                "Pmax",
                10,
                "integer",
                (0, 100),
                "Upper limit on active individuals retained in each species.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.diversity_plus = 1
        self.shift_severity = 1.0
        self.quantum_radius = 0.5 * self.shift_severity
        self.swarm_number = 1
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.active = np.ones(self.population_size, dtype=int)
        self.pop = self._generate_subpopulation(problem)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm = self.pop
        self.active = np.ones(self.population_size, dtype=int)
        species = self._species()

        self._apply_pmax(species, problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._apply_anti_convergence(species, problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        inertia = float(self.parameters["x"])
        cognitive = float(self.parameters["c1"])
        social = float(self.parameters["c2"])
        for seed, members in species:
            seed_position = swarm.pbest_positions[seed, :].copy()
            for member in members:
                swarm.lbest_position[member, :] = seed_position
                r1 = self.rng.random(self.dimension)
                r2 = self.rng.random(self.dimension)
                swarm.velocities[member, :] = inertia * (
                    swarm.velocities[member, :]
                    + cognitive
                    * r1
                    * (swarm.pbest_positions[member, :] - swarm.positions[member, :])
                    + social * r2 * (seed_position - swarm.positions[member, :])
                )
                swarm.positions[member, :] = (
                    swarm.positions[member, :] + swarm.velocities[member, :]
                )
                self._clip_member(swarm, member)
                values = evaluate_(problem, swarm.positions[member, :], 1)
                if problem.state.recent_change or problem.state.terminated:
                    return
                swarm.fitness[member] = self._first_value(values)

        self._update_pbests_from_fitness()
        self._reset_best(swarm)

    def on_environment_change(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm = self.pop
        species = self._species()
        for seed, members in species:
            for member in members:
                seed_position = swarm.pbest_positions[seed, :]
                offset = (2.0 * self.rng.random(self.dimension) - 1.0) * self.quantum_radius
                swarm.positions[member, :] = seed_position + offset
                values = evaluate_(problem, swarm.positions[member, :], 1)
                if problem.state.recent_change or problem.state.terminated:
                    return
                swarm.fitness[member] = self._first_value(values)
                swarm.pbest_fitness[member] = swarm.fitness[member]
                swarm.pbest_positions[member, :] = swarm.positions[member, :].copy()
        self._reset_best(swarm)

    def _generate_subpopulation(self, problem) -> IDSPSOSwarm:
        positions = self._random_positions(self.population_size)
        fitness = self._padded_fitness(evaluate_(problem, positions, self.population_size))
        pbest_positions = positions.copy()
        if problem.state.recent_change:
            fitness = np.full(self.population_size, -np.inf, dtype=float)
            pbest_fitness = fitness.copy()
        else:
            pbest_fitness = fitness.copy()
        best_id = self._safe_best_id(pbest_fitness)
        return IDSPSOSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            velocities=np.zeros((self.population_size, self.dimension), dtype=float),
            shifts=[],
            lbest_position=np.full((self.population_size, self.dimension), np.nan),
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
            best_id=best_id,
        )

    def _species(self) -> list[tuple[int, list[int]]]:
        swarm = self.pop
        order = np.argsort(np.nan_to_num(swarm.pbest_fitness, nan=-np.inf))[::-1]
        species: list[tuple[int, list[int]]] = []
        radius = float(self.parameters["Radius"])
        for particle in order:
            particle_id = int(particle)
            if not species:
                species.append((particle_id, [particle_id]))
                continue
            for seed, members in species:
                distance = float(
                    np.linalg.norm(
                        swarm.pbest_positions[seed, :]
                        - swarm.pbest_positions[particle_id, :]
                    )
                )
                if distance < radius:
                    members.append(particle_id)
                    break
            else:
                species.append((particle_id, [particle_id]))
        return species

    def _apply_pmax(
        self,
        species: list[tuple[int, list[int]]],
        problem,
    ) -> None:
        pmax = int(self.parameters["Pmax"])
        swarm = self.pop
        for _seed, members in species:
            if len(members) <= pmax:
                continue
            member_array = np.asarray(members, dtype=int)
            values = np.nan_to_num(swarm.pbest_fitness[member_array], nan=-np.inf)
            order = np.argsort(values)
            excess = len(members) - max(0, pmax)
            for local_idx in order[:excess]:
                member = int(member_array[int(local_idx)])
                self._reset_member(member, problem)
                if problem.state.recent_change or problem.state.terminated:
                    return

    def _apply_anti_convergence(
        self,
        species: list[tuple[int, list[int]]],
        problem,
    ) -> None:
        if not species:
            return
        swarm = self.pop
        seed_values = [
            float(np.nan_to_num(swarm.pbest_fitness[seed], nan=-np.inf))
            for seed, _members in species
        ]
        worst_index = int(np.argmin(seed_values))
        for member in species[worst_index][1]:
            self._reset_member(member, problem)
            if problem.state.recent_change or problem.state.terminated:
                return

    def _reset_member(self, member: int, problem) -> None:
        swarm = self.pop
        self.active[member] = 0
        swarm.positions[member, :] = self._random_positions(1)[0, :]
        values = evaluate_(problem, swarm.positions[member, :], 1)
        if problem.state.recent_change or problem.state.terminated:
            return
        swarm.fitness[member] = self._first_value(values)
        swarm.pbest_positions[member, :] = swarm.positions[member, :].copy()
        swarm.pbest_fitness[member] = swarm.fitness[member]
        swarm.velocities[member, :] = 0.0

    def _update_pbests_from_fitness(self) -> None:
        swarm = self.pop
        fitness = np.nan_to_num(swarm.fitness, nan=-np.inf)
        pbest = np.nan_to_num(swarm.pbest_fitness, nan=-np.inf)
        improved = fitness > pbest
        swarm.pbest_fitness[improved] = fitness[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]

    def _reset_best(self, swarm: IDSPSOSwarm) -> None:
        best_id = self._safe_best_id(swarm.pbest_fitness)
        swarm.best_id = best_id
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _clip_member(self, swarm: IDSPSOSwarm, member: int) -> None:
        high = swarm.positions[member, :] > self.max_coordinate
        low = swarm.positions[member, :] < self.min_coordinate
        clipped = high | low
        if np.any(high):
            swarm.positions[member, high] = self.max_coordinate
        if np.any(low):
            swarm.positions[member, low] = self.min_coordinate
        if np.any(clipped):
            swarm.velocities[member, clipped] = 0.0

    def _random_positions(self, rows: int) -> np.ndarray:
        return self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((rows, self.dimension))
        )

    def _padded_fitness(self, values: np.ndarray) -> np.ndarray:
        fitness = np.full(self.population_size, -np.inf, dtype=float)
        count = min(self.population_size, values.size)
        if count:
            fitness[:count] = np.nan_to_num(values[:count], nan=-np.inf)
        return fitness

    def _first_value(self, values: np.ndarray) -> float:
        if values.size == 0 or np.isnan(values[0]):
            return -np.inf
        return float(values[0])

    def _safe_best_id(self, values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        return int(np.argmax(np.nan_to_num(values, nan=-np.inf)))
