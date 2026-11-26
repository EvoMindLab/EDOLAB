"""PSPSO algorithm instance."""

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
from instances.algorithms.pspso.state import PSPSOSwarm


class PSPSOSwarmFactory:
    """Build PSPSO sub-swarms from sampled or preserved positions."""

    def __init__(self, algorithm: "PSPSOAlgorithm") -> None:
        self.algorithm = algorithm

    def __getattr__(self, name: str):
        return getattr(self.algorithm, name)

    def generate_subpopulations(
        self,
        positions: np.ndarray,
        problem,
    ) -> list[PSPSOSwarm]:
        positions = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        if positions.shape[0] == 0:
            return []

        values = evaluate_(problem, positions, positions.shape[0])
        if problem.state.recent_change:
            comparable_values = np.full(positions.shape[0], -np.inf, dtype=float)
            stored_values = comparable_values.copy()
        elif problem.state.terminated and not has_complete_values(
            values,
            positions.shape[0],
        ):
            complete = ~np.isnan(values)
            positions = positions[complete, :]
            stored_values = values[complete].copy()
            comparable_values = self._finite_fitness(stored_values)
        elif not has_complete_values(values, positions.shape[0]):
            complete = ~np.isnan(values)
            positions = positions[complete, :]
            stored_values = values[complete].copy()
            comparable_values = self._finite_fitness(stored_values)
        else:
            stored_values = values.copy()
            comparable_values = self._finite_fitness(values)

        if positions.shape[0] == 0:
            return []

        cluster_indices = PSPSOHelper.nearest_neighbor_clusters(
            positions,
            comparable_values,
            self.swarm_size,
        )
        swarms: list[PSPSOSwarm] = []
        for indices in cluster_indices:
            if indices.size == 0:
                continue
            group = positions[indices, :].copy()
            group_values = stored_values[indices].copy()
            pbest_fitness = self._finite_fitness(group_values)
            center = np.mean(group, axis=0)
            init_radius = PSPSOHelper.mean_radius(group, center)
            span = self.max_coordinate - self.min_coordinate
            velocities = -(span / 4.0) + (2.0 * span / 4.0) * self.rng.random(
                group.shape,
            )
            best_id = self._best_index(pbest_fitness)
            swarms.append(
                PSPSOSwarm(
                    positions=group,
                    velocities=velocities,
                    fitness=group_values,
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
        return swarms


@algorithm_registry.register("PSPSO")
class PSPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "PSPSO"

    def __init__(self) -> None:
        self.swarm_factory = PSPSOSwarmFactory(self)

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "SwarmNumber": ParameterSpec(
                "SwarmNumber",
                10,
                "integer",
                (1, 1000),
                "Initial number of sub-swarms.",
            ),
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                7,
                "integer",
                (1, 1000),
                "Particles per PSPSO sub-swarm.",
            ),
            "SwarmSize": ParameterSpec(
                "SwarmSize",
                7,
                "integer",
                (1, 1000),
                "MATLAB-compatible alias for particles per sub-swarm.",
            ),
            "w": ParameterSpec(
                "w",
                0.729843788,
                "numeric",
                (0, 2),
                "Constriction factor applied to the PSO update.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Cognitive coefficient.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Social coefficient.",
            ),
            "ConvergenceLimit": ParameterSpec(
                "ConvergenceLimit",
                0.01,
                "numeric",
                (0, 10),
                "Base radius threshold multiplied by sqrt(dimension).",
            ),
            "DiversityDegree": ParameterSpec(
                "DiversityDegree",
                0.7,
                "numeric",
                (0, 1),
                "Minimum active-particle fraction before regeneration.",
            ),
            "PerturbationFactor": ParameterSpec(
                "PerturbationFactor",
                0.025,
                "numeric",
                (0, 1),
                "Velocity perturbation fraction of the search span.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.initial_swarm_number = self._positive_integer_parameter("SwarmNumber")
        self._positive_integer_parameter("PopulationSize")
        self._positive_integer_parameter("SwarmSize")
        self.swarm_size = self._resolve_swarm_size()
        self.init_population_size = self.initial_swarm_number * self.swarm_size
        self.convergence_limit = float(self.parameters["ConvergenceLimit"]) * np.sqrt(
            self.dimension,
        )
        self.perturbation_range = float(self.parameters["PerturbationFactor"]) * (
            self.max_coordinate - self.min_coordinate
        )
        self.pso_operator = PSOOperator()

        init_positions = self.bounds.sample(self.init_population_size, self.rng)
        self.pop = self.swarm_factory.generate_subpopulations(init_positions, problem)
        self.swarm_number = len(self.pop)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        if not self.pop:
            self._regenerate_from_random_positions(problem)
            return

        for swarm in list(self.pop):
            if swarm.is_converged == 1:
                continue
            self._pso_move(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return

        self._refresh_centers()
        self._delete_overlapping_worst_swarms()
        if not self.pop:
            self._regenerate_from_random_positions(problem)
            return

        self._perturb_random_subpopulation(problem)
        if problem.state.recent_change or problem.state.terminated:
            return

        self._refresh_radii()
        newly_converged = self._detect_convergence()
        self._ensure_diversity(newly_converged, problem)

    def on_environment_change(self, problem) -> None:
        """PSPSO's MATLAB source has no explicit change reaction."""

    def _pso_move(self, swarm: PSPSOSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        constriction = float(self.parameters["w"])
        self.pso_operator.update_and_move(
            swarm,
            inertia=constriction,
            cognitive=constriction * float(self.parameters["c1"]),
            social=constriction * float(self.parameters["c2"]),
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
        improved = comparable > swarm.pbest_fitness
        if not np.any(improved):
            return
        swarm.pbest_fitness[improved] = comparable[improved]
        swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        self._reset_best(swarm)

    def _refresh_centers(self) -> None:
        for swarm in self.pop:
            if swarm.is_converged == 1:
                continue
            if swarm.positions.shape[0] == 0:
                swarm.center = np.zeros(self.dimension)
            else:
                swarm.center = update_center(swarm)

    def _delete_overlapping_worst_swarms(self) -> None:
        removed = True
        while removed:
            removed = False
            for left in range(self.swarm_number):
                if (
                    left >= self.swarm_number
                    or self.pop[left].positions.shape[0] == 0
                    or self.pop[left].is_converged == 1
                ):
                    continue
                for right in range(left + 1, self.swarm_number):
                    if (
                        right >= self.swarm_number
                        or self.pop[right].positions.shape[0] == 0
                        or self.pop[right].is_converged == 1
                    ):
                        continue
                    distance = float(
                        np.linalg.norm(
                            self.pop[left].best_position - self.pop[right].best_position,
                        ),
                    )
                    if (
                        distance >= self.pop[left].init_radius
                        or distance >= self.pop[right].init_radius
                    ):
                        continue
                    target = (
                        right
                        if float(self.pop[left].best_fitness)
                        > float(self.pop[right].best_fitness)
                        else left
                    )
                    del self.pop[target]
                    self.swarm_number = len(self.pop)
                    removed = True
                    break
                if removed:
                    break

    def _perturb_random_subpopulation(self, problem) -> None:
        if not self.pop:
            return
        index = int(self.rng.integers(0, self.swarm_number))
        swarm = self.pop[index]
        rows = swarm.pbest_positions.shape[0]
        if rows == 0:
            return

        values = evaluate_(problem, swarm.pbest_positions, rows)
        if problem.state.recent_change:
            return
        if problem.state.terminated and not has_complete_values(values, rows):
            return
        if has_complete_values(values, rows):
            swarm.pbest_fitness = self._finite_fitness(values)
            self._reset_best(swarm)

        perturbation = -self.perturbation_range + (
            2.0 * self.perturbation_range * self.rng.random(swarm.positions.shape)
        )
        swarm.velocities = swarm.velocities + perturbation

    def _refresh_radii(self) -> None:
        for swarm in self.pop:
            if swarm.is_converged == 1:
                continue
            if swarm.positions.shape[0] == 0:
                swarm.current_radius = 0.0
            else:
                swarm.current_radius = update_current_radius(swarm)

    def _detect_convergence(self) -> int:
        if not self.pop:
            return 0
        best_id = self._best_swarm_index()
        converged = 0
        for index, swarm in enumerate(self.pop):
            if swarm.current_radius < self.convergence_limit and index != best_id:
                swarm.is_converged = 1
                converged += 1
        return converged

    def _ensure_diversity(self, converged_count: int, problem) -> None:
        survived_particles = sum(
            swarm.positions.shape[0]
            for swarm in self.pop
            if swarm.is_converged == 0
        )
        threshold = self.init_population_size * float(self.parameters["DiversityDegree"])
        if survived_particles >= threshold:
            return

        saved_best: list[np.ndarray] = []
        remaining_to_remove = int(converged_count)
        while remaining_to_remove > 0:
            target = self._first_converged_swarm_index()
            if target is None:
                break
            saved_best.append(self.pop[target].best_position.copy())
            del self.pop[target]
            self.swarm_number = len(self.pop)
            remaining_to_remove -= 1

        saved_positions = (
            np.vstack(saved_best).reshape(-1, self.dimension)
            if saved_best
            else np.empty((0, self.dimension))
        )
        random_count = max(
            0,
            self.init_population_size - survived_particles - saved_positions.shape[0],
        )
        if random_count > 0:
            random_positions = self.bounds.sample(random_count, self.rng)
            add_positions = (
                np.vstack([random_positions, saved_positions])
                if saved_positions.size
                else random_positions
            )
        else:
            add_positions = saved_positions

        if add_positions.shape[0] == 0:
            return
        new_swarms = self.swarm_factory.generate_subpopulations(add_positions, problem)
        if problem.state.recent_change:
            return
        self.pop.extend(new_swarms)
        self.swarm_number = len(self.pop)

    def _regenerate_from_random_positions(self, problem) -> None:
        positions = self.bounds.sample(self.init_population_size, self.rng)
        self.pop = self.swarm_factory.generate_subpopulations(positions, problem)
        self.swarm_number = len(self.pop)

    def _first_converged_swarm_index(self) -> int | None:
        for index, swarm in enumerate(self.pop):
            if swarm.is_converged == 1:
                return index
        return None

    def _best_swarm_index(self) -> int:
        if not self.pop:
            return 0
        return int(
            np.argmax(
                [
                    float("-inf")
                    if swarm.best_fitness is None
                    else float(swarm.best_fitness)
                    for swarm in self.pop
                ],
            ),
        )

    def _reset_best(self, swarm: PSPSOSwarm) -> None:
        if swarm.pbest_fitness.size == 0:
            return
        best_id = self._best_index(swarm.pbest_fitness)
        swarm.best_id = best_id
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()

    def _resolve_swarm_size(self) -> int:
        default_size = self.parameter_specs()["SwarmSize"].default
        swarm_size = self._positive_integer_parameter("SwarmSize")
        population_size = self._positive_integer_parameter("PopulationSize")
        if swarm_size != default_size:
            return swarm_size
        return population_size

    def _positive_integer_parameter(self, name: str) -> int:
        value = self.parameters[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PSPSO parameter {name} must be a positive integer.") from exc
        if not np.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
            raise ValueError(f"PSPSO parameter {name} must be a positive integer.")
        return int(numeric)

    @staticmethod
    def _finite_fitness(values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf).copy()

    @staticmethod
    def _best_index(values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        comparable = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)
        return int(np.argmax(comparable))


class PSPSOHelper:
    """Stateless helper methods used by the PSPSO flow."""

    @staticmethod
    def nearest_neighbor_clusters(
        positions: np.ndarray,
        fitness: np.ndarray,
        swarm_size: int,
    ) -> list[np.ndarray]:
        positions = np.asarray(positions, dtype=float)
        fitness = np.nan_to_num(np.asarray(fitness, dtype=float), nan=-np.inf)
        swarm_size = max(1, int(swarm_size))
        if positions.ndim != 2 or positions.shape[0] == 0:
            return []

        order = np.argsort(fitness)[::-1]
        used = np.zeros(positions.shape[0], dtype=bool)
        clusters: list[np.ndarray] = []
        for seed in order:
            if used[seed]:
                continue
            current = [int(seed)]
            used[seed] = True
            available = np.flatnonzero(~used)
            if available.size:
                distances = np.linalg.norm(positions[available, :] - positions[seed, :], axis=1)
                closest_order = np.argsort(distances)
                take = available[closest_order[: max(0, swarm_size - 1)]]
                for index in take:
                    used[index] = True
                    current.append(int(index))
            clusters.append(np.asarray(current, dtype=int))
        return clusters

    @staticmethod
    def mean_radius(positions: np.ndarray, center: np.ndarray) -> float:
        positions = np.asarray(positions, dtype=float)
        if positions.shape[0] == 0:
            return 0.0
        return float(np.mean(np.linalg.norm(positions - center, axis=1)))
