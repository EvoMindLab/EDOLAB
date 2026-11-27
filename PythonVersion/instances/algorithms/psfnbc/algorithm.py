"""psfNBC algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.psfnbc.state import PsfNBCSwarm


@algorithm_registry.register("psfNBC")
class PsfNBCAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "psfNBC"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                100,
                "integer",
                (0, 1000),
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
            "ratio": ParameterSpec(
                "ratio",
                0.9,
                "numeric",
                (0, 1),
                "Proportion of retained particles used in species formation.",
            ),
            "mem_Maxsize": ParameterSpec(
                "mem_Maxsize",
                50,
                "integer",
                (0, 1000),
                "The maximum number of best solutions stored in memory.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.max_coordinate = float(problem.parameters["MaxCoordinate"])
        self.min_coordinate = float(problem.parameters["MinCoordinate"])
        self.diversity_plus = 1
        self.swarm_number = 1
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.ratio = float(self.parameters["ratio"])
        self.mem_max_size = max(0, int(self.parameters["mem_Maxsize"]))
        self.num_initial = self._initial_count()
        self.max_v = (self.max_coordinate - self.min_coordinate) / 2.0
        self.min_v = -self.max_v
        self.successfault = np.zeros(self.population_size, dtype=float)
        self.mem = np.empty((0, self.dimension), dtype=float)
        self.mem_fit = np.empty(0, dtype=float)
        self.seeds = np.empty(0, dtype=int)
        self.rs = self._species_radius(problem)
        self.pop = self._generate_subpopulation(problem)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return

        lower = self.min_coordinate
        upper = self.max_coordinate
        max_v = (upper - lower) / 2.0
        min_v = -max_v
        self.max_v = max_v
        self.min_v = min_v
        w = float(self.parameters["x"])
        c1 = float(self.parameters["c1"])
        c2 = float(self.parameters["c2"])
        p_max = 10
        self.rs = self._species_radius(problem)

        pop = self.pop
        p_self = pop.positions.copy()
        p_best = pop.pbest_positions.copy()
        p_v = pop.velocities.copy()
        p_fit = pop.fitness.copy()
        p_bestfit = pop.pbest_fitness.copy()
        successfault = self.successfault.copy()

        order = self._stable_descending_order(p_bestfit)
        p_bestfit = p_bestfit[order]
        p_fit = p_fit[order]
        p_v = p_v[order, :]
        p_best = p_best[order, :]
        p_self = p_self[order, :]
        successfault = successfault[order]

        matdis = self._pairwise_distances(p_best)
        memberships, seeds = self._build_species(matdis, self.num_initial)
        self.seeds = seeds.copy()
        self.num_initial = 0

        count = 1
        while memberships.size:
            species = memberships[memberships[:, 1] == count, 0].astype(int)
            if species.size == 0:
                memberships = memberships[memberships[:, 1] != count, :]
                count += 1
                continue

            if self._species_converged(matdis, species):
                seed = int(species[0])
                self._save_memory(p_best[seed, :], float(p_bestfit[seed]))
                for curr in species:
                    self._reinitialize_particle(
                        int(curr),
                        p_self,
                        p_best,
                        p_v,
                        p_fit,
                        p_bestfit,
                        successfault,
                        problem,
                    )
                    self.num_initial += 1
                    if problem.state.recent_change or problem.state.terminated:
                        break
            else:
                if species.size > p_max:
                    for curr in species[p_max:]:
                        self._reinitialize_particle(
                            int(curr),
                            p_self,
                            p_best,
                            p_v,
                            p_fit,
                            p_bestfit,
                            successfault,
                            problem,
                        )
                        self.num_initial += 1
                        if problem.state.recent_change or problem.state.terminated:
                            break
                    species = species[:p_max]

                if not (problem.state.recent_change or problem.state.terminated):
                    seed = int(species[0])
                    for local_index, curr_value in enumerate(species):
                        curr = int(curr_value)
                        use_quantum = (
                            local_index == species.size - 1 and successfault[seed] >= 3
                        )
                        if use_quantum:
                            xtmp = self.quantum(p_best[seed, :], 1.0)
                        elif local_index == 0:
                            p_v[curr, :] = (
                                w
                                * (
                                    p_v[curr, :]
                                    + c1
                                    * self.rng.random(self.dimension)
                                    * (p_best[curr, :] - p_self[curr, :])
                                )
                                + self.rng.standard_normal(self.dimension) * 0.2
                            )
                            xtmp = p_self[curr, :] + p_v[curr, :]
                        else:
                            p_v[curr, :] = w * (
                                p_v[curr, :]
                                + c1
                                * self.rng.random(self.dimension)
                                * (p_best[curr, :] - p_self[curr, :])
                                + c2
                                * self.rng.random(self.dimension)
                                * (p_best[seed, :] - p_self[curr, :])
                            )
                            xtmp = p_self[curr, :] + p_v[curr, :]

                        xtmp = self.boundary_check(xtmp)
                        p_v[curr, :] = xtmp - p_self[curr, :]
                        p_self[curr, :] = xtmp
                        value = self._evaluate_particle(problem, p_self[curr, :])
                        if value is not None:
                            p_fit[curr] = value

                        if use_quantum and value is not None:
                            if p_fit[curr] > p_bestfit[seed]:
                                p_v[curr, :] = p_self[curr, :] - p_best[seed, :]
                            else:
                                p_v[curr, :] = p_best[seed, :] - p_self[curr, :]

                        p_v[curr, :] = np.clip(p_v[curr, :], min_v, max_v)
                        if value is not None and p_fit[curr] > p_bestfit[curr]:
                            p_bestfit[curr] = p_fit[curr]
                            p_best[curr, :] = p_self[curr, :]
                            successfault[curr] = 0
                        else:
                            successfault[curr] += 1

                        if problem.state.recent_change or problem.state.terminated:
                            break

            self._write_population(p_self, p_fit, p_best, p_bestfit, p_v, successfault)
            memberships = memberships[memberships[:, 1] != count, :]
            count += 1
            if problem.state.recent_change or problem.state.terminated:
                return

        best_id = int(np.argmax(np.nan_to_num(self.pop.pbest_fitness, nan=-np.inf)))
        best_value = float(self.pop.pbest_fitness[best_id])
        if best_value > float(self.pop.best_fitness):
            self.pop.best_fitness = best_value
            self.pop.best_position = self.pop.pbest_positions[best_id, :].copy()

    def on_environment_change(self, problem) -> None:
        pop = self.pop
        p_self = pop.positions.copy()
        p_best = pop.pbest_positions.copy()
        p_bestfit = pop.pbest_fitness.copy()
        p_v = pop.velocities.copy()
        p_fit = pop.fitness.copy()

        for seed in self.seeds[: min(5, self.seeds.size)]:
            idx = int(seed)
            if 0 <= idx < p_best.shape[0]:
                self._save_memory(p_best[idx, :], float(p_bestfit[idx]))

        for idx in range(self.population_size):
            value = self._evaluate_particle(problem, p_self[idx, :])
            if value is not None:
                p_fit[idx] = value
            if problem.state.recent_change or problem.state.terminated:
                return

        order = self._stable_descending_order(p_fit)
        p_self = p_self[order, :]
        p_fit = p_fit[order]
        p_v = p_v[order, :]

        if self.mem.shape[0] > 0:
            for idx in range(self.mem.shape[0]):
                value = self._evaluate_particle(problem, self.mem[idx, :])
                if value is not None:
                    self.mem_fit[idx] = value
                if problem.state.recent_change or problem.state.terminated:
                    return
            mem_order = self._stable_descending_order(self.mem_fit)
            self.mem = self.mem[mem_order, :]
            self.mem_fit = self.mem_fit[mem_order]

            candidates: list[tuple[np.ndarray, float]] = [
                (self.mem[0, :].copy(), float(self.mem_fit[0]))
            ]
            for _ in range(2):
                sample = self.quantum(self.mem[0, :], 0.1 * self.rs)
                value = self._evaluate_particle(problem, sample)
                if value is None:
                    return
                candidates.append((sample, value))
                if problem.state.recent_change or problem.state.terminated:
                    return

            for target, (sample, value) in zip(
                range(self.population_size - 1, -1, -1),
                candidates,
            ):
                p_self[target, :] = sample
                p_fit[target] = value
                p_v[target, :] = 0.0

        p_bestfit = p_fit.copy()
        p_best = p_self.copy()
        self.successfault = np.zeros(self.population_size, dtype=float)
        self.num_initial = self._initial_count()

        pop.positions = p_self
        pop.pbest_positions = p_best
        pop.pbest_fitness = p_bestfit
        pop.fitness = p_fit
        pop.velocities = p_v
        best_id = int(np.argmax(np.nan_to_num(pop.pbest_fitness, nan=-np.inf)))
        pop.best_fitness = float(pop.pbest_fitness[best_id])
        pop.best_position = pop.pbest_positions[best_id, :].copy()

    def boundary_check(self, x: np.ndarray) -> np.ndarray:
        return PsfNBCHelper.boundary_check(
            np.asarray(x, dtype=float).copy(),
            self.min_coordinate,
            self.max_coordinate,
        )

    def quantum(self, core: np.ndarray, length: float) -> np.ndarray:
        direction = self.rng.standard_normal(self.dimension)
        norm = float(np.linalg.norm(direction))
        if norm != 0.0:
            direction = direction * (float(length) / norm)
        return self.boundary_check(np.asarray(core, dtype=float) + direction)

    def _generate_subpopulation(self, problem) -> PsfNBCSwarm:
        positions = self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random((self.population_size, self.dimension))
        )
        fitness = evaluate_(problem, positions, self.population_size)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        best_id = int(np.argmax(np.nan_to_num(pbest_fitness, nan=-np.inf)))
        velocities = self.rng.random((self.population_size, self.dimension)) * (
            self.max_v - self.min_v
        ) + self.min_v
        return PsfNBCSwarm(
            gbest_past_environment=np.full(self.dimension, np.nan),
            velocities=velocities,
            shifts=[],
            positions=positions,
            fitness=fitness,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_fitness=float(pbest_fitness[best_id]),
            best_position=pbest_positions[best_id, :].copy(),
        )

    def _reinitialize_particle(
        self,
        idx: int,
        p_self: np.ndarray,
        p_best: np.ndarray,
        p_v: np.ndarray,
        p_fit: np.ndarray,
        p_bestfit: np.ndarray,
        successfault: np.ndarray,
        problem,
    ) -> None:
        p_self[idx, :] = self.min_coordinate + (
            (self.max_coordinate - self.min_coordinate)
            * self.rng.random(self.dimension)
        )
        p_best[idx, :] = p_self[idx, :]
        p_v[idx, :] = self.rng.random(self.dimension) * (
            self.max_v - self.min_v
        ) + self.min_v
        value = self._evaluate_particle(problem, p_self[idx, :])
        if value is not None:
            p_fit[idx] = value
            p_bestfit[idx] = value
        successfault[idx] = 0

    def _evaluate_particle(self, problem, position: np.ndarray) -> float | None:
        values = problem.evaluate(np.asarray(position, dtype=float).reshape(1, -1))
        if values.size == 0:
            return None
        return float(values[0])

    def _write_population(
        self,
        p_self: np.ndarray,
        p_fit: np.ndarray,
        p_best: np.ndarray,
        p_bestfit: np.ndarray,
        p_v: np.ndarray,
        successfault: np.ndarray,
    ) -> None:
        self.pop.positions = p_self
        self.pop.fitness = p_fit
        self.pop.pbest_positions = p_best
        self.pop.pbest_fitness = p_bestfit
        self.pop.velocities = p_v
        self.successfault = successfault

    def _save_memory(self, position: np.ndarray, fitness: float) -> None:
        if self.mem_max_size <= 0:
            return
        position = np.asarray(position, dtype=float).reshape(1, self.dimension)
        if self.mem.shape[0] < self.mem_max_size:
            self.mem = np.vstack([self.mem, position])
            self.mem_fit = np.append(self.mem_fit, float(fitness))
            return
        distances = np.sum((self.mem - position) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        if self.mem_fit[nearest] < fitness:
            self.mem[nearest, :] = position[0, :]
            self.mem_fit[nearest] = float(fitness)

    def _build_species(
        self,
        matdis: np.ndarray,
        num_initial: int | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return PsfNBCHelper.build_species(
            matdis,
            self._safe_initial_count(num_initial),
            self.rng,
        )

    def _pairwise_distances(self, samples: np.ndarray) -> np.ndarray:
        delta = samples[:, None, :] - samples[None, :, :]
        return np.linalg.norm(delta, axis=2)

    def _species_converged(self, matdis: np.ndarray, species: np.ndarray) -> bool:
        if species.size <= 1:
            return False
        seed = int(species[0])
        return bool(float(np.mean(matdis[seed, species[1:]])) <= 0.0001)

    def _initial_count(self) -> int:
        return self._safe_initial_count(
            (1.0 - float(self.parameters["ratio"])) * self.population_size
        )

    def _safe_initial_count(self, value: int | float | None) -> int:
        if value is None:
            value = self.num_initial
        return max(0, min(self.population_size, int(np.floor(float(value) + 1e-12))))

    def _stable_descending_order(self, values: np.ndarray) -> np.ndarray:
        return PsfNBCHelper.stable_descending_order(values)

    def _species_radius(self, problem) -> float:
        peak_number = 1
        if hasattr(problem, "parameters"):
            peak_number = int(problem.parameters.get("PeakNumber", 1) or 1)
        peak_number = max(1, peak_number)
        return float(
            (self.max_coordinate - self.min_coordinate)
            * np.sqrt(self.dimension)
            / (2.0 * peak_number ** (1.0 / self.dimension))
        )


class PsfNBCHelper:
    """Stateless helpers used by psfNBC and its regression tests."""

    @staticmethod
    def build_species(
        matdis: np.ndarray,
        num_initial: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        matdis = np.asarray(matdis, dtype=float)
        n = int(matdis.shape[0])
        if n == 0:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)

        fu = 2.25
        fl = 1.25
        factor = fl + float(rng.random()) * (fu - fl)
        num_initial = max(0, min(n, int(num_initial)))
        partial = n - num_initial
        if num_initial > 0:
            partial += 1
        partial = max(1, min(n, int(partial)))

        parents = np.full(partial, -1, dtype=int)
        distances = np.zeros(partial, dtype=float)
        for idx in range(1, partial):
            previous = matdis[idx, :idx]
            parent = int(np.argmin(previous))
            parents[idx] = parent
            distances[idx] = float(previous[parent])

        if partial > 1:
            mean_distance = factor * float(np.mean(distances[1:]))
            cut = distances > mean_distance
            parents[cut] = -1
            distances[cut] = 0.0

        seeds = np.flatnonzero(parents == -1)
        if seeds.size == 0:
            seeds = np.array([0], dtype=int)

        memberships = np.zeros((n, 2), dtype=int)
        memberships[:, 0] = np.arange(n, dtype=int)
        for idx in range(n):
            nearest_seed = seeds[int(np.argmin(matdis[seeds, idx]))]
            memberships[idx, 1] = int(nearest_seed)

        for label, seed in enumerate(seeds, start=1):
            memberships[memberships[:, 1] == seed, 1] = label

        return memberships, seeds.astype(int)

    @staticmethod
    def stable_descending_order(values: np.ndarray) -> np.ndarray:
        sortable = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)
        return np.argsort(-sortable, kind="stable")

    @staticmethod
    def boundary_check(
        x: np.ndarray,
        lower_bound: float,
        upper_bound: float,
    ) -> np.ndarray:
        if upper_bound <= lower_bound:
            return np.full_like(x, lower_bound, dtype=float)
        while bool(np.any(x < lower_bound) or np.any(x > upper_bound)):
            x = np.where(x < lower_bound, 2.0 * lower_bound - x, x)
            x = np.where(x > upper_bound, 2.0 * upper_bound - x, x)
        return x
