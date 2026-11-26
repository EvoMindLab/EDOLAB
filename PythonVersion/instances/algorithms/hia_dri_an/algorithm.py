"""HIA_DRI_AN algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.hia_dri_an.state import HiaDriAnCMAState, HiaDriAnPopulation


@algorithm_registry.register("HIA_DRI_AN")
class HiaDriAnAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "HIA_DRI_AN"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                1000,
                "integer",
                (10, 5000),
                "Total population size (N).",
            ),
            "AccuracyThreshold": ParameterSpec(
                "AccuracyThreshold",
                2,
                "numeric",
                (0.1, 20),
                "Accuracy threshold (E) controlling alpha and diversity response.",
            ),
            "AlphaMin": ParameterSpec(
                "AlphaMin",
                2,
                "numeric",
                (0.1, 10),
                "Lower limit of niche division parameter alpha.",
            ),
            "AlphaMax": ParameterSpec(
                "AlphaMax",
                5,
                "numeric",
                (0.1, 20),
                "Upper limit of niche division parameter alpha.",
            ),
            "AROrder": ParameterSpec(
                "AROrder",
                3,
                "integer",
                (1, 10),
                "Order of the autoregression prediction model.",
            ),
            "DetectorRatio": ParameterSpec(
                "DetectorRatio",
                0.05,
                "numeric",
                (0.01, 0.5),
                "Fraction of population used as fixed change detectors.",
            ),
            "K": ParameterSpec(
                "K",
                0.729,
                "numeric",
                (0, 2),
                "PSO constriction factor.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Cognitive acceleration coefficient.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Social acceleration coefficient.",
            ),
            "GuideSwitchRatio": ParameterSpec(
                "GuideSwitchRatio",
                0.5,
                "numeric",
                (0, 1),
                "Fraction of MaxGenPSO where the guide switches from slbest to sgbest.",
            ),
            "CMAESBudgetRatio": ParameterSpec(
                "CMAESBudgetRatio",
                0.2,
                "numeric",
                (0, 0.5),
                "Fraction of remaining environment evaluations reserved for CMA-ES.",
            ),
            "BalancingTime": ParameterSpec(
                "BalancingTime",
                0.25,
                "numeric",
                (0, 1),
                "Fraction of MaxGenPSO where species balancing is triggered.",
            ),
            "MaxSpeciesSize": ParameterSpec(
                "MaxSpeciesSize",
                20,
                "integer",
                (5, 100),
                "Maximum allowed number of particles in one species.",
            ),
            "CMAESMaxSize": ParameterSpec(
                "CMAESMaxSize",
                30,
                "integer",
                (5, 100),
                "Maximum number of individuals per species participating in CMA-ES.",
            ),
            "CMAESInitSigma": ParameterSpec(
                "CMAESInitSigma",
                0.2,
                "numeric",
                (0.01, 1),
                "Initial CMA-ES step size.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.bounds = problem.bounds
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.population_size = max(1, int(self.parameters["PopulationSize"]))
        self.accuracy_threshold = float(self.parameters["AccuracyThreshold"])
        self.alpha_min = float(self.parameters["AlphaMin"])
        self.alpha_max = float(self.parameters["AlphaMax"])
        self.ar_order = max(1, int(self.parameters["AROrder"]))
        self.detector_ratio = float(self.parameters["DetectorRatio"])
        self.k = float(self.parameters["K"])
        self.c1 = float(self.parameters["c1"])
        self.c2 = float(self.parameters["c2"])
        self.guide_switch_ratio = float(self.parameters["GuideSwitchRatio"])
        self.cmaes_budget_ratio = float(self.parameters["CMAESBudgetRatio"])
        self.balancing_time = float(self.parameters["BalancingTime"])
        self.max_species_size = max(1, int(self.parameters["MaxSpeciesSize"]))
        if self.dimension < 3:
            self.cmaes_max_size = 20
            self.cmaes_init_sigma = 0.1
        else:
            self.cmaes_max_size = 30
            self.cmaes_init_sigma = 0.2

        self.phase = 1
        self.gen_pso = 0
        self.cmaes_used_fe = 0
        self.env_start_fe = 0
        self.ar = 0.0
        self.alpha = self.alpha_max
        self.history_optima: list[np.ndarray] = []
        self.history_best_set: list[np.ndarray] = []
        self.environment_count = 0
        self.last_prediction_set = np.empty((0, self.dimension), dtype=float)
        self.history_detector_fitness: list[np.ndarray] = []
        self.cma_state: list[HiaDriAnCMAState] = []
        self.swarm_number = 1

        self.pop = self._sub_population_generator(problem)
        num_detectors = max(1, int(np.ceil(self.detector_ratio * self.population_size)))
        self.detector_positions = self._random_positions(num_detectors)
        detector_fitness = evaluate_(problem, self.detector_positions, num_detectors)
        self.history_detector_fitness.append(detector_fitness.copy())
        self.env_start_fe = int(problem.state.fe)
        if problem.state.recent_change or problem.state.terminated:
            return
        self._configure_phase_budgets(problem)

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        if self.phase == 1 and self.max_gen_pso <= 0:
            self.phase = 2
            if not self.cma_state:
                self._init_cmaes_states()
        if self.phase == 1:
            self._phase1_nbnc_pso(problem)
        elif self.phase == 2:
            self._phase2_cmaes(problem)

    def on_environment_change(self, problem) -> None:
        if not hasattr(self, "pop") or problem.state.terminated:
            return
        self._change_reaction(problem)

    def _configure_phase_budgets(self, problem) -> None:
        change_frequency = int(problem.change_frequency)
        remaining_fe = change_frequency - (int(problem.state.fe) % change_frequency)
        if remaining_fe == change_frequency and int(problem.state.fe) == 0:
            remaining_fe = change_frequency
        remaining_fe = max(0, remaining_fe)
        self.fes_per_env = change_frequency
        self.environment_remaining_fe = remaining_fe
        self.cmaes_budget_fe = int(np.floor(self.cmaes_budget_ratio * remaining_fe))
        self.pso_budget_fe = max(0, remaining_fe - self.cmaes_budget_fe)
        self.max_gen_pso = int(np.floor(self.pso_budget_fe / self.population_size))

    def _sub_population_generator(self, problem) -> HiaDriAnPopulation:
        positions = self._random_positions(self.population_size)
        velocities = np.zeros((self.population_size, self.dimension), dtype=float)
        fitness = evaluate_(problem, positions, self.population_size)
        if problem.state.recent_change:
            fitness = np.full(self.population_size, -np.inf, dtype=float)
        else:
            fitness = np.nan_to_num(fitness, nan=-np.inf)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        best_id = int(np.argmax(pbest_fitness)) if pbest_fitness.size else 0
        best_position = pbest_positions[best_id, :].copy()
        best_fitness = float(pbest_fitness[best_id])
        return HiaDriAnPopulation(
            positions=positions,
            fitness=fitness,
            metadata={},
            velocities=velocities,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            best_position=best_position,
            best_fitness=best_fitness,
            slbest_positions=np.full((self.population_size, self.dimension), np.nan),
            sgbest_positions=np.full((self.population_size, self.dimension), np.nan),
            species_id=np.zeros(self.population_size, dtype=int),
            best_id=best_id,
        )

    def _phase1_nbnc_pso(self, problem) -> None:
        pop = self.pop
        n = pop.positions.shape[0]
        self.gen_pso += 1
        self._assign_nbnc_species_and_guides()

        balancing_gen = int(np.floor(self.balancing_time * self.max_gen_pso))
        if self.gen_pso == balancing_gen and balancing_gen > 0:
            self._balance_species()

        switch_gen = int(np.floor(self.guide_switch_ratio * self.max_gen_pso))
        if self.gen_pso <= switch_gen:
            guide = pop.slbest_positions
        else:
            guide = pop.sgbest_positions
        guide = np.where(np.isnan(guide), pop.positions, guide)
        r1 = self.rng.random((n, self.dimension))
        r2 = self.rng.random((n, self.dimension))
        pop.velocities = self.k * (
            pop.velocities
            + self.c1 * r1 * (pop.pbest_positions - pop.positions)
            + self.c2 * r2 * (guide - pop.positions)
        )
        pop.positions = pop.positions + pop.velocities
        upper = pop.positions > self.max_coordinate
        lower = pop.positions < self.min_coordinate
        pop.positions = self.bounds.clip(pop.positions)
        pop.velocities[upper | lower] = 0.0

        values = evaluate_(problem, pop.positions, n)
        if problem.state.recent_change or problem.state.terminated:
            return
        pop.fitness = np.nan_to_num(values, nan=-np.inf)
        improved = pop.fitness > pop.pbest_fitness
        pop.pbest_fitness[improved] = pop.fitness[improved]
        pop.pbest_positions[improved, :] = pop.positions[improved, :]
        self._refresh_global_best()

        if self.gen_pso >= self.max_gen_pso:
            self.phase = 2
            self._init_cmaes_states()

    def _phase2_cmaes(self, problem) -> None:
        n = self.pop.positions.shape[0]
        if not self.cma_state:
            self._evaluate_detector_subset(problem)
            return

        remain_budget = self.cmaes_budget_fe - self.cmaes_used_fe
        if remain_budget <= 0:
            self._evaluate_detector_subset(problem)
            return

        did_evaluate = False
        for species_index, state in enumerate(list(self.cma_state), start=1):
            if state.converged:
                continue
            if remain_budget < state.lambda_:
                break
            offspring = self._sample_cma_offspring(state)
            values = evaluate_(problem, offspring, state.lambda_)
            if problem.state.recent_change or problem.state.terminated:
                return
            fitness = np.nan_to_num(values, nan=-np.inf)
            self.cmaes_used_fe += state.lambda_
            remain_budget -= state.lambda_
            did_evaluate = True
            self._update_cma_state(state, offspring, fitness)
            self._write_cma_best_to_population(species_index, state)

        if not did_evaluate:
            self._evaluate_detector_subset(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
        self._refresh_global_best()

    def _assign_nbnc_species_and_guides(self) -> None:
        pop = self.pop
        positions = np.asarray(pop.positions, dtype=float)
        fitness = np.asarray(pop.fitness, dtype=float)
        n = positions.shape[0]
        distances = self._pairwise_distances(positions)
        mean_dis = self._mean_nearest_distance(distances)
        delta_alpha = self.alpha_max - self.alpha_min
        if self.ar < self.accuracy_threshold:
            alpha_ini = self.alpha_max - (delta_alpha / self.accuracy_threshold) * self.ar
        else:
            alpha_ini = self.alpha_min
        if self.gen_pso > 1 and self.max_gen_pso > 0:
            self.alpha = alpha_ini + self.gen_pso / self.max_gen_pso
        else:
            self.alpha = alpha_ini
        slbest, edges = self._compute_slbest_links(
            positions,
            fitness,
            threshold=float(self.alpha * mean_dis),
            distances=distances,
            return_edges=True,
        )
        species_id = self._species_from_edges(n, edges)
        species_id = self._dominance_merge(species_id, positions, fitness)
        pop.species_id = species_id
        sgbest = self._species_seed_positions(species_id, positions, fitness)
        pop.slbest_positions = positions[slbest, :].copy()
        pop.sgbest_positions = sgbest[species_id - 1, :].copy()

    def _balance_species(self) -> None:
        pop = self.pop
        species_id = pop.species_id.copy()
        if species_id.size == 0 or int(np.max(species_id)) <= 0:
            return
        distances = self._pairwise_distances(pop.positions)
        mean_dis = self._mean_nearest_distance(distances)
        species_count = int(np.max(species_id))
        seeds = self._species_seed_positions(species_id, pop.positions, pop.fitness)
        sizes = np.asarray([np.count_nonzero(species_id == s) for s in range(1, species_count + 1)])
        relocated: list[int] = []
        species = 1
        while species <= species_count:
            while sizes[species - 1] > self.max_species_size:
                members = np.flatnonzero(species_id == species)
                ordered = members[np.argsort(pop.fitness[members])[::-1]]
                idx = int(ordered[self.max_species_size])
                target_candidates = [
                    candidate
                    for candidate in range(species_count)
                    if candidate != species - 1 and sizes[candidate] < self.max_species_size
                ]
                if target_candidates:
                    smallest_idx = min(target_candidates, key=lambda candidate: sizes[candidate])
                    smallest = int(smallest_idx) + 1
                    target_seed = seeds[smallest_idx, :].copy()
                else:
                    species_count += 1
                    smallest = species_count
                    target_seed = pop.positions[idx, :].copy()
                    sizes = np.append(sizes, 0)
                    seeds = np.vstack([seeds, target_seed])
                species_id[idx] = smallest
                pop.positions[idx, :] = target_seed + (
                    2 * self.rng.random(self.dimension) - 1
                ) * (self.alpha * mean_dis)
                pop.positions[idx, :] = self.bounds.clip(pop.positions[idx, :])
                pop.pbest_positions[idx, :] = target_seed
                pop.pbest_fitness[idx] = -np.inf
                pop.fitness[idx] = -np.inf
                pop.velocities[idx, :] = 0.0
                sizes[species - 1] -= 1
                sizes[smallest - 1] += 1
                relocated.append(idx)
            species += 1
        pop.species_id = species_id
        if relocated:
            self._refresh_guides_for_existing_species()

    def _refresh_guides_for_existing_species(self) -> None:
        pop = self.pop
        distances = self._pairwise_distances(pop.positions)
        threshold = self.alpha * self._mean_nearest_distance(distances)
        slbest = self._compute_slbest_links(
            pop.positions,
            pop.fitness,
            threshold=threshold,
            distances=distances,
        )
        pop.slbest_positions = pop.positions[slbest, :].copy()
        seeds = self._species_seed_positions(pop.species_id, pop.positions, pop.fitness)
        pop.sgbest_positions = seeds[pop.species_id - 1, :].copy()

    def _init_cmaes_states(self) -> None:
        species_id = np.asarray(self.pop.species_id, dtype=int).copy()
        if species_id.size == 0 or int(np.max(species_id)) <= 0:
            species_id = np.ones(self.pop.positions.shape[0], dtype=int)
        species_id = self._merge_tiny_species(species_id)
        self.pop.species_id = species_id
        self.cma_state = []
        species_count = int(np.max(species_id)) if species_id.size else 0
        for species in range(1, species_count + 1):
            members = np.flatnonzero(species_id == species)
            if members.size == 0:
                continue
            lambda_raw = min(int(members.size), self.cmaes_max_size)
            lambda_ = max(lambda_raw, 4)
            mu = max(1, int(np.floor(lambda_ / 2)))
            fits = self.pop.pbest_fitness[members]
            best_local = int(np.argmax(fits))
            best_x = self.pop.pbest_positions[members[best_local], :].copy()
            best_fit = float(fits[best_local])
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
            weights = weights / np.sum(weights)
            mueff = float(1.0 / np.sum(weights**2))
            cs_param = float((mueff + 2) / (self.dimension + mueff + 5))
            damps = float(
                1
                + 2 * max(0.0, np.sqrt((mueff - 1) / (self.dimension + 1)) - 1)
                + cs_param
            )
            cc = float((4 + mueff / self.dimension) / (self.dimension + 4 + 2 * mueff / self.dimension))
            c1 = float(2 / ((self.dimension + 1.3) ** 2 + mueff))
            cmu = float(
                min(
                    1 - c1,
                    2 * (mueff - 2 + 1 / mueff) / ((self.dimension + 2) ** 2 + mueff),
                )
            )
            chi_n = float(
                np.sqrt(self.dimension)
                * (1 - 1 / (4 * self.dimension) + 1 / (21 * self.dimension**2))
            )
            self.cma_state.append(
                HiaDriAnCMAState(
                    lambda_=lambda_,
                    mu=mu,
                    mean=best_x.copy(),
                    sigma=float(self.cmaes_init_sigma),
                    covariance=np.eye(self.dimension),
                    b_matrix=np.eye(self.dimension),
                    d_vector=np.ones(self.dimension),
                    pc=np.zeros(self.dimension),
                    ps=np.zeros(self.dimension),
                    gen_count=0,
                    converged=False,
                    best_x=best_x.copy(),
                    best_fit=best_fit,
                    weights=weights,
                    mueff=mueff,
                    cs_param=cs_param,
                    damps=damps,
                    cc=cc,
                    c1=c1,
                    cmu=cmu,
                    chi_n=chi_n,
                )
            )

    def _change_reaction(self, problem) -> None:
        current_optima = self._current_species_optima()
        self.environment_count += 1
        env_idx = self.environment_count
        self.history_optima.append(current_optima.copy())
        self.history_best_set.append(current_optima.copy())

        detector_fitness = evaluate_(problem, self.detector_positions, self.detector_positions.shape[0])
        if problem.state.recent_change or problem.state.terminated:
            return
        self._store_detector_fitness(env_idx, detector_fitness)

        if self.last_prediction_set.size and current_optima.size:
            distances = np.linalg.norm(
                current_optima[:, None, :] - self.last_prediction_set[None, :, :],
                axis=2,
            )
            self.ar = float(np.mean(np.min(distances, axis=1)))
        else:
            self.ar = 0.0

        merged_set = self._prediction_set_for_next_environment(env_idx)
        self.last_prediction_set = merged_set.copy()
        self.pop.positions = self._build_reaction_population(merged_set)
        values = evaluate_(problem, self.pop.positions, self.population_size)
        if problem.state.recent_change or problem.state.terminated:
            return
        self.pop.fitness = np.nan_to_num(values, nan=-np.inf)
        self.pop.pbest_positions = self.pop.positions.copy()
        self.pop.pbest_fitness = self.pop.fitness.copy()
        self.pop.velocities = np.zeros((self.population_size, self.dimension), dtype=float)
        self._refresh_global_best()
        self.pop.species_id = np.zeros(self.population_size, dtype=int)
        self.pop.slbest_positions = np.full((self.population_size, self.dimension), np.nan)
        self.pop.sgbest_positions = np.full((self.population_size, self.dimension), np.nan)
        self.phase = 1
        self.gen_pso = 0
        self.cmaes_used_fe = 0
        self.env_start_fe = int(problem.state.fe)
        self.cma_state = []
        self._configure_phase_budgets(problem)

    def _prediction_set_for_next_environment(self, env_idx: int) -> np.ndarray:
        sets: list[np.ndarray] = []
        if env_idx >= 2:
            sol_t = self.history_optima[env_idx - 1]
            sol_t1 = self.history_optima[env_idx - 2]
            predicted = np.zeros_like(sol_t)
            for idx, row in enumerate(sol_t):
                nearest = int(np.argmin(np.linalg.norm(sol_t1 - row, axis=1)))
                predicted[idx, :] = row + (row - sol_t1[nearest, :])
            sets.append(self.bounds.clip(predicted))
        if env_idx >= self.ar_order + 1:
            sets.append(self._autoregression_prediction(env_idx))
        if env_idx >= 1:
            new_fitness = self.history_detector_fitness[env_idx]
            best_env = -1
            best_similarity = np.inf
            for past_idx in range(env_idx):
                if past_idx >= len(self.history_detector_fitness):
                    continue
                stored = self.history_detector_fitness[past_idx]
                if stored.size != new_fitness.size:
                    continue
                similarity = float(np.linalg.norm(new_fitness - stored))
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_env = past_idx
            if best_env >= 0:
                sets.append(self.history_optima[best_env])
        return self._deduplicate_rows(sets)

    def _autoregression_prediction(self, env_idx: int) -> np.ndarray:
        sol_t = self.history_optima[env_idx - 1]
        output = np.zeros_like(sol_t)
        p = self.ar_order
        for opt_idx, row in enumerate(sol_t):
            trajectory = np.zeros((env_idx, self.dimension), dtype=float)
            trajectory[env_idx - 1, :] = row
            for t in range(env_idx - 2, -1, -1):
                sol_prev = self.history_optima[t]
                nearest = int(np.argmin(np.linalg.norm(sol_prev - trajectory[t + 1, :], axis=1)))
                trajectory[t, :] = sol_prev[nearest, :]
            length = min(env_idx, 10)
            if length < p + 1:
                output[opt_idx, :] = row
                continue
            start = env_idx - length
            recent = trajectory[start:env_idx, :]
            n_rows = length - p
            for dim in range(self.dimension):
                psi = np.zeros(n_rows, dtype=float)
                phi = np.zeros((n_rows, p), dtype=float)
                for ridx in range(n_rows):
                    t_idx = length - ridx - 1
                    psi[ridx] = recent[t_idx, dim]
                    for j in range(p):
                        phi[ridx, j] = recent[t_idx - j - 1, dim]
                coeff = np.linalg.pinv(phi) @ psi
                history = np.asarray([trajectory[env_idx - j - 1, dim] for j in range(p)])
                pred = float(np.dot(coeff, history))
                residuals = psi - phi @ coeff
                sigma = float(np.sqrt(max(float(np.mean(residuals**2)), 0.0)))
                output[opt_idx, dim] = pred + self.rng.normal(0.0, sigma)
        return self.bounds.clip(output)

    def _build_reaction_population(self, merged_set: np.ndarray) -> np.ndarray:
        if merged_set.size == 0:
            return self._random_positions(self.population_size)
        n_set = merged_set.shape[0]
        ratio = min(self.ar / self.accuracy_threshold, 1.0) if self.accuracy_threshold > 0 else 1.0
        copies = max(0, int(np.floor((self.population_size * (1 - ratio) - n_set) / max(n_set, 1))))
        c_rows: list[np.ndarray] = []
        for row in merged_set:
            for _ in range(copies):
                c_rows.append(row + self.rng.normal(0.0, 0.5, self.dimension))
        c_set = self.bounds.clip(np.vstack(c_rows)) if c_rows else np.empty((0, self.dimension))
        used = n_set + c_set.shape[0]
        random_count = max(0, self.population_size - used)
        random_set = self._random_positions(random_count)
        new_pop = np.vstack([merged_set, c_set, random_set])
        if new_pop.shape[0] > self.population_size:
            new_pop = new_pop[: self.population_size, :]
        elif new_pop.shape[0] < self.population_size:
            new_pop = np.vstack(
                [new_pop, self._random_positions(self.population_size - new_pop.shape[0])]
            )
        return self.bounds.clip(new_pop)

    def _sample_cma_offspring(self, state: HiaDriAnCMAState) -> np.ndarray:
        z = self.rng.standard_normal((self.dimension, state.lambda_))
        steps = state.b_matrix @ (state.d_vector[:, None] * z)
        offspring = state.mean.reshape(1, -1) + state.sigma * steps.T
        return self.bounds.clip(offspring)

    def _update_cma_state(
        self,
        state: HiaDriAnCMAState,
        offspring: np.ndarray,
        fitness: np.ndarray,
    ) -> None:
        order = np.argsort(fitness)[::-1]
        sorted_fitness = fitness[order]
        sorted_offspring = offspring[order, :]
        if abs(float(sorted_fitness[0]) - float(sorted_fitness[-1])) < 1e-7:
            state.converged = True
        selected = sorted_offspring[: state.mu, :]
        old_mean = state.mean.copy()
        state.mean = state.weights @ selected
        sigma = max(float(state.sigma), np.finfo(float).eps)
        y = (state.mean - old_mean) / sigma
        inv_step = state.b_matrix @ ((state.b_matrix.T @ y) / state.d_vector)
        state.ps = (1 - state.cs_param) * state.ps + np.sqrt(
            state.cs_param * (2 - state.cs_param) * state.mueff
        ) * inv_step
        denom = np.sqrt(1 - (1 - state.cs_param) ** (2 * (state.gen_count + 1)))
        hsig = (
            np.linalg.norm(state.ps) / max(denom, np.finfo(float).eps) / state.chi_n
            < 1.4 + 2 / (self.dimension + 1)
        )
        state.pc = (1 - state.cc) * state.pc + float(hsig) * np.sqrt(
            state.cc * (2 - state.cc) * state.mueff
        ) * y
        artmp = (selected - old_mean.reshape(1, -1)).T / sigma
        weighted = artmp @ np.diag(state.weights) @ artmp.T
        state.covariance = (
            (1 - state.c1 - state.cmu + (1 - float(hsig)) * state.c1 * state.cc * (2 - state.cc))
            * state.covariance
            + state.c1 * np.outer(state.pc, state.pc)
            + state.cmu * weighted
        )
        state.sigma = float(
            state.sigma
            * np.exp((state.cs_param / state.damps) * (np.linalg.norm(state.ps) / state.chi_n - 1))
        )
        state.covariance = np.triu(state.covariance) + np.triu(state.covariance, 1).T
        eig_values, eig_vectors = np.linalg.eigh(state.covariance)
        state.d_vector = np.sqrt(np.maximum(eig_values, np.finfo(float).eps))
        state.b_matrix = eig_vectors
        state.gen_count += 1
        if float(sorted_fitness[0]) > state.best_fit:
            state.best_fit = float(sorted_fitness[0])
            state.best_x = sorted_offspring[0, :].copy()

    def _write_cma_best_to_population(self, species_index: int, state: HiaDriAnCMAState) -> None:
        members = np.flatnonzero(self.pop.species_id == species_index)
        if members.size == 0:
            return
        best_member = int(members[np.argmax(self.pop.pbest_fitness[members])])
        if state.best_fit > self.pop.pbest_fitness[best_member]:
            self.pop.positions[best_member, :] = state.best_x
            self.pop.fitness[best_member] = state.best_fit
            self.pop.pbest_fitness[best_member] = state.best_fit
            self.pop.pbest_positions[best_member, :] = state.best_x

    def _evaluate_detector_subset(self, problem) -> None:
        n = self.pop.positions.shape[0]
        count = max(1, int(np.ceil(self.detector_ratio * n)))
        count = min(count, n)
        indices = self.rng.choice(n, size=count, replace=False)
        values = evaluate_(problem, self.pop.positions[indices, :], count)
        if problem.state.recent_change or problem.state.terminated:
            return
        self.pop.fitness[indices] = np.nan_to_num(values, nan=-np.inf)

    def _current_species_optima(self) -> np.ndarray:
        species_id = np.asarray(self.pop.species_id, dtype=int)
        if species_id.size and int(np.max(species_id)) > 0:
            rows = []
            for species in range(1, int(np.max(species_id)) + 1):
                members = np.flatnonzero(species_id == species)
                if members.size == 0:
                    continue
                best = int(members[np.argmax(self.pop.pbest_fitness[members])])
                rows.append(self.pop.pbest_positions[best, :].copy())
            if rows:
                return np.vstack(rows)
        return self.pop.best_position.reshape(1, self.dimension).copy()

    def _store_detector_fitness(self, env_idx: int, values: np.ndarray) -> None:
        while len(self.history_detector_fitness) <= env_idx:
            self.history_detector_fitness.append(np.empty(0, dtype=float))
        self.history_detector_fitness[env_idx] = values.copy()

    def _deduplicate_rows(self, sets: list[np.ndarray]) -> np.ndarray:
        rows = [np.asarray(row, dtype=float).reshape(-1, self.dimension) for row in sets if row.size]
        if not rows:
            return np.empty((0, self.dimension), dtype=float)
        merged = self.bounds.clip(np.vstack(rows))
        keep = np.ones(merged.shape[0], dtype=bool)
        for idx in range(1, merged.shape[0]):
            for prev in range(idx):
                if keep[prev] and np.linalg.norm(merged[idx, :] - merged[prev, :]) < 1e-6:
                    keep[idx] = False
                    break
        return merged[keep, :]

    def _refresh_global_best(self) -> None:
        best_id = int(np.argmax(self.pop.pbest_fitness))
        self.pop.best_id = best_id
        self.pop.best_fitness = float(self.pop.pbest_fitness[best_id])
        self.pop.best_position = self.pop.pbest_positions[best_id, :].copy()

    def _merge_tiny_species(self, species_id: np.ndarray) -> np.ndarray:
        species_id = species_id.copy()
        species_count = int(np.max(species_id)) if species_id.size else 0
        if species_count <= 1:
            return species_id
        seeds = self._species_seed_positions(species_id, self.pop.positions, self.pop.fitness)
        sizes = np.asarray([np.count_nonzero(species_id == s) for s in range(1, species_count + 1)])
        for species in range(1, species_count + 1):
            if sizes[species - 1] >= 2:
                continue
            candidates = [s for s in range(1, species_count + 1) if s != species and sizes[s - 1] >= 2]
            if not candidates:
                continue
            distances = [np.linalg.norm(seeds[species - 1, :] - seeds[s - 1, :]) for s in candidates]
            target = candidates[int(np.argmin(distances))]
            species_id[species_id == species] = target
            sizes[target - 1] += sizes[species - 1]
            sizes[species - 1] = 0
        _, compact = np.unique(species_id, return_inverse=True)
        return compact.astype(int) + 1

    def _dominance_merge(
        self,
        species_id: np.ndarray,
        positions: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        species_count = int(np.max(species_id)) if species_id.size else 0
        if species_count <= 1:
            return species_id
        seeds = self._species_seed_positions(species_id, positions, fitness)
        min_fit = np.asarray(
            [np.min(fitness[species_id == s]) for s in range(1, species_count + 1)],
            dtype=float,
        )
        max_fit = np.asarray(
            [np.max(fitness[species_id == s]) for s in range(1, species_count + 1)],
            dtype=float,
        )
        merged_into = np.arange(species_count)
        for si in range(species_count):
            if merged_into[si] != si:
                continue
            candidates = [sj for sj in range(species_count) if sj != si and merged_into[sj] == sj]
            if not candidates:
                continue
            nearest = candidates[
                int(np.argmin([np.linalg.norm(seeds[si, :] - seeds[sj, :]) for sj in candidates]))
            ]
            if max_fit[si] < min_fit[nearest]:
                merged_into[si] = nearest
        merged = species_id.copy()
        for species in range(species_count):
            if merged_into[species] != species:
                merged[species_id == species + 1] = merged_into[species] + 1
        _, compact = np.unique(merged, return_inverse=True)
        return compact.astype(int) + 1

    def _species_seed_positions(
        self,
        species_id: np.ndarray,
        positions: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        species_count = int(np.max(species_id)) if species_id.size else 0
        seeds = np.zeros((species_count, self.dimension), dtype=float)
        for species in range(1, species_count + 1):
            members = np.flatnonzero(species_id == species)
            if members.size == 0:
                continue
            best = int(members[np.argmax(fitness[members])])
            seeds[species - 1, :] = positions[best, :]
        return seeds

    def _compute_slbest_links(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        *,
        threshold: float,
        distances: np.ndarray | None = None,
        return_edges: bool = False,
    ):
        n = positions.shape[0]
        if distances is None:
            distances = self._pairwise_distances(positions)
        sort_idx = np.argsort(fitness)[::-1]
        slbest = np.zeros(n, dtype=int)
        edges: list[tuple[int, int]] = []
        for rank, idx in enumerate(sort_idx):
            better = sort_idx[:rank]
            better = better[fitness[better] > fitness[idx]]
            if better.size == 0:
                slbest[idx] = idx
                continue
            local = distances[idx, better]
            best_pos = int(np.argmin(local))
            if float(local[best_pos]) < threshold:
                target = int(better[best_pos])
                slbest[idx] = target
                edges.append((idx, target))
            else:
                slbest[idx] = idx
        if return_edges:
            return slbest, edges
        return slbest

    def _species_from_edges(self, n: int, edges: list[tuple[int, int]]) -> np.ndarray:
        parent = list(range(n))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        for left, right in edges:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_left] = root_right
        roots = np.asarray([find(idx) for idx in range(n)], dtype=int)
        _, compact = np.unique(roots, return_inverse=True)
        return compact.astype(int) + 1

    def _pairwise_distances(self, positions: np.ndarray) -> np.ndarray:
        return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)

    def _mean_nearest_distance(self, distances: np.ndarray) -> float:
        n = distances.shape[0]
        if n <= 1:
            return 0.0
        work = distances.copy()
        np.fill_diagonal(work, np.inf)
        nearest = np.min(work, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if nearest.size == 0:
            return 0.0
        return float(np.mean(nearest))

    def _random_positions(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, self.dimension), dtype=float)
        return self.min_coordinate + (self.max_coordinate - self.min_coordinate) * self.rng.random(
            (count, self.dimension)
        )
