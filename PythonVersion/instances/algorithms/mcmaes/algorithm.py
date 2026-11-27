"""mCMAES algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import CMAESOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.mcmaes.state import MCMAESSubPopulation


@algorithm_registry.register("mCMAES")
class MCMAESAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "mCMAES"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSize": ParameterSpec(
                "PopulationSize",
                1,
                "integer",
                (1, 100),
                "Retained for configuration parity; mCMAES uses a fixed lambda=5 strategy.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.coordinate_span = float(np.mean(self.max_coordinate - self.min_coordinate))
        self.population_size = int(self.parameters["PopulationSize"])
        self.swarm_number = 1
        self.free_swarm_id = 0
        self.lambda_ = 5
        self.mu = int(np.floor(self.lambda_ / 2))
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1, dtype=float))
        self.weights = weights / np.sum(weights)
        self.mueff = float(np.sum(self.weights) ** 2 / np.sum(self.weights**2))
        self.cc = float(
            (4 + self.mueff / self.dimension)
            / (self.dimension + 4 + 2 * self.mueff / self.dimension)
        )
        self.cs = float((self.mueff + 2) / (self.dimension + self.mueff + 5))
        self.c1 = float(2 / ((self.dimension + 1.3) ** 2 + self.mueff))
        self.cmu = float(
            min(
                1 - self.c1,
                2
                * (self.mueff - 2 + 1 / self.mueff)
                / ((self.dimension + 2) ** 2 + self.mueff),
            )
        )
        self.damps = float(
            1
            + 2 * max(0.0, np.sqrt((self.mueff - 1) / (self.dimension + 1)) - 1)
            + self.cs
        )
        self.chi_n = float(
            np.sqrt(self.dimension)
            * (1 - 1 / (4 * self.dimension) + 1 / (21 * self.dimension**2))
        )
        self.min_exclusion_limit = self._threshold_for_swarm_count(10)
        self.exclusion_limit = self.min_exclusion_limit
        self.convergence_limit = self.exclusion_limit
        self.shift_severity = 1.0
        self.cmaes_operator = CMAESOperator()
        self.pop = [self._generate_subpopulation(problem)]

    def iterate(self, problem) -> None:
        previous_swarm_number = self.swarm_number
        last_offspring: np.ndarray | None = None
        for swarm in list(self.pop[: self.swarm_number]):
            last_offspring = self._move_subpopulation(swarm, problem)
            if problem.state.recent_change:
                return

        if self._apply_exclusion(problem):
            return

        if last_offspring is not None and self._free_swarm_converged(last_offspring):
            self.swarm_number += 1
            self.free_swarm_id = self.swarm_number - 1
            self.pop.append(self._generate_subpopulation(problem))
            if problem.state.recent_change:
                return

        if previous_swarm_number != self.swarm_number:
            self._refresh_thresholds()

    def on_environment_change(self, problem) -> None:
        shift_samples: list[float] = []
        for index, swarm in enumerate(self.pop):
            if index == self.free_swarm_id:
                continue
            if not np.isnan(swarm.gbest_past_environment).any():
                shift = float(np.linalg.norm(swarm.gbest_past_environment - swarm.mean))
                swarm.shifts.append(shift)
            shift_samples.extend(swarm.shifts)
        if shift_samples:
            self.shift_severity = float(np.mean(shift_samples))

        for index, swarm in enumerate(self.pop):
            if index == self.free_swarm_id:
                continue
            swarm.shift_severity = self.shift_severity
            self._reset_strategy_state(swarm, sigma=max(self.shift_severity / 2, np.finfo(float).eps))

        for swarm in self.pop:
            swarm.gbest_past_environment = swarm.mean.copy()
            swarm.fgbest_past_environment = swarm.mean_fitness

    def _generate_subpopulation(self, problem) -> MCMAESSubPopulation:
        mean = self.bounds.sample(1, self.rng)[0]
        values = evaluate_(problem, mean, 1)
        fitness = float(values[0]) if values.size and np.isfinite(values[0]) else -np.inf
        sigma = max(float(np.round(self.coordinate_span / 3)), np.finfo(float).eps)
        state = MCMAESSubPopulation(
            positions=mean.reshape(1, self.dimension).copy(),
            fitness=np.asarray([fitness], dtype=float),
            metadata={},
            sigma=sigma,
            pc=np.zeros(self.dimension),
            ps=np.zeros(self.dimension),
            b_matrix=np.eye(self.dimension),
            d_vector=np.ones(self.dimension),
            covariance=np.eye(self.dimension),
            invsqrt_covariance=np.eye(self.dimension),
            eigen_eval=0,
            eval_count=0,
            gbest_past_environment=np.full(self.dimension, np.nan),
            fgbest_past_environment=float("nan"),
            shifts=[],
            shift_severity=1.0,
            last_offspring_positions=None,
        )
        return state

    def _move_subpopulation(
        self,
        swarm: MCMAESSubPopulation,
        problem,
    ) -> np.ndarray | None:
        offspring_columns = self.cmaes_operator.sample_columns(
            swarm.mean,
            swarm.sigma,
            swarm.b_matrix,
            swarm.d_vector,
            self.lambda_,
            self.rng,
        )
        offspring_columns = self.cmaes_operator.reflect_bounds(
            offspring_columns,
            self.min_coordinate.reshape(-1, 1),
            self.max_coordinate.reshape(-1, 1),
        )
        offspring = offspring_columns.T
        swarm.last_offspring_positions = offspring.copy()

        offspring_fitness = evaluate_(problem, offspring, self.lambda_)
        if problem.state.recent_change or problem.state.terminated:
            return offspring
        if np.count_nonzero(np.isfinite(offspring_fitness)) < self.lambda_:
            return offspring
        swarm.eval_count += self.lambda_

        x_old = swarm.mean.copy()
        new_mean, order = self.cmaes_operator.recombine_columns(
            offspring_columns,
            offspring_fitness,
            self.weights,
        )

        mean_fitness = problem.evaluate(new_mean)
        if problem.state.recent_change:
            return offspring
        if mean_fitness.size == 0:
            return offspring
        swarm.mean = new_mean
        swarm.mean_fitness = float(mean_fitness[0])
        if problem.state.terminated:
            return offspring

        sigma = max(float(swarm.sigma), np.finfo(float).eps)
        update = self.cmaes_operator.update_distribution(
            mean=swarm.mean,
            old_mean=x_old,
            sigma=sigma,
            samples=offspring_columns,
            selected_order=order,
            weights=self.weights,
            pc=swarm.pc,
            ps=swarm.ps,
            covariance=swarm.covariance,
            invsqrt_covariance=swarm.invsqrt_covariance,
            eval_count=swarm.eval_count,
            lambda_=self.lambda_,
            mueff=self.mueff,
            cc=self.cc,
            cs=self.cs,
            c1=self.c1,
            cmu=self.cmu,
            damps=self.damps,
            chi_n=self.chi_n,
        )
        swarm.pc = np.asarray(update["pc"], dtype=float)
        swarm.ps = np.asarray(update["ps"], dtype=float)
        swarm.covariance = np.asarray(update["covariance"], dtype=float)
        swarm.sigma = float(update["sigma"])
        if swarm.eval_count - swarm.eigen_eval > self.lambda_ / (self.c1 + self.cmu) / self.dimension / 10:
            self._update_eigendecomposition(swarm)
        return offspring

    def _apply_exclusion(self, problem) -> bool:
        if self.swarm_number <= 1:
            return False
        remove = np.zeros(self.swarm_number, dtype=bool)
        for left in range(self.swarm_number - 1):
            for right in range(left + 1, self.swarm_number):
                distance = float(np.linalg.norm(self.pop[left].mean - self.pop[right].mean))
                if distance >= self.exclusion_limit:
                    continue
                if self.pop[left].mean_fitness < self.pop[right].mean_fitness:
                    target = left
                else:
                    target = right
                if target == self.free_swarm_id:
                    self.pop[target] = self._generate_subpopulation(problem)
                    if problem.state.recent_change:
                        return True
                else:
                    remove[target] = True

        for index in range(self.swarm_number - 1, -1, -1):
            if not remove[index]:
                continue
            del self.pop[index]
            if index < self.free_swarm_id:
                self.free_swarm_id -= 1
        self.swarm_number = len(self.pop)
        self.free_swarm_id = min(self.free_swarm_id, self.swarm_number - 1)
        return False

    def _free_swarm_converged(self, offspring: np.ndarray) -> bool:
        if offspring.shape[0] < 2:
            return True
        distances = np.linalg.norm(
            offspring[:, None, :] - offspring[None, :, :],
            axis=2,
        )
        return bool(np.all(distances <= self.convergence_limit))

    def _reset_strategy_state(self, swarm: MCMAESSubPopulation, sigma: float) -> None:
        swarm.sigma = float(sigma)
        swarm.pc = np.zeros(self.dimension)
        swarm.ps = np.zeros(self.dimension)
        swarm.b_matrix = np.eye(self.dimension)
        swarm.d_vector = np.ones(self.dimension)
        swarm.covariance = np.eye(self.dimension)
        swarm.invsqrt_covariance = np.eye(self.dimension)
        swarm.eigen_eval = 0
        swarm.eval_count = 0

    def _update_eigendecomposition(self, swarm: MCMAESSubPopulation) -> None:
        swarm.eigen_eval = swarm.eval_count
        basis, scales, invsqrt = self.cmaes_operator.eigensystem(swarm.covariance)
        swarm.b_matrix = basis
        swarm.d_vector = scales
        swarm.invsqrt_covariance = invsqrt

    def _refresh_thresholds(self) -> None:
        self.exclusion_limit = self._threshold_for_swarm_count(max(1, self.swarm_number))
        self.convergence_limit = self.exclusion_limit

    def _threshold_for_swarm_count(self, count: int) -> float:
        return MCMAESHelper.threshold_for_swarm_count(
            self.coordinate_span,
            self.dimension,
            count,
        )


class MCMAESHelper:
    """Stateless helper methods used by the mCMAES flow."""

    @staticmethod
    def threshold_for_swarm_count(
        coordinate_span: float,
        dimension: int,
        swarm_count: int,
    ) -> float:
        """Return the mCMAES exclusion/convergence threshold."""

        return float(0.5 * (coordinate_span / (swarm_count ** (1 / dimension))))
