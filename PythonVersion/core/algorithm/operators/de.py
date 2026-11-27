"""Reusable DE operators."""

from __future__ import annotations

import numpy as np

from core.algorithm.operators.base import Operator


class DEOperator(Operator):
    name = "de"

    def mutation_best_2(
        self,
        population: np.ndarray,
        best_position: np.ndarray,
        factors: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return DE/best/2 donors for each target row."""

        positions = np.asarray(population, dtype=float)
        best = np.asarray(best_position, dtype=float).reshape(1, -1)
        factors = np.asarray(factors, dtype=float).reshape(-1, 1)
        if positions.ndim != 2:
            raise ValueError("population must be a 2D array.")
        row_count = positions.shape[0]
        if row_count < 5:
            raise ValueError("DE/best/2 requires at least 5 individuals.")
        donors = np.empty_like(positions)
        for row in range(row_count):
            choices = np.delete(np.arange(row_count), row)
            selected = rng.choice(choices, size=4, replace=False)
            donors[row, :] = best + factors[row] * (
                positions[selected[0], :]
                + positions[selected[1], :]
                - positions[selected[2], :]
                - positions[selected[3], :]
            )
        return donors

    def binomial_crossover(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        crossover_rate: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = rng.random(target.shape) < crossover_rate
        if target.ndim == 1 and not mask.any():
            mask[rng.integers(0, target.size)] = True
        return np.where(mask, mutant, target)

    def binomial_crossover_population(
        self,
        targets: np.ndarray,
        donors: np.ndarray,
        crossover_rates: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Binomial crossover with one forced donor dimension per row."""

        targets = np.asarray(targets, dtype=float)
        donors = np.asarray(donors, dtype=float)
        rates = np.asarray(crossover_rates, dtype=float).reshape(-1, 1)
        if targets.shape != donors.shape:
            raise ValueError("targets and donors must have the same shape.")
        if targets.ndim != 2:
            raise ValueError("targets must be a 2D array.")
        offspring = targets.copy()
        row_count, dimension = targets.shape
        forced = rng.integers(0, dimension, size=row_count)
        offspring[np.arange(row_count), forced] = donors[np.arange(row_count), forced]
        mask = rng.random(targets.shape) < rates
        offspring[mask] = donors[mask]
        return offspring

    def midpoint_repair_bounds(
        self,
        offspring: np.ndarray,
        targets: np.ndarray,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
    ) -> np.ndarray:
        """Repair out-of-bounds DE offspring halfway toward the target."""

        repaired = np.asarray(offspring, dtype=float).copy()
        targets = np.asarray(targets, dtype=float)
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        low = repaired < lower_arr
        high = repaired > upper_arr
        if np.any(low):
            lower_full = np.broadcast_to(lower_arr, repaired.shape)
            repaired[low] = ((lower_full + targets) / 2.0)[low]
        if np.any(high):
            upper_full = np.broadcast_to(upper_arr, repaired.shape)
            repaired[high] = ((upper_full + targets) / 2.0)[high]
        return repaired

    def greedy_select(
        self,
        targets: np.ndarray,
        target_fitness: np.ndarray,
        offspring: np.ndarray,
        offspring_fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maximization greedy selection."""

        targets = np.asarray(targets, dtype=float).copy()
        target_fitness = np.asarray(target_fitness, dtype=float).copy()
        offspring = np.asarray(offspring, dtype=float)
        offspring_fitness = np.asarray(offspring_fitness, dtype=float)
        better = offspring_fitness > target_fitness
        targets[better, :] = offspring[better, :]
        target_fitness[better] = offspring_fitness[better]
        return targets, target_fitness, better


def binomial_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return DEOperator().binomial_crossover(target, mutant, crossover_rate, rng)
