"""Shared algorithm helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


class FixedSizeSwarmRepairer:
    """Resize clustered position groups into fixed-size swarms."""

    def __init__(
        self,
        population_size: int,
        dimension: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.population_size = max(1, int(population_size))
        self.dimension = int(dimension)
        self.rng = rng

    def repair(
        self,
        groups: list[np.ndarray],
        remain: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        groups = [np.asarray(group, dtype=float).copy() for group in groups if group.size]
        overflow_rows: list[np.ndarray] = []
        for index, group in enumerate(groups):
            if group.shape[0] <= self.population_size:
                continue
            centroid = np.mean(group, axis=0)
            order = np.argsort(np.linalg.norm(group - centroid, axis=1))
            keep = order[: self.population_size]
            remove = order[self.population_size :]
            overflow_rows.extend(group[remove, :])
            groups[index] = group[keep, :]

        fill_pool = stack_rows(overflow_rows, self.dimension)
        remain = np.asarray(remain, dtype=float).reshape(-1, self.dimension)
        if remain.size:
            fill_pool = np.vstack([fill_pool, remain]) if fill_pool.size else remain.copy()
        for index, group in enumerate(groups):
            deficit = self.population_size - group.shape[0]
            if deficit <= 0 or fill_pool.size == 0:
                continue
            centroid = np.mean(group, axis=0)
            order = np.argsort(np.linalg.norm(fill_pool - centroid, axis=1))
            take = order[: min(deficit, fill_pool.shape[0])]
            groups[index] = np.vstack([group, fill_pool[take, :]])
            fill_pool = np.delete(fill_pool, take, axis=0)

        while fill_pool.shape[0] >= self.population_size:
            groups.append(fill_pool[: self.population_size, :].copy())
            fill_pool = fill_pool[self.population_size :, :]

        valid: list[np.ndarray] = []
        remain_rows = [row for row in fill_pool] if fill_pool.size else []
        for group in groups:
            if group.shape[0] < self.population_size:
                remain_rows.extend(group)
            else:
                valid.append(group)

        if not valid and remain_rows:
            pool = stack_rows(remain_rows, self.dimension)
            if pool.shape[0] >= self.population_size:
                valid.append(pool[: self.population_size, :].copy())
                pool = pool[self.population_size :, :]
            else:
                pad_count = self.population_size - pool.shape[0]
                rng = self.rng or np.random.default_rng(0)
                pad = pool[rng.integers(0, pool.shape[0], size=pad_count), :]
                valid.append(np.vstack([pool, pad]))
                pool = np.empty((0, self.dimension))
            remain_rows = [row for row in pool] if pool.size else []

        return valid, stack_rows(remain_rows, self.dimension)


def evaluate_(problem: Any, x: np.ndarray, expected: int) -> np.ndarray:
    """Evaluate within the current budget window and pad unfinished slots."""

    values = problem.evaluate(x)
    if values.size == expected:
        return values
    output = np.full(expected, np.nan)
    output[: values.size] = values
    return output


def has_complete_values(values: np.ndarray, expected: int) -> bool:
    """Return true when an evaluation produced every requested finite-or-infinite slot."""

    data = np.asarray(values, dtype=float).reshape(-1)
    return data.size == int(expected) and not bool(np.any(np.isnan(data)))


def update_center(swarm: Any) -> np.ndarray:
    return np.mean(swarm.pbest_positions, axis=0)


def update_current_radius(swarm: Any) -> float:
    return float(np.mean(np.linalg.norm(swarm.pbest_positions - swarm.center, axis=1)))


def stack_rows(rows: list[np.ndarray], dimension: int) -> np.ndarray:
    if not rows:
        return np.empty((0, dimension))
    return np.vstack(rows).reshape(-1, dimension)
