"""Repulsion and boundary handling from Ahrari's RS-CMSA-ESII source.

Translated from Restart.m, Subpopulation.m and SubpopulationCMSA.m.
Original author: Ali Ahrari; CC BY-NC-SA (see the source audit).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ..state import ArchivedSolution, RestartState


def normalized_distances(points: np.ndarray, centre: np.ndarray, state: RestartState) -> np.ndarray:
    return np.linalg.norm((points - centre) @ state.invsqrt, axis=1) / state.sigma


def initial_centre(
    archive: list[ArchivedSolution], lower: np.ndarray, widths: np.ndarray,
    radius: float, reduction: float, rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Sample outside spherical archive exclusions, shrinking after 101 failures."""
    points = np.asarray([(item.position - lower) / widths for item in archive])
    taboo = np.asarray([item.taboo_distance for item in archive])
    rejected = 0
    while True:
        candidate = rng.random(len(lower))
        if not archive or np.all(np.linalg.norm(candidate - points, axis=1) > taboo * radius):
            return lower + candidate * widths, radius
        rejected += 1
        if rejected > 100:
            rejected = 0
            radius *= reduction


def repair_bounds(
    candidate: np.ndarray, centre: np.ndarray, lower: np.ndarray,
    upper: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Relocate violating coordinates in the source's centre-symmetric interval."""
    repaired = candidate.copy()
    high = repaired > upper
    lo = np.maximum(2 * centre[high] - upper[high], lower[high])
    repaired[high] = lo + rng.random(np.count_nonzero(high)) * (upper[high] - lo)
    low = repaired < lower
    hi = np.minimum(2 * centre[low] - lower[low], upper[low])
    repaired[low] = lower[low] + rng.random(np.count_nonzero(low)) * (hi - lower[low])
    return repaired


def critical_regions(
    archive: list[ArchivedSolution], state: RestartState,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Only better archive members repel; prioritize Gaussian criticality > .01."""
    better = [item for item in archive if item.fitness > state.best_fitness]
    if not better:
        return np.empty((0, len(state.mean))), np.empty(0), 0.0
    points = np.asarray([item.position for item in better])
    radii = np.asarray([item.taboo_distance for item in better])
    distances = normalized_distances(points, state.mean, state)
    criticality = ndtr(distances + radii) - ndtr(distances - radii)
    order = np.argsort(-criticality, kind="stable")
    selected = order[criticality[order] > .01]
    return points[selected], radii[selected], float(np.max(criticality))


def merge_candidate(archive: list[ArchivedSolution], state: RestartState) -> tuple[int, int]:
    """Return the highest mergeability archive index and count exceeding .5."""
    if not archive:
        return -1, 0
    positions = np.asarray([item.position for item in archive])
    distances = normalized_distances(positions, state.mean, state)
    numerators = np.asarray([item.taboo_distance + 1.0 for item in archive])
    # Coincident centres have infinite mergeability by the ratio's limit.
    scores = np.full(len(archive), np.inf)
    np.divide(numerators, distances, out=scores, where=distances != 0)
    return int(np.argmax(scores)), int(np.count_nonzero(scores > .5))
