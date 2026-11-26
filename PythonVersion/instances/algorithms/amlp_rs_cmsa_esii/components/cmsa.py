"""SubpopulationCMSA.m / SpecStrParCMSA.m (Ali Ahrari, CC BY-NC-SA 4.0).

The distributed main.m disables elitism. Individual step sizes self-adapt;
there are no CMA evolution paths or CSA updates in this core.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state import RestartState


@dataclass(frozen=True, slots=True)
class CMSAParameters:
    weights: np.ndarray
    mu_eff: float
    tau_sigma: float
    tau_cov: float

    @classmethod
    def create(cls, dimension: int, population_size: int, parent_ratio: float):
        mu = max(1, int(np.floor(.5 + population_size * parent_ratio)))
        weights = np.log(1 + mu) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        return cls(weights, mu_eff, np.sqrt(.5 * .5 / dimension),
                   1 + dimension * (1 + dimension) / (2 * mu_eff))


def initialize(mean: np.ndarray, sigma: float, widths: np.ndarray, radius: float):
    return RestartState(mean.copy(), sigma, np.diag(widths**2), np.eye(len(mean)),
                        widths.copy(), np.diag(1 / widths), radius)


def update(state: RestartState, candidates: np.ndarray, fitness: np.ndarray,
           individual_sigmas: np.ndarray, parameters: CMSAParameters,
           lower: np.ndarray, upper: np.ndarray) -> None:
    """Non-elitist recombination; corrected steps include boundary relocation."""
    order = np.argsort(-fitness, kind="stable")
    parents = order[:len(parameters.weights)]
    steps = (candidates[parents] - state.mean) / individual_sigmas[parents, None]
    state.mean = np.clip(parameters.weights @ candidates[parents], lower, upper)
    state.best_position = candidates[order[0]].copy()
    state.best_fitness = float(fitness[order[0]])
    # Algebraically identical to weighted product / (geomean(s) / old_sigma).
    logs = np.log(individual_sigmas)
    state.sigma *= float(np.exp(parameters.weights @ logs[parents] - np.mean(logs)))
    suggested = (steps.T * parameters.weights) @ steps
    rate = 1 / parameters.tau_cov
    covariance = (1 - rate) * state.covariance + rate * suggested
    state.covariance = .5 * (covariance + covariance.T)
    eigenvalues, state.basis = np.linalg.eigh(state.covariance)
    state.scales = np.sqrt(eigenvalues)
    state.invsqrt = (state.basis / state.scales) @ state.basis.T
    state.generation += 1

