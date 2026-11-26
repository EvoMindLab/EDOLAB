"""Positive-weight CMA-ES core using EDOLAB's shared operator.

EvOpt Report 2022002 specifies CMA-ES, half-population positive recombination,
and the fourth root of the ordinary CSA damping. The distributed CMSA source
does not include its CMA sibling; this component reconstructs that core from
the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.operators import CMAESOperator

from ..state import RestartState


@dataclass(frozen=True, slots=True)
class CMAParameters:
    weights: np.ndarray
    mueff: float
    cc: float
    cs: float
    c1: float
    cmu: float
    damps: float
    chi_n: float

    @classmethod
    def create(cls, dimension: int, population_size: int, parent_ratio: float) -> CMAParameters:
        parents = max(1, int(np.floor(.5 + population_size * parent_ratio)))
        weights = np.log(1 + parents) - np.log(np.arange(1, parents + 1))
        weights /= weights.sum()
        mueff = float(1 / (weights @ weights))
        cc = (4 + mueff / dimension) / (dimension + 4 + 2 * mueff / dimension)
        cs = (mueff + 2) / (dimension + mueff + 5)
        c1 = 2 / ((dimension + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dimension + 2)**2 + mueff))
        damping = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dimension + 1)) - 1) + cs
        chi_n = np.sqrt(dimension) * (1 - 1 / (4 * dimension) + 1 / (21 * dimension**2))
        return cls(weights, mueff, cc, cs, c1, cmu, float(damping**.25), float(chi_n))


def update(
    state: RestartState, candidates: np.ndarray, fitness: np.ndarray,
    parameters: CMAParameters, operator: CMAESOperator,
) -> None:
    """Commit one complete, paid generation; partial generations never get here."""
    mean, order = operator.recombine_columns(candidates.T, fitness, parameters.weights)
    result = operator.update_distribution(
        mean=mean, old_mean=state.mean, sigma=state.sigma, samples=candidates.T,
        selected_order=order, weights=parameters.weights, pc=state.pc, ps=state.ps,
        covariance=state.covariance, invsqrt_covariance=state.invsqrt,
        eval_count=(state.generation + 1) * len(candidates), lambda_=len(candidates),
        mueff=parameters.mueff, cc=parameters.cc, cs=parameters.cs, c1=parameters.c1,
        cmu=parameters.cmu, damps=parameters.damps, chi_n=parameters.chi_n,
    )
    state.mean = mean
    state.pc = result["pc"]
    state.ps = result["ps"]
    state.sigma = result["sigma"]
    state.covariance = result["covariance"]
    state.basis, state.scales, state.invsqrt = operator.eigensystem(state.covariance)
    state.generation += 1

