"""Reusable CMA-ES helpers."""

from __future__ import annotations

import numpy as np

from core.algorithm.operators.base import Operator


class CMAESOperator(Operator):
    name = "cmaes"

    def sample_columns(
        self,
        mean: np.ndarray,
        sigma: float,
        basis: np.ndarray,
        scales: np.ndarray,
        sample_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample CMA-ES offspring as MATLAB-style column vectors."""

        mean = np.asarray(mean, dtype=float).reshape(-1, 1)
        basis = np.asarray(basis, dtype=float)
        scales = np.asarray(scales, dtype=float).reshape(-1, 1)
        normals = rng.standard_normal((mean.shape[0], int(sample_count)))
        return mean + float(sigma) * basis @ (scales * normals)

    def sample_multivariate(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        sigma: float,
        sample_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.multivariate_normal(mean, covariance, size=sample_count) * sigma + mean

    def reflect_bounds(
        self,
        samples: np.ndarray,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
    ) -> np.ndarray:
        """Reflect samples at bounds, matching the mCMAES repair rule."""

        repaired = np.asarray(samples, dtype=float).copy()
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        high = repaired > upper_arr
        if np.any(high):
            upper_full = np.broadcast_to(upper_arr, repaired.shape)
            repaired[high] = (2.0 * upper_full - repaired)[high]
            low_after = repaired < lower_arr
            if np.any(low_after):
                lower_full = np.broadcast_to(lower_arr, repaired.shape)
                repaired[low_after] = lower_full[low_after]
        low = repaired < lower_arr
        if np.any(low):
            lower_full = np.broadcast_to(lower_arr, repaired.shape)
            repaired[low] = (2.0 * lower_full - repaired)[low]
            high_after = repaired > upper_arr
            if np.any(high_after):
                upper_full = np.broadcast_to(upper_arr, repaired.shape)
                repaired[high_after] = upper_full[high_after]
        return repaired

    def recombine_columns(
        self,
        samples: np.ndarray,
        fitness: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sort samples by descending fitness and return weighted mean."""

        samples = np.asarray(samples, dtype=float)
        fitness = np.asarray(fitness, dtype=float)
        weights = np.asarray(weights, dtype=float).reshape(-1, 1)
        order = np.argsort(-fitness)
        parents = samples[:, order[: weights.size]]
        mean = parents @ weights
        return mean.reshape(-1), order

    def update_distribution(
        self,
        *,
        mean: np.ndarray,
        old_mean: np.ndarray,
        sigma: float,
        samples: np.ndarray,
        selected_order: np.ndarray,
        weights: np.ndarray,
        pc: np.ndarray,
        ps: np.ndarray,
        covariance: np.ndarray,
        invsqrt_covariance: np.ndarray,
        eval_count: int,
        lambda_: int,
        mueff: float,
        cc: float,
        cs: float,
        c1: float,
        cmu: float,
        damps: float,
        chi_n: float,
    ) -> dict[str, np.ndarray | float | bool]:
        """Update CMA-ES paths, covariance, and step size for maximization."""

        mean = np.asarray(mean, dtype=float).reshape(-1)
        old_mean = np.asarray(old_mean, dtype=float).reshape(-1)
        pc = np.asarray(pc, dtype=float).reshape(-1)
        ps = np.asarray(ps, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)
        invsqrt_covariance = np.asarray(invsqrt_covariance, dtype=float)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        dimension = mean.size
        step = (mean - old_mean) / float(sigma)
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (
            invsqrt_covariance @ step
        )
        denominator = 1.0 - (1.0 - cs) ** (2.0 * eval_count / float(lambda_))
        hsig = (
            np.sum(ps * ps) / denominator / dimension
            < 2.0 + 4.0 / (dimension + 1.0)
        )
        pc = (1.0 - cc) * pc + float(hsig) * np.sqrt(cc * (2.0 - cc) * mueff) * step
        parents = np.asarray(samples, dtype=float)[:, selected_order[: weights.size]]
        artmp = (parents - old_mean.reshape(-1, 1)) / float(sigma)
        covariance = (
            (1.0 - c1 - cmu) * covariance
            + c1
            * (
                np.outer(pc, pc)
                + (1.0 - float(hsig)) * cc * (2.0 - cc) * covariance
            )
            + cmu * artmp @ np.diag(weights) @ artmp.T
        )
        sigma = float(sigma) * float(np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0)))
        return {
            "pc": pc,
            "ps": ps,
            "covariance": covariance,
            "sigma": sigma,
            "hsig": bool(hsig),
        }

    def eigensystem(
        self,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return basis, standard deviations, and inverse square-root covariance."""

        covariance = np.asarray(covariance, dtype=float)
        covariance = np.triu(covariance) + np.triu(covariance, 1).T
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 1e-30)
        scales = np.sqrt(values)
        invsqrt = vectors @ np.diag(scales ** -1.0) @ vectors.T
        return vectors, scales, invsqrt


def sample_multivariate(
    mean: np.ndarray,
    covariance: np.ndarray,
    sigma: float,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return CMAESOperator().sample_multivariate(mean, covariance, sigma, sample_count, rng)
