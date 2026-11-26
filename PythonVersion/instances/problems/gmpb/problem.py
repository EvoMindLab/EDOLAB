"""Standalone GMPB problem instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.compute import record_compute_use, select_torch_device
from core.parameters import ParameterSpec
from core.problem.base import BaseProblem
from core.problem.bounds import Bounds
from core.registry import problem_registry


@problem_registry.register("GMPB")
class GMPBProblem(BaseProblem):
    name: ClassVar[str] = "GMPB"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "MinCoordinate": ParameterSpec("MinCoordinate", -50.0, "numeric", (-1000, 1000)),
            "MaxCoordinate": ParameterSpec("MaxCoordinate", 50.0, "numeric", (-1000, 1000)),
            "Dimension": ParameterSpec("Dimension", 5, "integer", (2, 100)),
            "PeakNumber": ParameterSpec("PeakNumber", 10, "integer", (1, 500)),
            "ChangeFrequency": ParameterSpec("ChangeFrequency", 5000, "integer", (1, 100000)),
            "EnvironmentNumber": ParameterSpec("EnvironmentNumber", 100, "integer", (1, 10000)),
            "ShiftSeverity": ParameterSpec("ShiftSeverity", 1.0, "numeric", (0, 100)),
            "HeightSeverity": ParameterSpec("HeightSeverity", 7.0, "numeric", (0, 100)),
            "WidthSeverity": ParameterSpec("WidthSeverity", 1.0, "numeric", (0, 100)),
            "AngleSeverity": ParameterSpec("AngleSeverity", float(np.pi / 9), "numeric", (0, np.pi)),
            "TauSeverity": ParameterSpec("TauSeverity", 0.2, "numeric", (0, 1)),
            "EtaSeverity": ParameterSpec("EtaSeverity", 10.0, "numeric", (0, 100)),
            "EvaluationDevice": ParameterSpec(
                "EvaluationDevice",
                "auto",
                "option",
                ("auto", "cuda", "cuda:<id>", "mps", "cpu", "numpy"),
                "Component-level compute device for GMPB fitness evaluation.",
            ),
        }

    @property
    def dimension(self) -> int:
        return int(self.parameters["Dimension"])

    @property
    def bounds(self) -> Bounds:
        return Bounds.uniform(
            float(self.parameters["MinCoordinate"]),
            float(self.parameters["MaxCoordinate"]),
            self.dimension,
        )

    def reset(self, seed: int | None = None) -> None:
        self.initialize_runtime_state()
        self.rng = np.random.default_rng(seed)
        self._build_environments()

    def _shift_severity(self, environment_index: int) -> float:
        """Nominal displacement for one transition; research subclasses may schedule it."""
        return float(self.parameters["ShiftSeverity"])

    def _build_environments(self) -> None:
        env_count = self.environment_number
        peak_count = int(self.parameters["PeakNumber"])
        dim = self.dimension
        rng = self.rng
        bounds = self.bounds

        self.min_height = 30.0
        self.max_height = 70.0
        self.min_width = 1.0
        self.max_width = 12.0
        self.positions = np.full((env_count, peak_count, dim), np.nan, dtype=float)
        self.heights = np.full((env_count, peak_count), np.nan, dtype=float)
        self.widths = np.full((env_count, peak_count, dim), np.nan, dtype=float)
        self.initial_rotation = np.full((peak_count, dim, dim), np.nan, dtype=float)
        self.rotation_matrices: list[np.ndarray] = []
        self.angles = np.full((env_count, peak_count), np.nan, dtype=float)
        self.tau = np.full((env_count, peak_count), np.nan, dtype=float)
        self.eta = np.full((env_count, peak_count, 4), np.nan, dtype=float)
        self.optimum_values = np.full(env_count, np.nan, dtype=float)
        self.optimum_ids = np.full(env_count, -1, dtype=int)
        self._torch_env_cache: dict[tuple[int, str], tuple[object, ...]] = {}
        self._explicit_evaluation_backend: tuple[object | None, object | None, str] | None = None

        self.positions[0] = bounds.sample(peak_count, rng)
        self.heights[0] = self.min_height + (self.max_height - self.min_height) * rng.random(
            peak_count
        )
        self.widths[0] = self.min_width + (self.max_width - self.min_width) * rng.random(
            (peak_count, dim)
        )
        for peak in range(peak_count):
            q, _ = np.linalg.qr(rng.random((dim, dim)))
            self.initial_rotation[peak] = q
        self.rotation_matrices.append(self.initial_rotation.copy())
        self.angles[0] = -np.pi + 2 * np.pi * rng.random(peak_count)
        self.tau[0] = 0.1 + 0.9 * rng.random(peak_count)
        self.eta[0] = 50.0 * rng.random((peak_count, 4))
        self._record_optimum(0)

        for env in range(1, env_count):
            shift = rng.standard_normal((peak_count, dim))
            norm = np.linalg.norm(shift, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            shift = shift / norm * self._shift_severity(env)

            self.positions[env] = bounds.reflect(self.positions[env - 1] + shift)
            self.heights[env] = _reflect(
                self.heights[env - 1]
                + float(self.parameters["HeightSeverity"]) * rng.standard_normal(peak_count),
                self.min_height,
                self.max_height,
            )
            self.widths[env] = _reflect(
                self.widths[env - 1]
                + float(self.parameters["WidthSeverity"]) * rng.standard_normal((peak_count, dim)),
                self.min_width,
                self.max_width,
            )
            self.angles[env] = _reflect(
                self.angles[env - 1]
                + float(self.parameters["AngleSeverity"]) * rng.standard_normal(peak_count),
                -np.pi,
                np.pi,
            )
            self.tau[env] = _reflect(
                self.tau[env - 1]
                + float(self.parameters["TauSeverity"]) * rng.standard_normal(peak_count),
                0.1,
                1.0,
            )
            self.eta[env] = _reflect(
                self.eta[env - 1]
                + float(self.parameters["EtaSeverity"]) * rng.standard_normal((peak_count, 4)),
                0.0,
                50.0,
            )
            rotations = np.full((peak_count, dim, dim), np.nan, dtype=float)
            for peak in range(peak_count):
                rotations[peak] = self.initial_rotation[peak] @ _rotation(
                    self.angles[env, peak],
                    dim,
                    rng,
                )
            self.rotation_matrices.append(rotations)
            self._record_optimum(env)

    def _record_optimum(self, env: int) -> None:
        optimum_id = int(np.argmax(self.heights[env]))
        self.optimum_ids[env] = optimum_id
        self.optimum_values[env] = float(self.heights[env, optimum_id])

    def raw_fitness(self, x: np.ndarray) -> float:
        env = self.state.environment_index
        sample = np.asarray(x, dtype=float).reshape(-1)
        values = np.empty(int(self.parameters["PeakNumber"]), dtype=float)
        order = np.argsort(self.heights[env])[::-1]
        result = -np.inf
        for peak in order:
            if result >= self.heights[env, peak]:
                break
            delta = sample - self.positions[env, peak]
            transformed = _transform(
                self.rotation_matrices[env][peak] @ delta,
                self.tau[env, peak],
                self.eta[env, peak],
            )
            peak_value = self.heights[env, peak] - np.sqrt(
                transformed @ np.diag(self.widths[env, peak]) @ transformed
            )
            if peak_value > result:
                result = float(peak_value)
        return float(result)

    def raw_fitness_batch(self, x: np.ndarray) -> np.ndarray:
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        work_items = int(samples.shape[0]) * int(self.parameters["PeakNumber"]) * self.dimension
        torch, device, label = self._select_evaluation_backend(work_items)
        if torch is None or device is None:
            return super().raw_fitness_batch(samples)
        try:
            return self._raw_fitness_batch_torch(samples, torch, device)
        except Exception as exc:
            record_compute_use(
                "problem:GMPB",
                "numpy:cpu",
                f"{label} failed; using NumPy CPU: {type(exc).__name__}",
            )
            return super().raw_fitness_batch(samples)

    def _select_evaluation_backend(self, work_items: int):
        policy = str(self.parameters.get("EvaluationDevice", "auto"))
        normalized = policy.strip().lower()
        if normalized != "auto" and self._explicit_evaluation_backend is not None:
            return self._explicit_evaluation_backend
        selected = select_torch_device(
            "problem:GMPB",
            work_items=work_items,
            device=policy,
        )
        if normalized != "auto":
            self._explicit_evaluation_backend = selected
        return selected

    def _raw_fitness_batch_torch(self, samples: np.ndarray, torch, device) -> np.ndarray:
        env = self.state.environment_index
        positions, heights, widths, rotations, tau, eta = self._torch_environment(env, torch, device)
        with torch.inference_mode():
            sample_t = torch.as_tensor(samples, dtype=torch.float32, device=device)
            delta = sample_t[:, None, :] - positions[None, :, :]
            # The peak axis is the batch axis for the rotation matrices.
            # ``bmm`` is equivalent to the einsum used here historically but
            # avoids the general einsum parser/dispatch cost on every small
            # swarm evaluation.
            rotated = torch.bmm(
                delta.permute(1, 0, 2),
                rotations.transpose(1, 2),
            ).permute(1, 0, 2)
            transformed = _transform_torch(rotated, tau, eta, torch)
            quadratic = torch.sum(transformed * transformed * widths[None, :, :], dim=-1)
            peak_values = heights[None, :] - torch.sqrt(torch.clamp(quadratic, min=0.0))
            values = torch.max(peak_values, dim=1).values
            return values.detach().cpu().numpy().astype(float)

    def _torch_environment(self, env: int, torch, device) -> tuple[object, ...]:
        key = (int(env), str(device))
        cached = self._torch_env_cache.get(key)
        if cached is not None:
            return cached
        values = (
            torch.as_tensor(self.positions[env], dtype=torch.float32, device=device),
            torch.as_tensor(self.heights[env], dtype=torch.float32, device=device),
            torch.as_tensor(self.widths[env], dtype=torch.float32, device=device),
            torch.as_tensor(self.rotation_matrices[env], dtype=torch.float32, device=device),
            torch.as_tensor(self.tau[env], dtype=torch.float32, device=device),
            torch.as_tensor(self.eta[env], dtype=torch.float32, device=device),
        )
        self._torch_env_cache[key] = values
        return values

    def current_optimum_value(self) -> float:
        return float(self.optimum_values[self.state.environment_index])

    def global_optimum_basin_mask(
        self,
        x: np.ndarray,
        environment_index: int | None = None,
    ) -> np.ndarray:
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        env = self.state.environment_index if environment_index is None else int(environment_index)
        peak_positions = self.positions[env]
        distances = np.linalg.norm(
            samples[:, None, :] - peak_positions[None, :, :],
            axis=2,
        )
        nearest_peak_ids = np.argmin(distances, axis=1)
        return nearest_peak_ids == int(self.optimum_ids[env])

    def visualization_peaks(self, environment_number: int | None = None) -> dict[str, object]:
        """Expose MATLAB-style GMPB peak annotations for 2D landscapes."""
        if self.dimension != 2:
            return {}
        env_index = self.resolve_environment_index(environment_number)
        env_number = env_index + 1
        peak_positions = self.positions[env_index]
        peak_heights = self.heights[env_index]
        peak_values = self.static_fitness(peak_positions, env_number)
        visible_mask = np.isclose(peak_values, peak_heights, rtol=1e-5, atol=1e-4)
        optimum_id = int(self.optimum_ids[env_index])
        return {
            "global_optimum_id": optimum_id + 1,
            "global_optimum_position": peak_positions[optimum_id, :2].copy(),
            "visible_peak_ids": np.flatnonzero(visible_mask) + 1,
            "visible_peak_positions": peak_positions[visible_mask, :2].copy(),
        }


def _reflect(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    reflected = np.asarray(values, dtype=float).copy()
    high = reflected > upper
    reflected[high] = 2 * upper - reflected[high]
    low = reflected < lower
    reflected[low] = 2 * lower - reflected[low]
    return np.clip(reflected, lower, upper)


def _rotation(theta: float, dimension: int, rng: np.random.Generator) -> np.ndarray:
    pages = []
    for ii in range(dimension):
        for jj in range(ii + 1, dimension):
            matrix = np.eye(dimension)
            matrix[ii, ii] = np.cos(theta)
            matrix[jj, jj] = np.cos(theta)
            matrix[ii, jj] = np.sin(theta)
            matrix[jj, ii] = -np.sin(theta)
            pages.append(matrix)
    output = np.eye(dimension)
    if not pages:
        return output
    for index in rng.permutation(len(pages)):
        output = output @ pages[int(index)]
    return output


def _transform(x: np.ndarray, tau: float, eta: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    positive = x > 0
    if np.any(positive):
        z = np.log(x[positive])
        y[positive] = np.exp(z + tau * (np.sin(eta[0] * z) + np.sin(eta[1] * z)))
    negative = x < 0
    if np.any(negative):
        z = np.log(-x[negative])
        y[negative] = -np.exp(z + tau * (np.sin(eta[2] * z) + np.sin(eta[3] * z)))
    return y


def _transform_torch(x, tau, eta, torch):
    tiny = torch.finfo(x.dtype).tiny
    tau_b = tau.reshape(1, -1, 1)
    eta0 = eta[:, 0].reshape(1, -1, 1)
    eta1 = eta[:, 1].reshape(1, -1, 1)
    eta2 = eta[:, 2].reshape(1, -1, 1)
    eta3 = eta[:, 3].reshape(1, -1, 1)
    positive = x > 0
    nonzero = x != 0
    # Positive and negative coordinates share the same magnitude transform;
    # only their two eta coefficients differ.  Selecting those coefficients
    # first avoids evaluating both full log/sin/exp branches for every input.
    # This is algebraically identical to the historical two-branch formula
    # and leaves the zero-coordinate result unchanged.
    z = torch.log(torch.clamp(torch.abs(x), min=tiny))
    eta_first = torch.where(positive, eta0, eta2)
    eta_second = torch.where(positive, eta1, eta3)
    magnitude = torch.exp(
        z + tau_b * (torch.sin(eta_first * z) + torch.sin(eta_second * z))
    )
    transformed = torch.where(positive, magnitude, -magnitude)
    return torch.where(nonzero, transformed, x)
