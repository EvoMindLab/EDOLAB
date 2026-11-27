"""Object-oriented Generalized Dynamic Benchmark Generator (GDBG)."""

from __future__ import annotations

from typing import Callable, ClassVar

import numpy as np

from core.compute import record_compute_use, select_torch_device
from core.parameters import ParameterSpec
from core.problem.base import BaseProblem
from core.problem.bounds import Bounds
from core.registry import problem_registry


_GDBG_FUNCTION_RANGES = {
    1: (-100.0, 100.0),
    2: (-5.0, 5.0),
    3: (-0.5, 0.5),
    4: (-100.0, 100.0),
    5: (-32.0, 32.0),
}


@problem_registry.register("GDBG")
class GDBGProblem(BaseProblem):
    name: ClassVar[str] = "GDBG"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        entries = [
            (
                "MinCoordinate",
                -50,
                "numeric",
                (-1000, 1000),
                "Lower bound of the search space.",
            ),
            (
                "MaxCoordinate",
                50,
                "numeric",
                (-1000, 1000),
                "Upper bound of the search space.",
            ),
            (
                "Dimension",
                5,
                "integer",
                (2, 100),
                "Number of dimensions in the optimization problem.",
            ),
            (
                "PeakNumber",
                10,
                "integer",
                (1, 500),
                "Number of peaks in the dynamic environment.",
            ),
            (
                "ChangeFrequency",
                5000,
                "integer",
                (100, 20000),
                "Number of function evaluations between environment changes.",
            ),
            (
                "ShiftSeverity",
                1,
                "numeric",
                (0, 10),
                "Magnitude of positional shifts during environment changes.",
            ),
            (
                "EnvironmentNumber",
                100,
                "integer",
                (1, 1000),
                "Total number of distinct environmental states.",
            ),
            (
                "HeightSeverity",
                7,
                "numeric",
                (0, 20),
                "Intensity factor for peak height modifications.",
            ),
            (
                "WidthSeverity",
                1,
                "numeric",
                (0, 6),
                "Intensity factor for peak width modifications.",
            ),
            (
                "EvaluationDevice",
                "auto",
                "option",
                ("auto", "cuda", "cuda:<id>", "mps", "cpu", "numpy"),
                "Compute backend for GDBG fitness evaluation.",
            ),
        ]
        return {
            name: ParameterSpec(name, default, kind, bounds, description)
            for name, default, kind, bounds, description in entries
        }

    def __getattr__(self, name: str):
        parameters = self.__dict__.get("parameters", {})
        if name in parameters:
            return parameters[name]
        raise AttributeError(name)

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

    def _build_environments(self) -> None:
        env_count = self.environment_number
        peak_count = int(self.parameters["PeakNumber"])
        dimension = self.dimension
        rng = self.rng
        bounds = self.bounds

        self.min_height = 30.0
        self.max_height = 70.0
        self.min_width = 1.0
        self.max_width = 12.0
        self.min_function_id = 1
        self.max_function_id = 5
        self.benchmark_name = self.name

        self.peak_visibility = np.zeros((env_count, peak_count), dtype=int)
        self.optimum_values = np.full(env_count, np.nan, dtype=float)
        self.optimum_ids = np.full(env_count, -1, dtype=int)
        self.peaks_height = np.full((env_count, peak_count), np.nan, dtype=float)
        self.peaks_position = np.full((peak_count, dimension, env_count), np.nan, dtype=float)
        self.function_select = np.full((env_count, peak_count), -1, dtype=int)

        self.peaks_position[:, :, 0] = bounds.sample(peak_count, rng)
        self.peaks_height[0] = self.min_height + (
            self.max_height - self.min_height
        ) * rng.random(peak_count)
        self.optimum_ids[0] = int(np.argmax(self.peaks_height[0]))
        self.optimum_values[0] = float(self.peaks_height[0, self.optimum_ids[0]])
        self.function_select[0] = rng.integers(
            self.min_function_id,
            self.max_function_id + 1,
            size=peak_count,
        )

        self.rotation_matrix = self._generate_rotation_matrix(dimension, rng)
        self.theta = float(self._theta)
        self.peak_visibility[0] = self._peak_visibility(0)

        for env_index in range(1, env_count):
            self.function_select[env_index] = rng.integers(
                self.min_function_id,
                self.max_function_id + 1,
                size=peak_count,
            )
            shift_offset = rng.standard_normal((peak_count, dimension))
            norms = np.linalg.norm(shift_offset, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            shift = shift_offset / norms * float(self.parameters["ShiftSeverity"])

            peaks_position = self.peaks_position[:, :, env_index - 1] + shift
            peaks_height = self.peaks_height[env_index - 1] + (
                float(self.parameters["HeightSeverity"]) * rng.standard_normal(peak_count)
            )
            self.peaks_position[:, :, env_index] = bounds.reflect(peaks_position)
            self.peaks_height[env_index] = self._reflect(
                peaks_height,
                self.min_height,
                self.max_height,
            )
            self.optimum_ids[env_index] = int(np.argmax(self.peaks_height[env_index]))
            self.optimum_values[env_index] = float(
                self.peaks_height[env_index, self.optimum_ids[env_index]]
            )
            self.peak_visibility[env_index] = self._peak_visibility(env_index)

        self._publish_matlab_aliases()

    def _publish_matlab_aliases(self) -> None:
        self.BenchmarkName = self.benchmark_name
        self.MinHeight = self.min_height
        self.MaxHeight = self.max_height
        self.MinWidth = self.min_width
        self.MaxWidth = self.max_width
        self.MinFunctionID = self.min_function_id
        self.MaxFunctionID = self.max_function_id
        self.PeakVisibility = self.peak_visibility
        self.OptimumValue = self.optimum_values
        self.OptimumID = self.optimum_ids
        self.PeaksHeight = self.peaks_height
        self.PeaksPosition = self.peaks_position
        self.FunctionSelect = self.function_select
        self.M = self.rotation_matrix
        self.Theta = self.theta
        self._torch_env_cache: dict[tuple[int, str], tuple[object, ...]] = {}
        self._torch_basic_function_cache: dict[tuple[str, int], dict[str, object]] = {}
        self._explicit_evaluation_backend: tuple[object | None, object | None, str] | None = None

    def raw_fitness(self, x: np.ndarray) -> float:
        sample = np.asarray(x, dtype=float).reshape(-1)
        if sample.size != self.dimension:
            raise ValueError(f"GDBG expects dimension {self.dimension}.")
        sample = self.bounds.clip(sample)
        return self._fitness_in_environment(sample, int(self.state.environment_index))

    def raw_fitness_batch(self, x: np.ndarray) -> np.ndarray:
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if samples.shape[1] != self.dimension:
            raise ValueError(f"GDBG expects dimension {self.dimension}.")
        samples = self.bounds.clip(samples)
        device_policy = str(self.parameters.get("EvaluationDevice", "auto"))
        if device_policy.strip().lower() == "numpy":
            record_compute_use("problem:GDBG", "numpy:cpu")
            return self._raw_fitness_batch_numpy(samples)

        work_items = int(samples.shape[0]) * int(self.parameters["PeakNumber"]) * self.dimension
        torch, device, label = self._select_evaluation_backend(
            work_items,
            device_policy,
        )
        if torch is None or device is None:
            return self._raw_fitness_batch_numpy(samples)
        try:
            return self._raw_fitness_batch_torch(samples, torch, device)
        except Exception as exc:
            record_compute_use(
                "problem:GDBG",
                "numpy:cpu",
                f"{label} failed; using NumPy CPU: {type(exc).__name__}",
            )
            return self._raw_fitness_batch_numpy(samples)

    def _select_evaluation_backend(self, work_items: int, policy: str):
        normalized = policy.strip().lower()
        if normalized != "auto" and self._explicit_evaluation_backend is not None:
            return self._explicit_evaluation_backend
        selected = select_torch_device(
            "problem:GDBG",
            work_items=work_items,
            device=policy,
        )
        if normalized != "auto":
            self._explicit_evaluation_backend = selected
        return selected

    def _raw_fitness_batch_numpy(self, samples: np.ndarray) -> np.ndarray:
        env_index = int(self.state.environment_index)
        function_ids = self.function_select[env_index].astype(int)
        centers = self.peaks_position[:, :, env_index]
        heights = self.peaks_height[env_index]
        dimension = self.dimension

        delta = samples[:, None, :] - centers[None, :, :]
        weights = np.exp(-np.sqrt(np.sum(delta * delta, axis=2) / (2.0 * dimension)))
        max_weight = np.max(weights, axis=1, keepdims=True)
        weights = np.where(weights != max_weight, weights * (1.0 - max_weight**10), weights)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        weights = np.divide(
            weights,
            weight_sum,
            out=np.full_like(weights, 1.0 / weights.shape[1]),
            where=weight_sum != 0.0,
        )

        lower = np.array([_GDBG_FUNCTION_RANGES[int(fid)][0] for fid in function_ids], dtype=float)
        upper = np.array([_GDBG_FUNCTION_RANGES[int(fid)][1] for fid in function_ids], dtype=float)
        span = float(self.parameters["MaxCoordinate"]) - float(self.parameters["MinCoordinate"])
        lambdas = span / (upper - lower)
        transformed = np.einsum("npd,dk->npk", delta / lambdas[None, :, None], self.rotation_matrix)

        components = np.empty((samples.shape[0], function_ids.size), dtype=float)
        for function_id in range(self.min_function_id, self.max_function_id + 1):
            mask = function_ids == function_id
            if not np.any(mask):
                continue
            fi_max = self._basic_function_batch_numpy(
                function_id,
                np.full((1, dimension), _GDBG_FUNCTION_RANGES[function_id][1]) @ self.rotation_matrix,
            )[0]
            values = self._basic_function_batch_numpy(function_id, transformed[:, mask, :])
            components[:, mask] = -2000.0 * values / abs(float(fi_max)) + heights[mask]
        return np.sum(weights * components, axis=1).astype(float)

    def _raw_fitness_batch_torch(self, samples: np.ndarray, torch, device) -> np.ndarray:
        function_ids, centers, heights, lambdas, rotation, fi_max, basic_constants = self._torch_environment(
            int(self.state.environment_index),
            torch,
            device,
        )
        with torch.inference_mode():
            sample_t = torch.as_tensor(samples, dtype=torch.float64, device=device)
            delta = sample_t[:, None, :] - centers[None, :, :]
            weights = torch.exp(
                -torch.sqrt(torch.sum(delta * delta, dim=2) / (2.0 * float(self.dimension)))
            )
            max_weight = torch.max(weights, dim=1, keepdim=True).values
            weights = torch.where(weights != max_weight, weights * (1.0 - max_weight**10), weights)
            weight_sum = torch.sum(weights, dim=1, keepdim=True)
            uniform = torch.full_like(weights, 1.0 / float(weights.shape[1]))
            weights = torch.where(weight_sum != 0.0, weights / weight_sum, uniform)

            transformed = torch.matmul(
                delta / lambdas.reshape(1, -1, 1),
                rotation,
            )
            components = torch.empty(
                (sample_t.shape[0], function_ids.shape[0]),
                dtype=torch.float64,
                device=device,
            )
            for function_id in range(self.min_function_id, self.max_function_id + 1):
                mask = function_ids == int(function_id)
                if bool(torch.any(mask)):
                    values = self._basic_function_batch_torch(
                        function_id,
                        transformed[:, mask, :],
                        torch,
                        constants=basic_constants,
                    )
                    components[:, mask] = -2000.0 * values / torch.abs(fi_max[int(function_id)]) + heights[mask]
            result = torch.sum(weights * components, dim=1)
            return result.detach().cpu().numpy().astype(float)

    def _torch_environment(self, env_index: int, torch, device) -> tuple[object, ...]:
        key = (int(env_index), str(device))
        cached = self._torch_env_cache.get(key)
        if cached is not None:
            return cached

        function_ids_np = self.function_select[env_index].astype(int)
        lower = np.array([_GDBG_FUNCTION_RANGES[int(fid)][0] for fid in function_ids_np], dtype=float)
        upper = np.array([_GDBG_FUNCTION_RANGES[int(fid)][1] for fid in function_ids_np], dtype=float)
        span = float(self.parameters["MaxCoordinate"]) - float(self.parameters["MinCoordinate"])
        lambdas_np = span / (upper - lower)
        fi_max_np = np.zeros(self.max_function_id + 1, dtype=float)
        for function_id in range(self.min_function_id, self.max_function_id + 1):
            fi_max_np[function_id] = self._basic_function_batch_numpy(
                function_id,
                np.full((1, self.dimension), _GDBG_FUNCTION_RANGES[function_id][1]) @ self.rotation_matrix,
            )[0]
        basic_key = (str(device), self.dimension)
        basic_constants = self._torch_basic_function_cache.get(basic_key)
        if basic_constants is None:
            k = torch.arange(21, dtype=torch.float64, device=device)
            factors = 0.5**k
            frequencies = 3.0**k
            basic_constants = {
                "weierstrass_factors": factors,
                "weierstrass_frequencies": frequencies,
                "weierstrass_sum2": torch.sum(
                    factors * torch.cos(torch.pi * frequencies)
                ),
                "griewank_divisors": torch.sqrt(
                    torch.arange(
                        1,
                        self.dimension + 1,
                        dtype=torch.float64,
                        device=device,
                    )
                ),
            }
            self._torch_basic_function_cache[basic_key] = basic_constants
        values = (
            torch.as_tensor(function_ids_np, dtype=torch.long, device=device),
            torch.as_tensor(self.peaks_position[:, :, env_index], dtype=torch.float64, device=device),
            torch.as_tensor(self.peaks_height[env_index], dtype=torch.float64, device=device),
            torch.as_tensor(lambdas_np, dtype=torch.float64, device=device),
            torch.as_tensor(self.rotation_matrix, dtype=torch.float64, device=device),
            torch.as_tensor(fi_max_np, dtype=torch.float64, device=device),
            basic_constants,
        )
        self._torch_env_cache[key] = values
        return values

    def _fitness_in_environment(self, sample: np.ndarray, env_index: int) -> float:
        selected_indices = self.function_select[env_index].astype(int)
        centers = self.peaks_position[:, :, env_index]
        heights = self.peaks_height[env_index]
        peak_count = selected_indices.size
        sigma = np.ones(peak_count, dtype=float)

        weights = np.asarray(
            [
                self._weight(sample, centers[peak_index], sigma[peak_index])
                for peak_index in range(peak_count)
            ],
            dtype=float,
        )
        max_weight = float(np.max(weights))
        weights = np.where(weights != max_weight, weights * (1.0 - max_weight**10), weights)
        weight_sum = float(np.sum(weights))
        if weight_sum == 0.0:
            weights = np.full(peak_count, 1.0 / peak_count)
        else:
            weights = weights / weight_sum

        total = 0.0
        coordinate_span = float(self.parameters["MaxCoordinate"]) - float(
            self.parameters["MinCoordinate"]
        )
        for peak_index, function_id in enumerate(selected_indices):
            func, search_range = self._basic_function(int(function_id))
            lambda_i = sigma[peak_index] * (
                coordinate_span / (search_range[1] - search_range[0])
            )
            fi_max = func((search_range[1] * np.ones(self.dimension)) @ self.rotation_matrix)
            transformed = ((sample - centers[peak_index]) / lambda_i) @ self.rotation_matrix
            fi0 = 2000.0 * func(transformed) / abs(fi_max)
            component = -fi0 + heights[peak_index]
            total += weights[peak_index] * component
        return float(total)

    def _peak_visibility(self, env_index: int) -> np.ndarray:
        visibility = np.zeros(int(self.parameters["PeakNumber"]), dtype=int)
        for peak_index in range(int(self.parameters["PeakNumber"])):
            position = self.peaks_position[peak_index, :, env_index]
            value = self._fitness_in_environment(position, env_index)
            if abs(value - self.peaks_height[env_index, peak_index]) < 0.1:
                visibility[peak_index] = 1
        return visibility

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
        if samples.shape[1] != self.dimension:
            raise ValueError(f"GDBG expects dimension {self.dimension}.")
        samples = self.bounds.clip(samples)
        env = self.state.environment_index if environment_index is None else int(environment_index)
        peak_positions = self.peaks_position[:, :, env]
        distances = np.linalg.norm(samples[:, None, :] - peak_positions[None, :, :], axis=2)
        nearest_peak_ids = np.argmin(distances, axis=1)
        return nearest_peak_ids == int(self.optimum_ids[env])

    def visualization_peaks(self, environment_number: int | None = None) -> dict[str, object]:
        if self.dimension != 2:
            return {}
        env_index = self.resolve_environment_index(environment_number)
        optimum_id = int(self.optimum_ids[env_index])
        visible_mask = self.peak_visibility[env_index].astype(bool)
        peak_positions = self.peaks_position[:, :2, env_index]
        return {
            "global_optimum_id": optimum_id + 1,
            "global_optimum_position": peak_positions[optimum_id].copy(),
            "visible_peak_ids": np.flatnonzero(visible_mask) + 1,
            "visible_peak_positions": peak_positions[visible_mask].copy(),
        }

    @classmethod
    def _basic_function(cls, function_id: int) -> tuple[Callable[[np.ndarray], float], tuple[float, float]]:
        functions = {
            1: (cls._sphere, (-100.0, 100.0)),
            2: (cls._rastrigin, (-5.0, 5.0)),
            3: (cls._weierstrass, (-0.5, 0.5)),
            4: (cls._griewank, (-100.0, 100.0)),
            5: (cls._ackley, (-32.0, 32.0)),
        }
        return functions[int(function_id)]

    @staticmethod
    def _reflect(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        reflected = np.asarray(values, dtype=float).copy()
        high = reflected > upper
        reflected[high] = 2 * upper - reflected[high]
        low = reflected < lower
        reflected[low] = 2 * lower - reflected[low]
        return np.clip(reflected, lower, upper)

    def _generate_rotation_matrix(
        self,
        dimension: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        self._theta = 2.0 * np.pi * rng.random()
        output = np.eye(dimension)
        pair_count = int(2 * np.floor(1 + (dimension / 2 - 1) * rng.random()))
        available = list(range(dimension))
        selected: list[int] = []
        for _ in range(pair_count):
            if len(available) == 1:
                idx = 0
            else:
                idx = int(np.floor((len(available) - 1) * rng.random()))
            selected.append(available.pop(idx))

        for idx in range(0, len(selected) - 1, 2):
            left = selected[idx]
            right = selected[idx + 1]
            rotation = np.eye(dimension)
            rotation[left, left] = np.cos(self._theta)
            rotation[left, right] = -np.sin(self._theta)
            rotation[right, left] = np.sin(self._theta)
            rotation[right, right] = np.cos(self._theta)
            output = output @ rotation
        return output

    @staticmethod
    def _weight(x: np.ndarray, center: np.ndarray, sigma: float) -> float:
        return float(np.exp(-np.sqrt(np.sum((x - center) ** 2) / (2 * x.size * sigma**2))))

    @staticmethod
    def _sphere(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x**2))

    @staticmethod
    def _rastrigin(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))

    @staticmethod
    def _weierstrass(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        a = 0.5
        b = 3
        kmax = 20
        sum1 = 0.0
        for k in range(kmax + 1):
            sum1 += (a**k) * np.cos(2 * np.pi * (b**k) * (x + 0.5))
        sum2 = 0.0
        for k in range(kmax + 1):
            sum2 += (a**k) * np.cos(np.pi * (b**k))
        return float(np.sum(sum1) - x.size * sum2)

    @staticmethod
    def _griewank(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        sum_sq = np.sum(x**2)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))
        return float(1 + sum_sq / 4000 - prod_cos)

    @staticmethod
    def _ackley(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        n = x.size
        return float(-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e)

    @staticmethod
    def _basic_function_batch_numpy(function_id: int, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if int(function_id) == 1:
            return np.sum(values * values, axis=-1)
        if int(function_id) == 2:
            return np.sum(values * values - 10.0 * np.cos(2.0 * np.pi * values) + 10.0, axis=-1)
        if int(function_id) == 3:
            a = 0.5
            b = 3.0
            k = np.arange(21, dtype=float)
            factors = a**k
            frequencies = b**k
            sum1 = np.sum(
                factors.reshape((1,) * values.ndim + (21,))
                * np.cos(
                    2.0
                    * np.pi
                    * frequencies.reshape((1,) * values.ndim + (21,))
                    * (values[..., None] + 0.5)
                ),
                axis=(-2, -1),
            )
            sum2 = float(np.sum(factors * np.cos(np.pi * frequencies)))
            return sum1 - values.shape[-1] * sum2
        if int(function_id) == 4:
            dim = values.shape[-1]
            sum_sq = np.sum(values * values, axis=-1)
            prod_cos = np.prod(np.cos(values / np.sqrt(np.arange(1, dim + 1, dtype=float))), axis=-1)
            return 1.0 + sum_sq / 4000.0 - prod_cos
        if int(function_id) == 5:
            dim = values.shape[-1]
            sum_sq = np.sum(values * values, axis=-1)
            sum_cos = np.sum(np.cos(2.0 * np.pi * values), axis=-1)
            return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / dim)) - np.exp(sum_cos / dim) + 20.0 + np.e
        raise ValueError(f"Unknown GDBG function id {function_id}.")

    @staticmethod
    def _basic_function_batch_torch(function_id: int, x, torch, *, constants=None):
        if int(function_id) == 1:
            return torch.sum(x * x, dim=-1)
        if int(function_id) == 2:
            return torch.sum(x * x - 10.0 * torch.cos(2.0 * torch.pi * x) + 10.0, dim=-1)
        if int(function_id) == 3:
            if constants is None:
                k = torch.arange(21, dtype=x.dtype, device=x.device)
                factors = 0.5**k
                frequencies = 3.0**k
                sum2 = torch.sum(factors * torch.cos(torch.pi * frequencies))
            else:
                factors = constants["weierstrass_factors"]
                frequencies = constants["weierstrass_frequencies"]
                sum2 = constants["weierstrass_sum2"]
            sum1 = torch.sum(
                factors.reshape((1,) * x.ndim + (21,))
                * torch.cos(
                    2.0
                    * torch.pi
                    * frequencies.reshape((1,) * x.ndim + (21,))
                    * (x.unsqueeze(-1) + 0.5)
                ),
                dim=(-2, -1),
            )
            return sum1 - x.shape[-1] * sum2
        if int(function_id) == 4:
            dim = x.shape[-1]
            divisors = (
                torch.sqrt(torch.arange(1, dim + 1, dtype=x.dtype, device=x.device))
                if constants is None
                else constants["griewank_divisors"]
            )
            sum_sq = torch.sum(x * x, dim=-1)
            prod_cos = torch.prod(torch.cos(x / divisors), dim=-1)
            return 1.0 + sum_sq / 4000.0 - prod_cos
        if int(function_id) == 5:
            dim = x.shape[-1]
            sum_sq = torch.sum(x * x, dim=-1)
            sum_cos = torch.sum(torch.cos(2.0 * torch.pi * x), dim=-1)
            return -20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq / dim)) - torch.exp(sum_cos / dim) + 20.0 + torch.e
        raise ValueError(f"Unknown GDBG function id {function_id}.")
