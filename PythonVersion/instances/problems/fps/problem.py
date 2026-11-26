"""Object-oriented Free Peaks Benchmark (FPs) problem instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.parameters import ParameterSpec
from core.problem.base import BaseProblem
from core.problem.bounds import Bounds
from core.registry import problem_registry


@problem_registry.register("FPs")
class FPSProblem(BaseProblem):
    name: ClassVar[str] = "FPs"

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
                "Lambda",
                1,
                "boolean",
                (0, 1),
                "Direction correlation: 1=fixed direction, 0=random movement.",
            ),
            (
                "SpaceChange",
                1,
                "boolean",
                (0, 1),
                "Enable subspace reorganization each environment (1=on, 0=off).",
            ),
            (
                "FunctionChange",
                1,
                "boolean",
                (0, 1),
                "Enable function type changes each environment (1=on, 0=off).",
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
        # The MATLAB FPs generator accepts these parameters, then overwrites
        # them before environment generation. Preserve that runtime behavior.
        self.parameters["Lambda"] = 1
        self.parameters["SpaceChange"] = 0
        self.parameters["FunctionChange"] = 0

        env_count = self.environment_number
        peak_count = int(self.parameters["PeakNumber"])
        dimension = self.dimension
        rng = self.rng

        self.l_sv = -100.0
        self.u_sv = 100.0
        self.min_height = 0.0
        self.max_height = 100.0
        self.min_function_id = 1
        self.max_function_id = 8
        self.lowest_value = 0.0
        self.benchmark_name = self.name

        self.optimum_values = np.full(env_count, np.nan, dtype=float)
        self.optimum_ids = np.full(env_count, -1, dtype=int)
        self.peaks_height = np.full((env_count, peak_count), np.nan, dtype=float)
        self.peaks_position = np.full(
            (peak_count, dimension, env_count),
            np.nan,
            dtype=float,
        )
        self.peaks_position_map = np.full(
            (peak_count, dimension, env_count),
            np.nan,
            dtype=float,
        )
        self.function_select = np.full((env_count, peak_count), -1, dtype=int)
        self.subspace = np.full((dimension, 2, peak_count, env_count), np.nan, dtype=float)
        self.subspace_lowest = np.full((env_count, peak_count), np.nan, dtype=float)
        self.subspace_linkage = np.eye(peak_count, dtype=int)
        self.peak_visibility = np.ones((env_count, peak_count), dtype=int)
        self.subspace_weight = np.zeros((env_count, peak_count), dtype=float)

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

        self.subspace_weight[0] = np.ones(peak_count) / peak_count
        subspaces, linkage = self._kd_tree_partition(
            peak_count,
            np.tile(
                [self.parameters["MinCoordinate"], self.parameters["MaxCoordinate"]],
                (dimension, 1),
            ),
        )
        self.subspace_linkage = linkage
        self.subspace[:, :, :, 0] = self._subspaces_to_tensor(
            subspaces,
            dimension,
            peak_count,
        )

        for peak_index in range(peak_count):
            for dim_index in range(dimension):
                low, high = self.subspace[dim_index, :, peak_index, 0]
                self.peaks_position[peak_index, dim_index, 0] = (
                    low + (high - low) * rng.random()
                )

        self.peaks_position_map[:, :, 0], _ = self._transfer(
            self.peaks_position[:, :, 0],
            0,
            include_upper=False,
        )
        self.subspace_lowest[0] = self._subspace_lowest(0)

        for env_index in range(1, env_count):
            self.subspace[:, :, :, env_index] = self.subspace[:, :, :, env_index - 1]

            direction = -1.0 + 2.0 * rng.random((peak_count, dimension))
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            direction = direction / norms
            shift = float(self.parameters["ShiftSeverity"]) * rng.standard_normal()

            if env_index - 1 > 0:
                previous_step = (
                    self.peaks_position[:, :, env_index - 1]
                    - self.peaks_position[:, :, env_index - 2]
                )
                random_step = direction * (1 - int(self.parameters["Lambda"])) * shift
                self.peaks_position[:, :, env_index] = (
                    self.peaks_position[:, :, env_index - 1]
                    + previous_step * int(self.parameters["Lambda"])
                    + random_step
                )
            else:
                self.peaks_position[:, :, env_index] = (
                    self.peaks_position[:, :, env_index - 1] + direction * shift
                )

            self._fold_positions_into_subspaces(env_index)
            self._evolve_heights(env_index)

            if int(self.parameters["FunctionChange"]) == 1:
                self.function_select[env_index] = rng.integers(
                    self.min_function_id,
                    self.max_function_id + 1,
                    size=peak_count,
                )
            else:
                self.function_select[env_index] = self.function_select[env_index - 1]

            self.peaks_position_map[:, :, env_index], _ = self._transfer(
                self.peaks_position[:, :, env_index],
                env_index,
                include_upper=False,
            )
            self.subspace_lowest[env_index] = self._subspace_lowest(env_index)
            self.optimum_ids[env_index] = int(np.argmax(self.peaks_height[env_index]))
            self.optimum_values[env_index] = float(
                self.peaks_height[env_index, self.optimum_ids[env_index]]
            )

        self._publish_matlab_aliases()

    def _publish_matlab_aliases(self) -> None:
        self.BenchmarkName = self.benchmark_name
        self.MinHeight = self.min_height
        self.MaxHeight = self.max_height
        self.MinFunctionID = self.min_function_id
        self.MaxFunctionID = self.max_function_id
        self.LowestValue = self.lowest_value
        self.OptimumValue = self.optimum_values
        self.OptimumID = self.optimum_ids
        self.PeaksHeight = self.peaks_height
        self.PeaksPosition = self.peaks_position
        self.PeaksPositionMap = self.peaks_position_map
        self.FunctionSelect = self.function_select
        self.SubSpace = self.subspace
        self.SubSpaceLowest = self.subspace_lowest
        self.SubSpaceLinkage = self.subspace_linkage
        self.PeakVisibility = self.peak_visibility
        self.SubspaceWeight = self.subspace_weight

    def _evolve_heights(self, env_index: int) -> None:
        for peak_index in range(int(self.parameters["PeakNumber"])):
            value = self.peaks_height[env_index - 1, peak_index] + (
                float(self.parameters["HeightSeverity"]) * self.rng.standard_normal()
            )
            while value < self.min_height:
                value += abs(
                    float(self.parameters["HeightSeverity"]) * self.rng.standard_normal()
                )
            while value > self.max_height:
                value -= abs(
                    float(self.parameters["HeightSeverity"]) * self.rng.standard_normal()
                )
            self.peaks_height[env_index, peak_index] = value

    def raw_fitness(self, x: np.ndarray) -> float:
        sample = np.asarray(x, dtype=float).reshape(-1)
        if sample.size != self.dimension:
            raise ValueError(f"FPs expects dimension {self.dimension}.")
        sample = self.bounds.clip(sample)
        env_index = int(self.state.environment_index)
        mapped, subspace_ids = self._transfer(
            sample.reshape(1, -1),
            env_index,
            include_upper=True,
        )
        peak_id = int(subspace_ids[0])
        if peak_id < 0:
            peak_id = self._nearest_subspace(sample, env_index)
        function_id = int(self.function_select[env_index, peak_id])
        return float(
            self._select_fitness_func(
                mapped[0],
                peak_id,
                function_id,
                env_index,
                self.lowest_value,
            )
        )

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
            raise ValueError(f"FPs expects dimension {self.dimension}.")
        samples = np.clip(
            samples,
            float(self.parameters["MinCoordinate"]),
            float(self.parameters["MaxCoordinate"]),
        )
        env = (
            self.state.environment_index
            if environment_index is None
            else int(environment_index)
        )
        _, subspace_ids = self._transfer(samples, env, include_upper=True)
        for sample_index, subspace_id in enumerate(subspace_ids):
            if subspace_id < 0:
                subspace_ids[sample_index] = self._nearest_subspace(samples[sample_index], env)
        return subspace_ids == int(self.optimum_ids[env])

    def visualization_peaks(self, environment_number: int | None = None) -> dict[str, object]:
        if self.dimension != 2:
            return {}
        env_index = self.resolve_environment_index(environment_number)
        optimum_id = int(self.optimum_ids[env_index])
        visible_mask = self.peak_visibility[env_index].astype(bool)
        return {
            "global_optimum_id": optimum_id + 1,
            "global_optimum_position": self.peaks_position[
                optimum_id,
                :2,
                env_index,
            ].copy(),
            "visible_peak_ids": np.flatnonzero(visible_mask) + 1,
            "visible_peak_positions": self.peaks_position[visible_mask, :2, env_index].copy(),
        }

    @staticmethod
    def _kd_tree_partition(num: int, boundary: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        bounds = np.vstack([boundary[:, 0], boundary[:, 1]])
        leaves: list[np.ndarray] = []

        def divide(current_bounds: np.ndarray, count: int, depth: int) -> None:
            if count == 1:
                leaves.append(current_bounds.copy())
                return
            dim = depth % current_bounds.shape[1]
            if count % 2 == 1:
                left_count = (count - 1) // 2
                right_count = (count + 1) // 2
                split_ratio = left_count / count
            else:
                left_count = right_count = count // 2
                split_ratio = 0.5
            split = current_bounds[0, dim] + (
                current_bounds[1, dim] - current_bounds[0, dim]
            ) * split_ratio
            left = current_bounds.copy()
            right = current_bounds.copy()
            left[1, dim] = split
            right[0, dim] = split
            divide(left, left_count, depth + 1)
            divide(right, right_count, depth + 1)

        divide(bounds, num, 0)
        return leaves, FPSProblem._adjacency(leaves)

    @staticmethod
    def _subspaces_to_tensor(
        subspaces: list[np.ndarray],
        dimension: int,
        peak_count: int,
    ) -> np.ndarray:
        tensor = np.zeros((dimension, 2, peak_count), dtype=float)
        for peak_index, bounds in enumerate(subspaces):
            for dim_index in range(dimension):
                tensor[dim_index, :, peak_index] = bounds[:, dim_index]
        return tensor

    @staticmethod
    def _adjacency(subspaces: list[np.ndarray]) -> np.ndarray:
        count = len(subspaces)
        adjacency = np.eye(count, dtype=int)
        for left in range(count):
            for right in range(left + 1, count):
                if FPSProblem._are_adjacent(subspaces[left], subspaces[right]):
                    adjacency[left, right] = 1
                    adjacency[right, left] = 1
        return adjacency

    @staticmethod
    def _are_adjacent(left: np.ndarray, right: np.ndarray) -> bool:
        touching = 0
        for dim_index in range(left.shape[1]):
            if np.isclose(left[1, dim_index], right[0, dim_index]) or np.isclose(
                right[1, dim_index],
                left[0, dim_index],
            ):
                touching += 1
            elif (
                left[1, dim_index] <= right[0, dim_index]
                or right[1, dim_index] <= left[0, dim_index]
            ):
                return False
        return touching == 1

    def _transfer(
        self,
        x: np.ndarray,
        env_index: int,
        include_upper: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        mapped = np.zeros((samples.shape[0], self.dimension), dtype=float)
        subspace_ids = np.full(samples.shape[0], -1, dtype=int)
        for sample_index, sample in enumerate(samples):
            for peak_index in range(int(self.parameters["PeakNumber"])):
                lows = self.subspace[:, 0, peak_index, env_index]
                highs = self.subspace[:, 1, peak_index, env_index]
                if include_upper:
                    inside = np.all((sample >= lows) & (sample <= highs))
                else:
                    inside = np.all((sample >= lows) & (sample < highs))
                if inside:
                    subspace_ids[sample_index] = peak_index
                    mapped[sample_index] = self.l_sv + (
                        (self.u_sv - self.l_sv) * (sample - lows) / (highs - lows)
                    )
                    break
        return mapped, subspace_ids

    def _nearest_subspace(self, sample: np.ndarray, env_index: int) -> int:
        centers = []
        for peak_index in range(int(self.parameters["PeakNumber"])):
            lows = self.subspace[:, 0, peak_index, env_index]
            highs = self.subspace[:, 1, peak_index, env_index]
            centers.append((lows + highs) / 2.0)
        distances = [np.linalg.norm(sample - center) for center in centers]
        return int(np.argmin(distances))

    def _subspace_lowest(self, env_index: int) -> np.ndarray:
        previous_env = self.state.environment_index
        previous_lowest = self.subspace_lowest[env_index].copy()
        try:
            self.state.environment_index = env_index
            self.subspace_lowest[env_index] = np.nan
            boundary = np.zeros(
                (int(self.parameters["PeakNumber"]), self.dimension),
                dtype=float,
            )
            for peak_index in range(int(self.parameters["PeakNumber"])):
                boundary[peak_index] = self.subspace[:, 0, peak_index, env_index]
            return np.asarray([self.raw_fitness(row) for row in boundary], dtype=float)
        finally:
            self.subspace_lowest[env_index] = previous_lowest
            self.state.environment_index = previous_env

    def _fold_positions_into_subspaces(self, env_index: int) -> None:
        for peak_index in range(int(self.parameters["PeakNumber"])):
            for dim_index in range(self.dimension):
                low, high = self.subspace[dim_index, :, peak_index, env_index]
                value = self.peaks_position[peak_index, dim_index, env_index]
                if value > high:
                    value = low + ((high - low) ** 2) / (value - low)
                elif value < low:
                    value = low + (high - low) * (low - value) / (high - value)
                self.peaks_position[peak_index, dim_index, env_index] = value

    def _select_fitness_func(
        self,
        x: np.ndarray,
        peak_id: int,
        function_id: int,
        env_index: int,
        lowest_value: float,
    ) -> float:
        height = self.peaks_height[env_index, peak_id]
        position = self.peaks_position_map[peak_id, :, env_index]
        subspace = np.tile([self.l_sv, self.u_sv], (self.dimension, 1))
        if function_id == 1:
            value = self._s1(x, height, position, subspace)
        elif function_id in {2, 3}:
            value = self._s3(x, height, position, subspace)
        elif function_id in {4, 5}:
            value = self._s5(x, height, position, subspace)
        elif function_id == 6:
            value = self._s6(x, height, position, subspace, self.dimension)
        elif function_id == 7:
            value = self._s7(x, height, position, subspace, 50)
        elif function_id == 8:
            value = self._s8(x, height, position, subspace, 50, 3, 5.5)
        elif function_id == 9:
            value = self._s9(x, height, position, subspace)
        else:
            raise ValueError(f"Unknown FPs function id: {function_id}")

        lowest = self.subspace_lowest[env_index, peak_id]
        if not np.isnan(lowest):
            denominator = height - lowest
            if denominator != 0:
                value = lowest_value + (height - lowest_value) * (
                    (value - lowest) / denominator
                )
        return float(value)

    @staticmethod
    def _distance_like(x: np.ndarray, position: np.ndarray, subspace: np.ndarray) -> float:
        q = 1.0
        for index in range(x.size):
            if x[index] >= position[index]:
                q *= 1.0 - (x[index] - position[index]) / (
                    subspace[index, 1] - position[index]
                )
            else:
                q *= 1.0 - (position[index] - x[index]) / (
                    position[index] - subspace[index, 0]
                )
        return float(
            (1.0 - q) * np.sqrt(np.sum((subspace[:, 1] - subspace[:, 0]) ** 2))
        )

    @classmethod
    def _s1(
        cls,
        x: np.ndarray,
        height: float,
        position: np.ndarray,
        subspace: np.ndarray,
    ) -> float:
        return float(height - cls._distance_like(x, position, subspace))

    @classmethod
    def _s3(
        cls,
        x: np.ndarray,
        height: float,
        position: np.ndarray,
        subspace: np.ndarray,
    ) -> float:
        return float(height - np.sqrt(height * cls._distance_like(x, position, subspace)))

    @classmethod
    def _s5(
        cls,
        x: np.ndarray,
        height: float,
        position: np.ndarray,
        subspace: np.ndarray,
    ) -> float:
        dx = cls._distance_like(x, position, subspace)
        return float(height - dx**2 / height)

    @classmethod
    def _s6(
        cls,
        x: np.ndarray,
        height: float,
        position: np.ndarray,
        subspace: np.ndarray,
        dimension: int,
    ) -> float:
        dx = cls._distance_like(x, position, subspace)
        return float(height - np.exp(2.0 * np.sqrt(dx / np.sqrt(dimension))) + 1.0)

    @classmethod
    def _s7(
        cls,
        x: np.ndarray,
        height: float,
        position: np.ndarray,
        subspace: np.ndarray,
        radius: float,
    ) -> float:
        dx = cls._distance_like(x, position, subspace)
        if dx <= radius:
            return float(height * np.cos(np.pi * dx / radius))
        return float(-height - dx + radius)

    @classmethod
    def _s8(
        cls,
        x: np.ndarray,
        height: float,
        position: np.ndarray,
        subspace: np.ndarray,
        radius: float,
        m: float,
        n: float,
    ) -> float:
        dx = cls._distance_like(x, position, subspace)
        if dx <= radius:
            return float(
                height
                * (
                    np.cos(
                        m
                        * np.pi
                        * (dx - m * (m * dx) / radius * radius)
                        / radius
                    )
                    - n * (m * dx) / radius
                )
                / np.sqrt(dx + 1.0)
            )
        return float(height - n * m * np.sqrt(radius + 1.0) - dx + radius)

    @classmethod
    def _s9(
        cls,
        x: np.ndarray,
        height: float,
        position: np.ndarray,
        subspace: np.ndarray,
    ) -> float:
        dx = cls._distance_like(x, position, subspace)
        return float(height * np.exp(-(dx**2) / (2.0 * (200.0 / 5.0) ** 2)))
