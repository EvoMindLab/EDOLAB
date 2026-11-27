"""Object-oriented Moving Peaks Benchmark (MPB)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.parameters import ParameterSpec
from core.problem.base import BaseProblem
from core.problem.bounds import Bounds
from core.registry import problem_registry


@problem_registry.register("MPB")
class MPBProblem(BaseProblem):
    """Scenario 2 conical Moving Peaks Benchmark."""

    name: ClassVar[str] = "MPB"

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
        self.benchmark_name = self.name

        self.peak_visibility = np.zeros((env_count, peak_count), dtype=int)
        self.optimum_values = np.full(env_count, np.nan, dtype=float)
        self.optimum_ids = np.full(env_count, -1, dtype=int)
        self.peaks_height = np.full((env_count, peak_count), np.nan, dtype=float)
        self.peaks_position = np.full(
            (peak_count, dimension, env_count),
            np.nan,
            dtype=float,
        )
        self.peaks_width = np.full((env_count, peak_count), np.nan, dtype=float)

        self.peaks_position[:, :, 0] = bounds.sample(peak_count, rng)
        self.peaks_height[0] = self.min_height + (
            self.max_height - self.min_height
        ) * rng.random(peak_count)
        self.peaks_width[0] = self.min_width + (
            self.max_width - self.min_width
        ) * rng.random(peak_count)
        self._record_optimum(0)
        self.peak_visibility[0] = self._peak_visibility(0)

        for env_index in range(1, env_count):
            shift_offset = rng.standard_normal((peak_count, dimension))
            norms = np.linalg.norm(shift_offset, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            shift = shift_offset / norms * float(self.parameters["ShiftSeverity"])

            peaks_position = self.peaks_position[:, :, env_index - 1] + shift
            peaks_height = self.peaks_height[env_index - 1] + (
                float(self.parameters["HeightSeverity"]) * rng.standard_normal(peak_count)
            )
            peaks_width = self.peaks_width[env_index - 1] + (
                float(self.parameters["WidthSeverity"]) * rng.standard_normal(peak_count)
            )

            self.peaks_position[:, :, env_index] = bounds.reflect(peaks_position)
            self.peaks_height[env_index] = self._reflect(
                peaks_height,
                self.min_height,
                self.max_height,
            )
            self.peaks_width[env_index] = self._reflect(
                peaks_width,
                self.min_width,
                self.max_width,
            )
            self._record_optimum(env_index)
            self.peak_visibility[env_index] = self._peak_visibility(env_index)

        self._publish_matlab_aliases()

    def _publish_matlab_aliases(self) -> None:
        self.BenchmarkName = self.benchmark_name
        self.MinHeight = self.min_height
        self.MaxHeight = self.max_height
        self.MinWidth = self.min_width
        self.MaxWidth = self.max_width
        self.PeakVisibility = self.peak_visibility
        self.OptimumValue = self.optimum_values
        self.OptimumID = self.optimum_ids
        self.PeaksHeight = self.peaks_height
        self.PeaksPosition = self.peaks_position
        self.PeaksWidth = self.peaks_width

    def _record_optimum(self, env_index: int) -> None:
        optimum_id = int(np.argmax(self.peaks_height[env_index]))
        self.optimum_ids[env_index] = optimum_id
        self.optimum_values[env_index] = float(self.peaks_height[env_index, optimum_id])

    def raw_fitness(self, x: np.ndarray) -> float:
        sample = np.asarray(x, dtype=float).reshape(-1)
        if sample.size != self.dimension:
            raise ValueError(f"MPB expects dimension {self.dimension}.")
        sample = self.bounds.clip(sample)
        return self._fitness_in_environment(sample, int(self.state.environment_index))

    def _fitness_in_environment(self, sample: np.ndarray, env_index: int) -> float:
        positions = self.peaks_position[:, :, env_index]
        distances = np.linalg.norm(positions - sample.reshape(1, -1), axis=1)
        values = self.peaks_height[env_index] - self.peaks_width[env_index] * distances
        return float(np.max(values))

    def _peak_visibility(self, env_index: int) -> np.ndarray:
        visibility = np.zeros(int(self.parameters["PeakNumber"]), dtype=int)
        for peak_index in range(int(self.parameters["PeakNumber"])):
            position = self.peaks_position[peak_index, :, env_index]
            value = self._fitness_in_environment(position, env_index)
            if value == self.peaks_height[env_index, peak_index]:
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
            raise ValueError(f"MPB expects dimension {self.dimension}.")
        samples = self.bounds.clip(samples)
        env = (
            self.state.environment_index
            if environment_index is None
            else int(environment_index)
        )
        peak_positions = self.peaks_position[:, :, env]
        distances = np.linalg.norm(
            samples[:, None, :] - peak_positions[None, :, :],
            axis=2,
        )
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

    @staticmethod
    def _reflect(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        reflected = np.asarray(values, dtype=float).copy()
        high = reflected > upper
        reflected[high] = 2 * upper - reflected[high]
        low = reflected < lower
        reflected[low] = 2 * lower - reflected[low]
        return np.clip(reflected, lower, upper)
