"""Problem contracts and dynamic evaluation control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from core.parameters import ParameterSpec, resolve_parameters
from core.evaluation.base import MetricManager
from core.problem.bounds import Bounds
from core.registry import metric_registry
from utility.nbn import NBNContinuous


VISUALIZATION_FONT_FAMILY = "Times New Roman"


@dataclass(slots=True)
class ProblemState:
    fe: int = 0
    environment_index: int = 0
    recent_change: bool = False
    terminated: bool = False
    max_evals: int = 0
    current_error: np.ndarray | None = None
    current_performance: np.ndarray | None = None
    e_bbc: np.ndarray | None = None
    p_bbc: np.ndarray | None = None
    best_error: float = np.inf
    best_performance: float = -np.inf
    last_evaluation_x: np.ndarray | None = None
    last_evaluation_value: float = np.nan
    last_evaluation_environment_index: int = 0

    def acknowledge_change(self) -> None:
        self.recent_change = False
        self.best_error = np.inf
        self.best_performance = -np.inf

class BaseProblem(ABC):
    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        ...

    def __init__(self, parameters: dict[str, Any]):
        self.parameters = parameters
        self.state = ProblemState()
        self._metric_manager: Any | None = None
        self._metric_algorithm_context: Any | None = None

    @classmethod
    def create(cls, overrides: dict[str, Any] | None = None):
        return cls(resolve_parameters(cls.parameter_specs(), overrides))

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        ...

    @abstractmethod
    def raw_fitness(self, x: np.ndarray) -> float:
        ...

    def raw_fitness_batch(self, x: np.ndarray) -> np.ndarray:
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        return np.asarray([float(self.raw_fitness(sample)) for sample in samples], dtype=float)

    def current_optimum_value(self) -> float | None:
        return None

    def global_optimum_basin_mask(
        self,
        x: np.ndarray,
        environment_index: int | None = None,
    ) -> np.ndarray | None:
        return None

    @property
    def max_evals(self) -> int:
        return int(self.parameters["ChangeFrequency"] * self.parameters["EnvironmentNumber"])

    @property
    def change_frequency(self) -> int:
        return int(self.parameters["ChangeFrequency"])

    @property
    def environment_number(self) -> int:
        return int(self.parameters["EnvironmentNumber"])

    def resolve_environment_index(self, environment_number: int | None = None) -> int:
        """Resolve public 1-based environment number to internal 0-based index."""
        if environment_number is None:
            return int(self.state.environment_index)
        environment_number = int(environment_number)
        if environment_number < 1 or environment_number > self.environment_number:
            raise ValueError(
                f"environment_number must be in [1, {self.environment_number}], got {environment_number}."
            )
        return environment_number - 1

    def static_fitness(
        self,
        x: np.ndarray,
        environment_number: int | None = None,
    ) -> np.ndarray:
        """Evaluate raw fitness at an environment without changing FE or metrics."""
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        env_index = self.resolve_environment_index(environment_number)
        previous_env = self.state.environment_index
        try:
            self.state.environment_index = env_index
            return np.asarray(self.raw_fitness_batch(samples), dtype=float)
        finally:
            self.state.environment_index = previous_env

    def initialize_runtime_state(self) -> None:
        max_evals = self.max_evals
        env_number = self.environment_number
        self.state = ProblemState(
            max_evals=max_evals,
            current_error=np.full(max_evals, np.nan, dtype=float),
            current_performance=np.full(max_evals, np.nan, dtype=float),
            e_bbc=np.full(env_number, np.nan, dtype=float),
            p_bbc=np.full(env_number, np.nan, dtype=float),
        )

    def attach_metric_observer(self, metric_manager: Any, algorithm_context: Any) -> None:
        self._metric_manager = metric_manager
        self._metric_algorithm_context = algorithm_context

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        values: list[float] = []
        offset = 0
        while offset < samples.shape[0]:
            if self.state.terminated or self.state.recent_change:
                break
            if self.state.fe >= self.state.max_evals:
                self.state.terminated = True
                break
            remaining = samples.shape[0] - offset
            remaining_budget = self.state.max_evals - self.state.fe
            period_offset = self.state.fe % self.change_frequency
            remaining_before_change = self.change_frequency - period_offset
            count = min(remaining, remaining_budget, remaining_before_change)
            batch_values = self.raw_fitness_batch(samples[offset : offset + count])
            for sample, value in zip(samples[offset : offset + count], batch_values):
                scalar = float(value)
                values.append(scalar)
                self._record_evaluation(scalar, sample)
            offset += count
        return np.asarray(values, dtype=float)

    def _record_evaluation(self, value: float, sample: np.ndarray) -> None:
        self.state.last_evaluation_x = np.asarray(sample, dtype=float).copy()
        self.state.last_evaluation_value = float(value)
        self.state.last_evaluation_environment_index = int(self.state.environment_index)
        self.state.fe += 1
        MetricManager.record_default_metrics(self, value)
        self._notify_after_fe()
        environment_finished = self.state.fe % self.change_frequency == 0
        if environment_finished:
            self._notify_after_environment()

        if self.state.fe >= self.state.max_evals:
            self.state.terminated = True
            return

        if environment_finished:
            self.state.environment_index += 1
            self.state.recent_change = True

    def _notify_after_fe(self) -> None:
        if self._metric_manager is None:
            return
        self._metric_manager.sample_fe(self, self._metric_algorithm_context)

    def _notify_after_environment(self) -> None:
        if self._metric_manager is None:
            return
        self._metric_manager.sample_environment(self, self._metric_algorithm_context)

    def metrics(self, metric_names: list[str] | tuple[str, ...] | None = None) -> dict[str, float]:
        names = (
            tuple(metric_names)
            if metric_names is not None
            else MetricManager.default_summary_metric_names(self)
        )
        return {
            name: float(metric_registry.get(name)().compute(self, None))
            for name in names
        }

    def visualization_peaks(self, environment_number: int | None = None) -> dict[str, Any]:
        """Return optional 2D peak annotations for landscape visualization.

        Problem subclasses can override this to expose MATLAB-style
        ``OptimumID`` and visible peak information without making the base
        problem know benchmark-specific fields.
        """
        return {}

    def landscape_visualize(
        self,
        environment_number: int | None = None,
        *,
        resolution: int = 120,
        axes: Any | None = None,
        output_path: str | Path | None = None,
        show: bool = False,
        cmap: str = "cool",
        levels: int = 25,
        font_family: str = VISUALIZATION_FONT_FAMILY,
    ) -> Any:
        """Draw a 2D landscape contour for the selected environment.

        ``environment_number`` is public-facing and 1-based.  If omitted,
        the current problem environment is used.
        """
        if self.dimension != 2:
            raise ValueError("landscape_visualize is only available for 2D problems.")
        import matplotlib.pyplot as plt

        env_index = self.resolve_environment_index(environment_number)
        env_number = env_index + 1
        resolution = max(8, int(resolution))
        x_axis = np.linspace(float(self.bounds.lower[0]), float(self.bounds.upper[0]), resolution)
        y_axis = np.linspace(float(self.bounds.lower[1]), float(self.bounds.upper[1]), resolution)
        xx, yy = np.meshgrid(x_axis, y_axis)
        samples = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        values = self.static_fitness(samples, env_number).reshape(resolution, resolution)

        if axes is None:
            _, axes = plt.subplots()
        contour = axes.contour(xx, yy, values, levels=levels, cmap=cmap, linewidths=0.85)
        self._draw_landscape_peak_annotations(axes, env_number)
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")
        axes.set_title(f"{self.name} Landscape (Env {env_number})")
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
        figure = axes.figure
        if not getattr(axes, "_edolab_colorbar", False):
            figure.colorbar(contour, ax=axes, label="fitness")
            axes._edolab_colorbar = True
        _apply_matplotlib_font(figure, font_family)
        if output_path is not None:
            figure.savefig(output_path, bbox_inches="tight", dpi=160)
        if show:
            plt.show()
        return axes

    def _draw_landscape_peak_annotations(self, axes: Any, environment_number: int) -> None:
        peak_info = self.visualization_peaks(environment_number)
        if not peak_info:
            return

        plotted = False
        optimum_position = peak_info.get("global_optimum_position")
        if optimum_position is not None:
            optimum_position = np.asarray(optimum_position, dtype=float).reshape(-1)
            if optimum_position.size >= 2:
                axes.scatter(
                    [optimum_position[0]],
                    [optimum_position[1]],
                    marker="*",
                    s=220,
                    facecolors="red",
                    edgecolors="red",
                    linewidths=1.0,
                    zorder=7,
                    label="global optimum",
                )
                plotted = True

        visible_ids = np.asarray(peak_info.get("visible_peak_ids", ()), dtype=int).reshape(-1)
        global_optimum_id = peak_info.get("global_optimum_id")
        visible_positions = np.asarray(
            peak_info.get("visible_peak_positions", np.zeros((0, 2))),
            dtype=float,
        )
        if visible_positions.size:
            visible_positions = visible_positions.reshape(-1, visible_positions.shape[-1])
            if visible_positions.shape[1] >= 2:
                circle_positions = visible_positions
                if global_optimum_id is not None and visible_ids.size == visible_positions.shape[0]:
                    circle_positions = visible_positions[visible_ids != int(global_optimum_id)]
                elif optimum_position is not None and optimum_position.size >= 2:
                    is_optimum = np.isclose(
                        visible_positions[:, :2],
                        optimum_position[:2],
                        rtol=1e-7,
                        atol=1e-9,
                    ).all(axis=1)
                    circle_positions = visible_positions[~is_optimum]
                if circle_positions.size:
                    axes.scatter(
                        circle_positions[:, 0],
                        circle_positions[:, 1],
                        marker="o",
                        s=130,
                        facecolors="none",
                        edgecolors="#0057b8",
                        linewidths=2.2,
                        zorder=8,
                        label="visible peaks",
                    )
                    plotted = True

        if plotted:
            handles, labels = axes.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axes.legend(
                unique.values(),
                unique.keys(),
                loc="best",
                fontsize=8,
                frameon=True,
            )

    def build_nbn(
        self,
        environment_number: int | None = None,
        *,
        sample_count: int = 1600,
        seed: int | None = None,
        samples: np.ndarray | None = None,
        build_layout: bool = True,
    ) -> NBNContinuous:
        """Build a nearest-better network from static samples of this problem."""
        env_index = self.resolve_environment_index(environment_number)
        env_number = env_index + 1
        if samples is None:
            rng = np.random.default_rng(seed)
            samples = self.bounds.sample(int(sample_count), rng)
        else:
            samples = np.asarray(samples, dtype=float)
            if samples.ndim != 2 or samples.shape[1] != self.dimension:
                raise ValueError(
                    f"samples must have shape [N, {self.dimension}], got {samples.shape}."
                )
        fitness = self.static_fitness(samples, env_number)
        network = NBNContinuous(optimization_sense="max", build_layout=build_layout)
        network.set_samples(samples, fitness, environment=env_number)
        network.build()
        return network

    def nbn_visualize(
        self,
        environment_number: int | None = None,
        *,
        sample_count: int = 1600,
        seed: int | None = None,
        samples: np.ndarray | None = None,
        axes: Any | None = None,
        output_path: str | Path | None = None,
        show: bool = False,
        font_family: str = VISUALIZATION_FONT_FAMILY,
    ) -> Any:
        """Build and draw the tree-layout NBN Front/Top/Side views."""
        import matplotlib.pyplot as plt

        env_index = self.resolve_environment_index(environment_number)
        env_number = env_index + 1
        network = self.build_nbn(
            env_number,
            sample_count=sample_count,
            seed=seed,
            samples=samples,
            build_layout=True,
        )

        figure = network.plot_three_views(
            env_number,
            axes=axes,
            title_prefix=f"{self.name} NBN",
            font_family=font_family,
        )
        if output_path is not None:
            figure.savefig(output_path, bbox_inches="tight", dpi=160)
        if show:
            plt.show()
        return figure


def _apply_matplotlib_font(figure: Any, font_family: str) -> None:
    figure.patch.set_facecolor("white")
    for axes in figure.axes:
        axes.set_facecolor("white")
        text_items = [axes.title, axes.xaxis.label, axes.yaxis.label]
        if hasattr(axes, "zaxis"):
            text_items.append(axes.zaxis.label)
        for text in text_items:
            text.set_fontname(font_family)
        for label in axes.get_xticklabels() + axes.get_yticklabels():
            label.set_fontname(font_family)
        if hasattr(axes, "get_zticklabels"):
            for label in axes.get_zticklabels():
                label.set_fontname(font_family)
        legend = axes.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontname(font_family)
