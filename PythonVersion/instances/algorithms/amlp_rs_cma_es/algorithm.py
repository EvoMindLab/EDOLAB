"""AMLP-RS-CMA-ES: public AMLP/RS logic plus the 2022 GMPB report settings.

See README.md for the source-to-code map and reconstruction decisions. The
author's released 2022003 package uses CMSA, whereas 2022002 specifies CMA.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.algorithm.operators import CMAESOperator
from core.parameters import ParameterSpec
from core.registry import algorithm_registry

from .components import cma, niching
from .components.prediction import recommendations
from .metrics import METRICS
from .state import ArchivedSolution, RestartState


@algorithm_registry.register("AMLP_RS_CMA_ES")
class AMLPRSCMAESAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "AMLP_RS_CMA_ES"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSizeCoefficient": ParameterSpec("PopulationSizeCoefficient", 6., "numeric", (2., 100.)),
            "ParentRatio": ParameterSpec("ParentRatio", .5, "numeric", (.05, 1.)),
            "MaxPredictionLevel": ParameterSpec("MaxPredictionLevel", 10, "integer", (1, 100)),
            "InitialSigmaFraction": ParameterSpec("InitialSigmaFraction", .15, "numeric", (1e-8, 1.)),
            "PredictionErrorMultiplier": ParameterSpec("PredictionErrorMultiplier", .5, "numeric", (1e-8, 100.)),
            "TolHistFun": ParameterSpec("TolHistFun", 100., "numeric", (0., 1e10)),
        }

    @classmethod
    def custom_metric_classes(cls):
        return METRICS

    def initialize(self, problem) -> None:
        # Parameters are an external configuration boundary; internal state is
        # maintained by the restart lifecycle without repeated validation.
        for name, spec in self.parameter_specs().items():
            value = self.parameters[name]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not spec.bounds[0] <= value <= spec.bounds[1]:
                raise ValueError(f"{name} must be a finite number in {spec.bounds}")
        level = self.parameters["MaxPredictionLevel"]
        if int(level) != level:
            raise ValueError("MaxPredictionLevel must be an integer")
        self.dimension = problem.dimension
        self.lower = problem.bounds.lower.copy()
        self.upper = problem.bounds.upper.copy()
        self.widths = self.upper - self.lower
        self.population_size = int(np.floor(self.parameters["PopulationSizeCoefficient"] * np.sqrt(self.dimension)))
        self.cma_parameters = cma.CMAParameters.create(self.dimension, self.population_size, self.parameters["ParentRatio"])
        self.operator = CMAESOperator()
        self.max_prediction_level = int(level)
        self.max_sigma = float(self.parameters["InitialSigmaFraction"])
        self.tolerance = float(self.parameters["TolHistFun"])
        self.history_size = int(np.floor(10 + 30 * self.dimension / self.population_size))
        self.max_generations = int(np.ceil((70 + 35 * (self.dimension + 3)**2) / np.sqrt(self.population_size)))
        self.reduction = .99**(1 / self.dimension)
        self.archive_history: list[np.ndarray] = []
        self.recommended_centres = np.empty((0, self.dimension))
        self.recommended_sigmas = np.empty(0)
        self.recommended_levels = np.empty(0, dtype=int)
        self.prediction_level_counts = np.zeros(self.max_prediction_level, dtype=int)
        self.stop_counts: dict[str, int] = {}
        self.completed_restarts = 0
        self._reset_environment()
        self._start_restart()

    def _reset_environment(self) -> None:
        self.archive: list[ArchivedSolution] = []
        # The GMPB report retains every restart for prediction. The static
        # niching archive below preserves the released RS deduplication rule.
        self.prediction_archive: list[tuple[np.ndarray, float]] = []
        self.recommendation_index = 0
        self.environment_completed_restarts = 0
        self.default_taboo = 1.0
        self.initial_radius = np.sqrt(self.dimension) / 2
        self.best_environment_fitness = -np.inf
        self.state: RestartState | None = None

    def _start_restart(self) -> None:
        mean, radius = niching.initial_centre(
            self.archive, self.lower, self.widths, self.initial_radius, self.reduction, self.rng,
        )
        sigma = min(self.max_sigma, radius)
        if self.recommendation_index < len(self.recommended_centres):
            index = self.recommendation_index
            mean = np.clip(self.recommended_centres[index], self.lower, self.upper)
            sigma = float(self.recommended_sigmas[index])
            self.prediction_level_counts[self.recommended_levels[index] - 1] += 1
            self.recommendation_index += 1
        self.state = RestartState(
            mean=mean, sigma=sigma, covariance=np.diag(self.widths**2),
            basis=np.eye(self.dimension), scales=self.widths.copy(),
            invsqrt=np.diag(1 / self.widths), pc=np.zeros(self.dimension),
            ps=np.zeros(self.dimension), initial_radius=radius,
        )

    def _sample(self) -> np.ndarray:
        state = self.state
        taboo_points, taboo_radii, criticality = niching.critical_regions(self.archive, state)
        state.criticality_history.append(criticality)
        candidate, count = niching.merge_candidate(self.archive, state)
        state.merge_candidate_history.append(candidate)
        state.merge_count_history.append(count)
        samples = np.empty((self.population_size, self.dimension))
        temporary_reduction = 1.0
        for index in range(self.population_size):
            while True:
                sample = self.operator.sample_columns(
                    state.mean, state.sigma, state.basis, state.scales, 1, self.rng,
                )[:, 0]
                sample = niching.repair_bounds(sample, state.mean, self.lower, self.upper, self.rng)
                distances = niching.normalized_distances(taboo_points, sample, state)
                if np.all(distances >= taboo_radii * temporary_reduction):
                    samples[index] = sample
                    break
                temporary_reduction *= self.reduction
        return samples

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        if self.state is None:
            self._start_restart()
        state = self.state
        candidates = self._sample()
        values = problem.evaluate(candidates)
        evaluated = values.size
        if evaluated:
            best = int(np.argmax(values))
            if values[best] > state.best_fitness:
                state.best_fitness = float(values[best])
                state.best_position = candidates[best].copy()
        if problem.state.recent_change or problem.state.terminated:
            return
        cma.update(state, candidates[:evaluated], values, self.cma_parameters, self.operator)
        state.best_history.append(float(np.max(values)))
        state.median_history.append(float(np.median(values)))
        reason = self._stop_reason()
        if not reason:
            reason = self._check_merge(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        if not reason:
            reason = self._check_local_convergence()
        if reason:
            state.stop_reason = reason
            self.stop_counts[reason] = self.stop_counts.get(reason, 0) + 1
            self.prediction_archive.append((state.best_position.copy(), state.best_fitness))
            self.completed_restarts += 1
            self.environment_completed_restarts += 1
            self._update_archive(problem)
            # A hill-valley probe can finish an environment. Its old search
            # state must not initiate or evaluate another restart afterward.
            self.initial_radius = min(state.initial_radius * 1.04, np.sqrt(self.dimension) / 2)
            if self.archive:
                self.default_taboo = float(np.percentile([item.taboo_distance for item in self.archive], 10))
            self.best_environment_fitness = max(self.best_environment_fitness, state.best_fitness)
            self.state = None

    def _stop_reason(self) -> str:
        state = self.state
        if (np.max(state.scales) / np.min(state.scales))**2 > 1e14:
            return "condition"
        stagnation = int(np.floor(120 + .2 * state.generation + 30 * self.dimension / self.population_size))
        if state.generation >= stagnation:
            best = state.best_history[-stagnation:]
            median = state.median_history[-stagnation:]
            if max(np.median(best[-20:]) - np.median(best[:20]), np.median(median[-20:]) - np.median(median[:20])) < 0:
                return "stagnation"
        if state.generation >= self.history_size and np.ptp(state.best_history[-self.history_size:]) < self.tolerance:
            return "function_history"
        if np.max(state.scales) * state.sigma < 1e-12:
            return "step_size"
        if state.generation >= self.max_generations:
            return "generation_limit"
        return ""

    def _hill_valley(self, problem, first, first_fitness, second, second_fitness, *, random: bool) -> bool | None:
        """True means distinct basins; None means a paid probe hit a boundary."""
        for probe in range(10):
            ratio = .1 + .8 * self.rng.random() if random else (probe + .5) / 10
            position = first + ratio * (second - first)
            values = problem.evaluate(position)
            if problem.state.recent_change or problem.state.terminated:
                return None
            if values[0] < min(first_fitness, second_fitness) - self.tolerance:
                return True
        return False

    def _check_merge(self, problem) -> str:
        state = self.state
        if state.generation < self.history_size or state.generation <= state.check_merge_after:
            return ""
        if not all(count == 1 for count in state.merge_count_history[-self.history_size:]):
            return ""
        archived = self.archive[state.merge_candidate_history[-1]]
        distinct = self._hill_valley(problem, archived.position, archived.fitness, state.best_position, state.best_fitness, random=False)
        if distinct is None:
            return ""
        if distinct:
            state.check_merge_after = state.generation + .1 * self.history_size
            return ""
        state.best_position = archived.position.copy()
        state.best_fitness = archived.fitness
        return "merge"

    def _check_local_convergence(self) -> str:
        state = self.state
        window = int(np.floor(2 + .5 * self.history_size))
        if state.generation <= 2 + max(.5 * self.history_size, window) or not self.archive:
            return ""
        if max(state.criticality_history[-window:]) >= .01:
            return ""
        fluctuation = float(np.mean(np.abs(np.diff(state.best_history[-window:]))))
        gap = min(item.fitness for item in self.archive) - state.best_fitness - 1e-5
        return "local_convergence" if fluctuation < .04 * gap else ""

    def _update_archive(self, problem) -> None:
        """Released static RS archive; every restart is also retained separately."""
        state = self.state
        best = max(state.best_fitness, self.best_environment_fitness)
        self.archive = [item for item in self.archive if item.fitness + 1e-5 > best]
        desirable = state.generation > 1 and state.best_fitness + 1e-5 >= best
        if not desirable:
            if self.archive:
                factor = np.exp(-.5 / (np.sqrt(self.dimension) * len(self.archive)))
                for item in self.archive:
                    item.taboo_distance *= factor
            return
        if self.archive:
            distances = [np.linalg.norm(item.position - state.best_position) for item in self.archive]
            for index in np.argsort(distances, kind="stable")[:5]:
                item = self.archive[index]
                distinct = self._hill_valley(problem, item.position, item.fitness, state.best_position, state.best_fitness, random=True)
                if distinct is None:
                    return
                if not distinct:
                    if state.best_fitness > item.fitness + self.tolerance:
                        item.position = state.best_position.copy()
                        item.fitness = state.best_fitness
                    for other_index, other in enumerate(self.archive):
                        difference = 1.0 if other_index == index else -.5 / (len(self.archive) - 1)
                        other.taboo_distance *= np.exp(difference / np.sqrt(self.dimension))
                    return
        self.archive.append(ArchivedSolution(state.best_position.copy(), state.best_fitness, self.default_taboo))

    def on_environment_change(self, problem) -> None:
        # Include the evaluated prefix of a restart interrupted by the change.
        if self.state is not None and self.state.best_position is not None:
            self.prediction_archive.append((self.state.best_position.copy(), self.state.best_fitness))
        ranked = sorted(self.prediction_archive, key=lambda item: -item[1])
        positions = np.asarray([item[0] for item in ranked]).reshape(-1, self.dimension)
        self.archive_history.append((positions - self.lower) / self.widths)
        self.archive_history = self.archive_history[-(self.max_prediction_level + 1):]
        centres, sigmas, levels = recommendations(
            self.archive_history, self.max_prediction_level, self.max_sigma,
            float(self.parameters["PredictionErrorMultiplier"]),
        )
        self.recommended_centres = self.lower + centres * self.widths
        self.recommended_sigmas = sigmas
        self.recommended_levels = levels
        self._reset_environment()
        self._start_restart()

    def prepare_environment_metrics(self, problem) -> None:
        # Called at the environment's last paid FE, before on_environment_change.
        levels = self.recommended_levels[:self.recommendation_index]
        self.metric_state.update(
            amlp_archive_size=len(self.archive),
            amlp_completed_restarts=self.environment_completed_restarts,
            amlp_predicted_restarts=self.recommendation_index,
            amlp_mean_prediction_level=float(np.mean(levels)) if levels.size else float("nan"),
        )

    def run_once(self, *args, **kwargs):
        result = super().run_once(*args, **kwargs)
        result.raw.setdefault("event_streams", {})["amlp.audit"] = {
            "schema_version": 1,
            "events": [{
                "completed_restarts": self.completed_restarts,
                "stop_reasons": dict(self.stop_counts),
                "prediction_level_counts": self.prediction_level_counts.tolist(),
                "history_archive_sizes": [len(archive) for archive in self.archive_history],
                "core": "positive_weight_cma",
                "reference": "EvOpt 2022002; shared code EvOpt 2022003",
            }],
        }
        return result
