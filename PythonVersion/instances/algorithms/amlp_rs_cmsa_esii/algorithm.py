"""AMLP-RS-CMSA-ESII, translated from the author's distributed main.m.

Minimization comparisons are reversed for EDOLAB fitness maximization.
See README.md for notified changes and partial-evaluation adaptations.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry

from .components import cmsa, niching
from .components.prediction import recommendations
from .metrics import METRICS
from .state import ArchivedSolution, RestartState


@algorithm_registry.register("AMLP_RS_CMSA_ESII")
class AMLPRSCMSAESIIAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "AMLP_RS_CMSA_ESII"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "PopulationSizeCoefficient": ParameterSpec("PopulationSizeCoefficient", 9., "numeric", (2., 100.)),
            "ParentRatio": ParameterSpec("ParentRatio", .5, "numeric", (.05, 1.)),
            "InitialSigmaCoefficient": ParameterSpec("InitialSigmaCoefficient", 1., "numeric", (1e-8, 100.)),
            "InitialSigmaFraction": ParameterSpec("InitialSigmaFraction", .1, "numeric", (1e-8, 1.)),
            "MaxPredictionLevel": ParameterSpec("MaxPredictionLevel", 10, "integer", (1, 100)),
            "PredictionErrorMultiplier": ParameterSpec("PredictionErrorMultiplier", .3, "numeric", (1e-8, 100.)),
            "TolHistFun": ParameterSpec("TolHistFun", 1e-5, "numeric", (0., 1e10)),
            "ArchiveTolerance": ParameterSpec("ArchiveTolerance", .001, "numeric", (1e-12, 1e10)),
        }

    @classmethod
    def custom_metric_classes(cls):
        return METRICS

    def initialize(self, problem) -> None:
        for name, spec in self.parameter_specs().items():
            value = self.parameters[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not spec.bounds[0] <= value <= spec.bounds[1]:
                raise ValueError(f"{name} must be a finite number in {spec.bounds}")
            if spec.kind == "integer" and int(value) != value:
                raise ValueError(f"{name} must be an integer")
        self.dimension = problem.dimension
        self.lower = problem.bounds.lower.copy()
        self.upper = problem.bounds.upper.copy()
        self.widths = self.upper - self.lower
        self.population_size = int(np.floor(self.parameters["PopulationSizeCoefficient"] * np.sqrt(self.dimension)))
        self.cmsa_parameters = cmsa.CMSAParameters.create(self.dimension, self.population_size, self.parameters["ParentRatio"])
        self.max_prediction_level = int(self.parameters["MaxPredictionLevel"])
        self.max_sigma = float(self.parameters["InitialSigmaFraction"])
        self.tolerance = float(self.parameters["TolHistFun"])
        self.archive_tolerance = float(self.parameters["ArchiveTolerance"])
        self.history_size = int(np.floor(10 + 30 * self.dimension / self.population_size))
        self.reduction = .99**(1 / self.dimension)
        self.archive_history: list[np.ndarray] = []
        self.recommended_centres = np.empty((0, self.dimension))
        self.recommended_sigmas = np.empty(0)
        self.recommended_levels = np.empty(0, dtype=int)
        self.prediction_level_counts = np.zeros(self.max_prediction_level, dtype=int)
        self.stop_counts: dict[str, int] = {}
        self.completed_restarts = 0
        self.started_restarts = 0
        self.archive_sizes: list[int] = []
        self.paid_evaluations = dict(search=0, merge=0, archive=0)
        self.boundary_events = dict(search=0, merge=0, archive=0)
        self._reset_environment()
        self._start_restart()

    def _reset_environment(self) -> None:
        self.archive: list[ArchivedSolution] = []
        self.recommendation_index = 0
        self.environment_completed_restarts = 0
        self.default_taboo = 1.0
        self.initial_radius = np.sqrt(self.dimension) / 2
        self.best_environment_fitness = -np.inf  # completed restart bests only
        self.state: RestartState | None = None

    def _start_restart(self) -> None:
        # driver.m samples the random initialization even when AMLP replaces it.
        mean, radius = niching.initial_centre(
            self.archive, self.lower, self.widths, self.initial_radius, self.reduction, self.rng,
        )
        sigma = min(self.max_sigma, radius * self.parameters["InitialSigmaCoefficient"])
        if self.recommendation_index < len(self.recommended_centres):
            index = self.recommendation_index
            mean = np.clip(self.recommended_centres[index], self.lower, self.upper)
            sigma = float(self.recommended_sigmas[index])
            self.prediction_level_counts[self.recommended_levels[index] - 1] += 1
            self.recommendation_index += 1
        self.state = cmsa.initialize(mean, sigma, self.widths, radius)
        self.started_restarts += 1

    def _sample(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.state
        points, radii, criticality = niching.critical_regions(self.archive, state)
        # Subpopulation.m retains criticality when the better-archive set is
        # empty. It clears criticInd, so this only affects convergence history.
        if any(item.fitness > state.best_fitness for item in self.archive):
            state.last_criticality = criticality
        state.criticality_history.append(state.last_criticality)
        candidate, count = niching.merge_candidate(self.archive, state)
        state.merge_candidate_history.append(candidate)
        state.merge_count_history.append(count)
        samples = np.empty((self.population_size, self.dimension))
        sigmas = np.empty(self.population_size)
        temporary_reduction = 1.0
        for index in range(self.population_size):
            while True:
                sigma = state.sigma * np.exp(self.rng.normal() * self.cmsa_parameters.tau_sigma)
                step = state.basis @ (state.scales * self.rng.normal(size=self.dimension))
                sample = niching.repair_bounds(state.mean + sigma * step, state.mean, self.lower, self.upper, self.rng)
                if np.all(niching.normalized_distances(points, sample, state) >= radii * temporary_reduction):
                    samples[index], sigmas[index] = sample, sigma
                    break
                temporary_reduction *= self.reduction
        return samples, sigmas

    def _evaluate(self, problem, candidates, kind):
        values = problem.evaluate(candidates)
        self.paid_evaluations[kind] += values.size
        if problem.state.recent_change or problem.state.terminated:
            self.boundary_events[kind] += 1
        return values

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        if self.state is None:
            self._start_restart()
        candidates, sigmas = self._sample()
        values = self._evaluate(problem, candidates, "search")
        # Never rank unobserved individuals or mix environments. The interrupted
        # restart is discarded by driver.m before the prediction archive update.
        if problem.state.recent_change or problem.state.terminated:
            return
        cmsa.update(self.state, candidates[:values.size], values, sigmas[:values.size],
                    self.cmsa_parameters, self.lower, self.upper)
        state = self.state
        state.best_history.append(float(np.max(values)))
        state.median_history.append(float(np.median(values)))
        limit = max(self._stagnation_size(), self.history_size)
        state.best_history = state.best_history[-limit:]
        state.median_history = state.median_history[-limit:]
        reason = self._stop_reason()
        if not reason:
            reason = self._check_merge(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        if not reason:
            reason = self._check_local_convergence()
        if reason:
            self.stop_counts[reason] = self.stop_counts.get(reason, 0) + 1
            self.completed_restarts += 1
            self.environment_completed_restarts += 1
            self._update_archive(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
            self.initial_radius = min(state.initial_radius * 1.04, np.sqrt(self.dimension) / 2)
            self.default_taboo = float(np.percentile([item.taboo_distance for item in self.archive], 10))
            self.best_environment_fitness = max(self.best_environment_fitness, state.best_fitness)
            self.state = None

    def _stagnation_size(self):
        return int(np.floor(120 + .2 * self.state.generation + 30 * self.dimension / self.population_size))

    def _stop_reason(self) -> str:
        state = self.state
        if (np.max(state.scales) / np.min(state.scales))**2 > 1e14:
            return "condition"
        window = self._stagnation_size()
        if state.generation >= window:
            best, median = state.best_history[-window:], state.median_history[-window:]
            if max(np.median(best[-20:]) - np.median(best[:20]), np.median(median[-20:]) - np.median(median[:20])) < 0:
                return "stagnation"
        if state.generation >= self.history_size and np.ptp(state.best_history[-self.history_size:]) < self.tolerance:
            return "function_history"
        if np.max(state.scales) * state.sigma < 1e-12:
            return "step_size"
        # maxIterPar is declared but never used in the distributed source.
        return ""

    def _hill_valley(self, problem, first, first_fitness, second, second_fitness, *, random):
        """True: distinct; False: same basin; None: evaluation boundary."""
        kind = "archive" if random else "merge"
        for probe in range(10):
            ratio = .1 + .8 * self.rng.random() if random else (probe + .5) / 10
            values = self._evaluate(problem, first + ratio * (second - first), kind)
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
        # Source tests from archive endpoint toward current best (r=.05,.15,...).
        distinct = self._hill_valley(problem, archived.position, archived.fitness,
                                    state.best_position, state.best_fitness, random=False)
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
        gap = min(item.fitness for item in self.archive) - state.best_fitness - self.archive_tolerance
        return "local_convergence" if fluctuation < .04 * gap else ""

    def _update_archive(self, problem) -> None:
        """The ONE filtered archive used for both repulsion and prediction."""
        state = self.state
        best = max(state.best_fitness, self.best_environment_fitness)
        self.archive = [item for item in self.archive if item.fitness + self.archive_tolerance > best]
        desirable = state.generation > 1 and state.best_fitness + self.archive_tolerance >= best
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
                distinct = self._hill_valley(problem, item.position, item.fitness,
                                            state.best_position, state.best_fitness, random=True)
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
        # driver.m calls DynamicManager.update BEFORE archiving an interrupted
        # restart. Even a better paid prefix therefore does not enter history.
        ranked = sorted(self.archive, key=lambda item: -item.fitness)
        positions = np.asarray([item.position for item in ranked]).reshape(-1, self.dimension)
        self.archive_sizes.append(len(ranked))
        self.archive_history.append((positions - self.lower) / self.widths)
        self.archive_history = self.archive_history[-(self.max_prediction_level + 1):]
        centres, sigmas, levels = recommendations(self.archive_history, self.max_prediction_level,
                                                self.max_sigma, self.parameters["PredictionErrorMultiplier"])
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
        result.raw.setdefault("event_streams", {})["amlp_cmsa.audit"] = {
            "schema_version": 1,
            "events": [{
                "core": "source_cmsa_nonelitist", "completed_restarts": self.completed_restarts,
                "started_restarts": self.started_restarts, "stop_reasons": self.stop_counts,
                "prediction_level_counts": self.prediction_level_counts.tolist(),
                "archive_sizes_at_change": self.archive_sizes,
                "paid_evaluations": self.paid_evaluations, "boundary_events": self.boundary_events,
                "final_archive_size": len(self.archive), "known_changes": True,
            }],
        }
        return result
