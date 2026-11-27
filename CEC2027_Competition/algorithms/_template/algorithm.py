"""Template of a competition algorithm.

Copy this folder to ``CEC2027_Competition/algorithms/<your_algorithm>/`` (a name without a leading underscore), then:

1. Rename the class and give it a unique registry name below (it must not clash with ``python -m cli list``).
2. Declare every parameter in ``parameter_specs()``. All 24 instances are run with these defaults; tuning them per
   instance is not allowed.
3. Fill in ``initialize``, ``iterate`` and ``on_environment_change``.
4. Check it quickly with ``python -m CEC2027_Competition.run --algorithm <Name> --smoke``.

Rules that the platform relies on:

* Evaluate solutions only through ``evaluate_(problem, x, n)`` (or ``problem.evaluate(x)``, which it wraps). The
  platform counts every evaluation, ends each environment after ChangeFrequency evaluations and computes the offline
  error from what you evaluated.
* Fitness is maximized. When a batch is cut short by a change or by the end of the budget, ``evaluate_`` returns NaN
  for the solutions that were not evaluated, whereas ``problem.evaluate`` returns only the values of the evaluated
  ones (a shorter array).
* After every evaluation, return when ``problem.state.recent_change`` or ``problem.state.terminated`` is set. The
  platform then calls ``on_environment_change`` (after a change) and keeps calling ``iterate``.
* Use ``self.rng`` for every random number, so that the fixed competition seeds make runs reproducible.
* Use only ``problem.dimension``, ``problem.bounds`` and the fitness values you evaluate. Do not read the internal
  parameters of GMPB or DynPORTAL (peak positions, heights, optimum values, change parameters, ...).

The two examples next to this folder (example_mqso, example_mde) are complete algorithms written this way.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_


@algorithm_registry.register("MyAlgorithm")
class MyAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "MyAlgorithm"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        # ParameterSpec(name, default, kind, range, description); kind is "integer", "numeric", "boolean",
        # "option" or "string".
        return {
            "PopulationSize": ParameterSpec("PopulationSize", 10, "integer", (1, 1000), "Number of individuals."),
        }

    def initialize(self, problem) -> None:
        self.lower = problem.bounds.lower
        self.upper = problem.bounds.upper
        size = int(self.parameters["PopulationSize"])
        self.positions = problem.bounds.sample(size, self.rng)
        self.fitness = self._evaluate(problem, self.positions)

    def iterate(self, problem) -> None:
        # One iteration: here, a Gaussian step around every individual that keeps improvements.
        step = 0.01 * (self.upper - self.lower)
        trial = np.clip(self.positions + step * self.rng.standard_normal(self.positions.shape), self.lower, self.upper)
        values = self._evaluate(problem, trial)
        if problem.state.recent_change or problem.state.terminated:
            return
        better = values > self.fitness
        self.positions[better] = trial[better]
        self.fitness[better] = values[better]

    def on_environment_change(self, problem) -> None:
        # The landscape has changed: the stored fitness values are outdated. Re-evaluate them.
        self.fitness = self._evaluate(problem, self.positions)

    def _evaluate(self, problem, points: np.ndarray) -> np.ndarray:
        values = evaluate_(problem, points, points.shape[0])
        return np.nan_to_num(values, nan=-np.inf)
