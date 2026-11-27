"""SPMP: solution space partitioning-based multipopulation method.

M. Peng, C. Li, J. Wang, X. Cai, S. Zeng and S. Yang, "A Solution Space Partitioning-Based Multipopulation Method for
Dynamic Optimization," IEEE Transactions on Evolutionary Computation, vol. 30, no. 3, pp. 1299-1313, 2026,
doi: 10.1109/TEVC.2025.3597453.

Monitor ("Monitor", default true): components/monitor.py records, per environment, the evaluations by category and
their allocation among populations, the numbers of subspaces, basins and populations (active and hibernating), and
creations, removals and tracker events, as the event stream spmp.monitor and the trends in metrics.py. It only
reads state.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_, stack_rows
from utility.clustering import SingleLinkageClusterer
from instances.algorithms.spmp.state import (
    SPMPMapRecord,
    SPMPHiberArea,
    SPMPBoA,
    SPMPSpatialModelState,
    SPMPSwarm,
)
from instances.algorithms.spmp.components.partition import SPMPPartitioner
from instances.algorithms.spmp.components.monitor import MONITOR_STREAM, SPMPMonitor
from instances.algorithms.spmp.metrics import METRICS

SHIFT_WINDOW = 10


def _fitness_values(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float).reshape(-1)
    return np.where(np.isnan(raw), -np.inf, raw)


def _has_complete_values(values: np.ndarray, expected: int) -> bool:
    raw = np.asarray(values, dtype=float).reshape(-1)
    return raw.size == int(expected) and not bool(np.any(np.isnan(raw)))


def _best_index(values: np.ndarray) -> int:
    finite = _fitness_values(values)
    if finite.size == 0:
        return 0
    return int(np.argmax(finite))


def _make_spmp_swarm(
    positions: np.ndarray,
    fitness: np.ndarray,
    dimension: int,
    swarm_type: str = "Exploratory",
) -> SPMPSwarm:
    x = np.asarray(positions, dtype=float).reshape(-1, int(dimension))
    values = _fitness_values(fitness)
    if values.size < x.shape[0]:
        values = np.pad(values, (0, x.shape[0] - values.size), constant_values=-np.inf)
    values = values[: x.shape[0]]
    best_id = _best_index(values)
    center = np.mean(x, axis=0) if x.size else np.zeros(int(dimension), dtype=float)
    radius = float(np.mean(np.linalg.norm(x - center, axis=1))) if x.size else 0.0
    return SPMPSwarm(
        positions=x.copy(),
        fitness=values.copy(),
        metadata={},
        velocities=np.zeros_like(x),
        pbest_positions=x.copy(),
        pbest_fitness=values.copy(),
        best_position=x[best_id, :].copy(),
        best_fitness=float(values[best_id]),
        gbest_past_environment=np.full(int(dimension), np.nan),
        shifts=[],
        is_hibernated=0,
        stagnating_count=0,
        is_stagnated=0,
        best_id=best_id,
        center=center,
        init_radius=radius,
        current_radius=radius,
        swarm_type=swarm_type,
        stagnation_count=0,
        last_radius=radius,
    )


def spmp_record_to_history(
    history: np.ndarray,
    positions: np.ndarray,
    fitness: np.ndarray,
    fe: int,
    dimension: int,
    history_window_fe: int | float = np.inf,
    history_grid_coords: np.ndarray | None = None,
    distance_threshold: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Append non-duplicate samples and keep the short FE window."""

    dim = int(dimension)
    existing = np.asarray(history, dtype=float)
    if existing.size == 0:
        existing = np.empty((0, dim + 2), dtype=float)
    if (
        history_grid_coords is None
        or np.asarray(history_grid_coords).shape != (existing.shape[0], dim)
    ):
        grid = _history_grid_coords(existing[:, :dim], distance_threshold, dim)
    else:
        grid = np.asarray(history_grid_coords, dtype=np.int32).reshape(-1, dim)
    if np.isfinite(history_window_fe) and history_window_fe > 0 and existing.size:
        min_fe = int(fe) - int(history_window_fe) + 1
        keep = existing[:, dim + 1] >= min_fe
        existing = existing[keep, :]
        grid = grid[keep, :]

    samples = np.asarray(positions, dtype=float)
    if samples.size == 0:
        return existing, grid
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    values = np.asarray(fitness, dtype=float).reshape(-1)
    nx = int(samples.shape[0])
    if nx <= 0:
        return existing, grid

    rows: list[np.ndarray] = []
    row_grid: list[np.ndarray] = []
    batch_grid = _history_grid_coords(samples[:, :dim], distance_threshold, dim)
    threshold_sq = float(distance_threshold) ** 2
    for idx in range(nx):
        value = float(values[idx]) if idx < values.size else np.nan
        if np.isnan(value):
            continue
        point = samples[idx, :dim].copy()
        fe_i = int(fe) - nx + idx + 1
        duplicate = False
        if existing.shape[0] > 0:
            duplicate = _has_near_history_duplicate(
                point,
                batch_grid[idx],
                existing[:, :dim],
                grid,
                threshold_sq,
            )
        if not duplicate and row_grid:
            duplicate = _has_near_history_duplicate(
                point,
                batch_grid[idx],
                np.vstack(rows)[:, :dim],
                np.vstack(row_grid),
                threshold_sq,
            )
        if not duplicate:
            rows.append(np.concatenate([point, [value, float(fe_i)]]))
            row_grid.append(batch_grid[idx].copy())
    if not rows:
        return existing, grid
    return np.vstack([existing, np.vstack(rows)]), np.vstack([grid, np.vstack(row_grid)])


def _history_grid_coords(
    points: np.ndarray,
    cell_size: float,
    dimension: int,
) -> np.ndarray:
    samples = np.asarray(points, dtype=float)
    if samples.size == 0:
        return np.zeros((0, int(dimension)), dtype=np.int32)
    return np.floor(samples.reshape(-1, int(dimension)) / float(cell_size)).astype(np.int32)


def _has_near_history_duplicate(
    point: np.ndarray,
    grid_coord: np.ndarray,
    ref_points: np.ndarray,
    ref_grid: np.ndarray,
    threshold_sq: float,
) -> bool:
    if ref_points.size == 0:
        return False
    neighbor_mask = np.all(np.abs(ref_grid - grid_coord) <= 1, axis=1)
    if not np.any(neighbor_mask):
        return False
    diffs = ref_points[neighbor_mask, :] - point
    return bool(np.any(np.sum(diffs * diffs, axis=1) < threshold_sq))


def sub_population_generator_spmp(
    dimension: int,
    positions: np.ndarray,
    problem,
    evaluate=None,
    *,
    defer_on_change: bool = False,
) -> tuple[list[SPMPSwarm], np.ndarray, list[int]]:
    """Generate swarms from samples by single-linkage clustering; singletons form the unevaluated remainder.

    Returns the swarms, the remainder and, per swarm, the FE count at which its group was evaluated. With
    ``defer_on_change`` a group cut by an environment change and the groups after it return to
    the remainder unevaluated; otherwise the cut group becomes a swarm of -inf values and the rest is dropped.
    """

    evaluate = evaluate or (lambda points, count: evaluate_(problem, points, count))
    samples = np.asarray(positions, dtype=float).reshape(-1, int(dimension))
    groups, remain = SingleLinkageClusterer().groups_and_remain(samples)
    swarms: list[SPMPSwarm] = []
    stamps: list[int] = []
    remain_rows = [row for row in remain]

    for index, group in enumerate(groups):
        if group.shape[0] == 0:
            continue
        values = evaluate(group, group.shape[0])
        if problem.state.recent_change:
            if defer_on_change:
                for rest in groups[index:]:
                    remain_rows.extend(rest)
                break
            values = np.full(group.shape[0], -np.inf, dtype=float)
        elif problem.state.terminated and not _has_complete_values(values, group.shape[0]):
            return swarms, stack_rows(remain_rows, int(dimension)), stamps
        elif not _has_complete_values(values, group.shape[0]):
            remain_rows.extend(group)
            continue
        swarms.append(_make_spmp_swarm(group, values, int(dimension)))
        stamps.append(int(problem.state.fe))
        if problem.state.recent_change or problem.state.terminated:
            break
    return swarms, stack_rows(remain_rows, int(dimension)), stamps


@algorithm_registry.register("SPMP")
class SPMPAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "SPMP"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "initPopulationSize": ParameterSpec("initPopulationSize", 100, "integer", (0, 1000)),
            "w": ParameterSpec("w", 0.7298, "numeric", (0, 2)),
            "c1": ParameterSpec("c1", 1.496, "numeric", (0, 5)),
            "c2": ParameterSpec("c2", 1.496, "numeric", (0, 5)),
            "ConvergenceLimit": ParameterSpec("ConvergenceLimit", 0.00001, "numeric", (0, 0.1)),
            "diversityFactor": ParameterSpec("diversityFactor", 0.005, "numeric", (0, 1)),
            "StepIndivs": ParameterSpec("StepIndivs", 10, "integer", (0, 100)),
            "N_s_start": ParameterSpec("N_s_start", 100, "integer", (20, 2000)),
            "alpha": ParameterSpec("alpha", 0.4, "numeric", (0, 1)),
            "beta": ParameterSpec("beta", 0.4, "numeric", (0, 1)),
            "gamma": ParameterSpec("gamma", 0.2, "numeric", (0, 1)),
            "M_factor": ParameterSpec("M_factor", 8, "integer", (1, 100)),
            "BoATopK": ParameterSpec("BoATopK", 3, "integer", (1, 10)),
            "MAX_SWARMS": ParameterSpec("MAX_SWARMS", 10, "integer", (3, 30)),
            "DebugResourceAllocation": ParameterSpec("DebugResourceAllocation", 0, "integer", (0, 1)),
            "Monitor": ParameterSpec("Monitor", True, "boolean", None,
                "Passive per-environment monitoring; reads state only and never changes the search."),
        }

    @classmethod
    def custom_metric_classes(cls):
        return METRICS

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.Dimension = self.dimension
        self.bounds = problem.bounds
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float).reshape(self.dimension)
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float).reshape(self.dimension)
        self.MaxCoordinate = self.max_coordinate
        self.MinCoordinate = self.min_coordinate
        self.max_indivs = 1000
        self.min_indivs = 10
        self.shift_severity = 1.0
        self.hiber_step = 0.5
        self.first_update = 1
        self.off_peak = 3
        self.current_pop_size = 0
        self.previous_pop_size = 0
        self.iter_individuals_begin: list[float] = []
        self.iter_pop_size_end: list[int] = []
        self.adjust_iter = 0
        self.distance_accuracy = 0.2
        self.record_pop_info: list[int] = []
        self.avoidance_flag = 0
        self.domain_size = float(np.linalg.norm(self.max_coordinate - self.min_coordinate))
        self.domainS = self.domain_size

        if not isinstance(self.parameters["Monitor"], (bool, np.bool_)):
            raise ValueError("Monitor must be true or false")
        self._configure_tracker()
        # Displacement of the peaks per change (Euclidean); shift_severity is the prior.
        self.shift_estimate = self.shift_severity
        self.shift_observations: list[float] = []
        self.next_uid = 0
        self.monitor = SPMPMonitor(self, problem) if self.parameters["Monitor"] else None

        self.History = np.empty((0, self.dimension + 2), dtype=float)
        self.HistoryGridCoords = np.zeros((0, self.dimension), dtype=np.int32)
        self.BoAs: list[SPMPBoA] = []
        dim_ratio = 1.0 / (1.0 + np.log2(max(self.dimension, 1) + 1.0))
        self.HistoryWindowFE = max(1, int(np.ceil(problem.change_frequency * dim_ratio)))
        self.HistoryWindowFE = max(self.HistoryWindowFE, int(self.parameters["initPopulationSize"]))
        self.HistoryWindowFE = min(self.HistoryWindowFE, int(problem.change_frequency))
        self.SpatialModelState = self._default_spatial_model_state()
        self.SpatialModelState.HistoryWindowFE = self.HistoryWindowFE
        self.spatial_model_state = self.SpatialModelState

        self.BaseNomega = int(self.parameters["N_s_start"]) * self.dimension
        self.ConservativeNomega = max(4 * self.dimension, int(np.ceil(self.BaseNomega * dim_ratio)))
        self.N_omega = self.ConservativeNomega
        self.partitioner_cache: dict[int, SPMPPartitioner] = {}
        self.PartitionerCache = self.partitioner_cache
        self.Subspaces = []
        self.SubspaceLabels = np.zeros(0, dtype=int)
        self.SubspaceAnyBoA = np.zeros(0, dtype=bool)
        self._apply_partition_budget(self.N_omega)

        init_size = max(0, int(self.parameters["initPopulationSize"]))
        init_x = self.bounds.sample(init_size, self.rng)
        self.pop, self.remain_indivs, stamps = self._generate_swarms(init_x, problem, "init")
        self.remainIndivs = self.remain_indivs
        self._sync_swarm_number("init")
        if not problem.state.recent_change:
            self._record_new_swarms(self.pop, stamps, problem)

        self.map_pop_inds = [SPMPMapRecord() for _ in range(max(self.SwarmNumber, 1))]
        self.MapPopInds = self.map_pop_inds
        self.hiber_area: list[SPMPHiberArea] = []
        self.HiberArea = self.hiber_area
        self.LastGhostDiagnostics = self._default_ghost_diagnostics()
        self.LastBoADiagnostics = self._default_boa_diagnostics()
        self.LastSwarmTypeDiagnostics = self._default_swarm_type_diagnostics()

        self.estimate_boas()
        self.update_swarm_types()
        self.update_spatial_model_state()

    def iterate(self, problem) -> None:
        self._iterate(problem)
        if self.monitor is not None and not (problem.state.recent_change or problem.state.terminated):
            self.monitor.snapshot(problem)

    def _configure_tracker(self) -> None:
        """Cloud of the tracker: lambda samples per turn, step size adapted by the (1+lambda) success rule."""
        self.cloud_size = 4 + int(np.floor(3.0 * np.log(self.dimension)))
        self.success_target = 1.0 / (5.0 + np.sqrt(self.cloud_size) / 2.0)
        self.success_smoothing = self.success_target * self.cloud_size / (2.0 + self.success_target * self.cloud_size)
        self.step_damping = 1.0 + self.dimension / (2.0 * self.cloud_size)
        width = float(np.mean(self.max_coordinate - self.min_coordinate))
        self.cloud_radius_max = 0.25 * width
        self.cloud_radius_min = 1e-8 * width

    def _iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        self.domain_size = float(np.linalg.norm(self.max_coordinate - self.min_coordinate))
        self.domainS = self.domain_size
        self._iterative_components(problem)
        if problem.state.recent_change or problem.state.terminated:
            return
        self.update_swarm_types()
        self.update_spatial_model_state()
        self._enforce_swarm_limit()
        if problem.state.recent_change or problem.state.terminated:
            return
        self._diversity_monitor_and_adjust(problem)

    def on_environment_change(self, problem) -> None:
        if self.monitor is not None:
            self.monitor.close_environment(problem)
        if not self._restart_swarms(problem):
            return

        self.History = np.empty((0, self.dimension + 2), dtype=float)
        self.HistoryGridCoords = np.zeros((0, self.dimension), dtype=np.int32)
        for swarm in self.pop:
            kept = np.isfinite(swarm.pbest_fitness)
            self._record(swarm.pbest_positions[kept], swarm.pbest_fitness[kept], problem)
        self.hiber_area = []
        self.HiberArea = self.hiber_area

        self.estimate_boas()
        self.update_swarm_types()
        self.update_spatial_model_state()
        if not self.can_use_spatial_guidance() and self.N_omega != self.ConservativeNomega:
            self._apply_partition_budget(self.ConservativeNomega)
            self.estimate_boas()
            self.update_swarm_types()
            self.update_spatial_model_state()

    def _restart_swarms(self, problem) -> bool:
        """Change response: estimate the displacement, revalidate each swarm's best, restart around it."""
        observations = self._observe_displacement()
        radii: list[float] = []
        for swarm in self.pop:
            value = self._evaluate(problem, swarm.best_position, 1, "change", swarm)
            if problem.state.recent_change or np.isnan(value[0]):
                return False
            radii.append(self._restart_swarm(swarm, float(value[0])))
        if self.monitor is not None:
            self.monitor.note_restart(self.shift_estimate, observations, radii)
            self.monitor.count("restart", len(radii))
        return True

    def _observe_displacement(self) -> int:
        """Displacement of the peaks: how far the best solution of a swarm moved between the last two changes, for
        the swarms carried across both whose search scale at the end fell below half the displacement then assumed
        (they re-converged). The estimate is the median of the latest SHIFT_WINDOW such observations, so that a swarm
        that moved to another peak cannot set it alone. Without observations the estimate stays."""
        limit = 0.5 * self.shift_estimate
        observed = []
        for swarm in self.pop:
            old = swarm.gbest_past_environment
            if not np.all(np.isfinite(old)) or self._search_radius(swarm) >= limit:
                continue
            distance = float(np.linalg.norm(swarm.best_position - old))
            if distance > 0.0:
                observed.append(distance)
        if observed:
            self.shift_observations = (self.shift_observations + observed)[-SHIFT_WINDOW:]
            self.shift_estimate = float(np.median(self.shift_observations))
        return len(observed)

    def _restart_swarm(self, swarm: SPMPSwarm, value: float) -> float:
        """Keep the revalidated best as one particle; spread the others and set the cloud at the restart radius."""
        root = float(np.sqrt(self.dimension))
        radius = max(self.shift_estimate, self._search_radius(swarm))
        radius = min(max(radius, self.cloud_radius_min * root), self.cloud_radius_max * root)
        best = swarm.best_position.copy()
        rows = swarm.positions.shape[0]
        positions = best + (radius / root) * self.rng.standard_normal((rows, self.dimension))
        positions = np.clip(positions, self.min_coordinate, self.max_coordinate)
        positions[0] = best
        fitness = np.full(rows, -np.inf, dtype=float)
        fitness[0] = value
        swarm.gbest_past_environment = best.copy()
        swarm.positions = positions
        swarm.fitness = fitness
        swarm.pbest_positions = positions.copy()
        swarm.pbest_fitness = fitness.copy()
        swarm.velocities = np.zeros_like(positions)
        self._reset_best(swarm)
        swarm.cloud_radius = radius / root
        swarm.success_rate = self.success_target
        swarm.is_hibernated = 0
        swarm.is_stagnated = 0
        swarm.stagnating_count = 0
        swarm.stagnation_count = 0
        swarm.last_radius = 0.0
        self._refresh_radius(swarm)
        return radius

    def _search_radius(self, swarm: SPMPSwarm) -> float:
        """Euclidean scale of the swarm's search: the step size of its cloud."""
        return float(swarm.cloud_radius) * float(np.sqrt(self.dimension))

    def _iterative_components(self, problem) -> None:
        limit = float(self.parameters["ConvergenceLimit"])
        for swarm in list(self.pop):
            before_radius = swarm.current_radius
            if swarm.is_hibernated == 0:
                self._pso_move(swarm, problem)
                if problem.state.recent_change or problem.state.terminated:
                    return

            self._refresh_radius(swarm)
            if swarm.current_radius / (before_radius + 0.000001) < 0.9:
                swarm.stagnating_count = 0
            else:
                swarm.stagnating_count += 1

            self._cloud_move(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return

            radius = self._search_radius(swarm)
            if radius < limit:
                if swarm.is_hibernated == 0:
                    self._count("hibernate")
                swarm.is_hibernated = 1
                swarm.is_stagnated = 0
            if swarm.is_hibernated == 1:
                swarm.init_radius = self.distance_accuracy

        self._check_stagnant()
        self.apply_safeguard_exclusion()
        if self.can_use_spatial_guidance() and self.BoAs:
            self.apply_exclusion()
        if self.can_use_spatial_guidance() and self.BoAs:
            self.split_cross_boa_swarms(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
        self.apply_safeguard_exclusion()

    def _pso_move(self, swarm: SPMPSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        r1 = self.rng.random((rows, self.dimension))
        r2 = self.rng.random((rows, self.dimension))
        swarm.velocities = (
            float(self.parameters["w"]) * swarm.velocities
            + float(self.parameters["c1"]) * r1 * (swarm.pbest_positions - swarm.positions)
            + float(self.parameters["c2"]) * r2 * (swarm.best_position - swarm.positions)
        )
        swarm.positions = swarm.positions + swarm.velocities
        clipped = np.clip(swarm.positions, self.min_coordinate, self.max_coordinate)
        swarm.velocities[clipped != swarm.positions] = 0.0
        swarm.positions = clipped
        values = self._evaluate(problem, swarm.positions, rows, "pso", swarm)
        if problem.state.recent_change:
            return
        if problem.state.terminated and not _has_complete_values(values, rows):
            return
        self._record(swarm.positions, values, problem)
        swarm.fitness = _fitness_values(values)
        self._update_personal_bests(swarm)

    def _cloud_move(self, swarm: SPMPSwarm, problem) -> None:
        """Cloud: lambda samples around the swarm best at the swarm's own step size.

        The best sample moves the swarm's best particle there and replaces the swarm
        best when it improves on it. The step size follows the (1+lambda) success rule (Igel, Hansen and Roth 2007):
        the success rate p is smoothed with c_p = p_t*lambda/(2 + p_t*lambda) and the step size is multiplied by
        exp((p - p_t)/(d*(1 - p_t))) with p_t = 1/(5 + sqrt(lambda)/2) and d = 1 + D/(2*lambda).
        """
        if swarm.positions.shape[0] < 2 or swarm.is_hibernated == 1:
            return
        best_id = int(swarm.best_id)
        raw = swarm.pbest_positions[best_id] + swarm.cloud_radius * self.rng.standard_normal(
            (self.cloud_size, self.dimension)
        )
        candidates = np.clip(raw, self.min_coordinate, self.max_coordinate)
        values = self._evaluate(problem, candidates, self.cloud_size, "local", swarm)
        if problem.state.recent_change:
            return
        if problem.state.terminated and not _has_complete_values(values, self.cloud_size):
            return
        self._record(candidates, values, problem)
        chosen = _best_index(values)
        result = float(values[chosen])
        clipped = candidates[chosen] != raw[chosen]
        if np.any(clipped):
            swarm.velocities[best_id, clipped] = 0.0
        swarm.positions[best_id, :] = candidates[chosen]
        swarm.fitness[best_id] = result
        success = result > swarm.pbest_fitness[best_id]
        if success:
            swarm.pbest_fitness[best_id] = result
            swarm.pbest_positions[best_id, :] = candidates[chosen]
            self._reset_best(swarm)
        swarm.success_rate = (1.0 - self.success_smoothing) * swarm.success_rate + self.success_smoothing * float(success)
        if swarm.is_stagnated == 1:
            # The cloud takes the place of the stagnation move, which resets the counter after its step; without
            # the reset a swarm stays flagged, drops out of the diversity test, and the trigger fires every iteration.
            swarm.stagnating_count = 0
        swarm.cloud_radius = float(np.clip(
            swarm.cloud_radius * np.exp(
                (swarm.success_rate - self.success_target) / (self.step_damping * (1.0 - self.success_target))
            ),
            self.cloud_radius_min,
            self.cloud_radius_max,
        ))

    def _check_stagnant(self) -> None:
        active = [swarm.current_radius for swarm in self.pop if swarm.is_hibernated == 0]
        ave_radius = float(np.mean(active)) if active else 0.0
        factor = float(self.parameters["diversityFactor"])
        diversity_limit = factor * self.domainS
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                continue
            size = max(1, swarm.positions.shape[0])
            if (
                (
                    swarm.stagnating_count > size
                    and swarm.current_radius >= ave_radius
                    and swarm.current_radius >= diversity_limit
                )
                or swarm.stagnating_count > 10 * size
            ):
                swarm.is_stagnated = 1
            else:
                swarm.is_stagnated = 0

    def apply_safeguard_exclusion(self) -> None:
        idx = None
        while idx != -1:
            idx = -1
            for ii in range(self.SwarmNumber):
                if self.pop[ii].positions.shape[0] == 0:
                    continue
                for jj in range(ii + 1, self.SwarmNumber):
                    if self.pop[jj].positions.shape[0] == 0:
                        continue
                    distance = float(np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position))
                    if distance >= self.pop[ii].init_radius or distance >= self.pop[jj].init_radius:
                        continue
                    c1 = sum(
                        np.linalg.norm(self.pop[ii].center - row) < self.pop[ii].init_radius
                        for row in self.pop[jj].positions
                    )
                    c2 = sum(
                        np.linalg.norm(self.pop[jj].center - row) < self.pop[jj].init_radius
                        for row in self.pop[ii].positions
                    )
                    if c1 > 0 and c2 > 0:
                        if self.pop[ii].best_fitness > self.pop[jj].best_fitness:
                            self._activate_swarm(ii)
                            self._delete_swarm(jj)
                            idx = jj
                        else:
                            self._activate_swarm(jj)
                            self._delete_swarm(ii)
                            idx = ii
                        break
                if idx != -1:
                    break

    def apply_exclusion(self) -> None:
        if not self.BoAs or self.SwarmNumber < 2:
            return
        n_b = len(self.BoAs)
        l_max = float(np.linalg.norm(self.max_coordinate - self.min_coordinate))
        l_avg = float(np.mean(self.max_coordinate - self.min_coordinate))
        r_e = 0.1 * l_max
        r_s_min = 0.05 * l_avg
        boa_ids = np.zeros(self.SwarmNumber, dtype=int)
        r_s = np.full(self.SwarmNumber, 0.1 * l_avg, dtype=float)
        for idx, swarm in enumerate(self.pop):
            sub_id = int(self.partitioner.locate(swarm.best_position))
            if 0 < sub_id <= self.SubspaceLabels.size:
                boa_ids[idx] = int(self.SubspaceLabels[sub_id - 1])
            if boa_ids[idx] > 0 and boa_ids[idx] <= len(self.BoAs):
                boa = self.BoAs[boa_ids[idx] - 1]
                r_s[idx] = boa.vol * (l_avg / (2.0 * (n_b ** (1.0 / self.dimension))))
            else:
                for b_idx, boa in enumerate(self.BoAs, start=1):
                    if sub_id in set(np.asarray(boa.subspaces, dtype=int).tolist()):
                        boa_ids[idx] = b_idx
                        r_s[idx] = boa.vol * (l_avg / (2.0 * (n_b ** (1.0 / self.dimension))))
                        break
            r_s[idx] = max(r_s[idx], r_s_min)

        remove = np.zeros(self.SwarmNumber, dtype=bool)
        for i in range(self.SwarmNumber - 1):
            if remove[i]:
                continue
            for j in range(i + 1, self.SwarmNumber):
                if remove[j]:
                    continue
                distance = float(np.linalg.norm(self.pop[i].best_position - self.pop[j].best_position))
                should_merge = False
                if self.pop[i].swarm_type == "Exploratory" and self.pop[j].swarm_type == "Exploratory":
                    should_merge = distance < r_e
                elif self.pop[i].swarm_type == "Exploitative" and self.pop[j].swarm_type == "Exploitative":
                    should_merge = (
                        boa_ids[i] > 0
                        and boa_ids[i] == boa_ids[j]
                        and distance < max(r_s[i], r_s[j])
                    )
                if should_merge:
                    if self.pop[i].best_fitness > self.pop[j].best_fitness:
                        remove[j] = True
                    else:
                        remove[i] = True
                        break
        if np.any(remove):
            self.pop = [swarm for idx, swarm in enumerate(self.pop) if not remove[idx]]
            self._sync_swarm_number("boa_exclusion")

    def split_cross_boa_swarms(self, problem) -> None:
        if not self.BoAs or self.partitioner is None:
            return
        new_swarms: list[SPMPSwarm] = []
        remove_indices: list[int] = []
        for idx, swarm in enumerate(self.pop):
            if swarm.swarm_type == "Ghost":
                continue
            if swarm.stagnating_count < swarm.positions.shape[0]:
                continue
            if swarm.pbest_positions.shape[0] < 2:
                continue
            particle_boa = np.zeros(swarm.pbest_positions.shape[0], dtype=int)
            particle_subspaces = np.asarray(self.partitioner.locate(swarm.pbest_positions), dtype=int)
            valid = (particle_subspaces > 0) & (particle_subspaces <= self.SubspaceLabels.size)
            if self.SubspaceLabels.size:
                particle_boa[valid] = self.SubspaceLabels[particle_subspaces[valid] - 1]
            else:
                for p_idx, sub_id in enumerate(particle_subspaces):
                    for b_idx, boa in enumerate(self.BoAs, start=1):
                        if int(sub_id) in set(np.asarray(boa.subspaces, dtype=int).tolist()):
                            particle_boa[p_idx] = b_idx
                            break
            unique_boas = np.unique(particle_boa[particle_boa > 0])
            if unique_boas.size <= 1:
                continue
            remove_indices.append(idx)
            for boa_id in unique_boas:
                members = np.flatnonzero(particle_boa == boa_id)
                if members.size < 2:
                    continue
                positions = swarm.pbest_positions[members, :].copy()
                values = swarm.pbest_fitness[members].copy()
                new_swarm = _make_spmp_swarm(
                    positions,
                    values,
                    self.dimension,
                    swarm_type="Exploitative",
                )
                new_swarms.append(new_swarm)
        if remove_indices:
            for idx in reversed(remove_indices):
                del self.pop[idx]
            self.pop.extend(new_swarms)
            self._sync_swarm_number("split")

    def estimate_boas(self) -> None:
        previous_boas = list(self.BoAs) if hasattr(self, "BoAs") else []
        n_s = len(self.Subspaces)
        labels = np.zeros(n_s + 1, dtype=int)
        diag = self._default_boa_diagnostics()
        diag["NumSubspaces"] = n_s
        robust_scores = np.full(n_s + 1, -np.inf, dtype=float)
        best_scores = np.full(n_s + 1, -np.inf, dtype=float)
        counts = np.zeros(n_s + 1, dtype=int)
        sampled_mask = np.zeros(n_s + 1, dtype=bool)
        best_rows = np.full(n_s + 1, -1, dtype=int)

        if n_s == 0:
            self.BoAs = []
            self.SubspaceLabels = np.zeros(0, dtype=int)
            self.SubspaceAnyBoA = np.zeros(0, dtype=bool)
            self.LastBoADiagnostics = diag
            return

        if self.History.size:
            subs = np.asarray(
                self.partitioner.locate(self.History[:, : self.dimension]),
                dtype=int,
            )
            fits = self.History[:, self.dimension]
            for sub_id in range(1, n_s + 1):
                values = fits[subs == sub_id]
                values = values[~np.isnan(values)]
                counts[sub_id] = int(values.size)
                if values.size:
                    ordered = np.sort(values)[::-1]
                    top_k = min(int(self.parameters["BoATopK"]), ordered.size)
                    robust_scores[sub_id] = float(np.mean(ordered[:top_k]))
                    best_scores[sub_id] = float(ordered[0])
                    sampled_mask[sub_id] = True
                    rows = np.flatnonzero((subs == sub_id) & ~np.isnan(fits))
                    best_rows[sub_id] = int(rows[np.argmax(fits[rows])])

        for subspace in self.Subspaces:
            sid = int(subspace.index)
            subspace.S_i = int(counts[sid])
            subspace.F_i = float(robust_scores[sid]) if sampled_mask[sid] else 0.0

        sampled_ids = np.flatnonzero(sampled_mask)
        sampled_ids = sampled_ids[sampled_ids > 0]
        diag["SampledSubspaceCount"] = int(sampled_ids.size)
        diag["UnsampledSubspaceCount"] = int(n_s - sampled_ids.size)
        if sampled_ids.size == 0:
            self.BoAs = []
            self.SubspaceLabels = np.zeros(n_s, dtype=int)
            self.SubspaceAnyBoA = np.zeros(n_s, dtype=bool)
            for subspace in self.Subspaces:
                subspace.isOverlap = True
            diag["OverlapSubspaceCount"] = n_s
            diag["OverlapSubspaceRatio"] = 1.0
            self.LastBoADiagnostics = diag
            return

        parent = np.zeros(n_s + 1, dtype=int)
        mode_cache = np.zeros(n_s + 1, dtype=int)
        raw_label_by_cell = np.zeros(n_s + 1, dtype=int)
        for sub_id in sampled_ids:
            parent[sub_id] = self._select_ascent_parent(int(sub_id), robust_scores, counts)
        for sub_id in sampled_ids:
            mode_cache[sub_id] = self._resolve_mode(int(sub_id), parent, mode_cache)
        raw_mode_cells = np.unique(mode_cache[sampled_ids])
        raw_mode_to_id = {int(cell): idx + 1 for idx, cell in enumerate(raw_mode_cells)}
        for sub_id in sampled_ids:
            raw_label_by_cell[sub_id] = raw_mode_to_id[int(mode_cache[sub_id])]

        raw_basins: list[dict[str, object]] = []
        for raw_idx, mode_cell in enumerate(raw_mode_cells, start=1):
            members = np.flatnonzero(raw_label_by_cell == raw_idx)
            raw_basins.append(
                {
                    "modeCell": int(mode_cell),
                    "modeScore": float(robust_scores[int(mode_cell)]),
                    "members": members.astype(int),
                    "sampleCount": int(np.sum(counts[members])),
                }
            )
        diag["RawBasinCount"] = len(raw_basins)
        diag["MeanSampleSupport"] = float(np.mean(counts[sampled_ids]))
        diag["MaxSampleSupport"] = int(np.max(counts[sampled_ids]))

        boundary_saddle = np.full((len(raw_basins), len(raw_basins)), -np.inf)
        for sub_id in sampled_ids:
            for neighbor in self._neighbors(int(sub_id)):
                if neighbor <= sub_id or neighbor > n_s or not sampled_mask[neighbor]:
                    continue
                basin_i = raw_label_by_cell[sub_id] - 1
                basin_j = raw_label_by_cell[neighbor] - 1
                if basin_i == basin_j or basin_i < 0 or basin_j < 0:
                    continue
                saddle = min(robust_scores[sub_id], robust_scores[neighbor])
                if saddle > boundary_saddle[basin_i, basin_j]:
                    boundary_saddle[basin_i, basin_j] = saddle
                    boundary_saddle[basin_j, basin_i] = saddle

        sampled_scores = robust_scores[sampled_ids]
        score_scale = float(np.std(sampled_scores))
        if not score_scale > 0:
            score_scale = max(float(np.max(sampled_scores) - np.min(sampled_scores)), 1.0)
        support_scale = self._support_scale(counts[sampled_ids])
        clusters = [[idx] for idx in range(len(raw_basins))]
        active = np.ones(len(raw_basins), dtype=bool)
        raw_persistences: list[float] = []
        for cluster_id in range(len(raw_basins)):
            higher_id, best_saddle = self._find_higher_neighbor_cluster(
                cluster_id,
                clusters,
                active,
                raw_basins,
                boundary_saddle,
            )
            if higher_id >= 0 and np.isfinite(best_saddle):
                raw_persistences.append(
                    max(0.0, float(raw_basins[cluster_id]["modeScore"]) - best_saddle)
                )
        persistence_threshold = self._persistence_scale(np.asarray(raw_persistences), score_scale)
        diag["ScoreScale"] = score_scale
        diag["PersistenceThreshold"] = persistence_threshold

        while True:
            merge_from, merge_to = self._find_merge_candidate(
                clusters,
                active,
                raw_basins,
                boundary_saddle,
                persistence_threshold,
                support_scale,
            )
            if merge_from < 0:
                break
            clusters[merge_to].extend(clusters[merge_from])
            active[merge_from] = False

        active_ids = np.flatnonzero(active)
        diag["MergedClusterCount"] = int(active_ids.size)
        diag["MergedAwayCount"] = int(len(raw_basins) - active_ids.size)
        candidates: list[dict[str, object]] = []
        for cluster_id in active_ids:
            raw_ids = clusters[int(cluster_id)]
            mode_score, mode_cell, total_support = self._cluster_stats(raw_ids, raw_basins)
            member_cells = np.unique(
                np.concatenate(
                    [np.asarray(raw_basins[raw_id]["members"], dtype=int) for raw_id in raw_ids]
                )
            )
            higher_id, best_saddle = self._find_higher_neighbor_cluster(
                int(cluster_id),
                clusters,
                active,
                raw_basins,
                boundary_saddle,
            )
            persistence = np.inf if higher_id < 0 else max(0.0, mode_score - best_saddle)
            member_scores = robust_scores[member_cells]
            if member_scores.size == 0:
                continue
            if np.isfinite(persistence):
                core_floor = mode_score - max(0.5 * persistence, persistence_threshold)
            else:
                core_floor = mode_score - max(float(np.std(member_scores)), persistence_threshold)
            core_members = member_cells[member_scores >= core_floor - np.finfo(float).eps]
            if core_members.size == 0:
                core_members = np.asarray([mode_cell], dtype=int)

            sample_conf = total_support / max(total_support + support_scale, np.finfo(float).eps)
            if np.isfinite(persistence):
                persistence_conf = persistence / max(
                    persistence + persistence_threshold,
                    np.finfo(float).eps,
                )
            else:
                persistence_conf = 1.0
            spatial_conf = member_cells.size / max(
                member_cells.size + max(1, len(raw_ids)) + int(self.parameters["BoATopK"]),
                np.finfo(float).eps,
            )
            confidence = 0.5 * sample_conf + 0.35 * persistence_conf + 0.15 * spatial_conf
            occupied = self._basin_holds_swarm_best(member_cells)
            summary = {
                "modeCell": int(mode_cell),
                "modeScore": float(mode_score),
                "peakFitness": float(best_scores[mode_cell]),
                "sampleCount": int(total_support),
                "coreSubspaceCount": int(core_members.size),
                "allSubspaceCount": int(member_cells.size),
                "confidence": float(confidence),
                "persistence": float(persistence),
                "isOccupied": bool(occupied),
                "isGhost": not bool(occupied),
                "acceptance": "",
            }
            row = int(best_rows[int(mode_cell)])
            candidates.append(
                {
                    "summary": summary,
                    "coreMembers": core_members.astype(int),
                    "memberCells": member_cells.astype(int),
                    "peakPoint": self.History[row, : self.dimension].copy() if row >= 0 else None,
                }
            )

        ghost_reference = self._weighted_confidence_reference(candidates, only_ghost=True)
        accepted_reference = self._weighted_confidence_reference(candidates, only_ghost=False)
        diag["ConfidenceThreshold"] = ghost_reference

        boas: list[SPMPBoA] = []
        accepted: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        next_label = 0
        for candidate in candidates:
            summary = dict(candidate["summary"])
            core_members = np.asarray(candidate["coreMembers"], dtype=int)
            member_cells = np.asarray(candidate["memberCells"], dtype=int)
            accept_ghost = bool(summary["isGhost"]) and float(summary["confidence"]) >= ghost_reference
            if not (bool(summary["isOccupied"]) or accept_ghost):
                summary["acceptance"] = "rejected_adaptive_ghost"
                rejected.append(summary)
                continue

            next_label += 1
            labels[core_members] = next_label
            if bool(summary["isOccupied"]):
                summary["acceptance"] = "accepted_occupied"
            else:
                summary["acceptance"] = "accepted_adaptive_ghost"
            accepted.append(summary)
            boas.append(
                SPMPBoA(
                    subspaces=core_members,
                    allSubspaces=member_cells,
                    vol=float(sum(self.Subspaces[int(sid) - 1].vol for sid in core_members)),
                    allVol=float(sum(self.Subspaces[int(sid) - 1].vol for sid in member_cells)),
                    peakSubspace=int(summary["modeCell"]),
                    robustPeakScore=float(summary["modeScore"]),
                    peakFitness=float(summary["peakFitness"]),
                    isGhost=bool(summary["isGhost"]),
                    hasGenerated=False,
                    persistence=float(summary["persistence"]),
                    confidence=float(summary["confidence"]),
                    sampleCount=int(summary["sampleCount"]),
                    coreSubspaceCount=int(summary["coreSubspaceCount"]),
                    allSubspaceCount=int(summary["allSubspaceCount"]),
                    peakPoint=candidate["peakPoint"],
                )
            )

        self._inherit_boa_state(previous_boas, boas)
        self.BoAs = boas
        self.SubspaceLabels = labels[1:].copy()
        self.SubspaceAnyBoA = np.zeros(n_s, dtype=bool)
        for boa in boas:
            valid = np.asarray(boa.allSubspaces, dtype=int)
            valid = valid[(valid > 0) & (valid <= n_s)]
            self.SubspaceAnyBoA[valid - 1] = True
        for subspace in self.Subspaces:
            subspace.isOverlap = bool(labels[int(subspace.index)] == 0)

        diag["AcceptedBoACount"] = len(boas)
        diag["RejectedClusterCount"] = len(rejected)
        diag["OccupiedAcceptedCount"] = sum(not boa.isGhost for boa in boas)
        diag["GhostAcceptedCount"] = sum(boa.isGhost for boa in boas)
        diag["ReliableAcceptedCount"] = sum(boa.confidence >= accepted_reference for boa in boas)
        diag["OverlapSubspaceCount"] = int(np.sum(labels[1:] == 0))
        diag["OverlapSubspaceRatio"] = diag["OverlapSubspaceCount"] / max(n_s, 1)
        diag["TopBoAs"] = self._rank_boa_summaries(accepted)
        diag["TopRejectedBoAs"] = self._rank_boa_summaries(rejected)
        self.LastBoADiagnostics = diag

    def _select_ascent_parent(
        self,
        subspace_id: int,
        scores: np.ndarray,
        counts: np.ndarray,
    ) -> int:
        parent_id = int(subspace_id)
        best_score = float(scores[subspace_id])
        best_count = int(counts[subspace_id])
        for neighbor in self._neighbors(subspace_id):
            if neighbor <= 0 or neighbor >= scores.size or not np.isfinite(scores[neighbor]):
                continue
            if scores[neighbor] > best_score + np.finfo(float).eps:
                best_score = float(scores[neighbor])
                best_count = int(counts[neighbor])
                parent_id = int(neighbor)
            elif abs(scores[neighbor] - best_score) <= np.finfo(float).eps and counts[neighbor] > best_count:
                best_count = int(counts[neighbor])
                parent_id = int(neighbor)
        return parent_id

    def _neighbors(self, subspace_id: int) -> list[int]:
        """Cells adjacent to a cell for the basin model: the cells that share a face with it."""
        return self.partitioner.face_neighbors()[int(subspace_id)]

    def _resolve_mode(self, subspace_id: int, parent: np.ndarray, cache: np.ndarray) -> int:
        current = int(subspace_id)
        while True:
            if cache[current] != 0:
                return int(cache[current])
            if parent[current] == current or parent[current] == 0:
                return current
            current = int(parent[current])

    def _cluster_stats(
        self,
        raw_ids: list[int],
        raw_basins: list[dict[str, object]],
    ) -> tuple[float, int, int]:
        scores = np.asarray([float(raw_basins[idx]["modeScore"]) for idx in raw_ids])
        best = int(np.argmax(scores))
        basin = raw_basins[raw_ids[best]]
        support = int(sum(int(raw_basins[idx]["sampleCount"]) for idx in raw_ids))
        return float(scores[best]), int(basin["modeCell"]), support

    def _cluster_boundary_saddle(
        self,
        raw_ids_a: list[int],
        raw_ids_b: list[int],
        boundary_saddle: np.ndarray,
    ) -> float:
        saddle = -np.inf
        for raw_id in raw_ids_a:
            if len(raw_ids_b) == 0:
                continue
            candidate = float(np.max(boundary_saddle[raw_id, raw_ids_b]))
            if candidate > saddle:
                saddle = candidate
        return saddle

    def _find_higher_neighbor_cluster(
        self,
        cluster_id: int,
        clusters: list[list[int]],
        active: np.ndarray,
        raw_basins: list[dict[str, object]],
        boundary_saddle: np.ndarray,
    ) -> tuple[int, float]:
        cluster_score, _, _ = self._cluster_stats(clusters[cluster_id], raw_basins)
        higher = -1
        best_saddle = -np.inf
        for other_id in np.flatnonzero(active):
            other = int(other_id)
            if other == cluster_id:
                continue
            other_score, _, _ = self._cluster_stats(clusters[other], raw_basins)
            if other_score <= cluster_score + np.finfo(float).eps:
                continue
            candidate = self._cluster_boundary_saddle(
                clusters[cluster_id],
                clusters[other],
                boundary_saddle,
            )
            if candidate > best_saddle:
                best_saddle = candidate
                higher = other
        if not np.isfinite(best_saddle):
            return -1, -np.inf
        return higher, best_saddle

    def _find_merge_candidate(
        self,
        clusters: list[list[int]],
        active: np.ndarray,
        raw_basins: list[dict[str, object]],
        boundary_saddle: np.ndarray,
        persistence_threshold: float,
        min_support: float,
    ) -> tuple[int, int]:
        active_ids = np.flatnonzero(active)
        if active_ids.size == 0:
            return -1, -1
        scores = np.asarray(
            [self._cluster_stats(clusters[int(idx)], raw_basins)[0] for idx in active_ids],
            dtype=float,
        )
        for order_idx in np.argsort(scores):
            cluster_id = int(active_ids[order_idx])
            cluster_score, _, support = self._cluster_stats(clusters[cluster_id], raw_basins)
            higher_id, best_saddle = self._find_higher_neighbor_cluster(
                cluster_id,
                clusters,
                active,
                raw_basins,
                boundary_saddle,
            )
            if higher_id < 0:
                continue
            persistence = max(0.0, cluster_score - best_saddle)
            should_merge = (
                persistence <= persistence_threshold
                or (
                    support < min_support
                    and persistence <= 2.0 * persistence_threshold
                )
            )
            if should_merge:
                return cluster_id, higher_id
        return -1, -1

    def _support_scale(self, sample_counts: np.ndarray) -> float:
        values = np.asarray(sample_counts, dtype=float).reshape(-1)
        if values.size == 0:
            return 1.0
        return max(1.0, float(np.median(values)))

    def _persistence_scale(self, raw_persistences: np.ndarray, score_scale: float) -> float:
        values = np.asarray(raw_persistences, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return max(float(score_scale), float(np.finfo(float).eps))
        center = float(np.median(values))
        dispersion = float(np.median(np.abs(values - center)))
        return max(center + dispersion, float(np.finfo(float).eps))

    def _weighted_confidence_reference(
        self,
        candidates: list[dict[str, object]],
        *,
        only_ghost: bool,
    ) -> float:
        selected: list[dict[str, object]] = []
        for candidate in candidates:
            summary = candidate["summary"]
            if only_ghost and not bool(summary["isGhost"]):
                continue
            selected.append(summary)
        if not selected:
            return 0.0
        confidences = np.asarray([float(item["confidence"]) for item in selected], dtype=float)
        supports = np.asarray([float(item["sampleCount"]) for item in selected], dtype=float)
        total_support = float(np.sum(supports))
        if total_support <= 0:
            return float(np.mean(confidences))
        return float(np.sum(confidences * supports) / total_support)

    def _basin_holds_swarm_best(self, member_cells: np.ndarray) -> bool:
        """A basin is occupied when a swarm's best solution lies in one of its cells."""
        cells = set(np.asarray(member_cells, dtype=int).tolist())
        return any(int(self.partitioner.locate(swarm.best_position)) in cells for swarm in self.pop)

    def _inherit_boa_state(
        self,
        previous_boas: list[SPMPBoA],
        boas: list[SPMPBoA],
    ) -> None:
        if not previous_boas or not boas:
            return
        for boa in boas:
            cells = set(np.asarray(boa.allSubspaces, dtype=int).tolist())
            boa.hasGenerated = any(
                previous.hasGenerated
                and previous.peakPoint is not None
                and int(self.partitioner.locate(previous.peakPoint)) in cells
                for previous in previous_boas
            )

    def handle_ghost_boas(self, problem) -> None:
        diag = self._default_ghost_diagnostics()
        diag["TotalBoAs"] = len(self.BoAs)
        if not self.BoAs:
            diag["Reason"] = "no_boas"
            self.LastGhostDiagnostics = diag
            return
        candidates = [
            (idx, boa)
            for idx, boa in enumerate(self.BoAs)
            if boa.isGhost and boa.vol >= 0.01 and not boa.hasGenerated
        ]
        diag["EligibleCount"] = len(candidates)
        if not candidates:
            diag["Reason"] = "no_eligible_ghost_boa"
            self.LastGhostDiagnostics = diag
            return
        candidates.sort(key=lambda item: (item[1].confidence, item[1].peakFitness), reverse=True)
        diag["CandidatePreview"] = [
            {
                "idx": int(idx + 1),
                "peakFitness": float(boa.peakFitness),
                "confidence": float(boa.confidence),
                "sampleCount": int(boa.sampleCount),
            }
            for idx, boa in candidates[:3]
        ]
        for idx, boa in candidates[:1]:
            core = np.asarray(boa.subspaces, dtype=int)
            if core.size == 0:
                continue
            known = boa.peakPoint is not None
            if known:
                cell = self.Subspaces[int(boa.peakSubspace) - 1]
                new_x = cell.min + (cell.max - cell.min) * self.rng.random((2, self.dimension))
            else:
                new_x = np.zeros((3, self.dimension), dtype=float)
                chosen = self.rng.choice(core, size=3, replace=True)
                for row, sub_id in enumerate(chosen):
                    subspace = self.Subspaces[int(sub_id) - 1]
                    new_x[row] = subspace.min + (subspace.max - subspace.min) * self.rng.random(
                        self.dimension
                    )
            values = self._evaluate(problem, new_x, new_x.shape[0], "ghost")
            if problem.state.recent_change:
                diag["InterruptedByChange"] = bool(problem.state.recent_change)
                diag["Reason"] = "interrupted_by_change"
                self.LastGhostDiagnostics = diag
                return
            if problem.state.terminated and not _has_complete_values(values, new_x.shape[0]):
                diag["InterruptedByChange"] = False
                diag["Reason"] = "interrupted_by_termination"
                self.LastGhostDiagnostics = diag
                return
            self._record(new_x, values, problem)
            if known:
                new_x = np.vstack([boa.peakPoint, new_x])
                values = np.concatenate([[boa.peakFitness], values])
            swarm = _make_spmp_swarm(new_x, values, self.dimension, swarm_type="Ghost")
            self.pop.append(swarm)
            boa.hasGenerated = True
            diag["SelectedBoAIndices"].append(int(idx + 1))
            diag["DispatchedCount"] += 1
            diag["CreatedSwarmCount"] += 1
            if problem.state.terminated:
                break
        diag["Reason"] = "dispatched" if diag["DispatchedCount"] > 0 else "eligible_but_skipped"
        self._count("dispatch", diag["DispatchedCount"])
        self._sync_swarm_number("ghost")
        self.LastGhostDiagnostics = diag

    def calculate_subspace_uncertainty(self) -> None:
        if not self.Subspaces:
            return
        s_i = np.asarray([getattr(subspace, "S_i", 0) for subspace in self.Subspaces], dtype=float)
        f_i = np.asarray([getattr(subspace, "F_i", 0.0) for subspace in self.Subspaces], dtype=float)
        s_max, s_min = float(np.max(s_i)), float(np.min(s_i))
        f_max, f_min = float(np.max(f_i)), float(np.min(f_i))
        alpha = float(self.parameters["alpha"])
        beta = float(self.parameters["beta"])
        gamma = float(self.parameters["gamma"])
        for idx, subspace in enumerate(self.Subspaces):
            if s_max > s_min:
                s_term = (0.8 * s_i[idx] + 0.2 * s_min - s_max) / (s_min - s_max)
            else:
                s_term = 0.5
            c_term = max(0.0, (100.0 - min(s_i[idx], 100.0)) * 0.01)
            f_term = (f_i[idx] - f_min) / (f_max - f_min) if f_max > f_min else 0.5
            subspace.uncertainty = float(alpha * s_term + beta * c_term + gamma * f_term)

    def calculate_sampling_probs(self, values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float).reshape(-1)
        if vals.size == 0:
            return vals
        denom = float(np.max(vals) - np.min(vals))
        if denom < np.finfo(float).eps:
            return np.ones(vals.size, dtype=float) / vals.size
        probs = (0.8 * vals + 0.2 * float(np.max(vals)) - float(np.min(vals))) / denom
        probs = np.maximum(probs, 0.0)
        total = float(np.sum(probs))
        if total <= 0:
            return np.ones(vals.size, dtype=float) / vals.size
        return probs / total

    def sample_from_subspaces(
        self,
        subspace_ids: np.ndarray | list[int],
        probs: np.ndarray,
        count: int,
    ) -> np.ndarray:
        ids = np.asarray(subspace_ids, dtype=int).reshape(-1)
        if ids.size == 0 or int(count) <= 0:
            return np.empty((0, self.dimension), dtype=float)
        normalized = np.asarray(probs, dtype=float).reshape(-1)
        normalized = normalized / np.sum(normalized)
        chosen = self.rng.choice(ids, size=int(count), replace=True, p=normalized)
        samples = np.zeros((int(count), self.dimension), dtype=float)
        for row, sub_id in enumerate(chosen):
            subspace = self.Subspaces[int(sub_id) - 1]
            samples[row] = subspace.min + (subspace.max - subspace.min) * self.rng.random(
                self.dimension
            )
        return samples

    def update_swarm_types(self) -> None:
        if not self.Subspaces or self.partitioner is None or self.SubspaceLabels.size == 0:
            self.LastSwarmTypeDiagnostics = self._default_swarm_type_diagnostics()
            self.LastSwarmTypeDiagnostics["TotalSwarms"] = self.SwarmNumber
            return
        ghosts_to_remove: list[int] = []
        for idx, swarm in enumerate(self.pop):
            particle_subs = np.asarray(self.partitioner.locate(swarm.pbest_positions), dtype=int)
            labels = np.zeros(particle_subs.shape, dtype=int)
            valid = (particle_subs > 0) & (particle_subs <= self.SubspaceLabels.size)
            labels[valid] = self.SubspaceLabels[particle_subs[valid] - 1]
            positive = np.unique(labels[labels > 0])
            if swarm.swarm_type == "Ghost":
                if positive.size == 1 and np.all(labels == positive[0]):
                    swarm.swarm_type = "Exploitative"
                elif np.all(labels == 0) and not self._swarm_in_any_boa(particle_subs):
                    ghosts_to_remove.append(idx)
                continue
            if positive.size == 1 and np.all(labels == positive[0]):
                swarm.swarm_type = "Exploitative"
            else:
                swarm.swarm_type = "Exploratory"
        for idx in reversed(ghosts_to_remove):
            del self.pop[idx]
        self._sync_swarm_number("type_ghost")
        self.LastSwarmTypeDiagnostics = self._collect_swarm_type_diagnostics(
            len(ghosts_to_remove)
        )

    def global_swarm_elimination(self, max_swarms: int) -> None:
        if self.SwarmNumber <= int(max_swarms):
            return
        positions = np.vstack([swarm.best_position for swarm in self.pop])
        fitnesses = np.asarray([swarm.best_fitness for swarm in self.pop], dtype=float)
        fit_min, fit_max = float(np.min(fitnesses)), float(np.max(fitnesses))
        if fit_max > fit_min:
            norm_fit = (fitnesses - fit_min) / (fit_max - fit_min)
        else:
            norm_fit = np.full(fitnesses.shape, 0.5)
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        crowding = np.min(distances, axis=1)
        crowd_min, crowd_max = float(np.min(crowding)), float(np.max(crowding))
        if crowd_max > crowd_min:
            norm_crowd = (crowding - crowd_min) / (crowd_max - crowd_min)
        else:
            norm_crowd = np.full(crowding.shape, 0.5)
        scores = 0.5 * norm_fit + 0.5 * norm_crowd
        for idx, swarm in enumerate(self.pop):
            if swarm.swarm_type == "Ghost":
                scores[idx] *= 0.5
        remove_count = self.SwarmNumber - int(max_swarms)
        remove = set(np.argsort(scores)[:remove_count].tolist())
        self.pop = [swarm for idx, swarm in enumerate(self.pop) if idx not in remove]
        self._sync_swarm_number("cap")

    def update_spatial_model_state(self) -> None:
        state = self._default_spatial_model_state()
        state.HistoryWindowFE = int(self.HistoryWindowFE)
        diag = getattr(self, "LastBoADiagnostics", None)
        if not diag:
            state.Reason = "no_boa_diagnostics"
            self.SpatialModelState = state
            self.spatial_model_state = state
            return
        state.SampledRatio = diag["SampledSubspaceCount"] / max(diag["NumSubspaces"], 1)
        state.OverlapRatio = float(diag["OverlapSubspaceRatio"])
        state.ReliableBoACount = int(diag["ReliableAcceptedCount"])
        if self.BoAs:
            state.ConfidenceMean = float(np.mean([boa.confidence for boa in self.BoAs]))
        sampled_score = float(np.sqrt(max(state.SampledRatio, 0.0)))
        overlap_score = float(max(0.0, 1.0 - state.OverlapRatio))
        occupied_share = diag["OccupiedAcceptedCount"] / max(diag["AcceptedBoACount"], 1)
        accepted_density = min(1.0, diag["AcceptedBoACount"] / max(self.SwarmNumber, 1))
        confidence_score = (
            state.ConfidenceMean
            * float(np.sqrt(max(occupied_share, 0.0)))
            * float(accepted_density)
        )
        support_scale = max(1.0, float(np.sqrt(max(diag["MaxSampleSupport"], 0) + 1.0)))
        support_score = float(
            diag["MeanSampleSupport"]
            / max(diag["MeanSampleSupport"] + support_scale, np.finfo(float).eps)
        )
        state.Score = (
            0.30 * sampled_score
            + 0.15 * overlap_score
            + 0.35 * confidence_score
            + 0.20 * support_score
        )
        state.ReferenceScore = 0.5 * state.SampledRatio
        state.GuidanceWeight = min(
            1.0,
            max(0.0, state.Score * float(np.sqrt(max(confidence_score, 0.0)))),
        )
        state.GuidanceShare = min(
            1.0,
            max(0.0, 0.5 * state.GuidanceWeight + 0.25 * state.Score),
        )
        scores = [sampled_score, overlap_score, confidence_score, support_score]
        reasons = ["sampled_sparse", "high_overlap", "low_confidence", "low_support"]
        state.Reason = reasons[int(np.argmin(scores))]
        state.IsReliable = state.GuidanceWeight >= state.ReferenceScore
        if state.IsReliable:
            state.Mode = "guided"
        else:
            state.Mode = "conservative"
        self.SpatialModelState = state
        self.spatial_model_state = state

    def can_use_spatial_guidance(self) -> bool:
        return bool(getattr(self.SpatialModelState, "IsReliable", False))

    def _diversity_monitor_and_adjust(self, problem) -> None:
        active = [
            swarm.current_radius
            for swarm in self.pop
            if swarm.is_stagnated == 0 and swarm.is_hibernated == 0
        ]
        non_stagnated_radius = float(np.mean(active)) if active else 0.0
        if self.SwarmNumber > 0 and non_stagnated_radius >= float(self.parameters["diversityFactor"]) * self.domainS:
            return
        self._count("trigger")

        use_guidance = self.can_use_spatial_guidance()
        guidance_weight = self._guidance_weight()
        if use_guidance and self.BoAs:
            self.handle_ghost_boas(problem)
            if problem.state.recent_change or problem.state.terminated:
                return
        elif use_guidance:
            self._mark_ghost_dispatch_skipped("no_boa_available")
        else:
            self._mark_ghost_dispatch_skipped("conservative_mode")

        target_nomega = self.BaseNomega + int(self.parameters["M_factor"]) * self.SwarmNumber
        self.N_omega = int(round(self.ConservativeNomega + guidance_weight * (target_nomega - self.ConservativeNomega)))
        self._apply_partition_budget(self.N_omega)
        self.estimate_boas()
        self.update_swarm_types()
        self.update_spatial_model_state()
        use_guidance = self.can_use_spatial_guidance()
        guidance_share = self._guidance_share()

        self._remove_orphan_ghost_swarms()
        self.record_pop_info.append(self.SwarmNumber)
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                swarm.is_hibernated = 0
                swarm.stagnating_count = 0
                swarm.is_stagnated = 0
        self.hiber_area = []
        self.HiberArea = self.hiber_area

        self.iter_pop_size_end.append(self.SwarmNumber)
        if len(self.iter_pop_size_end) > 2:
            self.iter_pop_size_end.pop(0)
        if self.adjust_iter == 0:
            total = sum(swarm.positions.shape[0] for swarm in self.pop)
            self.iter_individuals_begin = [float(total)]

        self._update_memory()
        if self.first_update == 1:
            self.current_pop_size = self.iter_pop_size_end[-1]
            self.previous_pop_size = self.current_pop_size
            self.first_update = 0
        else:
            self.current_pop_size = self.iter_pop_size_end[-1]
            self.previous_pop_size = self.iter_pop_size_end[-2]

        record = self._map_record_for_count(self.current_pop_size)
        ratio = abs(self.current_pop_size - self.previous_pop_size) / self.off_peak
        next_individuals = record.mean + record.std_dev * self.rng.standard_normal()
        if ratio >= 1 or (0 < ratio and self.rng.random() <= ratio):
            next_individuals += int(self.parameters["StepIndivs"]) * (
                self.current_pop_size - self.previous_pop_size
            )

        total_survive = sum(swarm.positions.shape[0] for swarm in self.pop)
        if next_individuals <= total_survive:
            next_individuals = total_survive + int(self.parameters["StepIndivs"]) * 2
        next_individuals = int(
            np.ceil(min(self.max_indivs, max(self.min_indivs, next_individuals)))
        )
        self.iter_individuals_begin = [float(next_individuals)]

        num_remain = self.remain_indivs.shape[0]
        num_add = next_individuals - total_survive - num_remain
        guided_count = 0
        random_count = 0
        # When the pending remainder alone exceeds the shortfall, fill the shortfall from it.
        shortfall = next_individuals - total_survive
        pending = num_add <= 0 and num_remain > 0 and shortfall > 0
        if num_add > 0 or pending:
            if pending:
                add = np.empty((0, self.dimension))
            elif use_guidance:
                guided_count = int(round(guidance_share * num_add))
                guided_count = max(0, min(num_add, guided_count))
                random_count = num_add - guided_count
                parts: list[np.ndarray] = []
                if random_count > 0:
                    parts.append(self.bounds.sample(random_count, self.rng))
                if guided_count > 0:
                    self.calculate_subspace_uncertainty()
                    ids = np.arange(1, len(self.Subspaces) + 1, dtype=int)
                    uncertainties = np.asarray([sub.uncertainty for sub in self.Subspaces], dtype=float)
                    probs = self.calculate_sampling_probs(uncertainties)
                    parts.append(self.sample_from_subspaces(ids, probs, guided_count))
                add = np.vstack(parts) if parts else np.empty((0, self.dimension))
            else:
                random_count = num_add
                add = self.bounds.sample(num_add, self.rng)
            leftover = self.remain_indivs[shortfall:] if pending else np.empty((0, self.dimension))
            if num_remain:
                add = np.vstack([add, self.remain_indivs[:shortfall] if pending else self.remain_indivs])
            self.remain_indivs = np.empty((0, self.dimension))
            self.remainIndivs = self.remain_indivs
            swarms, remain, stamps = self._generate_swarms(add, problem, "replenish")
            self.remain_indivs = np.vstack([remain, leftover]) if leftover.size else remain
            self.remainIndivs = self.remain_indivs
            if not problem.state.recent_change:
                self._record_new_swarms(swarms, stamps, problem)
            self.pop.extend(swarms)
            self._sync_swarm_number("replenish")
            if problem.state.recent_change or problem.state.terminated:
                return

        self.LastResourceAllocationDiagnostics = {
            "NextIndividuals": int(next_individuals),
            "Added": max(0, int(num_add)),
            "Guided": int(guided_count),
            "Random": int(random_count),
            "Mode": self.SpatialModelState.Mode,
            "Swarms": self.SwarmNumber,
        }
        self.adjust_iter += 1

    def _remove_orphan_ghost_swarms(self) -> None:
        if not self.BoAs:
            return
        keep: list[SPMPSwarm] = []
        for swarm in self.pop:
            if swarm.swarm_type != "Ghost":
                keep.append(swarm)
                continue
            sub_ids = np.asarray(self.partitioner.locate(swarm.pbest_positions), dtype=int)
            if self.SubspaceAnyBoA.size:
                valid = (sub_ids > 0) & (sub_ids <= self.SubspaceAnyBoA.size)
                in_boa = bool(np.any(self.SubspaceAnyBoA[sub_ids[valid] - 1]))
            else:
                in_boa = any(
                    any(int(sub_id) in set(np.asarray(boa.allSubspaces, dtype=int).tolist()) for sub_id in sub_ids)
                    for boa in self.BoAs
                )
            if in_boa:
                keep.append(swarm)
        self.pop = keep
        self._sync_swarm_number("orphan_ghost")

    def _enforce_swarm_limit(self) -> None:
        limit = self._get_swarm_limit()
        if self.SwarmNumber > limit:
            self.global_swarm_elimination(limit)

    def _get_swarm_limit(self) -> int:
        max_swarms = int(self.parameters["MAX_SWARMS"])
        if max_swarms <= 0:
            return max(self.SwarmNumber, 1)
        guidance_weight = self._guidance_weight()
        limit = max_swarms + int(np.ceil((1.0 - guidance_weight) * max_swarms / 2.0))
        return max(max_swarms, limit)

    def _guidance_weight(self) -> float:
        weight = float(getattr(self.SpatialModelState, "GuidanceWeight", 0.0))
        return min(1.0, max(0.0, weight))

    def _guidance_share(self) -> float:
        share = float(getattr(self.SpatialModelState, "GuidanceShare", 0.0))
        return min(1.0, max(0.0, share))

    def _apply_partition_budget(self, target_budget: int | float) -> None:
        budget = max(1, int(np.ceil(target_budget)))
        self.N_omega = budget
        if budget not in self.partitioner_cache:
            self.partitioner_cache[budget] = SPMPPartitioner(
                self.dimension,
                self.min_coordinate,
                self.max_coordinate,
                budget,
            )
        self.partitioner = self.partitioner_cache[budget]
        self.Partitioner = self.partitioner
        self.Subspaces = self.partitioner.subspaces

    def _record(self, positions: np.ndarray, fitness: np.ndarray, problem, fe: int | None = None) -> None:
        self.History, self.HistoryGridCoords = spmp_record_to_history(
            self.History,
            positions,
            fitness,
            int(problem.state.fe if fe is None else fe),
            self.dimension,
            self.HistoryWindowFE,
            self.HistoryGridCoords,
        )

    def _record_new_swarms(self, swarms, stamps: list[int], problem) -> None:
        for swarm, stamp in zip(swarms, stamps):
            self._record(swarm.pbest_positions, swarm.pbest_fitness, problem, stamp)

    def _generate_swarms(self, positions: np.ndarray, problem, category: str):
        return sub_population_generator_spmp(
            self.dimension,
            positions,
            problem,
            lambda points, count: self._evaluate(problem, points, count, category),
            defer_on_change=True,
        )

    def _evaluate(self, problem, points: np.ndarray, expected: int, category: str, swarm=None) -> np.ndarray:
        """Every paid evaluation goes through here; the monitor is told who spent it."""
        before = problem.state.fe
        values = evaluate_(problem, points, expected)
        if self.monitor is not None:
            self.monitor.charge(category, None if swarm is None else swarm.uid, problem.state.fe - before)
        return values

    def _count(self, event: str, amount: int = 1) -> None:
        if self.monitor is not None:
            self.monitor.count(event, amount)

    def _refresh_radius(self, swarm: SPMPSwarm) -> None:
        if swarm.pbest_positions.shape[0] == 0:
            swarm.center = np.zeros(self.dimension, dtype=float)
            swarm.current_radius = 0.0
            return
        swarm.center = np.mean(swarm.pbest_positions, axis=0)
        swarm.current_radius = float(
            np.mean(np.linalg.norm(swarm.pbest_positions - swarm.center, axis=1))
        )

    def _update_personal_bests(self, swarm: SPMPSwarm) -> None:
        improved = swarm.fitness > swarm.pbest_fitness
        if np.any(improved):
            swarm.pbest_fitness[improved] = swarm.fitness[improved]
            swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        self._reset_best(swarm)

    def _reset_best(self, swarm: SPMPSwarm) -> None:
        best_id = _best_index(swarm.pbest_fitness)
        swarm.best_id = best_id
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()
        swarm.best_fitness = float(swarm.pbest_fitness[best_id])

    def _activate_swarm(self, index: int) -> None:
        self.pop[index].is_hibernated = 0
        self.pop[index].stagnating_count = 0
        self.pop[index].is_stagnated = 0

    def _delete_swarm(self, index: int) -> None:
        del self.pop[index]
        self._sync_swarm_number("safeguard")

    def _update_memory(self) -> None:
        record = self._map_record_for_count(self.SwarmNumber)
        record.num += 1
        value = float(self.iter_individuals_begin[-1]) if self.iter_individuals_begin else 0.0
        record.indivs.append(value)
        record.mean = ((record.mean * (record.num - 1)) + value) / record.num
        samples = np.asarray(record.indivs, dtype=float)
        record.std_dev = float(np.sqrt(np.mean((samples - record.mean) ** 2)))

    def _map_record_for_count(self, swarm_count: int) -> SPMPMapRecord:
        index = max(1, int(swarm_count)) - 1
        while index >= len(self.map_pop_inds):
            self.map_pop_inds.append(SPMPMapRecord())
        self.MapPopInds = self.map_pop_inds
        return self.map_pop_inds[index]

    def _swarm_in_any_boa(self, particle_subs: np.ndarray) -> bool:
        sub_ids = set(np.asarray(particle_subs, dtype=int).tolist())
        if self.SubspaceAnyBoA.size:
            valid = [sub_id for sub_id in sub_ids if 0 < sub_id <= self.SubspaceAnyBoA.size]
            if valid and bool(np.any(self.SubspaceAnyBoA[np.asarray(valid, dtype=int) - 1])):
                return True
        for boa in self.BoAs:
            if sub_ids & set(np.asarray(boa.allSubspaces, dtype=int).tolist()):
                return True
        return False

    def _rank_boa_summaries(self, summaries: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(
            summaries,
            key=lambda item: (
                float(item["confidence"]),
                int(item["sampleCount"]),
                float(item["modeScore"]),
            ),
            reverse=True,
        )

    def _mark_ghost_dispatch_skipped(self, reason: str) -> None:
        diag = self._default_ghost_diagnostics()
        diag["TotalBoAs"] = len(self.BoAs)
        diag["Reason"] = reason
        self.LastGhostDiagnostics = diag

    def _collect_swarm_type_diagnostics(self, removed_ghost_count: int) -> dict[str, object]:
        diag = self._default_swarm_type_diagnostics()
        diag["TotalSwarms"] = self.SwarmNumber
        diag["RemovedGhostCount"] = int(removed_ghost_count)
        for swarm in self.pop:
            if swarm.swarm_type == "Exploitative":
                diag["ExploitativeCount"] += 1
            elif swarm.swarm_type == "Exploratory":
                diag["ExploratoryCount"] += 1
            elif swarm.swarm_type == "Ghost":
                diag["GhostCount"] += 1
        return diag

    def _sync_swarm_number(self, cause: str = "") -> None:
        self.SwarmNumber = len(self.pop)
        self.swarm_number = self.SwarmNumber
        for swarm in self.pop:
            if swarm.uid < 0:
                self._adopt(swarm)
        if self.monitor is not None:
            self.monitor.sync(self.pop, cause)

    def _adopt(self, swarm: SPMPSwarm) -> None:
        """Bookkeeping for a new swarm: identity, and the cloud's step size at the swarm's spread per coordinate."""
        swarm.uid = self.next_uid
        self.next_uid += 1
        swarm.cloud_radius = float(np.clip(
            swarm.current_radius / np.sqrt(self.dimension), self.cloud_radius_min, self.cloud_radius_max
        ))
        swarm.success_rate = self.success_target

    def run_once(self, *args, **kwargs):
        result = super().run_once(*args, **kwargs)
        if self.monitor is not None:
            self.monitor.finish(self.problem)
            result.raw.setdefault("event_streams", {})[MONITOR_STREAM] = dict(
                schema_version=1, events=self.monitor.records
            )
        return result

    def _default_spatial_model_state(self) -> SPMPSpatialModelState:
        return SPMPSpatialModelState()

    def _default_boa_diagnostics(self) -> dict[str, object]:
        return {
            "NumSubspaces": 0,
            "SampledSubspaceCount": 0,
            "UnsampledSubspaceCount": 0,
            "RawBasinCount": 0,
            "MergedClusterCount": 0,
            "MergedAwayCount": 0,
            "AcceptedBoACount": 0,
            "RejectedClusterCount": 0,
            "OccupiedAcceptedCount": 0,
            "GhostAcceptedCount": 0,
            "ReliableAcceptedCount": 0,
            "OverlapSubspaceCount": 0,
            "OverlapSubspaceRatio": 1.0,
            "MeanSampleSupport": 0.0,
            "MaxSampleSupport": 0,
            "ScoreScale": 0.0,
            "PersistenceThreshold": 0.0,
            "ConfidenceThreshold": 0.0,
            "TopBoAs": [],
            "TopRejectedBoAs": [],
        }

    def _default_swarm_type_diagnostics(self) -> dict[str, object]:
        return {
            "TotalSwarms": 0,
            "ExploitativeCount": 0,
            "ExploratoryCount": 0,
            "GhostCount": 0,
            "RemovedGhostCount": 0,
            "MultiLabelSwarmCount": 0,
            "MixedOverlapSwarmCount": 0,
            "OverlapOnlySwarmCount": 0,
            "PureSingleBoASwarmCount": 0,
            "TopMixedSwarms": [],
        }

    def _default_ghost_diagnostics(self) -> dict[str, object]:
        return {
            "TotalBoAs": 0,
            "EligibleCount": 0,
            "DispatchedCount": 0,
            "CreatedSwarmCount": 0,
            "SelectedBoAIndices": [],
            "InterruptedByChange": False,
            "Reason": "",
            "CandidatePreview": [],
        }
