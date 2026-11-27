"""RAPSO algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import (
    evaluate_,
    stack_rows,
    update_center,
    update_current_radius,
)
from utility.clustering import SingleLinkageClusterer
from instances.algorithms.rapso.state import RAPSOHiberArea, RAPSOMapRecord, RAPSOSwarm


class RAPSOSwarmFactory:
    """Build RAPSO sub-swarms from sampled positions."""

    def __init__(self, algorithm: "RAPSOAlgorithm") -> None:
        self.algorithm = algorithm

    def __getattr__(self, name: str):
        return getattr(self.algorithm, name)

    def generate_subpopulations(
        self,
        positions: np.ndarray,
        problem,
    ) -> tuple[list[RAPSOSwarm], np.ndarray]:
        positions = np.asarray(positions, dtype=float).reshape(-1, self.dimension)
        groups, remain = SingleLinkageClusterer().groups_and_remain(positions)
        swarms: list[RAPSOSwarm] = []
        remain_rows = [row for row in remain]

        for group in groups:
            if group.shape[0] == 0:
                continue
            values = evaluate_(problem, group, group.shape[0])
            if values.size != group.shape[0] or np.all(np.isnan(values)):
                remain_rows.extend(group)
                continue

            center = np.mean(group, axis=0)
            init_radius = float(np.mean(np.linalg.norm(group - center, axis=1)))
            velocity = np.zeros_like(group)
            if problem.state.recent_change:
                values = np.full(group.shape[0], -np.inf, dtype=float)
            pbest_value = values.copy()
            best_id = int(np.argmax(np.nan_to_num(pbest_value, nan=-np.inf)))
            swarms.append(
                RAPSOSwarm(
                    positions=group.copy(),
                    gbest_past_environment=np.full(self.dimension, np.nan),
                    velocities=velocity,
                    shifts=[],
                    fitness=values,
                    pbest_positions=group.copy(),
                    is_hibernated=0,
                    stagnating_count=0,
                    is_stagnated=0,
                    pbest_fitness=pbest_value,
                    best_fitness=float(pbest_value[best_id]),
                    best_position=group[best_id, :].copy(),
                    best_id=best_id,
                    center=center,
                    init_radius=init_radius,
                    current_radius=init_radius,
                    ucb_pull_count=0,
                    ucb_reward_mean=0.0,
                    ucb_idle_rounds=0,
                )
            )
        return swarms, stack_rows(remain_rows, self.dimension)


@algorithm_registry.register("RAPSO")
class RAPSOAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "RAPSO"

    def __init__(self) -> None:
        self.swarm_factory = RAPSOSwarmFactory(self)

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "initPopulationSize": ParameterSpec(
                "initPopulationSize",
                100,
                "integer",
                (0, 1000),
                "Total number of individuals at initialization.",
            ),
            "w": ParameterSpec(
                "w",
                0.7298,
                "numeric",
                (0, 2),
                "Inertia weight balancing exploration and exploitation.",
            ),
            "c1": ParameterSpec(
                "c1",
                1.496,
                "numeric",
                (0, 5),
                "Cognitive acceleration coefficient scaling influence of personal best.",
            ),
            "c2": ParameterSpec(
                "c2",
                1.496,
                "numeric",
                (0, 5),
                "Social acceleration coefficient scaling influence of global best.",
            ),
            "ConvergenceLimit": ParameterSpec(
                "ConvergenceLimit",
                0.00001,
                "numeric",
                (0, 0.1),
                "Sub-swarm radius below which it is considered converged (hibernated).",
            ),
            "OverlapDegree": ParameterSpec(
                "OverlapDegree",
                0.1,
                "numeric",
                (0, 1),
                "Proportion of individuals that must overlap to trigger sub-swarm removal.",
            ),
            "diversityFactor": ParameterSpec(
                "diversityFactor",
                0.005,
                "numeric",
                (0, 1),
                "Configurable in MATLAB; RAPSO source still uses literal 0.005 in diversity checks.",
            ),
            "StepIndivs": ParameterSpec(
                "StepIndivs",
                10,
                "integer",
                (0, 100),
                "Base number of individuals used as a unit when increasing diversity.",
            ),
            "OpportunitySchedulingMode": ParameterSpec(
                "OpportunitySchedulingMode",
                "Origin",
                "option",
                ("Origin", "UCB"),
                "Swarm evolution-opportunity scheduling mode.",
            ),
            "EnableOpportunityDiagnostics": ParameterSpec(
                "EnableOpportunityDiagnostics",
                False,
                "boolean",
                (0, 1),
                "Record per-round evolution-opportunity diagnostics.",
            ),
            "UCBExplorationC": ParameterSpec(
                "UCBExplorationC",
                0.8,
                "numeric",
                (0, 5),
                "Exploration coefficient c in UCB scheduling.",
            ),
            "MaxIdleRounds": ParameterSpec(
                "MaxIdleRounds",
                3,
                "integer",
                (1, 20),
                "Maximum active idle rounds before starvation protection schedules a swarm.",
            ),
        }

    def initialize(self, problem) -> None:
        self.dimension = problem.dimension
        self.bounds = problem.bounds
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_indivs = 1000
        self.min_indivs = 10
        self.shift_severity = 1.0
        self.swarm_number = 0
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
        self.ucb_total_pulls = 0
        self.domain_size = float(np.linalg.norm(self.max_coordinate - self.min_coordinate))

        init_size = max(0, int(self.parameters["initPopulationSize"]))
        init_x = self.bounds.sample(init_size, self.rng)
        self.pop, self.remain_indivs = self.swarm_factory.generate_subpopulations(
            init_x,
            problem,
        )
        self.swarm_number = len(self.pop)
        self.map_pop_inds = [RAPSOMapRecord() for _ in range(max(self.swarm_number, 1))]
        self.hiber_area: list[RAPSOHiberArea] = []
        self.initialize_opportunity_diagnostics()

    def initialize_opportunity_diagnostics(self) -> None:
        self.opportunity_round_counter = 0
        if not bool(self.parameters.get("EnableOpportunityDiagnostics", False)):
            self.opportunity_diagnostics: dict[str, object] | list[object] = []
            return
        self.opportunity_diagnostics = {
            "mode": "opportunity_behavior_minimal",
            "schema_version": "v1",
            "scheduling_mode": self.parameters.get("OpportunitySchedulingMode", "Origin"),
            "snapshots": [],
        }

    def iterate(self, problem) -> None:
        if problem.state.recent_change or problem.state.terminated:
            return
        self.domain_size = float(np.linalg.norm(self.max_coordinate - self.min_coordinate))
        self.ensure_ucb_defaults()
        self.opportunity_round_counter += 1
        round_counter = self.opportunity_round_counter
        fe_round_start = problem.state.fe
        active_at_start = [
            index for index, swarm in enumerate(self.pop) if swarm.is_hibernated == 0
        ]
        idle_before = [self.pop[index].ucb_idle_rounds for index in active_at_start]
        pull_before = [self.pop[index].ucb_pull_count for index in active_at_start]
        exec_counts = [0 for _ in range(self.swarm_number)]
        rewards: list[float] = []
        guard_swarms: list[int] = []
        schedule: list[int] = []
        round_recent_change = False
        skipped_idle_update = False

        if self.uses_ucb_scheduling():
            for _ in range(len(active_at_start)):
                selected, used_guard = self.select_next_ucb_swarm(active_at_start, exec_counts)
                if selected < 0:
                    break
                schedule.append(selected)
                if used_guard:
                    guard_swarms.append(selected)
                reward = self.evolve_one_swarm(problem, selected)
                if selected < len(exec_counts):
                    exec_counts[selected] += 1
                rewards.append(reward)
                self.update_ucb_after_pull(selected, reward)
                if problem.state.recent_change or problem.state.terminated:
                    round_recent_change = bool(problem.state.recent_change)
                    skipped_idle_update = round_recent_change
                    break
            if not round_recent_change and not problem.state.terminated:
                self.update_ucb_idle_rounds(active_at_start, exec_counts)
        else:
            schedule = list(range(self.swarm_number))
            for index in range(self.swarm_number):
                reward = self.evolve_one_swarm(problem, index)
                if index < len(exec_counts):
                    exec_counts[index] += 1
                rewards.append(reward)
                if problem.state.recent_change or problem.state.terminated:
                    round_recent_change = bool(problem.state.recent_change)
                    break

        self.append_opportunity_diag_snapshot(
            problem,
            round_counter,
            fe_round_start,
            active_at_start,
            schedule,
            guard_swarms,
            idle_before,
            pull_before,
            exec_counts,
            rewards,
            round_recent_change,
            skipped_idle_update,
        )
        if round_recent_change or problem.state.terminated:
            return

        self._check_stagnant()
        self._check_overlapping()
        self._check_diversity(problem)

    def on_environment_change(self, problem) -> None:
        for swarm in self.pop:
            values = evaluate_(problem, swarm.positions, swarm.positions.shape[0])
            if problem.state.recent_change or problem.state.terminated:
                return
            if values.size != swarm.positions.shape[0] or np.all(np.isnan(values)):
                return
            swarm.fitness = values
            swarm.pbest_fitness = values.copy()
            swarm.pbest_positions = swarm.positions.copy()
            swarm.gbest_past_environment = swarm.best_position.copy()
            self._reset_best(swarm)
            swarm.velocities = -1 + 2 * self.rng.random(
                (swarm.positions.shape[0], self.dimension)
            )
            swarm.ucb_pull_count = 0
            swarm.ucb_reward_mean = 0.0
            swarm.ucb_idle_rounds = 0
        self.ucb_total_pulls = 0

    def ensure_ucb_defaults(self) -> None:
        self.parameters.setdefault("OpportunitySchedulingMode", "Origin")
        self.parameters.setdefault("EnableOpportunityDiagnostics", False)
        self.parameters.setdefault("UCBExplorationC", 0.8)
        self.parameters.setdefault("MaxIdleRounds", 3)
        if not hasattr(self, "opportunity_round_counter"):
            self.opportunity_round_counter = 0
        if not hasattr(self, "ucb_total_pulls"):
            self.ucb_total_pulls = 0
        for swarm in self.pop:
            if swarm.ucb_pull_count is None:
                swarm.ucb_pull_count = 0
            if swarm.ucb_reward_mean is None:
                swarm.ucb_reward_mean = 0.0
            if swarm.ucb_idle_rounds is None:
                swarm.ucb_idle_rounds = 0

    def uses_ucb_scheduling(self) -> bool:
        mode = str(self.parameters.get("OpportunitySchedulingMode", "Origin")).strip()
        return mode.lower() == "ucb"

    def compute_ucb_score(self, swarm: RAPSOSwarm) -> float:
        return float(
            swarm.ucb_reward_mean
            + float(self.parameters.get("UCBExplorationC", 0.8))
            * np.sqrt(np.log(self.ucb_total_pulls + 1) / (swarm.ucb_pull_count + 1))
        )

    def select_next_ucb_swarm(
        self,
        active_swarms_at_round_start: list[int] | None = None,
        exec_counts: list[int] | None = None,
    ) -> tuple[int, bool]:
        if active_swarms_at_round_start is None:
            active_swarms_at_round_start = [
                index for index, swarm in enumerate(self.pop) if swarm.is_hibernated == 0
            ]
        if exec_counts is None:
            exec_counts = [0 for _ in range(self.swarm_number)]

        candidates = [
            index
            for index in active_swarms_at_round_start
            if 0 <= index < len(self.pop)
            and index < len(exec_counts)
            and exec_counts[index] == 0
            and self.pop[index].is_hibernated == 0
        ]
        if not candidates:
            return -1, False

        max_idle_rounds = max(1, round(float(self.parameters.get("MaxIdleRounds", 3))))
        starved = [
            index
            for index in candidates
            if self.pop[index].ucb_idle_rounds >= max_idle_rounds
        ]
        if starved:
            return max(
                starved,
                key=lambda index: (self.pop[index].ucb_idle_rounds, self.compute_ucb_score(self.pop[index])),
            ), True
        return max(candidates, key=lambda index: self.compute_ucb_score(self.pop[index])), False

    def update_ucb_after_pull(self, swarm_index: int, reward: float) -> None:
        if swarm_index < 0 or swarm_index >= len(self.pop):
            return
        swarm = self.pop[swarm_index]
        pull_count = int(swarm.ucb_pull_count)
        new_count = pull_count + 1
        swarm.ucb_pull_count = new_count
        swarm.ucb_reward_mean = float(
            swarm.ucb_reward_mean + (reward - swarm.ucb_reward_mean) / new_count
        )
        swarm.ucb_idle_rounds = 0
        self.ucb_total_pulls += 1

    def update_ucb_idle_rounds(
        self,
        active_swarms_at_round_start: list[int],
        exec_counts: list[int],
    ) -> None:
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                swarm.ucb_idle_rounds = 0
        for index in active_swarms_at_round_start:
            if index >= len(self.pop) or self.pop[index].is_hibernated == 1:
                continue
            if index < len(exec_counts) and exec_counts[index] > 0:
                self.pop[index].ucb_idle_rounds = 0
            else:
                self.pop[index].ucb_idle_rounds += 1

    def append_opportunity_diag_snapshot(
        self,
        problem,
        round_counter: int,
        fe_round_start: int,
        active_swarms_at_round_start: list[int],
        schedule: list[int],
        guard_swarms: list[int],
        idle_before: list[int],
        pull_before: list[int],
        exec_counts: list[int],
        rewards: list[float],
        round_recent_change: bool,
        skipped_idle_update: bool,
    ) -> None:
        if not bool(self.parameters.get("EnableOpportunityDiagnostics", False)):
            return
        if not isinstance(getattr(self, "opportunity_diagnostics", None), dict):
            self.initialize_opportunity_diagnostics()
        active_exec_counts = [
            exec_counts[index] for index in active_swarms_at_round_start if index < len(exec_counts)
        ]
        max_exec = max(active_exec_counts) if active_exec_counts else 0
        total_exec = sum(active_exec_counts)
        snapshot = {
            "round_counter": round_counter,
            "environment": getattr(problem.state, "environment_index", 0),
            "fe_start": fe_round_start,
            "fe_end": problem.state.fe,
            "active_swarm_count": len(active_swarms_at_round_start),
            "pull_count": len(rewards),
            "unique_scheduled_active": sum(count > 0 for count in active_exec_counts),
            "zero_scheduled_active": sum(count == 0 for count in active_exec_counts),
            "max_exec_count": max_exec,
            "max_exec_share": max_exec / max(total_exec, 1),
            "guard_count": len(guard_swarms),
            "positive_reward_count": sum(reward > 0 for reward in rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "idle_mean_before": float(np.mean(idle_before)) if idle_before else 0.0,
            "idle_max_before": max(idle_before) if idle_before else 0,
            "pull_mean_before": float(np.mean(pull_before)) if pull_before else 0.0,
            "pull_max_before": max(pull_before) if pull_before else 0,
            "recent_change_triggered": bool(round_recent_change),
            "skipped_idle_update": bool(skipped_idle_update),
            "schedule": list(schedule),
            "guard_swarms": list(guard_swarms),
            "reward_per_pull": [float(reward) for reward in rewards],
            "exec_counts": active_exec_counts,
        }
        self.opportunity_diagnostics["snapshots"].append(snapshot)

    def evolve_one_swarm(self, problem, swarm_index: int) -> float:
        if swarm_index < 0 or swarm_index >= self.swarm_number:
            return 0.0
        swarm = self.pop[swarm_index]
        gbest_before = self.safe_gbest_value(swarm)
        fe_before = problem.state.fe
        before_radius = swarm.current_radius

        if swarm.is_hibernated == 0:
            self._pso_move(swarm, problem)
            if problem.state.recent_change or problem.state.terminated:
                return self.compute_swarm_reward(
                    gbest_before,
                    self.safe_gbest_value(swarm),
                    problem.state.fe - fe_before,
                )

        self._refresh_radius(swarm)
        if swarm.current_radius / (before_radius + 0.000001) < 0.9:
            swarm.stagnating_count = 0
        else:
            swarm.stagnating_count += 1

        self._brownian_or_stagnated_move(swarm, problem)
        if problem.state.recent_change or problem.state.terminated:
            return self.compute_swarm_reward(
                gbest_before,
                self.safe_gbest_value(swarm),
                problem.state.fe - fe_before,
            )

        if swarm.current_radius < float(self.parameters["ConvergenceLimit"]):
            swarm.is_hibernated = 1
            swarm.is_stagnated = 0
            if not self.hiber_area:
                self.hiber_area.append(RAPSOHiberArea(optima=swarm.best_position.copy()))
        if swarm.is_hibernated == 1:
            swarm.init_radius = self.distance_accuracy

        return self.compute_swarm_reward(
            gbest_before,
            self.safe_gbest_value(swarm),
            problem.state.fe - fe_before,
        )

    def _pso_move(self, swarm: RAPSOSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        r1 = self.rng.random((rows, self.dimension))
        r2 = self.rng.random((rows, self.dimension))
        swarm.velocities = float(self.parameters["w"]) * (
            swarm.velocities
            + float(self.parameters["c1"]) * r1 * (swarm.pbest_positions - swarm.positions)
            + float(self.parameters["c2"])
            * r2
            * (np.broadcast_to(swarm.best_position, swarm.positions.shape) - swarm.positions)
        )
        swarm.positions = swarm.positions + swarm.velocities
        self._clip_positions_and_velocity(swarm.positions, swarm.velocities)
        values = evaluate_(problem, swarm.positions, rows)
        if problem.state.recent_change:
            return
        if values.size != rows:
            swarm.fitness[: values.size] = values
            return
        swarm.fitness = values
        self._update_personal_bests(swarm)

    def _brownian_or_stagnated_move(self, swarm: RAPSOSwarm, problem) -> None:
        rows = swarm.positions.shape[0]
        if rows == 0:
            return
        if swarm.is_stagnated == 0:
            for _ in range(2):
                best_id = int(swarm.best_id)
                randval = self.rng.standard_normal()
                order = np.argsort(-np.asarray(swarm.pbest_fitness, dtype=float))
                second_id = int(order[1]) if rows > 1 else int(order[0])
                for column in range(self.dimension):
                    newpos1 = (
                        swarm.pbest_positions[int(order[0]), column]
                        + self.rng.standard_normal() * swarm.current_radius
                    )
                    swarm.positions[best_id, column] = newpos1 + randval * (
                        swarm.pbest_positions[second_id, column] - newpos1
                    )
                    if swarm.positions[best_id, column] > self.max_coordinate[column]:
                        swarm.positions[best_id, column] = self.max_coordinate[column]
                        swarm.velocities[best_id, column] = 0.0
                    elif swarm.positions[best_id, column] < self.min_coordinate[column]:
                        swarm.positions[best_id, column] = self.min_coordinate[column]
                        swarm.velocities[best_id, column] = 0.0
                value = evaluate_(problem, swarm.positions[best_id, :], 1)
                if problem.state.recent_change:
                    return
                if not value.size or np.isnan(value[0]):
                    continue
                best_result = float(value[0])
                swarm.fitness[best_id] = best_result
                if best_result > swarm.pbest_fitness[best_id]:
                    swarm.pbest_fitness[best_id] = best_result
                    swarm.pbest_positions[best_id, :] = swarm.positions[best_id, :]
                    swarm.best_position = swarm.pbest_positions[best_id, :].copy()
                    swarm.best_fitness = best_result
        else:
            best_id = int(swarm.best_id)
            randval = self.rng.standard_normal()
            order = np.argsort(-np.asarray(swarm.pbest_fitness, dtype=float))
            second_id = int(order[1]) if rows > 1 else int(order[0])
            for column in range(self.dimension):
                newpos1 = (
                    swarm.pbest_positions[int(order[0]), column]
                    + self.rng.standard_normal() * swarm.current_radius
                )
                newpos2 = newpos1 + randval * (
                    swarm.pbest_positions[second_id, column] - newpos1
                )
                swarm.pbest_positions[best_id, column] = newpos2
                if swarm.pbest_positions[best_id, column] > self.max_coordinate[column]:
                    swarm.pbest_positions[best_id, column] = self.max_coordinate[column]
                    swarm.velocities[best_id, column] = 0.0
                elif swarm.pbest_positions[best_id, column] < self.min_coordinate[column]:
                    swarm.pbest_positions[best_id, column] = self.min_coordinate[column]
                    swarm.velocities[best_id, column] = 0.0
            value = evaluate_(problem, swarm.positions[best_id, :], 1)
            if problem.state.recent_change:
                return
            if not value.size or np.isnan(value[0]):
                return
            stagnated_best_result = float(value[0])
            swarm.pbest_fitness[best_id] = stagnated_best_result
            swarm.fitness[best_id] = stagnated_best_result
            swarm.positions[best_id, :] = swarm.pbest_positions[best_id, :]
            self._reset_best(swarm)
            swarm.stagnating_count = 0

    @staticmethod
    def safe_gbest_value(swarm: RAPSOSwarm) -> float:
        value = swarm.best_fitness
        if value is None or not np.isfinite(value):
            return -np.inf
        return float(value)

    @staticmethod
    def compute_swarm_reward(gbest_before: float, gbest_after: float, delta_fe: int) -> float:
        if not np.isfinite(gbest_before) or not np.isfinite(gbest_after):
            return 0.0
        return max(0.0, gbest_after - gbest_before) / max(int(delta_fe), 1)

    def _check_stagnant(self) -> None:
        active = [swarm.current_radius for swarm in self.pop if swarm.is_hibernated == 0]
        ave_radius = float(np.mean(active)) if active else 0.0
        diversity_limit = 0.005 * self.domain_size
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

    def _check_overlapping(self) -> None:
        idx = None
        while idx != -1:
            idx = -1
            for ii in range(self.swarm_number):
                if self.pop[ii].positions.shape[0] == 0:
                    continue
                for jj in range(ii + 1, self.swarm_number):
                    if self.pop[jj].positions.shape[0] == 0:
                        continue
                    distance = float(
                        np.linalg.norm(self.pop[ii].best_position - self.pop[jj].best_position)
                    )
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

    def _check_diversity(self, problem) -> None:
        active = [
            swarm.current_radius
            for swarm in self.pop
            if swarm.is_stagnated == 0 and swarm.is_hibernated == 0
        ]
        non_stagnated_radius = float(np.mean(active)) if active else 0.0
        diversity_limit = 0.005 * self.domain_size
        if self.swarm_number > 0 and non_stagnated_radius >= diversity_limit:
            return

        self.record_pop_info.append(self.swarm_number)
        for swarm in self.pop:
            if swarm.is_hibernated == 1:
                swarm.is_hibernated = 0
        self.hiber_area = []

        self.iter_pop_size_end.append(self.swarm_number)
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
        if num_add > 0:
            add = self.bounds.sample(num_add, self.rng)
            add = np.vstack([add, self.remain_indivs]) if num_remain else add
            self._add_generated_swarms(add, problem)
        elif self.swarm_number == 0 and num_remain > 0:
            add = np.vstack(
                [self.bounds.sample(self.min_indivs, self.rng), self.remain_indivs]
            )
            self._add_generated_swarms(add, problem)

        self.adjust_iter += 1

    def _add_generated_swarms(self, positions: np.ndarray, problem) -> None:
        self.remain_indivs = np.empty((0, self.dimension))
        swarms, remain = self.swarm_factory.generate_subpopulations(positions, problem)
        self.remain_indivs = remain
        if problem.state.recent_change:
            return
        self.pop.extend(swarms)
        self.swarm_number = len(self.pop)

    def _update_memory(self) -> None:
        record = self._map_record_for_count(self.swarm_number)
        record.num += 1
        value = float(self.iter_individuals_begin[-1]) if self.iter_individuals_begin else 0.0
        record.indivs.append(value)
        record.mean = ((record.mean * (record.num - 1)) + value) / record.num
        samples = np.asarray(record.indivs, dtype=float)
        record.std_dev = float(np.sqrt(np.mean((samples - record.mean) ** 2)))

    def _map_record_for_count(self, swarm_count: int) -> RAPSOMapRecord:
        index = max(1, int(swarm_count)) - 1
        while index >= len(self.map_pop_inds):
            self.map_pop_inds.append(RAPSOMapRecord())
        return self.map_pop_inds[index]

    def _activate_swarm(self, index: int) -> None:
        self.pop[index].is_hibernated = 0
        self.pop[index].stagnating_count = 0
        self.pop[index].is_stagnated = 0

    def _delete_swarm(self, index: int) -> None:
        del self.pop[index]
        self.swarm_number = len(self.pop)

    def _refresh_radius(self, swarm: RAPSOSwarm) -> None:
        if swarm.positions.shape[0] == 0:
            swarm.center = np.zeros(self.dimension)
            swarm.current_radius = 0.0
            return
        swarm.center = update_center(swarm)
        swarm.current_radius = update_current_radius(swarm)

    def _update_personal_bests(self, swarm: RAPSOSwarm) -> None:
        improved = swarm.fitness > swarm.pbest_fitness
        if np.any(improved):
            swarm.pbest_fitness[improved] = swarm.fitness[improved]
            swarm.pbest_positions[improved, :] = swarm.positions[improved, :]
        self._reset_best(swarm)

    def _reset_best(self, swarm: RAPSOSwarm) -> None:
        fitness = np.asarray(swarm.pbest_fitness, dtype=float)
        best_id = int(np.argmax(np.nan_to_num(fitness, nan=-np.inf)))
        swarm.best_id = best_id
        swarm.best_position = swarm.pbest_positions[best_id, :].copy()
        swarm.best_fitness = float(fitness[best_id])

    def _clip_positions_and_velocity(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        high = positions > self.max_coordinate
        low = positions < self.min_coordinate
        if np.any(high):
            positions[high] = np.broadcast_to(self.max_coordinate, positions.shape)[high]
            velocities[high] = 0.0
        if np.any(low):
            positions[low] = np.broadcast_to(self.min_coordinate, positions.shape)[low]
            velocities[low] = 0.0
