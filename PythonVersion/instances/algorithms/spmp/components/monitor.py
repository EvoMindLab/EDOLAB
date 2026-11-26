"""Passive monitor of SPMP: what the algorithm holds and where its evaluations go.

The monitor only reads the algorithm's state and the problem's FE counter. It never evaluates, never draws from the
algorithm's RNG and never writes to the search state, so switching it off leaves every run bit-identical.

Two products:
  records    one dict per environment (the event stream ``spmp.monitor``), closed when the algorithm learns
             of the change, before its change response, and for the last environment at the end of the run;
  snapshots  the state at GRID points of every environment (the first iteration end at or after each point), from
             which the FE-axis trends are built; each record also carries its environment's points (``grid``).

Evaluation categories: init (initial populations), replenish (populations created when diversity is low), ghost
(populations dispatched to unoccupied basins), pso (particle moves), local (the cloud around the swarm best) and change (the change response). The last three are
also charged to the population that spent them, which gives the allocation among populations.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

MONITOR_STREAM = "spmp.monitor"
GRID = 20
CATEGORIES = ("init", "replenish", "ghost", "pso", "local", "change")
TRACKER_CATEGORIES = ("pso", "local", "change")
GRID_FIELDS = ("populations", "active", "inactive", "subspaces", "basins", "basins_ghost", "guided", "cloud_radius")


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def gini(values: np.ndarray) -> float:
    """Gini coefficient of nonnegative shares (0: equal, towards 1: one population takes everything)."""
    data = np.sort(np.asarray(values, dtype=float))
    if data.size == 0 or data.sum() <= 0:
        return 0.0
    ranks = np.arange(1, data.size + 1)
    return float((2.0 * np.sum(ranks * data) / (data.size * data.sum())) - (data.size + 1.0) / data.size)


class SPMPMonitor:
    def __init__(self, algorithm, problem) -> None:
        self.algorithm = algorithm
        self.frequency = int(problem.change_frequency)
        self.environments = int(problem.environment_number)
        self.records: list[dict] = []
        self.snapshots: list[tuple[int, dict]] = []
        self.known: set[int] = set()
        self.finished = False
        self._open(0)

    def _open(self, start_fe: int) -> None:
        self.start_fe = int(start_fe)
        self.fe_by: Counter = Counter()
        self.fe_by_uid: Counter = Counter()
        self.created: Counter = Counter()
        self.removed: Counter = Counter()
        self.removed_uids: set[int] = set()
        self.events: Counter = Counter()
        self.samples: list[dict] = []
        self.grid: list[dict] = []
        self.next_grid = 1
        self.restart: dict = {}

    # ---------------------------------------------------------------- charged by the algorithm
    def charge(self, category: str, uid: int | None, fes: int) -> None:
        if fes <= 0:
            return
        self.fe_by[category] += int(fes)
        if uid is not None and category in TRACKER_CATEGORIES:
            self.fe_by_uid[int(uid)] += int(fes)

    def count(self, event: str, amount: int = 1) -> None:
        self.events[event] += int(amount)

    def note_restart(self, shift_estimate: float, observations: int, radii: list[float]) -> None:
        self.restart = dict(shift_estimate=_round(shift_estimate), shift_observations=int(observations),
                            restart_radius=_round(float(np.median(radii))) if radii else None)

    def sync(self, population, cause: str) -> None:
        """Called after every change of the population list; attributes births and deaths to their cause."""
        current = {swarm.uid for swarm in population}
        born = current - self.known
        died = self.known - current
        if born:
            self.created[cause] += len(born)
        if died:
            self.removed[cause] += len(died)
            self.removed_uids |= died
        self.known = current

    # ---------------------------------------------------------------- read from the algorithm
    def _state(self) -> dict:
        algorithm = self.algorithm
        pop = algorithm.pop
        active = [swarm for swarm in pop if swarm.is_hibernated == 0]
        types = Counter(swarm.swarm_type for swarm in pop)
        diag = algorithm.LastBoADiagnostics
        boas = algorithm.BoAs
        cloud = [swarm.cloud_radius for swarm in active]
        return dict(
            populations=len(pop), active=len(active), inactive=len(pop) - len(active),
            stagnated=sum(int(swarm.is_stagnated) for swarm in pop),
            individuals=int(sum(swarm.positions.shape[0] for swarm in pop)),
            exploitative=types.get("Exploitative", 0), exploratory=types.get("Exploratory", 0),
            ghost_populations=types.get("Ghost", 0),
            subspaces=len(algorithm.Subspaces), basins=len(boas),
            basins_occupied=sum(not boa.isGhost for boa in boas), basins_ghost=sum(bool(boa.isGhost) for boa in boas),
            raw_basins=int(diag.get("RawBasinCount", 0)), sampled_cells=int(diag.get("SampledSubspaceCount", 0)),
            guided=int(algorithm.SpatialModelState.Mode == "guided"),
            cloud_radius=float(np.median(cloud)) if cloud else None,
        )

    def snapshot(self, problem) -> None:
        """At the end of an iteration: one sample for time averages, and the FE-grid points passed since the last."""
        state = self._state()
        self.samples.append(state)
        local = int(problem.state.fe) - self.start_fe
        while self.next_grid <= GRID and local >= self.next_grid * self.frequency / GRID:
            self._grid_point(state)

    def _grid_point(self, state: dict) -> None:
        self.snapshots.append((self.start_fe + int(round(self.next_grid * self.frequency / GRID)), state))
        self.grid.append(state)
        self.next_grid += 1

    def close_environment(self, problem) -> None:
        state = self._state()
        while self.next_grid <= GRID:
            self._grid_point(state)
        tracker = {uid: fes for uid, fes in self.fe_by_uid.items()}
        spent = np.asarray(sorted(tracker.values(), reverse=True), dtype=float)
        total_tracker = float(spent.sum())
        total = int(sum(self.fe_by.values()))
        samples = self.samples or [state]

        def mean(key):
            values = [sample[key] for sample in samples if sample[key] is not None]
            return _round(float(np.mean(values))) if values else None

        record = dict(
            e=len(self.records) + 1, fe=int(problem.state.fe), fe_total=total,
            fe_by={category: int(self.fe_by.get(category, 0)) for category in CATEGORIES},
            # Allocation of the tracker evaluations (particle moves, local move, change response) among populations.
            funded=int(spent.size),
            top1=_round(spent[0] / total_tracker) if total_tracker > 0 else None,
            top3=_round(spent[:3].sum() / total_tracker) if total_tracker > 0 else None,
            gini=_round(gini(spent)) if spent.size else None,
            fe_removed=int(sum(fes for uid, fes in tracker.items() if uid in self.removed_uids)),
            created=dict(self.created), removed=dict(self.removed), events=dict(self.events),
            iterations=len(self.samples),
            end={key: (_round(value) if isinstance(value, float) else value) for key, value in state.items()},
            mean={key: mean(key) for key in ("populations", "active", "inactive", "subspaces", "basins",
                                               "basins_ghost", "guided", "cloud_radius")},
            # The state at the GRID points of the environment (every 1/GRID of its budget).
            grid={key: [point[key] if not isinstance(point[key], float) else _round(point[key], 4)
                        for point in self.grid] for key in GRID_FIELDS},
            **self.restart,
        )
        self.records.append(record)
        self._open(int(problem.state.fe))

    def finish(self, problem) -> None:
        if not self.finished:
            self.finished = True
            if int(problem.state.fe) > self.start_fe:
                self.close_environment(problem)


__all__ = ["CATEGORIES", "GRID", "MONITOR_STREAM", "SPMPMonitor", "gini"]
