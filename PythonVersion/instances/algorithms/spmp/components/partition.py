"""SPMP space partition state and face adjacency."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utility.space_partition import SpacePartitioner


class SPMPPartitioner(SpacePartitioner):
    """Space partitioner extended with SPMP subspace state."""

    @dataclass(slots=True)
    class Subspace(SpacePartitioner.Subspace):
        isOverlap: bool = True
        uncertainty: float = 1.0
        sols: list[int] = field(default_factory=list)
        F_i: float = 0.0
        S_i: int = 0
        C_i: float = 0.0

        @property
        def vol(self) -> float:
            return float(self.volume)

    def face_neighbors(self) -> dict[int, list[int]]:
        """Neighbours sharing a (D-1)-dimensional face, by subspace index; built once per partition.

        The partitioner's neighbours are closed boxes that touch anywhere, corners included. In a coarse partition
        of five or more dimensions that links most cells to each other (at D = 20 and 2048 cells, every cell to every
        other), so steepest ascent over cells finds one or two basins. Two cells of the partition never overlap in
        volume, so a neighbour shares a face exactly when they touch in one coordinate only.
        """
        cached = getattr(self, "_face_neighbors", None)
        if cached is None:
            mins = np.vstack([subspace.min for subspace in self.subspaces])
            maxs = np.vstack([subspace.max for subspace in self.subspaces])
            tolerance = 1e-9 * (self.upper - self.lower)
            cached = {}
            for subspace in self.subspaces:
                ids = np.asarray(subspace.neighbors, dtype=int)
                if ids.size == 0:
                    cached[subspace.index] = []
                    continue
                overlap = np.minimum(subspace.max, maxs[ids - 1]) - np.maximum(subspace.min, mins[ids - 1])
                touching = np.sum(overlap <= tolerance, axis=1)
                cached[subspace.index] = ids[touching == 1].tolist()
            self._face_neighbors = cached
        return cached


SPMPSubspace = SPMPPartitioner.Subspace

__all__ = ["SPMPPartitioner", "SPMPSubspace"]
