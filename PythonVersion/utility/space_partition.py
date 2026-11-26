"""Object-oriented axis-aligned space partitioning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SpacePartitioner:
    """Recursive binary partitioner with stateful axis-aligned subspaces."""

    _VECTOR_ADJACENCY_THRESHOLD = 4096
    _VECTOR_ADJACENCY_BLOCK_SIZE = 512

    @dataclass(slots=True)
    class Subspace:
        index: int
        bounds: np.ndarray
        min: np.ndarray
        max: np.ndarray
        volume: float
        center: np.ndarray
        neighbors: list[int] = field(default_factory=list)

    @dataclass(slots=True)
    class _Node:
        bounds: np.ndarray
        depth: int
        index: int = 0
        leaf_count: int = 1
        split_dim: int = -1
        split_point: float = 0.0
        left: "SpacePartitioner._Node | None" = None
        right: "SpacePartitioner._Node | None" = None

        @property
        def is_leaf(self) -> bool:
            return self.left is None and self.right is None

    def __init__(
        self,
        dimension: int,
        lower: np.ndarray | float,
        upper: np.ndarray | float,
        partition_count: int,
    ) -> None:
        self.dimension = int(dimension)
        self.partition_count = max(1, int(partition_count))
        self.lower = self._coerce_bound(lower)
        self.upper = self._coerce_bound(upper)
        self.total_volume = float(np.prod(self.upper - self.lower))
        self._next_index = 0
        self.tree = self._divide(np.vstack([self.lower, self.upper]), self.partition_count, 0)
        self.subspaces: list[SpacePartitioner.Subspace] = []
        self._collect_subspaces(self.tree)
        self._by_index = {subspace.index: subspace for subspace in self.subspaces}
        self.adjacency_candidate_checks = 0
        self._compute_all_adjacencies()

    def locate(self, point: np.ndarray) -> np.ndarray | int:
        points = np.asarray(point, dtype=float)
        if points.ndim == 1:
            return int(self._locate_one(points, self.tree))
        return np.asarray([self._locate_one(row, self.tree) for row in points], dtype=int)

    def are_adjacent(self, idx1: int, idx2: int) -> bool:
        left = self._by_index[int(idx1)]
        right = self._by_index[int(idx2)]
        return self._bounds_overlap(left.min, left.max, right.min, right.max)

    def _coerce_bound(self, value: np.ndarray | float) -> np.ndarray:
        if np.asarray(value).ndim > 0:
            return np.asarray(value, dtype=float).reshape(self.dimension)
        return np.full(self.dimension, value, dtype=float)

    def _divide(
        self,
        bounds: np.ndarray,
        count: int,
        depth: int,
    ) -> _Node:
        if count <= 1:
            self._next_index += 1
            return self._Node(bounds=bounds.copy(), depth=depth, index=self._next_index)
        split_dim = depth % self.dimension
        if count % 2 == 1:
            left_count = count // 2
            right_count = count - left_count
            split_ratio = left_count / count
        else:
            left_count = right_count = count // 2
            split_ratio = 0.5
        split_point = bounds[0, split_dim] + (
            bounds[1, split_dim] - bounds[0, split_dim]
        ) * split_ratio
        left_bounds = bounds.copy()
        right_bounds = bounds.copy()
        left_bounds[1, split_dim] = split_point
        right_bounds[0, split_dim] = split_point
        return self._Node(
            bounds=bounds.copy(),
            depth=depth,
            leaf_count=int(left_count + right_count),
            split_dim=split_dim,
            split_point=float(split_point),
            left=self._divide(left_bounds, left_count, depth + 1),
            right=self._divide(right_bounds, right_count, depth + 1),
        )

    def _collect_subspaces(self, node: _Node) -> None:
        if node.is_leaf:
            mins = node.bounds[0].copy()
            maxs = node.bounds[1].copy()
            abs_volume = float(np.prod(maxs - mins))
            rel_volume = abs_volume / self.total_volume if self.total_volume > 0 else 0.0
            self.subspaces.append(
                self.Subspace(
                    index=node.index,
                    bounds=node.bounds.copy(),
                    min=mins,
                    max=maxs,
                    volume=rel_volume,
                    center=(mins + maxs) / 2.0,
                )
            )
            return
        assert node.left is not None and node.right is not None
        self._collect_subspaces(node.left)
        self._collect_subspaces(node.right)

    def _locate_one(self, point: np.ndarray, node: _Node) -> int:
        if np.any(point < self.lower - 1e-12) or np.any(point > self.upper + 1e-12):
            return 0
        if node.is_leaf:
            return int(node.index)
        assert node.left is not None and node.right is not None
        if point[node.split_dim] <= node.split_point:
            return self._locate_one(point, node.left)
        return self._locate_one(point, node.right)

    def _compute_all_adjacencies(self) -> None:
        for subspace in self.subspaces:
            subspace.neighbors = []
        self.adjacency_candidate_checks = 0
        if len(self.subspaces) >= self._VECTOR_ADJACENCY_THRESHOLD:
            self._compute_all_adjacencies_vectorized()
            return
        self._neighbor_sets: dict[int, set[int]] = {
            subspace.index: set() for subspace in self.subspaces
        }
        self._compute_adjacencies_from_tree(self.tree)
        for subspace in self.subspaces:
            subspace.neighbors = sorted(self._neighbor_sets[subspace.index])
        del self._neighbor_sets

    def _compute_all_adjacencies_vectorized(self) -> None:
        ids = np.asarray([subspace.index for subspace in self.subspaces], dtype=int)
        mins = np.vstack([subspace.min for subspace in self.subspaces])
        maxs = np.vstack([subspace.max for subspace in self.subspaces])
        count = ids.size
        neighbor_sets: dict[int, set[int]] = {int(index): set() for index in ids}
        columns = np.arange(count, dtype=int)
        block_size = min(self._VECTOR_ADJACENCY_BLOCK_SIZE, count)
        tolerance = 1e-9

        for start in range(0, count, block_size):
            end = min(start + block_size, count)
            overlap = np.ones((end - start, count), dtype=bool)
            left_min = mins[start:end]
            left_max = maxs[start:end]
            for dim in range(self.dimension):
                overlap &= left_max[:, dim, None] >= mins[None, :, dim] - tolerance
                overlap &= maxs[None, :, dim] >= left_min[:, dim, None] - tolerance
            overlap &= columns[None, :] > np.arange(start, end, dtype=int)[:, None]
            rows, cols = np.nonzero(overlap)
            for row, col in zip(rows.tolist(), cols.tolist()):
                left = int(ids[start + row])
                right = int(ids[col])
                neighbor_sets[left].add(right)
                neighbor_sets[right].add(left)

        for subspace in self.subspaces:
            subspace.neighbors = sorted(neighbor_sets[subspace.index])
        self.adjacency_candidate_checks = count * (count - 1) // 2

    def _compute_adjacencies_from_tree(self, node: _Node) -> None:
        if node.is_leaf:
            return
        assert node.left is not None and node.right is not None
        self._compute_adjacencies_from_tree(node.left)
        self._compute_adjacencies_from_tree(node.right)
        self._connect_adjacent_node_pairs(node.left, node.right)

    def _connect_adjacent_node_pairs(
        self,
        left_node: _Node,
        right_node: _Node,
    ) -> None:
        self.adjacency_candidate_checks += 1
        if not self._nodes_overlap(left_node, right_node):
            return
        if left_node.is_leaf and right_node.is_leaf:
            self._add_neighbor(left_node.index, right_node.index)
            return
        if right_node.is_leaf or (
            not left_node.is_leaf and left_node.leaf_count >= right_node.leaf_count
        ):
            assert left_node.left is not None and left_node.right is not None
            self._connect_adjacent_node_pairs(left_node.left, right_node)
            self._connect_adjacent_node_pairs(left_node.right, right_node)
            return
        assert right_node.left is not None and right_node.right is not None
        self._connect_adjacent_node_pairs(left_node, right_node.left)
        self._connect_adjacent_node_pairs(left_node, right_node.right)

    def _nodes_overlap(self, left_node: _Node, right_node: _Node) -> bool:
        return self._bounds_overlap(
            left_node.bounds[0],
            left_node.bounds[1],
            right_node.bounds[0],
            right_node.bounds[1],
        )

    def _bounds_overlap(
        self,
        left_min: np.ndarray,
        left_max: np.ndarray,
        right_min: np.ndarray,
        right_max: np.ndarray,
    ) -> bool:
        return bool(np.all((left_max >= right_min - 1e-9) & (right_max >= left_min - 1e-9)))

    def _add_neighbor(self, left_idx: int, right_idx: int) -> None:
        self._neighbor_sets[int(left_idx)].add(int(right_idx))
        self._neighbor_sets[int(right_idx)].add(int(left_idx))


__all__ = ["SpacePartitioner"]
