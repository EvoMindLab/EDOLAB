"""Object-oriented, library-backed clustering utilities."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
import numpy as np
from scipy.cluster.hierarchy import linkage as _linkage
from scipy.spatial.distance import pdist as _pdist
from scipy.spatial.distance import squareform as _squareform
from sklearn.cluster import AffinityPropagation as _AffinityPropagation
from sklearn.cluster import HDBSCAN as _HDBSCAN
from sklearn.cluster import KMeans as _KMeans
from sklearn.cluster import OPTICS as _OPTICS


@_dataclass(slots=True)
class ClusterResult:
    """Fitted clustering state exposed by utility clusterers."""

    labels: np.ndarray = _field(default_factory=lambda: np.empty(0, dtype=int))
    centers: np.ndarray = _field(default_factory=lambda: np.empty((0, 0), dtype=float))
    groups: list[np.ndarray] = _field(default_factory=list)
    remain: np.ndarray = _field(default_factory=lambda: np.empty((0, 0), dtype=float))


class BaseClusterer:
    """Shared OO surface for EDOLAB clustering strategies."""

    result: ClusterResult

    def fit(self, positions: np.ndarray) -> "BaseClusterer":
        raise NotImplementedError

    def fit_predict(self, positions: np.ndarray) -> np.ndarray:
        return self.fit(positions).result.labels

    def groups_and_remain(self, positions: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        fitted = self.fit(positions).result
        return fitted.groups, fitted.remain

    def _coerce_points(self, positions: np.ndarray) -> np.ndarray:
        points = np.asarray(positions, dtype=float)
        if points.ndim != 2:
            raise ValueError("positions must be a 2-D array")
        return points

    def _nan_to_noise(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=float)
        return np.where(np.isnan(raw), -np.inf, raw)

    def _centers_from_labels(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        previous_centers: np.ndarray,
    ) -> np.ndarray:
        centers = np.asarray(previous_centers, dtype=float).copy()
        for cluster_id in range(centers.shape[0]):
            members = positions[labels == cluster_id, :]
            if members.size:
                centers[cluster_id, :] = np.mean(members, axis=0)
        return centers

    def _centers_from_positive_labels(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        centers = [
            np.mean(positions[labels == label, :], axis=0)
            for label in np.unique(labels)
            if label > 0
        ]
        if not centers:
            return np.empty((0, positions.shape[1]))
        return np.vstack(centers)

    def _normalize_labels(
        self,
        labels: np.ndarray,
        cluster_count: int | None = None,
    ) -> np.ndarray:
        values = np.asarray(labels, dtype=int).reshape(-1)
        mapping: dict[int, int] = {}
        normalized = np.empty(values.shape[0], dtype=int)
        for index, label in enumerate(values):
            raw_label = int(label)
            if raw_label not in mapping:
                mapping[raw_label] = len(mapping)
            normalized[index] = mapping[raw_label]
        if cluster_count is not None and len(mapping) != int(cluster_count):
            raise ValueError("clustering produced an unexpected number of clusters")
        return normalized

    def _normalize_nonnegative_labels(self, labels: np.ndarray) -> np.ndarray:
        values = np.asarray(labels, dtype=int).reshape(-1)
        if values.size == 0:
            return values
        if np.all(values < 0):
            return np.zeros(values.size, dtype=int)
        output = np.zeros(values.size, dtype=int)
        for new_label, old_label in enumerate(label for label in np.unique(values) if label >= 0):
            output[values == old_label] = new_label
        return output

    def _positive_noise_zero_labels(
        self,
        labels: np.ndarray,
        *,
        min_cluster_size: int = 1,
    ) -> np.ndarray:
        values = np.asarray(labels, dtype=int).reshape(-1)
        output = np.zeros(values.shape[0], dtype=int)
        next_label = 1
        for old_label in np.unique(values):
            if int(old_label) < 0:
                continue
            members = values == int(old_label)
            if np.count_nonzero(members) < int(min_cluster_size):
                continue
            output[members] = next_label
            next_label += 1
        return output


@_dataclass(slots=True)
class AffinityPropagationClusterer(BaseClusterer):
    """Sklearn-backed Affinity Propagation using precomputed similarity."""

    damping: float = 0.5
    max_iter: int = 200
    convergence_iter: int = 50
    preference: float | None = None
    random_state: int | None = 0
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)

    def fit(self, positions: np.ndarray) -> "AffinityPropagationClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        if count == 0:
            self.result = ClusterResult(remain=points)
            return self
        if count == 1:
            self.result = ClusterResult(
                labels=np.zeros(1, dtype=int),
                centers=points.copy(),
                groups=[points.copy()],
                remain=np.empty((0, points.shape[1])),
            )
            return self

        similarity = self._negative_squared_similarity(points)
        preference = (
            float(np.median(similarity))
            if self.preference is None
            else float(self.preference)
        )
        model = _AffinityPropagation(
            affinity="precomputed",
            damping=float(self.damping),
            max_iter=max(1, int(self.max_iter)),
            convergence_iter=max(1, int(self.convergence_iter)),
            preference=preference,
            random_state=self.random_state,
        )
        labels = self._normalize_nonnegative_labels(model.fit_predict(similarity))
        groups = [
            points[labels == label, :].copy()
            for label in np.unique(labels)
            if label >= 0
        ]
        if not groups:
            labels = np.zeros(count, dtype=int)
            groups = [points.copy()]
        centers = self._centers_from_labels(
            points,
            labels,
            np.empty((int(np.max(labels)) + 1, points.shape[1])),
        )
        self.result = ClusterResult(
            labels=labels,
            centers=centers,
            groups=groups,
            remain=np.empty((0, points.shape[1])),
        )
        return self

    def _negative_squared_similarity(self, points: np.ndarray) -> np.ndarray:
        diff = points[:, None, :] - points[None, :, :]
        return -np.sum(diff * diff, axis=2)


@_dataclass(slots=True)
class KMeansClusterer(BaseClusterer):
    """Sklearn-backed KMeans with EDOLAB label and empty-cluster semantics."""

    cluster_count: int
    rng: np.random.Generator | None = None
    n_init: int = 3
    max_iter: int = 100
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)

    def fit(self, positions: np.ndarray) -> "KMeansClusterer":
        points = self._coerce_points(positions)
        sample_count = points.shape[0]
        cluster_count = int(self.cluster_count)
        if cluster_count < 1:
            raise ValueError("cluster_count must be positive")
        if cluster_count > sample_count:
            raise ValueError("cluster_count must be less than or equal to sample count")
        if cluster_count == 1:
            labels = np.zeros(sample_count, dtype=int)
            centers = np.mean(points, axis=0, keepdims=True)
        else:
            model = _KMeans(
                n_clusters=cluster_count,
                init="k-means++",
                n_init=max(1, int(self.n_init)),
                max_iter=max(1, int(self.max_iter)),
                random_state=self._seed_from_rng(),
                algorithm="lloyd",
            )
            labels = model.fit_predict(points).astype(int, copy=False)
            labels, centers = self.repair_empty_clusters(points, labels, model.cluster_centers_)
            labels = self._normalize_labels(labels, cluster_count)
            centers = self._centers_from_labels(points, labels, centers)

        self.result = ClusterResult(
            labels=labels,
            centers=centers,
            groups=[points[labels == idx, :].copy() for idx in range(cluster_count)],
            remain=np.empty((0, points.shape[1])),
        )
        return self

    def repair_empty_clusters(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = self._coerce_points(positions)
        repaired = np.asarray(labels, dtype=int).copy()
        repaired_centers = np.asarray(centers, dtype=float).copy()
        cluster_count = repaired_centers.shape[0]
        while True:
            counts = np.bincount(repaired, minlength=cluster_count)
            empty_clusters = np.flatnonzero(counts == 0)
            if empty_clusters.size == 0:
                return repaired, self._centers_from_labels(points, repaired, repaired_centers)
            distances = np.sum((points - repaired_centers[repaired]) ** 2, axis=1)
            for empty_cluster in empty_clusters:
                counts = np.bincount(repaired, minlength=cluster_count)
                donor_mask = counts[repaired] > 1
                if not np.any(donor_mask):
                    donor_mask = np.ones(repaired.shape[0], dtype=bool)
                donor_distances = np.where(donor_mask, distances, -np.inf)
                donor_id = int(np.argmax(donor_distances))
                repaired[donor_id] = int(empty_cluster)
                repaired_centers = self._centers_from_labels(
                    points,
                    repaired,
                    repaired_centers,
                )

    def _seed_from_rng(self) -> int:
        if self.rng is None:
            return 0
        return int(self.rng.integers(0, np.iinfo(np.int32).max))


@_dataclass(slots=True)
class HierarchicalClusterer(BaseClusterer):
    """SciPy-backed hierarchical grouping with EDOLAB stopping semantics."""

    linkage_method: str = "complete"
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)
    merge_order_: np.ndarray = _field(default_factory=lambda: np.empty((0, 4)), init=False)

    def fit(self, positions: np.ndarray) -> "HierarchicalClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        if count == 0:
            self.result = ClusterResult(remain=points)
            self.merge_order_ = np.empty((0, 4))
            return self
        if count == 1:
            self.result = ClusterResult(
                labels=np.zeros(1, dtype=int),
                remain=points.copy(),
            )
            self.merge_order_ = np.empty((0, 4))
            return self

        condensed_distances = _pdist(points, metric="euclidean")
        distances = _squareform(condensed_distances)
        self.merge_order_ = _linkage(condensed_distances, method=self.linkage_method)
        cluster_members: dict[int, list[int]] = {idx: [idx] for idx in range(count)}
        active_ids = list(range(count))
        d_intra = 0.0
        d_inter = float(np.sum(distances) / 2.0)

        for merge_index, (left_raw, right_raw, *_rest) in enumerate(self.merge_order_):
            if d_intra >= d_inter:
                break
            left_id = int(left_raw)
            right_id = int(right_raw)
            new_id = count + merge_index
            left_pos = active_ids.index(left_id)
            right_pos = active_ids.index(right_id)
            insert_pos = min(left_pos, right_pos)
            merged = cluster_members.pop(left_id) + cluster_members.pop(right_id)
            active_ids = [
                cluster_id
                for cluster_id in active_ids
                if cluster_id not in (left_id, right_id)
            ]
            active_ids.insert(insert_pos, new_id)
            cluster_members[new_id] = merged
            d_intra, d_inter = self._linkage_distance(
                [cluster_members[cluster_id] for cluster_id in active_ids],
                distances,
            )

        labels = np.zeros(count, dtype=int)
        groups: list[np.ndarray] = []
        remain_rows: list[np.ndarray] = []
        next_label = 1
        for cluster_id in active_ids:
            members = cluster_members[cluster_id]
            if len(members) == 1:
                remain_rows.append(points[members[0]].copy())
                continue
            labels[members] = next_label
            next_label += 1
            groups.append(points[members, :].copy())
        remain = np.vstack(remain_rows) if remain_rows else np.empty((0, points.shape[1]))
        self.result = ClusterResult(labels=labels, groups=groups, remain=remain)
        return self

    def _linkage_distance(
        self,
        groups: list[list[int]],
        distances: np.ndarray,
    ) -> tuple[float, float]:
        inter = 0.0
        intra = 0.0
        for ii in range(len(groups) - 1):
            for jj in range(ii + 1, len(groups)):
                min_distance = np.inf
                for kk in groups[ii]:
                    for gg in groups[jj]:
                        min_distance = min(min_distance, distances[kk, gg])
                inter += min_distance
        for group in groups:
            for kk in range(len(group) - 1):
                for gg in range(kk + 1, len(group)):
                    intra += distances[group[kk], group[gg]]
        return float(intra), float(inter)


@_dataclass(slots=True)
class GreedyHierarchicalClusterer(BaseClusterer):
    """Greedy hierarchical grouping with configurable minimum group size.

    The merge rule is intentionally independent of any algorithm instance: it
    repeatedly merges the pair whose complete-linkage distance is smallest, then
    stops once total intra-group distance reaches total nearest inter-group
    distance.
    """

    min_group_size: int = 2
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)

    def fit(self, positions: np.ndarray) -> "GreedyHierarchicalClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        if count == 0:
            self.result = ClusterResult(labels=np.zeros(0, dtype=int), remain=points)
            return self

        distances = self._pairwise_distances(points)
        group_ids = [[idx] for idx in range(count)]
        d_intra = 0.0
        d_inter = float(np.sum(distances) / 2.0)

        while len(group_ids) > 1 and d_intra < d_inter:
            left, right = self._next_merge_pair(group_ids, distances)
            if left < 0 or right < 0:
                break
            group_ids[left] = group_ids[left] + group_ids[right]
            del group_ids[right]
            d_intra, d_inter = self._intra_inter_distance(group_ids, distances)

        labels = np.zeros(count, dtype=int)
        groups: list[np.ndarray] = []
        remain_rows: list[np.ndarray] = []
        next_label = 1
        min_group_size = max(1, int(self.min_group_size))
        for ids in group_ids:
            if len(ids) < min_group_size:
                remain_rows.extend(points[ids, :])
                continue
            labels[ids] = next_label
            next_label += 1
            groups.append(points[ids, :].copy())

        remain = (
            np.vstack(remain_rows)
            if remain_rows
            else np.empty((0, points.shape[1]), dtype=float)
        )
        self.result = ClusterResult(
            labels=labels,
            centers=self._centers_from_positive_labels(points, labels),
            groups=groups,
            remain=remain,
        )
        return self

    def _next_merge_pair(
        self,
        group_ids: list[list[int]],
        distances: np.ndarray,
    ) -> tuple[int, int]:
        best_distance = np.inf
        best_pair = (-1, -1)
        for left in range(len(group_ids) - 1):
            for right in range(left + 1, len(group_ids)):
                distance = self._complete_linkage_distance(
                    group_ids[left],
                    group_ids[right],
                    distances,
                )
                if best_distance > distance:
                    best_distance = distance
                    best_pair = (left, right)
        return best_pair

    def _complete_linkage_distance(
        self,
        left_ids: list[int],
        right_ids: list[int],
        distances: np.ndarray,
    ) -> float:
        return float(np.max(distances[np.ix_(left_ids, right_ids)]))

    def _intra_inter_distance(
        self,
        group_ids: list[list[int]],
        distances: np.ndarray,
    ) -> tuple[float, float]:
        intra = 0.0
        inter = 0.0
        for left in range(len(group_ids) - 1):
            for right in range(left + 1, len(group_ids)):
                inter += float(np.min(distances[np.ix_(group_ids[left], group_ids[right])]))
        for ids in group_ids:
            for left in range(len(ids) - 1):
                for right in range(left + 1, len(ids)):
                    intra += float(distances[ids[left], ids[right]])
        return intra, inter

    def _pairwise_distances(self, points: np.ndarray) -> np.ndarray:
        diff = points[:, None, :] - points[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))


@_dataclass(slots=True)
class SingleLinkageClusterer(HierarchicalClusterer):
    """Single-linkage hierarchical grouping."""

    linkage_method: str = _field(default="single", init=False)


@_dataclass(slots=True)
class CompleteLinkageClusterer(HierarchicalClusterer):
    """Complete-linkage hierarchical grouping."""

    max_size: int = 0
    linkage_method: str = _field(default="complete", init=False)

    def fit(self, positions: np.ndarray) -> "CompleteLinkageClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        if count == 0:
            self.result = ClusterResult(remain=points)
            self.merge_order_ = np.empty((0, 4))
            return self
        if count == 1:
            self.result = ClusterResult(
                labels=np.zeros(1, dtype=int),
                centers=points.copy(),
                groups=[points.copy()],
                remain=np.empty((0, points.shape[1])),
            )
            self.merge_order_ = np.empty((0, 4))
            return self

        condensed_distances = _pdist(points, metric="euclidean")
        distances = _squareform(condensed_distances)
        self.merge_order_ = _linkage(condensed_distances, method=self.linkage_method)
        group_states = [
            {"positions": points[[idx], :].copy(), "ids": [idx]}
            for idx in range(count)
        ]

        while any(group["positions"].shape[0] == 1 for group in group_states):
            merge_left = -1
            merge_right = -1
            min_distance = np.inf
            for left in range(len(group_states)):
                for right in range(left + 1, len(group_states)):
                    next_size = (
                        group_states[left]["positions"].shape[0]
                        + group_states[right]["positions"].shape[0]
                    )
                    if int(self.max_size) > 0 and next_size > int(self.max_size):
                        continue
                    distance = self._complete_linkage_distance(
                        group_states[left]["ids"],
                        group_states[right]["ids"],
                        distances,
                    )
                    if min_distance > distance:
                        min_distance = distance
                        merge_left = left
                        merge_right = right
            if merge_left < 0:
                break
            group_states[merge_left]["positions"] = np.vstack(
                [
                    group_states[merge_left]["positions"],
                    group_states[merge_right]["positions"],
                ]
            )
            group_states[merge_left]["ids"] = (
                group_states[merge_left]["ids"] + group_states[merge_right]["ids"]
            )
            del group_states[merge_right]

        labels = np.zeros(count, dtype=int)
        groups = []
        for label, group in enumerate(group_states):
            labels[group["ids"]] = label
            groups.append(group["positions"].copy())
        centers = self._centers_from_labels(
            points,
            labels,
            np.empty((len(groups), points.shape[1])),
        )
        self.result = ClusterResult(
            labels=labels,
            centers=centers,
            groups=groups,
            remain=np.empty((0, points.shape[1])),
        )
        return self

    def _complete_linkage_distance(
        self,
        left_ids: list[int],
        right_ids: list[int],
        distances: np.ndarray,
    ) -> float:
        max_distance = 0.0
        for left in left_ids:
            for right in right_ids:
                max_distance = max(max_distance, float(distances[left, right]))
        return max_distance


@_dataclass(slots=True)
class DensityPeakClusterer(BaseClusterer):
    """Density-peak clustering used by DPCPSO."""

    peak_number: int = 1
    percentile: float = 2.0
    max_centers: int = 30
    min_group_size: int = 3
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)
    rho_: np.ndarray = _field(default_factory=lambda: np.empty(0), init=False)
    delta_: np.ndarray = _field(default_factory=lambda: np.empty(0), init=False)
    gamma_: np.ndarray = _field(default_factory=lambda: np.empty(0), init=False)
    cutoff_distance_: float = 0.0

    def fit(self, positions: np.ndarray) -> "DensityPeakClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        self.rho_ = np.empty(count)
        self.delta_ = np.empty(count)
        self.gamma_ = np.empty(count)
        self.cutoff_distance_ = 0.0
        if int(self.peak_number) <= 0:
            raise ValueError("DensityPeakClusterer requires peak_number >= 1.")
        if count == 0:
            self.result = ClusterResult(remain=points)
            return self
        if count <= 2:
            self.result = self._fallback_result(points)
            return self

        distances = self._squared_distances(points)
        cutoff_distance = self._cutoff_distance(distances)
        self.cutoff_distance_ = cutoff_distance
        finite_distances = distances[np.isfinite(distances)]
        max_distance = float(np.max(finite_distances)) if finite_distances.size else 0.0
        rho = np.sum(np.exp(-((distances / cutoff_distance) ** 2)), axis=1)
        rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(rho)[::-1]

        delta = np.full(count, max_distance, dtype=float)
        nearest_higher = np.zeros(count, dtype=int)
        for order_index in range(1, count):
            current = int(order[order_index])
            for higher_index in range(order_index):
                neighbor = int(order[higher_index])
                if distances[current, neighbor] < delta[current]:
                    delta[current] = distances[current, neighbor]
                    nearest_higher[current] = neighbor
        delta[int(order[0])] = float(np.max(delta)) if delta.size else max_distance

        gamma = rho * delta
        centers = self._select_centers(gamma, count)
        self.rho_ = rho
        self.delta_ = delta
        self.gamma_ = gamma
        if not centers:
            self.result = self._fallback_result(points)
            return self

        center_points = points[centers, :].copy()
        center_distances = np.linalg.norm(
            points[:, None, :] - center_points[None, :, :],
            axis=2,
        )
        cluster_indices = np.argmin(center_distances, axis=1)
        candidate_groups = [
            points[cluster_indices == index, :].copy()
            for index in range(len(centers))
        ]
        valid = [
            index
            for index, group in enumerate(candidate_groups)
            if group.shape[0] >= int(self.min_group_size)
        ]
        if not valid:
            self.result = self._fallback_result(points)
            return self

        labels = np.zeros(count, dtype=int)
        groups: list[np.ndarray] = []
        for label, group_index in enumerate(valid):
            labels[cluster_indices == group_index] = label
            groups.append(candidate_groups[group_index])
        self.result = ClusterResult(
            labels=labels,
            centers=center_points[valid, :].copy(),
            groups=groups,
            remain=np.empty((0, points.shape[1])),
        )
        return self

    def _fallback_result(self, points: np.ndarray) -> ClusterResult:
        labels = np.zeros(points.shape[0], dtype=int)
        center = np.mean(points, axis=0, keepdims=True)
        return ClusterResult(
            labels=labels,
            centers=center,
            groups=[points.copy()],
            remain=np.empty((0, points.shape[1])),
        )

    def _squared_distances(self, points: np.ndarray) -> np.ndarray:
        diff = points[:, None, :] - points[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        np.fill_diagonal(distances, np.inf)
        return distances

    def _cutoff_distance(self, distances: np.ndarray) -> float:
        sorted_distances = np.sort(distances.reshape(-1))
        position = int(np.round(sorted_distances.size * float(self.percentile) / 100.0))
        position = min(max(position, 1), sorted_distances.size) - 1
        cutoff = float(sorted_distances[position])
        finite_positive = distances[np.isfinite(distances) & (distances > 0.0)]
        if np.isfinite(cutoff) and cutoff > 0.0:
            return cutoff
        return float(np.min(finite_positive)) if finite_positive.size else 1.0

    def _select_centers(self, gamma: np.ndarray, count: int) -> list[int]:
        threshold = -6.077 * count * np.log(int(self.peak_number)) + 50.49 * count
        centers: list[int] = []
        for index in np.argsort(gamma)[::-1]:
            if gamma[int(index)] > threshold:
                centers.append(int(index))
            if len(centers) >= int(self.max_centers):
                break
        return centers


@_dataclass(slots=True)
class OPTICSClusterer(BaseClusterer):
    """Sklearn-backed OPTICS labels.

    ``cluster_method='dbscan'`` is less fragmented when a finite ``max_eps`` is
    provided; ``cluster_method='xi'`` exposes sklearn's automatic OPTICS split.
    """

    min_samples: int = 5
    min_cluster_size: int | None = None
    max_eps: float = np.inf
    xi: float = 0.05
    cluster_method: str = "dbscan"
    eps: float | None = None
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)

    def fit(self, positions: np.ndarray) -> "OPTICSClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        if count == 0:
            self.result = ClusterResult(labels=np.zeros(0, dtype=int), remain=points)
            return self
        min_samples = max(2, min(int(self.min_samples), count))
        cluster_size = (
            min_samples
            if self.min_cluster_size is None
            else int(self.min_cluster_size)
        )
        cluster_size = max(2, min(int(cluster_size), count))
        if count < min_samples or count < cluster_size:
            self.result = ClusterResult(
                labels=np.zeros(count, dtype=int),
                remain=points.copy(),
            )
            return self

        method = str(self.cluster_method).strip().lower()
        if method not in {"dbscan", "xi"}:
            method = "dbscan"
        eps = self.eps
        if method == "dbscan" and eps is None:
            eps = float(self.max_eps)
        model = _OPTICS(
            min_samples=min_samples,
            min_cluster_size=cluster_size,
            max_eps=float(self.max_eps),
            metric="euclidean",
            cluster_method=method,
            eps=eps,
            xi=float(self.xi),
        )
        labels = self._positive_noise_zero_labels(
            model.fit_predict(points),
            min_cluster_size=cluster_size,
        )
        groups = [
            points[labels == label, :].copy()
            for label in np.unique(labels)
            if label > 0
        ]
        remain = points[labels == 0, :].copy()
        self.result = ClusterResult(
            labels=labels,
            centers=self._centers_from_positive_labels(points, labels),
            groups=groups,
            remain=remain,
        )
        return self


@_dataclass(slots=True)
class MATLABOPTICSClusterer(BaseClusterer):
    """MATLAB ``optics_wrapper.m`` compatible OPTICS + DBSCAN extraction.

    ``min_samples`` follows the MATLAB wrapper's ``minPts`` convention and
    counts neighbors excluding the query point. Positive labels are 1-based and
    zero is noise, matching the MATLAB utility.
    """

    min_samples: int = 5
    max_eps: float | None = None
    min_cluster_size: int | None = None
    extraction_eps: float | None = None
    assign_noise_to_nearest_cluster: bool = True
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)
    details_: dict[str, object] = _field(default_factory=dict, init=False)
    cluster_info_: list[dict[str, object]] = _field(default_factory=list, init=False)

    def fit(self, positions: np.ndarray) -> "MATLABOPTICSClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        if count == 0:
            self.result = ClusterResult(labels=np.zeros(0, dtype=int), remain=points)
            self.details_ = self._empty_details(0)
            self.cluster_info_ = []
            return self

        min_pts = max(1, int(self.min_samples))
        min_cluster_size = min_pts if self.min_cluster_size is None else int(self.min_cluster_size)
        min_cluster_size = max(1, min_cluster_size)
        if count <= min_pts:
            labels = np.zeros(count, dtype=int)
            self.result = ClusterResult(labels=labels, remain=points.copy())
            self.details_ = self._empty_details(count)
            self.details_.update(
                {
                    "minPts": min_pts,
                    "minClusterSize": min_cluster_size,
                    "extractionMode": "dbscan_at_selected_eps",
                }
            )
            self.cluster_info_ = []
            return self

        distances = self._pairwise_distances(points)
        max_eps = self._resolve_max_eps(distances)
        sorted_distances = np.sort(distances, axis=1)
        core_distance = sorted_distances[:, min_pts - 1].copy()
        core_distance[core_distance > max_eps] = np.inf

        order, reachability, predecessor = self._optics_order(distances, core_distance, max_eps)
        auto_extraction = self.extraction_eps is None
        auto_selection: dict[str, object] = {
            "selected": False,
            "gapIndex": 0,
            "candidateCount": 0,
            "sortedValues": np.empty(0, dtype=float),
        }
        if auto_extraction:
            extraction_eps, auto_selection = self._choose_extraction_eps(
                reachability,
                core_distance,
                max_eps,
            )
            extraction_eps, auto_selection = self._refine_auto_extraction_eps(
                points,
                min_pts,
                extraction_eps,
                max_eps,
                min_cluster_size,
                auto_selection,
            )
        else:
            extraction_eps = float(self.extraction_eps)
            if not np.isfinite(extraction_eps) or extraction_eps <= 0:
                raise ValueError("extraction_eps must be None or a positive finite scalar")

        labels, removed_clusters = self._extract_labels_at_eps(
            points,
            extraction_eps,
            min_pts,
            min_cluster_size,
        )
        labels_before_noise_assignment = labels.copy()
        reassigned_noise_count = 0
        if bool(self.assign_noise_to_nearest_cluster):
            labels, reassigned_noise_count = self._assign_noise_to_nearest_cluster(points, labels)

        self.cluster_info_ = self._build_cluster_info(points, labels, order, reachability, core_distance)
        self.details_ = {
            "order": order.copy(),
            "reachability": reachability.copy(),
            "coreDistance": core_distance.copy(),
            "predecessor": predecessor.copy(),
            "maxEps": float(max_eps),
            "extractionEps": float(extraction_eps),
            "autoExtraction": bool(auto_extraction),
            "autoSelection": auto_selection,
            "minPts": int(min_pts),
            "minClusterSize": int(min_cluster_size),
            "removedClusters": np.asarray(removed_clusters, dtype=int),
            "extractionMode": "dbscan_at_selected_eps",
            "labelsBeforeNoiseAssignment": labels_before_noise_assignment,
            "assignNoiseToNearestCluster": bool(self.assign_noise_to_nearest_cluster),
            "reassignedNoiseCount": int(reassigned_noise_count),
        }

        groups = [
            points[labels == label, :].copy()
            for label in np.unique(labels)
            if label > 0
        ]
        remain = points[labels == 0, :].copy()
        centers = self._centers_from_positive_labels(points, labels)
        self.result = ClusterResult(labels=labels, centers=centers, groups=groups, remain=remain)
        return self

    @staticmethod
    def _empty_details(count: int) -> dict[str, object]:
        return {
            "order": np.arange(int(count), dtype=int),
            "reachability": np.full(int(count), np.inf),
            "coreDistance": np.full(int(count), np.inf),
            "predecessor": np.full(int(count), -1, dtype=int),
            "maxEps": np.inf,
            "extractionEps": np.inf,
            "autoExtraction": True,
            "autoSelection": {
                "selected": False,
                "gapIndex": 0,
                "candidateCount": 0,
                "sortedValues": np.empty(0, dtype=float),
            },
            "removedClusters": np.empty(0, dtype=int),
            "labelsBeforeNoiseAssignment": np.zeros(int(count), dtype=int),
            "assignNoiseToNearestCluster": True,
            "reassignedNoiseCount": 0,
        }

    @staticmethod
    def _pairwise_distances(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= 1:
            return np.full((points.shape[0], points.shape[0]), np.inf)
        distances = _squareform(_pdist(points, metric="euclidean"))
        np.fill_diagonal(distances, np.inf)
        return distances

    def _resolve_max_eps(self, distances: np.ndarray) -> float:
        if self.max_eps is not None:
            value = float(self.max_eps)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("max_eps must be None or a positive finite scalar")
            return value
        finite = distances[np.isfinite(distances)]
        return float(np.max(finite)) if finite.size else np.inf

    def _optics_order(
        self,
        distances: np.ndarray,
        core_distance: np.ndarray,
        max_eps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = distances.shape[0]
        processed = np.zeros(count, dtype=bool)
        reachability = np.full(count, np.inf, dtype=float)
        predecessor = np.full(count, -1, dtype=int)
        order: list[int] = []

        for point_idx in range(count):
            if processed[point_idx]:
                continue
            candidate_reachability = np.full(count, np.inf, dtype=float)
            candidate_predecessor = np.full(count, -1, dtype=int)
            in_seeds = np.zeros(count, dtype=bool)

            processed[point_idx] = True
            order.append(point_idx)
            if np.isfinite(core_distance[point_idx]) and core_distance[point_idx] <= max_eps:
                self._update_seeds(
                    point_idx,
                    processed,
                    candidate_reachability,
                    candidate_predecessor,
                    in_seeds,
                    distances,
                    core_distance,
                    max_eps,
                )

            while np.any(in_seeds):
                candidates = np.flatnonzero(in_seeds)
                best_relative = int(np.argmin(candidate_reachability[candidates]))
                seed_idx = int(candidates[best_relative])
                in_seeds[seed_idx] = False
                if processed[seed_idx]:
                    continue

                reachability[seed_idx] = candidate_reachability[seed_idx]
                predecessor[seed_idx] = candidate_predecessor[seed_idx]
                processed[seed_idx] = True
                order.append(seed_idx)
                if np.isfinite(core_distance[seed_idx]) and core_distance[seed_idx] <= max_eps:
                    self._update_seeds(
                        seed_idx,
                        processed,
                        candidate_reachability,
                        candidate_predecessor,
                        in_seeds,
                        distances,
                        core_distance,
                        max_eps,
                    )

        return np.asarray(order, dtype=int), reachability, predecessor

    @staticmethod
    def _update_seeds(
        source_idx: int,
        processed: np.ndarray,
        candidate_reachability: np.ndarray,
        candidate_predecessor: np.ndarray,
        in_seeds: np.ndarray,
        distances: np.ndarray,
        core_distance: np.ndarray,
        max_eps: float,
    ) -> None:
        neighbors = np.flatnonzero(distances[source_idx, :] <= max_eps)
        source_core_distance = float(core_distance[source_idx])
        for neighbor_idx in neighbors:
            if processed[int(neighbor_idx)]:
                continue
            new_reachability = max(source_core_distance, float(distances[source_idx, neighbor_idx]))
            if (
                not in_seeds[int(neighbor_idx)]
                or new_reachability < candidate_reachability[int(neighbor_idx)]
            ):
                candidate_reachability[int(neighbor_idx)] = new_reachability
                candidate_predecessor[int(neighbor_idx)] = int(source_idx)
                in_seeds[int(neighbor_idx)] = True

    @staticmethod
    def _choose_extraction_eps(
        reachability: np.ndarray,
        core_distance: np.ndarray,
        max_eps: float,
    ) -> tuple[float, dict[str, object]]:
        candidate_values = np.concatenate(
            [
                reachability[np.isfinite(reachability)],
                core_distance[np.isfinite(core_distance)],
            ]
        )
        candidate_values = candidate_values[candidate_values > 0]
        auto_selection: dict[str, object] = {
            "selected": False,
            "gapIndex": 0,
            "candidateCount": int(candidate_values.size),
            "sortedValues": np.empty(0, dtype=float),
        }
        if candidate_values.size == 0:
            return float(max_eps), auto_selection

        sorted_values = np.sort(candidate_values)
        auto_selection["sortedValues"] = sorted_values.copy()
        if sorted_values.size == 1:
            auto_selection["selected"] = True
            auto_selection["gapIndex"] = 1
            return float(sorted_values[0]), auto_selection

        log_values = np.log(sorted_values)
        gaps = np.diff(log_values)
        lower_idx = max(1, int(np.floor(0.10 * gaps.size)))
        upper_idx = min(gaps.size, max(lower_idx, int(np.ceil(0.90 * gaps.size))))
        window = gaps[lower_idx - 1 : upper_idx]
        relative_idx = int(np.argmax(window))
        gap_idx = lower_idx + relative_idx
        extraction_eps = float(np.exp((log_values[gap_idx - 1] + log_values[gap_idx]) / 2.0))
        extraction_eps = min(extraction_eps, float(max_eps))
        auto_selection["selected"] = True
        auto_selection["gapIndex"] = int(gap_idx)
        return extraction_eps, auto_selection

    def _refine_auto_extraction_eps(
        self,
        points: np.ndarray,
        min_pts: int,
        base_eps: float,
        max_eps: float,
        min_cluster_size: int,
        auto_selection: dict[str, object],
    ) -> tuple[float, dict[str, object]]:
        distances = self._pairwise_distances(points)
        sorted_distances = np.sort(distances, axis=1)
        finite_core = sorted_distances[:, min_pts - 1]
        finite_core = finite_core[np.isfinite(finite_core) & (finite_core > 0)]

        auto_selection = dict(auto_selection)
        auto_selection.update(
            {
                "baseEps": float(base_eps),
                "baseScore": -np.inf,
                "baseNoiseFraction": 1.0,
                "baseClusterCount": 0,
                "refined": False,
                "selectedStrategy": "largest_log_gap",
                "quantileLevels": np.empty(0, dtype=float),
                "quantileCandidates": np.empty(0, dtype=float),
                "candidateScores": np.empty(0, dtype=float),
                "candidateNoiseFractions": np.empty(0, dtype=float),
                "candidateLargestClusterFractions": np.empty(0, dtype=float),
                "candidateClusterCounts": np.empty(0, dtype=int),
            }
        )
        if finite_core.size == 0:
            return float(base_eps), auto_selection

        base_score, base_noise, base_largest, base_count = self._score_extraction(
            points,
            base_eps,
            min_pts,
            min_cluster_size,
        )
        auto_selection.update(
            {
                "baseScore": float(base_score),
                "baseNoiseFraction": float(base_noise),
                "baseLargestClusterFraction": float(base_largest),
                "baseClusterCount": int(base_count),
            }
        )

        quantile_levels = np.arange(0.15, 0.85 + 1e-12, 0.05)
        quantile_candidates = np.quantile(finite_core, quantile_levels)
        candidate_eps = np.unique(np.concatenate([[base_eps], quantile_candidates]))
        candidate_eps = candidate_eps[np.isfinite(candidate_eps) & (candidate_eps > 0)]
        candidate_eps = np.minimum(candidate_eps, max_eps)
        candidate_eps = np.unique(candidate_eps)
        if candidate_eps.size == 0:
            return float(base_eps), auto_selection

        scores = np.full(candidate_eps.size, -np.inf, dtype=float)
        noise_fractions = np.ones(candidate_eps.size, dtype=float)
        largest_fractions = np.ones(candidate_eps.size, dtype=float)
        cluster_counts = np.zeros(candidate_eps.size, dtype=int)
        for idx, candidate in enumerate(candidate_eps):
            scores[idx], noise_fractions[idx], largest_fractions[idx], cluster_counts[idx] = (
                self._score_extraction(points, float(candidate), min_pts, min_cluster_size)
            )

        auto_selection.update(
            {
                "quantileLevels": quantile_levels,
                "quantileCandidates": candidate_eps,
                "candidateScores": scores,
                "candidateNoiseFractions": noise_fractions,
                "candidateLargestClusterFractions": largest_fractions,
                "candidateClusterCounts": cluster_counts,
            }
        )

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_eps = float(candidate_eps[best_idx])
        should_refine = False
        if base_noise > 0.65 and best_score > base_score:
            should_refine = True
        elif best_score > base_score + 0.05:
            should_refine = True
        if should_refine:
            auto_selection["refined"] = True
            auto_selection["selectedStrategy"] = "coverage_dominance_refinement"
            auto_selection["selected"] = True
            return best_eps, auto_selection
        return float(base_eps), auto_selection

    def _score_extraction(
        self,
        points: np.ndarray,
        extraction_eps: float,
        min_pts: int,
        min_cluster_size: int,
    ) -> tuple[float, float, float, int]:
        labels, _ = self._extract_labels_at_eps(points, extraction_eps, min_pts, min_cluster_size)
        clustered = labels > 0
        noise_fraction = 1.0 - float(np.mean(clustered))
        positive = np.unique(labels[clustered])
        cluster_count = int(positive.size)
        if cluster_count == 0:
            return -np.inf, noise_fraction, 1.0, cluster_count
        sizes = np.asarray([np.count_nonzero(labels == label) for label in positive], dtype=float)
        largest_cluster_fraction = float(np.max(sizes) / labels.size)
        clustered_fraction = 1.0 - noise_fraction
        score = clustered_fraction * (1.0 - largest_cluster_fraction) - 0.002 * cluster_count
        return float(score), noise_fraction, largest_cluster_fraction, cluster_count

    def _extract_labels_at_eps(
        self,
        points: np.ndarray,
        extraction_eps: float,
        min_pts: int,
        min_cluster_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = self._local_dbscan(points, extraction_eps, min_pts)
        return self._filter_small_clusters(labels, min_cluster_size)

    @staticmethod
    def _local_dbscan(points: np.ndarray, eps_value: float, min_pts: int) -> np.ndarray:
        count = points.shape[0]
        labels = np.zeros(count, dtype=int)
        cluster_id = 0
        for point_idx in range(count):
            if labels[point_idx] != 0:
                continue
            neighbors = MATLABOPTICSClusterer._local_find_neighbors(points, point_idx, eps_value)
            if neighbors.size < int(min_pts):
                labels[point_idx] = 0
                continue
            cluster_id += 1
            labels[point_idx] = cluster_id
            queue = list(int(idx) for idx in neighbors)
            while queue:
                current_idx = queue.pop(0)
                if labels[current_idx] == 0:
                    labels[current_idx] = cluster_id
                if labels[current_idx] != cluster_id:
                    continue
                current_neighbors = MATLABOPTICSClusterer._local_find_neighbors(
                    points,
                    current_idx,
                    eps_value,
                )
                if current_neighbors.size >= int(min_pts):
                    unlabeled = current_neighbors[labels[current_neighbors] == 0]
                    queue.extend(int(idx) for idx in unlabeled)
        return labels

    @staticmethod
    def _local_find_neighbors(points: np.ndarray, point_idx: int, eps_value: float) -> np.ndarray:
        distances = np.linalg.norm(points - points[int(point_idx), :], axis=1)
        return np.flatnonzero((distances <= float(eps_value)) & (distances > 0.0))

    @staticmethod
    def _filter_small_clusters(
        labels: np.ndarray,
        min_cluster_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        output = np.asarray(labels, dtype=int).copy()
        removed: list[int] = []
        for cluster_id in np.unique(output[output > 0]):
            members = output == int(cluster_id)
            if np.count_nonzero(members) < int(min_cluster_size):
                output[members] = 0
                removed.append(int(cluster_id))

        kept = np.unique(output[output > 0])
        for new_id, old_id in enumerate(kept, start=1):
            output[output == int(old_id)] = int(new_id)
        return output, np.asarray(removed, dtype=int)

    @staticmethod
    def _assign_noise_to_nearest_cluster(
        points: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        output = np.asarray(labels, dtype=int).copy()
        cluster_ids = np.unique(output[output > 0])
        if cluster_ids.size == 0:
            return output, 0
        noise_members = np.flatnonzero(output == 0)
        if noise_members.size == 0:
            return output, 0
        centers = np.vstack([np.mean(points[output == label, :], axis=0) for label in cluster_ids])
        distances = np.linalg.norm(points[noise_members, None, :] - centers[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        output[noise_members] = cluster_ids[nearest]
        return output, int(noise_members.size)

    @staticmethod
    def _build_cluster_info(
        points: np.ndarray,
        labels: np.ndarray,
        order: np.ndarray,
        reachability: np.ndarray,
        core_distance: np.ndarray,
    ) -> list[dict[str, object]]:
        cluster_info: list[dict[str, object]] = []
        if labels.size == 0:
            return cluster_info
        order_position = np.zeros(labels.size, dtype=int)
        if order.size:
            order_position[np.asarray(order, dtype=int)] = np.arange(1, order.size + 1)
        for cluster_id in np.unique(labels[labels > 0]):
            members = np.flatnonzero(labels == int(cluster_id))
            member_reachability = reachability[members]
            member_core_distance = core_distance[members]
            finite_reachability = member_reachability[np.isfinite(member_reachability)]
            finite_core = member_core_distance[np.isfinite(member_core_distance)]
            positions = order_position[members]
            cluster_info.append(
                {
                    "id": int(cluster_id),
                    "size": int(members.size),
                    "center": np.mean(points[members, :], axis=0),
                    "members": members.copy(),
                    "medianReachability": (
                        float(np.median(finite_reachability))
                        if finite_reachability.size
                        else np.nan
                    ),
                    "medianCoreDistance": (
                        float(np.median(finite_core))
                        if finite_core.size
                        else np.nan
                    ),
                    "orderingSpan": (
                        np.asarray([int(np.min(positions)), int(np.max(positions))], dtype=int)
                        if positions.size
                        else np.asarray([0, 0], dtype=int)
                    ),
                }
            )
        return cluster_info


@_dataclass(slots=True)
class HDBSCANClusterer(BaseClusterer):
    """Sklearn-backed HDBSCAN clusterer."""

    min_samples: int = 5
    min_cluster_size: int = 20
    cluster_selection_epsilon: float = 0.0
    allow_single_cluster: bool = False
    result: ClusterResult = _field(default_factory=ClusterResult, init=False)

    def fit(self, positions: np.ndarray) -> "HDBSCANClusterer":
        points = self._coerce_points(positions)
        count = points.shape[0]
        if count == 0:
            self.result = ClusterResult(labels=np.zeros(0, dtype=int), remain=points)
            return self
        min_cluster_size = max(2, min(int(self.min_cluster_size), count))
        min_samples = max(1, min(int(self.min_samples), count))
        if count < min_cluster_size:
            self.result = ClusterResult(
                labels=np.zeros(count, dtype=int),
                remain=points.copy(),
            )
            return self

        model = _HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=max(0.0, float(self.cluster_selection_epsilon)),
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=bool(self.allow_single_cluster),
        )
        labels = self._positive_noise_zero_labels(
            model.fit_predict(points),
            min_cluster_size=min_cluster_size,
        )
        groups = [
            points[labels == label, :].copy()
            for label in np.unique(labels)
            if label > 0
        ]
        remain = points[labels == 0, :].copy()
        self.result = ClusterResult(
            labels=labels,
            centers=self._centers_from_positive_labels(points, labels),
            groups=groups,
            remain=remain,
        )
        return self


__all__ = [
    "AffinityPropagationClusterer",
    "BaseClusterer",
    "ClusterResult",
    "CompleteLinkageClusterer",
    "DensityPeakClusterer",
    "GreedyHierarchicalClusterer",
    "HierarchicalClusterer",
    "HDBSCANClusterer",
    "KMeansClusterer",
    "MATLABOPTICSClusterer",
    "OPTICSClusterer",
    "SingleLinkageClusterer",
]
