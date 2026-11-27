"""Nearest-better network for continuous optimization snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np


OptimizationSense = Literal["max", "min"]
NBN_FONT_FAMILY = "Times New Roman"
NBN_FITNESS_CMAP = "coolwarm"
NBN_TORCH_MIN_SAMPLE_COUNT = 512
NBN_TORCH_MAX_BLOCK_CELLS = 6_000_000
NBN_TORCH_MPS_MAX_BLOCK_CELLS = 2_000_000


class NBNContinuous:
    """Build and inspect nearest-better forests from continuous samples."""

    @dataclass(slots=True)
    class Snapshot:
        environment: int
        x: np.ndarray
        fitness: np.ndarray
        local_node_id: np.ndarray
        global_sample_index: np.ndarray
        parent: np.ndarray
        parent_global_index: np.ndarray
        parent_distance: np.ndarray
        root: np.ndarray
        root_global_index: np.ndarray
        basin: np.ndarray
        depth: np.ndarray
        path_distance: np.ndarray
        children: list[list[int]]
        root_node_ids: np.ndarray
        edge_list: np.ndarray
        normalized_fitness: np.ndarray
        layout_2d: np.ndarray
        labels: list[Any] = field(default_factory=list)

        @property
        def node_count(self) -> int:
            return int(self.x.shape[0])

        @property
        def dimension(self) -> int:
            return int(self.x.shape[1]) if self.x.ndim == 2 else 0

        @property
        def summary(self) -> dict[str, int | float]:
            largest_basin = 0
            if self.basin.size:
                _, counts = np.unique(self.basin, return_counts=True)
                largest_basin = int(np.max(counts))
            return {
                "environment": int(self.environment),
                "node_count": self.node_count,
                "edge_count": int(self.edge_list.shape[0]),
                "root_count": int(self.root_node_ids.size),
                "largest_basin_size": largest_basin,
            }

    @dataclass(slots=True)
    class NodeRow:
        node_id: int
        global_sample_index: int
        parent_node_id: int
        root_node_id: int
        basin: int
        depth: int
        parent_distance: float
        path_distance: float
        fitness: float
        normalized_fitness: float
        label: Any = None

    @dataclass(slots=True)
    class BasinRow:
        environment: int
        basin: int
        root_node_id: int
        best_node_id: int
        basin_size: int
        root_fitness: float

    @dataclass(slots=True)
    class Attachment:
        item_id: int
        environment: int
        label: str
        position: np.ndarray
        fitness: float
        node_id: int
        root_node_id: int
        node_distance: float
        behavior_tag: str

    def __init__(
        self,
        optimization_sense: str = "max",
        fitness_tolerance: float = 1e-12,
        distance_tolerance: float = 1e-12,
        build_layout: bool = True,
    ) -> None:
        self.optimization_sense = self._normalize_sense(optimization_sense)
        self.fitness_tolerance = float(fitness_tolerance)
        self.distance_tolerance = float(distance_tolerance)
        self.build_layout = bool(build_layout)
        self.clear_samples()

    def configure(
        self,
        optimization_sense: str | None = None,
        fitness_tolerance: float | None = None,
        distance_tolerance: float | None = None,
        build_layout: bool | None = None,
    ) -> None:
        if optimization_sense is not None:
            self.optimization_sense = self._normalize_sense(optimization_sense)
        if fitness_tolerance is not None:
            self.fitness_tolerance = float(fitness_tolerance)
        if distance_tolerance is not None:
            self.distance_tolerance = float(distance_tolerance)
        if build_layout is not None:
            self.build_layout = bool(build_layout)
        self.is_built = False

    def clear_samples(self) -> None:
        self.samples_x = np.zeros((0, 0), dtype=float)
        self.samples_fitness = np.zeros(0, dtype=float)
        self.samples_environment = np.zeros(0, dtype=int)
        self.sample_labels: list[Any] = []
        self.dimension = 0
        self.is_built = False
        self.snapshots: list[NBNContinuous.Snapshot] = []

    def set_samples(
        self,
        x: Any,
        fitness: Any,
        environment: Any | None = None,
        labels: list[Any] | None = None,
    ) -> None:
        self.clear_samples()
        self.add_samples(x, fitness, environment=environment, labels=labels)

    def add_samples(
        self,
        x: Any,
        fitness: Any,
        environment: Any | None = None,
        labels: list[Any] | None = None,
    ) -> None:
        data, values = self._validate_samples(x, fitness)
        if self.dimension == 0:
            self.dimension = data.shape[1]
            self.samples_x = np.zeros((0, self.dimension), dtype=float)
        elif data.shape[1] != self.dimension:
            raise ValueError(f"Expected sample dimension {self.dimension}, got {data.shape[1]}.")

        env = self._resolve_environment(environment, data.shape[0])
        labels = list(labels) if labels is not None else [None for _ in range(data.shape[0])]
        if len(labels) != data.shape[0]:
            raise ValueError("labels must provide exactly one entry per sample.")

        self.samples_x = np.vstack([self.samples_x, data])
        self.samples_fitness = np.concatenate([self.samples_fitness, values])
        self.samples_environment = np.concatenate([self.samples_environment, env])
        self.sample_labels.extend(labels)
        self.is_built = False
        self.snapshots = []

    def build_from_samples(self, x: Any, fitness: Any, **kwargs: Any) -> None:
        self.set_samples(x, fitness, **kwargs)
        self.build()

    def list_environments(self) -> np.ndarray:
        if self.samples_environment.size == 0:
            return np.zeros(0, dtype=int)
        return np.asarray(list(dict.fromkeys(self.samples_environment.tolist())), dtype=int)

    def build(self, build_layout: bool | None = None) -> None:
        if build_layout is None:
            build_layout = self.build_layout
        self.snapshots = []
        for env in self.list_environments():
            indices = np.flatnonzero(self.samples_environment == env)
            self.snapshots.append(self._build_snapshot(int(env), indices, bool(build_layout)))
        self.is_built = True

    def get_snapshot(self, environment: int | None = None) -> "NBNContinuous.Snapshot":
        if not self.is_built:
            self.build()
        if not self.snapshots:
            raise ValueError("No NBN snapshot is available. Add samples first.")
        if environment is None:
            if len(self.snapshots) != 1:
                raise ValueError("environment must be specified when multiple snapshots exist.")
            return self.snapshots[0]
        for snapshot in self.snapshots:
            if int(snapshot.environment) == int(environment):
                return snapshot
        raise ValueError(f"Environment {environment} is not present in the NBN.")

    def query_nearest(
        self,
        positions: Any,
        environment: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        snapshot = self.get_snapshot(environment)
        query = self._validate_query_positions(positions, self.dimension)
        if snapshot.x.size == 0:
            return np.zeros(query.shape[0], dtype=int), np.full(query.shape[0], np.nan)
        distances = self._pairwise_distances(query, snapshot.x)
        nearest = np.argmin(distances, axis=1)
        return nearest + 1, distances[np.arange(query.shape[0]), nearest]

    def attach_individuals(
        self,
        individuals: Sequence[Any],
        environment: int | None = None,
    ) -> list["NBNContinuous.Attachment"]:
        """Attach standard individuals to their nearest NBN nodes."""

        items = list(individuals)
        if not items:
            return []
        positions = np.vstack([self._individual_position(item) for item in items])
        node_ids, distances = self.query_nearest(positions, environment)
        snapshot = self.get_snapshot(environment)
        attachments: list[NBNContinuous.Attachment] = []
        for index, item in enumerate(items):
            node_id = int(node_ids[index])
            root_id = int(snapshot.root[node_id - 1])
            attachment = self.Attachment(
                item_id=index + 1,
                environment=int(snapshot.environment),
                label=self._individual_label(item, index + 1),
                position=positions[index].copy(),
                fitness=self._individual_fitness(item),
                node_id=node_id,
                root_node_id=root_id,
                node_distance=float(distances[index]),
                behavior_tag="root" if node_id == root_id else "basin-member",
            )
            attachments.append(attachment)
        return attachments

    def export_nodes(self, environment: int | None = None) -> list["NBNContinuous.NodeRow"]:
        snapshot = self.get_snapshot(environment)
        rows: list[NBNContinuous.NodeRow] = []
        for index, node_id in enumerate(snapshot.local_node_id):
            rows.append(
                self.NodeRow(
                    node_id=int(node_id),
                    global_sample_index=int(snapshot.global_sample_index[index]),
                    parent_node_id=int(snapshot.parent[index]),
                    root_node_id=int(snapshot.root[index]),
                    basin=int(snapshot.basin[index]),
                    depth=int(snapshot.depth[index]),
                    parent_distance=float(snapshot.parent_distance[index]),
                    path_distance=float(snapshot.path_distance[index]),
                    fitness=float(snapshot.fitness[index]),
                    normalized_fitness=float(snapshot.normalized_fitness[index]),
                    label=snapshot.labels[index] if index < len(snapshot.labels) else None,
                )
            )
        return rows

    def export_basins(self, environment: int | None = None) -> list["NBNContinuous.BasinRow"]:
        snapshot = self.get_snapshot(environment)
        rows: list[NBNContinuous.BasinRow] = []
        for basin_index, root_id in enumerate(snapshot.root_node_ids, start=1):
            mask = snapshot.root == root_id
            rows.append(
                self.BasinRow(
                    environment=int(snapshot.environment),
                    basin=basin_index,
                    root_node_id=int(root_id),
                    best_node_id=int(root_id),
                    basin_size=int(np.sum(mask)),
                    root_fitness=float(snapshot.fitness[int(root_id) - 1]),
                )
            )
        return rows

    def plot_three_views(
        self,
        environment: int | None = None,
        *,
        figure: Any | None = None,
        axes: Sequence[Any] | None = None,
        node_size: float = 18,
        root_node_size: float = 60,
        show_node_labels: bool = False,
        title_prefix: str = "NBN",
        font_family: str = NBN_FONT_FAMILY,
        cmap: str = NBN_FITNESS_CMAP,
    ) -> Any:
        """Draw front, top, and side NBN projections.

        The three views use ``Layout2D`` and normalized fitness only. They are
        rendered as equal-size 2D projections so the visible panels match in
        front, top, and side views without the extra padding introduced by
        Matplotlib 3D axes.
        """
        snapshot = self.get_snapshot(environment)
        import matplotlib.pyplot as plt

        if axes is None:
            if figure is None:
                figure = plt.figure(figsize=(9.2, 4.4))
            else:
                figure.clf()
            grid = figure.add_gridspec(
                1,
                3,
                left=0.045,
                right=0.955,
                bottom=0.16,
                top=0.84,
                wspace=0.16,
            )
            axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
        else:
            axes = list(axes)
            if len(axes) != 3:
                raise ValueError("plot_three_views requires exactly three axes.")
            figure = axes[0].figure

        layout = self._tree_layout_coordinates(snapshot)
        z_values = np.asarray(snapshot.normalized_fitness, dtype=float).reshape(-1)
        if z_values.size == 0:
            z_values = np.zeros(snapshot.node_count, dtype=float)

        self._draw_snapshot_projection_panel(
            snapshot,
            axes[0],
            x_values=layout[:, 1] if layout.size else np.zeros(0, dtype=float),
            y_values=z_values,
            node_size=node_size,
            root_node_size=root_node_size,
            show_node_labels=show_node_labels,
            xlabel="Layout Y",
            ylabel="Normalized fitness",
            title=f"{title_prefix} Front View (Env {snapshot.environment})",
            cmap=cmap,
            spokes_mode="fitness",
        )
        self._draw_snapshot_projection_panel(
            snapshot,
            axes[1],
            x_values=layout[:, 1] if layout.size else np.zeros(0, dtype=float),
            y_values=layout[:, 0] if layout.size else np.zeros(0, dtype=float),
            node_size=node_size,
            root_node_size=root_node_size,
            show_node_labels=False,
            xlabel="Layout Y",
            ylabel="Layout X",
            title=f"{title_prefix} Top View (Env {snapshot.environment})",
            cmap=cmap,
            spokes_mode="radial",
            y_axis_side="right",
        )
        self._draw_snapshot_projection_panel(
            snapshot,
            axes[2],
            x_values=layout[:, 0] if layout.size else np.zeros(0, dtype=float),
            y_values=z_values,
            node_size=node_size,
            root_node_size=root_node_size,
            show_node_labels=False,
            xlabel="Layout X",
            ylabel="Normalized fitness",
            title=f"{title_prefix} Side View (Env {snapshot.environment})",
            cmap=cmap,
            spokes_mode="fitness",
            invert_x=True,
            y_axis_side="right",
        )
        self._align_three_view_axes_boxes(axes)
        self._apply_matplotlib_font(figure, font_family)
        return figure

    def _tree_layout_coordinates(self, snapshot: "NBNContinuous.Snapshot") -> np.ndarray:
        if snapshot.layout_2d.size:
            return snapshot.layout_2d
        return self._fallback_layout(snapshot.node_count)

    def _draw_snapshot_projection_panel(
        self,
        snapshot: "NBNContinuous.Snapshot",
        axes: Any,
        *,
        x_values: np.ndarray,
        y_values: np.ndarray,
        node_size: float,
        root_node_size: float,
        show_node_labels: bool,
        xlabel: str,
        ylabel: str,
        title: str,
        cmap: str,
        spokes_mode: Literal["radial", "fitness"],
        invert_x: bool = False,
        y_axis_side: Literal["left", "right"] = "left",
    ) -> Any:
        axes.cla()
        z_values = np.asarray(snapshot.normalized_fitness, dtype=float).reshape(-1)
        if z_values.size == 0:
            z_values = np.zeros(snapshot.node_count, dtype=float)
        plot_layout = np.column_stack((x_values, y_values)) if snapshot.node_count else np.zeros((0, 2), dtype=float)

        for child, parent in snapshot.edge_list:
            child_i = int(child) - 1
            parent_i = int(parent) - 1
            axes.plot(
                [plot_layout[child_i, 0], plot_layout[parent_i, 0]],
                [plot_layout[child_i, 1], plot_layout[parent_i, 1]],
                "-",
                color=(0.80, 0.80, 0.80),
                linewidth=0.75,
            )

        root_mask = snapshot.parent == 0
        scatter = None
        if snapshot.node_count:
            scatter = axes.scatter(
                plot_layout[:, 0],
                plot_layout[:, 1],
                marker="o",
                s=node_size,
                c=z_values,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                edgecolors=(0.55, 0.55, 0.55),
                linewidths=0.5,
            )
        if np.any(root_mask):
            axes.scatter(
                plot_layout[root_mask, 0],
                plot_layout[root_mask, 1],
                marker="o",
                s=root_node_size,
                c=z_values[root_mask],
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                edgecolors=(0.0, 0.0, 0.0),
                linewidths=1.15,
            )
        self._draw_virtual_root_spokes_projection(snapshot, axes, plot_layout, z_values, spokes_mode)

        if show_node_labels:
            for idx in range(snapshot.node_count):
                axes.text(
                    plot_layout[idx, 0],
                    plot_layout[idx, 1],
                    f" {idx + 1}",
                    fontsize=8,
                )

        axes.grid(True)
        try:
            axes.set_box_aspect(1.0)
        except AttributeError:
            axes.set_aspect("auto")
        axes.set_xlabel(xlabel, labelpad=1)
        axes.set_ylabel(ylabel, labelpad=1)
        axes.xaxis.set_ticks_position("bottom")
        axes.xaxis.set_label_position("bottom")
        if y_axis_side == "right":
            axes.yaxis.tick_right()
            axes.yaxis.set_label_position("right")
        else:
            axes.yaxis.tick_left()
            axes.yaxis.set_label_position("left")
        axes.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=8, pad=1)
        axes.tick_params(axis="y", labelsize=8, pad=1)
        axes.set_title(title, fontsize=10, pad=1.2)
        self._set_2d_axes_tight(axes, plot_layout)
        if invert_x:
            axes.invert_xaxis()
        return scatter

    def _draw_virtual_root_spokes_projection(
        self,
        snapshot: "NBNContinuous.Snapshot",
        axes: Any,
        plot_layout: np.ndarray,
        z_values: np.ndarray,
        mode: Literal["radial", "fitness"],
    ) -> None:
        root_ids = np.asarray(snapshot.root_node_ids, dtype=int).reshape(-1)
        root_ids = root_ids[(root_ids >= 1) & (root_ids <= plot_layout.shape[0])]
        if root_ids.size <= 1:
            return
        root_indices = root_ids - 1
        for root_index in root_indices:
            root_xy = plot_layout[root_index]
            if mode == "fitness":
                axes.plot(
                    [0.0, root_xy[0]],
                    [root_xy[1], root_xy[1]],
                    "--",
                    color=(0.15, 0.15, 0.15),
                    linewidth=1.4,
                )
            else:
                axes.plot(
                    [0.0, root_xy[0]],
                    [0.0, root_xy[1]],
                    "--",
                    color=(0.15, 0.15, 0.15),
                    linewidth=1.4,
                )
        marker_y = float(np.mean(z_values[root_indices])) if mode == "fitness" else 0.0
        axes.plot(
            [0.0],
            [marker_y],
            marker="x",
            linestyle="none",
            markersize=8.0,
            color=(0.15, 0.15, 0.15),
            markeredgewidth=1.2,
        )

    def _draw_virtual_root_spokes(
        self,
        snapshot: "NBNContinuous.Snapshot",
        axes: Any,
        layout: np.ndarray,
        z_values: np.ndarray,
    ) -> None:
        root_ids = np.asarray(snapshot.root_node_ids, dtype=int).reshape(-1)
        root_ids = root_ids[(root_ids >= 1) & (root_ids <= layout.shape[0])]
        if root_ids.size <= 1:
            return
        root_indices = root_ids - 1
        is_3d = hasattr(axes, "zaxis")
        for root_index in root_indices:
            root_xy = layout[root_index]
            root_z = float(z_values[root_index])
            if is_3d:
                axes.plot(
                    [0.0, root_xy[0]],
                    [0.0, root_xy[1]],
                    [root_z, root_z],
                    "--",
                    color=(0.15, 0.15, 0.15),
                    linewidth=1.4,
                )
            else:
                axes.plot(
                    [0.0, root_xy[0]],
                    [0.0, root_xy[1]],
                    "--",
                    color=(0.15, 0.15, 0.15),
                    linewidth=1.4,
                )
        if is_3d:
            axes.plot(
                [0.0],
                [0.0],
                [float(np.mean(z_values[root_indices]))],
                marker="x",
                linestyle="none",
                markersize=8.0,
                color=(0.15, 0.15, 0.15),
                markeredgewidth=1.2,
            )
        else:
            axes.plot(
                [0.0],
                [0.0],
                marker="x",
                linestyle="none",
                markersize=8.0,
                color=(0.15, 0.15, 0.15),
                markeredgewidth=1.2,
            )

    def _individual_position(self, individual: Any) -> np.ndarray:
        if hasattr(individual, "position"):
            position = getattr(individual, "position")
        elif isinstance(individual, dict) and "position" in individual:
            position = individual["position"]
        else:
            position = individual
        return np.asarray(position, dtype=float).reshape(1, -1)

    def _individual_fitness(self, individual: Any) -> float:
        if hasattr(individual, "fitness"):
            value = getattr(individual, "fitness")
        elif isinstance(individual, dict) and "fitness" in individual:
            value = individual["fitness"]
        else:
            value = np.nan
        return float(value) if value is not None else np.nan

    def _individual_label(self, individual: Any, fallback_id: int) -> str:
        if hasattr(individual, "label"):
            return str(getattr(individual, "label"))
        if isinstance(individual, dict) and "label" in individual:
            return str(individual["label"])
        return f"individual {fallback_id}"

    def _build_snapshot(
        self,
        environment: int,
        global_indices: np.ndarray,
        build_layout: bool,
    ) -> "NBNContinuous.Snapshot":
        x = self.samples_x[global_indices]
        fitness = self.samples_fitness[global_indices]
        labels = [self.sample_labels[int(index)] for index in global_indices]
        parent, parent_distance = self._compute_parent_forest(x, fitness)
        root, depth, path_distance = self._trace_forest(parent, parent_distance)
        basin, root_ids = self._assign_basins(root)
        children = self._build_children(parent)
        normalized = self._normalize_fitness(fitness, self.optimization_sense)
        layout = (
            self._build_radial_layout(parent, children, parent_distance, root_ids)
            if build_layout
            else self._fallback_layout(x.shape[0])
        )
        edge_mask = parent > 0
        edge_list = np.column_stack((np.flatnonzero(edge_mask) + 1, parent[edge_mask])) if np.any(edge_mask) else np.zeros((0, 2), dtype=int)
        parent_global = np.zeros_like(parent)
        edge_indices = np.flatnonzero(parent > 0)
        if edge_indices.size:
            parent_global[edge_indices] = global_indices[parent[edge_indices] - 1] + 1
        return self.Snapshot(
            environment=environment,
            x=x.copy(),
            fitness=fitness.copy(),
            local_node_id=np.arange(1, x.shape[0] + 1, dtype=int),
            global_sample_index=global_indices + 1,
            parent=parent,
            parent_global_index=parent_global,
            parent_distance=parent_distance,
            root=root,
            root_global_index=global_indices[root - 1] + 1 if root.size else np.zeros(0, dtype=int),
            basin=basin,
            depth=depth,
            path_distance=path_distance,
            children=children,
            root_node_ids=root_ids,
            edge_list=edge_list,
            normalized_fitness=normalized,
            layout_2d=layout,
            labels=labels,
        )

    def _compute_parent_forest(
        self,
        x: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x.shape[0] >= NBN_TORCH_MIN_SAMPLE_COUNT:
            accelerated = self._compute_parent_forest_torch(x, fitness)
            if accelerated is not None:
                return accelerated
        return self._compute_parent_forest_numpy(x, fitness)

    def _compute_parent_forest_numpy(
        self,
        x: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = x.shape[0]
        parent = np.zeros(count, dtype=int)
        parent_distance = np.full(count, np.nan, dtype=float)
        ordered_ids = self._sort_sample_ids(fitness, self.optimization_sense)
        for order_index, node_id in enumerate(ordered_ids):
            better_candidates = ordered_ids[:order_index]
            if better_candidates.size == 0:
                continue
            better_mask = self._is_better(
                fitness[better_candidates],
                fitness[node_id],
                self.optimization_sense,
                self.fitness_tolerance,
            )
            better_candidates = better_candidates[better_mask]
            if better_candidates.size == 0:
                continue
            dist_sq = np.sum((x[better_candidates] - x[node_id]) ** 2, axis=1)
            best_dist_sq = float(np.min(dist_sq))
            tied = dist_sq <= best_dist_sq + self.distance_tolerance**2
            tied_candidates = better_candidates[tied]
            if tied_candidates.size == 1:
                nearest = int(tied_candidates[0])
            else:
                nearest = self._break_parent_tie(tied_candidates, fitness, self.optimization_sense)
            parent[node_id] = nearest + 1
            parent_distance[node_id] = float(np.sqrt(best_dist_sq))
        return parent, parent_distance

    def _compute_parent_forest_torch(
        self,
        x: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        torch, device = self._select_torch_parent_device(x.shape[0], x.shape[1])
        if torch is None or device is None:
            return None
        count = x.shape[0]
        dtype = torch.float32 if str(device).startswith("mps") else torch.float64
        max_block_cells = (
            NBN_TORCH_MPS_MAX_BLOCK_CELLS
            if str(device).startswith("mps")
            else NBN_TORCH_MAX_BLOCK_CELLS
        )
        ordered_ids = self._sort_sample_ids(fitness, self.optimization_sense)
        parent = np.zeros(count, dtype=int)
        parent_distance = np.full(count, np.nan, dtype=float)
        try:
            with torch.inference_mode():
                x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)
                fitness_t = torch.as_tensor(np.ascontiguousarray(fitness), dtype=dtype, device=device)
                ordered_t = torch.as_tensor(ordered_ids, dtype=torch.long, device=device)
                inf = torch.tensor(float("inf"), dtype=dtype, device=device)
                start = 0
                while start < count:
                    candidate_count = max(1, start + 1)
                    block_size = max(1, min(count - start, max_block_cells // candidate_count))
                    end = start + block_size
                    candidate_ids = ordered_t[:end]
                    target_ids = ordered_t[start:end]
                    candidate_x = x_t[candidate_ids]
                    target_x = x_t[target_ids]
                    diff = target_x[:, None, :] - candidate_x[None, :, :]
                    dist_sq = torch.sum(diff * diff, dim=2)
                    candidate_fitness = fitness_t[candidate_ids][None, :]
                    target_fitness = fitness_t[target_ids][:, None]
                    tolerance = float(self.fitness_tolerance)
                    if self.optimization_sense == "max":
                        better = candidate_fitness > target_fitness + tolerance
                    else:
                        better = candidate_fitness < target_fitness - tolerance
                    dist_sq = torch.where(better, dist_sq, inf)
                    best_dist_sq, best_positions = torch.min(dist_sq, dim=1)
                    valid = torch.isfinite(best_dist_sq)
                    if bool(torch.any(valid).detach().cpu().item()):
                        valid_np = valid.detach().cpu().numpy().astype(bool)
                        node_np = target_ids.detach().cpu().numpy()
                        parent_np = candidate_ids[best_positions].detach().cpu().numpy()
                        distance_np = best_dist_sq.detach().cpu().numpy()
                        selected_nodes = node_np[valid_np]
                        parent[selected_nodes] = parent_np[valid_np] + 1
                        parent_distance[selected_nodes] = np.sqrt(np.maximum(distance_np[valid_np], 0.0))
                    start = end
            return parent, parent_distance
        except Exception:
            return None


    @staticmethod
    def _set_3d_axes_tight(axes: Any, layout: np.ndarray, z_values: np.ndarray) -> None:
        if layout.size == 0 or z_values.size == 0:
            return
        xlim = NBNContinuous._padded_limits(layout[:, 0])
        ylim = NBNContinuous._padded_limits(layout[:, 1])
        zlim = NBNContinuous._padded_limits(z_values)
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)
        axes.set_zlim(*zlim)
        try:
            axes.set_box_aspect((1.0, 1.0, 1.0), zoom=1.08)
        except TypeError:
            try:
                axes.set_box_aspect((1.0, 1.0, 1.0))
            except AttributeError:
                pass
        except AttributeError:
            pass


    @staticmethod
    def _set_2d_axes_tight(axes: Any, layout: np.ndarray) -> None:
        if layout.size == 0:
            return
        xlim = NBNContinuous._padded_limits(layout[:, 0])
        ylim = NBNContinuous._padded_limits(layout[:, 1])
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)


    @staticmethod
    def _align_three_view_axes_boxes(axes: Sequence[Any]) -> None:
        reference = axes[0].get_position()
        for axes_item in axes[1:]:
            position = axes_item.get_position()
            axes_item.set_position([position.x0, reference.y0, reference.width, reference.height])


    @staticmethod
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


    @staticmethod
    def _padded_limits(values: np.ndarray) -> tuple[float, float]:
        low = float(np.min(values))
        high = float(np.max(values))
        if np.isclose(low, high):
            pad = max(1.0, abs(low) * 0.05)
        else:
            pad = 0.04 * (high - low)
        return low - pad, high + pad


    @staticmethod
    def _validate_samples(x: Any, fitness: Any) -> tuple[np.ndarray, np.ndarray]:
        data = np.asarray(x, dtype=float)
        if data.ndim != 2 or data.size == 0:
            raise ValueError("x must be a non-empty 2D matrix of shape [N, D].")
        values = np.asarray(fitness, dtype=float).reshape(-1)
        if values.size != data.shape[0]:
            raise ValueError("fitness must provide exactly one value per sample.")
        return data, values


    @staticmethod
    def _validate_query_positions(positions: Any, dimension: int) -> np.ndarray:
        query = np.asarray(positions, dtype=float)
        if query.size == 0:
            return np.zeros((0, dimension), dtype=float)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != dimension:
            raise ValueError(f"Expected query positions with dimension {dimension}, got {query.shape}.")
        return query


    @staticmethod
    def _resolve_environment(environment: Any | None, count: int) -> np.ndarray:
        if environment is None or np.asarray(environment).size == 0:
            return np.ones(count, dtype=int)
        env = np.asarray(environment, dtype=int).reshape(-1)
        if env.size == 1:
            return np.full(count, int(env[0]), dtype=int)
        if env.size != count:
            raise ValueError("environment must be scalar or have one value per sample.")
        return env


    @staticmethod
    def _normalize_sense(value: str) -> OptimizationSense:
        lowered = str(value).strip().lower()
        if lowered in {"max", "maximize", "maximization"}:
            return "max"
        if lowered in {"min", "minimize", "minimization"}:
            return "min"
        raise ValueError("optimization_sense must be 'max' or 'min'.")


    @staticmethod
    def _sort_sample_ids(fitness: np.ndarray, sense: OptimizationSense) -> np.ndarray:
        sample_ids = np.arange(fitness.size)
        score = -fitness if sense == "max" else fitness
        return np.lexsort((sample_ids, score))


    @staticmethod
    def _is_better(
        candidate_fitness: np.ndarray,
        base_fitness: float,
        sense: OptimizationSense,
        tolerance: float,
    ) -> np.ndarray:
        if sense == "max":
            return candidate_fitness > base_fitness + tolerance
        return candidate_fitness < base_fitness - tolerance


    @staticmethod
    def _break_parent_tie(
        candidate_ids: np.ndarray,
        fitness: np.ndarray,
        sense: OptimizationSense,
    ) -> int:
        score = -fitness[candidate_ids] if sense == "max" else fitness[candidate_ids]
        order = np.lexsort((candidate_ids, score))
        return int(candidate_ids[order[0]])


    @staticmethod
    def _trace_forest(parent: np.ndarray, parent_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = parent.size
        root = np.zeros(count, dtype=int)
        depth = np.zeros(count, dtype=int)
        path_distance = np.zeros(count, dtype=float)
        for idx in range(count):
            current = idx
            seen: set[int] = set()
            while parent[current] > 0 and current not in seen:
                seen.add(current)
                path_distance[idx] += float(parent_distance[current])
                current = int(parent[current]) - 1
                depth[idx] += 1
            root[idx] = current + 1
        return root, depth, path_distance


    @staticmethod
    def _assign_basins(root: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordered_root_ids: list[int] = []
        seen: set[int] = set()
        for value in root:
            root_id = int(value)
            if root_id not in seen:
                ordered_root_ids.append(root_id)
                seen.add(root_id)
        root_ids = np.asarray(ordered_root_ids, dtype=int)
        basin_map = {root_id: index + 1 for index, root_id in enumerate(root_ids)}
        basin = np.asarray([basin_map[int(value)] for value in root], dtype=int)
        return basin, root_ids


    @staticmethod
    def _build_children(parent: np.ndarray) -> list[list[int]]:
        children = [[] for _ in range(parent.size)]
        for child_index, parent_id in enumerate(parent, start=1):
            if parent_id > 0:
                children[int(parent_id) - 1].append(child_index)
        return children


    @staticmethod
    def _normalize_fitness(fitness: np.ndarray, sense: OptimizationSense) -> np.ndarray:
        values = np.asarray(fitness, dtype=float)
        if sense == "min":
            values = -values
        low = float(np.min(values))
        high = float(np.max(values))
        if np.isclose(low, high):
            return np.ones_like(values)
        return (values - low) / (high - low)


    @staticmethod
    def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2))

    @staticmethod
    def _select_torch_parent_device(count: int, dimension: int) -> tuple[Any | None, Any | None]:
        try:
            import torch
        except ImportError:
            return None, None
        work_items = int(count) * max(1, int(dimension))
        if torch.cuda.is_available():
            return torch, torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and work_items >= 10_000:
            return torch, torch.device("mps")
        if int(count) >= 2_000:
            return torch, torch.device("cpu")
        return None, None


    @staticmethod
    def _fallback_layout(count: int) -> np.ndarray:
        if count == 0:
            return np.zeros((0, 2), dtype=float)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))


    @staticmethod
    def _build_radial_layout(
        parent: np.ndarray,
        children: list[list[int]],
        parent_distance: np.ndarray,
        root_ids: np.ndarray,
    ) -> np.ndarray:
        count = parent.size
        if count == 0:
            return np.zeros((0, 2), dtype=float)
        root_ids = np.asarray(root_ids, dtype=int).reshape(-1)
        root_ids = root_ids[(root_ids >= 1) & (root_ids <= count)]
        if root_ids.size == 0:
            return NBNContinuous._fallback_layout(count)

        finite_distance = parent_distance[np.isfinite(parent_distance) & (parent_distance >= 0)]
        if finite_distance.size == 0:
            max_distance = 1.0
        else:
            max_distance = 2.0 * float(np.max(finite_distance))
            if not (np.isfinite(max_distance) and max_distance > 0):
                max_distance = 1.0

        edge_radius = np.zeros(count, dtype=float)
        edge_mask = parent > 0
        edge_radius[edge_mask] = parent_distance[edge_mask]
        edge_radius[~np.isfinite(edge_radius)] = max_distance
        if root_ids.size > 1:
            edge_radius[root_ids - 1] = max_distance
        else:
            edge_radius[root_ids - 1] = 0.0
        edge_radius = edge_radius / max_distance

        if root_ids.size <= 1:
            layout_root_id = int(root_ids[0])
            parent_ext = parent.astype(int).copy()
            children_ext = [list(child_ids) for child_ids in children]
            edge_radius_ext = edge_radius.copy()
            layout_ext = np.zeros((count, 2), dtype=float)
        else:
            layout_root_id = count + 1
            parent_ext = np.concatenate([parent.astype(int).copy(), np.array([0], dtype=int)])
            children_ext = [list(child_ids) for child_ids in children]
            children_ext.append(root_ids.astype(int).tolist())
            edge_radius_ext = np.concatenate([edge_radius, np.array([0.0])])
            layout_ext = np.zeros((count + 1, 2), dtype=float)
            parent_ext[root_ids - 1] = layout_root_id

        breadth_first_order = NBNContinuous._build_breadth_first_order(
            children_ext,
            layout_root_id,
        )
        leaf_count = np.ones(parent_ext.size, dtype=float)
        for node_id in reversed(breadth_first_order):
            child_ids = children_ext[int(node_id) - 1]
            if child_ids:
                leaf_count[int(node_id) - 1] = max(
                    1.0,
                    float(np.sum(leaf_count[np.asarray(child_ids, dtype=int) - 1])),
                )

        from_angle = np.zeros(parent_ext.size, dtype=float)
        to_angle = np.zeros(parent_ext.size, dtype=float)
        root_children = children_ext[layout_root_id - 1]
        to_angle[layout_root_id - 1] = 0.5 if len(root_children) == 1 else 1.0

        for parent_id in breadth_first_order:
            parent_zero = int(parent_id) - 1
            child_ids = children_ext[parent_zero]
            if not child_ids:
                continue
            theta_length = (to_angle[parent_zero] - from_angle[parent_zero]) / leaf_count[parent_zero]
            current_theta = from_angle[parent_zero]
            for child_id in child_ids:
                child_zero = int(child_id) - 1
                from_angle[child_zero] = current_theta
                current_theta = current_theta + theta_length * leaf_count[child_zero]
                to_angle[child_zero] = current_theta

        layout_ext[layout_root_id - 1] = np.array([0.0, 0.0])
        for node_id in breadth_first_order[1:]:
            node_zero = int(node_id) - 1
            parent_id = int(parent_ext[node_zero])
            angle = (from_angle[node_zero] + to_angle[node_zero]) / 2.0
            offset = NBNContinuous._polar_offset(edge_radius_ext[node_zero], angle)
            layout_ext[node_zero] = layout_ext[parent_id - 1] + offset

        radius = np.sqrt(np.sum(layout_ext[:, :2] ** 2, axis=1))
        max_radius = float(np.max(radius)) if radius.size else 0.0
        if np.isfinite(max_radius) and max_radius > 0:
            layout_ext[:, :2] = layout_ext[:, :2] / max_radius

        return layout_ext[:count]


    @staticmethod
    def _build_breadth_first_order(children: list[list[int]], root_id: int) -> list[int]:
        order: list[int] = []
        queue: list[int] = [int(root_id)]
        visited = {int(root_id)}
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for child_id in children[node_id - 1]:
                child_id = int(child_id)
                if child_id in visited:
                    continue
                queue.append(child_id)
                visited.add(child_id)
        return order


    @staticmethod
    def _polar_offset(radius: float, angle: float) -> np.ndarray:
        theta = float(angle) * 2.0 * np.pi
        return np.array([float(radius) * np.sin(theta), float(radius) * np.cos(theta)])
