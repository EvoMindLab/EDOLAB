"""Backend-agnostic benchmark visualization service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import instances  # noqa: F401  # populate problem_registry for standalone use
from core.parameters import ParameterSpec
from core.registry import problem_registry

NBN_SAMPLE_COUNT_MAX = 100_000


def list_problem_specs() -> list[dict[str, Any]]:
    """Return registered problem parameter specs as JSON-ready dictionaries."""
    problems: list[dict[str, Any]] = []
    for problem_name in problem_registry.names():
        problem_class = problem_registry.get(problem_name)
        specs = problem_class.parameter_specs()
        problems.append(
            {
                "name": problem_name,
                "parameters": [_parameter_spec_payload(spec) for spec in specs.values()],
            }
        )
    return problems


def render_landscape_contour(
    *,
    problem_name: str,
    parameters: dict[str, Any] | None,
    output_path: str | Path,
    environment_number: int = 1,
    seed: int | None = None,
    resolution: int = 120,
) -> dict[str, Any]:
    """Render a 2D landscape contour to ``output_path``."""
    _use_agg_backend()
    problem = _create_problem(problem_name, parameters, seed=seed)
    if problem.dimension != 2:
        raise ValueError("2D contour requires Dimension == 2.")
    import matplotlib.pyplot as plt

    axes = problem.landscape_visualize(
        environment_number,
        resolution=_clamp_int(resolution, 8, 240),
        output_path=output_path,
    )
    figure = axes.figure
    plt.close(figure)
    return _problem_payload(problem, environment_number, seed=seed, kind="contour")


def build_surface_model(
    *,
    problem_name: str,
    parameters: dict[str, Any] | None,
    environment_number: int = 1,
    seed: int | None = None,
    resolution: int = 64,
) -> dict[str, Any]:
    """Build an interactive x1/x2/fitness surface model.

    The web workbench only exposes full 2D benchmark surfaces. Higher
    dimensional landscapes should be inspected through NBN views instead.
    """
    problem = _create_problem(problem_name, parameters, seed=seed)
    if problem.dimension != 2:
        raise ValueError("Interactive surface requires Dimension == 2.")
    env_number = _resolve_environment_number(problem, environment_number)
    resolution = _clamp_int(resolution, 8, 140)
    bounds = problem.bounds
    x_axis = np.linspace(float(bounds.lower[0]), float(bounds.upper[0]), resolution)
    y_axis = np.linspace(float(bounds.lower[1]), float(bounds.upper[1]), resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    center = (np.asarray(bounds.lower, dtype=float) + np.asarray(bounds.upper, dtype=float)) / 2.0
    samples = np.tile(center.reshape(1, -1), (resolution * resolution, 1))
    samples[:, 0] = xx.reshape(-1)
    samples[:, 1] = yy.reshape(-1)
    values = problem.static_fitness(samples, env_number).reshape(resolution, resolution)
    finite = values[np.isfinite(values)]
    z_min = float(np.min(finite)) if finite.size else 0.0
    z_max = float(np.max(finite)) if finite.size else 0.0
    return {
        **_problem_payload(problem, env_number, seed=seed, kind="surface"),
        "x": x_axis.tolist(),
        "y": y_axis.tolist(),
        "z": values.tolist(),
        "z_min": z_min,
        "z_max": z_max,
        "slice_center": center.tolist(),
    }


def render_nbn_three_views(
    *,
    problem_name: str,
    parameters: dict[str, Any] | None,
    output_path: str | Path,
    environment_number: int = 1,
    seed: int | None = None,
    sample_count: int = 800,
) -> dict[str, Any]:
    """Render the NBN front/top/side views to ``output_path``."""
    _use_agg_backend()
    problem = _create_problem(problem_name, parameters, seed=seed)
    env_number = _resolve_environment_number(problem, environment_number)
    import matplotlib.pyplot as plt

    figure = problem.nbn_visualize(
        env_number,
        sample_count=_clamp_int(sample_count, 20, NBN_SAMPLE_COUNT_MAX),
        seed=seed,
        output_path=output_path,
    )
    plt.close(figure)
    return _problem_payload(problem, env_number, seed=seed, kind="nbn_three_views")


def build_nbn_model(
    *,
    problem_name: str,
    parameters: dict[str, Any] | None,
    environment_number: int = 1,
    seed: int | None = None,
    sample_count: int = 800,
) -> dict[str, Any]:
    """Build a compact interactive NBN 3D model."""
    problem = _create_problem(problem_name, parameters, seed=seed)
    env_number = _resolve_environment_number(problem, environment_number)
    network = problem.build_nbn(
        env_number,
        sample_count=_clamp_int(sample_count, 20, NBN_SAMPLE_COUNT_MAX),
        seed=seed,
        build_layout=True,
    )
    snapshot = network.get_snapshot(env_number)
    layout = snapshot.layout_2d if snapshot.layout_2d.size else np.zeros((snapshot.node_count, 2), dtype=float)
    normalized = np.asarray(snapshot.normalized_fitness, dtype=float).reshape(-1)
    if normalized.size == 0:
        normalized = np.zeros(snapshot.node_count, dtype=float)
    nodes = []
    for index, node_id in enumerate(snapshot.local_node_id):
        sample = snapshot.x[index].astype(float).tolist()
        nodes.append(
            {
                "id": int(node_id),
                "x": float(layout[index, 0]) if layout.size else 0.0,
                "y": float(layout[index, 1]) if layout.size else 0.0,
                "z": float(normalized[index]),
                "fitness": float(snapshot.fitness[index]),
                "normalized_fitness": float(normalized[index]),
                "basin": int(snapshot.basin[index]),
                "depth": int(snapshot.depth[index]),
                "is_root": bool(snapshot.parent[index] == 0),
                "sample": sample,
            }
        )
    edges = [
        {"source": int(child), "target": int(parent)}
        for child, parent in np.asarray(snapshot.edge_list, dtype=int).reshape(-1, 2)
    ]
    return {
        **_problem_payload(problem, env_number, seed=seed, kind="nbn_model"),
        "nodes": nodes,
        "edges": edges,
        "summary": snapshot.summary,
    }


def _create_problem(problem_name: str, parameters: dict[str, Any] | None, *, seed: int | None):
    problem_class = problem_registry.get(problem_name)
    coerced = _coerce_parameters(problem_class.parameter_specs(), parameters or {})
    problem = problem_class.create(coerced)
    problem.reset(seed)
    return problem


def _coerce_parameters(specs: dict[str, ParameterSpec], values: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in values.items():
        if key not in specs:
            coerced[key] = value
            continue
        coerced[key] = _coerce_value(value, specs[key])
    return coerced


def _coerce_value(value: Any, spec: ParameterSpec) -> Any:
    if value is None:
        return None
    if spec.kind == "integer":
        return int(value)
    if spec.kind == "numeric":
        return float(value)
    if spec.kind == "boolean":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    return value


def _problem_payload(problem: Any, environment_number: int, *, seed: int | None, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "problem_name": problem.name,
        "dimension": int(problem.dimension),
        "environment": _resolve_environment_number(problem, environment_number),
        "seed": seed,
        "problem_params": _jsonable(problem.parameters),
    }


def _parameter_spec_payload(spec: ParameterSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "default": _jsonable(spec.default),
        "kind": spec.kind,
        "bounds": _jsonable(spec.bounds),
        "description": spec.description,
    }


def _resolve_environment_number(problem: Any, environment_number: int) -> int:
    return int(problem.resolve_environment_index(int(environment_number)) + 1)


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _use_agg_backend() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
