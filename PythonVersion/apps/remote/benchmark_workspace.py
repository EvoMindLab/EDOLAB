"""Safe benchmark visualization boundary for the Relay application.

The numerical implementation remains in :mod:`core.benchmark.service`.  This
module deliberately owns only application-boundary concerns: request
validation, resource limits, managed artifact paths, and JSON-safe responses.
It has no dependency on the Relay database, execution controller, or SSH
transport.  Interactive requests call it directly in the local Relay process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
import shutil
import threading
from typing import Any, Mapping
from urllib.parse import quote
from uuid import uuid4

from core.benchmark import service as benchmark_service
from core.parameters import ParameterSpec
from core.registry import problem_registry


CONTOUR_RESOLUTION_MIN = 8
CONTOUR_RESOLUTION_MAX = 240
SURFACE_RESOLUTION_MIN = 8
SURFACE_RESOLUTION_MAX = 140
NBN_SAMPLE_COUNT_MIN = 20
# Interactive NBN requests execute synchronously in the local Relay process.
# Keep their quadratic nearest-better construction bounded independently from
# the larger limit retained by the reusable core/CLI service.
NBN_SAMPLE_COUNT_MAX = min(5_000, benchmark_service.NBN_SAMPLE_COUNT_MAX)
NBN_INTERACTIVE_COORDINATE_BUDGET = 500_000
SEED_MAX = (1 << 63) - 1
STRING_PARAMETER_MAX_LENGTH = 16_384
DEFAULT_MAX_ARTIFACT_DIRECTORIES = 128
_MANAGED_ARTIFACT_FILENAMES = frozenset({"contour.png", "nbn-three-views.png"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class BenchmarkWorkspaceError(ValueError):
    """Raised when an application-level benchmark request is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    """Validated, canonical input shared by all benchmark operations."""

    problem_name: str
    problem_params: dict[str, Any]
    environment_number: int
    seed: int | None
    resolution: int
    sample_count: int


class BenchmarkWorkspace:
    """Expose benchmark plots and models through a bounded artifact workspace."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        artifact_url_prefix: str = "/api/benchmarks/artifacts",
        max_nbn_samples: int = NBN_SAMPLE_COUNT_MAX,
        max_artifact_directories: int = DEFAULT_MAX_ARTIFACT_DIRECTORIES,
    ) -> None:
        root = Path(artifact_root).expanduser()
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except FileExistsError as exc:
            raise BenchmarkWorkspaceError(
                "artifact_root must be a regular directory, not a symlink or file."
            ) from exc
        if root.is_symlink() or not root.is_dir():
            raise BenchmarkWorkspaceError(
                "artifact_root must be a regular directory, not a symlink or file."
            )
        root.chmod(0o700)
        self.artifact_root = root.resolve(strict=True)
        self.artifact_url_prefix = "/" + artifact_url_prefix.strip("/")
        self.max_nbn_samples = _strict_int(
            max_nbn_samples,
            name="max_nbn_samples",
            minimum=NBN_SAMPLE_COUNT_MIN,
            maximum=NBN_SAMPLE_COUNT_MAX,
        )
        self.max_artifact_directories = _strict_int(
            max_artifact_directories,
            name="max_artifact_directories",
            minimum=1,
            maximum=10_000,
        )
        self._artifact_lock = threading.Lock()

    def limits(self) -> dict[str, int]:
        """Return public request limits for forms and API schema generation."""
        return {
            "contour_resolution_min": CONTOUR_RESOLUTION_MIN,
            "contour_resolution_max": CONTOUR_RESOLUTION_MAX,
            "surface_resolution_min": SURFACE_RESOLUTION_MIN,
            "surface_resolution_max": SURFACE_RESOLUTION_MAX,
            "nbn_sample_count_min": NBN_SAMPLE_COUNT_MIN,
            "nbn_sample_count_max": self.max_nbn_samples,
            "seed_max": SEED_MAX,
        }

    def list_problems(self) -> list[dict[str, Any]]:
        """List registered problems with JSON-safe parameter specifications."""
        entries: list[dict[str, Any]] = []
        for item in benchmark_service.list_problem_specs():
            specs = item.get("parameters", [])
            default_dimension = next(
                (spec.get("default") for spec in specs if spec.get("name") == "Dimension"),
                None,
            )
            entries.append(
                {
                    **item,
                    "default_dimension": default_dimension,
                    "supports_2d": any(
                        spec.get("name") == "Dimension"
                        and _bounds_include(spec.get("bounds"), 2)
                        for spec in specs
                    ),
                    "supports_evaluation_device": any(
                        spec.get("name") == "EvaluationDevice" for spec in specs
                    ),
                }
            )
        return _json_safe(entries)

    def validate_request(
        self,
        payload: Mapping[str, Any],
        *,
        operation: str,
    ) -> BenchmarkRequest:
        """Validate one local interactive request before calculation."""
        if operation not in {"contour", "surface", "nbn_views", "nbn_model"}:
            raise BenchmarkWorkspaceError(f"Unknown benchmark operation {operation!r}.")
        return self._validate_request(payload, operation=operation)

    def render_contour(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self.validate_request(payload, operation="contour")
        run_dir, target = self._new_artifact_target("contour")
        try:
            metadata = benchmark_service.render_landscape_contour(
                problem_name=request.problem_name,
                parameters=request.problem_params,
                output_path=target,
                environment_number=request.environment_number,
                seed=request.seed,
                resolution=request.resolution,
            )
            artifact = self._artifact_payload(target, run_dir.name)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return _json_safe({**metadata, "artifact": artifact})

    def build_surface(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self.validate_request(payload, operation="surface")
        model = benchmark_service.build_surface_model(
            problem_name=request.problem_name,
            parameters=request.problem_params,
            environment_number=request.environment_number,
            seed=request.seed,
            resolution=request.resolution,
        )
        return _json_safe(model)

    def render_nbn_views(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self.validate_request(payload, operation="nbn_views")
        run_dir, target = self._new_artifact_target("nbn-three-views")
        try:
            metadata = benchmark_service.render_nbn_three_views(
                problem_name=request.problem_name,
                parameters=request.problem_params,
                output_path=target,
                environment_number=request.environment_number,
                seed=request.seed,
                sample_count=request.sample_count,
            )
            artifact = self._artifact_payload(target, run_dir.name)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return _json_safe({**metadata, "artifact": artifact})

    def build_nbn_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self.validate_request(payload, operation="nbn_model")
        model = benchmark_service.build_nbn_model(
            problem_name=request.problem_name,
            parameters=request.problem_params,
            environment_number=request.environment_number,
            seed=request.seed,
            sample_count=request.sample_count,
        )
        return _json_safe(model)

    def resolve_artifact(self, storage_key: str) -> Path:
        """Resolve a public storage key without allowing traversal or symlinks."""
        if not isinstance(storage_key, str) or not storage_key.strip():
            raise BenchmarkWorkspaceError("artifact storage_key must be a non-empty string.")
        if "\x00" in storage_key or "\\" in storage_key:
            raise BenchmarkWorkspaceError("artifact storage_key contains invalid characters.")
        relative = Path(storage_key)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise BenchmarkWorkspaceError("artifact storage_key must be a safe relative path.")
        if (
            len(relative.parts) != 2
            or len(relative.parts[0]) != 32
            or any(character not in "0123456789abcdef" for character in relative.parts[0])
            or relative.parts[1] not in _MANAGED_ARTIFACT_FILENAMES
        ):
            raise BenchmarkWorkspaceError(
                "artifact storage_key is not a managed benchmark key."
            )
        candidate = self.artifact_root.joinpath(relative)
        cursor = self.artifact_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise BenchmarkWorkspaceError("artifact storage_key must not traverse symlinks.")
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Benchmark artifact does not exist: {storage_key}") from exc
        if not resolved.is_relative_to(self.artifact_root):
            raise BenchmarkWorkspaceError("artifact storage_key escapes the managed workspace.")
        if not resolved.is_file():
            raise BenchmarkWorkspaceError("artifact storage_key does not identify a PNG file.")
        _validate_png_signature(resolved)
        return resolved

    def _validate_request(self, payload: Mapping[str, Any], *, operation: str) -> BenchmarkRequest:
        if not isinstance(payload, Mapping):
            raise BenchmarkWorkspaceError("Benchmark payload must be an object.")
        allowed = {
            "problem_name",
            "problem_params",
            "environment_number",
            "seed",
            "resolution",
            "sample_count",
            "evaluation_device",
        }
        unknown_fields = sorted(set(payload) - allowed, key=str)
        if unknown_fields:
            raise BenchmarkWorkspaceError(
                "Unknown benchmark field(s): " + ", ".join(str(item) for item in unknown_fields)
            )

        problem_name = payload.get("problem_name")
        if not isinstance(problem_name, str) or not problem_name.strip():
            raise BenchmarkWorkspaceError("problem_name must be a non-empty string.")
        problem_name = problem_name.strip()
        if problem_name not in problem_registry:
            available = ", ".join(problem_registry.names())
            raise BenchmarkWorkspaceError(
                f"Unknown problem {problem_name!r}. Available: {available}."
            )

        problem_class = problem_registry.get(problem_name)
        specs = problem_class.parameter_specs()
        raw_params = payload.get("problem_params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, Mapping):
            raise BenchmarkWorkspaceError("problem_params must be an object.")
        unknown_params = sorted(set(raw_params) - set(specs), key=str)
        if unknown_params:
            raise BenchmarkWorkspaceError(
                "Unknown problem parameter(s): " + ", ".join(str(item) for item in unknown_params)
            )
        problem_params = {
            name: _coerce_parameter(raw_params[name], specs[name])
            for name in raw_params
        }

        evaluation_device = payload.get("evaluation_device")
        if evaluation_device is not None:
            if "EvaluationDevice" not in specs:
                if evaluation_device != "auto":
                    raise BenchmarkWorkspaceError(
                        f"{problem_name} does not expose an EvaluationDevice parameter."
                    )
            else:
                problem_params["EvaluationDevice"] = _coerce_parameter(
                    evaluation_device,
                    specs["EvaluationDevice"],
                )

        resolved_params = {name: spec.default for name, spec in specs.items()}
        resolved_params.update(problem_params)
        environment_count = _strict_int(
            resolved_params.get("EnvironmentNumber", 1),
            name="EnvironmentNumber",
            minimum=1,
        )
        environment_number = _strict_int(
            payload.get("environment_number", 1),
            name="environment_number",
            minimum=1,
            maximum=environment_count,
        )
        seed_value = payload.get("seed")
        seed = None if seed_value is None else _strict_int(
            seed_value,
            name="seed",
            minimum=0,
            maximum=SEED_MAX,
        )

        if operation == "contour":
            resolution = _strict_int(
                payload.get("resolution", 120),
                name="resolution",
                minimum=CONTOUR_RESOLUTION_MIN,
                maximum=CONTOUR_RESOLUTION_MAX,
            )
        elif operation == "surface":
            resolution = _strict_int(
                payload.get("resolution", 64),
                name="resolution",
                minimum=SURFACE_RESOLUTION_MIN,
                maximum=SURFACE_RESOLUTION_MAX,
            )
        else:
            resolution = _strict_int(
                payload.get("resolution", 64),
                name="resolution",
                minimum=SURFACE_RESOLUTION_MIN,
                maximum=SURFACE_RESOLUTION_MAX,
            )
        sample_count = _strict_int(
            payload.get("sample_count", 800),
            name="sample_count",
            minimum=NBN_SAMPLE_COUNT_MIN,
            maximum=self.max_nbn_samples,
        )

        dimension = _strict_int(resolved_params.get("Dimension"), name="Dimension", minimum=1)
        if (
            operation in {"nbn_views", "nbn_model"}
            and sample_count * dimension > NBN_INTERACTIVE_COORDINATE_BUDGET
        ):
            coordinate_count = sample_count * dimension
            raise BenchmarkWorkspaceError(
                "NBN interactive preview budget allows at most "
                f"{NBN_INTERACTIVE_COORDINATE_BUDGET} sample coordinates "
                "(sample_count * Dimension); received "
                f"{sample_count} * {dimension} = {coordinate_count}."
            )
        if operation in {"contour", "surface"} and dimension != 2:
            raise BenchmarkWorkspaceError(
                f"{operation} requires Dimension == 2; received Dimension={dimension}."
            )
        return BenchmarkRequest(
            problem_name=problem_name,
            problem_params=problem_params,
            environment_number=environment_number,
            seed=seed,
            resolution=resolution,
            sample_count=sample_count,
        )

    def _new_artifact_target(self, stem: str) -> tuple[Path, Path]:
        with self._artifact_lock:
            self._prune_artifact_directories()
            run_dir = (self.artifact_root / uuid4().hex).resolve()
            if not run_dir.is_relative_to(self.artifact_root):
                raise BenchmarkWorkspaceError("Generated artifact directory escaped the workspace.")
            run_dir.mkdir(mode=0o700)
        return run_dir, run_dir / f"{stem}.png"

    def _prune_artifact_directories(self) -> None:
        """Keep the interactive preview cache bounded without a job ledger."""

        managed = sorted(
            (
                item
                for item in self.artifact_root.iterdir()
                if not item.is_symlink()
                and item.is_dir()
                and len(item.name) == 32
                and all(character in "0123456789abcdef" for character in item.name)
            ),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        keep_before_create = max(self.max_artifact_directories - 1, 0)
        for stale in managed[keep_before_create:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _artifact_payload(self, target: Path, artifact_id: str) -> dict[str, Any]:
        resolved = target.resolve(strict=True)
        if not resolved.is_relative_to(self.artifact_root) or not resolved.is_file():
            raise BenchmarkWorkspaceError("Benchmark renderer did not create a managed artifact.")
        _validate_png_signature(resolved)
        storage_key = resolved.relative_to(self.artifact_root).as_posix()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "id": artifact_id,
            "storage_key": storage_key,
            "filename": resolved.name,
            "media_type": "image/png",
            "size_bytes": resolved.stat().st_size,
            "sha256": digest.hexdigest(),
            "url": f"{self.artifact_url_prefix}/{quote(storage_key, safe='/')}",
            "created_at": datetime.now(UTC).isoformat(),
        }


def _validate_png_signature(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(len(_PNG_SIGNATURE))
    if header != _PNG_SIGNATURE:
        raise BenchmarkWorkspaceError("Benchmark renderer produced an invalid PNG artifact.")


def _coerce_parameter(value: Any, spec: ParameterSpec) -> Any:
    name = spec.name
    if value is None:
        raise BenchmarkWorkspaceError(f"Parameter {name} cannot be null.")
    if spec.kind == "integer":
        result: Any = _strict_int(value, name=name)
        _validate_bounds(result, spec)
        return result
    if spec.kind == "numeric":
        if isinstance(value, bool):
            raise BenchmarkWorkspaceError(f"Parameter {name} must be numeric, not boolean.")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise BenchmarkWorkspaceError(f"Parameter {name} must be numeric.") from exc
        if not math.isfinite(result):
            raise BenchmarkWorkspaceError(f"Parameter {name} must be finite.")
        _validate_bounds(result, spec)
        return result
    if spec.kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
            return value.strip().lower() in {"true", "1"}
        raise BenchmarkWorkspaceError(f"Parameter {name} must be boolean.")
    if spec.kind == "option":
        if not isinstance(value, str) or not _option_allowed(value, spec.bounds):
            raise BenchmarkWorkspaceError(
                f"Parameter {name} must be one of the declared options {spec.bounds!r}."
            )
        return value
    if spec.kind == "string":
        if not isinstance(value, str):
            raise BenchmarkWorkspaceError(f"Parameter {name} must be a string.")
        if "\x00" in value or len(value) > STRING_PARAMETER_MAX_LENGTH:
            raise BenchmarkWorkspaceError(f"Parameter {name} contains invalid or excessive text.")
        return value
    raise BenchmarkWorkspaceError(f"Parameter {name} has unsupported kind {spec.kind!r}.")


def _strict_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise BenchmarkWorkspaceError(f"{name} must be an integer, not boolean.")
    if not isinstance(value, (int, float, str)):
        raise BenchmarkWorkspaceError(f"{name} must be an integer.")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise BenchmarkWorkspaceError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkWorkspaceError(f"{name} must be an integer.") from exc
    if isinstance(value, str) and value.strip() != str(result):
        raise BenchmarkWorkspaceError(f"{name} must be an integer.")
    if minimum is not None and result < minimum:
        raise BenchmarkWorkspaceError(f"{name} must be at least {minimum}; received {result}.")
    if maximum is not None and result > maximum:
        raise BenchmarkWorkspaceError(f"{name} must be at most {maximum}; received {result}.")
    return result


def _validate_bounds(value: int | float, spec: ParameterSpec) -> None:
    bounds = spec.bounds
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        return
    lower, upper = bounds
    if isinstance(lower, (int, float)) and value < lower:
        raise BenchmarkWorkspaceError(
            f"Parameter {spec.name} must be at least {lower}; received {value}."
        )
    if isinstance(upper, (int, float)) and value > upper:
        raise BenchmarkWorkspaceError(
            f"Parameter {spec.name} must be at most {upper}; received {value}."
        )


def _option_allowed(value: str, bounds: Any) -> bool:
    if not isinstance(bounds, (tuple, list)):
        return False
    for option in bounds:
        if value == option:
            return True
        if isinstance(option, str) and option.endswith("<id>"):
            prefix = option[: -len("<id>")]
            suffix = value[len(prefix) :] if value.startswith(prefix) else ""
            if suffix.isdigit():
                return True
    return False


def _bounds_include(bounds: Any, value: int | float) -> bool:
    return (
        isinstance(bounds, (tuple, list))
        and len(bounds) == 2
        and isinstance(bounds[0], (int, float))
        and isinstance(bounds[1], (int, float))
        and bounds[0] <= value <= bounds[1]
    )


def _json_safe(value: Any) -> Any:
    """Return a structure accepted by ``json.dumps(..., allow_nan=False)``."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    # NumPy scalar/array values are intentionally handled without importing
    # NumPy into this application boundary.
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise BenchmarkWorkspaceError(f"Benchmark response contains unsupported value {type(value).__name__}.")
