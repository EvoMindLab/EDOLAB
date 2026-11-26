"""Shared remote-environment dependency admission rules.

The full EDOLAB registry imports NumPy, SciPy, scikit-learn, and PyTorch before
a task can start.  These checks only import already-installed modules; they do
not install packages or resolve dependencies on the remote server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RuntimeDependencySpec:
    key: str
    distribution: str
    label: str
    imports: tuple[str, ...]
    required_attributes: tuple[tuple[str, str], ...] = ()

    def probe_tuple(
        self,
    ) -> tuple[str, str, str, tuple[str, ...], tuple[tuple[str, str], ...]]:
        return (
            self.key,
            self.distribution,
            self.label,
            self.imports,
            self.required_attributes,
        )


@dataclass(frozen=True, slots=True)
class RuntimeDependencyStatus:
    spec: RuntimeDependencySpec
    reported: bool
    available: bool
    version: str | None
    error: str | None

    @property
    def detail(self) -> str:
        if self.available:
            suffix = f" {self.version}" if self.version else ""
            return f"{self.spec.label}{suffix} imports successfully in the configured environment."
        if not self.reported:
            return (
                f"The latest probe did not verify {self.spec.label}; run the environment "
                "check again before scheduling."
            )
        if self.error:
            return f"{self.spec.label} import failed: {self.error}"
        return f"{self.spec.label} could not be imported in the configured environment."


REGISTRY_RUNTIME_DEPENDENCIES: tuple[RuntimeDependencySpec, ...] = (
    RuntimeDependencySpec(
        key="numpy",
        distribution="numpy",
        label="NumPy",
        imports=("numpy",),
    ),
    RuntimeDependencySpec(
        key="scipy",
        distribution="scipy",
        label="SciPy",
        imports=("scipy", "scipy.cluster.hierarchy", "scipy.io", "scipy.linalg", "scipy.spatial"),
    ),
    RuntimeDependencySpec(
        key="sklearn",
        distribution="scikit-learn",
        label="scikit-learn",
        imports=("sklearn", "sklearn.cluster"),
        required_attributes=(("sklearn.cluster", "HDBSCAN"),),
    ),
    RuntimeDependencySpec(
        key="torch",
        distribution="torch",
        label="PyTorch",
        imports=("torch",),
    ),
)


def runtime_dependency_probe_specs(
) -> tuple[tuple[str, str, str, tuple[str, ...], tuple[tuple[str, str], ...]], ...]:
    """Return a literal-safe shape embedded in the read-only SSH probe."""

    return tuple(spec.probe_tuple() for spec in REGISTRY_RUNTIME_DEPENDENCIES)


def runtime_dependency_statuses(
    environment_python: object,
) -> tuple[RuntimeDependencyStatus, ...]:
    """Normalize a probe report, treating legacy/missing evidence as failed."""

    dependencies: Mapping[str, Any] | None = None
    if isinstance(environment_python, Mapping):
        candidate = environment_python.get("runtime_dependencies")
        if isinstance(candidate, Mapping):
            dependencies = candidate

    statuses: list[RuntimeDependencyStatus] = []
    for spec in REGISTRY_RUNTIME_DEPENDENCIES:
        raw = dependencies.get(spec.key) if dependencies is not None else None
        reported = isinstance(raw, Mapping)
        statuses.append(
            RuntimeDependencyStatus(
                spec=spec,
                reported=reported,
                available=reported and raw.get("available") is True,
                version=_optional_text(raw.get("version")) if reported else None,
                error=_optional_text(raw.get("error")) if reported else None,
            )
        )
    return tuple(statuses)


def missing_runtime_dependency_messages(environment_python: object) -> list[str]:
    return [status.detail for status in runtime_dependency_statuses(environment_python) if not status.available]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
