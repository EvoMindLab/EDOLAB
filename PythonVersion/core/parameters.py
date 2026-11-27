"""Parameter helpers for the OO shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    default: Any
    kind: str
    bounds: Any = None
    description: str = ""


def resolve_parameters(
    specs: dict[str, ParameterSpec],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    unknown = [name for name in overrides if name not in specs]
    if unknown:
        raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    values = {name: spec.default for name, spec in specs.items()}
    values.update(overrides)
    return values


def parse_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_key_value(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Expected KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Expected non-empty key in {text!r}")
    return key, parse_value(value.strip())
