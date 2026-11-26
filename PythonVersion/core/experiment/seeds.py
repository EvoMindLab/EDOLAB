"""Seed resolution policy."""

from __future__ import annotations

import secrets
from collections.abc import Sequence


SeedInput = int | Sequence[int] | None


def resolve_seed_sequence(seed: SeedInput, run_number: int) -> list[int]:
    if run_number <= 0:
        raise ValueError("run_number must be positive.")
    if seed is None:
        return [secrets.randbits(31) for _ in range(run_number)]
    if isinstance(seed, int):
        return [seed + index for index in range(run_number)]
    values = [int(value) for value in seed]
    if len(values) != run_number:
        raise ValueError(
            f"Seed vector length must equal run_number. Got {len(values)} for {run_number} runs."
        )
    return values


def parse_seed_text(text: str | None) -> SeedInput:
    if text is None:
        return None
    stripped = text.strip()
    if stripped.lower() in {"", "none", "random"}:
        return None
    if "," in stripped:
        return [int(item.strip()) for item in stripped.split(",") if item.strip()]
    return int(stripped)

