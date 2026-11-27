"""AMLP translated from Ahrari's public Prediction.m and DynamicManager.m.

Source: EvOpt AMLP-RS-CMSA-ESII (2022), CC BY-NC-SA 4.0.
The distributed main.m selects at most ten prediction levels.
Coordinates supplied here are normalized by the search-domain widths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Forecast:
    positions: np.ndarray
    error_norms: np.ndarray
    level: int  # One-based, as in the author's MATLAB interface.


def predict(history: np.ndarray, max_level: int) -> Forecast:
    """Predict each level and select it using its latest available error.

    The highest estimable level borrows the preceding level's error when it
    has no observed error yet. The author's tie rule then promotes that level
    if the preceding one would otherwise be selected.
    """
    count, dimension = history.shape
    levels = min(max_level, count)
    predictions = np.full((levels, count + 1, dimension), np.nan)
    errors = np.full((levels, count, dimension), np.nan)
    recent = max(0, count - 1 - max_level)
    for time in range(recent, count):
        predictions[0, time + 1] = history[time]
        if time < count - 1:
            errors[0, time + 1] = history[time + 1] - predictions[0, time + 1]
    for level in range(1, levels):
        for time in range(max(recent, level), count):
            predictions[level, time + 1] = (
                predictions[level - 1, time + 1] + errors[level - 1, time]
            )
            if time < count - 1:
                errors[level, time + 1] = history[time + 1] - predictions[level, time + 1]
    if levels == count and levels > 1:
        errors[-1, -1] = errors[-2, -1]
    norms = np.linalg.norm(errors[:, -1], axis=1)
    best = 0 if count == 1 else int(np.argmin(norms))
    if count > 1 and best == levels - 2 and levels == count:
        best += 1
    return Forecast(predictions[:, -1].copy(), norms, best + 1)


def track_history(archives: list[np.ndarray], index: int) -> np.ndarray:
    """Backtrack a current solution through nearest neighbours in each archive."""
    history = np.empty((len(archives), archives[-1].shape[1]))
    history[-1] = archives[-1][index]
    for time in range(len(archives) - 2, -1, -1):
        distances = np.linalg.norm(archives[time] - history[time + 1], axis=1)
        history[time] = archives[time][int(np.argmin(distances))]
    return history


def recommendations(
    archives: list[np.ndarray], max_level: int, max_sigma: float, error_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized restart centres, step sizes and one-based levels."""
    available = 0
    for archive in reversed(archives[-(max_level + 1):]):
        if len(archive) == 0:
            break
        available += 1
    current = archives[-1]
    if available == 0:
        return current.copy(), np.empty(0), np.empty(0, dtype=int)
    if available == 1:
        return current.copy(), np.full(len(current), max_sigma), np.ones(len(current), dtype=int)
    centres = np.empty_like(current)
    sigmas = np.empty(len(current))
    chosen = np.empty(len(current), dtype=int)
    for index in range(len(current)):
        forecast = predict(track_history(archives[-available:], index), min(max_level, available))
        chosen[index] = forecast.level
        centres[index] = forecast.positions[forecast.level - 1]
        sigmas[index] = np.clip(
            error_multiplier * forecast.error_norms[forecast.level - 1], max_sigma * 1e-6, max_sigma,
        )
    return centres, sigmas, chosen
