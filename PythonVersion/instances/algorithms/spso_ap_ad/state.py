"""SPSO_AP_AD state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SPSOAPADParticle:
    position: np.ndarray
    pbest_position: np.ndarray
    velocity: np.ndarray
    fitness_value: float
    pbest_fitness: float
    processed: int = 0
    shifts: list[float] = field(default_factory=list)
    pbest_past_environment: np.ndarray | None = None

    @property
    def X(self) -> np.ndarray:
        return self.position

    @X.setter
    def X(self, value: np.ndarray) -> None:
        self.position = np.asarray(value, dtype=float).reshape(-1)

    @property
    def PbestPosition(self) -> np.ndarray:
        return self.pbest_position

    @PbestPosition.setter
    def PbestPosition(self, value: np.ndarray) -> None:
        self.pbest_position = np.asarray(value, dtype=float).reshape(-1)

    @property
    def Velocity(self) -> np.ndarray:
        return self.velocity

    @Velocity.setter
    def Velocity(self, value: np.ndarray) -> None:
        self.velocity = np.asarray(value, dtype=float).reshape(-1)

    @property
    def FitnessValue(self) -> float:
        return self.fitness_value

    @FitnessValue.setter
    def FitnessValue(self, value: float) -> None:
        self.fitness_value = float(value)

    @property
    def PbestFitness(self) -> float:
        return self.pbest_fitness

    @PbestFitness.setter
    def PbestFitness(self, value: float) -> None:
        self.pbest_fitness = float(value)

    @property
    def Processed(self) -> int:
        return self.processed

    @Processed.setter
    def Processed(self, value: int) -> None:
        self.processed = int(value)

    @property
    def Shifts(self) -> list[float]:
        return self.shifts

    @Shifts.setter
    def Shifts(self, value: float | list[float] | np.ndarray | None) -> None:
        if value is None:
            self.shifts = []
            return
        values = np.asarray(value, dtype=float).reshape(-1)
        self.shifts = [float(item) for item in values if np.isfinite(item)]

    @property
    def Pbest_past_environment(self) -> np.ndarray | None:
        return self.pbest_past_environment

    @Pbest_past_environment.setter
    def Pbest_past_environment(self, value: np.ndarray | None) -> None:
        if value is None:
            self.pbest_past_environment = None
        else:
            self.pbest_past_environment = np.asarray(value, dtype=float).reshape(-1)


@dataclass(slots=True)
class SPSOAPADSpecies:
    seed: int
    member: list[int]
    remove: int = 0
    Active: int = 1
    distance: float = 0.0

    @property
    def active(self) -> int:
        return self.Active

    @active.setter
    def active(self, value: int) -> None:
        self.Active = int(value)
