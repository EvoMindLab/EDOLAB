"""SPSO_AP_AD algorithm instance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from core.algorithm.base import BaseAlgorithm
from core.parameters import ParameterSpec
from core.registry import algorithm_registry
from instances.algorithms.common import evaluate_
from instances.algorithms.spso_ap_ad.state import SPSOAPADParticle, SPSOAPADSpecies


@algorithm_registry.register("SPSO_AP_AD")
class SPSOAPADAlgorithm(BaseAlgorithm):
    name: ClassVar[str] = "SPSO_AP_AD"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {
            "InitialPopulationSize": ParameterSpec(
                "InitialPopulationSize",
                50,
                "integer",
                (1, 1000),
                "Total number of particles in the initial population.",
            ),
            "SwarmMember": ParameterSpec(
                "SwarmMember",
                5,
                "integer",
                (1, 100),
                "Number of particles selected for each species.",
            ),
            "NewlyAddedPopulationSize": ParameterSpec(
                "NewlyAddedPopulationSize",
                5,
                "integer",
                (0, 100),
                "Number of random particles added when converged species remain below Nmax.",
            ),
            "x": ParameterSpec(
                "x",
                0.729843788,
                "numeric",
                (0, 2),
                "Constriction factor used in the PSO velocity update.",
            ),
            "c1": ParameterSpec(
                "c1",
                2.05,
                "numeric",
                (0, 5),
                "Cognitive acceleration coefficient.",
            ),
            "c2": ParameterSpec(
                "c2",
                2.05,
                "numeric",
                (0, 5),
                "Social acceleration coefficient toward the species seed.",
            ),
            "rho": ParameterSpec(
                "rho",
                0.7,
                "numeric",
                (0, 1),
                "Ratio used for the maximum deactivation threshold.",
            ),
            "mu": ParameterSpec(
                "mu",
                0.2,
                "numeric",
                (0, 1),
                "Ratio used for the minimum deactivation threshold.",
            ),
            "beta": ParameterSpec(
                "beta",
                1,
                "numeric",
                (0, 1),
                "Adaptive deactivation scaling factor.",
            ),
            "gama": ParameterSpec(
                "gama",
                0.1,
                "numeric",
                (0, 1),
                "Decay factor for beta; spelling follows the MATLAB source.",
            ),
            "Nmax": ParameterSpec(
                "Nmax",
                30,
                "integer",
                (1, 100),
                "Maximum number of species before anti-convergence restarts a species.",
            ),
        }

    @property
    def Particle(self) -> list[SPSOAPADParticle]:
        return self.particles

    @Particle.setter
    def Particle(self, value: list[SPSOAPADParticle]) -> None:
        self.particles = value

    @property
    def Dimension(self) -> int:
        return self.dimension

    @Dimension.setter
    def Dimension(self, value: int) -> None:
        self.dimension = int(value)

    @property
    def MinCoordinate(self) -> np.ndarray:
        return self.min_coordinate

    @MinCoordinate.setter
    def MinCoordinate(self, value) -> None:
        self.min_coordinate = np.asarray(value, dtype=float)

    @property
    def MaxCoordinate(self) -> np.ndarray:
        return self.max_coordinate

    @MaxCoordinate.setter
    def MaxCoordinate(self, value) -> None:
        self.max_coordinate = np.asarray(value, dtype=float)

    @property
    def ShiftSeverity(self) -> float:
        return self.shift_severity

    @ShiftSeverity.setter
    def ShiftSeverity(self, value: float) -> None:
        self.shift_severity = float(value)

    @property
    def ExclusionLimit(self) -> float:
        return self.exclusion_limit

    @ExclusionLimit.setter
    def ExclusionLimit(self, value: float) -> None:
        self.exclusion_limit = float(value)

    @property
    def GenerateRadious(self) -> float:
        return self.generate_radious

    @GenerateRadious.setter
    def GenerateRadious(self, value: float) -> None:
        self.generate_radious = float(value)

    @property
    def MaxDeactivation(self) -> float:
        return self.max_deactivation

    @MaxDeactivation.setter
    def MaxDeactivation(self, value: float) -> None:
        self.max_deactivation = float(value)

    @property
    def MinDeactivation(self) -> float:
        return self.min_deactivation

    @MinDeactivation.setter
    def MinDeactivation(self, value: float) -> None:
        self.min_deactivation = float(value)

    @property
    def CurrentDeactivation(self) -> float:
        return self.current_deactivation

    @CurrentDeactivation.setter
    def CurrentDeactivation(self, value: float) -> None:
        self.current_deactivation = float(value)

    @property
    def InitialPopulationSize(self) -> int:
        return self.initial_population_size

    @InitialPopulationSize.setter
    def InitialPopulationSize(self, value: int) -> None:
        self.initial_population_size = max(1, int(value))

    @property
    def SwarmMember(self) -> int:
        return self.swarm_member

    @SwarmMember.setter
    def SwarmMember(self, value: int) -> None:
        self.swarm_member = max(1, int(value))

    @property
    def NewlyAddedPopulationSize(self) -> int:
        return self.newly_added_population_size

    @NewlyAddedPopulationSize.setter
    def NewlyAddedPopulationSize(self, value: int) -> None:
        self.newly_added_population_size = max(0, int(value))

    @property
    def Nmax(self) -> int:
        return self.nmax

    @Nmax.setter
    def Nmax(self, value: int) -> None:
        self.nmax = max(1, int(value))

    def initialize(self, problem) -> None:
        self.dimension = int(problem.dimension)
        self.bounds = problem.bounds
        self.min_coordinate = np.asarray(self.bounds.lower, dtype=float)
        self.max_coordinate = np.asarray(self.bounds.upper, dtype=float)
        self.initial_population_size = max(
            1, int(self.parameters["InitialPopulationSize"])
        )
        self.swarm_member = max(1, int(self.parameters["SwarmMember"]))
        self.newly_added_population_size = max(
            0, int(self.parameters["NewlyAddedPopulationSize"])
        )
        self.nmax = max(1, int(self.parameters["Nmax"]))
        self.beta = float(self.parameters["beta"])
        self.gama = float(self.parameters["gama"])

        self.shift_severity = 1.0
        peak_number = max(1, int(problem.parameters.get("PeakNumber", 1)))
        initial_radius = self._radius_scale(peak_number)
        self.exclusion_limit = 0.5 * initial_radius
        self.generate_radious = 0.3 * initial_radius
        self.teta = self.shift_severity
        self.tracker: list[int] = []
        self.max_deactivation = float(self.parameters["rho"]) * self.shift_severity
        self.min_deactivation = float(self.parameters["mu"]) * np.sqrt(self.dimension)
        self.current_deactivation = self.max_deactivation

        self.particles = [
            self._generate_particle(problem)
            for _ in range(self.initial_population_size)
        ]
        self.species = self._create_species()

    def iterate(self, problem) -> None:
        if self._recent_or_terminated(problem):
            return

        species = self._create_species()
        previous_tracker_count = len(self.tracker)
        best_tracker_index = self._determine_trackers(species)

        removed_particle_indices = self._mark_exclusion(species)
        self.tracker = [
            tracker_index
            for tracker_index in self.tracker
            if species[tracker_index].remove != 1
        ]
        self._deactivate_trackers(
            species,
            previous_tracker_count=previous_tracker_count,
            best_tracker_index=best_tracker_index,
        )

        species = self._drop_removed_species(species)
        self._refresh_thresholds(len(species))
        all_converge = self._all_species_converged(species)

        if len(species) >= self.nmax and all_converge:
            all_converge = False
            if self._reset_worst_species(species, problem):
                return

        if self._move_active_species(species, removed_particle_indices, problem):
            return

        if removed_particle_indices:
            self._delete_particles(removed_particle_indices)
            species = self._create_species()

        if len(species) < self.nmax and all_converge:
            for _ in range(self.newly_added_population_size):
                self.particles.append(self._generate_particle(problem))
                if self._recent_or_terminated(problem):
                    self.species = self._create_species()
                    return

    def on_environment_change(self, problem) -> None:
        if getattr(problem.state, "terminated", False) or not self.particles:
            return

        species = self.species if getattr(self, "species", None) else self._create_species()
        self._determine_trackers(species)
        valid_shifts: list[float] = []

        for tracker_index in self.tracker:
            if tracker_index >= len(species):
                continue
            species_record = species[tracker_index]
            seed_particle = self.particles[species_record.seed]
            for local_index, member_index in enumerate(species_record.member):
                if member_index >= len(self.particles):
                    continue
                member = self.particles[member_index]
                past_position = member.pbest_past_environment
                if past_position is None:
                    continue
                past = np.asarray(past_position, dtype=float).reshape(-1)
                if past.size != self.dimension or not np.all(np.isfinite(past)):
                    continue
                shift = float(np.linalg.norm(past - seed_particle.pbest_position))
                if np.isfinite(shift):
                    seed_particle.shifts = [shift]
                    member.pbest_past_environment = None
                    if local_index != 0:
                        member.shifts = []
                    break
            if seed_particle.shifts:
                shift = float(seed_particle.shifts[-1])
                if np.isfinite(shift):
                    valid_shifts.append(shift)

        if valid_shifts:
            self.shift_severity = float(np.mean(valid_shifts))

        for tracker_index in self.tracker:
            if tracker_index >= len(species):
                continue
            species_record = species[tracker_index]
            seed_position = self.particles[species_record.seed].pbest_position.copy()
            for member_index in species_record.member:
                if member_index >= len(self.particles):
                    continue
                direction = self.rng.standard_normal(self.dimension)
                norm = float(np.linalg.norm(direction))
                if norm == 0.0:
                    direction = np.zeros(self.dimension, dtype=float)
                    direction[0] = 1.0
                    norm = 1.0
                shift = (direction / norm) * self.shift_severity
                self.particles[member_index].position = seed_position + shift
            seed_particle = self.particles[species_record.seed]
            seed_particle.position = seed_position.copy()
            seed_particle.pbest_past_environment = seed_position.copy()

        for species_record in species:
            for member_index in species_record.member:
                if member_index < len(self.particles):
                    self._clip_particle(self.particles[member_index])

        for particle in self.particles:
            values = evaluate_(problem, particle.position, 1)
            particle.fitness_value = self._first_value(values)
            particle.pbest_fitness = particle.fitness_value
            particle.pbest_position = particle.position.copy()
            if self._recent_or_terminated(problem):
                return

        self.max_deactivation = float(self.parameters["rho"]) * self.shift_severity
        self.current_deactivation = self.max_deactivation
        self.teta = self.shift_severity
        self.beta = 1.0

    def creating_species(self) -> list[SPSOAPADSpecies]:
        return self._create_species()

    def CreatingSpecies(self) -> list[SPSOAPADSpecies]:
        return self._create_species()

    def sub_population_generator(self, problem) -> SPSOAPADParticle:
        return self._generate_particle(problem)

    def SubPopulationGenerator_SPSO_AP_AD(self, problem) -> SPSOAPADParticle:
        return self._generate_particle(problem)

    def _generate_particle(self, problem) -> SPSOAPADParticle:
        position = self._random_positions(1)[0, :]
        values = evaluate_(problem, position, 1)
        fitness_value = self._first_value(values)
        if getattr(problem.state, "recent_change", False):
            fitness_value = -np.inf
        return SPSOAPADParticle(
            position=position,
            pbest_position=position.copy(),
            velocity=np.zeros(self.dimension, dtype=float),
            fitness_value=fitness_value,
            pbest_fitness=fitness_value,
            processed=0,
            shifts=[],
            pbest_past_environment=None,
        )

    def _create_species(self) -> list[SPSOAPADSpecies]:
        for particle in self.particles:
            particle.processed = 0
        if not self.particles:
            self.species = []
            return []

        fitness = np.asarray(
            [particle.pbest_fitness for particle in self.particles], dtype=float
        )
        comparable = np.nan_to_num(fitness, nan=-np.inf)
        order = np.argsort(-comparable, kind="mergesort")
        species: list[SPSOAPADSpecies] = []

        for seed_index in order:
            seed = int(seed_index)
            if self.particles[seed].processed != 0:
                continue
            self.particles[seed].processed = 1
            members = [seed]
            candidate_indices = [
                index
                for index, particle in enumerate(self.particles)
                if particle.processed == 0
            ]
            nearest_distances: list[float] = []
            if candidate_indices:
                seed_position = self.particles[seed].pbest_position
                distances = np.asarray(
                    [
                        np.linalg.norm(
                            seed_position - self.particles[candidate].pbest_position
                        )
                        for candidate in candidate_indices
                    ],
                    dtype=float,
                )
                nearest_order = np.argsort(distances, kind="mergesort")
                nearest_count = min(self.swarm_member - 1, len(candidate_indices))
                for local_index in nearest_order[:nearest_count]:
                    candidate = candidate_indices[int(local_index)]
                    members.append(candidate)
                    self.particles[candidate].processed = 1
                    nearest_distances.append(float(distances[int(local_index)]))
            distance = max(nearest_distances) if nearest_distances else 0.0
            species.append(
                SPSOAPADSpecies(
                    seed=seed,
                    member=members,
                    remove=0,
                    Active=1,
                    distance=float(distance),
                )
            )

        self.species = species
        return species

    def _determine_trackers(self, species: list[SPSOAPADSpecies]) -> int | None:
        self.tracker = []
        best_tracker_index: int | None = None
        best_tracker_fitness = -np.inf
        for index, species_record in enumerate(species):
            if not species_record.distance < self.teta:
                continue
            self.tracker.append(index)
            seed_fitness = float(self.particles[species_record.seed].pbest_fitness)
            if seed_fitness > best_tracker_fitness:
                best_tracker_fitness = seed_fitness
                best_tracker_index = index
        return best_tracker_index

    def _mark_exclusion(self, species: list[SPSOAPADSpecies]) -> list[int]:
        removed_particle_indices: list[int] = []
        for left in range(len(species)):
            for right in range(left + 1, len(species)):
                left_seed = self.particles[species[left].seed]
                right_seed = self.particles[species[right].seed]
                distance = float(
                    np.linalg.norm(
                        left_seed.pbest_position - right_seed.pbest_position
                    )
                )
                if distance >= self.exclusion_limit:
                    continue
                if left_seed.pbest_fitness < right_seed.pbest_fitness:
                    loser = left
                else:
                    loser = right
                species[loser].remove = 1
                removed_particle_indices.extend(species[loser].member)
        return removed_particle_indices

    def _deactivate_trackers(
        self,
        species: list[SPSOAPADSpecies],
        *,
        previous_tracker_count: int,
        best_tracker_index: int | None,
    ) -> None:
        if previous_tracker_count < len(self.tracker):
            self.current_deactivation = self.max_deactivation

        all_trackers_converged = 1
        if self.tracker:
            for tracker_index in self.tracker:
                if species[tracker_index].distance > self.current_deactivation:
                    all_trackers_converged = 0
                elif tracker_index != best_tracker_index:
                    species[tracker_index].Active = 0

        if all_trackers_converged == 1 and self.tracker:
            if self.current_deactivation > self.min_deactivation:
                self.beta = self.beta * self.gama
                self.current_deactivation = self.min_deactivation + (
                    self.max_deactivation - self.min_deactivation
                ) * self.beta

    def _drop_removed_species(
        self, species: list[SPSOAPADSpecies]
    ) -> list[SPSOAPADSpecies]:
        survivors = [record for record in species if record.remove != 1]
        self.species = survivors
        return survivors

    def _refresh_thresholds(self, species_count: int) -> None:
        radius = self._radius_scale(max(1, species_count))
        self.exclusion_limit = 0.5 * radius
        self.generate_radious = 0.3 * radius

    def _all_species_converged(self, species: list[SPSOAPADSpecies]) -> bool:
        if not species:
            return False
        return all(
            species_record.distance <= self.generate_radious
            for species_record in species
        )

    def _reset_worst_species(self, species: list[SPSOAPADSpecies], problem) -> bool:
        if not species:
            return False
        seed_fitness = np.asarray(
            [self.particles[record.seed].pbest_fitness for record in species],
            dtype=float,
        )
        worst_index = int(np.argmin(np.nan_to_num(seed_fitness, nan=-np.inf)))
        for member_index in list(species[worst_index].member):
            if member_index >= len(self.particles):
                continue
            self.particles[member_index] = self._generate_particle(problem)
            if self._recent_or_terminated(problem):
                self.species = self._create_species()
                return True
        species[worst_index].Active = 1
        return False

    def _move_active_species(
        self,
        species: list[SPSOAPADSpecies],
        removed_particle_indices: list[int],
        problem,
    ) -> bool:
        constriction = float(self.parameters["x"])
        cognitive = float(self.parameters["c1"])
        social = float(self.parameters["c2"])
        for species_record in species:
            if species_record.Active != 1:
                continue
            seed_position = self.particles[species_record.seed].pbest_position.copy()
            for member_index in species_record.member:
                if member_index >= len(self.particles):
                    continue
                particle = self.particles[member_index]
                r1 = self.rng.random(self.dimension)
                r2 = self.rng.random(self.dimension)
                particle.velocity = constriction * (
                    particle.velocity
                    + cognitive * r1 * (particle.pbest_position - particle.position)
                    + social * r2 * (seed_position - particle.position)
                )
                particle.position = particle.position + particle.velocity
                self._clip_particle(particle)
                values = evaluate_(problem, particle.position, 1)
                particle.fitness_value = self._first_value(values)
                if getattr(problem.state, "recent_change", False):
                    if removed_particle_indices:
                        self._delete_particles(removed_particle_indices)
                    self.species = self._create_species()
                    return True
                if particle.fitness_value > particle.pbest_fitness:
                    particle.pbest_fitness = particle.fitness_value
                    particle.pbest_position = particle.position.copy()
                if getattr(problem.state, "terminated", False):
                    return True
        return False

    def _delete_particles(self, indices: list[int]) -> None:
        for index in sorted(set(int(item) for item in indices), reverse=True):
            if 0 <= index < len(self.particles):
                del self.particles[index]

    def _clip_particle(self, particle: SPSOAPADParticle) -> None:
        lower = self._lower()
        upper = self._upper()
        high = particle.position > upper
        low = particle.position < lower
        clipped = high | low
        if np.any(high):
            particle.position[high] = np.broadcast_to(upper, particle.position.shape)[high]
        if np.any(low):
            particle.position[low] = np.broadcast_to(lower, particle.position.shape)[low]
        if np.any(clipped):
            particle.velocity[clipped] = 0.0

    def _random_positions(self, rows: int) -> np.ndarray:
        lower = self._lower()
        upper = self._upper()
        return lower + (upper - lower) * self.rng.random((rows, self.dimension))

    def _radius_scale(self, count: int) -> float:
        return float(self._domain_span() / (max(1, count) ** (1.0 / self.dimension)))

    def _domain_span(self) -> float:
        return float(np.max(self._upper() - self._lower()))

    def _lower(self) -> np.ndarray:
        return np.asarray(self.min_coordinate, dtype=float)

    def _upper(self) -> np.ndarray:
        return np.asarray(self.max_coordinate, dtype=float)

    @staticmethod
    def _first_value(values: np.ndarray) -> float:
        if values.size == 0:
            return -np.inf
        value = float(np.asarray(values, dtype=float).reshape(-1)[0])
        if np.isnan(value):
            return -np.inf
        return value

    @staticmethod
    def _recent_or_terminated(problem) -> bool:
        return bool(
            getattr(problem.state, "recent_change", False)
            or getattr(problem.state, "terminated", False)
        )


def sub_population_generator_spso_ap_ad(
    algorithm: SPSOAPADAlgorithm, problem
) -> SPSOAPADParticle:
    return algorithm._generate_particle(problem)


def creating_species_spso_ap_ad(
    algorithm: SPSOAPADAlgorithm,
) -> list[SPSOAPADSpecies]:
    return algorithm._create_species()


SubPopulationGenerator_SPSO_AP_AD = sub_population_generator_spso_ap_ad
CreatingSpecies = creating_species_spso_ap_ad
SPSO_AP_ADAlgorithm = SPSOAPADAlgorithm
