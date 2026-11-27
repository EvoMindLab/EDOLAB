"""Object-oriented DPORTAL problem instance.

Each environment is a PORTAL landscape turned into a maximisation problem:
component forms, neutralisation scalings, Givens rotations and the five
transformations follow the PORTAL reference code (``portal_core``). Points
outside the bounds are evaluated at their projection, as in the other bounded
generators of this platform.

The first environment is drawn as by the PORTAL2 generator with its default
modes: forms A/B with probability 1/2, kappa uniform in [1, 100], exponents
uniform in [0.2, 1.2], transformation parameters uniform in their ranges
(per-component values always uniform), and centres and heights redrawn until no
component covers another's centre. DynPORTAL keeps its own scale: the domain
[-50, 50]^D with centres anywhere in it, heights in [30, 70], neutralisation with
r_ref = 5 and Delta = 50 (PORTAL2: r_ref = 100, Delta uniform in [50, 200]), and
fully rotated components. Mode numbers already in use keep their DynPORTAL
meaning; PORTAL2's remaining modes have new numbers (see the ``*InitMode``
functions).

Every component starts with the log-sinusoidal transformation, whose amplitude mu
is drawn from [0.1, 1.0] (GMPB's irregularity range; PORTAL: [0.05, 0.5]), followed
by 0--3 further transformations of the other four types. The additive-periodic and
wavelet transformations are contracted towards the summit by the length scale
0.1, and the radial-polytrig amplitude is limited to [0.1, 0.5] (PORTAL: [0.4, 2.0]);
all other ranges are PORTAL's. Without these settings the near-summit surface is
smooth on most components, and radial-polytrig's concentric shells mislead
step-size and covariance adaptation.

Every parameter that changes between environments stays in its range: a value
that leaves it is reflected back. Every peak moves by a random walk without
inertia, so a step has exactly the stated severity; the shift severity is 1 as
in GMPB. Sequence changes have probability 0.1, act on the transformations after
the log-sinusoidal one, and keep 0--3 of them.

Environments are generated, stored and evaluated in float64 by default: the
composed transformations can amplify single-precision rounding. The explicit
``EvaluationPrecision=float32`` option stores and evaluates the environments in
single precision; rounding happens after all transitions are generated.

``_phi_component`` is the per-point reference of the objective. Evaluations
use the batched NumPy evaluator of ``evaluation.py``. It was measured faster
than Torch on CPU at every batch size, and than Torch on MPS, so the
``EvaluationDevice`` values ``auto``, ``cpu`` and ``numpy`` all select it.
Torch evaluates only an explicitly requested ``cuda`` or ``mps`` device.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from core.compute import normalize_torch_device_policy, record_compute_use, select_torch_device
from core.parameters import ParameterSpec
from core.problem.base import BaseProblem
from core.problem.bounds import Bounds
from core.registry import problem_registry

from .evaluation import TRANSFORMS, TorchEnvironment, prepare, tensor_source_key, use_float32


class _PortalStruct(dict):
    """Small attribute-access container for DPORTAL environment state.

    ``version`` counts the changes of its entries, so that an evaluator can
    tell without comparing them whether they are the ones it prepared.
    """

    version = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _counted(method):
    def changed(self, *args, **kwargs):
        self.__dict__["version"] = self.version + 1
        return method(self, *args, **kwargs)
    return changed


for _name in ("__setitem__", "__delitem__", "__ior__", "update", "pop", "popitem", "setdefault", "clear"):
    setattr(_PortalStruct, _name, _counted(getattr(dict, _name)))


@problem_registry.register("DPORTAL")
class DPORTALProblem(BaseProblem):
    name: ClassVar[str] = "DPORTAL"

    @classmethod
    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return _portal_parameter_specs()

    def __getattr__(self, name: str) -> Any:
        parameters = self.__dict__.get("parameters", {})
        if name in parameters:
            return parameters[name]
        raise AttributeError(name)

    @property
    def dimension(self) -> int:
        return int(self.parameters["Dimension"])

    @property
    def bounds(self) -> Bounds:
        return Bounds.uniform(
            float(self.parameters["MinCoordinate"]),
            float(self.parameters["MaxCoordinate"]),
            self.dimension,
        )

    def reset(self, seed: int | None = None) -> None:
        if self.EvaluationPrecision not in ("float32", "float64"):
            raise ValueError("DPORTAL EvaluationPrecision must be float32 or float64.")
        self.initialize_runtime_state()
        self.rng = np.random.default_rng(seed)
        self._torch_prepared = None
        try:
            policy = normalize_torch_device_policy(str(self.parameters.get("EvaluationDevice", "auto")))
        except ValueError:
            policy = "auto"
        # Only an explicitly requested accelerator replaces the NumPy evaluator.
        self._accelerator = None if policy in ("auto", "cpu", "numpy") else policy
        self._numpy_recorded = False
        self._build_environments()

    def _shift_severity(self, environment_index: int) -> float:
        """Nominal displacement; direction, inertia and reflection stay native."""
        return float(self.parameters["ShiftSeverity"])

    def _build_environments(self) -> None:
        env_count = self.environment_number
        dimension = self.dimension
        component_count = int(self.parameters["ComponentNumber"])

        self.params: list[_PortalStruct] = []
        first = _sample_portal_params(component_count, dimension, self, self.rng)
        self.params.append(first)
        for _env_index in range(1, env_count):
            self.params.append(_evolve_portal_params(
                self.params[-1], self, self.rng,
                shift_severity=self._shift_severity(_env_index),
            ))

        # Irregularity control: every transformation amplitude is scaled after all environments are drawn, so
        # the scale changes only the surfaces, not the centres, heights, shapes, changes or sequences of a seed.
        scale = float(self.TransformAmplitudeScale)
        if scale != 1.0:
            for params in self.params:
                _scale_transform_amplitudes(params, scale)
        length = float(self.TransformLengthScale)
        if length != 1.0:
            types = _length_types(self)
            for params in self.params:
                _shrink_transform_length(params, length, types)

        # Generate all transitions before rounding, so changing evaluation
        # precision does not change the generator's random walk or RNG state.
        if self.EvaluationPrecision == "float32":
            for params in self.params:
                use_float32(params)

        self.component_counts = np.asarray([param.c.shape[0] for param in self.params], dtype=int)
        self.optimum_values = np.asarray([np.max(param.beta) for param in self.params], dtype=float)
        self.optimum_ids = np.asarray([int(np.argmax(param.beta)) for param in self.params], dtype=int)
        self.peak_number = int(np.max(self.component_counts))
        self.parameters["PeakNumber"] = self.peak_number
        self.peak_visibility = np.zeros((env_count, self.peak_number), dtype=int)
        self.peaks_position = np.full((self.peak_number, dimension, env_count), np.nan, dtype=float)
        for env_index, param in enumerate(self.params):
            k_env = param.c.shape[0]
            self.peak_visibility[env_index, :k_env] = 1
            self.peaks_position[:k_env, :, env_index] = param.c
        self.Params = self.params
        self.K_each_env = self.component_counts
        self.OptimumValue = self.optimum_values
        self.OptimumID = self.optimum_ids
        self.PeakNumber = self.peak_number
        self.PeakVisibility = self.peak_visibility
        self.PeaksPosition = self.peaks_position

    def raw_fitness(self, x: np.ndarray) -> float:
        params = self.params[int(self.state.environment_index)]
        sample = np.asarray(x, dtype=params.c.dtype).reshape(-1)
        if sample.size != self.dimension:
            raise ValueError(f"DPORTAL expects dimension {self.dimension}.")
        return float(self._raw_fitness_batch_numpy(sample[None, :], params)[0])

    def raw_fitness_batch(self, x: np.ndarray) -> np.ndarray:
        params = self.params[int(self.state.environment_index)]
        samples = np.asarray(x, dtype=params.c.dtype)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        centers = params["c"]
        if samples.shape[1] != centers.shape[1]:
            raise ValueError(f"DPORTAL expects dimension {self.dimension}.")
        if self._accelerator is not None:
            torch, device, label = select_torch_device(
                "problem:DPORTAL", work_items=samples.size * centers.shape[0], allow_torch_cpu=False,
                device=self._accelerator)
            if torch is not None and device is not None:
                try:
                    return self._raw_fitness_batch_torch(samples, params, torch, device)
                except Exception as exc:
                    record_compute_use(
                        "problem:DPORTAL",
                        "numpy:cpu",
                        f"{label} failed; using NumPy CPU: {type(exc).__name__}",
                    )
        elif not self._numpy_recorded:
            self._numpy_recorded = True
            select_torch_device("problem:DPORTAL", device="numpy")
        return prepare(params).fitness(samples)

    def _raw_fitness_batch_numpy(self, samples: np.ndarray, params: _PortalStruct) -> np.ndarray:
        return prepare(params).fitness(samples)

    def _raw_fitness_batch_torch(self, samples: np.ndarray, params: _PortalStruct, torch, device) -> np.ndarray:
        # Cache only CPU views, which share NumPy storage. Other devices keep
        # reading current parameters on every call, as before.
        key = tensor_source_key(params) if str(device) == "cpu" else None
        cached = self._torch_prepared
        with torch.inference_mode():
            if key is not None and cached is not None and cached[0] == key:
                prepared = cached[1]
            else:
                prepared = TorchEnvironment(params, torch, device)
                self._torch_prepared = (key, prepared) if key is not None else None
            dtype = torch.float32 if params.c.dtype == np.float32 else torch.float64
            sample_t = torch.as_tensor(samples, dtype=dtype, device=device)
            clipped = torch.clamp(sample_t, min=prepared.bounds_min, max=prepared.bounds_max)
            values = _phi_components_batch_torch(clipped, params, torch, device, prepared)
            return torch.max(values, dim=1).values.detach().cpu().numpy().astype(float)

    def current_optimum_value(self) -> float:
        return float(self.optimum_values[self.state.environment_index])

    def global_optimum_basin_mask(
        self,
        x: np.ndarray,
        environment_index: int | None = None,
    ) -> np.ndarray:
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        env = self.state.environment_index if environment_index is None else int(environment_index)
        centers = self.params[env].c
        distances = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
        nearest_component_ids = np.argmin(distances, axis=1)
        return nearest_component_ids == int(self.optimum_ids[env])

    def visualization_peaks(self, environment_number: int | None = None) -> dict[str, object]:
        if self.dimension != 2:
            return {}
        env_index = self.resolve_environment_index(environment_number)
        param = self.params[env_index]
        optimum_id = int(self.optimum_ids[env_index])
        visible_mask = np.ones(param.c.shape[0], dtype=bool)
        return {
            "global_optimum_id": optimum_id + 1,
            "global_optimum_position": param.c[optimum_id, :2].copy(),
            "visible_peak_ids": np.flatnonzero(visible_mask) + 1,
            "visible_peak_positions": param.c[visible_mask, :2].copy(),
        }


def _portal_parameter_specs() -> dict[str, ParameterSpec]:
    pi = float(np.pi)
    entries = [
        ("MinCoordinate", -50, "numeric", [-1000, 1000]),
        ("MaxCoordinate", 50, "numeric", [-1000, 1000]),
        ("Dimension", 5, "integer", [2, 100]),
        ("ComponentNumber", 10, "integer", [1, 100]),
        ("EnvironmentNumber", 100, "integer", [1, 1000]),
        ("ChangeFrequency", 5000, "integer", [100, 20000]),
        ("EvaluationDevice", "auto", "option", ["auto", "cuda", "cuda:<id>", "mps", "cpu", "numpy"]),
        ("EvaluationPrecision", "float64", "option", ["float32", "float64"]),
        ("CenterInitMode", 1, "integer", [1, 4]),
        ("QuasiUniformPerturbation", 5, "numeric", [0, 30]),
        ("ExclusionZone", 20, "numeric", [0, 50]),
        ("KappaInitMode", 1, "integer", [1, 5]),
        ("KappaDistAlpha", 0.2, "numeric", [0.1, 10]),
        ("KappaDistBeta", 0.2, "numeric", [0.1, 10]),
        ("KappaMin", 1, "numeric", [0.01, 1000]),
        ("KappaMax", 100, "numeric", [0.01, 1000]),
        ("PInitMode", 1, "integer", [1, 5]),
        ("FixedP", 0.6, "numeric", [0.2, 1.2]),
        ("ExponentMin", 0.2, "numeric", [0.05, 5]),
        ("ExponentMax", 1.2, "numeric", [0.05, 5]),
        ("ExponentDistAlpha", 0.5, "numeric", [0.1, 10]),
        ("ExponentDistBeta", 0.5, "numeric", [0.1, 10]),
        ("BetaOffsetInitMode", 1, "integer", [1, 6]),
        ("FixedBetaOffset", 50, "numeric", [30, 70]),
        ("BetaOffsetDistAlpha", 0.2, "numeric", [0.1, 10]),
        ("BetaOffsetDistBeta", 0.2, "numeric", [0.1, 10]),
        ("BetaGap", 20, "numeric", [0, 40]),
        ("BetaOptGap", 1, "numeric", [0, 40]),
        ("TransformParamInitMode", 1, "integer", [1, 8]),
        ("TransformDistAlpha", 0.2, "numeric", [0.1, 10]),
        ("TransformDistBeta", 0.2, "numeric", [0.1, 10]),
        ("TransformParamReflect", 1, "boolean", [0, 1]),
        ("TransformAmplitudeScale", 1.0, "numeric", [0, 10]),
        ("TransformLengthScale", 0.1, "numeric", [0.01, 1]),
        ("TransformLengthTypes", "additive_periodic,wavelet_mod", "string", []),
        ("TransformRangeOverrides", "log_sinusoidal.mu=0.1:1.0;radial_polytrig.mu=0.1:0.5", "string", []),
        ("BaseTransform", "log_sinusoidal", "string", []),
        ("TransformTypePool", "additive_periodic,log_sinusoidal,wavelet_mod,tensor_interference,radial_polytrig", "string", []),
        ("BaselineInitMode", 1, "integer", [1, 6]),
        ("BaselineProbA", 0.5, "numeric", [0, 1]),
        ("BaselineSeq", "A,B", "string", []),
        ("NeutralizeDeltaMin", 50, "numeric", [1, 1000]),
        ("NeutralizeDeltaMax", 50, "numeric", [1, 1000]),
        ("NeutralizeRRef", 5, "numeric", [0.1, 100]),
        ("EnsureNoCover", 1, "boolean", [0, 1]),
        ("NoCoverMaxAttempts", 1000, "integer", [1, 100000]),
        ("NoCoverTolerance", 1e-6, "numeric", [0, 1]),
        ("TransformSeqInitMode", 1, "integer", [1, 6]),
        ("TransformSeqCount", 1, "integer", [0, 10]),
        ("TransformSeqCustom", "", "string", []),
        ("RotationInitMode", 2, "integer", [1, 7]),
        ("RotationProb", 0.5, "numeric", [0, 1]),
        ("SpecificAngleDeg", 45, "numeric", [-180, 180]),
        ("CenterChangeEnable", 1, "boolean", [0, 1]),
        ("CenterStrategyMode", 1, "integer", [1, 4]),
        ("CenterChangeMode", 1, "integer", [1, 3]),
        ("ShiftSeverity", 1, "numeric", [0, 10]),
        ("CenterLambda_Homo", 0.0, "numeric", [0, 1]),
        ("CenterAngleSensitivity_Homo", pi / 10, "numeric", [0, pi]),
        ("CenterAttract_Homo", 0.1, "numeric", [0, 1]),
        ("CenterRepel_Homo", 10, "numeric", [0, 100]),
        ("CenterLambda", 0.0, "numeric", [0, 1]),
        ("CenterAngleSensitivity", pi / 10, "numeric", [0, pi]),
        ("CenterAttract", 0.1, "numeric", [0, 1]),
        ("CenterRepel", 10, "numeric", [0, 100]),
        ("CenterModeProb_MomentumWalk", 0.34, "numeric", [0, 1]),
        ("CenterModeProb_Circular", 0.33, "numeric", [0, 1]),
        ("CenterModeProb_AR", 0.33, "numeric", [0, 1]),
        ("CenterModeCustom", "1,2", "string", []),
        ("BetaChangeEnable", 1, "boolean", [0, 1]),
        ("BetaStrategyMode", 1, "integer", [1, 4]),
        ("BetaChangeMode", 1, "integer", [1, 2]),
        ("BetaSeverity", 8, "numeric", [0, 20]),
        ("BetaLambda_Homo", 0.0, "numeric", [0, 1]),
        ("BetaOmegaMode_Homo", 1, "integer", [1, 3]),
        ("BetaOmega_Homo", pi / 4, "numeric", [0, pi]),
        ("BetaAmplitudeMode_Homo", 1, "integer", [1, 3]),
        ("BetaAmplitude_Homo", 10, "numeric", [0, 30]),
        ("BetaLambda", 0.0, "numeric", [0, 1]),
        ("BetaOmegaMode", 1, "integer", [1, 3]),
        ("BetaOmega", pi / 4, "numeric", [0, pi]),
        ("BetaOmegaMin", 0.1, "numeric", [0, pi]),
        ("BetaOmegaMax", 2.0, "numeric", [0, pi]),
        ("BetaOmegaArray", "0.5,0.5,0.5,0.5,0.5", "string", []),
        ("BetaAmplitudeMode", 1, "integer", [1, 3]),
        ("BetaAmplitude", 10, "numeric", [0, 30]),
        ("BetaAmplitudeMin", 1, "numeric", [0, 20]),
        ("BetaAmplitudeMax", 10, "numeric", [0, 30]),
        ("BetaAmplitudeArray", "5,5,5,5,5", "string", []),
        ("BetaModeProb_Walk", 0.5, "numeric", [0, 1]),
        ("BetaModeProb_Periodic", 0.5, "numeric", [0, 1]),
        ("BetaModeCustom", "1,2", "string", []),
        ("KappaChangeEnable", 1, "boolean", [0, 1]),
        ("KappaStrategyMode", 1, "integer", [1, 4]),
        ("KappaChangeMode", 1, "integer", [1, 3]),
        ("KappaSeverity", 1.0, "numeric", [0, 5]),
        ("KappaLambda_Homo", 0.0, "numeric", [0, 1]),
        ("KappaLambda", 0.0, "numeric", [0, 1]),
        ("KappaModeProb_Walk", 0.34, "numeric", [0, 1]),
        ("KappaModeProb_Homo", 0.33, "numeric", [0, 1]),
        ("KappaModeProb_Diff", 0.33, "numeric", [0, 1]),
        ("KappaModeCustom", "1,2,3,1,2", "string", []),
        ("KappaSwitchProb", 0.3, "numeric", [0, 1]),
        ("ExponentChangeEnable", 1, "boolean", [0, 1]),
        ("PSeverity", 0.1, "numeric", [0, 0.5]),
        ("ExponentStrategyMode", 1, "integer", [1, 4]),
        ("ExponentChangeMode", 1, "integer", [1, 3]),
        ("ExponentLambda_Homo", 0.0, "numeric", [0, 1]),
        ("ExponentOmega", pi / 10, "numeric", [0, pi]),
        ("ExponentAmplitude", 0.2, "numeric", [0, 1]),
        ("ExponentLambda", 0.0, "numeric", [0, 1]),
        ("ExponentModeProb_Walk", 0.34, "numeric", [0, 1]),
        ("ExponentModeProb_Homo", 0.33, "numeric", [0, 1]),
        ("ExponentModeProb_Diff", 0.33, "numeric", [0, 1]),
        ("ExponentModeCustom", "1,2,3,1,2", "string", []),
        ("ExponentSwitchProb", 0.3, "numeric", [0, 1]),
        ("RotationAngleEnable", 1, "boolean", [0, 1]),
        ("AngleSeverity", pi / 8, "numeric", [0, pi]),
        ("RotationChangeMode", 1, "integer", [1, 2]),
        ("RotationTopologyEnable", 0, "boolean", [0, 1]),
        ("RotationFlipProb", 0.1, "numeric", [0, 1]),
        ("TopologySwitchProb", 0.1, "numeric", [0, 1]),
        ("TopologyModePool", "2,3,4,5,6,7", "string", []),
        ("FormSwitchEnable", 0, "boolean", [0, 1]),
        ("FormSwitchProb", 0.1, "numeric", [0, 0.5]),
        ("TransformSeqChangeEnable", 1, "boolean", [0, 1]),
        ("AddTransformProb", 0.1, "numeric", [0, 0.3]),
        ("RemoveTransformProb", 0.1, "numeric", [0, 0.3]),
        ("ReplaceTransformProb", 0.1, "numeric", [0, 0.3]),
        ("ShuffleTransformProb", 0.1, "numeric", [0, 0.3]),
        ("TransformSeqMinLen", 0, "integer", [0, 5]),
        ("TransformSeqMaxLen", 3, "integer", [1, 10]),
        ("TransformParamChangeEnable", 1, "boolean", [0, 1]),
        ("TransformParamLambda", 0.5, "numeric", [0, 1]),
        ("AP_mu_plus_Severity", 0.03, "numeric", [0, 0.6]),
        ("AP_mu_minus_Severity", 0.03, "numeric", [0, 0.6]),
        ("AP_gamma_plus_Severity", 0.01, "numeric", [0, 0.2]),
        ("AP_gamma_minus_Severity", 0.01, "numeric", [0, 0.2]),
        ("AP_omega_plus_Severity", 0.05, "numeric", [0, 1.0]),
        ("AP_omega_minus_Severity", 0.05, "numeric", [0, 1.0]),
        ("LS_mu_plus_Severity", 0.0225, "numeric", [0, 0.45]),
        ("LS_mu_minus_Severity", 0.0225, "numeric", [0, 0.45]),
        ("LS_omega_plus_Severity", 2.25, "numeric", [0, 45]),
        ("LS_omega_minus_Severity", 2.25, "numeric", [0, 45]),
        ("WM_mu_plus_Severity", 2.0, "numeric", [0, 40]),
        ("WM_mu_minus_Severity", 2.0, "numeric", [0, 40]),
        ("WM_omega_plus_Severity", 0.035, "numeric", [0, 0.7]),
        ("WM_omega_minus_Severity", 0.035, "numeric", [0, 0.7]),
        ("WM_eta_Severity", 0.7, "numeric", [0, 14]),
        ("TI_mu_plus_Severity", 0.5, "numeric", [0, 10]),
        ("TI_mu_minus_Severity", 0.5, "numeric", [0, 10]),
        ("TI_omega_plus_Severity", 0.03, "numeric", [0, 0.6]),
        ("TI_omega_minus_Severity", 0.03, "numeric", [0, 0.6]),
        ("RP_mu_Severity", 0.08, "numeric", [0, 1.6]),
        ("RP_p_Severity", 0.015, "numeric", [0, 0.3]),
        ("RP_q_Severity", 0.04, "numeric", [0, 0.8]),
        ("RP_omega_Severity", 0.5, "numeric", [0, 10]),
        ("ComponentCountChangeEnable", 0, "boolean", [0, 1]),
        ("AddComponentProb", 0.05, "numeric", [0, 0.5]),
        ("RemoveComponentProb", 0.05, "numeric", [0, 0.5]),
        ("K_min", 2, "integer", [1, 10]),
        ("K_max", 20, "integer", [5, 50]),
    ]
    specs: dict[str, ParameterSpec] = {}
    for name, value, value_type, value_range in entries:
        metadata = _DPORTAL_PARAMETER_METADATA.get(name, {})
        description = _parameter_description(name, metadata)
        specs[name] = ParameterSpec(name, value, value_type, tuple(value_range), description)
    return specs


def _parameter_description(name: str, metadata: dict[str, str]) -> str:
    if not metadata:
        return f"DPORTAL parameter {name}."
    details = []
    parent = metadata.get("parent")
    required = metadata.get("required")
    if parent:
        details.append(f"parent={parent}")
    if required:
        details.append(f"required={required}")
    return f"DPORTAL parameter {name} ({'; '.join(details)})."


_DPORTAL_PARAMETER_METADATA = {
    "BetaGap": {"parent": "BetaOffsetInitMode", "required": "BetaOffsetInitMode==6"},
    "BetaOptGap": {"parent": "BetaOffsetInitMode", "required": "BetaOffsetInitMode==6"},
    "BaselineProbA": {"parent": "BaselineInitMode", "required": "BaselineInitMode==5"},
    "BaselineSeq": {"parent": "BaselineInitMode", "required": "BaselineInitMode==6"},
    "TransformSeqCount": {"parent": "TransformSeqInitMode", "required": "TransformSeqInitMode==5"},
    "TransformSeqCustom": {"parent": "TransformSeqInitMode", "required": "TransformSeqInitMode==6"},
    "NoCoverMaxAttempts": {"parent": "EnsureNoCover", "required": "EnsureNoCover==1"},
    "NoCoverTolerance": {"parent": "EnsureNoCover", "required": "EnsureNoCover==1"},
    "QuasiUniformPerturbation": {
        "parent": "CenterInitMode",
        "required": "CenterInitMode==4",
    },
    "ExclusionZone": {"parent": "CenterInitMode", "required": "CenterInitMode==2"},
    "KappaDistAlpha": {"parent": "KappaInitMode", "required": "KappaInitMode==2"},
    "KappaDistBeta": {"parent": "KappaInitMode", "required": "KappaInitMode==2"},
    "FixedP": {"parent": "PInitMode", "required": "PInitMode==2"},
    "ExponentDistAlpha": {"parent": "PInitMode", "required": "PInitMode==4"},
    "ExponentDistBeta": {"parent": "PInitMode", "required": "PInitMode==4"},
    "FixedBetaOffset": {
        "parent": "BetaOffsetInitMode",
        "required": "BetaOffsetInitMode==3",
    },
    "BetaOffsetDistAlpha": {
        "parent": "BetaOffsetInitMode",
        "required": "BetaOffsetInitMode==4",
    },
    "BetaOffsetDistBeta": {
        "parent": "BetaOffsetInitMode",
        "required": "BetaOffsetInitMode==4",
    },
    "TransformDistAlpha": {
        "parent": "TransformParamInitMode",
        "required": "TransformParamInitMode==4",
    },
    "TransformDistBeta": {
        "parent": "TransformParamInitMode",
        "required": "TransformParamInitMode==4",
    },
    "RotationProb": {"parent": "RotationInitMode", "required": "RotationInitMode==3"},
    "SpecificAngleDeg": {
        "parent": "RotationInitMode",
        "required": "RotationInitMode==5",
    },
    "CenterStrategyMode": {
        "parent": "CenterChangeEnable",
        "required": "CenterChangeEnable==1",
    },
    "CenterChangeMode": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==1",
    },
    "ShiftSeverity": {
        "parent": "CenterChangeEnable",
        "required": "CenterChangeEnable==1",
    },
    "CenterLambda_Homo": {
        "parent": "CenterChangeMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==1 && CenterChangeMode==1",
    },
    "CenterAngleSensitivity_Homo": {
        "parent": "CenterChangeMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==1 && CenterChangeMode==2",
    },
    "CenterAttract_Homo": {
        "parent": "CenterChangeMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==1 && CenterChangeMode==3",
    },
    "CenterRepel_Homo": {
        "parent": "CenterChangeMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==1 && CenterChangeMode==3",
    },
    "CenterLambda": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode>=2",
    },
    "CenterAngleSensitivity": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode>=2",
    },
    "CenterAttract": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode>=2",
    },
    "CenterRepel": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode>=2",
    },
    "CenterModeProb_MomentumWalk": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==2",
    },
    "CenterModeProb_Circular": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==2",
    },
    "CenterModeProb_AR": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==2",
    },
    "CenterModeCustom": {
        "parent": "CenterStrategyMode",
        "required": "CenterChangeEnable==1 && CenterStrategyMode==3",
    },
    "BetaStrategyMode": {
        "parent": "BetaChangeEnable",
        "required": "BetaChangeEnable==1",
    },
    "BetaChangeMode": {
        "parent": "BetaStrategyMode",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==1",
    },
    "BetaSeverity": {"parent": "BetaChangeEnable", "required": "BetaChangeEnable==1"},
    "BetaLambda_Homo": {
        "parent": "BetaChangeMode",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==1 && BetaChangeMode==1",
    },
    "BetaOmegaMode_Homo": {
        "parent": "BetaChangeMode",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==1 && BetaChangeMode==2",
    },
    "BetaOmega_Homo": {
        "parent": "BetaOmegaMode_Homo",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==1 && BetaChangeMode==2 && BetaOmegaMode_Homo==1",
    },
    "BetaAmplitudeMode_Homo": {
        "parent": "BetaChangeMode",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==1 && BetaChangeMode==2",
    },
    "BetaAmplitude_Homo": {
        "parent": "BetaAmplitudeMode_Homo",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==1 && BetaChangeMode==2 && BetaAmplitudeMode_Homo==1",
    },
    "BetaLambda": {
        "parent": "BetaStrategyMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3)",
    },
    "BetaOmegaMode": {
        "parent": "BetaStrategyMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3)",
    },
    "BetaOmega": {
        "parent": "BetaOmegaMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaOmegaMode==1",
    },
    "BetaOmegaMin": {
        "parent": "BetaOmegaMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaOmegaMode==2",
    },
    "BetaOmegaMax": {
        "parent": "BetaOmegaMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaOmegaMode==2",
    },
    "BetaOmegaArray": {
        "parent": "BetaOmegaMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaOmegaMode==3",
    },
    "BetaAmplitudeMode": {
        "parent": "BetaStrategyMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3)",
    },
    "BetaAmplitude": {
        "parent": "BetaAmplitudeMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaAmplitudeMode==1",
    },
    "BetaAmplitudeMin": {
        "parent": "BetaAmplitudeMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaAmplitudeMode==2",
    },
    "BetaAmplitudeMax": {
        "parent": "BetaAmplitudeMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaAmplitudeMode==2",
    },
    "BetaAmplitudeArray": {
        "parent": "BetaAmplitudeMode",
        "required": "BetaChangeEnable==1 && (BetaStrategyMode==2 || BetaStrategyMode==3) && BetaAmplitudeMode==3",
    },
    "BetaModeProb_Walk": {
        "parent": "BetaStrategyMode",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==2",
    },
    "BetaModeProb_Periodic": {
        "parent": "BetaStrategyMode",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==2",
    },
    "BetaModeCustom": {
        "parent": "BetaStrategyMode",
        "required": "BetaChangeEnable==1 && BetaStrategyMode==3",
    },
    "KappaStrategyMode": {
        "parent": "KappaChangeEnable",
        "required": "KappaChangeEnable==1",
    },
    "KappaChangeMode": {
        "parent": "KappaStrategyMode",
        "required": "KappaChangeEnable==1 && KappaStrategyMode==1",
    },
    "KappaSeverity": {
        "parent": "KappaChangeEnable",
        "required": "KappaChangeEnable==1",
    },
    "KappaLambda_Homo": {
        "parent": "KappaChangeMode",
        "required": "KappaChangeEnable==1 && KappaStrategyMode==1 && KappaChangeMode==1",
    },
    "KappaLambda": {
        "parent": "KappaStrategyMode",
        "required": "KappaChangeEnable==1 && (KappaStrategyMode==2 || KappaStrategyMode==3)",
    },
    "KappaModeProb_Walk": {
        "parent": "KappaStrategyMode",
        "required": "KappaChangeEnable==1 && KappaStrategyMode==2",
    },
    "KappaModeProb_Homo": {
        "parent": "KappaStrategyMode",
        "required": "KappaChangeEnable==1 && KappaStrategyMode==2",
    },
    "KappaModeProb_Diff": {
        "parent": "KappaStrategyMode",
        "required": "KappaChangeEnable==1 && KappaStrategyMode==2",
    },
    "KappaModeCustom": {
        "parent": "KappaStrategyMode",
        "required": "KappaChangeEnable==1 && KappaStrategyMode==3",
    },
    "KappaSwitchProb": {
        "parent": "KappaStrategyMode",
        "required": "KappaChangeEnable==1 && KappaStrategyMode==4",
    },
    "PSeverity": {
        "parent": "ExponentChangeEnable",
        "required": "ExponentChangeEnable==1",
    },
    "ExponentStrategyMode": {
        "parent": "ExponentChangeEnable",
        "required": "ExponentChangeEnable==1",
    },
    "ExponentChangeMode": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1 && ExponentStrategyMode==1",
    },
    "ExponentLambda_Homo": {
        "parent": "ExponentChangeMode",
        "required": "ExponentChangeEnable==1 && ExponentStrategyMode==1 && ExponentChangeMode==1",
    },
    "ExponentOmega": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1",
    },
    "ExponentAmplitude": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1",
    },
    "ExponentLambda": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1 && (ExponentStrategyMode==2 || ExponentStrategyMode==3)",
    },
    "ExponentModeProb_Walk": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1 && ExponentStrategyMode==2",
    },
    "ExponentModeProb_Homo": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1 && ExponentStrategyMode==2",
    },
    "ExponentModeProb_Diff": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1 && ExponentStrategyMode==2",
    },
    "ExponentModeCustom": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1 && ExponentStrategyMode==3",
    },
    "ExponentSwitchProb": {
        "parent": "ExponentStrategyMode",
        "required": "ExponentChangeEnable==1 && ExponentStrategyMode==4",
    },
    "RotationChangeMode": {
        "parent": "RotationAngleEnable",
        "required": "RotationAngleEnable==1",
    },
    "AngleSeverity": {
        "parent": "RotationAngleEnable",
        "required": "RotationAngleEnable==1",
    },
    "RotationFlipProb": {
        "parent": "RotationTopologyEnable",
        "required": "RotationTopologyEnable==1",
    },
    "FormSwitchProb": {"parent": "FormSwitchEnable", "required": "FormSwitchEnable==1"},
    "AddTransformProb": {
        "parent": "TransformSeqChangeEnable",
        "required": "TransformSeqChangeEnable==1",
    },
    "RemoveTransformProb": {
        "parent": "TransformSeqChangeEnable",
        "required": "TransformSeqChangeEnable==1",
    },
    "ReplaceTransformProb": {
        "parent": "TransformSeqChangeEnable",
        "required": "TransformSeqChangeEnable==1",
    },
    "ShuffleTransformProb": {
        "parent": "TransformSeqChangeEnable",
        "required": "TransformSeqChangeEnable==1",
    },
    "TransformParamLambda": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "AP_mu_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "AP_mu_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "AP_gamma_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "AP_gamma_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "AP_omega_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "AP_omega_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "LS_mu_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "LS_mu_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "LS_omega_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "LS_omega_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "WM_mu_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "WM_mu_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "WM_omega_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "WM_omega_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "WM_eta_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "TI_mu_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "TI_mu_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "TI_omega_plus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "TI_omega_minus_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "RP_mu_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "RP_p_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "RP_q_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "RP_omega_Severity": {
        "parent": "TransformParamChangeEnable",
        "required": "TransformParamChangeEnable==1",
    },
    "AddComponentProb": {
        "parent": "ComponentCountChangeEnable",
        "required": "ComponentCountChangeEnable==1",
    },
    "RemoveComponentProb": {
        "parent": "ComponentCountChangeEnable",
        "required": "ComponentCountChangeEnable==1",
    },
    "K_min": {
        "parent": "ComponentCountChangeEnable",
        "required": "ComponentCountChangeEnable==1",
    },
    "K_max": {
        "parent": "ComponentCountChangeEnable",
        "required": "ComponentCountChangeEnable==1",
    },
}


def _sample_portal_params(k_count: int, dimension: int, problem: DPORTALProblem, rng) -> _PortalStruct:
    params = _PortalStruct()
    params.d = dimension
    params.bounds_min = float(problem.MinCoordinate)
    params.bounds_max = float(problem.MaxCoordinate)
    params.c = _init_centers(k_count, dimension, problem, rng)
    params.beta = _init_beta(k_count, problem, rng)
    params.kappa_plus = _init_kappa(k_count, dimension, problem, rng)
    params.kappa_minus = _init_kappa(k_count, dimension, problem, rng)
    params.p_single = _init_exponents(k_count, 1, problem, rng).reshape(-1)
    params.p_plus = _init_exponents(k_count, dimension, problem, rng)
    params.p_minus = _init_exponents(k_count, dimension, problem, rng)
    params.form = _sample_forms(k_count, problem, rng)
    params.neutralize_enable = True
    # A component falls by Delta where every rotated coordinate is r_ref from the summit; Delta is drawn even
    # when its range is a single value, so the random stream does not depend on the range.
    delta_low, delta_high = float(problem.NeutralizeDeltaMin), float(problem.NeutralizeDeltaMax)
    params.Delta = delta_low + (delta_high - delta_low) * rng.random(k_count)
    params.r_ref = float(problem.NeutralizeRRef) + 0 * rng.random(k_count)
    params.R = _init_rotations(k_count, dimension, problem, rng)
    params.transform_seq = _init_transform_sequences(k_count, problem, rng)
    _init_transform_parameters(params, problem, rng)
    _recompute_scalings(params)
    if bool(problem.EnsureNoCover):
        _redraw_until_uncovered(params, problem, rng)
    params.center_velocity = np.zeros((k_count, dimension), dtype=float)
    params.beta_velocity = np.zeros(k_count, dtype=float)
    params.kappa_plus_velocity = np.zeros((k_count, dimension), dtype=float)
    params.kappa_minus_velocity = np.zeros((k_count, dimension), dtype=float)
    params.p_plus_velocity = np.zeros((k_count, dimension), dtype=float)
    params.p_minus_velocity = np.zeros((k_count, dimension), dtype=float)
    params.p_single_velocity = np.zeros(k_count, dtype=float)
    params.phase = 2 * np.pi * rng.random(k_count)
    if int(problem.RotationChangeMode) == 2:
        # GMPB's rotation change: the initial rotation and one angle per component (drawn only in this mode, so
        # the random stream of mode 1 is unchanged).
        params.R0 = params.R.copy()
        params.rotation_angle = rng.uniform(-np.pi, np.pi, k_count)
    params.center_modes = _mode_assignments(
        k_count,
        int(problem.CenterStrategyMode),
        int(problem.CenterChangeMode),
        [problem.CenterModeProb_MomentumWalk, problem.CenterModeProb_Circular, problem.CenterModeProb_AR],
        str(problem.CenterModeCustom),
        rng,
        3,
    )
    params.beta_modes = _mode_assignments(
        k_count,
        int(problem.BetaStrategyMode),
        int(problem.BetaChangeMode),
        [problem.BetaModeProb_Walk, problem.BetaModeProb_Periodic],
        str(problem.BetaModeCustom),
        rng,
        2,
    )
    params.kappa_modes = _mode_assignments(
        k_count,
        int(problem.KappaStrategyMode),
        int(problem.KappaChangeMode),
        [problem.KappaModeProb_Walk, problem.KappaModeProb_Homo, problem.KappaModeProb_Diff],
        str(problem.KappaModeCustom),
        rng,
        3,
    )
    params.exponent_modes = _mode_assignments(
        k_count,
        int(problem.ExponentStrategyMode),
        int(problem.ExponentChangeMode),
        [problem.ExponentModeProb_Walk, problem.ExponentModeProb_Homo, problem.ExponentModeProb_Diff],
        str(problem.ExponentModeCustom),
        rng,
        3,
    )
    return params


def _evolve_portal_params(previous: _PortalStruct, problem: DPORTALProblem, rng,
                          *, shift_severity: float | None = None) -> _PortalStruct:
    current = _copy_params(previous)

    if bool(problem.CenterChangeEnable):
        current.c, current.center_velocity = _evolve_centers(
            current, problem, rng, shift_severity=shift_severity,
        )
        current.c = _reflect(current.c, current.bounds_min, current.bounds_max)
    if bool(problem.BetaChangeEnable):
        current.beta, current.beta_velocity = _evolve_scalar_modes(
            current.beta,
            current.beta_velocity,
            current.beta_modes,
            float(problem.BetaSeverity),
            float(_beta_lambda(problem)),
            *_HEIGHT_RANGE,
            rng,
            phase=current.phase,
            omega=float(_beta_omega(problem)),
            amplitude=float(_beta_amplitude(problem)),
        )
    if bool(problem.KappaChangeEnable):
        current.kappa_plus, current.kappa_plus_velocity = _evolve_matrix_modes(
            current.kappa_plus,
            current.kappa_plus_velocity,
            current.kappa_modes,
            float(problem.KappaSeverity),
            float(_kappa_lambda(problem)),
            float(problem.KappaMin),
            float(problem.KappaMax),
            rng,
        )
        current.kappa_minus, current.kappa_minus_velocity = _evolve_matrix_modes(
            current.kappa_minus,
            current.kappa_minus_velocity,
            current.kappa_modes,
            float(problem.KappaSeverity),
            float(_kappa_lambda(problem)),
            float(problem.KappaMin),
            float(problem.KappaMax),
            rng,
        )
    if bool(problem.ExponentChangeEnable):
        current.p_plus, current.p_plus_velocity = _evolve_matrix_modes(
            current.p_plus,
            current.p_plus_velocity,
            current.exponent_modes,
            float(problem.PSeverity),
            float(_exponent_lambda(problem)),
            *_exponent_range(problem),
            rng,
        )
        current.p_minus, current.p_minus_velocity = _evolve_matrix_modes(
            current.p_minus,
            current.p_minus_velocity,
            current.exponent_modes,
            float(problem.PSeverity),
            float(_exponent_lambda(problem)),
            *_exponent_range(problem),
            rng,
        )
        current.p_single, current.p_single_velocity = _evolve_scalar_modes(
            current.p_single,
            current.p_single_velocity,
            current.exponent_modes,
            float(problem.PSeverity),
            float(_exponent_lambda(problem)),
            *_exponent_range(problem),
            rng,
            phase=current.phase,
            omega=float(problem.ExponentOmega),
            amplitude=float(problem.ExponentAmplitude),
        )
    if bool(problem.RotationAngleEnable) and int(problem.RotationChangeMode) == 2:
        current.rotation_angle = _reflect(
            current.rotation_angle + float(problem.AngleSeverity) * rng.standard_normal(current.c.shape[0]), -np.pi, np.pi)
        current.R = np.asarray([initial @ _all_plane_rotation(angle, current.d, rng)
                                for initial, angle in zip(current.R0, current.rotation_angle)])
    elif bool(problem.RotationAngleEnable):
        current.R = _evolve_rotations(current.R, float(problem.AngleSeverity), rng)
    if bool(problem.RotationTopologyEnable) and rng.random() < float(problem.RotationFlipProb):
        current.R = _init_rotations(current.c.shape[0], int(problem.Dimension), problem, rng)
        if int(problem.RotationChangeMode) == 2:
            current.R0 = current.R.copy()
    if bool(problem.FormSwitchEnable):
        switches = rng.random(current.form.shape[0]) < float(problem.FormSwitchProb)
        current.form[switches] = 3 - current.form[switches]
    if bool(problem.TransformSeqChangeEnable):
        current.transform_seq = _evolve_transform_sequences(current.transform_seq, problem, rng)
    if bool(problem.TransformParamChangeEnable):
        _evolve_transform_parameters(current, problem, rng)
    if bool(problem.ComponentCountChangeEnable):
        current = _evolve_component_count(current, problem, rng)
    _recompute_scalings(current)
    return current


def _sample_forms(k_count: int, problem: DPORTALProblem, rng) -> np.ndarray:
    """Form A (1) or B (2). Modes 1, 5 and 6 are PORTAL2's modes 1, 2 and 4; PORTAL2's mode 3 is 2 or 3 here."""
    mode = int(problem.BaselineInitMode)
    if mode == 2:
        return np.ones(k_count, dtype=int)
    if mode == 3:
        return 2 * np.ones(k_count, dtype=int)
    if mode == 4:
        return np.asarray([(idx % 2) + 1 for idx in range(k_count)], dtype=int)
    if mode == 5:
        return np.where(rng.random(k_count) < float(problem.BaselineProbA), 1, 2)
    if mode == 6:
        forms = [{"A": 1, "B": 2}[item.strip().upper()] for item in str(problem.BaselineSeq).split(",") if item.strip()]
        if len(forms) != k_count:
            raise ValueError(f"DPORTAL BaselineSeq must list one form per component ({k_count}); received {len(forms)}.")
        return np.asarray(forms, dtype=int)
    return rng.integers(1, 3, size=k_count)


def _redraw_until_uncovered(params: _PortalStruct, problem: DPORTALProblem, rng) -> None:
    """PORTAL2's no-cover condition: redraw centres and heights until no component exceeds another's height at
    that component's centre, so that every component is a visible peak of the first environment."""
    tolerance = float(problem.NoCoverTolerance)
    attempts = int(problem.NoCoverMaxAttempts)
    for attempt in range(attempts):
        values = _phi_components_batch(np.asarray(params.c, dtype=float), params)
        np.fill_diagonal(values, -np.inf)
        covered = np.flatnonzero(values.max(axis=1) > params.beta + tolerance)
        if covered.size == 0:
            return
        if attempt + 1 < attempts:
            params.c = _init_centers(params.c.shape[0], params.c.shape[1], problem, rng)
            params.beta = _init_beta(params.c.shape[0], problem, rng)
    raise RuntimeError(
        f"DPORTAL could not draw {params.c.shape[0]} uncovered components in {int(problem.NoCoverMaxAttempts)} "
        f"attempts (covered: {(covered + 1).tolist()[:8]}); raise NoCoverMaxAttempts or set EnsureNoCover=false.")


def _reflect(values, lower: float, upper: float):
    """Mirror every value that leaves [lower, upper] at the bound it crossed, repeatedly, until it lies inside."""
    reflected = np.asarray(values, dtype=float).copy()
    if upper <= lower:
        return np.full_like(reflected, lower)
    while True:
        high = reflected > upper
        reflected[high] = 2 * upper - reflected[high]
        low = reflected < lower
        reflected[low] = 2 * lower - reflected[low]
        if not (high.any() or low.any()):
            return reflected


def _init_centers(k_count: int, dimension: int, problem: DPORTALProblem, rng) -> np.ndarray:
    lower = float(problem.MinCoordinate)
    upper = float(problem.MaxCoordinate)
    span = upper - lower
    mode = int(problem.CenterInitMode)
    if mode == 3:
        center = lower + span * rng.random(dimension)
        return np.repeat(center[None, :], k_count, axis=0)
    if mode == 4:
        centers = lower + span * _halton_sequence(k_count, dimension)
        perturbation = float(problem.QuasiUniformPerturbation)
        if perturbation > 0:
            perturb = perturbation * (2 * rng.random((k_count, dimension)) - 1)
        else:
            perturb = 0.02 * span * (2 * rng.random((k_count, dimension)) - 1)
        return np.clip(centers + perturb, lower, upper)
    if mode == 2:
        exclusion = float(problem.ExclusionZone)
        lower_range = lower + ((-exclusion) - lower) * rng.random((k_count, dimension))
        upper_range = exclusion + (upper - exclusion) * rng.random((k_count, dimension))
        selector = rng.integers(0, 2, size=(k_count, dimension))
        return selector * lower_range + (1 - selector) * upper_range
    return lower + span * rng.random((k_count, dimension))


# Heights of DynPORTAL (PORTAL2 draws depths in [0, 10] for minimisation; DynPORTAL keeps GMPB's scale).
_HEIGHT_RANGE = (30, 70)


def _init_beta(k_count: int, problem: DPORTALProblem, rng) -> np.ndarray:
    """Modes 1-5 are PORTAL2's; mode 6 is PORTAL2's optimum-protected mode, mirrored for maximisation."""
    low, high = _HEIGHT_RANGE
    span = high - low
    mode = int(problem.BetaOffsetInitMode)
    if mode == 2:
        return np.full(k_count, low + span * rng.random(), dtype=float)
    if mode == 3:
        return np.full(k_count, float(problem.FixedBetaOffset), dtype=float)
    if mode == 4:
        return low + span * rng.beta(float(problem.BetaOffsetDistAlpha), float(problem.BetaOffsetDistBeta), k_count)
    if mode == 5:
        return rng.permutation(np.linspace(low, high, k_count))
    if mode == 6:
        # One component is the highest; every other lies BetaOptGap to BetaGap below it.
        gap = min(float(problem.BetaGap), span)
        opt_gap = min(float(problem.BetaOptGap), gap)
        top = high - (span - gap) * rng.random()
        heights = top - opt_gap - (gap - opt_gap) * rng.random(k_count)
        heights[int(rng.integers(k_count))] = top
        return heights
    return low + span * rng.random(k_count)


def _init_kappa(k_count: int, dimension: int, problem: DPORTALProblem, rng) -> np.ndarray:
    mode = int(problem.KappaInitMode)
    low, high = float(problem.KappaMin), float(problem.KappaMax)
    if mode == 2:
        return low + (high - low) * rng.beta(float(problem.KappaDistAlpha), float(problem.KappaDistBeta), (k_count, dimension))
    if mode == 3:  # isotropic per component
        return np.repeat(low + (high - low) * rng.random((k_count, 1)), dimension, axis=1)
    if mode == 4:
        values = np.linspace(low, high, dimension)
        return np.asarray([rng.permutation(values) for _ in range(k_count)], dtype=float)
    if mode == 5:
        return np.ones((k_count, dimension), dtype=float)
    return low + (high - low) * rng.random((k_count, dimension))


def _exponent_range(problem: DPORTALProblem) -> tuple[float, float]:
    return float(problem.ExponentMin), float(problem.ExponentMax)


def _init_exponents(k_count: int, dimension: int, problem: DPORTALProblem, rng) -> np.ndarray:
    low, high = _exponent_range(problem)
    # Rounded so that the default span is exactly 1.
    span = round(high - low, 12)
    mode = int(problem.PInitMode)
    if mode == 2:
        return np.full((k_count, dimension), float(problem.FixedP), dtype=float)
    if mode == 3:  # gradient over the dimensions
        return np.tile(np.linspace(low, high, dimension), (k_count, 1))
    if mode == 4:
        return low + span * rng.beta(float(problem.ExponentDistAlpha), float(problem.ExponentDistBeta), (k_count, dimension))
    if mode == 5:
        if dimension == 1:
            return low + span * rng.random((k_count, dimension))
        values = np.linspace(low, high, dimension)
        return np.asarray([rng.permutation(values) for _ in range(k_count)], dtype=float)
    return low + span * rng.random((k_count, dimension))


def _init_rotations(k_count: int, dimension: int, problem: DPORTALProblem, rng) -> np.ndarray:
    mode = int(problem.RotationInitMode)
    rotations = np.repeat(np.eye(dimension)[None, :, :], k_count, axis=0)
    if mode == 1 or dimension < 2:
        return rotations
    for index in range(k_count):
        rotations[index] = _givens_rotation(_rotation_angles(dimension, mode, problem, rng))
    return rotations


def _rotation_angles(dimension: int, mode: int, problem: DPORTALProblem, rng) -> np.ndarray:
    """Upper-triangular Givens angles (radians) of one component, by the PORTAL rotation modes."""
    angles = np.zeros((dimension, dimension), dtype=float)
    pairs = [(left, right) for left in range(dimension - 1) for right in range(left + 1, dimension)]
    if mode == 2:  # fully connected
        for pair in pairs:
            angles[pair] = rng.uniform(-np.pi, np.pi)
    elif mode == 3:  # each pair with probability RotationProb
        for pair in pairs:
            if rng.random() < float(problem.RotationProb):
                angles[pair] = rng.uniform(-np.pi, np.pi)
    elif mode == 4:  # one angle shared by every pair
        shared = rng.uniform(-np.pi, np.pi)
        for pair in pairs:
            angles[pair] = shared
    elif mode == 5:  # the given angle on every pair
        for pair in pairs:
            angles[pair] = np.deg2rad(float(problem.SpecificAngleDeg))
    elif mode == 6:  # chain
        for left in range(dimension - 1):
            angles[left, left + 1] = rng.uniform(-np.pi, np.pi)
    elif mode == 7:  # block-diagonal over a random grouping of the variables
        block_count = 3 if dimension >= 30 else 2
        sizes = [dimension // block_count + (block < dimension % block_count) for block in range(block_count)]
        order = rng.permutation(dimension)
        block_angles = np.deg2rad([45.0, 135.0, 22.5, 112.5])
        start = 0
        for block, size in enumerate(sizes):
            group = sorted(int(variable) for variable in order[start:start + size])
            start += size
            for position, left in enumerate(group):
                for right in group[position + 1:]:
                    angles[left, right] = block_angles[block % block_angles.size]
    return angles


def _givens_rotation(angles: np.ndarray) -> np.ndarray:
    dimension = angles.shape[0]
    rotation = np.eye(dimension)
    for left in range(dimension - 1):
        for right in range(left + 1, dimension):
            if abs(angles[left, right]) < 1e-15:
                continue
            rotation = rotation @ _plane_rotation(dimension, angles[left, right], left, right)
    return rotation


def _halton_sequence(count: int, dimension: int) -> np.ndarray:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    output = np.zeros((count, dimension), dtype=float)
    for dim in range(dimension):
        base = primes[min(dim, len(primes) - 1)]
        for index in range(count):
            output[index, dim] = _halton_element(index + 1, base)
    return output


def _halton_element(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    current = int(index)
    while current > 0:
        value += factor * (current % base)
        current //= base
        factor /= base
    return value


def _plane_rotation(dimension: int, angle: float, left: int, right: int) -> np.ndarray:
    matrix = np.eye(dimension)
    c = np.cos(angle)
    s = np.sin(angle)
    matrix[left, left] = c
    matrix[right, right] = c
    matrix[left, right] = -s
    matrix[right, left] = s
    return matrix


_TRANSFORM_NAMES = ["additive_periodic", "log_sinusoidal", "wavelet_mod", "tensor_interference", "radial_polytrig"]


def _transform_pool(problem: DPORTALProblem) -> list[str]:
    """Transformation types drawn for new or replaced sequence entries (all five by default, in this order)."""
    pool = [name.strip() for name in str(problem.TransformTypePool).split(",") if name.strip()]
    unknown = sorted(set(pool) - set(_TRANSFORM_NAMES))
    if not pool or unknown:
        raise ValueError(f"DPORTAL TransformTypePool must list known transformations; unknown {unknown}.")
    return pool


def _base_transform(problem: DPORTALProblem) -> str:
    base = str(problem.BaseTransform).strip()
    if base and base not in _TRANSFORM_NAMES:
        raise ValueError(f"DPORTAL BaseTransform must be empty or one of {_TRANSFORM_NAMES}; received {base!r}.")
    return base


def _extra_pool(problem: DPORTALProblem) -> list[str]:
    """Types drawn for sequence entries: the pool, without the base transformation when there is one."""
    base = _base_transform(problem)
    names = [name for name in _transform_pool(problem) if name != base]
    if not names:
        raise ValueError("DPORTAL TransformTypePool must contain a type other than BaseTransform.")
    return names


def _init_transform_sequences(k_count: int, problem: DPORTALProblem, rng) -> list[list[str]]:
    """With a BaseTransform every sequence starts with it, and the modes below draw the further entries."""
    base = _base_transform(problem)
    sequences = _draw_transform_sequences(k_count, problem, rng)
    return [[base, *sequence] for sequence in sequences] if base else sequences


def _draw_transform_sequences(k_count: int, problem: DPORTALProblem, rng) -> list[list[str]]:
    """Mode 1 draws TransformSeqMinLen-TransformSeqMaxLen transformations (PORTAL2 mode 1); 2 none (PORTAL2 mode 3);
    3 and 4 one and two; 5 TransformSeqCount (PORTAL2 mode 2); 6 the list TransformSeqCustom (PORTAL2 mode 4)."""
    mode = int(problem.TransformSeqInitMode)
    names = _extra_pool(problem)
    if mode == 2:
        return [[] for _ in range(k_count)]
    if mode == 6:
        # Components separated by ";", transformations by ","; a single entry applies to every component.
        custom = [[name.strip() for name in part.split(",") if name.strip()] for part in str(problem.TransformSeqCustom).split(";")]
        unknown = sorted({name for part in custom for name in part} - set(_TRANSFORM_NAMES))
        if unknown:
            raise ValueError(f"DPORTAL TransformSeqCustom has unknown transformations {unknown}.")
        if len(custom) == 1:
            custom = custom * k_count
        if len(custom) != k_count:
            raise ValueError(f"DPORTAL TransformSeqCustom must give one or {k_count} sequences; received {len(custom)}.")
        return custom
    sequences = []
    for index in range(k_count):
        if mode == 3:
            length = 1
        elif mode == 4:
            length = 2
        elif mode == 5:
            length = int(problem.TransformSeqCount)
        else:
            length = int(rng.integers(int(problem.TransformSeqMinLen), int(problem.TransformSeqMaxLen) + 1))
        sequences.append([str(rng.choice(names)) for _ in range(length)])
    return sequences


# Sampling ranges of the PORTAL reference generator, by transformation and parameter family
# (``mu_plus`` and ``mu_minus`` share the range of ``mu``; ``omega1_*`` and ``omega2_*`` that of ``omega``).
_TRANSFORM_RANGES = {
    "T_additive_periodic": {"mu": (0.1, 0.7), "gamma": (0.002, 0.2), "omega": (0.05, 1.0)},
    "T_log_sinusoidal": {"mu": (0.05, 0.5), "omega": (5, 50)},
    "T_wavelet_mod": {"mu": (10, 50), "omega": (0.3, 1.0), "eta": (10, 24)},
    "T_tensor_interference": {"mu": (10, 20), "omega": (0.1, 0.7)},
    "T_radial_polytrig": {"mu": (0.4, 2.0), "p": (0.4, 0.7), "q": (0.4, 1.2), "omega": (0.1, 10)},
}


def _transform_range(block: str, key: str, ranges=_TRANSFORM_RANGES) -> tuple[float, float]:
    return ranges[block][key.split("_")[0].rstrip("12")]


def _transform_ranges(problem: DPORTALProblem) -> dict:
    """PORTAL's ranges with the TransformRangeOverrides, e.g. "log_sinusoidal.mu=0.1:1.0;radial_polytrig.mu=0.1:0.5"."""
    ranges = {block: dict(families) for block, families in _TRANSFORM_RANGES.items()}
    for entry in filter(None, (part.strip() for part in str(problem.TransformRangeOverrides).split(";"))):
        try:
            target, bounds = entry.split("=")
            name, family = target.strip().split(".")
            low, high = (float(value) for value in bounds.split(":"))
        except ValueError as error:
            raise ValueError(f"DPORTAL TransformRangeOverrides entry {entry!r} is not type.family=low:high.") from error
        block = f"T_{name}"
        if block not in ranges or family not in ranges[block] or not 0 <= low <= high:
            raise ValueError(f"DPORTAL TransformRangeOverrides entry {entry!r} names no range or has low > high.")
        ranges[block][family] = (low, high)
    return ranges


def _init_transform_parameters(params: _PortalStruct, problem: DPORTALProblem, rng) -> None:
    k_count, dimension = params.c.shape
    ranges = _transform_ranges(problem)
    AP, LS, WM, TI, RP = (ranges[name] for name in (
        "T_additive_periodic", "T_log_sinusoidal", "T_wavelet_mod", "T_tensor_interference", "T_radial_polytrig"))
    params.T_additive_periodic = _PortalStruct(
        mu_plus=_transform_values(k_count, dimension, problem, rng, "AP_mu_plus_Severity", *AP["mu"]),
        mu_minus=_transform_values(k_count, dimension, problem, rng, "AP_mu_minus_Severity", *AP["mu"]),
        gamma_plus=_transform_values(k_count, dimension, problem, rng, "AP_gamma_plus_Severity", *AP["gamma"]),
        gamma_minus=_transform_values(k_count, dimension, problem, rng, "AP_gamma_minus_Severity", *AP["gamma"]),
        omega_plus=_transform_values(k_count, dimension, problem, rng, "AP_omega_plus_Severity", *AP["omega"]),
        omega_minus=_transform_values(k_count, dimension, problem, rng, "AP_omega_minus_Severity", *AP["omega"]),
    )
    params.T_log_sinusoidal = _PortalStruct(
        mu_plus=_transform_values(k_count, dimension, problem, rng, "LS_mu_plus_Severity", *LS["mu"]),
        mu_minus=_transform_values(k_count, dimension, problem, rng, "LS_mu_minus_Severity", *LS["mu"]),
        omega1_plus=_transform_values(k_count, dimension, problem, rng, "LS_omega_plus_Severity", *LS["omega"]),
        omega1_minus=_transform_values(k_count, dimension, problem, rng, "LS_omega_minus_Severity", *LS["omega"]),
        omega2_plus=_transform_values(k_count, dimension, problem, rng, "LS_omega_plus_Severity", *LS["omega"]),
        omega2_minus=_transform_values(k_count, dimension, problem, rng, "LS_omega_minus_Severity", *LS["omega"]),
        # Scale of the origin gate; a constant of the reference evaluator.
        sigma=np.full(k_count, 0.1, dtype=float),
    )
    params.T_wavelet_mod = _PortalStruct(
        mu_plus=_transform_values(k_count, dimension, problem, rng, "WM_mu_plus_Severity", *WM["mu"]),
        mu_minus=_transform_values(k_count, dimension, problem, rng, "WM_mu_minus_Severity", *WM["mu"]),
        omega_plus=_transform_values(k_count, dimension, problem, rng, "WM_omega_plus_Severity", *WM["omega"]),
        omega_minus=_transform_values(k_count, dimension, problem, rng, "WM_omega_minus_Severity", *WM["omega"]),
        eta=_transform_column_values(k_count, problem, rng, *WM["eta"]),
        eps=1e-12,
    )
    params.T_tensor_interference = _PortalStruct(
        mu_plus=_transform_values(k_count, dimension, problem, rng, "TI_mu_plus_Severity", *TI["mu"]),
        mu_minus=_transform_values(k_count, dimension, problem, rng, "TI_mu_minus_Severity", *TI["mu"]),
        omega_plus=_transform_values(k_count, dimension, problem, rng, "TI_omega_plus_Severity", *TI["omega"]),
        omega_minus=_transform_values(k_count, dimension, problem, rng, "TI_omega_minus_Severity", *TI["omega"]),
    )
    params.T_radial_polytrig = _PortalStruct(
        mu=_transform_column_values(k_count, problem, rng, *RP["mu"]),
        p=_transform_column_values(k_count, problem, rng, *RP["p"]),
        q=_transform_column_values(k_count, problem, rng, *RP["q"]),
        omega=_transform_column_values(k_count, problem, rng, *RP["omega"]),
        eps=1e-12,
    )


def _scale_transform_amplitudes(params: _PortalStruct, scale: float) -> None:
    """Multiply the amplitude mu of every transformation; mu = 0 leaves a coordinate unchanged."""
    for block in _TRANSFORM_RANGES:
        group = params[block]
        for key in [key for key in group if key == "mu" or key.startswith("mu_")]:
            group[key] = group[key] * scale


def _length_types(problem: DPORTALProblem) -> set[str]:
    types = {name.strip() for name in str(problem.TransformLengthTypes).split(",") if name.strip()}
    allowed = set(_TRANSFORM_NAMES) - {"log_sinusoidal"}
    if not types <= allowed:
        raise ValueError(f"DPORTAL TransformLengthTypes must list types among {sorted(allowed)}.")
    return types


def _shrink_transform_length(params: _PortalStruct, length: float, types) -> None:
    """Replace each listed transformation T by length * T(a / length): its pattern contracts towards the summit by
    the factor length, with the relative distortion unchanged. The log-sinusoidal map is self-similar and is kept."""
    if "additive_periodic" in types:
        group = params.T_additive_periodic
        for key in ("gamma_plus", "gamma_minus", "omega_plus", "omega_minus"):
            group[key] = group[key] / length
    for name in ("wavelet_mod", "tensor_interference"):
        if name in types:
            group = params[f"T_{name}"]
            for side in ("plus", "minus"):
                group[f"omega_{side}"] = group[f"omega_{side}"] / length
                group[f"mu_{side}"] = group[f"mu_{side}"] * length
    if "radial_polytrig" in types:
        group = params.T_radial_polytrig
        group.mu = group.mu * length ** (1 - group.p)
        group.omega = group.omega * length ** (-group.q)


def _transform_values(k_count, dimension, problem, rng, severity_name, low, high):
    """(K, D) values of one transformation parameter. Modes 1, 2, 6, 7, 8 and 4 are PORTAL2's modes 1-6
    (independent, shared by the components, shared by the dimensions, one global value, linear with a permutation
    per component, Beta); 3 (midpoint) and 5 (linear, unpermuted) are DynPORTAL's."""
    del severity_name
    mode = int(problem.TransformParamInitMode)
    if mode == 2:
        values = low + (high - low) * rng.random(dimension)
        return np.repeat(values[None, :], k_count, axis=0)
    if mode == 3:
        return np.full((k_count, dimension), (low + high) / 2, dtype=float)
    if mode == 4:
        return low + (high - low) * rng.beta(float(problem.TransformDistAlpha), float(problem.TransformDistBeta), (k_count, dimension))
    if mode == 5:
        return np.tile(np.linspace(low, high, dimension), (k_count, 1))
    if mode == 6:
        return np.repeat(low + (high - low) * rng.random((k_count, 1)), dimension, axis=1)
    if mode == 7:
        return np.full((k_count, dimension), low + (high - low) * rng.random(), dtype=float)
    if mode == 8:
        values = np.linspace(low, high, dimension)
        return np.asarray([rng.permutation(values) for _ in range(k_count)], dtype=float)
    return low + (high - low) * rng.random((k_count, dimension))


def _transform_column_values(k_count, problem, rng, low, high):
    """Per-component values (wavelet eta, radial-polytrig parameters): uniform as in PORTAL2, except under
    DynPORTAL's modes 3-5, which draw them with the distribution of the mode."""
    if int(problem.TransformParamInitMode) in (3, 4, 5):
        return _transform_values(k_count, 1, problem, rng, "", low, high).reshape(k_count)
    return low + (high - low) * rng.random(k_count)


def _recompute_scalings(params: _PortalStruct) -> None:
    k_count, dimension = params.c.shape
    params.rho_single = np.full(k_count, np.nan, dtype=params.c.dtype)
    params.rho_plus = np.full((k_count, dimension), np.nan, dtype=params.c.dtype)
    params.rho_minus = np.full((k_count, dimension), np.nan, dtype=params.c.dtype)
    if not bool(getattr(params, "neutralize_enable", True)):
        form_a = params.form == 1
        params.rho_plus[form_a, :] = 1.0
        params.rho_minus[form_a, :] = 1.0
        params.rho_single[~form_a] = 1.0
        return

    for component_index in range(k_count):
        kappa_average_by_dimension = (
            params.kappa_plus[component_index, :] + params.kappa_minus[component_index, :]
        ) / 2.0
        if int(params.form[component_index]) == 1:
            kappa_average = float(np.mean(kappa_average_by_dimension))
            for dim in range(dimension):
                params.rho_plus[component_index, dim] = (
                    params.Delta[component_index] / dimension
                ) / (
                    kappa_average
                    * (params.r_ref[component_index] ** (2 * params.p_plus[component_index, dim]))
                )
                params.rho_minus[component_index, dim] = (
                    params.Delta[component_index] / dimension
                ) / (
                    kappa_average
                    * (params.r_ref[component_index] ** (2 * params.p_minus[component_index, dim]))
                )
        else:
            exponent = max(float(params.p_single[component_index]), np.finfo(float).eps)
            sum_term = float(np.sum(kappa_average_by_dimension ** (1 / exponent)))
            params.rho_single[component_index] = params.Delta[component_index] / (
                (params.r_ref[component_index] ** 2 * sum_term) ** exponent
            )


def _copy_params(previous: _PortalStruct) -> _PortalStruct:
    current = _PortalStruct()
    for key, value in previous.items():
        if isinstance(value, np.ndarray):
            current[key] = value.copy()
        elif isinstance(value, list):
            current[key] = [list(item) if isinstance(item, list) else item for item in value]
        elif isinstance(value, _PortalStruct):
            current[key] = _copy_params(value)
        else:
            current[key] = value
    return current


def _mode_assignments(k_count, strategy, homogeneous_mode, probabilities, custom, rng, mode_count):
    if strategy == 1:
        return np.full(k_count, int(homogeneous_mode), dtype=int)
    if strategy == 3:
        values = _parse_int_list(custom)
        if not values:
            values = list(range(1, mode_count + 1))
        return np.asarray([values[index % len(values)] for index in range(k_count)], dtype=int)
    if strategy == 4:
        return rng.integers(1, mode_count + 1, size=k_count)
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.full(mode_count, 1 / mode_count)
    return rng.choice(np.arange(1, mode_count + 1), size=k_count, p=probs)


def _parse_int_list(value: str) -> list[int]:
    output = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            output.append(int(float(item)))
    return output


def _evolve_centers(current: _PortalStruct, problem: DPORTALProblem, rng,
                    *, shift_severity: float | None = None):
    centers = current.c.copy()
    velocity = current.center_velocity.copy()
    mean_center = centers.mean(axis=0)
    for index, mode in enumerate(current.center_modes):
        if mode == 2 and current.d >= 2:
            angle = float(problem.CenterAngleSensitivity_Homo if problem.CenterStrategyMode == 1 else problem.CenterAngleSensitivity)
            centered = centers[index] - mean_center
            rotated = _plane_rotation(current.d, angle, 0, 1) @ centered
            step = rotated - centered
        elif mode == 3:
            direction = centers[index] - mean_center
            norm = np.linalg.norm(direction)
            if norm <= np.finfo(float).eps:
                direction = _unit_vectors(rng.standard_normal((1, current.d)))[0]
            else:
                direction = direction / norm
            step = direction * float(problem.CenterRepel_Homo if problem.CenterStrategyMode == 1 else problem.CenterRepel) * 0.01
        else:
            severity = float(problem.ShiftSeverity) if shift_severity is None else shift_severity
            random_step = _unit_vectors(rng.standard_normal((1, current.d)))[0] * severity
            inertia = float(problem.CenterLambda_Homo if problem.CenterStrategyMode == 1 else problem.CenterLambda)
            velocity[index] = inertia * velocity[index] + (1 - inertia) * random_step
            step = velocity[index]
        centers[index] = centers[index] + step
    return centers, velocity


def _evolve_scalar_modes(values, velocity, modes, severity, inertia, lower, upper, rng, phase=None, omega=0.0, amplitude=0.0):
    new_values = values.copy()
    new_velocity = velocity.copy()
    for index, mode in enumerate(modes):
        if mode == 2 and phase is not None:
            step = amplitude * np.sin(phase[index] + omega)
        elif mode == 3:
            step = severity * (np.mean(values) - values[index])
        else:
            new_velocity[index] = inertia * new_velocity[index] + (1 - inertia) * severity * rng.standard_normal()
            step = new_velocity[index]
        new_values[index] += step
    return _reflect(new_values, lower, upper), new_velocity


def _evolve_matrix_modes(values, velocity, modes, severity, inertia, lower, upper, rng):
    new_values = values.copy()
    new_velocity = velocity.copy()
    for index, mode in enumerate(modes):
        if mode == 2:
            step = severity * (np.mean(values, axis=0) - values[index])
        elif mode == 3:
            direction = values[index] - np.mean(values, axis=0)
            step = severity * direction
        else:
            new_velocity[index] = (
                inertia * new_velocity[index] + (1 - inertia) * severity * rng.standard_normal(values.shape[1])
            )
            step = new_velocity[index]
        new_values[index] += step
    return _reflect(new_values, lower, upper), new_velocity


def _evolve_rotations(rotations: np.ndarray, severity: float, rng) -> np.ndarray:
    output = rotations.copy()
    dimension = output.shape[1]
    if dimension < 2:
        return output
    for index in range(output.shape[0]):
        left, right = rng.choice(dimension, size=2, replace=False)
        output[index] = _plane_rotation(dimension, severity * rng.standard_normal(), int(left), int(right)) @ output[index]
    return output


def _all_plane_rotation(angle: float, dimension: int, rng) -> np.ndarray:
    """GMPB's rotation: every coordinate plane turned by the same angle, the planes in a random order."""
    pages = []
    for left in range(dimension):
        for right in range(left + 1, dimension):
            page = np.eye(dimension)
            page[left, left] = page[right, right] = np.cos(angle)
            page[left, right], page[right, left] = np.sin(angle), -np.sin(angle)
            pages.append(page)
    output = np.eye(dimension)
    for index in rng.permutation(len(pages)):
        output = output @ pages[int(index)]
    return output


def _evolve_transform_sequences(sequences: list[list[str]], problem: DPORTALProblem, rng) -> list[list[str]]:
    """Changes act on the entries after a BaseTransform, which stays first."""
    names = _extra_pool(problem)
    base = _base_transform(problem)
    output = [list(seq[1:] if base else seq) for seq in sequences]
    shortest = int(problem.TransformSeqMinLen)
    for seq in output:
        if rng.random() < float(problem.RemoveTransformProb) and len(seq) > shortest:
            seq.pop(int(rng.integers(0, len(seq))))
        if rng.random() < float(problem.AddTransformProb) and len(seq) < int(problem.TransformSeqMaxLen):
            seq.append(str(rng.choice(names)))
        if rng.random() < float(problem.ReplaceTransformProb) and seq:
            seq[int(rng.integers(0, len(seq)))] = str(rng.choice(names))
        if rng.random() < float(problem.ShuffleTransformProb) and len(seq) > 1:
            rng.shuffle(seq)
    return [[base, *seq] for seq in output] if base else output


def _evolve_transform_parameters(params: _PortalStruct, problem: DPORTALProblem, rng) -> None:
    # Without reflection a parameter only stays nonnegative and can drift out of its PORTAL range.
    reflect = bool(problem.TransformParamReflect)
    ranges = _transform_ranges(problem)

    def step(block, key, value, severity):
        moved = value + severity * rng.standard_normal(value.shape)
        return _reflect(moved, *_transform_range(block, key, ranges)) if reflect else np.maximum(0, moved)

    for group_name, block in [
        ("AP", "T_additive_periodic"),
        ("LS", "T_log_sinusoidal"),
        ("WM", "T_wavelet_mod"),
        ("TI", "T_tensor_interference"),
    ]:
        group = params[block]
        for key, value in list(group.items()):
            if not isinstance(value, np.ndarray):
                continue
            if group_name == "WM" and key == "eta":
                severity = float(problem.WM_eta_Severity)
            elif group_name == "LS" and key == "sigma":
                continue
            else:
                severity = _severity_for_transform(group_name, key, problem)
            group[key] = step(block, key, value, severity)
    rp = params.T_radial_polytrig
    for key, severity_name in [
        ("mu", "RP_mu_Severity"),
        ("p", "RP_p_Severity"),
        ("q", "RP_q_Severity"),
        ("omega", "RP_omega_Severity"),
    ]:
        rp[key] = step("T_radial_polytrig", key, rp[key], float(getattr(problem, severity_name)))


def _severity_for_transform(group_name: str, key: str, problem: DPORTALProblem) -> float:
    mapping = {
        "AP": {"mu_plus": "AP_mu_plus_Severity", "mu_minus": "AP_mu_minus_Severity", "gamma_plus": "AP_gamma_plus_Severity", "gamma_minus": "AP_gamma_minus_Severity", "omega_plus": "AP_omega_plus_Severity", "omega_minus": "AP_omega_minus_Severity"},
        "LS": {"mu_plus": "LS_mu_plus_Severity", "mu_minus": "LS_mu_minus_Severity", "omega1_plus": "LS_omega_plus_Severity", "omega1_minus": "LS_omega_minus_Severity", "omega2_plus": "LS_omega_plus_Severity", "omega2_minus": "LS_omega_minus_Severity"},
        "WM": {"mu_plus": "WM_mu_plus_Severity", "mu_minus": "WM_mu_minus_Severity", "omega_plus": "WM_omega_plus_Severity", "omega_minus": "WM_omega_minus_Severity"},
        "TI": {"mu_plus": "TI_mu_plus_Severity", "mu_minus": "TI_mu_minus_Severity", "omega_plus": "TI_omega_plus_Severity", "omega_minus": "TI_omega_minus_Severity"},
    }
    return float(getattr(problem, mapping[group_name][key]))


def _evolve_component_count(current: _PortalStruct, problem: DPORTALProblem, rng) -> _PortalStruct:
    count = current.c.shape[0]
    if count > int(problem.K_min) and rng.random() < float(problem.RemoveComponentProb):
        keep = np.ones(count, dtype=bool)
        keep[int(rng.integers(0, count))] = False
        return _filter_components(current, keep)
    if count < int(problem.K_max) and rng.random() < float(problem.AddComponentProb):
        new_component = _sample_portal_params(1, int(problem.Dimension), problem, rng)
        return _append_component(current, new_component)
    return current


def _filter_components(params: _PortalStruct, keep: np.ndarray) -> _PortalStruct:
    for key, value in list(params.items()):
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == keep.size:
            params[key] = value[keep].copy()
        elif isinstance(value, list) and len(value) == keep.size:
            params[key] = [item for item, flag in zip(value, keep, strict=True) if flag]
        elif isinstance(value, _PortalStruct):
            for subkey, subvalue in list(value.items()):
                if isinstance(subvalue, np.ndarray) and subvalue.shape and subvalue.shape[0] == keep.size:
                    value[subkey] = subvalue[keep].copy()
    return params


def _append_component(params: _PortalStruct, new_component: _PortalStruct) -> _PortalStruct:
    for key, value in list(params.items()):
        if key in new_component and isinstance(value, np.ndarray):
            params[key] = np.concatenate([value, new_component[key]], axis=0)
        elif key in new_component and isinstance(value, list):
            params[key] = value + new_component[key]
        elif isinstance(value, _PortalStruct) and key in new_component:
            for subkey, subvalue in list(value.items()):
                if isinstance(subvalue, np.ndarray):
                    value[subkey] = np.concatenate([subvalue, new_component[key][subkey]], axis=0)
    return params


def _unit_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return values / norms


def _beta_lambda(problem: DPORTALProblem) -> float:
    return problem.BetaLambda_Homo if int(problem.BetaStrategyMode) == 1 else problem.BetaLambda


def _beta_omega(problem: DPORTALProblem) -> float:
    return problem.BetaOmega_Homo if int(problem.BetaStrategyMode) == 1 else problem.BetaOmega


def _beta_amplitude(problem: DPORTALProblem) -> float:
    return problem.BetaAmplitude_Homo if int(problem.BetaStrategyMode) == 1 else problem.BetaAmplitude


def _kappa_lambda(problem: DPORTALProblem) -> float:
    return problem.KappaLambda_Homo if int(problem.KappaStrategyMode) == 1 else problem.KappaLambda


def _exponent_lambda(problem: DPORTALProblem) -> float:
    return problem.ExponentLambda_Homo if int(problem.ExponentStrategyMode) == 1 else problem.ExponentLambda


def _phi_components_batch(samples: np.ndarray, params: _PortalStruct) -> np.ndarray:
    """Component values (N, K) of points that already lie within the bounds."""
    return prepare(params).components(samples)


def _phi_components_batch_torch(samples, params: _PortalStruct, torch, device, prepared=None):
    if prepared is None:
        prepared = TorchEnvironment(params, torch, device)
    rows = prepared.rows
    component_values = []
    for component_index in range(int(params.c.shape[0])):
        z = (samples - prepared.centers[component_index]) @ prepared.rotations[component_index]
        z = _apply_transform_seq_batch_torch(z, params, params.transform_seq[component_index], component_index, torch, device, prepared)
        if int(params.form[component_index]) == 1:
            positive = z >= 0
            h = torch.where(
                positive,
                rows["kappa_plus"][component_index],
                rows["kappa_minus"][component_index],
            )
            p_i = torch.where(
                positive,
                rows["p_plus"][component_index],
                rows["p_minus"][component_index],
            )
            rho_i = torch.where(
                positive,
                rows["rho_plus"][component_index],
                rows["rho_minus"][component_index],
            )
            penalty = torch.sum(rho_i * h * (torch.abs(z) ** (2 * p_i)), dim=1)
            component_values.append(float(params.beta[component_index]) - penalty)
            continue

        exponent = max(float(params.p_single[component_index]), np.finfo(float).eps)
        h = torch.where(
            z >= 0,
            rows["kappa_plus"][component_index],
            rows["kappa_minus"][component_index],
        )
        qsum = torch.sum((h ** (1 / exponent)) * (torch.abs(z) ** 2), dim=1)
        component_values.append(
            float(params.beta[component_index]) - float(params.rho_single[component_index]) * (qsum ** exponent)
        )
    return torch.stack(component_values, dim=1)


def _apply_transform_seq_batch_torch(a, params: _PortalStruct, sequence, component_index: int, torch, device, prepared=None):
    z = a
    for name in sequence:
        z = _apply_transform_once_batch_torch(z, params, str(name).lower(), component_index, torch, device, prepared)
    return z


def _apply_transform_once_batch(a: np.ndarray, params: _PortalStruct, name: str, component_index: int) -> np.ndarray:
    """One named transformation of (N, D) points of one component, by the batched evaluator."""
    points = np.asarray(a)
    if name not in TRANSFORMS:
        return points.copy()
    tables, transform = TRANSFORMS[name]
    block = np.ascontiguousarray(points.T)[:, None, :]
    return transform(block, *tables(params, np.asarray([component_index]), points.dtype))[:, 0, :].T


def _apply_transform_once_batch_torch(a, params: _PortalStruct, name: str, component_index: int, torch, device, prepared=None):
    if prepared is None:
        prepared = TorchEnvironment(params, torch, device)
    rows = prepared.transforms.get(name)
    epslog = 1e-12
    if name == "none":
        return a
    if name == "additive_periodic":
        group = params.T_additive_periodic
        positive = a >= 0
        mu = torch.where(
            positive,
            rows["mu_plus"][component_index],
            rows["mu_minus"][component_index],
        )
        gamma = torch.where(
            positive,
            rows["gamma_plus"][component_index],
            rows["gamma_minus"][component_index],
        )
        omega = torch.where(
            positive,
            rows["omega_plus"][component_index],
            rows["omega_minus"][component_index],
        )
        abs_a = torch.abs(a)
        g = a * (1 - torch.exp(-gamma * abs_a)) * torch.sin(omega * abs_a)
        return a + mu * g
    if name == "log_sinusoidal":
        group = params.T_log_sinusoidal
        positive = a > 0
        mu = torch.where(
            positive,
            rows["mu_plus"][component_index],
            rows["mu_minus"][component_index],
        )
        omega1 = torch.where(
            positive,
            rows["omega1_plus"][component_index],
            rows["omega1_minus"][component_index],
        )
        omega2 = torch.where(
            positive,
            rows["omega2_plus"][component_index],
            rows["omega2_minus"][component_index],
        )
        sigma2 = max(float(group.sigma[component_index]) ** 2, np.finfo(float).tiny)
        log_abs = torch.log(torch.abs(a) + epslog)
        gate = 1.0 - torch.exp(-(a * a) / sigma2)
        modulation = gate * mu * (torch.sin(omega1 * log_abs) + torch.sin(omega2 * log_abs))
        return torch.sign(a) * torch.exp(log_abs + modulation)
    if name == "wavelet_mod":
        group = params.T_wavelet_mod
        positive = a >= 0
        mu = torch.where(
            positive,
            rows["mu_plus"][component_index],
            rows["mu_minus"][component_index],
        )
        omega = torch.where(
            positive,
            rows["omega_plus"][component_index],
            rows["omega_minus"][component_index],
        )
        epss = float(group.eps)
        length = float(group.eta[component_index]) / torch.clamp(omega, min=epss)
        t = torch.abs(a)
        skip = (t < epss) | (omega <= 0) | (mu == 0) | (length <= 0)
        ratio = t / torch.where(skip, torch.ones_like(length), length)
        envelope = ratio ** 2 * torch.exp(-(ratio ** 2))
        phi = envelope * torch.sin(omega * t - np.pi / 2)
        candidate = a + mu * phi * (a / (t + epss))
        return torch.where(skip, a, candidate)
    if name == "tensor_interference":
        group = params.T_tensor_interference
        positive = a >= 0
        mu = torch.where(
            positive,
            rows["mu_plus"][component_index],
            rows["mu_minus"][component_index],
        )
        omega = torch.where(
            positive,
            rows["omega_plus"][component_index],
            rows["omega_minus"][component_index],
        )
        sin_terms = torch.sin(omega * a)
        squares = sin_terms**2
        scale = 2 ** ((params.d - 1) / 2.0)
        phi_even = torch.ones_like(a)
        for other in range(params.d):
            factor = squares[:, other:other + 1].expand_as(a).clone()
            factor[:, other] = 1.0
            phi_even = phi_even * factor
        candidate = a + mu * scale * sin_terms * phi_even
        return torch.where((mu == 0) | (omega <= 0), a, candidate)
    if name == "radial_polytrig":
        group = params.T_radial_polytrig
        radius = torch.linalg.norm(a, dim=1)
        epss = float(group.eps)
        safe_radius = radius + epss
        phi = (radius ** float(group.p[component_index])) * torch.sin(
            float(group.omega[component_index]) * (radius ** float(group.q[component_index]))
        )
        candidate = a + float(group.mu[component_index]) * phi[:, None] * (a / safe_radius[:, None])
        return torch.where((radius < epss)[:, None], a, candidate)
    return a


def _phi_component(x, params: _PortalStruct, component_index: int) -> float:
    x = np.asarray(x, dtype=params.c.dtype)
    center = params.c[component_index, :]
    a = params.R[component_index, :, :] @ (x - center)
    z = _apply_transform_seq(a, params, params.transform_seq[component_index], component_index)
    if int(params.form[component_index]) == 1:
        acc = 0.0
        for index in range(params.d):
            if z[index] >= 0:
                h = params.kappa_plus[component_index, index]
                p_i = params.p_plus[component_index, index]
                rho_i = params.rho_plus[component_index, index]
            else:
                h = params.kappa_minus[component_index, index]
                p_i = params.p_minus[component_index, index]
                rho_i = params.rho_minus[component_index, index]
            acc += rho_i * h * (abs(z[index]) ** (2 * p_i))
        return float(params.beta[component_index] - acc)

    qsum = 0.0
    for index in range(params.d):
        h = (
            params.kappa_plus[component_index, index]
            if z[index] >= 0
            else params.kappa_minus[component_index, index]
        )
        qsum += h ** (1 / max(params.p_single[component_index], np.finfo(float).eps)) * (
            abs(z[index]) ** 2
        )
    return float(
        params.beta[component_index]
        - params.rho_single[component_index] * (qsum ** params.p_single[component_index])
    )


def _apply_transform_seq(a, params: _PortalStruct, sequence, component_index: int):
    z = np.asarray(a).copy()
    for name in sequence:
        z = _apply_transform_once(z, params, str(name).lower(), component_index)
    return z


def _apply_transform_once(a, params: _PortalStruct, name: str, component_index: int):
    z = np.asarray(a).copy()
    epslog = 1e-12
    if name == "none":
        return z
    if name == "additive_periodic":
        group = params.T_additive_periodic
        for i in range(params.d):
            if a[i] >= 0:
                mu = group.mu_plus[component_index, i]
                gamma = group.gamma_plus[component_index, i]
                omega = group.omega_plus[component_index, i]
            else:
                mu = group.mu_minus[component_index, i]
                gamma = group.gamma_minus[component_index, i]
                omega = group.omega_minus[component_index, i]
            g = a[i] * (1 - np.exp(-gamma * abs(a[i]))) * np.sin(omega * abs(a[i]))
            z[i] = a[i] + mu * g
        return z
    if name == "log_sinusoidal":
        group = params.T_log_sinusoidal
        sigma2 = max(float(group.sigma[component_index]) ** 2, np.finfo(float).tiny)
        for i in range(params.d):
            if a[i] == 0:
                z[i] = 0
                continue
            abs_ai = abs(a[i])
            log_abs = np.log(abs_ai + epslog)
            if a[i] > 0:
                mu = group.mu_plus[component_index, i]
                omega1 = group.omega1_plus[component_index, i]
                omega2 = group.omega2_plus[component_index, i]
            else:
                mu = group.mu_minus[component_index, i]
                omega1 = group.omega1_minus[component_index, i]
                omega2 = group.omega2_minus[component_index, i]
            gate = 1.0 - np.exp(-(a[i] * a[i]) / sigma2)
            modulation = gate * mu * (np.sin(omega1 * log_abs) + np.sin(omega2 * log_abs))
            z[i] = np.sign(a[i]) * np.exp(log_abs + modulation)
        return z
    if name == "wavelet_mod":
        group = params.T_wavelet_mod
        eta = group.eta[component_index]
        epss = group.eps
        for i in range(params.d):
            t = abs(a[i])
            if t < epss:
                z[i] = a[i]
                continue
            if a[i] >= 0:
                mu = group.mu_plus[component_index, i]
                omega = group.omega_plus[component_index, i]
            else:
                mu = group.mu_minus[component_index, i]
                omega = group.omega_minus[component_index, i]
            length = eta / max(omega, epss)
            if omega <= 0 or mu == 0 or length <= 0:
                z[i] = a[i]
                continue
            env = (t / length) ** 2 * np.exp(-((t / length) ** 2))
            phi = env * np.sin(omega * t - np.pi / 2)
            z[i] = a[i] + mu * phi * (a[i] / (t + epss))
        return z
    if name == "tensor_interference":
        group = params.T_tensor_interference
        for i in range(params.d):
            ai = a[i]
            if ai >= 0:
                mu_i = group.mu_plus[component_index, i]
                omega_i = group.omega_plus[component_index, i]
            else:
                mu_i = group.mu_minus[component_index, i]
                omega_i = group.omega_minus[component_index, i]
            if mu_i == 0 or omega_i <= 0:
                z[i] = ai
                continue
            phi_even = 1.0
            for j in range(params.d):
                if j == i:
                    continue
                omega_j = (
                    group.omega_plus[component_index, j]
                    if a[j] >= 0
                    else group.omega_minus[component_index, j]
                )
                phi_even *= np.sin(omega_j * a[j]) ** 2
            z[i] = ai + mu_i * 2 ** ((params.d - 1) / 2.0) * np.sin(omega_i * ai) * phi_even
        return z
    if name == "radial_polytrig":
        group = params.T_radial_polytrig
        radius = float(np.linalg.norm(a))
        if radius < group.eps:
            return z
        phi = radius ** group.p[component_index] * np.sin(group.omega[component_index] * radius ** group.q[component_index])
        return z + group.mu[component_index] * phi * (a / (radius + group.eps))
    return z
