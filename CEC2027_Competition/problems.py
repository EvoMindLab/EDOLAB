"""The 24 competition problems F1, ..., F12 (GMPB) and P1, ..., P12 (DynPORTAL) as registered problems.

PROTECTED FILE. Do not modify it.

Each problem is its generator with every parameter fixed to the values of instances.py; its parameter declarations
show those values as defaults. Creating it with a different value of any parameter raises an error, so ``F1`` is always
the same instance wherever a problem name is accepted (the competition runner, ``python -m cli run --problem F1``,
batch task files).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from core.parameters import ParameterSpec, resolve_parameters
from core.registry import problem_registry
from instances.problems.dportal.problem import DPORTALProblem
from instances.problems.gmpb.problem import GMPBProblem

from CEC2027_Competition.instances import INSTANCES

GENERATORS = {"GMPB": GMPBProblem, "DPORTAL": DPORTALProblem}


def _competition_problem(name: str, generator: str, fixed: dict[str, Any]) -> type:
    base = GENERATORS[generator]

    def parameter_specs(cls) -> dict[str, ParameterSpec]:
        return {key: dataclasses.replace(spec, default=fixed[key]) for key, spec in base.parameter_specs().items()}

    def create(cls, overrides: dict[str, Any] | None = None):
        changed = sorted(key for key, value in (overrides or {}).items() if key not in fixed or fixed[key] != value)
        if changed:
            raise ValueError(f"{name} is a fixed CEC 2027 competition problem; its parameters cannot be changed "
                             f"({', '.join(changed)}). Use {generator} to experiment with other settings.")
        return cls(resolve_parameters(cls.parameter_specs(), None))

    problem = type(f"CEC2027Problem{name}", (base,), {
        "__doc__": f"CEC 2027 competition problem {name}: {generator} with the parameters of instances.py.",
        "__module__": __name__,
        "name": name,
        "generator": generator,
        "parameter_specs": classmethod(parameter_specs),
        "create": classmethod(create),
    })
    globals()[problem.__name__] = problem
    return problem_registry.register(name, problem)


PROBLEMS = {name: _competition_problem(name, generator, parameters)
            for name, (generator, parameters) in INSTANCES.items()}
