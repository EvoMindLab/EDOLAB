"""Concrete EDOLAB OO instances."""

import instances.algorithms  # noqa: F401
import instances.problems  # noqa: F401

# The competition folder next to PythonVersion/ (CEC2027_Competition/) registers its problems F1-F12 and P1-P12 and
# its algorithms when it is importable, i.e. when running from the root of the EDOLAB repository.
try:
    import CEC2027_Competition.registration  # noqa: F401
except ModuleNotFoundError as _error:
    if _error.name != "CEC2027_Competition":
        raise
