"""Registers the competition problems (F1-F12, P1-P12) and the competition algorithms with the platform.

The platform imports this module whenever CEC2027_Competition is importable (from the root of the EDOLAB repository),
so the 24 problems and the algorithms in algorithms/ are available wherever a problem or algorithm name is accepted.
"""

import CEC2027_Competition.algorithms  # noqa: F401
import CEC2027_Competition.problems  # noqa: F401
