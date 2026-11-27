"""CEC 2027 Competition on Dynamic Optimization Problems.

PROTECTED FILE. Do not modify it.

This folder sits next to PythonVersion/ in the EDOLAB repository and runs on that Python version: importing it puts
the sibling PythonVersion/ first on the module search path, so the competition always uses the platform of the same
checkout. ``instances.py`` holds the 24 problem instances, ``run.py`` runs an algorithm on them and writes the files of a
submission, ``verify.py`` checks that the protected files are unmodified, and ``algorithms/`` holds the participants'
algorithms with two examples and a template. See README.md.
"""

import sys
from pathlib import Path

_PLATFORM = Path(__file__).resolve().parents[1] / "PythonVersion"
if (_PLATFORM / "core").is_dir() and str(_PLATFORM) not in sys.path:
    sys.path.insert(0, str(_PLATFORM))
