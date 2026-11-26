"""Competition algorithms.

Every folder here that contains an ``algorithm.py`` is imported, so the algorithm in it registers itself under the
name given to ``@algorithm_registry.register``. Folders whose name starts with an underscore (such as ``_template``)
are skipped. The platform imports this package together with its own algorithms, so a competition algorithm can be
used everywhere an algorithm name is accepted: the competition runner, ``python -m cli run``, batch task files and the
Relay console.
"""

from importlib import import_module
from pathlib import Path

for _folder in sorted(Path(__file__).parent.iterdir()):
    if _folder.is_dir() and not _folder.name.startswith(("_", ".")) and (_folder / "algorithm.py").is_file():
        import_module(f"{__name__}.{_folder.name}.algorithm")
