"""Check that the protected files of the competition are unmodified.

Protected are the platform core, the two benchmark generators (GMPB and DynPORTAL) of the Python version of EDOLAB,
and the instance definitions and scripts of this folder. Their SHA-256 digests are listed in protected.sha256 (platform
files as ``PythonVersion/...``, files of this folder as ``CEC2027_Competition/...``). Line endings are normalized first,
so a checkout with Windows line endings verifies too.

    python -m CEC2027_Competition.verify          (from the root of the EDOLAB repository)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import core  # the platform this folder runs on (see __init__.py)

COMPETITION_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = Path(core.__file__).resolve().parents[1]
MANIFEST = COMPETITION_ROOT / "protected.sha256"
PLATFORM_PROTECTED = ("core", "instances/problems/gmpb", "instances/problems/dportal")
COMPETITION_PROTECTED = ("instances.py", "problems.py", "registration.py", "run.py", "verify.py", "__init__.py")


def _python_files(root: Path, entries: tuple[str, ...]) -> list[str]:
    files: set[str] = set()
    for entry in entries:
        path = root / entry
        if path.is_dir():
            files.update(item.relative_to(root).as_posix() for item in path.rglob("*.py")
                         if "__pycache__" not in item.parts)
        elif path.is_file():
            files.add(entry)
    return sorted(files)


def protected_files(platform_root: Path = PLATFORM_ROOT, competition_root: Path = COMPETITION_ROOT) -> dict[str, Path]:
    """Manifest name -> file path of every protected file."""
    files = {f"PythonVersion/{name}": platform_root / name for name in _python_files(platform_root, PLATFORM_PROTECTED)}
    files.update({f"CEC2027_Competition/{name}": competition_root / name
                  for name in _python_files(competition_root, COMPETITION_PROTECTED)})
    return dict(sorted(files.items()))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def compute(platform_root: Path = PLATFORM_ROOT, competition_root: Path = COMPETITION_ROOT) -> dict[str, str]:
    return {name: digest(path) for name, path in protected_files(platform_root, competition_root).items()}


def write_manifest(platform_root: Path = PLATFORM_ROOT, competition_root: Path = COMPETITION_ROOT) -> Path:
    """Write protected.sha256 (used when the release is built)."""
    manifest = competition_root / MANIFEST.name
    manifest.write_text("".join(f"{value}  {name}\n" for name, value in compute(platform_root, competition_root).items()),
                        encoding="utf-8")
    return manifest


def read_manifest(manifest: Path = MANIFEST) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value, name = line.split(maxsplit=1)
            expected[name.strip()] = value
    return expected


def compare(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    """One message per modified, missing or added protected file."""
    problems = [f"modified: {name}" for name in sorted(expected) if name in actual and actual[name] != expected[name]]
    problems += [f"missing: {name}" for name in sorted(set(expected) - set(actual))]
    problems += [f"added: {name}" for name in sorted(set(actual) - set(expected))]
    return problems


def check() -> list[str]:
    """Problems of the platform and folder in use; empty when every protected file is intact."""
    return compare(read_manifest(), compute())


def manifest_digest(manifest: Path = MANIFEST) -> str:
    """Digest of protected.sha256 itself, recorded with every competition result."""
    return digest(manifest)


def main() -> int:
    problems = check()
    if problems:
        print("The protected competition files differ from the release:")
        for problem in problems:
            print(f"  {problem}")
        print("Restore them (for example with `git checkout -- <file>`) before running the competition.")
        return 1
    print(f"All {len(read_manifest())} protected files are unmodified (manifest {manifest_digest()[:16]}).")
    print(f"Platform: {PLATFORM_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
