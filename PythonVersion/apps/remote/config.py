"""Local paths and single-controller locking for EDOLAB Relay."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile
from typing import IO


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT_MARKER = ".edolab-relay-root"
_DATA_ROOT_ENTRIES = frozenset(
    {
        _DATA_ROOT_MARKER,
        ".DS_Store",
        "control.lock",
        "control.sqlite",
        "control.sqlite-shm",
        "control.sqlite-wal",
        "known_hosts",
        "local_execution",
        "analyses",
        "runs",
        "snapshots",
        "ssh_mux",
        "trash",
        "working",
    }
)


@dataclass(frozen=True, slots=True)
class RelayPaths:
    repository_root: Path
    data_root: Path
    database: Path
    known_hosts: Path
    ssh_mux: Path
    snapshots: Path
    working: Path
    runs: Path
    analyses: Path
    trash: Path
    frontend_dist: Path

    @property
    def local_execution(self) -> Path:
        """Private workspace used by the explicit local-process target."""

        return self.data_root / "local_execution"

    @property
    def imported_results(self) -> Path:
        """Independent, user-selected result library used by preview/plot/SA."""

        return self.analyses / "results"

    @classmethod
    def resolve(
        cls,
        *,
        repository_root: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> "RelayPaths":
        configured_repository = (
            repository_root
            or os.environ.get("EDOLAB_RELAY_REPOSITORY_ROOT")
            or REPOSITORY_ROOT
        )
        repository = Path(configured_repository).expanduser().resolve()
        configured_root = data_root or os.environ.get("EDOLAB_RELAY_DATA_ROOT")
        local_data = Path(
            configured_root or repository / "results" / "remote_control"
        ).expanduser().resolve()
        frontend = Path(
            os.environ.get("EDOLAB_RELAY_FRONTEND_DIST")
            or repository / "apps" / "remote" / "frontend" / "dist"
        ).expanduser().resolve()
        return cls(
            repository_root=repository,
            data_root=local_data,
            database=local_data / "control.sqlite",
            known_hosts=local_data / "known_hosts",
            ssh_mux=local_data / "ssh_mux",
            snapshots=local_data / "snapshots",
            working=local_data / "working",
            runs=local_data / "runs",
            analyses=local_data / "analyses",
            trash=local_data / "trash",
            frontend_dist=frontend,
        )

    def ensure(self) -> None:
        self._ensure_dedicated_data_root()
        for directory in (
            self.ssh_mux,
            self.snapshots,
            self.working,
            self.runs,
            self.analyses,
            self.imported_results,
            self.trash,
            self.local_execution,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if directory.is_symlink() or not directory.is_dir():
                raise RuntimeError(f"Relay state path is not a safe directory: {directory}")
            directory.chmod(0o700)
        if self.known_hosts.is_symlink():
            raise RuntimeError(
                f"Relay known_hosts path must not be a symlink: {self.known_hosts}"
            )
        self.known_hosts.touch(mode=0o600, exist_ok=True)
        if not self.known_hosts.is_file():
            raise RuntimeError(
                f"Relay known_hosts path is not a regular file: {self.known_hosts}"
            )
        self.known_hosts.chmod(0o600)

    def _ensure_dedicated_data_root(self) -> None:
        """Adopt only a dedicated state directory, never a broad user path."""

        root = self.data_root
        repository = self.repository_root.resolve()
        home = Path.home().resolve()
        temporary_root = Path(tempfile.gettempdir()).resolve()
        filesystem_root = Path(root.anchor).resolve()
        if root in {filesystem_root, home, repository, temporary_root} or _contains(
            root, repository
        ):
            raise RuntimeError(
                "Relay data_root must be a dedicated subdirectory, not a filesystem, "
                "home, temporary, repository, or repository-ancestor directory: "
                f"{root}"
            )

        existed = root.exists() or root.is_symlink()
        if existed:
            if root.is_symlink() or not root.is_dir():
                raise RuntimeError(f"Relay data root is not a safe directory: {root}")
            unexpected = sorted(
                entry.name
                for entry in root.iterdir()
                if entry.name not in _DATA_ROOT_ENTRIES
                and not entry.name.startswith("controller.lease")
            )
            if unexpected:
                preview = ", ".join(unexpected[:5])
                raise RuntimeError(
                    "Relay data_root is not a dedicated Relay directory; unexpected "
                    f"entries were found: {preview}"
                )
        else:
            root.mkdir(parents=True, mode=0o700)

        root_info = root.stat(follow_symlinks=False)
        if hasattr(os, "getuid") and root_info.st_uid != os.getuid():
            raise RuntimeError(
                f"Relay data_root must be owned by the current user: {root}"
            )
        root.chmod(0o700)

        marker = root / _DATA_ROOT_MARKER
        if marker.is_symlink():
            raise RuntimeError(f"Relay data root marker is unsafe: {marker}")
        flags = os.O_WRONLY | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(marker, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"Relay data root marker is unsafe: {marker}")
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)


def _contains(directory: Path, path: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


class ControllerAlreadyRunning(RuntimeError):
    """Raised when another Relay process already owns the data directory."""


class ControllerLock:
    """Advisory process lock shared by the Python and Docker launchers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle: IO[str] | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ControllerAlreadyRunning(
                f"Unsafe Relay controller lock path: {self.path}"
            )
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
            handle = os.fdopen(descriptor, "a+", encoding="utf-8")
            self.path.chmod(0o600)
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError, OSError) as exc:
            if "handle" in locals():
                handle.close()
            raise ControllerAlreadyRunning(
                f"Another EDOLAB Relay controller is using {self.path.parent}."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "ControllerLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
