#!/usr/bin/env python3
"""One-command launcher for the local EDOLAB Relay control plane.

The launcher locates the conda ``EDOLAB`` interpreter without using
``conda run``, refreshes the editable installation, verifies the registry, and
then opens the production UI on localhost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Sequence
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parent
ENVIRONMENT_NAME = os.environ.get("EDOLAB_CONDA_ENV", "EDOLAB")
CONTAINER_MODE = os.environ.get("EDOLAB_RELAY_CONTAINER") == "1"
LEASE_STALE_SECONDS = 120.0
LEASE_HEARTBEAT_SECONDS = 5.0
SUPERVISED_CHILD_ENV = "EDOLAB_RELAY_SUPERVISED_CHILD"


class _ControllerLease:
    """Cross-host/container single-controller lease for one Relay data root.

    Docker Desktop bind mounts do not reliably share advisory ``flock`` state
    with macOS.  Directory creation and rename are filesystem operations, so a
    small heartbeat lease remains atomic across the native and Docker launchers.
    """

    def __init__(
        self,
        data_root: Path,
        *,
        stale_after: float = LEASE_STALE_SECONDS,
        heartbeat_interval: float = LEASE_HEARTBEAT_SECONDS,
        on_lost: Callable[[], None] | None = None,
    ) -> None:
        if stale_after <= heartbeat_interval * 2:
            raise ValueError("controller lease timeout must exceed two heartbeats")
        self.data_root = data_root
        self.path = data_root / "controller.lease"
        self.token = uuid.uuid4().hex
        self.stale_after = stale_after
        self.heartbeat_interval = heartbeat_interval
        self.on_lost = on_lost or self._terminate_owner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._acquired = False

    def acquire(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.data_root.is_symlink() or not self.data_root.is_dir():
            raise RuntimeError(f"unsafe Relay data root: {self.data_root}")
        self.data_root.chmod(0o700)
        for _attempt in range(4):
            try:
                self.path.mkdir(mode=0o700)
            except FileExistsError:
                if self.path.is_symlink() or not self.path.is_dir():
                    raise RuntimeError(
                        f"unsafe Relay controller lease path: {self.path}"
                    )
                age = self._age_seconds(self.path)
                if age <= self.stale_after:
                    owner = self._owner_summary(self.path)
                    raise RuntimeError(
                        "Another EDOLAB Relay controller is active for "
                        f"{self.data_root}{owner}."
                    )
                stale = self.data_root / f"controller.lease.stale.{uuid.uuid4().hex}"
                try:
                    self.path.rename(stale)
                except FileNotFoundError:
                    continue
                self._remove_known_lease(stale)
                continue
            self._initialize_owned_lease()
            return
        raise RuntimeError(f"could not acquire Relay controller lease: {self.path}")

    def release(self) -> None:
        if not self._acquired:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.heartbeat_interval + 1.0)
        try:
            if self._owner_token(self.path) == self.token:
                self._remove_known_lease(self.path)
        finally:
            self._acquired = False
            self._thread = None

    def _initialize_owned_lease(self) -> None:
        try:
            metadata = {
                "token": self.token,
                "pid": os.getpid(),
                "mode": "docker" if CONTAINER_MODE else "native",
                "started_at": time.time(),
            }
            owner = self.path / "owner.json"
            with owner.open("x", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True)
                handle.write("\n")
            heartbeat = self.path / "heartbeat"
            heartbeat.touch(mode=0o600, exist_ok=False)
            self._acquired = True
            self._thread = threading.Thread(
                target=self._heartbeat,
                name="relay-controller-lease",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self._remove_known_lease(self.path)
            raise

    def _heartbeat(self) -> None:
        heartbeat = self.path / "heartbeat"
        while not self._stop.wait(self.heartbeat_interval):
            try:
                if self._owner_token(self.path) != self.token:
                    self.on_lost()
                    return
                os.utime(heartbeat, follow_symlinks=False)
            except (FileNotFoundError, OSError):
                self.on_lost()
                return

    @staticmethod
    def _terminate_owner() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    @staticmethod
    def _owner_token(path: Path) -> str | None:
        try:
            payload = json.loads((path / "owner.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return None
        token = payload.get("token") if isinstance(payload, dict) else None
        return token if isinstance(token, str) else None

    @staticmethod
    def _owner_summary(path: Path) -> str:
        try:
            payload = json.loads((path / "owner.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        return f" (mode={payload.get('mode', 'unknown')}, pid={payload.get('pid', '?')})"

    @staticmethod
    def _age_seconds(path: Path) -> float:
        heartbeat = path / "heartbeat"
        try:
            timestamp = heartbeat.stat(follow_symlinks=False).st_mtime
        except FileNotFoundError:
            timestamp = path.stat(follow_symlinks=False).st_mtime
        return max(0.0, time.time() - timestamp)

    @staticmethod
    def _remove_known_lease(path: Path) -> None:
        try:
            entries = tuple(path.iterdir())
        except FileNotFoundError:
            return
        allowed = {"owner.json", "heartbeat"}
        if any(entry.name not in allowed or entry.is_symlink() for entry in entries):
            raise RuntimeError(f"unsafe files in Relay controller lease: {path}")
        for entry in entries:
            if not entry.is_file():
                raise RuntimeError(f"unsafe entry in Relay controller lease: {entry}")
            entry.unlink()
        path.rmdir()


def _looks_like_research_checkout(path: Path) -> bool:
    """Return whether *path* can safely be used as the snapshot source."""

    return (
        (path / "pyproject.toml").is_file()
        and (path / "core").is_dir()
        and (path / "instances").is_dir()
        and (path / "tasks").is_dir()
    )


def _configure_research_repository() -> Path:
    """Select the checkout whose current files Relay will snapshot.

    Editable/source launches use the checkout containing this launcher.  A
    wheel-installed ``edolab-relay`` instead uses the current working
    directory, so installing the controller can never silently turn the
    Python ``site-packages`` directory into the research source tree.
    """

    configured = os.environ.get("EDOLAB_RELAY_REPOSITORY_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    if not configured:
        candidates.extend((REPOSITORY_ROOT, Path.cwd()))
    for candidate in candidates:
        resolved = candidate.resolve()
        if _looks_like_research_checkout(resolved):
            os.environ["EDOLAB_RELAY_REPOSITORY_ROOT"] = str(resolved)
            return resolved
    if configured:
        detail = f"configured path is not an EDOLAB checkout: {configured}"
    else:
        detail = "run the command from an EDOLAB checkout"
    raise RuntimeError(
        f"Could not locate the research repository ({detail}); set "
        "EDOLAB_RELAY_REPOSITORY_ROOT=/absolute/path/to/EDOLAB-Python-OO."
    )


def _is_target_python(path: Path) -> bool:
    current = path.resolve()
    override = os.environ.get("EDOLAB_PYTHON")
    if override and current == Path(override).expanduser().resolve(strict=False):
        return True
    normalized = str(current).replace("\\", "/")
    return f"/envs/{ENVIRONMENT_NAME}/bin/python" in normalized


def _conda_environment_python() -> Path:
    override = os.environ.get("EDOLAB_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        raise RuntimeError(f"EDOLAB_PYTHON is not executable: {candidate}")

    candidates: list[Path] = []
    conda = shutil.which("conda")
    if conda:
        completed = subprocess.run(
            [conda, "info", "--base"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            shell=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            candidates.append(
                Path(completed.stdout.strip())
                / "envs"
                / ENVIRONMENT_NAME
                / "bin"
                / "python"
            )
    candidates.extend(
        [
            Path(f"/opt/anaconda3/envs/{ENVIRONMENT_NAME}/bin/python"),
            Path.home() / "anaconda3" / "envs" / ENVIRONMENT_NAME / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / ENVIRONMENT_NAME / "bin" / "python",
            Path(f"/opt/conda/envs/{ENVIRONMENT_NAME}/bin/python"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError(
        f"Could not find conda environment {ENVIRONMENT_NAME!r}. "
        "Create it with Python 3.11 or set EDOLAB_PYTHON=/absolute/path/to/python."
    )


def _reexec_in_edolab(argv: Sequence[str]) -> None:
    if CONTAINER_MODE or _is_target_python(Path(sys.executable)):
        return
    python = _conda_environment_python()
    environment = os.environ.copy()
    prefix = python.parent.parent
    environment["CONDA_DEFAULT_ENV"] = ENVIRONMENT_NAME
    environment["CONDA_PREFIX"] = str(prefix)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    os.execve(
        str(python),
        [str(python), str(Path(__file__).resolve()), *argv],
        environment,
    )


def _run_preflight() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"EDOLAB Relay requires Python 3.11; current interpreter is {sys.version.split()[0]}."
        )
    source_launch = (REPOSITORY_ROOT / "pyproject.toml").is_file()
    if (
        os.environ.get("EDOLAB_RELAY_SKIP_EDITABLE_INSTALL") != "1"
        and source_launch
    ):
        print("EDOLAB Relay: refreshing the editable installation…", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(REPOSITORY_ROOT)],
            cwd=REPOSITORY_ROOT,
            check=False,
            shell=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"editable installation failed with exit code {completed.returncode}"
            )
    print("EDOLAB Relay: checking algorithm and problem registry…", flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "cli", "list"],
        cwd=REPOSITORY_ROOT,
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(f"registry check failed with exit code {completed.returncode}")
    frontend = Path(
        os.environ.get("EDOLAB_RELAY_FRONTEND_DIST")
        or REPOSITORY_ROOT / "apps" / "remote" / "frontend" / "dist"
    ).expanduser()
    index = frontend / "index.html"
    if not index.is_file():
        raise RuntimeError(
            "production frontend is missing; run `npm ci && npm run build` "
            "in apps/remote/frontend"
        )


def _relay_arguments(argv: Sequence[str]) -> list[str]:
    arguments = list(argv)
    no_browser = "--no-open-browser" in arguments
    arguments = [item for item in arguments if item != "--no-open-browser"]
    if not no_browser and "--open-browser" not in arguments:
        arguments.append("--open-browser")
    return ["start", *arguments]


def _requested_data_root(argv: Sequence[str]) -> str | None:
    arguments = list(argv)
    for index, item in enumerate(arguments):
        if item.startswith("--data-root="):
            return item.split("=", 1)[1]
        if item == "--data-root" and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _run_supervised(arguments: Sequence[str], lease: _ControllerLease) -> int:
    environment = os.environ.copy()
    environment[SUPERVISED_CHILD_ENV] = f"{os.getpid()}:{lease.token}"
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        cwd=Path.cwd(),
        env=environment,
    )
    handled = (signal.SIGINT, signal.SIGTERM)
    previous = {number: signal.getsignal(number) for number in handled}

    def forward(number: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(number)

    try:
        for number in handled:
            signal.signal(number, forward)
        return_code = child.wait()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if child.poll() is None:
            child.terminate()
            child.wait()
    return 128 - return_code if return_code < 0 else return_code


def _validate_supervised_child(arguments: Sequence[str], marker: str) -> None:
    try:
        parent_text, token = marker.split(":", 1)
        parent_pid = int(parent_text)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("invalid Relay supervisor marker") from exc
    if parent_pid != os.getppid() or not token:
        raise RuntimeError("Relay controller child is not owned by its supervisor")
    from apps.remote.config import RelayPaths

    paths = RelayPaths.resolve(data_root=_requested_data_root(arguments))
    if _ControllerLease._owner_token(paths.data_root / "controller.lease") != token:
        raise RuntimeError("Relay controller lease changed before child startup")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    child_marker = os.environ.pop(SUPERVISED_CHILD_ENV, None)
    lease: _ControllerLease | None = None
    try:
        _reexec_in_edolab(arguments)
        wants_help = any(item in {"-h", "--help"} for item in arguments)
        if not wants_help:
            _configure_research_repository()
            if child_marker is not None:
                _validate_supervised_child(arguments, child_marker)
            else:
                _run_preflight()
                from apps.remote.config import RelayPaths

                paths = RelayPaths.resolve(data_root=_requested_data_root(arguments))
                paths.ensure()
                lease = _ControllerLease(paths.data_root)
                lease.acquire()
                return _run_supervised(arguments, lease)
        from apps.remote.__main__ import main as relay_main

        return relay_main(_relay_arguments(arguments))
    except RuntimeError as exc:
        print(f"EDOLAB Relay: {exc}", file=sys.stderr)
        return 2
    finally:
        if lease is not None:
            lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
