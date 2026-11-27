"""Hardened OpenSSH transport used by the remote execution service.

Only the system ``ssh`` client is used.  Local commands are always launched as
an argv vector with ``shell=False``.  OpenSSH necessarily invokes the account's
login shell for a remote command; :meth:`SSHClient.run_script` keeps that safe by
building the command from individually shell-quoted tokens and sends Python
source over stdin instead of interpolating it into the command line.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import threading
import time
from typing import Any, Sequence

from apps.remote.durability import fsync_directory, fsync_file
from apps.remote.environment import runtime_dependency_probe_specs


_SAFE_USER = re.compile(r"^[A-Za-z0-9_.-]+$")
_CONTROL_HASH_BYTES = 40
_CONTROL_PATH_MAX_BYTES = 80
_CONTROL_TEMP_SUFFIX_RESERVE = 20
# host_key_fingerprint answers keyed by the known-hosts file identity; see SSHClient.host_key_fingerprint.
_FINGERPRINT_CACHE: dict[tuple, str | None] = {}
_FINGERPRINT_LOCK = threading.Lock()

# The SSH probe runs under a non-interactive login environment.  Conda's shell
# function and its base ``bin`` directory are therefore commonly absent from
# PATH even though an interactive session can use Conda.  Keep discovery in
# the probe only; the selected absolute executable is later frozen with the run
# and passed to the remote worker launcher.
REMOTE_CONDA_EXECUTABLE_CANDIDATES = (
    "~/miniconda3/bin/conda",
    "~/miniforge3/bin/conda",
    "~/mambaforge/bin/conda",
    "~/anaconda3/bin/conda",
    "/opt/conda/bin/conda",
    "/opt/miniconda3/bin/conda",
    "/opt/miniforge3/bin/conda",
    "/opt/mambaforge/bin/conda",
    "/opt/anaconda3/bin/conda",
    "/usr/local/conda/bin/conda",
    "/usr/local/miniconda3/bin/conda",
    "/usr/local/miniforge3/bin/conda",
    "/usr/local/mambaforge/bin/conda",
    "/usr/local/anaconda3/bin/conda",
)


class SSHError(RuntimeError):
    """Base error for local validation and SSH transport failures."""


class SSHCommandError(SSHError):
    """A remote command exited unsuccessfully."""

    def __init__(self, message: str, result: "SSHResult") -> None:
        super().__init__(message)
        self.result = result
        self.returncode = result.returncode


@dataclass(frozen=True, slots=True, kw_only=True)
class SSHConnection:
    """Connection settings with no persisted key material."""

    host: str
    username: str
    key_path: str | Path
    known_hosts_path: str | Path
    control_dir: str | Path
    port: int = 22
    connect_timeout_seconds: int = 10
    server_alive_interval_seconds: int = 15
    server_alive_count_max: int = 3
    control_persist_seconds: int = 300
    ssh_binary: str = "ssh"
    remote_python: str = "python3"

    def __post_init__(self) -> None:
        host = str(self.host).strip()
        username = str(self.username).strip()
        if not host or host.startswith("-") or any(char.isspace() for char in host):
            raise ValueError("host must be a non-empty hostname or address")
        if any(char in host for char in ("\x00", "\r", "\n")):
            raise ValueError("host contains an invalid control character")
        if not _SAFE_USER.fullmatch(username):
            raise ValueError("username contains unsupported characters")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        for name in (
            "connect_timeout_seconds",
            "server_alive_interval_seconds",
            "server_alive_count_max",
            "control_persist_seconds",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not str(self.remote_python).strip():
            raise ValueError("remote_python must not be empty")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "username", username)
        object.__setattr__(self, "key_path", Path(self.key_path).expanduser())
        object.__setattr__(
            self, "known_hosts_path", Path(self.known_hosts_path).expanduser()
        )
        object.__setattr__(self, "control_dir", Path(self.control_dir).expanduser())

    @classmethod
    def for_server(cls, server: Any, state_dir: str | Path, **overrides: Any) -> "SSHConnection":
        """Create settings from an ``apps.remote.domain.Server``-like object."""

        state = Path(state_dir).expanduser()
        server_id = str(getattr(server, "id", "server"))
        values: dict[str, Any] = {
            "host": server.host,
            "username": server.username,
            "key_path": server.ssh_key_path,
            "port": server.port,
            # RelayPaths owns one private OpenSSH known_hosts file and a
            # per-server mux subdirectory.  Keeping this convention here
            # avoids treating the known_hosts file as a directory.
            "known_hosts_path": state / "known_hosts",
            "control_dir": state / "ssh_mux" / server_id,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SSHResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")

    def check_returncode(self) -> "SSHResult":
        if self.returncode:
            detail = self.stderr_text.strip() or self.stdout_text.strip()
            message = f"SSH command exited with code {self.returncode}"
            if detail:
                message = f"{message}: {detail}"
            raise SSHCommandError(message, self)
        return self


@dataclass(frozen=True, slots=True)
class TransferResult:
    path: Path | str
    size_bytes: int
    sha256: str
    duration_seconds: float


def validate_private_key(path: str | Path) -> Path:
    """Resolve and validate an OpenSSH private key without changing it."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        info = resolved.stat()
    except FileNotFoundError as exc:
        raise SSHError(f"SSH private key does not exist: {candidate}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise SSHError(f"SSH private key is not a regular file: {candidate}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise SSHError(f"SSH private key is not owned by the current user: {candidate}")
    mode = stat.S_IMODE(info.st_mode)
    if not mode & stat.S_IRUSR:
        raise SSHError(f"SSH private key is not readable by its owner: {candidate}")
    if mode & 0o077:
        raise SSHError(
            f"SSH private key permissions are too open ({mode:#o}); use chmod 600 or 400"
        )
    return resolved


class SSHClient:
    """Small synchronous transport around a reusable OpenSSH control socket."""

    def __init__(self, connection: SSHConnection) -> None:
        self.connection = connection
        self.key_path = validate_private_key(connection.key_path)
        self.known_hosts_path = _prepare_private_file(connection.known_hosts_path)
        self.control_dir = _prepare_private_directory(connection.control_dir)
        self.control_path = _safe_control_path(self.control_dir)
        binary = shutil.which(connection.ssh_binary)
        if binary is None:
            raise SSHError(f"OpenSSH client was not found: {connection.ssh_binary}")
        self.ssh_binary = binary

    @property
    def destination(self) -> str:
        return f"{self.connection.username}@{self.connection.host}"

    def base_argv(self) -> list[str]:
        """Return the complete hardened argv prefix, ending in the destination."""

        connection = self.connection
        return [
            self.ssh_binary,
            "-F",
            "/dev/null",
            "-T",
            "-p",
            str(connection.port),
            "-i",
            str(self.key_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            f"ConnectTimeout={connection.connect_timeout_seconds}",
            "-o",
            f"ServerAliveInterval={connection.server_alive_interval_seconds}",
            "-o",
            f"ServerAliveCountMax={connection.server_alive_count_max}",
            "-o",
            f"UserKnownHostsFile={self.known_hosts_path}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            # accept-new accepts only an unseen key; OpenSSH still rejects any
            # changed key, so the first connection is convenient and every
            # subsequent connection remains pinned to this private file.
            "StrictHostKeyChecking=accept-new",
            "-o",
            "HashKnownHosts=yes",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPersist={connection.control_persist_seconds}",
            "-o",
            f"ControlPath={self.control_path}",
            "--",
            self.destination,
        ]

    def command_argv(self, remote_argv: Sequence[str]) -> list[str]:
        tokens = _validate_remote_argv(remote_argv)
        return [*self.base_argv(), shlex.join(tokens)]

    def run_command(
        self,
        remote_argv: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> SSHResult:
        argv = self.command_argv(remote_argv)
        started = time.monotonic()
        completed = subprocess.run(
            argv,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False,
        )
        result = SSHResult(
            argv=tuple(argv),
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=time.monotonic() - started,
        )
        return result.check_returncode() if check else result

    def run_script(
        self,
        script: str | bytes,
        args: Sequence[str] = (),
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> SSHResult:
        """Run Python source from stdin with safely quoted positional arguments."""

        source = script.encode("utf-8") if isinstance(script, str) else bytes(script)
        return self.run_command(
            [self.connection.remote_python, "-", *[str(value) for value in args]],
            input_bytes=source,
            timeout=timeout,
            check=check,
        )

    def probe(
        self,
        remote_path: str,
        conda_env: str = "EDOLAB",
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        result = self.run_script(
            REMOTE_PROBE_SCRIPT,
            [remote_path, conda_env],
            timeout=timeout,
        )
        lines = [line for line in result.stdout_text.splitlines() if line.strip()]
        if not lines:
            raise SSHError("remote probe returned no JSON")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise SSHError("remote probe returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise SSHError("remote probe returned an unsupported payload")
        payload["fingerprint"] = self.host_key_fingerprint()
        return payload

    def sample_resources(
        self,
        *,
        timeout: float = 10.0,
        expected_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Return a lightweight, read-only CPU/memory/GPU sample.

        Resource polling deliberately does not repeat the environment, Conda,
        filesystem, Docker, or tool discovery performed by :meth:`probe`.
        When the service has already pinned a host fingerprint, require the
        local known-hosts entry to match before opening another SSH command.
        This prevents a deleted known-hosts file from silently turning a live
        monitor refresh into a new TOFU decision.
        """

        if expected_fingerprint:
            pinned = self.host_key_fingerprint()
            if pinned != expected_fingerprint:
                detail = pinned or "unavailable"
                raise SSHError(
                    "pinned SSH host-key fingerprint is unavailable or changed "
                    f"(expected {expected_fingerprint}, found {detail})"
                )
        result = self.run_script(REMOTE_RESOURCE_SCRIPT, timeout=timeout)
        lines = [line for line in result.stdout_text.splitlines() if line.strip()]
        if not lines:
            raise SSHError("remote resource sample returned no JSON")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise SSHError("remote resource sample returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise SSHError("remote resource sample returned an unsupported payload")
        return payload

    def host_key_fingerprint(self) -> str | None:
        """Return the pinned host key's SHA-256 fingerprint, if available.

        The resource monitor asks before every sample of every target. The answer only changes when the
        known-hosts file does, so it is cached on the file's identity (inode, size, mtime): two ssh-keygen
        processes per sample were most of the Relay's fork load.
        """

        ssh_keygen = shutil.which("ssh-keygen")
        if ssh_keygen is None:
            return None
        lookup = (
            self.connection.host
            if self.connection.port == 22
            else f"[{self.connection.host}]:{self.connection.port}"
        )
        try:
            info = Path(self.known_hosts_path).stat()
        except OSError:
            return None
        key = (ssh_keygen, str(self.known_hosts_path), lookup, info.st_ino, info.st_size, info.st_mtime_ns)
        with _FINGERPRINT_LOCK:
            if key in _FINGERPRINT_CACHE:
                return _FINGERPRINT_CACHE[key]
        value = self._host_key_fingerprint_uncached(ssh_keygen, lookup)
        with _FINGERPRINT_LOCK:
            if len(_FINGERPRINT_CACHE) >= 256:
                _FINGERPRINT_CACHE.clear()
            _FINGERPRINT_CACHE[key] = value
        return value

    def _host_key_fingerprint_uncached(self, ssh_keygen: str, lookup: str) -> str | None:
        found = subprocess.run(
            [ssh_keygen, "-F", lookup, "-f", str(self.known_hosts_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
        if found.returncode:
            return None
        key_lines = [
            line
            for line in found.stdout.splitlines()
            if line.strip() and not line.lstrip().startswith(b"#")
        ]
        for key_line in key_lines:
            fingerprint = subprocess.run(
                [ssh_keygen, "-lf", "-", "-E", "sha256"],
                input=key_line + b"\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
            )
            if fingerprint.returncode:
                continue
            for token in fingerprint.stdout.decode("utf-8", errors="replace").split():
                if token.startswith("SHA256:"):
                    return token
        return None

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        mode: int = 0o600,
        timeout: float | None = None,
    ) -> TransferResult:
        """Stream one file to an atomically replaced remote destination."""

        source = Path(local_path)
        if source.is_symlink() or not source.is_file():
            raise SSHError(f"upload source is not a regular file: {source}")
        info = source.stat()
        digest = _sha256_file(source)
        upload_program = (
            "import base64;exec(compile(base64.b64decode("
            + repr(base64.b64encode(_REMOTE_FILE_UPLOAD_SCRIPT.encode("utf-8")).decode("ascii"))
            + "),'<edolab-upload>','exec'))"
        )
        argv = self.command_argv(
            [
                self.connection.remote_python,
                "-c",
                upload_program,
                remote_path,
                str(info.st_size),
                digest,
                format(mode & 0o777, "o"),
            ]
        )
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        with source.open("rb") as handle:
            process = subprocess.Popen(
                argv,
                stdin=handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except BaseException:
                _kill_and_reap(process)
                raise
        result = SSHResult(
            argv=tuple(argv),
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
        ).check_returncode()
        try:
            remote = json.loads(result.stdout_text)
        except json.JSONDecodeError as exc:
            raise SSHError("upload completed without a valid verification response") from exc
        if remote.get("sha256") != digest or int(remote.get("size_bytes", -1)) != info.st_size:
            raise SSHError("remote upload verification did not match the local file")
        return TransferResult(remote_path, info.st_size, digest, result.duration_seconds)

    def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
        *,
        timeout: float | None = None,
    ) -> TransferResult:
        """Stream one remote file to a verified, atomically replaced local file."""

        metadata_result = self.run_script(
            _REMOTE_FILE_METADATA_SCRIPT,
            [remote_path],
            timeout=timeout,
        )
        try:
            metadata = json.loads(metadata_result.stdout_text)
            expected_size = int(metadata["size_bytes"])
            expected_sha256 = str(metadata["sha256"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SSHError("remote file metadata was invalid") from exc

        destination = Path(local_path)
        parent = destination.parent
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise SSHError(f"download destination directory is unsafe: {parent}")
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise SSHError(f"download destination directory is unsafe: {parent}")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise SSHError(f"download destination is unsafe: {destination}")
        partial = destination.with_name(f".{destination.name}.part-{os.getpid()}")
        argv = self.command_argv(
            [self.connection.remote_python, "-", remote_path]
        )
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        try:
            with partial.open("xb") as handle:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    shell=False,
                )
                try:
                    # Let ``communicate`` own stdin for its entire lifetime.
                    # Manually closing the pipe before calling it is unsafe on
                    # Python 3.11: the first call fails while flushing the
                    # closed pipe, marks communication as started, and a retry
                    # then accesses an uninitialised ``_fileobj2output`` map.
                    # stdout still streams directly to ``handle``; only this
                    # small, fixed helper script and stderr use pipe buffers.
                    _, stderr = process.communicate(
                        input=_REMOTE_FILE_DOWNLOAD_SCRIPT.encode("utf-8"),
                        timeout=timeout,
                    )
                except BaseException:
                    _kill_and_reap(process)
                    raise
            if process.returncode:
                result = SSHResult(
                    tuple(argv),
                    int(process.returncode),
                    b"",
                    stderr,
                    time.monotonic() - started,
                )
                result.check_returncode()
            actual_size = partial.stat().st_size
            actual_sha256 = _sha256_file(partial)
            if actual_size != expected_size or actual_sha256 != expected_sha256:
                raise SSHError("downloaded file failed size or SHA-256 verification")
            # The verified bytes must reach stable storage before the atomic
            # name is published.  The parent fsync persists that publication.
            fsync_file(partial)
            os.replace(partial, destination)
            fsync_directory(parent)
        except BaseException:
            if process is not None:
                _kill_and_reap(process)
            partial.unlink(missing_ok=True)
            raise
        return TransferResult(
            destination,
            expected_size,
            expected_sha256,
            time.monotonic() - started,
        )

    def close_master(self) -> SSHResult:
        """Ask OpenSSH to close this connection's reusable control master."""

        argv = [*self.base_argv()[:-2], "-O", "exit", "--", self.destination]
        started = time.monotonic()
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
        return SSHResult(
            tuple(argv),
            int(completed.returncode),
            completed.stdout,
            completed.stderr,
            time.monotonic() - started,
        )


def _validate_remote_argv(values: Sequence[str]) -> list[str]:
    if not values:
        raise ValueError("remote argv must not be empty")
    result = [str(value) for value in values]
    if any("\x00" in value or "\n" in value or "\r" in value for value in result):
        raise ValueError("remote argv contains a control character")
    if not result[0].strip():
        raise ValueError("remote executable must not be empty")
    return result


def _prepare_private_directory(path: str | Path) -> Path:
    target = Path(path).expanduser()
    # ``Path.exists()`` is false for a dangling symlink.  Check the link
    # itself first so ``mkdir(..., exist_ok=True)`` cannot fail with an opaque
    # FileExistsError or, if the target appears later, redirect SSH state.
    if target.is_symlink():
        raise SSHError(f"SSH state directory must not be a symlink: {target}")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not target.is_dir():
        raise SSHError(f"SSH state path is not a directory: {target}")
    info = target.stat()
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise SSHError(f"SSH state directory is not owned by the current user: {target}")
    target.chmod(0o700)
    return target.resolve()


def _safe_control_path(control_dir: Path) -> Path:
    """Return an OpenSSH mux path that fits conservative AF_UNIX limits.

    ``%C`` expands to a 40-character connection hash.  OpenSSH may first bind a
    temporary socket with an additional random suffix before atomically moving
    it into place, so the final ``%C`` path alone is not a sufficient length
    check.  Keep at least 20 bytes free below conservative macOS AF_UNIX limits.
    The scientific state remains under Relay's data root; only disposable SSH
    mux sockets use a private, content-addressed directory below ``/tmp``.
    """

    direct = control_dir / "%C"
    expanded_length = len(
        os.fsencode(str(control_dir / ("x" * _CONTROL_HASH_BYTES)))
    )
    if expanded_length <= _CONTROL_PATH_MAX_BYTES:
        return direct
    owner = str(os.getuid()) if hasattr(os, "getuid") else "user"
    digest = hashlib.sha256(os.fsencode(str(control_dir))).hexdigest()[:12]
    short = Path("/tmp") / f"er-{owner}-{digest}"
    prepared = _prepare_private_directory(short)
    candidate = prepared / "%C"
    candidate_length = len(
        os.fsencode(str(prepared / ("x" * _CONTROL_HASH_BYTES)))
    )
    if candidate_length > _CONTROL_PATH_MAX_BYTES:
        raise SSHError("SSH ControlPath remains too long for a Unix-domain socket")
    if candidate_length + _CONTROL_TEMP_SUFFIX_RESERVE > 103:
        raise SSHError("SSH ControlPath leaves no room for OpenSSH's temporary suffix")
    return candidate


def _prepare_private_file(path: str | Path) -> Path:
    target = Path(path).expanduser()
    _prepare_private_directory(target.parent)
    if target.is_symlink():
        raise SSHError(f"known_hosts must not be a symlink: {target}")
    flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise SSHError(f"known_hosts must be a regular file: {target}")
    finally:
        os.close(descriptor)
    target.chmod(0o600)
    return target.resolve()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kill_and_reap(process: subprocess.Popen[Any]) -> None:
    """Best-effort terminate a failed transfer and always reap its SSH child."""

    try:
        if process.poll() is None:
            process.kill()
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.stdin = None
    except Exception:
        pass
    try:
        process.communicate(timeout=5)
    except Exception:
        try:
            process.wait(timeout=5)
        except Exception:
            pass


# Kept as a separate constant so callers/tests can inspect the exact remote
# implementation.  It is a fixed, individually quoted ``python -c`` argument;
# stdin remains available for the file bytes themselves.
_REMOTE_FILE_UPLOAD_SCRIPT = r'''
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

destination = Path(sys.argv[1]).expanduser()
expected_size = int(sys.argv[2])
expected_sha256 = sys.argv[3]
mode = int(sys.argv[4], 8)
parent = destination.parent
if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
    raise RuntimeError("remote upload parent is unsafe")
parent.mkdir(parents=True, exist_ok=True)
if parent.is_symlink() or not parent.is_dir():
    raise RuntimeError("remote upload parent is unsafe")
if destination.is_symlink() or (destination.exists() and not destination.is_file()):
    raise RuntimeError("remote upload destination is unsafe")
if destination.exists():
    existing_digest = hashlib.sha256()
    existing_size = 0
    with destination.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            existing_digest.update(chunk)
            existing_size += len(chunk)
    if existing_size == expected_size and existing_digest.hexdigest() == expected_sha256:
        os.chmod(destination, mode)
        print(json.dumps({
            "reused": True,
            "size_bytes": existing_size,
            "sha256": existing_digest.hexdigest(),
        }, sort_keys=True))
        raise SystemExit(0)
partial = destination.with_name("." + destination.name + ".part-" + uuid.uuid4().hex)
digest = hashlib.sha256()
size = 0
try:
    with partial.open("xb") as handle:
        while True:
            chunk = sys.stdin.buffer.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise RuntimeError("uploaded content did not match declared metadata")
    os.chmod(partial, mode)
    os.replace(partial, destination)
finally:
    try:
        partial.unlink()
    except FileNotFoundError:
        pass
print(json.dumps({"reused": False, "size_bytes": size, "sha256": digest.hexdigest()}, sort_keys=True))
'''

_REMOTE_FILE_METADATA_SCRIPT = r'''
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
if path.is_symlink() or not path.is_file():
    raise RuntimeError("remote path is not a regular file")
digest = hashlib.sha256()
size = 0
with path.open("rb") as handle:
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
print(json.dumps({"size_bytes": size, "sha256": digest.hexdigest()}, sort_keys=True))
'''

_REMOTE_FILE_DOWNLOAD_SCRIPT = r'''
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1]).expanduser()
if path.is_symlink() or not path.is_file():
    raise RuntimeError("remote path is not a regular file")
with path.open("rb") as handle:
    shutil.copyfileobj(handle, sys.stdout.buffer, length=1024 * 1024)
'''


# This probe has no write operations.  In particular, writability is inferred
# from permissions on the nearest existing parent instead of creating a marker
# file in the requested remote workspace.
_REMOTE_PROBE_SCRIPT_TEMPLATE = r'''
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


def command_info(name, args=("--version",), timeout=4.0):
    executable = shutil.which(name)
    result = {"installed": executable is not None, "path": executable, "version": None}
    if executable is None:
        return result
    try:
        completed = subprocess.run(
            [executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout.strip().splitlines()
        result["version"] = output[0] if output else None
        result["returncode"] = completed.returncode
    except Exception as exc:
        result["error"] = str(exc)
    return result


def cpu_times():
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except Exception:
        return None


def cpu_percent():
    before = cpu_times()
    if before is None:
        if platform.system() == "Darwin":
            try:
                completed = subprocess.run(
                    ["top", "-l", "2", "-n", "0", "-stats", "cpu"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=4,
                    check=False,
                )
                for line in reversed(completed.stdout.splitlines()):
                    if "CPU usage:" not in line:
                        continue
                    match = re.search(r"([0-9.]+)%\s*idle", line)
                    if match:
                        return round(100.0 - float(match.group(1)), 2)
            except Exception:
                pass
        return None
    time.sleep(0.12)
    after = cpu_times()
    if after is None:
        return None
    total = after[0] - before[0]
    idle = after[1] - before[1]
    return round(100.0 * (total - idle) / total, 2) if total > 0 else None


def physical_cores():
    try:
        pairs = set()
        physical = None
        core = None
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines() + [""]:
            if not line.strip():
                if core is not None:
                    pairs.add((physical or "0", core))
                physical = core = None
            elif line.startswith("physical id"):
                physical = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
        if pairs:
            return len(pairs)
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
        except Exception:
            pass
    return None


def memory_info():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
    except Exception:
        pass
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    if total is None and platform.system() == "Darwin":
        try:
            total = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
            ).strip())
            vm_output = subprocess.check_output(
                ["vm_stat"], text=True, timeout=3
            )
            page_match = re.search(r"page size of (\d+) bytes", vm_output)
            page_size = int(page_match.group(1)) if page_match else 4096
            pages = {}
            for line in vm_output.splitlines()[1:]:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                match = re.search(r"\d+", raw.replace(".", ""))
                if match:
                    pages[key.strip()] = int(match.group(0))
            available = page_size * sum(
                pages.get(key, 0)
                for key in (
                    "Pages free",
                    "Pages inactive",
                    "Pages speculative",
                    "Pages purgeable",
                )
            )
        except Exception:
            pass
    return {"total_bytes": total, "available_bytes": available}


def nearest_existing(path):
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def disk_info(requested):
    path = Path(requested).expanduser()
    existing = nearest_existing(path)
    result = {
        "requested_path": str(path),
        "existing_path": str(existing),
        "path_exists": path.exists(),
        "writable": os.access(existing, os.W_OK | os.X_OK),
    }
    try:
        usage = shutil.disk_usage(existing)
        result.update(total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free)
        stats = os.statvfs(existing)
        read_only = bool(stats.f_flag & getattr(os, "ST_RDONLY", 1))
        result.update(
            inodes_total=int(stats.f_files),
            inodes_free=int(stats.f_favail),
            read_only=read_only,
        )
        result["writable"] = result["writable"] and not read_only
    except Exception as exc:
        result["error"] = str(exc)
    return result


def nvidia_info():
    executable = shutil.which("nvidia-smi")
    result = {"installed": executable is not None, "usable": False, "path": executable, "devices": []}
    if executable is None:
        return result
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu"
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=6,
            check=False,
        )
        result["returncode"] = completed.returncode
        if completed.returncode != 0:
            result["error"] = completed.stderr.strip() or completed.stdout.strip()
            return result
        devices = []
        names = fields.split(",")
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(",")]
            item = dict(zip(names, values))
            for key in ("index", "memory.total", "memory.used", "utilization.gpu", "temperature.gpu"):
                try:
                    item[key] = int(item[key])
                except (KeyError, ValueError):
                    pass
            devices.append(item)
        result["devices"] = devices
        result["usable"] = bool(devices)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def conda_info(environment_name):
    result = {
        "installed": False,
        "path": None,
        "version": None,
        "environment": environment_name,
        "environment_available": False,
        "environments": [],
        "environment_python": None,
    }
    candidates = []
    path_candidate = shutil.which("conda")
    if path_candidate:
        candidates.append((Path(path_candidate), "path"))
    for candidate_text in __EDOLAB_REMOTE_CONDA_CANDIDATES__:
        discovery = "home_fallback" if candidate_text.startswith("~/") else "system_fallback"
        candidates.append((Path(candidate_text).expanduser(), discovery))

    attempts = []
    seen = set()
    first_usable = None
    first_environment_failure = None
    dependency_specs = __EDOLAB_REMOTE_RUNTIME_DEPENDENCIES__
    probe_source = """
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import sys

dependency_specs = %r
dependencies = {}
for key, distribution, label, imports, required_attributes in dependency_specs:
    report = {
        "available": False,
        "distribution": distribution,
        "label": label,
        "version": None,
        "error": None,
    }
    imported = {}
    try:
        for module_name in imports:
            imported[module_name] = importlib.import_module(module_name)
        for module_name, attribute in required_attributes:
            module = imported.get(module_name) or importlib.import_module(module_name)
            getattr(module, attribute)
        try:
            report["version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            primary = imported.get(imports[0]) if imports else None
            report["version"] = getattr(primary, "__version__", None)
        report["available"] = True
    except Exception as exc:
        report["error"] = (type(exc).__name__ + ": " + str(exc))[:1000]
    dependencies[key] = report

print(json.dumps({
    "executable": sys.executable,
    "version": platform.python_version(),
    "pip_available": importlib.util.find_spec("pip") is not None,
    "runtime_dependencies": dependencies,
    "registry_runtime_ready": all(item["available"] for item in dependencies.values()),
}, sort_keys=True, separators=(",", ":")))
""" % (dependency_specs,)
    for candidate, discovery in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        path_text = str(resolved)
        if path_text in seen or not resolved.is_file() or not os.access(resolved, os.X_OK):
            continue
        seen.add(path_text)
        result["installed"] = True
        attempt = {"path": path_text, "discovery": discovery}
        try:
            version_probe = subprocess.run(
                [path_text, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=4,
                check=False,
            )
            version_lines = version_probe.stdout.strip().splitlines()
            version = version_lines[0] if version_lines else None
            attempt.update(version=version, returncode=version_probe.returncode)
            if version_probe.returncode != 0:
                attempt["error"] = version_probe.stdout.strip()
                attempts.append(attempt)
                continue

            environment_probe = subprocess.run(
                [path_text, "env", "list", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
                check=False,
            )
            if environment_probe.returncode != 0:
                attempt["env_error"] = environment_probe.stderr.strip() or environment_probe.stdout.strip()
                attempts.append(attempt)
                continue
            payload = json.loads(environment_probe.stdout)
            environments = payload.get("envs", [])
            if not isinstance(environments, list):
                raise ValueError("conda env list did not return an envs list")
            environments = [str(item) for item in environments]
            available = any(Path(item).name == environment_name for item in environments)
            attempt.update(environments=environments, environment_available=available)
            candidate_result = {
                "path": path_text,
                "version": version,
                "returncode": version_probe.returncode,
                "discovery": discovery,
                "environments": environments,
            }
            if first_usable is None:
                first_usable = candidate_result
            if not available:
                attempts.append(attempt)
                continue

            python_probe = subprocess.run(
                [path_text, "run", "--no-capture-output", "-n", environment_name, "python", "-c", probe_source],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=12,
                check=False,
            )
            candidate_result["environment_available"] = True
            if python_probe.returncode == 0:
                lines = [line for line in python_probe.stdout.splitlines() if line.strip()]
                environment_python = json.loads(lines[-1]) if lines else None
                if not isinstance(environment_python, dict):
                    raise ValueError("conda run did not return a Python report")
                result.update(candidate_result)
                result["environment_python"] = environment_python
                attempts.append(attempt)
                result["candidates"] = attempts
                return result
            candidate_result["environment_python_error"] = (
                python_probe.stderr.strip() or python_probe.stdout.strip()
            )
            attempt["environment_python_error"] = candidate_result["environment_python_error"]
            if first_environment_failure is None:
                first_environment_failure = candidate_result
            attempts.append(attempt)
        except Exception as exc:
            attempt["error"] = str(exc)
            attempts.append(attempt)

    if first_environment_failure is not None:
        result.update(first_environment_failure)
    elif first_usable is not None:
        result.update(first_usable)
    if attempts:
        result["candidates"] = attempts
    return result


def docker_info():
    result = command_info("docker", ("--version",))
    result["usable"] = False
    if not result["installed"]:
        return result
    try:
        completed = subprocess.run(
            [result["path"], "info", "--format", "{{json .ServerVersion}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=6,
            check=False,
        )
        result["usable"] = completed.returncode == 0
        if completed.returncode != 0:
            result["daemon_error"] = completed.stderr.strip()
    except Exception as exc:
        result["daemon_error"] = str(exc)
    return result


remote_path = sys.argv[1] if len(sys.argv) > 1 else "."
conda_environment = sys.argv[2] if len(sys.argv) > 2 else "EDOLAB"
logical = os.cpu_count()
try:
    loads = os.getloadavg()
except (AttributeError, OSError):
    loads = (None, None, None)

payload = {
    "schema_version": 1,
    "hostname": platform.node(),
    "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    "cpu": {
        "logical_cores": logical,
        "physical_cores": physical_cores(),
        "percent": cpu_percent(),
        "load_1m": loads[0],
        "load_5m": loads[1],
        "load_15m": loads[2],
        "load_ratio": (loads[0] / logical) if loads[0] is not None and logical else None,
    },
    "memory": memory_info(),
    "disk": disk_info(remote_path),
    "gpu": {"nvidia_smi": nvidia_info()},
    "python": {
        "installed": True,
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    },
    "conda": conda_info(conda_environment),
    "docker": docker_info(),
    "tools": {
        "rsync": command_info("rsync"),
        "tar": command_info("tar"),
        "git": command_info("git"),
    },
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''

_REMOTE_CONDA_CANDIDATES_PLACEHOLDER = "__EDOLAB_REMOTE_CONDA_CANDIDATES__"
if _REMOTE_PROBE_SCRIPT_TEMPLATE.count(_REMOTE_CONDA_CANDIDATES_PLACEHOLDER) != 1:
    raise RuntimeError("remote Conda candidate placeholder is missing or duplicated")
_REMOTE_RUNTIME_DEPENDENCIES_PLACEHOLDER = "__EDOLAB_REMOTE_RUNTIME_DEPENDENCIES__"
if _REMOTE_PROBE_SCRIPT_TEMPLATE.count(_REMOTE_RUNTIME_DEPENDENCIES_PLACEHOLDER) != 1:
    raise RuntimeError("remote runtime dependency placeholder is missing or duplicated")
REMOTE_PROBE_SCRIPT = _REMOTE_PROBE_SCRIPT_TEMPLATE.replace(
    _REMOTE_CONDA_CANDIDATES_PLACEHOLDER,
    repr(REMOTE_CONDA_EXECUTABLE_CANDIDATES),
).replace(
    _REMOTE_RUNTIME_DEPENDENCIES_PLACEHOLDER,
    repr(runtime_dependency_probe_specs()),
)


# Lightweight read-only sampling for the live server monitor.  Keep this
# separate from REMOTE_PROBE_SCRIPT so a three-second UI refresh does not run
# Conda discovery, disk checks, or registry/environment validation repeatedly.
REMOTE_RESOURCE_SCRIPT = r'''
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time


def cpu_times():
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except Exception:
        return None


def cpu_percent():
    before = cpu_times()
    if before is None:
        if platform.system() == "Darwin":
            try:
                completed = subprocess.run(
                    ["top", "-l", "2", "-n", "0", "-stats", "cpu"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=4,
                    check=False,
                )
                for line in reversed(completed.stdout.splitlines()):
                    if "CPU usage:" not in line:
                        continue
                    match = re.search(r"([0-9.]+)%\s*idle", line)
                    if match:
                        return round(100.0 - float(match.group(1)), 2)
            except Exception:
                pass
        return None
    time.sleep(0.12)
    after = cpu_times()
    if after is None:
        return None
    total = after[0] - before[0]
    idle = after[1] - before[1]
    return round(100.0 * (total - idle) / total, 2) if total > 0 else None


def memory_info():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
    except Exception:
        pass
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    if total is None and platform.system() == "Darwin":
        try:
            total = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
            ).strip())
            vm_output = subprocess.check_output(
                ["vm_stat"], text=True, timeout=3
            )
            page_match = re.search(r"page size of (\d+) bytes", vm_output)
            page_size = int(page_match.group(1)) if page_match else 4096
            pages = {}
            for line in vm_output.splitlines()[1:]:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                match = re.search(r"\d+", raw.replace(".", ""))
                if match:
                    pages[key.strip()] = int(match.group(0))
            available = page_size * sum(
                pages.get(key, 0)
                for key in (
                    "Pages free",
                    "Pages inactive",
                    "Pages speculative",
                    "Pages purgeable",
                )
            )
        except Exception:
            pass
    return {"total_bytes": total, "available_bytes": available}


def nvidia_info():
    executable = shutil.which("nvidia-smi")
    result = {"usable": False, "devices": []}
    if executable is None:
        result["error"] = "nvidia-smi is not installed"
        return result
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            result["error"] = completed.stderr.strip() or completed.stdout.strip()
            return result
        names = fields.split(",")
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            item = dict(zip(names, (value.strip() for value in line.split(","))))
            for key in ("index", "memory.total", "memory.used", "utilization.gpu", "temperature.gpu"):
                try:
                    item[key] = int(item[key])
                except (KeyError, ValueError):
                    pass
            result["devices"].append(item)
        result["usable"] = bool(result["devices"])
    except Exception as exc:
        result["error"] = str(exc)
    return result


logical = os.cpu_count()
try:
    loads = os.getloadavg()
except (AttributeError, OSError):
    loads = (None, None, None)

payload = {
    "schema_version": 1,
    "cpu": {
        "logical_cores": logical,
        "percent": cpu_percent(),
        "load_1m": loads[0],
        "load_5m": loads[1],
        "load_15m": loads[2],
        "load_ratio": (loads[0] / logical) if loads[0] is not None and logical else None,
    },
    "memory": memory_info(),
    "gpu": {"nvidia_smi": nvidia_info()},
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''


__all__ = [
    "REMOTE_CONDA_EXECUTABLE_CANDIDATES",
    "REMOTE_PROBE_SCRIPT",
    "REMOTE_RESOURCE_SCRIPT",
    "SSHClient",
    "SSHCommandError",
    "SSHConnection",
    "SSHError",
    "SSHResult",
    "TransferResult",
    "validate_private_key",
]
