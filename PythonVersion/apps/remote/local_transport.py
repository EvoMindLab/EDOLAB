"""Explicit local-process transport for the unified Relay compute pool.

The local target deliberately implements the same narrow file/script boundary
as the OpenSSH transport.  Consequently local jobs still run from an immutable
snapshot in a marker-owned workspace, detach into their own process group, can
be recovered after Relay restarts, and pass through the same artifact gates.
No SSH key, loopback SSH daemon, or source-checkout execution is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence
import uuid

from apps.remote.durability import fsync_directory, fsync_file
from apps.remote.ssh import (
    REMOTE_PROBE_SCRIPT,
    REMOTE_RESOURCE_SCRIPT,
    SSHResult,
    TransferResult,
)


class LocalTransportError(RuntimeError):
    """A deterministic local transport or filesystem operation failed."""


def _hardware_address() -> str | None:
    """The MAC address, or None when Python can only offer a random stand-in (multicast bit set)."""

    node = uuid.getnode()
    return None if (node >> 40) & 1 else f"{node:012x}"


def local_machine_fingerprint() -> str:
    """Return a stable, non-secret identity for the current OS account/host."""

    identity = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "node": _hardware_address(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "LOCAL-SHA256:" + hashlib.sha256(encoded).hexdigest()


class LocalProcessClient:
    """Run Relay's hardened workspace scripts directly on this machine."""

    def __init__(self, *, python_executable: str | Path | None = None) -> None:
        candidate = Path(python_executable or sys.executable).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise LocalTransportError(
                f"Local Python executable does not exist: {candidate}"
            ) from exc
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise LocalTransportError(
                f"Local Python executable is not executable: {resolved}"
            )
        self.python_executable = resolved

    def run_command(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> SSHResult:
        tokens = tuple(str(value) for value in argv)
        if not tokens or not tokens[0].strip():
            raise ValueError("local argv must not be empty")
        if any("\x00" in value for value in tokens):
            raise ValueError("local argv contains a NUL byte")
        started = time.monotonic()
        completed = subprocess.run(
            tokens,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False,
        )
        result = SSHResult(
            argv=tokens,
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=time.monotonic() - started,
        )
        if check and result.returncode:
            detail = result.stderr_text.strip() or result.stdout_text.strip()
            message = f"Local command exited with code {result.returncode}"
            if detail:
                message += f": {detail}"
            raise LocalTransportError(message)
        return result

    def run_script(
        self,
        script: str | bytes,
        args: Sequence[str] = (),
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> SSHResult:
        source = script.encode("utf-8") if isinstance(script, str) else bytes(script)
        return self.run_command(
            [str(self.python_executable), "-", *[str(value) for value in args]],
            input_bytes=source,
            timeout=timeout,
            check=check,
        )

    def probe(
        self,
        local_path: str,
        conda_env: str = "EDOLAB",
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        result = self.run_script(
            REMOTE_PROBE_SCRIPT,
            [local_path, conda_env],
            timeout=timeout,
        )
        payload = self._last_json_object(result, "local environment probe")
        if payload.get("schema_version") != 1:
            raise LocalTransportError("local probe returned an unsupported payload")
        payload["fingerprint"] = local_machine_fingerprint()
        payload["transport"] = "local"
        return payload

    def sample_resources(
        self,
        *,
        timeout: float = 10.0,
        expected_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        fingerprint = local_machine_fingerprint()
        if expected_fingerprint and expected_fingerprint != fingerprint:
            raise LocalTransportError(
                "The local machine identity differs from the identity pinned by "
                "this compute target."
            )
        result = self.run_script(REMOTE_RESOURCE_SCRIPT, timeout=timeout)
        payload = self._last_json_object(result, "local resource sample")
        if payload.get("schema_version") != 1:
            raise LocalTransportError(
                "local resource sample returned an unsupported payload"
            )
        return payload

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        mode: int = 0o600,
        timeout: float | None = None,
    ) -> TransferResult:
        del timeout  # Local copies are bounded by the calling operation lifecycle.
        source = Path(local_path)
        if source.is_symlink() or not source.is_file():
            raise LocalTransportError(
                f"local transfer source is not a regular file: {source}"
            )
        destination = self._safe_destination(remote_path)
        expected_size, expected_sha256 = _file_identity(source)
        if destination.is_file() and not destination.is_symlink():
            actual_size, actual_sha256 = _file_identity(destination)
            if (actual_size, actual_sha256) == (expected_size, expected_sha256):
                destination.chmod(mode & 0o777)
                return TransferResult(
                    destination,
                    expected_size,
                    expected_sha256,
                    0.0,
                )

        started = time.monotonic()
        partial = destination.with_name(
            f".{destination.name}.part-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with source.open("rb") as source_handle, partial.open("xb") as target:
                shutil.copyfileobj(source_handle, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            actual_size, actual_sha256 = _file_identity(partial)
            if (actual_size, actual_sha256) != (expected_size, expected_sha256):
                raise LocalTransportError(
                    "local copied file failed size or SHA-256 verification"
                )
            partial.chmod(mode & 0o777)
            os.replace(partial, destination)
            fsync_directory(destination.parent)
        finally:
            partial.unlink(missing_ok=True)
        return TransferResult(
            destination,
            expected_size,
            expected_sha256,
            time.monotonic() - started,
        )

    def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
        *,
        timeout: float | None = None,
    ) -> TransferResult:
        del timeout
        source = Path(remote_path).expanduser()
        if source.is_symlink() or not source.is_file():
            raise LocalTransportError(
                f"local artifact source is not a regular file: {source}"
            )
        destination = self._safe_destination(local_path)
        expected_size, expected_sha256 = _file_identity(source)
        started = time.monotonic()
        partial = destination.with_name(
            f".{destination.name}.part-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with source.open("rb") as source_handle, partial.open("xb") as target:
                shutil.copyfileobj(source_handle, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            actual_size, actual_sha256 = _file_identity(partial)
            if (actual_size, actual_sha256) != (expected_size, expected_sha256):
                raise LocalTransportError(
                    "local artifact copy failed size or SHA-256 verification"
                )
            fsync_file(partial)
            os.replace(partial, destination)
            fsync_directory(destination.parent)
        finally:
            partial.unlink(missing_ok=True)
        return TransferResult(
            destination,
            expected_size,
            expected_sha256,
            time.monotonic() - started,
        )

    @staticmethod
    def _safe_destination(value: str | Path) -> Path:
        destination = Path(value).expanduser()
        if not destination.is_absolute():
            destination = destination.resolve()
        parent = destination.parent
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise LocalTransportError(
                f"local transfer parent is unsafe: {parent}"
            )
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if parent.is_symlink() or not parent.is_dir():
            raise LocalTransportError(
                f"local transfer parent is unsafe: {parent}"
            )
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise LocalTransportError(
                f"local transfer destination is unsafe: {destination}"
            )
        return destination

    @staticmethod
    def _last_json_object(result: SSHResult, operation: str) -> dict[str, Any]:
        lines = [line for line in result.stdout_text.splitlines() if line.strip()]
        if not lines:
            raise LocalTransportError(f"{operation} returned no JSON")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise LocalTransportError(f"{operation} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise LocalTransportError(f"{operation} returned a non-object payload")
        return payload


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


__all__ = [
    "LocalProcessClient",
    "LocalTransportError",
    "local_machine_fingerprint",
]
