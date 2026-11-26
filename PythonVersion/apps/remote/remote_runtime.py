"""Marker-owned remote workspace lifecycle and verified artifact collection."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any, Callable, Mapping

from apps.remote.durability import fsync_directories
from apps.remote.domain import stable_json_dumps
from apps.remote.ssh import SSHClient, SSHError


@dataclass(frozen=True, slots=True)
class RemoteWorkspace:
    root: str
    run_dir: str
    previous_manifest: Mapping[str, Any] | None
    base_snapshot_id: str | None = None
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class RemotePoll:
    alive: bool
    phase: str
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    stdout_offset: int
    stderr_offset: int
    stdout_more: bool
    stderr_more: bool
    resource: Mapping[str, Any]
    process_identity_valid: bool = True


@dataclass(frozen=True, slots=True)
class RemoteArtifact:
    remote_path: str
    size_bytes: int
    sha256: str
    exit_code: int | None
    bundle_validation: Mapping[str, Any]
    manifest: Mapping[str, Any]


class RemoteEnvironmentPreflightError(RuntimeError):
    """Exact uploaded source cannot start in the frozen remote environment."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def prepare_remote_workspace(
    client: SSHClient,
    *,
    remote_root: str,
    installation_id: str,
    run_id: str,
    snapshot_id: str | None = None,
) -> RemoteWorkspace:
    payload = _json_result(
        client.run_script(
            PREPARE_REMOTE_WORKSPACE_SCRIPT,
            [remote_root, installation_id, run_id, snapshot_id or "-"],
            timeout=30,
        ).stdout_text
    )
    return RemoteWorkspace(
        root=str(payload["root"]),
        run_dir=str(payload["run_dir"]),
        previous_manifest=payload.get("previous_manifest"),
        base_snapshot_id=(
            str(payload["base_snapshot_id"])
            if payload.get("base_snapshot_id")
            else None
        ),
        cache_hit=bool(payload.get("cache_hit", False)),
    )


def upload_and_verify_source(
    client: SSHClient,
    *,
    workspace: RemoteWorkspace,
    payload_path: str | Path | None,
    manifest_path: str | Path,
    task_path: str | Path,
    snapshot_id: str,
    sync_plan: Mapping[str, Any],
    stage_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    incoming = f"{workspace.run_dir}/incoming"
    transfer_paths = tuple(str(path) for path in sync_plan.get("transfer_paths") or ())
    target_snapshot_id = str(sync_plan.get("target_snapshot_id") or "")
    base_snapshot_id = str(sync_plan.get("base_snapshot_id") or "") or None
    if target_snapshot_id != snapshot_id:
        raise ValueError("sync plan target does not match the frozen snapshot")
    if base_snapshot_id != workspace.base_snapshot_id:
        raise ValueError("sync plan base does not match the prepared remote cache")
    payload = Path(payload_path) if payload_path is not None else None
    if transfer_paths and payload is None:
        raise ValueError("sync plan requires a source payload")
    if not transfer_paths and payload is not None:
        raise ValueError("source payload is unnecessary for an empty content delta")

    client.upload_file(
        manifest_path,
        f"{incoming}/manifest.json",
        mode=0o600,
        timeout=120,
    )
    client.upload_file(
        task_path,
        f"{workspace.run_dir}/meta/task.jsonl",
        mode=0o600,
        timeout=120,
    )
    transfer_detail = {
        **dict(sync_plan),
        "payload_size_bytes": payload.stat().st_size if payload is not None else 0,
    }
    if stage_callback is not None:
        stage_callback("transfer", transfer_detail)
    if payload is not None:
        # Bound a half-open connection without treating a healthy but slow WAN
        # as failed.  Unlike the old full-snapshot path this timeout is based on
        # only the changed content archive.
        payload_timeout = max(
            120.0,
            min(7200.0, 60.0 + payload.stat().st_size / (16 * 1024)),
        )
        client.upload_file(
            payload,
            f"{incoming}/source-delta.tar.gz",
            mode=0o600,
            timeout=payload_timeout,
        )
    if stage_callback is not None:
        stage_callback("verify", transfer_detail)
    result = _json_result(
        client.run_script(
            VERIFY_REMOTE_SOURCE_SCRIPT,
            [
                workspace.run_dir,
                snapshot_id,
                base_snapshot_id or "-",
                stable_json_dumps(transfer_paths),
            ],
            timeout=900,
        ).stdout_text
    )
    result["plan"] = dict(sync_plan)
    result["payload_size_bytes"] = transfer_detail["payload_size_bytes"]
    return result


def launch_remote_worker(
    client: SSHClient,
    *,
    workspace: RemoteWorkspace,
    conda_env: str,
    conda_executable: str,
    max_parallel_runs: int,
    install_editable: bool = True,
    worker_kind: str = "batch",
) -> int:
    if worker_kind not in {"batch", "benchmark", "plot"}:
        raise ValueError(f"unsupported worker kind: {worker_kind}")
    preflight = _json_result(
        client.run_script(
            PREFLIGHT_REMOTE_SOURCE_SCRIPT,
            [
                workspace.run_dir,
                conda_env,
                conda_executable,
                "editable" if install_editable else "isolated",
                worker_kind,
            ],
            timeout=300,
        ).stdout_text
    )
    if preflight.get("ok") is not True:
        raise RemoteEnvironmentPreflightError(
            str(preflight.get("code") or "ENV_SOURCE_PREFLIGHT"),
            str(
                preflight.get("message")
                or "The exact uploaded source failed remote environment preflight."
            ),
            preflight,
        )
    payload = _json_result(
        client.run_script(
            LAUNCH_REMOTE_WORKER_SCRIPT,
            [
                workspace.run_dir,
                conda_env,
                str(int(max_parallel_runs)),
                conda_executable,
                "editable" if install_editable else "isolated",
                worker_kind,
            ],
            timeout=30,
        ).stdout_text
    )
    return int(payload["pid"])


def discover_remote_worker(
    client: SSHClient,
    *,
    workspace: RemoteWorkspace,
) -> int:
    """Recover a marker-owned worker PID after local controller state loss."""

    payload = _json_result(
        client.run_script(
            DISCOVER_REMOTE_WORKER_SCRIPT,
            [workspace.run_dir],
            timeout=20,
        ).stdout_text
    )
    return int(payload["pid"])


def poll_remote_worker(
    client: SSHClient,
    *,
    workspace: RemoteWorkspace,
    remote_pid: int,
    stdout_offset: int,
    stderr_offset: int,
) -> RemotePoll:
    payload = _json_result(
        client.run_script(
            POLL_REMOTE_WORKER_SCRIPT,
            [
                workspace.run_dir,
                str(int(remote_pid)),
                str(int(stdout_offset)),
                str(int(stderr_offset)),
            ],
            timeout=30,
        ).stdout_text
    )
    return RemotePoll(
        alive=bool(payload.get("alive")),
        phase=str(payload.get("phase") or "unknown"),
        exit_code=(
            int(payload["exit_code"])
            if payload.get("exit_code") is not None
            else None
        ),
        stdout=base64.b64decode(payload.get("stdout_b64") or ""),
        stderr=base64.b64decode(payload.get("stderr_b64") or ""),
        stdout_offset=int(payload.get("stdout_offset") or 0),
        stderr_offset=int(payload.get("stderr_offset") or 0),
        stdout_more=bool(payload.get("stdout_more")),
        stderr_more=bool(payload.get("stderr_more")),
        resource=dict(payload.get("resource") or {}),
        process_identity_valid=bool(payload.get("process_identity_valid", True)),
    )


def cancel_remote_worker(
    client: SSHClient,
    *,
    workspace: RemoteWorkspace,
    remote_pid: int,
    force: bool = False,
) -> dict[str, Any]:
    return _json_result(
        client.run_script(
            CANCEL_REMOTE_WORKER_SCRIPT,
            [workspace.run_dir, str(int(remote_pid)), "KILL" if force else "TERM"],
            timeout=20,
        ).stdout_text
    )


def build_remote_artifact(
    client: SSHClient,
    *,
    workspace: RemoteWorkspace,
    worker_kind: str = "batch",
) -> RemoteArtifact:
    if worker_kind not in {"batch", "benchmark", "plot"}:
        raise ValueError(f"unsupported worker kind: {worker_kind}")
    payload = _json_result(
        client.run_script(
            BUILD_REMOTE_ARTIFACT_SCRIPT,
            [workspace.run_dir, worker_kind],
            # Large research bundles may need substantial time for two full
            # SQLite checks, hashing, and compression.  SSH keepalives still
            # bound dead transports; this ceiling avoids relaunching a healthy
            # collector every five minutes.
            timeout=28_800,
        ).stdout_text
    )
    return RemoteArtifact(
        remote_path=str(payload["archive_path"]),
        size_bytes=int(payload["size_bytes"]),
        sha256=str(payload["sha256"]),
        exit_code=(
            int(payload["exit_code"])
            if payload.get("exit_code") is not None
            else None
        ),
        bundle_validation=dict(payload.get("bundle_validation") or {}),
        manifest=dict(payload["manifest"]),
    )


def extract_verified_artifact(
    archive_path: str | Path,
    destination: str | Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    archive = Path(archive_path)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                relative = _safe_relative(member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise SSHError(f"unsafe artifact archive member: {member.name}")
                output = target / relative
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise SSHError(f"unsupported artifact archive member: {member.name}")
                output.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise SSHError(f"artifact member cannot be read: {member.name}")
                with output.open("xb") as destination_handle:
                    shutil.copyfileobj(source, destination_handle, length=1024 * 1024)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())

        expected_files = list(manifest.get("files") or [])
        expected_paths: set[str] = set()
        verified: list[dict[str, Any]] = []
        for item in expected_files:
            relative_text = str(item["path"])
            relative = _safe_relative(relative_text)
            expected_paths.add(relative.as_posix())
            path = target / relative
            if not path.is_file() or path.is_symlink():
                raise SSHError(f"artifact file is missing: {relative_text}")
            size = path.stat().st_size
            digest = _sha256_file(path)
            if size != int(item["size_bytes"]) or digest != str(item["sha256"]):
                raise SSHError(f"artifact checksum mismatch: {relative_text}")
            verified.append(
                {"path": relative.as_posix(), "size_bytes": size, "sha256": digest}
            )
        actual_paths = {
            path.relative_to(target).as_posix()
            for path in target.rglob("*")
            if path.is_file()
        }
        allowed_extra = {"meta/artifact-manifest.json"}
        unexpected = actual_paths - expected_paths - allowed_extra
        if unexpected:
            raise SSHError(
                "artifact archive contains unmanifested files: "
                + ", ".join(sorted(unexpected))
            )
        # Each file was fsynced before close.  Persist every directory entry
        # created by extraction, plus the target entry in its working parent,
        # only after the manifest/checksum pass has accepted the full tree.
        directories = [target]
        directories.extend(path for path in target.rglob("*") if path.is_dir())
        fsync_directories(directories, include_parents=(target.parent,))
        return verified
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def cleanup_remote_workspace(
    client: SSHClient,
    *,
    workspace: RemoteWorkspace,
    installation_id: str,
    run_id: str,
) -> dict[str, Any]:
    return _json_result(
        client.run_script(
            CLEANUP_REMOTE_WORKSPACE_SCRIPT,
            [workspace.root, installation_id, run_id],
            timeout=60,
        ).stdout_text
    )


def _json_result(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise SSHError("remote helper returned no JSON")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SSHError("remote helper returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SSHError("remote helper returned a non-object payload")
    return payload


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SSHError(f"unsafe relative path: {value}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_REMOTE_SOURCE_CACHE_HELPERS = r'''
def relay_snapshot_id(value):
    text = str(value or "")
    if not re.fullmatch(r"[a-f0-9]{64}", text):
        raise RuntimeError("invalid source snapshot id")
    return text

def relay_safe_relative(value):
    text = str(value)
    if "\\" in text or any(token in text for token in ("\x00", "\n", "\r")):
        raise RuntimeError("unsafe source path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise RuntimeError("unsafe source path")
    return relative

def relay_manifest_files(manifest):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise RuntimeError("unsupported source manifest schema")
    files = {}
    for raw in manifest["files"]:
        if not isinstance(raw, dict):
            raise RuntimeError("source manifest file entry is invalid")
        relative = relay_safe_relative(raw.get("path")).as_posix()
        if relative in files:
            raise RuntimeError("source manifest contains a duplicate path")
        sha256 = str(raw.get("sha256") or "")
        if not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise RuntimeError("source manifest contains an invalid checksum")
        size = int(raw.get("size_bytes", -1))
        mode = int(raw.get("mode", -1))
        if size < 0 or mode < 0 or mode > 0o7777:
            raise RuntimeError("source manifest contains invalid file metadata")
        files[relative] = {"path": relative, "sha256": sha256, "size_bytes": size, "mode": mode}
    paths = set(files)
    for relative in paths:
        parts = PurePosixPath(relative).parts
        for index in range(1, len(parts)):
            if PurePosixPath(*parts[:index]).as_posix() in paths:
                raise RuntimeError("source manifest contains a file/directory conflict")
    return files

def relay_manifest_hash(manifest):
    canonical = {
        "schema_version": manifest.get("schema_version"),
        "source": manifest.get("source"),
        "files": manifest.get("files"),
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def relay_validate_manifest(manifest, expected_snapshot_id):
    snapshot_id = relay_snapshot_id(expected_snapshot_id)
    if not isinstance(manifest, dict):
        raise RuntimeError("source manifest is not an object")
    if relay_manifest_hash(manifest) != snapshot_id:
        raise RuntimeError("source manifest identity mismatch")
    if manifest.get("manifest_sha256") != snapshot_id or manifest.get("snapshot_id") != snapshot_id:
        raise RuntimeError("source manifest claimed identity mismatch")
    relay_manifest_files(manifest)
    declared = manifest.get("source_archive")
    if not isinstance(declared, dict):
        raise RuntimeError("source manifest archive metadata is missing")
    if str(declared.get("name") or "") != "source.tar.gz":
        raise RuntimeError("source manifest archive name is invalid")
    if int(declared.get("size_bytes", -1)) < 0 or not re.fullmatch(r"[a-f0-9]{64}", str(declared.get("sha256") or "")):
        raise RuntimeError("source manifest archive metadata is invalid")
    return manifest

def relay_digest(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()

def relay_cache_entry_exists(cache_root, snapshot_id):
    snapshot_id = relay_snapshot_id(snapshot_id)
    return any(
        path.exists() or path.is_symlink()
        for path in (cache_root / "v2" / snapshot_id, cache_root / snapshot_id)
    )

def relay_load_cache(cache_root, snapshot_id, verify_archive=False):
    snapshot_id = relay_snapshot_id(snapshot_id)
    v2_root = cache_root / "v2"
    if v2_root.exists() or v2_root.is_symlink():
        if v2_root.is_symlink() or not v2_root.is_dir():
            raise RuntimeError("remote source tree-cache path is unsafe")
    v2_entry = v2_root / snapshot_id
    legacy_entry = cache_root / snapshot_id
    if v2_entry.exists() or v2_entry.is_symlink():
        entry = v2_entry
    else:
        entry = legacy_entry
    if entry.is_symlink() or not entry.is_dir():
        raise RuntimeError("remote source cache entry is missing or unsafe")
    manifest_path = entry / "manifest.json"
    archive_path = entry / "source.tar.gz"
    tree_path = entry / "tree"
    index_path = entry / "tree-index.json"
    marker_path = entry / "cache.json"
    for name, path in (("manifest", manifest_path), ("marker", marker_path)):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("remote source cache " + name + " is missing or unsafe")
    manifest = relay_validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), snapshot_id)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    schema_version = int(marker.get("schema_version", -1)) if isinstance(marker, dict) else -1
    if tree_path.exists() or tree_path.is_symlink():
        if tree_path.is_symlink() or not tree_path.is_dir():
            raise RuntimeError("remote source cache tree is unsafe")
        tree = tree_path
    else:
        tree = None
    if archive_path.exists() or archive_path.is_symlink():
        if archive_path.is_symlink() or not archive_path.is_file():
            raise RuntimeError("remote source cache archive is unsafe")
        archive = archive_path
    else:
        archive = None

    if schema_version == 1:
        declared = manifest["source_archive"]
        expected_marker = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "archive_sha256": declared["sha256"],
            "archive_size_bytes": int(declared["size_bytes"]),
        }
        if marker != expected_marker or archive is None:
            raise RuntimeError("remote source cache marker identity mismatch")
        if verify_archive:
            size, digest = relay_digest(archive)
            if size != expected_marker["archive_size_bytes"] or digest != expected_marker["archive_sha256"]:
                raise RuntimeError("remote source cache archive checksum mismatch")
    elif schema_version == 2:
        if entry != v2_entry:
            raise RuntimeError("remote source tree cache is outside the v2 namespace")
        if index_path.is_symlink() or not index_path.is_file():
            raise RuntimeError("remote source cache tree index is missing or unsafe")
        index_size, index_sha256 = relay_digest(index_path)
        expected_marker = {
            "schema_version": 2,
            "snapshot_id": snapshot_id,
            "manifest_sha256": snapshot_id,
            "storage": "tree",
            "file_count": len(relay_manifest_files(manifest)),
            "tree_index_size_bytes": index_size,
            "tree_index_sha256": index_sha256,
        }
        if marker != expected_marker or tree is None:
            raise RuntimeError("remote source cache marker identity mismatch")
    else:
        raise RuntimeError("unsupported remote source cache schema")
    return manifest, tree, archive, schema_version
'''


PREPARE_REMOTE_WORKSPACE_SCRIPT = r'''
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
''' + _REMOTE_SOURCE_CACHE_HELPERS + r'''
requested, installation_id, run_id = sys.argv[1:4]
requested_snapshot_id = sys.argv[4] if len(sys.argv) > 4 else "-"
if not re.fullmatch(r"[A-Za-z0-9_.-]{8,128}", installation_id):
    raise RuntimeError("invalid installation id")
if not re.fullmatch(r"[A-Fa-f0-9]{16,64}", run_id):
    raise RuntimeError("invalid run id")
if requested_snapshot_id != "-":
    requested_snapshot_id = relay_snapshot_id(requested_snapshot_id)
if any(token in requested for token in ("\x00", "\n", "\r", "$", "*", "?", "[", "]")):
    raise RuntimeError("unsafe remote root")
requested_path = Path(requested).expanduser()
if requested_path.is_symlink():
    raise RuntimeError("remote root must not be a symlink")
root = requested_path.resolve(strict=False)
home = Path.home().resolve()
if root == Path("/") or root == home:
    raise RuntimeError("remote root is too broad")
root.mkdir(parents=True, exist_ok=True, mode=0o700)
if root.is_symlink() or not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
    raise RuntimeError("remote root is not writable")
marker = root / ".edolab-relay-root.json"
expected = {"schema_version": 1, "installation_id": installation_id}
if marker.exists():
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError("remote root marker is not a regular file")
    if json.loads(marker.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("remote root belongs to another Relay installation")
else:
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
runs = root / "runs"
runs.mkdir(mode=0o700, exist_ok=True)
if runs.is_symlink() or not runs.is_dir():
    raise RuntimeError("remote runs path is not a regular directory")
cache_root = root / ".edolab-relay-snapshots"
cache_root.mkdir(mode=0o700, exist_ok=True)
if cache_root.is_symlink() or not cache_root.is_dir():
    raise RuntimeError("remote source cache path is unsafe")
tree_cache_root = cache_root / "v2"
tree_cache_root.mkdir(mode=0o700, exist_ok=True)
if tree_cache_root.is_symlink() or not tree_cache_root.is_dir():
    raise RuntimeError("remote source tree-cache path is unsafe")
cleanup_state = root / ".edolab-relay-cleanup"
if cleanup_state.exists() and (cleanup_state.is_symlink() or not cleanup_state.is_dir()):
    raise RuntimeError("remote cleanup state path is unsafe")
cleanup_record = cleanup_state / (run_id + ".json")
if cleanup_record.exists():
    if cleanup_record.is_symlink() or not cleanup_record.is_file():
        raise RuntimeError("remote cleanup record is unsafe")
    raise RuntimeError("remote run id was already cleaned and cannot be reused")
run_dir = runs / run_id
run_marker = run_dir / ".edolab-relay-run.json"
run_expected = {"schema_version": 1, "installation_id": installation_id, "run_id": run_id}
if run_dir.exists():
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("existing remote run path is not a regular directory")
    if run_marker.is_symlink() or not run_marker.is_file() or json.loads(run_marker.read_text(encoding="utf-8")) != run_expected:
        raise RuntimeError("existing remote run directory is not owned by this run")
else:
    run_dir.mkdir(mode=0o700)
    run_marker.write_text(json.dumps(run_expected, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.chmod(run_marker, 0o600)
if run_dir.resolve().parent != runs.resolve():
    raise RuntimeError("remote run directory escapes the owned runs path")
for name in ("incoming", "logs", "meta", "results"):
    child = run_dir / name
    child.mkdir(mode=0o700, exist_ok=True)
    if child.is_symlink() or not child.is_dir():
        raise RuntimeError("remote run child is not a regular directory: " + name)

previous = None
base_snapshot_id = None
cache_hit = False
if requested_snapshot_id != "-" and relay_cache_entry_exists(cache_root, requested_snapshot_id):
    previous, _, _, _ = relay_load_cache(cache_root, requested_snapshot_id, verify_archive=False)
    base_snapshot_id = requested_snapshot_id
    cache_hit = True
else:
    last_manifest = root / ".last-source-manifest.json"
    if last_manifest.exists() and (last_manifest.is_symlink() or not last_manifest.is_file()):
        raise RuntimeError("remote source-history manifest is unsafe")
    if last_manifest.is_file():
        try:
            last = json.loads(last_manifest.read_text(encoding="utf-8"))
            candidate = relay_snapshot_id(last.get("snapshot_id") or last.get("manifest_sha256"))
            previous, _, _, _ = relay_load_cache(cache_root, candidate, verify_archive=False)
            base_snapshot_id = candidate
        except Exception:
            previous = None
            base_snapshot_id = None
print(json.dumps({
    "root": str(root),
    "run_dir": str(run_dir),
    "previous_manifest": previous,
    "base_snapshot_id": base_snapshot_id,
    "cache_hit": cache_hit,
}, sort_keys=True, separators=(",", ":")))
'''


VERIFY_REMOTE_SOURCE_SCRIPT = r'''
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
''' + _REMOTE_SOURCE_CACHE_HELPERS + r'''
def relay_expected_directories(files):
    directories = set()
    for relative in files:
        parts = PurePosixPath(relative).parts
        for index in range(1, len(parts)):
            directories.add(PurePosixPath(*parts[:index]).as_posix())
    return directories

def relay_tree_layout(root):
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("source tree is missing or unsafe")
    files = set()
    directories = set()
    for path in root.rglob("*"):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("source tree contains a symlink")
        if stat.S_ISREG(info.st_mode):
            files.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            directories.add(relative)
        else:
            raise RuntimeError("source tree contains an unsupported file type")
    return files, directories

def relay_cache_file_mode(item):
    # Cache files are never executed in place.  Keep them owner-readable so a
    # later clone can copy unusual write-only source files, while removing all
    # write bits from the shared cache representation.
    return (int(item["mode"]) & ~0o222) | 0o400

def relay_validate_tree_shape(root, expected, *, cache_modes, source_root=None):
    files, directories = relay_tree_layout(root)
    if files != set(expected) or directories != relay_expected_directories(expected):
        raise RuntimeError("materialized source tree differs from the manifest")
    observed = {}
    for relative, item in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        info = path.lstat()
        required_mode = relay_cache_file_mode(item) if cache_modes else int(item["mode"])
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != int(item["size_bytes"])
            or stat.S_IMODE(info.st_mode) != required_mode
        ):
            raise RuntimeError("materialized source metadata differs from the manifest")
        observed[relative] = info
        if source_root is not None:
            source = source_root.joinpath(*PurePosixPath(relative).parts)
            source_info = source.lstat()
            if (info.st_dev, info.st_ino) == (source_info.st_dev, source_info.st_ino):
                raise RuntimeError("source clone unexpectedly shares a file inode")
    return observed

def relay_extract_archive(archive_path, destination, expected, *, verify_content, exact_tree):
    seen = set()
    hashed_bytes = 0
    # Streaming mode inflates the gzip exactly once.  Delta members are the
    # only source bytes read on the normal v2 path.
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            relative = relay_safe_relative(member.name).as_posix()
            if relative in seen:
                raise RuntimeError("source archive contains a duplicate member")
            if relative not in expected:
                raise RuntimeError("source archive contains an unexpected member")
            if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("source archive contains an unsafe member")
            item = expected[relative]
            if member.size != item["size_bytes"] or (member.mode & 0o7777) != item["mode"]:
                raise RuntimeError("source archive metadata differs from the manifest")
            target = destination.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            current = destination
            for part in PurePosixPath(relative).parts[:-1]:
                current = current / part
                if current.is_symlink() or not current.is_dir():
                    raise RuntimeError("source archive parent path is unsafe")
            if target.exists() or target.is_symlink():
                raise RuntimeError("source archive would overwrite an existing path")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError("source archive member is unreadable")
            digest = hashlib.sha256() if verify_content else None
            size = 0
            with target.open("xb") as output:
                while True:
                    chunk = extracted.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    if digest is not None:
                        digest.update(chunk)
                    size += len(chunk)
            os.chmod(target, item["mode"])
            if size != item["size_bytes"]:
                raise RuntimeError("source archive member size changed during extraction")
            if digest is not None:
                hashed_bytes += size
                if digest.hexdigest() != item["sha256"]:
                    raise RuntimeError("source delta checksum mismatch")
            seen.add(relative)
    if seen != set(expected):
        raise RuntimeError("source archive member set differs from the transfer plan")
    if exact_tree and relay_tree_layout(destination)[0] != set(expected):
        raise RuntimeError("source archive file set differs from the manifest")
    return hashed_bytes

def relay_prune_empty_directories(root):
    directories = [path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

def relay_remove_paths(root, paths):
    for relative in sorted(paths, key=lambda value: len(PurePosixPath(value).parts), reverse=True):
        target = root.joinpath(*relay_safe_relative(relative).parts)
        if target.is_symlink():
            raise RuntimeError("refusing to remove a symlink from the source partial")
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            try:
                target.rmdir()
            except OSError as exc:
                raise RuntimeError("source path conflict could not be removed") from exc
    relay_prune_empty_directories(root)

def relay_json_bytes(payload):
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def relay_write_json_file(path, payload):
    encoded = relay_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())

def relay_fsync_directory(path):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def relay_fsync_tree(root):
    files, directories = relay_tree_layout(root)
    for relative in sorted(files):
        path = root.joinpath(*PurePosixPath(relative).parts)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError("source tree changed during fsync")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for relative in sorted(directories, key=lambda value: len(PurePosixPath(value).parts), reverse=True):
        relay_fsync_directory(root.joinpath(*PurePosixPath(relative).parts))
    relay_fsync_directory(root)

def relay_discard_tree(path):
    if not (path.exists() or path.is_symlink()):
        return
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("refusing to discard an unsafe tree")
    def make_writable(function, raw_path, exc_info):
        item = Path(raw_path)
        if item.is_symlink():
            item.unlink()
            return
        os.chmod(item, 0o700 if item.is_dir() else 0o600)
        function(raw_path)
    shutil.rmtree(path, onerror=make_writable)

def relay_apply_manifest_modes(root, files):
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise RuntimeError("source mode restoration found an unsafe root")
    os.chmod(root, 0o700)
    directories = relay_expected_directories(files)
    for relative in sorted(directories, key=lambda value: len(PurePosixPath(value).parts)):
        directory = root.joinpath(*PurePosixPath(relative).parts)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError("source mode restoration found an unsafe directory")
        os.chmod(directory, 0o700)
    for relative, item in files.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != int(item["size_bytes"])
        ):
            raise RuntimeError("source mode restoration found an unsafe file")
        os.chmod(path, int(item["mode"]))

def relay_freeze_cache_tree(root, files):
    relay_apply_manifest_modes(root, files)
    for relative, item in files.items():
        os.chmod(root.joinpath(*PurePosixPath(relative).parts), relay_cache_file_mode(item))
    directories = relay_expected_directories(files)
    for relative in sorted(directories, key=lambda value: len(PurePosixPath(value).parts), reverse=True):
        os.chmod(root.joinpath(*PurePosixPath(relative).parts), 0o555)
    os.chmod(root, 0o555)

def relay_stat_identity(info):
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        stat.S_IMODE(info.st_mode),
        int(info.st_nlink),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )

def relay_copy_file_descriptor(source_descriptor, destination_descriptor):
    os.lseek(source_descriptor, 0, os.SEEK_SET)
    os.ftruncate(destination_descriptor, 0)
    os.lseek(destination_descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_descriptor, 1024 * 1024)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(destination_descriptor, view)
            if written <= 0:
                raise RuntimeError("source clone write made no progress")
            view = view[written:]

def relay_clone_tree(source, destination, files, source_seal):
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("source clone destination already exists")
    if set(source_seal) != set(files):
        raise RuntimeError("source clone seal differs from the manifest")
    destination.mkdir(mode=0o700)
    for relative in sorted(
        relay_expected_directories(files),
        key=lambda value: len(PurePosixPath(value).parts),
    ):
        directory = destination.joinpath(*PurePosixPath(relative).parts)
        directory.mkdir(mode=0o700)
    used_copy = False
    reflink_available = True
    ficlone = 0x40049409
    fallback_errors = {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTTY,
        errno.EXDEV,
        errno.EPERM,
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        getattr(errno, "ENOTSUP", errno.EINVAL),
    }
    try:
        for relative, item in sorted(files.items()):
            source_path = source.joinpath(*PurePosixPath(relative).parts)
            destination_path = destination.joinpath(*PurePosixPath(relative).parts)
            source_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                source_flags |= os.O_NOFOLLOW
            source_descriptor = os.open(source_path, source_flags)
            destination_descriptor = None
            try:
                source_before = os.fstat(source_descriptor)
                if (
                    not stat.S_ISREG(source_before.st_mode)
                    or source_before.st_nlink != 1
                    or relay_stat_identity(source_before)
                    != relay_stat_identity(source_seal[relative])
                ):
                    raise RuntimeError("source cache changed before cloning")
                destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    destination_flags |= os.O_NOFOLLOW
                destination_descriptor = os.open(
                    destination_path, destination_flags, 0o600
                )
                if reflink_available:
                    try:
                        fcntl.ioctl(destination_descriptor, ficlone, source_descriptor)
                    except OSError as exc:
                        if exc.errno not in fallback_errors:
                            raise
                        if exc.errno != errno.EINVAL:
                            reflink_available = False
                        used_copy = True
                        relay_copy_file_descriptor(
                            source_descriptor, destination_descriptor
                        )
                else:
                    used_copy = True
                    relay_copy_file_descriptor(
                        source_descriptor, destination_descriptor
                    )
                os.fchmod(destination_descriptor, relay_cache_file_mode(item))
                destination_info = os.fstat(destination_descriptor)
                source_after = os.fstat(source_descriptor)
                path_after = source_path.lstat()
                if (
                    relay_stat_identity(source_after)
                    != relay_stat_identity(source_before)
                    or relay_stat_identity(path_after)
                    != relay_stat_identity(source_before)
                    or not stat.S_ISREG(destination_info.st_mode)
                    or destination_info.st_nlink != 1
                    or destination_info.st_size != int(item["size_bytes"])
                    or (destination_info.st_dev, destination_info.st_ino)
                    == (source_before.st_dev, source_before.st_ino)
                ):
                    raise RuntimeError("source cache changed while cloning")
            finally:
                if destination_descriptor is not None:
                    os.close(destination_descriptor)
                os.close(source_descriptor)
        relay_validate_tree_shape(
            destination, files, cache_modes=True, source_root=source
        )
    except BaseException:
        relay_discard_tree(destination)
        raise
    return "copy" if used_copy else "reflink"

def relay_tree_index_payload(snapshot_id, tree, files):
    relay_validate_tree_shape(tree, files, cache_modes=True)
    entries = []
    for relative, item in sorted(files.items()):
        info = tree.joinpath(*PurePosixPath(relative).parts).lstat()
        entries.append({
            "path": relative,
            "size_bytes": int(info.st_size),
            "manifest_mode": int(item["mode"]),
            "cache_mode": stat.S_IMODE(info.st_mode),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "nlink": int(info.st_nlink),
            "mtime_ns": int(info.st_mtime_ns),
            "ctime_ns": int(info.st_ctime_ns),
        })
    return {"schema_version": 1, "snapshot_id": snapshot_id, "files": entries}

def relay_validate_tree_cache(tree, manifest):
    snapshot_id = relay_snapshot_id(manifest["snapshot_id"])
    files = relay_manifest_files(manifest)
    index_path = tree.parent / "tree-index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise RuntimeError("remote source cache tree index is missing or unsafe")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("schema_version") != 1 or index.get("snapshot_id") != snapshot_id or not isinstance(index.get("files"), list):
        raise RuntimeError("remote source cache tree index is invalid")
    indexed = {}
    for raw in index["files"]:
        if not isinstance(raw, dict):
            raise RuntimeError("remote source cache tree index entry is invalid")
        relative = relay_safe_relative(raw.get("path")).as_posix()
        if relative in indexed:
            raise RuntimeError("remote source cache tree index contains a duplicate path")
        indexed[relative] = raw
    if set(indexed) != set(files):
        raise RuntimeError("remote source cache tree index differs from the manifest")
    observed = relay_validate_tree_shape(tree, files, cache_modes=True)
    for relative, item in files.items():
        raw = indexed[relative]
        info = observed[relative]
        expected = {
            "path": relative,
            "size_bytes": int(item["size_bytes"]),
            "manifest_mode": int(item["mode"]),
            "cache_mode": relay_cache_file_mode(item),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "nlink": 1,
            "mtime_ns": int(info.st_mtime_ns),
            "ctime_ns": int(info.st_ctime_ns),
        }
        if raw != expected:
            raise RuntimeError("remote source cache tree seal mismatch")
    return observed

def relay_publish_tree_cache(cache_root, snapshot_id, manifest, staging_entry):
    files = relay_manifest_files(manifest)
    tree = staging_entry / "tree"
    relay_freeze_cache_tree(tree, files)
    relay_validate_tree_shape(tree, files, cache_modes=True)
    relay_fsync_tree(tree)
    index = relay_tree_index_payload(snapshot_id, tree, files)
    index_bytes = relay_json_bytes(index)
    relay_write_json_file(staging_entry / "manifest.json", manifest)
    relay_write_json_file(staging_entry / "tree-index.json", index)
    relay_write_json_file(staging_entry / "cache.json", {
        "schema_version": 2,
        "snapshot_id": snapshot_id,
        "manifest_sha256": snapshot_id,
        "storage": "tree",
        "file_count": len(files),
        "tree_index_size_bytes": len(index_bytes),
        "tree_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
    })
    relay_fsync_directory(staging_entry)
    v2_root = cache_root / "v2"
    target = v2_root / snapshot_id
    try:
        os.replace(staging_entry, target)
        relay_fsync_directory(v2_root)
        published = True
    except OSError:
        if not (target.exists() or target.is_symlink()):
            raise
        relay_load_cache(cache_root, snapshot_id, verify_archive=False)
        relay_discard_tree(staging_entry)
        published = False
    loaded_manifest, loaded_tree, _, schema_version = relay_load_cache(cache_root, snapshot_id, verify_archive=False)
    if schema_version != 2 or loaded_tree is None:
        raise RuntimeError("published remote source tree cache is unavailable")
    loaded_seal = relay_validate_tree_cache(loaded_tree, loaded_manifest)
    return loaded_manifest, loaded_tree, published, loaded_seal

def relay_ensure_cache_tree(cache_root, snapshot_id, run_id, metrics):
    manifest, tree, archive, schema_version = relay_load_cache(cache_root, snapshot_id, verify_archive=False)
    if schema_version == 2:
        seal = relay_validate_tree_cache(tree, manifest)
        return manifest, tree, False, seal
    if schema_version != 1 or archive is None:
        raise RuntimeError("remote source cache cannot be materialized")
    declared = manifest["source_archive"]
    archive_size, archive_sha256 = relay_digest(archive)
    metrics["archive_reads"] += 1
    metrics["archive_hashed_bytes"] += archive_size
    if archive_size != int(declared["size_bytes"]) or archive_sha256 != declared["sha256"]:
        raise RuntimeError("remote source cache archive checksum mismatch")
    v2_root = cache_root / "v2"
    staging = v2_root / (".partial-promote-" + run_id + "-" + snapshot_id)
    if staging.exists() or staging.is_symlink():
        relay_discard_tree(staging)
    staging.mkdir(mode=0o700)
    tree = staging / "tree"
    tree.mkdir(mode=0o700)
    try:
        hashed = relay_extract_archive(
            archive,
            tree,
            relay_manifest_files(manifest),
            verify_content=True,
            exact_tree=True,
        )
        metrics["archive_reads"] += 1
        metrics["content_hashed_bytes"] += hashed
        loaded_manifest, loaded_tree, _, loaded_seal = relay_publish_tree_cache(
            cache_root, snapshot_id, manifest, staging
        )
    finally:
        if staging.exists() or staging.is_symlink():
            relay_discard_tree(staging)
    return loaded_manifest, loaded_tree, True, loaded_seal

run_dir = Path(sys.argv[1])
snapshot_id = relay_snapshot_id(sys.argv[2])
base_snapshot_id = None if sys.argv[3] == "-" else relay_snapshot_id(sys.argv[3])
planned_base_snapshot_id = base_snapshot_id
try:
    requested_transfer = json.loads(sys.argv[4])
except Exception as exc:
    raise RuntimeError("source transfer plan is invalid") from exc
if not isinstance(requested_transfer, list):
    raise RuntimeError("source transfer plan must contain a path list")
requested_transfer = [relay_safe_relative(value).as_posix() for value in requested_transfer]
if len(requested_transfer) != len(set(requested_transfer)):
    raise RuntimeError("source transfer plan contains duplicate paths")
if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.parent.name != "runs":
    raise RuntimeError("unsafe remote run directory")
runs_dir = run_dir.parent
root = runs_dir.parent
if runs_dir.is_symlink() or not runs_dir.is_dir() or run_dir.resolve() != runs_dir.resolve() / run_dir.name:
    raise RuntimeError("remote run directory escapes the owned runs path")
run_marker = run_dir / ".edolab-relay-run.json"
root_marker = root / ".edolab-relay-root.json"
if run_marker.is_symlink() or not run_marker.is_file() or root_marker.is_symlink() or not root_marker.is_file():
    raise RuntimeError("remote Relay ownership marker is missing or unsafe")
run_identity = json.loads(run_marker.read_text(encoding="utf-8"))
root_identity = json.loads(root_marker.read_text(encoding="utf-8"))
if run_identity.get("run_id") != run_dir.name or root_identity.get("installation_id") != run_identity.get("installation_id"):
    raise RuntimeError("remote root/run marker ownership mismatch")
incoming = run_dir / "incoming"
meta = run_dir / "meta"
for name, child in (("incoming", incoming), ("meta", meta)):
    if child.is_symlink() or not child.is_dir() or child.resolve().parent != run_dir.resolve():
        raise RuntimeError("remote run child is unsafe: " + name)
manifest_path = incoming / "manifest.json"
delta_path = incoming / "source-delta.tar.gz"
task_path = meta / "task.jsonl"
for name, path in (("source manifest", manifest_path), ("task file", task_path)):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("remote " + name + " is missing or unsafe")
manifest = relay_validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), snapshot_id)
target_files = relay_manifest_files(manifest)
cache_root = root / ".edolab-relay-snapshots"
if cache_root.is_symlink() or not cache_root.is_dir():
    raise RuntimeError("remote source cache path is unsafe")
tree_cache_root = cache_root / "v2"
tree_cache_root.mkdir(mode=0o700, exist_ok=True)
if tree_cache_root.is_symlink() or not tree_cache_root.is_dir():
    raise RuntimeError("remote source tree-cache path is unsafe")
source = run_dir / "source"
partial = run_dir / ".source-partial"
run_manifest_path = meta / "source-manifest.json"
empty_metrics = {
    "archive_reads": 0,
    "archive_writes": 0,
    "archive_hashed_bytes": 0,
    "content_hashed_bytes": 0,
}
if source.exists():
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError("remote source path is unsafe")
    if run_manifest_path.is_file() and not run_manifest_path.is_symlink():
        existing_manifest = relay_validate_manifest(json.loads(run_manifest_path.read_text(encoding="utf-8")), snapshot_id)
        pid_path = meta / "pid"
        process_path = meta / "process.json"
        if pid_path.exists() or pid_path.is_symlink() or process_path.exists() or process_path.is_symlink():
            for name, path in (("pid", pid_path), ("process identity", process_path)):
                if (path.exists() or path.is_symlink()) and (path.is_symlink() or not path.is_file()):
                    raise RuntimeError("remote " + name + " marker is unsafe")
            print(json.dumps({
                "verified": True,
                "snapshot_id": snapshot_id,
                "strategy": "existing_run_copy",
                "cache_storage": "tree_v2",
                "cache_materialization": "existing",
                "run_materialization": "existing",
                "file_count": len(relay_manifest_files(existing_manifest)),
                "transferred_files": 0,
                **empty_metrics,
            }, sort_keys=True, separators=(",", ":")))
            raise SystemExit(0)
        # A pre-launch source copy is disposable.  Rebuild it from the sealed
        # cache on retry so run files need no expensive per-file fsync and a
        # power loss cannot make a durable manifest bless incomplete bytes.
        run_manifest_path.unlink()
        relay_discard_tree(source)
    else:
        pid_path = meta / "pid"
        process_path = meta / "process.json"
        if pid_path.exists() or pid_path.is_symlink() or process_path.exists() or process_path.is_symlink():
            raise RuntimeError("refusing to replace source after remote launch")
        relay_discard_tree(source)
if partial.exists() or partial.is_symlink():
    relay_discard_tree(partial)

strategy = "cache"
base_files = {}
cache_materialization = "existing"
run_materialization = None
legacy_promoted = False
cache_race = False
metrics = dict(empty_metrics)
transferred_file_count = len(requested_transfer)
applied_transfer_file_count = len(requested_transfer)
target_tree = None
target_cache_staging = tree_cache_root / (".partial-" + run_dir.name + "-" + snapshot_id)
if target_cache_staging.exists() or target_cache_staging.is_symlink():
    relay_discard_tree(target_cache_staging)
try:
    if relay_cache_entry_exists(cache_root, snapshot_id):
        target_manifest, target_tree, target_promoted, target_seal = relay_ensure_cache_tree(
            cache_root, snapshot_id, run_dir.name, metrics
        )
        legacy_promoted = target_promoted
        if relay_manifest_files(target_manifest) != target_files:
            raise RuntimeError("remote source cache manifest differs from the target")
        if requested_transfer:
            # Another run may publish this exact target after the current run
            # prepared its verified full/delta plan and uploaded the payload.
            # Validate that persisted plan against its original baseline, then
            # reuse the winner instead of turning a harmless race into a
            # permanently failing retry.
            if planned_base_snapshot_id is None:
                planned_base_files = {}
                strategy = "full"
            else:
                planned_base_manifest, _, base_promoted, _ = relay_ensure_cache_tree(
                    cache_root, planned_base_snapshot_id, run_dir.name, metrics
                )
                planned_base_files = relay_manifest_files(planned_base_manifest)
                legacy_promoted = legacy_promoted or base_promoted
                strategy = "delta"
            expected_race_transfer = {
                path for path, item in target_files.items()
                if path not in planned_base_files
                or item["sha256"] != planned_base_files[path]["sha256"]
                or item["size_bytes"] != planned_base_files[path]["size_bytes"]
            }
            if set(requested_transfer) != expected_race_transfer:
                raise RuntimeError("source payload file set differs from the verified cache-race plan")
            cache_race = True
            cache_materialization = "race_existing"
            applied_transfer_file_count = 0
        else:
            base_snapshot_id = snapshot_id
        requested_transfer = []
        if target_promoted and not cache_race:
            cache_materialization = "legacy_extract"
    else:
        base_tree = None
        base_seal = None
        if base_snapshot_id is not None:
            base_manifest, base_tree, promoted, base_seal = relay_ensure_cache_tree(
                cache_root, base_snapshot_id, run_dir.name, metrics
            )
            base_files = relay_manifest_files(base_manifest)
            legacy_promoted = promoted
            strategy = "delta"
        else:
            strategy = "full"
        expected_transfer = {
            path for path, item in target_files.items()
            if path not in base_files
            or item["sha256"] != base_files[path]["sha256"]
            or item["size_bytes"] != base_files[path]["size_bytes"]
        }
        if set(requested_transfer) != expected_transfer:
            raise RuntimeError("source payload file set differs from the verified manifest delta")

        target_cache_staging.mkdir(mode=0o700)
        staging_tree = target_cache_staging / "tree"
        if base_tree is None:
            staging_tree.mkdir(mode=0o700)
            cache_materialization = "extract"
        else:
            cache_materialization = relay_clone_tree(
                base_tree, staging_tree, base_files, base_seal
            )
            relay_apply_manifest_modes(staging_tree, base_files)
        removed = set(base_files) - set(target_files)
        relay_remove_paths(staging_tree, removed)
        relay_remove_paths(staging_tree, expected_transfer)
        if expected_transfer:
            if delta_path.is_symlink() or not delta_path.is_file():
                raise RuntimeError("remote source delta is missing or unsafe")
            metrics["content_hashed_bytes"] += relay_extract_archive(
                delta_path,
                staging_tree,
                {path: target_files[path] for path in expected_transfer},
                verify_content=True,
                exact_tree=False,
            )
        elif delta_path.exists() or delta_path.is_symlink():
            if delta_path.is_symlink() or not delta_path.is_file():
                raise RuntimeError("remote source delta path is unsafe")
            delta_path.unlink()
        relay_apply_manifest_modes(staging_tree, target_files)
        _, target_tree, target_published, target_seal = relay_publish_tree_cache(
            cache_root, snapshot_id, manifest, target_cache_staging
        )
        if not target_published:
            cache_race = True
            cache_materialization = "race_existing"

    run_materialization = relay_clone_tree(
        target_tree, partial, target_files, target_seal
    )
    relay_apply_manifest_modes(partial, target_files)
    os.replace(partial, source)
    relay_fsync_directory(run_dir)
    run_manifest_temporary = meta / (".source-manifest-" + run_dir.name + ".tmp")
    if run_manifest_temporary.exists() or run_manifest_temporary.is_symlink():
        if run_manifest_temporary.is_symlink() or not run_manifest_temporary.is_file():
            raise RuntimeError("remote run source manifest temporary path is unsafe")
        run_manifest_temporary.unlink()
    relay_write_json_file(run_manifest_temporary, manifest)
    os.replace(run_manifest_temporary, run_manifest_path)
    relay_fsync_directory(meta)
    last = root / ".last-source-manifest.json"
    last_temporary = root / (".last-source-manifest-" + run_dir.name + ".tmp")
    for path in (last, last_temporary):
        if (path.exists() or path.is_symlink()) and (path.is_symlink() or not path.is_file()):
            raise RuntimeError("remote source-history path is unsafe")
    if last_temporary.exists():
        last_temporary.unlink()
    relay_write_json_file(last_temporary, manifest)
    os.replace(last_temporary, last)
    relay_fsync_directory(root)
finally:
    if partial.exists() or partial.is_symlink():
        relay_discard_tree(partial)
    if target_cache_staging.exists() or target_cache_staging.is_symlink():
        relay_discard_tree(target_cache_staging)
for path in (delta_path, manifest_path):
    if path.exists() and not path.is_symlink() and path.is_file():
        path.unlink()
print(json.dumps({
    "verified": True,
    "snapshot_id": snapshot_id,
    "base_snapshot_id": base_snapshot_id,
    "strategy": strategy,
    "cache_storage": "tree_v2",
    "cache_materialization": cache_materialization,
    "run_materialization": run_materialization,
    "legacy_promoted": legacy_promoted,
    "cache_race": cache_race,
    "file_count": len(target_files),
    "transferred_files": transferred_file_count,
    "applied_transfer_files": applied_transfer_file_count,
    "reused_files": len(target_files) - applied_transfer_file_count,
    **metrics,
}, sort_keys=True, separators=(",", ":")))
'''


_PROCESS_IDENTITY_HELPERS = r'''
def relay_process_start_time(value):
    try:
        tail = Path(f"/proc/{value}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
        return tail[19]
    except Exception:
        return None

def relay_pid_exists(value):
    try:
        os.kill(value, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def relay_group_exists(value):
    try:
        os.killpg(value, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def relay_group_has_token(pgid, token):
    proc = Path("/proc")
    if not token or not proc.is_dir():
        return None
    expected = ("EDOLAB_RELAY_PROCESS_TOKEN=" + token).encode("utf-8")
    group_seen = False
    unreadable = False
    try:
        entries = tuple(proc.iterdir())
    except Exception:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            group_seen = True
            environment = (entry / "environ").read_bytes().split(b"\0")
            if expected in environment:
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError, ValueError, IndexError):
            unreadable = True
    if unreadable:
        return None
    return False if group_seen else None

def relay_process_group_state(process_identity):
    pid = int(process_identity.get("pid", -1))
    pgid = int(process_identity.get("pgid", -1))
    if pid <= 0 or pgid <= 0:
        raise RuntimeError("remote process identity is incomplete")
    recorded_start = process_identity.get("start_time")
    current_start = relay_process_start_time(pid)
    if current_start is not None:
        if recorded_start is not None and current_start != recorded_start:
            raise RuntimeError("stored remote PID has been reused by another process")
        try:
            current_pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError) as exc:
            raise RuntimeError("remote process identity cannot be confirmed") from exc
        if current_pgid != pgid:
            raise RuntimeError("stored remote PID belongs to a different process group")
        return True, True
    if recorded_start is None and relay_pid_exists(pid):
        try:
            current_pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError) as exc:
            raise RuntimeError("remote process identity cannot be confirmed") from exc
        if current_pgid != pgid:
            raise RuntimeError("stored remote PID belongs to a different process group")
        return True, True
    elif recorded_start is not None and relay_pid_exists(pid):
        raise RuntimeError("remote process exists but its start time cannot be verified")
    if not relay_group_exists(pgid):
        return True, False
    token_match = relay_group_has_token(
        pgid, str(process_identity.get("process_token") or "")
    )
    if token_match is True:
        return True, True
    if token_match is False:
        raise RuntimeError("stored remote process group has been reused by another process")
    # On systems without a readable /proc (for example macOS), retaining a
    # conservative live-but-unverified state is safer than declaring the group
    # dead and deleting its workspace.  Cancellation refuses to signal such a
    # group, while polling and cleanup continue to protect its files.
    return False, True
'''


PREFLIGHT_REMOTE_SOURCE_SCRIPT = r'''
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


def emit(payload):
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def failure(code, message, *, step=None, completed=None, error=None):
    details = {
        "ok": False,
        "code": code,
        "message": message,
        "step": step,
    }
    if completed is not None:
        details.update(
            returncode=int(completed.returncode),
            stdout=(completed.stdout or "")[-4000:],
            stderr=(completed.stderr or "")[-4000:],
        )
    if error is not None:
        details["error"] = (type(error).__name__ + ": " + str(error))[:4000]
    emit(details)
    raise SystemExit(0)


def run_step(command, *, cwd, timeout, code, message, step):
    environment = os.environ.copy()
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        failure(code, message, step=step, error=exc)
    if completed.returncode != 0:
        failure(code, message, step=step, completed=completed)
    return completed


try:
    run_dir = Path(sys.argv[1])
    conda_env = sys.argv[2]
    conda_executable_argument = sys.argv[3]
    execution_mode = sys.argv[4] if len(sys.argv) > 4 else "editable"
    if execution_mode not in {"editable", "isolated"}:
        failure("ENV_EXECUTION_MODE", "The requested execution mode is unsupported.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", conda_env):
        failure("ENV_CONDA_NAME", "The configured Conda environment name is unsafe.")
    if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.parent.name != "runs":
        failure("ENV_SOURCE_PREFLIGHT", "The marker-owned remote run directory is unsafe.")
    runs_dir = run_dir.parent
    root = runs_dir.parent
    if runs_dir.is_symlink() or not runs_dir.is_dir():
        failure("ENV_SOURCE_PREFLIGHT", "The marker-owned runs directory is unsafe.")
    if run_dir.resolve() != runs_dir.resolve() / run_dir.name:
        failure("ENV_SOURCE_PREFLIGHT", "The remote run directory escapes its owned root.")
    marker = run_dir / ".edolab-relay-run.json"
    root_marker = root / ".edolab-relay-root.json"
    if marker.is_symlink() or not marker.is_file():
        failure("ENV_SOURCE_PREFLIGHT", "The remote run marker is missing or unsafe.")
    if root_marker.is_symlink() or not root_marker.is_file():
        failure("ENV_SOURCE_PREFLIGHT", "The remote root marker is missing or unsafe.")
    run_identity = json.loads(marker.read_text(encoding="utf-8"))
    root_identity = json.loads(root_marker.read_text(encoding="utf-8"))
    if run_identity.get("run_id") != run_dir.name:
        failure("ENV_SOURCE_PREFLIGHT", "The remote run marker identity does not match.")
    if root_identity.get("installation_id") != run_identity.get("installation_id"):
        failure("ENV_SOURCE_PREFLIGHT", "The remote root/run ownership does not match.")

    # Idempotent launch retry: once a process identity exists, the hardened
    # launch script owns recovery and no environment mutation is repeated.
    process_file = run_dir / "meta" / "process.json"
    pid_file = run_dir / "meta" / "pid"
    if process_file.exists() or process_file.is_symlink() or pid_file.exists() or pid_file.is_symlink():
        emit({"ok": True, "skipped": "existing_process_identity"})
        raise SystemExit(0)

    source = run_dir / "source"
    if source.is_symlink() or not source.is_dir() or source.resolve().parent != run_dir.resolve():
        failure("ENV_SOURCE_PREFLIGHT", "The verified remote source directory is unsafe.")
    conda_candidate = Path(conda_executable_argument)
    if not conda_candidate.is_absolute():
        failure(
            "ENV_CONDA_EXECUTABLE",
            "The frozen Conda executable path is not absolute.",
        )
    try:
        conda_executable = conda_candidate.resolve(strict=True)
    except OSError as exc:
        failure(
            "ENV_CONDA_EXECUTABLE",
            "The frozen Conda executable is unavailable.",
            error=exc,
        )
    if not conda_executable.is_file() or not os.access(conda_executable, os.X_OK):
        failure(
            "ENV_CONDA_EXECUTABLE",
            "The frozen Conda executable is not an executable regular file.",
        )

    prefix = [
        str(conda_executable),
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
    ]
    run_step(
        [
            *prefix,
            "-c",
            (
                "import os,sys; "
                f"assert os.environ.get('CONDA_DEFAULT_ENV') == {conda_env!r}; "
                "assert sys.version_info[:2] == (3,11), sys.version"
            ),
        ],
        cwd=source,
        timeout=30,
        code="ENV_PYTHON_VERSION",
        message="Python 3.11 in the configured Conda environment could not be verified.",
        step="python",
    )
    if execution_mode == "editable":
        run_step(
            [
                *prefix,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                ".",
            ],
            cwd=source,
            timeout=180,
            code="ENV_EDITABLE_INSTALL",
            message=(
                "The exact uploaded snapshot could not be installed editable without "
                "dependency resolution; no remote dependencies were downloaded."
            ),
            step="editable_install",
        )
    registry = run_step(
        [*prefix, "-m", "cli", "list"],
        cwd=source,
        timeout=120,
        code="ENV_REGISTRY",
        message=(
            "The exact uploaded snapshot could not import the EDOLAB algorithm, "
            "problem, and metric registry in the configured environment."
        ),
        step="registry",
    )
    emit(
        {
            "ok": True,
            "registry_output": (registry.stdout or "")[-4000:],
            # Dependency resolution remains disabled in both modes.  SSH may
            # install the exact snapshot editable with --no-deps, while local
            # execution imports directly from its frozen source directory.
            "dependency_install": "disabled",
            "source_install": "editable-no-deps" if execution_mode == "editable" else "isolated-source",
        }
    )
except SystemExit:
    raise
except Exception as exc:
    failure(
        "ENV_SOURCE_PREFLIGHT",
        "The exact uploaded snapshot environment preflight failed unexpectedly.",
        error=exc,
    )
'''


LAUNCH_REMOTE_WORKER_SCRIPT = r'''
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import subprocess
import sys
''' + _PROCESS_IDENTITY_HELPERS + r'''
run_dir = Path(sys.argv[1])
conda_env = sys.argv[2]
max_parallel = int(sys.argv[3])
conda_executable_argument = sys.argv[4]
execution_mode = sys.argv[5] if len(sys.argv) > 5 else "editable"
worker_kind = sys.argv[6] if len(sys.argv) > 6 else "batch"
if not re.fullmatch(r"[A-Za-z0-9_.-]+", conda_env):
    raise RuntimeError("unsafe conda environment name")
if max_parallel < 1:
    raise RuntimeError("max_parallel_runs must be positive")
if execution_mode not in {"editable", "isolated"}:
    raise RuntimeError("unsupported execution mode")
if worker_kind not in {"batch", "benchmark", "plot"}:
    raise RuntimeError("unsupported worker kind")
if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.parent.name != "runs":
    raise RuntimeError("unsafe remote run directory")
runs_dir = run_dir.parent
root = runs_dir.parent
if runs_dir.is_symlink() or not runs_dir.is_dir():
    raise RuntimeError("remote runs path is unsafe")
if run_dir.resolve() != runs_dir.resolve() / run_dir.name:
    raise RuntimeError("remote run directory escapes the owned runs path")
marker = run_dir / ".edolab-relay-run.json"
if marker.is_symlink() or not marker.is_file():
    raise RuntimeError("remote run marker is missing or unsafe")
run_identity = json.loads(marker.read_text(encoding="utf-8"))
if run_identity.get("run_id") != run_dir.name:
    raise RuntimeError("remote run marker identity mismatch")
root_marker = root / ".edolab-relay-root.json"
if root_marker.is_symlink() or not root_marker.is_file():
    raise RuntimeError("remote root marker is missing or unsafe")
root_identity = json.loads(root_marker.read_text(encoding="utf-8"))
if root_identity.get("installation_id") != run_identity.get("installation_id"):
    raise RuntimeError("remote root/run marker ownership mismatch")
source = run_dir / "source"
task = run_dir / "meta" / "task.jsonl"
output = run_dir / "results" / "bundle"
phase = run_dir / "meta" / "phase"
exit_file = run_dir / "meta" / "exit_code"
finished = run_dir / "meta" / "finished_at"
cancel_requested = run_dir / "meta" / "cancel-requested"
launch = run_dir / "meta" / "launch.sh"
pid_file = run_dir / "meta" / "pid"
process_file = run_dir / "meta" / "process.json"
launch_gate = run_dir / "meta" / "launch.ready"
process_temporary = process_file.with_suffix(".json.tmp")
pid_temporary = pid_file.with_suffix(".tmp")
gate_temporary = launch_gate.with_suffix(".tmp")
for name in ("logs", "meta", "results"):
    child = run_dir / name
    if child.is_symlink() or not child.is_dir() or child.resolve().parent != run_dir.resolve():
        raise RuntimeError("remote run child is unsafe: " + name)
if source.is_symlink() or not source.is_dir() or source.resolve().parent != run_dir.resolve():
    raise RuntimeError("remote source directory is unsafe")
if task.is_symlink() or not task.is_file():
    raise RuntimeError("remote task file is missing or unsafe")
if output.exists() and (output.is_symlink() or not output.is_dir()):
    raise RuntimeError("remote result bundle path is unsafe")
for path in (
    phase,
    exit_file,
    finished,
    cancel_requested,
    launch,
    pid_file,
    process_file,
    launch_gate,
    process_temporary,
    pid_temporary,
    gate_temporary,
    run_dir / "logs" / "stdout.log",
    run_dir / "logs" / "stderr.log",
):
    if path.is_symlink():
        raise RuntimeError("remote worker state path is a symlink: " + path.name)
    if path.exists() and not path.is_file():
        raise RuntimeError("remote worker state path is not a regular file: " + path.name)

if process_file.exists() or process_file.is_symlink() or pid_file.exists() or pid_file.is_symlink():
    if process_file.is_symlink() or not process_file.is_file():
        raise RuntimeError("existing remote process identity is incomplete or unsafe")
    process_identity = json.loads(process_file.read_text(encoding="utf-8"))
    existing_pid = int(process_identity.get("pid", -1))
    if pid_file.exists():
        if pid_file.is_symlink() or not pid_file.is_file():
            raise RuntimeError("existing pid file is unsafe")
        if int(pid_file.read_text(encoding="ascii").strip()) != existing_pid:
            raise RuntimeError("existing remote process identity does not match pid file")
    elif existing_pid > 0:
        pid_file.write_text(str(existing_pid) + "\n", encoding="ascii")
    if existing_pid <= 0:
        raise RuntimeError("existing remote process identity does not match pid file")
    identity_valid, process_alive = relay_process_group_state(process_identity)
    if process_alive and not identity_valid:
        raise RuntimeError(
            "existing remote process group is live but its identity cannot be confirmed"
        )
    process_token = str(process_identity.get("process_token") or "")
    if process_alive and process_token:
        if launch_gate.exists():
            if launch_gate.read_text(encoding="ascii").strip() != process_token:
                raise RuntimeError("remote launch gate does not match process identity")
        else:
            gate_temporary.write_text(process_token + "\n", encoding="ascii")
            os.replace(gate_temporary, launch_gate)
    if not process_alive and not exit_file.is_file():
        raise RuntimeError(
            "stored remote worker is no longer alive and did not write an exit status"
        )
    print(json.dumps({"pid": existing_pid, "recovered": True}, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0)
for stale in (launch_gate, process_temporary, pid_temporary, gate_temporary):
    if stale.exists():
        stale.unlink()
conda_candidate = Path(conda_executable_argument)
if not conda_candidate.is_absolute():
    raise RuntimeError(
        "EDOLAB_ERROR ENV_CONDA_EXECUTABLE: the probed Conda executable path is not absolute"
    )
try:
    conda_executable = conda_candidate.resolve(strict=True)
except OSError as exc:
    raise RuntimeError(
        "EDOLAB_ERROR ENV_CONDA_EXECUTABLE: the probed Conda executable is unavailable"
    ) from exc
if not conda_executable.is_file() or not os.access(conda_executable, os.X_OK):
    raise RuntimeError(
        "EDOLAB_ERROR ENV_CONDA_EXECUTABLE: the probed Conda executable is not an executable file"
    )
conda_info = subprocess.run(
    [str(conda_executable), "info", "--base"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=12,
    check=False,
)
if conda_info.returncode != 0 or not conda_info.stdout.strip():
    detail = (conda_info.stderr.strip() or conda_info.stdout.strip())[:1000]
    raise RuntimeError(
        "EDOLAB_ERROR ENV_CONDA_INFO: the probed Conda executable could not report its base"
        + (": " + detail if detail else "")
    )
conda_base_candidate = Path(conda_info.stdout.strip()).expanduser()
if not conda_base_candidate.is_absolute():
    raise RuntimeError(
        "EDOLAB_ERROR ENV_CONDA_INFO: Conda reported a non-absolute base directory"
    )
try:
    conda_base = conda_base_candidate.resolve(strict=True)
except OSError as exc:
    raise RuntimeError(
        "EDOLAB_ERROR ENV_CONDA_INFO: the reported Conda base directory is unavailable"
    ) from exc
conda_hook = conda_base / "etc" / "profile.d" / "conda.sh"
if not conda_hook.is_file():
    raise RuntimeError(
        "EDOLAB_ERROR ENV_CONDA_HOOK: the reported Conda base does not provide etc/profile.d/conda.sh"
    )
q = shlex.quote
python_check = (
    "import os,sys; "
    f"assert os.environ.get('CONDA_DEFAULT_ENV') == {conda_env!r}; "
    "assert sys.version_info[:2] == (3,11), sys.version"
)
editable_install = (
    "python -m pip install --no-deps --no-build-isolation -e . || "
    "{ echo 'EDOLAB_ERROR ENV_EDITABLE_INSTALL: editable install failed' >&2; exit 44; }"
    if execution_mode == "editable"
    else (
        "export PYTHONPATH=" + q(str(source)) + "${PYTHONPATH:+:$PYTHONPATH}\n"
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import core\n"
        "source = Path(" + repr(str(source)) + ").resolve()\n"
        "module = Path(core.__file__).resolve()\n"
        "if not module.is_relative_to(source):\n"
        "    raise SystemExit(f'core imported outside frozen source: {module}')\n"
        "PY"
    )
)
if worker_kind == "batch":
    worker_command = (
        "python -u -m apps.remote.worker --task-file " + q(str(task))
        + " --output-dir " + q(str(output))
        + " --max-parallel-runs " + str(max_parallel)
    )
else:
    worker_module = (
        "apps.remote.benchmark_worker"
        if worker_kind == "benchmark"
        else "apps.remote.plot_worker"
    )
    worker_command = (
        "python -u -m " + worker_module + " --spec-file " + q(str(task))
        + " --output-dir " + q(str(output))
    )
script = f"""#!/usr/bin/env bash
set -Eeuo pipefail
umask 0077
phase_file={q(str(phase))}
exit_file={q(str(exit_file))}
finished_file={q(str(finished))}
cancel_file={q(str(cancel_requested))}
set_phase() {{ printf '%s\\n' "$1" > "${{phase_file}}.tmp"; mv "${{phase_file}}.tmp" "$phase_file"; }}
on_exit() {{ code=$?; canceled=0; if [ -f "$cancel_file" ] && [ ! -L "$cancel_file" ]; then canceled=1; if [ "$code" -eq 0 ]; then code=130; fi; fi; printf '%s\\n' "$code" > "${{exit_file}}.tmp"; mv "${{exit_file}}.tmp" "$exit_file"; date -u +%Y-%m-%dT%H:%M:%SZ > "${{finished_file}}.tmp"; mv "${{finished_file}}.tmp" "$finished_file"; if [ "$canceled" -eq 1 ]; then set_phase canceled; elif [ "$code" -eq 0 ]; then set_phase finished; else set_phase failed; fi; }}
trap on_exit EXIT
set_phase preparing
source {q(str(conda_hook))}
conda activate {q(conda_env)} || {{ echo 'EDOLAB_ERROR ENV_CONDA_ACTIVATE: cannot activate configured environment' >&2; exit 42; }}
python -c {q(python_check)} || {{ echo 'EDOLAB_ERROR ENV_PYTHON_VERSION: Python 3.11 is required' >&2; exit 43; }}
cd {q(str(source))}
{editable_install}
python -m cli list || {{ echo 'EDOLAB_ERROR ENV_REGISTRY: registry import failed' >&2; exit 45; }}
python -m pip freeze > {q(str(run_dir / 'meta' / 'pip-freeze.txt'))}
python - <<'PY' > {q(str(run_dir / 'meta' / 'python.json'))}
import json, os, platform, sys
print(json.dumps({{"executable": sys.executable, "version": platform.python_version(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV")}}, sort_keys=True))
PY
set_phase running
{worker_command}
"""
launch.write_text(script, encoding="utf-8")
os.chmod(launch, 0o700)
stdout = (run_dir / "logs" / "stdout.log").open("ab", buffering=0)
stderr = (run_dir / "logs" / "stderr.log").open("ab", buffering=0)
process_token = secrets.token_hex(32)
environment = os.environ.copy()
environment["EDOLAB_RELAY_PROCESS_TOKEN"] = process_token
environment["EDOLAB_RELAY_RUN_DIR"] = str(run_dir)
gate_command = (
    "while [ \"$(cat " + q(str(launch_gate)) + " 2>/dev/null || true)\" != "
    + q(process_token)
    + " ]; do sleep 0.05; done; exec bash "
    + q(str(launch))
)
process = subprocess.Popen(
    ["bash", "-c", gate_command],
    cwd=str(run_dir),
    stdin=subprocess.DEVNULL,
    stdout=stdout,
    stderr=stderr,
    start_new_session=True,
    close_fds=True,
    env=environment,
)
process_identity = {
    "pid": process.pid,
    "pgid": os.getpgid(process.pid),
    "start_time": relay_process_start_time(process.pid),
    "process_token": process_token,
}
try:
    process_temporary.write_text(json.dumps(process_identity, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(process_temporary, process_file)
    pid_temporary.write_text(str(process.pid) + "\n", encoding="ascii")
    os.replace(pid_temporary, pid_file)
    gate_temporary.write_text(process_token + "\n", encoding="ascii")
    os.replace(gate_temporary, launch_gate)
except BaseException:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except Exception:
        pass
    for failed in (process_temporary, process_file, pid_temporary, pid_file, gate_temporary, launch_gate):
        try:
            failed.unlink()
        except FileNotFoundError:
            pass
    raise
print(json.dumps({"pid": process.pid}, sort_keys=True, separators=(",", ":")))
'''


DISCOVER_REMOTE_WORKER_SCRIPT = r'''
import json
import os
from pathlib import Path
import sys
''' + _PROCESS_IDENTITY_HELPERS + r'''
run_dir = Path(sys.argv[1])
if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.parent.name != "runs":
    raise RuntimeError("unsafe remote run directory")
runs_dir = run_dir.parent
root = runs_dir.parent
if runs_dir.is_symlink() or not runs_dir.is_dir():
    raise RuntimeError("remote runs path is unsafe")
if run_dir.resolve() != runs_dir.resolve() / run_dir.name:
    raise RuntimeError("remote run directory escapes the owned runs path")
marker = run_dir / ".edolab-relay-run.json"
root_marker = root / ".edolab-relay-root.json"
if marker.is_symlink() or not marker.is_file():
    raise RuntimeError("remote run marker is missing or unsafe")
if root_marker.is_symlink() or not root_marker.is_file():
    raise RuntimeError("remote root marker is missing or unsafe")
run_identity = json.loads(marker.read_text(encoding="utf-8"))
root_identity = json.loads(root_marker.read_text(encoding="utf-8"))
if run_identity.get("run_id") != run_dir.name:
    raise RuntimeError("remote run marker identity mismatch")
if root_identity.get("installation_id") != run_identity.get("installation_id"):
    raise RuntimeError("remote root/run marker ownership mismatch")
meta = run_dir / "meta"
if meta.is_symlink() or not meta.is_dir() or meta.resolve().parent != run_dir.resolve():
    raise RuntimeError("remote metadata directory is unsafe")
pid_path = meta / "pid"
process_path = meta / "process.json"
if pid_path.is_symlink() or process_path.is_symlink() or not pid_path.is_file() or not process_path.is_file():
    raise RuntimeError("remote process identity is missing or unsafe")
pid = int(pid_path.read_text(encoding="ascii").strip())
process_identity = json.loads(process_path.read_text(encoding="utf-8"))
if pid <= 0 or int(process_identity.get("pid", -1)) != pid:
    raise RuntimeError("remote process identity does not match PID")

identity_valid, process_alive = relay_process_group_state(process_identity)
print(json.dumps({"pid": pid, "alive": process_alive, "process_identity_valid": identity_valid}, sort_keys=True, separators=(",", ":")))
'''


POLL_REMOTE_WORKER_SCRIPT = r'''
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
''' + _PROCESS_IDENTITY_HELPERS + r'''
run_dir = Path(sys.argv[1])
pid = int(sys.argv[2])
stdout_offset = max(0, int(sys.argv[3]))
stderr_offset = max(0, int(sys.argv[4]))
if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.parent.name != "runs":
    raise RuntimeError("unsafe remote run directory")
runs_dir = run_dir.parent
root = runs_dir.parent
if runs_dir.is_symlink() or not runs_dir.is_dir():
    raise RuntimeError("remote runs path is unsafe")
if run_dir.resolve() != runs_dir.resolve() / run_dir.name:
    raise RuntimeError("remote run directory escapes the owned runs path")
marker = run_dir / ".edolab-relay-run.json"
if marker.is_symlink() or not marker.is_file():
    raise RuntimeError("remote run marker is missing or unsafe")
run_identity = json.loads(marker.read_text(encoding="utf-8"))
if run_identity.get("run_id") != run_dir.name:
    raise RuntimeError("remote run marker identity mismatch")
root_marker = root / ".edolab-relay-root.json"
if root_marker.is_symlink() or not root_marker.is_file():
    raise RuntimeError("remote root marker is missing or unsafe")
root_identity = json.loads(root_marker.read_text(encoding="utf-8"))
if root_identity.get("installation_id") != run_identity.get("installation_id"):
    raise RuntimeError("remote root/run marker ownership mismatch")
meta = run_dir / "meta"
logs = run_dir / "logs"
for name, child in (("meta", meta), ("logs", logs)):
    if child.is_symlink() or not child.is_dir() or child.resolve().parent != run_dir.resolve():
        raise RuntimeError("remote run child is unsafe: " + name)
pid_path = meta / "pid"
process_path = meta / "process.json"
if pid_path.is_symlink() or process_path.is_symlink() or not pid_path.is_file() or not process_path.is_file():
    raise RuntimeError("remote process identity is missing or unsafe")
stored_pid = int(pid_path.read_text().strip())
if stored_pid != pid:
    raise RuntimeError("remote PID does not belong to this run")
process_identity = json.loads(process_path.read_text(encoding="utf-8"))
if int(process_identity.get("pid", -1)) != pid:
    raise RuntimeError("remote process identity does not match PID")

identity_valid, process_alive = relay_process_group_state(process_identity)

def chunk(path, offset, limit=262144):
    if not path.exists():
        return b"", 0, False
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("unsafe remote log path")
    size = path.stat().st_size
    start = min(offset, size)
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(limit)
    next_offset = start + len(data)
    return data, next_offset, next_offset < size

def cpu_sample():
    def read():
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = [int(item) for item in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    try:
        first = read(); time.sleep(0.10); second = read()
        total = second[0] - first[0]; idle = second[1] - first[1]
        return 100.0 * (total - idle) / total if total > 0 else None
    except Exception:
        return None

def memory():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1); values[key] = int(value.strip().split()[0]) * 1024
    except Exception:
        pass
    return {"total_bytes": values.get("MemTotal"), "available_bytes": values.get("MemAvailable", values.get("MemFree"))}

def gpu():
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=False)
        if completed.returncode:
            return []
        devices = []
        for line in completed.stdout.splitlines():
            values = [item.strip() for item in line.split(",")]
            if len(values) == 6:
                devices.append({"index": int(values[0]), "name": values[1], "memory_total_mb": int(values[2]), "memory_used_mb": int(values[3]), "utilization_percent": int(values[4]), "temperature_c": int(values[5])})
        return devices
    except Exception:
        return []

stdout, stdout_next, stdout_more = chunk(logs / "stdout.log", stdout_offset)
stderr, stderr_next, stderr_more = chunk(logs / "stderr.log", stderr_offset)
phase_path = meta / "phase"
exit_path = meta / "exit_code"
for name, path in (("phase", phase_path), ("exit code", exit_path)):
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise RuntimeError("remote " + name + " path is unsafe")
logical = os.cpu_count()
try:
    loads = os.getloadavg()
except Exception:
    loads = (None, None, None)
resource = {"cpu_logical_cores": logical, "cpu_percent": cpu_sample(), "load_1m": loads[0], "load_5m": loads[1], "load_15m": loads[2], "load_ratio": (loads[0] / logical) if loads[0] is not None and logical else None, "memory": memory(), "gpus": gpu()}
print(json.dumps({"alive": process_alive, "process_identity_valid": identity_valid, "phase": phase_path.read_text().strip() if phase_path.exists() and not phase_path.is_symlink() else "starting", "exit_code": int(exit_path.read_text().strip()) if exit_path.exists() and not exit_path.is_symlink() else None, "stdout_b64": base64.b64encode(stdout).decode("ascii"), "stderr_b64": base64.b64encode(stderr).decode("ascii"), "stdout_offset": stdout_next, "stderr_offset": stderr_next, "stdout_more": stdout_more, "stderr_more": stderr_more, "resource": resource}, sort_keys=True, separators=(",", ":")))
'''


CANCEL_REMOTE_WORKER_SCRIPT = r'''
import json
import os
from pathlib import Path
import signal
import stat
import sys
''' + _PROCESS_IDENTITY_HELPERS + r'''
run_dir = Path(sys.argv[1])
pid = int(sys.argv[2])
signal_name = sys.argv[3]
if signal_name not in {"TERM", "KILL"}:
    raise RuntimeError("unsupported cancellation signal")
if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.parent.name != "runs":
    raise RuntimeError("unsafe remote run directory")
runs_dir = run_dir.parent
root = runs_dir.parent
if runs_dir.is_symlink() or not runs_dir.is_dir():
    raise RuntimeError("remote runs path is unsafe")
if run_dir.resolve() != runs_dir.resolve() / run_dir.name:
    raise RuntimeError("remote run directory escapes the owned runs path")
marker = run_dir / ".edolab-relay-run.json"
if marker.is_symlink() or not marker.is_file():
    raise RuntimeError("remote run marker is missing or unsafe")
run_identity = json.loads(marker.read_text(encoding="utf-8"))
if run_identity.get("run_id") != run_dir.name:
    raise RuntimeError("remote run marker identity mismatch")
root_marker = root / ".edolab-relay-root.json"
if root_marker.is_symlink() or not root_marker.is_file():
    raise RuntimeError("remote root marker is missing or unsafe")
root_identity = json.loads(root_marker.read_text(encoding="utf-8"))
if root_identity.get("installation_id") != run_identity.get("installation_id"):
    raise RuntimeError("remote root/run marker ownership mismatch")
meta = run_dir / "meta"
if meta.is_symlink() or not meta.is_dir() or meta.resolve().parent != run_dir.resolve():
    raise RuntimeError("remote metadata directory is unsafe")
pid_path = meta / "pid"
process_path = meta / "process.json"
if pid_path.is_symlink() or process_path.is_symlink() or not pid_path.is_file() or not process_path.is_file():
    raise RuntimeError("remote process identity is missing or unsafe")
if int(pid_path.read_text().strip()) != pid:
    raise RuntimeError("remote PID does not belong to this run")
process_identity = json.loads(process_path.read_text(encoding="utf-8"))
if int(process_identity.get("pid", -1)) != pid:
    raise RuntimeError("remote process identity does not match PID")

identity_valid, process_exists = relay_process_group_state(process_identity)
if process_exists and not identity_valid:
    raise RuntimeError("refusing to signal an unverified remote process group")
cancel_requested = meta / "cancel-requested"
if cancel_requested.is_symlink() or (
    cancel_requested.exists() and not cancel_requested.is_file()
):
    raise RuntimeError("remote cancellation marker is unsafe")
created = False
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_CLOEXEC"):
    flags |= os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
try:
    descriptor = os.open(cancel_requested, flags, 0o600)
    created = True
except FileExistsError:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(cancel_requested, flags)
try:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise RuntimeError("remote cancellation marker is unsafe")
    if created:
        os.write(descriptor, b"cancel-requested\n")
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory_flags = os.O_RDONLY
if hasattr(os, "O_CLOEXEC"):
    directory_flags |= os.O_CLOEXEC
if hasattr(os, "O_DIRECTORY"):
    directory_flags |= os.O_DIRECTORY
directory_descriptor = os.open(meta, directory_flags)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
chosen = signal.SIGKILL if signal_name == "KILL" else signal.SIGTERM
if process_exists:
    try:
        os.killpg(int(process_identity["pgid"]), chosen)
        sent = True
    except ProcessLookupError:
        sent = False
else:
    sent = False
print(json.dumps({"sent": sent, "signal": signal_name}, sort_keys=True, separators=(",", ":")))
'''


BUILD_REMOTE_ARTIFACT_SCRIPT = r'''
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tarfile

run_dir = Path(sys.argv[1])
worker_kind = sys.argv[2] if len(sys.argv) > 2 else "batch"
if worker_kind not in {"batch", "benchmark", "plot"}:
    raise RuntimeError("unsupported worker kind")
if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.parent.name != "runs":
    raise RuntimeError("unsafe remote run directory")
root = run_dir.parent.parent
if run_dir.parent.is_symlink() or not run_dir.parent.is_dir():
    raise RuntimeError("remote runs path is unsafe")
if run_dir.resolve() != run_dir.parent.resolve() / run_dir.name:
    raise RuntimeError("remote run directory escapes the owned runs path")
marker = run_dir / ".edolab-relay-run.json"
if marker.is_symlink() or not marker.is_file():
    raise RuntimeError("remote run marker is missing or unsafe")
run_identity = json.loads(marker.read_text(encoding="utf-8"))
if run_identity.get("run_id") != run_dir.name:
    raise RuntimeError("remote run marker identity mismatch")
root_marker = root / ".edolab-relay-root.json"
if root_marker.is_symlink() or not root_marker.is_file():
    raise RuntimeError("remote root marker is missing or unsafe")
root_identity = json.loads(root_marker.read_text(encoding="utf-8"))
if root_identity.get("installation_id") != run_identity.get("installation_id"):
    raise RuntimeError("remote root/run marker ownership mismatch")
for name in ("logs", "meta", "results"):
    child = run_dir / name
    if child.is_symlink() or not child.is_dir() or child.resolve().parent != run_dir.resolve():
        raise RuntimeError("remote artifact directory is unsafe: " + name)
meta = run_dir / "meta"
exit_path = meta / "exit_code"
if (exit_path.exists() or exit_path.is_symlink()) and (exit_path.is_symlink() or not exit_path.is_file()):
    raise RuntimeError("remote exit-code path is unsafe")
cancel_path = meta / "cancel-requested"
if (cancel_path.exists() or cancel_path.is_symlink()) and (cancel_path.is_symlink() or not cancel_path.is_file()):
    raise RuntimeError("remote cancellation marker is unsafe")
cancel_requested = cancel_path.is_file()
exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
if cancel_requested and exit_code in (None, 0):
    exit_code = 130
manifest_path = meta / "artifact-manifest.json"
manifest_temporary = meta / ".artifact-manifest.json.tmp"
archive = meta / "artifact.tar.gz"
partial = meta / ".artifact.tar.gz.part"
for temporary in (partial, manifest_temporary):
    if temporary.exists() or temporary.is_symlink():
        if temporary.is_symlink() or not temporary.is_file():
            raise RuntimeError("remote artifact temporary path is unsafe: " + temporary.name)
        temporary.unlink()
for existing in (manifest_path, archive):
    if (existing.exists() or existing.is_symlink()) and (existing.is_symlink() or not existing.is_file()):
        raise RuntimeError("remote artifact output path is unsafe: " + existing.name)

def digest(path):
    value = hashlib.sha256(); size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk: break
            value.update(chunk); size += len(chunk)
    return size, value.hexdigest()

validation = {"required": exit_code == 0, "ok": False, "errors": [], "worker_kind": worker_kind}
bundle = run_dir / "results" / "bundle"
if worker_kind == "batch":
    summary = bundle / "summary.sqlite"
    trend = bundle / "trend.sqlite"
    if summary.is_file() and trend.is_file():
        try:
            metas = []
            for path in (summary, trend):
                if path.is_symlink(): raise RuntimeError(path.name + " is a symlink")
                uri = path.resolve().as_uri() + "?mode=ro"
                connection = sqlite3.connect(uri, uri=True)
                try:
                    check = [row[0] for row in connection.execute("pragma quick_check")]
                    if check != ["ok"]: raise RuntimeError(path.name + " quick_check: " + "; ".join(check))
                    metas.append({key: json.loads(value) for key, value in connection.execute("select key, value from meta")})
                finally:
                    connection.close()
            if metas[0].get("bundle_id") != metas[1].get("bundle_id"): raise RuntimeError("bundle IDs differ")
            if exit_code == 0 and metas[0].get("status") != "completed": raise RuntimeError("successful worker produced a non-completed bundle")
            validation.update(ok=True, bundle_id=metas[0].get("bundle_id"), status=metas[0].get("status"))
        except Exception as exc:
            validation["errors"].append(str(exc))
    elif exit_code == 0:
        validation["errors"].append("successful worker did not produce both SQLite files")
    else:
        validation["errors"].append("partial or missing result bundle after worker failure")
else:
    result_path = bundle / "workload-result.json"
    worker_status_path = bundle / "worker-status.json"
    scientific_manifest_path = bundle / "artifact-manifest.json"
    task_path = run_dir / "meta" / "task.jsonl"
    try:
        for path in (result_path, worker_status_path, scientific_manifest_path, task_path):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("workload file is missing or unsafe: " + path.name)
        workload_result = json.loads(result_path.read_text(encoding="utf-8"))
        worker_status = json.loads(worker_status_path.read_text(encoding="utf-8"))
        scientific_manifest = json.loads(scientific_manifest_path.read_text(encoding="utf-8"))
        expected_spec_sha = digest(task_path)[1]
        if workload_result.get("schema_version") != 1: raise RuntimeError("unsupported workload result schema")
        if workload_result.get("status") != "completed": raise RuntimeError("workload result is not completed")
        if workload_result.get("workload_kind") != worker_kind: raise RuntimeError("workload result kind mismatch")
        allowed_operations = {"contour", "surface", "nbn_views", "nbn_model"} if worker_kind == "benchmark" else {"trend", "boxplot"}
        if workload_result.get("operation") not in allowed_operations: raise RuntimeError("workload operation is invalid")
        if workload_result.get("spec_sha256") != expected_spec_sha: raise RuntimeError("workload spec checksum mismatch")
        if worker_status.get("event") != "worker_completed" or worker_status.get("status") != "completed": raise RuntimeError("worker did not record completion")
        if worker_status.get("spec_sha256") != expected_spec_sha: raise RuntimeError("worker spec checksum mismatch")
        if scientific_manifest.get("spec_sha256") != expected_spec_sha: raise RuntimeError("scientific manifest spec checksum mismatch")
        declared = workload_result.get("artifacts")
        if not isinstance(declared, list) or not declared: raise RuntimeError("workload result has no declared artifacts")
        if scientific_manifest.get("files") != declared: raise RuntimeError("scientific artifact manifests differ")
        for item in declared:
            if not isinstance(item, dict): raise RuntimeError("invalid scientific artifact descriptor")
            relative = Path(str(item.get("path") or ""))
            if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise RuntimeError("unsafe scientific artifact path")
            path = bundle.joinpath(*relative.parts)
            if path.is_symlink() or not path.is_file() or path.resolve().parent == run_dir.resolve():
                raise RuntimeError("scientific artifact is missing or unsafe: " + relative.as_posix())
            if not path.resolve().is_relative_to(bundle.resolve()):
                raise RuntimeError("scientific artifact escapes result bundle")
            size, value = digest(path)
            if size != int(item.get("size_bytes", -1)) or value != item.get("sha256"):
                raise RuntimeError("scientific artifact checksum mismatch: " + relative.as_posix())
        validation.update(ok=True, status="completed", operation=workload_result.get("operation"), spec_sha256=expected_spec_sha)
    except Exception as exc:
        validation["errors"].append(str(exc))
if validation["required"] and not validation["ok"]:
    raise RuntimeError("result bundle validation failed: " + "; ".join(validation["errors"]))

files = []
excluded = {manifest_path, manifest_temporary, archive, partial}
for name in ("results", "logs", "meta"):
    root = run_dir / name
    if not root.exists(): continue
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("refusing to archive a symlink: " + str(path))
        if path.is_file() and path not in excluded:
            size, value = digest(path)
            files.append({"path": path.relative_to(run_dir).as_posix(), "size_bytes": size, "sha256": value})
manifest = {"schema_version": 1, "worker_kind": worker_kind, "exit_code": exit_code, "cancel_requested": cancel_requested, "bundle_validation": validation, "files": files}
manifest_temporary.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
os.chmod(manifest_temporary, 0o600)
os.replace(manifest_temporary, manifest_path)
with tarfile.open(partial, "x:gz") as handle:
    for item in files:
        handle.add(run_dir / item["path"], arcname=item["path"], recursive=False)
    handle.add(manifest_path, arcname="meta/artifact-manifest.json", recursive=False)
os.chmod(partial, 0o600)
os.replace(partial, archive)
size, value = digest(archive)
print(json.dumps({"archive_path": str(archive), "size_bytes": size, "sha256": value, "exit_code": exit_code, "bundle_validation": validation, "manifest": manifest}, sort_keys=True, separators=(",", ":")))
'''


CLEANUP_REMOTE_WORKSPACE_SCRIPT = r'''
import json
import os
from pathlib import Path
import re
import shutil
import sys
''' + _PROCESS_IDENTITY_HELPERS + r'''
requested_root = Path(sys.argv[1]).expanduser()
if requested_root.is_symlink():
    raise RuntimeError("remote root must not be a symlink")
root = requested_root.resolve()
installation_id = sys.argv[2]
run_id = sys.argv[3]
if root == Path("/") or root == Path.home().resolve():
    raise RuntimeError("remote root is too broad")
if not re.fullmatch(r"[A-Fa-f0-9]{16,64}", run_id):
    raise RuntimeError("invalid run id")
root_marker = root / ".edolab-relay-root.json"
expected_root = {"schema_version": 1, "installation_id": installation_id}
if root_marker.is_symlink() or not root_marker.is_file() or json.loads(root_marker.read_text(encoding="utf-8")) != expected_root:
    raise RuntimeError("remote root marker mismatch")
runs_dir = root / "runs"
if runs_dir.exists() and (runs_dir.is_symlink() or not runs_dir.is_dir()):
    raise RuntimeError("remote runs path is unsafe")
run_dir = runs_dir / run_id
if runs_dir.exists() and run_dir.resolve(strict=False).parent != runs_dir.resolve():
    raise RuntimeError("remote run path escapes the owned runs directory")
expected_run = {"schema_version": 1, "installation_id": installation_id, "run_id": run_id}
run_marker = run_dir / ".edolab-relay-run.json"
cleanup_dir = root / ".edolab-relay-cleanup"
if cleanup_dir.exists() and (cleanup_dir.is_symlink() or not cleanup_dir.is_dir()):
    raise RuntimeError("remote cleanup state path is unsafe")
cleanup_dir.mkdir(mode=0o700, exist_ok=True)
intent = cleanup_dir / (run_id + ".json")
deleting = cleanup_dir / (run_id + ".deleting")
expected_intent = {**expected_run, "status": "deleting"}
if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
    raise RuntimeError("remote Python cannot provide symlink-safe recursive cleanup")

if intent.exists():
    if intent.is_symlink() or not intent.is_file():
        raise RuntimeError("remote cleanup intent is unsafe")
    intent_payload = json.loads(intent.read_text(encoding="utf-8"))
    if {key: intent_payload.get(key) for key in expected_intent} not in (expected_intent, {**expected_run, "status": "deleted"}):
        raise RuntimeError("remote cleanup intent mismatch")

if not run_dir.exists():
    if deleting.exists():
        if deleting.is_symlink() or not deleting.is_dir() or not intent.exists():
            raise RuntimeError("remote deleting path is unsafe")
        if deleting.resolve().parent != cleanup_dir.resolve():
            raise RuntimeError("remote deleting path escapes cleanup state")
        shutil.rmtree(deleting)
    deleted_payload = {**expected_run, "status": "deleted"}
    temporary = cleanup_dir / (run_id + ".tmp")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise RuntimeError("remote cleanup temporary path is unsafe")
    temporary.write_text(json.dumps(deleted_payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, intent)
    print(json.dumps({"deleted": True, "already_absent": True, "run_dir": str(run_dir)}, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0)

if run_dir.is_symlink() or not run_dir.is_dir():
    raise RuntimeError("remote run path is unsafe")
if run_marker.is_symlink() or not run_marker.is_file() or json.loads(run_marker.read_text(encoding="utf-8")) != expected_run:
    raise RuntimeError("remote run marker mismatch")
meta = run_dir / "meta"
if meta.is_symlink() or not meta.is_dir() or meta.resolve().parent != run_dir.resolve():
    raise RuntimeError("remote metadata directory is unsafe")
pid_path = meta / "pid"
process_path = meta / "process.json"
pid_present = pid_path.exists() or pid_path.is_symlink()
process_present = process_path.exists() or process_path.is_symlink()
if pid_present != process_present:
    raise RuntimeError("remote process identity is incomplete; cleanup must be retried")
if pid_present:
    if pid_path.is_symlink() or process_path.is_symlink() or not pid_path.is_file() or not process_path.is_file():
        raise RuntimeError("remote process identity is unsafe")
    pid = int(pid_path.read_text(encoding="ascii").strip())
    process_identity = json.loads(process_path.read_text(encoding="utf-8"))
    if pid <= 0 or int(process_identity.get("pid", -1)) != pid:
        raise RuntimeError("remote process identity does not match PID")
    _, process_alive = relay_process_group_state(process_identity)
    if process_alive:
        raise RuntimeError("refusing to cleanup a remote workspace with a live worker process group")
if deleting.exists():
    raise RuntimeError("remote deleting path already exists while run directory is present")
if not intent.exists():
    descriptor = os.open(intent, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(expected_intent, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
os.replace(run_dir, deleting)
if deleting.is_symlink() or not deleting.is_dir() or deleting.resolve().parent != cleanup_dir.resolve():
    raise RuntimeError("renamed remote deletion target is unsafe")
shutil.rmtree(deleting)
deleted_payload = {**expected_run, "status": "deleted"}
temporary = cleanup_dir / (run_id + ".tmp")
if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
    raise RuntimeError("remote cleanup temporary path is unsafe")
temporary.write_text(json.dumps(deleted_payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
os.replace(temporary, intent)
print(json.dumps({"deleted": not run_dir.exists() and not deleting.exists(), "already_absent": False, "run_dir": str(run_dir)}, sort_keys=True, separators=(",", ":")))
'''


__all__ = [
    "DISCOVER_REMOTE_WORKER_SCRIPT",
    "RemoteArtifact",
    "RemotePoll",
    "RemoteWorkspace",
    "build_remote_artifact",
    "cancel_remote_worker",
    "cleanup_remote_workspace",
    "discover_remote_worker",
    "extract_verified_artifact",
    "launch_remote_worker",
    "poll_remote_worker",
    "prepare_remote_workspace",
    "upload_and_verify_source",
]
