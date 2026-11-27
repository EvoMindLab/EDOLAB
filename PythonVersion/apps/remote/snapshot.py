"""Content-addressed runtime snapshots for reproducible remote execution."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
from typing import Any, Mapping, Sequence

from apps.remote.durability import fsync_directory, fsync_file
from apps.remote.domain import stable_json_dumps, utc_now
from apps.remote.performance import measured, metrics


_ROOT_FILES = frozenset({"pyproject.toml", "cli.py", "apps/__init__.py"})
_RUNTIME_PREFIXES = (
    "core/",
    "instances/",
    "utility/",
    "scripts/",
)
_REMOTE_PREFIX = "apps/remote/"
_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "frontend",
        "node_modules",
        "results",
        "tests",
    }
)
_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".swp", "~")
_TEST_REFERENCE_PATHS: frozenset[str] = frozenset()
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_SECRET_BASENAMES = frozenset(
    {".env", "credentials.json", "service-account.json", "authorized_keys"}
)
_PRIVATE_KEY_PREFIXES = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")
_PEM_BEGIN = b"-----BEGIN "
_PEM_PRIVATE_KEY_END = b"PRIVATE KEY-----"
_PRIVATE_KEY_MARKERS = tuple(
    _PEM_BEGIN + key_type + _PEM_PRIVATE_KEY_END
    for key_type in (b"OPENSSH ", b"RSA ", b"EC ", b"")
)
_SNAPSHOT_CACHE_LOCK = threading.Lock()


class SnapshotError(RuntimeError):
    """Base class for source discovery and snapshot construction failures."""


class SnapshotSecurityError(SnapshotError):
    """A source path could not be archived without following an unsafe link."""


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    snapshot_id: str
    directory: Path
    archive_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    archive_sha256: str
    archive_size_bytes: int

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ManifestDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[FileChange, ...]
    source_changed: bool

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed or self.source_changed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": [
                {"path": item.path, "before": dict(item.before), "after": dict(item.after)}
                for item in self.changed
            ],
            "source_changed": self.source_changed,
            "is_empty": self.is_empty,
        }


class SnapshotBuilder:
    """Build an immutable archive from Git-known runtime files.

    The candidate set comes from ``git ls-files --cached --others
    --exclude-standard``: tracked files and non-ignored untracked files are both
    represented.  A narrow allowlist then prevents local results, tests, caches,
    and the retired frontend from entering a remote run.
    """

    def __init__(
        self,
        source_root: str | Path,
        output_root: str | Path | None = None,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.output_root = (
            Path(output_root).expanduser().resolve() if output_root is not None else None
        )
        if not self.source_root.is_dir():
            raise SnapshotError(f"source root is not a directory: {self.source_root}")
        # Fails unless the source folder is inside a Git worktree; it may be a sub-folder of it.
        self._git_text("rev-parse", "--show-toplevel")

    def selected_paths(self) -> tuple[str, ...]:
        """Return sorted, safe POSIX paths selected for the runtime snapshot."""

        output = self._git_bytes(
            "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        )
        paths: set[str] = set()
        for raw in output.split(b"\0"):
            if not raw:
                continue
            try:
                relative = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SnapshotSecurityError("Git returned a non-UTF-8 path") from exc
            normalized = _normalize_relative_path(relative)
            if _runtime_path_allowed(normalized):
                paths.add(normalized)
        return tuple(sorted(paths))

    @measured("snapshot.build")
    def build(self, output_root: str | Path | None = None) -> SourceSnapshot:
        destination_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else self.output_root
        )
        if destination_root is None:
            raise ValueError("an output_root is required")
        destination_root.mkdir(parents=True, exist_ok=True)

        candidates = self.selected_paths()
        if not candidates:
            raise SnapshotError("no runtime files were selected")
        git_before = self._git_state(paths=candidates)
        temporary = Path(
            tempfile.mkdtemp(prefix=".snapshot-build-", dir=destination_root)
        )
        tree = temporary / "tree"
        tree.mkdir(mode=0o700)
        try:
            files: list[dict[str, Any]] = []
            for relative in candidates:
                source = self.source_root / relative
                target = tree / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                metadata = _copy_regular_file_without_links(
                    source,
                    target,
                    self.source_root,
                )
                metadata["path"] = relative
                files.append(metadata)

            git_after_base = self._git_state(paths=candidates)
            # Commit/branch/status/diff must describe the bytes just copied.  A
            # concurrent checkout or edit is surfaced instead of silently
            # labelling a mixed snapshot with the wrong Git provenance.
            if git_before != git_after_base:
                raise SnapshotError("Git worktree changed while building the snapshot")
            git_after = self._git_state(files=files, paths=candidates)

            source_metadata = {
                "git_commit": git_after[0],
                "git_branch": git_after[1],
                "git_dirty": git_after[2],
                "git_diff_sha256": git_after[3],
            }
            canonical = {
                "schema_version": 1,
                "source": source_metadata,
                "files": files,
            }
            manifest_sha256 = canonical_manifest_hash(canonical)
            snapshot_id = manifest_sha256

            final = destination_root / snapshot_id
            with _SNAPSHOT_CACHE_LOCK:
                if final.exists():
                    existing = self._validated_cached_manifest(final, snapshot_id)
                    self._verify_sources(candidates, files, git_after)
                    shutil.rmtree(temporary)
                    fsync_file(final / "source.tar.gz")
                    fsync_file(final / "manifest.json")
                    fsync_directory(final)
                    fsync_directory(destination_root)
                    fsync_directory(destination_root.parent)
                    metrics.increment("snapshot.cache_hit")
                    return SourceSnapshot(
                        snapshot_id=snapshot_id, directory=final,
                        archive_path=final / "source.tar.gz",
                        manifest_path=final / "manifest.json", manifest=existing,
                        archive_sha256=str(existing["source_archive"]["sha256"]),
                        archive_size_bytes=int(existing["source_archive"]["size_bytes"]),
                    )
            metrics.increment("snapshot.cache_miss")

            archive_path = temporary / "source.tar.gz"
            _write_deterministic_archive(tree, files, archive_path)
            # Re-read every selected source file after the complete archive has
            # been written.  ``git status``/``git diff`` cannot detect content
            # changes to an already-untracked file, so without this pass a
            # concurrent editor could otherwise produce a time-mixed snapshot.
            self._verify_sources(candidates, files, git_after)
            archive_sha256 = _sha256_file(archive_path)
            archive_size = archive_path.stat().st_size
            manifest = {
                **canonical,
                "snapshot_id": snapshot_id,
                "manifest_sha256": manifest_sha256,
                "created_at": utc_now(),
                "source_archive": {
                    "name": "source.tar.gz",
                    "sha256": archive_sha256,
                    "size_bytes": archive_size,
                },
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                stable_json_dumps(manifest) + "\n",
                encoding="utf-8",
            )
            shutil.rmtree(tree)
            # Publish only a snapshot whose archive, manifest, and containing
            # directory have reached stable storage.  The second pass below
            # also covers a concurrently-created cache entry reused here.
            fsync_file(archive_path)
            fsync_file(manifest_path)
            fsync_directory(temporary)

            final = destination_root / snapshot_id
            with _SNAPSHOT_CACHE_LOCK:
                if final.exists():
                    self._validated_cached_manifest(final, snapshot_id)
                    shutil.rmtree(temporary)
                else:
                    os.replace(temporary, final)

            fsync_file(final / "source.tar.gz")
            fsync_file(final / "manifest.json")
            fsync_directory(final)
            fsync_directory(destination_root)
            fsync_directory(destination_root.parent)
            final_manifest = load_manifest(final / "manifest.json")
            archive = final / "source.tar.gz"
            return SourceSnapshot(
                snapshot_id=snapshot_id,
                directory=final,
                archive_path=archive,
                manifest_path=final / "manifest.json",
                manifest=final_manifest,
                archive_sha256=str(final_manifest["source_archive"]["sha256"]),
                archive_size_bytes=int(final_manifest["source_archive"]["size_bytes"]),
            )
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    @staticmethod
    def _validated_cached_manifest(final: Path, snapshot_id: str) -> dict[str, Any]:
        if final.is_symlink() or (final / "manifest.json").is_symlink() or (final / "source.tar.gz").is_symlink():
            raise SnapshotError(f"snapshot cache is symlinked: {final}")
        existing = load_manifest(final / "manifest.json")
        if canonical_manifest_hash(existing) != snapshot_id:
            raise SnapshotError(f"snapshot cache is corrupt: {final}")
        archive = final / "source.tar.gz"
        metadata = existing.get("source_archive", {})
        if archive.stat().st_size != metadata.get("size_bytes") or _sha256_file(archive) != metadata.get("sha256"):
            raise SnapshotError(f"snapshot archive is corrupt: {final}")
        return existing

    def _verify_sources(self, candidates, files, git_after) -> None:
        verified_files = []
        for relative in candidates:
            metadata = _read_regular_file_metadata_without_links(self.source_root / relative, self.source_root)
            metadata["path"] = relative
            verified_files.append(metadata)
        if verified_files != files or self.selected_paths() != candidates:
            raise SnapshotError("source files changed while building the snapshot")
        if self._git_state(files=verified_files, paths=candidates) != git_after:
            raise SnapshotError("Git worktree changed while building the snapshot")

    def _git_state(
        self,
        *,
        files: Sequence[Mapping[str, Any]] | None = None,
        paths: Sequence[str],
    ) -> tuple[str | None, str | None, bool, str]:
        commit = self._git_optional_text("rev-parse", "HEAD")
        branch = self._git_optional_text("branch", "--show-current") or None
        pathspecs = tuple(f":(literal){path}" for path in paths)
        status_payload = self._git_bytes(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *pathspecs,
        )
        if commit is None:
            diff_payload = b""
        else:
            diff_payload = self._git_bytes(
                "diff", "--binary", "HEAD", "--", *pathspecs
            )
        digest = hashlib.sha256()
        digest.update(diff_payload)
        digest.update(b"\0status\0")
        digest.update(status_payload)
        if files is not None:
            # Adds exact bytes for non-ignored untracked files, which ``git
            # diff HEAD`` cannot represent by itself.
            tracked = set(self._tracked_paths())
            untracked_files = [entry for entry in files if entry["path"] not in tracked]
            digest.update(b"\0untracked\0")
            digest.update(stable_json_dumps(untracked_files).encode("utf-8"))
        return commit, branch, bool(status_payload), digest.hexdigest()

    def _tracked_paths(self) -> tuple[str, ...]:
        output = self._git_bytes("ls-files", "--cached", "-z")
        return tuple(
            raw.decode("utf-8") for raw in output.split(b"\0") if raw
        )

    def _git_bytes(self, *arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(self.source_root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
        if completed.returncode:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise SnapshotError(f"git {' '.join(arguments)} failed: {message}")
        return completed.stdout

    def _git_text(self, *arguments: str) -> str:
        return self._git_bytes(*arguments).decode("utf-8").strip()

    def _git_optional_text(self, *arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(self.source_root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
        if completed.returncode:
            return None
        return completed.stdout.decode("utf-8").strip() or None


def canonical_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash only canonical source identity fields, excluding derived metadata."""

    canonical = {
        "schema_version": manifest.get("schema_version"),
        "source": manifest.get("source"),
        "files": manifest.get("files"),
    }
    return hashlib.sha256(stable_json_dumps(canonical).encode("utf-8")).hexdigest()


def load_manifest(path_or_manifest: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_manifest, Mapping):
        return dict(path_or_manifest)
    path = Path(path_or_manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot manifest must contain a JSON object")
    claimed = payload.get("manifest_sha256")
    actual = canonical_manifest_hash(payload)
    if claimed is not None and claimed != actual:
        raise SnapshotError("snapshot manifest SHA-256 is invalid")
    return payload


def diff_manifests(
    before: str | Path | Mapping[str, Any],
    after: str | Path | Mapping[str, Any],
) -> ManifestDiff:
    old = load_manifest(before)
    new = load_manifest(after)
    old_files = {str(item["path"]): item for item in old.get("files", [])}
    new_files = {str(item["path"]): item for item in new.get("files", [])}
    added = tuple(sorted(set(new_files) - set(old_files)))
    removed = tuple(sorted(set(old_files) - set(new_files)))
    changed = tuple(
        FileChange(path, old_files[path], new_files[path])
        for path in sorted(set(old_files) & set(new_files))
        if _file_identity(old_files[path]) != _file_identity(new_files[path])
    )
    return ManifestDiff(
        added=added,
        removed=removed,
        changed=changed,
        source_changed=old.get("source") != new.get("source"),
    )


def write_snapshot_delta_archive(
    source_archive: str | Path,
    manifest: Mapping[str, Any],
    paths: Sequence[str],
    destination: str | Path,
) -> dict[str, Any]:
    """Write a deterministic archive containing only selected snapshot files.

    The bytes are read from the already-frozen full snapshot archive rather than
    from the mutable working tree.  This keeps retries reproducible while making
    the WAN transfer proportional to the actual content changes.
    """

    source = Path(source_archive)
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise SnapshotError(f"delta archive destination already exists: {target}")
    if not source.is_file() or source.is_symlink():
        raise SnapshotError(f"snapshot archive is missing or unsafe: {source}")

    loaded = load_manifest(manifest)
    claimed = str(loaded.get("manifest_sha256") or "")
    if claimed != canonical_manifest_hash(loaded):
        raise SnapshotError("snapshot manifest SHA-256 is invalid")

    file_entries: dict[str, Mapping[str, Any]] = {}
    for item in loaded.get("files") or []:
        relative = _normalize_relative_path(str(item["path"]))
        if relative in file_entries:
            raise SnapshotError(f"snapshot manifest contains a duplicate path: {relative}")
        file_entries[relative] = item

    selected: list[str] = []
    seen_requested: set[str] = set()
    for value in paths:
        relative = _normalize_relative_path(str(value))
        if relative in seen_requested:
            raise SnapshotError(f"delta path is duplicated: {relative}")
        if relative not in file_entries:
            raise SnapshotError(f"delta path is absent from the snapshot: {relative}")
        seen_requested.add(relative)
        selected.append(relative)
    selected.sort()

    target.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    try:
        with target.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as output_archive:
                    with tarfile.open(source, mode="r|gz") as input_archive:
                        for member in input_archive:
                            relative = _normalize_relative_path(member.name)
                            if relative not in seen_requested:
                                continue
                            if relative in found:
                                raise SnapshotError(
                                    f"snapshot archive contains a duplicate member: {relative}"
                                )
                            if not member.isfile() or member.issym() or member.islnk():
                                raise SnapshotError(
                                    f"snapshot archive member is not a regular file: {relative}"
                                )
                            source_handle = input_archive.extractfile(member)
                            if source_handle is None:
                                raise SnapshotError(
                                    f"snapshot archive member cannot be read: {relative}"
                                )
                            item = file_entries[relative]
                            info = tarfile.TarInfo(relative)
                            info.size = int(item["size_bytes"])
                            info.mode = int(item["mode"])
                            info.mtime = 0
                            info.uid = 0
                            info.gid = 0
                            info.uname = ""
                            info.gname = ""
                            output_archive.addfile(info, source_handle)
                            found.add(relative)
                            # The source archive is a locally generated,
                            # checksum-verified snapshot. Once every requested
                            # member has been copied, inflating the remaining
                            # unchanged files would make a tiny delta scale with
                            # the complete source tree.
                            if found == seen_requested:
                                break
            raw.flush()
            os.fsync(raw.fileno())
        if found != seen_requested:
            missing = ", ".join(sorted(seen_requested - found))
            raise SnapshotError(f"snapshot archive is missing delta members: {missing}")
        return {
            "paths": selected,
            "file_count": len(selected),
            "logical_size_bytes": sum(
                int(file_entries[path]["size_bytes"]) for path in selected
            ),
            "archive_size_bytes": target.stat().st_size,
            "archive_sha256": _sha256_file(target),
        }
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _runtime_path_allowed(path: str) -> bool:
    pure = PurePosixPath(path)
    name = pure.name.lower()
    if any(part in _EXCLUDED_PARTS for part in pure.parts):
        return False
    if path.endswith(_EXCLUDED_SUFFIXES):
        return False
    if path in _TEST_REFERENCE_PATHS:
        return False
    if (
        name in _SECRET_BASENAMES
        or name.startswith(".env.")
        or name.endswith(_SECRET_SUFFIXES)
        or name.startswith(_PRIVATE_KEY_PREFIXES)
    ):
        return False
    if path in _ROOT_FILES:
        return True
    if path.startswith(_REMOTE_PREFIX):
        # The new web frontend is built and deployed independently.  Only the
        # Python worker/control plane belongs in a runtime source snapshot.
        return path.endswith(".py")
    return any(path.startswith(prefix) for prefix in _RUNTIME_PREFIXES)


def _normalize_relative_path(path: str) -> str:
    if "\x00" in path or "\\" in path:
        raise SnapshotSecurityError(f"unsafe Git path: {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise SnapshotSecurityError(f"unsafe Git path: {path!r}")
    normalized = pure.as_posix()
    if normalized.startswith("-"):
        raise SnapshotSecurityError(f"unsafe Git path: {path!r}")
    return normalized


def _copy_regular_file_without_links(
    source: Path,
    destination: Path,
    source_root: Path,
) -> dict[str, Any]:
    try:
        relative = source.relative_to(source_root)
    except ValueError as exc:
        raise SnapshotSecurityError(f"source escaped worktree: {source}") from exc
    current = source_root
    for part in relative.parts:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SnapshotSecurityError(f"snapshot refuses symlink: {relative}")
    source_info = source.lstat()
    if not stat.S_ISREG(source_info.st_mode):
        raise SnapshotSecurityError(f"snapshot source is not a regular file: {relative}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    size = 0
    marker_tail = b""
    try:
        opened_info = os.fstat(descriptor)
        if not stat.S_ISREG(opened_info.st_mode):
            raise SnapshotSecurityError(f"snapshot source changed type: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as source_handle:
            with destination.open("xb") as destination_handle:
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    marker_window = marker_tail + chunk
                    if any(marker in marker_window for marker in _PRIVATE_KEY_MARKERS):
                        raise SnapshotSecurityError(
                            f"snapshot refuses private-key content: {relative}"
                        )
                    marker_tail = marker_window[-64:]
                    destination_handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        final_info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current_info = source.lstat()
    identity_before = (opened_info.st_dev, opened_info.st_ino, opened_info.st_size, opened_info.st_mtime_ns)
    identity_after = (final_info.st_dev, final_info.st_ino, final_info.st_size, final_info.st_mtime_ns)
    identity_current = (
        current_info.st_dev,
        current_info.st_ino,
        current_info.st_size,
        current_info.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_current:
        raise SnapshotError(f"source file changed while being copied: {relative}")
    mode = stat.S_IMODE(source_info.st_mode)
    destination.chmod(mode)
    return {"sha256": digest.hexdigest(), "size_bytes": size, "mode": mode}


def _read_regular_file_metadata_without_links(
    source: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Hash one source file without following links and detect in-read edits."""

    try:
        relative = source.relative_to(source_root)
    except ValueError as exc:
        raise SnapshotSecurityError(f"source escaped worktree: {source}") from exc
    current = source_root
    for part in relative.parts:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SnapshotSecurityError(f"snapshot refuses symlink: {relative}")
    source_info = source.lstat()
    if not stat.S_ISREG(source_info.st_mode):
        raise SnapshotSecurityError(f"snapshot source is not a regular file: {relative}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    size = 0
    marker_tail = b""
    try:
        opened_info = os.fstat(descriptor)
        if not stat.S_ISREG(opened_info.st_mode):
            raise SnapshotSecurityError(f"snapshot source changed type: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as source_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                marker_window = marker_tail + chunk
                if any(marker in marker_window for marker in _PRIVATE_KEY_MARKERS):
                    raise SnapshotSecurityError(
                        f"snapshot refuses private-key content: {relative}"
                    )
                marker_tail = marker_window[-64:]
                digest.update(chunk)
                size += len(chunk)
        final_info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current_info = source.lstat()
    identity_before = (
        opened_info.st_dev,
        opened_info.st_ino,
        opened_info.st_size,
        opened_info.st_mtime_ns,
    )
    identity_after = (
        final_info.st_dev,
        final_info.st_ino,
        final_info.st_size,
        final_info.st_mtime_ns,
    )
    identity_current = (
        current_info.st_dev,
        current_info.st_ino,
        current_info.st_size,
        current_info.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_current:
        raise SnapshotError(f"source file changed while being verified: {relative}")
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "mode": stat.S_IMODE(source_info.st_mode),
    }


def _write_deterministic_archive(
    tree: Path,
    files: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for item in files:
                    relative = str(item["path"])
                    data_path = tree / relative
                    info = tarfile.TarInfo(relative)
                    info.size = int(item["size_bytes"])
                    info.mode = int(item["mode"])
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with data_path.open("rb") as handle:
                        archive.addfile(info, handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(item: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return item.get("sha256"), item.get("size_bytes"), item.get("mode")


__all__ = [
    "FileChange",
    "ManifestDiff",
    "SnapshotBuilder",
    "SnapshotError",
    "SnapshotSecurityError",
    "SourceSnapshot",
    "canonical_manifest_hash",
    "diff_manifests",
    "load_manifest",
    "write_snapshot_delta_archive",
]


def load_source_snapshot(output_root: Path, snapshot_id: str) -> SourceSnapshot:
    """Reuse one validated immutable source snapshot."""
    if len(snapshot_id) != 64 or any(c not in '0123456789abcdef' for c in snapshot_id):
        raise SnapshotError('Invalid source snapshot identity')
    directory = output_root / snapshot_id
    manifest = SnapshotBuilder._validated_cached_manifest(directory, snapshot_id)
    archive = manifest['source_archive']
    return SourceSnapshot(snapshot_id, directory, directory/'source.tar.gz',
                          directory/'manifest.json', manifest, archive['sha256'], archive['size_bytes'])
