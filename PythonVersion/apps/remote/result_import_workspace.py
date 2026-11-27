"""Safe, database-neutral copy boundary for existing result bundles.

This module validates a quiescent EDOLAB bundle and atomically publishes the
two SQLite databases into an empty Relay-managed staging directory.  External
imports reject every path below Relay's data root by default; the explicit
``allow_managed_source`` gate exists only for copying a checksum-verified run
into the independent result library.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

from apps.remote.durability import fsync_directory, fsync_directory_tree
from core.result.store import SUMMARY_DB_NAME, TREND_DB_NAME
from core.result.validation import (
    ResultBundleValidation,
    validate_result_bundle,
)


RESULT_DATABASE_NAMES = (SUMMARY_DB_NAME, TREND_DB_NAME)
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_COPY_CHUNK_BYTES = 1024 * 1024


class ResultImportWorkspaceError(ValueError):
    """Raised when an external bundle cannot be imported safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ImportedResultFile:
    """One checksum-verified file suitable for transfer/artifact records."""

    filename: str
    destination_path: Path
    transfer_path: str
    artifact_relative_path: str
    size_bytes: int
    sha256: str

    def manifest_item(self) -> dict[str, Any]:
        return {
            "path": self.transfer_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    def artifact_descriptor(self) -> dict[str, Any]:
        return {
            "relative_path": self.artifact_relative_path,
            "staging_path": str(self.destination_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ImportedResultBundle:
    """Verified import evidence for a caller-created synthetic completed run."""

    source_bundle: Path
    staging_root: Path
    bundle_dir: Path
    validation: ResultBundleValidation
    files: tuple[ImportedResultFile, ...]

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    @property
    def transfer_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": "local_result_import",
            "bundle_id": self.validation.bundle_id,
            "files": [item.manifest_item() for item in self.files],
        }

    @property
    def artifact_descriptors(self) -> list[dict[str, Any]]:
        return [item.artifact_descriptor() for item in self.files]

    def as_dict(self) -> dict[str, Any]:
        validation = self.validation.as_dict()
        validation.pop("output_dir", None)
        return {
            "source_bundle": str(self.source_bundle),
            "staging_root": str(self.staging_root),
            "bundle_dir": str(self.bundle_dir),
            "validation": validation,
            "total_size_bytes": self.total_size_bytes,
            "transfer_manifest": self.transfer_manifest,
            "artifacts": self.artifact_descriptors,
        }


@dataclass(frozen=True, slots=True)
class _PathIdentity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_PathIdentity":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size=int(value.st_size),
            modified_ns=int(value.st_mtime_ns),
            changed_ns=int(value.st_ctime_ns),
        )


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    directory: _PathIdentity
    files: dict[str, _PathIdentity]


class ResultImportWorkspace:
    """Import completed bundles into empty directories managed by one Relay."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = _existing_safe_directory(
            data_root,
            code="unsafe_data_root",
            label="Relay data root",
        )

    def import_bundle(
        self,
        source_bundle: str | Path,
        staging_root: str | Path,
        *,
        allow_managed_source: bool = False,
    ) -> ImportedResultBundle:
        """Validate and atomically stage one external completed result bundle.

        ``staging_root`` must already exist, be empty, and be a strict
        descendant of ``data_root``.  On success it contains exactly
        ``remote/results/bundle/{summary.sqlite,trend.sqlite}``.  On any failure
        it is returned to its original empty state.
        """

        source = self._source_bundle(
            source_bundle,
            allow_managed_source=allow_managed_source,
        )
        staging = self._staging_root(staging_root)
        if _paths_overlap(source, staging):
            raise ResultImportWorkspaceError(
                "source_staging_overlap",
                "Result bundle source and import staging directory must not overlap.",
            )
        source_snapshot = _capture_source_snapshot(source)
        try:
            source_validation = validate_result_bundle(
                source,
                require_completed=True,
            )
        except Exception as exc:
            raise ResultImportWorkspaceError(
                "invalid_result_bundle",
                f"External result bundle validation failed: {exc}",
            ) from exc
        _assert_source_unchanged(source, source_snapshot)

        staging_identity = _directory_anchor(staging)
        temporary_root: Path | None = None
        published = False
        try:
            temporary_root = Path(
                tempfile.mkdtemp(prefix=".result-import-", dir=staging)
            )
            temporary_root.chmod(0o700)
            temporary_remote = temporary_root / "remote"
            temporary_bundle = temporary_remote / "results" / "bundle"
            temporary_bundle.mkdir(parents=True, mode=0o700)
            for directory in (
                temporary_remote,
                temporary_remote / "results",
                temporary_bundle,
            ):
                directory.chmod(0o700)

            copied: list[ImportedResultFile] = []
            for filename in RESULT_DATABASE_NAMES:
                source_file = source / filename
                destination_file = temporary_bundle / filename
                size_bytes, digest = _copy_regular_file(
                    source_file,
                    destination_file,
                    expected_identity=source_snapshot.files[filename],
                )
                copied.append(
                    ImportedResultFile(
                        filename=filename,
                        destination_path=staging
                        / "remote"
                        / "results"
                        / "bundle"
                        / filename,
                        transfer_path=f"results/bundle/{filename}",
                        artifact_relative_path=f"remote/results/bundle/{filename}",
                        size_bytes=size_bytes,
                        sha256=digest,
                    )
                )

            _assert_source_unchanged(source, source_snapshot)
            try:
                copied_validation = validate_result_bundle(
                    temporary_bundle,
                    require_completed=True,
                )
            except Exception as exc:
                raise ResultImportWorkspaceError(
                    "copied_bundle_validation_failed",
                    f"Copied result bundle validation failed: {exc}",
                ) from exc
            if _validation_identity(copied_validation) != _validation_identity(
                source_validation
            ):
                raise ResultImportWorkspaceError(
                    "source_changed_during_import",
                    "Copied bundle identity differs from the bundle validated before import.",
                )

            fsync_directory_tree(temporary_remote, include_parent=True)
            _assert_source_unchanged(source, source_snapshot)
            if _directory_anchor(staging) != staging_identity:
                raise ResultImportWorkspaceError(
                    "staging_changed_during_import",
                    "Relay staging directory changed while the bundle was imported.",
                )
            visible_entries = list(staging.iterdir())
            if visible_entries != [temporary_root]:
                raise ResultImportWorkspaceError(
                    "staging_changed_during_import",
                    "Relay staging directory is no longer exclusively owned by this import.",
                )

            final_remote = staging / "remote"
            os.rename(temporary_remote, final_remote)
            published = True
            temporary_root.rmdir()
            fsync_directory_tree(final_remote, include_parent=True)
            fsync_directory(staging)
            final_bundle = final_remote / "results" / "bundle"
            try:
                final_validation = validate_result_bundle(
                    final_bundle,
                    require_completed=True,
                )
            except Exception as exc:
                raise ResultImportWorkspaceError(
                    "published_bundle_validation_failed",
                    f"Published result bundle validation failed: {exc}",
                ) from exc
            if _validation_identity(final_validation) != _validation_identity(
                copied_validation
            ):
                raise ResultImportWorkspaceError(
                    "published_bundle_changed",
                    "Published result bundle identity differs from its staged validation.",
                )
            for item in copied:
                size_bytes, digest = _hash_regular_file(item.destination_path)
                if size_bytes != item.size_bytes or digest != item.sha256:
                    raise ResultImportWorkspaceError(
                        "published_bundle_changed",
                        f"Published result file checksum changed: {item.filename}",
                    )
            return ImportedResultBundle(
                source_bundle=source,
                staging_root=staging,
                bundle_dir=final_bundle,
                validation=final_validation,
                files=tuple(copied),
            )
        except BaseException as exc:
            try:
                if published:
                    _remove_owned_tree(staging / "remote", staging)
                if temporary_root is not None:
                    _remove_owned_tree(temporary_root, staging)
                fsync_directory(staging)
            except BaseException as cleanup_exc:
                raise ResultImportWorkspaceError(
                    "import_cleanup_failed",
                    "Result import failed and its private staging tree could not be cleaned.",
                ) from cleanup_exc
            if isinstance(exc, ResultImportWorkspaceError):
                raise
            if not isinstance(exc, Exception):
                raise
            raise ResultImportWorkspaceError(
                "result_import_failed",
                f"Result bundle import failed: {exc}",
            ) from exc

    def _source_bundle(
        self,
        value: str | Path,
        *,
        allow_managed_source: bool,
    ) -> Path:
        source = _existing_safe_directory(
            value,
            code="unsafe_source_bundle",
            label="External result bundle",
        )
        if not allow_managed_source and _paths_overlap(source, self.data_root):
            raise ResultImportWorkspaceError(
                "source_inside_data_root",
                "External result bundle must not be inside, contain, or equal the Relay data root.",
            )
        for entry in source.iterdir():
            if entry.is_symlink():
                raise ResultImportWorkspaceError(
                    "unsafe_source_bundle",
                    f"External result bundle contains a symlink: {entry.name}",
                )
        for filename in RESULT_DATABASE_NAMES:
            path = source / filename
            try:
                info = path.lstat()
            except FileNotFoundError as exc:
                raise ResultImportWorkspaceError(
                    "missing_result_file",
                    f"External result bundle is missing {filename}.",
                ) from exc
            if not stat.S_ISREG(info.st_mode):
                raise ResultImportWorkspaceError(
                    "unsafe_source_bundle",
                    f"External result file is not a regular file: {filename}",
                )
            for suffix in _SQLITE_SIDECAR_SUFFIXES:
                sidecar = source / f"{filename}{suffix}"
                if sidecar.exists() or sidecar.is_symlink():
                    raise ResultImportWorkspaceError(
                        "source_bundle_not_quiescent",
                        f"External result database has a live SQLite sidecar: {sidecar.name}",
                    )
        return source

    def _staging_root(self, value: str | Path) -> Path:
        staging = _existing_safe_directory(
            value,
            code="unsafe_staging_root",
            label="Relay staging root",
        )
        try:
            relative = staging.relative_to(self.data_root)
        except ValueError as exc:
            raise ResultImportWorkspaceError(
                "staging_outside_data_root",
                "Result import staging root must be inside the Relay data root.",
            ) from exc
        if not relative.parts:
            raise ResultImportWorkspaceError(
                "unsafe_staging_root",
                "Relay data root itself cannot be used as an import staging directory.",
            )
        if any(staging.iterdir()):
            raise ResultImportWorkspaceError(
                "staging_not_empty",
                "Result import staging root must be empty.",
            )
        return staging


def _existing_safe_directory(
    value: str | Path,
    *,
    code: str,
    label: str,
) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ResultImportWorkspaceError(code, f"{label} must be an absolute path.")
    if ".." in raw.parts:
        raise ResultImportWorkspaceError(code, f"{label} must not contain '..'.")
    _reject_symlink_chain(raw, code=code, label=label)
    try:
        resolved = raw.resolve(strict=True)
        info = resolved.lstat()
    except OSError as exc:
        raise ResultImportWorkspaceError(code, f"{label} does not exist.") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ResultImportWorkspaceError(code, f"{label} must be a regular directory.")
    return resolved


def _reject_symlink_chain(path: Path, *, code: str, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise ResultImportWorkspaceError(code, f"{label} does not exist.") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ResultImportWorkspaceError(
                code,
                f"{label} must not contain symlinked path components.",
            )


def _capture_source_snapshot(source: Path) -> _SourceSnapshot:
    return _SourceSnapshot(
        directory=_PathIdentity.from_stat(source.lstat()),
        files={
            filename: _PathIdentity.from_stat((source / filename).lstat())
            for filename in RESULT_DATABASE_NAMES
        },
    )


def _assert_source_unchanged(source: Path, expected: _SourceSnapshot) -> None:
    try:
        current = _capture_source_snapshot(source)
    except OSError as exc:
        raise ResultImportWorkspaceError(
            "source_changed_during_import",
            "External result bundle disappeared while it was imported.",
        ) from exc
    if current != expected:
        raise ResultImportWorkspaceError(
            "source_changed_during_import",
            "External result bundle changed while it was imported.",
        )


def _copy_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_identity: _PathIdentity,
) -> tuple[int, str]:
    source_flags = os.O_RDONLY
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        source_flags |= os.O_CLOEXEC
        destination_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        destination_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    destination_descriptor = -1
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        opened = _PathIdentity.from_stat(os.fstat(source_descriptor))
        if opened != expected_identity or not stat.S_ISREG(opened.mode):
            raise ResultImportWorkspaceError(
                "source_changed_during_import",
                f"External result file changed before copying: {source.name}",
            )
        destination_descriptor = os.open(
            destination,
            destination_flags,
            0o600,
        )
        os.fchmod(destination_descriptor, 0o600)
        while True:
            chunk = os.read(source_descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise OSError("short write while copying result database")
                remaining = remaining[written:]
        if _PathIdentity.from_stat(os.fstat(source_descriptor)) != expected_identity:
            raise ResultImportWorkspaceError(
                "source_changed_during_import",
                f"External result file changed while copying: {source.name}",
            )
        os.fsync(destination_descriptor)
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    copied = destination.lstat()
    if not stat.S_ISREG(copied.st_mode) or int(copied.st_size) != size_bytes:
        raise ResultImportWorkspaceError(
            "copied_file_invalid",
            f"Copied result file is missing or incomplete: {destination.name}",
        )
    return size_bytes, digest.hexdigest()


def _hash_regular_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        opened = _PathIdentity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(opened.mode):
            raise ResultImportWorkspaceError(
                "published_bundle_changed",
                f"Published result path is not a regular file: {path.name}",
            )
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
        if _PathIdentity.from_stat(os.fstat(descriptor)) != opened:
            raise ResultImportWorkspaceError(
                "published_bundle_changed",
                f"Published result file changed while hashing: {path.name}",
            )
    finally:
        os.close(descriptor)
    return size_bytes, digest.hexdigest()


def _validation_identity(validation: ResultBundleValidation) -> tuple[Any, ...]:
    return (
        validation.bundle_id,
        validation.format_version,
        validation.mode,
        validation.status,
        validation.task_count,
        validation.runset_count,
        validation.expected_runs,
        validation.completed_runs,
        validation.stored_runs,
        validation.trend_metric_count,
    )


def _directory_anchor(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    return (
        int(info.st_dev),
        int(info.st_ino),
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _remove_owned_tree(path: Path, expected_parent: Path) -> None:
    if path.parent != expected_parent:
        raise OSError(f"refusing to clean an unexpected import path: {path}")
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


__all__ = [
    "ImportedResultBundle",
    "ImportedResultFile",
    "RESULT_DATABASE_NAMES",
    "ResultImportWorkspace",
    "ResultImportWorkspaceError",
]
