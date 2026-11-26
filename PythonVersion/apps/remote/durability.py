"""Narrow filesystem durability primitives for scientific run artifacts.

Closing a file only hands dirty pages to the kernel, and ``os.replace`` only
changes the in-memory directory entry.  Relay uses these helpers at the small
number of boundaries where a power loss must not make a locally verified
result disappear after its remote copy has been deleted.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Iterable


def fsync_file(path: str | Path) -> None:
    """Flush one existing regular file without following a final symlink."""

    target = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"durability target is not a regular file: {target}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: str | Path) -> None:
    """Flush one directory entry set without following a final symlink."""

    target = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(f"durability target is not a directory: {target}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directories(
    directories: Iterable[str | Path],
    *,
    include_parents: Iterable[str | Path] = (),
) -> None:
    """Flush unique directories deepest-first, then explicit parent anchors."""

    unique = {Path(path) for path in directories}
    for directory in sorted(unique, key=lambda item: len(item.parts), reverse=True):
        fsync_directory(directory)
    for parent in dict.fromkeys(Path(path) for path in include_parents):
        if parent not in unique:
            fsync_directory(parent)


def fsync_directory_tree(root: str | Path, *, include_parent: bool = True) -> None:
    """Flush every directory below ``root`` and optionally its parent anchor."""

    target = Path(root)
    if target.is_symlink() or not target.is_dir():
        raise OSError(f"durability tree root is not a safe directory: {target}")
    directories = [target]
    for path in target.rglob("*"):
        if path.is_symlink():
            raise OSError(f"durability tree contains a symlink: {path}")
        if path.is_dir():
            directories.append(path)
    fsync_directories(
        directories,
        include_parents=(target.parent,) if include_parent else (),
    )


__all__ = [
    "fsync_directories",
    "fsync_directory",
    "fsync_directory_tree",
    "fsync_file",
]
