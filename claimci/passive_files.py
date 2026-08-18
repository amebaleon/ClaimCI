"""Descriptor-bound reads for passive, repository-confined files."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class PassiveFileError(ValueError):
    """A passive path could not be captured without crossing a trust boundary."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PassiveFileCapture:
    """One bounded byte snapshot tied to the opened regular-file identity."""

    content: bytes
    size: int
    sha256: str


def _path_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _file_path_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_open_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Fields reported consistently by pathname and handle stats on Windows."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _open_passive(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def capture_confined_regular_file(
    root: Path,
    relative: str,
    *,
    max_bytes: int,
) -> PassiveFileCapture:
    """Capture a stable regular file without trusting a checked pathname later."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise TypeError("passive file byte bound must be a non-negative integer")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise PassiveFileError("passive file path is not portable", code="unsafe_path")
    lexical = PurePosixPath(relative)
    if lexical.is_absolute() or any(part in {"", ".", ".."} for part in lexical.parts):
        raise PassiveFileError("passive file path is not confined", code="unsafe_path")

    try:
        confined_root = Path(root).resolve(strict=True)
        root_before = os.lstat(confined_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise PassiveFileError("passive file root is unavailable", code="unavailable") from error
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise PassiveFileError("passive file root is not a trusted directory", code="symlink")

    current = confined_root
    component_identities: list[tuple[Path, tuple[int, int, int]]] = []
    try:
        for index, component in enumerate(lexical.parts):
            current = current / component
            observed = os.lstat(current)
            if stat.S_ISLNK(observed.st_mode):
                raise PassiveFileError(
                    "passive file path contains a symbolic link",
                    code="symlink",
                )
            if index < len(lexical.parts) - 1 and not stat.S_ISDIR(observed.st_mode):
                raise PassiveFileError(
                    "passive file parent is not a directory",
                    code="not_regular",
                )
            component_identities.append((current, _path_identity(observed)))
        resolved = current.resolve(strict=True)
        resolved.relative_to(confined_root)
        if resolved != current:
            raise PassiveFileError(
                "passive file path is not a direct confined path",
                code="outside",
            )
        before = os.lstat(current)
        if not stat.S_ISREG(before.st_mode):
            raise PassiveFileError(
                "passive file is not a regular file",
                code="not_regular",
            )
        if before.st_size > max_bytes:
            raise PassiveFileError(
                "passive file exceeds the byte bound",
                code="too_large",
            )
        descriptor = _open_passive(current)
    except PassiveFileError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PassiveFileError("passive file could not be opened", code="unavailable") from error

    opened: os.stat_result | None = None
    opened_after: os.stat_result | None = None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PassiveFileError(
                "passive file descriptor is not regular",
                code="not_regular",
            )
        if _file_open_identity(opened) != _file_open_identity(before):
            raise PassiveFileError(
                "passive file identity changed during open",
                code="changed",
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            block = os.read(descriptor, min(64 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        content = b"".join(chunks)
        if len(content) > max_bytes:
            raise PassiveFileError(
                "passive file changed beyond the byte bound",
                code="too_large",
            )
        opened_after = os.fstat(descriptor)
        if _file_open_identity(opened_after) != _file_open_identity(opened):
            raise PassiveFileError(
                "passive file descriptor changed during capture",
                code="changed",
            )
    except PassiveFileError:
        raise
    except OSError as error:
        raise PassiveFileError("passive file read failed", code="unavailable") from error
    finally:
        os.close(descriptor)

    assert opened is not None
    assert opened_after is not None
    try:
        root_after = os.lstat(confined_root)
        if _path_identity(root_after) != _path_identity(root_before):
            raise PassiveFileError(
                "passive file root identity changed during capture",
                code="changed",
            )
        for component, identity in component_identities[:-1]:
            if _path_identity(os.lstat(component)) != identity:
                raise PassiveFileError(
                    "passive file parent identity changed during capture",
                    code="changed",
                )
        after = os.lstat(current)
    except PassiveFileError:
        raise
    except OSError as error:
        raise PassiveFileError(
            "passive file identity could not be rechecked",
            code="changed",
        ) from error
    if _file_path_identity(after) != _file_path_identity(before):
        raise PassiveFileError(
            "passive file pathname changed during capture",
            code="changed",
        )
    if _file_open_identity(after) != _file_open_identity(opened_after):
        raise PassiveFileError(
            "passive file identity changed during capture",
            code="changed",
        )
    if len(content) != opened.st_size:
        raise PassiveFileError(
            "passive file size changed during capture",
            code="changed",
        )
    return PassiveFileCapture(
        content=content,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


__all__ = [
    "PassiveFileCapture",
    "PassiveFileError",
    "capture_confined_regular_file",
]
