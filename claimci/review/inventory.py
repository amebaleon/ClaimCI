"""Bounded, metadata-only Git change inventory construction."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import subprocess
import tempfile
import threading
import time
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    ExactMaterialOmission,
    MAX_CHANGE_INVENTORY_ENTRIES,
    ReviewError,
    SnapshotIdentity,
    SnapshotRole,
)


MAX_CHANGESET_METADATA_BYTES = 1024 * 1024
MAX_SELECTED_MATERIAL_BYTES = 16 * 1024 * 1024
_GIT_TIMEOUT_SECONDS = 10.0
_MAX_GIT_ERROR_BYTES = 4_096
_MAX_SNAPSHOT_INDEX_BYTES = MAX_SELECTED_MATERIAL_BYTES
_MAX_DECODED_INDEX_PATH_BYTES = _MAX_SNAPSHOT_INDEX_BYTES
_MAX_SNAPSHOT_INDEX_ENTRIES = 262_144
_MAX_WORKTREE_METADATA_ENTRIES = 262_144
_UINTMAX_MAX = (1 << 64) - 1
_WINDOWS_REPARSE_POINT = 0x400
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)


class InventoryVerificationError(ReviewError):
    """Typed, content-free failure at the trusted inventory boundary."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code.startswith("PREFLIGHT_G1_"):
            raise ValueError("inventory verification code is invalid")
        self.code = code
        super().__init__(message)


class _GitCommandError(Exception):
    """A Git invocation failed without retaining its untrusted diagnostics."""


class _GitOutputLimitError(Exception):
    """A Git invocation exceeded its bounded retained output."""


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _untrusted_execution_roots(root: Path) -> tuple[Path, ...]:
    try:
        roots = {root.resolve(strict=True), Path.cwd().resolve(strict=True)}
    except OSError as exc:
        raise _GitCommandError("git execution roots unavailable") from exc
    return tuple(roots)


def _trusted_path_directories(root: Path) -> tuple[Path, ...]:
    search_path = os.environ.get("PATH")
    if not search_path:
        raise _GitCommandError("git unavailable")
    excluded = _untrusted_execution_roots(root)
    directories: list[Path] = []
    seen: set[Path] = set()
    for raw_directory in search_path.split(os.pathsep):
        if not raw_directory:
            continue
        directory = Path(raw_directory)
        if not directory.is_absolute():
            continue
        try:
            directory = directory.resolve(strict=True)
        except OSError:
            continue
        if (
            not directory.is_dir()
            or directory in seen
            or any(_path_is_within(directory, item) for item in excluded)
        ):
            continue
        seen.add(directory)
        directories.append(directory)
    if not directories:
        raise _GitCommandError("trusted executable search path unavailable")
    return tuple(directories)


def _git_environment(root: Path) -> dict[str, str]:
    environment = {
        name: value
        for name in (
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
        )
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "LC_ALL": "C",
            "LANG": "C",
            "PATH": os.pathsep.join(
                str(directory) for directory in _trusted_path_directories(root)
            ),
        }
    )
    return environment


def _drain_bounded(
    stream: object,
    limit: int,
    sink: bytearray,
    overflow: threading.Event,
) -> None:
    try:
        while len(sink) <= limit:
            remaining = limit + 1 - len(sink)
            chunk = stream.read(min(65_536, remaining))  # type: ignore[attr-defined]
            if not chunk:
                return
            sink.extend(chunk)
            if len(sink) > limit:
                overflow.set()
                return
    finally:
        stream.close()  # type: ignore[attr-defined]


def _resolve_git_executable(root: Path | None = None) -> Path:
    execution_root = Path.cwd() if root is None else root
    excluded = _untrusted_execution_roots(execution_root)
    names = ("git.exe",) if os.name == "nt" else ("git",)
    for directory in _trusted_path_directories(execution_root):
        for name in names:
            try:
                candidate = (directory / name).resolve(strict=True)
            except OSError:
                continue
            if (
                candidate.is_file()
                and os.access(candidate, os.X_OK)
                and not any(_path_is_within(candidate, item) for item in excluded)
            ):
                return candidate
    raise _GitCommandError("git unavailable")


def _run_git(
    root: Path,
    arguments: tuple[str, ...],
    *,
    stdout_limit: int,
    stdin_bytes: bytes | None = None,
) -> bytes:
    executable = _resolve_git_executable(root)
    argv = (
        str(executable),
        "-c",
        "core.quotepath=false",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        *arguments,
    )
    environment = _git_environment(root)
    input_stream = None
    try:
        if stdin_bytes is not None:
            if len(stdin_bytes) > _MAX_SNAPSHOT_INDEX_BYTES:
                raise _GitOutputLimitError("git input exceeded bound")
            input_stream = tempfile.TemporaryFile()
            input_stream.write(stdin_bytes)
            input_stream.seek(0)
        process = subprocess.Popen(
            argv,
            cwd=str(root),
            stdin=input_stream if input_stream is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            shell=False,
        )
    except _GitOutputLimitError:
        raise
    except (OSError, ValueError) as exc:
        raise _GitCommandError("git unavailable") from exc
    finally:
        if input_stream is not None:
            input_stream.close()
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise _GitCommandError("git pipes unavailable")

    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    readers = (
        threading.Thread(
            target=_drain_bounded,
            args=(process.stdout, stdout_limit, stdout, stdout_overflow),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded,
            args=(process.stderr, _MAX_GIT_ERROR_BYTES, stderr, stderr_overflow),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    timed_out = False
    while process.poll() is None:
        if stdout_overflow.is_set() or stderr_overflow.is_set():
            process.kill()
            break
        if time.monotonic() >= deadline:
            timed_out = True
            process.kill()
            break
        time.sleep(0.01)
    try:
        return_code = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait(timeout=1.0)
        timed_out = True
    for reader in readers:
        reader.join(timeout=1.0)

    if any(reader.is_alive() for reader in readers):
        raise _GitCommandError("git output reader did not terminate")
    if stdout_overflow.is_set() or stderr_overflow.is_set():
        raise _GitOutputLimitError("git output exceeded bound")
    if timed_out or return_code != 0:
        raise _GitCommandError("git command failed")
    return bytes(stdout)


@dataclass(frozen=True)
class _IndexEntryMetadata:
    path: str
    mode: int
    skip_worktree: bool
    ctime_seconds: int
    ctime_nanoseconds: int
    mtime_seconds: int
    mtime_nanoseconds: int
    device: int
    inode: int
    uid: int
    gid: int
    size: int


@dataclass(frozen=True)
class _IndexSnapshot:
    path: Path
    encoded: bytes
    modified_time_ns: int
    entries: tuple[_IndexEntryMetadata, ...]


def _read_index_file(path: Path) -> tuple[bytes, int]:
    descriptor = -1
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        if path.is_symlink():
            raise OSError("index path is a symlink")
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > _MAX_SNAPSHOT_INDEX_BYTES
        ):
            raise OSError("index file is invalid or oversized")
        encoded = bytearray()
        while len(encoded) <= _MAX_SNAPSHOT_INDEX_BYTES:
            remaining = _MAX_SNAPSHOT_INDEX_BYTES + 1 - len(encoded)
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            encoded.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise _GitCommandError("snapshot index metadata unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        len(encoded) > _MAX_SNAPSHOT_INDEX_BYTES
        or len(encoded) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise _GitCommandError("snapshot index changed during verification")
    return bytes(encoded), before.st_mtime_ns


def _decode_v4_remove_count(encoded: bytes, offset: int, end: int) -> tuple[int, int]:
    if offset >= end:
        raise _GitCommandError("snapshot index metadata malformed")
    current = encoded[offset]
    offset += 1
    value = current & 0x7F
    while current & 0x80:
        value += 1
        if offset >= end or value > (_UINTMAX_MAX >> 7):
            raise _GitCommandError("snapshot index metadata malformed")
        current = encoded[offset]
        offset += 1
        value = (value << 7) | (current & 0x7F)
    return value, offset


def _validate_index_path(path: str) -> None:
    for component in path.split("/"):
        folded = component.casefold()
        device_name = folded.split(".", 1)[0].rstrip(" ")
        if (
            folded == ".git"
            or component.endswith((".", " "))
            or ":" in component
            or device_name in _WINDOWS_RESERVED_NAMES
        ):
            raise _GitCommandError("snapshot index metadata malformed")


def _is_reparse_point(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _has_indexed_descendant(indexed_paths: tuple[str, ...], path: str) -> bool:
    prefix = f"{path}/"
    position = bisect_left(indexed_paths, prefix)
    return position < len(indexed_paths) and indexed_paths[position].startswith(prefix)


def _parse_index(
    path: Path,
    encoded: bytes,
    modified_time_ns: int,
    object_format: str,
) -> _IndexSnapshot:
    hash_name, hash_bytes = ("sha1", 20) if object_format == "sha1" else ("sha256", 32)
    if len(encoded) < 12 + hash_bytes:
        raise _GitCommandError("snapshot index metadata malformed")
    payload = encoded[:-hash_bytes]
    if hashlib.new(hash_name, payload).digest() != encoded[-hash_bytes:]:
        raise _GitCommandError("snapshot index checksum mismatch")
    signature, version, entry_count = struct.unpack(">4sII", payload[:12])
    if (
        signature != b"DIRC"
        or version not in {2, 3, 4}
        or entry_count > _MAX_SNAPSHOT_INDEX_ENTRIES
    ):
        raise _GitCommandError("snapshot index metadata unsupported")

    offset = 12
    previous_path = b""
    decoded_path_bytes = 0
    gitlink_paths: list[str] = []
    entries: list[_IndexEntryMetadata] = []
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    for _ in range(entry_count):
        if time.monotonic() >= deadline:
            raise _GitCommandError("snapshot index metadata scan exceeded its bound")
        entry_start = offset
        fixed_size = 40 + hash_bytes + 2
        if offset + fixed_size > len(payload):
            raise _GitCommandError("snapshot index metadata malformed")
        values = struct.unpack(">10I", payload[offset : offset + 40])
        offset += 40 + hash_bytes
        flags = struct.unpack(">H", payload[offset : offset + 2])[0]
        offset += 2
        extended_flags = 0
        if flags & 0x4000:
            if version < 3 or offset + 2 > len(payload):
                raise _GitCommandError("snapshot index metadata malformed")
            extended_flags = struct.unpack(">H", payload[offset : offset + 2])[0]
            offset += 2
        if (
            flags & 0x8000
            or flags & 0x3000
            or extended_flags & ~0x4000
        ):
            raise _GitCommandError("snapshot index contains unsafe validity bits")

        if version == 4:
            remove_count, offset = _decode_v4_remove_count(
                payload, offset, len(payload)
            )
            terminator = payload.find(b"\0", offset)
            if terminator < 0 or remove_count > len(previous_path):
                raise _GitCommandError("snapshot index metadata malformed")
            raw_path = previous_path[: len(previous_path) - remove_count] + payload[
                offset:terminator
            ]
            offset = terminator + 1
        else:
            terminator = payload.find(b"\0", offset)
            if terminator < 0:
                raise _GitCommandError("snapshot index metadata malformed")
            raw_path = payload[offset:terminator]
            offset = terminator + 1
            padding = (-(offset - entry_start)) % 8
            if offset + padding > len(payload) or any(
                payload[offset : offset + padding]
            ):
                raise _GitCommandError("snapshot index metadata malformed")
            offset += padding
        declared_path_bytes = flags & 0x0FFF
        if declared_path_bytes < 0x0FFF and declared_path_bytes != len(raw_path):
            raise _GitCommandError("snapshot index metadata malformed")
        if previous_path and raw_path <= previous_path:
            raise _GitCommandError("snapshot index metadata malformed")
        decoded_path_bytes += len(raw_path)
        if decoded_path_bytes > _MAX_DECODED_INDEX_PATH_BYTES:
            raise _GitCommandError("snapshot index decoded path bound exceeded")
        previous_path = raw_path
        try:
            decoded_path = raw_path.decode("utf-8", errors="strict")
            _validate_index_path(decoded_path)
            validated_path = ChangeEntry(decoded_path, ChangeStatus.MODIFIED).path
        except (UnicodeDecodeError, ReviewError) as exc:
            raise _GitCommandError("snapshot index metadata malformed") from exc
        mode = values[6]
        if mode not in {0o100644, 0o100755, 0o120000, 0o160000}:
            raise _GitCommandError("snapshot index mode unsupported")
        if mode == 0o160000:
            gitlink_paths.append(validated_path)
        entries.append(
            _IndexEntryMetadata(
                path=validated_path,
                mode=mode,
                skip_worktree=bool(extended_flags & 0x4000),
                ctime_seconds=values[0],
                ctime_nanoseconds=values[1],
                mtime_seconds=values[2],
                mtime_nanoseconds=values[3],
                device=values[4],
                inode=values[5],
                uid=values[7],
                gid=values[8],
                size=values[9],
            )
        )

    indexed_paths = tuple(entry.path for entry in entries)
    for gitlink_path in gitlink_paths:
        if time.monotonic() >= deadline:
            raise _GitCommandError("snapshot index metadata scan exceeded its bound")
        if _has_indexed_descendant(indexed_paths, gitlink_path):
            raise _GitCommandError("snapshot index metadata malformed")
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise _GitCommandError("snapshot index extension metadata malformed")
        extension = payload[offset : offset + 4]
        extension_size = struct.unpack(">I", payload[offset + 4 : offset + 8])[0]
        offset += 8
        if (
            not extension
            or not 65 <= extension[0] <= 90
            or extension == b"FSMN"
            or offset + extension_size > len(payload)
        ):
            raise _GitCommandError("snapshot index extension unsupported")
        offset += extension_size
    return _IndexSnapshot(path, encoded, modified_time_ns, tuple(entries))


def _snapshot_index_metadata(root: Path) -> _IndexSnapshot:
    try:
        encoded_index_path = _run_git(
            root,
            ("rev-parse", "--path-format=absolute", "--git-path", "index"),
            stdout_limit=4_096,
        )
        encoded_object_format = _run_git(
            root,
            ("rev-parse", "--show-object-format"),
            stdout_limit=16,
        )
        encoded_shared_index_path = _run_git(
            root,
            ("rev-parse", "--shared-index-path"),
            stdout_limit=4_096,
        )
        index_text = encoded_index_path.decode("utf-8", errors="strict").strip()
        object_format = encoded_object_format.decode("ascii", errors="strict").strip()
    except (_GitCommandError, _GitOutputLimitError, UnicodeDecodeError) as exc:
        raise _GitCommandError("snapshot index metadata unavailable") from exc
    index_path = Path(index_text)
    if not index_text or not index_path.is_absolute() or object_format not in {
        "sha1",
        "sha256",
    }:
        raise _GitCommandError("snapshot index metadata unavailable")
    encoded, modified_time_ns = _read_index_file(index_path)
    if encoded_shared_index_path.strip():
        raise _GitCommandError("snapshot split index unsupported")
    parsed = _parse_index(
        index_path,
        encoded,
        modified_time_ns,
        object_format,
    )
    return _IndexSnapshot(
        index_path,
        encoded,
        modified_time_ns,
        parsed.entries,
    )


def _tracked_lstat(root: Path, path: str, directories: set[Path]) -> os.stat_result:
    current = root
    components = path.split("/")
    try:
        for component in components[:-1]:
            current /= component
            if current in directories:
                continue
            metadata = os.lstat(current)
            if not stat.S_ISDIR(metadata.st_mode) or _is_reparse_point(metadata):
                raise OSError("tracked path has a non-directory parent")
            directories.add(current)
        return os.lstat(current / components[-1])
    except OSError as exc:
        raise _GitCommandError("snapshot tracked path metadata unavailable") from exc


def _verify_index_entry_metadata(
    entry: _IndexEntryMetadata,
    metadata: os.stat_result,
    index_modified_time_ns: int,
) -> None:
    if entry.mode == 0o160000:
        if not stat.S_ISDIR(metadata.st_mode) or _is_reparse_point(metadata):
            raise _GitCommandError("snapshot gitlink mount is dirty")
        return
    expected_type = stat.S_ISLNK if entry.mode == 0o120000 else stat.S_ISREG
    if not expected_type(metadata.st_mode):
        raise _GitCommandError("snapshot tracked path type is dirty")
    index_size = metadata.st_size if metadata.st_size <= 0xFFFFFFFF else 0
    if index_size != entry.size:
        raise _GitCommandError("snapshot tracked path size is dirty")
    identity_mismatch = (
        divmod(metadata.st_mtime_ns, 1_000_000_000)
        != (entry.mtime_seconds, entry.mtime_nanoseconds)
        or divmod(metadata.st_ctime_ns, 1_000_000_000)
        != (entry.ctime_seconds, entry.ctime_nanoseconds)
    )
    actual_identity = (
        metadata.st_dev & 0xFFFFFFFF,
        metadata.st_ino & 0xFFFFFFFF,
        metadata.st_uid & 0xFFFFFFFF,
        metadata.st_gid & 0xFFFFFFFF,
    )
    expected_identity = (entry.device, entry.inode, entry.uid, entry.gid)
    if os.name == "nt":
        identity_mismatch = identity_mismatch or any(
            expected and actual != expected
            for actual, expected in zip(actual_identity, expected_identity, strict=True)
        )
    else:
        identity_mismatch = identity_mismatch or actual_identity != expected_identity
    if identity_mismatch:
        raise _GitCommandError("snapshot tracked path stat metadata is dirty")
    if os.name != "nt" and entry.mode in {0o100644, 0o100755}:
        if bool(metadata.st_mode & stat.S_IXUSR) != (entry.mode == 0o100755):
            raise _GitCommandError("snapshot tracked path mode is dirty")
    entry_modified_time_ns = (
        entry.mtime_seconds * 1_000_000_000 + entry.mtime_nanoseconds
    )
    if entry_modified_time_ns >= index_modified_time_ns:
        raise _GitCommandError("snapshot tracked path metadata is racy")


def _verify_index_entry_stat(
    root: Path,
    entry: _IndexEntryMetadata,
    index_modified_time_ns: int,
    directories: set[Path],
) -> None:
    metadata = _tracked_lstat(root, entry.path, directories)
    _verify_index_entry_metadata(entry, metadata, index_modified_time_ns)


def _verify_sparse_omissions(
    root: Path, entries: tuple[_IndexEntryMetadata, ...]
) -> None:
    sparse_paths = tuple(
        entry.path for entry in entries if entry.skip_worktree
    )
    if not sparse_paths:
        return
    encoded_paths = b"".join(
        path.encode("utf-8") + b"\0" for path in sparse_paths
    )
    try:
        matched = _run_git(
            root,
            ("sparse-checkout", "check-rules", "-z"),
            stdout_limit=len(encoded_paths),
            stdin_bytes=encoded_paths,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise _GitCommandError(
            "snapshot sparse omissions are not authorized"
        ) from exc
    if matched:
        raise _GitCommandError("snapshot sparse omissions are inconsistent")


def _verify_no_untracked_material(
    root: Path,
    entries: tuple[_IndexEntryMetadata, ...],
    index_modified_time_ns: int,
) -> None:
    entries_by_path = {entry.path: entry for entry in entries}
    indexed_paths = tuple(entries_by_path)
    pending: list[tuple[str, Path, bool]] = [("", root, False)]
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    observed = 0
    observed_indexed_paths: set[str] = set()
    try:
        while pending:
            relative_parent, directory, wholly_untracked = pending.pop()
            with os.scandir(directory) as children:
                for child in children:
                    observed += 1
                    if (
                        observed > _MAX_WORKTREE_METADATA_ENTRIES
                        or time.monotonic() >= deadline
                    ):
                        raise OSError("worktree metadata scan exceeded its bound")
                    relative = (
                        child.name
                        if not relative_parent
                        else f"{relative_parent}/{child.name}"
                    )
                    if not relative_parent and child.name == ".git":
                        continue
                    child_metadata = child.stat(follow_symlinks=False)
                    is_directory = stat.S_ISDIR(
                        child_metadata.st_mode
                    ) and not _is_reparse_point(child_metadata)
                    entry = entries_by_path.get(relative)
                    if entry is not None:
                        _verify_index_entry_metadata(
                            entry,
                            child_metadata,
                            index_modified_time_ns,
                        )
                        observed_indexed_paths.add(relative)
                        continue
                    if wholly_untracked:
                        if not is_directory:
                            raise _GitCommandError("snapshot worktree is dirty")
                        pending.append((relative, Path(child.path), True))
                    elif _has_indexed_descendant(indexed_paths, relative):
                        if not is_directory:
                            raise _GitCommandError("snapshot worktree is dirty")
                        pending.append((relative, Path(child.path), False))
                    elif not is_directory:
                        raise _GitCommandError("snapshot worktree is dirty")
                    else:
                        pending.append((relative, Path(child.path), True))
    except OSError as exc:
        raise _GitCommandError("snapshot worktree metadata unavailable") from exc
    required_paths = {
        entry.path for entry in entries if not entry.skip_worktree
    }
    if not required_paths.issubset(observed_indexed_paths):
        raise _GitCommandError("snapshot tracked path metadata unavailable")


def _verify_tracked_worktree(root: Path) -> _IndexSnapshot:
    snapshot = _snapshot_index_metadata(root)
    _verify_sparse_omissions(root, snapshot.entries)
    directories = {root}
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    for entry in snapshot.entries:
        if entry.skip_worktree:
            continue
        if time.monotonic() >= deadline:
            raise _GitCommandError("snapshot tracked metadata scan exceeded its bound")
        _verify_index_entry_stat(
            root,
            entry,
            snapshot.modified_time_ns,
            directories,
        )
    _verify_no_untracked_material(
        root,
        snapshot.entries,
        snapshot.modified_time_ns,
    )
    return snapshot


def _verify_snapshot(identity: SnapshotIdentity) -> None:
    try:
        resolved_head = _run_git(
            identity.root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            stdout_limit=129,
        ).strip()
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot Git identity could not be verified",
        ) from exc
    try:
        actual_sha = resolved_head.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot Git identity could not be verified",
        ) from exc
    if actual_sha != identity.sha:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_SHA_MISMATCH",
            "snapshot SHA identity mismatch",
        )

    try:
        index_snapshot = _verify_tracked_worktree(identity.root)
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot worktree is dirty or unavailable",
        ) from exc

    try:
        _run_git(
            identity.root,
            (
                "diff-index",
                "--cached",
                "--quiet",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "--ignore-submodules=none",
                "HEAD",
                "--",
            ),
            stdout_limit=0,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot index is dirty or unavailable",
        ) from exc
    try:
        current_index, _modified_time_ns = _read_index_file(index_snapshot.path)
    except _GitCommandError as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot index changed or became unavailable",
        ) from exc
    if current_index != index_snapshot.encoded:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot index changed during verification",
        )


def _require_commit_object(root: Path, sha: str) -> None:
    try:
        _run_git(
            root,
            ("cat-file", "-e", f"{sha}^{{commit}}"),
            stdout_limit=0,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "required Git commit object is unavailable",
        ) from exc


def _verify_coordinates(
    requested_base: SnapshotIdentity,
    comparison_base: SnapshotIdentity,
    head: SnapshotIdentity,
    comparison_basis: ComparisonBasis,
) -> None:
    identities = (requested_base, comparison_base, head)
    expected_roles = (
        SnapshotRole.REQUESTED_BASE,
        SnapshotRole.COMPARISON_BASE,
        SnapshotRole.HEAD,
    )
    if not all(isinstance(identity, SnapshotIdentity) for identity in identities):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot identities are invalid",
        )
    if tuple(identity.role for identity in identities) != expected_roles:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot roles do not match their coordinates",
        )
    if not isinstance(comparison_basis, ComparisonBasis):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_COMPARISON_BASE_MISMATCH",
            "comparison basis is invalid",
        )

    shas_by_root: dict[Path, str] = {}
    for identity in identities:
        prior = shas_by_root.setdefault(identity.root, identity.sha)
        if prior != identity.sha:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_SNAPSHOT_SHA_MISMATCH",
                "one snapshot root cannot identify different commits",
            )
        _verify_snapshot(identity)

    _require_commit_object(head.root, requested_base.sha)
    _require_commit_object(head.root, comparison_base.sha)
    if comparison_basis is ComparisonBasis.DIRECT_BASE:
        if comparison_base.sha != requested_base.sha:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_COMPARISON_BASE_MISMATCH",
                "direct comparison base does not match requested base",
            )
        return

    try:
        encoded = _run_git(
            head.root,
            ("merge-base", "--all", requested_base.sha, head.sha),
            stdout_limit=130,
        )
        lines = encoded.decode("ascii").splitlines()
    except (_GitCommandError, _GitOutputLimitError, UnicodeDecodeError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_COMPARISON_BASE_MISMATCH",
            "unique Git merge base could not be verified",
        ) from exc
    if len(lines) != 1:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_COMPARISON_BASE_MISMATCH",
            "unique Git merge base could not be verified",
        )
    if lines[0] != comparison_base.sha:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_COMPARISON_BASE_MISMATCH",
            "declared comparison base does not match Git merge base",
        )


def _parse_change_entries(encoded: bytes) -> tuple[ChangeEntry, ...]:
    if not encoded:
        return ()
    if not encoded.endswith(b"\0"):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "Git change metadata is malformed",
        )
    fields = encoded[:-1].split(b"\0")
    if len(fields) % 2 != 0:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "Git change metadata is malformed",
        )
    entry_count = len(fields) // 2
    if entry_count > MAX_CHANGE_INVENTORY_ENTRIES:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGESET_PATH_LIMIT",
            "Git change metadata exceeds the entry limit",
        )

    statuses = {
        b"A": ChangeStatus.ADDED,
        b"M": ChangeStatus.MODIFIED,
        b"D": ChangeStatus.DELETED,
    }
    entries: list[ChangeEntry] = []
    for index in range(0, len(fields), 2):
        status = statuses.get(fields[index])
        if status is None:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_CHANGE_INVENTORY_STATUS_INVALID",
                "Git change metadata has an unsupported status",
            )
        try:
            path = fields[index + 1].decode("utf-8", errors="strict")
            entries.append(ChangeEntry(path=path, status=status))
        except (UnicodeDecodeError, ReviewError) as exc:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH",
                "Git change metadata contains an invalid path",
            ) from exc
    return tuple(sorted(entries, key=lambda entry: (entry.path, entry.status.value)))


def build_git_change_inventory(
    requested_base: SnapshotIdentity,
    comparison_base: SnapshotIdentity,
    head: SnapshotIdentity,
    comparison_basis: ComparisonBasis,
) -> ChangeInventory:
    """Build a complete A/M/D inventory from trusted local Git metadata."""

    _verify_coordinates(requested_base, comparison_base, head, comparison_basis)
    try:
        encoded = _run_git(
            head.root,
            (
                "diff-tree",
                "-r",
                "--no-commit-id",
                "--name-status",
                "-z",
                "--no-renames",
                "--no-ext-diff",
                "--ignore-submodules=none",
                comparison_base.sha,
                head.sha,
            ),
            stdout_limit=MAX_CHANGESET_METADATA_BYTES,
        )
    except _GitOutputLimitError as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGESET_METADATA_BYTES_LIMIT",
            "Git change metadata exceeds the metadata limit",
        ) from exc
    except _GitCommandError as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_UNAVAILABLE",
            "Git change metadata is unavailable",
        ) from exc
    entries = _parse_change_entries(encoded)
    folded_paths = tuple(entry.path.casefold() for entry in entries)
    if len(set(folded_paths)) != len(folded_paths):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_DUPLICATE_PATH",
            "Git change metadata contains duplicate paths",
        )
    try:
        return ChangeInventory(
            schema_version=1,
            requested_base_sha=requested_base.sha,
            comparison_base_sha=comparison_base.sha,
            head_sha=head.sha,
            comparison_basis=comparison_basis,
            source=ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH,
            declared_entry_count=len(entries),
            complete=True,
            entries=entries,
        )
    except ReviewError as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "Git change metadata could not form a valid inventory",
        ) from exc


def _git_blob_descriptor(
    root: Path,
    commit_sha: str,
    path: str,
) -> tuple[int, str] | None:
    """Return one exact committed regular blob's size and object ID."""

    try:
        encoded = _run_git(
            root,
            (
                "ls-tree",
                "-l",
                "-z",
                "--full-tree",
                commit_sha,
                "--",
                path,
            ),
            stdout_limit=8_192,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected material Git identity is unavailable",
        ) from exc
    if not encoded:
        return None
    records = encoded.split(b"\0")
    if len(records) != 2 or records[1] or b"\t" not in records[0]:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "selected material Git metadata is malformed",
        )
    metadata, raw_path = records[0].split(b"\t", 1)
    fields = metadata.split()
    try:
        decoded_path = raw_path.decode("utf-8", errors="strict")
        size = int(fields[3].decode("ascii"))
        object_id = fields[2].decode("ascii")
    except (IndexError, UnicodeDecodeError, ValueError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "selected material Git metadata is malformed",
        ) from exc
    if (
        len(fields) != 4
        or fields[0] not in {b"100644", b"100755"}
        or fields[1] != b"blob"
        or decoded_path != path
        or size < 0
    ):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "selected material is not an exact regular Git blob",
        )
    if len(object_id) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in object_id
    ):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "selected material Git object ID is malformed",
        )
    return size, object_id


def git_blob_descriptor(
    identity: SnapshotIdentity,
    path: str,
) -> tuple[int, str] | None:
    """Return bounded metadata for one canonical path at an exact snapshot."""

    if not isinstance(identity, SnapshotIdentity):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected material snapshot identity is invalid",
        )
    try:
        canonical = ChangeEntry(path=path, status=ChangeStatus.MODIFIED).path
    except ReviewError as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH",
            "selected material path is invalid",
        ) from exc
    if canonical != path:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH",
            "selected material path is not canonical",
        )
    return _git_blob_descriptor(identity.root, identity.sha, canonical)


def git_blob_object_id(
    identity: SnapshotIdentity,
    path: str,
) -> str | None:
    """Return one exact regular blob ID without requiring its content or size."""

    if not isinstance(identity, SnapshotIdentity):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected material snapshot identity is invalid",
        )
    try:
        canonical = ChangeEntry(path=path, status=ChangeStatus.MODIFIED).path
    except ReviewError as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH",
            "selected material path is invalid",
        ) from exc
    if canonical != path:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_INVALID_PATH",
            "selected material path is not canonical",
        )
    try:
        encoded = _run_git(
            identity.root,
            (
                "ls-tree",
                "-z",
                "--full-tree",
                identity.sha,
                "--",
                canonical,
            ),
            stdout_limit=8_192,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected material Git identity is unavailable",
        ) from exc
    if not encoded:
        return None
    records = encoded.split(b"\0")
    if len(records) != 2 or records[1] or b"\t" not in records[0]:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "selected material Git metadata is malformed",
        )
    metadata, raw_path = records[0].split(b"\t", 1)
    fields = metadata.split()
    try:
        decoded_path = raw_path.decode("utf-8", errors="strict")
        object_id = fields[2].decode("ascii")
    except (IndexError, UnicodeDecodeError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "selected material Git metadata is malformed",
        ) from exc
    if (
        len(fields) != 3
        or fields[0] not in {b"100644", b"100755"}
        or fields[1] != b"blob"
        or decoded_path != canonical
        or len(object_id) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in object_id)
    ):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
            "selected material is not an exact regular Git blob",
        )
    return object_id


def exact_git_material_omission_matches(
    omission: ExactMaterialOmission,
    identity: SnapshotIdentity,
) -> bool:
    """Bind a sparse size fact to its source descriptor and target tree entry."""

    if (
        not isinstance(omission, ExactMaterialOmission)
        or not isinstance(identity, SnapshotIdentity)
        or omission.source.role is not omission.role
        or identity.role is not omission.role
        or omission.source.sha != identity.sha
    ):
        return False
    try:
        source_descriptor = git_blob_descriptor(omission.source, omission.path)
        target_object_id = git_blob_object_id(identity, omission.path)
    except InventoryVerificationError:
        return False
    return (
        source_descriptor == (omission.observed, omission.object_id)
        and target_object_id == omission.object_id
    )


def _read_git_blob(
    identity: SnapshotIdentity,
    path: str,
    *,
    max_bytes: int,
    maximum_allowed_bytes: int,
) -> bytes | None:
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or not 0 <= max_bytes <= maximum_allowed_bytes
    ):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected Git blob read bound is invalid",
        )
    descriptor = git_blob_descriptor(identity, path)
    if descriptor is None or descriptor[0] > max_bytes:
        return None
    size, object_id = descriptor
    try:
        content = _run_git(
            identity.root,
            ("cat-file", "blob", object_id),
            stdout_limit=max_bytes,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected Git blob content is unavailable",
        ) from exc
    if len(content) != size:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected Git blob content size is inconsistent",
        )
    return content


def read_git_blob(
    identity: SnapshotIdentity,
    path: str,
    *,
    max_bytes: int,
) -> bytes | None:
    """Read one exact metadata blob within the one-mebibyte metadata cap."""

    return _read_git_blob(
        identity,
        path,
        max_bytes=max_bytes,
        maximum_allowed_bytes=MAX_CHANGESET_METADATA_BYTES,
    )


def read_git_material_blob(
    identity: SnapshotIdentity,
    path: str,
    *,
    max_bytes: int,
) -> bytes | None:
    """Read one selected exact Git blob within the materialization byte cap."""

    return _read_git_blob(
        identity,
        path,
        max_bytes=max_bytes,
        maximum_allowed_bytes=MAX_SELECTED_MATERIAL_BYTES,
    )


def verify_git_material_identities(
    comparison_base: SnapshotIdentity,
    head: SnapshotIdentity,
    inventory: ChangeInventory,
    selected_paths: Sequence[str],
    head_identities: Mapping[str, tuple[int, str, str]],
    comparison_identities: Mapping[str, tuple[int, str, str] | None],
    *,
    max_files: int,
) -> None:
    """Bind selected descriptor captures to exact comparison/head blob IDs."""

    selected = tuple(selected_paths)
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected material identity bound is invalid",
        )
    if (
        not isinstance(comparison_base, SnapshotIdentity)
        or not isinstance(head, SnapshotIdentity)
        or not isinstance(inventory, ChangeInventory)
        or not isinstance(selected_paths, Sequence)
        or isinstance(selected_paths, (str, bytes))
        or len(selected) > max_files
        or len(set(selected)) != len(selected)
        or set(head_identities) != set(selected)
        or set(comparison_identities) != set(selected)
    ):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "selected material identity contract is invalid",
        )

    status_by_path = {entry.path: entry.status for entry in inventory.entries}

    def matches_blob(
        captured: tuple[int, str, str],
        blob: tuple[int, str],
    ) -> bool:
        size, sha1, sha256 = captured
        blob_size, object_id = blob
        return size == blob_size and object_id == (
            sha1 if len(object_id) == 40 else sha256
        )

    for path in selected:
        status = status_by_path.get(path)
        if status not in {None, ChangeStatus.ADDED, ChangeStatus.MODIFIED}:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
                "selected material has an invalid exact inventory status",
            )
        head_blob = _git_blob_descriptor(
            head.root,
            head.sha,
            path,
        )
        if head_blob is None:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
                "selected material is absent from the exact head commit",
            )
        if not matches_blob(head_identities[path], head_blob):
            raise InventoryVerificationError(
                "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
                "selected material does not match the exact head commit",
            )

        comparison_blob = _git_blob_descriptor(
            comparison_base.root,
            comparison_base.sha,
            path,
        )
        expected_comparison = comparison_identities[path]
        if status is ChangeStatus.ADDED:
            if comparison_blob is not None or expected_comparison is not None:
                raise InventoryVerificationError(
                    "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
                    "added material exists in the exact comparison commit",
                )
        elif (
            comparison_blob is None
            or expected_comparison is None
            or not matches_blob(expected_comparison, comparison_blob)
        ):
            raise InventoryVerificationError(
                "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
                "selected material does not match the exact comparison commit",
            )
        elif status is None and comparison_blob != head_blob:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_CHANGE_INVENTORY_MALFORMED",
                "supplemental material is not unchanged across the comparison",
            )
