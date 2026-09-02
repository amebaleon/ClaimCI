"""Bounded, metadata-only Git change inventory construction."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ComparisonBasis,
    MAX_CHANGE_INVENTORY_ENTRIES,
    ReviewError,
    SnapshotIdentity,
    SnapshotRole,
)


MAX_CHANGESET_METADATA_BYTES = 1024 * 1024
_GIT_TIMEOUT_SECONDS = 10.0
_MAX_GIT_ERROR_BYTES = 4_096


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


def _git_environment() -> dict[str, str]:
    environment = {
        name: value
        for name in (
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATH",
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


def _run_git(root: Path, arguments: tuple[str, ...], *, stdout_limit: int) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise _GitCommandError("git unavailable")
    argv = (
        str(Path(executable).resolve()),
        "-c",
        "core.quotepath=false",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        *arguments,
    )
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(),
            shell=False,
        )
    except (OSError, ValueError) as exc:
        raise _GitCommandError("git unavailable") from exc
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
        _run_git(
            identity.root,
            ("diff-index", "--cached", "--quiet", "HEAD", "--"),
            stdout_limit=0,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot index is dirty or unavailable",
        ) from exc

    try:
        worktree_changes = _run_git(
            identity.root,
            (
                "ls-files",
                "--modified",
                "--deleted",
                "--others",
                "--directory",
                "--no-empty-directory",
                "-z",
            ),
            stdout_limit=1,
        )
    except (_GitCommandError, _GitOutputLimitError) as exc:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot worktree is dirty or unavailable",
        ) from exc
    if worktree_changes:
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "snapshot worktree is dirty",
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
                "--ignore-submodules=all",
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
