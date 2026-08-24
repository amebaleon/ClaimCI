"""Trusted descriptor-bound streaming sources for passive artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath

from .contracts import (
    AnalysisContractError,
    ArtifactCandidate,
    GitCommitSha,
    RepositoryIdentity,
)


_READ_CHUNK_BYTES = 64 * 1024
_TRACE_PREFIX = b"claimci-evidence-trace-bytes-v1\0"
_MAX_STREAM_BYTES = 16 * 1024 * 1024 * 1024


class StreamingArtifactError(AnalysisContractError):
    """A streaming artifact could not be consumed safely."""


class StreamingIntegrityError(StreamingArtifactError):
    """A streamed source failed exact size, hash, or identity validation."""


class StreamingLimitError(StreamingArtifactError):
    """A streamed source exceeded one of Core's explicit work budgets."""


class ScanPurpose(str, Enum):
    INTEGRITY_ONLY = "integrity_only"
    SCHEMA = "schema"
    SELECTED_OBSERVATIONS = "selected_observations"
    DATASET_RECORDS = "dataset_records"


class ScanState(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE_BUDGET = "incomplete_budget"
    INCOMPLETE_LIMIT = "incomplete_limit"
    FAILED = "failed"


class ScanReason(str, Enum):
    COMPLETE = "complete"
    SEMANTIC_BYTE_BUDGET = "semantic_byte_budget"
    RECORD_BUDGET = "record_budget"
    OBSERVATION_BUDGET = "observation_budget"
    LOGICAL_RECORD_LIMIT = "logical_record_limit"
    COLUMN_LIMIT = "column_limit"
    NODE_LIMIT = "node_limit"
    DEPTH_LIMIT = "depth_limit"
    CANDIDATE_LIMIT = "candidate_limit"
    SCRATCH_BUDGET = "scratch_budget"
    MALFORMED = "malformed"
    INVALID_UTF8 = "invalid_utf8"
    INTEGRITY_BUDGET = "integrity_budget"
    SIZE_MISMATCH = "size_mismatch"
    SHA_MISMATCH = "sha_mismatch"
    SOURCE_CHANGED = "source_changed"


def _positive_int(value: object, label: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise AnalysisContractError(
            f"{label} must be between one and {maximum}"
        )
    return value


@dataclass(frozen=True, slots=True)
class StreamingLimits:
    """Core-owned deterministic work budgets for streamed artifacts."""

    max_integrity_bytes: int = 1024 * 1024 * 1024
    max_semantic_bytes: int = 512 * 1024 * 1024
    max_semantic_records: int = 250_000
    max_logical_record_bytes: int = 1024 * 1024
    max_columns: int = 256
    max_nodes_per_record: int = 100_000
    max_metric_candidates: int = 32
    max_observations: int = 100_000
    max_scratch_bytes: int = 512 * 1024 * 1024
    max_hash_previews: int = 128

    def __post_init__(self) -> None:
        for label, value, maximum in (
            ("max_integrity_bytes", self.max_integrity_bytes, _MAX_STREAM_BYTES),
            ("max_semantic_bytes", self.max_semantic_bytes, _MAX_STREAM_BYTES),
            ("max_semantic_records", self.max_semantic_records, 10_000_000),
            (
                "max_logical_record_bytes",
                self.max_logical_record_bytes,
                16 * 1024 * 1024,
            ),
            ("max_columns", self.max_columns, 256),
            ("max_nodes_per_record", self.max_nodes_per_record, 1_000_000),
            ("max_metric_candidates", self.max_metric_candidates, 64),
            ("max_observations", self.max_observations, 1_000_000),
            ("max_scratch_bytes", self.max_scratch_bytes, _MAX_STREAM_BYTES),
            ("max_hash_previews", self.max_hash_previews, 1024),
        ):
            _positive_int(value, label, maximum)


@dataclass(frozen=True, slots=True)
class ScanCompleteness:
    """Exact integrity and semantic-completeness facts for one scan purpose."""

    purpose: ScanPurpose
    state: ScanState
    reason: ScanReason
    records_scanned: int
    semantic_bytes: int
    integrity_bytes: int
    expected_bytes: int
    integrity_verified: bool
    peak_buffer_bytes: int
    scratch_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, ScanPurpose):
            raise TypeError("scan purpose must be ScanPurpose")
        if not isinstance(self.state, ScanState):
            raise TypeError("scan state must be ScanState")
        if not isinstance(self.reason, ScanReason):
            raise TypeError("scan reason must be ScanReason")
        for label, value in (
            ("records_scanned", self.records_scanned),
            ("semantic_bytes", self.semantic_bytes),
            ("integrity_bytes", self.integrity_bytes),
            ("expected_bytes", self.expected_bytes),
            ("peak_buffer_bytes", self.peak_buffer_bytes),
            ("scratch_bytes", self.scratch_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AnalysisContractError(
                    f"scan {label} must be a non-negative integer"
                )
        if not isinstance(self.integrity_verified, bool):
            raise TypeError("scan integrity_verified must be boolean")
        if self.integrity_verified and self.integrity_bytes != self.expected_bytes:
            raise AnalysisContractError(
                "verified scan byte count must equal expected bytes"
            )
        if self.semantic_bytes > self.integrity_bytes:
            raise AnalysisContractError(
                "scan semantic bytes cannot exceed integrity bytes"
            )
        if self.state is ScanState.COMPLETE:
            if not self.integrity_verified or self.reason is not ScanReason.COMPLETE:
                raise AnalysisContractError(
                    "complete scan requires verified integrity and complete reason"
                )
        elif self.reason is ScanReason.COMPLETE:
            raise AnalysisContractError(
                "incomplete or failed scan cannot use the complete reason"
            )

    @property
    def complete(self) -> bool:
        return self.state is ScanState.COMPLETE


def _path_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_open_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Fields reported consistently by pathname and descriptor stats on Windows."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _canonical_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StreamingIntegrityError("artifact source path is not portable")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StreamingIntegrityError("artifact source path is not confined")
    return path


def _resolved_root(value: Path) -> tuple[Path, tuple[int, int, int]]:
    try:
        root = Path(value).resolve(strict=True)
        observed = os.lstat(root)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise StreamingIntegrityError("artifact source root is unavailable") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise StreamingIntegrityError("artifact source root is not trusted")
    return root, _path_identity(observed)


def _open_descriptor(
    source: "ArtifactSource",
) -> tuple[
    int,
    Path,
    os.stat_result,
    tuple[tuple[Path, tuple[int, int, int]], ...],
]:
    root, current_root_identity = _resolved_root(source._root)
    if root != source._root or current_root_identity != source._root_identity:
        raise StreamingIntegrityError("artifact source root identity changed")
    current = root
    components: list[tuple[Path, tuple[int, int, int]]] = []
    try:
        parts = _canonical_relative(str(source.candidate.path)).parts
        for index, component in enumerate(parts):
            current = current / component
            observed = os.lstat(current)
            if stat.S_ISLNK(observed.st_mode):
                raise StreamingIntegrityError(
                    "artifact source path contains a symbolic link"
                )
            if index < len(parts) - 1 and not stat.S_ISDIR(observed.st_mode):
                raise StreamingIntegrityError(
                    "artifact source parent is not a directory"
                )
            components.append((current, _path_identity(observed)))
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
        if resolved != current:
            raise StreamingIntegrityError("artifact source path escaped confinement")
        before = os.lstat(current)
        if not stat.S_ISREG(before.st_mode):
            raise StreamingIntegrityError("artifact source is not a regular file")
        if before.st_size != source.candidate.size:
            raise StreamingIntegrityError("artifact source size changed")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(current, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_open_identity(
            opened
        ) != _file_open_identity(before):
            os.close(descriptor)
            raise StreamingIntegrityError("artifact source identity changed during open")
        return descriptor, current, opened, tuple(components)
    except StreamingIntegrityError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise StreamingIntegrityError("artifact source could not be opened") from error


@dataclass(frozen=True, slots=True, init=False)
class ArtifactSource:
    """One factory-issued, exact-head, reopenable confined artifact source."""

    repository: RepositoryIdentity
    head_sha: GitCommitSha
    candidate: ArtifactCandidate
    _root: Path = field(repr=False, compare=False)
    _root_identity: tuple[int, int, int] = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("ArtifactSource values require the trusted snapshot factory")

    @classmethod
    def _from_snapshot(
        cls,
        repository: RepositoryIdentity,
        head_sha: GitCommitSha,
        root: Path,
        candidate: ArtifactCandidate,
    ) -> "ArtifactSource":
        instance = object.__new__(cls)
        resolved, identity = _resolved_root(root)
        object.__setattr__(instance, "repository", repository)
        object.__setattr__(instance, "head_sha", head_sha)
        object.__setattr__(instance, "candidate", candidate)
        object.__setattr__(instance, "_root", resolved)
        object.__setattr__(instance, "_root_identity", identity)
        descriptor, _path, _opened, _components = _open_descriptor(instance)
        os.close(descriptor)
        return instance

    def open_scan(
        self,
        purpose: ScanPurpose,
        limits: StreamingLimits = StreamingLimits(),
    ) -> "ArtifactScan":
        if not isinstance(purpose, ScanPurpose):
            raise TypeError("artifact scan purpose must be ScanPurpose")
        if type(limits) is not StreamingLimits:
            raise TypeError("artifact scan limits must be StreamingLimits")
        return ArtifactScan._open(self, purpose, limits)


def artifact_source_from_snapshot(
    repository: RepositoryIdentity,
    head_sha: GitCommitSha,
    root: Path,
    candidate: ArtifactCandidate,
) -> ArtifactSource:
    """Issue one source from trusted repository/head/root/candidate inputs."""

    if type(repository) is not RepositoryIdentity:
        raise TypeError("artifact source repository must be RepositoryIdentity")
    if not isinstance(head_sha, GitCommitSha):
        head_sha = GitCommitSha(head_sha)
    if type(candidate) is not ArtifactCandidate:
        raise TypeError("artifact source candidate must be ArtifactCandidate")
    return ArtifactSource._from_snapshot(repository, head_sha, Path(root), candidate)


@dataclass(slots=True, init=False)
class ArtifactScan:
    """One descriptor-bound streaming pass with incremental integrity state."""

    source: ArtifactSource
    purpose: ScanPurpose
    limits: StreamingLimits
    maximum_read_request: int
    _descriptor: int
    _path: Path
    _opened: os.stat_result
    _components: tuple[tuple[Path, tuple[int, int, int]], ...]
    _raw_hash: object
    _trace_hash: object
    _integrity_bytes: int
    _semantic_bytes: int
    _peak_buffer_bytes: int
    _eof: bool
    _finished: bool
    _closed: bool

    @classmethod
    def _open(
        cls,
        source: ArtifactSource,
        purpose: ScanPurpose,
        limits: StreamingLimits,
    ) -> "ArtifactScan":
        descriptor, path, opened, components = _open_descriptor(source)
        instance = object.__new__(cls)
        instance.source = source
        instance.purpose = purpose
        instance.limits = limits
        instance.maximum_read_request = 0
        instance._descriptor = descriptor
        instance._path = path
        instance._opened = opened
        instance._components = components
        instance._raw_hash = hashlib.sha256()
        instance._trace_hash = hashlib.sha256(_TRACE_PREFIX)
        instance._integrity_bytes = 0
        instance._semantic_bytes = 0
        instance._peak_buffer_bytes = 0
        instance._eof = False
        instance._finished = False
        instance._closed = False
        return instance

    def __enter__(self) -> "ArtifactScan":
        if self._closed:
            raise StreamingIntegrityError("artifact scan is already closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def trace_sha256(self) -> str:
        if not self._finished:
            raise StreamingIntegrityError("trace digest requires a finished scan")
        return self._trace_hash.hexdigest()

    def read_chunk(self, maximum: int = _READ_CHUNK_BYTES) -> bytes:
        if self._closed or self._finished:
            raise StreamingIntegrityError("artifact scan is not readable")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= _READ_CHUNK_BYTES
        ):
            raise StreamingLimitError("stream read request exceeds the chunk bound")
        remaining = self.limits.max_integrity_bytes - self._integrity_bytes
        request = min(maximum, max(1, remaining))
        self.maximum_read_request = max(self.maximum_read_request, request)
        try:
            block = os.read(self._descriptor, request)
        except OSError as error:
            raise StreamingIntegrityError("artifact source read failed") from error
        if not block:
            self._eof = True
            return b""
        self._integrity_bytes += len(block)
        self._raw_hash.update(block)
        self._trace_hash.update(block)
        if self._integrity_bytes > self.limits.max_integrity_bytes:
            raise StreamingLimitError("artifact exceeds the integrity-I/O budget")
        return block

    def note_semantic_bytes(self, count: int, *, buffered: int = 0) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TypeError("semantic byte count must be a non-negative integer")
        if isinstance(buffered, bool) or not isinstance(buffered, int) or buffered < 0:
            raise TypeError("buffered byte count must be a non-negative integer")
        self._semantic_bytes += count
        self._peak_buffer_bytes = max(self._peak_buffer_bytes, buffered)

    def _drain(self) -> None:
        while not self._eof:
            self.read_chunk(_READ_CHUNK_BYTES)

    def _verify_stable_source(self) -> None:
        try:
            opened_after = os.fstat(self._descriptor)
            if _file_identity(opened_after) != _file_identity(self._opened):
                raise StreamingIntegrityError(
                    "artifact source descriptor changed during scan"
                )
            root, root_identity = _resolved_root(self.source._root)
            if root != self.source._root or root_identity != self.source._root_identity:
                raise StreamingIntegrityError(
                    "artifact source root changed during scan"
                )
            for component, identity in self._components[:-1]:
                if _path_identity(os.lstat(component)) != identity:
                    raise StreamingIntegrityError(
                        "artifact source parent changed during scan"
                    )
            if _file_open_identity(os.lstat(self._path)) != _file_open_identity(
                self._opened
            ):
                raise StreamingIntegrityError(
                    "artifact source pathname changed during scan"
                )
        except StreamingIntegrityError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise StreamingIntegrityError(
                "artifact source identity could not be revalidated"
            ) from error

    def finish(
        self,
        *,
        records_scanned: int,
        semantic_complete: bool,
        state: ScanState | None = None,
        reason: ScanReason | None = None,
        scratch_bytes: int = 0,
    ) -> ScanCompleteness:
        if self._closed or self._finished:
            raise StreamingIntegrityError("artifact scan cannot be finished twice")
        if not isinstance(semantic_complete, bool):
            raise TypeError("semantic_complete must be boolean")
        self._drain()
        self._verify_stable_source()
        expected = self.source.candidate.size
        if self._integrity_bytes != expected:
            raise StreamingIntegrityError("artifact integrity size mismatch")
        if self._raw_hash.hexdigest() != self.source.candidate.sha256:
            raise StreamingIntegrityError("artifact integrity sha256 mismatch")
        if semantic_complete:
            resolved_state = ScanState.COMPLETE
            resolved_reason = ScanReason.COMPLETE
        else:
            resolved_state = state or ScanState.INCOMPLETE_BUDGET
            resolved_reason = reason or ScanReason.SEMANTIC_BYTE_BUDGET
            if resolved_state is ScanState.COMPLETE:
                raise AnalysisContractError(
                    "an incomplete semantic scan cannot claim complete state"
                )
        report = ScanCompleteness(
            purpose=self.purpose,
            state=resolved_state,
            reason=resolved_reason,
            records_scanned=records_scanned,
            semantic_bytes=self._semantic_bytes,
            integrity_bytes=self._integrity_bytes,
            expected_bytes=expected,
            integrity_verified=True,
            peak_buffer_bytes=self._peak_buffer_bytes,
            scratch_bytes=scratch_bytes,
        )
        self._finished = True
        return report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._descriptor)


__all__ = [
    "ArtifactScan",
    "ArtifactSource",
    "ScanCompleteness",
    "ScanPurpose",
    "ScanReason",
    "ScanState",
    "StreamingArtifactError",
    "StreamingIntegrityError",
    "StreamingLimitError",
    "StreamingLimits",
    "artifact_source_from_snapshot",
]
