"""Bounded temporary dataset indexing for native Audit consumption."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from claimci.dataset_check import canonicalize_sample
from claimci.models import ClaimCIError
from claimci.parsing import unique_json_object

from .artifact_source import (
    ArtifactSource,
    ScanCompleteness,
    ScanPurpose,
    ScanReason,
    ScanState,
    StreamingArtifactError,
    StreamingLimits,
)
from .contracts import (
    AnalysisContractError,
    ArtifactKind,
    GitCommitSha,
    RepositoryIdentity,
    RepositoryPath,
    Sha256Digest,
)
from .adapters.streaming import _ScanStop, _binary_lines, _validate_record_graph


_SCHEMA_SQL = """
CREATE TABLE dataset_hash_counts (
  experiment TEXT NOT NULL CHECK (experiment IN ('baseline','candidate')),
  split TEXT NOT NULL CHECK (split IN ('train','eval')),
  digest TEXT NOT NULL,
  multiplicity INTEGER NOT NULL,
  PRIMARY KEY (experiment, split, digest)
)
""".strip()
_UPSERT_SQL = """
INSERT INTO dataset_hash_counts (experiment, split, digest, multiplicity)
VALUES (?, ?, ?, ?)
ON CONFLICT (experiment, split, digest)
DO UPDATE SET multiplicity = multiplicity + excluded.multiplicity
""".strip()
_DELETE_SPLIT_SQL = (
    "DELETE FROM dataset_hash_counts WHERE experiment = ? AND split = ?"
)
_COUNT_SQL = """
SELECT COALESCE(SUM(multiplicity), 0)
FROM dataset_hash_counts
WHERE experiment = ? AND split = ?
""".strip()
_MULTISET_SQL = """
SELECT digest, multiplicity
FROM dataset_hash_counts
WHERE experiment = ? AND split = ?
ORDER BY digest
""".strip()
_OVERLAP_COUNT_SQL = """
SELECT COALESCE(SUM(
  CASE WHEN train.multiplicity < evaluation.multiplicity
       THEN train.multiplicity ELSE evaluation.multiplicity END
), 0), COUNT(*)
FROM dataset_hash_counts AS train
JOIN dataset_hash_counts AS evaluation
  ON train.experiment = evaluation.experiment
 AND train.digest = evaluation.digest
WHERE train.experiment = ?
  AND train.split = 'train'
  AND evaluation.split = 'eval'
""".strip()
_OVERLAP_PREVIEW_SQL = """
SELECT train.digest
FROM dataset_hash_counts AS train
JOIN dataset_hash_counts AS evaluation
  ON train.experiment = evaluation.experiment
 AND train.digest = evaluation.digest
WHERE train.experiment = ?
  AND train.split = 'train'
  AND evaluation.split = 'eval'
ORDER BY train.digest
LIMIT ?
""".strip()
_MATCHING_EVAL_SQL = """
SELECT COALESCE(SUM(
  CASE WHEN baseline.multiplicity < candidate.multiplicity
       THEN baseline.multiplicity ELSE candidate.multiplicity END
), 0)
FROM dataset_hash_counts AS baseline
JOIN dataset_hash_counts AS candidate
  ON baseline.digest = candidate.digest
WHERE baseline.experiment = 'baseline'
  AND baseline.split = 'eval'
  AND candidate.experiment = 'candidate'
  AND candidate.split = 'eval'
""".strip()
_SIDE_ONLY_COUNT_SQL = """
SELECT COALESCE(SUM(
  CASE WHEN source.multiplicity > COALESCE(other.multiplicity, 0)
       THEN source.multiplicity - COALESCE(other.multiplicity, 0) ELSE 0 END
), 0)
FROM dataset_hash_counts AS source
LEFT JOIN dataset_hash_counts AS other
  ON other.experiment = ?
 AND other.split = 'eval'
 AND other.digest = source.digest
WHERE source.experiment = ?
  AND source.split = 'eval'
""".strip()
_SIDE_ONLY_PREVIEW_SQL = """
SELECT source.digest,
       source.multiplicity - COALESCE(other.multiplicity, 0) AS difference
FROM dataset_hash_counts AS source
LEFT JOIN dataset_hash_counts AS other
  ON other.experiment = ?
 AND other.split = 'eval'
 AND other.digest = source.digest
WHERE source.experiment = ?
  AND source.split = 'eval'
  AND source.multiplicity > COALESCE(other.multiplicity, 0)
ORDER BY source.digest
LIMIT ?
""".strip()

_BATCH_RECORDS = 128
_EXPERIMENTS = ("baseline", "candidate")
_SPLITS = ("train", "eval")


class _ScratchFull(StreamingArtifactError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetSplitScan:
    experiment: str
    split: str
    path: RepositoryPath
    artifact_sha256: Sha256Digest
    artifact_size: int
    source_trace_sha256: Sha256Digest
    completeness: ScanCompleteness
    indexed_records: int

    def __post_init__(self) -> None:
        if self.experiment not in _EXPERIMENTS or self.split not in _SPLITS:
            raise AnalysisContractError("dataset scan role is invalid")
        if not isinstance(self.path, RepositoryPath):
            raise TypeError("dataset scan path must be RepositoryPath")
        if not isinstance(self.artifact_sha256, Sha256Digest):
            raise TypeError("dataset scan sha256 must be Sha256Digest")
        if not isinstance(self.source_trace_sha256, Sha256Digest):
            raise TypeError("dataset scan trace sha256 must be Sha256Digest")
        if type(self.completeness) is not ScanCompleteness:
            raise TypeError("dataset scan completeness is invalid")
        if (
            isinstance(self.artifact_size, bool)
            or not isinstance(self.artifact_size, int)
            or self.artifact_size < 0
            or self.artifact_size != self.completeness.expected_bytes
        ):
            raise AnalysisContractError("dataset scan artifact size is invalid")
        if (
            isinstance(self.indexed_records, bool)
            or not isinstance(self.indexed_records, int)
            or self.indexed_records < 0
            or self.indexed_records > self.completeness.records_scanned
        ):
            raise AnalysisContractError("dataset indexed record count is invalid")
        if not self.completeness.integrity_verified:
            raise AnalysisContractError(
                "dataset scan facts require verified artifact integrity"
            )


@dataclass(frozen=True, slots=True, init=False)
class DatasetMultisetIdentity:
    record_count: int
    canonical_sha256: str

    def __init__(self) -> None:
        raise TypeError("DatasetMultisetIdentity requires a complete dataset scan")

    @classmethod
    def _from_store(
        cls,
        *,
        record_count: int,
        canonical_sha256: str,
    ) -> DatasetMultisetIdentity:
        if (
            isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count < 1
        ):
            raise AnalysisContractError("dataset identity record count is invalid")
        if (
            not isinstance(canonical_sha256, str)
            or len(canonical_sha256) != 64
            or any(character not in "0123456789abcdef" for character in canonical_sha256)
        ):
            raise AnalysisContractError("dataset identity digest is invalid")
        instance = object.__new__(cls)
        object.__setattr__(instance, "record_count", record_count)
        object.__setattr__(instance, "canonical_sha256", canonical_sha256)
        return instance


@dataclass(frozen=True, slots=True)
class DatasetOverlapFact:
    experiment: str
    train_scan: DatasetSplitScan
    eval_scan: DatasetSplitScan
    train_count: int
    eval_count: int
    overlap_count: int
    overlapping_hashes: tuple[str, ...]
    overlapping_distinct_count: int

    @property
    def complete(self) -> bool:
        return (
            self.train_scan.completeness.complete
            and self.eval_scan.completeness.complete
        )


@dataclass(frozen=True, slots=True)
class DatasetAlignmentFact:
    baseline: DatasetMultisetIdentity
    candidate: DatasetMultisetIdentity
    matching_count: int
    baseline_only_count: int
    candidate_only_count: int
    baseline_only: tuple[tuple[str, int], ...]
    candidate_only: tuple[tuple[str, int], ...]


class _DatasetSpill:
    def __init__(self, scratch_root: Path, maximum_bytes: int) -> None:
        raw_root = Path(scratch_root)
        try:
            raw_stat = os.lstat(raw_root)
            root = raw_root.resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise StreamingArtifactError("dataset scratch root is unavailable") from error
        if stat.S_ISLNK(raw_stat.st_mode) or not root.is_dir():
            raise StreamingArtifactError("dataset scratch root is not trusted")
        self.root = root
        self.maximum_bytes = maximum_bytes
        self.peak_bytes = 0
        self.exhausted = False
        self.path = root / f"claimci-dataset-{secrets.token_hex(16)}.sqlite3"
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0),
            0o600,
        )
        os.close(descriptor)
        try:
            os.chmod(self.path, 0o600)
            self.connection = sqlite3.connect(self.path)
            self.connection.execute("PRAGMA page_size = 4096")
            page_budget = max(1, maximum_bytes // 4096)
            self.connection.execute(f"PRAGMA max_page_count = {page_budget:d}")
            self.connection.execute("PRAGMA journal_mode = MEMORY")
            self.connection.execute("PRAGMA temp_store = MEMORY")
            self.connection.execute(_SCHEMA_SQL)
            self.connection.commit()
            self._measure()
        except sqlite3.DatabaseError as error:
            self.close()
            raise _ScratchFull(
                "dataset spill scratch budget cannot hold the fixed schema"
            ) from error
        except Exception:
            self.close()
            raise

    def _measure(self) -> int:
        size = sum(
            path.stat().st_size
            for path in self._paths()
            if path.exists()
        )
        self.peak_bytes = max(self.peak_bytes, size)
        if size > self.maximum_bytes:
            self.exhausted = True
            raise _ScratchFull("dataset spill exceeded its scratch budget")
        return size

    def _paths(self) -> tuple[Path, ...]:
        return (
            self.path,
            Path(str(self.path) + "-journal"),
            Path(str(self.path) + "-wal"),
            Path(str(self.path) + "-shm"),
        )

    def add_batch(
        self,
        experiment: str,
        split: str,
        digests: list[str],
    ) -> int:
        if not digests:
            return 0
        if self.exhausted:
            raise _ScratchFull("dataset spill scratch budget is exhausted")
        counts = Counter(digests)
        parameters = tuple(
            (experiment, split, digest, multiplicity)
            for digest, multiplicity in sorted(counts.items())
        )
        try:
            self.connection.executemany(_UPSERT_SQL, parameters)
            self.connection.commit()
            self._measure()
        except (sqlite3.DatabaseError, _ScratchFull) as error:
            self.connection.rollback()
            self.exhausted = True
            raise _ScratchFull("dataset spill scratch budget is exhausted") from error
        return len(digests)

    def delete_split(self, experiment: str, split: str) -> None:
        self.connection.execute(_DELETE_SPLIT_SQL, (experiment, split))
        self.connection.commit()

    def count(self, experiment: str, split: str) -> int:
        row = self.connection.execute(_COUNT_SQL, (experiment, split)).fetchone()
        assert row is not None
        return int(row[0])

    def multiset_identity(
        self,
        experiment: str,
        split: str,
    ) -> DatasetMultisetIdentity:
        digest = hashlib.sha256()
        digest.update(b"[")
        first = True
        record_count = 0
        for raw_digest, multiplicity in self.connection.execute(
            _MULTISET_SQL,
            (experiment, split),
        ):
            if not first:
                digest.update(b",")
            material = json.dumps(
                {"sha256": raw_digest, "count": multiplicity},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            digest.update(material)
            first = False
            record_count += int(multiplicity)
        digest.update(b"]")
        return DatasetMultisetIdentity._from_store(
            record_count=record_count,
            canonical_sha256=digest.hexdigest(),
        )

    def overlap_fact(
        self,
        experiment: str,
        train_scan: DatasetSplitScan,
        eval_scan: DatasetSplitScan,
        preview_limit: int,
    ) -> DatasetOverlapFact:
        row = self.connection.execute(_OVERLAP_COUNT_SQL, (experiment,)).fetchone()
        assert row is not None
        hashes = tuple(
            item[0]
            for item in self.connection.execute(
                _OVERLAP_PREVIEW_SQL,
                (experiment, preview_limit),
            )
        )
        return DatasetOverlapFact(
            experiment=experiment,
            train_scan=train_scan,
            eval_scan=eval_scan,
            train_count=self.count(experiment, "train"),
            eval_count=self.count(experiment, "eval"),
            overlap_count=int(row[0]),
            overlapping_hashes=hashes,
            overlapping_distinct_count=int(row[1]),
        )

    def alignment_fact(
        self,
        baseline: DatasetMultisetIdentity,
        candidate: DatasetMultisetIdentity,
        preview_limit: int,
    ) -> DatasetAlignmentFact:
        matching = self.connection.execute(_MATCHING_EVAL_SQL).fetchone()
        baseline_only = self.connection.execute(
            _SIDE_ONLY_COUNT_SQL,
            ("candidate", "baseline"),
        ).fetchone()
        candidate_only = self.connection.execute(
            _SIDE_ONLY_COUNT_SQL,
            ("baseline", "candidate"),
        ).fetchone()
        assert matching is not None and baseline_only is not None and candidate_only is not None

        def preview(source: str, other: str) -> tuple[tuple[str, int], ...]:
            return tuple(
                (row[0], int(row[1]))
                for row in self.connection.execute(
                    _SIDE_ONLY_PREVIEW_SQL,
                    (other, source, preview_limit),
                )
            )

        return DatasetAlignmentFact(
            baseline=baseline,
            candidate=candidate,
            matching_count=int(matching[0]),
            baseline_only_count=int(baseline_only[0]),
            candidate_only_count=int(candidate_only[0]),
            baseline_only=preview("baseline", "candidate"),
            candidate_only=preview("candidate", "baseline"),
        )

    def close(self) -> None:
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None
        path = getattr(self, "path", None)
        if path is not None:
            for target in self._paths():
                target.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True, init=False)
class DatasetScanAuditContext:
    repository: RepositoryIdentity
    head_sha: GitCommitSha
    scans: tuple[DatasetSplitScan, ...]
    _store: _DatasetSpill = field(repr=False, compare=False)
    _active: bool = field(repr=False, compare=False)
    _preview_limit: int = field(repr=False, compare=False)
    _manifest_paths: tuple[tuple[str, str, RepositoryPath], ...] = field(
        repr=False,
        compare=False,
    )

    def __init__(self) -> None:
        raise TypeError(
            "DatasetScanAuditContext requires verified streaming materialization"
        )

    @classmethod
    def _from_scans(
        cls,
        *,
        repository: RepositoryIdentity,
        head_sha: GitCommitSha,
        scans: tuple[DatasetSplitScan, ...],
        store: _DatasetSpill,
        preview_limit: int,
        manifest_paths: tuple[tuple[str, str, RepositoryPath], ...],
    ) -> DatasetScanAuditContext:
        instance = object.__new__(cls)
        object.__setattr__(instance, "repository", repository)
        object.__setattr__(instance, "head_sha", head_sha)
        object.__setattr__(instance, "scans", scans)
        object.__setattr__(instance, "_store", store)
        object.__setattr__(instance, "_active", True)
        object.__setattr__(instance, "_preview_limit", preview_limit)
        object.__setattr__(instance, "_manifest_paths", manifest_paths)
        return instance

    def _require_active(self) -> None:
        if not self._active:
            raise AnalysisContractError("dataset Audit context is no longer active")

    def scan(self, experiment: str, split: str) -> DatasetSplitScan:
        self._require_active()
        matches = tuple(
            item
            for item in self.scans
            if item.experiment == experiment and item.split == split
        )
        if len(matches) != 1:
            raise AnalysisContractError("dataset Audit scan key is invalid")
        return matches[0]

    def overlap_fact(self, experiment: str) -> DatasetOverlapFact:
        self._require_active()
        return self._store.overlap_fact(
            experiment,
            self.scan(experiment, "train"),
            self.scan(experiment, "eval"),
            self._preview_limit,
        )

    def evaluation_identity(
        self,
        experiment: str,
    ) -> DatasetMultisetIdentity | None:
        self._require_active()
        scan = self.scan(experiment, "eval")
        if not scan.completeness.complete:
            return None
        return self._store.multiset_identity(experiment, "eval")

    def evaluation_alignment_fact(self) -> DatasetAlignmentFact | None:
        self._require_active()
        baseline = self.evaluation_identity("baseline")
        candidate = self.evaluation_identity("candidate")
        if baseline is None or candidate is None:
            return None
        return self._store.alignment_fact(
            baseline,
            candidate,
            self._preview_limit,
        )

    def validate_paths(
        self,
        root: Path,
        paths: dict[tuple[str, str], Path],
    ) -> None:
        self._require_active()
        resolved_root = Path(root).resolve(strict=True)
        expected = {
            (experiment, split): path
            for experiment, split, path in self._manifest_paths
        }
        if set(paths) != set(expected):
            raise ClaimCIError(
                "dataset scan context does not match the confined manifest"
            )
        for key, path in paths.items():
            try:
                relative = Path(path).resolve(strict=False).relative_to(
                    resolved_root
                ).as_posix()
            except (OSError, RuntimeError, ValueError) as error:
                raise ClaimCIError(
                    "dataset scan context does not match the confined manifest"
                ) from error
            if expected[key] != RepositoryPath(relative):
                raise ClaimCIError(
                    "dataset scan context does not match the confined manifest"
                )

    def _deactivate(self) -> None:
        object.__setattr__(self, "_active", False)


def _parse_dataset_value(raw: bytes, limits: StreamingLimits) -> object:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise _ScanStop(ScanState.FAILED, ScanReason.INVALID_UTF8) from error

    def reject_constant(_token: str) -> object:
        raise ValueError("non-finite dataset value")

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_json_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED) from error
    _validate_record_graph(value, limits)
    return value


def _flush(
    store: _DatasetSpill,
    experiment: str,
    split: str,
    pending: list[str],
) -> int:
    indexed = store.add_batch(experiment, split, pending)
    pending.clear()
    return indexed


def _scan_dataset(
    source: ArtifactSource,
    *,
    experiment: str,
    split: str,
    store: _DatasetSpill,
    limits: StreamingLimits,
) -> DatasetSplitScan:
    if type(source) is not ArtifactSource:
        raise TypeError("streaming dataset inputs must be ArtifactSource")
    if (
        source.candidate.kind is not ArtifactKind.DATASET
        or PurePosixPath(str(source.candidate.path)).suffix.casefold() != ".jsonl"
    ):
        raise AnalysisContractError(
            "streaming dataset input must be a fixed JSONL dataset source"
        )
    records = 0
    indexed = 0
    pending: list[str] = []
    with source.open_scan(ScanPurpose.DATASET_RECORDS, limits) as scan:
        try:
            if store.exhausted:
                raise _ScanStop(
                    ScanState.INCOMPLETE_BUDGET,
                    ScanReason.SCRATCH_BUDGET,
                )
            for raw, physical_bytes in _binary_lines(
                scan,
                maximum=limits.max_logical_record_bytes,
            ):
                if scan.semantic_bytes + physical_bytes > limits.max_semantic_bytes:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.SEMANTIC_BYTE_BUDGET,
                    )
                if not raw.strip():
                    scan.note_semantic_bytes(physical_bytes, buffered=len(raw))
                    continue
                if records >= limits.max_semantic_records:
                    raise _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.RECORD_BUDGET,
                    )
                scan.note_semantic_bytes(physical_bytes, buffered=len(raw))
                value = _parse_dataset_value(raw, limits)
                pending.append(hashlib.sha256(canonicalize_sample(value)).hexdigest())
                records += 1
                if len(pending) >= _BATCH_RECORDS:
                    indexed += _flush(store, experiment, split, pending)
            indexed += _flush(store, experiment, split, pending)
            if records == 0:
                raise _ScanStop(ScanState.FAILED, ScanReason.MALFORMED)
            report = scan.finish(
                records_scanned=records,
                semantic_complete=True,
                scratch_bytes=store.peak_bytes,
            )
        except _ScratchFull:
            pending.clear()
            report = scan.finish(
                records_scanned=records,
                semantic_complete=False,
                state=ScanState.INCOMPLETE_BUDGET,
                reason=ScanReason.SCRATCH_BUDGET,
                scratch_bytes=min(store.peak_bytes, limits.max_scratch_bytes),
            )
        except _ScanStop as stop:
            if stop.state is ScanState.FAILED:
                pending.clear()
                store.delete_split(experiment, split)
                indexed = 0
            else:
                try:
                    indexed += _flush(store, experiment, split, pending)
                except _ScratchFull:
                    stop = _ScanStop(
                        ScanState.INCOMPLETE_BUDGET,
                        ScanReason.SCRATCH_BUDGET,
                    )
            report = scan.finish(
                records_scanned=records,
                semantic_complete=False,
                state=stop.state,
                reason=stop.reason,
                scratch_bytes=min(store.peak_bytes, limits.max_scratch_bytes),
            )
        except (ClaimCIError, AnalysisContractError, TypeError, ValueError):
            pending.clear()
            store.delete_split(experiment, split)
            indexed = 0
            report = scan.finish(
                records_scanned=records,
                semantic_complete=False,
                state=ScanState.FAILED,
                reason=ScanReason.MALFORMED,
                scratch_bytes=min(store.peak_bytes, limits.max_scratch_bytes),
            )
    return DatasetSplitScan(
        experiment=experiment,
        split=split,
        path=source.candidate.path,
        artifact_sha256=source.candidate.sha256,
        artifact_size=source.candidate.size,
        source_trace_sha256=Sha256Digest(scan.trace_sha256),
        completeness=report,
        indexed_records=indexed,
    )


@contextmanager
def stream_dataset_audit_context(
    *,
    baseline_train: ArtifactSource,
    baseline_eval: ArtifactSource,
    candidate_train: ArtifactSource,
    candidate_eval: ArtifactSource,
    scratch_root: Path,
    limits: StreamingLimits = StreamingLimits(),
    manifest_paths: Mapping[tuple[str, str], RepositoryPath] | None = None,
) -> Iterator[DatasetScanAuditContext]:
    """Yield a live factory-only Audit context and always remove its spill."""

    if type(limits) is not StreamingLimits:
        raise TypeError("streaming dataset limits must be StreamingLimits")
    keyed = (
        ("baseline", "train", baseline_train),
        ("baseline", "eval", baseline_eval),
        ("candidate", "train", candidate_train),
        ("candidate", "eval", candidate_eval),
    )
    if any(type(source) is not ArtifactSource for _, _, source in keyed):
        raise TypeError("all streaming dataset inputs must be ArtifactSource")
    repositories = {source.repository for _, _, source in keyed}
    heads = {source.head_sha for _, _, source in keyed}
    if len(repositories) != 1 or len(heads) != 1:
        raise AnalysisContractError(
            "streaming dataset sources must share one repository and exact head"
        )
    source_paths = {
        (experiment, split): source.candidate.path
        for experiment, split, source in keyed
    }
    if manifest_paths is None:
        fixed_manifest_paths = source_paths
    else:
        if not isinstance(manifest_paths, Mapping) or set(manifest_paths) != set(
            source_paths
        ):
            raise AnalysisContractError(
                "dataset manifest aliases must cover every exact scan source"
            )
        fixed_manifest_paths = {
            key: (
                value
                if isinstance(value, RepositoryPath)
                else RepositoryPath(value)
            )
            for key, value in manifest_paths.items()
        }
        if len(set(fixed_manifest_paths.values())) != len(fixed_manifest_paths):
            raise AnalysisContractError("dataset manifest aliases must be unique")
    store = _DatasetSpill(Path(scratch_root), limits.max_scratch_bytes)
    context: DatasetScanAuditContext | None = None
    try:
        scans = tuple(
            _scan_dataset(
                source,
                experiment=experiment,
                split=split,
                store=store,
                limits=limits,
            )
            for experiment, split, source in keyed
        )
        context = DatasetScanAuditContext._from_scans(
            repository=next(iter(repositories)),
            head_sha=next(iter(heads)),
            scans=scans,
            store=store,
            preview_limit=limits.max_hash_previews,
            manifest_paths=tuple(
                (experiment, split, fixed_manifest_paths[(experiment, split)])
                for experiment, split, _source in keyed
            ),
        )
        yield context
    finally:
        if context is not None:
            context._deactivate()
        store.close()


__all__ = [
    "DatasetAlignmentFact",
    "DatasetMultisetIdentity",
    "DatasetOverlapFact",
    "DatasetScanAuditContext",
    "DatasetSplitScan",
    "stream_dataset_audit_context",
]
