from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import claimci.analysis as analysis
from claimci.analysis import (
    AnalysisContractError,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    FieldProvenance,
    GitCommitSha,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    Sha256Digest,
)
from claimci.analysis.artifact_source import (
    ArtifactSource,
    ScanCompleteness,
    ScanPurpose,
    ScanReason,
    ScanState,
    StreamingLimits,
    StreamingIntegrityError,
    StreamingLimitError,
    artifact_source_from_snapshot,
)
from claimci.analysis.discovery.models import DiscoveryLimits
from claimci.analysis.discovery.repository import (
    ArtifactInspectionBudget,
    collect_repository_context,
    inspect_artifact,
)
from claimci.passive_files import inspect_confined_regular_file


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI")
HEAD = GitCommitSha("3" * 40)


def _candidate(content: bytes, *, path: str = "results/eval.jsonl") -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=ArtifactKind.RESULTS,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="streaming source fixture",
        relevant_claim_ids=("claim-streaming",),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "streaming source fixture",
            RepositoryPath(path),
            "fixture:streaming-source",
        ),
    )


def _source(tmp_path: Path, content: bytes) -> ArtifactSource:
    target = tmp_path / "results" / "eval.jsonl"
    target.parent.mkdir()
    target.write_bytes(content)
    return artifact_source_from_snapshot(
        REPOSITORY,
        HEAD,
        tmp_path,
        _candidate(content),
    )


def test_trusted_source_is_factory_only_and_verifies_incremental_integrity(
    tmp_path: Path,
) -> None:
    content = b'{"run":1,"acc":0.9}\n' * 10_000
    source = _source(tmp_path, content)

    with pytest.raises(TypeError, match="trusted snapshot factory"):
        ArtifactSource()

    with source.open_scan(
        ScanPurpose.INTEGRITY_ONLY,
        StreamingLimits(max_integrity_bytes=len(content)),
    ) as scan:
        while scan.read_chunk(64 * 1024):
            pass
        report = scan.finish(records_scanned=0, semantic_complete=True)

    assert source.repository == REPOSITORY
    assert source.head_sha == HEAD
    assert source.candidate == _candidate(content)
    assert report.purpose is ScanPurpose.INTEGRITY_ONLY
    assert report.state is ScanState.COMPLETE
    assert report.reason is ScanReason.COMPLETE
    assert report.integrity_verified is True
    assert report.integrity_bytes == len(content)
    assert report.expected_bytes == len(content)
    assert scan.maximum_read_request == 64 * 1024


def test_confined_inspection_hashes_large_file_without_returning_content(
    tmp_path: Path,
) -> None:
    content = b"0123456789abcdef" * 1_000_000
    target = tmp_path / "results" / "large.jsonl"
    target.parent.mkdir()
    target.write_bytes(content)

    inspection = inspect_confined_regular_file(
        tmp_path,
        "results/large.jsonl",
        max_bytes=len(content),
    )

    assert inspection.size == len(content)
    assert inspection.sha256 == hashlib.sha256(content).hexdigest()
    assert not hasattr(inspection, "content")


def test_discovery_streams_supported_line_artifact_above_bytes_threshold(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    content = b'{"run":1,"accuracy":0.9}\n' * 100_000
    target = head / "results" / "large.jsonl"
    target.parent.mkdir()
    target.write_bytes(content)
    limits = DiscoveryLimits(
        max_artifact_bytes=1024 * 1024,
        max_stream_artifact_bytes=4 * 1024 * 1024,
        max_stream_total_bytes=4 * 1024 * 1024,
    )
    context = collect_repository_context(head, limits=limits)

    inspection = inspect_artifact(
        context,
        RepositoryPath("results/large.jsonl"),
        limits=limits,
    )

    assert inspection.size == len(content)
    assert inspection.sha256 == hashlib.sha256(content).hexdigest()


def test_discovery_rejects_stream_before_exceeding_aggregate_integrity_budget(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    content = b'{"accuracy":0.9}\n' * 100_000
    target = head / "large.jsonl"
    target.write_bytes(content)
    limits = DiscoveryLimits(
        max_artifact_bytes=1024,
        max_stream_artifact_bytes=len(content),
        max_stream_total_bytes=len(content),
    )
    context = collect_repository_context(head, limits=limits)
    budget = ArtifactInspectionBudget(remaining_bytes=len(content) - 1)

    outcome = inspect_artifact(
        context,
        RepositoryPath("large.jsonl"),
        limits=limits,
        budget=budget,
    )

    assert outcome.reason == "artifact exceeds the streaming discovery work budget"
    assert budget.remaining_bytes == len(content) - 1


def test_streaming_contract_is_exported_from_analysis_package() -> None:
    assert analysis.ArtifactSource is ArtifactSource
    assert analysis.ScanCompleteness is ScanCompleteness
    assert analysis.artifact_source_from_snapshot is artifact_source_from_snapshot


def test_source_fails_closed_for_exact_size_sha_mismatch(tmp_path: Path) -> None:
    expected = b'{"accuracy":0.9}\n'
    source = _source(tmp_path, expected)
    (tmp_path / "results" / "eval.jsonl").write_bytes(b'{"accuracy":0.8}\n')

    with source.open_scan(ScanPurpose.SCHEMA) as scan:
        while scan.read_chunk():
            pass
        with pytest.raises(StreamingIntegrityError, match="sha256 mismatch"):
            scan.finish(records_scanned=1, semantic_complete=True)


def test_source_rejects_size_change_before_open(tmp_path: Path) -> None:
    content = b'{"accuracy":0.9}\n'
    source = _source(tmp_path, content)
    (tmp_path / "results" / "eval.jsonl").write_bytes(content + b"x")

    with pytest.raises(StreamingIntegrityError, match="size changed"):
        source.open_scan(ScanPurpose.SCHEMA)


def test_semantic_budget_stays_incomplete_after_integrity_drain(tmp_path: Path) -> None:
    content = b'{"accuracy":0.9}\n' * 100
    source = _source(tmp_path, content)
    limits = StreamingLimits(
        max_integrity_bytes=len(content),
        max_semantic_bytes=64,
    )

    with source.open_scan(ScanPurpose.SCHEMA, limits) as scan:
        prefix = scan.read_chunk(64)
        scan.note_semantic_bytes(len(prefix), buffered=len(prefix))
        report = scan.finish(
            records_scanned=3,
            semantic_complete=False,
            state=ScanState.INCOMPLETE_BUDGET,
            reason=ScanReason.SEMANTIC_BYTE_BUDGET,
        )

    assert report.integrity_verified is True
    assert report.integrity_bytes == len(content)
    assert report.semantic_bytes == 64
    assert report.state is ScanState.INCOMPLETE_BUDGET
    assert report.complete is False


def test_source_integrity_budget_exhaustion_never_returns_evidence(
    tmp_path: Path,
) -> None:
    content = b'{"accuracy":0.9}\n' * 100
    source = _source(tmp_path, content)

    with source.open_scan(
        ScanPurpose.INTEGRITY_ONLY,
        StreamingLimits(max_integrity_bytes=len(content) - 1),
    ) as scan:
        with pytest.raises(StreamingLimitError, match="integrity-I/O budget"):
            scan.finish(records_scanned=0, semantic_complete=True)


def test_scan_completeness_rejects_false_complete_and_semantic_overread() -> None:
    with pytest.raises(AnalysisContractError, match="complete scan requires"):
        ScanCompleteness(
            purpose=ScanPurpose.SCHEMA,
            state=ScanState.COMPLETE,
            reason=ScanReason.COMPLETE,
            records_scanned=1,
            semantic_bytes=1,
            integrity_bytes=1,
            expected_bytes=1,
            integrity_verified=False,
            peak_buffer_bytes=1,
        )

    with pytest.raises(AnalysisContractError, match="semantic bytes"):
        ScanCompleteness(
            purpose=ScanPurpose.SCHEMA,
            state=ScanState.INCOMPLETE_BUDGET,
            reason=ScanReason.SEMANTIC_BYTE_BUDGET,
            records_scanned=1,
            semantic_bytes=2,
            integrity_bytes=1,
            expected_bytes=1,
            integrity_verified=True,
            peak_buffer_bytes=1,
        )
