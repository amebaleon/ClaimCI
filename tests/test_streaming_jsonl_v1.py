from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from claimci.analysis import (
    AdapterMatch,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    ScanPurpose,
    ScanReason,
    ScanState,
    Sha256Digest,
    StreamingLimits,
    approve_metric_binding,
    artifact_source_from_snapshot,
    extract_bound_metric_evidence,
    extract_metric_candidate_scan,
    extract_metric_candidates,
    resolve_metric_binding,
)
from claimci.analysis.adapters import (
    MAX_LOGICAL_RECORD_BYTES,
    extract_registered_source,
    scan_jsonl_observations,
    scan_jsonl_schema,
)


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI")
HEAD = GitCommitSha("4" * 40)


def _candidate(
    content: bytes,
    *,
    path: str = "results/runs.jsonl",
    kind: ArtifactKind = ArtifactKind.RESULTS,
) -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="streaming JSONL fixture",
        relevant_claim_ids=("claim-streaming",),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "streaming JSONL fixture",
            RepositoryPath(path),
            "fixture:streaming-jsonl",
        ),
    )


def _source(
    tmp_path: Path,
    content: bytes,
    *,
    path: str = "results/runs.jsonl",
    kind: ArtifactKind = ArtifactKind.RESULTS,
):
    target = tmp_path.joinpath(*RepositoryPath(path).split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return artifact_source_from_snapshot(
        REPOSITORY,
        HEAD,
        tmp_path,
        _candidate(content, path=path, kind=kind),
    )


def _selected_match(schema, raw_metric_name: str) -> AdapterMatch:
    candidate = next(
        item for item in schema.candidates if item.raw_metric_name == raw_metric_name
    )
    support = tuple(
        item
        for item in schema.match.mappings
        if item.target_field in {"metric_name", "run_id", "seed"}
    )
    return AdapterMatch(
        schema.match.adapter_id,
        schema.match.path,
        schema.match.confidence,
        (
            *support,
            FieldMapping("metric_value", candidate.selector, candidate.provenance),
        ),
        schema.match.match_evidence,
    )


def test_large_jsonl_discovers_schema_candidates_without_retaining_records(
    tmp_path: Path,
) -> None:
    padding = "x" * 700
    content = b"".join(
        json.dumps(
            {"run": index, "acc": 0.8, "loss": 0.2, "padding": padding},
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for index in range(12_500)
    )
    assert len(content) > 8 * 1024 * 1024
    source = _source(tmp_path, content)

    result = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.BASELINE,
    )

    assert result.completeness.state is ScanState.COMPLETE
    assert result.completeness.integrity_verified is True
    assert [item.raw_metric_name for item in result.candidates] == ["acc", "loss"]
    assert result.completeness.peak_buffer_bytes <= MAX_LOGICAL_RECORD_BYTES
    assert result.retained_raw_records == 0


def test_incomplete_jsonl_never_issues_complete_candidates(tmp_path: Path) -> None:
    content = b"".join(
        f'{{"run":{index},"acc":0.9}}\n'.encode() for index in range(20)
    )
    source = _source(tmp_path, content)

    outcome = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.BASELINE,
        limits=StreamingLimits(max_semantic_records=10),
    )

    assert outcome.candidates == ()
    assert outcome.completeness.state is ScanState.INCOMPLETE_BUDGET
    assert outcome.completeness.reason is ScanReason.RECORD_BUDGET
    assert outcome.completeness.integrity_verified is True
    assert outcome.completeness.records_scanned == 10


def test_stream_and_bytes_metric_candidates_have_exact_identity_parity(
    tmp_path: Path,
) -> None:
    content = (
        b'{"seed":1,"acc":0.81,"loss":0.4}\n'
        b'{"seed":2,"acc":0.83,"loss":0.3}\n'
    )
    source = _source(tmp_path, content)
    passive = PassiveArtifact(source.candidate, content)

    streamed = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.CANDIDATE,
    )
    bounded = extract_metric_candidates(
        passive,
        role=ExperimentRole.CANDIDATE,
    )

    assert streamed.candidates == bounded
    assert streamed.completeness.state is ScanState.COMPLETE


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b'{"acc":0.8}\n\n{"acc":0.9}\n', ScanReason.MALFORMED),
        (b'{"acc":0.8,"acc":0.9}\n', ScanReason.MALFORMED),
        (b'{"acc":NaN}\n', ScanReason.MALFORMED),
        (b'{"acc":"\xff"}\n', ScanReason.INVALID_UTF8),
    ],
)
def test_malformed_jsonl_fails_without_provisional_candidates(
    tmp_path: Path,
    content: bytes,
    reason: ScanReason,
) -> None:
    source = _source(tmp_path, content)

    outcome = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.BASELINE,
    )

    assert outcome.candidates == ()
    assert outcome.completeness.state is ScanState.FAILED
    assert outcome.completeness.reason is reason
    assert outcome.completeness.integrity_verified is True


def test_logical_record_limit_and_candidate_limit_are_incomplete_not_complete(
    tmp_path: Path,
) -> None:
    long_source = _source(
        tmp_path / "long",
        b'{"padding":"' + (b"x" * 65) + b'","acc":0.9}\n',
    )
    long_outcome = extract_metric_candidate_scan(
        long_source,
        role=ExperimentRole.BASELINE,
        limits=StreamingLimits(max_logical_record_bytes=64),
    )
    assert long_outcome.candidates == ()
    assert long_outcome.completeness.state is ScanState.INCOMPLETE_LIMIT
    assert long_outcome.completeness.reason is ScanReason.LOGICAL_RECORD_LIMIT

    exploding = {
        f"metric_{index}": index / 10 for index in range(5)
    }
    candidate_source = _source(
        tmp_path / "candidates",
        json.dumps(exploding, separators=(",", ":")).encode() + b"\n",
    )
    candidate_outcome = extract_metric_candidate_scan(
        candidate_source,
        role=ExperimentRole.BASELINE,
        limits=StreamingLimits(max_metric_candidates=4),
    )
    assert candidate_outcome.candidates == ()
    assert candidate_outcome.completeness.state is ScanState.INCOMPLETE_LIMIT
    assert candidate_outcome.completeness.reason is ScanReason.CANDIDATE_LIMIT


def test_selected_observations_require_a_complete_scan(tmp_path: Path) -> None:
    content = b"".join(
        f'{{"run":{index},"acc":0.{index + 1}}}\n'.encode()
        for index in range(5)
    )
    source = _source(tmp_path, content)
    schema = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.CANDIDATE,
    )
    match = _selected_match(schema, "acc")

    complete = scan_jsonl_observations(source, match)
    incomplete = scan_jsonl_observations(
        source,
        match,
        limits=StreamingLimits(max_observations=3),
    )

    assert complete.evidence is not None
    assert len(complete.evidence.observations) == 5
    assert complete.evidence.scan_completeness == complete.completeness
    assert complete.completeness.state is ScanState.COMPLETE
    assert incomplete.evidence is None
    assert incomplete.completeness.state is ScanState.INCOMPLETE_BUDGET
    assert incomplete.completeness.reason is ScanReason.OBSERVATION_BUDGET
    assert incomplete.completeness.integrity_verified is True


def test_registry_streams_dataset_identity_without_claiming_record_completeness(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path,
        b'{"prompt":"a"}\n{"prompt":"b"}\n',
        path="data/train.jsonl",
        kind=ArtifactKind.DATASET,
    )

    outcome = extract_registered_source(source)

    assert outcome is not None
    assert outcome.evidence is not None
    assert outcome.completeness.purpose is ScanPurpose.INTEGRITY_ONLY
    assert outcome.completeness.state is ScanState.COMPLETE
    assert outcome.evidence.scan_completeness == outcome.completeness


def test_schema_scan_is_fixed_jsonl_registry_data(tmp_path: Path) -> None:
    source = _source(tmp_path, b'{"acc":0.9}\n')

    outcome = scan_jsonl_schema(source)

    assert outcome.match is not None
    assert outcome.match.adapter_id == "claimci-jsonl-v1"
    assert outcome.retained_raw_records == 0


@pytest.mark.parametrize(
    ("content", "limits", "reason"),
    [
        (
            b'{"outer":{"inner":{"acc":0.9}}}\n',
            StreamingLimits(max_depth=2),
            ScanReason.DEPTH_LIMIT,
        ),
        (
            b'{"acc":0.9,"loss":0.2}\n',
            StreamingLimits(max_nodes_per_record=4),
            ScanReason.NODE_LIMIT,
        ),
        (
            b'{"acc":0.9}\n{"acc":0.8}\n',
            StreamingLimits(max_semantic_bytes=5),
            ScanReason.SEMANTIC_BYTE_BUDGET,
        ),
    ],
)
def test_jsonl_per_record_and_semantic_work_bounds_fail_closed(
    tmp_path: Path,
    content: bytes,
    limits: StreamingLimits,
    reason: ScanReason,
) -> None:
    outcome = extract_metric_candidate_scan(
        _source(tmp_path, content),
        role=ExperimentRole.BASELINE,
        limits=limits,
    )

    assert outcome.candidates == ()
    assert outcome.completeness.reason is reason
    assert outcome.completeness.integrity_verified is True


def test_bound_metric_source_revalidates_then_canonicalizes_after_complete_scan(
    tmp_path: Path,
) -> None:
    baseline_source = _source(
        tmp_path,
        b'{"run":1,"acc":0.71}\n{"run":2,"acc":0.72}\n',
        path="results/baseline.jsonl",
    )
    candidate_source = _source(
        tmp_path,
        b'{"run":1,"acc":0.81}\n{"run":2,"acc":0.82}\n',
        path="results/candidate.jsonl",
    )
    baseline = extract_metric_candidate_scan(
        baseline_source,
        role=ExperimentRole.BASELINE,
    )
    candidate = extract_metric_candidate_scan(
        candidate_source,
        role=ExperimentRole.CANDIDATE,
    )
    pending = resolve_metric_binding(
        (*baseline.candidates, *candidate.candidates),
        repository=REPOSITORY,
        head_sha=HEAD,
        canonical_metric="accuracy",
    )
    assert pending.question is not None
    binding = approve_metric_binding(
        pending.question,
        proposal_id=pending.question.proposals[0].proposal_id,
        approved_by="owner:amebaleon",
    )

    evidence = extract_bound_metric_evidence(
        baseline_source,
        binding,
        role=ExperimentRole.BASELINE,
    )

    assert evidence.scan_completeness is not None
    assert evidence.scan_completeness.state is ScanState.COMPLETE
    assert {item.metric_name for item in evidence.observations} == {"accuracy"}
