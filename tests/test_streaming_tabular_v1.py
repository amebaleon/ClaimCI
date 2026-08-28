from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    AdapterMatch,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    PassiveArtifact,
    ProvenanceKind,
    RepositoryIdentity,
    RepositoryPath,
    ScanReason,
    ScanState,
    SelectorKind,
    Sha256Digest,
    StreamingLimits,
    TablePredicate,
    TableScalarType,
    TableSelector,
    artifact_source_from_snapshot,
    approve_metric_binding,
    extract_bound_metric_evidence,
    extract_metric_candidate_scan,
    extract_metric_candidates,
    resolve_metric_binding,
)
from claimci.analysis.adapters import (
    CsvAdapter,
    TsvAdapter,
    scan_delimited_observations,
    scan_delimited_schema,
)


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI")
HEAD = GitCommitSha("5" * 40)


def _candidate(content: bytes, path: str) -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=ArtifactKind.RESULTS,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="streaming table fixture",
        relevant_claim_ids=("claim-table",),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "streaming table fixture",
            RepositoryPath(path),
            "fixture:streaming-table",
        ),
    )


def _source(tmp_path: Path, content: bytes, suffix: str):
    path = f"results/runs.{suffix}"
    target = tmp_path / "results" / f"runs.{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return artifact_source_from_snapshot(
        REPOSITORY,
        HEAD,
        tmp_path,
        _candidate(content, path),
    )


def _field_match(source, column: str) -> AdapterMatch:
    adapter_id = f"claimci-{source.candidate.path.rsplit('.', 1)[1]}-v1"
    provenance = FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        f"adapter={adapter_id}; selector={column}; "
        f"sha256={source.candidate.sha256}; inferred=false",
        source.candidate.path,
        f"{adapter_id}:{str(source.candidate.sha256)[:16]}",
    )
    return AdapterMatch(
        adapter_id,
        source.candidate.path,
        Confidence(0.95),
        (
            FieldMapping(
                "metric_value",
                EvidenceSelector(SelectorKind.COLUMN, column, provenance),
                provenance,
            ),
        ),
        (provenance,),
    )


@pytest.mark.parametrize(("suffix", "delimiter"), [("csv", ","), ("tsv", "\t")])
def test_large_table_discovers_numeric_columns_without_row_retention(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
) -> None:
    padding = "x" * 700
    header = delimiter.join(("run", "accuracy", "f1", "loss", "note")) + "\n"
    rows = "".join(
        delimiter.join((str(index), "0.8", "0.7", "0.2", padding)) + "\n"
        for index in range(12_500)
    )
    content = (header + rows).encode()
    assert len(content) > 8 * 1024 * 1024

    outcome = extract_metric_candidate_scan(
        _source(tmp_path, content, suffix),
        role=ExperimentRole.CANDIDATE,
    )

    assert outcome.completeness.state is ScanState.COMPLETE
    assert tuple(item.raw_metric_name for item in outcome.candidates) == (
        "accuracy",
        "f1",
        "loss",
    )
    assert outcome.retained_raw_rows == 0


@pytest.mark.parametrize(("suffix", "delimiter"), [("csv", ","), ("tsv", "\t")])
def test_stream_and_bytes_table_candidates_have_exact_identity_parity(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
) -> None:
    content = (
        f"seed{delimiter}acc{delimiter}loss\n"
        f"1{delimiter}0.81{delimiter}0.4\n"
        f"2{delimiter}0.83{delimiter}0.3\n"
    ).encode()
    source = _source(tmp_path, content, suffix)

    streamed = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.BASELINE,
    )
    bounded = extract_metric_candidates(
        passive_artifact(source.candidate, content),
        role=ExperimentRole.BASELINE,
    )

    assert streamed.candidates == bounded


@pytest.mark.parametrize(("suffix", "delimiter"), [("csv", ","), ("tsv", "\t")])
def test_passive_and_streaming_table_evidence_ids_share_one_v2_selector_identity(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
) -> None:
    content = (
        f"role{delimiter}run{delimiter}seed{delimiter}acc\n"
        f"baseline{delimiter}baseline-run{delimiter}1{delimiter}0.70\n"
        f"candidate{delimiter}candidate-run{delimiter}2{delimiter}0.80\n"
    ).encode()
    source = _source(tmp_path, content, suffix)
    passive = passive_artifact(
        source.candidate,
        content,
        repository=source.repository,
        snapshot_role=source.snapshot_role,
        commit=source.head_sha,
    )
    adapter = CsvAdapter() if suffix == "csv" else TsvAdapter()
    adapter_id = adapter.adapter_id

    def match_for(role: str) -> AdapterMatch:
        provenance = FieldProvenance(
            ProvenanceKind.PROVIDER_PROPOSAL,
            f"shared table selector; sha256={source.candidate.sha256}",
            source.candidate.path,
            f"fixture:table:{role}",
        )
        return AdapterMatch(
            adapter_id,
            source.candidate.path,
            Confidence(0.95),
            (
                FieldMapping(
                    "metric_value",
                    TableSelector(
                        "acc",
                        (TablePredicate("role", TableScalarType.STRING, role),),
                        1,
                        provenance,
                    ),
                    provenance,
                ),
                FieldMapping(
                    "run_id",
                    EvidenceSelector(SelectorKind.COLUMN, "run", provenance),
                    provenance,
                ),
                FieldMapping(
                    "seed",
                    EvidenceSelector(SelectorKind.COLUMN, "seed", provenance),
                    provenance,
                ),
            ),
            (provenance,),
        )

    baseline_match = match_for("baseline")
    candidate_match = match_for("candidate")
    baseline = adapter.extract(passive, baseline_match)
    candidate = adapter.extract(passive, candidate_match)
    streamed = scan_delimited_observations(source, candidate_match)

    assert baseline.evidence_id.startswith("evidence-v2-")
    assert candidate.evidence_id.startswith("evidence-v2-")
    assert baseline.evidence_id != candidate.evidence_id
    assert streamed.evidence is not None
    assert streamed.completeness.state is ScanState.COMPLETE
    assert streamed.evidence.evidence_id == candidate.evidence_id
    assert streamed.evidence.scan_completeness == streamed.completeness


def test_table_selector_requires_complete_exact_cardinality(tmp_path: Path) -> None:
    content = b"split,acc\neval,0.8\neval,0.9\ntrain,0.7\n"
    source = _source(tmp_path, content, "csv")
    provenance = FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        f"exact table selector; sha256={source.candidate.sha256}",
        source.candidate.path,
        "fixture:table-selector",
    )
    selector = TableSelector(
        "acc",
        (TablePredicate("split", TableScalarType.STRING, "eval"),),
        2,
        provenance,
    )
    match = AdapterMatch(
        "claimci-csv-v1",
        source.candidate.path,
        Confidence(0.95),
        (FieldMapping("metric_value", selector, provenance),),
        (provenance,),
    )

    incomplete = scan_delimited_observations(
        source,
        match,
        limits=StreamingLimits(max_semantic_records=1),
    )
    complete = scan_delimited_observations(source, match)

    assert incomplete.evidence is None
    assert incomplete.completeness.state is ScanState.INCOMPLETE_BUDGET
    assert incomplete.completeness.integrity_verified is True
    assert complete.evidence is not None
    assert len(complete.evidence.observations) == 2
    assert complete.evidence.scan_completeness == complete.completeness

    selected_candidate = extract_metric_candidate_scan(
        source,
        role=ExperimentRole.CANDIDATE,
        adapter_match=match,
    )
    assert selected_candidate.completeness.state is ScanState.COMPLETE
    assert len(selected_candidate.candidates) == 1
    assert (
        selected_candidate.candidates[0].selector
        == complete.evidence.adapter_match.mappings[0].selector
    )


def test_table_selector_cardinality_mismatch_never_returns_first_match(
    tmp_path: Path,
) -> None:
    content = b"split,acc\neval,0.8\neval,0.9\n"
    source = _source(tmp_path, content, "csv")
    provenance = FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        f"exact selector; sha256={source.candidate.sha256}",
        source.candidate.path,
        "fixture:cardinality",
    )
    match = AdapterMatch(
        "claimci-csv-v1",
        source.candidate.path,
        Confidence(0.95),
        (
            FieldMapping(
                "metric_value",
                TableSelector(
                    "acc",
                    (TablePredicate("split", TableScalarType.STRING, "eval"),),
                    1,
                    provenance,
                ),
                provenance,
            ),
        ),
        (provenance,),
    )

    outcome = scan_delimited_observations(source, match)

    assert outcome.evidence is None
    assert outcome.completeness.state is ScanState.FAILED
    assert outcome.completeness.reason is ScanReason.SELECTOR_MISMATCH


def test_delimited_column_limit_accepts_256_and_rejects_257(tmp_path: Path) -> None:
    accepted_header = ",".join(f"c{index}" for index in range(256))
    accepted_row = ",".join("1" for _ in range(256))
    accepted = scan_delimited_schema(
        _source(
            tmp_path / "accepted",
            f"{accepted_header}\n{accepted_row}\n".encode(),
            "csv",
        )
    )
    rejected_header = ",".join(f"c{index}" for index in range(257))
    rejected_row = ",".join("1" for _ in range(257))
    rejected = scan_delimited_schema(
        _source(
            tmp_path / "rejected",
            f"{rejected_header}\n{rejected_row}\n".encode(),
            "csv",
        )
    )

    assert accepted.completeness.state is ScanState.COMPLETE
    assert rejected.completeness.state is ScanState.INCOMPLETE_LIMIT
    assert rejected.completeness.reason is ScanReason.COLUMN_LIMIT


@pytest.mark.parametrize(
    "content",
    [
        b"acc,acc\n0.8,0.9\n",
        b"acc,\n0.8,0.9\n",
        b"acc,loss\n0.8\n",
        b"acc\n\xff\n",
        b"acc\n0.8\x00\n",
    ],
)
def test_malformed_delimited_input_returns_no_schema(
    tmp_path: Path,
    content: bytes,
) -> None:
    outcome = scan_delimited_schema(_source(tmp_path, content, "csv"))

    assert outcome.header == ()
    assert outcome.completeness.state is ScanState.FAILED
    assert outcome.completeness.integrity_verified is True


def test_quoted_newlines_are_one_bounded_logical_row(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        b'note,acc\n"first line\nsecond line",0.8\nplain,0.9\n',
        "csv",
    )

    schema = scan_delimited_schema(source)
    evidence = scan_delimited_observations(source, _field_match(source, "acc"))

    assert schema.completeness.records_scanned == 2
    assert evidence.evidence is not None
    assert [item.metric_value for item in evidence.evidence.observations] == [0.8, 0.9]


def test_cumulative_quoted_record_limit_fails_closed(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        b'note,acc\n"1234567890\n1234567890",0.8\n',
        "csv",
    )

    outcome = scan_delimited_schema(
        source,
        limits=StreamingLimits(max_logical_record_bytes=16),
    )

    assert outcome.completeness.state is ScanState.INCOMPLETE_LIMIT
    assert outcome.completeness.reason is ScanReason.LOGICAL_RECORD_LIMIT


def test_exact_table_selector_binding_revalidates_on_streamed_source(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path,
        b"role,acc\nbaseline,0.7\ncandidate,0.8\n",
        "csv",
    )

    def selected(role: ExperimentRole, value: str):
        provenance = FieldProvenance(
            ProvenanceKind.ADAPTER_EXTRACTION,
            f"exact role selector; sha256={source.candidate.sha256}",
            source.candidate.path,
            f"fixture:role:{value}",
        )
        match = AdapterMatch(
            "claimci-csv-v1",
            source.candidate.path,
            Confidence(0.95),
            (
                FieldMapping(
                    "metric_value",
                    TableSelector(
                        "acc",
                        (
                            TablePredicate(
                                "role",
                                TableScalarType.STRING,
                                value,
                            ),
                        ),
                        1,
                        provenance,
                    ),
                    provenance,
                ),
            ),
            (provenance,),
        )
        outcome = extract_metric_candidate_scan(
            source,
            role=role,
            adapter_match=match,
        )
        assert len(outcome.candidates) == 1
        return outcome.candidates[0]

    baseline = selected(ExperimentRole.BASELINE, "baseline")
    candidate = selected(ExperimentRole.CANDIDATE, "candidate")
    pending = resolve_metric_binding(
        (baseline, candidate),
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
        source,
        binding,
        role=ExperimentRole.BASELINE,
    )

    assert evidence.scan_completeness is not None
    assert [item.metric_name for item in evidence.observations] == ["accuracy"]
    assert [item.metric_value for item in evidence.observations] == [0.7]
