"""Immutable bounded evidence trace contracts and authority boundaries."""

from __future__ import annotations

import dataclasses
import json

import pytest

from claimci.analysis import (
    AdvisoryResearchInterpretation,
    AnalysisAuthority,
    ArtifactKind,
    BoundedValueRepresentation,
    Confidence,
    DeterministicAuditTrace,
    EvidenceTraceBundle,
    EvidenceTraceEntry,
    ExperimentRole,
    FieldProvenance,
    GitCommitSha,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    TraceAuthorityClass,
    TraceCompleteness,
    TraceContractError,
    TraceRecordKind,
    TraceSelector,
    TraceValueType,
    to_jsonable,
)
from claimci.models import AuditResult, Verdict


HEAD_SHA = GitCommitSha("a" * 40)


def _audit_result() -> AuditResult:
    return AuditResult(verdict=Verdict.SUPPORTED, findings=())


def _passive_entry(*, provenance: ProvenanceKind) -> EvidenceTraceEntry:
    values = BoundedValueRepresentation.from_value([0.6, 0.9])
    return EvidenceTraceEntry(
        trace_id="evidence-baseline-results",
        record_kind=TraceRecordKind.PASSIVE_SOURCE_EVIDENCE,
        authority=TraceAuthorityClass.PASSIVE_SOURCE,
        head_sha=HEAD_SHA,
        provenance_kind=provenance,
        detail_code="results.metric_values",
        artifact_path=RepositoryPath("results/baseline.json"),
        artifact_sha256=Sha256Digest("b" * 64),
        artifact_size=128,
        artifact_kind=ArtifactKind.RESULTS,
        adapter_id="claimci-json-v1",
        adapter_version="v1",
        selectors=(
            TraceSelector("metric_value", SelectorKind.JSON_POINTER, "/accuracy"),
        ),
        role=ExperimentRole.BASELINE,
        source_value=values,
        normalized_value=values,
        consumer_rule_ids=("RESULT.RECOMPUTED",),
    )


def test_bounded_value_representation_commits_all_values_without_copying_strings() -> None:
    representation = BoundedValueRepresentation.from_value(
        {
            "metric": [0.6, 0.9],
            "model": "customer-private-model-name",
            "enabled": True,
        }
    )

    assert representation.value_type is TraceValueType.MAPPING
    assert representation.count == 4
    assert representation.direct_values == (0.6, 0.9, True)
    assert len(str(representation.canonical_sha256)) == 64
    serialized = json.dumps(to_jsonable(representation), sort_keys=True)
    assert "customer-private-model-name" not in serialized
    assert "0.6" in serialized


def test_value_preview_omits_integers_that_are_not_exact_in_worker_json() -> None:
    representation = BoundedValueRepresentation.from_value(10**100)

    assert representation.value_type is TraceValueType.NUMBER
    assert representation.count == 1
    assert representation.direct_values == ()
    assert len(str(representation.canonical_sha256)) == 64


def test_deep_value_commitment_fails_as_a_controlled_trace_limit() -> None:
    value: object = 1
    for _ in range(2_000):
        value = [value]

    with pytest.raises(TraceContractError):
        BoundedValueRepresentation.from_value(value)


def test_trace_contracts_are_deeply_immutable_and_json_safe() -> None:
    entry = _passive_entry(provenance=ProvenanceKind.ADAPTER_EXTRACTION)
    authority = DeterministicAuditTrace.from_audit_result(_audit_result())
    bundle = EvidenceTraceBundle(
        head_sha=HEAD_SHA,
        completeness=TraceCompleteness.COMPLETE,
        entries=(entry,),
        deterministic_authority=authority,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.detail_code = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.entries = ()  # type: ignore[misc]
    payload = to_jsonable(bundle)
    assert json.loads(json.dumps(payload))["head_sha"] == str(HEAD_SHA)
    assert payload["deterministic_authority"]["authority"] == "deterministic"


def test_provider_provenance_cannot_be_passive_or_deterministic_evidence() -> None:
    with pytest.raises(TraceContractError, match="provider|passive"):
        _passive_entry(provenance=ProvenanceKind.PROVIDER_PROPOSAL)

    with pytest.raises(TraceContractError, match="provider|deterministic"):
        EvidenceTraceEntry(
            trace_id="derived-provider",
            record_kind=TraceRecordKind.DERIVED_DETERMINISTIC_EVIDENCE,
            authority=TraceAuthorityClass.DETERMINISTIC_DERIVATION,
            head_sha=HEAD_SHA,
            provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
            detail_code="result.summary",
            normalized_value=BoundedValueRepresentation.from_value(0.3),
            consumer_rule_ids=("RESULT.RECOMPUTED",),
            source_trace_ids=("evidence-baseline-results",),
        )


def test_record_kind_and_authority_class_are_not_interchangeable() -> None:
    with pytest.raises(TraceContractError, match="authority"):
        EvidenceTraceEntry(
            trace_id="approval-is-not-verdict",
            record_kind=TraceRecordKind.APPROVED_MAPPING,
            authority=TraceAuthorityClass.DETERMINISTIC_DERIVATION,
            head_sha=HEAD_SHA,
            provenance_kind=ProvenanceKind.USER_APPROVED,
            detail_code="mapping.approval",
        )

    proposal = EvidenceTraceEntry(
        trace_id="proposal-1",
        record_kind=TraceRecordKind.LLM_SEMANTIC_PROPOSAL,
        authority=TraceAuthorityClass.NON_AUTHORITATIVE_INPUT,
        head_sha=HEAD_SHA,
        provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
        detail_code="semantic.proposal",
        source_value=BoundedValueRepresentation.from_value("provider prose"),
    )
    assert proposal.authority is TraceAuthorityClass.NON_AUTHORITATIVE_INPUT


def test_deterministic_trace_authority_requires_an_actual_audit_result() -> None:
    with pytest.raises(TypeError, match="from_audit_result"):
        DeterministicAuditTrace()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="AuditResult"):
        DeterministicAuditTrace.from_audit_result(object())  # type: ignore[arg-type]

    authority = DeterministicAuditTrace.from_audit_result(_audit_result())
    assert authority.authority is AnalysisAuthority.DETERMINISTIC
    assert authority.verdict is Verdict.SUPPORTED
    assert len(str(authority.audit_sha256)) == 64


def test_advisory_trace_has_no_verdict_field() -> None:
    interpretation = AdvisoryResearchInterpretation(
        summary="Advisory context only.",
        interpretations=("One interpretation.",),
        missing_evidence=(),
        confidence=Confidence(0.7),
    )
    authority = EvidenceTraceBundle(
        head_sha=HEAD_SHA,
        completeness=TraceCompleteness.COMPLETE,
        entries=(_passive_entry(provenance=ProvenanceKind.ADAPTER_EXTRACTION),),
        deterministic_authority=DeterministicAuditTrace.from_audit_result(
            _audit_result()
        ),
    ).with_advisory(interpretation)

    advisory = authority.advisory_interpretation
    assert advisory is not None
    assert advisory.authority is AnalysisAuthority.ADVISORY
    assert "verdict" not in to_jsonable(advisory)


def test_complete_bundle_rejects_unknown_dependencies_and_over_bound_json() -> None:
    entry = EvidenceTraceEntry(
        trace_id="derived-missing-source",
        record_kind=TraceRecordKind.DERIVED_DETERMINISTIC_EVIDENCE,
        authority=TraceAuthorityClass.DETERMINISTIC_DERIVATION,
        head_sha=HEAD_SHA,
        provenance_kind=None,
        detail_code="result.summary",
        normalized_value=BoundedValueRepresentation.from_value(0.3),
        consumer_rule_ids=("RESULT.RECOMPUTED",),
        source_trace_ids=("missing-source",),
    )
    with pytest.raises(TraceContractError, match="source trace"):
        EvidenceTraceBundle(
            head_sha=HEAD_SHA,
            completeness=TraceCompleteness.COMPLETE,
            entries=(entry,),
            deterministic_authority=DeterministicAuditTrace.from_audit_result(
                _audit_result()
            ),
        )

    huge = "x" * 513
    with pytest.raises(TraceContractError, match="bounded|detail"):
        dataclasses.replace(
            _passive_entry(provenance=ProvenanceKind.ADAPTER_EXTRACTION),
            detail_code=huge,
        )

    wide = dataclasses.replace(
        _passive_entry(provenance=ProvenanceKind.ADAPTER_EXTRACTION),
        selectors=tuple(
            TraceSelector(
                f"config.field_{index}",
                SelectorKind.JSON_POINTER,
                f"/{index}-" + "x" * 900,
            )
            for index in range(32)
        ),
    )
    with pytest.raises(TraceContractError, match="16384-byte"):
        EvidenceTraceBundle(
            head_sha=HEAD_SHA,
            completeness=TraceCompleteness.COMPLETE,
            entries=(wide,),
            deterministic_authority=DeterministicAuditTrace.from_audit_result(
                _audit_result()
            ),
        )


def test_bounded_and_unavailable_completeness_are_explicit() -> None:
    deterministic = DeterministicAuditTrace.from_audit_result(_audit_result())
    omitted_digest = Sha256Digest("c" * 64)
    bounded = EvidenceTraceBundle(
        head_sha=HEAD_SHA,
        completeness=TraceCompleteness.BOUNDED,
        entries=(),
        deterministic_authority=deterministic,
        omitted_entry_count=4,
        omitted_entries_sha256=omitted_digest,
        reason_code="trace_size_limit",
    )
    unavailable = EvidenceTraceBundle(
        head_sha=HEAD_SHA,
        completeness=TraceCompleteness.UNAVAILABLE,
        entries=(),
        deterministic_authority=deterministic,
        reason_code="trace_construction_unavailable",
    )

    assert bounded.completeness is TraceCompleteness.BOUNDED
    assert unavailable.completeness is TraceCompleteness.UNAVAILABLE


def test_fixture_provenance_helper_stays_type_distinct() -> None:
    provenance = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "validated proposal only",
        source_id="provider-1",
    )
    assert provenance.kind is ProvenanceKind.PROVIDER_PROPOSAL
