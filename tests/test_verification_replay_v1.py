"""Canonical pre-Audit snapshot and post-Audit Replay Recipe regressions."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from claimci.analysis import (
    AnalysisAuthority,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    ConfigValue,
    DatasetSplit,
    EvidenceSelector,
    EvidenceTraceBundle,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    ProvenanceKind,
    GitCommitSha,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
)
from claimci.analysis.replay import (
    ReplayAuditCommitment,
    ReplayEngineProvenance,
    ReplayRecipe,
    ReplayTraceReference,
)
from claimci.analysis.verification import (
    AuditSemanticsCompatibility,
    MeasurementProtocolReference,
    VerificationArtifactIdentity,
    VerificationSnapshotCapability,
    _artifact_semantic_projections,
    _artifact_identity_from_materialized,
    VerificationInputSnapshot,
    verification_snapshot_json_bytes,
)
from claimci.models import AuditResult, Finding, Impact, Severity, Verdict


def _passive_result(
    content: bytes = b'{"accuracy": 0.8}\n',
    *,
    path: str = "bench/results.json",
) -> PassiveArtifact:
    candidate = ArtifactCandidate(
        path=RepositoryPath(path),
        kind=ArtifactKind.RESULTS,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.99),
        discovery_reason="exact deterministic fixture",
        relevant_claim_ids=("claim-1",),
        provenance=FieldProvenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            "exact deterministic fixture",
            RepositoryPath(path),
        ),
    )
    return PassiveArtifact(candidate, content)


def _bound_result(
    passive: PassiveArtifact,
    *,
    role: ExperimentRole = ExperimentRole.BASELINE,
    expression: str = "/accuracy",
    value: float = 0.8,
) -> tuple[NormalizedEvidence, ArtifactBinding]:
    provenance = FieldProvenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        "fixed result extraction",
        passive.candidate.path,
    )
    mapping = FieldMapping(
        "metric_value",
        EvidenceSelector(
            SelectorKind.JSON_POINTER,
            expression,
            provenance,
        ),
        provenance,
    )
    from claimci.analysis import AdapterMatch

    match = AdapterMatch(
        "claimci-structured-results-v1",
        passive.candidate.path,
        Confidence(0.99),
        (mapping,),
        (provenance,),
    )
    evidence = NormalizedEvidence(
        evidence_id=f"evidence-{role.value}",
        artifact=passive.candidate,
        adapter_match=match,
        observations=(
            NormalizedObservation(
                metric_name="accuracy",
                metric_value=value,
                run_id="run-1",
                seed=1,
                experiment_role=role,
                provenance=provenance,
            ),
        ),
    )
    binding = ArtifactBinding(
        passive.candidate.path,
        ArtifactKind.RESULTS,
        role,
        match.adapter_id,
        match.mappings,
        provenance,
    )
    return evidence, binding


def _audit_result(verdict: Verdict = Verdict.NOT_SUPPORTED) -> AuditResult:
    finding = Finding(
        rule_id="RESULT.IMPROVEMENT_NOT_MET",
        severity=Severity.CRITICAL,
        title="Improvement threshold was not met",
        explanation="The deterministic comparison did not meet the threshold.",
        evidence={"observed": 0.01, "required": 0.05},
        impact=Impact.INVALIDATES,
    )
    return AuditResult(
        verdict=verdict,
        findings=(finding,),
        manifest_path=Path("<ephemeral>/research.yaml"),
        metric="accuracy",
        minimum_improvement=0.05,
    )


def test_snapshot_authority_records_are_immutable_final_and_constructor_sealed() -> None:
    for contract in (
        VerificationArtifactIdentity,
        MeasurementProtocolReference,
        AuditSemanticsCompatibility,
    ):
        with pytest.raises(TypeError):
            contract()  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="final"):
            type(f"Forged{contract.__name__}", (contract,), {})


def test_materialized_artifact_identity_separates_source_extraction_and_audit_semantics() -> None:
    passive = _passive_result()
    evidence, binding = _bound_result(passive)

    identity = _artifact_identity_from_materialized(
        evidence=evidence,
        binding=binding,
        passive=passive,
        extraction_projection={"metric": "accuracy", "value": 0.8},
        audit_semantics_projection={"values": [0.8], "seeds": [1]},
        evidence_projector_version="claimci.projector.training-results.v1",
    )

    assert identity.source_sha256 == passive.candidate.sha256
    assert identity.extraction_sha256 == Sha256Digest(
        "bfc536933bd19fb1cd08d5a52e15cab9b5e5cd645b433a86cf6b8f7557190b56"
    )
    assert identity.audit_semantics_sha256 == Sha256Digest(
        "bf9383aef7d0840eb0d75c7e4c0177223df71261492880ae4f4fdfcc7315b0e0"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.path = RepositoryPath("other.json")  # type: ignore[misc]


def test_formatting_only_source_change_does_not_change_audit_semantics() -> None:
    first = _passive_result(b'{"accuracy":0.8}\n')
    second = _passive_result(b'{ "accuracy" : 0.8 }\n')
    first_evidence, first_binding = _bound_result(first)
    second_evidence, second_binding = _bound_result(second)

    identities = tuple(
        _artifact_identity_from_materialized(
            evidence=evidence,
            binding=binding,
            passive=passive,
            extraction_projection={"metric": "accuracy", "value": 0.8},
            audit_semantics_projection={"values": [0.8], "seeds": [1]},
            evidence_projector_version="claimci.projector.training-results.v1",
        )
        for passive, evidence, binding in (
            (first, first_evidence, first_binding),
            (second, second_evidence, second_binding),
        )
    )

    assert identities[0].source_sha256 != identities[1].source_sha256
    assert identities[0].extraction_sha256 == identities[1].extraction_sha256
    assert identities[0].audit_semantics_sha256 == identities[1].audit_semantics_sha256


def test_benchmark_audit_semantics_exclude_unconsumed_profile_keys() -> None:
    passive = _passive_result()
    evidence, binding = _bound_result(passive)
    provenance = evidence.observations[0].provenance
    evidence = dataclasses.replace(
        evidence,
        observations=(
            dataclasses.replace(
                evidence.observations[0],
                config_values=(
                    ConfigValue("benchmark.not_consumed", "provider-prose", provenance),
                ),
            ),
        ),
    )

    _extraction, semantics, _version = _artifact_semantic_projections(
        evidence,
        binding,
        passive,
        metric="accuracy",
        benchmark_profile=True,
    )

    assert semantics["profile_config"] == ()


def test_replay_audit_commitment_requires_an_actual_audit_result() -> None:
    with pytest.raises(TypeError, match="actual AuditResult"):
        ReplayAuditCommitment.from_audit_result(
            {"verdict": "SUPPORTED"}  # type: ignore[arg-type]
        )

    commitment = ReplayAuditCommitment.from_audit_result(_audit_result())

    assert commitment.verdict is Verdict.NOT_SUPPORTED
    assert commitment.authority is AnalysisAuthority.DETERMINISTIC
    assert commitment.applied_rule_ids == ("RESULT.IMPROVEMENT_NOT_MET",)
    assert commitment.stable_audit_sha256 == Sha256Digest(
        "cb752c140efaa99aaf332b1ea16992432d931e64746690ddc11494537ebe75f7"
    )


def test_engine_build_provenance_is_separate_from_audit_semantics() -> None:
    first = ReplayEngineProvenance.current(
        source_revision="a" * 40,
        distribution_sha256="1" * 64,
    )
    second = ReplayEngineProvenance.current(
        source_revision="b" * 40,
        distribution_sha256="2" * 64,
    )

    assert first != second
    assert first.package_version == second.package_version == "0.2.0"
    assert not hasattr(first, "audit_semantics_sha256")


def test_replay_recipe_never_authorizes_experiment_execution() -> None:
    assert ReplayRecipe.experiment_replay_supported is False


def test_snapshot_unavailable_is_factory_only_and_cannot_create_replay() -> None:
    with pytest.raises(TypeError):
        VerificationInputSnapshot()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="final"):
        class ForgedSnapshot(VerificationInputSnapshot):
            pass

    snapshot = VerificationInputSnapshot.unavailable(
        repository=RepositoryIdentity("amebaleon", "ClaimCI"),
        head_sha=GitCommitSha("a" * 40),
        pr_number=22,
        reason_code="snapshot_unavailable",
    )
    result = _audit_result()
    commitment = ReplayAuditCommitment.from_audit_result(result)
    trace = EvidenceTraceBundle.unavailable(
        head_sha=GitCommitSha("a" * 40),
        result=result,
    )
    reference = ReplayTraceReference.from_trace(trace, commitment)

    assert snapshot.capability is VerificationSnapshotCapability.UNAVAILABLE
    assert b'"capability":"unavailable"' in verification_snapshot_json_bytes(snapshot)
    with pytest.raises(ValueError, match="unavailable"):
        ReplayRecipe.from_execution(
            input_snapshot=snapshot,
            engine_provenance=ReplayEngineProvenance.current(),
            audit_commitment=commitment,
            trace_reference=reference,
        )


def test_unavailable_snapshot_rejects_malformed_pr_and_base_identity() -> None:
    repository = RepositoryIdentity("amebaleon", "ClaimCI")
    head_sha = GitCommitSha("a" * 40)

    for pr_number in (False, 0, -1, "22"):
        with pytest.raises((TypeError, ValueError), match="pr_number"):
            VerificationInputSnapshot.unavailable(
                repository=repository,
                head_sha=head_sha,
                pr_number=pr_number,  # type: ignore[arg-type]
            )
    with pytest.raises(TypeError, match="base_sha"):
        VerificationInputSnapshot.unavailable(
            repository=repository,
            head_sha=head_sha,
            base_sha="b" * 40,  # type: ignore[arg-type]
        )


def test_trace_reference_rejects_a_different_actual_audit_commitment() -> None:
    first = _audit_result(Verdict.NOT_SUPPORTED)
    second = dataclasses.replace(
        _audit_result(Verdict.INSUFFICIENT_EVIDENCE),
        findings=(),
    )
    trace = EvidenceTraceBundle.unavailable(
        head_sha=GitCommitSha("a" * 40),
        result=first,
    )

    with pytest.raises(ValueError, match="does not match"):
        ReplayTraceReference.from_trace(
            trace,
            ReplayAuditCommitment.from_audit_result(second),
        )


def test_replay_authority_records_are_final_and_constructor_sealed() -> None:
    for contract in (
        ReplayEngineProvenance,
        ReplayAuditCommitment,
        ReplayTraceReference,
        ReplayRecipe,
    ):
        with pytest.raises(TypeError):
            contract()  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="final"):
            type(f"Forged{contract.__name__}", (contract,), {})
