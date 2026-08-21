"""Hosted-runner integration and accepted ClaimCI v0.3 scenarios A through I."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from claimci.analysis import (
    AnalysisState,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    ClaimFieldTarget,
    ConfigValue,
    EvidenceObligationDecision,
    EvidenceObligationReason,
    EvidenceObligationState,
    EvidenceSelector,
    EvidenceTraceBundle,
    DeterministicAuditOutcome,
    ExperimentRole,
    FieldMapping,
    GitCommitSha,
    MappingCandidate,
    PassiveArtifact,
    PlanningRequest,
    ProvenanceKind,
    RepositoryPath,
    RepositoryIdentity,
    SelectorKind,
    Sha256Digest,
    TraceCompleteness,
    UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER,
    UnifiedAnalysisResult,
    claim_evidence_policy,
    compile_audit_claim,
    plan_ephemeral_audit,
    recover_scientific_claim,
    run_unified_analysis,
)
from claimci.analysis.adapters import extract_registered_artifact
from claimci.analysis.materialize import MaterializationPartial
from claimci.audit import audit_research
from claimci.models import AuditResult, Verdict
from claimci.report import render_json
from claimci.review import ProviderUsage, ReviewConfig
from claimci.review.provider import ProviderResponse, StructuredRequest

from test_ephemeral_materialize_v03 import _plan_fixture


class IntegrationProvider:
    def __init__(self, *, disagree: bool = False, fail: bool = False) -> None:
        self.disagree = disagree
        self.fail = fail
        self.calls: list[StructuredRequest] = []

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        raise AssertionError("hosted integration must not perform claim extraction")

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("private provider failure detail")
        payload = {
            "summary": (
                "I call this supported."
                if self.disagree
                else "The deterministic evidence has been interpreted."
            ),
            "interpretations": [
                "SUPPORTED"
                if self.disagree
                else "The Audit findings remain authoritative."
            ],
            "missing_evidence": [],
            "confidence": 0.8,
        }
        return ProviderResponse(
            json.dumps(payload),
            "fake",
            "fake-model",
            usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def _request_from_plan(plan: object) -> PlanningRequest:
    evidence = (*plan.baseline_evidence, *plan.candidate_evidence)
    assert isinstance(plan.selected_mapping, MappingCandidate)
    scientific_claim = plan.scientific_claim or recover_scientific_claim(plan.claim)
    assert scientific_claim is not None
    return PlanningRequest(
        repository=plan.repository,
        pr_number=plan.pr_number,
        head_sha=plan.head_sha,
        claim=plan.claim,
        audit_claim=plan.audit_claim,
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=(plan.selected_mapping,),
        scientific_claim=scientific_claim,
        claim_policy=claim_evidence_policy(scientific_claim),
    )


def _fixture_request(
    tmp_path: Path,
    **kwargs: object,
) -> tuple[PlanningRequest, object, Path, Path]:
    plan, runtime, checkout, scratch = _plan_fixture(tmp_path, **kwargs)
    return _request_from_plan(plan), runtime, checkout, scratch


def _without_datasets(request: PlanningRequest) -> PlanningRequest:
    evidence = tuple(
        item
        for item in request.normalized_evidence
        if item.artifact.kind is not ArtifactKind.DATASET
    )
    mapping = request.mapping_candidates[0]
    changed_mapping = dataclasses.replace(
        mapping,
        bindings=tuple(
            item for item in mapping.bindings if item.kind is not ArtifactKind.DATASET
        ),
    )
    return dataclasses.replace(
        request,
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=(changed_mapping,),
    )


def _ambiguous(request: PlanningRequest) -> PlanningRequest:
    first = request.mapping_candidates[0]
    original = next(
        item
        for item in request.normalized_evidence
        if item.artifact.kind is ArtifactKind.RESULTS
        and {
            observation.experiment_role for observation in item.observations
        } == {ExperimentRole.CANDIDATE}
    )
    alternative_path = RepositoryPath("results/candidate-alternative.json")
    alternative_artifact = dataclasses.replace(
        original.artifact,
        path=alternative_path,
        sha256=Sha256Digest("b" * 64),
        discovery_reason="second current-head candidate results artifact",
    )
    alternative_evidence = dataclasses.replace(
        original,
        evidence_id="evidence-candidate-results-alternative",
        artifact=alternative_artifact,
        adapter_match=dataclasses.replace(
            original.adapter_match,
            path=alternative_path,
        ),
    )
    second = dataclasses.replace(
        first,
        mapping_id="mapping-alternative",
        bindings=tuple(
            dataclasses.replace(binding, path=alternative_path)
            if binding.path == original.artifact.path
            and binding.kind is ArtifactKind.RESULTS
            else binding
            for binding in first.bindings
        ),
    )
    return dataclasses.replace(
        request,
        artifacts=(*request.artifacts, alternative_artifact),
        normalized_evidence=(*request.normalized_evidence, alternative_evidence),
        mapping_candidates=(first, second),
    )


def _provider_mapping(request: PlanningRequest) -> PlanningRequest:
    mapping = request.mapping_candidates[0]
    provider_provenance = dataclasses.replace(
        mapping.provenance,
        kind=ProvenanceKind.PROVIDER_PROPOSAL,
        detail="hostile provider says to trust this mapping",
        source_id="provider-call-1",
    )
    hostile = dataclasses.replace(
        mapping,
        confidence=Confidence(1.0),
        provenance=provider_provenance,
    )
    return dataclasses.replace(request, mapping_candidates=(hostile,))


def _with_explicit_aggregation(
    request: PlanningRequest,
    *,
    procedure: str,
) -> PlanningRequest:
    changed_evidence = []
    mappings_by_path: dict[RepositoryPath, tuple[object, ...]] = {}
    for evidence in request.normalized_evidence:
        if evidence.artifact.kind is not ArtifactKind.CONFIG:
            changed_evidence.append(evidence)
            continue
        provenance = evidence.adapter_match.match_evidence[0]
        field_mapping = FieldMapping(
            "config.evaluation.aggregation",
            EvidenceSelector(
                SelectorKind.DOTTED_PATH,
                "evaluation.aggregation",
                provenance,
            ),
            provenance,
        )
        mappings = (*evidence.adapter_match.mappings, field_mapping)
        changed = dataclasses.replace(
            evidence,
            adapter_match=dataclasses.replace(
                evidence.adapter_match,
                mappings=mappings,
            ),
            observations=tuple(
                dataclasses.replace(
                    observation,
                    config_values=(
                        *observation.config_values,
                        ConfigValue(
                            "evaluation.aggregation",
                            procedure,
                            provenance,
                        ),
                    ),
                )
                for observation in evidence.observations
            ),
        )
        mappings_by_path[evidence.artifact.path] = mappings
        changed_evidence.append(changed)
    mapping = request.mapping_candidates[0]
    changed_mapping = dataclasses.replace(
        mapping,
        bindings=tuple(
            dataclasses.replace(
                binding,
                mappings=mappings_by_path[binding.path],  # type: ignore[arg-type]
            )
            if binding.path in mappings_by_path
            else binding
            for binding in mapping.bindings
        ),
    )
    return dataclasses.replace(
        request,
        normalized_evidence=tuple(changed_evidence),
        mapping_candidates=(changed_mapping,),
    )


def _update_artifact_bytes(
    request: PlanningRequest,
    checkout: Path,
    *,
    path: str,
    content: bytes,
    evidence_transform: object | None = None,
) -> PlanningRequest:
    source = checkout / Path(path)
    source.write_bytes(content)
    target = next(
        item
        for item in request.normalized_evidence
        if str(item.artifact.path) == path
    )
    artifact = dataclasses.replace(
        target.artifact,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
    )
    if target.artifact.kind is ArtifactKind.DATASET:
        changed = extract_registered_artifact(PassiveArtifact(artifact, content))
        assert changed is not None
        assert evidence_transform is None
    else:
        changed = dataclasses.replace(target, artifact=artifact)
    if evidence_transform is not None:
        changed = evidence_transform(changed)
    evidence = tuple(
        changed if item is target else item for item in request.normalized_evidence
    )
    return dataclasses.replace(
        request,
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
    )


def test_state_successful_audit_with_disabled_review_is_partial_with_authority(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=False),
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert result.research_interpretation is None
    assert result.evidence_trace is not None
    assert result.evidence_trace.completeness is TraceCompleteness.COMPLETE
    assert result.evidence_trace.advisory_interpretation is None
    assert result.evidence_obligations is not None
    assert result.evidence_obligations.decision is EvidenceObligationDecision.READY


def test_state_review_failure_is_partial_without_provider_error_leak(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(fail=True),
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert "private provider" not in (result.unavailable_reason or "")


def test_state_runtime_integrity_failure_is_unavailable(tmp_path: Path) -> None:
    request, runtime, checkout, _scratch = _fixture_request(tmp_path)
    (checkout / "results" / "candidate.json").write_text(
        "changed",
        encoding="utf-8",
    )

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.UNAVAILABLE
    assert result.authoritative_verdict is None
    assert result.evidence_obligations is None


def test_state_unexpected_executor_failure_is_fail_closed_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.integration as integration

    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    monkeypatch.setattr(
        integration,
        "execute_ephemeral_audit_with_trace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private executor failure detail")
        ),
    )

    result = integration.run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.UNAVAILABLE
    assert result.authoritative_verdict is None
    assert "private executor" not in (result.unavailable_reason or "")


def test_untyped_materialization_partial_is_fail_closed_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    def fail_without_a_validated_slot(*_args: object, **_kwargs: object) -> object:
        raise MaterializationPartial("private untyped representation detail")

    monkeypatch.setattr(
        "claimci.analysis.integration.execute_ephemeral_audit_with_trace",
        fail_without_a_validated_slot,
    )

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.UNAVAILABLE
    assert result.authoritative_verdict is None
    assert result.evidence_obligations is None
    assert "private" not in (result.unavailable_reason or "")


def test_scenario_a_zero_manifest_normalized_evidence_runs_full_analysis(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    provider = IntegrationProvider()

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert result.state is AnalysisState.COMPLETE
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert result.research_interpretation is not None
    assert result.evidence_trace is not None
    assert result.evidence_trace.completeness is TraceCompleteness.COMPLETE
    assert result.evidence_trace.advisory_interpretation is not None
    assert result.verification_input_snapshot is not None
    assert result.replay_recipe is not None
    assert (
        result.replay_recipe.audit_commitment.verdict
        is result.authoritative_verdict
    )
    with pytest.raises((TypeError, ValueError), match="snapshot|Replay|deterministic"):
        dataclasses.replace(result, deterministic=None)
    different_same_verdict = DeterministicAuditOutcome.from_audit_result(
        AuditResult(
            verdict=Verdict.NOT_SUPPORTED,
            findings=(),
            manifest_path=Path("different/research.yaml"),
            metric="accuracy",
            minimum_improvement=0.05,
        )
    )
    with pytest.raises(ValueError, match="commitment"):
        dataclasses.replace(result, deterministic=different_same_verdict)
    assert not hasattr(result.evidence_trace.advisory_interpretation, "verdict")
    assert len(provider.calls) == 1


def test_advisory_trace_failure_preserves_the_real_verdict_and_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    def fail_advisory_trace(
        self: EvidenceTraceBundle,
        _interpretation: object,
    ) -> EvidenceTraceBundle:
        raise RuntimeError("trace-only advisory failure")

    monkeypatch.setattr(EvidenceTraceBundle, "with_advisory", fail_advisory_trace)
    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.COMPLETE
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert result.research_interpretation is not None
    assert result.evidence_trace is not None
    assert result.evidence_trace.completeness is TraceCompleteness.UNAVAILABLE
    assert result.evidence_trace.advisory_interpretation is not None


def test_scenario_b_native_manifest_preserves_audit_bytes(study_factory) -> None:
    manifest = study_factory(candidate_config_updates={"training_steps": 300})

    before = render_json(audit_research(manifest))
    after = render_json(audit_research(manifest))

    assert before == after


def test_scenario_c_ambiguous_candidate_mapping_asks_without_verdict(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    result = run_unified_analysis(
        _ambiguous(request),
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.MAPPING_NEEDED
    assert result.mapping_question is not None
    assert result.authoritative_verdict is None
    assert result.evidence_trace is None
    assert result.evidence_obligations is not None
    assert (
        result.evidence_obligations.decision
        is EvidenceObligationDecision.MAPPING_NEEDED
    )
    assert result.mapping_question.blocking_obligation_ids == (
        result.evidence_obligations.blocking_obligation_ids
    )


def test_mapping_question_cannot_be_relinked_to_an_unrelated_obligation_bundle(
    tmp_path: Path,
) -> None:
    request, _runtime, _checkout, _scratch = _fixture_request(tmp_path)
    planning = plan_ephemeral_audit(_ambiguous(request))
    assert planning.mapping_question is not None
    assert planning.evidence_obligations is not None
    unrelated = dataclasses.replace(
        planning.mapping_question,
        question_id="mapping-unrelated-question",
    )

    with pytest.raises(ValueError, match="obligation|question|link"):
        UnifiedAnalysisResult(
            state=AnalysisState.MAPPING_NEEDED,
            mapping_question=unrelated,
            evidence_obligations=planning.evidence_obligations,
        )


def test_scenario_d_missing_dataset_is_pre_audit_partial(tmp_path: Path) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    result = run_unified_analysis(
        _without_datasets(request),
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.authoritative_verdict is None
    assert {item.kind for item in result.missing_evidence} == {ArtifactKind.DATASET}
    assert result.evidence_obligations is not None
    assert result.evidence_obligations.decision is EvidenceObligationDecision.PARTIAL


def test_native_config_representation_failure_is_a_specific_artifact_obligation(
    tmp_path: Path,
) -> None:
    request, runtime, checkout, _scratch = _fixture_request(tmp_path)

    def conflicting_config(evidence):
        observation = evidence.observations[0]
        provenance = observation.config_values[0].provenance
        return dataclasses.replace(
            evidence,
            observations=(
                dataclasses.replace(
                    observation,
                    config_values=(
                        ConfigValue("training_steps", 100, provenance),
                        ConfigValue("training_steps.value", 100, provenance),
                    ),
                ),
            ),
        )

    request = _update_artifact_bytes(
        request,
        checkout,
        path="configs/candidate.json",
        content=b'{"training_steps":100,"training_steps.value":100}\n',
        evidence_transform=conflicting_config,
    )
    provider = IntegrationProvider()

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.authoritative_verdict is None
    assert result.mapping_question is None
    assert provider.calls == []
    assert result.evidence_obligations is not None
    assert result.evidence_obligations.decision is EvidenceObligationDecision.PARTIAL
    blocker = next(
        item
        for item in result.evidence_obligations.obligations
        if item.obligation_id == "artifact.candidate.config"
    )
    assert blocker.state is EvidenceObligationState.UNSUPPORTED
    assert (
        blocker.reason
        is EvidenceObligationReason.NATIVE_REPRESENTATION_NOT_SUPPORTED
    )
    assert {item.kind for item in result.missing_evidence} == {ArtifactKind.CONFIG}
    assert ArtifactKind.MANIFEST not in {item.kind for item in result.missing_evidence}


@pytest.mark.parametrize(
    "text",
    [
        "Candidate accuracy is at least 0.90.",
        "The candidate generalizes across unseen domains.",
        "The candidate uses 40% less memory.",
    ],
)
def test_each_unsupported_primary_is_exact_partial_and_never_reaches_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    reference = dataclasses.replace(
        request.claim,
        text=text,
    )
    scientific_claim = recover_scientific_claim(reference)
    assert scientific_claim is not None
    unsupported = dataclasses.replace(
        request,
        claim=reference,
        audit_claim=None,
        scientific_claim=scientific_claim,
        claim_policy=claim_evidence_policy(scientific_claim),
    )

    def fail_if_executed(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unsupported claim reached deterministic Audit")

    monkeypatch.setattr(
        "claimci.analysis.integration.execute_ephemeral_audit_with_trace",
        fail_if_executed,
    )

    result = run_unified_analysis(
        unsupported,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.unavailable_reason == UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER
    assert result.authoritative_verdict is None
    assert result.deterministic is None
    assert result.mapping_question is None
    assert result.evidence_obligations is not None
    assert (
        result.evidence_obligations.compiler_state
        is EvidenceObligationState.UNSUPPORTED
    )


def test_unsupported_upstream_procedure_never_reaches_audit_or_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    reference = dataclasses.replace(
        request.claim,
        text=(
            "Using median across supplied runs, accuracy improved from 0.60 "
            "to 0.90 by at least 0.05."
        ),
    )
    scientific_claim = recover_scientific_claim(reference)
    assert scientific_claim is not None
    blocked = dataclasses.replace(
        request,
        claim=reference,
        audit_claim=compile_audit_claim(scientific_claim),
        scientific_claim=scientific_claim,
        claim_policy=claim_evidence_policy(scientific_claim),
    )
    provider = IntegrationProvider()

    def fail_if_executed(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unsupported upstream procedure reached Audit")

    monkeypatch.setattr(
        "claimci.analysis.integration.execute_ephemeral_audit_with_trace",
        fail_if_executed,
    )

    result = run_unified_analysis(
        blocked,
        runtime,
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.unavailable_reason == "measurement_procedure_not_supported"
    assert result.authoritative_verdict is None
    assert result.mapping_question is None
    assert provider.calls == []


def test_supported_upstream_reduction_reaches_real_audit_with_one_review_call(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    request = _with_explicit_aggregation(
        request,
        procedure="arithmetic_mean_v1",
    )
    reference = dataclasses.replace(
        request.claim,
        text=(
            "Using the arithmetic mean across supplied runs, accuracy improved "
            "from 0.60 to 0.90 by at least 0.05."
        ),
    )
    scientific_claim = recover_scientific_claim(reference)
    assert scientific_claim is not None
    request = dataclasses.replace(
        request,
        claim=reference,
        audit_claim=compile_audit_claim(scientific_claim),
        scientific_claim=scientific_claim,
        claim_policy=claim_evidence_policy(scientific_claim),
    )
    provider = IntegrationProvider()

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert result.state is AnalysisState.COMPLETE
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert result.deterministic is not None
    measurement = result.deterministic.payload["measurement_drift"]
    assert measurement["claimci_verification_reduction"] == "arithmetic_mean_v1"
    retry = next(
        item
        for item in measurement["findings"]
        if item["component_kind"] == "retry_aggregation"
    )
    assert retry["state"] == "verified"
    assert len(provider.calls) == 1


def test_scenario_e_advisory_disagreement_cannot_override_audit(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(disagree=True),
    )

    assert result.state is AnalysisState.COMPLETE
    assert result.authoritative_verdict is Verdict.NOT_SUPPORTED
    assert result.research_interpretation is not None
    assert result.research_interpretation.interpretations == ("SUPPORTED",)


def test_scenario_f_explicit_threshold_reaches_native_audit(tmp_path: Path) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.deterministic is not None
    assert result.deterministic.payload["claim"]["minimum_improvement"] == 0.05


def test_scenario_g_vague_improvement_never_invents_threshold(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    vague = dataclasses.replace(
        request,
        claim=dataclasses.replace(request.claim, text="Accuracy improved."),
        audit_claim=dataclasses.replace(
            request.audit_claim,
            minimum_absolute_improvement=None,
            threshold_provenance=None,
        ),
        scientific_claim=None,
        claim_policy=None,
    )
    vague_scientific = recover_scientific_claim(vague.claim)
    assert vague_scientific is not None
    vague = dataclasses.replace(
        vague,
        scientific_claim=vague_scientific,
        claim_policy=claim_evidence_policy(vague_scientific),
    )

    result = run_unified_analysis(
        vague,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.authoritative_verdict is None
    assert result.missing_evidence == ()
    assert result.unavailable_reason == "required_threshold_not_recovered"
    assert result.evidence_obligations is not None
    threshold = next(
        item
        for item in result.evidence_obligations.obligations
        if item.obligation_id == "claim.threshold"
    )
    assert isinstance(threshold.target, ClaimFieldTarget)
    assert threshold.state is EvidenceObligationState.MISSING
    assert threshold.reason is EvidenceObligationReason.REQUIRED_THRESHOLD_NOT_RECOVERED


def test_scenario_h_identical_inputs_produce_stable_plan_and_result(
    tmp_path: Path,
) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)

    first_plan = plan_ephemeral_audit(request)
    second_plan = plan_ephemeral_audit(request)
    first = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )
    second = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert first_plan.plan is not None and second_plan.plan is not None
    assert first_plan.plan.plan_id == second_plan.plan.plan_id
    assert first.authoritative_verdict == second.authoritative_verdict
    assert first.deterministic is not None and second.deterministic is not None
    assert first.deterministic.payload == second.deterministic.payload
    assert first.research_interpretation == second.research_interpretation


def test_scenario_i_hostile_provider_mapping_never_executes(tmp_path: Path) -> None:
    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    provider = IntegrationProvider()

    result = run_unified_analysis(
        _provider_mapping(request),
        runtime,
        ReviewConfig(enabled=True),
        provider=provider,
    )

    assert result.state is AnalysisState.MAPPING_NEEDED
    assert result.authoritative_verdict is None
    assert provider.calls == []
    assert result.evidence_obligations is not None


def test_completed_insufficient_audit_plus_review_is_pipeline_complete(
    tmp_path: Path,
) -> None:
    request, runtime, checkout, _scratch = _fixture_request(tmp_path)

    def same_compute(evidence):
        observation = evidence.observations[0]
        values = tuple(
            dataclasses.replace(item, value=100)
            if item.key == "training_steps"
            else item
            for item in observation.config_values
        )
        return dataclasses.replace(
            evidence,
            observations=(dataclasses.replace(observation, config_values=values),),
        )

    request = _update_artifact_bytes(
        request,
        checkout,
        path="configs/candidate.json",
        content=b'{"training_steps":100}\n',
        evidence_transform=same_compute,
    )
    request = _update_artifact_bytes(
        request,
        checkout,
        path="data/candidate-train.jsonl",
        content=b'{"id":"candidate-train"}\n',
    )

    result = run_unified_analysis(
        request,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.COMPLETE
    assert result.authoritative_verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.research_interpretation is not None
    assert result.evidence_obligations is not None
    assert result.evidence_obligations.decision is EvidenceObligationDecision.READY
