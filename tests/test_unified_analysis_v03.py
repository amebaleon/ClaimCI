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
    ExperimentRole,
    GitCommitSha,
    MappingCandidate,
    PassiveArtifact,
    PlanningRequest,
    ProvenanceKind,
    RepositoryPath,
    RepositoryIdentity,
    Sha256Digest,
    UnifiedAnalysisResult,
    plan_ephemeral_audit,
    run_unified_analysis,
)
from claimci.analysis.adapters import extract_registered_artifact
from claimci.audit import audit_research
from claimci.models import Verdict
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
    return PlanningRequest(
        repository=plan.repository,
        pr_number=plan.pr_number,
        head_sha=plan.head_sha,
        claim=plan.claim,
        audit_claim=plan.audit_claim,
        artifacts=tuple(item.artifact for item in evidence),
        normalized_evidence=evidence,
        mapping_candidates=(plan.selected_mapping,),
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


def test_state_unexpected_executor_failure_is_fail_closed_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claimci.analysis.integration as integration

    request, runtime, _checkout, _scratch = _fixture_request(tmp_path)
    monkeypatch.setattr(
        integration,
        "execute_ephemeral_audit",
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
    assert len(provider.calls) == 1


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
    )

    result = run_unified_analysis(
        vague,
        runtime,
        ReviewConfig(enabled=True),
        provider=IntegrationProvider(),
    )

    assert result.state is AnalysisState.PARTIAL
    assert result.authoritative_verdict is None
    assert any("threshold" in item.description for item in result.missing_evidence)


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
