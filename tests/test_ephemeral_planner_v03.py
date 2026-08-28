"""Pure planning behavior for zero-configuration ClaimCI analysis."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from tests.analysis_occurrence_support import passive_artifact

from claimci.analysis import (
    AbsoluteMetricClaim,
    AdapterMatch,
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    ArtifactSnapshotRole,
    ArtifactEvidenceSlot,
    Confidence,
    ConfigValue,
    DatasetSplit,
    DatasetReference,
    EvidenceSelector,
    EvidenceObligationDecision,
    EvidenceObligationReason,
    EvidenceObligationState,
    ClaimFieldTarget,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingChoice,
    MappingQuestion,
    MappingTrust,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    passive_artifact_from_snapshot,
    PlanningRequest,
    PlanningState,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    TablePredicate,
    TableScalarType,
    TableSelector,
    UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER,
    audit_relevant_claim_projection,
    plan_ephemeral_audit,
    planning_request_from_discovery,
    to_jsonable,
)
from claimci.analysis.adapters import CsvAdapter, JsonAdapter
from claimci.analysis.adapters.core import _evidence_id
from claimci.review.models import ClaimDirection, ClaimType, SourceKind, SourceLocation
from claimci.analysis.discovery import ClaimedValue, DiscoveredClaim, DiscoveryResult


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI-Demo")
HEAD_SHA = GitCommitSha("a" * 40)
CLAIM_ID = "claim-1"


def _provenance(
    kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
    *,
    source_path: str | None = "CLAIM.md",
    source_id: str = "source-1",
) -> FieldProvenance:
    return FieldProvenance(
        kind=kind,
        detail="validated planner fixture",
        source_path=None if source_path is None else RepositoryPath(source_path),
        source_id=source_id,
    )


def _artifact(
    path: str,
    kind: ArtifactKind,
    content: bytes = b"{}\n",
    *,
    claim_ids: tuple[str, ...] = (CLAIM_ID,),
) -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.95),
        discovery_reason="fixture artifact",
        relevant_claim_ids=claim_ids,
        provenance=_provenance(source_path=path, source_id=f"artifact:{path}"),
    )


def _mapping_for_artifact(
    artifact: ArtifactCandidate,
    role: ExperimentRole,
    *,
    target: str,
) -> FieldMapping:
    provenance = _provenance(
        ProvenanceKind.ADAPTER_EXTRACTION,
        source_path=str(artifact.path),
        source_id=f"adapter:{artifact.path}",
    )
    return FieldMapping(
        target_field=target,
        selector=EvidenceSelector(
            kind=SelectorKind.DOTTED_PATH,
            expression=target,
            provenance=provenance,
        ),
        provenance=provenance,
    )


def _evidence(
    path: str,
    kind: ArtifactKind,
    role: ExperimentRole,
    *,
    split: str | None = None,
    claim_ids: tuple[str, ...] = (CLAIM_ID,),
) -> NormalizedEvidence:
    artifact = _artifact(path, kind, claim_ids=claim_ids)
    target = {
        ArtifactKind.RESULTS: "metric_value",
        ArtifactKind.CONFIG: "config.training_steps",
        ArtifactKind.DATASET: "dataset.path",
    }[kind]
    field_mapping = _mapping_for_artifact(artifact, role, target=target)
    match = AdapterMatch(
        adapter_id="fixture.adapter",
        path=artifact.path,
        confidence=Confidence(0.95),
        mappings=(field_mapping,),
        match_evidence=(field_mapping.provenance,),
    )
    if kind is ArtifactKind.RESULTS:
        observation = NormalizedObservation(
            provenance=field_mapping.provenance,
            metric_name="accuracy",
            metric_value=0.7 if role is ExperimentRole.BASELINE else 0.8,
            seed=1,
            experiment_role=role,
        )
    elif kind is ArtifactKind.CONFIG:
        observation = NormalizedObservation(
            provenance=field_mapping.provenance,
            experiment_role=role,
            config_values=(
                ConfigValue("training_steps", 10, field_mapping.provenance),
            ),
        )
    else:
        observation = NormalizedObservation(
            provenance=field_mapping.provenance,
            experiment_role=role,
            dataset_references=(
                DatasetReference(artifact.path, split, field_mapping.provenance),
            ),
        )
    return NormalizedEvidence(
        evidence_id="evidence-" + str(artifact.path).replace("/", "-"),
        artifact=artifact,
        adapter_match=match,
        observations=(observation,),
    )


def _binding(evidence: NormalizedEvidence, role: ExperimentRole) -> ArtifactBinding:
    references = tuple(
        reference
        for observation in evidence.observations
        for reference in observation.dataset_references
    )
    dataset_split = (
        DatasetSplit(references[0].split)
        if evidence.artifact.kind is ArtifactKind.DATASET
        and len(references) == 1
        and references[0].split is not None
        else None
    )
    return ArtifactBinding(
        path=evidence.artifact.path,
        kind=evidence.artifact.kind,
        role=role,
        adapter_id=evidence.adapter_match.adapter_id,
        mappings=evidence.adapter_match.mappings,
        provenance=evidence.adapter_match.match_evidence[0],
        dataset_split=dataset_split,
    )


def _claim(
    text: str = "Accuracy improved from 0.71 to 0.79 by at least 0.05.",
    *,
    minimum: float | None = 0.05,
    provenance_kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> DiscoveredClaim:
    from claimci.analysis import ClaimReference

    provenance = _provenance(provenance_kind)
    return DiscoveredClaim(
        reference=ClaimReference(
            claim_id=CLAIM_ID,
            text=text,
            source_path=RepositoryPath("CLAIM.md"),
            confidence=Confidence(0.95),
            provenance=provenance,
        ),
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="candidate",
        source=SourceLocation(
            source_id="source-1",
            kind=SourceKind.REPOSITORY_FILE,
            path="CLAIM.md",
            start_line=1,
            end_line=1,
        ),
        metric="accuracy",
        direction=ClaimDirection.HIGHER,
        baseline_value=ClaimedValue(0.71, None, provenance),
        candidate_value=ClaimedValue(0.79, None, provenance),
        minimum_improvement=None
        if minimum is None
        else ClaimedValue(minimum, None, provenance),
    )


def _discovery(
    claim: DiscoveredClaim,
    evidence: tuple[NormalizedEvidence, ...],
    mapping_candidates: tuple[MappingCandidate, ...] = (),
) -> DiscoveryResult:
    artifacts = tuple(item.artifact for item in evidence)
    return DiscoveryResult(
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=7,
        repository_paths=(RepositoryPath("CLAIM.md"),) + tuple(
            item.path for item in artifacts
        ),
        changed_paths=(RepositoryPath("CLAIM.md"),),
        claims=(claim,),
        artifacts=artifacts,
        mapping_candidates=mapping_candidates,
        approved_mapping=None,
        mapping_question=None,
        artifact_issues=(),
    )


def test_discovery_boundary_preserves_reference_values_and_explicit_threshold() -> None:
    claim = _claim()
    evidence = (
        _evidence("results/baseline.json", ArtifactKind.RESULTS, ExperimentRole.BASELINE),
        _evidence("results/candidate.json", ArtifactKind.RESULTS, ExperimentRole.CANDIDATE),
    )
    discovery = _discovery(claim, evidence)

    request = planning_request_from_discovery(
        discovery,
        claim_id=CLAIM_ID,
        normalized_evidence=evidence,
    )

    assert request.repository == REPOSITORY
    assert request.head_sha == HEAD_SHA
    assert request.pr_number == 7
    assert request.claim is claim.reference
    assert request.audit_claim.metric == "accuracy"
    assert request.audit_claim.direction.value == "higher"
    assert request.audit_claim.minimum_absolute_improvement == 0.05
    assert tuple(item.role for item in request.audit_claim.claimed_values) == (
        ExperimentRole.BASELINE,
        ExperimentRole.CANDIDATE,
    )
    assert audit_relevant_claim_projection(request.audit_claim) == {
        "claim_id": CLAIM_ID,
        "metric": "accuracy",
        "direction": "higher",
        "minimum_absolute_improvement": 0.05,
    }
    assert request.scientific_claim is claim.scientific_claim
    assert request.claim_policy is not None
    assert request.claim_policy.deterministic_compiler_id == "metric-improvement-v0"


def test_recognized_unsupported_claim_is_partial_before_evidence_or_mapping() -> None:
    claim = _claim(
        "Candidate accuracy is at least 0.90.",
        minimum=0.90,
        provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
    )
    request = _request(claim=claim, mappings=())

    assert request.scientific_claim is not None
    assert type(request.scientific_claim.primary) is AbsoluteMetricClaim
    assert request.audit_claim is None

    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.reason == UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER
    assert outcome.missing_evidence == ()
    assert outcome.mapping_question is None
    assert outcome.plan is None
    assert outcome.evidence_obligations is not None
    assert (
        outcome.evidence_obligations.compiler_state
        is EvidenceObligationState.UNSUPPORTED
    )
    assert outcome.evidence_obligations.obligations
    assert not any(
        isinstance(item.target, ArtifactEvidenceSlot)
        for item in outcome.evidence_obligations.obligations
    )


def test_unrepresentable_metric_threshold_is_exact_compiler_partial() -> None:
    request = _request(
        claim=_claim(
            "Accuracy improved by at least 5 percentage points.",
            minimum=5.0,
        ),
        mappings=(),
    )

    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.reason == UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER
    assert outcome.mapping_question is None
    assert outcome.plan is None


def test_held_out_constraint_does_not_change_audit_claim_or_plan_identity() -> None:
    plain = _request(
        claim=_claim(
            "Accuracy improved from 0.71 to 0.79 by at least 0.05."
        )
    )
    held_out = _request(
        claim=_claim(
            "On held-out data, accuracy improved from 0.71 to 0.79 "
            "by at least 0.05."
        )
    )

    assert plain.audit_claim == held_out.audit_claim
    assert json.dumps(
        to_jsonable(plain.audit_claim),
        sort_keys=True,
        separators=(",", ":"),
    ) == json.dumps(
        to_jsonable(held_out.audit_claim),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert plain.scientific_claim is not None
    assert held_out.scientific_claim is not None
    assert plain.scientific_claim.constraints == ()
    assert [item.kind.value for item in held_out.scientific_claim.constraints] == [
        "held_out"
    ]

    plain_outcome = plan_ephemeral_audit(plain)
    held_out_outcome = plan_ephemeral_audit(held_out)

    assert plain_outcome.plan is not None
    assert held_out_outcome.plan is not None
    assert plain_outcome.plan.plan_id == held_out_outcome.plan.plan_id
    assert held_out_outcome.plan.evidence_obligations is not None
    assert any(
        item.obligation_id == "claim.constraint.held_out"
        and item.state is EvidenceObligationState.SATISFIED
        for item in held_out_outcome.plan.evidence_obligations.obligations
    )
    artifact_obligations = tuple(
        item
        for item in held_out_outcome.plan.evidence_obligations.obligations
        if isinstance(item.target, ArtifactEvidenceSlot)
    )
    assert len(artifact_obligations) == 8
    assert all(
        support.kind is item.target.kind
        and support.role is item.target.role
        and support.dataset_split is item.target.dataset_split
        for item in artifact_obligations
        for support in item.support_references
    )
    assert {
        (item.target.role, item.target.dataset_split)
        for item in artifact_obligations
        if item.target.kind is ArtifactKind.DATASET
    } == {
        (ExperimentRole.BASELINE, DatasetSplit.TRAIN),
        (ExperimentRole.BASELINE, DatasetSplit.EVAL),
        (ExperimentRole.CANDIDATE, DatasetSplit.TRAIN),
        (ExperimentRole.CANDIDATE, DatasetSplit.EVAL),
    }


def test_discovery_boundary_does_not_turn_from_to_values_into_threshold() -> None:
    claim = _claim(
        "Accuracy improved from 0.71 to 0.79.",
        minimum=0.08,
    )
    evidence = (
        _evidence("results/baseline.json", ArtifactKind.RESULTS, ExperimentRole.BASELINE),
        _evidence("results/candidate.json", ArtifactKind.RESULTS, ExperimentRole.CANDIDATE),
    )

    request = planning_request_from_discovery(
        _discovery(claim, evidence),
        claim_id=CLAIM_ID,
        normalized_evidence=evidence,
    )

    assert request.audit_claim.minimum_absolute_improvement is None
    assert request.audit_claim.threshold_provenance is None
    assert tuple(item.value for item in request.audit_claim.claimed_values) == (
        0.71,
        0.79,
    )


def test_discovery_boundary_rejects_unissued_normalized_evidence() -> None:
    claim = _claim()
    issued = (
        _evidence("results/baseline.json", ArtifactKind.RESULTS, ExperimentRole.BASELINE),
        _evidence("results/candidate.json", ArtifactKind.RESULTS, ExperimentRole.CANDIDATE),
    )
    unissued = _evidence(
        "results/other.json",
        ArtifactKind.RESULTS,
        ExperimentRole.CANDIDATE,
    )

    with pytest.raises((TypeError, ValueError), match="issued|artifact|discovery"):
        planning_request_from_discovery(
            _discovery(claim, issued),
            claim_id=CLAIM_ID,
            normalized_evidence=(*issued, unissued),
        )


def test_discovery_boundary_rejects_artifact_outside_issued_repository_index() -> None:
    claim = _claim()
    evidence = (
        _evidence(
            "results/baseline.json",
            ArtifactKind.RESULTS,
            ExperimentRole.BASELINE,
        ),
        _evidence(
            "results/candidate.json",
            ArtifactKind.RESULTS,
            ExperimentRole.CANDIDATE,
        ),
    )
    with pytest.raises((TypeError, ValueError), match="issued|repository|artifact"):
        dataclasses.replace(
            _discovery(claim, evidence),
            repository_paths=(RepositoryPath("CLAIM.md"),),
        )


def test_discovery_boundary_requires_the_concrete_discovery_result_contract() -> None:
    claim = _claim()
    evidence = (
        _evidence("results/baseline.json", ArtifactKind.RESULTS, ExperimentRole.BASELINE),
        _evidence("results/candidate.json", ArtifactKind.RESULTS, ExperimentRole.CANDIDATE),
    )
    concrete = _discovery(claim, evidence)
    structural_clone = SimpleNamespace(
        **{
            field.name: getattr(concrete, field.name)
            for field in dataclasses.fields(DiscoveryResult)
        }
    )

    with pytest.raises(TypeError, match="DiscoveryResult"):
        planning_request_from_discovery(
            structural_clone,
            claim_id=CLAIM_ID,
            normalized_evidence=evidence,
        )


def test_discovery_boundary_scopes_repository_wide_inputs_to_selected_claim() -> None:
    other_claim_id = "claim-2"
    selected_claim = _claim()
    other_claim = dataclasses.replace(
        selected_claim,
        reference=dataclasses.replace(
            selected_claim.reference,
            claim_id=other_claim_id,
            text="F1 improved by at least 0.03.",
        ),
        metric="f1",
        minimum_improvement=ClaimedValue(
            0.03,
            None,
            selected_claim.reference.provenance,
        ),
        scientific_claim=None,
    )
    selected_evidence = (
        _evidence("accuracy/baseline_results.json", ArtifactKind.RESULTS, ExperimentRole.BASELINE),
        _evidence("accuracy/candidate_results.json", ArtifactKind.RESULTS, ExperimentRole.CANDIDATE),
    )
    unrelated_evidence = (
        _evidence(
            "f1/baseline_results.json",
            ArtifactKind.RESULTS,
            ExperimentRole.BASELINE,
            claim_ids=(other_claim_id,),
        ),
        _evidence(
            "f1/candidate_results.json",
            ArtifactKind.RESULTS,
            ExperimentRole.CANDIDATE,
            claim_ids=(other_claim_id,),
        ),
    )
    selected_mapping = _mapping("accuracy-mapping", selected_evidence)
    unrelated_mapping = _mapping("f1-mapping", unrelated_evidence)
    unrelated_question = MappingQuestion(
        question_id="question-f1-results",
        prompt="Which file contains the candidate F1 results?",
        choices=(
            MappingChoice(
                "f1-choice-1",
                "first F1 result",
                (_binding(unrelated_evidence[0], ExperimentRole.CANDIDATE),),
            ),
            MappingChoice(
                "f1-choice-2",
                "second F1 result",
                (_binding(unrelated_evidence[1], ExperimentRole.CANDIDATE),),
            ),
        ),
        relevant_claim_id=other_claim_id,
    )
    all_evidence = (*selected_evidence, *unrelated_evidence)
    discovery = DiscoveryResult(
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=7,
        repository_paths=(RepositoryPath("CLAIM.md"),) + tuple(
            item.artifact.path for item in all_evidence
        ),
        changed_paths=(RepositoryPath("CLAIM.md"),),
        claims=(selected_claim, other_claim),
        artifacts=tuple(item.artifact for item in all_evidence),
        mapping_candidates=(selected_mapping, unrelated_mapping),
        approved_mapping=None,
        mapping_question=unrelated_question,
        artifact_issues=(),
    )

    request = planning_request_from_discovery(
        discovery,
        claim_id=CLAIM_ID,
        normalized_evidence=all_evidence,
    )

    assert request.artifacts == tuple(item.artifact for item in selected_evidence)
    assert request.normalized_evidence == selected_evidence
    assert len(request.mapping_candidates) == 1
    assert {
        (binding.path, binding.kind)
        for binding in request.mapping_candidates[0].bindings
    } == {
        (item.artifact.path, item.artifact.kind) for item in selected_evidence
    }
    assert request.upstream_mapping_question is None


def test_discovery_boundary_scopes_shared_question_choices_to_selected_claim() -> None:
    other_claim_id = "claim-2"
    selected_claim = _claim()
    selected = _evidence(
        "accuracy/candidate_results.json",
        ArtifactKind.RESULTS,
        ExperimentRole.CANDIDATE,
    )
    unrelated = _evidence(
        "f1/candidate_results.json",
        ArtifactKind.RESULTS,
        ExperimentRole.CANDIDATE,
        claim_ids=(other_claim_id,),
    )
    question = MappingQuestion(
        question_id="question-shared-results",
        prompt="Which candidate result should be used?",
        choices=(
            MappingChoice(
                "selected-result",
                "accuracy result",
                (_binding(selected, ExperimentRole.CANDIDATE),),
            ),
            MappingChoice(
                "unrelated-result",
                "F1 result",
                (_binding(unrelated, ExperimentRole.CANDIDATE),),
            ),
        ),
    )
    discovery = DiscoveryResult(
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=7,
        repository_paths=(
            RepositoryPath("CLAIM.md"),
            selected.artifact.path,
            unrelated.artifact.path,
        ),
        changed_paths=(RepositoryPath("CLAIM.md"),),
        claims=(selected_claim,),
        artifacts=(selected.artifact, unrelated.artifact),
        mapping_candidates=(),
        approved_mapping=None,
        mapping_question=question,
        artifact_issues=(),
    )

    request = planning_request_from_discovery(
        discovery,
        claim_id=CLAIM_ID,
        normalized_evidence=(selected, unrelated),
    )

    assert request.upstream_mapping_question is None


def _complete_evidence() -> tuple[NormalizedEvidence, ...]:
    return (
        _evidence(
            "results/baseline.json",
            ArtifactKind.RESULTS,
            ExperimentRole.BASELINE,
        ),
        _evidence(
            "results/candidate.json",
            ArtifactKind.RESULTS,
            ExperimentRole.CANDIDATE,
        ),
        _evidence(
            "configs/baseline.yaml",
            ArtifactKind.CONFIG,
            ExperimentRole.BASELINE,
        ),
        _evidence(
            "configs/candidate.yaml",
            ArtifactKind.CONFIG,
            ExperimentRole.CANDIDATE,
        ),
        _evidence(
            "data/baseline-train.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.BASELINE,
            split="train",
        ),
        _evidence(
            "data/baseline-eval.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.BASELINE,
            split="eval",
        ),
        _evidence(
            "data/candidate-train.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.CANDIDATE,
            split="train",
        ),
        _evidence(
            "data/candidate-eval.jsonl",
            ArtifactKind.DATASET,
            ExperimentRole.CANDIDATE,
            split="eval",
        ),
    )


def _mapping(
    mapping_id: str,
    evidence: tuple[NormalizedEvidence, ...],
    *,
    confidence: float = 0.95,
    trust: MappingTrust = MappingTrust.INFERRED,
    provenance_kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> MappingCandidate:
    return MappingCandidate(
        mapping_id=mapping_id,
        bindings=tuple(
            _binding(item, item.observations[0].experiment_role)
            for item in evidence
        ),
        confidence=Confidence(confidence),
        trust=trust,
        provenance=_provenance(
            provenance_kind,
            source_path="research.yaml"
            if provenance_kind is ProvenanceKind.MANIFEST_HINT
            else "CLAIM.md",
            source_id=mapping_id,
        ),
    )


def _request(
    *,
    claim: DiscoveredClaim | None = None,
    evidence: tuple[NormalizedEvidence, ...] | None = None,
    mappings: tuple[MappingCandidate, ...] | None = None,
    approved: RepoMapping | None = None,
    upstream_question: MappingQuestion | None = None,
) -> PlanningRequest:
    chosen_claim = claim or _claim()
    chosen_evidence = evidence or _complete_evidence()
    chosen_mappings = mappings
    if chosen_mappings is None:
        chosen_mappings = (_mapping("mapping-primary", chosen_evidence),)
    discovery = dataclasses.replace(
        _discovery(chosen_claim, chosen_evidence, chosen_mappings),
        approved_mapping=approved,
        mapping_question=upstream_question,
    )
    return planning_request_from_discovery(
        discovery,
        claim_id=CLAIM_ID,
        normalized_evidence=chosen_evidence,
    )


def _v2_json_passive(
    path: str,
    content: bytes = b'{"accuracy":0.9}\n',
    *,
    snapshot_role: ArtifactSnapshotRole = ArtifactSnapshotRole.HEAD,
    commit: GitCommitSha = HEAD_SHA,
) -> PassiveArtifact:
    candidate = _artifact(path, ArtifactKind.RESULTS, content)
    return passive_artifact_from_snapshot(
        REPOSITORY,
        snapshot_role,
        commit,
        candidate,
        content,
    )


def test_v2_occurrence_identity_matrix_preserves_planning_collision_checks() -> None:
    adapter = JsonAdapter()
    first_passive = _v2_json_passive("results/first.json")
    first_match = adapter.probe(first_passive)
    assert first_match is not None
    first = adapter.extract(first_passive, first_match)

    different_path_passive = _v2_json_passive("results/second.json")
    different_path_match = adapter.probe(different_path_passive)
    assert different_path_match is not None
    different_path = adapter.extract(different_path_passive, different_path_match)
    assert first.artifact.sha256 == different_path.artifact.sha256
    assert first.evidence_id != different_path.evidence_id

    base_passive = _v2_json_passive(
        "results/first.json",
        snapshot_role=ArtifactSnapshotRole.BASE,
    )
    base_match = adapter.probe(base_passive)
    assert base_match is not None
    base = adapter.extract(base_passive, base_match)
    assert base.evidence_id != first.evidence_id

    reread_passive = _v2_json_passive("results/first.json")
    reread_match = adapter.probe(reread_passive)
    assert reread_match is not None
    assert adapter.extract(reread_passive, reread_match).evidence_id == first.evidence_id

    alternate_mapping = dataclasses.replace(
        first_match.mappings[0],
        selector=dataclasses.replace(
            first_match.mappings[0].selector,
            expression="/different_metric",
        ),
    )
    assert _evidence_id(
        adapter.adapter_id,
        adapter.semantic_version,
        first_passive,
        (alternate_mapping,),
    ) != first.evidence_id

    request = _request(
        evidence=(
            *_complete_evidence(),
            first,
            different_path,
        )
    )
    assert request.normalized_evidence[-2:] == (first, different_path)

    with pytest.raises(AnalysisContractError, match="unique"):
        dataclasses.replace(
            request,
            normalized_evidence=(*request.normalized_evidence, first),
        )


def test_shared_benchmark_table_selectors_bind_baseline_and_candidate_independently() -> None:
    content = b"commit,accuracy\nbaseline,0.70\ncandidate,0.80\n"
    artifact = _artifact(
        "benchmarks/shared.csv",
        ArtifactKind.RESULTS,
        content,
    )
    passive = passive_artifact(artifact, content)
    adapter = CsvAdapter()

    def selected(role: ExperimentRole, key: str) -> tuple[NormalizedEvidence, ArtifactBinding]:
        provenance = FieldProvenance(
            ProvenanceKind.ADAPTER_EXTRACTION,
            f"exact table selector; sha256={artifact.sha256}",
            artifact.path,
            f"selector:{key}",
        )
        mapping = FieldMapping(
            "metric_value",
            TableSelector(
                "accuracy",
                (TablePredicate("commit", TableScalarType.STRING, key),),
                1,
                provenance,
            ),
            provenance,
        )
        evidence = adapter.extract(
            passive,
            AdapterMatch(
                adapter.adapter_id,
                artifact.path,
                Confidence(0.95),
                (mapping,),
                (provenance,),
            ),
        )
        return evidence, ArtifactBinding(
            artifact.path,
            artifact.kind,
            role,
            adapter.adapter_id,
            evidence.adapter_match.mappings,
            evidence.adapter_match.match_evidence[0],
        )

    baseline_result, baseline_binding = selected(
        ExperimentRole.BASELINE,
        "baseline",
    )
    candidate_result, candidate_binding = selected(
        ExperimentRole.CANDIDATE,
        "candidate",
    )
    ordinary = tuple(
        item
        for item in _complete_evidence()
        if item.artifact.kind is not ArtifactKind.RESULTS
    )
    evidence = (baseline_result, candidate_result, *ordinary)
    mapping_provenance = _provenance(
        ProvenanceKind.DETERMINISTIC_DISCOVERY,
        source_path=str(artifact.path),
        source_id="shared-table-mapping",
    )
    mapping = MappingCandidate(
        "mapping-shared-table",
        (
            baseline_binding,
            candidate_binding,
            *(
                _binding(item, role)
                for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
                for item in ordinary
                if any(
                    observation.experiment_role is role
                    for observation in item.observations
                )
            ),
        ),
        Confidence(0.95),
        MappingTrust.INFERRED,
        mapping_provenance,
    )
    base = _request()
    request = dataclasses.replace(
        base,
        artifacts=(artifact, *(item.artifact for item in ordinary)),
        normalized_evidence=evidence,
        mapping_candidates=(mapping,),
    )

    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert tuple(item.evidence_id for item in outcome.plan.baseline_evidence).count(
        baseline_result.evidence_id
    ) == 1
    assert tuple(item.evidence_id for item in outcome.plan.candidate_evidence).count(
        candidate_result.evidence_id
    ) == 1

    alternative = dataclasses.replace(
        mapping,
        mapping_id="mapping-shared-table-swapped",
        bindings=(
            dataclasses.replace(
                baseline_binding,
                mappings=candidate_binding.mappings,
            ),
            dataclasses.replace(
                candidate_binding,
                mappings=baseline_binding.mappings,
            ),
            *mapping.bindings[2:],
        ),
    )
    ambiguous = plan_ephemeral_audit(
        dataclasses.replace(
            request,
            mapping_candidates=(mapping, alternative),
        )
    )

    assert ambiguous.state is PlanningState.MAPPING_NEEDED
    assert ambiguous.mapping_question is not None
    assert len(ambiguous.mapping_question.choices) == 2
    assert ambiguous.plan is None


def _identity_only_dataset_evidence(
    evidence: tuple[NormalizedEvidence, ...],
) -> tuple[NormalizedEvidence, ...]:
    values: list[NormalizedEvidence] = []
    for item in evidence:
        if item.artifact.kind is not ArtifactKind.DATASET:
            values.append(item)
            continue
        observations = tuple(
            dataclasses.replace(
                observation,
                experiment_role=ExperimentRole.UNSPECIFIED,
                dataset_references=tuple(
                    dataclasses.replace(reference, split=None)
                    for reference in observation.dataset_references
                ),
            )
            for observation in item.observations
        )
        values.append(dataclasses.replace(item, observations=observations))
    return tuple(values)


def _with_aggregation(
    evidence: tuple[NormalizedEvidence, ...],
    *,
    procedure: str = "arithmetic_mean_v1",
) -> tuple[NormalizedEvidence, ...]:
    values: list[NormalizedEvidence] = []
    for item in evidence:
        if item.artifact.kind is not ArtifactKind.CONFIG:
            values.append(item)
            continue
        provenance = item.adapter_match.match_evidence[0]
        mapping = FieldMapping(
            "config.evaluation.aggregation",
            EvidenceSelector(
                SelectorKind.DOTTED_PATH,
                "evaluation.aggregation",
                provenance,
            ),
            provenance,
        )
        match = dataclasses.replace(
            item.adapter_match,
            mappings=(*item.adapter_match.mappings, mapping),
        )
        observations = tuple(
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
            for observation in item.observations
        )
        values.append(
            dataclasses.replace(item, adapter_match=match, observations=observations)
        )
    return tuple(values)


def test_split_bearing_mapping_makes_identity_only_dataset_evidence_ready() -> None:
    evidence_with_splits = _complete_evidence()
    mapping = _mapping("mapping-split-identity", evidence_with_splits)
    identity_only = _identity_only_dataset_evidence(evidence_with_splits)

    outcome = plan_ephemeral_audit(
        _request(evidence=identity_only, mappings=(mapping,))
    )

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert {
        binding.dataset_split
        for binding in outcome.plan.selected_mapping.bindings
        if binding.kind is ArtifactKind.DATASET
    } == {DatasetSplit.TRAIN, DatasetSplit.EVAL}


def test_explicit_arithmetic_mean_is_ready_only_with_both_recovered_procedures() -> None:
    evidence = _with_aggregation(_complete_evidence())
    claim = _claim(
        "Using the arithmetic mean across supplied runs, accuracy improved "
        "from 0.71 to 0.79 by at least 0.05."
    )

    outcome = plan_ephemeral_audit(
        _request(
            claim=claim,
            evidence=evidence,
            mappings=(_mapping("mapping-mean", evidence),),
        )
    )

    assert outcome.state is PlanningState.READY
    assert outcome.evidence_obligations is not None
    procedure = next(
        item
        for item in outcome.evidence_obligations.obligations
        if item.obligation_id == "measurement.retry_aggregation"
    )
    assert procedure.state is EvidenceObligationState.SATISFIED


def test_missing_required_upstream_procedure_is_partial_not_mean_fallback() -> None:
    claim = _claim(
        "Using the arithmetic mean across supplied runs, accuracy improved "
        "from 0.71 to 0.79 by at least 0.05."
    )

    outcome = plan_ephemeral_audit(_request(claim=claim))

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.plan is None
    assert outcome.mapping_question is None
    assert outcome.reason == "required_measurement_procedure_not_recovered"


@pytest.mark.parametrize(
    "procedure",
    [
        "median",
        "weighted mean",
        "best-of-N",
        "retry filtering",
        "adjudication",
        "geometric mean",
    ],
)
def test_explicit_unsupported_upstream_procedure_is_pre_audit_partial(
    procedure: str,
) -> None:
    claim = _claim(
        f"Using {procedure} across supplied runs, accuracy improved "
        "from 0.71 to 0.79 by at least 0.05."
    )

    outcome = plan_ephemeral_audit(_request(claim=claim))

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.plan is None
    assert outcome.mapping_question is None
    assert outcome.reason == "measurement_procedure_not_supported"


def test_required_procedure_ambiguity_uses_one_bounded_mapping_question() -> None:
    evidence = _with_aggregation(_complete_evidence())
    original_candidate = next(
        item
        for item in evidence
        if item.artifact.kind is ArtifactKind.CONFIG
        and item.observations[0].experiment_role is ExperimentRole.CANDIDATE
    )
    alternative = _with_aggregation(
        (
            _evidence(
                "configs/candidate-alternative.yaml",
                ArtifactKind.CONFIG,
                ExperimentRole.CANDIDATE,
            ),
        )
    )[0]
    second_evidence = tuple(
        alternative if item is original_candidate else item for item in evidence
    )
    all_evidence = (*evidence, alternative)
    claim = _claim(
        "Using the arithmetic mean across supplied runs, accuracy improved "
        "from 0.71 to 0.79 by at least 0.05."
    )
    first_mapping = _mapping("mapping-mean-first", evidence)
    second_mapping = _mapping("mapping-mean-second", second_evidence)

    outcome = plan_ephemeral_audit(
        _request(
            claim=claim,
            evidence=all_evidence,
            mappings=(first_mapping, second_mapping),
        )
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None
    assert len(outcome.mapping_question.choices) == 2
    assert "measurement.retry_aggregation" in (
        outcome.mapping_question.blocking_obligation_ids
    )

    approved = RepoMapping.approve(
        REPOSITORY,
        first_mapping,
        approved_by="pilot-owner",
    )
    resumed = plan_ephemeral_audit(
        _request(
            claim=claim,
            evidence=all_evidence,
            mappings=(first_mapping, second_mapping),
            approved=approved,
        )
    )
    assert resumed.state is PlanningState.READY
    assert resumed.plan is not None
    assert type(resumed.plan.selected_mapping) is RepoMapping


def test_missing_required_procedure_dominates_otherwise_resolvable_mapping() -> None:
    evidence = _complete_evidence()
    alternative = _evidence(
        "configs/candidate-alternative.yaml",
        ArtifactKind.CONFIG,
        ExperimentRole.CANDIDATE,
    )
    original_candidate = next(
        item
        for item in evidence
        if item.artifact.kind is ArtifactKind.CONFIG
        and item.observations[0].experiment_role is ExperimentRole.CANDIDATE
    )
    second_evidence = tuple(
        alternative if item is original_candidate else item for item in evidence
    )
    claim = _claim(
        "Using the arithmetic mean across supplied runs, accuracy improved "
        "from 0.71 to 0.79 by at least 0.05."
    )

    outcome = plan_ephemeral_audit(
        _request(
            claim=claim,
            evidence=(*evidence, alternative),
            mappings=(
                _mapping("mapping-missing-procedure-first", evidence),
                _mapping("mapping-missing-procedure-second", second_evidence),
            ),
        )
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None
    assert outcome.reason == "required_measurement_procedure_not_recovered"


def test_non_authoritative_semantic_proposal_is_carried_to_the_plan_only() -> None:
    request = dataclasses.replace(
        _request(),
        semantic_proposal_provenance=_provenance(
            ProvenanceKind.PROVIDER_PROPOSAL,
            source_path=None,
            source_id="semantic-call-1",
        ),
    )

    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert outcome.plan.semantic_proposal_provenance is request.semantic_proposal_provenance
    assert outcome.plan.selected_mapping is request.mapping_candidates[0]

    with pytest.raises(AnalysisContractError, match="provider provenance"):
        dataclasses.replace(
            request,
            semantic_proposal_provenance=_provenance(
                ProvenanceKind.DETERMINISTIC_DISCOVERY,
                source_id="not-a-provider-call",
            ),
        )


def test_resolvable_upstream_dataset_question_precedes_splitless_partial() -> None:
    evidence_with_splits = _complete_evidence()
    identity_only = _identity_only_dataset_evidence(evidence_with_splits)
    complete_mapping = _mapping("mapping-question-source", evidence_with_splits)
    train = next(
        item
        for item in complete_mapping.bindings
        if item.kind is ArtifactKind.DATASET
        and item.role is ExperimentRole.CANDIDATE
        and item.dataset_split is DatasetSplit.TRAIN
    )
    alternative = dataclasses.replace(
        train,
        path=RepositoryPath("data/candidate-eval.jsonl"),
    )
    question = MappingQuestion(
        question_id="question-candidate-train",
        prompt="Which file contains the candidate train dataset?",
        choices=(
            MappingChoice("choice-candidate-train-1", str(train.path), (train,)),
            MappingChoice(
                "choice-candidate-train-2",
                str(alternative.path),
                (alternative,),
            ),
        ),
        relevant_claim_id=CLAIM_ID,
    )
    incomplete_mapping = dataclasses.replace(
        complete_mapping,
        bindings=tuple(
            item for item in complete_mapping.bindings if item is not train
        ),
    )

    outcome = plan_ephemeral_audit(
        _request(
            evidence=identity_only,
            mappings=(incomplete_mapping,),
            upstream_question=question,
        )
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.mapping_question is not None
    assert outcome.mapping_question.prompt == question.prompt


def test_dataset_split_changes_ephemeral_plan_identity() -> None:
    evidence_with_splits = _complete_evidence()
    identity_only = _identity_only_dataset_evidence(evidence_with_splits)
    original = _mapping("mapping-split-sensitive", evidence_with_splits)
    baseline_datasets = [
        binding
        for binding in original.bindings
        if binding.kind is ArtifactKind.DATASET
        and binding.role is ExperimentRole.BASELINE
    ]
    swapped_bindings = tuple(
        dataclasses.replace(
            binding,
            dataset_split=(
                DatasetSplit.EVAL
                if binding.dataset_split is DatasetSplit.TRAIN
                else DatasetSplit.TRAIN
            ),
        )
        if binding in baseline_datasets
        else binding
        for binding in original.bindings
    )
    swapped = dataclasses.replace(original, bindings=swapped_bindings)

    first = plan_ephemeral_audit(
        _request(evidence=identity_only, mappings=(original,))
    )
    second = plan_ephemeral_audit(
        _request(evidence=identity_only, mappings=(swapped,))
    )

    assert first.state is PlanningState.READY
    assert second.state is PlanningState.READY
    assert first.plan is not None and second.plan is not None
    assert first.plan.plan_id != second.plan.plan_id


def test_threshold_free_claim_is_partial_before_mapping_resolution() -> None:
    first = _mapping("mapping-first", _complete_evidence())
    second = dataclasses.replace(first, mapping_id="mapping-second")

    outcome = plan_ephemeral_audit(
        _request(
            claim=_claim("Accuracy improved.", minimum=None),
            mappings=(first, second),
        )
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.plan is None
    assert outcome.mapping_question is None
    assert outcome.missing_evidence == ()
    assert outcome.reason == "required_threshold_not_recovered"
    assert outcome.evidence_obligations is not None
    threshold = next(
        item
        for item in outcome.evidence_obligations.obligations
        if item.obligation_id == "claim.threshold"
    )
    assert isinstance(threshold.target, ClaimFieldTarget)
    assert threshold.state is EvidenceObligationState.MISSING
    assert (
        threshold.reason
        is EvidenceObligationReason.REQUIRED_THRESHOLD_NOT_RECOVERED
    )


def test_missing_dataset_is_partial_without_a_mapping_question() -> None:
    evidence = tuple(
        item for item in _complete_evidence() if item.artifact.kind is not ArtifactKind.DATASET
    )
    mapping = _mapping("mapping-without-datasets", evidence)

    outcome = plan_ephemeral_audit(_request(evidence=evidence, mappings=(mapping,)))

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None
    assert {
        (item.kind, item.role)
        for item in outcome.missing_evidence
    } == {
        (ArtifactKind.DATASET, ExperimentRole.BASELINE),
        (ArtifactKind.DATASET, ExperimentRole.CANDIDATE),
    }
    assert outcome.evidence_obligations is not None
    assert outcome.evidence_obligations.decision is EvidenceObligationDecision.PARTIAL
    assert {
        item.obligation_id
        for item in outcome.evidence_obligations.obligations
        if item.state is EvidenceObligationState.MISSING
        and isinstance(item.target, ArtifactEvidenceSlot)
    } == {
        "artifact.baseline.dataset.train",
        "artifact.baseline.dataset.eval",
        "artifact.candidate.dataset.train",
        "artifact.candidate.dataset.eval",
    }
    comparison = next(
        item
        for item in outcome.evidence_obligations.obligations
        if item.obligation_id == "comparison.readiness"
    )
    assert comparison.state is EvidenceObligationState.MISSING


def test_missing_one_roles_identity_only_datasets_is_typed_partial_not_question() -> None:
    evidence_with_splits = _complete_evidence()
    identity_only = _identity_only_dataset_evidence(evidence_with_splits)
    complete_mapping = _mapping("mapping-complete", evidence_with_splits)
    evidence = tuple(
        item
        for item in identity_only
        if not (
            item.artifact.kind is ArtifactKind.DATASET
            and "candidate" in str(item.artifact.path)
        )
    )
    mapping = dataclasses.replace(
        complete_mapping,
        mapping_id="mapping-missing-candidate-datasets",
        bindings=tuple(
            item
            for item in complete_mapping.bindings
            if not (
                item.kind is ArtifactKind.DATASET
                and item.role is ExperimentRole.CANDIDATE
            )
        ),
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(mapping,))
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None
    assert {
        (item.kind, item.role)
        for item in outcome.missing_evidence
    } == {(ArtifactKind.DATASET, ExperimentRole.CANDIDATE)}


def test_unrelated_upstream_question_cannot_hide_missing_candidate_datasets() -> None:
    evidence_with_splits = _complete_evidence()
    identity_only = _identity_only_dataset_evidence(evidence_with_splits)
    complete_mapping = _mapping("mapping-complete", evidence_with_splits)
    evidence = tuple(
        item
        for item in identity_only
        if not (
            item.artifact.kind is ArtifactKind.DATASET
            and "candidate" in str(item.artifact.path)
        )
    )
    mapping = dataclasses.replace(
        complete_mapping,
        mapping_id="mapping-missing-candidate-datasets",
        bindings=tuple(
            item
            for item in complete_mapping.bindings
            if not (
                item.kind is ArtifactKind.DATASET
                and item.role is ExperimentRole.CANDIDATE
            )
        ),
    )
    candidate_result = next(
        item
        for item in mapping.bindings
        if item.kind is ArtifactKind.RESULTS
        and item.role is ExperimentRole.CANDIDATE
    )
    baseline_result = next(
        item
        for item in mapping.bindings
        if item.kind is ArtifactKind.RESULTS
        and item.role is ExperimentRole.BASELINE
    )
    result_question = MappingQuestion(
        question_id="question-candidate-results",
        prompt="Which file contains the candidate results?",
        choices=(
            MappingChoice(
                "choice-candidate-results",
                str(candidate_result.path),
                (candidate_result,),
            ),
            MappingChoice(
                "choice-baseline-as-candidate-results",
                str(baseline_result.path),
                (
                    dataclasses.replace(
                        baseline_result,
                        role=ExperimentRole.CANDIDATE,
                    ),
                ),
            ),
        ),
        relevant_claim_id=CLAIM_ID,
    )

    outcome = plan_ephemeral_audit(
        _request(
            evidence=evidence,
            mappings=(mapping,),
            upstream_question=result_question,
        )
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None
    assert {
        (item.kind, item.role)
        for item in outcome.missing_evidence
    } == {(ArtifactKind.DATASET, ExperimentRole.CANDIDATE)}


def test_incomplete_mapping_candidates_cannot_union_into_fake_dataset_coverage() -> None:
    evidence_with_splits = _complete_evidence()
    identity_only = _identity_only_dataset_evidence(evidence_with_splits)
    complete = _mapping("mapping-complete", evidence_with_splits)
    baseline_only = dataclasses.replace(
        complete,
        mapping_id="mapping-baseline-datasets-only",
        bindings=tuple(
            item
            for item in complete.bindings
            if not (
                item.kind is ArtifactKind.DATASET
                and item.role is ExperimentRole.CANDIDATE
            )
        ),
    )
    candidate_only = dataclasses.replace(
        complete,
        mapping_id="mapping-candidate-datasets-only",
        bindings=tuple(
            item
            for item in complete.bindings
            if not (
                item.kind is ArtifactKind.DATASET
                and item.role is ExperimentRole.BASELINE
            )
        ),
    )

    outcome = plan_ephemeral_audit(
        _request(
            evidence=identity_only,
            mappings=(baseline_only, candidate_only),
        )
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None
    assert outcome.missing_evidence == ()
    assert outcome.evidence_obligations is not None
    assert any(
        item.state is EvidenceObligationState.AMBIGUOUS
        and item.reason is EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED
        for item in outcome.evidence_obligations.obligations
    )


@pytest.mark.parametrize("missing_split", [DatasetSplit.TRAIN, DatasetSplit.EVAL])
def test_missing_one_candidate_dataset_split_is_typed_partial(
    missing_split: DatasetSplit,
) -> None:
    evidence_with_splits = _complete_evidence()
    identity_only = _identity_only_dataset_evidence(evidence_with_splits)
    complete_mapping = _mapping("mapping-complete", evidence_with_splits)
    missing_binding = next(
        item
        for item in complete_mapping.bindings
        if item.kind is ArtifactKind.DATASET
        and item.role is ExperimentRole.CANDIDATE
        and item.dataset_split is missing_split
    )
    evidence = tuple(
        item
        for item in identity_only
        if item.artifact.path != missing_binding.path
    )
    mapping = dataclasses.replace(
        complete_mapping,
        mapping_id=f"mapping-missing-candidate-{missing_split.value}",
        bindings=tuple(
            item
            for item in complete_mapping.bindings
            if item is not missing_binding
        ),
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(mapping,))
    )

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None
    assert {
        (item.kind, item.role)
        for item in outcome.missing_evidence
    } == {(ArtifactKind.DATASET, ExperimentRole.CANDIDATE)}


def test_sole_eligible_inferred_mapping_produces_ready_head_bound_plan() -> None:
    request = _request()

    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert outcome.plan.head_sha == request.head_sha
    assert outcome.plan.selected_mapping == request.mapping_candidates[0]
    assert outcome.plan.missing_evidence == ()


def test_manifest_hint_can_auto_plan_only_after_the_standard_hosted_gate() -> None:
    evidence = _complete_evidence()
    hint = _mapping(
        "manifest-hint",
        evidence,
        confidence=0.99,
        trust=MappingTrust.MANIFEST_HINT,
        provenance_kind=ProvenanceKind.MANIFEST_HINT,
    )

    outcome = plan_ephemeral_audit(_request(evidence=evidence, mappings=(hint,)))

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert type(outcome.plan.selected_mapping) is MappingCandidate
    assert outcome.plan.selected_mapping.trust is MappingTrust.MANIFEST_HINT


def test_low_confidence_manifest_hint_requires_clarification() -> None:
    evidence = _complete_evidence()
    hint = _mapping(
        "manifest-hint-low-confidence",
        evidence,
        confidence=0.899,
        trust=MappingTrust.MANIFEST_HINT,
        provenance_kind=ProvenanceKind.MANIFEST_HINT,
    )

    outcome = plan_ephemeral_audit(_request(evidence=evidence, mappings=(hint,)))

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None


def test_applicable_explicit_repo_mapping_is_selected() -> None:
    evidence = _complete_evidence()
    candidate = _mapping(
        "provider-proposal",
        evidence,
        confidence=1.0,
        provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
    )
    approved = RepoMapping.approve(
        REPOSITORY,
        candidate,
        approved_by="owner:amebaleon",
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(candidate,), approved=approved)
    )

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert outcome.plan.selected_mapping is approved
    assert outcome.evidence_obligations is not None
    assert outcome.evidence_obligations.decision is EvidenceObligationDecision.READY
    assert not hasattr(outcome.evidence_obligations, "verdict")


def test_ambiguous_strong_mappings_return_bounded_question_without_verdict() -> None:
    primary = _complete_evidence()
    alternative_result = _evidence(
        "results/candidate-alternative.json",
        ArtifactKind.RESULTS,
        ExperimentRole.CANDIDATE,
    )
    evidence = (*primary, alternative_result)
    first = _mapping("mapping-first", primary)
    second = MappingCandidate(
        mapping_id="mapping-second",
        bindings=tuple(
            _binding(item, item.observations[0].experiment_role)
            for item in (*primary[:1], alternative_result, *primary[2:])
        ),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=_provenance(source_id="mapping-second"),
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(first, second))
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None
    assert 2 <= len(outcome.mapping_question.choices) <= 8
    assert outcome.evidence_obligations is not None
    assert outcome.evidence_obligations.blocking_obligation_ids == (
        "artifact.candidate.results.metric",
    )
    assert outcome.mapping_question.blocking_obligation_ids == (
        "artifact.candidate.results.metric",
    )


def test_provider_mapping_at_full_confidence_never_auto_executes() -> None:
    evidence = _complete_evidence()
    provider_mapping = _mapping(
        "provider-proposal",
        evidence,
        confidence=1.0,
        provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(provider_mapping,))
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None
    assert outcome.evidence_obligations is not None
    assert all(
        item.state is EvidenceObligationState.AMBIGUOUS
        for item in outcome.evidence_obligations.obligations
        if isinstance(item.target, ArtifactEvidenceSlot)
    )


def test_low_confidence_inferred_mapping_requires_clarification() -> None:
    evidence = _complete_evidence()
    low_confidence = _mapping(
        "low-confidence",
        evidence,
        confidence=0.899,
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(low_confidence,))
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.mapping_question is not None
    assert outcome.evidence_obligations is not None
    assert (
        outcome.evidence_obligations.decision
        is EvidenceObligationDecision.MAPPING_NEEDED
    )
    assert outcome.mapping_question.blocking_obligation_ids
    assert outcome.mapping_question.blocking_obligation_ids == (
        outcome.evidence_obligations.blocking_obligation_ids
    )


def test_unissued_selector_is_partial_and_cannot_be_made_executable_by_approval() -> None:
    evidence = _complete_evidence()
    valid = _mapping("mapping-primary", evidence)
    invalid = dataclasses.replace(
        valid,
        bindings=(
            dataclasses.replace(valid.bindings[0], adapter_id="different.adapter"),
            *valid.bindings[1:],
        ),
    )

    outcome = plan_ephemeral_audit(_request(evidence=evidence, mappings=(invalid,)))

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.mapping_question is None
    assert outcome.evidence_obligations is not None
    assert any(
        item.reason is EvidenceObligationReason.SELECTOR_NOT_RECOVERABLE
        for item in outcome.evidence_obligations.obligations
    )


def test_manifest_hint_conflicting_with_strong_candidate_requires_question() -> None:
    primary = _complete_evidence()
    alternative_results = _evidence(
        "results/candidate-alternative.json",
        ArtifactKind.RESULTS,
        ExperimentRole.CANDIDATE,
    )
    evidence = (*primary, alternative_results)
    hint = _mapping(
        "manifest-hint",
        primary,
        trust=MappingTrust.MANIFEST_HINT,
        provenance_kind=ProvenanceKind.MANIFEST_HINT,
    )
    alternative_evidence = tuple(
        alternative_results
        if item.artifact.path == RepositoryPath("results/candidate.json")
        else item
        for item in primary
    )
    alternative = _mapping("strong-alternative", alternative_evidence)

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(hint, alternative))
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.mapping_question is not None


def test_runtime_valid_approved_mapping_precedes_unapproved_conflicts() -> None:
    primary = _complete_evidence()
    alternative_results = _evidence(
        "results/candidate-alternative.json",
        ArtifactKind.RESULTS,
        ExperimentRole.CANDIDATE,
    )
    evidence = (*primary, alternative_results)
    source = _mapping("approved-source", primary)
    approved = RepoMapping.approve(
        REPOSITORY,
        source,
        approved_by="owner:amebaleon",
    )
    conflicting = MappingCandidate(
        mapping_id="strong-conflict",
        bindings=tuple(
            _binding(item, item.observations[0].experiment_role)
            for item in (*primary[:1], alternative_results, *primary[2:])
        ),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=_provenance(source_id="strong-conflict"),
    )

    outcome = plan_ephemeral_audit(
        _request(
            evidence=evidence,
            mappings=(conflicting,),
            approved=approved,
        )
    )

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert outcome.plan.selected_mapping is approved
    assert outcome.mapping_question is None


def test_invalid_approved_mapping_blocks_automatic_fallback() -> None:
    evidence = _complete_evidence()
    valid = _mapping("valid-inferred", evidence)
    invalid_source = dataclasses.replace(
        valid,
        mapping_id="stale-approved-source",
        bindings=(
            dataclasses.replace(
                valid.bindings[0],
                adapter_id="stale.adapter",
            ),
            *valid.bindings[1:],
        ),
    )
    approved = RepoMapping.approve(
        REPOSITORY,
        invalid_source,
        approved_by="owner:amebaleon",
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(valid,), approved=approved)
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None


def test_approved_mapping_with_unissued_exact_head_path_is_unavailable() -> None:
    evidence = _complete_evidence()
    valid = _mapping("valid-inferred", evidence)
    stale_source = dataclasses.replace(
        valid,
        mapping_id="stale-head-source",
        bindings=(
            dataclasses.replace(
                valid.bindings[0],
                path=RepositoryPath("results/removed-at-head.json"),
            ),
            *valid.bindings[1:],
        ),
    )
    approved = RepoMapping.approve(
        REPOSITORY,
        stale_source,
        approved_by="owner:amebaleon",
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(valid,), approved=approved)
    )

    assert outcome.state is PlanningState.UNAVAILABLE
    assert outcome.plan is None
    assert outcome.mapping_question is None
    assert outcome.evidence_obligations is None
    assert outcome.reason == "approved mapping is stale for the exact-head snapshot"


def test_planning_request_rejects_provider_modified_obligation_policy() -> None:
    request = _request()
    assert request.claim_policy is not None
    hostile_policy = dataclasses.replace(
        request.claim_policy,
        obligation_template_ids=("claim.metric",),
    )

    with pytest.raises(ValueError, match="policy|scientific claim"):
        dataclasses.replace(request, claim_policy=hostile_policy)


def test_generic_unspecified_evidence_uses_only_explicit_mapping_roles() -> None:
    role_by_path = {
        item.artifact.path: (
            ExperimentRole.BASELINE
            if "baseline" in str(item.artifact.path)
            else ExperimentRole.CANDIDATE
        )
        for item in _complete_evidence()
    }
    evidence = tuple(
        dataclasses.replace(
            item,
            observations=tuple(
                dataclasses.replace(
                    observation,
                    experiment_role=ExperimentRole.UNSPECIFIED,
                )
                for observation in item.observations
            ),
        )
        for item in _complete_evidence()
    )
    explicit = MappingCandidate(
        mapping_id="explicit-generic-roles",
        bindings=tuple(
            _binding(item, role_by_path[item.artifact.path]) for item in evidence
        ),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=_provenance(source_id="explicit-generic-roles"),
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(explicit,))
    )

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert {
        item.observations[0].experiment_role
        for item in (*outcome.plan.baseline_evidence, *outcome.plan.candidate_evidence)
    } == {ExperimentRole.UNSPECIFIED}


def test_mapping_with_unissued_selector_or_binding_cannot_reach_approval_or_audit() -> None:
    evidence = _complete_evidence()
    valid = _mapping("mapping-primary", evidence)
    invalid_binding = dataclasses.replace(
        valid.bindings[0],
        adapter_id="different.adapter",
    )
    invalid = dataclasses.replace(
        valid,
        bindings=(invalid_binding, *valid.bindings[1:]),
    )

    outcome = plan_ephemeral_audit(_request(evidence=evidence, mappings=(invalid,)))

    assert outcome.state is PlanningState.PARTIAL
    assert outcome.plan is None
    assert outcome.mapping_question is None


def test_planning_outcome_enforces_exactly_one_state_payload() -> None:
    from claimci.analysis import PlanningOutcome

    with pytest.raises((TypeError, ValueError), match="ready|plan|state"):
        PlanningOutcome(PlanningState.READY)
    with pytest.raises((TypeError, ValueError), match="mapping|question|state"):
        PlanningOutcome(PlanningState.MAPPING_NEEDED)
    with pytest.raises((TypeError, ValueError), match="partial|missing|reason|state"):
        PlanningOutcome(PlanningState.PARTIAL)


def _expected_plan_id(request: PlanningRequest, mapping: MappingCandidate) -> str:
    artifacts = {
        (item.artifact.path, item.artifact.kind): item.artifact
        for item in request.normalized_evidence
    }
    bindings = []
    for binding in mapping.bindings:
        artifact = artifacts[(binding.path, binding.kind)]
        bindings.append(
            {
                "path": str(binding.path),
                "sha256": str(artifact.sha256),
                "kind": binding.kind.value,
                "role": binding.role.value,
                "adapter_id": binding.adapter_id,
                "dataset_split": (
                    binding.dataset_split.value
                    if binding.dataset_split is not None
                    else None
                ),
                "selectors": [
                    {
                        "target": item.target_field,
                        "kind": item.selector.kind.value,
                        "expression": item.selector.expression,
                    }
                    for item in sorted(
                        binding.mappings,
                        key=lambda item: (
                            item.target_field,
                            item.selector.kind.value,
                            item.selector.expression,
                        ),
                    )
                ],
            }
        )
    projection = {
        "repository": {
            "owner": request.repository.owner,
            "name": request.repository.name,
        },
        "pr_number": request.pr_number,
        "head_sha": str(request.head_sha),
        "claim": audit_relevant_claim_projection(request.audit_claim),
        "mapping": {
            "mapping_id": mapping.mapping_id,
            "trust": mapping.trust.value,
            "source_provenance_kind": mapping.provenance.kind.value,
            "source_id": mapping.provenance.source_id,
        },
        "bindings": sorted(
            bindings,
            key=lambda item: (
                str(item["path"]),
                str(item["kind"]),
                str(item["role"]),
                str(item["dataset_split"]),
            ),
        ),
    }
    canonical = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "plan-" + hashlib.sha256(canonical).hexdigest()[:24]


def test_plan_id_matches_canonical_audit_relevant_projection() -> None:
    request = _request()
    mapping = request.mapping_candidates[0]

    outcome = plan_ephemeral_audit(request)

    assert outcome.plan is not None
    assert outcome.plan.plan_id == _expected_plan_id(request, mapping)


def test_plan_id_ignores_descriptive_claim_and_discovery_prose() -> None:
    evidence = _complete_evidence()
    mapping = _mapping("mapping-primary", evidence)
    original = _request(evidence=evidence, mappings=(mapping,))
    changed_claim = dataclasses.replace(
        original.claim,
        text="Different provider prose: accuracy rises by at least 0.05.",
    )
    changed_audit_claim = dataclasses.replace(
        original.audit_claim,
        claimed_values=tuple(
            dataclasses.replace(item, value=item.value + 100)
            for item in original.audit_claim.claimed_values
        ),
    )
    changed_mapping = dataclasses.replace(
        original.mapping_candidates[0],
        provenance=dataclasses.replace(
            original.mapping_candidates[0].provenance,
            detail="different unstable provider explanation",
        ),
        bindings=tuple(reversed(original.mapping_candidates[0].bindings)),
    )
    changed_request = dataclasses.replace(
        original,
        claim=changed_claim,
        audit_claim=changed_audit_claim,
        mapping_candidates=(changed_mapping,),
        scientific_claim=None,
        claim_policy=None,
    )

    first = plan_ephemeral_audit(original)
    second = plan_ephemeral_audit(changed_request)

    assert first.plan is not None
    assert second.plan is not None
    assert first.plan.plan_id == second.plan.plan_id


@pytest.mark.parametrize(
    "change",
    ["repository", "head", "metric", "direction", "threshold", "mapping_id"],
)
def test_plan_id_changes_for_deterministic_inputs(change: str) -> None:
    original = _request()
    changed = original
    if change == "repository":
        changed = dataclasses.replace(
            original,
            repository=RepositoryIdentity("amebaleon", "other-repo"),
        )
    elif change == "head":
        changed = dataclasses.replace(original, head_sha=GitCommitSha("b" * 40))
    elif change == "metric":
        changed = dataclasses.replace(
            original,
            audit_claim=dataclasses.replace(original.audit_claim, metric="f1"),
            scientific_claim=None,
            claim_policy=None,
        )
    elif change == "direction":
        from claimci.models import Direction

        changed = dataclasses.replace(
            original,
            audit_claim=dataclasses.replace(
                original.audit_claim,
                direction=Direction.LOWER,
            ),
            scientific_claim=None,
            claim_policy=None,
        )
    elif change == "threshold":
        changed = dataclasses.replace(
            original,
            audit_claim=dataclasses.replace(
                original.audit_claim,
                minimum_absolute_improvement=0.06,
            ),
            scientific_claim=None,
            claim_policy=None,
        )
    else:
        changed_mapping = dataclasses.replace(
            original.mapping_candidates[0],
            mapping_id="mapping-renamed",
        )
        changed = dataclasses.replace(
            original,
            mapping_candidates=(changed_mapping,),
        )

    first = plan_ephemeral_audit(original)
    second = plan_ephemeral_audit(changed)

    assert first.plan is not None
    assert second.plan is not None
    assert first.plan.plan_id != second.plan.plan_id
