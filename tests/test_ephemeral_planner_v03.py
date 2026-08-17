"""Pure planning behavior for zero-configuration ClaimCI analysis."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

import pytest

from claimci.analysis import (
    AdapterMatch,
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    Confidence,
    ConfigValue,
    DatasetReference,
    EvidenceSelector,
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
    PlanningRequest,
    PlanningState,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    audit_relevant_claim_projection,
    plan_ephemeral_audit,
    planning_request_from_discovery,
    to_jsonable,
)
from claimci.review.models import ClaimDirection, ClaimType, SourceKind, SourceLocation


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


@dataclass(frozen=True, slots=True)
class FakeClaimedValue:
    value: float
    unit: str | None
    provenance: FieldProvenance


@dataclass(frozen=True, slots=True)
class FakeDiscoveredClaim:
    reference: object
    claim_type: ClaimType
    subject: str
    source: SourceLocation
    metric: str | None
    direction: ClaimDirection
    baseline_value: FakeClaimedValue | None = None
    candidate_value: FakeClaimedValue | None = None
    minimum_improvement: FakeClaimedValue | None = None
    qualifiers: tuple[str, ...] = ()
    evidence_hints: tuple[RepositoryPath, ...] = ()


@dataclass(frozen=True, slots=True)
class FakeDiscoveryResult:
    repository: RepositoryIdentity
    head_sha: GitCommitSha
    pr_number: int | None
    repository_paths: tuple[RepositoryPath, ...]
    changed_paths: tuple[RepositoryPath, ...]
    claims: tuple[FakeDiscoveredClaim, ...]
    artifacts: tuple[ArtifactCandidate, ...]
    mapping_candidates: tuple[MappingCandidate, ...]
    approved_mapping: RepoMapping | None = None
    mapping_question: MappingQuestion | None = None
    artifact_issues: tuple[object, ...] = ()

    @property
    def preferred_mapping(self) -> object:
        raise AssertionError("planner must not use preferred_mapping as authority")


def _artifact(path: str, kind: ArtifactKind, content: bytes = b"{}\n") -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=kind,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.95),
        discovery_reason="fixture artifact",
        relevant_claim_ids=(CLAIM_ID,),
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
) -> NormalizedEvidence:
    artifact = _artifact(path, kind)
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
    return ArtifactBinding(
        path=evidence.artifact.path,
        kind=evidence.artifact.kind,
        role=role,
        adapter_id=evidence.adapter_match.adapter_id,
        mappings=evidence.adapter_match.mappings,
        provenance=evidence.adapter_match.match_evidence[0],
    )


def _claim(
    text: str = "Accuracy improved by at least 0.05.",
    *,
    minimum: float | None = 0.05,
    provenance_kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> FakeDiscoveredClaim:
    from claimci.analysis import ClaimReference

    provenance = _provenance(provenance_kind)
    return FakeDiscoveredClaim(
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
        baseline_value=FakeClaimedValue(0.71, None, provenance),
        candidate_value=FakeClaimedValue(0.79, None, provenance),
        minimum_improvement=None
        if minimum is None
        else FakeClaimedValue(minimum, None, provenance),
    )


def _discovery(
    claim: FakeDiscoveredClaim,
    evidence: tuple[NormalizedEvidence, ...],
    mapping_candidates: tuple[MappingCandidate, ...] = (),
) -> FakeDiscoveryResult:
    artifacts = tuple(item.artifact for item in evidence)
    return FakeDiscoveryResult(
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
    discovery = dataclasses.replace(
        _discovery(claim, evidence),
        repository_paths=(RepositoryPath("CLAIM.md"),),
    )

    with pytest.raises((TypeError, ValueError), match="issued|repository|artifact"):
        planning_request_from_discovery(
            discovery,
            claim_id=CLAIM_ID,
            normalized_evidence=evidence,
        )


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
    claim: FakeDiscoveredClaim | None = None,
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
    assert any("threshold" in item.description for item in outcome.missing_evidence)


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


def test_sole_eligible_inferred_mapping_produces_ready_head_bound_plan() -> None:
    request = _request()

    outcome = plan_ephemeral_audit(request)

    assert outcome.state is PlanningState.READY
    assert outcome.plan is not None
    assert outcome.plan.head_sha == request.head_sha
    assert outcome.plan.selected_mapping == request.mapping_candidates[0]
    assert outcome.plan.missing_evidence == ()


def test_manifest_hint_can_guide_ephemeral_plan_without_trust_elevation() -> None:
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


def test_ambiguous_strong_mappings_return_bounded_question_without_verdict() -> None:
    evidence = _complete_evidence()
    first = _mapping("mapping-first", evidence)
    second_bindings = tuple(reversed(first.bindings))
    second = dataclasses.replace(
        first,
        mapping_id="mapping-second",
        bindings=second_bindings,
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(first, second))
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None
    assert 2 <= len(outcome.mapping_question.choices) <= 8


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


def test_manifest_hint_conflicting_with_strong_candidate_requires_question() -> None:
    evidence = _complete_evidence()
    hint = _mapping(
        "manifest-hint",
        evidence,
        trust=MappingTrust.MANIFEST_HINT,
        provenance_kind=ProvenanceKind.MANIFEST_HINT,
    )
    alternative = dataclasses.replace(
        _mapping("strong-alternative", evidence),
        bindings=tuple(reversed(hint.bindings)),
    )

    outcome = plan_ephemeral_audit(
        _request(evidence=evidence, mappings=(hint, alternative))
    )

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.mapping_question is not None


def test_approved_mapping_conflict_is_not_silently_resolved() -> None:
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

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.mapping_question is not None


def test_mapping_with_unissued_selector_or_binding_cannot_auto_execute() -> None:
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

    assert outcome.state is PlanningState.MAPPING_NEEDED
    assert outcome.plan is None
    assert outcome.mapping_question is not None


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
        mapping,
        provenance=dataclasses.replace(
            mapping.provenance,
            detail="different unstable provider explanation",
        ),
        bindings=tuple(reversed(mapping.bindings)),
    )
    changed_request = dataclasses.replace(
        original,
        claim=changed_claim,
        audit_claim=changed_audit_claim,
        mapping_candidates=(changed_mapping,),
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
        )
    elif change == "direction":
        from claimci.models import Direction

        changed = dataclasses.replace(
            original,
            audit_claim=dataclasses.replace(
                original.audit_claim,
                direction=Direction.LOWER,
            ),
        )
    elif change == "threshold":
        changed = dataclasses.replace(
            original,
            audit_claim=dataclasses.replace(
                original.audit_claim,
                minimum_absolute_improvement=0.06,
            ),
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
