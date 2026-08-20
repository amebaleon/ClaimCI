from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import FrozenInstanceError

import pytest

from claimci.analysis import (
    AdapterMatch,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactEvidenceSlot,
    ArtifactEvidenceSupport,
    ArtifactKind,
    CanonicalScientificClaim,
    ClaimFieldKind,
    ClaimFieldSupport,
    ClaimFieldTarget,
    Confidence,
    DatasetSplit,
    EvidenceObligationEffect,
    EvidenceObligation,
    EvidenceObligationBundle,
    EvidenceObligationDecision,
    EvidenceObligationReason,
    EvidenceObligationState,
    EvidenceSelector,
    EvaluationConstraintKind,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    MappingCandidate,
    MappingTrust,
    NormalizedEvidence,
    NormalizedObservation,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    assess_evidence_obligations,
    claim_field_support,
    claim_evidence_policy,
    compile_audit_claim,
    evidence_obligations_json_bytes,
    recover_scientific_claim,
    to_jsonable,
    validated_artifact_support,
)
from claimci.analysis.obligations import _build_obligation_bundle
from claimci.analysis.contracts import ClaimReference


REPOSITORY = RepositoryIdentity("amebaleon", "ClaimCI-Demo")


def _provenance(
    kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
    *,
    path: str = "CLAIM.md",
    source_id: str = "source-claim",
) -> FieldProvenance:
    return FieldProvenance(
        kind=kind,
        detail="validated obligation fixture",
        source_path=RepositoryPath(path),
        source_id=source_id,
    )


def _claim(
    text: str = (
        "On the held-out test set, accuracy improved from 0.70 to 0.80 "
        "by at least 0.05."
    ),
    *,
    provenance_kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> CanonicalScientificClaim:
    reference = ClaimReference(
        claim_id="claim-1",
        text=text,
        source_path=RepositoryPath("CLAIM.md"),
        confidence=Confidence(0.95),
        provenance=_provenance(provenance_kind),
    )
    recovered = recover_scientific_claim(reference)
    assert type(recovered) is CanonicalScientificClaim
    return recovered


def _result_evidence(
    *,
    role: ExperimentRole = ExperimentRole.BASELINE,
    provenance_kind: ProvenanceKind = ProvenanceKind.ADAPTER_EXTRACTION,
) -> tuple[NormalizedEvidence, ArtifactBinding, MappingCandidate]:
    path = RepositoryPath(f"results/{role.value}.json")
    content = b'{"accuracy": 0.7}\n'
    provenance = _provenance(
        provenance_kind,
        path=str(path),
        source_id=f"adapter:{path}",
    )
    selector = EvidenceSelector(
        kind=SelectorKind.JSON_POINTER,
        expression="/accuracy",
        provenance=provenance,
    )
    field_mapping = FieldMapping(
        target_field="metric_value",
        selector=selector,
        provenance=provenance,
    )
    artifact = ArtifactCandidate(
        path=path,
        kind=ArtifactKind.RESULTS,
        sha256=Sha256Digest(hashlib.sha256(content).hexdigest()),
        size=len(content),
        confidence=Confidence(0.95),
        discovery_reason="validated result fixture",
        relevant_claim_ids=("claim-1",),
        provenance=provenance,
    )
    match = AdapterMatch(
        adapter_id="claimci-json-v1",
        path=path,
        confidence=Confidence(0.95),
        mappings=(field_mapping,),
        match_evidence=(provenance,),
    )
    evidence = NormalizedEvidence(
        evidence_id=f"evidence-{role.value}-results",
        artifact=artifact,
        adapter_match=match,
        observations=(
            NormalizedObservation(
                provenance=provenance,
                metric_name="accuracy",
                metric_value=0.7,
                experiment_role=role,
            ),
        ),
    )
    binding = ArtifactBinding(
        path=path,
        kind=ArtifactKind.RESULTS,
        role=role,
        adapter_id=match.adapter_id,
        mappings=match.mappings,
        provenance=provenance,
    )
    mapping = MappingCandidate(
        mapping_id=f"mapping-{role.value}",
        bindings=(binding,),
        confidence=Confidence(0.95),
        trust=MappingTrust.INFERRED,
        provenance=_provenance(
            ProvenanceKind.DETERMINISTIC_DISCOVERY,
            source_id=f"mapping-{role.value}",
        ),
    )
    return evidence, binding, mapping


def test_canonical_enums_and_targets_are_typed_and_immutable() -> None:
    assert {item.value for item in EvidenceObligationState} == {
        "satisfied",
        "missing",
        "ambiguous",
        "unsupported",
    }
    assert EvidenceObligationReason.VALIDATED_SUPPORT_BOUND.value == (
        "validated_support_bound"
    )
    assert EvidenceObligationEffect.BLOCKS_PARTIAL.value == "blocks_partial"

    field = ClaimFieldTarget(ClaimFieldKind.THRESHOLD)
    slot = ArtifactEvidenceSlot(
        ArtifactKind.DATASET,
        ExperimentRole.CANDIDATE,
        DatasetSplit.EVAL,
    )
    with pytest.raises(FrozenInstanceError):
        field.field = ClaimFieldKind.METRIC  # type: ignore[misc]
    with pytest.raises(ValueError, match="dataset split"):
        ArtifactEvidenceSlot(
            ArtifactKind.CONFIG,
            ExperimentRole.BASELINE,
            DatasetSplit.TRAIN,
        )
    with pytest.raises(TypeError, match="ArtifactKind"):
        ArtifactEvidenceSlot(  # type: ignore[arg-type]
            "results",
            ExperimentRole.BASELINE,
        )
    with pytest.raises(TypeError, match="ExperimentRole"):
        ArtifactEvidenceSlot(  # type: ignore[arg-type]
            ArtifactKind.RESULTS,
            "baseline",
        )
    assert slot.dataset_split is DatasetSplit.EVAL


def test_claim_field_support_is_factory_only_and_never_fake_artifact_evidence() -> None:
    claim = _claim()

    metric = claim_field_support(claim, ClaimFieldKind.METRIC)
    threshold = claim_field_support(claim, ClaimFieldKind.THRESHOLD)
    held_out = claim_field_support(
        claim,
        ClaimFieldKind.EVALUATION_CONSTRAINT,
        constraint_kind=EvaluationConstraintKind.HELD_OUT,
    )

    assert type(metric) is ClaimFieldSupport
    assert threshold.field is ClaimFieldKind.THRESHOLD
    assert held_out.constraint_kind is EvaluationConstraintKind.HELD_OUT
    assert not hasattr(threshold, "artifact_kind")
    assert set(to_jsonable(threshold)) == {
        "support_id",
        "claim_id",
        "field",
        "constraint_kind",
    }
    with pytest.raises(TypeError, match="factory"):
        ClaimFieldSupport()  # type: ignore[call-arg]


def test_claim_field_support_re_recovers_source_and_rejects_provider_forgery() -> None:
    claim = _claim(provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL)
    assert claim_field_support(claim, ClaimFieldKind.METRIC).claim_id == "claim-1"

    assert hasattr(claim.primary, "metric")
    forged = dataclasses.replace(
        claim,
        primary=dataclasses.replace(claim.primary, metric="loss"),  # type: ignore[call-arg]
    )
    with pytest.raises(ValueError, match="source|recover|canonical"):
        claim_field_support(forged, ClaimFieldKind.METRIC)
    with pytest.raises(ValueError, match="not present|required"):
        claim_field_support(
            _claim("Accuracy improved from 0.70 to 0.80."),
            ClaimFieldKind.THRESHOLD,
        )


def test_artifact_support_references_exact_validated_evidence_and_binding_ids() -> None:
    evidence, binding, mapping = _result_evidence()

    support = validated_artifact_support(
        evidence,
        binding,
        mapping,
        metric="accuracy",
    )

    assert type(support) is ArtifactEvidenceSupport
    assert support.evidence_id == evidence.evidence_id
    assert support.support_id.startswith("support-artifact-")
    assert support.binding_id.startswith("binding-")
    assert support.kind is ArtifactKind.RESULTS
    assert support.role is ExperimentRole.BASELINE
    assert support.dataset_split is None
    serialized = to_jsonable(support)
    assert set(serialized) == {"support_id", "evidence_id", "binding_id"}
    assert str(evidence.artifact.path) not in str(serialized)
    assert str(evidence.artifact.sha256) not in str(serialized)
    with pytest.raises(TypeError, match="factory"):
        ArtifactEvidenceSupport()  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="target|role|support"):
        EvidenceObligation(
            obligation_id="artifact.candidate.results.metric",
            target=ArtifactEvidenceSlot(
                ArtifactKind.RESULTS,
                ExperimentRole.CANDIDATE,
            ),
            state=EvidenceObligationState.SATISFIED,
            reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
            effect=EvidenceObligationEffect.NONE,
            support_references=(support,),
        )


def test_provider_artifact_support_requires_explicit_repo_mapping_approval() -> None:
    evidence, binding, _ = _result_evidence(
        provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
    )
    provider_mapping = MappingCandidate(
        mapping_id="mapping-provider",
        bindings=(binding,),
        confidence=Confidence(0.99),
        trust=MappingTrust.INFERRED,
        provenance=_provenance(
            ProvenanceKind.PROVIDER_PROPOSAL,
            source_id="provider-call-1",
        ),
    )

    with pytest.raises(ValueError, match="provider|approval|trust"):
        validated_artifact_support(
            evidence,
            binding,
            provider_mapping,
            metric="accuracy",
        )

    approved = RepoMapping.approve(
        REPOSITORY,
        provider_mapping,
        approved_by="owner:amebaleon",
    )
    support = validated_artifact_support(
        evidence,
        binding,
        approved,
        metric="accuracy",
    )
    assert type(support) is ArtifactEvidenceSupport


@pytest.mark.parametrize("mutation", ["role", "selector", "metric"])
def test_artifact_support_fails_closed_on_unrecovered_binding_identity(
    mutation: str,
) -> None:
    evidence, binding, mapping = _result_evidence()
    changed_binding = binding
    changed_metric = "accuracy"
    if mutation == "role":
        changed_binding = dataclasses.replace(
            binding,
            role=ExperimentRole.CANDIDATE,
        )
    elif mutation == "selector":
        provenance = binding.provenance
        changed_binding = dataclasses.replace(
            binding,
            mappings=(
                FieldMapping(
                    "metric_value",
                    EvidenceSelector(
                        SelectorKind.JSON_POINTER,
                        "/loss",
                        provenance,
                    ),
                    provenance,
                ),
            ),
        )
    else:
        changed_metric = "loss"

    with pytest.raises(ValueError, match="binding|selector|metric|role"):
        validated_artifact_support(
            evidence,
            changed_binding,
            mapping,
            metric=changed_metric,
        )


def test_metric_policy_owns_stable_templates_without_fake_manifest_fields() -> None:
    policy = claim_evidence_policy(_claim())

    assert policy.obligation_template_ids == (
        "claim.metric",
        "claim.direction",
        "claim.threshold",
        "claim.constraint.held_out",
        "artifact.baseline.results.metric",
        "artifact.candidate.results.metric",
        "artifact.baseline.config",
        "artifact.candidate.config",
        "artifact.baseline.dataset.train",
        "artifact.baseline.dataset.eval",
        "artifact.candidate.dataset.train",
        "artifact.candidate.dataset.eval",
        "comparison.readiness",
    )
    assert all("manifest" not in item for item in policy.obligation_template_ids)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Candidate accuracy is at least 0.90.",
            ("claim.metric", "claim.quantitative_bound"),
        ),
        ("The candidate generalizes across unseen domains.", ()),
        ("The candidate uses 40% less memory.", ("claim.quantitative_bound",)),
    ],
)
def test_unsupported_policies_do_not_fabricate_executable_artifact_templates(
    text: str,
    expected: tuple[str, ...],
) -> None:
    policy = claim_evidence_policy(_claim(text))

    assert policy.deterministic_compiler_id is None
    assert policy.obligation_template_ids == expected
    assert not any(item.startswith("artifact.") for item in policy.obligation_template_ids)


def _supported_leaf(obligation_id: str, field: ClaimFieldKind) -> EvidenceObligation:
    support = claim_field_support(_claim(), field)
    return EvidenceObligation(
        obligation_id=obligation_id,
        target=ClaimFieldTarget(field),
        state=EvidenceObligationState.SATISFIED,
        reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
        effect=EvidenceObligationEffect.NONE,
        support_references=(support,),
    )


def _missing_leaf(obligation_id: str) -> EvidenceObligation:
    return EvidenceObligation(
        obligation_id=obligation_id,
        target=ArtifactEvidenceSlot(
            ArtifactKind.RESULTS,
            ExperimentRole.CANDIDATE,
        ),
        state=EvidenceObligationState.MISSING,
        reason=EvidenceObligationReason.REQUIRED_ARTIFACT_NOT_FOUND,
        effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
    )


def _composite(
    obligation_id: str,
    dependencies: tuple[str, ...],
) -> EvidenceObligation:
    return EvidenceObligation(
        obligation_id=obligation_id,
        target=None,
        state=EvidenceObligationState.SATISFIED,
        reason=EvidenceObligationReason.VALIDATED_DEPENDENCIES_SATISFIED,
        effect=EvidenceObligationEffect.NONE,
        dependency_ids=dependencies,
    )


def test_bundle_orders_graph_and_derives_composite_and_partial_blockers() -> None:
    policy = claim_evidence_policy(_claim())
    bundle = _build_obligation_bundle(
        claim_id="claim-1",
        policy=policy,
        compiler_supported=True,
        obligations=(
            _composite(
                "comparison.readiness",
                ("claim.metric", "artifact.candidate.results.metric"),
            ),
            _missing_leaf("artifact.candidate.results.metric"),
            _supported_leaf("claim.metric", ClaimFieldKind.METRIC),
        ),
        mapping_question_id=None,
    )

    assert type(bundle) is EvidenceObligationBundle
    assert tuple(item.obligation_id for item in bundle.obligations) == (
        "artifact.candidate.results.metric",
        "claim.metric",
        "comparison.readiness",
    )
    composite = bundle.obligations[-1]
    assert composite.state is EvidenceObligationState.MISSING
    assert composite.reason is EvidenceObligationReason.DEPENDENCY_MISSING
    assert composite.effect is EvidenceObligationEffect.BLOCKS_PARTIAL
    assert bundle.blocking_obligation_ids == (
        "artifact.candidate.results.metric",
    )
    assert bundle.decision is EvidenceObligationDecision.PARTIAL


@pytest.mark.parametrize("defect", ["duplicate", "dangling", "cycle"])
def test_obligation_graph_rejects_duplicate_dangling_and_cyclic_ids(
    defect: str,
) -> None:
    policy = claim_evidence_policy(_claim())
    leaf = _supported_leaf("claim.metric", ClaimFieldKind.METRIC)
    if defect == "duplicate":
        obligations = (leaf, leaf)
    elif defect == "dangling":
        obligations = (leaf, _composite("comparison.readiness", ("missing",)))
    else:
        obligations = (
            _composite("composite.a", ("composite.b",)),
            _composite("composite.b", ("composite.a",)),
        )

    with pytest.raises(ValueError, match=defect):
        _build_obligation_bundle(
            claim_id="claim-1",
            policy=policy,
            compiler_supported=True,
            obligations=obligations,
            mapping_question_id=None,
        )


@pytest.mark.parametrize(
    "blocking_state,blocking_reason",
    [
        (
            EvidenceObligationState.MISSING,
            EvidenceObligationReason.REQUIRED_ARTIFACT_NOT_FOUND,
        ),
        (
            EvidenceObligationState.UNSUPPORTED,
            EvidenceObligationReason.ARTIFACT_FORMAT_NOT_SUPPORTED,
        ),
    ],
)
def test_missing_or_unsupported_dominates_mapping_ambiguity(
    blocking_state: EvidenceObligationState,
    blocking_reason: EvidenceObligationReason,
) -> None:
    policy = claim_evidence_policy(_claim())
    ambiguous = EvidenceObligation(
        obligation_id="artifact.baseline.results.metric",
        target=ArtifactEvidenceSlot(
            ArtifactKind.RESULTS,
            ExperimentRole.BASELINE,
        ),
        state=EvidenceObligationState.AMBIGUOUS,
        reason=EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES,
        effect=EvidenceObligationEffect.REQUIRES_MAPPING,
    )

    blocking = EvidenceObligation(
        obligation_id="artifact.candidate.results.metric",
        target=ArtifactEvidenceSlot(
            ArtifactKind.RESULTS,
            ExperimentRole.CANDIDATE,
        ),
        state=blocking_state,
        reason=blocking_reason,
        effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
    )
    mixed = _build_obligation_bundle(
        claim_id="claim-1",
        policy=policy,
        compiler_supported=True,
        obligations=(ambiguous, blocking),
        mapping_question_id="question-1",
    )
    ambiguity_only = _build_obligation_bundle(
        claim_id="claim-1",
        policy=policy,
        compiler_supported=True,
        obligations=(ambiguous,),
        mapping_question_id="question-1",
    )

    assert mixed.decision is EvidenceObligationDecision.PARTIAL
    assert mixed.mapping_question_id is None
    assert ambiguity_only.decision is EvidenceObligationDecision.MAPPING_NEEDED
    assert ambiguity_only.mapping_question_id == "question-1"


def test_unsupported_compiler_is_structured_partial_without_artifact_obligations() -> None:
    claim = _claim("Candidate accuracy is at least 0.90.")
    policy = claim_evidence_policy(claim)
    bundle = _build_obligation_bundle(
        claim_id="claim-1",
        policy=policy,
        compiler_supported=False,
        obligations=(
            EvidenceObligation(
                obligation_id="claim.metric",
                target=ClaimFieldTarget(ClaimFieldKind.METRIC),
                state=EvidenceObligationState.SATISFIED,
                reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
                effect=EvidenceObligationEffect.NONE,
                support_references=(
                    claim_field_support(claim, ClaimFieldKind.METRIC),
                ),
            ),
        ),
        mapping_question_id=None,
    )

    assert bundle.compiler_state is EvidenceObligationState.UNSUPPORTED
    assert bundle.compiler_reason is EvidenceObligationReason.CLAIM_TYPE_NOT_EXECUTABLE
    assert bundle.decision is EvidenceObligationDecision.PARTIAL
    assert not any(
        isinstance(item.target, ArtifactEvidenceSlot) for item in bundle.obligations
    )


def test_obligation_serialization_is_deterministic_minimal_and_bounded() -> None:
    policy = claim_evidence_policy(_claim())
    bundle = _build_obligation_bundle(
        claim_id="claim-1",
        policy=policy,
        compiler_supported=True,
        obligations=(_supported_leaf("claim.metric", ClaimFieldKind.METRIC),),
        mapping_question_id=None,
    )

    first = evidence_obligations_json_bytes(bundle)
    second = evidence_obligations_json_bytes(bundle)

    assert first == second
    assert b"validated obligation fixture" not in first
    assert b"CLAIM.md" not in first
    assert len(bundle.obligations) <= 16


def test_obligation_contracts_cannot_carry_verdict_authority() -> None:
    forbidden = {"verdict", "authority", "severity", "impact"}
    for contract in (
        ClaimFieldTarget,
        ArtifactEvidenceSlot,
        ClaimFieldSupport,
        ArtifactEvidenceSupport,
        EvidenceObligation,
        EvidenceObligationBundle,
    ):
        assert forbidden.isdisjoint(item.name for item in dataclasses.fields(contract))


@pytest.mark.parametrize(
    "state,reason,effect",
    [
        (
            EvidenceObligationState.SATISFIED,
            EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
            EvidenceObligationEffect.NONE,
        ),
        (
            EvidenceObligationState.MISSING,
            EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES,
            EvidenceObligationEffect.BLOCKS_PARTIAL,
        ),
        (
            EvidenceObligationState.AMBIGUOUS,
            EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES,
            EvidenceObligationEffect.NONE,
        ),
    ],
)
def test_provider_cannot_inject_satisfied_reason_or_effect_state(
    state: EvidenceObligationState,
    reason: EvidenceObligationReason,
    effect: EvidenceObligationEffect,
) -> None:
    with pytest.raises(ValueError, match="state|support|inconsistent"):
        EvidenceObligation(
            obligation_id="artifact.baseline.results.metric",
            target=ArtifactEvidenceSlot(
                ArtifactKind.RESULTS,
                ExperimentRole.BASELINE,
            ),
            state=state,
            reason=reason,
            effect=effect,
        )


def test_provider_cannot_invent_support_identity_or_construct_bundle() -> None:
    with pytest.raises(TypeError, match="factory"):
        ClaimFieldSupport()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="factory"):
        ArtifactEvidenceSupport()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="engine"):
        EvidenceObligationBundle()  # type: ignore[call-arg]


@pytest.mark.parametrize("attack", ["upgrade_unsupported", "change_metric"])
def test_provider_cannot_upgrade_or_rewrite_the_deterministic_compiler(
    attack: str,
) -> None:
    executable = _claim()
    audit_claim = compile_audit_claim(executable)
    claim = executable
    if attack == "upgrade_unsupported":
        claim = _claim("Candidate accuracy is at least 0.90.")
    else:
        audit_claim = dataclasses.replace(audit_claim, metric="loss")

    with pytest.raises(ValueError, match="compiler|audit claim|claim semantics"):
        assess_evidence_obligations(
            claim=claim,
            policy=claim_evidence_policy(claim),
            audit_claim=audit_claim,
            artifacts=(),
            normalized_evidence=(),
            mapping_candidates=(),
            selected_mapping=None,
            mapping_question=None,
            ambiguity_reason=None,
        )


def test_leaf_support_must_match_the_exact_claim_field_target() -> None:
    metric_support = claim_field_support(_claim(), ClaimFieldKind.METRIC)

    with pytest.raises(ValueError, match="target|support|field"):
        EvidenceObligation(
            obligation_id="claim.threshold",
            target=ClaimFieldTarget(ClaimFieldKind.THRESHOLD),
            state=EvidenceObligationState.SATISFIED,
            reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
            effect=EvidenceObligationEffect.NONE,
            support_references=(metric_support,),
        )


def test_bundle_rejects_claim_support_from_another_claim_identity() -> None:
    other = dataclasses.replace(_claim().reference, claim_id="claim-other")
    other_claim = recover_scientific_claim(other)
    assert other_claim is not None
    leaf = EvidenceObligation(
        obligation_id="claim.metric",
        target=ClaimFieldTarget(ClaimFieldKind.METRIC),
        state=EvidenceObligationState.SATISFIED,
        reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
        effect=EvidenceObligationEffect.NONE,
        support_references=(
            claim_field_support(other_claim, ClaimFieldKind.METRIC),
        ),
    )

    with pytest.raises(ValueError, match="claim|support"):
        _build_obligation_bundle(
            claim_id="claim-1",
            policy=claim_evidence_policy(_claim()),
            compiler_supported=True,
            obligations=(leaf,),
            mapping_question_id=None,
        )
