"""Deterministic Stale Evidence Detection v1 regressions."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from claimci.analysis import (
    ArtifactRelocation,
    ArtifactKind,
    AuditSemanticsCompatibility,
    DatasetSplit,
    EvidenceTraceBundle,
    ExperimentRole,
    GitCommitSha,
    MappingTrust,
    RequiredAction,
    RepositoryIdentity,
    RepositoryPath,
    ReverificationDecision,
    STALE_EVIDENCE_VERSION,
    SelectorKind,
    Sha256Digest,
    StaleEvidenceAssessment,
    StalenessReason,
    StalenessState,
    VerificationArtifactIdentity,
    VerificationInputSnapshot,
    VerificationSelectorIdentity,
    VerificationSnapshotCapability,
    VerificationTablePredicateIdentity,
    assess_stale_evidence,
    to_jsonable,
)
from claimci.analysis.replay import (
    ReplayAuditCommitment,
    ReplayEngineProvenance,
    ReplayRecipe,
    ReplayTraceReference,
)
from claimci.analysis.verification import (
    EvidenceObligationIdentity,
    EvidenceProfileIdentity,
    MeasurementProtocolReference,
    VerificationClaimIdentity,
    VerificationMappingIdentity,
    _snapshot_instance,
)
from claimci.analysis.profiles import EvidenceProfileId
from claimci.models import AuditResult, Verdict


def _sha(label: str) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(label.encode("utf-8")).hexdigest())


def _artifact(
    *,
    path: str = "results/baseline.json",
    role: ExperimentRole = ExperimentRole.BASELINE,
    kind: ArtifactKind = ArtifactKind.RESULTS,
    split: DatasetSplit | None = None,
    source: str = "source-a",
    extraction: str = "extraction-a",
    semantics: str = "semantics-a",
    selector: str = "/accuracy",
    adapter_id: str = "claimci-structured-results-v1",
    adapter_version: str | None = "v1",
    projector: str = "claimci.projector.training-results.v1",
    selector_identity: VerificationSelectorIdentity | None = None,
) -> VerificationArtifactIdentity:
    selected_identity = selector_identity or VerificationSelectorIdentity(
        "metric_value",
        SelectorKind.JSON_POINTER,
        selector,
    )
    instance = object.__new__(VerificationArtifactIdentity)
    values = {
        "verification_evidence_id": "verification-evidence-"
        + str(_sha(f"{path}:{role.value}:{split}:{selector}:{source}:{semantics}")),
        "path": RepositoryPath(path),
        "kind": kind,
        "role": role,
        "dataset_split": split,
        "source_sha256": _sha(source),
        "extraction_sha256": _sha(extraction),
        "audit_semantics_sha256": _sha(semantics),
        "adapter_id": adapter_id,
        "adapter_semantic_version": adapter_version,
        "evidence_projector_version": projector,
        "selectors": (selected_identity,),
    }
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _table_selector(model: str) -> VerificationSelectorIdentity:
    return VerificationSelectorIdentity(
        target_field="metric_value",
        kind=SelectorKind.COLUMN,
        expression="accuracy",
        table_predicates=(
            VerificationTablePredicateIdentity("model", "string", model),
        ),
        expected_cardinality=1,
    )


def _claim(statement: str = "claim-a", audit_spec: str = "spec-a") -> VerificationClaimIdentity:
    instance = object.__new__(VerificationClaimIdentity)
    object.__setattr__(instance, "claim_id", "claim-1")
    object.__setattr__(instance, "canonical_claim_sha256", _sha(statement))
    object.__setattr__(instance, "audit_spec_sha256", _sha(audit_spec))
    return instance


def _mapping(
    artifacts: tuple[VerificationArtifactIdentity, ...],
    *,
    source_mapping_id: str = "mapping-1",
    trust: MappingTrust = MappingTrust.INFERRED,
) -> VerificationMappingIdentity:
    instance = object.__new__(VerificationMappingIdentity)
    object.__setattr__(instance, "source_mapping_id", source_mapping_id)
    object.__setattr__(instance, "trust", trust)
    object.__setattr__(
        instance,
        "binding_evidence_ids",
        tuple(item.verification_evidence_id for item in artifacts),
    )
    object.__setattr__(
        instance,
        "bindings_sha256",
        _sha("bindings:" + ":".join(str(item.path) for item in artifacts)),
    )
    object.__setattr__(
        instance,
        "approval_sha256",
        _sha(f"approval:{source_mapping_id}")
        if trust is MappingTrust.USER_APPROVED
        else None,
    )
    return instance


def _profile(token: str = "profile-a", *, complete: bool = True) -> EvidenceProfileIdentity:
    instance = object.__new__(EvidenceProfileIdentity)
    object.__setattr__(instance, "selection_id", "selection-1")
    object.__setattr__(instance, "profile_id", EvidenceProfileId.TRAINING_EXPERIMENT_V0)
    object.__setattr__(instance, "profile_version", 0)
    object.__setattr__(instance, "benchmark_variant", None)
    object.__setattr__(instance, "result_form", None)
    object.__setattr__(
        instance,
        "activation_policy_id",
        "claimci.profile.training-experiment.v0",
    )
    object.__setattr__(instance, "complete", complete)
    object.__setattr__(instance, "profile_semantics_sha256", _sha(token))
    return instance


def _obligations(token: str = "obligations-a") -> EvidenceObligationIdentity:
    instance = object.__new__(EvidenceObligationIdentity)
    object.__setattr__(instance, "policy_id", "claimci.obligations.v1")
    object.__setattr__(instance, "bundle_sha256", _sha(f"bundle:{token}"))
    object.__setattr__(instance, "obligation_semantics_sha256", _sha(token))
    object.__setattr__(instance, "obligation_count", 3)
    return instance


def _measurement(semantic: str, source: str) -> MeasurementProtocolReference:
    instance = object.__new__(MeasurementProtocolReference)
    object.__setattr__(
        instance,
        "semantic_protocol_id",
        f"measurement-semantic-{_sha(semantic)}",
    )
    object.__setattr__(instance, "source_snapshot_id", f"measurement-source-{_sha(source)}")
    return instance


def _compatibility(token: str = "compatibility-a") -> AuditSemanticsCompatibility:
    return AuditSemanticsCompatibility._from_fixed_versions(
        audit_policy_semantics=f"claimci.audit.{token}.v1",
        rule_set_semantics="claimci.audit-rules.v1",
        claim_compiler_semantics="claimci.claim-compiler.v0",
        evidence_projector_versions=("claimci.projector.training-results.v1",),
        adapter_semantic_versions=("claimci-structured-results-v1:v1",),
        evidence_profile_semantics=str(_sha("profile-a")),
    )


def _snapshot(
    *,
    head: str = "a" * 40,
    base: str | None = "b" * 40,
    repository_name: str = "ClaimCI",
    artifacts: tuple[VerificationArtifactIdentity, ...] | None = None,
    statement: str = "claim-a",
    audit_spec: str = "spec-a",
    profile: str = "profile-a",
    profile_complete: bool = True,
    obligations: str = "obligations-a",
    measurement_semantic: str = "measurement-a",
    measurement_source: str = "source-snapshot-a",
    compatibility: str = "compatibility-a",
    capability: VerificationSnapshotCapability = VerificationSnapshotCapability.COMPLETE,
    snapshot_token: str = "snapshot-a",
    mapping_identity: VerificationMappingIdentity | None = None,
) -> VerificationInputSnapshot:
    artifact_values = (_artifact(),) if artifacts is None else artifacts
    details = artifact_values if capability is VerificationSnapshotCapability.COMPLETE else ()
    claim = _claim(statement, audit_spec)
    profile_identity = _profile(profile, complete=profile_complete)
    obligation_identity = _obligations(obligations)
    baseline = _measurement(measurement_semantic, measurement_source + ":baseline")
    candidate = _measurement(measurement_semantic, measurement_source + ":candidate")
    audit_compatibility = _compatibility(compatibility)
    return _snapshot_instance(
        repository=RepositoryIdentity("amebaleon", repository_name),
        pr_number=23,
        head_sha=GitCommitSha(head),
        base_sha=None if base is None else GitCommitSha(base),
        claim_identity=claim,
        mapping_identity=(
            _mapping(artifact_values)
            if mapping_identity is None
            else mapping_identity
        ),
        artifacts=details,
        artifact_count=len(artifact_values),
        artifacts_sha256=_sha(
            "artifacts:"
            + ":".join(item.verification_evidence_id for item in artifact_values)
        ),
        profile_identity=profile_identity,
        obligation_identity=obligation_identity,
        baseline_measurement=baseline,
        candidate_measurement=candidate,
        audit_compatibility=audit_compatibility,
        captured_audit_input_sha256=_sha(
            "captured:"
            f"{statement}:{audit_spec}:{profile}:{obligations}:"
            f"{measurement_semantic}:{compatibility}:"
            + ":".join(str(item.audit_semantics_sha256) for item in artifact_values)
        ),
        comparison_frame_sha256=_sha(
            f"frame:{audit_spec}:{profile}:{measurement_semantic}:{compatibility}:"
            + ":".join(str(item.audit_semantics_sha256) for item in artifact_values)
        ),
        input_snapshot_sha256=_sha(snapshot_token),
        capability=capability,
        reason_code=(
            None
            if capability is VerificationSnapshotCapability.COMPLETE
            else "snapshot_size_limit"
        ),
    )


def _recipe(
    snapshot: VerificationInputSnapshot,
    *,
    source_revision: str = "1" * 40,
    verdict: Verdict = Verdict.SUPPORTED,
) -> ReplayRecipe:
    result = AuditResult(
        verdict=verdict,
        findings=(),
        manifest_path=Path("<ephemeral>/research.yaml"),
        metric="accuracy",
        minimum_improvement=0.05,
    )
    commitment = ReplayAuditCommitment.from_audit_result(result)
    trace = EvidenceTraceBundle.unavailable(head_sha=snapshot.head_sha, result=result)
    return ReplayRecipe.from_execution(
        input_snapshot=snapshot,
        engine_provenance=ReplayEngineProvenance.current(source_revision=source_revision),
        audit_commitment=commitment,
        trace_reference=ReplayTraceReference.from_trace(trace, commitment),
    )


def test_canonical_staleness_states_decisions_and_actions_are_stable() -> None:
    assert STALE_EVIDENCE_VERSION == "claimci.stale-evidence.v1"
    assert tuple(item.value for item in StalenessState) == (
        "fresh",
        "byte_only_change",
        "identity_relocation",
        "material_change",
        "indeterminate",
        "legacy_unavailable",
    )
    assert tuple(item.value for item in ReverificationDecision) == (
        "no_reverify_trigger_detected",
        "reverify_required",
        "undetermined",
    )
    assert tuple(item.value for item in RequiredAction) == (
        "none",
        "reverify",
        "remap",
        "supply_provenance",
    )


def test_stale_evidence_assessment_is_metadata_only_immutable_and_final() -> None:
    with pytest.raises(TypeError):
        StaleEvidenceAssessment()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="final"):
        type("ForgedStaleEvidenceAssessment", (StaleEvidenceAssessment,), {})
    with pytest.raises(TypeError):
        ArtifactRelocation()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="final"):
        type("ForgedArtifactRelocation", (ArtifactRelocation,), {})

    assert StaleEvidenceAssessment.scheduled_audit_continues is True
    assert StaleEvidenceAssessment.authorizes_audit_skip is False
    assert "verdict" not in {item.name for item in dataclasses.fields(StaleEvidenceAssessment)}
    assert "finding" not in {item.name for item in dataclasses.fields(StaleEvidenceAssessment)}
    assert "impact" not in {item.name for item in dataclasses.fields(StaleEvidenceAssessment)}


def test_legacy_result_without_replay_is_unavailable_for_staleness() -> None:
    assessment = assess_stale_evidence(None, _snapshot())

    assert assessment.state is StalenessState.LEGACY_UNAVAILABLE
    assert assessment.decision is ReverificationDecision.UNDETERMINED
    assert assessment.required_action is RequiredAction.SUPPLY_PROVENANCE
    assert assessment.reason_codes == (StalenessReason.LEGACY_REPLAY_UNAVAILABLE,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        assessment.state = StalenessState.FRESH  # type: ignore[misc]
    payload = to_jsonable(assessment)
    assert payload["state"] == "legacy_unavailable"  # type: ignore[index]
    assert "verdict" not in payload


def test_unrelated_head_and_base_change_with_identical_inputs_is_fresh() -> None:
    previous = _snapshot(
        head="a" * 40,
        base="b" * 40,
        measurement_source="unchanged-source-snapshot",
        snapshot_token="old-head-snapshot",
    )
    current = _snapshot(
        head="c" * 40,
        base="d" * 40,
        measurement_source="unchanged-source-snapshot",
        snapshot_token="new-head-snapshot",
    )

    assessment = assess_stale_evidence(_recipe(previous), current)

    assert assessment.state is StalenessState.FRESH
    assert assessment.decision is ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED
    assert assessment.required_action is RequiredAction.NONE
    assert StalenessReason.HEAD_ONLY_CHANGE in assessment.reason_codes


def test_measurement_source_change_is_retained_when_head_also_changes() -> None:
    previous = _snapshot(
        head="a" * 40,
        measurement_source="source-at-old-head",
        snapshot_token="old-head-source",
    )
    current = _snapshot(
        head="c" * 40,
        measurement_source="source-at-new-head",
        snapshot_token="new-head-source",
    )

    assessment = assess_stale_evidence(_recipe(previous), current)

    assert assessment.state is StalenessState.BYTE_ONLY_CHANGE
    assert assessment.decision is ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED
    assert StalenessReason.MEASUREMENT_SOURCE_SNAPSHOT_CHANGED in assessment.reason_codes


@pytest.mark.parametrize(
    ("previous", "current", "reason"),
    (
        (
            _snapshot(statement="claim-a", snapshot_token="claim-old"),
            _snapshot(statement="claim-b", snapshot_token="claim-new"),
            StalenessReason.CLAIM_STATEMENT_CHANGED,
        ),
        (
            _snapshot(audit_spec="spec-a", snapshot_token="spec-old"),
            _snapshot(audit_spec="spec-b", snapshot_token="spec-new"),
            StalenessReason.AUDIT_CLAIM_SPEC_CHANGED,
        ),
        (
            _snapshot(profile="profile-a", snapshot_token="profile-old"),
            _snapshot(profile="profile-b", snapshot_token="profile-new"),
            StalenessReason.EVIDENCE_PROFILE_CHANGED,
        ),
        (
            _snapshot(obligations="obligations-a", snapshot_token="obligations-old"),
            _snapshot(obligations="obligations-b", snapshot_token="obligations-new"),
            StalenessReason.EVIDENCE_OBLIGATIONS_CHANGED,
        ),
        (
            _snapshot(measurement_semantic="measurement-a", snapshot_token="measurement-old"),
            _snapshot(measurement_semantic="measurement-b", snapshot_token="measurement-new"),
            StalenessReason.MEASUREMENT_SEMANTIC_PROTOCOL_CHANGED,
        ),
        (
            _snapshot(compatibility="compatibility-a", snapshot_token="compatibility-old"),
            _snapshot(compatibility="compatibility-b", snapshot_token="compatibility-new"),
            StalenessReason.AUDIT_SEMANTICS_COMPATIBILITY_CHANGED,
        ),
    ),
)
def test_semantic_verification_input_changes_require_reverification(
    previous: VerificationInputSnapshot,
    current: VerificationInputSnapshot,
    reason: StalenessReason,
) -> None:
    assessment = assess_stale_evidence(_recipe(previous), current)

    assert assessment.state is StalenessState.MATERIAL_CHANGE
    assert assessment.decision is ReverificationDecision.REVERIFY_REQUIRED
    assert assessment.required_action is RequiredAction.REVERIFY
    assert reason in assessment.reason_codes


def test_measurement_source_only_change_is_not_semantic_measurement_drift() -> None:
    previous = _snapshot(
        measurement_source="measurement-source-a",
        snapshot_token="measurement-source-old",
    )
    current = _snapshot(
        measurement_source="measurement-source-b",
        snapshot_token="measurement-source-new",
    )

    assessment = assess_stale_evidence(_recipe(previous), current)

    assert assessment.state is StalenessState.BYTE_ONLY_CHANGE
    assert assessment.decision is ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED
    assert assessment.required_action is RequiredAction.NONE
    assert assessment.reason_codes == (
        StalenessReason.MEASUREMENT_SOURCE_SNAPSHOT_CHANGED,
    )


def test_core_build_change_is_ignored_when_audit_semantics_compatibility_is_same() -> None:
    snapshot = _snapshot()

    assessment = assess_stale_evidence(
        _recipe(snapshot, source_revision="1" * 40),
        snapshot,
    )

    assert assessment.state is StalenessState.FRESH
    assert assessment.decision is ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED


def test_previous_audit_verdict_is_not_reused_as_staleness_authority() -> None:
    snapshot = _snapshot()

    supported = assess_stale_evidence(
        _recipe(snapshot, verdict=Verdict.SUPPORTED),
        snapshot,
    )
    insufficient = assess_stale_evidence(
        _recipe(snapshot, verdict=Verdict.INSUFFICIENT_EVIDENCE),
        snapshot,
    )

    assert supported.state is insufficient.state is StalenessState.FRESH
    assert supported.decision is insufficient.decision
    assert supported.comparison_sha256 == insufficient.comparison_sha256


def test_unavailable_or_incomplete_current_snapshot_is_indeterminate() -> None:
    previous = _snapshot()
    unavailable = VerificationInputSnapshot.unavailable(
        repository=RepositoryIdentity("amebaleon", "ClaimCI"),
        head_sha=GitCommitSha("c" * 40),
    )
    identity_only = _snapshot(
        capability=VerificationSnapshotCapability.IDENTITY_ONLY,
        snapshot_token="identity-only",
    )
    incomplete_profile = _snapshot(
        profile_complete=False,
        snapshot_token="incomplete-profile",
    )

    for current in (unavailable, identity_only, incomplete_profile):
        assessment = assess_stale_evidence(_recipe(previous), current)
        assert assessment.state is StalenessState.INDETERMINATE
        assert assessment.decision is ReverificationDecision.UNDETERMINED
        assert assessment.required_action is RequiredAction.SUPPLY_PROVENANCE


def test_repository_identity_change_is_indeterminate_not_a_relocation() -> None:
    assessment = assess_stale_evidence(
        _recipe(_snapshot(repository_name="ClaimCI")),
        _snapshot(repository_name="other-repository", snapshot_token="other-repo"),
    )

    assert assessment.state is StalenessState.INDETERMINATE
    assert assessment.required_action is RequiredAction.SUPPLY_PROVENANCE
    assert assessment.reason_codes == (
        StalenessReason.REPOSITORY_IDENTITY_CHANGED,
    )


def test_source_byte_change_with_stable_extraction_and_audit_semantics_is_byte_only() -> None:
    previous_artifact = _artifact(source="source-a")
    current_artifact = _artifact(source="source-b")
    previous = _snapshot(
        artifacts=(previous_artifact,),
        measurement_source="source-a",
        snapshot_token="byte-old",
    )
    current = _snapshot(
        artifacts=(current_artifact,),
        measurement_source="source-b",
        snapshot_token="byte-new",
    )

    assessment = assess_stale_evidence(_recipe(previous), current)

    assert assessment.state is StalenessState.BYTE_ONLY_CHANGE
    assert assessment.decision is ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED
    assert assessment.required_action is RequiredAction.NONE
    assert StalenessReason.SOURCE_BYTES_CHANGED in assessment.reason_codes


def test_order_insensitive_dataset_row_reordering_is_byte_only() -> None:
    previous_artifact = _artifact(
        path="data/baseline-eval.jsonl",
        kind=ArtifactKind.DATASET,
        split=DatasetSplit.EVAL,
        source="rows-a-b",
        extraction="canonical-dataset-extraction",
        semantics="record-multiset-a-b",
        adapter_id="claimci-jsonl-dataset-v1",
        projector="claimci.projector.dataset-jsonl.v1",
    )
    current_artifact = _artifact(
        path="data/baseline-eval.jsonl",
        kind=ArtifactKind.DATASET,
        split=DatasetSplit.EVAL,
        source="rows-b-a",
        extraction="canonical-dataset-extraction",
        semantics="record-multiset-a-b",
        adapter_id="claimci-jsonl-dataset-v1",
        projector="claimci.projector.dataset-jsonl.v1",
    )

    assessment = assess_stale_evidence(
        _recipe(
            _snapshot(
                artifacts=(previous_artifact,),
                measurement_source="rows-a-b",
                snapshot_token="rows-old",
            )
        ),
        _snapshot(
            artifacts=(current_artifact,),
            measurement_source="rows-b-a",
            snapshot_token="rows-new",
        ),
    )

    assert assessment.state is StalenessState.BYTE_ONLY_CHANGE
    assert assessment.required_action is RequiredAction.NONE


def test_changed_artifact_audit_semantics_require_reverification() -> None:
    previous = _snapshot(
        artifacts=(_artifact(semantics="metric-0.8"),),
        snapshot_token="semantics-old",
    )
    current = _snapshot(
        artifacts=(_artifact(semantics="metric-0.9"),),
        snapshot_token="semantics-new",
    )

    assessment = assess_stale_evidence(_recipe(previous), current)

    assert assessment.state is StalenessState.MATERIAL_CHANGE
    assert assessment.decision is ReverificationDecision.REVERIFY_REQUIRED
    assert StalenessReason.ARTIFACT_AUDIT_SEMANTICS_CHANGED in assessment.reason_codes


def test_extraction_change_without_audit_semantic_change_is_indeterminate() -> None:
    previous = _snapshot(
        artifacts=(_artifact(extraction="projection-a"),),
        snapshot_token="extraction-old",
    )
    current = _snapshot(
        artifacts=(_artifact(extraction="projection-b"),),
        snapshot_token="extraction-new",
    )

    assessment = assess_stale_evidence(_recipe(previous), current)

    assert assessment.state is StalenessState.INDETERMINATE
    assert assessment.decision is ReverificationDecision.UNDETERMINED
    assert assessment.required_action is RequiredAction.SUPPLY_PROVENANCE
    assert StalenessReason.EXTRACTION_IDENTITY_CHANGED in assessment.reason_codes


@pytest.mark.parametrize(
    ("before", "after", "reason"),
    (
        (
            _artifact(adapter_id="claimci-structured-results-v1", adapter_version="v1"),
            _artifact(adapter_id="claimci-structured-results-v2", adapter_version="v2"),
            StalenessReason.ADAPTER_SEMANTICS_CHANGED,
        ),
        (
            _artifact(projector="claimci.projector.training-results.v1"),
            _artifact(projector="claimci.projector.training-results.v2"),
            StalenessReason.EVIDENCE_PROJECTOR_CHANGED,
        ),
    ),
)
def test_adapter_or_projector_semantic_change_is_material(
    before: VerificationArtifactIdentity,
    after: VerificationArtifactIdentity,
    reason: StalenessReason,
) -> None:
    assessment = assess_stale_evidence(
        _recipe(_snapshot(artifacts=(before,), snapshot_token="adapter-old")),
        _snapshot(artifacts=(after,), snapshot_token="adapter-new"),
    )

    assert assessment.state is StalenessState.MATERIAL_CHANGE
    assert assessment.required_action is RequiredAction.REVERIFY
    assert reason in assessment.reason_codes


def test_unversioned_adapter_is_incomplete_projector_coverage() -> None:
    artifact = _artifact(adapter_id="fixture.results", adapter_version=None)
    assessment = assess_stale_evidence(
        _recipe(_snapshot(artifacts=(artifact,), snapshot_token="unversioned-old")),
        _snapshot(
            head="c" * 40,
            artifacts=(artifact,),
            measurement_source="source-at-new-head",
            snapshot_token="unversioned-new",
        ),
    )

    assert assessment.state is StalenessState.INDETERMINATE
    assert assessment.required_action is RequiredAction.SUPPLY_PROVENANCE
    assert StalenessReason.REQUIRED_SEMANTIC_COVERAGE_INCOMPLETE in assessment.reason_codes


def test_exact_unversioned_snapshot_remains_indeterminate() -> None:
    artifact = _artifact(adapter_id="fixture.results", adapter_version=None)
    snapshot = _snapshot(artifacts=(artifact,), snapshot_token="unversioned-exact")

    assessment = assess_stale_evidence(_recipe(snapshot), snapshot)

    assert assessment.state is StalenessState.INDETERMINATE
    assert assessment.required_action is RequiredAction.SUPPLY_PROVENANCE


def test_binding_role_change_is_material_even_with_equal_bytes_and_values() -> None:
    before = _artifact(role=ExperimentRole.BASELINE)
    after = _artifact(role=ExperimentRole.CANDIDATE)

    assessment = assess_stale_evidence(
        _recipe(_snapshot(artifacts=(before,), snapshot_token="role-old")),
        _snapshot(artifacts=(after,), snapshot_token="role-new"),
    )

    assert assessment.state is StalenessState.MATERIAL_CHANGE
    assert StalenessReason.MAPPING_BINDING_CHANGED in assessment.reason_codes


def test_mapping_approval_provenance_change_does_not_change_input_semantics() -> None:
    artifact = _artifact()
    previous_mapping = _mapping(
        (artifact,),
        source_mapping_id="mapping-inferred",
        trust=MappingTrust.INFERRED,
    )
    current_mapping = _mapping(
        (artifact,),
        source_mapping_id="mapping-approved",
        trust=MappingTrust.USER_APPROVED,
    )

    assessment = assess_stale_evidence(
        _recipe(
            _snapshot(
                artifacts=(artifact,),
                mapping_identity=previous_mapping,
                snapshot_token="mapping-old",
            )
        ),
        _snapshot(
            artifacts=(artifact,),
            mapping_identity=current_mapping,
            snapshot_token="mapping-new",
        ),
    )

    assert assessment.state is StalenessState.FRESH
    assert assessment.required_action is RequiredAction.NONE
    assert StalenessReason.MAPPING_PROVENANCE_CHANGED in assessment.reason_codes


def test_same_table_selector_change_is_material_even_when_bytes_are_identical() -> None:
    previous_artifacts = (
        _artifact(
            path="bench/results.csv",
            kind=ArtifactKind.BENCHMARK,
            role=ExperimentRole.BASELINE,
            source="shared-table",
            semantics="baseline-0.8",
            selector_identity=_table_selector("baseline"),
            adapter_id="claimci-benchmark-table-v0",
            adapter_version="v0",
            projector="claimci.projector.benchmark-evidence.v1",
        ),
        _artifact(
            path="bench/results.csv",
            kind=ArtifactKind.BENCHMARK,
            role=ExperimentRole.CANDIDATE,
            source="shared-table",
            semantics="candidate-0.9",
            selector_identity=_table_selector("candidate"),
            adapter_id="claimci-benchmark-table-v0",
            adapter_version="v0",
            projector="claimci.projector.benchmark-evidence.v1",
        ),
    )
    current_artifacts = (
        previous_artifacts[0],
        _artifact(
            path="bench/results.csv",
            kind=ArtifactKind.BENCHMARK,
            role=ExperimentRole.CANDIDATE,
            source="shared-table",
            semantics="candidate-0.9",
            selector_identity=_table_selector("candidate-v2"),
            adapter_id="claimci-benchmark-table-v0",
            adapter_version="v0",
            projector="claimci.projector.benchmark-evidence.v1",
        ),
    )

    assessment = assess_stale_evidence(
        _recipe(_snapshot(artifacts=previous_artifacts, snapshot_token="selector-old")),
        _snapshot(artifacts=current_artifacts, snapshot_token="selector-new"),
    )

    assert assessment.state is StalenessState.MATERIAL_CHANGE
    assert assessment.required_action is RequiredAction.REVERIFY
    assert StalenessReason.SELECTOR_IDENTITY_CHANGED in assessment.reason_codes


def test_unique_semantically_equivalent_relocation_requires_remap_not_reverify() -> None:
    previous_artifact = _artifact(path="old/results.json")
    relocated = _artifact(path="new/results.json")
    assessment = assess_stale_evidence(
        _recipe(
            _snapshot(
                artifacts=(previous_artifact,),
                measurement_source="old-path",
                snapshot_token="relocation-old",
            )
        ),
        _snapshot(
            artifacts=(relocated,),
            measurement_source="new-path",
            snapshot_token="relocation-new",
        ),
    )

    assert assessment.state is StalenessState.IDENTITY_RELOCATION
    assert assessment.decision is ReverificationDecision.NO_REVERIFY_TRIGGER_DETECTED
    assert assessment.required_action is RequiredAction.REMAP
    assert len(assessment.relocations) == 1
    assert assessment.relocations[0].previous_path == RepositoryPath("old/results.json")
    assert assessment.relocations[0].current_path == RepositoryPath("new/results.json")


def test_multiple_equivalent_relocation_candidates_are_ambiguous() -> None:
    previous_artifact = _artifact(path="old/results.json")
    first = _artifact(path="new/results-a.json")
    second = _artifact(path="new/results-b.json")
    assessment = assess_stale_evidence(
        _recipe(
            _snapshot(
                artifacts=(previous_artifact,),
                measurement_source="old-path",
                snapshot_token="ambiguous-old",
            )
        ),
        _snapshot(
            artifacts=(first,),
            measurement_source="new-path",
            snapshot_token="ambiguous-new",
        ),
        issued_candidates=(second,),
    )

    assert assessment.state is StalenessState.INDETERMINATE
    assert assessment.decision is ReverificationDecision.UNDETERMINED
    assert assessment.required_action is RequiredAction.REMAP
    assert StalenessReason.RELOCATION_AMBIGUOUS in assessment.reason_codes


def test_missing_relocation_with_different_current_semantics_is_material() -> None:
    previous_artifact = _artifact(path="old/results.json", semantics="metric-0.8")
    changed = _artifact(path="new/results.json", semantics="metric-0.9")
    assessment = assess_stale_evidence(
        _recipe(
            _snapshot(
                artifacts=(previous_artifact,),
                measurement_source="old-path",
                snapshot_token="missing-old",
            )
        ),
        _snapshot(
            artifacts=(changed,),
            measurement_source="new-path",
            snapshot_token="missing-new",
        ),
    )

    assert assessment.state is StalenessState.MATERIAL_CHANGE
    assert assessment.required_action is RequiredAction.REVERIFY
    assert StalenessReason.RELOCATION_NOT_FOUND in assessment.reason_codes


def test_provider_shaped_relocation_candidate_cannot_establish_equivalence() -> None:
    previous = _snapshot(artifacts=(_artifact(path="old/results.json"),))
    current = _snapshot(artifacts=(), snapshot_token="provider-current")

    with pytest.raises(TypeError, match="exact identities"):
        assess_stale_evidence(
            _recipe(previous),
            current,
            issued_candidates=(
                {
                    "path": "new/results.json",
                    "audit_semantics_sha256": str(_sha("semantics-a")),
                    "authority": "deterministic",
                },
            ),  # type: ignore[arg-type]
        )
    RequiredAction,
    ReverificationDecision,
    StaleEvidenceAssessment,
    StalenessReason,
    StalenessState,
