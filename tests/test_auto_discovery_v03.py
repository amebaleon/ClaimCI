from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from claimci.analysis import (
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    ClaimReference,
    Confidence,
    ExperimentRole,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingTrust,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    Sha256Digest,
)
from claimci.analysis.discovery import (
    ArtifactIssue,
    ClaimedValue,
    DiscoveredClaim,
    DiscoveryLimits,
    DiscoveryResult,
)
from claimci.review.models import ClaimDirection, ClaimType, SourceKind, SourceLocation


REPOSITORY = RepositoryIdentity(owner="amebaleon", name="ClaimCI-Demo")
HEAD_SHA = GitCommitSha("1" * 40)
ZERO_DIGEST = Sha256Digest("0" * 64)


def _provenance(
    kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
    *,
    path: str | None = None,
    detail: str = "test provenance",
) -> FieldProvenance:
    return FieldProvenance(
        kind=kind,
        detail=detail,
        source_path=None if path is None else RepositoryPath(path),
    )


def _binding(
    path: str,
    kind: ArtifactKind,
    role: ExperimentRole,
    *,
    provenance: FieldProvenance | None = None,
) -> ArtifactBinding:
    return ArtifactBinding(
        path=RepositoryPath(path),
        kind=kind,
        role=role,
        adapter_id=None,
        mappings=(),
        provenance=provenance or _provenance(path=path),
    )


def _mapping(
    mapping_id: str,
    *,
    trust: MappingTrust,
    confidence: float,
    path: str,
) -> MappingCandidate:
    provenance_kind = (
        ProvenanceKind.MANIFEST_HINT
        if trust is MappingTrust.MANIFEST_HINT
        else ProvenanceKind.DETERMINISTIC_DISCOVERY
    )
    provenance = _provenance(
        provenance_kind,
        path=path,
        detail=f"{trust.value} mapping",
    )
    return MappingCandidate(
        mapping_id=mapping_id,
        bindings=(
            _binding(
                path,
                ArtifactKind.RESULTS,
                ExperimentRole.CANDIDATE,
                provenance=provenance,
            ),
        ),
        confidence=Confidence(confidence),
        trust=trust,
        provenance=provenance,
    )


def _claim() -> DiscoveredClaim:
    provenance = _provenance(path="README.md")
    reference = ClaimReference(
        claim_id="claim-accuracy",
        text="Accuracy improved 71 to 79.",
        source_path=RepositoryPath("README.md"),
        confidence=Confidence(0.95),
        provenance=provenance,
    )
    source = SourceLocation(
        source_id="source-readme",
        kind=SourceKind.REPOSITORY_FILE,
        path="README.md",
        start_line=1,
        end_line=1,
    )
    return DiscoveredClaim(
        reference=reference,
        claim_type=ClaimType.METRIC_IMPROVEMENT,
        subject="candidate",
        source=source,
        metric="accuracy",
        direction=ClaimDirection.HIGHER,
        baseline_value=ClaimedValue(71, None, provenance),
        candidate_value=ClaimedValue(79, None, provenance),
    )


def _artifact(path: str, claim_id: str = "claim-accuracy") -> ArtifactCandidate:
    return ArtifactCandidate(
        path=RepositoryPath(path),
        kind=ArtifactKind.RESULTS,
        sha256=ZERO_DIGEST,
        size=0,
        confidence=Confidence(0.8),
        discovery_reason="obvious results filename",
        relevant_claim_ids=(claim_id,),
        provenance=_provenance(path=path),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_artifact_candidates", 0),
        ("max_mapping_candidates", True),
        ("max_artifact_bytes", 0),
        ("max_change_comparison_files", -1),
        ("max_change_comparison_bytes", 0),
        ("max_provider_claims", 65),
        ("max_provider_mappings", 33),
    ],
)
def test_discovery_limits_reject_malformed_or_unbounded_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DiscoveryLimits(**{field: value})


def test_discovery_models_are_immutable_and_validate_finite_claim_values() -> None:
    claim = _claim()

    with pytest.raises(FrozenInstanceError):
        claim.metric = "loss"  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        ClaimedValue(float("nan"), None, _provenance())


def test_discovery_result_prefers_approved_mapping_over_manifest_and_inference() -> None:
    inferred = _mapping(
        "mapping-inferred",
        trust=MappingTrust.INFERRED,
        confidence=1.0,
        path="candidate_results.json",
    )
    manifest = _mapping(
        "mapping-manifest",
        trust=MappingTrust.MANIFEST_HINT,
        confidence=0.99,
        path="manifest_candidate_results.json",
    )
    approved_source = _mapping(
        "mapping-approved-source",
        trust=MappingTrust.INFERRED,
        confidence=0.4,
        path="approved_candidate_results.json",
    )
    approved = RepoMapping.approve(
        REPOSITORY,
        approved_source,
        approved_by="repository-owner",
    )
    claim = _claim()

    result = DiscoveryResult(
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=7,
        repository_paths=(
            RepositoryPath("README.md"),
            RepositoryPath("approved_candidate_results.json"),
            RepositoryPath("candidate_results.json"),
            RepositoryPath("manifest_candidate_results.json"),
        ),
        changed_paths=(RepositoryPath("README.md"),),
        claims=(claim,),
        artifacts=(_artifact("candidate_results.json"),),
        mapping_candidates=(inferred, manifest),
        approved_mapping=approved,
        mapping_question=None,
        artifact_issues=(),
    )

    assert result.preferred_mapping is approved


def test_discovery_result_prefers_manifest_hint_over_higher_confidence_inference() -> None:
    inferred = _mapping(
        "mapping-inferred",
        trust=MappingTrust.INFERRED,
        confidence=1.0,
        path="candidate_results.json",
    )
    manifest = _mapping(
        "mapping-manifest",
        trust=MappingTrust.MANIFEST_HINT,
        confidence=0.99,
        path="manifest_candidate_results.json",
    )

    result = DiscoveryResult(
        repository=REPOSITORY,
        head_sha=HEAD_SHA,
        pr_number=None,
        repository_paths=(
            RepositoryPath("candidate_results.json"),
            RepositoryPath("manifest_candidate_results.json"),
        ),
        changed_paths=(),
        claims=(),
        artifacts=(),
        mapping_candidates=(inferred, manifest),
        approved_mapping=None,
        mapping_question=None,
        artifact_issues=(),
    )

    assert result.preferred_mapping is manifest


def test_discovery_result_rejects_duplicate_identifiers_and_foreign_approval() -> None:
    claim = _claim()
    artifact = _artifact("candidate_results.json")
    approved_source = _mapping(
        "mapping-approved-source",
        trust=MappingTrust.INFERRED,
        confidence=0.8,
        path="candidate_results.json",
    )
    foreign = RepoMapping.approve(
        RepositoryIdentity(owner="other", name="repository"),
        approved_source,
        approved_by="repository-owner",
    )

    with pytest.raises(ValueError, match="repository"):
        DiscoveryResult(
            repository=REPOSITORY,
            head_sha=HEAD_SHA,
            pr_number=1,
            repository_paths=(RepositoryPath("candidate_results.json"),),
            changed_paths=(),
            claims=(),
            artifacts=(artifact,),
            mapping_candidates=(),
            approved_mapping=foreign,
            mapping_question=None,
            artifact_issues=(),
        )

    with pytest.raises(ValueError, match="unique"):
        DiscoveryResult(
            repository=REPOSITORY,
            head_sha=HEAD_SHA,
            pr_number=1,
            repository_paths=(RepositoryPath("README.md"),),
            changed_paths=(),
            claims=(claim, claim),
            artifacts=(),
            mapping_candidates=(),
            approved_mapping=None,
            mapping_question=None,
            artifact_issues=(),
        )


def test_artifact_issue_requires_a_confined_path_and_bounded_reason() -> None:
    issue = ArtifactIssue(RepositoryPath("results/too-large.json"), "file too large")
    assert issue.path == "results/too-large.json"

    with pytest.raises(ValueError):
        ArtifactIssue(RepositoryPath("results.json"), "x" * 4_097)

