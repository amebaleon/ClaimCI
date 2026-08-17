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
    DiscoveryError,
    DiscoveryLimits,
    DiscoveryResult,
)
from claimci.analysis.discovery.repository import (
    collect_repository_context,
    inspect_artifact,
    read_artifact_text,
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


def test_repository_context_reuses_bounded_review_index_and_changed_sources(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    (base / "README.md").write_text("Old claim.\n", encoding="utf-8")
    (head / "README.md").write_text("Accuracy improved.\n", encoding="utf-8")
    (head / "candidate_results.json").write_text("{}\n", encoding="utf-8")

    context = collect_repository_context(
        head,
        base_root=base,
        pr_title="Candidate study",
        pr_description="Bounded description",
        limits=DiscoveryLimits(),
    )

    assert context.repository_paths == (
        RepositoryPath("README.md"),
        RepositoryPath("candidate_results.json"),
    )
    assert context.changed_paths == (RepositoryPath("README.md"),)
    assert tuple(source.text for source in context.source_bundle.sources) == (
        "Candidate study",
        "Bounded description",
        "Accuracy improved.\n",
    )


def test_repository_context_excludes_symlinked_files_and_directories(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    head = tmp_path / "head"
    outside.mkdir()
    head.mkdir()
    (outside / "secret.json").write_text('{"secret": true}\n', encoding="utf-8")
    (head / "safe_results.json").write_text("{}\n", encoding="utf-8")
    try:
        (head / "linked_results.json").symlink_to(outside / "secret.json")
        (head / "linked_directory").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    context = collect_repository_context(head, limits=DiscoveryLimits())

    assert context.repository_paths == (RepositoryPath("safe_results.json"),)


def test_artifact_inspection_rejects_unindexed_paths_and_isolates_oversize(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    (head / "small_results.json").write_bytes(b"{}\n")
    (head / "large_results.json").write_bytes(b"x" * 33)
    limits = DiscoveryLimits(max_artifact_bytes=32)
    context = collect_repository_context(head, limits=limits)

    small = inspect_artifact(
        context,
        RepositoryPath("small_results.json"),
        limits=limits,
    )
    large = inspect_artifact(
        context,
        RepositoryPath("large_results.json"),
        limits=limits,
    )

    assert small.path == "small_results.json"
    assert small.size == 3
    assert small.sha256 == Sha256Digest(
        "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
    )
    assert large == ArtifactIssue(
        RepositoryPath("large_results.json"),
        "artifact exceeds the 32-byte discovery limit",
    )
    with pytest.raises(DiscoveryError, match="indexed"):
        inspect_artifact(
            context,
            RepositoryPath("missing.json"),
            limits=limits,
        )


def test_artifact_text_isolates_invalid_utf8_without_losing_other_artifacts(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    (head / "good.yaml").write_text("metric: accuracy\n", encoding="utf-8")
    (head / "bad.yaml").write_bytes(b"\xff\xfe\x00")
    context = collect_repository_context(head, limits=DiscoveryLimits())

    good = read_artifact_text(
        context,
        RepositoryPath("good.yaml"),
        limits=DiscoveryLimits(),
    )
    bad = read_artifact_text(
        context,
        RepositoryPath("bad.yaml"),
        limits=DiscoveryLimits(),
    )

    assert good[1] == "metric: accuracy\n"
    assert bad == ArtifactIssue(
        RepositoryPath("bad.yaml"),
        "artifact is not valid UTF-8 text",
    )


def test_repository_context_is_deterministic_for_creation_order(tmp_path: Path) -> None:
    head = tmp_path / "head"
    head.mkdir()
    for name in ("z_results.json", "A_config.yaml", "middle.md"):
        (head / name).write_text(f"{name}\n", encoding="utf-8")

    first = collect_repository_context(head, limits=DiscoveryLimits())
    second = collect_repository_context(head, limits=DiscoveryLimits())

    assert first.repository_paths == second.repository_paths == (
        RepositoryPath("A_config.yaml"),
        RepositoryPath("middle.md"),
        RepositoryPath("z_results.json"),
    )


def test_repository_context_rejects_repository_depth_beyond_existing_bound(
    tmp_path: Path,
) -> None:
    head = tmp_path / "head"
    head.mkdir()
    cursor = head
    for _ in range(66):
        cursor = cursor / "d"
        cursor.mkdir()
    (cursor / "results.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DiscoveryError, match="repository"):
        collect_repository_context(head, limits=DiscoveryLimits())
