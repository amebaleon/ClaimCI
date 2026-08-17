"""Public orchestration for one bounded zero-configuration discovery pass."""

from __future__ import annotations

from pathlib import Path

from claimci.analysis.contracts import (
    GitCommitSha,
    RepoMapping,
    RepositoryIdentity,
)

from .artifacts import discover_artifacts
from .claims import discover_claims
from .mappings import resolve_mappings
from .models import DiscoveryLimits, DiscoveryResult
from .repository import collect_repository_context


def discover_repository(
    head_root: Path,
    *,
    repository: RepositoryIdentity,
    head_sha: GitCommitSha,
    pr_number: int | None = None,
    base_root: Path | None = None,
    pr_title: str = "",
    pr_description: str = "",
    approved_mapping: RepoMapping | None = None,
    provider_claim_payload: object | None = None,
    provider_mapping_payload: object | None = None,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> DiscoveryResult:
    """Discover passive claim/evidence proposals without executing authority."""

    if not isinstance(repository, RepositoryIdentity):
        raise TypeError("repository must be RepositoryIdentity")
    if not isinstance(head_sha, GitCommitSha):
        head_sha = GitCommitSha(head_sha)
    if not isinstance(limits, DiscoveryLimits):
        raise TypeError("limits must be DiscoveryLimits")
    context = collect_repository_context(
        head_root,
        base_root=base_root,
        pr_title=pr_title,
        pr_description=pr_description,
        limits=limits,
    )
    claims = discover_claims(
        context,
        provider_payload=provider_claim_payload,
        limits=limits,
    )
    artifact_discovery = discover_artifacts(
        context,
        claims,
        approved_mapping=approved_mapping,
        limits=limits,
    )
    mapping_resolution = resolve_mappings(
        artifact_discovery.artifacts,
        claims,
        artifact_discovery.manifest_mappings,
        repository=repository,
        approved_mapping=approved_mapping,
        provider_payload=provider_mapping_payload,
        limits=limits,
    )
    return DiscoveryResult(
        repository=repository,
        head_sha=head_sha,
        pr_number=pr_number,
        repository_paths=context.repository_paths,
        changed_paths=context.changed_paths,
        claims=claims,
        artifacts=artifact_discovery.artifacts,
        mapping_candidates=mapping_resolution.candidates,
        approved_mapping=mapping_resolution.approved_mapping,
        mapping_question=mapping_resolution.question,
        artifact_issues=artifact_discovery.issues,
    )


__all__ = ["discover_repository"]
