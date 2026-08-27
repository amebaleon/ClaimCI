from __future__ import annotations

from claimci.analysis import (
    ArtifactCandidate,
    ArtifactSnapshotRole,
    GitCommitSha,
    PassiveArtifact,
    RepositoryIdentity,
    passive_artifact_from_snapshot,
)


TEST_REPOSITORY = RepositoryIdentity("claimci-tests", "fixture")
TEST_HEAD = GitCommitSha("a" * 40)


def passive_artifact(
    candidate: ArtifactCandidate,
    content: bytes,
    *,
    repository: RepositoryIdentity = TEST_REPOSITORY,
    snapshot_role: ArtifactSnapshotRole = ArtifactSnapshotRole.HEAD,
    commit: GitCommitSha = TEST_HEAD,
) -> PassiveArtifact:
    return passive_artifact_from_snapshot(
        repository,
        snapshot_role,
        commit,
        candidate,
        content,
    )
