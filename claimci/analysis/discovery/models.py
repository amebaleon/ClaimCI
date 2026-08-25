"""Immutable values returned by ClaimCI zero-configuration discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass

from claimci.analysis.confidence import Confidence
from claimci.analysis.claim_types import (
    CanonicalScientificClaim,
    recover_scientific_claim,
)
from claimci.analysis.contracts import (
    ArtifactCandidate,
    ClaimReference,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingQuestion,
    MappingTrust,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
)
from claimci.review.models import ClaimDirection, ClaimType, SourceLocation


class DiscoveryError(ValueError):
    """A controlled discovery input, repository, or proposal failure."""


def _bounded_positive_int(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 1 <= value <= maximum:
        raise DiscoveryError(f"{label} must be between 1 and {maximum}")
    return value


def _bounded_text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiscoveryError(f"{label} must be non-empty text")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        raise DiscoveryError(f"{label} must be bounded printable text")
    return value


@dataclass(frozen=True, slots=True)
class DiscoveryLimits:
    """Bounds below the fixed repository-index limits used by Review."""

    max_artifact_candidates: int = 64
    max_mapping_candidates: int = 16
    max_artifact_bytes: int = 16 * 1024 * 1024
    max_stream_artifact_bytes: int = 1024 * 1024 * 1024
    max_stream_total_bytes: int = 4 * 1024 * 1024 * 1024
    max_change_comparison_files: int = 128
    max_change_comparison_bytes: int = 16 * 1024 * 1024
    max_provider_claims: int = 64
    max_provider_mappings: int = 32

    def __post_init__(self) -> None:
        _bounded_positive_int(
            self.max_artifact_candidates,
            "max_artifact_candidates",
            256,
        )
        _bounded_positive_int(
            self.max_mapping_candidates,
            "max_mapping_candidates",
            64,
        )
        _bounded_positive_int(
            self.max_artifact_bytes,
            "max_artifact_bytes",
            16 * 1024 * 1024,
        )
        _bounded_positive_int(
            self.max_stream_artifact_bytes,
            "max_stream_artifact_bytes",
            16 * 1024 * 1024 * 1024,
        )
        _bounded_positive_int(
            self.max_stream_total_bytes,
            "max_stream_total_bytes",
            64 * 1024 * 1024 * 1024,
        )
        if self.max_stream_artifact_bytes < self.max_artifact_bytes:
            raise DiscoveryError(
                "max_stream_artifact_bytes must be at least max_artifact_bytes"
            )
        if self.max_stream_total_bytes < self.max_stream_artifact_bytes:
            raise DiscoveryError(
                "max_stream_total_bytes must be at least max_stream_artifact_bytes"
            )
        _bounded_positive_int(
            self.max_change_comparison_files,
            "max_change_comparison_files",
            512,
        )
        _bounded_positive_int(
            self.max_change_comparison_bytes,
            "max_change_comparison_bytes",
            16 * 1024 * 1024,
        )
        _bounded_positive_int(
            self.max_provider_claims,
            "max_provider_claims",
            64,
        )
        _bounded_positive_int(
            self.max_provider_mappings,
            "max_provider_mappings",
            32,
        )


@dataclass(frozen=True, slots=True)
class ClaimedValue:
    """One numeric value explicitly present in a trusted source quotation."""

    value: float
    unit: str | None
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("claimed value must be a number")
        normalized = float(self.value)
        if not math.isfinite(normalized):
            raise DiscoveryError("claimed value must be finite")
        object.__setattr__(self, "value", normalized)
        if self.unit is not None:
            _bounded_text(self.unit, "claimed value unit", 64)
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("claimed value provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class DiscoveredClaim:
    """One validated deterministic or provider-proposed scientific claim."""

    reference: ClaimReference
    claim_type: ClaimType
    subject: str
    source: SourceLocation
    metric: str | None = None
    direction: ClaimDirection = ClaimDirection.NOT_APPLICABLE
    baseline_value: ClaimedValue | None = None
    candidate_value: ClaimedValue | None = None
    minimum_improvement: ClaimedValue | None = None
    qualifiers: tuple[str, ...] = ()
    evidence_hints: tuple[RepositoryPath, ...] = ()
    scientific_claim: CanonicalScientificClaim | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ClaimReference):
            raise TypeError("discovered claim reference must be ClaimReference")
        if not isinstance(self.claim_type, ClaimType):
            raise TypeError("discovered claim type must be ClaimType")
        _bounded_text(self.subject, "claim subject", 1_024)
        if not isinstance(self.source, SourceLocation):
            raise TypeError("discovered claim source must be SourceLocation")
        if self.metric is not None:
            _bounded_text(self.metric, "claim metric", 256)
        if not isinstance(self.direction, ClaimDirection):
            raise TypeError("discovered claim direction must be ClaimDirection")
        for label, value in (
            ("baseline_value", self.baseline_value),
            ("candidate_value", self.candidate_value),
            ("minimum_improvement", self.minimum_improvement),
        ):
            if value is not None and not isinstance(value, ClaimedValue):
                raise TypeError(f"{label} must be ClaimedValue or null")
        if (self.baseline_value is None) != (self.candidate_value is None):
            raise DiscoveryError(
                "baseline and candidate claimed values must appear together"
            )
        if not isinstance(self.qualifiers, tuple) or not all(
            isinstance(item, str) and bool(item.strip()) and len(item) <= 1_024
            for item in self.qualifiers
        ):
            raise DiscoveryError(
                "claim qualifiers must be a tuple of bounded non-empty strings"
            )
        if len(set(self.qualifiers)) != len(self.qualifiers):
            raise DiscoveryError("claim qualifiers must be unique")
        if not isinstance(self.evidence_hints, tuple) or not all(
            isinstance(item, RepositoryPath) for item in self.evidence_hints
        ):
            raise TypeError("claim evidence_hints must be RepositoryPath values")
        if len(set(self.evidence_hints)) != len(self.evidence_hints):
            raise DiscoveryError("claim evidence_hints must be unique")
        if self.source.kind.value == "repository_file":
            if self.reference.source_path != self.source.path:
                raise DiscoveryError(
                    "claim reference and source location paths must match"
                )
        elif self.reference.source_path is not None:
            raise DiscoveryError("pull-request claim reference cannot carry a path")
        recovered = recover_scientific_claim(self.reference)
        if self.scientific_claim is None:
            object.__setattr__(self, "scientific_claim", recovered)
        elif type(self.scientific_claim) is not CanonicalScientificClaim:
            raise TypeError(
                "discovered scientific_claim must be CanonicalScientificClaim or null"
            )
        elif self.scientific_claim != recovered:
            raise DiscoveryError(
                "discovered scientific claim conflicts with source recovery"
            )


@dataclass(frozen=True, slots=True)
class ArtifactIssue:
    """One safely isolated repository artifact and its bounded reason."""

    path: RepositoryPath
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, RepositoryPath):
            object.__setattr__(self, "path", RepositoryPath(self.path))
        _bounded_text(self.reason, "artifact issue reason", 4_096)


def _tuple_of(values: object, item_type: type, label: str) -> tuple:
    if not isinstance(values, tuple) or not all(
        isinstance(item, item_type) for item in values
    ):
        raise TypeError(f"{label} must be a tuple of {item_type.__name__} values")
    return values


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Immutable output of one bounded discovery pass."""

    repository: RepositoryIdentity
    head_sha: GitCommitSha
    pr_number: int | None
    repository_paths: tuple[RepositoryPath, ...]
    changed_paths: tuple[RepositoryPath, ...]
    claims: tuple[DiscoveredClaim, ...]
    artifacts: tuple[ArtifactCandidate, ...]
    mapping_candidates: tuple[MappingCandidate, ...]
    approved_mapping: RepoMapping | None
    mapping_question: MappingQuestion | None
    artifact_issues: tuple[ArtifactIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.repository, RepositoryIdentity):
            raise TypeError("discovery repository must be RepositoryIdentity")
        if not isinstance(self.head_sha, GitCommitSha):
            object.__setattr__(self, "head_sha", GitCommitSha(self.head_sha))
        if self.pr_number is not None and (
            isinstance(self.pr_number, bool)
            or not isinstance(self.pr_number, int)
            or self.pr_number < 1
        ):
            raise DiscoveryError("discovery pr_number must be a positive integer")
        _tuple_of(self.repository_paths, RepositoryPath, "repository_paths")
        _tuple_of(self.changed_paths, RepositoryPath, "changed_paths")
        _tuple_of(self.claims, DiscoveredClaim, "claims")
        _tuple_of(self.artifacts, ArtifactCandidate, "artifacts")
        _tuple_of(self.mapping_candidates, MappingCandidate, "mapping_candidates")
        _tuple_of(self.artifact_issues, ArtifactIssue, "artifact_issues")

        if len(set(self.repository_paths)) != len(self.repository_paths):
            raise DiscoveryError("repository_paths must be unique")
        if not set(self.changed_paths).issubset(self.repository_paths):
            raise DiscoveryError("changed_paths must be issued repository paths")
        claim_ids = tuple(item.reference.claim_id for item in self.claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise DiscoveryError("claim identifiers must be unique")
        artifact_paths = tuple(item.path for item in self.artifacts)
        if len(set(artifact_paths)) != len(artifact_paths):
            raise DiscoveryError("artifact candidate paths must be unique")
        mapping_ids = tuple(item.mapping_id for item in self.mapping_candidates)
        if len(set(mapping_ids)) != len(mapping_ids):
            raise DiscoveryError("mapping candidate identifiers must be unique")
        issue_paths = tuple(item.path for item in self.artifact_issues)
        if len(set(issue_paths)) != len(issue_paths):
            raise DiscoveryError("artifact issue paths must be unique")
        issued = set(self.repository_paths)
        if any(item.path not in issued for item in self.artifacts):
            raise DiscoveryError("artifact candidates must use issued repository paths")
        if self.approved_mapping is not None:
            if not isinstance(self.approved_mapping, RepoMapping):
                raise TypeError("approved_mapping must be RepoMapping or null")
            if self.approved_mapping.repository != self.repository:
                raise DiscoveryError(
                    "approved mapping repository does not match discovery repository"
                )
        if self.mapping_question is not None:
            if not isinstance(self.mapping_question, MappingQuestion):
                raise TypeError("mapping_question must be MappingQuestion or null")
            relevant = self.mapping_question.relevant_claim_id
            if relevant is not None and relevant not in set(claim_ids):
                raise DiscoveryError("mapping question cites an unknown claim")

    @property
    def preferred_mapping(self) -> RepoMapping | MappingCandidate | None:
        """Return the highest-precedence proposal without elevating its trust."""

        if self.approved_mapping is not None:
            return self.approved_mapping
        if not self.mapping_candidates:
            return None
        tier = {
            MappingTrust.MANIFEST_HINT: 0,
            MappingTrust.INFERRED: 1,
        }
        return min(
            self.mapping_candidates,
            key=lambda item: (
                tier[item.trust],
                -item.confidence.value,
                item.mapping_id,
            ),
        )


__all__ = [
    "ArtifactIssue",
    "ClaimedValue",
    "DiscoveredClaim",
    "DiscoveryError",
    "DiscoveryLimits",
    "DiscoveryResult",
]
