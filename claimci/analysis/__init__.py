"""Shared immutable contracts for ClaimCI zero-configuration analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .confidence import Confidence
from .claims import (
    audit_claim_spec_from_research_spec,
    audit_relevant_claim_projection,
)
from .contracts import (
    Adapter,
    AdapterMatch,
    AdvisoryResearchInterpretation,
    AnalysisAuthority,
    AnalysisContractError,
    AnalysisState,
    ArtifactCandidate,
    ArtifactBinding,
    ArtifactKind,
    AuditClaimSpec,
    ClaimReference,
    ClaimedMetricValue,
    ComputeEvidence,
    ConfigValue,
    DatasetReference,
    DeterministicAuditOutcome,
    EphemeralAuditPlan,
    EvidenceSelector,
    ExperimentRole,
    FieldMapping,
    FieldProvenance,
    GitCommitSha,
    MappingCandidate,
    MappingChoice,
    MappingQuestion,
    MappingTrust,
    MissingEvidence,
    NormalizedEvidence,
    NormalizedObservation,
    PassiveArtifact,
    ProvenanceKind,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    UnifiedAnalysisResult,
    to_jsonable,
)
from .planner import (
    PlanningOutcome,
    PlanningRequest,
    PlanningState,
    derive_ephemeral_plan_id,
    plan_ephemeral_audit,
    planning_request_from_discovery,
)
from .materialize import (
    MaterializationError,
    MaterializationLimits,
    MaterializationPartial,
    MaterializationUnavailable,
    RuntimeExecutionContext,
    execute_ephemeral_audit,
)

if TYPE_CHECKING:
    from claimci.review.models import ReviewConfig
    from claimci.review.provider import ReviewerProvider


def run_unified_analysis(
    request: PlanningRequest,
    runtime: RuntimeExecutionContext,
    review_config: "ReviewConfig",
    *,
    provider: "ReviewerProvider | None" = None,
) -> UnifiedAnalysisResult:
    """Load the hosted orchestration lazily to keep Review imports acyclic."""

    from .integration import run_unified_analysis as implementation

    return implementation(
        request,
        runtime,
        review_config,
        provider=provider,
    )

__all__ = [
    "Adapter",
    "AdapterMatch",
    "AdvisoryResearchInterpretation",
    "AnalysisAuthority",
    "AnalysisContractError",
    "AnalysisState",
    "ArtifactCandidate",
    "ArtifactBinding",
    "ArtifactKind",
    "AuditClaimSpec",
    "ClaimReference",
    "ClaimedMetricValue",
    "ComputeEvidence",
    "Confidence",
    "ConfigValue",
    "DatasetReference",
    "DeterministicAuditOutcome",
    "EphemeralAuditPlan",
    "EvidenceSelector",
    "ExperimentRole",
    "FieldMapping",
    "FieldProvenance",
    "GitCommitSha",
    "MappingCandidate",
    "MappingChoice",
    "MappingQuestion",
    "MappingTrust",
    "MaterializationError",
    "MaterializationLimits",
    "MaterializationPartial",
    "MaterializationUnavailable",
    "MissingEvidence",
    "NormalizedEvidence",
    "NormalizedObservation",
    "PassiveArtifact",
    "PlanningOutcome",
    "PlanningRequest",
    "PlanningState",
    "ProvenanceKind",
    "RepoMapping",
    "RepositoryIdentity",
    "RepositoryPath",
    "RuntimeExecutionContext",
    "SelectorKind",
    "Sha256Digest",
    "UnifiedAnalysisResult",
    "to_jsonable",
    "audit_claim_spec_from_research_spec",
    "audit_relevant_claim_projection",
    "derive_ephemeral_plan_id",
    "execute_ephemeral_audit",
    "plan_ephemeral_audit",
    "planning_request_from_discovery",
    "run_unified_analysis",
]
