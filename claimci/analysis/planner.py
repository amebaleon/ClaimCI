"""Pure trust selection and ephemeral audit planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from claimci.review.models import SourceKind, SourceLocation

from .claim_types import (
    CanonicalScientificClaim,
    ClaimEvidencePolicy,
    MetricImprovementClaim,
    UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER,
    UnsupportedDeterministicClaimCompiler,
    claim_evidence_policy,
    compile_audit_claim,
)
from .confidence import Confidence
from .contracts import (
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    AuditClaimSpec,
    ClaimReference,
    DatasetSplit,
    EphemeralAuditPlan,
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
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    ProvenanceKind,
)
from .claims import audit_relevant_claim_projection
from .obligations import (
    EvidenceObligationBundle,
    EvidenceObligationDecision,
    EvidenceObligationReason,
    assess_evidence_obligations,
    legacy_missing_evidence,
)

if TYPE_CHECKING:
    from claimci.analysis.discovery import DiscoveryResult


class PlanningState(str, Enum):
    READY = "ready"
    MAPPING_NEEDED = "mapping_needed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    repository: RepositoryIdentity
    pr_number: int | None
    head_sha: GitCommitSha
    claim: ClaimReference
    audit_claim: AuditClaimSpec | None
    artifacts: tuple[ArtifactCandidate, ...]
    normalized_evidence: tuple[NormalizedEvidence, ...]
    mapping_candidates: tuple[MappingCandidate, ...]
    approved_mapping: RepoMapping | None = None
    upstream_mapping_question: MappingQuestion | None = None
    semantic_proposal_provenance: FieldProvenance | None = None
    scientific_claim: CanonicalScientificClaim | None = None
    claim_policy: ClaimEvidencePolicy | None = None

    def __post_init__(self) -> None:
        if type(self.repository) is not RepositoryIdentity:
            raise TypeError("planning repository must be RepositoryIdentity")
        if self.pr_number is not None and (
            isinstance(self.pr_number, bool)
            or not isinstance(self.pr_number, int)
            or self.pr_number < 1
        ):
            raise AnalysisContractError(
                "planning pr_number must be a positive integer"
            )
        if not isinstance(self.head_sha, GitCommitSha):
            object.__setattr__(self, "head_sha", GitCommitSha(self.head_sha))
        if type(self.claim) is not ClaimReference:
            raise TypeError("planning claim must be ClaimReference")
        if self.audit_claim is not None and type(self.audit_claim) is not AuditClaimSpec:
            raise TypeError("planning audit_claim must be AuditClaimSpec or null")
        if (
            self.audit_claim is not None
            and self.claim.claim_id != self.audit_claim.claim_id
        ):
            raise AnalysisContractError("planning claim identifiers must match")
        if not isinstance(self.artifacts, tuple) or not all(
            type(item) is ArtifactCandidate for item in self.artifacts
        ):
            raise TypeError("planning artifacts must be ArtifactCandidate values")
        artifact_paths = tuple(item.path for item in self.artifacts)
        if len(set(artifact_paths)) != len(artifact_paths):
            raise AnalysisContractError("planning artifact paths must be unique")
        if any(
            self.claim.claim_id not in item.relevant_claim_ids
            for item in self.artifacts
        ):
            raise AnalysisContractError(
                "planning artifacts must be relevant to the selected claim"
            )
        if not isinstance(self.normalized_evidence, tuple) or not all(
            type(item) is NormalizedEvidence for item in self.normalized_evidence
        ):
            raise TypeError(
                "planning normalized_evidence must be NormalizedEvidence values"
            )
        issued = set(self.artifacts)
        if any(item.artifact not in issued for item in self.normalized_evidence):
            raise AnalysisContractError(
                "normalized evidence artifact was not issued by discovery"
            )
        evidence_ids = tuple(item.evidence_id for item in self.normalized_evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise AnalysisContractError(
                "planning normalized evidence identifiers must be unique"
            )
        if not isinstance(self.mapping_candidates, tuple) or not all(
            type(item) is MappingCandidate for item in self.mapping_candidates
        ):
            raise TypeError(
                "planning mapping_candidates must be MappingCandidate values"
            )
        mapping_ids = tuple(item.mapping_id for item in self.mapping_candidates)
        if len(set(mapping_ids)) != len(mapping_ids):
            raise AnalysisContractError(
                "planning mapping candidate identifiers must be unique"
            )
        if self.approved_mapping is not None:
            if type(self.approved_mapping) is not RepoMapping:
                raise TypeError("approved planning mapping must be RepoMapping")
            if self.approved_mapping.repository != self.repository:
                raise AnalysisContractError(
                    "approved mapping repository does not match planning repository"
                )
        if self.upstream_mapping_question is not None and type(
            self.upstream_mapping_question
        ) is not MappingQuestion:
            raise TypeError("upstream mapping question must be MappingQuestion")
        if self.semantic_proposal_provenance is not None:
            if type(self.semantic_proposal_provenance) is not FieldProvenance:
                raise TypeError(
                    "semantic proposal provenance must be FieldProvenance or null"
                )
            if (
                self.semantic_proposal_provenance.kind
                is not ProvenanceKind.PROVIDER_PROPOSAL
            ):
                raise AnalysisContractError(
                    "semantic proposal trace input must remain provider provenance"
                )
        if (self.scientific_claim is None) != (self.claim_policy is None):
            raise AnalysisContractError(
                "planning scientific claim and claim policy must appear together"
            )
        if self.scientific_claim is not None:
            if type(self.scientific_claim) is not CanonicalScientificClaim:
                raise TypeError(
                    "planning scientific_claim must be CanonicalScientificClaim or null"
                )
            if type(self.claim_policy) is not ClaimEvidencePolicy:
                raise TypeError(
                    "planning claim_policy must be ClaimEvidencePolicy or null"
                )
            if self.scientific_claim.reference != self.claim:
                raise AnalysisContractError(
                    "planning scientific claim reference must match its claim"
                )
            if self.claim_policy != claim_evidence_policy(self.scientific_claim):
                raise AnalysisContractError(
                    "planning claim policy must match canonical claim semantics"
                )
            if type(self.scientific_claim.primary) is MetricImprovementClaim:
                try:
                    compiled = compile_audit_claim(self.scientific_claim)
                except UnsupportedDeterministicClaimCompiler:
                    if self.audit_claim is not None:
                        raise AnalysisContractError(
                            "unsupported scientific claim cannot carry an audit claim"
                        )
                else:
                    if self.audit_claim != compiled:
                        raise AnalysisContractError(
                            "planning audit claim must match the deterministic compiler"
                        )
            elif self.audit_claim is not None:
                raise AnalysisContractError(
                    "unsupported scientific claim cannot carry an audit claim"
                )
        elif self.audit_claim is None:
            raise AnalysisContractError(
                "legacy planning request requires an audit claim"
            )


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    state: PlanningState
    plan: EphemeralAuditPlan | None = None
    mapping_question: MappingQuestion | None = None
    missing_evidence: tuple[MissingEvidence, ...] = ()
    reason: str | None = None
    evidence_obligations: EvidenceObligationBundle | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, PlanningState):
            raise TypeError("planning state must be PlanningState")
        if self.plan is not None and type(self.plan) is not EphemeralAuditPlan:
            raise TypeError("planning plan must be EphemeralAuditPlan or null")
        if self.mapping_question is not None and type(
            self.mapping_question
        ) is not MappingQuestion:
            raise TypeError("planning question must be MappingQuestion or null")
        if not isinstance(self.missing_evidence, tuple) or not all(
            type(item) is MissingEvidence for item in self.missing_evidence
        ):
            raise TypeError(
                "planning missing_evidence must be MissingEvidence values"
            )
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise AnalysisContractError("planning reason must be non-empty text")
        if self.evidence_obligations is not None and type(
            self.evidence_obligations
        ) is not EvidenceObligationBundle:
            raise TypeError(
                "planning evidence_obligations must be EvidenceObligationBundle"
            )

        if self.state is PlanningState.READY:
            if (
                self.plan is None
                or self.mapping_question is not None
                or self.missing_evidence
                or self.reason is not None
            ):
                raise AnalysisContractError(
                    "ready planning requires only one validated plan"
                )
            if self.evidence_obligations is not None and (
                self.evidence_obligations.decision
                is not EvidenceObligationDecision.READY
            ):
                raise AnalysisContractError(
                    "ready planning requires satisfied evidence obligations"
                )
            if (
                self.evidence_obligations is not None
                and self.plan is not None
                and self.plan.evidence_obligations != self.evidence_obligations
            ):
                raise AnalysisContractError(
                    "ready planning outcome and plan obligations must match"
                )
        elif self.state is PlanningState.MAPPING_NEEDED:
            if (
                self.mapping_question is None
                or self.plan is not None
                or self.missing_evidence
                or self.reason is not None
            ):
                raise AnalysisContractError(
                    "mapping-needed planning requires only one mapping question"
                )
            if self.evidence_obligations is not None and (
                self.evidence_obligations.decision
                is not EvidenceObligationDecision.MAPPING_NEEDED
            ):
                raise AnalysisContractError(
                    "mapping-needed planning requires ambiguous obligations"
                )
            if self.evidence_obligations is not None and (
                self.evidence_obligations.mapping_question_id
                != self.mapping_question.question_id
                or self.evidence_obligations.blocking_obligation_ids
                != self.mapping_question.blocking_obligation_ids
            ):
                raise AnalysisContractError(
                    "mapping question linkage does not match evidence obligations"
                )
        elif self.state is PlanningState.PARTIAL:
            if (
                self.plan is not None
                or self.mapping_question is not None
                or (not self.missing_evidence and self.reason is None)
            ):
                raise AnalysisContractError(
                    "partial planning requires missing evidence or a limitation"
                )
            if self.evidence_obligations is not None and (
                self.evidence_obligations.decision
                is not EvidenceObligationDecision.PARTIAL
            ):
                raise AnalysisContractError(
                    "partial planning requires blocking evidence obligations"
                )
        elif (
            self.reason is None
            or self.plan is not None
            or self.mapping_question is not None
            or self.missing_evidence
        ):
            raise AnalysisContractError(
                "unavailable planning requires only a failure reason"
            )


def _binding_key(binding: ArtifactBinding) -> tuple[RepositoryPath, ArtifactKind]:
    return binding.path, binding.kind


def _field_mapping_projection(
    mappings: tuple[FieldMapping, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                item.target_field,
                item.selector.kind.value,
                item.selector.expression,
            )
            for item in mappings
        )
    )


def _hydrate_binding(
    binding: ArtifactBinding,
    evidence_by_key: dict[
        tuple[RepositoryPath, ArtifactKind],
        NormalizedEvidence,
    ],
) -> ArtifactBinding:
    evidence = evidence_by_key.get(_binding_key(binding))
    if evidence is None:
        return binding
    runtime = evidence.adapter_match
    if binding.adapter_id is not None and binding.adapter_id != runtime.adapter_id:
        return binding
    if binding.mappings and _field_mapping_projection(
        binding.mappings
    ) != _field_mapping_projection(runtime.mappings):
        return binding
    return replace(
        binding,
        adapter_id=runtime.adapter_id,
        mappings=runtime.mappings,
    )


def _scoped_mapping_id(
    claim_id: str,
    candidate: MappingCandidate,
    bindings: tuple[ArtifactBinding, ...],
) -> str:
    material = {
        "claim_id": claim_id,
        "trust": candidate.trust.value,
        "provenance_kind": candidate.provenance.kind.value,
        "bindings": sorted(_binding_signature(item) for item in bindings),
    }
    canonical = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "mapping-scoped-" + hashlib.sha256(canonical).hexdigest()[:16]


def _scoped_candidate_confidence(
    candidate: MappingCandidate,
) -> Confidence:
    # Claim scoping narrows an already-issued mapping; it does not create a new
    # inference or justify changing the issuer's confidence assessment.
    return candidate.confidence


def _scope_mapping_candidates(
    candidates: tuple[MappingCandidate, ...],
    *,
    claim_id: str,
    selected_artifacts: tuple[ArtifactCandidate, ...],
    selected_evidence: tuple[NormalizedEvidence, ...],
) -> tuple[MappingCandidate, ...]:
    selected_keys = {
        (item.path, item.kind) for item in selected_artifacts
    }
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item for item in selected_evidence
    }
    scoped: list[MappingCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        if type(candidate) is not MappingCandidate:
            raise TypeError("discovery mapping candidates must be MappingCandidate values")
        bindings = tuple(
            _hydrate_binding(binding, evidence_by_key)
            for binding in candidate.bindings
            if _binding_key(binding) in selected_keys
        )
        if not bindings:
            continue
        signature = (
            candidate.trust,
            candidate.provenance.kind,
            tuple(sorted(_binding_signature(item) for item in bindings)),
        )
        if signature in seen:
            continue
        seen.add(signature)
        scoped.append(
            MappingCandidate(
                mapping_id=_scoped_mapping_id(claim_id, candidate, bindings),
                bindings=bindings,
                confidence=_scoped_candidate_confidence(candidate),
                trust=candidate.trust,
                provenance=candidate.provenance,
            )
        )
    return tuple(scoped)


def _scope_approved_mapping(
    approved: RepoMapping | None,
    *,
    all_artifacts: tuple[ArtifactCandidate, ...],
    selected_artifacts: tuple[ArtifactCandidate, ...],
    selected_evidence: tuple[NormalizedEvidence, ...],
) -> RepoMapping | None:
    if approved is None:
        return None
    if type(approved) is not RepoMapping:
        raise TypeError("discovery approved mapping must be RepoMapping")
    all_keys = {(item.path, item.kind) for item in all_artifacts}
    selected_keys = {(item.path, item.kind) for item in selected_artifacts}
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item for item in selected_evidence
    }
    selected_bindings: list[ArtifactBinding] = []
    stale = False
    for binding in approved.bindings:
        key = _binding_key(binding)
        if key in selected_keys:
            selected_bindings.append(_hydrate_binding(binding, evidence_by_key))
        elif key not in all_keys:
            stale = True
    if stale:
        return approved
    if not selected_bindings:
        return None
    try:
        return approved.scope_to_runtime_bindings(tuple(selected_bindings))
    except (AnalysisContractError, TypeError):
        # Preserve the invalid approval so planning cannot silently fall back
        # to a lower-trust automatic proposal.
        return approved


def _scoped_choice_id(
    claim_id: str,
    bindings: tuple[ArtifactBinding, ...],
) -> str:
    canonical = json.dumps(
        {
            "claim_id": claim_id,
            "bindings": sorted(_binding_signature(item) for item in bindings),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "choice-scoped-" + hashlib.sha256(canonical).hexdigest()[:16]


def _scope_mapping_question(
    question: MappingQuestion | None,
    *,
    claim_id: str,
    selected_artifacts: tuple[ArtifactCandidate, ...],
    selected_evidence: tuple[NormalizedEvidence, ...],
) -> MappingQuestion | None:
    if question is None:
        return None
    if type(question) is not MappingQuestion:
        raise TypeError("discovery mapping question must be MappingQuestion")
    if question.relevant_claim_id not in {None, claim_id}:
        return None
    selected_keys = {(item.path, item.kind) for item in selected_artifacts}
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item for item in selected_evidence
    }
    choices: list[MappingChoice] = []
    seen: set[tuple[tuple[object, ...], ...]] = set()
    for choice in question.choices:
        bindings = tuple(
            _hydrate_binding(binding, evidence_by_key)
            for binding in choice.bindings
            if _binding_key(binding) in selected_keys
        )
        if not bindings:
            continue
        signature = tuple(sorted(_binding_signature(item) for item in bindings))
        if signature in seen:
            continue
        seen.add(signature)
        choices.append(
            MappingChoice(
                choice_id=_scoped_choice_id(claim_id, bindings),
                label=choice.label,
                bindings=bindings,
            )
        )
    if not 2 <= len(choices) <= 8:
        return None
    question_material = "|".join(item.choice_id for item in choices)
    return MappingQuestion(
        question_id="question-scoped-"
        + hashlib.sha256(question_material.encode("utf-8")).hexdigest()[:16],
        prompt=question.prompt,
        choices=tuple(choices),
        relevant_claim_id=claim_id,
    )


def planning_request_from_discovery(
    discovery: "DiscoveryResult",
    *,
    claim_id: str,
    normalized_evidence: tuple[NormalizedEvidence, ...],
) -> PlanningRequest:
    """Convert one issued discovery claim without elevating mapping trust."""

    from .discovery.models import DiscoveredClaim, DiscoveryResult

    if type(discovery) is not DiscoveryResult:
        raise TypeError("planning conversion requires concrete DiscoveryResult")
    repository = discovery.repository
    head_sha = discovery.head_sha
    pr_number = discovery.pr_number
    claims = discovery.claims
    artifacts = discovery.artifacts
    repository_paths = discovery.repository_paths
    selected = tuple(
        item
        for item in claims
        if type(item) is DiscoveredClaim and item.reference.claim_id == claim_id
    )
    if len(selected) != 1:
        raise AnalysisContractError(
            "discovery must issue exactly one selected claim identifier"
        )
    discovered_claim = selected[0]
    reference = discovered_claim.reference
    if type(reference) is not ClaimReference:
        raise TypeError("discovered claim reference must be ClaimReference")
    source = discovered_claim.source
    if type(source) is not SourceLocation:
        raise TypeError("discovered claim source must be SourceLocation")
    if source.kind is SourceKind.REPOSITORY_FILE:
        if source.path is None or reference.source_path != RepositoryPath(source.path):
            raise AnalysisContractError(
                "discovered claim source path does not match its reference"
            )
        if reference.source_path not in set(repository_paths):
            raise AnalysisContractError(
                "discovered claim source must be an issued repository path"
            )
    elif reference.source_path is not None:
        raise AnalysisContractError(
            "pull-request discovered claim cannot carry a repository path"
        )
    evidence_hints = discovered_claim.evidence_hints
    if not isinstance(evidence_hints, tuple) or not all(
        isinstance(item, RepositoryPath) for item in evidence_hints
    ):
        raise TypeError("discovered claim evidence_hints must be RepositoryPath values")
    if any(item not in set(repository_paths) for item in evidence_hints):
        raise AnalysisContractError(
            "discovered claim evidence hints must be issued repository paths"
        )
    scientific_claim = discovered_claim.scientific_claim
    if type(scientific_claim) is not CanonicalScientificClaim:
        raise AnalysisContractError(
            "selected claim has no canonical scientific claim representation"
        )
    policy = claim_evidence_policy(scientific_claim)
    try:
        audit_claim: AuditClaimSpec | None = compile_audit_claim(scientific_claim)
    except UnsupportedDeterministicClaimCompiler:
        audit_claim = None
    if not isinstance(artifacts, tuple) or not all(
        type(item) is ArtifactCandidate for item in artifacts
    ):
        raise TypeError("discovery artifacts must be ArtifactCandidate values")
    if any(item.path not in set(repository_paths) for item in artifacts):
        raise AnalysisContractError(
            "discovery artifacts must use issued repository paths"
        )
    if not isinstance(normalized_evidence, tuple) or not all(
        type(item) is NormalizedEvidence for item in normalized_evidence
    ):
        raise TypeError("normalized_evidence must contain NormalizedEvidence values")
    issued_artifacts = set(artifacts)
    if any(item.artifact not in issued_artifacts for item in normalized_evidence):
        raise AnalysisContractError(
            "normalized evidence artifact was not issued by discovery"
        )
    selected_artifacts = tuple(
        item for item in artifacts if reference.claim_id in item.relevant_claim_ids
    )
    selected_artifact_set = set(selected_artifacts)
    selected_evidence = tuple(
        item for item in normalized_evidence if item.artifact in selected_artifact_set
    )
    scoped_candidates = _scope_mapping_candidates(
        discovery.mapping_candidates,
        claim_id=reference.claim_id,
        selected_artifacts=selected_artifacts,
        selected_evidence=selected_evidence,
    )
    scoped_approved = _scope_approved_mapping(
        discovery.approved_mapping,
        all_artifacts=artifacts,
        selected_artifacts=selected_artifacts,
        selected_evidence=selected_evidence,
    )
    scoped_question = _scope_mapping_question(
        discovery.mapping_question,
        claim_id=reference.claim_id,
        selected_artifacts=selected_artifacts,
        selected_evidence=selected_evidence,
    )
    return PlanningRequest(
        repository=repository,
        pr_number=pr_number,
        head_sha=head_sha,
        claim=reference,
        audit_claim=audit_claim,
        artifacts=selected_artifacts,
        normalized_evidence=selected_evidence,
        mapping_candidates=scoped_candidates,
        approved_mapping=scoped_approved,
        upstream_mapping_question=scoped_question,
        scientific_claim=scientific_claim,
        claim_policy=policy,
    )


def plan_ephemeral_audit(request: PlanningRequest) -> PlanningOutcome:
    """Select one trusted mapping and produce a head-bound ephemeral plan."""

    if type(request) is not PlanningRequest:
        raise TypeError("ephemeral planning requires PlanningRequest")

    if request.scientific_claim is None or request.claim_policy is None:
        return _plan_legacy_request(request)

    if request.approved_mapping is not None and any(
        (binding.path, binding.kind)
        not in {(item.path, item.kind) for item in request.artifacts}
        for binding in request.approved_mapping.bindings
    ):
        return PlanningOutcome(
            state=PlanningState.UNAVAILABLE,
            reason="approved mapping is stale for the exact-head snapshot",
        )

    if request.audit_claim is None:
        obligations = assess_evidence_obligations(
            claim=request.scientific_claim,
            policy=request.claim_policy,
            audit_claim=None,
            artifacts=request.artifacts,
            normalized_evidence=request.normalized_evidence,
            mapping_candidates=request.mapping_candidates,
            selected_mapping=None,
            mapping_question=None,
            ambiguity_reason=None,
        )
        return PlanningOutcome(
            state=PlanningState.PARTIAL,
            reason=UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER,
            evidence_obligations=obligations,
        )

    selected = _select_mapping(request)
    valid_candidates = tuple(
        item
        for item in request.mapping_candidates
        if _mapping_is_runtime_revalidatable(item, request)
    )
    question: MappingQuestion | None = None
    ambiguity_reason: EvidenceObligationReason | None = None
    if selected is None:
        if request.upstream_mapping_question is not None and _question_is_useful(
            request.upstream_mapping_question,
            request,
        ):
            question = request.upstream_mapping_question
            ambiguity_reason = EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES
        else:
            candidate_question = _mapping_question(
                valid_candidates,
                request.claim.claim_id,
            )
            if candidate_question is not None and _question_is_useful(
                candidate_question,
                request,
            ):
                question = candidate_question
            if request.approved_mapping is not None:
                ambiguity_reason = (
                    EvidenceObligationReason.APPROVED_MAPPING_NOT_APPLICABLE
                    if question is not None
                    else None
                )
            elif len({_mapping_signature(item) for item in valid_candidates}) > 1:
                ambiguity_reason = (
                    EvidenceObligationReason.CONFLICTING_VALIDATED_MAPPINGS
                    if question is not None
                    else EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED
                )
            elif valid_candidates:
                ambiguity_reason = (
                    EvidenceObligationReason.EXPLICIT_MAPPING_APPROVAL_REQUIRED
                    if question is not None
                    else EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED
                )

    obligations = assess_evidence_obligations(
        claim=request.scientific_claim,
        policy=request.claim_policy,
        audit_claim=request.audit_claim,
        artifacts=request.artifacts,
        normalized_evidence=request.normalized_evidence,
        mapping_candidates=request.mapping_candidates,
        selected_mapping=selected,
        mapping_question=question,
        ambiguity_reason=ambiguity_reason,
    )
    missing = legacy_missing_evidence(obligations)
    if obligations.decision is EvidenceObligationDecision.PARTIAL:
        reason = None
        if not missing:
            blocker_by_id = {
                item.obligation_id: item
                for item in obligations.obligations
                if item.target is not None
            }
            reason = next(
                (
                    blocker_by_id[item].reason.value
                    for item in obligations.blocking_obligation_ids
                    if item in blocker_by_id
                ),
                "no bounded mapping clarification can produce a viable plan",
            )
        return PlanningOutcome(
            state=PlanningState.PARTIAL,
            missing_evidence=missing,
            reason=reason,
            evidence_obligations=obligations,
        )
    if obligations.decision is EvidenceObligationDecision.MAPPING_NEEDED:
        if question is None:
            return PlanningOutcome(
                state=PlanningState.UNAVAILABLE,
                reason="obligation mapping linkage is unavailable",
            )
        linked_question = replace(
            question,
            blocking_obligation_ids=obligations.blocking_obligation_ids,
        )
        return PlanningOutcome(
            state=PlanningState.MAPPING_NEEDED,
            mapping_question=linked_question,
            evidence_obligations=obligations,
        )
    if selected is None:
        return PlanningOutcome(
            state=PlanningState.UNAVAILABLE,
            reason="satisfied obligations have no selected mapping",
        )

    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item
        for item in request.normalized_evidence
    }
    baseline: list[NormalizedEvidence] = []
    candidate: list[NormalizedEvidence] = []
    for binding in selected.bindings:
        evidence = evidence_by_key[(binding.path, binding.kind)]
        target = baseline if binding.role is ExperimentRole.BASELINE else candidate
        if evidence not in target:
            target.append(evidence)
    baseline.sort(key=lambda item: item.evidence_id)
    candidate.sort(key=lambda item: item.evidence_id)
    mapping_provenance = _mapping_provenance(selected)
    confidence = _plan_confidence(request, selected, (*baseline, *candidate))
    plan = EphemeralAuditPlan(
        plan_id=_plan_id(request, selected, (*baseline, *candidate)),
        repository=request.repository,
        pr_number=request.pr_number,
        head_sha=request.head_sha,
        claim=request.claim,
        baseline_evidence=tuple(baseline),
        candidate_evidence=tuple(candidate),
        mapping_provenance=mapping_provenance,
        missing_evidence=(),
        confidence=confidence,
        audit_claim=request.audit_claim,
        selected_mapping=selected,
        semantic_proposal_provenance=request.semantic_proposal_provenance,
        scientific_claim=request.scientific_claim,
        claim_policy=request.claim_policy,
        evidence_obligations=obligations,
    )
    return PlanningOutcome(
        state=PlanningState.READY,
        plan=plan,
        evidence_obligations=obligations,
    )


def _plan_legacy_request(request: PlanningRequest) -> PlanningOutcome:
    """Preserve the pre-claim-taxonomy construction seam for existing callers."""

    if request.audit_claim is None:
        return PlanningOutcome(
            state=PlanningState.PARTIAL,
            reason=UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER,
        )
    missing = _planning_missing_evidence(request)
    if missing:
        return PlanningOutcome(
            state=PlanningState.PARTIAL,
            missing_evidence=missing,
        )
    selected = _select_mapping(request)
    if selected is None:
        question = request.upstream_mapping_question or _mapping_question(
            request.mapping_candidates,
            request.claim.claim_id,
        )
        if question is None:
            return PlanningOutcome(
                state=PlanningState.PARTIAL,
                reason="no bounded mapping clarification can produce a viable plan",
            )
        return PlanningOutcome(
            state=PlanningState.MAPPING_NEEDED,
            mapping_question=question,
        )
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item
        for item in request.normalized_evidence
    }
    baseline: list[NormalizedEvidence] = []
    candidate: list[NormalizedEvidence] = []
    for binding in selected.bindings:
        evidence = evidence_by_key[(binding.path, binding.kind)]
        target = baseline if binding.role is ExperimentRole.BASELINE else candidate
        if evidence not in target:
            target.append(evidence)
    baseline.sort(key=lambda item: item.evidence_id)
    candidate.sort(key=lambda item: item.evidence_id)
    plan = EphemeralAuditPlan(
        plan_id=_plan_id(request, selected, (*baseline, *candidate)),
        repository=request.repository,
        pr_number=request.pr_number,
        head_sha=request.head_sha,
        claim=request.claim,
        baseline_evidence=tuple(baseline),
        candidate_evidence=tuple(candidate),
        mapping_provenance=_mapping_provenance(selected),
        missing_evidence=(),
        confidence=_plan_confidence(request, selected, (*baseline, *candidate)),
        audit_claim=request.audit_claim,
        selected_mapping=selected,
        semantic_proposal_provenance=request.semantic_proposal_provenance,
    )
    return PlanningOutcome(state=PlanningState.READY, plan=plan)


def _binding_supports_native_slot(
    binding: ArtifactBinding,
    request: PlanningRequest,
) -> bool:
    evidence = next(
        (
            item
            for item in request.normalized_evidence
            if item.artifact.path == binding.path
            and item.artifact.kind is binding.kind
        ),
        None,
    )
    if evidence is None or not _evidence_supports_role(evidence, binding.role):
        return False
    if (
        evidence.adapter_match.adapter_id != binding.adapter_id
        or evidence.adapter_match.mappings != binding.mappings
    ):
        return False
    if binding.kind is ArtifactKind.RESULTS:
        return any(
            item.metric_name == request.audit_claim.metric
            for item in evidence.observations
        )
    if binding.kind is ArtifactKind.CONFIG:
        return any(item.config_values for item in evidence.observations)
    return any(
        reference.path == binding.path
        for observation in evidence.observations
        for reference in observation.dataset_references
    )


def _bindings_can_produce_plan(
    bindings: tuple[ArtifactBinding, ...],
    request: PlanningRequest,
) -> bool:
    if request.audit_claim is None:
        return False
    for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
        for kind in (ArtifactKind.RESULTS, ArtifactKind.CONFIG):
            matching = tuple(
                item
                for item in bindings
                if item.role is role and item.kind is kind
            )
            if not matching or not all(
                _binding_supports_native_slot(item, request) for item in matching
            ):
                return False
        for split in (DatasetSplit.TRAIN, DatasetSplit.EVAL):
            matching = tuple(
                item
                for item in bindings
                if item.role is role
                and item.kind is ArtifactKind.DATASET
                and item.dataset_split is split
            )
            if len(matching) != 1 or not _binding_supports_native_slot(
                matching[0], request
            ):
                return False
    return True


def _question_is_useful(
    question: MappingQuestion,
    request: PlanningRequest,
) -> bool:
    bases = tuple(item.bindings for item in request.mapping_candidates) or ((),)
    for choice in question.choices:
        for base in bases:
            by_slot: dict[tuple[ArtifactKind, ExperimentRole, DatasetSplit | None], ArtifactBinding] = {
                (item.kind, item.role, item.dataset_split): item for item in base
            }
            for binding in choice.bindings:
                by_slot[(binding.kind, binding.role, binding.dataset_split)] = binding
            if _bindings_can_produce_plan(tuple(by_slot.values()), request):
                return True
    return False


def _planning_missing_evidence(
    request: PlanningRequest,
) -> tuple[MissingEvidence, ...]:
    missing: list[MissingEvidence] = []
    if request.audit_claim.minimum_absolute_improvement is None:
        missing.append(
            MissingEvidence(
                kind=ArtifactKind.MANIFEST,
                role=ExperimentRole.UNSPECIFIED,
                description=(
                    "an explicit absolute improvement threshold is required by "
                    "the current deterministic engine"
                ),
                claim_id=request.claim.claim_id,
            )
        )

    dataset_profiles = _potential_dataset_profiles(request)
    role_has_dataset_slots = {
        role: any(
            _profile_has_dataset_slots(profile, role)
            for profile in dataset_profiles
        )
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
    }
    has_coherent_dataset_mapping = any(
        all(
            _profile_has_dataset_slots(profile, role)
            for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
        )
        for profile in dataset_profiles
    )

    for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
        role_evidence = tuple(
            item
            for item in request.normalized_evidence
            if _evidence_supports_role(item, role)
        )
        for kind in (ArtifactKind.RESULTS, ArtifactKind.CONFIG):
            if not any(item.artifact.kind is kind for item in role_evidence):
                missing.append(
                    MissingEvidence(
                        kind=kind,
                        role=role,
                        description=f"{role.value} {kind.value} evidence is missing",
                        claim_id=request.claim.claim_id,
                    )
                )
        if not role_has_dataset_slots[role]:
            missing.append(
                MissingEvidence(
                    kind=ArtifactKind.DATASET,
                    role=role,
                    description=(
                        f"{role.value} train and evaluation dataset evidence "
                        "is required"
                    ),
                    claim_id=request.claim.claim_id,
                )
            )
    if (
        not has_coherent_dataset_mapping
        and all(role_has_dataset_slots.values())
    ):
        missing.append(
            MissingEvidence(
                kind=ArtifactKind.DATASET,
                role=ExperimentRole.UNSPECIFIED,
                description=(
                    "no single bounded mapping supplies train and evaluation "
                    "dataset evidence for both experiment roles"
                ),
                claim_id=request.claim.claim_id,
            )
        )
    return tuple(missing)


def _potential_dataset_profiles(
    request: PlanningRequest,
) -> tuple[dict[tuple[ExperimentRole, DatasetSplit], int], ...]:
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item
        for item in request.normalized_evidence
    }
    bases: list[tuple[ArtifactBinding, ...]] = [
        mapping.bindings for mapping in request.mapping_candidates
    ]
    if request.approved_mapping is not None:
        bases.append(request.approved_mapping.bindings)
    if not bases:
        bases.append(())
    additions = (
        tuple(choice.bindings for choice in request.upstream_mapping_question.choices)
        if request.upstream_mapping_question is not None
        else ((),)
    )
    profiles: list[dict[tuple[ExperimentRole, DatasetSplit], int]] = []
    for base in bases:
        for addition in additions:
            counts: dict[tuple[ExperimentRole, DatasetSplit], int] = {}
            for binding in (*base, *addition):
                evidence = evidence_by_key.get((binding.path, binding.kind))
                if (
                    binding.kind is not ArtifactKind.DATASET
                    or binding.role not in {
                        ExperimentRole.BASELINE,
                        ExperimentRole.CANDIDATE,
                    }
                    or binding.dataset_split not in {
                        DatasetSplit.TRAIN,
                        DatasetSplit.EVAL,
                    }
                    or evidence is None
                    or not _evidence_supports_role(evidence, binding.role)
                ):
                    continue
                key = (binding.role, binding.dataset_split)
                counts[key] = counts.get(key, 0) + 1
            profiles.append(counts)
    return tuple(profiles)


def _profile_has_dataset_slots(
    profile: dict[tuple[ExperimentRole, DatasetSplit], int],
    role: ExperimentRole,
) -> bool:
    return all(
        profile.get((role, split), 0) == 1
        for split in (DatasetSplit.TRAIN, DatasetSplit.EVAL)
    )


def _evidence_supports_role(
    evidence: NormalizedEvidence,
    role: ExperimentRole,
) -> bool:
    roles = {
        item.experiment_role
        for item in evidence.observations
        if item.experiment_role is not ExperimentRole.UNSPECIFIED
    }
    return not roles or roles == {role}


def _binding_signature(binding: ArtifactBinding) -> tuple[object, ...]:
    return (
        str(binding.path),
        binding.kind.value,
        binding.role.value,
        binding.adapter_id,
        binding.dataset_split.value if binding.dataset_split is not None else None,
        tuple(
            sorted(
                (
                    item.target_field,
                    item.selector.kind.value,
                    item.selector.expression,
                )
                for item in binding.mappings
            )
        ),
    )


def _mapping_signature(
    mapping: MappingCandidate | RepoMapping,
) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted(_binding_signature(item) for item in mapping.bindings))


def _mapping_is_runtime_revalidatable(
    mapping: MappingCandidate | RepoMapping,
    request: PlanningRequest,
) -> bool:
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item
        for item in request.normalized_evidence
    }
    if not mapping.bindings:
        return False
    for binding in mapping.bindings:
        if binding.role not in {
            ExperimentRole.BASELINE,
            ExperimentRole.CANDIDATE,
        }:
            return False
        evidence = evidence_by_key.get((binding.path, binding.kind))
        if evidence is None:
            return False
        if request.claim.claim_id not in evidence.artifact.relevant_claim_ids:
            return False
        if binding.adapter_id != evidence.adapter_match.adapter_id:
            return False
        if binding.mappings != evidence.adapter_match.mappings:
            return False
        if not _evidence_supports_role(evidence, binding.role):
            return False
        if binding.kind is ArtifactKind.DATASET and any(
            reference.path != evidence.artifact.path
            for observation in evidence.observations
            for reference in observation.dataset_references
        ):
            return False
    return _mapping_covers_native_inputs(mapping, request)


def _mapping_covers_native_inputs(
    mapping: MappingCandidate | RepoMapping,
    request: PlanningRequest,
) -> bool:
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item
        for item in request.normalized_evidence
    }
    for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
        role_bindings = tuple(item for item in mapping.bindings if item.role is role)
        if not all(
            any(item.kind is kind for item in role_bindings)
            for kind in (ArtifactKind.RESULTS, ArtifactKind.CONFIG)
        ):
            return False
        dataset_bindings = tuple(
            item for item in role_bindings if item.kind is ArtifactKind.DATASET
        )
        split_counts = {
            split: sum(item.dataset_split is split for item in dataset_bindings)
            for split in (DatasetSplit.TRAIN, DatasetSplit.EVAL)
        }
        if split_counts != {DatasetSplit.TRAIN: 1, DatasetSplit.EVAL: 1}:
            return False
    return True


def _select_mapping(
    request: PlanningRequest,
) -> MappingCandidate | RepoMapping | None:
    valid_candidates = tuple(
        item
        for item in request.mapping_candidates
        if _mapping_is_runtime_revalidatable(item, request)
    )
    approved = request.approved_mapping
    if approved is not None:
        if _mapping_is_runtime_revalidatable(approved, request):
            return approved
        return None

    signatures = {_mapping_signature(item) for item in valid_candidates}
    if len(signatures) != 1:
        return None
    eligible = tuple(
        item
        for item in valid_candidates
        if item.confidence.value >= 0.90
        and item.provenance.kind is not ProvenanceKind.PROVIDER_PROPOSAL
    )
    if not eligible:
        return None
    provenance_priority = {
        ProvenanceKind.DETERMINISTIC_DISCOVERY: 0,
        ProvenanceKind.MANIFEST_HINT: 1,
    }
    return min(
        eligible,
        key=lambda item: (
            provenance_priority.get(item.provenance.kind, 2),
            -item.confidence.value,
            item.mapping_id,
        ),
    )


def _mapping_question(
    candidates: tuple[MappingCandidate, ...],
    claim_id: str,
) -> MappingQuestion | None:
    if not candidates or len(candidates) > 8:
        return None
    ordered = tuple(sorted(candidates, key=lambda item: item.mapping_id))
    choices = tuple(
        MappingChoice(
            choice_id=f"mapping-{index + 1}",
            label=f"Use mapping {item.mapping_id}",
            bindings=item.bindings,
        )
        for index, item in enumerate(ordered)
    )
    if len(choices) == 1:
        candidate = ordered[0]
        choices = (
            MappingChoice(
                choice_id="approve-mapping-1",
                label=f"Explicitly approve mapping {candidate.mapping_id}",
                bindings=candidate.bindings,
            ),
            MappingChoice(
                choice_id="reject-mapping-1",
                label=f"Do not use mapping {candidate.mapping_id}",
                bindings=candidate.bindings,
            ),
        )
    digest = hashlib.sha256(
        "|".join(item.mapping_id for item in ordered).encode("utf-8")
    ).hexdigest()[:16]
    return MappingQuestion(
        question_id=f"mapping-{digest}",
        prompt="Which validated repository mapping should ClaimCI use?",
        choices=choices,
        relevant_claim_id=claim_id,
    )


def _mapping_provenance(
    mapping: MappingCandidate | RepoMapping,
) -> tuple[FieldProvenance, ...]:
    if type(mapping) is RepoMapping:
        return (mapping.approval_provenance,)
    return (mapping.provenance,)


def _plan_confidence(
    request: PlanningRequest,
    mapping: MappingCandidate | RepoMapping,
    evidence: tuple[NormalizedEvidence, ...],
) -> Confidence:
    values = [request.claim.confidence.value]
    if type(mapping) is MappingCandidate:
        values.append(mapping.confidence.value)
    values.extend(item.artifact.confidence.value for item in evidence)
    values.extend(item.adapter_match.confidence.value for item in evidence)
    return Confidence(min(values))


def _selected_mapping_identity(
    mapping: MappingCandidate | RepoMapping,
) -> dict[str, object]:
    if type(mapping) is RepoMapping:
        return {
            "mapping_id": mapping.source_mapping_id,
            "trust": mapping.trust.value,
            "source_trust": mapping.source_trust.value,
            "approval_source_id": mapping.approval_provenance.source_id,
        }
    return {
        "mapping_id": mapping.mapping_id,
        "trust": mapping.trust.value,
        "source_provenance_kind": mapping.provenance.kind.value,
        "source_id": mapping.provenance.source_id,
    }


def _plan_id(
    request: PlanningRequest,
    mapping: MappingCandidate | RepoMapping,
    evidence: tuple[NormalizedEvidence, ...],
) -> str:
    if request.audit_claim is None:
        raise AnalysisContractError(
            "plan identity requires a compiled deterministic claim"
        )
    return _plan_id_from_components(
        repository=request.repository,
        pr_number=request.pr_number,
        head_sha=request.head_sha,
        audit_claim=request.audit_claim,
        mapping=mapping,
        evidence=evidence,
    )


def _plan_id_from_components(
    *,
    repository: RepositoryIdentity,
    pr_number: int | None,
    head_sha: GitCommitSha,
    audit_claim: AuditClaimSpec,
    mapping: MappingCandidate | RepoMapping,
    evidence: tuple[NormalizedEvidence, ...],
) -> str:
    artifacts = {
        (item.artifact.path, item.artifact.kind): item.artifact
        for item in evidence
    }
    bindings: list[dict[str, object]] = []
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
            "owner": repository.owner,
            "name": repository.name,
        },
        "pr_number": pr_number,
        "head_sha": str(head_sha),
        "claim": audit_relevant_claim_projection(audit_claim),
        "mapping": _selected_mapping_identity(mapping),
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


def derive_ephemeral_plan_id(plan: EphemeralAuditPlan) -> str:
    """Recompute one plan's canonical audit-relevant identity."""

    if type(plan) is not EphemeralAuditPlan:
        raise TypeError("plan identity derivation requires EphemeralAuditPlan")
    if plan.audit_claim is None or plan.selected_mapping is None:
        raise AnalysisContractError(
            "plan identity derivation requires executable claim and mapping"
        )
    return _plan_id_from_components(
        repository=plan.repository,
        pr_number=plan.pr_number,
        head_sha=plan.head_sha,
        audit_claim=plan.audit_claim,
        mapping=plan.selected_mapping,
        evidence=(*plan.baseline_evidence, *plan.candidate_evidence),
    )


__all__ = [
    "derive_ephemeral_plan_id",
    "PlanningOutcome",
    "PlanningRequest",
    "PlanningState",
    "plan_ephemeral_audit",
    "planning_request_from_discovery",
]
