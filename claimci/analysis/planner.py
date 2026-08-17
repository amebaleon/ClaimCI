"""Pure trust selection and ephemeral audit planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING

from claimci.models import Direction
from claimci.review.models import ClaimDirection, ClaimType

from .confidence import Confidence
from .contracts import (
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    AuditClaimSpec,
    ClaimReference,
    ClaimedMetricValue,
    EphemeralAuditPlan,
    ExperimentRole,
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
    ProvenanceKind,
)
from .claims import audit_relevant_claim_projection

if TYPE_CHECKING:
    from claimci.analysis.discovery import DiscoveryResult


_BOUNDED_THRESHOLD = re.compile(
    r"(?:\bat\s+least\b|\bminimum(?:\s+of)?\b|>=|\bno\s+less\s+than\b)"
    r"\s*(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))",
    flags=re.IGNORECASE,
)


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
    audit_claim: AuditClaimSpec
    artifacts: tuple[ArtifactCandidate, ...]
    normalized_evidence: tuple[NormalizedEvidence, ...]
    mapping_candidates: tuple[MappingCandidate, ...]
    approved_mapping: RepoMapping | None = None
    upstream_mapping_question: MappingQuestion | None = None

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
        if type(self.audit_claim) is not AuditClaimSpec:
            raise TypeError("planning audit_claim must be AuditClaimSpec")
        if self.claim.claim_id != self.audit_claim.claim_id:
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


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    state: PlanningState
    plan: EphemeralAuditPlan | None = None
    mapping_question: MappingQuestion | None = None
    missing_evidence: tuple[MissingEvidence, ...] = ()
    reason: str | None = None

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
        elif self.state is PlanningState.PARTIAL:
            if (
                self.plan is not None
                or self.mapping_question is not None
                or (not self.missing_evidence and self.reason is None)
            ):
                raise AnalysisContractError(
                    "partial planning requires missing evidence or a limitation"
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


def _discovery_attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise TypeError(f"discovery result is missing {name}") from error


def _claimed_value(
    value: object,
    role: ExperimentRole,
) -> ClaimedMetricValue:
    number = _discovery_attribute(value, "value")
    unit = _discovery_attribute(value, "unit")
    provenance = _discovery_attribute(value, "provenance")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise TypeError("discovery claimed value must be numeric")
    if not math.isfinite(float(number)):
        raise AnalysisContractError("discovery claimed value must be finite")
    if unit is not None and not isinstance(unit, str):
        raise TypeError("discovery claimed value unit must be text or null")
    if not isinstance(provenance, FieldProvenance):
        raise TypeError("discovery claimed value provenance is invalid")
    return ClaimedMetricValue(
        role=role,
        value=float(number),
        unit=unit,
        provenance=provenance,
    )


def _validated_explicit_threshold(
    source_text: str,
    minimum: object | None,
) -> tuple[float | None, FieldProvenance | None]:
    if minimum is None:
        return None, None
    match = _BOUNDED_THRESHOLD.search(source_text)
    if match is None:
        return None, None
    raw_value = _discovery_attribute(minimum, "value")
    unit = _discovery_attribute(minimum, "unit")
    provenance = _discovery_attribute(minimum, "provenance")
    if unit is not None:
        # Auto Discovery does not yet normalize percentage-point units into
        # native metric units, so conversion would be lossy.
        return None, None
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise TypeError("discovery minimum improvement must be numeric")
    try:
        source_value = Decimal(match.group("value"))
        discovered_value = Decimal(str(float(raw_value)))
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise AnalysisContractError("discovery threshold is invalid") from error
    if not source_value.is_finite() or source_value != discovered_value:
        raise AnalysisContractError(
            "discovery threshold does not match the bounded source expression"
        )
    if source_value < 0:
        raise AnalysisContractError("discovery threshold must be non-negative")
    if not isinstance(provenance, FieldProvenance):
        raise TypeError("discovery threshold provenance is invalid")
    return float(source_value), provenance


def planning_request_from_discovery(
    discovery: "DiscoveryResult",
    *,
    claim_id: str,
    normalized_evidence: tuple[NormalizedEvidence, ...],
) -> PlanningRequest:
    """Convert one issued discovery claim without elevating mapping trust."""

    repository = _discovery_attribute(discovery, "repository")
    head_sha = _discovery_attribute(discovery, "head_sha")
    pr_number = _discovery_attribute(discovery, "pr_number")
    claims = _discovery_attribute(discovery, "claims")
    artifacts = _discovery_attribute(discovery, "artifacts")
    mapping_candidates = _discovery_attribute(discovery, "mapping_candidates")
    approved_mapping = _discovery_attribute(discovery, "approved_mapping")
    mapping_question = _discovery_attribute(discovery, "mapping_question")
    if type(repository) is not RepositoryIdentity:
        raise TypeError("discovery repository must be RepositoryIdentity")
    if not isinstance(head_sha, GitCommitSha):
        head_sha = GitCommitSha(head_sha)
    if not isinstance(claims, tuple):
        raise TypeError("discovery claims must be a tuple")
    selected = tuple(
        item
        for item in claims
        if isinstance(getattr(item, "reference", None), ClaimReference)
        and item.reference.claim_id == claim_id
    )
    if len(selected) != 1:
        raise AnalysisContractError(
            "discovery must issue exactly one selected claim identifier"
        )
    discovered_claim = selected[0]
    reference = _discovery_attribute(discovered_claim, "reference")
    if type(reference) is not ClaimReference:
        raise TypeError("discovered claim reference must be ClaimReference")
    if _discovery_attribute(discovered_claim, "claim_type") is not ClaimType.METRIC_IMPROVEMENT:
        raise AnalysisContractError(
            "current deterministic planner supports metric improvement claims"
        )
    metric = _discovery_attribute(discovered_claim, "metric")
    if not isinstance(metric, str) or not metric.strip():
        raise AnalysisContractError("discovered audit metric is unavailable")
    discovered_direction = _discovery_attribute(discovered_claim, "direction")
    direction = {
        ClaimDirection.HIGHER: Direction.HIGHER,
        ClaimDirection.LOWER: Direction.LOWER,
    }.get(discovered_direction)
    if direction is None:
        raise AnalysisContractError("discovered audit direction is unavailable")
    baseline = _discovery_attribute(discovered_claim, "baseline_value")
    candidate = _discovery_attribute(discovered_claim, "candidate_value")
    claimed_values: tuple[ClaimedMetricValue, ...] = ()
    if (baseline is None) != (candidate is None):
        raise AnalysisContractError(
            "discovered baseline and candidate values must appear together"
        )
    if baseline is not None and candidate is not None:
        claimed_values = (
            _claimed_value(baseline, ExperimentRole.BASELINE),
            _claimed_value(candidate, ExperimentRole.CANDIDATE),
        )
    threshold, threshold_provenance = _validated_explicit_threshold(
        reference.text,
        _discovery_attribute(discovered_claim, "minimum_improvement"),
    )
    audit_claim = AuditClaimSpec(
        claim_id=reference.claim_id,
        metric=metric,
        direction=direction,
        minimum_absolute_improvement=threshold,
        metric_provenance=reference.provenance,
        direction_provenance=reference.provenance,
        threshold_provenance=threshold_provenance,
        claimed_values=claimed_values,
    )
    if not isinstance(artifacts, tuple) or not all(
        type(item) is ArtifactCandidate for item in artifacts
    ):
        raise TypeError("discovery artifacts must be ArtifactCandidate values")
    if any(reference.claim_id not in item.relevant_claim_ids for item in artifacts):
        raise AnalysisContractError(
            "discovery artifacts must be relevant to the selected claim"
        )
    return PlanningRequest(
        repository=repository,
        pr_number=pr_number,
        head_sha=head_sha,
        claim=reference,
        audit_claim=audit_claim,
        artifacts=artifacts,  # type: ignore[arg-type]
        normalized_evidence=normalized_evidence,
        mapping_candidates=mapping_candidates,  # type: ignore[arg-type]
        approved_mapping=approved_mapping,  # type: ignore[arg-type]
        upstream_mapping_question=mapping_question,  # type: ignore[arg-type]
    )


def plan_ephemeral_audit(request: PlanningRequest) -> PlanningOutcome:
    """Select one trusted mapping and produce a head-bound ephemeral plan."""

    if type(request) is not PlanningRequest:
        raise TypeError("ephemeral planning requires PlanningRequest")

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
    )
    return PlanningOutcome(state=PlanningState.READY, plan=plan)


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
        dataset_splits = {
            reference.split.casefold()
            for item in role_evidence
            if item.artifact.kind is ArtifactKind.DATASET
            for observation in item.observations
            for reference in observation.dataset_references
            if reference.split is not None
        }
        if not {"train", "eval"}.issubset(dataset_splits):
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
    return tuple(missing)


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
        splits: set[str] = set()
        for binding in role_bindings:
            if binding.kind is not ArtifactKind.DATASET:
                continue
            evidence = evidence_by_key.get((binding.path, binding.kind))
            if evidence is None:
                continue
            splits.update(
                reference.split.casefold()
                for observation in evidence.observations
                for reference in observation.dataset_references
                if reference.split is not None
            )
        if not {"train", "eval"}.issubset(splits):
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
    if approved is not None and _mapping_is_runtime_revalidatable(approved, request):
        approved_signature = _mapping_signature(approved)
        strong_conflicts = tuple(
            item
            for item in valid_candidates
            if item.confidence.value >= 0.90
            and _mapping_signature(item) != approved_signature
        )
        if not strong_conflicts:
            return approved
        return None

    if len(valid_candidates) != 1:
        return None
    candidate = valid_candidates[0]
    strong_others = tuple(
        item
        for item in request.mapping_candidates
        if item is not candidate and item.confidence.value >= 0.90
    )
    if strong_others:
        return None
    if candidate.trust is MappingTrust.MANIFEST_HINT:
        return candidate
    if candidate.confidence.value < 0.90:
        return None
    if candidate.provenance.kind is ProvenanceKind.PROVIDER_PROPOSAL:
        return None
    return candidate


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
        "mapping": _selected_mapping_identity(mapping),
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


__all__ = [
    "PlanningOutcome",
    "PlanningRequest",
    "PlanningState",
    "plan_ephemeral_audit",
    "planning_request_from_discovery",
]
