"""Hosted-runner orchestration for one unified zero-configuration analysis."""

from __future__ import annotations

from claimci.review.analysis_bridge import AnalysisReviewContext, run_analysis_review
from claimci.review.models import ReviewConfig
from claimci.review.provider import ReviewerProvider

from .contracts import (
    AnalysisState,
    ArtifactKind,
    DeterministicAuditOutcome,
    EphemeralAuditPlan,
    ExperimentRole,
    FieldProvenance,
    MissingEvidence,
    UnifiedAnalysisResult,
)
from .materialize import (
    MaterializationPartial,
    MaterializationUnavailable,
    RuntimeExecutionContext,
    execute_ephemeral_audit_with_trace,
)
from .planner import PlanningRequest, PlanningState, plan_ephemeral_audit


_PLANNING_UNAVAILABLE = "analysis planning is unavailable"
_EXECUTION_UNAVAILABLE = "trusted deterministic execution is unavailable"
_REPRESENTATION_PARTIAL = (
    "validated evidence cannot be represented by the current deterministic engine"
)
_REVIEW_PARTIAL = "advisory Research Review is unavailable"


def _unavailable(reason: str) -> UnifiedAnalysisResult:
    return UnifiedAnalysisResult(
        state=AnalysisState.UNAVAILABLE,
        unavailable_reason=reason,
    )


def _evidence_provenance(
    plan: EphemeralAuditPlan,
) -> tuple[FieldProvenance, ...]:
    values: list[FieldProvenance] = list(plan.mapping_provenance)
    for evidence in (*plan.baseline_evidence, *plan.candidate_evidence):
        values.append(evidence.artifact.provenance)
        values.extend(evidence.adapter_match.match_evidence)
        for mapping in evidence.adapter_match.mappings:
            values.extend((mapping.provenance, mapping.selector.provenance))
        for observation in evidence.observations:
            values.append(observation.provenance)
            values.extend(item.provenance for item in observation.config_values)
            values.extend(item.provenance for item in observation.dataset_references)
            values.extend(item.provenance for item in observation.compute_evidence)
    unique: list[FieldProvenance] = []
    for item in values:
        if item not in unique:
            unique.append(item)
    return tuple(unique)


def run_unified_analysis(
    request: PlanningRequest,
    runtime: RuntimeExecutionContext,
    review_config: ReviewConfig,
    *,
    provider: ReviewerProvider | None = None,
) -> UnifiedAnalysisResult:
    """Run planner, existing Audit, and one advisory synthesis in order."""

    if type(request) is not PlanningRequest:
        raise TypeError("unified analysis requires PlanningRequest")
    if type(runtime) is not RuntimeExecutionContext:
        raise TypeError("unified analysis requires RuntimeExecutionContext")
    if type(review_config) is not ReviewConfig:
        raise TypeError("unified analysis requires ReviewConfig")

    try:
        planning = plan_ephemeral_audit(request)
    except Exception:
        return _unavailable(_PLANNING_UNAVAILABLE)

    if planning.state is PlanningState.MAPPING_NEEDED:
        if planning.mapping_question is None:
            return _unavailable(_PLANNING_UNAVAILABLE)
        return UnifiedAnalysisResult(
            state=AnalysisState.MAPPING_NEEDED,
            mapping_question=planning.mapping_question,
        )
    if planning.state is PlanningState.PARTIAL:
        return UnifiedAnalysisResult(
            state=AnalysisState.PARTIAL,
            unavailable_reason=(
                None if planning.missing_evidence else _REPRESENTATION_PARTIAL
            ),
            missing_evidence=planning.missing_evidence,
        )
    if planning.state is PlanningState.UNAVAILABLE:
        return _unavailable(_PLANNING_UNAVAILABLE)
    if planning.plan is None:
        return _unavailable(_PLANNING_UNAVAILABLE)
    plan = planning.plan

    try:
        execution = execute_ephemeral_audit_with_trace(plan, runtime)
        audit_result = execution.audit_result
        evidence_trace = execution.trace
        deterministic = DeterministicAuditOutcome.from_audit_result(audit_result)
    except MaterializationPartial:
        return UnifiedAnalysisResult(
            state=AnalysisState.PARTIAL,
            missing_evidence=(
                MissingEvidence(
                    kind=ArtifactKind.MANIFEST,
                    role=ExperimentRole.UNSPECIFIED,
                    description=_REPRESENTATION_PARTIAL,
                    claim_id=plan.claim.claim_id,
                ),
            ),
        )
    except MaterializationUnavailable:
        return _unavailable(_EXECUTION_UNAVAILABLE)
    except Exception:
        return _unavailable(_EXECUTION_UNAVAILABLE)

    try:
        if plan.audit_claim is None:
            return UnifiedAnalysisResult(
                state=AnalysisState.PARTIAL,
                deterministic=deterministic,
                unavailable_reason=_REVIEW_PARTIAL,
                evidence_trace=evidence_trace,
            )
        review_context = AnalysisReviewContext(
            claim=plan.claim,
            audit_claim=plan.audit_claim,
            evidence_provenance=_evidence_provenance(plan),
            deterministic=deterministic,
            missing_evidence=plan.missing_evidence,
        )
        interpretation = run_analysis_review(
            review_context,
            review_config,
            provider=provider,
        )
    except Exception:
        return UnifiedAnalysisResult(
            state=AnalysisState.PARTIAL,
            deterministic=deterministic,
            unavailable_reason=_REVIEW_PARTIAL,
            missing_evidence=plan.missing_evidence,
            evidence_trace=evidence_trace,
        )

    try:
        evidence_trace = evidence_trace.with_advisory(interpretation)
    except Exception:
        from .trace import EvidenceTraceBundle

        evidence_trace = EvidenceTraceBundle.unavailable(
            head_sha=plan.head_sha,
            result=audit_result,
            interpretation=interpretation,
        )

    return UnifiedAnalysisResult(
        state=AnalysisState.COMPLETE,
        deterministic=deterministic,
        research_interpretation=interpretation,
        missing_evidence=plan.missing_evidence,
        evidence_trace=evidence_trace,
    )


__all__ = ["run_unified_analysis"]
