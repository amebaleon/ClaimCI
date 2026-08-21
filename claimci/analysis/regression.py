"""Contextual historical transition classification over trusted Audit commitments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from claimci.models import Verdict

from .contracts import (
    AnalysisState,
    ExperimentRole,
    Sha256Digest,
    UnifiedAnalysisResult,
)
from .replay import ReplayRecipe
from .verification import VerificationInputSnapshot, VerificationSnapshotCapability


class ExecutionClassification(str, Enum):
    DETERMINISTIC_COMPLETE = "deterministic_complete"
    PRE_AUDIT_EVIDENCE_PARTIAL = "pre_audit_evidence_partial"
    MAPPING_NEEDED = "mapping_needed"
    POST_AUDIT_ADVISORY_PARTIAL = "post_audit_advisory_partial"
    OPERATIONAL_UNAVAILABLE = "operational_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RUNNER_SNAPSHOT_UNAVAILABLE = "runner_snapshot_unavailable"


class OutcomeComparability(str, Enum):
    EXACT = "exact"
    CAPTURED_SCOPE = "captured_scope"
    CLAIM_CHANGED = "claim_changed"
    MEASUREMENT_CHANGED = "measurement_changed"
    BASELINE_CHANGED = "baseline_changed"
    ENGINE_CHANGED = "engine_changed"
    INDETERMINATE = "indeterminate"
    LEGACY_UNAVAILABLE = "legacy_unavailable"


class VerdictTransitionKind(str, Enum):
    UNCHANGED = "unchanged"
    SUPPORTED_TO_NOT_SUPPORTED = "supported_to_not_supported"
    SUPPORTED_TO_INSUFFICIENT = "supported_to_insufficient"
    SUPPORTED_TO_NO_VERDICT = "supported_to_no_verdict"
    NOT_SUPPORTED_TO_SUPPORTED = "not_supported_to_supported"
    NOT_SUPPORTED_TO_INSUFFICIENT = "not_supported_to_insufficient"
    NOT_SUPPORTED_TO_NO_VERDICT = "not_supported_to_no_verdict"
    INSUFFICIENT_TO_SUPPORTED = "insufficient_to_supported"
    INSUFFICIENT_TO_NOT_SUPPORTED = "insufficient_to_not_supported"
    INSUFFICIENT_TO_NO_VERDICT = "insufficient_to_no_verdict"
    NO_VERDICT_TO_SUPPORTED = "no_verdict_to_supported"
    NO_VERDICT_TO_NOT_SUPPORTED = "no_verdict_to_not_supported"
    NO_VERDICT_TO_INSUFFICIENT = "no_verdict_to_insufficient"
    COMPLETENESS_REGRESSION = "completeness_regression"
    COMPLETENESS_RECOVERY = "completeness_recovery"
    RAW_TRANSITION_NOT_COMPARABLE = "raw_transition_not_comparable"


class RegressionClassification(str, Enum):
    REGRESSION = "regression"
    EVIDENCE_REGRESSION = "evidence_regression"
    COVERAGE_REGRESSION = "coverage_regression"
    RESOLUTION = "resolution"
    UNCHANGED = "unchanged"
    NOT_COMPARABLE = "not_comparable"


_UNAVAILABLE_EXECUTIONS = frozenset(
    {
        ExecutionClassification.OPERATIONAL_UNAVAILABLE,
        ExecutionClassification.PROVIDER_UNAVAILABLE,
        ExecutionClassification.RUNNER_SNAPSHOT_UNAVAILABLE,
    }
)


@dataclass(frozen=True, slots=True, init=False)
class HistoricalAnalysisOutcome:
    """One validated lifecycle outcome with optional exact Replay authority."""

    execution: ExecutionClassification
    verdict: Verdict | None
    replay_recipe: ReplayRecipe | None

    def __init__(self) -> None:
        raise TypeError(
            "HistoricalAnalysisOutcome must be created from UnifiedAnalysisResult"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("HistoricalAnalysisOutcome is final")

    @classmethod
    def from_unified_result(
        cls,
        result: UnifiedAnalysisResult,
        *,
        unavailable_classification: ExecutionClassification | None = None,
    ) -> "HistoricalAnalysisOutcome":
        if type(result) is not UnifiedAnalysisResult:
            raise TypeError(
                "historical outcome requires an exact UnifiedAnalysisResult"
            )
        if unavailable_classification is not None and type(
            unavailable_classification
        ) is not ExecutionClassification:
            raise TypeError(
                "unavailable classification must be ExecutionClassification"
            )

        if result.state is AnalysisState.UNAVAILABLE:
            if unavailable_classification not in _UNAVAILABLE_EXECUTIONS:
                raise ValueError(
                    "unavailable analysis requires a typed unavailable classification"
                )
            execution = unavailable_classification
        else:
            if unavailable_classification is not None:
                raise ValueError(
                    "unavailable classification is only valid for unavailable analysis"
                )
            if result.state is AnalysisState.COMPLETE:
                execution = ExecutionClassification.DETERMINISTIC_COMPLETE
            elif result.state is AnalysisState.MAPPING_NEEDED:
                execution = ExecutionClassification.MAPPING_NEEDED
            elif result.state is AnalysisState.PARTIAL:
                execution = (
                    ExecutionClassification.POST_AUDIT_ADVISORY_PARTIAL
                    if result.deterministic is not None
                    else ExecutionClassification.PRE_AUDIT_EVIDENCE_PARTIAL
                )
            else:  # pragma: no cover - AnalysisState is a closed enum
                raise ValueError("unsupported unified analysis lifecycle state")

        instance = object.__new__(HistoricalAnalysisOutcome)
        object.__setattr__(instance, "execution", execution)
        object.__setattr__(
            instance,
            "verdict",
            None if result.deterministic is None else result.deterministic.verdict,
        )
        object.__setattr__(instance, "replay_recipe", result.replay_recipe)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class VerdictRegressionReport:
    """Non-authoritative metadata describing one historical transition."""

    previous_execution: ExecutionClassification
    current_execution: ExecutionClassification
    comparability: OutcomeComparability
    transition_kind: VerdictTransitionKind
    classification: RegressionClassification
    previous_verdict: Verdict | None
    current_verdict: Verdict | None
    previous_audit_sha256: Sha256Digest | None
    current_audit_sha256: Sha256Digest | None
    previous_comparison_frame_sha256: Sha256Digest | None
    current_comparison_frame_sha256: Sha256Digest | None
    previous_captured_audit_input_sha256: Sha256Digest | None
    current_captured_audit_input_sha256: Sha256Digest | None

    def __init__(self) -> None:
        raise TypeError(
            "VerdictRegressionReport must be created by the regression classifier"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("VerdictRegressionReport is final")


_RAW_TRANSITIONS = {
    (None, None): VerdictTransitionKind.UNCHANGED,
    (None, Verdict.SUPPORTED): VerdictTransitionKind.NO_VERDICT_TO_SUPPORTED,
    (None, Verdict.NOT_SUPPORTED): (
        VerdictTransitionKind.NO_VERDICT_TO_NOT_SUPPORTED
    ),
    (None, Verdict.INSUFFICIENT_EVIDENCE): (
        VerdictTransitionKind.NO_VERDICT_TO_INSUFFICIENT
    ),
    (Verdict.SUPPORTED, None): VerdictTransitionKind.SUPPORTED_TO_NO_VERDICT,
    (Verdict.NOT_SUPPORTED, None): (
        VerdictTransitionKind.NOT_SUPPORTED_TO_NO_VERDICT
    ),
    (Verdict.INSUFFICIENT_EVIDENCE, None): (
        VerdictTransitionKind.INSUFFICIENT_TO_NO_VERDICT
    ),
    (Verdict.SUPPORTED, Verdict.SUPPORTED): VerdictTransitionKind.UNCHANGED,
    (Verdict.SUPPORTED, Verdict.NOT_SUPPORTED): (
        VerdictTransitionKind.SUPPORTED_TO_NOT_SUPPORTED
    ),
    (Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE): (
        VerdictTransitionKind.SUPPORTED_TO_INSUFFICIENT
    ),
    (Verdict.NOT_SUPPORTED, Verdict.SUPPORTED): (
        VerdictTransitionKind.NOT_SUPPORTED_TO_SUPPORTED
    ),
    (Verdict.NOT_SUPPORTED, Verdict.NOT_SUPPORTED): (
        VerdictTransitionKind.UNCHANGED
    ),
    (Verdict.NOT_SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE): (
        VerdictTransitionKind.NOT_SUPPORTED_TO_INSUFFICIENT
    ),
    (Verdict.INSUFFICIENT_EVIDENCE, Verdict.SUPPORTED): (
        VerdictTransitionKind.INSUFFICIENT_TO_SUPPORTED
    ),
    (Verdict.INSUFFICIENT_EVIDENCE, Verdict.NOT_SUPPORTED): (
        VerdictTransitionKind.INSUFFICIENT_TO_NOT_SUPPORTED
    ),
    (Verdict.INSUFFICIENT_EVIDENCE, Verdict.INSUFFICIENT_EVIDENCE): (
        VerdictTransitionKind.UNCHANGED
    ),
}


def _raw_verdict_transition(
    previous: Verdict | None,
    current: Verdict | None,
) -> VerdictTransitionKind:
    """Name a raw verdict movement without assigning regression authority."""

    if previous is not None and type(previous) is not Verdict:
        raise TypeError("previous raw verdict must be Verdict or null")
    if current is not None and type(current) is not Verdict:
        raise TypeError("current raw verdict must be Verdict or null")
    return _RAW_TRANSITIONS[(previous, current)]


def _snapshot(outcome: HistoricalAnalysisOutcome) -> VerificationInputSnapshot | None:
    recipe = outcome.replay_recipe
    return None if recipe is None else recipe.input_snapshot


def _has_actual_audit(outcome: HistoricalAnalysisOutcome) -> bool:
    recipe = outcome.replay_recipe
    return (
        outcome.verdict is not None
        and type(recipe) is ReplayRecipe
        and recipe.audit_commitment.verdict is outcome.verdict
        and recipe.input_snapshot.capability
        is not VerificationSnapshotCapability.UNAVAILABLE
    )


def _baseline_semantics(snapshot: VerificationInputSnapshot) -> tuple[object, ...]:
    return tuple(
        (
            item.kind.value,
            item.role.value,
            None if item.dataset_split is None else item.dataset_split.value,
            item.adapter_id,
            item.adapter_semantic_version,
            item.evidence_projector_version,
            item.selectors,
            item.audit_semantics_sha256,
        )
        for item in snapshot.artifacts
        if item.role in {ExperimentRole.BASELINE, ExperimentRole.REFERENCE}
    )


def _comparability(
    previous: HistoricalAnalysisOutcome,
    current: HistoricalAnalysisOutcome,
) -> OutcomeComparability:
    if previous.verdict is None or current.verdict is None:
        return OutcomeComparability.INDETERMINATE
    if not _has_actual_audit(previous) or not _has_actual_audit(current):
        return OutcomeComparability.LEGACY_UNAVAILABLE
    previous_snapshot = _snapshot(previous)
    current_snapshot = _snapshot(current)
    assert previous_snapshot is not None
    assert current_snapshot is not None

    previous_claim = previous_snapshot.claim_identity
    current_claim = current_snapshot.claim_identity
    if previous_claim is None or current_claim is None:
        return OutcomeComparability.LEGACY_UNAVAILABLE
    if previous_claim.audit_spec_sha256 != current_claim.audit_spec_sha256:
        return OutcomeComparability.CLAIM_CHANGED

    previous_measurements = (
        previous_snapshot.baseline_measurement,
        previous_snapshot.candidate_measurement,
    )
    current_measurements = (
        current_snapshot.baseline_measurement,
        current_snapshot.candidate_measurement,
    )
    if any(item is None for item in (*previous_measurements, *current_measurements)):
        return OutcomeComparability.LEGACY_UNAVAILABLE
    if tuple(item.semantic_protocol_id for item in previous_measurements if item) != tuple(
        item.semantic_protocol_id for item in current_measurements if item
    ):
        return OutcomeComparability.MEASUREMENT_CHANGED

    previous_profile = previous_snapshot.profile_identity
    current_profile = current_snapshot.profile_identity
    if previous_profile is None or current_profile is None:
        return OutcomeComparability.LEGACY_UNAVAILABLE
    if (
        previous_profile.profile_semantics_sha256
        != current_profile.profile_semantics_sha256
    ):
        return OutcomeComparability.INDETERMINATE

    previous_compatibility = previous_snapshot.audit_compatibility
    current_compatibility = current_snapshot.audit_compatibility
    if previous_compatibility is None or current_compatibility is None:
        return OutcomeComparability.LEGACY_UNAVAILABLE
    if (
        previous_compatibility.audit_semantics_sha256
        != current_compatibility.audit_semantics_sha256
    ):
        return OutcomeComparability.ENGINE_CHANGED

    previous_frame = previous_snapshot.comparison_frame_sha256
    current_frame = current_snapshot.comparison_frame_sha256
    previous_input = previous_snapshot.captured_audit_input_sha256
    current_input = current_snapshot.captured_audit_input_sha256
    if None in {previous_frame, current_frame, previous_input, current_input}:
        return OutcomeComparability.LEGACY_UNAVAILABLE
    if previous_frame == current_frame:
        return (
            OutcomeComparability.EXACT
            if previous_input == current_input
            else OutcomeComparability.CAPTURED_SCOPE
        )
    if (
        previous_snapshot.capability is VerificationSnapshotCapability.COMPLETE
        and current_snapshot.capability is VerificationSnapshotCapability.COMPLETE
        and _baseline_semantics(previous_snapshot)
        != _baseline_semantics(current_snapshot)
    ):
        return OutcomeComparability.BASELINE_CHANGED
    return OutcomeComparability.INDETERMINATE


def _compatible(comparability: OutcomeComparability) -> bool:
    return comparability in {
        OutcomeComparability.EXACT,
        OutcomeComparability.CAPTURED_SCOPE,
    }


def _classification_for_two_audits(
    transition: VerdictTransitionKind,
) -> RegressionClassification:
    if transition is VerdictTransitionKind.SUPPORTED_TO_NOT_SUPPORTED:
        return RegressionClassification.REGRESSION
    if transition in {
        VerdictTransitionKind.SUPPORTED_TO_INSUFFICIENT,
        VerdictTransitionKind.NOT_SUPPORTED_TO_INSUFFICIENT,
    }:
        return RegressionClassification.EVIDENCE_REGRESSION
    if transition in {
        VerdictTransitionKind.NOT_SUPPORTED_TO_SUPPORTED,
        VerdictTransitionKind.INSUFFICIENT_TO_SUPPORTED,
        VerdictTransitionKind.INSUFFICIENT_TO_NOT_SUPPORTED,
        VerdictTransitionKind.COMPLETENESS_RECOVERY,
    }:
        return RegressionClassification.RESOLUTION
    return RegressionClassification.UNCHANGED


def _report(
    previous: HistoricalAnalysisOutcome,
    current: HistoricalAnalysisOutcome,
    *,
    comparability: OutcomeComparability,
    transition: VerdictTransitionKind,
    classification: RegressionClassification,
) -> VerdictRegressionReport:
    previous_recipe = previous.replay_recipe
    current_recipe = current.replay_recipe
    previous_snapshot = _snapshot(previous)
    current_snapshot = _snapshot(current)
    instance = object.__new__(VerdictRegressionReport)
    values = {
        "previous_execution": previous.execution,
        "current_execution": current.execution,
        "comparability": comparability,
        "transition_kind": transition,
        "classification": classification,
        "previous_verdict": previous.verdict,
        "current_verdict": current.verdict,
        "previous_audit_sha256": (
            None
            if previous_recipe is None
            else previous_recipe.audit_commitment.stable_audit_sha256
        ),
        "current_audit_sha256": (
            None
            if current_recipe is None
            else current_recipe.audit_commitment.stable_audit_sha256
        ),
        "previous_comparison_frame_sha256": (
            None if previous_snapshot is None else previous_snapshot.comparison_frame_sha256
        ),
        "current_comparison_frame_sha256": (
            None if current_snapshot is None else current_snapshot.comparison_frame_sha256
        ),
        "previous_captured_audit_input_sha256": (
            None
            if previous_snapshot is None
            else previous_snapshot.captured_audit_input_sha256
        ),
        "current_captured_audit_input_sha256": (
            None
            if current_snapshot is None
            else current_snapshot.captured_audit_input_sha256
        ),
    }
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def classify_verdict_regression(
    previous: HistoricalAnalysisOutcome,
    current: HistoricalAnalysisOutcome,
) -> VerdictRegressionReport:
    """Classify one historical transition without changing current authority."""

    if type(previous) is not HistoricalAnalysisOutcome or type(
        current
    ) is not HistoricalAnalysisOutcome:
        raise TypeError(
            "verdict regression requires exact HistoricalAnalysisOutcome values"
        )
    comparability = _comparability(previous, current)
    transition = _raw_verdict_transition(previous.verdict, current.verdict)
    two_audits = _has_actual_audit(previous) and _has_actual_audit(current)

    if two_audits:
        if not _compatible(comparability):
            transition = VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
            classification = RegressionClassification.NOT_COMPARABLE
        else:
            if previous.verdict is current.verdict:
                if (
                    previous.execution
                    is ExecutionClassification.DETERMINISTIC_COMPLETE
                    and current.execution
                    is ExecutionClassification.POST_AUDIT_ADVISORY_PARTIAL
                ):
                    transition = VerdictTransitionKind.COMPLETENESS_REGRESSION
                elif (
                    previous.execution
                    is ExecutionClassification.POST_AUDIT_ADVISORY_PARTIAL
                    and current.execution
                    is ExecutionClassification.DETERMINISTIC_COMPLETE
                ):
                    transition = VerdictTransitionKind.COMPLETENESS_RECOVERY
            classification = _classification_for_two_audits(transition)
    elif (
        _has_actual_audit(previous)
        and current.execution
        is ExecutionClassification.PRE_AUDIT_EVIDENCE_PARTIAL
    ):
        classification = RegressionClassification.COVERAGE_REGRESSION
    elif (
        previous.execution
        is ExecutionClassification.PRE_AUDIT_EVIDENCE_PARTIAL
        and _has_actual_audit(current)
    ):
        transition = VerdictTransitionKind.COMPLETENESS_RECOVERY
        classification = RegressionClassification.RESOLUTION
    else:
        if comparability is OutcomeComparability.LEGACY_UNAVAILABLE:
            transition = VerdictTransitionKind.RAW_TRANSITION_NOT_COMPARABLE
        classification = RegressionClassification.NOT_COMPARABLE

    return _report(
        previous,
        current,
        comparability=comparability,
        transition=transition,
        classification=classification,
    )


__all__ = [
    "ExecutionClassification",
    "HistoricalAnalysisOutcome",
    "OutcomeComparability",
    "RegressionClassification",
    "VerdictRegressionReport",
    "VerdictTransitionKind",
    "classify_verdict_regression",
]
