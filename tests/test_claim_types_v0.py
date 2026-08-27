from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from claimci.analysis import (
    AbsoluteMetricClaim,
    CanonicalScientificClaim,
    ClaimBoundComparator,
    ClaimEvidencePolicy,
    ClaimReference,
    Confidence,
    EvaluationConstraint,
    EvaluationConstraintKind,
    ExperimentRole,
    FieldProvenance,
    GeneralizationClaim,
    GenericQuantitativeClaim,
    MetricImprovementClaim,
    PrimaryClaimKind,
    ProvenanceKind,
    RepositoryPath,
    UnsupportedDeterministicClaimCompiler,
    claim_evidence_policy,
    claim_semantic_projection,
    compile_audit_claim,
    recover_scientific_claim,
    to_jsonable,
)
from claimci.models import Direction


def _reference(
    text: str,
    *,
    provenance_kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> ClaimReference:
    return ClaimReference(
        claim_id="claim-source-bound",
        text=text,
        source_path=RepositoryPath("CLAIM.md"),
        confidence=Confidence(0.95),
        provenance=FieldProvenance(
            provenance_kind,
            "source-bound claim fixture",
            RepositoryPath("CLAIM.md"),
            "source-claim",
        ),
    )


def test_held_out_metric_improvement_has_primary_precedence_and_constraint() -> None:
    claim = recover_scientific_claim(
        _reference(
            "On the held-out test set, accuracy improved from 0.70 to 0.80 "
            "by at least 0.05."
        )
    )

    assert type(claim) is CanonicalScientificClaim
    assert type(claim.primary) is MetricImprovementClaim
    assert claim.primary.metric == "accuracy"
    assert claim.primary.direction is Direction.HIGHER
    assert claim.primary.baseline_value is not None
    assert claim.primary.baseline_value.value == 0.70
    assert claim.primary.candidate_value is not None
    assert claim.primary.candidate_value.value == 0.80
    assert claim.primary.minimum_improvement is not None
    assert claim.primary.minimum_improvement.value == 0.05
    assert claim.constraints == (
        EvaluationConstraint(
            kind=EvaluationConstraintKind.HELD_OUT,
            provenance=claim.reference.provenance,
        ),
    )


@pytest.mark.parametrize(
    "text,comparator",
    [
        ("Candidate accuracy is at least 0.90.", ClaimBoundComparator.AT_LEAST),
        ("Candidate accuracy is 0.90.", ClaimBoundComparator.EQUAL),
    ],
)
def test_exact_single_role_bound_is_absolute_metric_claim(
    text: str,
    comparator: ClaimBoundComparator,
) -> None:
    claim = recover_scientific_claim(_reference(text))

    assert claim is not None
    assert type(claim.primary) is AbsoluteMetricClaim
    assert claim.primary.metric == "accuracy"
    assert claim.primary.role is ExperimentRole.CANDIDATE
    assert claim.primary.comparator is comparator
    assert claim.primary.bound.value == 0.90


def test_broad_generalization_precedes_generic_quantitative() -> None:
    claim = recover_scientific_claim(
        _reference("The candidate generalizes across 12 unseen domains.")
    )

    assert claim is not None
    assert type(claim.primary) is GeneralizationClaim


def test_other_finite_bounded_language_is_generic_quantitative() -> None:
    claim = recover_scientific_claim(
        _reference("The candidate uses 40% less memory.")
    )

    assert claim is not None
    assert type(claim.primary) is GenericQuantitativeClaim
    assert tuple(item.value for item in claim.primary.quantities) == (40.0,)
    assert tuple(item.unit for item in claim.primary.quantities) == ("%",)


@pytest.mark.parametrize(
    "text",
    [
        "Accuracy was 0.8 or 0.9 depending on the run.",
        "Accuracy improved by 0.05 while loss decreased by 0.10.",
        "Candidate accuracy is at least NaN.",
        "Candidate accuracy might be higher or lower.",
    ],
)
def test_malformed_or_ambiguous_quantitative_language_is_not_recognized(
    text: str,
) -> None:
    assert recover_scientific_claim(_reference(text)) is None


def test_metric_compiler_recovers_only_source_bound_fields() -> None:
    claim = recover_scientific_claim(
        _reference("Loss decreased from 0.40 to 0.20 by at least 0.10.")
    )

    assert claim is not None
    audit = compile_audit_claim(claim)
    assert audit.metric == "loss"
    assert audit.direction is Direction.LOWER
    assert audit.minimum_absolute_improvement == 0.10
    assert tuple(item.value for item in audit.claimed_values) == (0.40, 0.20)


@pytest.mark.parametrize(
    ("text", "metric", "direction", "baseline", "candidate"),
    (
        (
            "accuracy improved from 0.60 to 0.70",
            "accuracy",
            Direction.HIGHER,
            0.60,
            0.70,
        ),
        (
            "candidate improves accuracy from 0.60 to 0.70",
            "accuracy",
            Direction.HIGHER,
            0.60,
            0.70,
        ),
        (
            "the candidate improves accuracy from 0.60 to 0.70",
            "accuracy",
            Direction.HIGHER,
            0.60,
            0.70,
        ),
        (
            "model improves accuracy from 0.60 to 0.70",
            "accuracy",
            Direction.HIGHER,
            0.60,
            0.70,
        ),
        (
            "candidate increases accuracy from 0.60 to 0.70",
            "accuracy",
            Direction.HIGHER,
            0.60,
            0.70,
        ),
        (
            "candidate raises accuracy from 0.60 to 0.70",
            "accuracy",
            Direction.HIGHER,
            0.60,
            0.70,
        ),
        (
            "candidate grows accuracy from 0.60 to 0.70",
            "accuracy",
            Direction.HIGHER,
            0.60,
            0.70,
        ),
        (
            "candidate reduces loss from 0.40 to 0.30",
            "loss",
            Direction.LOWER,
            0.40,
            0.30,
        ),
        (
            "candidate decreases loss from 0.40 to 0.30",
            "loss",
            Direction.LOWER,
            0.40,
            0.30,
        ),
        (
            "candidate lowers loss from 0.40 to 0.30",
            "loss",
            Direction.LOWER,
            0.40,
            0.30,
        ),
        (
            "candidate drops loss from 0.40 to 0.30",
            "loss",
            Direction.LOWER,
            0.40,
            0.30,
        ),
    ),
)
def test_metric_improvement_recovery_uses_one_bounded_grammar(
    text: str,
    metric: str,
    direction: Direction,
    baseline: float,
    candidate: float,
) -> None:
    recovered = recover_scientific_claim(_reference(text))

    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.metric == metric
    assert recovered.primary.direction is direction
    assert recovered.primary.baseline_value is not None
    assert recovered.primary.candidate_value is not None
    assert recovered.primary.baseline_value.value == baseline
    assert recovered.primary.candidate_value.value == candidate


def test_subject_first_threshold_uses_existing_explicit_continuation() -> None:
    recovered = recover_scientific_claim(
        _reference(
            "candidate improves accuracy from 0.60 to 0.70; "
            "improved by at least 0.05"
        )
    )

    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.minimum_improvement is not None
    assert recovered.primary.minimum_improvement.value == 0.05


def test_subject_first_inline_threshold_remains_explicit() -> None:
    recovered = recover_scientific_claim(
        _reference(
            "candidate improves accuracy from 0.60 to 0.70 by at least 0.05"
        )
    )

    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.minimum_improvement is not None
    assert recovered.primary.minimum_improvement.value == 0.05


def test_subject_first_lower_threshold_uses_existing_explicit_continuation() -> None:
    recovered = recover_scientific_claim(
        _reference(
            "candidate reduces loss from 0.40 to 0.30; "
            "reduced by at least 0.05"
        )
    )

    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.direction is Direction.LOWER
    assert recovered.primary.minimum_improvement is not None
    assert recovered.primary.minimum_improvement.value == 0.05


def test_subject_first_does_not_derive_threshold_from_arithmetic() -> None:
    recovered = recover_scientific_claim(
        _reference(
            "candidate improves accuracy from 0.60 to 0.70; "
            "0.70 - 0.60 = 0.10"
        )
    )

    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.minimum_improvement is None


@pytest.mark.parametrize(
    "text",
    (
        (
            "candidate improves accuracy from 0.60 to 0.70, at least 2 GPUs "
            "were used"
        ),
        (
            "candidate improves accuracy from 0.60 to 0.70, minimum of 3 epochs "
            "were run"
        ),
        (
            "candidate improves accuracy from 0.60 to 0.70: no less than 4 "
            "training runs were completed"
        ),
        (
            "candidate improves accuracy from 0.60 to 0.70 (at least 5 GPU-hours "
            "were used)"
        ),
    ),
)
def test_subject_first_does_not_borrow_unrelated_clause_threshold(
    text: str,
) -> None:
    recovered = recover_scientific_claim(_reference(text))

    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    assert recovered.primary.baseline_value is not None
    assert recovered.primary.baseline_value.value == 0.60
    assert recovered.primary.candidate_value is not None
    assert recovered.primary.candidate_value.value == 0.70
    assert recovered.primary.minimum_improvement is None


@pytest.mark.parametrize(
    "text",
    (
        (
            "candidate improves accuracy from 0.60 to 0.70 and loss from "
            "0.40 to 0.30"
        ),
        (
            "candidate improves accuracy from 0.60 to 0.70; improved by at "
            "least 0.05; improved by at least 0.06"
        ),
        (
            "candidate improves accuracy from 0.60 to 0.70; improved by at "
            "least 0.05, minimum 0.06"
        ),
        (
            "candidate improves accuracy from 0.60 to 0.70; improved by at "
            "least 0.05 and no less than 0.06"
        ),
        (
            "candidate says model improves accuracy from 0.60 to 0.70 while "
            "candidate reduces loss from 0.40 to 0.30"
        ),
    ),
)
def test_subject_first_rejects_ambiguous_metric_improvement(text: str) -> None:
    assert recover_scientific_claim(_reference(text)) is None


@pytest.mark.parametrize(
    "text",
    (
        "candidates improves accuracy from 0.60 to 0.70",
        "candidate improve accuracy from 0.60 to 0.70",
        "candidate improvesaccuracy from 0.60 to 0.70",
        "candidate's improves accuracy from 0.60 to 0.70",
        "system improves accuracy from 0.60 to 0.70",
    ),
)
def test_subject_first_rejects_unapproved_subject_or_verb_boundaries(
    text: str,
) -> None:
    assert recover_scientific_claim(_reference(text)) is None


@pytest.mark.parametrize(
    "text,expected_threshold",
    [
        (
            "accuracy improved from 0.60 to 0.70; improved by at least 0.05",
            0.05,
        ),
        (
            "f1 improved from 0.50 to 0.70; improved by at least 0.10",
            0.10,
        ),
        (
            "precision improved from 0.50 to 0.65; improved by at least 0.10",
            0.10,
        ),
        (
            "auc improved from 0.60 to 0.75; improved by at least 0.10",
            0.10,
        ),
        (
            "recall improved from 0.55 to 0.70; improved by at least 0.10",
            0.10,
        ),
    ],
)
def test_frozen_metric_e2e_titles_recover_explicit_continuation_threshold(
    text: str,
    expected_threshold: float,
) -> None:
    claim = recover_scientific_claim(_reference(text))

    assert claim is not None
    audit = compile_audit_claim(claim)
    assert audit.minimum_absolute_improvement == expected_threshold


@pytest.mark.parametrize(
    "text",
    [
        "Accuracy improved from 0.60 to 0.70.",
        "Accuracy improved from 0.60 to 0.70; improved substantially.",
        "Accuracy improved from 0.60 to 0.70; decreased by at least 0.10.",
        "Accuracy improved from 0.60 to 0.70; improved by at least 0.05 or 0.10.",
        "Accuracy improved from 0.60 to 0.70; loss improved by at least 0.10.",
    ],
)
def test_metric_compiler_does_not_invent_or_borrow_a_continuation_threshold(
    text: str,
) -> None:
    claim = recover_scientific_claim(_reference(text))

    if claim is not None:
        assert compile_audit_claim(claim).minimum_absolute_improvement is None


def test_metric_compiler_does_not_choose_an_ambiguous_continuation_threshold() -> None:
    claim = recover_scientific_claim(
        _reference(
            "Accuracy improved from 0.60 to 0.70; "
            "improved by at least 0.10 or 10%."
        )
    )

    assert claim is not None
    assert compile_audit_claim(claim).minimum_absolute_improvement is None


def test_relative_percentage_is_not_conflated_with_absolute_metric_delta() -> None:
    absolute = recover_scientific_claim(
        _reference(
            "Accuracy improved from 0.60 to 0.70; improved by at least 0.10."
        )
    )
    relative = recover_scientific_claim(
        _reference(
            "Accuracy improved from 60% to 70%; improved by at least 10%."
        )
    )
    percentage_points = recover_scientific_claim(
        _reference(
            "Accuracy improved from 60% to 70%; "
            "improved by at least 10 percentage points."
        )
    )

    assert absolute is not None
    assert absolute.primary.minimum_improvement is not None
    assert absolute.primary.minimum_improvement.unit is None
    assert compile_audit_claim(absolute).minimum_absolute_improvement == 0.10
    assert relative is not None
    assert relative.primary.minimum_improvement is not None
    assert relative.primary.minimum_improvement.unit == "%"
    assert percentage_points is not None
    assert percentage_points.primary.minimum_improvement is not None
    assert percentage_points.primary.minimum_improvement.unit == "percentage points"
    for claim in (relative, percentage_points):
        with pytest.raises(
            UnsupportedDeterministicClaimCompiler,
            match="^unsupported_deterministic_claim_compiler$",
        ):
            compile_audit_claim(claim)


def test_metric_compiler_does_not_borrow_unrelated_values_or_thresholds() -> None:
    claim = recover_scientific_claim(
        _reference(
            "Accuracy improved, while loss moved from 0.40 to 0.20 "
            "and remained at least 0.10."
        )
    )

    assert claim is not None
    audit = compile_audit_claim(claim)
    assert audit.metric == "accuracy"
    assert audit.claimed_values == ()
    assert audit.minimum_absolute_improvement is None


def test_metric_compiler_rejects_threshold_units_the_audit_cannot_represent() -> None:
    claim = recover_scientific_claim(
        _reference("Accuracy improved by at least 5 percentage points.")
    )

    assert claim is not None
    with pytest.raises(
        UnsupportedDeterministicClaimCompiler,
        match="^unsupported_deterministic_claim_compiler$",
    ):
        compile_audit_claim(claim)


@pytest.mark.parametrize(
    "text",
    [
        "Candidate accuracy is at least 0.90.",
        "The candidate generalizes across unseen domains.",
        "The candidate uses 40% less memory.",
    ],
)
def test_recognized_unsupported_primary_has_no_deterministic_compiler(
    text: str,
) -> None:
    claim = recover_scientific_claim(_reference(text))

    assert claim is not None
    with pytest.raises(
        UnsupportedDeterministicClaimCompiler,
        match="^unsupported_deterministic_claim_compiler$",
    ):
        compile_audit_claim(claim)


def test_provider_provenance_cannot_upgrade_absolute_bound_to_metric_improvement() -> None:
    claim = recover_scientific_claim(
        _reference(
            "Candidate accuracy is at least 0.90.",
            provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
        )
    )

    assert claim is not None
    assert type(claim.primary) is AbsoluteMetricClaim
    with pytest.raises(UnsupportedDeterministicClaimCompiler):
        compile_audit_claim(claim)


def test_forged_provider_metric_fields_cannot_override_source_recovery() -> None:
    recovered = recover_scientific_claim(
        _reference(
            "Accuracy improved by at least 0.05.",
            provenance_kind=ProvenanceKind.PROVIDER_PROPOSAL,
        )
    )
    assert recovered is not None
    assert type(recovered.primary) is MetricImprovementClaim
    forged = dataclasses.replace(
        recovered,
        primary=dataclasses.replace(
            recovered.primary,
            metric="f1",
            direction=Direction.LOWER,
        ),
    )

    with pytest.raises(ValueError, match="source|recover|canonical"):
        compile_audit_claim(forged)


def test_policy_seam_identifies_compiler_and_owned_obligation_templates() -> None:
    executable = recover_scientific_claim(
        _reference("Accuracy improved by at least 0.05 on held-out data.")
    )
    unsupported = recover_scientific_claim(
        _reference("Candidate accuracy is at least 0.90.")
    )
    assert executable is not None
    assert unsupported is not None

    executable_policy = claim_evidence_policy(executable)
    unsupported_policy = claim_evidence_policy(unsupported)

    assert type(executable_policy) is ClaimEvidencePolicy
    assert executable_policy.primary_kind is PrimaryClaimKind.METRIC_IMPROVEMENT
    assert executable_policy.constraint_kinds == (
        EvaluationConstraintKind.HELD_OUT,
    )
    assert executable_policy.deterministic_compiler_id == "metric-improvement-v0"
    assert unsupported_policy.primary_kind is PrimaryClaimKind.ABSOLUTE_METRIC
    assert unsupported_policy.deterministic_compiler_id is None
    assert executable_policy.obligation_template_ids[0:3] == (
        "claim.metric",
        "claim.direction",
        "claim.threshold",
    )
    assert executable_policy.obligation_template_ids[-1] == "comparison.readiness"
    assert unsupported_policy.obligation_template_ids == (
        "claim.metric",
        "claim.quantitative_bound",
    )
    assert not hasattr(executable_policy, "obligations")


def test_claim_contracts_are_immutable_json_safe_and_have_no_authority_fields() -> None:
    claim = recover_scientific_claim(
        _reference("Accuracy improved by at least 0.05 on held-out data.")
    )
    assert claim is not None

    with pytest.raises(FrozenInstanceError):
        claim.constraints = ()  # type: ignore[misc]

    public = to_jsonable(claim)
    assert public["primary"]["kind"] == "metric_improvement"
    assert public["constraints"][0]["kind"] == "held_out"
    assert claim_semantic_projection(claim)["constraints"] == ["held_out"]
    forbidden = {"verdict", "authority", "impact", "severity"}
    for contract in (
        MetricImprovementClaim,
        AbsoluteMetricClaim,
        GeneralizationClaim,
        GenericQuantitativeClaim,
        EvaluationConstraint,
        ClaimEvidencePolicy,
    ):
        assert forbidden.isdisjoint(field.name for field in dataclasses.fields(contract))
