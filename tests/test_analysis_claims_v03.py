"""Deterministic claim-to-test contracts for ClaimCI v0.3."""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from claimci.analysis import (
    AnalysisContractError,
    AuditClaimSpec,
    ClaimedMetricValue,
    ExperimentRole,
    FieldProvenance,
    ProvenanceKind,
    audit_claim_spec_from_research_spec,
    audit_relevant_claim_projection,
    to_jsonable,
)
from claimci.models import Direction, ExperimentPaths, ResearchSpec


def _provenance(
    kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_DISCOVERY,
) -> FieldProvenance:
    return FieldProvenance(
        kind=kind,
        detail="validated claim field",
        source_id="claim-source",
    )


def _spec(**overrides: object) -> AuditClaimSpec:
    provenance = _provenance()
    values: dict[str, object] = {
        "claim_id": "claim-1",
        "metric": "accuracy",
        "direction": Direction.HIGHER,
        "minimum_absolute_improvement": 0.05,
        "metric_provenance": provenance,
        "direction_provenance": provenance,
        "threshold_provenance": provenance,
        "claimed_values": (),
    }
    values.update(overrides)
    return AuditClaimSpec(**values)  # type: ignore[arg-type]


def test_claimed_metric_values_are_descriptive_and_do_not_change_audit_projection() -> None:
    provenance = _provenance(ProvenanceKind.PROVIDER_PROPOSAL)
    first = _spec(
        claimed_values=(
            ClaimedMetricValue(
                role=ExperimentRole.BASELINE,
                value=0.71,
                unit="%",
                provenance=provenance,
            ),
            ClaimedMetricValue(
                role=ExperimentRole.CANDIDATE,
                value=0.79,
                unit="%",
                provenance=provenance,
            ),
        ),
    )
    second = dataclasses.replace(first, claimed_values=())

    assert audit_relevant_claim_projection(first) == {
        "claim_id": "claim-1",
        "metric": "accuracy",
        "direction": "higher",
        "minimum_absolute_improvement": 0.05,
    }
    assert audit_relevant_claim_projection(first) == audit_relevant_claim_projection(
        second
    )
    assert "claimed_values" not in audit_relevant_claim_projection(first)


def test_claimed_metric_value_is_frozen_json_safe_and_may_omit_raw_token() -> None:
    value = ClaimedMetricValue(
        role=ExperimentRole.BASELINE,
        value=71,
        unit="%",
        provenance=_provenance(),
    )

    assert value.value == 71.0
    assert value.raw is None
    assert to_jsonable(value) == {
        "role": "baseline",
        "value": 71.0,
        "unit": "%",
        "provenance": {
            "kind": "deterministic_discovery",
            "detail": "validated claim field",
            "source_path": None,
            "source_id": "claim-source",
        },
        "raw": None,
    }
    with pytest.raises(FrozenInstanceError):
        value.value = 72  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"role": ExperimentRole.UNSPECIFIED},
        {"role": ExperimentRole.REFERENCE},
        {"value": float("nan")},
        {"value": float("inf")},
        {"unit": ""},
        {"raw": ""},
    ],
)
def test_claimed_metric_value_rejects_non_descriptive_roles_and_malformed_values(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "role": ExperimentRole.BASELINE,
        "value": 0.71,
        "unit": None,
        "provenance": _provenance(),
        "raw": None,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        ClaimedMetricValue(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "threshold,provenance",
    [(None, _provenance()), (0.05, None)],
)
def test_audit_claim_spec_requires_threshold_and_provenance_together(
    threshold: float | None,
    provenance: FieldProvenance | None,
) -> None:
    with pytest.raises(AnalysisContractError):
        _spec(
            minimum_absolute_improvement=threshold,
            threshold_provenance=provenance,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"claim_id": ""},
        {"metric": ""},
        {"metric": "accuracy\nloss"},
        {"direction": "higher"},
        {"minimum_absolute_improvement": -0.01},
        {"minimum_absolute_improvement": float("nan")},
        {"claimed_values": []},
    ],
)
def test_audit_claim_spec_rejects_malformed_fields(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _spec(**changes)


def test_audit_claim_spec_contains_no_authority_or_interpretation_fields() -> None:
    assert {field.name for field in dataclasses.fields(AuditClaimSpec)} == {
        "claim_id",
        "metric",
        "direction",
        "minimum_absolute_improvement",
        "metric_provenance",
        "direction_provenance",
        "threshold_provenance",
        "claimed_values",
    }


def test_native_research_spec_converts_to_manifest_hint_claim_without_changing_values() -> None:
    paths = ExperimentPaths(
        config=Path("baseline/config.yaml"),
        results=Path("baseline/results.json"),
        train_dataset=Path("baseline/train.jsonl"),
        eval_dataset=Path("baseline/eval.jsonl"),
    )
    native = ResearchSpec(
        manifest_path=Path("study/research.yaml"),
        metric="loss",
        minimum_improvement=0.125,
        baseline=paths,
        candidate=paths,
        direction=Direction.LOWER,
    )

    converted = audit_claim_spec_from_research_spec(native)

    assert converted.claim_id == "manifest:study/research.yaml"
    assert converted.metric == "loss"
    assert converted.direction is Direction.LOWER
    assert converted.minimum_absolute_improvement == 0.125
    assert converted.claimed_values == ()
    assert converted.metric_provenance.kind is ProvenanceKind.MANIFEST_HINT
    assert converted.direction_provenance.kind is ProvenanceKind.MANIFEST_HINT
    assert converted.threshold_provenance is not None
    assert converted.threshold_provenance.kind is ProvenanceKind.MANIFEST_HINT

