"""Wave 1 contract tests for the LLM-safe review model layer.

These tests deliberately exercise the review-only value objects and their
trusted validation boundary.  The deterministic ClaimCI models remain a
separate authority and must not leak into ``claimci.review.models``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import math
from pathlib import Path
from decimal import Decimal

import pytest


models = importlib.import_module("claimci.review.models")


def test_supported_claim_type_vocabulary_is_complete() -> None:
    expected = {
        "METRIC_IMPROVEMENT": "metric_improvement",
        "COMPUTE_EQUIVALENCE": "compute_equivalence",
        "HELD_OUT_EVALUATION": "held_out_evaluation",
        "RESOURCE_REDUCTION": "resource_reduction",
        "COMPONENT_CAUSALITY": "component_causality",
        "NO_EXTERNAL_REWARD": "no_external_reward",
        "IMPLEMENTATION_CLAIM": "implementation_claim",
        "OTHER_SCIENTIFIC": "other_scientific",
    }

    claim_type = models.ClaimType
    assert {member.name: member.value for member in claim_type} == expected


def test_review_direction_and_magnitude_enums_are_restricted() -> None:
    assert {member.name: member.value for member in models.ClaimDirection} == {
        "HIGHER": "higher",
        "LOWER": "lower",
        "NOT_APPLICABLE": "not_applicable",
    }
    assert {member.name: member.value for member in models.MagnitudeKind} == {
        "ABSOLUTE": "absolute",
        "RELATIVE": "relative",
        "UNSPECIFIED": "unspecified",
    }


def test_review_limits_default_to_the_frozen_safe_budget() -> None:
    limits = models.ReviewLimits()

    assert limits.max_calls == 2
    assert limits.max_context_chars == 60_000
    assert limits.max_output_chars == 12_000
    assert limits.max_files == 24
    assert limits.max_file_chars == 16_000
    assert limits.max_output_tokens_per_call == 2_000
    assert limits.timeout_seconds == 30
    assert getattr(limits, "retries", 0) == 0

    # Limits are trusted configuration, not mutable request state.
    if dataclasses.is_dataclass(limits) and getattr(limits, "__dataclass_params__").frozen:
        with pytest.raises(dataclasses.FrozenInstanceError):
            limits.max_calls = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_calls", 3),
        ("max_calls", 0),
        ("max_context_chars", 60_001),
        ("max_context_chars", 0),
        ("max_output_chars", 0),
        ("max_files", 0),
        ("max_file_chars", 0),
        ("max_output_tokens_per_call", 0),
        ("timeout_seconds", 0),
    ],
)
def test_review_limits_reject_unsafe_budget_values(field: str, value: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        models.ReviewLimits(**{field: value})


def test_review_limits_reject_bool_as_numeric_budget() -> None:
    with pytest.raises((TypeError, ValueError)):
        models.ReviewLimits(max_calls=True)


def test_source_location_is_relative_and_line_bounded() -> None:
    source_kind = models.SourceKind
    # The enum names are part of the source contract; the exact wire values
    # are asserted by source-collection tests through the returned records.
    assert hasattr(source_kind, "PULL_REQUEST_TITLE")
    assert hasattr(source_kind, "PULL_REQUEST_DESCRIPTION")
    assert hasattr(source_kind, "REPOSITORY_FILE")

    location = models.SourceLocation(
        source_id="src-paper",
        kind=source_kind.REPOSITORY_FILE,
        path="docs/paper.md",
        start_line=2,
        end_line=4,
    )
    assert location.path == "docs/paper.md"
    assert location.start_line == 2
    assert location.end_line == 4

    with pytest.raises((TypeError, ValueError)):
        models.SourceLocation(
            source_id="src-paper",
            kind=source_kind.REPOSITORY_FILE,
            path="../outside.md",
            start_line=1,
            end_line=1,
        )
    with pytest.raises((TypeError, ValueError)):
        models.SourceLocation(
            source_id="src-paper",
            kind=source_kind.REPOSITORY_FILE,
            path="docs/paper.md",
            start_line=0,
            end_line=1,
        )
    with pytest.raises((TypeError, ValueError)):
        models.SourceLocation(
            source_id="src-paper",
            kind=source_kind.REPOSITORY_FILE,
            path="docs/paper.md",
            start_line=3,
            end_line=2,
        )


def test_magnitude_rejects_nonfinite_values_and_invalid_kinds() -> None:
    with pytest.raises((TypeError, ValueError)):
        models.ClaimMagnitude(
            raw="inf%",
            value=math.inf,
            unit="%",
            kind=models.MagnitudeKind.RELATIVE,
        )
    with pytest.raises((TypeError, ValueError)):
        models.ClaimMagnitude(
            raw="10%",
            value=10.0,
            unit="%",
            kind="ratio",
        )


def test_review_models_do_not_import_deterministic_authority_types() -> None:
    module_path = Path(models.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_modules = {
        "claimci.models",
        "claimci.audit",
        "claimci.report",
        "claimci.github",
    }
    forbidden_names = {"Verdict", "Finding", "Severity", "Impact", "AuditResult"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
            assert not ({alias.name for alias in node.names} & forbidden_names)
        elif isinstance(node, ast.Import):
            assert not ({alias.name for alias in node.names} & forbidden_modules)

    # Keep the assertion about accidental authority fields useful even if the
    # implementation uses postponed annotations.
    for name, value in vars(models).items():
        if isinstance(value, type) and value.__module__ == models.__name__:
            annotations = getattr(value, "__annotations__", {})
            assert not (set(annotations) & forbidden_names), name
            assert not any(
                forbidden in repr(annotations)
                for forbidden in forbidden_names
            ), name


def test_usage_models_are_observability_only_and_frozen() -> None:
    usage = models.ProviderUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        estimated_cost_usd=Decimal("0.001"),
    )
    assert usage.total_tokens == 15
    assert usage.estimated_cost_usd == Decimal("0.001")
    with pytest.raises(dataclasses.FrozenInstanceError):
        usage.total_tokens = 99  # type: ignore[misc]

    record = models.ProviderCallRecord(
        task="extract_claims",
        provider="fake",
        model="test-model",
        request_id="req-1",
        usage=usage,
        input_chars=100,
        output_chars=50,
    )
    assert record.task == "extract_claims"
    assert record.usage is usage
