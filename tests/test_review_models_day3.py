"""Wave 1 contract tests for the LLM-safe research-review model layer."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import math
from decimal import Decimal
from pathlib import Path

import pytest
import yaml


models = importlib.import_module("claimci.review.models")
config = importlib.import_module("claimci.review.config")


def test_supported_claim_type_vocabulary_is_complete() -> None:
    assert {member.name: member.value for member in models.ClaimType} == {
        "METRIC_IMPROVEMENT": "metric_improvement",
        "COMPUTE_EQUIVALENCE": "compute_equivalence",
        "HELD_OUT_EVALUATION": "held_out_evaluation",
        "RESOURCE_REDUCTION": "resource_reduction",
        "COMPONENT_CAUSALITY": "component_causality",
        "NO_EXTERNAL_REWARD": "no_external_reward",
        "IMPLEMENTATION_CLAIM": "implementation_claim",
        "OTHER_SCIENTIFIC": "other_scientific",
    }


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

    if dataclasses.is_dataclass(limits) and getattr(limits, "__dataclass_params__").frozen:
        with pytest.raises(dataclasses.FrozenInstanceError):
            limits.max_calls = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_calls", 3),
        ("max_calls", 2.1),
        ("max_calls", 0),
        ("max_context_chars", 60_001),
        ("max_context_chars", 60_000.5),
        ("max_context_chars", 0),
        ("max_output_chars", 0),
        ("max_files", 0),
        ("max_file_chars", 0),
        ("max_output_tokens_per_call", 0),
        ("timeout_seconds", 0),
        ("timeout_seconds", 10**400),
    ],
)
def test_review_limits_reject_unsafe_budget_values(field: str, value: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        models.ReviewLimits(**{field: value})


def test_review_limits_reject_bool_as_numeric_budget() -> None:
    with pytest.raises((TypeError, ValueError)):
        models.ReviewLimits(max_calls=True)


def _write_review_config(root: Path, payload: object) -> Path:
    path = root / ".claimci" / "review.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_missing_review_config_is_strictly_default_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key-must-not-enable-review")
    loaded = config.load_review_config(tmp_path)
    assert loaded.enabled is False
    assert loaded.policy == "advisory"
    assert loaded.provider == "openai"
    assert loaded.model == "gpt-5.6-terra"
    assert loaded.limits == models.ReviewLimits()


def test_enabled_review_config_requires_the_complete_frozen_shape(tmp_path: Path) -> None:
    path = _write_review_config(
        tmp_path,
        {
            "schema_version": 1,
            "enabled": True,
            "policy": "advisory",
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "limits": {
                "max_calls": 2,
                "max_context_chars": 60_000,
                "max_output_chars": 12_000,
                "max_files": 24,
                "max_file_chars": 16_000,
                "max_output_tokens_per_call": 2_000,
                "timeout_seconds": 30,
            },
        },
    )
    loaded = config.load_review_config(tmp_path, config_path=path)
    assert loaded == models.ReviewConfig(
        schema_version=1,
        enabled=True,
        policy="advisory",
        provider="openai",
        model="gpt-5.6-terra",
        limits=models.ReviewLimits(),
    )


def test_explicitly_disabled_review_config_never_opts_in(tmp_path: Path) -> None:
    _write_review_config(
        tmp_path,
        {
            "schema_version": 1,
            "enabled": False,
            "policy": "advisory",
            "provider": "openai",
            "model": "gpt-5.6-terra",
        },
    )
    loaded = config.load_review_config(tmp_path)
    assert loaded.enabled is False
    assert loaded.limits == models.ReviewLimits()
    if dataclasses.is_dataclass(loaded) and getattr(loaded, "__dataclass_params__").frozen:
        with pytest.raises(dataclasses.FrozenInstanceError):
            loaded.enabled = True  # type: ignore[misc]


def test_review_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / ".claimci" / "review.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("schema_version: 1\nenabled: true\nenabled: false\n", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        config.load_review_config(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "enabled": True, "policy": "strict"},
        {"schema_version": 1, "enabled": "true"},
        {"schema_version": 1, "enabled": True, "unknown": 1},
        {"schema_version": 1, "enabled": True, "limits": {"unknown": 1}},
        {"schema_version": 2, "enabled": True},
        {"schema_version": 1.0, "enabled": False},
        {"schema_version": 1, "enabled": True, "limits": {"max_calls": 3}},
        {"schema_version": 1, "enabled": True, "limits": {"max_context_chars": 60_001}},
        [],
        "enabled: true",
    ],
)
def test_review_config_rejects_unknown_fields_invalid_values_and_nonmappings(
    tmp_path: Path, payload: object,
) -> None:
    if isinstance(payload, dict):
        _write_review_config(tmp_path, payload)
    else:
        path = tmp_path / ".claimci" / "review.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("- enabled\n" if isinstance(payload, list) else payload, encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        config.load_review_config(tmp_path)


def test_environment_model_override_cannot_hide_invalid_yaml_model_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAIMCI_OPENAI_MODEL", "trusted-runtime-model")
    _write_review_config(
        tmp_path,
        {
            "schema_version": 1,
            "enabled": True,
            "policy": "advisory",
            "provider": "openai",
            "model": [],
            "limits": {
                "max_calls": 2,
                "max_context_chars": 60_000,
                "max_output_chars": 12_000,
                "max_files": 24,
                "max_file_chars": 16_000,
                "max_output_tokens_per_call": 2_000,
                "timeout_seconds": 30,
            },
        },
    )

    with pytest.raises((TypeError, ValueError)):
        config.load_review_config(tmp_path)


def test_source_location_is_relative_and_line_bounded() -> None:
    assert {member.name: member.value for member in models.SourceKind} == {
        "PULL_REQUEST_TITLE": "pull_request_title",
        "PULL_REQUEST_DESCRIPTION": "pull_request_description",
        "REPOSITORY_FILE": "repository_file",
    }
    location = models.SourceLocation(
        source_id="src-paper",
        kind=models.SourceKind.REPOSITORY_FILE,
        path="docs/paper.md",
        start_line=2,
        end_line=4,
    )
    assert location.path == "docs/paper.md"
    assert location.start_line == 2
    assert location.end_line == 4
    with pytest.raises((TypeError, ValueError)):
        models.SourceLocation("src-paper", models.SourceKind.REPOSITORY_FILE, "../outside.md", 1, 1)
    with pytest.raises((TypeError, ValueError)):
        models.SourceLocation("src-paper", models.SourceKind.REPOSITORY_FILE, "docs/paper.md", 0, 1)
    with pytest.raises((TypeError, ValueError)):
        models.SourceLocation("src-paper", models.SourceKind.REPOSITORY_FILE, "docs/paper.md", 3, 2)
    with pytest.raises((TypeError, ValueError)):
        models.SourceLocation("", models.SourceKind.REPOSITORY_FILE, "docs/paper.md", 1, 1)
    for unsafe_path in ("C:/outside.md", "docs/\x00paper.md"):
        with pytest.raises((TypeError, ValueError)):
            models.SourceLocation("src-paper", models.SourceKind.REPOSITORY_FILE, unsafe_path, 1, 1)


@pytest.mark.parametrize("claim_type", list(models.ClaimType))
def test_scientific_claim_exposes_required_llm_safe_fields(claim_type) -> None:
    source = models.SourceLocation(
        source_id="src-title",
        kind=models.SourceKind.PULL_REQUEST_TITLE,
        path=None,
        start_line=1,
        end_line=1,
    )
    claim = models.ScientificClaim(
        claim_id="claim-0001",
        source_text="The candidate improves accuracy.",
        claim_type=claim_type,
        subject="candidate model",
        source=source,
        metric="accuracy" if claim_type is models.ClaimType.METRIC_IMPROVEMENT else None,
        direction=(
            models.ClaimDirection.HIGHER
            if claim_type is models.ClaimType.METRIC_IMPROVEMENT
            else models.ClaimDirection.NOT_APPLICABLE
        ),
        qualifiers=("on the held-out split",),
        confidence=0.75,
        evidence_hints=(),
    )
    assert claim.claim_id == "claim-0001"
    assert claim.source_text.startswith("The candidate")
    assert claim.source is source
    assert 0 <= claim.confidence <= 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("claim_id", ""),
        ("source_text", ""),
        ("subject", ""),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("confidence", math.nan),
    ],
)
def test_scientific_claim_rejects_missing_or_invalid_required_fields(field, value) -> None:
    source = models.SourceLocation("src-title", models.SourceKind.PULL_REQUEST_TITLE, None, 1, 1)
    kwargs = {
        "claim_id": "claim-0001",
        "source_text": "A claim",
        "claim_type": models.ClaimType.OTHER_SCIENTIFIC,
        "subject": "subject",
        "source": source,
        "confidence": 0.5,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        models.ScientificClaim(**kwargs)


def test_magnitude_rejects_nonfinite_values_and_invalid_kinds() -> None:
    unspecified = models.ClaimMagnitude(raw="not quantified")
    assert unspecified.kind is models.MagnitudeKind.UNSPECIFIED
    assert unspecified.value is None
    with pytest.raises((TypeError, ValueError)):
        models.ClaimMagnitude(raw="inf%", value=math.inf, unit="%", kind=models.MagnitudeKind.RELATIVE)
    with pytest.raises((TypeError, ValueError)):
        models.ClaimMagnitude(raw="10%", value=10.0, unit="%", kind="ratio")


def test_review_models_do_not_import_deterministic_authority_types() -> None:
    module_path = Path(models.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_modules = {"claimci.models", "claimci.audit", "claimci.report", "claimci.github"}
    forbidden_names = {"Verdict", "Finding", "Severity", "Impact", "AuditResult"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
            assert not ({alias.name for alias in node.names} & forbidden_names)
        elif isinstance(node, ast.Import):
            assert not ({alias.name for alias in node.names} & forbidden_modules)
        elif isinstance(node, ast.Call):
            called = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            assert called not in forbidden_names
    for name, value in vars(models).items():
        if isinstance(value, type) and value.__module__ == models.__name__:
            annotations = getattr(value, "__annotations__", {})
            assert not (set(annotations) & forbidden_names), name
            assert not any(forbidden in repr(annotations) for forbidden in forbidden_names), name


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


def test_pathological_token_counts_normalize_to_unavailable_observability() -> None:
    huge = 10**5_000

    usage = models.ProviderUsage(
        input_tokens=huge,
        output_tokens=huge,
        total_tokens=huge,
    )

    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
