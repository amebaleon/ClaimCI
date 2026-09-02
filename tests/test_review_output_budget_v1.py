"""Regression tests for production-scale research-review output budgets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
import pytest

from claimci.review.config import load_review_config
from claimci.review.models import (
    ProviderUsage,
    ReviewConfig,
    ReviewLimits,
    ReviewStatus,
    SourceBundle,
    SourceKind,
)
from claimci.review.openai_provider import OpenAIReviewerProvider
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.provider import ProviderResponse, StructuredRequest
from claimci.review.request_budget import (
    MAX_SYNTHESIS_AUDIT_CHARS,
    allocate_synthesis_inputs,
    bound_synthesis_audits,
    build_extraction_request_parts,
    logical_request_chars,
    output_budget_ready,
)
from claimci.review.sources import _record


def _request(task: str = "extract_claims") -> StructuredRequest:
    return StructuredRequest(
        task=task,
        payload={"sources": []},
        schema={"type": "object", "additionalProperties": False},
        max_output_tokens=5_000,
    )


def test_review_limits_default_to_material_claim_and_task_specific_budgets() -> None:
    limits = ReviewLimits()

    assert limits.max_claims == 16
    assert limits.extraction_max_output_tokens == 5_000
    assert limits.synthesis_max_output_tokens == 4_000
    assert limits.max_output_chars == 24_000


def test_request_policy_and_contracts_have_one_provider_free_authority() -> None:
    import claimci.review.orchestrator as orchestrator
    import claimci.review.provider as provider
    import claimci.review.request_budget as budget

    assert provider.REVIEW_SYSTEM_POLICY is budget.REVIEW_SYSTEM_POLICY
    assert not hasattr(orchestrator, "_EXTRACTION_CONTRACT")
    assert not hasattr(orchestrator, "_SYNTHESIS_CONTRACT")
    assert not hasattr(orchestrator, "_EXTRACTION_SCHEMA")
    assert not hasattr(orchestrator, "_SYNTHESIS_SCHEMA")


def test_synthesis_audit_allocation_keeps_only_complete_bounded_snapshots() -> None:
    small = {"manifest_path": "research.yaml", "findings": []}
    oversized = {
        "manifest_path": "large/research.yaml",
        "findings": [{"explanation": "x" * MAX_SYNTHESIS_AUDIT_CHARS}],
    }

    retained, omitted = bound_synthesis_audits((small, oversized))

    assert retained == (small,)
    assert omitted == 1


def test_exact_synthesis_allocator_packs_all_input_classes_and_counts_omissions() -> None:
    claim_ids = tuple(f"claim-{index:016x}" for index in range(16))
    claims = tuple(
        {
            "claim_id": claim_id,
            "source_text": "The benchmark improves accuracy.",
            "subject": f"candidate-{index}",
            "evidence_hints": [
                f"outside/{index:02}-{hint:02}.json" for hint in range(32)
            ],
        }
        for index, claim_id in enumerate(claim_ids)
    )
    evidence = (
        {
            "evidence_id": "evidence-table-one",
            "claim_ids": [claim_ids[0]],
            "path": "results.md",
            "excerpt": "first table",
        },
        {
            "evidence_id": "evidence-table-two",
            "claim_ids": [claim_ids[0]],
            "path": "results.md",
            "excerpt": "second table",
        },
    )
    owners = {
        claim_id: (
            ["evidence-table-one", "evidence-table-two"]
            if claim_id == claim_ids[0]
            else []
        )
        for claim_id in claim_ids
    }
    constraints = {
        claim_ids[0]: {
            "required_interpretation": "Use both distinct tables.",
            "required_citations": ["evidence-table-one", "evidence-table-two"],
        }
    }
    missing = tuple(
        {
            "claim_id": claim_id,
            "reason": "unresolved_provider_hint",
            "requested_path": f"outside/{claim_index:02}-{hint:02}.json",
            "description": "The exact provider hint was outside the issued scope.",
        }
        for claim_index, claim_id in enumerate(claim_ids)
        for hint in range(32)
    )
    audits = (
        {"manifest_path": "one/research.yaml", "findings": []},
        {
            "manifest_path": "two/research.yaml",
            "findings": [{"explanation": "x" * MAX_SYNTHESIS_AUDIT_CHARS}],
        },
    )

    allocation = allocate_synthesis_inputs(
        claims,
        evidence,
        owners,
        constraints,
        missing,
        audits,
        max_chars=60_000,
    )

    assert logical_request_chars(
        allocation.parts.task,
        allocation.parts.payload,
        allocation.parts.schema,
    ) <= 60_000
    assert [row["evidence_id"] for row in allocation.parts.payload["evidence"]] == [
        "evidence-table-one",
        "evidence-table-two",
    ]
    assert allocation.parts.payload["claims"] == list(claims)
    assert allocation.omitted_counts["missing_evidence"] == (
        len(missing) - len(allocation.parts.payload["missing_evidence"])
    )
    assert allocation.omitted_counts["missing_evidence"] > 0
    assert allocation.omitted_counts["deterministic_audits"] == 1


def test_preflight_output_reserve_requires_the_full_frozen_two_call_budget() -> None:
    assert output_budget_ready(ReviewLimits()) is True
    assert output_budget_ready(ReviewLimits(max_output_chars=23_999)) is False
    assert (
        output_budget_ready(ReviewLimits(extraction_max_output_tokens=4_999))
        is False
    )
    assert (
        output_budget_ready(ReviewLimits(synthesis_max_output_tokens=3_999))
        is False
    )
    with pytest.raises(ValueError, match="max_output_chars"):
        ReviewLimits(max_output_chars=24_001)


@pytest.mark.parametrize("max_claims", [15, 16])
def test_extraction_schema_accepts_every_valid_claim_cap_boundary(
    max_claims: int,
) -> None:
    parts = build_extraction_request_parts(SourceBundle(), max_claims)

    assert parts.schema["properties"]["claims"]["maxItems"] == max_claims


def test_review_limits_reject_seventeen_claims() -> None:
    with pytest.raises(ValueError, match="max_claims"):
        ReviewLimits(max_claims=17)


@pytest.mark.parametrize("target", [59_999, 60_000, 60_001])
def test_exact_extraction_logical_character_boundaries(target: int) -> None:
    empty_record = _record(SourceKind.PULL_REQUEST_DESCRIPTION, None, "")
    empty_bundle = SourceBundle(
        sources=(empty_record,),
        total_chars=0,
    )
    empty_parts = build_extraction_request_parts(empty_bundle, 16)
    fixed = logical_request_chars(
        empty_parts.task, empty_parts.payload, empty_parts.schema
    )
    text = "x" * (target - fixed)
    record = _record(SourceKind.PULL_REQUEST_DESCRIPTION, None, text)
    bundle = SourceBundle(sources=(record,), total_chars=len(text))
    parts = build_extraction_request_parts(bundle, 16)

    assert logical_request_chars(parts.task, parts.payload, parts.schema) == target


def test_enabled_config_accepts_new_budgets_and_legacy_config_remains_readable(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".claimci"
    config_dir.mkdir()
    config_path = config_dir / "review.yaml"
    common = {
        "max_calls": 2,
        "max_context_chars": 60_000,
        "max_output_chars": 24_000,
        "max_files": 24,
        "max_file_chars": 16_000,
        "timeout_seconds": 30,
    }
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "enabled": True,
                "policy": "advisory",
                "provider": "openai",
                "model": "gpt-5.6-terra",
                "limits": {
                    **common,
                    "max_claims": 16,
                    "extraction_max_output_tokens": 5_000,
                    "synthesis_max_output_tokens": 4_000,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_review_config(tmp_path)

    assert loaded.limits == ReviewLimits()

    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "enabled": True,
                "policy": "advisory",
                "provider": "openai",
                "model": "gpt-5.6-terra",
                "limits": {
                    **{**common, "max_output_chars": 12_000},
                    "max_output_tokens_per_call": 2_000,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    legacy = load_review_config(tmp_path)

    assert legacy.limits.max_claims == 16
    assert legacy.limits.extraction_max_output_tokens == 2_000
    assert legacy.limits.synthesis_max_output_tokens == 2_000
    assert legacy.limits.max_output_chars == 12_000


class _IncompleteResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="response-incomplete",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output_text='{"claims":',
            usage=SimpleNamespace(
                input_tokens=6_998,
                output_tokens=2_000,
                total_tokens=8_998,
            ),
        )


def test_openai_adapter_preserves_incomplete_max_output_tokens_status() -> None:
    client = SimpleNamespace(responses=_IncompleteResponses())
    provider = OpenAIReviewerProvider(
        model="gpt-5.6-terra",
        timeout_seconds=30,
        max_output_tokens=5_000,
        client=client,
    )

    response = provider.extract_claims(_request())

    assert response.complete is False
    assert response.incomplete_reason == "max_output_tokens"
    assert response.output_text == '{"claims":'
    assert response.usage.total_tokens == 8_998


class _RecordingProvider:
    def __init__(self, *, incomplete_task: str | None = None) -> None:
        self.incomplete_task = incomplete_task
        self.calls: list[StructuredRequest] = []

    def _response(self, request: StructuredRequest, output: str) -> ProviderResponse:
        self.calls.append(request)
        incomplete = request.task == self.incomplete_task
        return ProviderResponse(
            output_text='{"claims":' if incomplete else output,
            provider="fake",
            model="fake-model",
            request_id=f"request-{len(self.calls)}",
            usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            complete=not incomplete,
            incomplete_reason="max_output_tokens" if incomplete else None,
        )

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        return self._response(request, json.dumps({"claims": []}))

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        return self._response(request, json.dumps({"interpretations": []}))


def _run(tmp_path: Path, provider: _RecordingProvider):
    (tmp_path / "README.md").write_text(
        "Bounded review context.\n", encoding="utf-8"
    )
    return run_review(
        ReviewInputs(
            repository_root=tmp_path,
            pr_title="The rollup reads 4.0x fewer rows.",
            pr_description="The benchmark reports zero correctness mismatches.",
        ),
        ReviewConfig(enabled=True, limits=ReviewLimits()),
        provider=provider,
    )


def test_orchestrator_uses_material_claim_cap_and_task_specific_output_budgets(
    tmp_path: Path,
) -> None:
    provider = _RecordingProvider()

    result = _run(tmp_path, provider)

    assert result.status is ReviewStatus.COMPLETE
    assert [request.max_output_tokens for request in provider.calls] == [5_000, 4_000]
    extraction = provider.calls[0]
    assert extraction.schema["properties"]["claims"]["maxItems"] == 16
    assert extraction.payload["max_claims"] == 16
    prioritization = extraction.payload["extraction_contract"]["claim_prioritization"]
    assert "correctness" in prioritization
    assert "benchmark" in prioritization
    assert "experimental fairness" in prioritization
    assert "production" in prioritization
    assert "causal" in prioritization


def test_incomplete_extraction_returns_explicit_partial_without_parsing_truncated_json(
    tmp_path: Path,
) -> None:
    provider = _RecordingProvider(incomplete_task="extract_claims")

    result = _run(tmp_path, provider)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "CLAIM_EXTRACTION_TRUNCATED"
    assert [request.task for request in provider.calls] == ["extract_claims"]
    assert len(result.provider_calls) == 1


def test_incomplete_synthesis_preserves_extracted_state_as_explicit_partial(
    tmp_path: Path,
) -> None:
    provider = _RecordingProvider(incomplete_task="synthesize_review")

    result = _run(tmp_path, provider)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_TRUNCATED"
    assert [request.task for request in provider.calls] == [
        "extract_claims",
        "synthesize_review",
    ]
    assert len(result.provider_calls) == 2
