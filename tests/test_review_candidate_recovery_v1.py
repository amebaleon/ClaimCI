"""Regression tests for semantic claim-extraction recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claimci.review.models import ProviderUsage, ReviewConfig, ReviewLimits, ReviewStatus
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.provider import ProviderResponse, StructuredRequest
from claimci.review.sources import (
    collect_review_sources,
    validate_claim_candidates_best_effort,
)


def _candidate(
    source_id: str,
    source_text: str,
    *,
    start_line: int = 1,
    end_line: int = 1,
    subject: str = "rollup row reduction",
) -> dict[str, Any]:
    return {
        "source_text": source_text,
        "claim_type": "resource_reduction",
        "subject": subject,
        "metric": "rows read",
        "direction": "lower",
        "claimed_magnitude": {
            "raw": "4.0x fewer rows",
            "value": 4.0,
            "unit": "x",
            "kind": "relative",
        },
        "qualifiers": ["90-day local cohort"],
        "source": {
            "source_id": source_id,
            "start_line": start_line,
            "end_line": end_line,
        },
        "confidence": 0.98,
        "evidence_hints": [],
    }


def test_best_effort_validation_recovers_a_unique_exact_quote_location(
    tmp_path: Path,
) -> None:
    bundle = collect_review_sources(
        tmp_path,
        pr_description=(
            "Introductory line.\n"
            "The rollup reads 4.0x fewer rows for the 90-day benchmark."
        ),
    )
    source = bundle.sources[0]
    payload = {
        "claims": [
            _candidate(
                source.source_id,
                "The rollup reads 4.0x fewer rows for the 90-day benchmark.",
                start_line=1,
                end_line=1,
            )
        ]
    }

    validation = validate_claim_candidates_best_effort(payload, bundle, max_claims=16)

    assert validation.rejected_count == 0
    assert len(validation.claims) == 1
    assert validation.claims[0].source.start_line == 2
    assert validation.claims[0].source.end_line == 2


def test_best_effort_validation_rejects_only_the_bad_candidate(
    tmp_path: Path,
) -> None:
    text = "The rollup reads 4.0x fewer rows for the 90-day benchmark."
    bundle = collect_review_sources(tmp_path, pr_description=text)
    source = bundle.sources[0]
    payload = {
        "claims": [
            _candidate(source.source_id, text),
            _candidate(
                source.source_id,
                "This paraphrase does not occur in the issued source.",
                subject="invalid paraphrase",
            ),
        ]
    }

    validation = validate_claim_candidates_best_effort(payload, bundle, max_claims=16)

    assert len(validation.claims) == 1
    assert validation.rejected_count == 1
    assert validation.rejection_reasons == ("source_quote_mismatch",)


class _MixedExtractionProvider:
    def __init__(
        self,
        *,
        all_invalid: bool = False,
        invalid_synthesis: bool = False,
    ) -> None:
        self.all_invalid = all_invalid
        self.invalid_synthesis = invalid_synthesis
        self.calls: list[StructuredRequest] = []

    @staticmethod
    def _response(task: str, payload: object) -> ProviderResponse:
        return ProviderResponse(
            output_text=json.dumps(payload),
            provider="fake",
            model="fake-model",
            request_id=f"request-{task}",
            usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        self.calls.append(request)
        source = request.payload["sources"][0]
        source_id = source["source_id"]
        text = source["text"]
        invalid = _candidate(
            source_id,
            "This paraphrase does not occur in the issued source.",
            subject="invalid paraphrase",
        )
        claims = [invalid] if self.all_invalid else [_candidate(source_id, text), invalid]
        return self._response(request.task, {"claims": claims})

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        self.calls.append(request)
        claims = request.payload["claims"]
        if self.invalid_synthesis:
            interpretations = [
                {
                    "claim_id": "unknown-claim",
                    "interpretation": "Invalid claim identity.",
                    "citations": [],
                    "missing_evidence": ["Executed result artifact."],
                    "unsupported_inferences": [],
                    "confidence": 0.5,
                }
            ]
        else:
            interpretations = [
                {
                    "claim_id": claim["claim_id"],
                    "interpretation": "The claim needs an executed result artifact.",
                    "citations": [],
                    "missing_evidence": ["Executed result artifact."],
                    "unsupported_inferences": [],
                    "confidence": 0.8,
                }
                for claim in claims
            ]
        return self._response(request.task, {"interpretations": interpretations})


def _run(tmp_path: Path, provider: _MixedExtractionProvider):
    (tmp_path / "README.md").write_text(
        "Bounded review context.\n", encoding="utf-8"
    )
    return run_review(
        ReviewInputs(
            repository_root=tmp_path,
            pr_description="The rollup reads 4.0x fewer rows for the 90-day benchmark.",
        ),
        ReviewConfig(enabled=True, limits=ReviewLimits()),
        provider=provider,
    )


def test_orchestrator_reviews_valid_claims_and_reports_rejected_candidates(
    tmp_path: Path,
) -> None:
    provider = _MixedExtractionProvider()

    result = _run(tmp_path, provider)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "CLAIM_CANDIDATES_REJECTED"
    assert len(result.claims) == 1
    assert len(result.interpretations) == 1
    assert len(result.provider_calls) == 2


def test_all_invalid_candidates_remain_unavailable_and_skip_synthesis(
    tmp_path: Path,
) -> None:
    provider = _MixedExtractionProvider(all_invalid=True)

    result = _run(tmp_path, provider)

    assert result.status is ReviewStatus.UNAVAILABLE
    assert result.error_code == "REVIEW_UNAVAILABLE"
    assert result.claims == ()
    assert [request.task for request in provider.calls] == ["extract_claims"]


def test_invalid_synthesis_is_partial_and_fail_closed(
    tmp_path: Path,
) -> None:
    provider = _MixedExtractionProvider(invalid_synthesis=True)

    result = _run(tmp_path, provider)

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_INVALID"
    assert len(result.claims) == 1
    assert result.interpretations == ()
    assert len(result.provider_calls) == 2
