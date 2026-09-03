"""Regression tests for fail-closed partial synthesis handling."""

from __future__ import annotations

import json
from pathlib import Path

from claimci.review.models import ProviderUsage, ReviewConfig, ReviewLimits, ReviewStatus
from claimci.review.orchestrator import ReviewInputs, run_review
from claimci.review.provider import ProviderResponse, StructuredRequest


class _InvalidSynthesisProvider:
    def __init__(self) -> None:
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
        return self._response(
            request.task,
            {
                "claims": [
                    {
                        "source_text": source["text"],
                        "claim_type": "resource_reduction",
                        "subject": "rollup benchmark row reduction",
                        "metric": "rows read",
                        "direction": "lower",
                        "claimed_magnitude": {
                            "raw": "4.0x fewer rows",
                            "value": 4.0,
                            "unit": "x",
                            "kind": "relative",
                        },
                        "qualifiers": ["90-day benchmark"],
                        "source": {
                            "source_id": source["source_id"],
                            "start_line": 1,
                            "end_line": 1,
                        },
                        "confidence": 0.98,
                        "evidence_hints": [],
                    }
                ]
            },
        )

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        self.calls.append(request)
        return self._response(
            request.task,
            {
                "interpretations": [
                    {
                        "claim_id": "claim-not-issued",
                        "interpretation": "This row must not be accepted.",
                        "citations": ["evidence-not-issued"],
                        "missing_evidence": [],
                        "unsupported_inferences": [],
                        "confidence": 0.5,
                    }
                ]
            },
        )


def test_invalid_synthesis_is_partial_and_preserves_trusted_extraction(
    tmp_path: Path,
) -> None:
    document = tmp_path / "docs" / "review.md"
    document.parent.mkdir(parents=True)
    document.write_text(
        "Bounded review context.\n", encoding="utf-8"
    )
    provider = _InvalidSynthesisProvider()

    result = run_review(
        ReviewInputs(
            repository_root=tmp_path,
            pr_description=(
                "The rollup reads 4.0x fewer rows for the 90-day benchmark; "
                "see docs/review.md."
            ),
        ),
        ReviewConfig(enabled=True, limits=ReviewLimits()),
        provider=provider,
    )

    assert result.status is ReviewStatus.PARTIAL
    assert result.error_code == "SYNTHESIS_INVALID"
    assert len(result.claims) == 1
    assert result.interpretations == ()
    assert len(result.provider_calls) == 2
