"""Regression tests for claim-heavy provider request timeouts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from claimci.review.models import ReviewLimits
from claimci.review.openai_provider import OpenAIReviewerProvider
from claimci.review.provider import StructuredRequest


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="response-complete",
            status="completed",
            output_text='{"claims": []}',
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )


def test_review_limits_allow_a_90_second_claim_heavy_request() -> None:
    limits = ReviewLimits(timeout_seconds=90)

    assert limits.timeout_seconds == 90


def test_openai_adapter_passes_a_90_second_timeout_to_the_request() -> None:
    responses = _Responses()
    provider = OpenAIReviewerProvider(
        model="gpt-5.6-terra",
        timeout_seconds=90,
        max_output_tokens=5_000,
        client=SimpleNamespace(responses=responses),
    )
    request = StructuredRequest(
        task="extract_claims",
        payload={"sources": []},
        schema={"type": "object", "additionalProperties": False},
        max_output_tokens=5_000,
    )

    response = provider.extract_claims(request)

    assert response.complete is True
    assert responses.calls[0]["timeout"] == 90
