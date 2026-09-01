"""Day 3 provider-boundary regression tests.

The provider is deliberately a very small seam: trusted orchestration owns
validation and evidence discovery, while an adapter only performs the two
structured operations.  These tests use an injected client and never import
or contact a real network service.
"""

from __future__ import annotations

import inspect
import math
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest

from claimci.review.openai_provider import OpenAIReviewerProvider
from claimci.review.models import ProviderUsage
from claimci.review.provider import (
    ProviderResponse,
    ReviewerProvider,
    StructuredRequest,
)


def test_reviewer_provider_has_exactly_the_two_provider_neutral_operations() -> None:
    """No arbitrary model tools or third operation may expand the trust boundary."""

    public = {
        name
        for name, member in ReviewerProvider.__dict__.items()
        if not name.startswith("_") and callable(member)
    }

    assert public == {"extract_claims", "synthesize_review"}
    assert set(inspect.signature(ReviewerProvider.extract_claims).parameters) == {
        "self",
        "request",
    }
    assert set(inspect.signature(ReviewerProvider.synthesize_review).parameters) == {
        "self",
        "request",
    }
    for forbidden in ("complete", "call", "run_tool", "tools", "filesystem", "web_search"):
        assert not hasattr(ReviewerProvider, forbidden)


def test_structured_request_is_provider_neutral_and_immutable() -> None:
    request = StructuredRequest(
        task="extract_claims",
        payload={"title": "A claim"},
        schema={"type": "object", "additionalProperties": False},
        max_output_tokens=2000,
    )

    assert request.task == "extract_claims"
    assert request.payload == {"title": "A claim"}
    assert request.schema["additionalProperties"] is False
    assert request.max_output_tokens == 2000
    with pytest.raises(FrozenInstanceError):
        request.task = "synthesize_review"  # type: ignore[misc]


def test_provider_response_records_raw_output_and_usage_without_authority_fields() -> None:
    usage = ProviderUsage(
        input_tokens=31,
        output_tokens=17,
        total_tokens=48,
        estimated_cost_usd=0.00042,
    )
    response = ProviderResponse(
        output_text='{"claims": []}',
        provider="fake",
        model="test-model",
        request_id="req-7",
        usage=usage,
    )

    assert response.output_text == '{"claims": []}'
    assert response.provider == "fake"
    assert response.model == "test-model"
    assert response.request_id == "req-7"
    assert response.usage is usage
    # Provider envelopes must not become a back door for deterministic state.
    assert not hasattr(response, "verdict")
    assert not hasattr(response, "findings")
    assert not hasattr(response, "severity")
    assert not hasattr(response, "impact")
    assert not hasattr(response, "threshold")


class _FakeResponses:
    def __init__(self, output_text: str = '{"claims": []}') -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="openai-request-1",
            output_text=self.output_text,
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                total_tokens=18,
            ),
        )


class _FakeClient:
    def __init__(self, output_text: str = '{"claims": []}') -> None:
        self.responses = _FakeResponses(output_text)


def _request(task: str = "extract_claims") -> StructuredRequest:
    return StructuredRequest(
        task=task,
        payload={"sources": []},
        schema={"type": "object", "additionalProperties": False},
        max_output_tokens=123,
    )


def test_openai_adapter_uses_injected_client_and_never_grants_tools() -> None:
    """The optional adapter must be testable without SDK credentials/network."""

    client = _FakeClient()
    provider = OpenAIReviewerProvider(
        model="test-model",
        timeout_seconds=4.5,
        max_output_tokens=123,
        client=client,
    )

    response = provider.extract_claims(_request())

    assert response.output_text == '{"claims": []}'
    assert response.provider == "openai"
    assert response.model == "test-model"
    assert response.request_id == "openai-request-1"
    assert len(client.responses.calls) == 1
    kwargs = client.responses.calls[0]
    assert kwargs["model"] == "test-model"
    assert kwargs["max_output_tokens"] == 123
    assert kwargs["timeout"] == 4.5
    # Responses API calls are structured-output only in Day 3: no arbitrary
    # tools, shell, filesystem, web, or function access.
    assert kwargs.get("tools") in (None, [])
    assert "functions" not in kwargs
    assert "web_search" not in kwargs
    assert "file_search" not in kwargs


def test_openai_adapter_performs_the_same_single_operation_without_retry() -> None:
    class FailingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: Any) -> Any:
            self.calls += 1
            raise TimeoutError("provider timed out")

    client = SimpleNamespace(responses=FailingResponses())
    provider = OpenAIReviewerProvider(
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=123,
        client=client,
    )

    with pytest.raises(Exception, match=r"(?i)timeout|timed out"):
        provider.extract_claims(_request())
    assert client.responses.calls == 1


@pytest.mark.parametrize("operation", ["extract_claims", "synthesize_review"])
def test_openai_adapter_refusal_or_provider_error_is_controlled_and_not_repaired(
    operation: str,
) -> None:
    class RefusingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: Any) -> Any:
            self.calls += 1
            raise RuntimeError("refusal: request rejected")

    client = SimpleNamespace(responses=RefusingResponses())
    provider = OpenAIReviewerProvider(
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=123,
        client=client,
    )

    request = _request(operation)
    with pytest.raises(Exception, match=r"(?i)refusal|rejected|provider"):
        getattr(provider, operation)(request)
    assert client.responses.calls == 1


def test_openai_adapter_missing_sdk_is_lazy_and_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing review config must not import an optional SDK or make a call."""

    monkeypatch.setitem(sys.modules, "openai", None)
    provider = OpenAIReviewerProvider(
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=123,
        client=None,
    )

    with pytest.raises(Exception, match=r"(?i)sdk|openai|install|provider"):
        provider.extract_claims(_request())


@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf, 121.0])
def test_openai_adapter_rejects_nonfinite_or_out_of_policy_timeout(
    timeout: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpenAIReviewerProvider(
            model="test-model",
            timeout_seconds=timeout,
            max_output_tokens=123,
            client=_FakeClient(),
        )


def test_openai_adapter_usage_overflow_is_observability_only() -> None:
    """Implausible provider counters cannot make a valid response unavailable."""

    class HugeUsageResponses:
        def create(self, **_: Any) -> Any:
            return SimpleNamespace(
                id="openai-request-huge-usage",
                output_text='{"claims": []}',
                usage=SimpleNamespace(
                    input_tokens=10**100_000,
                    output_tokens=1,
                    total_tokens=10**100_000,
                ),
            )

    provider = OpenAIReviewerProvider(
        model="gpt-5.6-terra",
        timeout_seconds=1,
        max_output_tokens=123,
        client=SimpleNamespace(responses=HugeUsageResponses()),
    )

    response = provider.extract_claims(_request())

    assert response.output_text == '{"claims": []}'
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens == 1
    assert response.usage.total_tokens is None
    assert response.usage.estimated_cost_usd is None
