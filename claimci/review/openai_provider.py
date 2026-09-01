"""Optional OpenAI Responses API adapter for ClaimCI research review."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from decimal import Decimal, DecimalException
from enum import Enum
from typing import Any

from .models import MAX_RECORDED_TOKEN_COUNT, ProviderUsage, ReviewError
from .provider import ProviderResponse, REVIEW_SYSTEM_POLICY, StructuredRequest


_PRICES_PER_MILLION: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-5.6-terra": (Decimal("2.50"), Decimal("15.00")),
    "gpt-5.6-luna": (Decimal("1.00"), Decimal("6.00")),
    "gpt-5.6-sol": (Decimal("5.00"), Decimal("30.00")),
}

def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _usage_value(usage: object, name: str) -> int | None:
    value = getattr(usage, name, None)
    return (
        value
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= MAX_RECORDED_TOKEN_COUNT
        )
        else None
    )


def _estimated_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> Decimal | None:
    prices = _PRICES_PER_MILLION.get(model)
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    try:
        return (
            Decimal(input_tokens) * prices[0] + Decimal(output_tokens) * prices[1]
        ) / Decimal(1_000_000)
    except (DecimalException, OverflowError, ValueError):
        # Provider usage is observability only; unusable counters must not
        # turn otherwise valid structured output into a failed review.
        return None


class OpenAIReviewerProvider:
    """Two-operation OpenAI adapter with lazy credentials/SDK and no tools."""

    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        client: object | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ReviewError("OpenAI model must be non-empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or not 0 < float(timeout_seconds) <= 120
        ):
            raise ReviewError("OpenAI timeout must be finite and from 0 through 120")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            raise ReviewError("OpenAI max_output_tokens must be positive")
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_tokens = max_output_tokens
        self._provided_client = client
        self._lazy_client: object | None = None

    def _client(self) -> object:
        if self._provided_client is not None:
            return self._provided_client
        if self._lazy_client is not None:
            return self._lazy_client
        try:
            from openai import OpenAI
        except (ImportError, ModuleNotFoundError) as exc:
            raise ReviewError(
                "OpenAI SDK is not installed; install ClaimCI with the llm extra"
            ) from exc
        try:
            # The SDK reads OPENAI_API_KEY from the environment. ClaimCI never
            # reads, logs, serializes, or writes the credential itself.
            self._lazy_client = OpenAI(max_retries=0)
        except Exception as exc:
            raise ReviewError(f"OpenAI provider could not be initialized: {type(exc).__name__}") from exc
        return self._lazy_client

    def _perform(self, request: StructuredRequest) -> ProviderResponse:
        if not isinstance(request, StructuredRequest):
            raise ReviewError("OpenAI provider requires a StructuredRequest")
        output_tokens = min(request.max_output_tokens, self.max_output_tokens)
        payload = json.dumps(
            _plain(request.payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        client = self._client()
        try:
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": REVIEW_SYSTEM_POLICY}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": payload}],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": request.task,
                        "strict": True,
                        "schema": _plain(request.schema),
                    }
                },
                reasoning={"effort": "low"},
                max_output_tokens=output_tokens,
                timeout=self.timeout_seconds,
            )
        except Exception:
            # Preserve the provider exception type for callers while making no
            # retry and never echoing request content or credentials.
            raise
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise ReviewError("OpenAI provider returned no structured text output")

        status = getattr(response, "status", None)
        if status in (None, "completed"):
            complete = True
            incomplete_reason = None
        elif status == "incomplete":
            complete = False
            details = getattr(response, "incomplete_details", None)
            raw_reason = getattr(details, "reason", None)
            incomplete_reason = (
                raw_reason
                if isinstance(raw_reason, str) and raw_reason.strip() and len(raw_reason) <= 128
                else "unknown"
            )
        else:
            raise ReviewError("OpenAI provider returned a non-completed response")

        raw_usage = getattr(response, "usage", None)
        input_tokens = _usage_value(raw_usage, "input_tokens")
        output_count = _usage_value(raw_usage, "output_tokens")
        total_tokens = _usage_value(raw_usage, "total_tokens")
        usage = ProviderUsage(
            input_tokens=input_tokens,
            output_tokens=output_count,
            total_tokens=total_tokens,
            estimated_cost_usd=_estimated_cost(self.model, input_tokens, output_count),
        )
        return ProviderResponse(
            output_text=output_text,
            provider="openai",
            model=self.model,
            request_id=getattr(response, "id", None),
            usage=usage,
            complete=complete,
            incomplete_reason=incomplete_reason,
        )

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        return self._perform(request)

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        return self._perform(request)
