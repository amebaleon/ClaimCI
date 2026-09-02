"""Provider-neutral structured-response boundary for research review."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import ProviderUsage, ReviewError
from .request_budget import REVIEW_SYSTEM_POLICY


@dataclass(frozen=True)
class StructuredRequest:
    task: str
    payload: Mapping[str, Any]
    schema: Mapping[str, Any]
    max_output_tokens: int

    def __post_init__(self) -> None:
        if self.task not in {"extract_claims", "synthesize_review"}:
            raise ReviewError("structured request has an unsupported task")
        if not isinstance(self.payload, Mapping):
            raise ReviewError("structured request payload must be a mapping")
        if not isinstance(self.schema, Mapping):
            raise ReviewError("structured request schema must be a mapping")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ReviewError("structured request max_output_tokens must be positive")


@dataclass(frozen=True)
class ProviderResponse:
    output_text: str
    provider: str
    model: str
    request_id: str | None = None
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    complete: bool = True
    incomplete_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_text, str):
            raise ReviewError("provider output must be text")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ReviewError("provider name must be non-empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ReviewError("provider model must be non-empty")
        if self.request_id is not None and not isinstance(self.request_id, str):
            raise ReviewError("provider request_id must be text or null")
        if not isinstance(self.usage, ProviderUsage):
            raise ReviewError("provider usage must be ProviderUsage")
        if not isinstance(self.complete, bool):
            raise ReviewError("provider completion state must be a boolean")
        if self.incomplete_reason is not None and (
            not isinstance(self.incomplete_reason, str)
            or not self.incomplete_reason.strip()
            or len(self.incomplete_reason) > 128
        ):
            raise ReviewError("provider incomplete reason must be bounded text or null")
        if self.complete and self.incomplete_reason is not None:
            raise ReviewError("a complete provider response cannot have an incomplete reason")


class ReviewerProvider(Protocol):
    """The complete provider surface: exactly extraction and synthesis."""

    def extract_claims(self, request: StructuredRequest) -> ProviderResponse:
        ...

    def synthesize_review(self, request: StructuredRequest) -> ProviderResponse:
        ...
