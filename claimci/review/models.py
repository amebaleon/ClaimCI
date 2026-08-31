"""LLM-safe value objects for ClaimCI research review.

This module intentionally does not import the deterministic ClaimCI model
layer.  Provider-originated data can describe claims and interpretations, but
it has no type path for constructing blocking scientific authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath


class ReviewError(ValueError):
    """A controlled review configuration, source, or provider error."""


class ClaimType(str, Enum):
    METRIC_IMPROVEMENT = "metric_improvement"
    COMPUTE_EQUIVALENCE = "compute_equivalence"
    HELD_OUT_EVALUATION = "held_out_evaluation"
    RESOURCE_REDUCTION = "resource_reduction"
    COMPONENT_CAUSALITY = "component_causality"
    NO_EXTERNAL_REWARD = "no_external_reward"
    IMPLEMENTATION_CLAIM = "implementation_claim"
    OTHER_SCIENTIFIC = "other_scientific"


class ClaimDirection(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"
    NOT_APPLICABLE = "not_applicable"


class MagnitudeKind(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    UNSPECIFIED = "unspecified"


class SourceKind(str, Enum):
    PULL_REQUEST_TITLE = "pull_request_title"
    PULL_REQUEST_DESCRIPTION = "pull_request_description"
    REPOSITORY_FILE = "repository_file"


class ReviewStatus(str, Enum):
    DISABLED = "DISABLED"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_RECORDED_TOKEN_COUNT = 1_000_000_000_000


def _nonempty_text(value: object, label: str, *, max_chars: int = 16_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError(f"{label} must be a non-empty string")
    if len(value) > max_chars:
        raise ReviewError(f"{label} exceeds {max_chars} characters")
    return value


def _relative_path(value: object, label: str) -> str:
    text = _nonempty_text(value, label, max_chars=4_096)
    if "\x00" in text:
        raise ReviewError(f"{label} contains NUL")
    if "\\" in text:
        raise ReviewError(f"{label} must use unambiguous forward slashes")
    portable = text
    if (
        PurePosixPath(portable).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or ".." in portable.split("/")
    ):
        raise ReviewError(f"{label} must be a confined relative path")
    normalized = PurePosixPath(portable).as_posix()
    if normalized in {"", "."}:
        raise ReviewError(f"{label} must name a file")
    return normalized


def _positive_int(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewError(f"{label} must be an integer")
    if not 1 <= value <= maximum:
        raise ReviewError(f"{label} must be between 1 and {maximum}")
    return value


@dataclass(frozen=True)
class ReviewLimits:
    max_calls: int = 2
    max_context_chars: int = 60_000
    max_output_chars: int = 24_000
    max_files: int = 24
    max_file_chars: int = 16_000
    max_claims: int = 16
    extraction_max_output_tokens: int = 5_000
    synthesis_max_output_tokens: int = 4_000
    timeout_seconds: float = 30.0
    retries: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.max_calls, "max_calls", 2)
        _positive_int(self.max_context_chars, "max_context_chars", 60_000)
        _positive_int(self.max_output_chars, "max_output_chars", 24_000)
        _positive_int(self.max_files, "max_files", 24)
        _positive_int(self.max_file_chars, "max_file_chars", 16_000)
        _positive_int(self.max_claims, "max_claims", 16)
        _positive_int(
            self.extraction_max_output_tokens,
            "extraction_max_output_tokens",
            5_000,
        )
        _positive_int(
            self.synthesis_max_output_tokens,
            "synthesis_max_output_tokens",
            4_000,
        )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise ReviewError("timeout_seconds must be a finite number from 0 through 30")
        try:
            timeout = float(self.timeout_seconds)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ReviewError(
                "timeout_seconds must be a finite number from 0 through 30"
            ) from exc
        if not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ReviewError("timeout_seconds must be a finite number from 0 through 30")
        if isinstance(self.retries, bool) or self.retries != 0:
            raise ReviewError("provider retries must be zero")


@dataclass(frozen=True)
class ReviewConfig:
    schema_version: int = 1
    enabled: bool = False
    policy: str = "advisory"
    provider: str = "openai"
    model: str = "gpt-5.6-terra"
    limits: ReviewLimits = field(default_factory=ReviewLimits)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ReviewError("review schema_version must be 1")
        if not isinstance(self.enabled, bool):
            raise ReviewError("review enabled must be a boolean")
        if self.policy != "advisory":
            raise ReviewError("Day 3 review policy must be advisory")
        _nonempty_text(self.provider, "review provider", max_chars=64)
        _nonempty_text(self.model, "review model", max_chars=128)
        if not isinstance(self.limits, ReviewLimits):
            raise ReviewError("review limits must be ReviewLimits")


@dataclass(frozen=True)
class SourceLocation:
    source_id: str
    kind: SourceKind
    path: str | None
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        _nonempty_text(self.source_id, "source_id", max_chars=128)
        if not isinstance(self.kind, SourceKind):
            raise ReviewError("source kind is invalid")
        if self.kind is SourceKind.REPOSITORY_FILE:
            normalized = _relative_path(self.path, "source path")
            object.__setattr__(self, "path", normalized)
        elif self.path is not None:
            raise ReviewError("pull-request metadata sources cannot declare a path")
        if isinstance(self.start_line, bool) or not isinstance(self.start_line, int):
            raise ReviewError("source start_line must be an integer")
        if isinstance(self.end_line, bool) or not isinstance(self.end_line, int):
            raise ReviewError("source end_line must be an integer")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ReviewError("source line span is invalid")


@dataclass(frozen=True)
class ClaimMagnitude:
    raw: str
    value: float | None = None
    unit: str | None = None
    kind: MagnitudeKind = MagnitudeKind.UNSPECIFIED

    def __post_init__(self) -> None:
        _nonempty_text(self.raw, "claimed magnitude", max_chars=512)
        if not isinstance(self.kind, MagnitudeKind):
            raise ReviewError("claimed magnitude kind is invalid")
        if self.value is not None:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(float(self.value))
            ):
                raise ReviewError("claimed magnitude value must be finite")
        if self.unit is not None:
            _nonempty_text(self.unit, "claimed magnitude unit", max_chars=128)


@dataclass(frozen=True)
class ScientificClaim:
    claim_id: str
    source_text: str
    claim_type: ClaimType
    subject: str
    source: SourceLocation
    metric: str | None = None
    direction: ClaimDirection = ClaimDirection.NOT_APPLICABLE
    claimed_magnitude: ClaimMagnitude | None = None
    qualifiers: tuple[str, ...] = ()
    confidence: float = 0.0
    evidence_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_text(self.claim_id, "claim_id", max_chars=128)
        _nonempty_text(self.source_text, "claim source_text")
        if not isinstance(self.claim_type, ClaimType):
            raise ReviewError("claim type is invalid")
        _nonempty_text(self.subject, "claim subject", max_chars=1_024)
        if not isinstance(self.source, SourceLocation):
            raise ReviewError("claim source must be a SourceLocation")
        if self.metric is not None:
            _nonempty_text(self.metric, "claim metric", max_chars=256)
        if not isinstance(self.direction, ClaimDirection):
            raise ReviewError("claim direction is invalid")
        if self.claimed_magnitude is not None and not isinstance(
            self.claimed_magnitude, ClaimMagnitude
        ):
            raise ReviewError("claimed_magnitude is invalid")
        if not isinstance(self.qualifiers, tuple) or not all(
            isinstance(item, str) and item.strip() and len(item) <= 1_024
            for item in self.qualifiers
        ):
            raise ReviewError("claim qualifiers must be bounded non-empty strings")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0 <= float(self.confidence) <= 1
        ):
            raise ReviewError("claim confidence must be finite from 0 through 1")
        if not isinstance(self.evidence_hints, tuple) or not all(
            isinstance(item, str) and item.strip() and len(item) <= 4_096
            for item in self.evidence_hints
        ):
            raise ReviewError("evidence hints must be bounded non-empty strings")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    kind: SourceKind
    path: str | None
    text: str
    sha256: str

    def __post_init__(self) -> None:
        _nonempty_text(self.source_id, "source_id", max_chars=128)
        if not isinstance(self.kind, SourceKind):
            raise ReviewError("source kind is invalid")
        if self.kind is SourceKind.REPOSITORY_FILE:
            object.__setattr__(self, "path", _relative_path(self.path, "source path"))
        elif self.path is not None:
            raise ReviewError("pull-request metadata sources cannot declare a path")
        if not isinstance(self.text, str):
            raise ReviewError("source text must be a string")
        if not isinstance(self.sha256, str) or not _HEX_DIGEST.fullmatch(self.sha256):
            raise ReviewError("source sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SourceBundle:
    sources: tuple[SourceRecord, ...] = ()
    repository_paths: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    total_chars: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple) or not all(
            isinstance(item, SourceRecord) for item in self.sources
        ):
            raise ReviewError("sources must be SourceRecord values")
        if not isinstance(self.repository_paths, tuple):
            raise ReviewError("repository_paths must be a tuple")
        normalized = tuple(_relative_path(path, "repository path") for path in self.repository_paths)
        if normalized != self.repository_paths:
            object.__setattr__(self, "repository_paths", normalized)
        if not isinstance(self.changed_paths, tuple):
            raise ReviewError("changed_paths must be a tuple")
        normalized_changed = tuple(
            _relative_path(path, "changed repository path")
            for path in self.changed_paths
        )
        if not set(normalized_changed).issubset(set(normalized)):
            raise ReviewError("changed_paths must be issued repository paths")
        if normalized_changed != self.changed_paths:
            object.__setattr__(self, "changed_paths", normalized_changed)
        if (
            isinstance(self.total_chars, bool)
            or not isinstance(self.total_chars, int)
            or self.total_chars < 0
            or self.total_chars != sum(len(source.text) for source in self.sources)
        ):
            raise ReviewError("source total_chars is inconsistent")


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None

    def __post_init__(self) -> None:
        for label in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, label)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ReviewError(f"{label} must be a non-negative integer or null")
            if value is not None and value > MAX_RECORDED_TOKEN_COUNT:
                # Provider usage is observability only. Implausibly large
                # counters are represented as unavailable instead of making
                # JSON/Markdown integer conversion fail.
                object.__setattr__(self, label, None)
        if self.estimated_cost_usd is not None:
            try:
                cost = Decimal(str(self.estimated_cost_usd))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ReviewError("estimated_cost_usd must be a decimal") from exc
            if not cost.is_finite() or cost < 0:
                raise ReviewError("estimated_cost_usd must be finite and non-negative")
            object.__setattr__(self, "estimated_cost_usd", cost)


@dataclass(frozen=True)
class ProviderCallRecord:
    task: str
    provider: str
    model: str
    request_id: str | None
    usage: ProviderUsage
    input_chars: int
    output_chars: int

    def __post_init__(self) -> None:
        _nonempty_text(self.task, "provider task", max_chars=64)
        _nonempty_text(self.provider, "provider name", max_chars=64)
        _nonempty_text(self.model, "provider model", max_chars=128)
        if self.request_id is not None:
            _nonempty_text(self.request_id, "provider request_id", max_chars=256)
        if not isinstance(self.usage, ProviderUsage):
            raise ReviewError("provider call usage must be ProviderUsage")
        for label in ("input_chars", "output_chars"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReviewError(f"{label} must be a non-negative integer")
