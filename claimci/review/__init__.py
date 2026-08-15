"""Opt-in, advisory LLM research-review layer for ClaimCI.

The deterministic auditor remains in :mod:`claimci.audit`.  This package keeps
review-only models and provider output separate from ClaimCI's authoritative
finding and verdict types.
"""

from .models import (
    ClaimDirection,
    ClaimMagnitude,
    ClaimType,
    MagnitudeKind,
    ProviderCallRecord,
    ProviderUsage,
    ReviewConfig,
    ReviewError,
    ReviewLimits,
    ReviewStatus,
    ScientificClaim,
    SourceBundle,
    SourceKind,
    SourceLocation,
    SourceRecord,
)

__all__ = [
    "ClaimDirection",
    "ClaimMagnitude",
    "ClaimType",
    "MagnitudeKind",
    "ProviderCallRecord",
    "ProviderUsage",
    "ReviewConfig",
    "ReviewError",
    "ReviewLimits",
    "ReviewStatus",
    "ScientificClaim",
    "SourceBundle",
    "SourceKind",
    "SourceLocation",
    "SourceRecord",
]
