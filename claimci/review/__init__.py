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
from .analysis_bridge import (
    AnalysisReviewContext,
    run_analysis_review,
    validate_scientific_claim_for_audit,
)

__all__ = [
    "AnalysisReviewContext",
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
    "run_analysis_review",
    "validate_scientific_claim_for_audit",
]
