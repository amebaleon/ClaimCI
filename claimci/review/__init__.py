"""Opt-in, advisory LLM research-review layer for ClaimCI.

The deterministic auditor remains in :mod:`claimci.audit`.  This package keeps
review-only models and provider output separate from ClaimCI's authoritative
finding and verdict types.
"""

from .models import (
    ChangeEntry,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    ClaimDirection,
    ClaimMagnitude,
    ClaimType,
    ComparisonBasis,
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
    SnapshotIdentity,
    SnapshotRole,
)
from .analysis_bridge import (
    AnalysisReviewContext,
    run_analysis_review,
    validate_scientific_claim_for_audit,
)
from .inventory import build_git_change_inventory

__all__ = [
    "AnalysisReviewContext",
    "ChangeEntry",
    "ChangeInventory",
    "ChangeInventorySource",
    "ChangeStatus",
    "ClaimDirection",
    "ClaimMagnitude",
    "ClaimType",
    "ComparisonBasis",
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
    "SnapshotIdentity",
    "SnapshotRole",
    "build_git_change_inventory",
    "run_analysis_review",
    "validate_scientific_claim_for_audit",
]
