"""Validated claim conversion helpers for deterministic analysis."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from claimci.models import ResearchSpec

from .contracts import (
    AuditClaimSpec,
    FieldProvenance,
    ProvenanceKind,
    RepositoryPath,
)


def audit_relevant_claim_projection(spec: AuditClaimSpec) -> dict[str, object]:
    """Return only claim fields that affect deterministic Audit behavior."""

    if type(spec) is not AuditClaimSpec:
        raise TypeError("audit claim projection requires AuditClaimSpec")
    return {
        "claim_id": spec.claim_id,
        "metric": spec.metric,
        "direction": spec.direction.value,
        "minimum_absolute_improvement": spec.minimum_absolute_improvement,
    }


def _manifest_claim_id(path_text: str) -> str:
    candidate = f"manifest:{path_text}"
    if len(candidate) <= 128 and not any(ord(character) < 32 for character in candidate):
        return candidate
    digest = hashlib.sha256(path_text.encode("utf-8")).hexdigest()[:24]
    return f"manifest:{digest}"


def audit_claim_spec_from_research_spec(spec: ResearchSpec) -> AuditClaimSpec:
    """Represent one already parsed user manifest as an audit claim policy."""

    if type(spec) is not ResearchSpec:
        raise TypeError("native claim conversion requires ResearchSpec")
    path_text = PurePosixPath(spec.manifest_path.as_posix()).as_posix()
    source_path: RepositoryPath | None = None
    if not spec.manifest_path.is_absolute():
        try:
            source_path = RepositoryPath(path_text)
        except ValueError:
            source_path = None
    claim_id = _manifest_claim_id(path_text)
    provenance = FieldProvenance(
        kind=ProvenanceKind.MANIFEST_HINT,
        detail="validated native research manifest claim",
        source_path=source_path,
        source_id=claim_id,
    )
    return AuditClaimSpec(
        claim_id=claim_id,
        metric=spec.metric,
        direction=spec.direction,
        minimum_absolute_improvement=spec.minimum_improvement,
        metric_provenance=provenance,
        direction_provenance=provenance,
        threshold_provenance=provenance,
    )


__all__ = [
    "audit_claim_spec_from_research_spec",
    "audit_relevant_claim_projection",
]
