"""Deterministic claim heuristics and strict provider proposal validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from claimci.analysis.confidence import Confidence
from claimci.analysis.claim_types import (
    AbsoluteMetricClaim,
    CanonicalScientificClaim,
    ClaimQuantity,
    MetricImprovementClaim,
    PrimaryClaimKind,
    _TrustedSourceDocument,
    _issue_trusted_source_document,
    recover_scientific_claim,
)
from claimci.analysis.contracts import (
    ClaimReference,
    FieldProvenance,
    ProvenanceKind,
    RepositoryPath,
)
from claimci.review.models import (
    ClaimDirection,
    ClaimMagnitude,
    ClaimType,
    MagnitudeKind,
    ReviewError,
    ScientificClaim,
    SourceKind,
    SourceLocation,
    SourceRecord,
)
from claimci.review.sources import validate_claim_candidates

from .models import ClaimedValue, DiscoveredClaim, DiscoveryError, DiscoveryLimits
from .repository import RepositoryContext


MAX_DISCOVERED_CLAIMS = 64
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _ClaimMatch:
    claim_type: ClaimType
    subject: str
    metric: str | None
    direction: ClaimDirection
    confidence: float
    pattern: str


def _classify_line(text: str) -> _ClaimMatch | None:
    lowered = _WHITESPACE.sub(" ", text.casefold())
    if "compute" in lowered and any(
        term in lowered for term in ("equivalent", "equal", "same", "matching")
    ):
        return _ClaimMatch(
            ClaimType.COMPUTE_EQUIVALENCE,
            "candidate",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.82,
            "compute_equivalence",
        )
    if any(term in lowered for term in ("held-out", "held out", "unseen")) and any(
        term in lowered for term in ("eval", "evaluation", "test", "data", "set")
    ):
        return _ClaimMatch(
            ClaimType.HELD_OUT_EVALUATION,
            "evaluation",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.84,
            "held_out_evaluation",
        )
    if re.search(r"\b(?:reduce[ds]?|lower(?:ed|s)?|decreas(?:e|ed|es))\b", lowered) and any(
        term in lowered
        for term in ("latency", "memory", "cost", "resource", "parameters", "storage")
    ):
        return _ClaimMatch(
            ClaimType.RESOURCE_REDUCTION,
            "candidate",
            None,
            ClaimDirection.LOWER,
            0.80,
            "resource_reduction",
        )
    if any(term in lowered for term in ("ablation", "component")) and any(
        term in lowered for term in ("cause", "causal", "drives", "responsible")
    ):
        return _ClaimMatch(
            ClaimType.COMPONENT_CAUSALITY,
            "candidate component",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.80,
            "component_causality",
        )
    if any(
        term in lowered
        for term in (
            "no external reward",
            "without external reward",
            "does not use external reward",
        )
    ):
        return _ClaimMatch(
            ClaimType.NO_EXTERNAL_REWARD,
            "training",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.88,
            "no_external_reward",
        )
    if re.search(
        r"\b(?:implements?|implemented|implementation|adds? support|now supports?)\b",
        lowered,
    ):
        return _ClaimMatch(
            ClaimType.IMPLEMENTATION_CLAIM,
            "implementation",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.76,
            "implementation_claim",
        )
    if any(
        term in lowered
        for term in (
            "scientific effect",
            "statistically significant",
            "generalizes",
            "generalises",
            "robustness",
        )
    ):
        return _ClaimMatch(
            ClaimType.OTHER_SCIENTIFIC,
            "research claim",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.68,
            "other_scientific",
        )
    return None


def _canonical_fallback_match(
    text: str,
    record: SourceRecord,
) -> _ClaimMatch | None:
    """Recognize new canonical primaries that lack a legacy Review category."""

    provisional_provenance = _provenance(
        record,
        kind=ProvenanceKind.DETERMINISTIC_DISCOVERY,
        detail="deterministic canonical claim recognition",
    )
    provisional = ClaimReference(
        claim_id="claim-provisional",
        text=text,
        source_path=None if record.path is None else RepositoryPath(record.path),
        confidence=Confidence(0.75),
        provenance=provisional_provenance,
    )
    canonical = recover_scientific_claim(provisional)
    if canonical is None:
        return None
    primary = canonical.primary
    if primary.kind is PrimaryClaimKind.METRIC_IMPROVEMENT:
        return _ClaimMatch(
            ClaimType.METRIC_IMPROVEMENT,
            "candidate",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.70,
            "canonical_metric_improvement",
        )
    if primary.kind is PrimaryClaimKind.ABSOLUTE_METRIC:
        return _ClaimMatch(
            ClaimType.OTHER_SCIENTIFIC,
            "candidate",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.84,
            "absolute_metric",
        )
    if primary.kind is PrimaryClaimKind.GENERALIZATION:
        return _ClaimMatch(
            ClaimType.OTHER_SCIENTIFIC,
            "candidate",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.74,
            "generalization",
        )
    if primary.kind is PrimaryClaimKind.GENERIC_QUANTITATIVE:
        return _ClaimMatch(
            ClaimType.OTHER_SCIENTIFIC,
            "quantitative claim",
            None,
            ClaimDirection.NOT_APPLICABLE,
            0.72,
            "generic_quantitative",
        )
    return None


def _canonical_confidence(canonical: CanonicalScientificClaim) -> Confidence:
    primary = canonical.primary
    if type(primary) is MetricImprovementClaim:
        value = (
            0.95
            if primary.baseline_value is not None
            else 0.90
            if primary.minimum_improvement is not None
            else 0.70
        )
    elif type(primary) is AbsoluteMetricClaim:
        value = 0.84
    elif primary.kind is PrimaryClaimKind.GENERALIZATION:
        value = 0.74
    else:
        value = 0.72
    return Confidence(value)


def _recover_final_reference(
    reference: ClaimReference,
    *,
    trusted_source_document: _TrustedSourceDocument | None = None,
    primary_span: tuple[int, int] | None = None,
) -> tuple[ClaimReference, CanonicalScientificClaim | None]:
    canonical = recover_scientific_claim(
        reference,
        trusted_source_document=trusted_source_document,
        primary_span=primary_span,
    )
    if canonical is None:
        return reference, None
    confidence = _canonical_confidence(canonical)
    if confidence != reference.confidence:
        reference = replace(reference, confidence=confidence)
        canonical = recover_scientific_claim(
            reference,
            trusted_source_document=trusted_source_document,
            primary_span=primary_span,
        )
    return reference, canonical


def _canonical_metric_projection(
    canonical: CanonicalScientificClaim,
) -> tuple[str | None, ClaimDirection]:
    primary = canonical.primary
    if type(primary) is MetricImprovementClaim:
        return primary.metric, ClaimDirection(primary.direction.value)
    if type(primary) is AbsoluteMetricClaim:
        return primary.metric, ClaimDirection.NOT_APPLICABLE
    return None, ClaimDirection.NOT_APPLICABLE


def _provenance(
    record: SourceRecord,
    *,
    kind: ProvenanceKind,
    detail: str,
) -> FieldProvenance:
    return FieldProvenance(
        kind=kind,
        detail=detail,
        source_path=None if record.path is None else RepositoryPath(record.path),
        source_id=record.source_id,
    )


def _claim_id(
    *,
    kind: ProvenanceKind,
    source_id: str,
    start_line: int,
    claim_type: ClaimType,
    text: str,
) -> str:
    material = json.dumps(
        {
            "kind": kind.value,
            "source_id": source_id,
            "start_line": start_line,
            "claim_type": claim_type.value,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "claim-" + hashlib.sha256(material).hexdigest()[:16]


def _claimed_value(quantity: ClaimQuantity | None) -> ClaimedValue | None:
    if quantity is None:
        return None
    return ClaimedValue(quantity.value, quantity.unit, quantity.provenance)


def _source_lines(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return original source lines with the exact spans used for certification."""

    lines: list[tuple[int, int, str]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        end = offset + len(raw_line)
        line_end = end - 1 if raw_line.endswith("\n") else end
        lines.append((offset, line_end, text[offset:line_end]))
        offset = end
    if offset < len(text):
        lines.append((offset, len(text), text[offset:]))
    return tuple(lines)


def _deterministic_claims(context: RepositoryContext) -> list[DiscoveredClaim]:
    claims: list[DiscoveredClaim] = []
    for record in context.source_bundle.sources:
        trusted_source_document = _issue_trusted_source_document(
            source_id=record.source_id,
            sha256=record.sha256,
            text=record.text,
        )
        for line_number, (start, end, source_text) in enumerate(
            _source_lines(record.text),
            start=1,
        ):
            if not source_text.strip():
                continue
            matched = _canonical_fallback_match(source_text, record)
            if matched is None:
                matched = _classify_line(source_text)
            if matched is None:
                continue
            provenance = _provenance(
                record,
                kind=ProvenanceKind.DETERMINISTIC_DISCOVERY,
                detail=f"deterministic claim pattern: {matched.pattern}",
            )
            claim_id = _claim_id(
                kind=ProvenanceKind.DETERMINISTIC_DISCOVERY,
                source_id=record.source_id,
                start_line=line_number,
                claim_type=matched.claim_type,
                text=source_text,
            )
            source = SourceLocation(
                source_id=record.source_id,
                kind=record.kind,
                path=record.path,
                start_line=line_number,
                end_line=line_number,
            )
            reference = ClaimReference(
                claim_id=claim_id,
                text=source_text,
                source_path=None if record.path is None else RepositoryPath(record.path),
                confidence=Confidence(matched.confidence),
                provenance=provenance,
            )
            reference, canonical = _recover_final_reference(
                reference,
                trusted_source_document=trusted_source_document,
                primary_span=(start, end),
            )
            primary = (
                canonical.primary
                if canonical is not None
                and type(canonical.primary) is MetricImprovementClaim
                else None
            )
            if canonical is not None:
                metric, direction = _canonical_metric_projection(canonical)
            elif matched.claim_type is ClaimType.METRIC_IMPROVEMENT:
                continue
            else:
                metric = matched.metric
                direction = matched.direction
            claims.append(
                DiscoveredClaim(
                    reference=reference,
                    claim_type=matched.claim_type,
                    subject=matched.subject,
                    source=source,
                    metric=metric,
                    direction=direction,
                    baseline_value=(
                        _claimed_value(primary.baseline_value)
                        if primary is not None
                        else None
                    ),
                    candidate_value=(
                        _claimed_value(primary.candidate_value)
                        if primary is not None
                        else None
                    ),
                    minimum_improvement=(
                        _claimed_value(primary.minimum_improvement)
                        if primary is not None
                        else None
                    ),
                    scientific_claim=canonical,
                )
            )
            if len(claims) >= MAX_DISCOVERED_CLAIMS:
                return claims
    return claims


def _provider_claims(
    context: RepositoryContext,
    payload: object,
    limits: DiscoveryLimits,
) -> list[DiscoveredClaim]:
    if isinstance(payload, Mapping):
        raw_claims = payload.get("claims")
        if isinstance(raw_claims, list) and len(raw_claims) > limits.max_provider_claims:
            raise DiscoveryError("provider claim proposal exceeds the configured bound")
    try:
        validated = validate_claim_candidates(payload, context.source_bundle)
    except (ReviewError, TypeError, ValueError) as error:
        raise DiscoveryError("provider claim proposal is invalid") from error
    if len(validated) > limits.max_provider_claims:
        raise DiscoveryError("provider claim proposal exceeds the configured bound")
    issued = set(context.repository_paths)
    discovered: list[DiscoveredClaim] = []
    for claim in validated:
        hints: list[RepositoryPath] = []
        try:
            for raw in claim.evidence_hints:
                path = RepositoryPath(raw)
                if path not in issued:
                    raise DiscoveryError(
                        "provider evidence hint is not an indexed repository path"
                    )
                hints.append(path)
        except (TypeError, ValueError) as error:
            raise DiscoveryError("provider evidence hint is unsafe") from error
        record = next(
            item
            for item in context.source_bundle.sources
            if item.source_id == claim.source.source_id
        )
        provenance = _provenance(
            record,
            kind=ProvenanceKind.PROVIDER_PROPOSAL,
            detail="validated provider claim proposal",
        )
        reference = ClaimReference(
            claim_id=claim.claim_id,
            text=claim.source_text,
            source_path=None
            if claim.source.path is None
            else RepositoryPath(claim.source.path),
            confidence=Confidence(claim.confidence),
            provenance=provenance,
        )
        canonical = recover_scientific_claim(reference)
        primary = (
            canonical.primary
            if canonical is not None
            and type(canonical.primary) is MetricImprovementClaim
            else None
        )
        if claim.claim_type is ClaimType.METRIC_IMPROVEMENT and primary is None:
            raise DiscoveryError("provider claim source semantics are ambiguous")
        if canonical is not None:
            metric, direction = _canonical_metric_projection(canonical)
        else:
            metric = claim.metric
            direction = claim.direction
        discovered.append(
            DiscoveredClaim(
                reference=reference,
                claim_type=claim.claim_type,
                subject=claim.subject,
                source=claim.source,
                metric=metric,
                direction=direction,
                baseline_value=(
                    _claimed_value(primary.baseline_value)
                    if primary is not None
                    else None
                ),
                candidate_value=(
                    _claimed_value(primary.candidate_value)
                    if primary is not None
                    else None
                ),
                minimum_improvement=(
                    _claimed_value(primary.minimum_improvement)
                    if primary is not None
                    else None
                ),
                qualifiers=claim.qualifiers,
                evidence_hints=tuple(hints),
            )
        )
    return discovered


def _dedupe_key(claim: DiscoveredClaim) -> tuple[str, object, str, int]:
    semantic_kind: object = claim.claim_type
    if claim.scientific_claim is not None:
        semantic_kind = claim.scientific_claim.primary.kind
    return (
        _WHITESPACE.sub(" ", claim.reference.text.strip().casefold()),
        semantic_kind,
        claim.source.source_id,
        claim.source.start_line,
    )


def discover_claims(
    context: RepositoryContext,
    *,
    provider_payload: object | None = None,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> tuple[DiscoveredClaim, ...]:
    """Return deterministic claims followed by strictly validated proposals."""

    if not isinstance(context, RepositoryContext):
        raise TypeError("repository context must be RepositoryContext")
    if not isinstance(limits, DiscoveryLimits):
        raise TypeError("discovery limits must be DiscoveryLimits")
    candidates = _deterministic_claims(context)
    if provider_payload is not None:
        candidates.extend(_provider_claims(context, provider_payload, limits))

    source_order = {
        source.source_id: index
        for index, source in enumerate(context.source_bundle.sources)
    }
    unique: dict[tuple[str, object, str, int], DiscoveredClaim] = {}
    for claim in candidates:
        unique.setdefault(_dedupe_key(claim), claim)
    return tuple(
        sorted(
            unique.values(),
            key=lambda claim: (
                source_order[claim.source.source_id],
                claim.source.start_line,
                claim.claim_type.value,
                claim.reference.text.casefold(),
                claim.reference.claim_id,
            ),
        )[:MAX_DISCOVERED_CLAIMS]
    )


def as_scientific_claim(claim: DiscoveredClaim) -> ScientificClaim:
    """Adapt a validated discovery claim to existing evidence routing."""

    if not isinstance(claim, DiscoveredClaim):
        raise TypeError("claim must be DiscoveredClaim")
    magnitude = None
    if claim.minimum_improvement is not None:
        unit = claim.minimum_improvement.unit
        raw = f"at least {claim.minimum_improvement.value:g}"
        if unit:
            raw += f" {unit}"
        magnitude = ClaimMagnitude(
            raw=raw,
            value=claim.minimum_improvement.value,
            unit=unit,
            kind=MagnitudeKind.ABSOLUTE,
        )
    return ScientificClaim(
        claim_id=claim.reference.claim_id,
        source_text=claim.reference.text,
        claim_type=claim.claim_type,
        subject=claim.subject,
        source=claim.source,
        metric=claim.metric,
        direction=claim.direction,
        claimed_magnitude=magnitude,
        qualifiers=claim.qualifiers,
        confidence=claim.reference.confidence.value,
        evidence_hints=tuple(str(path) for path in claim.evidence_hints),
    )


__all__ = ["as_scientific_claim", "discover_claims"]
