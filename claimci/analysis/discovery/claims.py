"""Deterministic claim heuristics and strict provider proposal validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from claimci.analysis.confidence import Confidence
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
_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_METRIC_PAIR = re.compile(
    rf"\b(?P<metric>[A-Za-z][A-Za-z0-9_.-]{{0,63}})\s+"
    rf"(?:score\s+)?(?:improved|increased|rose|grew)\s+"
    rf"(?:from\s+)?(?P<baseline>{_NUMBER})\s*(?P<baseline_unit>%?)\s*"
    rf"(?:->|→|to)\s*(?P<candidate>{_NUMBER})\s*(?P<candidate_unit>%?)",
    flags=re.IGNORECASE,
)
_METRIC_VAGUE = re.compile(
    r"\b(?P<metric>[A-Za-z][A-Za-z0-9_.-]{0,63})\s+"
    r"(?:score\s+)?(?:improved|increased|rose|grew)\b",
    flags=re.IGNORECASE,
)
_VALUE_PAIR = re.compile(
    rf"(?P<baseline>{_NUMBER})\s*(?P<baseline_unit>%?)\s*"
    rf"(?:->|→|to)\s*(?P<candidate>{_NUMBER})\s*(?P<candidate_unit>%?)",
    flags=re.IGNORECASE,
)
_MINIMUM = re.compile(
    rf"(?:improved|increased|rose|grew)\s+by\s+at\s+least\s+"
    rf"(?P<value>{_NUMBER})"
    rf"(?:\s*(?P<unit>%|percentage\s+points?|points?))?(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _ClaimMatch:
    claim_type: ClaimType
    subject: str
    metric: str | None
    direction: ClaimDirection
    confidence: float
    pattern: str


def _normalized_unit(raw: str | None) -> str | None:
    if not raw:
        return None
    normalized = _WHITESPACE.sub(" ", raw.strip().casefold())
    if normalized.startswith("percentage point"):
        return "percentage points"
    if normalized.startswith("point"):
        return "points"
    return "%" if normalized == "%" else normalized


def _explicit_pair(text: str, *, require_metric: bool) -> re.Match[str] | None:
    return (_METRIC_PAIR if require_metric else _VALUE_PAIR).search(text)


def _explicit_minimum(text: str) -> re.Match[str] | None:
    return _MINIMUM.search(text)


def _classify_line(text: str) -> _ClaimMatch | None:
    metric = _METRIC_VAGUE.search(text)
    if metric is not None:
        explicit = _METRIC_PAIR.search(text)
        threshold = _MINIMUM.search(text)
        confidence = 0.95 if explicit is not None else 0.90 if threshold else 0.70
        return _ClaimMatch(
            claim_type=ClaimType.METRIC_IMPROVEMENT,
            subject="candidate",
            metric=metric.group("metric").casefold(),
            direction=ClaimDirection.HIGHER,
            confidence=confidence,
            pattern="metric_explicit" if explicit else "metric_threshold" if threshold else "metric_vague",
        )

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


def _claimed_values(
    text: str,
    provenance: FieldProvenance,
    *,
    require_metric: bool,
) -> tuple[ClaimedValue | None, ClaimedValue | None, ClaimedValue | None]:
    pair = _explicit_pair(text, require_metric=require_metric)
    baseline = candidate = None
    if pair is not None:
        baseline = ClaimedValue(
            float(pair.group("baseline")),
            _normalized_unit(pair.group("baseline_unit")),
            provenance,
        )
        candidate = ClaimedValue(
            float(pair.group("candidate")),
            _normalized_unit(pair.group("candidate_unit")),
            provenance,
        )
    threshold = _explicit_minimum(text)
    minimum = None
    if threshold is not None:
        minimum = ClaimedValue(
            float(threshold.group("value")),
            _normalized_unit(threshold.group("unit")),
            provenance,
        )
    return baseline, candidate, minimum


def _deterministic_claims(context: RepositoryContext) -> list[DiscoveredClaim]:
    claims: list[DiscoveredClaim] = []
    for record in context.source_bundle.sources:
        lines = record.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for line_number, source_text in enumerate(lines, start=1):
            if not source_text.strip():
                continue
            matched = _classify_line(source_text)
            if matched is None:
                continue
            provenance = _provenance(
                record,
                kind=ProvenanceKind.DETERMINISTIC_DISCOVERY,
                detail=f"deterministic claim pattern: {matched.pattern}",
            )
            baseline, candidate, minimum = _claimed_values(
                source_text,
                provenance,
                require_metric=True,
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
            claims.append(
                DiscoveredClaim(
                    reference=reference,
                    claim_type=matched.claim_type,
                    subject=matched.subject,
                    source=source,
                    metric=matched.metric,
                    direction=matched.direction,
                    baseline_value=baseline,
                    candidate_value=candidate,
                    minimum_improvement=minimum,
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
        baseline, candidate, minimum = _claimed_values(
            claim.source_text,
            provenance,
            require_metric=False,
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
        discovered.append(
            DiscoveredClaim(
                reference=reference,
                claim_type=claim.claim_type,
                subject=claim.subject,
                source=claim.source,
                metric=claim.metric,
                direction=claim.direction,
                baseline_value=baseline,
                candidate_value=candidate,
                minimum_improvement=minimum,
                qualifiers=claim.qualifiers,
                evidence_hints=tuple(hints),
            )
        )
    return discovered


def _dedupe_key(claim: DiscoveredClaim) -> tuple[str, ClaimType, str, int]:
    return (
        _WHITESPACE.sub(" ", claim.reference.text.strip().casefold()),
        claim.claim_type,
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
    unique: dict[tuple[str, ClaimType, str, int], DiscoveredClaim] = {}
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
