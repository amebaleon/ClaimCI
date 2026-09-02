"""Validated claim conversion and one-call advisory analysis synthesis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping

from claimci.analysis.claim_types import (
    ClaimTypeContractError,
    MetricImprovementClaim,
    UnsupportedDeterministicClaimCompiler,
    compile_audit_claim,
    recover_scientific_claim,
)
from claimci.analysis.claims import audit_relevant_claim_projection
from claimci.analysis.confidence import Confidence
from claimci.analysis.contracts import (
    AdvisoryResearchInterpretation,
    AuditClaimSpec,
    ClaimReference,
    DeterministicAuditOutcome,
    FieldProvenance,
    MissingEvidence,
    ProvenanceKind,
    RepositoryPath,
    to_jsonable,
)
from claimci.models import Direction
from claimci.parsing import unique_json_object

from .models import (
    ClaimDirection,
    ClaimType,
    MagnitudeKind,
    ReviewConfig,
    ReviewError,
    ScientificClaim,
    SourceBundle,
)
from .provider import ProviderResponse, ReviewerProvider, StructuredRequest


_ANALYSIS_SYNTHESIS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "interpretations",
        "missing_evidence",
        "confidence",
    ],
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 16_000},
        "interpretations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {"type": "string", "minLength": 1, "maxLength": 16_000},
        },
        "missing_evidence": {
            "type": "array",
            "maxItems": 64,
            "items": {"type": "string", "minLength": 1, "maxLength": 4_096},
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


@dataclass(frozen=True, slots=True)
class AnalysisReviewContext:
    """Immutable inputs for one advisory synthesis over an existing plan lane."""

    claim: ClaimReference
    audit_claim: AuditClaimSpec
    evidence_provenance: tuple[FieldProvenance, ...]
    deterministic: DeterministicAuditOutcome | None
    missing_evidence: tuple[MissingEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.claim) is not ClaimReference:
            raise TypeError("analysis review claim must be ClaimReference")
        if type(self.audit_claim) is not AuditClaimSpec:
            raise TypeError("analysis review audit_claim must be AuditClaimSpec")
        if self.claim.claim_id != self.audit_claim.claim_id:
            raise ReviewError("analysis review claim identifiers must match")
        if not isinstance(self.evidence_provenance, tuple) or not all(
            type(item) is FieldProvenance for item in self.evidence_provenance
        ):
            raise TypeError(
                "analysis review evidence_provenance must be FieldProvenance values"
            )
        if self.deterministic is not None and type(
            self.deterministic
        ) is not DeterministicAuditOutcome:
            raise TypeError(
                "analysis review deterministic input must be a real snapshot"
            )
        if not isinstance(self.missing_evidence, tuple) or not all(
            type(item) is MissingEvidence for item in self.missing_evidence
        ):
            raise TypeError(
                "analysis review missing_evidence must be MissingEvidence values"
            )
        if self.deterministic is None and not self.missing_evidence:
            raise ReviewError(
                "analysis review requires deterministic evidence or a typed gap"
            )
        if self.deterministic is not None:
            claim_payload = self.deterministic.payload.get("claim")
            if not isinstance(claim_payload, Mapping):
                raise ReviewError("deterministic snapshot claim policy is unavailable")
            direction = claim_payload.get("direction", Direction.HIGHER.value)
            if (
                claim_payload.get("metric") != self.audit_claim.metric
                or claim_payload.get("minimum_improvement")
                != self.audit_claim.minimum_absolute_improvement
                or direction != self.audit_claim.direction.value
            ):
                raise ReviewError(
                    "deterministic snapshot does not match the validated audit claim"
                )


def _source_quote(claim: ScientificClaim, sources: SourceBundle) -> str:
    records = tuple(
        item for item in sources.sources if item.source_id == claim.source.source_id
    )
    if len(records) != 1:
        raise ReviewError("scientific claim cites an unknown or duplicate source")
    record = records[0]
    if record.kind is not claim.source.kind or record.path != claim.source.path:
        raise ReviewError("scientific claim source identity does not match")
    lines = record.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if claim.source.end_line > len(lines):
        raise ReviewError("scientific claim source span is outside the source")
    return "\n".join(
        lines[claim.source.start_line - 1 : claim.source.end_line]
    )


def validate_scientific_claim_for_audit(
    claim: ScientificClaim,
    sources: SourceBundle,
) -> AuditClaimSpec:
    """Validate one provider-originated claim into a claim-to-test contract."""

    if type(claim) is not ScientificClaim:
        raise TypeError("audit claim conversion requires ScientificClaim")
    if type(sources) is not SourceBundle:
        raise TypeError("audit claim conversion requires SourceBundle")
    if claim.source_text != _source_quote(claim, sources):
        raise ReviewError("scientific claim source quote does not match")
    source_path = (
        None if claim.source.path is None else RepositoryPath(claim.source.path)
    )
    provenance = FieldProvenance(
        ProvenanceKind.PROVIDER_PROPOSAL,
        "provider claim validated against an issued source span",
        source_path,
        claim.source.source_id,
    )
    reference = ClaimReference(
        claim_id=claim.claim_id,
        text=claim.source_text,
        source_path=source_path,
        confidence=Confidence(claim.confidence),
        provenance=provenance,
    )
    canonical = recover_scientific_claim(reference)
    if canonical is None:
        raise ReviewError("scientific claim source semantics are ambiguous")
    try:
        compiled = compile_audit_claim(canonical)
    except UnsupportedDeterministicClaimCompiler as error:
        raise ReviewError(str(error)) from error
    except ClaimTypeContractError as error:
        raise ReviewError(f"scientific claim source semantics are invalid: {error}") from error
    if type(canonical.primary) is not MetricImprovementClaim:
        raise ReviewError("deterministic planner supports metric improvement claims")
    if claim.claim_type is not ClaimType.METRIC_IMPROVEMENT:
        raise ReviewError("scientific claim type conflicts with its source")
    if claim.metric is None or claim.metric.casefold() != compiled.metric:
        raise ReviewError("scientific claim metric conflicts with its source")
    direction = {
        ClaimDirection.HIGHER: Direction.HIGHER,
        ClaimDirection.LOWER: Direction.LOWER,
    }.get(claim.direction)
    if direction is None or direction is not compiled.direction:
        raise ReviewError("scientific claim direction conflicts with its source")
    if compiled.minimum_absolute_improvement is not None:
        magnitude = claim.claimed_magnitude
        if (
            magnitude is None
            or magnitude.kind is not MagnitudeKind.ABSOLUTE
            or magnitude.value is None
            or magnitude.unit is not None
            or not math.isfinite(float(magnitude.value))
            or float(magnitude.value) != compiled.minimum_absolute_improvement
            or magnitude.raw not in claim.source_text
        ):
            raise ReviewError(
                "scientific claim threshold magnitude conflicts with its source"
            )
    return compiled


def _strict_json(text: str) -> object:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON value {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_json_object,
            parse_constant=reject_constant,
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        raise ReviewError(f"analysis synthesis output is invalid JSON: {error}") from error


def _bounded_strings(
    value: object,
    label: str,
    *,
    maximum_count: int,
    maximum_chars: int,
    require_one: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_count:
        raise ReviewError(f"analysis synthesis {label} must be a bounded list")
    if require_one and not value:
        raise ReviewError(f"analysis synthesis {label} must not be empty")
    if not all(
        isinstance(item, str) and item.strip() and len(item) <= maximum_chars
        for item in value
    ):
        raise ReviewError(f"analysis synthesis {label} contains invalid text")
    if any(
        any(0xD800 <= ord(character) <= 0xDFFF for character in item)
        for item in value
    ):
        raise ReviewError(f"analysis synthesis {label} contains invalid Unicode")
    return tuple(value)


def _parse_advisory(value: object) -> AdvisoryResearchInterpretation:
    expected = {
        "summary",
        "interpretations",
        "missing_evidence",
        "confidence",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ReviewError(
            "analysis synthesis output fields do not match the advisory schema"
        )
    summary = value["summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 16_000:
        raise ReviewError("analysis synthesis summary is invalid")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in summary):
        raise ReviewError("analysis synthesis summary contains invalid Unicode")
    confidence = value["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
    ):
        raise ReviewError("analysis synthesis confidence is invalid")
    try:
        bounded_confidence = Confidence(confidence)
    except (TypeError, ValueError) as error:
        raise ReviewError("analysis synthesis confidence is invalid") from error
    return AdvisoryResearchInterpretation(
        summary=summary,
        interpretations=_bounded_strings(
            value["interpretations"],
            "interpretations",
            maximum_count=16,
            maximum_chars=16_000,
            require_one=True,
        ),
        missing_evidence=_bounded_strings(
            value["missing_evidence"],
            "missing_evidence",
            maximum_count=64,
            maximum_chars=4_096,
        ),
        confidence=bounded_confidence,
    )


def run_analysis_review(
    context: AnalysisReviewContext,
    config: ReviewConfig,
    *,
    provider: ReviewerProvider | None = None,
) -> AdvisoryResearchInterpretation:
    """Perform exactly one advisory synthesis without extraction or re-Audit."""

    if type(context) is not AnalysisReviewContext:
        raise TypeError("analysis review requires AnalysisReviewContext")
    if type(config) is not ReviewConfig:
        raise TypeError("analysis review requires ReviewConfig")
    if not config.enabled:
        raise ReviewError("analysis review is disabled")
    if provider is None:
        if config.provider != "openai":
            raise ReviewError("configured analysis review provider is unavailable")
        from .openai_provider import OpenAIReviewerProvider

        provider = OpenAIReviewerProvider(
            model=config.model,
            timeout_seconds=config.limits.timeout_seconds,
            max_output_tokens=config.limits.synthesis_max_output_tokens,
        )
    deterministic = None
    if context.deterministic is not None:
        serialized = to_jsonable(context.deterministic)
        if not isinstance(serialized, dict):
            raise ReviewError("deterministic snapshot serialization is invalid")
        deterministic_payload = serialized.get("payload")
        if not isinstance(deterministic_payload, dict):
            raise ReviewError("deterministic snapshot payload is invalid")
        deterministic_payload.pop("manifest", None)
        deterministic = serialized
    payload = {
        "policy": {
            "mode": "advisory",
            "untrusted_content": True,
            "deterministic_authority": "read_only",
            "tools": "none",
            "compatibility_manifest": "internal_not_user_authored",
        },
        "claim": to_jsonable(context.claim),
        "audit_claim": audit_relevant_claim_projection(context.audit_claim),
        "evidence_provenance": to_jsonable(context.evidence_provenance),
        "deterministic": deterministic,
        "missing_evidence": to_jsonable(context.missing_evidence),
    }
    try:
        context_chars = len(
            json.dumps(
                {"payload": payload, "schema": _ANALYSIS_SYNTHESIS_SCHEMA},
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ReviewError(f"analysis review context is not serializable: {error}") from error
    if context_chars > config.limits.max_context_chars:
        raise ReviewError("analysis review context exceeds the configured limit")
    request = StructuredRequest(
        task="synthesize_review",
        payload=payload,
        schema=_ANALYSIS_SYNTHESIS_SCHEMA,
        max_output_tokens=config.limits.synthesis_max_output_tokens,
    )
    try:
        response = provider.synthesize_review(request)
    except Exception as error:
        raise ReviewError(
            f"analysis synthesis provider is unavailable ({type(error).__name__})"
        ) from error
    if type(response) is not ProviderResponse:
        raise ReviewError("analysis synthesis provider returned an invalid response")
    if not response.complete:
        if response.incomplete_reason == "max_output_tokens":
            raise ReviewError("analysis synthesis reached the provider output-token limit")
        raise ReviewError("analysis synthesis did not return a complete structured response")
    if len(response.output_text) > config.limits.max_output_chars:
        raise ReviewError("analysis synthesis output exceeds the configured limit")
    return _parse_advisory(_strict_json(response.output_text))


__all__ = [
    "AnalysisReviewContext",
    "run_analysis_review",
    "validate_scientific_claim_for_audit",
]
