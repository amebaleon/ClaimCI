"""Fixed two-call research-review orchestration."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, DecimalException
from enum import Enum
from pathlib import Path
from typing import Any

from claimci.parsing import unique_json_object

from .evidence import EvidenceBundle, discover_evidence
from .models import (
    ClaimType,
    ProviderCallRecord,
    ProviderUsage,
    ReviewConfig,
    ReviewError,
    ReviewStatus,
    ScientificClaim,
    SourceBundle,
)
from .openai_provider import OpenAIReviewerProvider
from .provider import (
    ProviderResponse,
    REVIEW_SYSTEM_POLICY,
    ReviewerProvider,
    StructuredRequest,
)
from .sources import (
    collect_review_sources,
    validate_claim_candidates_best_effort,
)
from .tools import (
    DeterministicAuditSnapshot,
    ManifestAuditBundle,
    ManifestAuditPlan,
    discover_manifests,
    plan_manifest_audits,
    run_manifest_audits,
    select_relevant_manifest_audit_plans,
)


MAX_PROVIDER_JSON_DEPTH = 64


@dataclass(frozen=True)
class ReviewInputs:
    repository_root: Path
    base_root: Path | None = None
    pr_title: str = ""
    pr_description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path):
            object.__setattr__(self, "repository_root", Path(self.repository_root))
        if self.base_root is not None and not isinstance(self.base_root, Path):
            object.__setattr__(self, "base_root", Path(self.base_root))
        if not isinstance(self.pr_title, str) or not isinstance(self.pr_description, str):
            raise ReviewError("pull-request title and description must be strings")


@dataclass(frozen=True)
class ClaimInterpretation:
    claim_id: str
    interpretation: str
    citations: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    unsupported_inferences: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class ResearchReview:
    status: ReviewStatus
    claims: tuple[ScientificClaim, ...] = ()
    interpretations: tuple[ClaimInterpretation, ...] = ()
    evidence: EvidenceBundle = EvidenceBundle()
    deterministic_audits: tuple[DeterministicAuditSnapshot, ...] = ()
    provider_calls: tuple[ProviderCallRecord, ...] = ()
    usage: ProviderUsage = ProviderUsage()
    error_code: str | None = None
    error_message: str | None = None


_EXTRACTION_CONTRACT: dict[str, str] = {
    "claim_selection": (
        "Extract only explicit, scientifically verifiable statements present "
        "in an issued source. Do not emit duplicate claims."
    ),
    "claim_prioritization": (
        "Return at most max_claims material claims. Prioritize claims affecting "
        "correctness, benchmark or performance conclusions, experimental fairness, "
        "production or deployment conclusions, and causal conclusions. Ignore minor "
        "implementation statements unless they materially support one of those claims."
    ),
    "source_location": (
        "Use only an issued source_id and 1-based inclusive start_line/end_line "
        "values that are inside that source."
    ),
    "source_text": (
        "Copy exactly the complete source line or consecutive complete source "
        "lines selected by start_line/end_line, including punctuation; do not "
        "paraphrase, trim, join partial lines, or return a substring."
    ),
    "normalized_fields": (
        "Use non-empty strings for subject and every present metric, magnitude "
        "text/unit, qualifier, and evidence hint. Use null or [] instead of "
        "empty strings."
    ),
    "authority": (
        "Do not emit verdicts, findings, severity, impact, thresholds, or "
        "deterministic evidence."
    ),
}


_SYNTHESIS_CONTRACT: dict[str, str] = {
    "claim_coverage": (
        "Return exactly one interpretation for every issued claim_id, with no "
        "unknown, omitted, or duplicate claim IDs."
    ),
    "citations": (
        "Citations may contain only issued evidence_id values assigned to that claim. "
        "For each claim_id, copy only from that claim's list in "
        "evidence_ids_by_claim_id, and use an empty list when its allowed list is "
        "empty. Do not cite rule IDs, paths, manifests, deterministic finding IDs, "
        "or an evidence_id assigned to another claim."
    ),
    "missing_evidence": (
        "When issued evidence does not support a statement, use an empty list for "
        "citations and describe the gap in missing_evidence or unsupported_inferences. "
        "Scope every gap to the analyzed snapshot; never claim that evidence is absent "
        "from the original repository."
    ),
    "evidence_provenance": (
        "reported_measurement is summary support, not proof of execution. "
        "executable_benchmark_definition describes a runnable method, not a completed "
        "run. Only executed_result_artifact represents an available executed-result "
        "artifact; never upgrade another provenance category."
    ),
    "authority": (
        "Produce advisory interpretation only. Deterministic audit snapshots are "
        "read-only authority and must not be rewritten, upgraded, or contradicted."
    ),
}


_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_text",
                    "claim_type",
                    "subject",
                    "metric",
                    "direction",
                    "claimed_magnitude",
                    "qualifiers",
                    "source",
                    "confidence",
                    "evidence_hints",
                ],
                "properties": {
                    "source_text": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "metric_improvement",
                            "compute_equivalence",
                            "held_out_evaluation",
                            "resource_reduction",
                            "component_causality",
                            "no_external_reward",
                            "implementation_claim",
                            "other_scientific",
                        ],
                    },
                    "subject": {"type": "string"},
                    "metric": {"type": ["string", "null"]},
                    "direction": {
                        "type": "string",
                        "enum": ["higher", "lower", "not_applicable"],
                    },
                    "claimed_magnitude": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["raw", "value", "unit", "kind"],
                                "properties": {
                                    "raw": {"type": "string"},
                                    "value": {"type": ["number", "null"]},
                                    "unit": {"type": ["string", "null"]},
                                    "kind": {
                                        "type": "string",
                                        "enum": ["absolute", "relative", "unspecified"],
                                    },
                                },
                            },
                        ]
                    },
                    "qualifiers": {"type": "array", "items": {"type": "string"}},
                    "source": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_id", "start_line", "end_line"],
                        "properties": {
                            "source_id": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_hints": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def _extraction_schema(max_claims: int) -> dict[str, Any]:
    if (
        isinstance(max_claims, bool)
        or not isinstance(max_claims, int)
        or not 1 <= max_claims <= 16
    ):
        raise ReviewError("max_claims must be an integer from 1 through 16")
    schema = copy.deepcopy(_EXTRACTION_SCHEMA)
    schema["properties"]["claims"]["maxItems"] = max_claims
    return schema


_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["interpretations"],
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "claim_id",
                    "interpretation",
                    "citations",
                    "missing_evidence",
                    "unsupported_inferences",
                    "confidence",
                ],
                "properties": {
                    "claim_id": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "missing_evidence": {"type": "array", "items": {"type": "string"}},
                    "unsupported_inferences": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}


def _evidence_ids_by_claim_id(
    claims: Sequence[ScientificClaim],
    evidence: EvidenceBundle,
) -> dict[str, list[str]]:
    return {
        claim.claim_id: sorted(
            reference.evidence_id
            for reference in evidence.references
            if claim.claim_id in reference.claim_ids
        )
        for claim in claims
    }


def _synthesis_schema(
    claims: Sequence[ScientificClaim],
    evidence_ids_by_claim_id: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Constrain each row to issued IDs using the provider-supported subset."""

    schema = copy.deepcopy(_SYNTHESIS_SCHEMA)
    interpretations = schema["properties"]["interpretations"]
    interpretations["minItems"] = len(claims)
    interpretations["maxItems"] = len(claims)
    alternatives: list[dict[str, Any]] = []
    row_template = _SYNTHESIS_SCHEMA["properties"]["interpretations"]["items"]
    for claim in claims:
        row = copy.deepcopy(row_template)
        row["properties"]["claim_id"] = {
            "type": "string",
            "enum": [claim.claim_id],
        }
        allowed = list(evidence_ids_by_claim_id[claim.claim_id])
        citations: dict[str, Any] = {
            "type": "array",
            "maxItems": len(allowed),
            "items": {"type": "string"},
        }
        if allowed:
            citations["items"]["enum"] = allowed
        row["properties"]["citations"] = citations
        alternatives.append(row)
    if alternatives:
        interpretations["items"] = {"anyOf": alternatives}
    return schema


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _serialized_chars(value: object) -> int:
    return len(
        json.dumps(
            _plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _request_chars(task: str, payload: object, schema: object) -> int:
    """Count the complete logical provider context, not only repository data."""

    return _serialized_chars(
        {
            "system_policy": REVIEW_SYSTEM_POLICY,
            "task": task,
            "input": payload,
            "strict_response_schema": schema,
        }
    )


def _reject_constant(token: str) -> None:
    raise ReviewError(f"provider output contains non-finite number {token}")


def _validate_depth(value: object, depth: int = 0) -> None:
    if depth > MAX_PROVIDER_JSON_DEPTH:
        raise ReviewError("provider output exceeds maximum nesting depth")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReviewError("provider output object keys must be strings")
            _validate_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_depth(item, depth + 1)


def _parse_json(text: str) -> object:
    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_json_object,
            parse_constant=_reject_constant,
        )
        _validate_depth(value)
        return value
    except ReviewError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        raise ReviewError("provider returned malformed structured output") from exc


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 64:
        raise ReviewError(f"{label} must be a bounded list")
    if not all(isinstance(item, str) and item.strip() and len(item) <= 4_096 for item in value):
        raise ReviewError(f"{label} contains an invalid string")
    if any(
        any(0xD800 <= ord(character) <= 0xDFFF for character in item)
        for item in value
    ):
        raise ReviewError(f"{label} contains an invalid Unicode surrogate")
    return tuple(value)


def _parse_interpretations(
    value: object,
    claims: tuple[ScientificClaim, ...],
    evidence: EvidenceBundle,
) -> tuple[ClaimInterpretation, ...]:
    if not isinstance(value, Mapping) or set(value) != {"interpretations"}:
        raise ReviewError("synthesis response must contain only interpretations")
    rows = value["interpretations"]
    if not isinstance(rows, list) or len(rows) > len(claims):
        raise ReviewError("synthesis interpretations must be a bounded list")
    claim_ids = {claim.claim_id for claim in claims}
    evidence_claims = {
        item.evidence_id: frozenset(item.claim_ids)
        for item in evidence.references
    }
    expected = {
        "claim_id",
        "interpretation",
        "citations",
        "missing_evidence",
        "unsupported_inferences",
        "confidence",
    }
    seen: set[str] = set()
    parsed: list[ClaimInterpretation] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ReviewError("synthesis interpretation fields are invalid")
        claim_id = row["claim_id"]
        if not isinstance(claim_id, str) or claim_id not in claim_ids or claim_id in seen:
            raise ReviewError("synthesis cites an unknown or duplicate claim")
        interpretation = row["interpretation"]
        if not isinstance(interpretation, str) or not interpretation.strip() or len(interpretation) > 8_000:
            raise ReviewError("synthesis interpretation text is invalid")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in interpretation):
            raise ReviewError("synthesis interpretation contains an invalid Unicode surrogate")
        citations = _string_tuple(row["citations"], "synthesis citations")
        if any(
            citation not in evidence_claims
            or claim_id not in evidence_claims[citation]
            for citation in citations
        ):
            raise ReviewError(
                "synthesis cites unknown or claim-mismatched repository evidence"
            )
        confidence = row["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ReviewError("synthesis confidence is invalid")
        parsed.append(
            ClaimInterpretation(
                claim_id=claim_id,
                interpretation=interpretation,
                citations=citations,
                missing_evidence=_string_tuple(row["missing_evidence"], "missing evidence"),
                unsupported_inferences=_string_tuple(
                    row["unsupported_inferences"], "unsupported inferences"
                ),
                confidence=float(confidence),
            )
        )
        seen.add(claim_id)
    if seen != claim_ids:
        raise ReviewError("synthesis must interpret every accepted claim exactly once")
    return tuple(parsed)


def _aggregate_usage(calls: Sequence[ProviderCallRecord]) -> ProviderUsage:
    def total(field: str) -> int | None:
        values = [getattr(call.usage, field) for call in calls]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    totals_are_consistent = all(
        call.usage.input_tokens is not None
        and call.usage.output_tokens is not None
        and call.usage.total_tokens
        == call.usage.input_tokens + call.usage.output_tokens
        for call in calls
    )

    costs = [call.usage.estimated_cost_usd for call in calls]
    aggregate_cost: Decimal | None = None
    if costs and all(cost is not None for cost in costs):
        try:
            aggregate_cost = sum(
                (cost for cost in costs if cost is not None),
                Decimal(0),
            )
        except DecimalException:
            # Cost is observability only. Decimal context overflow must never
            # change review status or escape the controlled failure boundary.
            aggregate_cost = None
    return ProviderUsage(
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        total_tokens=total("total_tokens") if totals_are_consistent else None,
        estimated_cost_usd=aggregate_cost,
    )


def _result(
    status: ReviewStatus,
    *,
    claims: tuple[ScientificClaim, ...] = (),
    interpretations: tuple[ClaimInterpretation, ...] = (),
    evidence: EvidenceBundle = EvidenceBundle(),
    deterministic_audits: tuple[DeterministicAuditSnapshot, ...] = (),
    calls: Sequence[ProviderCallRecord] = (),
    error_code: str | None = None,
    error_message: str | None = None,
) -> ResearchReview:
    call_tuple = tuple(calls)
    return ResearchReview(
        status=status,
        claims=claims,
        interpretations=interpretations,
        evidence=evidence,
        deterministic_audits=deterministic_audits,
        provider_calls=call_tuple,
        usage=_aggregate_usage(call_tuple),
        error_code=error_code,
        error_message=error_message,
    )


def _record_call(
    task: str,
    response: ProviderResponse,
    input_chars: int,
) -> ProviderCallRecord:
    if not isinstance(response, ProviderResponse):
        raise ReviewError("provider returned an invalid response envelope")
    return ProviderCallRecord(
        task=task,
        provider=response.provider,
        model=response.model,
        request_id=response.request_id,
        usage=response.usage,
        input_chars=input_chars,
        output_chars=len(response.output_text),
    )


def _sources_after_manifest_reservation(
    sources: SourceBundle,
    audit_plans: Sequence[ManifestAuditPlan],
    *,
    max_files: int,
) -> SourceBundle:
    """Keep metadata and only repository sources left after audit priority."""

    selected = {
        path
        for plan in audit_plans
        for path in plan.paths
    }
    kept = []
    for source in sources.sources:
        if source.path is None:
            kept.append(source)
            continue
        if source.path in selected:
            kept.append(source)
            continue
        if len(selected) >= max_files:
            continue
        kept.append(source)
        selected.add(source.path)
    return SourceBundle(
        sources=tuple(kept),
        repository_paths=sources.repository_paths,
        changed_paths=sources.changed_paths,
        total_chars=sum(len(source.text) for source in kept),
    )


def _manifest_priority_paths(
    claims: Sequence[ScientificClaim],
    audit_bundles: Sequence[ManifestAuditBundle],
    *,
    repository_paths: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Associate trusted manifest inputs with claims they can actually inform."""

    indexed = set(repository_paths)
    routed: dict[str, tuple[str, ...]] = {}
    generally_auditable = {
        ClaimType.METRIC_IMPROVEMENT,
        ClaimType.COMPUTE_EQUIVALENCE,
        ClaimType.HELD_OUT_EVALUATION,
        ClaimType.COMPONENT_CAUSALITY,
    }
    for claim in claims:
        paths: list[str] = []
        for bundle in audit_bundles:
            metric_matches = (
                claim.metric is not None
                and claim.metric.casefold() == bundle.snapshot.metric.casefold()
            )
            if not metric_matches and claim.claim_type not in generally_auditable:
                continue
            for path in bundle.paths:
                if path in indexed and path not in paths:
                    paths.append(path)
        if paths:
            routed[claim.claim_id] = tuple(paths)
    return routed


def run_review(
    inputs: ReviewInputs,
    config: ReviewConfig,
    *,
    provider: ReviewerProvider | None = None,
) -> ResearchReview:
    """Run the fixed two-call advisory review state machine."""

    if not isinstance(config, ReviewConfig):
        raise ReviewError("config must be ReviewConfig")
    if not config.enabled:
        return _result(ReviewStatus.DISABLED)
    if not isinstance(inputs, ReviewInputs):
        raise ReviewError("inputs must be ReviewInputs")
    if provider is None:
        if config.provider != "openai":
            return _result(
                ReviewStatus.UNAVAILABLE,
                error_code="PROVIDER_UNSUPPORTED",
                error_message="Configured review provider is not available.",
            )
        provider = OpenAIReviewerProvider(
            model=config.model,
            timeout_seconds=config.limits.timeout_seconds,
            max_output_tokens=max(
                config.limits.extraction_max_output_tokens,
                config.limits.synthesis_max_output_tokens,
            ),
        )

    calls: list[ProviderCallRecord] = []
    try:
        sources = collect_review_sources(
            inputs.repository_root,
            base_root=inputs.base_root,
            pr_title=inputs.pr_title,
            pr_description=inputs.pr_description,
            limits=config.limits,
        )
        manifest_candidates = discover_manifests(
            inputs.repository_root,
            sources.repository_paths,
            max_manifests=min(4, config.limits.max_files),
        )
        audit_plans = select_relevant_manifest_audit_plans(
            plan_manifest_audits(
                inputs.repository_root,
                manifest_candidates,
                limits=config.limits,
            ),
            changed_paths=sources.changed_paths,
        )
        manifest_candidates = tuple(
            plan.manifest_path for plan in audit_plans
        )
        sources = _sources_after_manifest_reservation(
            sources,
            audit_plans,
            max_files=config.limits.max_files,
        )
        extraction_schema = _extraction_schema(config.limits.max_claims)
        extraction_payload = {
            "policy": {
                "mode": "advisory",
                "untrusted_content": True,
                "deterministic_authority": "ClaimCI Audit only",
            },
            "extraction_contract": dict(_EXTRACTION_CONTRACT),
            "max_claims": config.limits.max_claims,
            "sources": _plain(sources.sources),
            "repository_paths": _plain(sources.repository_paths),
        }
        extraction_chars = _request_chars(
            "extract_claims", extraction_payload, extraction_schema
        )
        if extraction_chars > config.limits.max_context_chars:
            return _result(
                ReviewStatus.UNAVAILABLE,
                error_code="CONTEXT_LIMIT",
                error_message="Extraction context exceeds the configured limit.",
            )
        extraction_request = StructuredRequest(
            task="extract_claims",
            payload=extraction_payload,
            schema=extraction_schema,
            max_output_tokens=config.limits.extraction_max_output_tokens,
        )
        extraction_response = provider.extract_claims(extraction_request)
        extraction_call = _record_call(
            "extract_claims", extraction_response, extraction_chars
        )
        calls.append(extraction_call)
        if not extraction_response.complete:
            truncated = extraction_response.incomplete_reason == "max_output_tokens"
            return _result(
                ReviewStatus.PARTIAL,
                calls=calls,
                error_code=(
                    "CLAIM_EXTRACTION_TRUNCATED"
                    if truncated
                    else "CLAIM_EXTRACTION_INCOMPLETE"
                ),
                error_message=(
                    "Claim extraction reached the provider output-token limit before "
                    "a complete structured response was returned."
                    if truncated
                    else "Claim extraction did not return a complete structured response."
                ),
            )
        if extraction_call.output_chars > config.limits.max_output_chars:
            raise ReviewError("provider extraction output exceeds configured limit")
        claim_validation = validate_claim_candidates_best_effort(
            _parse_json(extraction_response.output_text),
            sources,
            max_claims=config.limits.max_claims,
        )
        claims = claim_validation.claims
        rejected_claim_candidates = claim_validation.rejected_count
        if rejected_claim_candidates and not claims:
            raise ReviewError(
                "all extracted claim candidates failed deterministic source validation"
            )
        deterministic_audits = run_manifest_audits(
            inputs.repository_root,
            manifest_candidates,
            limits=config.limits,
        )
        plans_by_manifest = {
            plan.manifest_path: plan for plan in audit_plans
        }
        audit_bundles = tuple(
            ManifestAuditBundle(snapshot=snapshot, paths=plan.paths)
            for snapshot in deterministic_audits
            if (plan := plans_by_manifest.get(snapshot.manifest_path)) is not None
        )
        selected_paths = {
            source.path for source in sources.sources if source.path is not None
        }
        selected_paths.update(
            path
            for bundle in audit_bundles
            for path in bundle.paths
            if path in sources.repository_paths
        )
        priority_paths = _manifest_priority_paths(
            claims,
            audit_bundles,
            repository_paths=sources.repository_paths,
        )
        evidence = discover_evidence(
            inputs.repository_root,
            claims,
            sources.repository_paths,
            limits=config.limits,
            priority_paths=priority_paths,
            selected_paths=tuple(sorted(selected_paths)),
            changed_paths=sources.changed_paths,
        )
        if config.limits.max_calls < 2:
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="CALL_LIMIT",
                error_message="Synthesis was skipped by the configured call limit.",
            )
        evidence_ids_by_claim_id = _evidence_ids_by_claim_id(claims, evidence)
        synthesis_schema = _synthesis_schema(claims, evidence_ids_by_claim_id)
        synthesis_payload = {
            "policy": {
                "mode": "advisory",
                "deterministic_authority": "read_only",
            },
            "synthesis_contract": dict(_SYNTHESIS_CONTRACT),
            "claims": _plain(claims),
            "evidence": _plain(evidence.references),
            "evidence_ids_by_claim_id": evidence_ids_by_claim_id,
            "missing_evidence": _plain(evidence.missing),
            "deterministic_audits": _plain(deterministic_audits),
        }
        synthesis_chars = _request_chars(
            "synthesize_review", synthesis_payload, synthesis_schema
        )
        if extraction_chars + synthesis_chars > config.limits.max_context_chars:
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="CONTEXT_LIMIT",
                error_message="Synthesis context exceeds the configured limit.",
            )
        synthesis_request = StructuredRequest(
            task="synthesize_review",
            payload=synthesis_payload,
            schema=synthesis_schema,
            max_output_tokens=config.limits.synthesis_max_output_tokens,
        )
        synthesis_response = provider.synthesize_review(synthesis_request)
        synthesis_call = _record_call(
            "synthesize_review", synthesis_response, synthesis_chars
        )
        calls.append(synthesis_call)
        if not synthesis_response.complete:
            truncated = synthesis_response.incomplete_reason == "max_output_tokens"
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code=("SYNTHESIS_TRUNCATED" if truncated else "SYNTHESIS_INCOMPLETE"),
                error_message=(
                    "Review synthesis reached the provider output-token limit before "
                    "a complete structured response was returned."
                    if truncated
                    else "Review synthesis did not return a complete structured response."
                ),
            )
        if sum(call.output_chars for call in calls) > config.limits.max_output_chars:
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="SYNTHESIS_OUTPUT_LIMIT",
                error_message=(
                    "Review synthesis exceeded the configured aggregate output limit; "
                    "no synthesis interpretation was accepted."
                ),
            )
        try:
            interpretations = _parse_interpretations(
                _parse_json(synthesis_response.output_text), claims, evidence
            )
        except ReviewError as exc:
            validation_message = (
                "Review synthesis failed deterministic citation ownership validation; "
                "no synthesis interpretation was accepted."
                if "claim-mismatched repository evidence" in str(exc)
                else (
                    "Review synthesis failed deterministic claim coverage validation; "
                    "no synthesis interpretation was accepted."
                    if (
                        "unknown or duplicate claim" in str(exc)
                        or "interpret every accepted claim" in str(exc)
                    )
                    else (
                        "Review synthesis returned complete output that failed "
                        "deterministic claim or citation validation; no synthesis "
                        "interpretation was accepted."
                    )
                )
            )
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="SYNTHESIS_INVALID",
                error_message=validation_message,
            )
        if rejected_claim_candidates:
            return _result(
                ReviewStatus.PARTIAL,
                claims=claims,
                interpretations=interpretations,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="CLAIM_CANDIDATES_REJECTED",
                error_message=(
                    f"{rejected_claim_candidates} extracted claim candidate(s) failed "
                    "deterministic source validation and were excluded; every retained "
                    "claim was reviewed."
                ),
            )
        return _result(
            ReviewStatus.COMPLETE,
            claims=claims,
            interpretations=interpretations,
            evidence=evidence,
            deterministic_audits=deterministic_audits,
            calls=calls,
        )
    except Exception as exc:
        # Provider/model text is never included in the safe status message.
        return _result(
            ReviewStatus.UNAVAILABLE,
            calls=calls,
            error_code="REVIEW_UNAVAILABLE",
            error_message=f"Research review is unavailable ({type(exc).__name__}).",
        )
