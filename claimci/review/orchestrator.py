"""Fixed two-call research-review orchestration."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, DecimalException
from pathlib import Path, PurePosixPath
from typing import Any

from claimci.parsing import unique_json_object

from .evidence import (
    EvidenceBundle,
    EvidenceKind,
    EvidenceLocality,
    EvidenceReference,
    discover_evidence,
)
from .models import (
    ClaimType,
    ChangeInventory,
    DeclaredReviewCoordinates,
    GateDisposition,
    MagnitudeKind,
    PreflightGateResult,
    ProviderCallRecord,
    ProviderUsage,
    ReviewConfig,
    ReviewError,
    ReviewInventoryFailure,
    ReviewPreflight,
    ReviewStatus,
    ScientificClaim,
    ScopeIssue,
    SnapshotIdentity,
    SourceBundle,
)
from .provider import (
    ProviderResponse,
    ReviewerProvider,
    StructuredRequest,
)
from .request_budget import (
    allocate_synthesis_inputs,
    build_extraction_request_parts,
    build_synthesis_request_parts,
    extraction_schema as shared_extraction_schema,
    logical_request_chars,
    plain as shared_plain,
    serialized_chars as shared_serialized_chars,
    synthesis_schema as shared_synthesis_schema,
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
    revalidate_manifest_audit_input_identities,
    run_manifest_audits,
    select_relevant_manifest_audit_plans,
)


MAX_PROVIDER_JSON_DEPTH = 64


def OpenAIReviewerProvider(**kwargs: Any) -> ReviewerProvider:
    """Construct the default adapter lazily, after provider-free preflight."""

    from .openai_provider import OpenAIReviewerProvider as Provider

    return Provider(**kwargs)


@dataclass(frozen=True)
class ReviewInputs:
    repository_root: Path
    base_root: Path | None = None
    pr_title: str = ""
    pr_description: str = ""
    requested_base: SnapshotIdentity | None = None
    comparison_base: SnapshotIdentity | None = None
    head: SnapshotIdentity | None = None
    inventory: ChangeInventory | object | None = None
    coordinates: DeclaredReviewCoordinates | None = None
    inventory_failure: ReviewInventoryFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path):
            object.__setattr__(self, "repository_root", Path(self.repository_root))
        if self.base_root is not None and not isinstance(self.base_root, Path):
            object.__setattr__(self, "base_root", Path(self.base_root))
        if not isinstance(self.pr_title, str) or not isinstance(self.pr_description, str):
            raise ReviewError("pull-request title and description must be strings")
        if self.coordinates is not None and not isinstance(
            self.coordinates, DeclaredReviewCoordinates
        ):
            raise ReviewError("declared review coordinates are invalid")
        if self.inventory_failure is not None and not isinstance(
            self.inventory_failure, ReviewInventoryFailure
        ):
            raise ReviewError("review inventory failure is invalid")
        if self.inventory is not None and self.inventory_failure is not None:
            raise ReviewError("review inventory and failure are mutually exclusive")


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
    preflight: ReviewPreflight | None = None
    claims: tuple[ScientificClaim, ...] = ()
    interpretations: tuple[ClaimInterpretation, ...] = ()
    evidence: EvidenceBundle = EvidenceBundle()
    deterministic_audits: tuple[DeterministicAuditSnapshot, ...] = ()
    provider_calls: tuple[ProviderCallRecord, ...] = ()
    usage: ProviderUsage = ProviderUsage()
    error_code: str | None = None
    error_message: str | None = None


def _extraction_schema(max_claims: int) -> dict[str, Any]:
    return shared_extraction_schema(max_claims)


def _citable_reference(reference: EvidenceReference) -> bool:
    return reference.excerpt_locality is not EvidenceLocality.UNLOCALIZED_PREFIX


_STANDALONE_TEST_ANNOTATION = re.compile(r"[ \t]*@Test[ \t]*\Z")
_TEST_COUNT_METRICS = frozenset(
    {
        "test",
        "tests",
        "test count",
        "test case",
        "test cases",
        "test case count",
    }
)
_TEST_COUNT_UNITS = frozenset(
    {"case", "cases", "test", "tests", "test case", "test cases"}
)


def _normalized_words(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.casefold().split())


def _is_test_count_metric(value: str | None) -> bool:
    normalized = _normalized_words(value)
    if normalized in _TEST_COUNT_METRICS:
        return True
    return bool(
        normalized
        and (
            normalized.startswith("test cases covering ")
            or normalized.startswith("tests covering ")
        )
    )


def _java_standalone_test_count(text: str) -> int | None:
    """Count standalone ``@Test`` code lines with a narrow lexical scan.

    This is deliberately not a Java parser. Comments and literals are masked,
    and malformed or unterminated lexical state yields no whole-file fact.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Java translates Unicode escapes before lexical analysis. Supporting that
    # transformation would exceed this deliberately narrow grammar, so any
    # such escape makes the count unavailable instead of risking a false fact.
    if re.search(r"\\u+[0-9a-fA-F]{4}", normalized):
        return None
    masked: list[str] = []
    state = "code"
    index = 0
    while index < len(normalized):
        character = normalized[index]
        following = normalized[index + 1] if index + 1 < len(normalized) else ""

        if state == "code":
            if character == "/" and following == "/":
                masked.extend((" ", " "))
                state = "line_comment"
                index += 2
                continue
            if character == "/" and following == "*":
                masked.extend((" ", " "))
                state = "block_comment"
                index += 2
                continue
            if normalized.startswith('\"\"\"', index):
                masked.extend((" ", " ", " "))
                state = "text_block"
                index += 3
                continue
            if character == '"':
                masked.append(" ")
                state = "string"
                index += 1
                continue
            if character == "'":
                masked.append(" ")
                state = "character"
                index += 1
                continue
            masked.append(character)
            index += 1
            continue

        if state == "line_comment":
            if character == "\n":
                masked.append("\n")
                state = "code"
            else:
                masked.append(" ")
            index += 1
            continue

        if state == "block_comment":
            if character == "*" and following == "/":
                masked.extend((" ", " "))
                state = "code"
                index += 2
            else:
                masked.append("\n" if character == "\n" else " ")
                index += 1
            continue

        if state == "text_block":
            if normalized.startswith('\"\"\"', index):
                masked.extend((" ", " ", " "))
                state = "code"
                index += 3
            else:
                masked.append("\n" if character == "\n" else " ")
                index += 1
            continue

        if state in {"string", "character"}:
            terminator = '"' if state == "string" else "'"
            if character == "\n":
                return None
            if character == "\\":
                if not following or following == "\n":
                    return None
                masked.extend((" ", " "))
                index += 2
                continue
            masked.append(" ")
            index += 1
            if character == terminator:
                state = "code"
            continue

        return None

    if state not in {"code", "line_comment"}:
        return None
    return sum(
        1
        for line in "".join(masked).splitlines()
        if _STANDALONE_TEST_ANNOTATION.fullmatch(line)
    )


def _java_test_cardinality_constraints(
    claims: Sequence[ScientificClaim],
    evidence: EvidenceBundle,
) -> dict[str, dict[str, Any]]:
    constraints: dict[str, dict[str, Any]] = {}
    for claim in claims:
        magnitude = claim.claimed_magnitude
        if (
            claim.claim_type is not ClaimType.IMPLEMENTATION_CLAIM
            or magnitude is None
            or magnitude.kind is not MagnitudeKind.ABSOLUTE
            or magnitude.value is None
            or not float(magnitude.value).is_integer()
            or float(magnitude.value) < 0
            or _normalized_words(magnitude.unit) not in _TEST_COUNT_UNITS
            or not _is_test_count_metric(claim.metric)
        ):
            continue
        subject = claim.subject.strip()
        hinted_paths = {
            path
            for path in claim.evidence_hints
            if PurePosixPath(path).suffix.casefold() == ".java"
            and PurePosixPath(path).stem == subject
        }
        if len(hinted_paths) != 1:
            continue
        hinted_path = next(iter(hinted_paths))
        references = [
            reference
            for reference in evidence.references
            if reference.path == hinted_path
            and reference.kind is EvidenceKind.TEST
            and claim.claim_id in reference.claim_ids
        ]
        reference = references[0] if len(references) == 1 else None
        claimed_count = int(magnitude.value)
        observed_count = (
            _java_standalone_test_count(reference.excerpt)
            if reference is not None
            and reference.excerpt_complete
            and reference.excerpt_locality is EvidenceLocality.COMPLETE_FILE
            else None
        )
        if observed_count is None:
            state = "incomplete"
            required_interpretation = (
                "ClaimCI cannot verify the exact whole-file test count because "
                "complete claim-owned Java test evidence was not available in the "
                "analyzed snapshot."
            )
            required_citations: list[str] = []
        elif observed_count != claimed_count:
            state = "contradicted"
            required_interpretation = (
                f"The complete source contains {observed_count} standalone @Test "
                f"annotation lines, contradicting the claimed count of "
                f"{claimed_count} test cases."
            )
            required_citations = [reference.evidence_id]
        else:
            state = "corroborated"
            required_interpretation = (
                f"The complete source contains {observed_count} standalone @Test "
                f"annotation lines, matching the claimed count of "
                f"{claimed_count} test cases; this is source support, not evidence "
                "that the tests were executed."
            )
            required_citations = [reference.evidence_id]
        constraints[claim.claim_id] = {
            "kind": "java_test_cardinality",
            "state": state,
            "claimed_count": claimed_count,
            "observed_count": observed_count,
            "evidence_id": (
                reference.evidence_id if reference is not None else None
            ),
            "required_interpretation": required_interpretation,
            "required_citations": required_citations,
        }
    return constraints


def _interpretation_constraints(
    claims: Sequence[ScientificClaim],
    evidence: EvidenceBundle,
    evidence_ids_by_claim_id: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    constraints: dict[str, dict[str, Any]] = {}
    for claim in claims:
        unlocalized_ids = sorted(
            reference.evidence_id
            for reference in evidence.references
            if claim.claim_id in reference.claim_ids
            and reference.excerpt_locality is EvidenceLocality.UNLOCALIZED_PREFIX
        )
        if unlocalized_ids and not evidence_ids_by_claim_id[claim.claim_id]:
            constraints[claim.claim_id] = {
                "kind": "unlocalized_evidence",
                "state": "incomplete",
                "evidence_ids": unlocalized_ids,
                "required_interpretation": (
                    "ClaimCI cannot verify this claim from the exact source/test "
                    "evidence because the available bounded excerpts could not be "
                    "localized to the material region."
                ),
                "required_citations": [],
            }
    # The narrower whole-file count fact takes precedence when it applies.
    constraints.update(_java_test_cardinality_constraints(claims, evidence))
    return constraints


def _evidence_ids_by_claim_id(
    claims: Sequence[ScientificClaim],
    evidence: EvidenceBundle,
) -> dict[str, list[str]]:
    return {
        claim.claim_id: sorted(
            reference.evidence_id
            for reference in evidence.references
            if claim.claim_id in reference.claim_ids and _citable_reference(reference)
        )
        for claim in claims
    }


def _synthesis_schema(
    claims: Sequence[ScientificClaim],
    evidence_ids_by_claim_id: Mapping[str, Sequence[str]],
    interpretation_constraints_by_claim_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Constrain each row to issued IDs using the provider-supported subset."""
    return shared_synthesis_schema(
        claims,
        evidence_ids_by_claim_id,
        interpretation_constraints_by_claim_id,
    )


def _plain(value: Any) -> Any:
    return shared_plain(value)


def _serialized_chars(value: object) -> int:
    return shared_serialized_chars(value)


def _request_chars(task: str, payload: object, schema: object) -> int:
    """Count the complete logical context for this provider request."""

    return logical_request_chars(task, payload, schema)


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
    interpretation_constraints_by_claim_id: Mapping[str, Mapping[str, Any]] | None = None,
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
        if _citable_reference(item)
    }
    constraints = interpretation_constraints_by_claim_id or {}
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
        constraint = constraints.get(claim_id)
        if constraint is not None and (
            interpretation != constraint["required_interpretation"]
            or citations != tuple(constraint["required_citations"])
        ):
            raise ReviewError(
                "synthesis violates a deterministic advisory interpretation constraint"
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
    preflight: ReviewPreflight | None = None,
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
        preflight=preflight,
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


def _record_runtime_gate3_omissions(
    preflight: ReviewPreflight | None,
    omissions: Sequence[tuple[str, str, int]],
) -> ReviewPreflight | None:
    """Attach exact post-extraction allocation losses to the returned scope."""

    material = tuple(item for item in omissions if item[2] > 0)
    if preflight is None or not material:
        return preflight
    gate3 = preflight.gates[2]
    if gate3.disposition not in {
        GateDisposition.PASS_COMPLETE,
        GateDisposition.PASS_PARTIAL,
    }:
        raise ReviewError("runtime omissions require a provider-ready Gate 3")
    new_issues = tuple(
        ScopeIssue(code=code, observed=count, limit=0)
        for code, _metric, count in material
    )
    reasons = tuple(
        sorted(
            set((*gate3.reasons, *new_issues)),
            key=lambda issue: (
                issue.code,
                issue.path or "",
                -1 if issue.observed is None else issue.observed,
                -1 if issue.limit is None else issue.limit,
            ),
        )
    )
    metrics = dict(gate3.metrics)
    metrics.update({metric: count for _code, metric, count in material})
    revised_gate3 = PreflightGateResult(
        gate=3,
        disposition=GateDisposition.PASS_PARTIAL,
        reasons=reasons,
        metrics=metrics,
    )
    scope = preflight.scope
    if scope is not None:
        scope = replace(
            scope,
            complete=False,
            issues=tuple(
                sorted(
                    set((*scope.issues, *new_issues)),
                    key=lambda issue: (
                        issue.code,
                        issue.path or "",
                        -1 if issue.observed is None else issue.observed,
                        -1 if issue.limit is None else issue.limit,
                    ),
                )
            ),
        )
    return replace(
        preflight,
        gates=(*preflight.gates[:2], revised_gate3),
        review_status_ceiling=ReviewStatus.PARTIAL,
        scope=scope,
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
    preflight: ReviewPreflight | None = None

    def finish(status: ReviewStatus, **kwargs: Any) -> ResearchReview:
        return _result(status, preflight=preflight, **kwargs)

    if not config.enabled:
        return finish(ReviewStatus.DISABLED)
    if not isinstance(inputs, ReviewInputs):
        raise ReviewError("inputs must be ReviewInputs")
    declared = any(
        value is not None
        for value in (
            inputs.requested_base,
            inputs.comparison_base,
            inputs.head,
            inputs.inventory,
            inputs.coordinates,
            inputs.inventory_failure,
        )
    )
    from .preflight import (
        legacy_preflight_failure,
        preflight_review,
        scope_source_bundle,
    )

    if declared:
        preflight = preflight_review(inputs, config)
    else:
        # A legacy caller keeps the established permissive review path when
        # material-seed routing alone cannot establish the new coordinate-
        # bound contract.  When the bounded legacy planner is provider-ready,
        # retain that scope identity and consume its exact shortlist.
        planned_legacy = preflight_review(inputs, config)
        preflight = (
            planned_legacy
            if planned_legacy.ready_for_provider
            else legacy_preflight_failure(inputs, config)
        )
    if preflight is not None and not preflight.ready_for_provider:
        failure = next(
            gate for gate in preflight.gates if gate.disposition.value == "fail"
        )
        return finish(
            ReviewStatus.UNAVAILABLE,
            error_code=failure.reasons[0].code,
            error_message="Research review preflight did not pass.",
        )
    if preflight is not None:
        assert preflight.scope is not None
        preflight_sources = scope_source_bundle(preflight.scope)
        evidence_limits = replace(
            config.limits,
            max_context_chars=max(
                1,
                sum(
                    chars
                    for _path, chars in preflight.scope.materialized_path_chars
                ),
            ),
        )
    else:
        preflight_sources = None
        evidence_limits = config.limits

    calls: list[ProviderCallRecord] = []
    try:
        sources = (
            preflight_sources
            if preflight_sources is not None
            else collect_review_sources(
                inputs.repository_root,
                base_root=inputs.base_root,
                pr_title=inputs.pr_title,
                pr_description=inputs.pr_description,
                limits=config.limits,
            )
        )
        manifest_candidates = discover_manifests(
            inputs.repository_root,
            sources.repository_paths,
            max_manifests=min(4, config.limits.max_files),
        )
        audit_issued_paths = (
            sources.repository_paths if preflight_sources is not None else None
        )
        audit_selected_paths = (
            preflight.scope.selected_paths
            if preflight is not None and preflight.scope is not None
            else ()
        )
        audit_plans = select_relevant_manifest_audit_plans(
            plan_manifest_audits(
                inputs.repository_root,
                manifest_candidates,
                limits=config.limits,
                selected_paths=audit_selected_paths,
                issued_paths=audit_issued_paths,
            ),
            changed_paths=sources.changed_paths,
        )
        omitted_audit_plan_count = 0
        omitted_audit_dependency_count = 0
        invalidated_audit_plan_count = 0
        if preflight_sources is not None:
            issued_paths = set(preflight_sources.repository_paths)
            retained_audit_plans = tuple(
                plan
                for plan in audit_plans
                if set(plan.paths).issubset(issued_paths)
            )
            omitted_audit_plans = tuple(
                plan for plan in audit_plans if plan not in retained_audit_plans
            )
            omitted_audit_plan_count = len(omitted_audit_plans)
            omitted_audit_dependency_count = sum(
                len(set(plan.paths) - issued_paths) for plan in omitted_audit_plans
            )
            audit_plans = retained_audit_plans
            preflight = _record_runtime_gate3_omissions(
                preflight,
                (
                    (
                        "PREFLIGHT_G3_AUDIT_PLAN_OMITTED",
                        "omitted_audit_plan_count",
                        omitted_audit_plan_count,
                    ),
                    (
                        "PREFLIGHT_G3_AUDIT_DEPENDENCY_OMITTED",
                        "omitted_audit_dependency_count",
                        omitted_audit_dependency_count,
                    ),
                ),
            )
        manifest_candidates = tuple(
            plan.manifest_path for plan in audit_plans
        )
        if preflight_sources is None:
            sources = _sources_after_manifest_reservation(
                sources,
                audit_plans,
                max_files=config.limits.max_files,
            )
        # Deterministic authority consumes the exact reserved inputs before an
        # untrusted provider can mutate the checkout.  Later provider-time
        # validation is identity-only and never re-plans or re-runs Audit.
        deterministic_audits = run_manifest_audits(
            inputs.repository_root,
            manifest_candidates,
            limits=config.limits,
            selected_paths=audit_selected_paths,
            reserved_plans=audit_plans,
            issued_paths=audit_issued_paths,
        )
        completed_audit_manifests = {
            snapshot.manifest_path for snapshot in deterministic_audits
        }
        invalidated_audit_plan_count = sum(
            plan.manifest_path not in completed_audit_manifests
            for plan in audit_plans
        )
        preflight = _record_runtime_gate3_omissions(
            preflight,
            (
                (
                    "PREFLIGHT_G3_AUDIT_PLAN_INVALIDATED",
                    "invalidated_audit_plan_count",
                    invalidated_audit_plan_count,
                ),
            ),
        )
        if invalidated_audit_plan_count:
            return finish(
                ReviewStatus.PARTIAL,
                deterministic_audits=deterministic_audits,
                error_code="AUDIT_PLANS_INVALIDATED",
                error_message=(
                    f"{invalidated_audit_plan_count} pre-provider deterministic "
                    "audit plan(s) failed exact revalidation or execution and "
                    "were omitted from this bounded advisory review."
                ),
            )
        plans_by_manifest = {
            plan.manifest_path: plan for plan in audit_plans
        }
        audit_bundles = tuple(
            ManifestAuditBundle(snapshot=snapshot, paths=plan.paths)
            for snapshot in deterministic_audits
            if (plan := plans_by_manifest.get(snapshot.manifest_path)) is not None
        )
        extraction_parts = build_extraction_request_parts(
            sources, config.limits.max_claims
        )
        extraction_chars = _request_chars(
            extraction_parts.task,
            extraction_parts.payload,
            extraction_parts.schema,
        )
        if extraction_chars > config.limits.max_context_chars:
            return finish(
                ReviewStatus.UNAVAILABLE,
                error_code="CONTEXT_LIMIT",
                error_message="Extraction context exceeds the configured limit.",
            )
        if provider is None:
            if config.provider != "openai":
                return finish(
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
        extraction_request = StructuredRequest(
            task=extraction_parts.task,
            payload=extraction_parts.payload,
            schema=extraction_parts.schema,
            max_output_tokens=config.limits.extraction_max_output_tokens,
        )
        extraction_response = provider.extract_claims(extraction_request)
        extraction_call = _record_call(
            "extract_claims", extraction_response, extraction_chars
        )
        calls.append(extraction_call)
        if not extraction_response.complete:
            truncated = extraction_response.incomplete_reason == "max_output_tokens"
            return finish(
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
        retained_audit_plans = revalidate_manifest_audit_input_identities(
            inputs.repository_root,
            audit_plans,
            limits=config.limits,
            issued_paths=audit_issued_paths,
        )
        retained_audit_keys = {
            (plan.manifest_path, plan.paths) for plan in retained_audit_plans
        }
        invalidated_audit_plan_count = sum(
            (plan.manifest_path, plan.paths) not in retained_audit_keys
            for plan in audit_plans
        )
        if invalidated_audit_plan_count:
            retained_manifests = {
                plan.manifest_path for plan in retained_audit_plans
            }
            deterministic_audits = tuple(
                snapshot
                for snapshot in deterministic_audits
                if snapshot.manifest_path in retained_manifests
            )
            preflight = _record_runtime_gate3_omissions(
                preflight,
                (
                    (
                        "PREFLIGHT_G3_AUDIT_PLAN_INVALIDATED",
                        "invalidated_audit_plan_count",
                        invalidated_audit_plan_count,
                    ),
                ),
            )
            return finish(
                ReviewStatus.PARTIAL,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="AUDIT_PLANS_INVALIDATED",
                error_message=(
                    f"{invalidated_audit_plan_count} exact deterministic Audit "
                    "input bundle(s) changed or became unavailable after the "
                    "pre-provider snapshot; synthesis was skipped."
                ),
            )
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
        selected_paths = (
            set(preflight.scope.selected_paths)
            if preflight is not None and preflight.scope is not None
            else {
                source.path
                for source in sources.sources
                if source.path is not None
            }
        )
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
            limits=evidence_limits,
            priority_paths=priority_paths,
            selected_paths=tuple(sorted(selected_paths)),
            changed_paths=sources.changed_paths,
            base_root=(
                inputs.comparison_base.root
                if preflight_sources is not None
                and inputs.comparison_base is not None
                else inputs.base_root
            ),
        )
        if config.limits.max_calls < 2:
            return finish(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="CALL_LIMIT",
                error_message="Synthesis was skipped by the configured call limit.",
            )
        evidence_ids_by_claim_id = _evidence_ids_by_claim_id(claims, evidence)
        interpretation_constraints_by_claim_id = _interpretation_constraints(
            claims,
            evidence,
            evidence_ids_by_claim_id,
        )
        synthesis_allocation = allocate_synthesis_inputs(
            claims,
            tuple(
                reference
                for reference in evidence.references
                if _citable_reference(reference)
            ),
            evidence_ids_by_claim_id,
            interpretation_constraints_by_claim_id,
            evidence.missing,
            deterministic_audits,
            max_chars=config.limits.max_context_chars,
        )
        synthesis_parts = synthesis_allocation.parts
        omission_counts = synthesis_allocation.omitted_counts
        preflight = _record_runtime_gate3_omissions(
            preflight,
            (
                (
                    "PREFLIGHT_G3_SYNTHESIS_EVIDENCE_OMITTED",
                    "synthesis_evidence_omitted_count",
                    omission_counts["evidence"],
                ),
                (
                    "PREFLIGHT_G3_SYNTHESIS_CONSTRAINT_OMITTED",
                    "synthesis_constraint_omitted_count",
                    omission_counts["interpretation_constraints"],
                ),
                (
                    "PREFLIGHT_G3_SYNTHESIS_AUDIT_OMITTED",
                    "synthesis_audit_omitted_count",
                    omission_counts["deterministic_audits"],
                ),
                (
                    "PREFLIGHT_G3_SYNTHESIS_MISSING_EVIDENCE_OMITTED",
                    "synthesis_missing_evidence_omitted_count",
                    omission_counts["missing_evidence"],
                ),
            ),
        )
        synthesis_chars = _request_chars(
            synthesis_parts.task,
            synthesis_parts.payload,
            synthesis_parts.schema,
        )
        if synthesis_chars > config.limits.max_context_chars:
            return finish(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="CONTEXT_LIMIT",
                error_message="Synthesis context exceeds the configured limit.",
            )
        synthesis_request = StructuredRequest(
            task=synthesis_parts.task,
            payload=synthesis_parts.payload,
            schema=synthesis_parts.schema,
            max_output_tokens=config.limits.synthesis_max_output_tokens,
        )
        synthesis_response = provider.synthesize_review(synthesis_request)
        synthesis_call = _record_call(
            "synthesize_review", synthesis_response, synthesis_chars
        )
        calls.append(synthesis_call)
        if not synthesis_response.complete:
            truncated = synthesis_response.incomplete_reason == "max_output_tokens"
            return finish(
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
            return finish(
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
                _parse_json(synthesis_response.output_text),
                claims,
                evidence,
                synthesis_parts.payload[
                    "interpretation_constraints_by_claim_id"
                ],
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
            return finish(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="SYNTHESIS_INVALID",
                error_message=validation_message,
            )
        if rejected_claim_candidates:
            return finish(
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
        if any(omission_counts.values()):
            return finish(
                ReviewStatus.PARTIAL,
                claims=claims,
                interpretations=interpretations,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="SYNTHESIS_INPUTS_OMITTED",
                error_message=(
                    "Bounded advisory synthesis omitted exact input counts: "
                    f"evidence={omission_counts['evidence']}, "
                    "interpretation_constraints="
                    f"{omission_counts['interpretation_constraints']}, "
                    f"missing_evidence={omission_counts['missing_evidence']}, "
                    "deterministic_audits="
                    f"{omission_counts['deterministic_audits']}. Full evidence and "
                    "deterministic Audit results remain in the review record."
                ),
            )
        if evidence.routing_incomplete:
            return finish(
                ReviewStatus.PARTIAL,
                claims=claims,
                interpretations=interpretations,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="EVIDENCE_ROUTING_INCOMPLETE",
                error_message=(
                    "One or more provider-suggested evidence hints remained "
                    "unresolved after deterministic evidence routing; every retained "
                    "claim was interpreted, but the advisory review is incomplete."
                ),
            )
        if omitted_audit_plan_count:
            return finish(
                ReviewStatus.PARTIAL,
                claims=claims,
                interpretations=interpretations,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="AUDIT_PLANS_OMITTED",
                error_message=(
                    f"{omitted_audit_plan_count} deterministic audit plan(s) with "
                    f"{omitted_audit_dependency_count} unissued dependency path(s) "
                    "were omitted from this bounded advisory scope."
                ),
            )
        return finish(
            (
                preflight.review_status_ceiling
                if preflight is not None
                else ReviewStatus.COMPLETE
            ),
            claims=claims,
            interpretations=interpretations,
            evidence=evidence,
            deterministic_audits=deterministic_audits,
            calls=calls,
        )
    except Exception as exc:
        # Provider/model text is never included in the safe status message.
        return finish(
            ReviewStatus.UNAVAILABLE,
            calls=calls,
            error_code="REVIEW_UNAVAILABLE",
            error_message=f"Research review is unavailable ({type(exc).__name__}).",
        )
