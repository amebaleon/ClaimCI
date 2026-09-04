"""Fixed two-call research-review orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, DecimalException
from pathlib import Path, PurePosixPath
from typing import Any

from claimci.passive_files import (
    PassiveFileError,
    capture_confined_regular_file,
    inspect_confined_regular_file,
)
from claimci.parsing import unique_json_object

from .evidence import (
    _safe_relative,
    changed_region_excerpt_from_text,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLocality,
    EvidenceMaterialIdentity,
    EvidenceMaterialInvalidated,
    EvidenceReference,
    discover_evidence,
)
from .models import (
    ClaimType,
    ChangeInventory,
    ChangeInventorySource,
    ChangeStatus,
    DeclaredReviewCoordinates,
    ExactMaterialOmission,
    GateDisposition,
    MagnitudeKind,
    PreflightGateResult,
    ProviderCallRecord,
    ProviderLifecycle,
    ProviderUsage,
    ReviewConfig,
    ReviewError,
    ReviewInventoryFailure,
    ReviewMaterialKind,
    ReviewPreflight,
    ReviewScope,
    ReviewStatus,
    ScientificClaim,
    ScopeIssue,
    SnapshotIdentity,
    SnapshotRole,
    SourceBundle,
    SourceKind,
    SourceRecord,
)
from .path_policy import classify_review_material
from .inventory import (
    exact_git_material_omission_matches,
    InventoryVerificationError,
    git_blob_descriptor,
    verify_git_material_identities,
)
from .provider import (
    ProviderResponse,
    ReviewerProvider,
    StructuredRequest,
)
from .request_budget import (
    MAX_DETERMINISTIC_JAVA_TEST_COUNT,
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
    MAX_SOURCE_FILE_BYTES,
    collect_review_sources,
    validate_claim_candidates_best_effort,
)
from .tools import (
    MAX_MANIFEST_AUDITS,
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
    exact_material_omissions: tuple[ExactMaterialOmission, ...] = ()

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
        if (
            not isinstance(self.exact_material_omissions, tuple)
            or len(self.exact_material_omissions) > 24
            or not all(
                isinstance(item, ExactMaterialOmission)
                for item in self.exact_material_omissions
            )
            or len({item.path for item in self.exact_material_omissions})
            != len(self.exact_material_omissions)
        ):
            raise ReviewError("exact material omissions are invalid")


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
    provider_lifecycle: ProviderLifecycle = ProviderLifecycle.NOT_ATTEMPTED
    provider_attempt_count: int = 0
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.provider_attempt_count, bool)
            or not isinstance(self.provider_attempt_count, int)
            or not 0 <= self.provider_attempt_count <= 2
        ):
            raise ReviewError("provider attempt count must be from zero through two")
        completed = len(self.provider_calls)
        expected_attempts = (
            completed + 1
            if self.provider_lifecycle is ProviderLifecycle.FAILED_BEFORE_RESPONSE
            else completed
        )
        if self.provider_attempt_count != expected_attempts:
            raise ReviewError("provider lifecycle and attempt count are inconsistent")
        if (
            self.provider_lifecycle is ProviderLifecycle.NOT_ATTEMPTED
            and self.provider_attempt_count != 0
        ):
            raise ReviewError("unattempted provider lifecycle cannot contain calls")
        if (
            self.provider_lifecycle is ProviderLifecycle.RESPONSE_RECEIVED
            and self.provider_attempt_count == 0
        ):
            raise ReviewError("received provider lifecycle requires a response")


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
            # Java text blocks permit escaped quotes specifically so an
            # embedded quote run does not become the closing delimiter.  This
            # narrow scanner does not implement the JLS escape transformation;
            # fail closed on every text-block escape instead of exposing string
            # content as code and emitting a false whole-file count.
            if character == "\\":
                return None
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
        if float(magnitude.value) > MAX_DETERMINISTIC_JAVA_TEST_COUNT:
            constraints[claim.claim_id] = {
                "kind": "java_test_cardinality",
                "state": "out_of_bounds",
                "claimed_count": None,
                "observed_count": None,
                "evidence_id": None,
                "limit": MAX_DETERMINISTIC_JAVA_TEST_COUNT,
                "required_interpretation": (
                    "ClaimCI cannot verify the claimed whole-file test count "
                    "because it exceeds the maximum count representable by a "
                    "complete file within the bounded inspection size."
                ),
                "required_citations": [],
            }
            continue
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


def _order_synthesis_evidence(
    claims: Sequence[ScientificClaim],
    references: Sequence[EvidenceReference],
    *,
    priority_paths: Mapping[str, Sequence[str]],
    selected_paths: Sequence[str],
) -> tuple[EvidenceReference, ...]:
    """Rank already-authorized evidence for bounded synthesis first-fit.

    Discovery remains the authority for reference ownership and routing.  This
    ordering only gives bounded synthesis its strongest existing references
    first: claim-owned trusted audit inputs, exact SOURCE/TEST hints, and then
    SOURCE references owned by implementation claims.  Prioritized ties follow
    the frozen preflight rank; all other references retain producer order.
    """

    claims_by_id = {claim.claim_id: claim for claim in claims}
    trusted_paths_by_claim = {
        claim_id: frozenset(paths)
        for claim_id, paths in priority_paths.items()
    }
    exact_paths_by_claim = {
        claim.claim_id: frozenset(
            normalized
            for hint in claim.evidence_hints
            if (normalized := _safe_relative(hint)) is not None
        )
        for claim in claims
    }
    implementation_claim_ids = {
        claim.claim_id
        for claim in claims
        if claim.claim_type is ClaimType.IMPLEMENTATION_CLAIM
    }
    selected_rank = {path: index for index, path in enumerate(selected_paths)}
    unranked = len(selected_rank)
    prioritized: tuple[list[tuple[int, EvidenceReference]], ...] = (
        [],
        [],
        [],
    )
    remaining: list[EvidenceReference] = []

    for producer_index, reference in enumerate(references):
        owned_claim_ids = tuple(
            claim_id
            for claim_id in reference.claim_ids
            if claim_id in claims_by_id
        )
        if any(
            reference.path in trusted_paths_by_claim.get(claim_id, ())
            for claim_id in owned_claim_ids
        ):
            tier = 0
        elif reference.kind in {EvidenceKind.SOURCE, EvidenceKind.TEST} and any(
            reference.path in exact_paths_by_claim[claim_id]
            for claim_id in owned_claim_ids
        ):
            tier = 1
        elif (
            reference.kind is EvidenceKind.SOURCE
            and bool(set(owned_claim_ids) & implementation_claim_ids)
        ):
            tier = 2
        else:
            remaining.append(reference)
            continue
        prioritized[tier].append((producer_index, reference))

    def ranked(
        values: list[tuple[int, EvidenceReference]],
    ) -> tuple[EvidenceReference, ...]:
        return tuple(
            reference
            for _producer_index, reference in sorted(
                values,
                key=lambda item: (
                    selected_rank.get(item[1].path, unranked),
                    item[0],
                    item[1].path,
                    item[1].evidence_id,
                ),
            )
        )

    return (
        *ranked(prioritized[0]),
        *ranked(prioritized[1]),
        *ranked(prioritized[2]),
        *remaining,
    )


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
    issued_evidence_ids_by_claim_id: Mapping[str, Sequence[str]] | None = None,
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
    if issued_evidence_ids_by_claim_id is None:
        issued_evidence_claims = {
            claim_id: frozenset(
                evidence_id
                for evidence_id, owners in evidence_claims.items()
                if claim_id in owners
            )
            for claim_id in claim_ids
        }
    else:
        if (
            not isinstance(issued_evidence_ids_by_claim_id, Mapping)
            or set(issued_evidence_ids_by_claim_id) != claim_ids
        ):
            raise ReviewError("synthesis issued evidence ownership is invalid")
        issued_evidence_claims: dict[str, frozenset[str]] = {}
        for claim_id, raw_ids in issued_evidence_ids_by_claim_id.items():
            if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
                raise ReviewError("synthesis issued evidence ownership is invalid")
            allowed = tuple(raw_ids)
            if (
                any(
                    not isinstance(evidence_id, str)
                    or evidence_id not in evidence_claims
                    or claim_id not in evidence_claims[evidence_id]
                    for evidence_id in allowed
                )
                or len(set(allowed)) != len(allowed)
            ):
                raise ReviewError("synthesis issued evidence ownership is invalid")
            issued_evidence_claims[claim_id] = frozenset(allowed)
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
            or citation not in issued_evidence_claims[claim_id]
            for citation in citations
        ):
            raise ReviewError(
                "synthesis cites unknown, omitted, or claim-mismatched repository evidence"
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
    provider_lifecycle: ProviderLifecycle = ProviderLifecycle.NOT_ATTEMPTED,
    provider_attempt_count: int = 0,
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
        usage=(
            ProviderUsage()
            if provider_lifecycle is ProviderLifecycle.FAILED_BEFORE_RESPONSE
            else _aggregate_usage(call_tuple)
        ),
        provider_lifecycle=provider_lifecycle,
        provider_attempt_count=provider_attempt_count,
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


def _merge_ready_runtime_gate3(
    refreshed: ReviewPreflight,
    previous: ReviewPreflight | None,
) -> ReviewPreflight:
    """Keep prior runtime Gate 3 facts on a newly refreshed ready record."""

    if previous is None:
        return refreshed
    if not refreshed.ready_for_provider or not previous.ready_for_provider:
        raise ReviewError("runtime Gate 3 merge requires ready preflights")
    prior_gate3 = previous.gates[2]
    refreshed_gate3 = refreshed.gates[2]
    reasons = tuple(
        sorted(
            set((*prior_gate3.reasons, *refreshed_gate3.reasons)),
            key=lambda issue: (
                issue.code,
                issue.path or "",
                -1 if issue.observed is None else issue.observed,
                -1 if issue.limit is None else issue.limit,
            ),
        )
    )
    disposition = (
        GateDisposition.PASS_PARTIAL
        if reasons
        else GateDisposition.PASS_COMPLETE
    )
    gate3 = PreflightGateResult(
        gate=3,
        disposition=disposition,
        reasons=reasons,
        metrics={**dict(prior_gate3.metrics), **dict(refreshed_gate3.metrics)},
    )
    scope = refreshed.scope
    if scope is not None and previous.scope is not None:
        scope = replace(
            scope,
            complete=scope.complete and previous.scope.complete and not reasons,
            issues=tuple(
                sorted(
                    set((*scope.issues, *previous.scope.issues)),
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
        refreshed,
        gates=(*refreshed.gates[:2], gate3),
        review_status_ceiling=(
            ReviewStatus.PARTIAL
            if disposition is GateDisposition.PASS_PARTIAL
            else ReviewStatus.COMPLETE
        ),
        scope=scope,
    )


@dataclass(frozen=True)
class _MaterialPathIdentity:
    """Content identity/state for one already-authorized repository path."""

    state: str
    size: int | None = None
    sha256: str | None = None
    git_blob_sha1: str | None = None
    git_blob_sha256: str | None = None


def _capture_material_path_identity(
    root: Path,
    path: str,
) -> _MaterialPathIdentity:
    """Inspect one frozen path without discovering or retaining new material."""

    try:
        inspection = inspect_confined_regular_file(
            root,
            path,
            max_bytes=MAX_SOURCE_FILE_BYTES,
        )
    except PassiveFileError as exc:
        return _MaterialPathIdentity(state=f"error:{exc.code}")
    except (OSError, RuntimeError, TypeError, ValueError):
        return _MaterialPathIdentity(state="error:inspection")
    return _MaterialPathIdentity(
        state="present",
        size=inspection.size,
        sha256=inspection.sha256,
        git_blob_sha1=inspection.git_blob_sha1,
        git_blob_sha256=inspection.git_blob_sha256,
    )


def _capture_material_scope_identities(
    head_root: Path,
    selected_paths: Sequence[str],
    *,
    comparison_base_root: Path | None,
) -> tuple[
    tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None], ...
]:
    """Capture identities for only the deterministic, bounded selected list."""

    if not isinstance(selected_paths, Sequence) or isinstance(
        selected_paths, (str, bytes)
    ):
        raise ReviewError("selected material paths must be a sequence")
    selected = tuple(selected_paths)
    if any(not isinstance(path, str) or not path for path in selected):
        raise ReviewError("selected material path is invalid")
    if len(selected) > 24 or len(set(selected)) != len(selected):
        raise ReviewError("selected material paths exceed the fixed review bound")
    captured: list[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ] = []
    for path in selected:
        head_identity = _capture_material_path_identity(head_root, path)
        base_identity = (
            None
            if comparison_base_root is None
            else _capture_material_path_identity(comparison_base_root, path)
        )
        captured.append((path, head_identity, base_identity))
    return tuple(captured)


def _material_scope_capture_failure_paths(
    captured: Sequence[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ],
    inventory: ChangeInventory,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate captured states against authoritative changed-path statuses."""

    if not isinstance(inventory, ChangeInventory):
        raise ReviewError("selected material inventory is invalid")
    status_by_path = {entry.path: entry.status for entry in inventory.entries}
    failures: list[str] = []
    head_failures: list[str] = []

    def fail(path: str, *, head: bool = False) -> None:
        if path not in failures:
            failures.append(path)
        if head and path not in head_failures:
            head_failures.append(path)

    for path, head_identity, base_identity in captured:
        status = status_by_path.get(path)
        if status not in {None, ChangeStatus.ADDED, ChangeStatus.MODIFIED}:
            fail(path, head=True)
            continue
        if head_identity.state != "present":
            fail(path, head=True)
        if status in {None, ChangeStatus.MODIFIED}:
            if base_identity is None or base_identity.state != "present":
                fail(path)
            elif status is None and base_identity != head_identity:
                # A selected path absent from the complete change inventory is
                # an unchanged deterministic supplement, never an implicit
                # changed candidate.
                fail(path)
        elif (
            base_identity is not None
            and base_identity.state != "error:unavailable"
        ):
            # An added path must remain absent from the comparison snapshot.
            # Symlinks, oversized files, inspection errors, or present bytes are
            # inconsistent with the authoritative inventory and fail closed.
            fail(path)
    return tuple(failures), tuple(head_failures)


def _verify_material_git_binding(
    inputs: ReviewInputs,
    scope: ReviewScope,
    captured: Sequence[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ],
) -> None:
    """Prove descriptor-captured material equals exact declared Git blobs."""

    if scope.inventory.source is not ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH:
        return
    if not (
        isinstance(inputs.comparison_base, SnapshotIdentity)
        and isinstance(inputs.head, SnapshotIdentity)
    ):
        raise InventoryVerificationError(
            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
            "declared material snapshots are unavailable",
        )
    head_identities: dict[str, tuple[int, str, str]] = {}
    comparison_identities: dict[str, tuple[int, str, str] | None] = {}
    for path, head_identity, base_identity in captured:
        if (
            head_identity.state != "present"
            or head_identity.size is None
            or head_identity.git_blob_sha1 is None
            or head_identity.git_blob_sha256 is None
        ):
            raise InventoryVerificationError(
                "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
                "selected head material identity is unavailable",
            )
        head_identities[path] = (
            head_identity.size,
            head_identity.git_blob_sha1,
            head_identity.git_blob_sha256,
        )
        if base_identity is None or base_identity.state == "error:unavailable":
            comparison_identities[path] = None
        elif (
            base_identity.state == "present"
            and base_identity.size is not None
            and base_identity.git_blob_sha1 is not None
            and base_identity.git_blob_sha256 is not None
        ):
            comparison_identities[path] = (
                base_identity.size,
                base_identity.git_blob_sha1,
                base_identity.git_blob_sha256,
            )
        else:
            raise InventoryVerificationError(
                "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
                "selected comparison material identity is unavailable",
            )
    verify_git_material_identities(
        inputs.comparison_base,
        inputs.head,
        scope.inventory,
        tuple(path for path, _head, _base in captured),
        head_identities,
        comparison_identities,
        max_files=max(1, len(captured)),
    )


def _coordinate_failure_preflight(
    preflight: ReviewPreflight,
    code: str,
) -> ReviewPreflight:
    """Replace any materialized record with one exact Gate 1 failure."""

    gate1 = PreflightGateResult(
        gate=1,
        disposition=GateDisposition.FAIL,
        reasons=(ScopeIssue(code=code),),
        metrics={},
    )
    skipped = tuple(
        PreflightGateResult(
            gate=gate,
            disposition=GateDisposition.NOT_EVALUATED,
            reasons=(
                ScopeIssue(code="PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"),
            ),
            metrics={},
        )
        for gate in (2, 3)
    )
    return ReviewPreflight(
        schema_version=preflight.schema_version,
        requested_base_sha=preflight.requested_base_sha,
        comparison_base_sha=preflight.comparison_base_sha,
        head_sha=preflight.head_sha,
        gates=(gate1, *skipped),
        ready_for_provider=False,
        review_status_ceiling=ReviewStatus.UNAVAILABLE,
        scope=None,
        comparison_basis=preflight.comparison_basis,
    )


def _material_scope_drifted_paths(
    before: Sequence[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ],
    after: Sequence[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return head and base drift separately in deterministic selected order."""

    before_by_path = {path: (head, base) for path, head, base in before}
    after_by_path = {path: (head, base) for path, head, base in after}
    head_drifted = tuple(
        path
        for path, _head, _base in before
        if path not in after_by_path
        or before_by_path[path][0] != after_by_path[path][0]
    )
    base_drifted = tuple(
        path
        for path, _head, _base in before
        if path not in after_by_path
        or before_by_path[path][1] != after_by_path[path][1]
    )
    return head_drifted, base_drifted


def _sanitize_scope_material_paths(
    scope: ReviewScope,
    paths: Sequence[str],
) -> ReviewScope:
    """Remove every scope fact derived from unbound material paths."""

    invalid = set(paths)
    expanded = True
    while expanded:
        expanded = False
        for group in scope.atomic_path_groups:
            if invalid.intersection(group) and not set(group).issubset(invalid):
                invalid.update(group)
                expanded = True
    removed_source_ids = {
        source.source_id
        for source in scope.sources
        if source.kind is SourceKind.REPOSITORY_FILE and source.path in invalid
    }
    retained_selected = tuple(
        path for path in scope.selected_paths if path not in invalid
    )
    retained_path_chars = tuple(
        item for item in scope.materialized_path_chars if item[0] not in invalid
    )
    removed_chars = sum(
        chars
        for path, chars in scope.materialized_path_chars
        if path in invalid
    )
    return replace(
        scope,
        issued_paths=tuple(
            path for path in scope.issued_paths if path not in invalid
        ),
        issued_changed_paths=tuple(
            path for path in scope.issued_changed_paths if path not in invalid
        ),
        selected_paths=retained_selected,
        sources=tuple(
            source
            for source in scope.sources
            if source.kind is not SourceKind.REPOSITORY_FILE
            or source.path not in invalid
        ),
        seeds=tuple(
            seed
            for seed in scope.seeds
            if seed.origin_source_id not in removed_source_ids
        ),
        complete=False,
        issues=tuple(issue for issue in scope.issues if issue.path not in invalid),
        materialized_chars=max(0, scope.materialized_chars - removed_chars),
        materialized_path_chars=retained_path_chars,
        atomic_path_groups=tuple(
            group
            for group in scope.atomic_path_groups
            if not invalid.intersection(group)
        ),
    )


def _sanitize_preflight_material_facts(
    preflight: ReviewPreflight,
    scope: ReviewScope,
    paths: Sequence[str],
) -> ReviewPreflight:
    """Replace unprovable material facts with one content-free Gate 2 failure."""

    del scope
    invalid_count = len(set(paths))
    if invalid_count < 1:
        raise ReviewError("material invalidation requires at least one path")
    if preflight.gates[0].disposition is GateDisposition.FAIL:
        return replace(preflight, scope=None)
    gate2 = PreflightGateResult(
        gate=2,
        disposition=GateDisposition.FAIL,
        reasons=(
            ScopeIssue(
                code="PREFLIGHT_G2_MATERIAL_SOURCE_INVALIDATED",
                observed=invalid_count,
                limit=0,
            ),
        ),
        metrics={"material_source_invalidated_count": invalid_count},
    )
    gate3 = PreflightGateResult(
        gate=3,
        disposition=GateDisposition.NOT_EVALUATED,
        reasons=(ScopeIssue(code="PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"),),
        metrics={},
    )
    return ReviewPreflight(
        schema_version=preflight.schema_version,
        requested_base_sha=preflight.requested_base_sha,
        comparison_base_sha=preflight.comparison_base_sha,
        head_sha=preflight.head_sha,
        gates=(preflight.gates[0], gate2, gate3),
        ready_for_provider=False,
        review_status_ceiling=ReviewStatus.UNAVAILABLE,
        scope=None,
        comparison_basis=preflight.comparison_basis,
    )


_UNMATERIALIZED_SCOPE_ISSUE_CODES = frozenset(
    {
        "PREFLIGHT_G2_CANDIDATE_TOO_LARGE",
        "PREFLIGHT_G2_CANDIDATE_UNREADABLE",
        "PREFLIGHT_G2_COMPARISON_CANDIDATE_TOO_LARGE",
    }
)


def _unmaterialized_material_issue_paths(scope: ReviewScope) -> tuple[str, ...]:
    """Return attempted candidates whose material never entered the scope."""

    selected = set(scope.selected_paths)
    return tuple(
        sorted(
            {
                issue.path
                for issue in scope.issues
                if issue.code in _UNMATERIALIZED_SCOPE_ISSUE_CODES
                and issue.path is not None
                and issue.path not in selected
            }
        )
    )


def _revalidate_unmaterialized_material_issues(
    inputs: ReviewInputs,
    scope: ReviewScope,
) -> tuple[str, ...]:
    """Reproduce content-derived omissions without retaining their content."""

    if not isinstance(inputs.head, SnapshotIdentity):
        return _unmaterialized_material_issue_paths(scope)
    issue_codes_by_path: dict[str, set[str]] = {}
    for issue in scope.issues:
        if (
            issue.path is not None
            and issue.code in _UNMATERIALIZED_SCOPE_ISSUE_CODES
            and issue.path not in scope.selected_paths
        ):
            issue_codes_by_path.setdefault(issue.path, set()).add(issue.code)

    invalidated: list[str] = []
    for path in sorted(issue_codes_by_path):
        expected = issue_codes_by_path[path]
        if len(expected) != 1:
            invalidated.append(path)
            continue
        expected_code = next(iter(expected))
        if expected_code in {
            "PREFLIGHT_G2_CANDIDATE_TOO_LARGE",
            "PREFLIGHT_G2_COMPARISON_CANDIDATE_TOO_LARGE",
        }:
            expected_role = (
                SnapshotRole.HEAD
                if expected_code == "PREFLIGHT_G2_CANDIDATE_TOO_LARGE"
                else SnapshotRole.COMPARISON_BASE
            )
            omission = next(
                (
                    item
                    for item in inputs.exact_material_omissions
                    if item.path == path
                    and item.code == expected_code
                    and item.role is expected_role
                ),
                None,
            )
            issue = next(
                (
                    item
                    for item in scope.issues
                    if item.path == path and item.code == expected_code
                ),
                None,
            )
            identity = (
                inputs.head
                if expected_role is SnapshotRole.HEAD
                else inputs.comparison_base
            )
            inventory_status = next(
                (
                    entry.status
                    for entry in scope.inventory.entries
                    if entry.path == path
                ),
                None,
            )
            status_is_valid = (
                inventory_status is ChangeStatus.MODIFIED
                if expected_role is SnapshotRole.COMPARISON_BASE
                else inventory_status is not ChangeStatus.DELETED
            )
            expected_sha = (
                scope.inventory.head_sha
                if expected_role is SnapshotRole.HEAD
                else scope.inventory.comparison_base_sha
            )
            if (
                omission is not None
                and issue is not None
                and issue.observed == omission.observed
                and issue.limit == omission.limit
                and isinstance(identity, SnapshotIdentity)
                and identity.role is expected_role
                and identity.sha == expected_sha
                and status_is_valid
            ):
                if exact_git_material_omission_matches(omission, identity):
                    continue
        if expected_code == "PREFLIGHT_G2_COMPARISON_CANDIDATE_TOO_LARGE":
            status = next(
                (
                    entry.status
                    for entry in scope.inventory.entries
                    if entry.path == path
                ),
                None,
            )
            if (
                status is not ChangeStatus.MODIFIED
                or not isinstance(inputs.comparison_base, SnapshotIdentity)
            ):
                invalidated.append(path)
                continue
            try:
                descriptor = git_blob_descriptor(inputs.comparison_base, path)
            except InventoryVerificationError:
                invalidated.append(path)
                continue
            if descriptor is None or descriptor[0] <= MAX_SOURCE_FILE_BYTES:
                invalidated.append(path)
            continue
        if expected_code == "PREFLIGHT_G2_CANDIDATE_TOO_LARGE":
            try:
                descriptor = git_blob_descriptor(inputs.head, path)
            except InventoryVerificationError:
                invalidated.append(path)
                continue
            if descriptor is None or descriptor[0] <= MAX_SOURCE_FILE_BYTES:
                invalidated.append(path)
            continue

        try:
            first = capture_confined_regular_file(
                inputs.repository_root,
                path,
                max_bytes=MAX_SOURCE_FILE_BYTES,
            )
            first.content.decode("utf-8", errors="strict")
        except UnicodeError:
            pass
        except (
            PassiveFileError,
            OSError,
            RecursionError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            invalidated.append(path)
            continue
        else:
            invalidated.append(path)
            continue
        try:
            descriptor = git_blob_descriptor(inputs.head, path)
            if descriptor is None or descriptor[0] != first.size:
                invalidated.append(path)
                continue
            header = f"blob {first.size}\0".encode("ascii")
            sha1 = hashlib.sha1(
                header + first.content,
                usedforsecurity=False,
            ).hexdigest()
            sha256 = hashlib.sha256(header + first.content).hexdigest()
            object_id = descriptor[1]
            if object_id != (sha1 if len(object_id) == 40 else sha256):
                invalidated.append(path)
                continue
            second = capture_confined_regular_file(
                inputs.repository_root,
                path,
                max_bytes=MAX_SOURCE_FILE_BYTES,
            )
            if second.size != first.size or second.sha256 != first.sha256:
                invalidated.append(path)
                continue
            second.content.decode("utf-8", errors="strict")
        except UnicodeError:
            continue
        except (
            InventoryVerificationError,
            PassiveFileError,
            OSError,
            RecursionError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            invalidated.append(path)
            continue
        invalidated.append(path)
    return tuple(invalidated)


def _bind_scope_materialization(
    head_root: Path,
    scope: ReviewScope,
    captured: Sequence[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ],
    *,
    comparison_base_root: Path | None,
) -> tuple[ReviewScope, tuple[str, ...], tuple[str, ...]]:
    """Bind every material decision and provider-visible source to exact bytes.

    ``SourceRecord.sha256`` identifies its normalized, possibly truncated text,
    while ``_MaterialPathIdentity.sha256`` identifies the complete raw file.
    Re-capture every selected path and reproduce preflight's materialization
    and changed-region decision so transient head/base reads cannot authorize a
    provider request. Documents additionally bind their issued text records.
    """

    identities = {path: (head, base) for path, head, base in captured}
    allocated_chars = dict(scope.materialized_path_chars)
    status_by_path = {entry.path: entry.status for entry in scope.inventory.entries}
    expected_document_paths = tuple(
        path
        for path in scope.selected_paths
        if classify_review_material(path) is ReviewMaterialKind.DOCUMENT
    )
    repository_records = tuple(
        source
        for source in scope.sources
        if source.kind is SourceKind.REPOSITORY_FILE
    )
    records_by_path: dict[str, list[SourceRecord]] = {}
    for source in repository_records:
        assert source.path is not None
        records_by_path.setdefault(source.path, []).append(source)

    drifted: set[str] = set()
    head_drifted: set[str] = set()
    expected_set = set(expected_document_paths)
    actual_set = set(records_by_path)
    drifted.update(expected_set.symmetric_difference(actual_set))
    drifted.update(
        path for path, records in records_by_path.items() if len(records) != 1
    )

    locality_codes = {
        "PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE",
        "PREFLIGHT_G3_SELECTED_SOURCE_CHAR_LIMIT",
    }
    locality_issues_by_path = {
        path: {
            issue
            for issue in scope.issues
            if issue.path == path and issue.code in locality_codes
        }
        for path in scope.selected_paths
    }

    for path in scope.selected_paths:
        identity_pair = identities.get(path)
        if identity_pair is None or identity_pair[0].state != "present":
            drifted.add(path)
            head_drifted.add(path)
            continue
        head_identity, base_identity = identity_pair
        try:
            head_capture = capture_confined_regular_file(
                head_root,
                path,
                max_bytes=MAX_SOURCE_FILE_BYTES,
            )
        except (
            PassiveFileError,
            OSError,
            RecursionError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            drifted.add(path)
            head_drifted.add(path)
            continue
        if (
            head_capture.size != head_identity.size
            or head_capture.sha256 != head_identity.sha256
        ):
            drifted.add(path)
            head_drifted.add(path)
            continue
        try:
            head_text = head_capture.content.decode("utf-8", errors="strict")
            head_text = head_text.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeError:
            drifted.add(path)
            continue
        char_count = allocated_chars.get(path)
        if char_count is None or len(head_text) < char_count:
            drifted.add(path)
            continue

        expected_locality_code: str | None = None
        expected_material_text = head_text
        if len(head_text) > char_count:
            recorded_locality = locality_issues_by_path.get(path, set())
            if len(recorded_locality) != 1:
                drifted.add(path)
                continue
            recorded_issue = next(iter(recorded_locality))
            representation_limit = recorded_issue.limit
            if (
                isinstance(representation_limit, bool)
                or not isinstance(representation_limit, int)
                or representation_limit < char_count
                or representation_limit < 1
            ):
                drifted.add(path)
                continue
            changed_region: tuple[str, int, int] | None = None
            if base_identity is None:
                changed_region = None
            elif (
                base_identity.state == "error:unavailable"
                and status_by_path.get(path) is ChangeStatus.ADDED
            ):
                changed_region = changed_region_excerpt_from_text(
                    head_text,
                    None,
                    char_limit=representation_limit,
                )
            elif (
                base_identity.state == "present"
                and comparison_base_root is not None
            ):
                try:
                    base_capture = capture_confined_regular_file(
                        comparison_base_root,
                        path,
                        max_bytes=MAX_SOURCE_FILE_BYTES,
                    )
                except (
                    PassiveFileError,
                    OSError,
                    RecursionError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    drifted.add(path)
                    continue
                if (
                    base_capture.size != base_identity.size
                    or base_capture.sha256 != base_identity.sha256
                ):
                    drifted.add(path)
                    continue
                try:
                    base_text = base_capture.content.decode(
                        "utf-8", errors="strict"
                    )
                    base_text = base_text.replace("\r\n", "\n").replace(
                        "\r", "\n"
                    )
                except UnicodeError:
                    base_text = None
                    changed_region = None
                else:
                    changed_region = changed_region_excerpt_from_text(
                        head_text,
                        base_text,
                        char_limit=representation_limit,
                    )
            else:
                drifted.add(path)
                continue
            localized = changed_region is not None
            expected_material_text = (
                changed_region[0]
                if changed_region is not None
                else head_text[:representation_limit]
            )
            if len(expected_material_text) != char_count:
                drifted.add(path)
                continue
            expected_locality_code = (
                "PREFLIGHT_G3_SELECTED_SOURCE_CHAR_LIMIT"
                if localized
                else "PREFLIGHT_G2_EXCERPT_LOCALITY_UNAVAILABLE"
            )
        expected_locality_issues = (
            set()
            if expected_locality_code is None
            else {
                ScopeIssue(
                    code=expected_locality_code,
                    path=path,
                    observed=len(head_text),
                    limit=(
                        next(iter(locality_issues_by_path[path])).limit
                        if expected_locality_code is not None
                        else None
                    ),
                )
            }
        )
        if locality_issues_by_path.get(path, set()) != expected_locality_issues:
            drifted.add(path)

        if path in expected_set:
            records = records_by_path.get(path, ())
            if (
                len(records) != 1
                or records[0].text != expected_material_text
                or records[0].sha256
                != hashlib.sha256(records[0].text.encode("utf-8")).hexdigest()
            ):
                drifted.add(path)

    if not drifted:
        return scope, (), ()

    sanitized = _sanitize_scope_material_paths(scope, drifted)
    return sanitized, tuple(sorted(drifted)), tuple(sorted(head_drifted))


def _retain_audits_after_material_drift(
    snapshots: Sequence[DeterministicAuditSnapshot],
    plans: Sequence[ManifestAuditPlan],
    head_drifted_paths: Sequence[str],
) -> tuple[DeterministicAuditSnapshot, ...]:
    """Retain Audits whose authoritative head inputs remain unchanged."""

    drifted = set(head_drifted_paths)
    invalidated_manifests = {
        plan.manifest_path
        for plan in plans
        if drifted.intersection(plan.paths)
    }
    return tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.manifest_path not in invalidated_manifests
    )


def _audit_plan_drifted_paths(
    plans: Sequence[ManifestAuditPlan],
    material_identities: Sequence[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ],
) -> tuple[str, ...]:
    """Compare planned Audit inputs with the initial preflight-bound head state."""

    expected = {
        path: head.sha256 if head.state == "present" else None
        for path, head, _base in material_identities
    }
    drifted: set[str] = set()
    for plan in plans:
        # Empty identities remain a compatibility seam for manually supplied
        # plans; every production plan carries a complete bounded identity set.
        if not plan.input_sha256:
            continue
        for path, sha256 in plan.input_sha256:
            if path not in expected:
                if sha256 is not None:
                    drifted.add(path)
                continue
            if expected[path] != sha256:
                drifted.add(path)
    return tuple(sorted(drifted))


def _evidence_material_identities(
    captured: Sequence[
        tuple[str, _MaterialPathIdentity, _MaterialPathIdentity | None]
    ],
) -> tuple[EvidenceMaterialIdentity, ...]:
    """Adapt validated bounded identities for descriptor-bound evidence reads."""

    identities: list[EvidenceMaterialIdentity] = []
    for path, head, base in captured:
        if head.state != "present" or head.sha256 is None or head.size is None:
            raise ReviewError("selected head evidence identity is unavailable")
        if base is None:
            base_sha256 = None
            base_size = None
            base_bound = False
            base_absent = False
        elif base.state == "error:unavailable":
            base_sha256 = None
            base_size = None
            base_bound = True
            base_absent = True
        elif base.state == "present" and base.sha256 is not None and base.size is not None:
            base_sha256 = base.sha256
            base_size = base.size
            base_bound = True
            base_absent = False
        else:
            raise ReviewError("selected base evidence identity is unavailable")
        identities.append(
            EvidenceMaterialIdentity(
                path=path,
                head_sha256=head.sha256,
                head_size=head.size,
                base_sha256=base_sha256,
                base_size=base_size,
                base_bound=base_bound,
                base_absent=base_absent,
            )
        )
    return tuple(identities)


def _evidence_identity_drifted_paths(
    evidence: EvidenceBundle,
    identities: Sequence[EvidenceMaterialIdentity],
) -> tuple[str, ...]:
    """Reject evidence envelopes not tied to the authorized full-file identity."""

    expected = {
        identity.path: (identity.head_sha256, identity.head_size)
        for identity in identities
    }
    return tuple(
        sorted(
            {
                reference.path
                for reference in evidence.references
                if expected.get(reference.path)
                != (reference.sha256, reference.size)
            }
        )
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
    preflight_only: bool = False,
) -> ResearchReview:
    """Run the fixed two-call advisory review state machine."""

    if not isinstance(config, ReviewConfig):
        raise ReviewError("config must be ReviewConfig")
    if not isinstance(preflight_only, bool):
        raise ReviewError("preflight_only must be a boolean")
    preflight: ReviewPreflight | None = None
    provider_lifecycle = ProviderLifecycle.NOT_ATTEMPTED
    provider_attempt_count = 0

    def finish(status: ReviewStatus, **kwargs: Any) -> ResearchReview:
        return _result(
            status,
            preflight=preflight,
            provider_lifecycle=provider_lifecycle,
            provider_attempt_count=provider_attempt_count,
            **kwargs,
        )

    if not config.enabled:
        return finish(ReviewStatus.DISABLED)
    if not isinstance(inputs, ReviewInputs):
        raise ReviewError("inputs must be ReviewInputs")
    from .preflight import (
        _plan_preflight_review,
        revalidate_preflight_coordinates,
        scope_source_bundle,
    )

    preflight = _plan_preflight_review(inputs, config)
    comparison_base_root = (
        inputs.comparison_base.root
        if inputs.comparison_base is not None
        else inputs.base_root
    )
    if (
        preflight is not None
        and preflight.ready_for_provider
        and preflight.scope is not None
    ):
        invalidated_issue_paths = _revalidate_unmaterialized_material_issues(
            inputs,
            preflight.scope,
        )
        if invalidated_issue_paths:
            sanitized_scope = _sanitize_scope_material_paths(
                preflight.scope,
                invalidated_issue_paths,
            )
            preflight = _sanitize_preflight_material_facts(
                preflight,
                sanitized_scope,
                invalidated_issue_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                error_code="PREFLIGHT_G2_MATERIAL_SOURCE_INVALIDATED",
                error_message=(
                    "A material omission could not be bound to the exact "
                    "repository bytes; provider execution was skipped."
                ),
            )
    if preflight is not None and not preflight.ready_for_provider:
        failure = next(
            gate for gate in preflight.gates if gate.disposition.value == "fail"
        )
        if failure.gate == 1:
            # A failed coordinate/inventory gate cannot authorize retaining any
            # materialized repository content from that untrusted scope.
            preflight = replace(preflight, scope=None)
        elif preflight.scope is not None:
            failed_scope = preflight.scope
            selected = tuple(sorted(failed_scope.selected_paths))
            unmaterialized_issue_paths = _unmaterialized_material_issue_paths(
                failed_scope
            )
            hash_bindable_issue_paths = tuple(
                path
                for path in unmaterialized_issue_paths
                if any(
                    issue.path == path
                    and issue.code == "PREFLIGHT_G2_CANDIDATE_UNREADABLE"
                    for issue in failed_scope.issues
                )
            )
            bound_paths = tuple(
                sorted(set(selected) | set(hash_bindable_issue_paths))
            )
            captured = ()
            invalidated_paths: set[str] = set()
            try:
                captured = _capture_material_scope_identities(
                    inputs.repository_root,
                    bound_paths,
                    comparison_base_root=comparison_base_root,
                )
                failed_scope, drifted, _head_drifted = _bind_scope_materialization(
                    inputs.repository_root,
                    failed_scope,
                    captured,
                    comparison_base_root=comparison_base_root,
                )
            except ReviewError:
                drifted = tuple(
                    sorted(
                        set(selected)
                        | {
                            source.path
                            for source in failed_scope.sources
                            if source.kind is SourceKind.REPOSITORY_FILE
                            and source.path is not None
                        }
                    )
                )
                failed_scope = _sanitize_scope_material_paths(
                    failed_scope,
                    drifted,
                )
            invalidated_paths.update(drifted)
            invalidated_paths.update(
                _revalidate_unmaterialized_material_issues(inputs, failed_scope)
            )

            # A descriptor-consistent filesystem snapshot must still belong to
            # the declared Git HEAD. Revalidate without rematerializing, then
            # recapture to close a restore-during-Git-check ABA window.
            preflight = replace(preflight, scope=failed_scope)
            preflight = revalidate_preflight_coordinates(inputs, preflight)
            if preflight.gates[0].disposition is not GateDisposition.FAIL:
                try:
                    _verify_material_git_binding(
                        inputs,
                        failed_scope,
                        captured,
                    )
                except InventoryVerificationError as exc:
                    preflight = _coordinate_failure_preflight(
                        preflight,
                        exc.code,
                    )
            if preflight.gates[0].disposition is not GateDisposition.FAIL:
                invalidated_paths.update(
                    _revalidate_unmaterialized_material_issues(
                        inputs,
                        failed_scope,
                    )
                )
                try:
                    recaptured = _capture_material_scope_identities(
                        inputs.repository_root,
                        bound_paths,
                        comparison_base_root=comparison_base_root,
                    )
                    recapture_failures, _recapture_head_failures = (
                        _material_scope_capture_failure_paths(
                            recaptured,
                            failed_scope.inventory,
                        )
                    )
                    recaptured_head_drift, recaptured_base_drift = (
                        _material_scope_drifted_paths(captured, recaptured)
                    )
                    invalidated_paths.update(recapture_failures)
                    invalidated_paths.update(recaptured_head_drift)
                    invalidated_paths.update(recaptured_base_drift)
                except ReviewError:
                    invalidated_paths.update(bound_paths)
            if (
                preflight.gates[0].disposition is not GateDisposition.FAIL
                and invalidated_paths
            ):
                failed_scope = _sanitize_scope_material_paths(
                    failed_scope,
                    tuple(sorted(invalidated_paths)),
                )
                preflight = _sanitize_preflight_material_facts(
                    preflight,
                    failed_scope,
                    tuple(sorted(invalidated_paths)),
                )
        else:
            preflight = revalidate_preflight_coordinates(inputs, preflight)
        failure = next(
            gate for gate in preflight.gates if gate.disposition is GateDisposition.FAIL
        )
        return finish(
            ReviewStatus.UNAVAILABLE,
            error_code=failure.reasons[0].code,
            error_message="Research review preflight did not pass.",
        )
    if preflight is not None:
        assert preflight.scope is not None
        planned_scope = preflight.scope
        selected_material_paths = tuple(sorted(planned_scope.selected_paths))
        materialized_scope_identities = ()
        try:
            materialized_scope_identities = _capture_material_scope_identities(
                inputs.repository_root,
                selected_material_paths,
                comparison_base_root=comparison_base_root,
            )
            (
                initial_capture_failure_paths,
                _initial_head_failure_paths,
            ) = _material_scope_capture_failure_paths(
                materialized_scope_identities,
                planned_scope.inventory,
            )
        except ReviewError:
            initial_capture_failure_paths = tuple(selected_material_paths)
        try:
            (
                planned_scope,
                source_scope_drifted_paths,
                _source_head_drifted_paths,
            ) = _bind_scope_materialization(
                inputs.repository_root,
                planned_scope,
                materialized_scope_identities,
                comparison_base_root=comparison_base_root,
            )
        except ReviewError:
            source_scope_drifted_paths = tuple(
                sorted(
                    set(selected_material_paths)
                    | {
                        source.path
                        for source in planned_scope.sources
                        if source.kind is SourceKind.REPOSITORY_FILE
                        and source.path is not None
                    }
                )
            )
            planned_scope = _sanitize_scope_material_paths(
                planned_scope,
                source_scope_drifted_paths,
            )
        preflight = replace(preflight, scope=planned_scope)
        preflight = revalidate_preflight_coordinates(inputs, preflight)
        if preflight.gates[0].disposition is GateDisposition.FAIL:
            failure = preflight.gates[0]
            return finish(
                ReviewStatus.UNAVAILABLE,
                error_code=failure.reasons[0].code,
                error_message=(
                    "Research review coordinates changed after materialization; "
                    "provider execution was skipped."
                ),
            )
        initially_unbound_paths = tuple(
            sorted(
                set(initial_capture_failure_paths)
                | set(source_scope_drifted_paths)
            )
        )
        if initially_unbound_paths:
            planned_scope = _sanitize_scope_material_paths(
                planned_scope,
                initially_unbound_paths,
            )
            preflight = _sanitize_preflight_material_facts(
                preflight,
                planned_scope,
                initially_unbound_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                error_code="PREFLIGHT_G2_MATERIAL_SOURCE_INVALIDATED",
                error_message=(
                    f"{len(initially_unbound_paths)} selected material path(s) "
                    "could not be bound to exact repository bytes; provider "
                    "execution was skipped."
                ),
            )
        try:
            _verify_material_git_binding(
                inputs,
                planned_scope,
                materialized_scope_identities,
            )
        except InventoryVerificationError as exc:
            preflight = _coordinate_failure_preflight(preflight, exc.code)
            return finish(
                ReviewStatus.UNAVAILABLE,
                error_code=exc.code,
                error_message=(
                    "Selected review material did not match the exact declared "
                    "Git objects; provider execution was skipped."
                ),
            )

        post_git_capture_failure_paths: tuple[str, ...] = ()
        post_git_identity_drifted_paths: tuple[str, ...] = ()
        try:
            post_git_material_identities = _capture_material_scope_identities(
                inputs.repository_root,
                selected_material_paths,
                comparison_base_root=comparison_base_root,
            )
            post_git_capture_failure_paths, _post_git_head_failures = (
                _material_scope_capture_failure_paths(
                    post_git_material_identities,
                    planned_scope.inventory,
                )
            )
            post_git_head_drifted, post_git_base_drifted = (
                _material_scope_drifted_paths(
                    materialized_scope_identities,
                    post_git_material_identities,
                )
            )
            post_git_identity_drifted_paths = tuple(
                sorted(set(post_git_head_drifted) | set(post_git_base_drifted))
            )
        except ReviewError:
            post_git_capture_failure_paths = tuple(selected_material_paths)
        initial_capture_failure_paths = tuple(
            sorted(
                set(initial_capture_failure_paths)
                | set(post_git_capture_failure_paths)
            )
        )
        source_scope_drifted_paths = tuple(
            sorted(
                set(source_scope_drifted_paths)
                | set(post_git_identity_drifted_paths)
            )
        )
        unbound_initial_paths = tuple(
            sorted(
                set(initial_capture_failure_paths)
                | set(source_scope_drifted_paths)
            )
        )
        if unbound_initial_paths:
            planned_scope = _sanitize_scope_material_paths(
                planned_scope,
                unbound_initial_paths,
            )
            preflight = _sanitize_preflight_material_facts(
                preflight,
                planned_scope,
                unbound_initial_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                error_code="PREFLIGHT_G2_MATERIAL_SOURCE_INVALIDATED",
                error_message=(
                    f"{len(unbound_initial_paths)} selected material path(s) "
                    "could not be bound to exact repository bytes; provider "
                    "execution was skipped."
                ),
            )
        if preflight_only:
            return finish(preflight.review_status_ceiling)
        preflight_sources = scope_source_bundle(planned_scope)
        evidence_limits = replace(
            config.limits,
            max_context_chars=max(
                1,
                sum(
                    chars
                    for _path, chars in planned_scope.materialized_path_chars
                ),
            ),
        )
    else:
        planned_scope = None
        selected_material_paths = ()
        comparison_base_root = inputs.base_root
        materialized_scope_identities = ()
        preflight_sources = None
        evidence_limits = config.limits

    calls: list[ProviderCallRecord] = []
    provider_failure_handler: Any = None
    failure_claims: tuple[ScientificClaim, ...] = ()
    failure_evidence = EvidenceBundle()
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
            max_manifests=min(MAX_MANIFEST_AUDITS, config.limits.max_files),
        )
        audit_issued_paths = None
        if preflight_sources is not None:
            assert preflight is not None and preflight.scope is not None
            audit_issued_paths = tuple(
                dict.fromkeys(
                    (
                        *sources.repository_paths,
                        *(
                            path
                            for group in preflight.scope.atomic_path_groups
                            for path in group
                        ),
                    )
                )
            )
        audit_selected_paths = (
            preflight.scope.selected_paths
            if preflight is not None and preflight.scope is not None
            else ()
        )
        audit_relevance_paths = (
            tuple(entry.path for entry in preflight.scope.inventory.entries)
            if preflight is not None and preflight.scope is not None
            else sources.changed_paths
        )
        audit_plans = select_relevant_manifest_audit_plans(
            plan_manifest_audits(
                inputs.repository_root,
                manifest_candidates,
                limits=config.limits,
                selected_paths=audit_selected_paths,
                issued_paths=audit_issued_paths,
            ),
            changed_paths=audit_relevance_paths,
        )
        omitted_audit_plan_count = 0
        omitted_audit_dependency_count = 0
        invalidated_audit_plan_count = 0
        if preflight_sources is not None:
            issued_paths = set(audit_issued_paths or ())
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
        audit_plan_drifted_paths = _audit_plan_drifted_paths(
            audit_plans,
            materialized_scope_identities,
        )
        if audit_plan_drifted_paths:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                audit_plan_drifted_paths,
            )
            assert preflight is not None and preflight.scope is not None
            preflight = _sanitize_preflight_material_facts(
                preflight,
                preflight.scope,
                audit_plan_drifted_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                error_code="MATERIAL_SCOPE_INVALIDATED",
                error_message=(
                    "A deterministic Audit plan observed selected head bytes "
                    "that differ from the provider-free materialization; "
                    "provider execution was skipped."
                ),
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
            # A missing Audit snapshot can mean either a stable local execution
            # failure or that the live checkout stopped matching the bound
            # private-mirror inputs while Audit was running.  Prove the former
            # before returning a scope that still describes provider-eligible
            # material.  The Git check and trailing recapture close the same
            # descriptor/Git/descriptor ABA window as the normal provider path.
            assert preflight.scope is not None
            audit_failure_scope = preflight.scope
            audit_failure_invalidated: set[str] = set()
            audit_failure_head_invalidated: set[str] = set()
            audit_failure_identities = ()
            try:
                audit_failure_identities = _capture_material_scope_identities(
                    inputs.repository_root,
                    selected_material_paths,
                    comparison_base_root=comparison_base_root,
                )
                (
                    audit_failure_capture_paths,
                    audit_failure_capture_head_paths,
                ) = _material_scope_capture_failure_paths(
                    audit_failure_identities,
                    planned_scope.inventory,
                )
                (
                    audit_failure_head_drifted,
                    audit_failure_base_drifted,
                ) = _material_scope_drifted_paths(
                    materialized_scope_identities,
                    audit_failure_identities,
                )
                (
                    audit_failure_scope,
                    audit_failure_source_drifted,
                    audit_failure_source_head_drifted,
                ) = _bind_scope_materialization(
                    inputs.repository_root,
                    audit_failure_scope,
                    audit_failure_identities,
                    comparison_base_root=comparison_base_root,
                )
                audit_failure_invalidated.update(audit_failure_capture_paths)
                audit_failure_invalidated.update(audit_failure_head_drifted)
                audit_failure_invalidated.update(audit_failure_base_drifted)
                audit_failure_invalidated.update(audit_failure_source_drifted)
                audit_failure_head_invalidated.update(
                    audit_failure_capture_head_paths
                )
                audit_failure_head_invalidated.update(audit_failure_head_drifted)
                audit_failure_head_invalidated.update(
                    audit_failure_source_head_drifted
                )
            except ReviewError:
                audit_failure_invalidated.update(selected_material_paths)
                audit_failure_head_invalidated.update(selected_material_paths)
                audit_failure_scope = _sanitize_scope_material_paths(
                    audit_failure_scope,
                    selected_material_paths,
                )
            audit_failure_invalidated.update(
                _revalidate_unmaterialized_material_issues(
                    inputs,
                    audit_failure_scope,
                )
            )
            if audit_failure_invalidated:
                deterministic_audits = _retain_audits_after_material_drift(
                    deterministic_audits,
                    audit_plans,
                    tuple(sorted(audit_failure_head_invalidated)),
                )
                preflight = _sanitize_preflight_material_facts(
                    preflight,
                    audit_failure_scope,
                    tuple(sorted(audit_failure_invalidated)),
                )
                return finish(
                    ReviewStatus.UNAVAILABLE,
                    deterministic_audits=deterministic_audits,
                    error_code="MATERIAL_SCOPE_INVALIDATED",
                    error_message=(
                        "Selected material changed while deterministic Audit "
                        "was running; provider execution was skipped."
                    ),
                )

            preflight = replace(preflight, scope=audit_failure_scope)
            preflight = revalidate_preflight_coordinates(inputs, preflight)
            if preflight.gates[0].disposition is GateDisposition.FAIL:
                deterministic_audits = ()
                failure = preflight.gates[0]
                return finish(
                    ReviewStatus.UNAVAILABLE,
                    deterministic_audits=deterministic_audits,
                    error_code=failure.reasons[0].code,
                    error_message=(
                        "Research review coordinates changed after deterministic "
                        "Audit; provider execution was skipped."
                    ),
                )
            try:
                _verify_material_git_binding(
                    inputs,
                    audit_failure_scope,
                    audit_failure_identities,
                )
            except InventoryVerificationError as exc:
                deterministic_audits = ()
                preflight = _coordinate_failure_preflight(preflight, exc.code)
                return finish(
                    ReviewStatus.UNAVAILABLE,
                    deterministic_audits=deterministic_audits,
                    error_code=exc.code,
                    error_message=(
                        "Selected review material no longer matched the exact "
                        "declared Git objects after deterministic Audit."
                    ),
                )

            post_audit_invalidated: set[str] = set()
            post_audit_head_invalidated: set[str] = set()
            try:
                post_audit_identities = _capture_material_scope_identities(
                    inputs.repository_root,
                    selected_material_paths,
                    comparison_base_root=comparison_base_root,
                )
                post_audit_failures, post_audit_head_failures = (
                    _material_scope_capture_failure_paths(
                        post_audit_identities,
                        planned_scope.inventory,
                    )
                )
                post_audit_head_drifted, post_audit_base_drifted = (
                    _material_scope_drifted_paths(
                        audit_failure_identities,
                        post_audit_identities,
                    )
                )
                post_audit_invalidated.update(post_audit_failures)
                post_audit_invalidated.update(post_audit_head_drifted)
                post_audit_invalidated.update(post_audit_base_drifted)
                post_audit_head_invalidated.update(post_audit_head_failures)
                post_audit_head_invalidated.update(post_audit_head_drifted)
            except ReviewError:
                post_audit_invalidated.update(selected_material_paths)
                post_audit_head_invalidated.update(selected_material_paths)
            post_audit_invalidated.update(
                _revalidate_unmaterialized_material_issues(
                    inputs,
                    audit_failure_scope,
                )
            )
            if post_audit_invalidated:
                deterministic_audits = _retain_audits_after_material_drift(
                    deterministic_audits,
                    audit_plans,
                    tuple(sorted(post_audit_head_invalidated)),
                )
                preflight = _sanitize_preflight_material_facts(
                    preflight,
                    audit_failure_scope,
                    tuple(sorted(post_audit_invalidated)),
                )
                return finish(
                    ReviewStatus.UNAVAILABLE,
                    deterministic_audits=deterministic_audits,
                    error_code="MATERIAL_SCOPE_INVALIDATED",
                    error_message=(
                        "Selected material changed while deterministic Audit "
                        "authority was being revalidated; provider execution "
                        "was skipped."
                    ),
                )
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
        assert planned_scope is not None
        try:
            pre_provider_material_identities = _capture_material_scope_identities(
                inputs.repository_root,
                selected_material_paths,
                comparison_base_root=comparison_base_root,
            )
            (
                material_capture_failure_paths,
                material_capture_head_failure_paths,
            ) = _material_scope_capture_failure_paths(
                pre_provider_material_identities,
                planned_scope.inventory,
            )
        except ReviewError:
            pre_provider_material_identities = ()
            material_capture_failure_paths = tuple(selected_material_paths)
            material_capture_head_failure_paths = tuple(selected_material_paths)
        if pre_provider_material_identities:
            (
                head_material_drifted_paths,
                base_material_drifted_paths,
            ) = _material_scope_drifted_paths(
                materialized_scope_identities,
                pre_provider_material_identities,
            )
        else:
            head_material_drifted_paths = tuple(selected_material_paths)
            base_material_drifted_paths = ()
        material_drifted_paths = tuple(
            sorted(
                set(head_material_drifted_paths)
                | set(base_material_drifted_paths)
            )
        )

        # Re-run the complete free gate sequence after deterministic Audit and
        # before provider construction. A coordinate failure must replace the
        # stale passing record with its exact authoritative gate result.
        refreshed_preflight = _plan_preflight_review(inputs, config)
        if not refreshed_preflight.ready_for_provider:
            preflight = refreshed_preflight
            failure = next(
                gate
                for gate in preflight.gates
                if gate.disposition is GateDisposition.FAIL
            )
            if failure.gate == 1:
                # Exact-coordinate failure means no prior Audit can retain
                # authoritative provenance, regardless of matching path hashes.
                deterministic_audits = ()
            else:
                # A later routing/budget failure does not erase an Audit whose
                # exact head inputs remain stable. Re-capture after the failed
                # refresh as well, so a mutation at its return boundary cannot
                # preserve a stale authoritative snapshot.
                failed_refresh_identities = ()
                try:
                    failed_refresh_identities = _capture_material_scope_identities(
                        inputs.repository_root,
                        selected_material_paths,
                        comparison_base_root=comparison_base_root,
                    )
                    (
                        _failed_refresh_capture_paths,
                        failed_refresh_head_failure_paths,
                    ) = _material_scope_capture_failure_paths(
                        failed_refresh_identities,
                        planned_scope.inventory,
                    )
                    (
                        failed_refresh_head_drifted_paths,
                        _failed_refresh_base_drifted_paths,
                    ) = _material_scope_drifted_paths(
                        materialized_scope_identities,
                        failed_refresh_identities,
                    )
                except ReviewError:
                    failed_refresh_head_failure_paths = tuple(
                        selected_material_paths
                    )
                    failed_refresh_head_drifted_paths = tuple(
                        selected_material_paths
                    )
                failed_source_head_drifted_paths: tuple[str, ...] = ()
                if preflight.scope is not None:
                    failed_scope = preflight.scope
                    try:
                        (
                            sanitized_failed_scope,
                            failed_source_drifted_paths,
                            failed_source_head_drifted_paths,
                        ) = _bind_scope_materialization(
                            inputs.repository_root,
                            failed_scope,
                            failed_refresh_identities,
                            comparison_base_root=comparison_base_root,
                        )
                    except ReviewError:
                        failed_source_drifted_paths = tuple(
                            sorted(
                                set(selected_material_paths)
                                | {
                                    source.path
                                    for source in failed_scope.sources
                                    if source.kind is SourceKind.REPOSITORY_FILE
                                    and source.path is not None
                                }
                            )
                        )
                        failed_source_head_drifted_paths = tuple(
                            selected_material_paths
                        )
                        sanitized_failed_scope = _sanitize_scope_material_paths(
                            failed_scope,
                            failed_source_drifted_paths,
                        )
                    failed_issue_drifted_paths = (
                        _revalidate_unmaterialized_material_issues(
                            inputs,
                            sanitized_failed_scope,
                        )
                    )
                    failed_scope_drifted_paths = tuple(
                        sorted(
                            set(failed_source_drifted_paths)
                            | set(failed_issue_drifted_paths)
                        )
                    )
                    if failed_scope_drifted_paths:
                        sanitized_failed_scope = _sanitize_scope_material_paths(
                            sanitized_failed_scope,
                            failed_scope_drifted_paths,
                        )
                        preflight = _sanitize_preflight_material_facts(
                            preflight,
                            sanitized_failed_scope,
                            failed_scope_drifted_paths,
                        )
                    else:
                        preflight = replace(
                            preflight,
                            scope=sanitized_failed_scope,
                        )
                invalid_audit_head_paths = tuple(
                    sorted(
                        set(material_capture_head_failure_paths)
                        | set(head_material_drifted_paths)
                        | set(failed_refresh_head_failure_paths)
                        | set(failed_refresh_head_drifted_paths)
                        | set(failed_source_head_drifted_paths)
                    )
                )
                deterministic_audits = _retain_audits_after_material_drift(
                    deterministic_audits,
                    audit_plans,
                    invalid_audit_head_paths,
                )
            failure = next(
                gate
                for gate in preflight.gates
                if gate.disposition is GateDisposition.FAIL
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                error_code=failure.reasons[0].code,
                error_message=(
                    "Research review runtime preflight did not pass; provider "
                    "execution was skipped."
                ),
            )
        assert refreshed_preflight.scope is not None
        refreshed_scope = refreshed_preflight.scope
        refreshed_issue_drifted_paths = _revalidate_unmaterialized_material_issues(
            inputs,
            refreshed_scope,
        )
        if refreshed_issue_drifted_paths:
            refreshed_scope = _sanitize_scope_material_paths(
                refreshed_scope,
                refreshed_issue_drifted_paths,
            )
            preflight = _sanitize_preflight_material_facts(
                refreshed_preflight,
                refreshed_scope,
                refreshed_issue_drifted_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                error_code="PREFLIGHT_G2_MATERIAL_SOURCE_INVALIDATED",
                error_message=(
                    "A refreshed material omission could not be bound to the "
                    "exact repository bytes; provider execution was skipped."
                ),
            )
        pre_refresh_capture_failure_paths = material_capture_failure_paths
        pre_refresh_head_failure_paths = material_capture_head_failure_paths
        pre_refresh_head_drifted_paths = head_material_drifted_paths
        pre_refresh_base_drifted_paths = base_material_drifted_paths
        try:
            pre_provider_material_identities = _capture_material_scope_identities(
                inputs.repository_root,
                selected_material_paths,
                comparison_base_root=comparison_base_root,
            )
            (
                post_refresh_capture_failure_paths,
                post_refresh_head_failure_paths,
            ) = _material_scope_capture_failure_paths(
                pre_provider_material_identities,
                planned_scope.inventory,
            )
            (
                post_refresh_head_drifted_paths,
                post_refresh_base_drifted_paths,
            ) = _material_scope_drifted_paths(
                materialized_scope_identities,
                pre_provider_material_identities,
            )
        except ReviewError:
            pre_provider_material_identities = ()
            post_refresh_capture_failure_paths = tuple(selected_material_paths)
            post_refresh_head_failure_paths = tuple(selected_material_paths)
            post_refresh_head_drifted_paths = tuple(selected_material_paths)
            post_refresh_base_drifted_paths = ()
        material_capture_failure_paths = tuple(
            sorted(
                set(pre_refresh_capture_failure_paths)
                | set(post_refresh_capture_failure_paths)
            )
        )
        material_capture_head_failure_paths = tuple(
            sorted(
                set(pre_refresh_head_failure_paths)
                | set(post_refresh_head_failure_paths)
            )
        )
        head_material_drifted_paths = tuple(
            sorted(
                set(pre_refresh_head_drifted_paths)
                | set(post_refresh_head_drifted_paths)
            )
        )
        base_material_drifted_paths = tuple(
            sorted(
                set(pre_refresh_base_drifted_paths)
                | set(post_refresh_base_drifted_paths)
            )
        )
        material_drifted_paths = tuple(
            sorted(
                set(head_material_drifted_paths)
                | set(base_material_drifted_paths)
            )
        )
        (
            refreshed_scope,
            source_scope_drifted_paths,
            source_head_drifted_paths,
        ) = _bind_scope_materialization(
            inputs.repository_root,
            refreshed_scope,
            pre_provider_material_identities,
            comparison_base_root=comparison_base_root,
        )
        refreshed_preflight = replace(
            refreshed_preflight,
            scope=refreshed_scope,
        )
        head_material_drifted_paths = tuple(
            sorted(
                set(head_material_drifted_paths)
                | set(source_head_drifted_paths)
            )
        )
        material_drifted_paths = tuple(
            sorted(
                set(material_drifted_paths)
                | set(source_scope_drifted_paths)
            )
        )
        if material_capture_failure_paths:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                material_capture_head_failure_paths,
            )
            preflight = _sanitize_preflight_material_facts(
                refreshed_preflight,
                refreshed_scope,
                material_capture_failure_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                error_code="MATERIAL_SCOPE_CAPTURE_FAILED",
                error_message=(
                    f"{len(material_capture_failure_paths)} selected material "
                    "path(s) could not be captured before provider execution."
                ),
            )
        if refreshed_scope != planned_scope:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                head_material_drifted_paths,
            )
            invalidated_paths = material_drifted_paths or selected_material_paths
            preflight = _sanitize_preflight_material_facts(
                refreshed_preflight,
                refreshed_scope,
                invalidated_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                error_code="MATERIAL_SCOPE_INVALIDATED",
                error_message=(
                    "The exact selected material scope changed after its "
                    "provider-free preflight; provider execution was skipped."
                ),
            )
        if material_drifted_paths:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                head_material_drifted_paths,
            )
            assert preflight is not None and preflight.scope is not None
            preflight = _sanitize_preflight_material_facts(
                preflight,
                preflight.scope,
                material_drifted_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                error_code="MATERIAL_SCOPE_INVALIDATED",
                error_message=(
                    "Selected material changed after provider-free "
                    "materialization; provider execution was skipped."
                ),
            )
        evidence_material_identities = _evidence_material_identities(
            pre_provider_material_identities
        )
        frozen_material_identities = pre_provider_material_identities

        def finish_provider_failure(
            exc: Exception,
            *,
            retained_claims: tuple[ScientificClaim, ...] = (),
            retained_evidence: EvidenceBundle = EvidenceBundle(),
        ) -> ResearchReview:
            """Fail closed while retaining only freshly revalidated authority."""

            nonlocal preflight
            safe_claims = retained_claims
            safe_evidence = retained_evidence
            safe_audits = deterministic_audits
            invalid_paths: set[str] = set()
            invalid_head_paths: set[str] = set()

            try:
                if preflight is None or preflight.scope is None:
                    raise ReviewError("provider failure has no bound scope")
                preflight = revalidate_preflight_coordinates(inputs, preflight)
                if preflight.gates[0].disposition is GateDisposition.FAIL:
                    preflight = replace(preflight, scope=None)
                    safe_claims = ()
                    safe_evidence = EvidenceBundle()
                    safe_audits = ()
                else:
                    assert preflight.scope is not None
                    post_failure_identities = _capture_material_scope_identities(
                        inputs.repository_root,
                        selected_material_paths,
                        comparison_base_root=comparison_base_root,
                    )
                    capture_failures, capture_head_failures = (
                        _material_scope_capture_failure_paths(
                            post_failure_identities,
                            planned_scope.inventory,
                        )
                    )
                    head_drifted, base_drifted = _material_scope_drifted_paths(
                        frozen_material_identities,
                        post_failure_identities,
                    )
                    invalid_paths.update(capture_failures)
                    invalid_paths.update(head_drifted)
                    invalid_paths.update(base_drifted)
                    invalid_head_paths.update(capture_head_failures)
                    invalid_head_paths.update(head_drifted)

                    if not invalid_paths:
                        _verify_material_git_binding(
                            inputs,
                            preflight.scope,
                            post_failure_identities,
                        )
                        recaptured_identities = (
                            _capture_material_scope_identities(
                                inputs.repository_root,
                                selected_material_paths,
                                comparison_base_root=comparison_base_root,
                            )
                        )
                        recapture_failures, recapture_head_failures = (
                            _material_scope_capture_failure_paths(
                                recaptured_identities,
                                planned_scope.inventory,
                            )
                        )
                        recaptured_head_drifted, recaptured_base_drifted = (
                            _material_scope_drifted_paths(
                                frozen_material_identities,
                                recaptured_identities,
                            )
                        )
                        invalid_paths.update(recapture_failures)
                        invalid_paths.update(recaptured_head_drifted)
                        invalid_paths.update(recaptured_base_drifted)
                        invalid_head_paths.update(recapture_head_failures)
                        invalid_head_paths.update(recaptured_head_drifted)

                    if invalid_paths:
                        safe_claims = ()
                        safe_evidence = EvidenceBundle()
                        safe_audits = _retain_audits_after_material_drift(
                            safe_audits,
                            audit_plans,
                            tuple(sorted(invalid_head_paths)),
                        )
                        sanitized_scope = _sanitize_scope_material_paths(
                            preflight.scope,
                            tuple(sorted(invalid_paths)),
                        )
                        preflight = _sanitize_preflight_material_facts(
                            preflight,
                            sanitized_scope,
                            tuple(sorted(invalid_paths)),
                        )
                    else:
                        retained_plans = (
                            revalidate_manifest_audit_input_identities(
                                inputs.repository_root,
                                audit_plans,
                                limits=config.limits,
                                issued_paths=audit_issued_paths,
                            )
                        )
                        retained_keys = {
                            (plan.manifest_path, plan.paths)
                            for plan in retained_plans
                        }
                        invalidated_plans = tuple(
                            plan
                            for plan in audit_plans
                            if (plan.manifest_path, plan.paths)
                            not in retained_keys
                        )
                        safe_audits = tuple(
                            snapshot
                            for snapshot in safe_audits
                            if any(
                                plan.manifest_path == snapshot.manifest_path
                                and (plan.manifest_path, plan.paths)
                                in retained_keys
                                for plan in audit_plans
                            )
                        )
                        invalidated_count = len(audit_plans) - len(
                            retained_plans
                        )
                        if invalidated_count:
                            safe_claims = ()
                            safe_evidence = EvidenceBundle()
                            invalidated_paths = tuple(
                                sorted(
                                    {
                                        path
                                        for plan in invalidated_plans
                                        for path in plan.paths
                                    }
                                )
                            )
                            assert preflight.scope is not None
                            preflight = _sanitize_preflight_material_facts(
                                preflight,
                                preflight.scope,
                                invalidated_paths,
                            )
            except InventoryVerificationError as verification_error:
                if preflight is not None:
                    preflight = _coordinate_failure_preflight(
                        preflight,
                        verification_error.code,
                    )
                safe_claims = ()
                safe_evidence = EvidenceBundle()
                safe_audits = ()
            except Exception:
                if preflight is not None:
                    if preflight.scope is not None and selected_material_paths:
                        sanitized_scope = _sanitize_scope_material_paths(
                            preflight.scope,
                            selected_material_paths,
                        )
                        preflight = _sanitize_preflight_material_facts(
                            preflight,
                            sanitized_scope,
                            selected_material_paths,
                        )
                    else:
                        preflight = _coordinate_failure_preflight(
                            preflight,
                            "PREFLIGHT_G1_SNAPSHOT_IDENTITY_UNVERIFIED",
                        )
                safe_claims = ()
                safe_evidence = EvidenceBundle()
                safe_audits = ()

            # Provider/model text is never included in the safe status message.
            return finish(
                ReviewStatus.UNAVAILABLE,
                claims=safe_claims,
                evidence=safe_evidence,
                deterministic_audits=safe_audits,
                calls=calls,
                error_code="REVIEW_UNAVAILABLE",
                error_message=(
                    "Research review is unavailable "
                    f"({type(exc).__name__})."
                ),
            )

        provider_failure_handler = finish_provider_failure

        def validate_provider_response_authority(
            *,
            retained_claims: tuple[ScientificClaim, ...] = (),
            retained_evidence: EvidenceBundle = EvidenceBundle(),
        ) -> ResearchReview | None:
            """Revalidate frozen authority after every returned response."""

            nonlocal preflight, deterministic_audits

            def capture_drift() -> tuple[set[str], set[str]]:
                captured = _capture_material_scope_identities(
                    inputs.repository_root,
                    selected_material_paths,
                    comparison_base_root=comparison_base_root,
                )
                failures, head_failures = _material_scope_capture_failure_paths(
                    captured,
                    planned_scope.inventory,
                )
                head_drifted, base_drifted = _material_scope_drifted_paths(
                    frozen_material_identities,
                    captured,
                )
                return (
                    set(failures) | set(head_drifted) | set(base_drifted),
                    set(head_failures) | set(head_drifted),
                )

            def material_invalidated(
                invalid: set[str],
                invalid_head: set[str],
            ) -> ResearchReview:
                nonlocal preflight, deterministic_audits
                deterministic_audits = _retain_audits_after_material_drift(
                    deterministic_audits,
                    audit_plans,
                    tuple(sorted(invalid_head)),
                )
                if preflight is None or preflight.scope is None:
                    raise ReviewError("material invalidation has no bound scope")
                preflight = _sanitize_preflight_material_facts(
                    preflight,
                    preflight.scope,
                    tuple(sorted(invalid)),
                )
                return finish(
                    ReviewStatus.UNAVAILABLE,
                    deterministic_audits=deterministic_audits,
                    calls=calls,
                    error_code="MATERIAL_SCOPE_INVALIDATED",
                    error_message=(
                        "Selected material changed while a provider response "
                        "was pending; derived advisory data was discarded."
                    ),
                )

            try:
                if preflight is None or preflight.scope is None:
                    raise ReviewError("provider response has no bound scope")
                preflight = revalidate_preflight_coordinates(inputs, preflight)
                if preflight.gates[0].disposition is GateDisposition.FAIL:
                    preflight = replace(preflight, scope=None)
                    deterministic_audits = ()
                    return finish(
                        ReviewStatus.UNAVAILABLE,
                        calls=calls,
                        error_code=preflight.gates[0].reasons[0].code,
                        error_message=(
                            "Exact review coordinates changed while a provider "
                            "response was pending."
                        ),
                    )

                invalid_paths, invalid_head_paths = capture_drift()
                if not invalid_paths:
                    assert preflight.scope is not None
                    _verify_material_git_binding(
                        inputs,
                        preflight.scope,
                        _capture_material_scope_identities(
                            inputs.repository_root,
                            selected_material_paths,
                            comparison_base_root=comparison_base_root,
                        ),
                    )
                    recaptured_paths, recaptured_head_paths = capture_drift()
                    invalid_paths.update(recaptured_paths)
                    invalid_head_paths.update(recaptured_head_paths)
                if invalid_paths:
                    return material_invalidated(
                        invalid_paths,
                        invalid_head_paths,
                    )

                retained_plans = revalidate_manifest_audit_input_identities(
                    inputs.repository_root,
                    audit_plans,
                    limits=config.limits,
                    issued_paths=audit_issued_paths,
                )
                retained_keys = {
                    (plan.manifest_path, plan.paths) for plan in retained_plans
                }
                invalidated_plans = tuple(
                    plan
                    for plan in audit_plans
                    if (plan.manifest_path, plan.paths) not in retained_keys
                )
                invalidated_count = len(audit_plans) - len(retained_plans)
                if invalidated_count:
                    deterministic_audits = tuple(
                        snapshot
                        for snapshot in deterministic_audits
                        if any(
                            plan.manifest_path == snapshot.manifest_path
                            and (plan.manifest_path, plan.paths)
                            in retained_keys
                            for plan in audit_plans
                        )
                    )
                    invalidated_paths = tuple(
                        sorted(
                            {
                                path
                                for plan in invalidated_plans
                                for path in plan.paths
                            }
                        )
                    )
                    assert preflight is not None and preflight.scope is not None
                    preflight = _sanitize_preflight_material_facts(
                        preflight,
                        preflight.scope,
                        invalidated_paths,
                    )
                    return finish(
                        ReviewStatus.UNAVAILABLE,
                        deterministic_audits=deterministic_audits,
                        calls=calls,
                        error_code="AUDIT_PLANS_INVALIDATED",
                        error_message=(
                            f"{invalidated_count} deterministic Audit input "
                            "bundle(s) changed while a provider response was "
                            "pending."
                        ),
                    )

                final_paths, final_head_paths = capture_drift()
                if final_paths:
                    return material_invalidated(
                        final_paths,
                        final_head_paths,
                    )
                return None
            except InventoryVerificationError as exc:
                if preflight is not None:
                    preflight = _coordinate_failure_preflight(
                        preflight,
                        exc.code,
                    )
                deterministic_audits = ()
                return finish(
                    ReviewStatus.UNAVAILABLE,
                    calls=calls,
                    error_code=exc.code,
                    error_message=(
                        "Selected review material no longer matched the exact "
                        "declared Git objects."
                    ),
                )
            except Exception as exc:
                return finish_provider_failure(
                    exc,
                    retained_claims=retained_claims,
                    retained_evidence=retained_evidence,
                )

        sources = scope_source_bundle(refreshed_scope)
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
                deterministic_audits=deterministic_audits,
                error_code="CONTEXT_LIMIT",
                error_message="Extraction context exceeds the configured limit.",
            )
        if provider is None:
            if config.provider != "openai":
                return finish(
                    ReviewStatus.UNAVAILABLE,
                    deterministic_audits=deterministic_audits,
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
        provider_lifecycle = ProviderLifecycle.FAILED_BEFORE_RESPONSE
        provider_attempt_count += 1
        try:
            extraction_response = provider.extract_claims(extraction_request)
        except Exception as exc:
            return finish_provider_failure(exc)
        extraction_call = _record_call(
            "extract_claims", extraction_response, extraction_chars
        )
        calls.append(extraction_call)
        provider_lifecycle = ProviderLifecycle.RESPONSE_RECEIVED
        response_authority_failure = validate_provider_response_authority()
        if response_authority_failure is not None:
            return response_authority_failure
        if not extraction_response.complete:
            truncated = extraction_response.incomplete_reason == "max_output_tokens"
            return finish(
                ReviewStatus.PARTIAL,
                calls=calls,
                deterministic_audits=deterministic_audits,
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
            invalidated_plans = tuple(
                plan
                for plan in audit_plans
                if (plan.manifest_path, plan.paths)
                not in retained_audit_keys
            )
            invalidated_paths = tuple(
                sorted(
                    {
                        path
                        for plan in invalidated_plans
                        for path in plan.paths
                    }
                )
            )
            assert preflight is not None and preflight.scope is not None
            preflight = _sanitize_preflight_material_facts(
                preflight,
                preflight.scope,
                invalidated_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="AUDIT_PLANS_INVALIDATED",
                error_message=(
                    f"{invalidated_audit_plan_count} exact deterministic Audit "
                    "input bundle(s) changed or became unavailable after the "
                    "pre-provider snapshot; synthesis was skipped."
                ),
            )
        try:
            post_material_identities = _capture_material_scope_identities(
                inputs.repository_root,
                selected_material_paths,
                comparison_base_root=comparison_base_root,
            )
            (
                head_material_drifted_paths,
                base_material_drifted_paths,
            ) = _material_scope_drifted_paths(
                frozen_material_identities,
                post_material_identities,
            )
        except ReviewError:
            head_material_drifted_paths = tuple(selected_material_paths)
            base_material_drifted_paths = ()
        material_drifted_paths = tuple(
            sorted(
                set(head_material_drifted_paths)
                | set(base_material_drifted_paths)
            )
        )
        if material_drifted_paths:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                head_material_drifted_paths,
            )
            assert preflight is not None and preflight.scope is not None
            preflight = _sanitize_preflight_material_facts(
                preflight,
                preflight.scope,
                material_drifted_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="MATERIAL_SCOPE_INVALIDATED",
                error_message=(
                    "A selected material identity changed or became unavailable "
                    "after extraction; evidence discovery and synthesis were "
                    "skipped."
                ),
            )
        claim_validation = validate_claim_candidates_best_effort(
            _parse_json(extraction_response.output_text),
            sources,
            max_claims=config.limits.max_claims,
        )
        claims = claim_validation.claims
        failure_claims = claims
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
        try:
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
                materialized_path_chars=(
                    preflight.scope.materialized_path_chars
                    if preflight is not None and preflight.scope is not None
                    else None
                ),
                material_identities=evidence_material_identities,
            )
        except EvidenceMaterialInvalidated as exc:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                (exc.path,) if exc.head else (),
            )
            assert preflight is not None and preflight.scope is not None
            preflight = _sanitize_preflight_material_facts(
                preflight,
                preflight.scope,
                (exc.path,),
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="MATERIAL_SCOPE_INVALIDATED",
                error_message=(
                    "Selected material changed during descriptor-bound evidence "
                    "discovery; synthesis was skipped."
                ),
            )
        failure_evidence = evidence
        evidence_identity_drifted_paths = _evidence_identity_drifted_paths(
            evidence,
            evidence_material_identities,
        )
        if evidence_identity_drifted_paths:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                evidence_identity_drifted_paths,
            )
            assert preflight is not None and preflight.scope is not None
            preflight = _sanitize_preflight_material_facts(
                preflight,
                preflight.scope,
                evidence_identity_drifted_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="MATERIAL_SCOPE_INVALIDATED",
                error_message=(
                    "Evidence identities did not match the frozen selected "
                    "material; synthesis was skipped."
                ),
            )
        try:
            post_evidence_material_identities = _capture_material_scope_identities(
                inputs.repository_root,
                selected_material_paths,
                comparison_base_root=comparison_base_root,
            )
            (
                post_evidence_capture_failures,
                post_evidence_head_failures,
            ) = _material_scope_capture_failure_paths(
                post_evidence_material_identities,
                planned_scope.inventory,
            )
            (
                post_evidence_head_drifted_paths,
                post_evidence_base_drifted_paths,
            ) = _material_scope_drifted_paths(
                frozen_material_identities,
                post_evidence_material_identities,
            )
        except ReviewError:
            post_evidence_capture_failures = tuple(selected_material_paths)
            post_evidence_head_failures = tuple(selected_material_paths)
            post_evidence_head_drifted_paths = tuple(selected_material_paths)
            post_evidence_base_drifted_paths = ()
        post_evidence_drifted_paths = tuple(
            sorted(
                set(post_evidence_capture_failures)
                | set(post_evidence_head_drifted_paths)
                | set(post_evidence_base_drifted_paths)
            )
        )
        if post_evidence_drifted_paths:
            deterministic_audits = _retain_audits_after_material_drift(
                deterministic_audits,
                audit_plans,
                tuple(
                    sorted(
                        set(post_evidence_head_failures)
                        | set(post_evidence_head_drifted_paths)
                    )
                ),
            )
            assert preflight is not None and preflight.scope is not None
            preflight = _sanitize_preflight_material_facts(
                preflight,
                preflight.scope,
                post_evidence_drifted_paths,
            )
            return finish(
                ReviewStatus.UNAVAILABLE,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="MATERIAL_SCOPE_INVALIDATED",
                error_message=(
                    "Selected material changed during evidence processing; "
                    "synthesis was skipped."
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
        synthesis_evidence = _order_synthesis_evidence(
            claims,
            tuple(
                reference
                for reference in evidence.references
                if _citable_reference(reference)
            ),
            priority_paths=priority_paths,
            selected_paths=(
                preflight.scope.selected_paths
                if preflight is not None and preflight.scope is not None
                else tuple(sorted(selected_paths))
            ),
        )
        synthesis_allocation = allocate_synthesis_inputs(
            claims,
            synthesis_evidence,
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
        if omission_counts["interpretation_constraints"]:
            return finish(
                ReviewStatus.PARTIAL,
                claims=claims,
                evidence=evidence,
                deterministic_audits=deterministic_audits,
                calls=calls,
                error_code="SYNTHESIS_INPUTS_OMITTED",
                error_message=(
                    "Bounded advisory synthesis was skipped because deterministic "
                    "interpretation constraints could not be issued with all of "
                    "their required evidence. Exact omitted input counts: "
                    f"evidence={omission_counts['evidence']}, "
                    "interpretation_constraints="
                    f"{omission_counts['interpretation_constraints']}, "
                    f"missing_evidence={omission_counts['missing_evidence']}, "
                    "deterministic_audits="
                    f"{omission_counts['deterministic_audits']}. Full evidence and "
                    "deterministic Audit results remain in the review record."
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
        provider_lifecycle = ProviderLifecycle.FAILED_BEFORE_RESPONSE
        provider_attempt_count += 1
        try:
            synthesis_response = provider.synthesize_review(synthesis_request)
        except Exception as exc:
            return finish_provider_failure(
                exc,
                retained_claims=claims,
                retained_evidence=evidence,
            )
        synthesis_call = _record_call(
            "synthesize_review", synthesis_response, synthesis_chars
        )
        calls.append(synthesis_call)
        provider_lifecycle = ProviderLifecycle.RESPONSE_RECEIVED
        response_authority_failure = validate_provider_response_authority(
            retained_claims=claims,
            retained_evidence=evidence,
        )
        if response_authority_failure is not None:
            return response_authority_failure
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
                synthesis_parts.payload["evidence_ids_by_claim_id"],
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
        if provider_failure_handler is not None:
            return provider_failure_handler(
                exc,
                retained_claims=failure_claims,
                retained_evidence=failure_evidence,
            )
        # Provider/model text is never included in the safe status message.
        return finish(
            ReviewStatus.UNAVAILABLE,
            calls=calls,
            error_code="REVIEW_UNAVAILABLE",
            error_message=f"Research review is unavailable ({type(exc).__name__}).",
        )
