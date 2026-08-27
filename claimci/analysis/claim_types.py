"""Canonical source-bound scientific claim taxonomy and v0 compiler."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from claimci.models import Direction

from .contracts import (
    AuditClaimSpec,
    ClaimReference,
    ClaimedMetricValue,
    ExperimentRole,
    FieldProvenance,
)


UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER = (
    "unsupported_deterministic_claim_compiler"
)

_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_METRIC_IMPROVEMENT = re.compile(
    r"\b(?P<metric>[A-Za-z][A-Za-z0-9_.-]{0,63})\s+"
    r"(?:score\s+)?"
    r"(?P<verb>improved|increased|rose|grew|decreased|fell|dropped|reduced)\b",
    flags=re.IGNORECASE,
)
_SUBJECT_METRIC_IMPROVEMENT = re.compile(
    r"\b(?:the\s+)?(?:candidate|model)\s+"
    r"(?P<verb>improves|increases|raises|grows|reduces|decreases|lowers|drops)\s+"
    r"(?P<metric>[A-Za-z][A-Za-z0-9_.-]{0,63})\b",
    flags=re.IGNORECASE,
)
_VALUE_PAIR = re.compile(
    rf"(?:from\s+)?(?P<baseline>{_NUMBER})\s*(?P<baseline_unit>%?)\s*"
    rf"(?:->|→|to)\s*(?P<candidate>{_NUMBER})\s*(?P<candidate_unit>%?)",
    flags=re.IGNORECASE,
)
_MINIMUM = re.compile(
    rf"(?:\bby\s+)?(?:\bat\s+least\b|\bminimum(?:\s+of)?\b|>=|"
    rf"\bno\s+less\s+than\b)\s*(?P<value>{_NUMBER})"
    rf"(?:\s*(?P<unit>%|percentage\s+points?|points?))?(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_MINIMUM_CONTINUATION = re.compile(
    rf"^\s*(?P<verb>improved|increased|rose|grew|decreased|fell|dropped|reduced)"
    rf"\s+by\s+(?:at\s+least|minimum(?:\s+of)?|>=|no\s+less\s+than)\s*"
    rf"(?P<value>{_NUMBER})"
    rf"(?:\s*(?P<unit>%|percentage\s+points?|points?))?\s*\.?\s*$",
    flags=re.IGNORECASE,
)
_THRESHOLD_TRAILING = re.compile(
    rf"^\s*(?:(?:across|aggregated|as|for|in|on|over|under|using|via|with)\b"
    rf"(?:\s+(?:{_NUMBER}|[A-Za-z][A-Za-z0-9_.-]{{0,63}})){{1,16}})?\s*$",
    flags=re.IGNORECASE,
)
_ABSOLUTE_METRIC = re.compile(
    rf"\b(?P<role>baseline|candidate)\s+"
    rf"(?P<metric>[A-Za-z][A-Za-z0-9_.-]{{0,63}})\s+"
    rf"(?:score\s+)?(?P<comparator>"
    rf"(?:(?:is|was|reached|achieved)\s+)?(?:at\s+least|at\s+most|"
    rf"no\s+less\s+than|no\s+more\s+than|>=|<=)|"
    rf"equals?|=|is|was|reached|achieved)\s*"
    rf"(?P<value>{_NUMBER})(?:\s*(?P<unit>%|[A-Za-z][A-Za-z0-9_.-]{{0,31}}))?",
    flags=re.IGNORECASE,
)
_GENERALIZATION = re.compile(
    r"\b(?:generaliz(?:e|es|ed|ation)|generalisation)\b",
    flags=re.IGNORECASE,
)
_GENERALIZATION_SCOPE = re.compile(
    r"\b(?:across|unseen|new|other|out-of-domain|out\s+of\s+domain|"
    r"domains?|datasets?|tasks?|populations?)\b",
    flags=re.IGNORECASE,
)
_HELD_OUT = re.compile(r"\bheld(?:-|\s+)out\b", flags=re.IGNORECASE)
_NUMBER_WITH_UNIT = re.compile(
    rf"(?P<value>{_NUMBER})(?:\s*(?P<unit>%|[A-Za-z][A-Za-z0-9_.-]{{0,31}}))?",
    flags=re.IGNORECASE,
)
_GENERIC_BOUND = re.compile(
    r"(?:\bat\s+least\b|\bat\s+most\b|\bno\s+less\s+than\b|"
    r"\bno\s+more\s+than\b|>=|<=|\b(?:less|more)\b|"
    r"\b(?:increase|increased|reduction|reduced|decrease|decreased)\b)",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_CLAUSE_BOUNDARY = re.compile(r"[;!?\n]|\.(?=\s|$)")
_LOWER_VERBS = frozenset(
    {
        "decreased",
        "fell",
        "dropped",
        "reduced",
        "reduces",
        "decreases",
        "lowers",
        "drops",
    }
)


class ClaimTypeContractError(ValueError):
    """A canonical claim value is malformed or conflicts with its source."""


class UnsupportedDeterministicClaimCompiler(ClaimTypeContractError):
    """A recognized primary claim has no deterministic v0 compiler."""


class PrimaryClaimKind(str, Enum):
    METRIC_IMPROVEMENT = "metric_improvement"
    ABSOLUTE_METRIC = "absolute_metric"
    GENERALIZATION = "generalization"
    GENERIC_QUANTITATIVE = "generic_quantitative"


class ClaimBoundComparator(str, Enum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    EQUAL = "equal"


class EvaluationConstraintKind(str, Enum):
    HELD_OUT = "held_out"


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ClaimTypeContractError(f"{label} must be bounded non-empty text")
    if value != value.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ClaimTypeContractError(f"{label} must be canonical text")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ClaimTypeContractError(f"{label} must be finite")
    return normalized


def _unit(raw: str | None) -> str | None:
    if not raw:
        return None
    normalized = _WHITESPACE.sub(" ", raw.strip().casefold())
    if normalized.startswith("percentage point"):
        return "percentage points"
    if normalized.startswith("point"):
        return "points"
    return normalized


@dataclass(frozen=True, slots=True)
class ClaimQuantity:
    value: float
    unit: str | None
    raw: str
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, "claim quantity"))
        if self.unit is not None:
            _bounded_text(self.unit, "claim quantity unit", maximum=64)
        _bounded_text(self.raw, "claim quantity raw token", maximum=512)
        if type(self.provenance) is not FieldProvenance:
            raise TypeError("claim quantity provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class MetricImprovementClaim:
    metric: str
    direction: Direction
    metric_provenance: FieldProvenance
    direction_provenance: FieldProvenance
    baseline_value: ClaimQuantity | None = None
    candidate_value: ClaimQuantity | None = None
    minimum_improvement: ClaimQuantity | None = None
    kind: PrimaryClaimKind = field(
        default=PrimaryClaimKind.METRIC_IMPROVEMENT,
        init=False,
    )

    def __post_init__(self) -> None:
        metric = _bounded_text(self.metric, "metric improvement metric", maximum=256)
        if metric != metric.casefold():
            raise ClaimTypeContractError("metric improvement metric must be canonical")
        if type(self.direction) is not Direction:
            raise TypeError("metric improvement direction must be Direction")
        for value in (self.metric_provenance, self.direction_provenance):
            if type(value) is not FieldProvenance:
                raise TypeError("metric improvement provenance must be FieldProvenance")
        if (self.baseline_value is None) != (self.candidate_value is None):
            raise ClaimTypeContractError(
                "metric improvement values must appear as a baseline/candidate pair"
            )
        for value in (
            self.baseline_value,
            self.candidate_value,
            self.minimum_improvement,
        ):
            if value is not None and type(value) is not ClaimQuantity:
                raise TypeError("metric improvement values must be ClaimQuantity values")
        if (
            self.minimum_improvement is not None
            and self.minimum_improvement.value < 0
        ):
            raise ClaimTypeContractError(
                "minimum metric improvement must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class AbsoluteMetricClaim:
    metric: str
    role: ExperimentRole
    comparator: ClaimBoundComparator
    bound: ClaimQuantity
    metric_provenance: FieldProvenance
    comparator_provenance: FieldProvenance
    kind: PrimaryClaimKind = field(default=PrimaryClaimKind.ABSOLUTE_METRIC, init=False)

    def __post_init__(self) -> None:
        metric = _bounded_text(self.metric, "absolute metric", maximum=256)
        if metric != metric.casefold():
            raise ClaimTypeContractError("absolute metric must be canonical")
        if self.role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}:
            raise ClaimTypeContractError("absolute metric role must be a single run role")
        if type(self.comparator) is not ClaimBoundComparator:
            raise TypeError("absolute metric comparator must be ClaimBoundComparator")
        if type(self.bound) is not ClaimQuantity:
            raise TypeError("absolute metric bound must be ClaimQuantity")
        for value in (self.metric_provenance, self.comparator_provenance):
            if type(value) is not FieldProvenance:
                raise TypeError("absolute metric provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class GeneralizationClaim:
    subject: str
    provenance: FieldProvenance
    kind: PrimaryClaimKind = field(default=PrimaryClaimKind.GENERALIZATION, init=False)

    def __post_init__(self) -> None:
        _bounded_text(self.subject, "generalization subject", maximum=1_024)
        if type(self.provenance) is not FieldProvenance:
            raise TypeError("generalization provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class GenericQuantitativeClaim:
    subject: str
    quantities: tuple[ClaimQuantity, ...]
    provenance: FieldProvenance
    kind: PrimaryClaimKind = field(
        default=PrimaryClaimKind.GENERIC_QUANTITATIVE,
        init=False,
    )

    def __post_init__(self) -> None:
        _bounded_text(self.subject, "generic quantitative subject", maximum=1_024)
        if (
            not isinstance(self.quantities, tuple)
            or not 1 <= len(self.quantities) <= 8
            or not all(type(item) is ClaimQuantity for item in self.quantities)
        ):
            raise ClaimTypeContractError(
                "generic quantitative quantities must be a bounded ClaimQuantity tuple"
            )
        if type(self.provenance) is not FieldProvenance:
            raise TypeError("generic quantitative provenance must be FieldProvenance")


PrimaryScientificClaim: TypeAlias = (
    MetricImprovementClaim
    | AbsoluteMetricClaim
    | GeneralizationClaim
    | GenericQuantitativeClaim
)


@dataclass(frozen=True, slots=True)
class EvaluationConstraint:
    kind: EvaluationConstraintKind
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        if type(self.kind) is not EvaluationConstraintKind:
            raise TypeError("evaluation constraint kind must be EvaluationConstraintKind")
        if type(self.provenance) is not FieldProvenance:
            raise TypeError("evaluation constraint provenance must be FieldProvenance")


ClaimConstraint: TypeAlias = EvaluationConstraint


@dataclass(frozen=True, slots=True)
class CanonicalScientificClaim:
    reference: ClaimReference
    primary: PrimaryScientificClaim
    constraints: tuple[ClaimConstraint, ...] = ()

    def __post_init__(self) -> None:
        if type(self.reference) is not ClaimReference:
            raise TypeError("canonical claim reference must be ClaimReference")
        if type(self.primary) not in {
            MetricImprovementClaim,
            AbsoluteMetricClaim,
            GeneralizationClaim,
            GenericQuantitativeClaim,
        }:
            raise TypeError("canonical claim primary type is invalid")
        if not isinstance(self.constraints, tuple) or not all(
            type(item) is EvaluationConstraint for item in self.constraints
        ):
            raise TypeError("canonical claim constraints must be EvaluationConstraint values")
        kinds = tuple(item.kind for item in self.constraints)
        if len(set(kinds)) != len(kinds):
            raise ClaimTypeContractError("canonical claim constraints must be unique")


@dataclass(frozen=True, slots=True)
class ClaimEvidencePolicy:
    policy_id: str
    primary_kind: PrimaryClaimKind
    constraint_kinds: tuple[EvaluationConstraintKind, ...]
    deterministic_compiler_id: str | None
    obligation_template_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _bounded_text(self.policy_id, "claim evidence policy_id", maximum=128)
        if type(self.primary_kind) is not PrimaryClaimKind:
            raise TypeError("claim evidence primary kind must be PrimaryClaimKind")
        if not isinstance(self.constraint_kinds, tuple) or not all(
            type(item) is EvaluationConstraintKind for item in self.constraint_kinds
        ):
            raise TypeError(
                "claim evidence constraint kinds must be EvaluationConstraintKind values"
            )
        if len(set(self.constraint_kinds)) != len(self.constraint_kinds):
            raise ClaimTypeContractError(
                "claim evidence constraint kinds must be unique"
            )
        if self.deterministic_compiler_id is not None:
            _bounded_text(
                self.deterministic_compiler_id,
                "deterministic compiler_id",
                maximum=128,
            )
        if (
            not isinstance(self.obligation_template_ids, tuple)
            or len(self.obligation_template_ids) > 16
            or not all(
                isinstance(item, str)
                and item
                and item == item.strip()
                and len(item) <= 128
                and not any(
                    ord(character) < 32 or ord(character) == 127
                    for character in item
                )
                for item in self.obligation_template_ids
            )
        ):
            raise ClaimTypeContractError(
                "claim evidence obligation templates must be bounded identifiers"
            )
        if len(set(self.obligation_template_ids)) != len(
            self.obligation_template_ids
        ):
            raise ClaimTypeContractError(
                "claim evidence obligation templates must be unique"
            )


def _quantity(match: re.Match[str], provenance: FieldProvenance) -> ClaimQuantity:
    raw_value = match.group("value")
    raw_unit = match.groupdict().get("unit")
    unit = _unit(raw_unit)
    raw = raw_value if not raw_unit else f"{raw_value} {raw_unit}"
    return ClaimQuantity(float(raw_value), unit, raw, provenance)


def _paired_quantity(
    match: re.Match[str],
    prefix: str,
    provenance: FieldProvenance,
) -> ClaimQuantity:
    raw_value = match.group(prefix)
    raw_unit = match.group(f"{prefix}_unit")
    unit = _unit(raw_unit)
    raw = raw_value if not raw_unit else f"{raw_value} {raw_unit}"
    return ClaimQuantity(float(raw_value), unit, raw, provenance)


def _constraints(
    text: str,
    provenance: FieldProvenance,
) -> tuple[ClaimConstraint, ...]:
    if _HELD_OUT.search(text) is None:
        return ()
    return (
        EvaluationConstraint(
            kind=EvaluationConstraintKind.HELD_OUT,
            provenance=provenance,
        ),
    )


def _only_separators(value: str) -> bool:
    return not any(character.isalnum() or character == "_" for character in value)


def _only_whitespace(value: str) -> bool:
    return not value.strip()


def _valid_threshold_trailing(value: str) -> bool:
    return _THRESHOLD_TRAILING.fullmatch(value) is not None


def _verb_direction(verb: str) -> Direction:
    return (
        Direction.LOWER
        if verb.casefold() in _LOWER_VERBS
        else Direction.HIGHER
    )


def _recover_primary(
    text: str,
    provenance: FieldProvenance,
) -> PrimaryScientificClaim | None:
    improvements = tuple(
        sorted(
            (
                *_METRIC_IMPROVEMENT.finditer(text),
                *_SUBJECT_METRIC_IMPROVEMENT.finditer(text),
            ),
            key=lambda match: match.start(),
        )
    )
    if len(improvements) == 1:
        improvement = improvements[0]
        tail = text[improvement.start() :]
        continuation_threshold: re.Match[str] | None = None
        boundary = _CLAUSE_BOUNDARY.search(tail)
        if boundary is not None:
            if boundary.group(0) == ";":
                continuation = tail[boundary.end() :]
                candidate = _MINIMUM_CONTINUATION.fullmatch(continuation)
                if (
                    candidate is not None
                    and _verb_direction(candidate.group("verb"))
                    is _verb_direction(improvement.group("verb"))
                ):
                    continuation_threshold = candidate
                elif len(tuple(_MINIMUM.finditer(continuation))) > 1:
                    return None
            tail = tail[: boundary.start()]
        pairs = tuple(_VALUE_PAIR.finditer(tail))
        thresholds = tuple(_MINIMUM.finditer(tail))
        if len(pairs) > 1 or len(thresholds) + int(
            continuation_threshold is not None
        ) > 1:
            return None
        improvement_end = improvement.end() - improvement.start()
        pair = pairs[0] if pairs else None
        if pair is not None and not _only_separators(
            tail[improvement_end : pair.start()]
        ):
            pair = None
        threshold = thresholds[0] if thresholds else continuation_threshold
        threshold_anchor = improvement_end if pair is None else pair.end()
        if thresholds and not _only_whitespace(
            tail[threshold_anchor : threshold.start()]
        ):
            threshold = None
        if (
            threshold is not None
            and threshold is not continuation_threshold
            and not _valid_threshold_trailing(tail[threshold.end() :])
        ):
            threshold = None
        direction = _verb_direction(improvement.group("verb"))
        baseline = candidate = None
        if pair is not None:
            baseline = _paired_quantity(pair, "baseline", provenance)
            candidate = _paired_quantity(pair, "candidate", provenance)
        minimum = _quantity(threshold, provenance) if threshold is not None else None
        return MetricImprovementClaim(
            metric=improvement.group("metric").casefold(),
            direction=direction,
            metric_provenance=provenance,
            direction_provenance=provenance,
            baseline_value=baseline,
            candidate_value=candidate,
            minimum_improvement=minimum,
        )
    if len(improvements) > 1:
        return None

    absolute = tuple(_ABSOLUTE_METRIC.finditer(text))
    if len(absolute) == 1:
        match = absolute[0]
        raw_comparator = _WHITESPACE.sub(
            " ",
            match.group("comparator").casefold(),
        )
        if raw_comparator.endswith(("at least", "no less than", ">=")):
            comparator = ClaimBoundComparator.AT_LEAST
        elif raw_comparator.endswith(("at most", "no more than", "<=")):
            comparator = ClaimBoundComparator.AT_MOST
        else:
            comparator = ClaimBoundComparator.EQUAL
        return AbsoluteMetricClaim(
            metric=match.group("metric").casefold(),
            role=ExperimentRole(match.group("role").casefold()),
            comparator=comparator,
            bound=_quantity(match, provenance),
            metric_provenance=provenance,
            comparator_provenance=provenance,
        )
    if len(absolute) > 1:
        return None

    if _GENERALIZATION.search(text) is not None and _GENERALIZATION_SCOPE.search(
        text
    ) is not None:
        return GeneralizationClaim(subject="candidate", provenance=provenance)

    numbers = tuple(_NUMBER_WITH_UNIT.finditer(text))
    if len(numbers) == 1 and _GENERIC_BOUND.search(text) is not None:
        return GenericQuantitativeClaim(
            subject="quantitative claim",
            quantities=(_quantity(numbers[0], provenance),),
            provenance=provenance,
        )
    return None


def recover_scientific_claim(
    reference: ClaimReference,
) -> CanonicalScientificClaim | None:
    """Recover one canonical claim solely from a validated source reference."""

    if type(reference) is not ClaimReference:
        raise TypeError("claim recovery requires ClaimReference")
    if "\x00" in reference.text:
        return None
    text = _WHITESPACE.sub(" ", reference.text.strip())
    if not text:
        return None
    primary = _recover_primary(text, reference.provenance)
    if primary is None:
        return None
    return CanonicalScientificClaim(
        reference=reference,
        primary=primary,
        constraints=_constraints(text, reference.provenance),
    )


def claim_evidence_policy(claim: CanonicalScientificClaim) -> ClaimEvidencePolicy:
    """Return the stable v0 policy seam without evaluating obligations."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("claim evidence policy requires CanonicalScientificClaim")
    kind = claim.primary.kind
    compiler = None
    if type(claim.primary) is MetricImprovementClaim and (
        claim.primary.minimum_improvement is None
        or claim.primary.minimum_improvement.unit is None
    ):
        compiler = "metric-improvement-v0"
    from .measurement import recover_upstream_procedure_requirement

    procedure_requirement = recover_upstream_procedure_requirement(claim.reference)
    templates: list[str] = []
    if type(claim.primary) is MetricImprovementClaim:
        templates.extend(("claim.metric", "claim.direction", "claim.threshold"))
    elif type(claim.primary) is AbsoluteMetricClaim:
        templates.extend(("claim.metric", "claim.quantitative_bound"))
    elif type(claim.primary) is GenericQuantitativeClaim:
        templates.append("claim.quantitative_bound")
    if any(
        item.kind is EvaluationConstraintKind.HELD_OUT
        for item in claim.constraints
    ):
        templates.append("claim.constraint.held_out")
    if compiler is not None:
        templates.extend(
            (
                "artifact.baseline.results.metric",
                "artifact.candidate.results.metric",
                "artifact.baseline.config",
                "artifact.candidate.config",
                "artifact.baseline.dataset.train",
                "artifact.baseline.dataset.eval",
                "artifact.candidate.dataset.train",
                "artifact.candidate.dataset.eval",
            )
        )
        if procedure_requirement is not None:
            templates.append("measurement.retry_aggregation")
        templates.append("comparison.readiness")
    policy_id = f"claimci.claim-policy.{kind.value}.v0"
    if procedure_requirement is not None:
        policy_id += "+measurement-procedure-v1"
    return ClaimEvidencePolicy(
        policy_id=policy_id,
        primary_kind=kind,
        constraint_kinds=tuple(item.kind for item in claim.constraints),
        deterministic_compiler_id=compiler,
        obligation_template_ids=tuple(templates),
    )


def compile_audit_claim(claim: CanonicalScientificClaim) -> AuditClaimSpec:
    """Compile only an exact source-recovered metric-improvement claim."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("deterministic claim compiler requires CanonicalScientificClaim")
    recovered = recover_scientific_claim(claim.reference)
    if recovered != claim:
        raise ClaimTypeContractError(
            "canonical claim does not match independent source recovery"
        )
    primary = claim.primary
    if type(primary) is not MetricImprovementClaim:
        raise UnsupportedDeterministicClaimCompiler(
            UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER
        )
    if (
        primary.minimum_improvement is not None
        and primary.minimum_improvement.unit is not None
    ):
        raise UnsupportedDeterministicClaimCompiler(
            UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER
        )
    claimed_values: tuple[ClaimedMetricValue, ...] = ()
    if primary.baseline_value is not None and primary.candidate_value is not None:
        claimed_values = (
            ClaimedMetricValue(
                role=ExperimentRole.BASELINE,
                value=primary.baseline_value.value,
                unit=primary.baseline_value.unit,
                provenance=primary.baseline_value.provenance,
            ),
            ClaimedMetricValue(
                role=ExperimentRole.CANDIDATE,
                value=primary.candidate_value.value,
                unit=primary.candidate_value.unit,
                provenance=primary.candidate_value.provenance,
            ),
        )
    threshold = (
        None
        if primary.minimum_improvement is None
        else primary.minimum_improvement.value
    )
    threshold_provenance = (
        None
        if primary.minimum_improvement is None
        else primary.minimum_improvement.provenance
    )
    return AuditClaimSpec(
        claim_id=claim.reference.claim_id,
        metric=primary.metric,
        direction=primary.direction,
        minimum_absolute_improvement=threshold,
        metric_provenance=primary.metric_provenance,
        direction_provenance=primary.direction_provenance,
        threshold_provenance=threshold_provenance,
        claimed_values=claimed_values,
    )


def _quantity_projection(value: ClaimQuantity | None) -> object:
    if value is None:
        return None
    return {"value": value.value, "unit": value.unit}


def claim_semantic_projection(claim: CanonicalScientificClaim) -> dict[str, object]:
    """Return a bounded semantic projection for provenance, never authority."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("claim semantic projection requires CanonicalScientificClaim")
    primary = claim.primary
    projection: dict[str, object] = {"kind": primary.kind.value}
    if type(primary) is MetricImprovementClaim:
        projection.update(
            {
                "metric": primary.metric,
                "direction": primary.direction.value,
                "baseline_value": _quantity_projection(primary.baseline_value),
                "candidate_value": _quantity_projection(primary.candidate_value),
                "minimum_improvement": _quantity_projection(
                    primary.minimum_improvement
                ),
            }
        )
    elif type(primary) is AbsoluteMetricClaim:
        projection.update(
            {
                "metric": primary.metric,
                "role": primary.role.value,
                "comparator": primary.comparator.value,
                "bound": _quantity_projection(primary.bound),
            }
        )
    elif type(primary) is GeneralizationClaim:
        projection["subject"] = primary.subject
    else:
        projection.update(
            {
                "subject": primary.subject,
                "quantities": [
                    _quantity_projection(item) for item in primary.quantities
                ],
            }
        )
    result: dict[str, object] = {
        "primary": projection,
        "constraints": [item.kind.value for item in claim.constraints],
    }
    from .measurement import recover_upstream_procedure_requirement

    procedure = recover_upstream_procedure_requirement(claim.reference)
    if procedure is not None:
        result["measurement_requirements"] = [
            {
                "component_kind": "retry_aggregation",
                "procedure": procedure.procedure.value,
            }
        ]
    return result


__all__ = [
    "AbsoluteMetricClaim",
    "CanonicalScientificClaim",
    "ClaimBoundComparator",
    "ClaimConstraint",
    "ClaimEvidencePolicy",
    "ClaimQuantity",
    "ClaimTypeContractError",
    "EvaluationConstraint",
    "EvaluationConstraintKind",
    "GeneralizationClaim",
    "GenericQuantitativeClaim",
    "MetricImprovementClaim",
    "PrimaryClaimKind",
    "PrimaryScientificClaim",
    "UNSUPPORTED_DETERMINISTIC_CLAIM_COMPILER",
    "UnsupportedDeterministicClaimCompiler",
    "claim_evidence_policy",
    "claim_semantic_projection",
    "compile_audit_claim",
    "recover_scientific_claim",
]
