"""Bounded deterministic evidence-obligation contracts and support identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeAlias

from .claim_types import (
    AbsoluteMetricClaim,
    CanonicalScientificClaim,
    ClaimEvidencePolicy,
    EvaluationConstraintKind,
    GenericQuantitativeClaim,
    MetricImprovementClaim,
    UnsupportedDeterministicClaimCompiler,
    claim_semantic_projection,
    compile_audit_claim,
    recover_scientific_claim,
)
from .contracts import (
    AnalysisContractError,
    ArtifactBinding,
    ArtifactCandidate,
    ArtifactKind,
    AuditClaimSpec,
    DatasetSplit,
    ExperimentRole,
    MappingCandidate,
    MappingQuestion,
    MissingEvidence,
    NormalizedEvidence,
    ProvenanceKind,
    RepoMapping,
)


class EvidenceObligationState(str, Enum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


class EvidenceObligationReason(str, Enum):
    VALIDATED_SUPPORT_BOUND = "validated_support_bound"
    VALIDATED_DEPENDENCIES_SATISFIED = "validated_dependencies_satisfied"
    REQUIRED_ARTIFACT_NOT_FOUND = "required_artifact_not_found"
    REQUIRED_CLAIM_FIELD_NOT_RECOVERED = "required_claim_field_not_recovered"
    REQUIRED_METRIC_NOT_RECOVERED = "required_metric_not_recovered"
    REQUIRED_THRESHOLD_NOT_RECOVERED = "required_threshold_not_recovered"
    REQUIRED_ROLE_NOT_RECOVERED = "required_role_not_recovered"
    REQUIRED_SPLIT_NOT_RECOVERED = "required_split_not_recovered"
    DEPENDENCY_MISSING = "dependency_missing"
    MULTIPLE_VALIDATED_CANDIDATES = "multiple_validated_candidates"
    CONFLICTING_VALIDATED_MAPPINGS = "conflicting_validated_mappings"
    EXPLICIT_MAPPING_APPROVAL_REQUIRED = "explicit_mapping_approval_required"
    APPROVED_MAPPING_NOT_APPLICABLE = "approved_mapping_not_applicable"
    CLARIFICATION_NOT_BOUNDED = "clarification_not_bounded"
    CLAIM_TYPE_NOT_EXECUTABLE = "claim_type_not_executable"
    ARTIFACT_FORMAT_NOT_SUPPORTED = "artifact_format_not_supported"
    ARTIFACT_EXCEEDS_PASSIVE_LIMIT = "artifact_exceeds_passive_limit"
    ADAPTER_NOT_AVAILABLE = "adapter_not_available"
    SELECTOR_NOT_RECOVERABLE = "selector_not_recoverable"
    NATIVE_REPRESENTATION_NOT_SUPPORTED = "native_representation_not_supported"
    DEPENDENCY_UNSUPPORTED = "dependency_unsupported"


class EvidenceObligationEffect(str, Enum):
    NONE = "none"
    BLOCKS_PARTIAL = "blocks_partial"
    REQUIRES_MAPPING = "requires_mapping"


class EvidenceObligationDecision(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    MAPPING_NEEDED = "mapping_needed"


class ClaimFieldKind(str, Enum):
    METRIC = "metric"
    DIRECTION = "direction"
    THRESHOLD = "threshold"
    QUANTITATIVE_BOUND = "quantitative_bound"
    EVALUATION_CONSTRAINT = "evaluation_constraint"


@dataclass(frozen=True, slots=True)
class ClaimFieldTarget:
    field: ClaimFieldKind
    constraint_kind: EvaluationConstraintKind | None = None

    def __post_init__(self) -> None:
        if type(self.field) is not ClaimFieldKind:
            raise TypeError("claim field target must use ClaimFieldKind")
        if self.constraint_kind is not None and type(
            self.constraint_kind
        ) is not EvaluationConstraintKind:
            raise TypeError(
                "claim field target constraint must use EvaluationConstraintKind"
            )
        if (self.field is ClaimFieldKind.EVALUATION_CONSTRAINT) != (
            self.constraint_kind is not None
        ):
            raise AnalysisContractError(
                "only an evaluation-constraint target may carry a constraint kind"
            )


@dataclass(frozen=True, slots=True)
class ArtifactEvidenceSlot:
    kind: ArtifactKind
    role: ExperimentRole
    dataset_split: DatasetSplit | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not ArtifactKind:
            raise TypeError("artifact evidence slot kind must use ArtifactKind")
        if type(self.role) is not ExperimentRole:
            raise TypeError("artifact evidence slot role must use ExperimentRole")
        if self.kind not in {
            ArtifactKind.RESULTS,
            ArtifactKind.CONFIG,
            ArtifactKind.DATASET,
        }:
            raise AnalysisContractError(
                "artifact evidence slot must be a native passive input kind"
            )
        if self.role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}:
            raise AnalysisContractError(
                "artifact evidence slot role must be baseline or candidate"
            )
        if self.dataset_split is not None and type(
            self.dataset_split
        ) is not DatasetSplit:
            raise TypeError("artifact evidence slot split must use DatasetSplit")
        if (self.kind is ArtifactKind.DATASET) != (
            self.dataset_split is not None
        ):
            raise AnalysisContractError(
                "dataset split is required only for a dataset evidence slot"
            )


ObligationTarget: TypeAlias = ClaimFieldTarget | ArtifactEvidenceSlot


def _digest(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}-" + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True, slots=True, init=False)
class ClaimFieldSupport:
    support_id: str
    claim_id: str
    field: ClaimFieldKind
    constraint_kind: EvaluationConstraintKind | None

    def __init__(self) -> None:
        raise TypeError("ClaimFieldSupport must be created through its factory")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ClaimFieldSupport is final")


def _claim_has_field(
    claim: CanonicalScientificClaim,
    field: ClaimFieldKind,
    constraint_kind: EvaluationConstraintKind | None,
) -> bool:
    primary = claim.primary
    if field is ClaimFieldKind.METRIC:
        return type(primary) in {MetricImprovementClaim, AbsoluteMetricClaim}
    if field is ClaimFieldKind.DIRECTION:
        return type(primary) is MetricImprovementClaim
    if field is ClaimFieldKind.THRESHOLD:
        return (
            type(primary) is MetricImprovementClaim
            and primary.minimum_improvement is not None
        )
    if field is ClaimFieldKind.QUANTITATIVE_BOUND:
        return type(primary) in {AbsoluteMetricClaim, GenericQuantitativeClaim}
    return constraint_kind is not None and any(
        item.kind is constraint_kind for item in claim.constraints
    )


def claim_field_support(
    claim: CanonicalScientificClaim,
    field: ClaimFieldKind,
    *,
    constraint_kind: EvaluationConstraintKind | None = None,
) -> ClaimFieldSupport:
    """Create support only from independently re-recovered claim semantics."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("claim field support requires CanonicalScientificClaim")
    if type(field) is not ClaimFieldKind:
        raise TypeError("claim field support requires ClaimFieldKind")
    target = ClaimFieldTarget(field, constraint_kind)
    recovered = recover_scientific_claim(claim.reference)
    if recovered != claim:
        raise AnalysisContractError(
            "canonical claim does not match independent source recovery"
        )
    if not _claim_has_field(claim, field, constraint_kind):
        raise AnalysisContractError("required claim field is not present in source")
    material = {
        "claim_id": claim.reference.claim_id,
        "field": field.value,
        "constraint_kind": (
            target.constraint_kind.value if target.constraint_kind is not None else None
        ),
        "source_path": (
            str(claim.reference.source_path)
            if claim.reference.source_path is not None
            else None
        ),
        "source_id": claim.reference.provenance.source_id,
        "source_kind": claim.reference.provenance.kind.value,
        "source_text_sha256": hashlib.sha256(
            claim.reference.text.encode("utf-8")
        ).hexdigest(),
        "semantics": claim_semantic_projection(claim),
    }
    instance = object.__new__(ClaimFieldSupport)
    object.__setattr__(instance, "support_id", _digest("support-claim", material))
    object.__setattr__(instance, "claim_id", claim.reference.claim_id)
    object.__setattr__(instance, "field", field)
    object.__setattr__(instance, "constraint_kind", constraint_kind)
    return instance


@dataclass(frozen=True, slots=True, init=False)
class ArtifactEvidenceSupport:
    support_id: str
    evidence_id: str
    binding_id: str
    kind: ArtifactKind
    role: ExperimentRole
    dataset_split: DatasetSplit | None

    def __init__(self) -> None:
        raise TypeError("ArtifactEvidenceSupport must be created through its factory")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ArtifactEvidenceSupport is final")


def _mapping_identity(mapping: MappingCandidate | RepoMapping) -> object:
    if type(mapping) is RepoMapping:
        return {
            "source_mapping_id": mapping.source_mapping_id,
            "source_trust": mapping.source_trust.value,
            "trust": mapping.trust.value,
            "approval_source_id": mapping.approval_provenance.source_id,
        }
    return {
        "mapping_id": mapping.mapping_id,
        "trust": mapping.trust.value,
        "provenance_kind": mapping.provenance.kind.value,
        "source_id": mapping.provenance.source_id,
    }


def _binding_material(binding: ArtifactBinding) -> object:
    return {
        "path": str(binding.path),
        "kind": binding.kind.value,
        "role": binding.role.value,
        "adapter_id": binding.adapter_id,
        "dataset_split": (
            binding.dataset_split.value if binding.dataset_split is not None else None
        ),
        "mappings": [
            {
                "target": item.target_field,
                "kind": item.selector.kind.value,
                "expression": item.selector.expression,
            }
            for item in binding.mappings
        ],
    }


def _mapping_is_trusted_for_support(
    mapping: MappingCandidate | RepoMapping,
) -> bool:
    if type(mapping) is RepoMapping:
        return True
    if type(mapping) is not MappingCandidate or mapping.confidence.value < 0.90:
        return False
    provenance = (
        mapping.provenance,
        *(item.provenance for item in mapping.bindings),
        *(
            field.provenance
            for binding in mapping.bindings
            for field in binding.mappings
        ),
        *(
            field.selector.provenance
            for binding in mapping.bindings
            for field in binding.mappings
        ),
    )
    return not any(item.kind is ProvenanceKind.PROVIDER_PROPOSAL for item in provenance)


def validated_artifact_support(
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
    mapping: MappingCandidate | RepoMapping,
    *,
    metric: str,
) -> ArtifactEvidenceSupport:
    """Commit one support identity after exact mapping/evidence revalidation."""

    if type(evidence) is not NormalizedEvidence:
        raise TypeError("artifact support requires NormalizedEvidence")
    if type(binding) is not ArtifactBinding:
        raise TypeError("artifact support requires ArtifactBinding")
    if type(mapping) not in {MappingCandidate, RepoMapping}:
        raise TypeError("artifact support requires a selected mapping")
    if not isinstance(metric, str) or not metric or metric != metric.strip():
        raise AnalysisContractError("artifact support metric must be canonical text")
    if binding not in mapping.bindings:
        raise AnalysisContractError("artifact binding is not in the selected mapping")
    if not _mapping_is_trusted_for_support(mapping):
        raise AnalysisContractError(
            "provider mapping requires explicit approval before artifact support"
        )
    if (
        evidence.artifact.path != binding.path
        or evidence.artifact.kind is not binding.kind
        or evidence.adapter_match.path != binding.path
        or evidence.adapter_match.adapter_id != binding.adapter_id
        or evidence.adapter_match.mappings != binding.mappings
    ):
        raise AnalysisContractError(
            "artifact evidence does not match the exact validated binding or selector"
        )
    roles = {
        item.experiment_role
        for item in evidence.observations
        if item.experiment_role is not ExperimentRole.UNSPECIFIED
    }
    if roles and roles != {binding.role}:
        raise AnalysisContractError("artifact evidence role conflicts with its binding")
    if binding.kind is ArtifactKind.RESULTS and not any(
        item.metric_name == metric and item.metric_value is not None
        for item in evidence.observations
    ):
        raise AnalysisContractError("required result metric is not recoverable")
    if binding.kind is ArtifactKind.CONFIG and not any(
        item.config_values for item in evidence.observations
    ):
        raise AnalysisContractError("required config evidence is not recoverable")
    if binding.kind is ArtifactKind.DATASET:
        if binding.dataset_split is None:
            raise AnalysisContractError("required dataset split is not recoverable")
        if not any(
            reference.path == binding.path
            for observation in evidence.observations
            for reference in observation.dataset_references
        ):
            raise AnalysisContractError("required dataset identity is not recoverable")
    binding_material = {
        "artifact_sha256": str(evidence.artifact.sha256),
        "artifact_size": evidence.artifact.size,
        "evidence_id": evidence.evidence_id,
        "binding": _binding_material(binding),
        "mapping": _mapping_identity(mapping),
    }
    binding_id = _digest("binding", binding_material)
    instance = object.__new__(ArtifactEvidenceSupport)
    object.__setattr__(
        instance,
        "support_id",
        _digest(
            "support-artifact",
            {"evidence_id": evidence.evidence_id, "binding_id": binding_id},
        ),
    )
    object.__setattr__(instance, "evidence_id", evidence.evidence_id)
    object.__setattr__(instance, "binding_id", binding_id)
    object.__setattr__(instance, "kind", binding.kind)
    object.__setattr__(instance, "role", binding.role)
    object.__setattr__(instance, "dataset_split", binding.dataset_split)
    return instance


ObligationSupportReference: TypeAlias = ClaimFieldSupport | ArtifactEvidenceSupport


_MISSING_REASONS = frozenset(
    {
        EvidenceObligationReason.REQUIRED_ARTIFACT_NOT_FOUND,
        EvidenceObligationReason.REQUIRED_CLAIM_FIELD_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_METRIC_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_THRESHOLD_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_ROLE_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_SPLIT_NOT_RECOVERED,
        EvidenceObligationReason.DEPENDENCY_MISSING,
    }
)
_AMBIGUOUS_REASONS = frozenset(
    {
        EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES,
        EvidenceObligationReason.CONFLICTING_VALIDATED_MAPPINGS,
        EvidenceObligationReason.EXPLICIT_MAPPING_APPROVAL_REQUIRED,
        EvidenceObligationReason.APPROVED_MAPPING_NOT_APPLICABLE,
        EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED,
    }
)
_UNSUPPORTED_REASONS = frozenset(
    {
        EvidenceObligationReason.CLAIM_TYPE_NOT_EXECUTABLE,
        EvidenceObligationReason.ARTIFACT_FORMAT_NOT_SUPPORTED,
        EvidenceObligationReason.ARTIFACT_EXCEEDS_PASSIVE_LIMIT,
        EvidenceObligationReason.ADAPTER_NOT_AVAILABLE,
        EvidenceObligationReason.SELECTOR_NOT_RECOVERABLE,
        EvidenceObligationReason.NATIVE_REPRESENTATION_NOT_SUPPORTED,
        EvidenceObligationReason.DEPENDENCY_UNSUPPORTED,
    }
)


def _canonical_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AnalysisContractError(f"{label} must be a canonical identifier")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceObligation:
    obligation_id: str
    target: ObligationTarget | None
    state: EvidenceObligationState
    reason: EvidenceObligationReason
    effect: EvidenceObligationEffect
    support_references: tuple[ObligationSupportReference, ...] = ()
    dependency_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _canonical_id(self.obligation_id, "obligation_id")
        if self.target is not None and type(self.target) not in {
            ClaimFieldTarget,
            ArtifactEvidenceSlot,
        }:
            raise TypeError("obligation target is invalid")
        if type(self.state) is not EvidenceObligationState:
            raise TypeError("obligation state must use EvidenceObligationState")
        if type(self.reason) is not EvidenceObligationReason:
            raise TypeError("obligation reason must use EvidenceObligationReason")
        if type(self.effect) is not EvidenceObligationEffect:
            raise TypeError("obligation effect must use EvidenceObligationEffect")
        if (
            not isinstance(self.support_references, tuple)
            or len(self.support_references) > 8
            or not all(
                type(item) in {ClaimFieldSupport, ArtifactEvidenceSupport}
                for item in self.support_references
            )
        ):
            raise AnalysisContractError(
                "obligation support references exceed their structural bound"
            )
        support_ids = tuple(item.support_id for item in self.support_references)
        if len(set(support_ids)) != len(support_ids):
            raise AnalysisContractError("obligation support references must be unique")
        if type(self.target) is ClaimFieldTarget and any(
            type(item) is not ClaimFieldSupport
            or item.field is not self.target.field
            or item.constraint_kind is not self.target.constraint_kind
            for item in self.support_references
        ):
            raise AnalysisContractError(
                "claim support does not match its exact field target"
            )
        if type(self.target) is ArtifactEvidenceSlot and any(
            type(item) is not ArtifactEvidenceSupport
            or item.kind is not self.target.kind
            or item.role is not self.target.role
            or item.dataset_split is not self.target.dataset_split
            for item in self.support_references
        ):
            raise AnalysisContractError(
                "artifact target requires artifact evidence support"
            )
        if (
            not isinstance(self.dependency_ids, tuple)
            or len(self.dependency_ids) > 16
        ):
            raise AnalysisContractError(
                "obligation dependencies exceed their structural bound"
            )
        for dependency in self.dependency_ids:
            _canonical_id(dependency, "obligation dependency_id")
        if len(set(self.dependency_ids)) != len(self.dependency_ids):
            raise AnalysisContractError("obligation dependencies must be unique")
        if self.obligation_id in self.dependency_ids:
            raise AnalysisContractError("obligation dependency cycle is invalid")
        if (self.target is None) == (not self.dependency_ids):
            raise AnalysisContractError(
                "an obligation must be either one target leaf or one composite"
            )
        composite = self.target is None
        if self.state is EvidenceObligationState.SATISFIED:
            expected_reason = (
                EvidenceObligationReason.VALIDATED_DEPENDENCIES_SATISFIED
                if composite
                else EvidenceObligationReason.VALIDATED_SUPPORT_BOUND
            )
            if self.reason is not expected_reason or self.effect is not EvidenceObligationEffect.NONE:
                raise AnalysisContractError("satisfied obligation state is inconsistent")
            if composite == bool(self.support_references):
                raise AnalysisContractError(
                    "a satisfied leaf requires support and a composite cannot carry support"
                )
        elif self.state is EvidenceObligationState.MISSING:
            if (
                self.reason not in _MISSING_REASONS
                or self.effect is not EvidenceObligationEffect.BLOCKS_PARTIAL
                or self.support_references
            ):
                raise AnalysisContractError("missing obligation state is inconsistent")
        elif self.state is EvidenceObligationState.UNSUPPORTED:
            if (
                self.reason not in _UNSUPPORTED_REASONS
                or self.effect is not EvidenceObligationEffect.BLOCKS_PARTIAL
                or self.support_references
            ):
                raise AnalysisContractError("unsupported obligation state is inconsistent")
        elif (
            self.reason not in _AMBIGUOUS_REASONS
            or self.effect
            not in {
                EvidenceObligationEffect.BLOCKS_PARTIAL,
                EvidenceObligationEffect.REQUIRES_MAPPING,
            }
            or self.support_references
        ):
            raise AnalysisContractError("ambiguous obligation state is inconsistent")


@dataclass(frozen=True, slots=True, init=False)
class EvidenceObligationBundle:
    claim_id: str
    policy_id: str
    compiler_state: EvidenceObligationState
    compiler_reason: EvidenceObligationReason
    compiler_effect: EvidenceObligationEffect
    obligations: tuple[EvidenceObligation, ...]
    blocking_obligation_ids: tuple[str, ...]
    decision: EvidenceObligationDecision
    mapping_question_id: str | None
    version: int = field(default=1, init=False)

    def __init__(self) -> None:
        raise TypeError("EvidenceObligationBundle must be created by the engine")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EvidenceObligationBundle is final")


def _validate_graph(obligations: tuple[EvidenceObligation, ...]) -> None:
    identifiers = tuple(item.obligation_id for item in obligations)
    if len(set(identifiers)) != len(identifiers):
        raise AnalysisContractError("duplicate obligation IDs are invalid")
    known = set(identifiers)
    if any(
        dependency not in known
        for item in obligations
        for dependency in item.dependency_ids
    ):
        raise AnalysisContractError("dangling obligation dependency is invalid")
    dependencies = {item.obligation_id: item.dependency_ids for item in obligations}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise AnalysisContractError("cycle in obligation dependencies is invalid")
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in dependencies[identifier]:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in identifiers:
        visit(identifier)


def _derived_composite(
    item: EvidenceObligation,
    by_id: dict[str, EvidenceObligation],
    cache: dict[str, EvidenceObligation],
) -> EvidenceObligation:
    cached = cache.get(item.obligation_id)
    if cached is not None:
        return cached
    if item.target is not None:
        cache[item.obligation_id] = item
        return item
    children = tuple(
        _derived_composite(by_id[dependency], by_id, cache)
        for dependency in item.dependency_ids
    )
    if any(child.state is EvidenceObligationState.UNSUPPORTED for child in children):
        derived = replace(
            item,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.DEPENDENCY_UNSUPPORTED,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    elif any(child.state is EvidenceObligationState.MISSING for child in children):
        derived = replace(
            item,
            state=EvidenceObligationState.MISSING,
            reason=EvidenceObligationReason.DEPENDENCY_MISSING,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    elif any(child.state is EvidenceObligationState.AMBIGUOUS for child in children):
        mapping_only = all(
            child.state is EvidenceObligationState.SATISFIED
            or child.effect is EvidenceObligationEffect.REQUIRES_MAPPING
            for child in children
        )
        ambiguous = next(
            child for child in children if child.state is EvidenceObligationState.AMBIGUOUS
        )
        derived = replace(
            item,
            state=EvidenceObligationState.AMBIGUOUS,
            reason=(
                ambiguous.reason
                if mapping_only
                else EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED
            ),
            effect=(
                EvidenceObligationEffect.REQUIRES_MAPPING
                if mapping_only
                else EvidenceObligationEffect.BLOCKS_PARTIAL
            ),
        )
    else:
        derived = replace(
            item,
            state=EvidenceObligationState.SATISFIED,
            reason=EvidenceObligationReason.VALIDATED_DEPENDENCIES_SATISFIED,
            effect=EvidenceObligationEffect.NONE,
        )
    cache[item.obligation_id] = derived
    return derived


def _build_obligation_bundle(
    *,
    claim_id: str,
    policy: ClaimEvidencePolicy,
    compiler_supported: bool,
    obligations: tuple[EvidenceObligation, ...],
    mapping_question_id: str | None,
) -> EvidenceObligationBundle:
    """Internal factory used only after deterministic obligation instantiation."""

    _canonical_id(claim_id, "obligation bundle claim_id")
    if type(policy) is not ClaimEvidencePolicy:
        raise TypeError("obligation bundle policy must be ClaimEvidencePolicy")
    if type(compiler_supported) is not bool:
        raise TypeError("compiler support must be boolean")
    if (
        not isinstance(obligations, tuple)
        or len(obligations) > 16
        or not all(type(item) is EvidenceObligation for item in obligations)
    ):
        raise AnalysisContractError("obligation bundle exceeds its structural bound")
    if sum(len(item.support_references) for item in obligations) > 32:
        raise AnalysisContractError("obligation support bundle exceeds its bound")
    if mapping_question_id is not None:
        _canonical_id(mapping_question_id, "obligation mapping_question_id")
    _validate_graph(obligations)
    by_id = {item.obligation_id: item for item in obligations}
    cache: dict[str, EvidenceObligation] = {}
    derived = tuple(
        sorted(
            (_derived_composite(item, by_id, cache) for item in obligations),
            key=lambda item: item.obligation_id,
        )
    )
    if any(
        support.claim_id != claim_id
        for item in derived
        for support in item.support_references
        if type(support) is ClaimFieldSupport
    ):
        raise AnalysisContractError(
            "claim support identity does not match the obligation bundle claim"
        )
    leaves = tuple(item for item in derived if item.target is not None)
    blockers = tuple(
        item.obligation_id
        for item in leaves
        if item.state is not EvidenceObligationState.SATISFIED
    )
    compiler_state = (
        EvidenceObligationState.SATISFIED
        if compiler_supported
        else EvidenceObligationState.UNSUPPORTED
    )
    compiler_reason = (
        EvidenceObligationReason.VALIDATED_DEPENDENCIES_SATISFIED
        if compiler_supported
        else EvidenceObligationReason.CLAIM_TYPE_NOT_EXECUTABLE
    )
    compiler_effect = (
        EvidenceObligationEffect.NONE
        if compiler_supported
        else EvidenceObligationEffect.BLOCKS_PARTIAL
    )
    if (
        not compiler_supported
        or any(item.effect is EvidenceObligationEffect.BLOCKS_PARTIAL for item in leaves)
    ):
        decision = EvidenceObligationDecision.PARTIAL
        linked_question = None
    elif any(
        item.effect is EvidenceObligationEffect.REQUIRES_MAPPING for item in leaves
    ):
        decision = (
            EvidenceObligationDecision.MAPPING_NEEDED
            if mapping_question_id is not None
            else EvidenceObligationDecision.PARTIAL
        )
        linked_question = (
            mapping_question_id
            if decision is EvidenceObligationDecision.MAPPING_NEEDED
            else None
        )
    else:
        decision = EvidenceObligationDecision.READY
        linked_question = None
    instance = object.__new__(EvidenceObligationBundle)
    object.__setattr__(instance, "claim_id", claim_id)
    object.__setattr__(instance, "policy_id", policy.policy_id)
    object.__setattr__(instance, "compiler_state", compiler_state)
    object.__setattr__(instance, "compiler_reason", compiler_reason)
    object.__setattr__(instance, "compiler_effect", compiler_effect)
    object.__setattr__(instance, "obligations", derived)
    object.__setattr__(instance, "blocking_obligation_ids", blockers)
    object.__setattr__(instance, "decision", decision)
    object.__setattr__(instance, "mapping_question_id", linked_question)
    object.__setattr__(instance, "version", 1)
    return instance


def evidence_obligations_json_bytes(bundle: EvidenceObligationBundle) -> bytes:
    """Serialize one structurally bounded bundle without a transport allocation."""

    if type(bundle) is not EvidenceObligationBundle:
        raise TypeError("obligation serialization requires EvidenceObligationBundle")
    from .contracts import to_jsonable

    return json.dumps(
        to_jsonable(bundle),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _claim_target(template_id: str) -> ClaimFieldTarget:
    if template_id == "claim.metric":
        return ClaimFieldTarget(ClaimFieldKind.METRIC)
    if template_id == "claim.direction":
        return ClaimFieldTarget(ClaimFieldKind.DIRECTION)
    if template_id == "claim.threshold":
        return ClaimFieldTarget(ClaimFieldKind.THRESHOLD)
    if template_id == "claim.quantitative_bound":
        return ClaimFieldTarget(ClaimFieldKind.QUANTITATIVE_BOUND)
    if template_id == "claim.constraint.held_out":
        return ClaimFieldTarget(
            ClaimFieldKind.EVALUATION_CONSTRAINT,
            EvaluationConstraintKind.HELD_OUT,
        )
    raise AnalysisContractError("claim policy contains an unknown field template")


def _artifact_target(template_id: str) -> ArtifactEvidenceSlot:
    parts = template_id.split(".")
    if len(parts) not in {3, 4} or parts[0] != "artifact":
        raise AnalysisContractError("claim policy contains an unknown artifact template")
    role = ExperimentRole(parts[1])
    if parts[2] == "results" and parts[3:] == ["metric"]:
        return ArtifactEvidenceSlot(ArtifactKind.RESULTS, role)
    if parts[2] == "config" and len(parts) == 3:
        return ArtifactEvidenceSlot(ArtifactKind.CONFIG, role)
    if parts[2] == "dataset" and len(parts) == 4:
        return ArtifactEvidenceSlot(
            ArtifactKind.DATASET,
            role,
            DatasetSplit(parts[3]),
        )
    raise AnalysisContractError("claim policy contains an unknown artifact template")


def _missing_claim_reason(field: ClaimFieldKind) -> EvidenceObligationReason:
    if field is ClaimFieldKind.METRIC:
        return EvidenceObligationReason.REQUIRED_METRIC_NOT_RECOVERED
    if field is ClaimFieldKind.THRESHOLD:
        return EvidenceObligationReason.REQUIRED_THRESHOLD_NOT_RECOVERED
    return EvidenceObligationReason.REQUIRED_CLAIM_FIELD_NOT_RECOVERED


def _claim_obligation(
    obligation_id: str,
    claim: CanonicalScientificClaim,
) -> EvidenceObligation:
    target = _claim_target(obligation_id)
    try:
        support = claim_field_support(
            claim,
            target.field,
            constraint_kind=target.constraint_kind,
        )
    except AnalysisContractError:
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=target,
            state=EvidenceObligationState.MISSING,
            reason=_missing_claim_reason(target.field),
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    return EvidenceObligation(
        obligation_id=obligation_id,
        target=target,
        state=EvidenceObligationState.SATISFIED,
        reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
        effect=EvidenceObligationEffect.NONE,
        support_references=(support,),
    )


def _evidence_matches_binding(
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
) -> bool:
    if (
        evidence.artifact.path != binding.path
        or evidence.artifact.kind is not binding.kind
        or evidence.adapter_match.adapter_id != binding.adapter_id
        or evidence.adapter_match.mappings != binding.mappings
    ):
        return False
    roles = {
        item.experiment_role
        for item in evidence.observations
        if item.experiment_role is not ExperimentRole.UNSPECIFIED
    }
    return not roles or roles == {binding.role}


def _matching_bindings(
    slot: ArtifactEvidenceSlot,
    mapping: MappingCandidate | RepoMapping,
) -> tuple[ArtifactBinding, ...]:
    return tuple(
        item
        for item in mapping.bindings
        if item.kind is slot.kind
        and item.role is slot.role
        and item.dataset_split is slot.dataset_split
    )


def _binding_support_signature(binding: ArtifactBinding) -> tuple[object, ...]:
    return (
        str(binding.path),
        binding.kind.value,
        binding.role.value,
        binding.adapter_id,
        binding.dataset_split.value if binding.dataset_split is not None else "",
        tuple(
            sorted(
                (
                    item.target_field,
                    item.selector.kind.value,
                    item.selector.expression,
                )
                for item in binding.mappings
            )
        ),
    )


def _question_common_slot_signatures(
    slot: ArtifactEvidenceSlot,
    question: MappingQuestion,
) -> tuple[tuple[object, ...], ...] | None:
    signatures = tuple(
        tuple(
            sorted(
                _binding_support_signature(binding)
                for binding in choice.bindings
                if binding.kind is slot.kind
                and binding.role is slot.role
                and binding.dataset_split is slot.dataset_split
            )
        )
        for choice in question.choices
    )
    if not signatures or not signatures[0] or any(
        item != signatures[0] for item in signatures[1:]
    ):
        return None
    return signatures[0]


def _selected_artifact_obligation(
    obligation_id: str,
    slot: ArtifactEvidenceSlot,
    *,
    selected_mapping: MappingCandidate | RepoMapping,
    evidence_by_key: dict[tuple[object, ArtifactKind], NormalizedEvidence],
    metric: str,
) -> EvidenceObligation:
    bindings = _matching_bindings(slot, selected_mapping)
    supports: list[ArtifactEvidenceSupport] = []
    try:
        for binding in bindings:
            evidence = evidence_by_key.get((binding.path, binding.kind))
            if evidence is None:
                continue
            supports.append(
                validated_artifact_support(
                    evidence,
                    binding,
                    selected_mapping,
                    metric=metric,
                )
            )
    except AnalysisContractError:
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=slot,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.SELECTOR_NOT_RECOVERABLE,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    if len(supports) > 8:
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=slot,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.NATIVE_REPRESENTATION_NOT_SUPPORTED,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    if supports:
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=slot,
            state=EvidenceObligationState.SATISFIED,
            reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
            effect=EvidenceObligationEffect.NONE,
            support_references=tuple(supports),
        )
    reason = (
        EvidenceObligationReason.REQUIRED_SPLIT_NOT_RECOVERED
        if slot.kind is ArtifactKind.DATASET
        else EvidenceObligationReason.REQUIRED_ROLE_NOT_RECOVERED
    )
    return EvidenceObligation(
        obligation_id=obligation_id,
        target=slot,
        state=EvidenceObligationState.MISSING,
        reason=reason,
        effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
    )


def _unselected_artifact_obligation(
    obligation_id: str,
    slot: ArtifactEvidenceSlot,
    *,
    artifacts: tuple[ArtifactCandidate, ...],
    normalized_evidence: tuple[NormalizedEvidence, ...],
    mapping_candidates: tuple[MappingCandidate, ...],
    mapping_question: MappingQuestion | None,
    ambiguity_reason: EvidenceObligationReason | None,
    metric: str,
) -> EvidenceObligation:
    relevant_artifacts = tuple(item for item in artifacts if item.kind is slot.kind)
    if not relevant_artifacts:
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=slot,
            state=EvidenceObligationState.MISSING,
            reason=EvidenceObligationReason.REQUIRED_ARTIFACT_NOT_FOUND,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    evidence_by_key = {
        (item.artifact.path, item.artifact.kind): item for item in normalized_evidence
    }
    relevant_evidence = tuple(
        item for item in normalized_evidence if item.artifact.kind is slot.kind
    )
    if not relevant_evidence:
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=slot,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.ADAPTER_NOT_AVAILABLE,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    binding_candidates = tuple(
        (candidate, binding)
        for candidate in mapping_candidates
        for binding in _matching_bindings(slot, candidate)
    )
    question_bindings = (
        ()
        if mapping_question is None
        else tuple(
            binding
            for choice in mapping_question.choices
            for binding in choice.bindings
            if binding.kind is slot.kind
            and binding.role is slot.role
            and binding.dataset_split is slot.dataset_split
        )
    )
    if not binding_candidates and not question_bindings:
        same_kind = tuple(
            binding
            for candidate in mapping_candidates
            for binding in candidate.bindings
            if binding.kind is slot.kind
        )
        reason = (
            EvidenceObligationReason.REQUIRED_SPLIT_NOT_RECOVERED
            if slot.kind is ArtifactKind.DATASET
            and any(item.role is slot.role for item in same_kind)
            else EvidenceObligationReason.REQUIRED_ROLE_NOT_RECOVERED
        )
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=slot,
            state=EvidenceObligationState.MISSING,
            reason=reason,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    valid_bindings = tuple(
        (candidate, binding, evidence)
        for candidate, binding in binding_candidates
        if (evidence := evidence_by_key.get((binding.path, binding.kind))) is not None
        and _evidence_matches_binding(evidence, binding)
        and (
            slot.kind is not ArtifactKind.RESULTS
            or any(item.metric_name == metric for item in evidence.observations)
        )
    )
    valid_question_bindings = tuple(
        binding
        for binding in question_bindings
        if (evidence := evidence_by_key.get((binding.path, binding.kind))) is not None
        and _evidence_matches_binding(evidence, binding)
        and (
            slot.kind is not ArtifactKind.RESULTS
            or any(item.metric_name == metric for item in evidence.observations)
        )
    )
    if not valid_bindings and not valid_question_bindings:
        return EvidenceObligation(
            obligation_id=obligation_id,
            target=slot,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.SELECTOR_NOT_RECOVERABLE,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    if (
        mapping_question is not None
        and ambiguity_reason
        in {
            EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES,
            EvidenceObligationReason.CONFLICTING_VALIDATED_MAPPINGS,
        }
    ):
        common = _question_common_slot_signatures(slot, mapping_question)
        if common is not None:
            supports: list[ArtifactEvidenceSupport] = []
            for signature in common:
                supported = None
                for candidate, binding, evidence in sorted(
                    valid_bindings,
                    key=lambda item: (
                        item[0].mapping_id,
                        _binding_support_signature(item[1]),
                    ),
                ):
                    if _binding_support_signature(binding) != signature:
                        continue
                    try:
                        supported = validated_artifact_support(
                            evidence,
                            binding,
                            candidate,
                            metric=metric,
                        )
                    except AnalysisContractError:
                        continue
                    break
                if supported is None:
                    supports = []
                    break
                supports.append(supported)
            if supports:
                return EvidenceObligation(
                    obligation_id=obligation_id,
                    target=slot,
                    state=EvidenceObligationState.SATISFIED,
                    reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
                    effect=EvidenceObligationEffect.NONE,
                    support_references=tuple(supports),
                )
    return EvidenceObligation(
        obligation_id=obligation_id,
        target=slot,
        state=EvidenceObligationState.AMBIGUOUS,
        reason=EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED,
        effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
    )


def assess_evidence_obligations(
    *,
    claim: CanonicalScientificClaim,
    policy: ClaimEvidencePolicy,
    audit_claim: AuditClaimSpec | None,
    artifacts: tuple[ArtifactCandidate, ...],
    normalized_evidence: tuple[NormalizedEvidence, ...],
    mapping_candidates: tuple[MappingCandidate, ...],
    selected_mapping: MappingCandidate | RepoMapping | None,
    mapping_question: MappingQuestion | None,
    ambiguity_reason: EvidenceObligationReason | None,
) -> EvidenceObligationBundle:
    """Instantiate the canonical policy from validated Core planning inputs."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("obligation assessment requires CanonicalScientificClaim")
    if type(policy) is not ClaimEvidencePolicy:
        raise TypeError("obligation assessment requires ClaimEvidencePolicy")
    from .claim_types import claim_evidence_policy

    if policy != claim_evidence_policy(claim):
        raise AnalysisContractError("obligation policy does not match claim semantics")
    if audit_claim is not None and type(audit_claim) is not AuditClaimSpec:
        raise TypeError("obligation audit claim must be AuditClaimSpec or null")
    try:
        expected_audit_claim = compile_audit_claim(claim)
    except UnsupportedDeterministicClaimCompiler:
        if audit_claim is not None:
            raise AnalysisContractError(
                "unsupported claim compiler cannot be upgraded with an audit claim"
            )
    else:
        if audit_claim != expected_audit_claim:
            raise AnalysisContractError(
                "obligation audit claim does not match canonical claim semantics"
            )
    if not isinstance(artifacts, tuple) or not all(
        type(item) is ArtifactCandidate for item in artifacts
    ):
        raise TypeError("obligation artifacts must be ArtifactCandidate values")
    if not isinstance(normalized_evidence, tuple) or not all(
        type(item) is NormalizedEvidence for item in normalized_evidence
    ):
        raise TypeError("obligation evidence must be NormalizedEvidence values")
    if not isinstance(mapping_candidates, tuple) or not all(
        type(item) is MappingCandidate for item in mapping_candidates
    ):
        raise TypeError("obligation mappings must be MappingCandidate values")
    if selected_mapping is not None and type(selected_mapping) not in {
        MappingCandidate,
        RepoMapping,
    }:
        raise TypeError("obligation selected mapping is invalid")
    if mapping_question is not None and type(mapping_question) is not MappingQuestion:
        raise TypeError("obligation mapping question is invalid")
    if ambiguity_reason is not None and ambiguity_reason not in _AMBIGUOUS_REASONS:
        raise AnalysisContractError("obligation ambiguity reason is invalid")
    if (mapping_question is not None or ambiguity_reason is not None) and selected_mapping is not None:
        raise AnalysisContractError("selected mapping cannot remain ambiguous")

    compiler_supported = audit_claim is not None
    obligations: list[EvidenceObligation] = []
    artifact_templates: list[str] = []
    for template_id in policy.obligation_template_ids:
        if template_id.startswith("claim."):
            obligations.append(_claim_obligation(template_id, claim))
        elif template_id.startswith("artifact."):
            artifact_templates.append(template_id)
        elif template_id != "comparison.readiness":
            raise AnalysisContractError("claim policy contains an unknown template")

    if artifact_templates:
        if audit_claim is None:
            raise AnalysisContractError(
                "unsupported compiler policy cannot instantiate artifact obligations"
            )
        evidence_by_key = {
            (item.artifact.path, item.artifact.kind): item
            for item in normalized_evidence
        }
        for template_id in artifact_templates:
            slot = _artifact_target(template_id)
            if selected_mapping is not None:
                obligation = _selected_artifact_obligation(
                    template_id,
                    slot,
                    selected_mapping=selected_mapping,
                    evidence_by_key=evidence_by_key,
                    metric=audit_claim.metric,
                )
            else:
                obligation = _unselected_artifact_obligation(
                    template_id,
                    slot,
                    artifacts=artifacts,
                    normalized_evidence=normalized_evidence,
                    mapping_candidates=mapping_candidates,
                    mapping_question=mapping_question,
                    ambiguity_reason=ambiguity_reason,
                    metric=audit_claim.metric,
                )
                if (
                    ambiguity_reason is not None
                    and obligation.state is EvidenceObligationState.AMBIGUOUS
                ):
                    obligation = replace(
                        obligation,
                        reason=(
                            ambiguity_reason
                            if mapping_question is not None
                            else EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED
                        ),
                        effect=(
                            EvidenceObligationEffect.REQUIRES_MAPPING
                            if mapping_question is not None
                            else EvidenceObligationEffect.BLOCKS_PARTIAL
                        ),
                    )
            obligations.append(obligation)

    if "comparison.readiness" in policy.obligation_template_ids:
        dependencies = tuple(
            item
            for item in policy.obligation_template_ids
            if item != "comparison.readiness"
        )
        obligations.append(
            EvidenceObligation(
                obligation_id="comparison.readiness",
                target=None,
                state=EvidenceObligationState.SATISFIED,
                reason=EvidenceObligationReason.VALIDATED_DEPENDENCIES_SATISFIED,
                effect=EvidenceObligationEffect.NONE,
                dependency_ids=dependencies,
            )
        )
    return _build_obligation_bundle(
        claim_id=claim.reference.claim_id,
        policy=policy,
        compiler_supported=compiler_supported,
        obligations=tuple(obligations),
        mapping_question_id=(
            mapping_question.question_id if mapping_question is not None else None
        ),
    )


def legacy_missing_evidence(
    bundle: EvidenceObligationBundle,
) -> tuple[MissingEvidence, ...]:
    """Project only truthful artifact blockers into the legacy compatibility type."""

    if type(bundle) is not EvidenceObligationBundle:
        raise TypeError("legacy projection requires EvidenceObligationBundle")
    values: list[MissingEvidence] = []
    for item in bundle.obligations:
        if (
            type(item.target) is not ArtifactEvidenceSlot
            or item.state
            not in {EvidenceObligationState.MISSING, EvidenceObligationState.UNSUPPORTED}
        ):
            continue
        slot = item.target
        split = (
            f" {slot.dataset_split.value}"
            if slot.dataset_split is not None
            else ""
        )
        limitation = (
            "is missing"
            if item.state is EvidenceObligationState.MISSING
            else "is not supported by the current passive representation"
        )
        values.append(
            MissingEvidence(
                kind=slot.kind,
                role=slot.role,
                description=f"{slot.role.value}{split} {slot.kind.value} evidence {limitation}",
                claim_id=bundle.claim_id,
            )
        )
    return tuple(values)


def _with_native_representation_failure(
    bundle: EvidenceObligationBundle,
    policy: ClaimEvidencePolicy,
    *,
    kind: ArtifactKind,
    role: ExperimentRole,
    dataset_split: DatasetSplit | None,
) -> EvidenceObligationBundle:
    """Derive a pre-Audit unsupported leaf from a typed materialization limit."""

    if type(bundle) is not EvidenceObligationBundle:
        raise TypeError("native representation failure requires an obligation bundle")
    if type(policy) is not ClaimEvidencePolicy or bundle.policy_id != policy.policy_id:
        raise AnalysisContractError(
            "native representation failure policy does not match obligations"
        )
    if bundle.decision is not EvidenceObligationDecision.READY:
        raise AnalysisContractError(
            "native representation failure requires previously ready obligations"
        )
    if type(kind) is not ArtifactKind or type(role) is not ExperimentRole:
        raise TypeError("native representation failure requires a typed artifact slot")
    if dataset_split is not None and type(dataset_split) is not DatasetSplit:
        raise TypeError("native representation failure split must use DatasetSplit")
    changed = False
    obligations: list[EvidenceObligation] = []
    for item in bundle.obligations:
        target = item.target
        if (
            type(target) is ArtifactEvidenceSlot
            and target.kind is kind
            and target.role is role
            and (dataset_split is None or target.dataset_split is dataset_split)
        ):
            changed = True
            item = replace(
                item,
                state=EvidenceObligationState.UNSUPPORTED,
                reason=EvidenceObligationReason.NATIVE_REPRESENTATION_NOT_SUPPORTED,
                effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
                support_references=(),
            )
        obligations.append(item)
    if not changed:
        raise AnalysisContractError(
            "native representation failure has no matching artifact obligation"
        )
    return _build_obligation_bundle(
        claim_id=bundle.claim_id,
        policy=policy,
        compiler_supported=True,
        obligations=tuple(obligations),
        mapping_question_id=None,
    )


__all__ = [
    "ArtifactEvidenceSlot",
    "ArtifactEvidenceSupport",
    "ClaimFieldKind",
    "ClaimFieldSupport",
    "ClaimFieldTarget",
    "EvidenceObligationDecision",
    "EvidenceObligationEffect",
    "EvidenceObligation",
    "EvidenceObligationBundle",
    "EvidenceObligationReason",
    "EvidenceObligationState",
    "ObligationSupportReference",
    "ObligationTarget",
    "assess_evidence_obligations",
    "claim_field_support",
    "evidence_obligations_json_bytes",
    "legacy_missing_evidence",
    "validated_artifact_support",
]
