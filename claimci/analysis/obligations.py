"""Bounded deterministic evidence-obligation contracts and support identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeAlias

from claimci.measurement import (
    MeasurementComponentKind,
    UpstreamAggregationProcedure,
)

from .claim_types import (
    AbsoluteMetricClaim,
    CanonicalScientificClaim,
    ClaimEvidencePolicy,
    EvaluationConstraintKind,
    GenericQuantitativeClaim,
    MetricImprovementClaim,
    UnsupportedDeterministicClaimCompiler,
    claim_semantic_projection,
    claim_evidence_policy,
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
from .evidence_identity import (
    evidence_for_binding,
    evidence_matches_binding as _runtime_evidence_matches_binding,
    field_mapping_material,
    field_mapping_projection,
)
from .measurement import recover_upstream_procedure_requirement
from .profiles import (
    BenchmarkResultForm,
    EvidenceProfileId,
    EvidenceProfileSelection,
    ProfileRoleMode,
    ProfileSupportKind,
    ProfiledEvidencePolicy,
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
    REQUIRED_MEASUREMENT_PROCEDURE_NOT_RECOVERED = (
        "required_measurement_procedure_not_recovered"
    )
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
    MEASUREMENT_PROCEDURE_NOT_SUPPORTED = "measurement_procedure_not_supported"
    DEPENDENCY_UNSUPPORTED = "dependency_unsupported"
    REQUIRED_PROFILE_EVIDENCE_NOT_RECOVERED = (
        "required_profile_evidence_not_recovered"
    )
    PROFILE_EVIDENCE_NOT_SUPPORTED = "profile_evidence_not_supported"
    PROFILE_EVIDENCE_AMBIGUOUS = "profile_evidence_ambiguous"


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


@dataclass(frozen=True, slots=True)
class MeasurementComponentTarget:
    component_kind: MeasurementComponentKind
    required_procedure: UpstreamAggregationProcedure

    def __post_init__(self) -> None:
        if type(self.component_kind) is not MeasurementComponentKind:
            raise TypeError("measurement target kind must use MeasurementComponentKind")
        if self.component_kind is not MeasurementComponentKind.RETRY_AGGREGATION:
            raise AnalysisContractError(
                "v1 measurement obligation target must be retry/aggregation"
            )
        if type(self.required_procedure) is not UpstreamAggregationProcedure:
            raise TypeError(
                "measurement target procedure must use UpstreamAggregationProcedure"
            )


@dataclass(frozen=True, slots=True, init=False)
class ProfileEvidenceTarget:
    profile_id: EvidenceProfileId
    slot_id: str
    role_mode: ProfileRoleMode
    allowed_artifact_kinds: tuple[ArtifactKind, ...]
    allowed_support_kinds: tuple[ProfileSupportKind, ...]
    activation_policy_id: str

    def __init__(self) -> None:
        raise TypeError("ProfileEvidenceTarget must come from fixed profile policy")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ProfileEvidenceTarget is final")


_PROFILE_TARGET_SPECS: dict[
    str,
    tuple[ProfileRoleMode, tuple[ArtifactKind, ...], tuple[ProfileSupportKind, ...]],
] = {
    "benchmark.subject": (
        ProfileRoleMode.PAIRED_BASELINE_CANDIDATE,
        (ArtifactKind.BENCHMARK, ArtifactKind.CONFIG, ArtifactKind.RESULTS),
        (ProfileSupportKind.STRUCTURED_COMPONENT,),
    ),
    "benchmark.results": (
        ProfileRoleMode.PAIRED_BASELINE_CANDIDATE,
        (ArtifactKind.BENCHMARK, ArtifactKind.RESULTS),
        (
            ProfileSupportKind.NORMALIZED_METRIC,
            ProfileSupportKind.DETERMINISTIC_DERIVATION,
        ),
    ),
    "benchmark.workload": (
        ProfileRoleMode.SHARED_REFERENCE_OR_PAIRED,
        (
            ArtifactKind.BENCHMARK,
            ArtifactKind.CONFIG,
            ArtifactKind.DATASET,
            ArtifactKind.DOCUMENT,
            ArtifactKind.RESULTS,
        ),
        (ProfileSupportKind.ARTIFACT_BINDING, ProfileSupportKind.STRUCTURED_COMPONENT),
    ),
    "benchmark.measurement_config": (
        ProfileRoleMode.PAIRED_BASELINE_CANDIDATE,
        (ArtifactKind.BENCHMARK, ArtifactKind.CONFIG, ArtifactKind.RESULTS),
        (ProfileSupportKind.STRUCTURED_COMPONENT,),
    ),
    "measurement.metric_identity": (
        ProfileRoleMode.PAIRED_BASELINE_CANDIDATE,
        (ArtifactKind.BENCHMARK, ArtifactKind.RESULTS, ArtifactKind.CONFIG),
        (ProfileSupportKind.NORMALIZED_METRIC, ProfileSupportKind.STRUCTURED_COMPONENT),
    ),
    "benchmark.run_protocol": (
        ProfileRoleMode.PAIRED_BASELINE_CANDIDATE,
        (ArtifactKind.BENCHMARK, ArtifactKind.CONFIG, ArtifactKind.RESULTS),
        (ProfileSupportKind.STRUCTURED_COMPONENT,),
    ),
    "benchmark.environment": (
        ProfileRoleMode.SHARED_REFERENCE_OR_PAIRED,
        (
            ArtifactKind.BENCHMARK,
            ArtifactKind.CONFIG,
            ArtifactKind.DOCUMENT,
            ArtifactKind.RESULTS,
        ),
        (ProfileSupportKind.STRUCTURED_COMPONENT,),
    ),
    "benchmark.evaluator": (
        ProfileRoleMode.SHARED_REFERENCE_OR_PAIRED,
        (
            ArtifactKind.BENCHMARK,
            ArtifactKind.CONFIG,
            ArtifactKind.DOCUMENT,
            ArtifactKind.RESULTS,
        ),
        (
            ProfileSupportKind.ARTIFACT_BINDING,
            ProfileSupportKind.STRUCTURED_COMPONENT,
            ProfileSupportKind.DETERMINISTIC_DERIVATION,
        ),
    ),
}


def profile_evidence_target(
    selection: EvidenceProfileSelection,
    slot_id: str,
) -> ProfileEvidenceTarget:
    """Instantiate one fixed target from an engine-created Benchmark selection."""

    if type(selection) is not EvidenceProfileSelection:
        raise TypeError("profile target requires EvidenceProfileSelection")
    if selection.profile_id is not EvidenceProfileId.BENCHMARK_MEASUREMENT_V0:
        raise AnalysisContractError("Training uses its exact legacy artifact targets")
    if slot_id not in _PROFILE_TARGET_SPECS:
        raise AnalysisContractError("profile target slot is not in the fixed registry")
    role_mode, artifact_kinds, support_kinds = _PROFILE_TARGET_SPECS[slot_id]
    instance = object.__new__(ProfileEvidenceTarget)
    object.__setattr__(instance, "profile_id", selection.profile_id)
    object.__setattr__(instance, "slot_id", slot_id)
    object.__setattr__(instance, "role_mode", role_mode)
    object.__setattr__(instance, "allowed_artifact_kinds", artifact_kinds)
    object.__setattr__(instance, "allowed_support_kinds", support_kinds)
    object.__setattr__(instance, "activation_policy_id", selection.activation_policy_id)
    return instance


ObligationTarget: TypeAlias = (
    ClaimFieldTarget
    | ArtifactEvidenceSlot
    | MeasurementComponentTarget
    | ProfileEvidenceTarget
)


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
        "mappings": [field_mapping_material(item) for item in binding.mappings],
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


@dataclass(frozen=True, slots=True, init=False)
class ProfileEvidenceSupport:
    support_id: str
    profile_id: EvidenceProfileId
    slot_id: str
    role_mode: ProfileRoleMode
    support_kind: ProfileSupportKind
    source_binding_ids: tuple[str, ...]
    semantic_projection: tuple[tuple[str, object], ...]

    def __init__(self) -> None:
        raise TypeError("ProfileEvidenceSupport must be created through its factory")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ProfileEvidenceSupport is final")


def _profile_binding_id(
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
    mapping: MappingCandidate | RepoMapping,
) -> str:
    return _digest(
        "binding-profile",
        {
            "artifact_sha256": str(evidence.artifact.sha256),
            "artifact_size": evidence.artifact.size,
            "evidence_id": evidence.evidence_id,
            "binding": _binding_material(binding),
            "mapping": _mapping_identity(mapping),
        },
    )


def _profile_projection(
    target: ProfileEvidenceTarget,
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
    *,
    claim_metric: str,
) -> tuple[tuple[str, object], ...]:
    values: list[tuple[str, object]] = []
    if any(
        item.kind is not ProvenanceKind.ADAPTER_EXTRACTION
        for item in evidence.adapter_match.match_evidence
    ):
        raise AnalysisContractError(
            "only adapter-extracted evidence can create profile support"
        )
    selected_keys = {
        value
        for mapping in binding.mappings
        for value in (mapping.target_field, mapping.selector.expression)
    }
    metric_selected = any(
        mapping.target_field == "metric_value" for mapping in binding.mappings
    )
    if (
        target.slot_id == "benchmark.workload"
        and binding.kind is ArtifactKind.DATASET
    ):
        references = tuple(
            reference
            for observation in evidence.observations
            for reference in observation.dataset_references
            if reference.path == binding.path
        )
        if len(references) != 1:
            raise AnalysisContractError(
                "profile workload dataset must identify one exact passive artifact"
            )
        values.extend(
            (
                (f"{binding.role.value}.dataset_path", str(binding.path)),
                (
                    f"{binding.role.value}.dataset_sha256",
                    str(evidence.artifact.sha256),
                ),
                (f"{binding.role.value}.dataset_size", evidence.artifact.size),
                (
                    f"{binding.role.value}.dataset_split",
                    references[0].split,
                ),
            )
        )
    for observation in evidence.observations:
        if observation.provenance.kind is not ProvenanceKind.ADAPTER_EXTRACTION:
            raise AnalysisContractError(
                "only adapter-extracted values can create profile support"
            )
        if metric_selected and target.slot_id in {
            "benchmark.results",
            "measurement.metric_identity",
        }:
            if observation.metric_name is not None:
                values.append((f"{binding.role.value}.metric_name", observation.metric_name))
                if target.slot_id == "benchmark.results":
                    values.append((f"{binding.role.value}.metric_value", observation.metric_value))
        prefix = target.slot_id + "."
        for item in observation.config_values:
            if item.provenance.kind is not ProvenanceKind.ADAPTER_EXTRACTION:
                raise AnalysisContractError(
                    "only adapter-extracted values can create profile support"
                )
            if item.key in selected_keys and item.key.startswith(prefix):
                values.append((f"{binding.role.value}.{item.key}", item.value))
    return tuple(values)


def validated_profile_evidence_support(
    *,
    target: ProfileEvidenceTarget,
    selection: EvidenceProfileSelection,
    normalized_evidence: tuple[NormalizedEvidence, ...],
    selected_mapping: MappingCandidate | RepoMapping,
    claim_metric: str,
    required_procedure: UpstreamAggregationProcedure | None = None,
) -> ProfileEvidenceSupport:
    """Bind one grouped profile slot to exact trusted normalized evidence."""

    if type(target) is not ProfileEvidenceTarget:
        raise TypeError("profile support requires ProfileEvidenceTarget")
    if type(selection) is not EvidenceProfileSelection:
        raise TypeError("profile support requires EvidenceProfileSelection")
    if (
        target.profile_id is not selection.profile_id
        or target.activation_policy_id != selection.activation_policy_id
    ):
        raise AnalysisContractError("profile support target does not match selection")
    if type(selected_mapping) not in {MappingCandidate, RepoMapping}:
        raise TypeError("profile support requires one selected mapping")
    if not _mapping_is_trusted_for_support(selected_mapping):
        raise AnalysisContractError(
            "provider profile mapping requires explicit RepoMapping approval"
        )
    if not isinstance(normalized_evidence, tuple) or not all(
        type(item) is NormalizedEvidence for item in normalized_evidence
    ):
        raise TypeError("profile support requires normalized evidence")
    if not isinstance(claim_metric, str) or not claim_metric.strip():
        raise AnalysisContractError("profile support claim metric is invalid")
    if required_procedure is not None:
        if type(required_procedure) is not UpstreamAggregationProcedure:
            raise TypeError("profile support procedure is invalid")
        if target.slot_id != "benchmark.run_protocol":
            raise AnalysisContractError(
                "only the Benchmark run-protocol slot may require aggregation"
            )
        if (
            _mapping_procedure_status(
                selected_mapping,
                normalized_evidence,
                required_procedure,
            )
            != "satisfied"
        ):
            raise AnalysisContractError(
                "required benchmark aggregation is not independently recoverable"
            )
    projection: list[tuple[str, object]] = []
    binding_ids: list[str] = []
    represented_roles: set[ExperimentRole] = set()
    supporting_kinds: set[ArtifactKind] = set()
    for binding in selected_mapping.bindings:
        if binding.kind not in target.allowed_artifact_kinds:
            continue
        evidence = evidence_for_binding(binding, normalized_evidence)
        if evidence is None or not _evidence_matches_binding(evidence, binding):
            continue
        values = _profile_projection(
            target,
            evidence,
            binding,
            claim_metric=claim_metric,
        )
        if not values:
            continue
        projection.extend(values)
        binding_ids.append(_profile_binding_id(evidence, binding, selected_mapping))
        represented_roles.add(binding.role)
        supporting_kinds.add(binding.kind)
    paired = {
        ExperimentRole.BASELINE,
        ExperimentRole.CANDIDATE,
    }.issubset(represented_roles)
    shared = ExperimentRole.REFERENCE in represented_roles
    if target.role_mode is ProfileRoleMode.PAIRED_BASELINE_CANDIDATE and not paired:
        raise AnalysisContractError("profile slot needs baseline and candidate support")
    if target.role_mode is ProfileRoleMode.SHARED_REFERENCE and not shared:
        raise AnalysisContractError("profile slot needs shared reference support")
    if target.role_mode is ProfileRoleMode.SHARED_REFERENCE_OR_PAIRED and not (
        shared or paired
    ):
        raise AnalysisContractError("profile slot needs shared or paired support")
    if not projection or not binding_ids:
        raise AnalysisContractError("profile evidence is not independently recoverable")
    if required_procedure is not None:
        projection.append(
            ("required_upstream_aggregation", required_procedure.value)
        )
    if len(binding_ids) > 16 or len(projection) > 64:
        raise AnalysisContractError("profile support exceeds its structural bound")
    ordered_projection = tuple(sorted(set(projection), key=lambda item: (item[0], repr(item[1]))))
    ordered_bindings = tuple(sorted(set(binding_ids)))
    if (
        target.slot_id == "benchmark.workload"
        and ArtifactKind.DATASET in supporting_kinds
    ):
        support_kind = ProfileSupportKind.ARTIFACT_BINDING
    elif (
        target.slot_id == "benchmark.evaluator"
        and selection.result_form is BenchmarkResultForm.DETERMINISTIC_DERIVATION
    ):
        support_kind = ProfileSupportKind.DETERMINISTIC_DERIVATION
    elif target.slot_id in {"benchmark.results", "measurement.metric_identity"}:
        support_kind = ProfileSupportKind.NORMALIZED_METRIC
    else:
        support_kind = ProfileSupportKind.STRUCTURED_COMPONENT
    if support_kind not in target.allowed_support_kinds:
        raise AnalysisContractError("profile support kind is not allowed for its slot")
    material = {
        "profile_id": target.profile_id.value,
        "slot_id": target.slot_id,
        "activation": target.activation_policy_id,
        "role_mode": target.role_mode.value,
        "support_kind": support_kind.value,
        "bindings": ordered_bindings,
        "projection": ordered_projection,
    }
    instance = object.__new__(ProfileEvidenceSupport)
    object.__setattr__(instance, "support_id", _digest("support-profile", material))
    object.__setattr__(instance, "profile_id", target.profile_id)
    object.__setattr__(instance, "slot_id", target.slot_id)
    object.__setattr__(instance, "role_mode", target.role_mode)
    object.__setattr__(instance, "support_kind", support_kind)
    object.__setattr__(instance, "source_binding_ids", ordered_bindings)
    object.__setattr__(instance, "semantic_projection", ordered_projection)
    return instance


_PROCEDURE_CONFIG_KEYS = frozenset(
    {
        "evaluation.aggregation",
        "evaluation.aggregation_procedure",
        "evaluation.retry_aggregation",
        "benchmark.run_protocol.aggregation",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class MeasurementProcedureSupport:
    support_id: str
    procedure: UpstreamAggregationProcedure
    evidence_ids: tuple[str, ...]
    binding_ids: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("MeasurementProcedureSupport must be created through its factory")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementProcedureSupport is final")


def _procedure_values(
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
) -> tuple[object, ...]:
    selected_keys = {
        value
        for mapping in binding.mappings
        for value in (mapping.target_field, mapping.selector.expression)
    }
    values = tuple(
        config.value
        for observation in evidence.observations
        for config in observation.config_values
        if config.key in _PROCEDURE_CONFIG_KEYS and config.key in selected_keys
    )
    unique = {
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ): value
        for value in values
    }
    return tuple(unique[key] for key in sorted(unique))


def _procedure_bindings(
    mapping: MappingCandidate | RepoMapping,
    normalized_evidence: tuple[NormalizedEvidence, ...],
) -> tuple[tuple[NormalizedEvidence, ArtifactBinding], ...]:
    values: list[tuple[NormalizedEvidence, ArtifactBinding]] = []
    for binding in mapping.bindings:
        if (
            binding.kind not in {ArtifactKind.CONFIG, ArtifactKind.BENCHMARK}
            or binding.role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}
        ):
            continue
        evidence = evidence_for_binding(binding, normalized_evidence)
        if evidence is not None and _evidence_matches_binding(evidence, binding):
            values.append((evidence, binding))
    return tuple(values)


def _mapping_procedure_status(
    mapping: MappingCandidate | RepoMapping,
    normalized_evidence: tuple[NormalizedEvidence, ...],
    procedure: UpstreamAggregationProcedure,
) -> str:
    by_role: dict[ExperimentRole, list[object]] = {
        ExperimentRole.BASELINE: [],
        ExperimentRole.CANDIDATE: [],
    }
    for evidence, binding in _procedure_bindings(mapping, normalized_evidence):
        by_role[binding.role].extend(_procedure_values(evidence, binding))
    if any(not values for values in by_role.values()):
        return "missing"
    flattened = tuple(value for values in by_role.values() for value in values)
    if any(not isinstance(value, str) for value in flattened):
        return "unsupported"
    if any(len(values) != 1 for values in by_role.values()):
        return "ambiguous"
    expected = procedure.value
    return "satisfied" if all(value == expected for value in flattened) else "unsupported"


def validated_measurement_procedure_support(
    *,
    claim: CanonicalScientificClaim,
    procedure: UpstreamAggregationProcedure,
    normalized_evidence: tuple[NormalizedEvidence, ...],
    selected_mapping: MappingCandidate | RepoMapping,
) -> MeasurementProcedureSupport:
    """Create procedure support only from source recovery and trusted bindings."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("measurement procedure support requires CanonicalScientificClaim")
    if type(claim.primary) is not MetricImprovementClaim:
        raise AnalysisContractError(
            "measurement procedure support requires metric improvement semantics"
        )
    if type(procedure) is not UpstreamAggregationProcedure:
        raise TypeError("measurement procedure support requires its canonical enum")
    requirement = recover_upstream_procedure_requirement(claim.reference)
    if requirement is None or requirement.procedure is not procedure:
        raise AnalysisContractError(
            "measurement procedure does not match exact source recovery"
        )
    if not requirement.deterministically_supported:
        raise AnalysisContractError("measurement procedure is not natively supported")
    if type(selected_mapping) not in {MappingCandidate, RepoMapping}:
        raise TypeError("measurement procedure support requires a selected mapping")
    if not _mapping_is_trusted_for_support(selected_mapping):
        raise AnalysisContractError(
            "provider measurement mapping requires explicit approval"
        )
    if not isinstance(normalized_evidence, tuple) or not all(
        type(item) is NormalizedEvidence for item in normalized_evidence
    ):
        raise TypeError("measurement procedure support requires normalized evidence")
    if (
        _mapping_procedure_status(selected_mapping, normalized_evidence, procedure)
        != "satisfied"
    ):
        raise AnalysisContractError(
            "required measurement procedure is not independently recoverable"
        )
    support_records: list[tuple[str, str, str]] = []
    for evidence, binding in _procedure_bindings(selected_mapping, normalized_evidence):
        if any(
            provenance.kind is not ProvenanceKind.ADAPTER_EXTRACTION
            for observation in evidence.observations
            for provenance in (
                observation.provenance,
                *(item.provenance for item in observation.config_values),
            )
        ):
            raise AnalysisContractError(
                "provider values cannot create measurement procedure support"
            )
        if _procedure_values(evidence, binding):
            if binding.kind is ArtifactKind.CONFIG:
                support = validated_artifact_support(
                    evidence,
                    binding,
                    selected_mapping,
                    metric=claim.primary.metric,
                )
                support_records.append(
                    (support.evidence_id, support.binding_id, support.support_id)
                )
            else:
                binding_id = _profile_binding_id(
                    evidence,
                    binding,
                    selected_mapping,
                )
                support_records.append(
                    (
                        evidence.evidence_id,
                        binding_id,
                        _digest(
                            "support-benchmark-procedure",
                            {
                                "evidence_id": evidence.evidence_id,
                                "binding_id": binding_id,
                                "procedure": procedure.value,
                            },
                        ),
                    )
                )
    if len(support_records) != 2:
        raise AnalysisContractError(
            "measurement procedure needs one exact config support per role"
        )
    ordered = tuple(sorted(support_records, key=lambda item: item[1]))
    material = {
        "claim_id": claim.reference.claim_id,
        "claim_source_sha256": requirement.source_text_sha256,
        "procedure": procedure.value,
        "supports": [item[2] for item in ordered],
    }
    instance = object.__new__(MeasurementProcedureSupport)
    object.__setattr__(
        instance,
        "support_id",
        _digest("support-measurement", material),
    )
    object.__setattr__(instance, "procedure", procedure)
    object.__setattr__(
        instance,
        "evidence_ids",
        tuple(item[0] for item in ordered),
    )
    object.__setattr__(
        instance,
        "binding_ids",
        tuple(item[1] for item in ordered),
    )
    return instance


ObligationSupportReference: TypeAlias = (
    ClaimFieldSupport
    | ArtifactEvidenceSupport
    | MeasurementProcedureSupport
    | ProfileEvidenceSupport
)


_MISSING_REASONS = frozenset(
    {
        EvidenceObligationReason.REQUIRED_ARTIFACT_NOT_FOUND,
        EvidenceObligationReason.REQUIRED_CLAIM_FIELD_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_METRIC_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_THRESHOLD_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_ROLE_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_SPLIT_NOT_RECOVERED,
        EvidenceObligationReason.REQUIRED_MEASUREMENT_PROCEDURE_NOT_RECOVERED,
        EvidenceObligationReason.DEPENDENCY_MISSING,
        EvidenceObligationReason.REQUIRED_PROFILE_EVIDENCE_NOT_RECOVERED,
    }
)
_AMBIGUOUS_REASONS = frozenset(
    {
        EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES,
        EvidenceObligationReason.CONFLICTING_VALIDATED_MAPPINGS,
        EvidenceObligationReason.EXPLICIT_MAPPING_APPROVAL_REQUIRED,
        EvidenceObligationReason.APPROVED_MAPPING_NOT_APPLICABLE,
        EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED,
        EvidenceObligationReason.PROFILE_EVIDENCE_AMBIGUOUS,
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
        EvidenceObligationReason.MEASUREMENT_PROCEDURE_NOT_SUPPORTED,
        EvidenceObligationReason.DEPENDENCY_UNSUPPORTED,
        EvidenceObligationReason.PROFILE_EVIDENCE_NOT_SUPPORTED,
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
            MeasurementComponentTarget,
            ProfileEvidenceTarget,
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
                type(item)
                in {
                    ClaimFieldSupport,
                    ArtifactEvidenceSupport,
                    MeasurementProcedureSupport,
                    ProfileEvidenceSupport,
                }
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
        if type(self.target) is MeasurementComponentTarget and any(
            type(item) is not MeasurementProcedureSupport
            or item.procedure is not self.target.required_procedure
            for item in self.support_references
        ):
            raise AnalysisContractError(
                "measurement target requires exact procedure support"
            )
        if type(self.target) is ProfileEvidenceTarget and any(
            type(item) is not ProfileEvidenceSupport
            or item.profile_id is not self.target.profile_id
            or item.slot_id != self.target.slot_id
            or item.role_mode is not self.target.role_mode
            for item in self.support_references
        ):
            raise AnalysisContractError(
                "profile target requires exact profile evidence support"
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
    policy: ClaimEvidencePolicy | ProfiledEvidencePolicy,
    compiler_supported: bool,
    obligations: tuple[EvidenceObligation, ...],
    mapping_question_id: str | None,
) -> EvidenceObligationBundle:
    """Internal factory used only after deterministic obligation instantiation."""

    _canonical_id(claim_id, "obligation bundle claim_id")
    if type(policy) not in {ClaimEvidencePolicy, ProfiledEvidencePolicy}:
        raise TypeError("obligation bundle policy is invalid")
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
    if not _runtime_evidence_matches_binding(evidence, binding):
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
        field_mapping_projection(binding.mappings),
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
    normalized_evidence: tuple[NormalizedEvidence, ...],
    metric: str,
) -> EvidenceObligation:
    bindings = _matching_bindings(slot, selected_mapping)
    supports: list[ArtifactEvidenceSupport] = []
    try:
        for binding in bindings:
            evidence = evidence_for_binding(binding, normalized_evidence)
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
        if (evidence := evidence_for_binding(binding, normalized_evidence)) is not None
        and _evidence_matches_binding(evidence, binding)
        and (
            slot.kind is not ArtifactKind.RESULTS
            or any(item.metric_name == metric for item in evidence.observations)
        )
    )
    valid_question_bindings = tuple(
        binding
        for binding in question_bindings
        if (evidence := evidence_for_binding(binding, normalized_evidence)) is not None
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


def _measurement_procedure_obligation(
    *,
    claim: CanonicalScientificClaim,
    normalized_evidence: tuple[NormalizedEvidence, ...],
    mapping_candidates: tuple[MappingCandidate, ...],
    selected_mapping: MappingCandidate | RepoMapping | None,
    mapping_question: MappingQuestion | None,
    ambiguity_reason: EvidenceObligationReason | None,
) -> EvidenceObligation:
    requirement = recover_upstream_procedure_requirement(claim.reference)
    if requirement is None:
        raise AnalysisContractError(
            "measurement obligation has no source-recovered requirement"
        )
    target = MeasurementComponentTarget(
        MeasurementComponentKind.RETRY_AGGREGATION,
        requirement.procedure,
    )
    if not requirement.deterministically_supported:
        return EvidenceObligation(
            obligation_id="measurement.retry_aggregation",
            target=target,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.MEASUREMENT_PROCEDURE_NOT_SUPPORTED,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    if selected_mapping is not None:
        status = _mapping_procedure_status(
            selected_mapping,
            normalized_evidence,
            requirement.procedure,
        )
        if status == "satisfied":
            try:
                support = validated_measurement_procedure_support(
                    claim=claim,
                    procedure=requirement.procedure,
                    normalized_evidence=normalized_evidence,
                    selected_mapping=selected_mapping,
                )
            except AnalysisContractError:
                return EvidenceObligation(
                    obligation_id="measurement.retry_aggregation",
                    target=target,
                    state=EvidenceObligationState.UNSUPPORTED,
                    reason=EvidenceObligationReason.MEASUREMENT_PROCEDURE_NOT_SUPPORTED,
                    effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
                )
            return EvidenceObligation(
                obligation_id="measurement.retry_aggregation",
                target=target,
                state=EvidenceObligationState.SATISFIED,
                reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
                effect=EvidenceObligationEffect.NONE,
                support_references=(support,),
            )
        if status == "missing":
            return EvidenceObligation(
                obligation_id="measurement.retry_aggregation",
                target=target,
                state=EvidenceObligationState.MISSING,
                reason=(
                    EvidenceObligationReason.REQUIRED_MEASUREMENT_PROCEDURE_NOT_RECOVERED
                ),
                effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
            )
        if status == "ambiguous":
            return EvidenceObligation(
                obligation_id="measurement.retry_aggregation",
                target=target,
                state=EvidenceObligationState.AMBIGUOUS,
                reason=EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED,
                effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
            )
        return EvidenceObligation(
            obligation_id="measurement.retry_aggregation",
            target=target,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.MEASUREMENT_PROCEDURE_NOT_SUPPORTED,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )

    statuses = tuple(
        _mapping_procedure_status(
            candidate,
            normalized_evidence,
            requirement.procedure,
        )
        for candidate in mapping_candidates
    )
    if "satisfied" in statuses:
        if mapping_question is not None:
            reason = (
                ambiguity_reason
                if ambiguity_reason in _AMBIGUOUS_REASONS
                else EvidenceObligationReason.MULTIPLE_VALIDATED_CANDIDATES
            )
            return EvidenceObligation(
                obligation_id="measurement.retry_aggregation",
                target=target,
                state=EvidenceObligationState.AMBIGUOUS,
                reason=reason,
                effect=EvidenceObligationEffect.REQUIRES_MAPPING,
            )
        return EvidenceObligation(
            obligation_id="measurement.retry_aggregation",
            target=target,
            state=EvidenceObligationState.AMBIGUOUS,
            reason=EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    if "unsupported" in statuses or "ambiguous" in statuses:
        return EvidenceObligation(
            obligation_id="measurement.retry_aggregation",
            target=target,
            state=EvidenceObligationState.UNSUPPORTED,
            reason=EvidenceObligationReason.MEASUREMENT_PROCEDURE_NOT_SUPPORTED,
            effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
        )
    return EvidenceObligation(
        obligation_id="measurement.retry_aggregation",
        target=target,
        state=EvidenceObligationState.MISSING,
        reason=EvidenceObligationReason.REQUIRED_MEASUREMENT_PROCEDURE_NOT_RECOVERED,
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
    measurement_template = False
    for template_id in policy.obligation_template_ids:
        if template_id.startswith("claim."):
            obligations.append(_claim_obligation(template_id, claim))
        elif template_id.startswith("artifact."):
            artifact_templates.append(template_id)
        elif template_id == "measurement.retry_aggregation":
            measurement_template = True
        elif template_id != "comparison.readiness":
            raise AnalysisContractError("claim policy contains an unknown template")

    if artifact_templates:
        if audit_claim is None:
            raise AnalysisContractError(
                "unsupported compiler policy cannot instantiate artifact obligations"
            )
        for template_id in artifact_templates:
            slot = _artifact_target(template_id)
            if selected_mapping is not None:
                obligation = _selected_artifact_obligation(
                    template_id,
                    slot,
                    selected_mapping=selected_mapping,
                    normalized_evidence=normalized_evidence,
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

    if measurement_template:
        obligations.append(
            _measurement_procedure_obligation(
                claim=claim,
                normalized_evidence=normalized_evidence,
                mapping_candidates=mapping_candidates,
                selected_mapping=selected_mapping,
                mapping_question=mapping_question,
                ambiguity_reason=ambiguity_reason,
            )
        )

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


def assess_profiled_evidence_obligations(
    *,
    claim: CanonicalScientificClaim,
    policy: ProfiledEvidencePolicy,
    audit_claim: AuditClaimSpec | None,
    artifacts: tuple[ArtifactCandidate, ...],
    normalized_evidence: tuple[NormalizedEvidence, ...],
    mapping_candidates: tuple[MappingCandidate, ...],
    selected_mapping: MappingCandidate | RepoMapping | None,
    mapping_question: MappingQuestion | None,
    ambiguity_reason: EvidenceObligationReason | None,
) -> EvidenceObligationBundle:
    """Assess one fixed evidence profile without changing legacy assessment."""

    if type(policy) is not ProfiledEvidencePolicy:
        raise TypeError("profiled obligation assessment requires ProfiledEvidencePolicy")
    if policy.claim_policy != claim_evidence_policy(claim):
        raise AnalysisContractError("profiled policy does not match claim semantics")
    if policy.profile_id is EvidenceProfileId.TRAINING_EXPERIMENT_V0:
        return assess_evidence_obligations(
            claim=claim,
            policy=policy.claim_policy,
            audit_claim=audit_claim,
            artifacts=artifacts,
            normalized_evidence=normalized_evidence,
            mapping_candidates=mapping_candidates,
            selected_mapping=selected_mapping,
            mapping_question=mapping_question,
            ambiguity_reason=ambiguity_reason,
        )
    selection = policy.profile_selection
    if selection.profile_id is not EvidenceProfileId.BENCHMARK_MEASUREMENT_V0:
        raise AnalysisContractError("profiled obligation selection is invalid")
    if audit_claim is None or type(audit_claim) is not AuditClaimSpec:
        raise AnalysisContractError("Benchmark obligations require an executable claim")
    if audit_claim != compile_audit_claim(claim):
        raise AnalysisContractError("Benchmark obligations require the canonical Audit claim")

    obligations: list[EvidenceObligation] = [
        _claim_obligation(template_id, claim)
        for template_id in policy.claim_template_ids
    ]
    profile_ids = tuple(
        item for item in policy.profile_template_ids if item.startswith("benchmark.")
    )
    if "measurement.metric_identity" in policy.profile_template_ids:
        profile_ids = (*profile_ids, "measurement.metric_identity")
    procedure_requirement = recover_upstream_procedure_requirement(claim.reference)
    for slot_id in profile_ids:
        target = profile_evidence_target(selection, slot_id)
        if selected_mapping is not None:
            if (
                slot_id == "benchmark.run_protocol"
                and procedure_requirement is not None
                and not procedure_requirement.deterministically_supported
            ):
                obligation = EvidenceObligation(
                    obligation_id=slot_id,
                    target=target,
                    state=EvidenceObligationState.UNSUPPORTED,
                    reason=EvidenceObligationReason.MEASUREMENT_PROCEDURE_NOT_SUPPORTED,
                    effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
                )
            else:
                try:
                    support = validated_profile_evidence_support(
                        target=target,
                        selection=selection,
                        normalized_evidence=normalized_evidence,
                        selected_mapping=selected_mapping,
                        claim_metric=audit_claim.metric,
                        required_procedure=(
                            procedure_requirement.procedure
                            if slot_id == "benchmark.run_protocol"
                            and procedure_requirement is not None
                            else None
                        ),
                    )
                except AnalysisContractError:
                    obligation = EvidenceObligation(
                        obligation_id=slot_id,
                        target=target,
                        state=EvidenceObligationState.MISSING,
                        reason=(
                            EvidenceObligationReason.REQUIRED_PROFILE_EVIDENCE_NOT_RECOVERED
                        ),
                        effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
                    )
                else:
                    obligation = EvidenceObligation(
                        obligation_id=slot_id,
                        target=target,
                        state=EvidenceObligationState.SATISFIED,
                        reason=EvidenceObligationReason.VALIDATED_SUPPORT_BOUND,
                        effect=EvidenceObligationEffect.NONE,
                        support_references=(support,),
                    )
        else:
            if mapping_candidates and mapping_question is not None:
                obligation = EvidenceObligation(
                    obligation_id=slot_id,
                    target=target,
                    state=EvidenceObligationState.AMBIGUOUS,
                    reason=(
                        ambiguity_reason
                        if ambiguity_reason in _AMBIGUOUS_REASONS
                        else EvidenceObligationReason.PROFILE_EVIDENCE_AMBIGUOUS
                    ),
                    effect=EvidenceObligationEffect.REQUIRES_MAPPING,
                )
            elif mapping_candidates:
                obligation = EvidenceObligation(
                    obligation_id=slot_id,
                    target=target,
                    state=EvidenceObligationState.AMBIGUOUS,
                    reason=EvidenceObligationReason.CLARIFICATION_NOT_BOUNDED,
                    effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
                )
            else:
                obligation = EvidenceObligation(
                    obligation_id=slot_id,
                    target=target,
                    state=EvidenceObligationState.MISSING,
                    reason=(
                        EvidenceObligationReason.REQUIRED_PROFILE_EVIDENCE_NOT_RECOVERED
                    ),
                    effect=EvidenceObligationEffect.BLOCKS_PARTIAL,
                )
        obligations.append(obligation)

    dependencies = tuple(item.obligation_id for item in obligations)
    obligations.append(
        EvidenceObligation(
            obligation_id=policy.comparison_template_id,
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
        compiler_supported=True,
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
    "MeasurementComponentTarget",
    "MeasurementProcedureSupport",
    "ProfileEvidenceSupport",
    "ProfileEvidenceTarget",
    "EvidenceObligationDecision",
    "EvidenceObligationEffect",
    "EvidenceObligation",
    "EvidenceObligationBundle",
    "EvidenceObligationReason",
    "EvidenceObligationState",
    "ObligationSupportReference",
    "ObligationTarget",
    "assess_evidence_obligations",
    "assess_profiled_evidence_obligations",
    "claim_field_support",
    "evidence_obligations_json_bytes",
    "legacy_missing_evidence",
    "profile_evidence_target",
    "validated_artifact_support",
    "validated_measurement_procedure_support",
    "validated_profile_evidence_support",
]
