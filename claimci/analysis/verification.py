"""Trusted pre-Audit verification input identities and commitments."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from claimci.dataset_check import hash_sample
from claimci.measurement import MeasurementProtocolIdentity, MeasurementProtocolPair
from claimci.parsing import unique_json_object, validate_json_graph

from .contracts import (
    AnalysisContractError,
    AuditClaimSpec,
    ArtifactBinding,
    ArtifactKind,
    DatasetSplit,
    EphemeralAuditPlan,
    ExperimentRole,
    FieldMapping,
    GitCommitSha,
    MappingCandidate,
    MappingTrust,
    NormalizedEvidence,
    PassiveArtifact,
    RepoMapping,
    RepositoryIdentity,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    TableSelector,
    field_mapping_identity,
)
from .claims import audit_relevant_claim_projection
from .claim_types import claim_semantic_projection
from .obligations import EvidenceObligationBundle
from .profiles import (
    _BENCHMARK_PROFILE_FIELDS,
    _canonical_profile_scalar,
    BenchmarkResultForm,
    EvidenceProfileId,
    EvidenceProfileSelection,
)


VERIFICATION_INPUT_SNAPSHOT_VERSION = "claimci.verification-input-snapshot.v1"
AUDIT_SEMANTICS_COMPATIBILITY_VERSION = "claimci.audit-semantics-compatibility.v1"
VERIFICATION_INPUT_COMPLETE_MAX_BYTES = 65_536
VERIFICATION_INPUT_MAX_ARTIFACTS = 32
_ADAPTER_VERSION = re.compile(r"(?:^|[-_.])(v[0-9]+)\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")


class VerificationContractError(ValueError):
    """A verification identity violates its trusted bounded contract."""


class VerificationSnapshotCapability(str, Enum):
    COMPLETE = "complete"
    IDENTITY_ONLY = "identity_only"
    UNAVAILABLE = "unavailable"


def _jsonable(value: object, *, state: list[int]) -> object:
    state[0] += 1
    if state[0] > 100_000:
        raise VerificationContractError("verification projection has too many values")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise VerificationContractError("verification numbers must be finite")
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value, state=state)
    if isinstance(value, (RepositoryPath, Sha256Digest, GitCommitSha)):
        return str(value)
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise VerificationContractError("verification mapping keys must be text")
        return {
            key: _jsonable(item, state=state)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item, state=state) for item in value]
    if dataclasses.is_dataclass(value) and type(value).__module__.startswith(
        ("claimci.analysis", "claimci.measurement")
    ):
        return {
            item.name: _jsonable(getattr(value, item.name), state=state)
            for item in dataclasses.fields(value)
        }
    raise TypeError("value is not an approved verification projection")


def _canonical_json_bytes(value: object) -> bytes:
    """Encode one finite, stable verification projection."""

    try:
        return json.dumps(
            _jsonable(value, state=[0]),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, OverflowError) as error:
        raise VerificationContractError(
            "verification projection is not finite canonical JSON"
        ) from error


def _digest(domain: str, value: object) -> Sha256Digest:
    if not isinstance(domain, str) or not domain:
        raise TypeError("verification digest domain must be text")
    material = domain.encode("ascii") + b"\0" + _canonical_json_bytes(value)
    return Sha256Digest(hashlib.sha256(material).hexdigest())


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise VerificationContractError(f"{label} is not a canonical identifier")
    return value


@dataclass(frozen=True, slots=True)
class VerificationTablePredicateIdentity:
    column: str
    scalar_type: str
    canonical_value: str

    def __post_init__(self) -> None:
        _identifier(self.column, "verification table predicate column")
        _identifier(self.scalar_type, "verification table predicate scalar type")
        if not isinstance(self.canonical_value, str) or len(self.canonical_value) > 1_024:
            raise VerificationContractError(
                "verification table predicate value exceeds its bound"
            )


@dataclass(frozen=True, slots=True)
class VerificationSelectorIdentity:
    target_field: str
    kind: SelectorKind
    expression: str
    table_predicates: tuple[VerificationTablePredicateIdentity, ...] = ()
    expected_cardinality: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.target_field, "verification selector target")
        if type(self.kind) is not SelectorKind:
            raise TypeError("verification selector kind must be SelectorKind")
        if (
            not isinstance(self.expression, str)
            or not self.expression
            or len(self.expression) > 1_024
        ):
            raise VerificationContractError("verification selector expression is invalid")
        if (
            not isinstance(self.table_predicates, tuple)
            or len(self.table_predicates) > 8
            or not all(
                type(item) is VerificationTablePredicateIdentity
                for item in self.table_predicates
            )
        ):
            raise VerificationContractError("verification selector predicates exceed their bound")
        if self.table_predicates:
            if self.kind is not SelectorKind.COLUMN:
                raise VerificationContractError("table predicates require a column selector")
            if (
                isinstance(self.expected_cardinality, bool)
                or not isinstance(self.expected_cardinality, int)
                or not 1 <= self.expected_cardinality <= 32
            ):
                raise VerificationContractError("table selector cardinality is invalid")
        elif self.expected_cardinality is not None:
            raise VerificationContractError("non-table selector cannot carry cardinality")


def _selector_identity(mapping: FieldMapping) -> VerificationSelectorIdentity:
    if type(mapping) is not FieldMapping:
        raise TypeError("verification selector requires an exact FieldMapping")
    selector = mapping.selector
    if type(selector) is TableSelector:
        return VerificationSelectorIdentity(
            target_field=mapping.target_field,
            kind=selector.kind,
            expression=selector.expression,
            table_predicates=tuple(
                VerificationTablePredicateIdentity(
                    item.column,
                    item.scalar_type.value,
                    item.canonical_value,
                )
                for item in selector.predicates
            ),
            expected_cardinality=selector.expected_cardinality,
        )
    return VerificationSelectorIdentity(
        target_field=mapping.target_field,
        kind=selector.kind,
        expression=selector.expression,
    )


@dataclass(frozen=True, slots=True, init=False)
class VerificationArtifactIdentity:
    verification_evidence_id: str
    path: RepositoryPath
    kind: ArtifactKind
    role: ExperimentRole
    dataset_split: DatasetSplit | None
    source_sha256: Sha256Digest
    extraction_sha256: Sha256Digest
    audit_semantics_sha256: Sha256Digest
    adapter_id: str
    adapter_semantic_version: str | None
    evidence_projector_version: str
    selectors: tuple[VerificationSelectorIdentity, ...]

    def __init__(self) -> None:
        raise TypeError("VerificationArtifactIdentity must be created by materialization")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("VerificationArtifactIdentity is final")


def _artifact_identity_from_materialized(
    *,
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
    passive: PassiveArtifact,
    extraction_projection: object,
    audit_semantics_projection: object,
    evidence_projector_version: str,
) -> VerificationArtifactIdentity:
    """Create one identity only from an exact captured, bound adapter result."""

    if type(evidence) is not NormalizedEvidence:
        raise TypeError("verification artifact requires exact NormalizedEvidence")
    if type(binding) is not ArtifactBinding:
        raise TypeError("verification artifact requires exact ArtifactBinding")
    if type(passive) is not PassiveArtifact:
        raise TypeError("verification artifact requires exact PassiveArtifact")
    _identifier(evidence_projector_version, "evidence projector version")
    if evidence.artifact != passive.candidate:
        raise AnalysisContractError("verification artifact is not the captured candidate")
    if (
        binding.path != evidence.artifact.path
        or binding.kind is not evidence.artifact.kind
        or binding.adapter_id is None
        or binding.adapter_id != evidence.adapter_match.adapter_id
        or binding.mappings != evidence.adapter_match.mappings
    ):
        raise AnalysisContractError("verification binding does not match extracted evidence")
    selectors = tuple(
        _selector_identity(item)
        for item in sorted(binding.mappings, key=field_mapping_identity)
    )
    extraction = _digest(
        "claimci.verification-artifact.extraction.v1",
        extraction_projection,
    )
    audit_semantics = _digest(
        "claimci.verification-artifact.audit-semantics.v1",
        audit_semantics_projection,
    )
    adapter_match = _ADAPTER_VERSION.search(binding.adapter_id)
    identity_material = {
        "path": str(binding.path),
        "kind": binding.kind.value,
        "role": binding.role.value,
        "dataset_split": None if binding.dataset_split is None else binding.dataset_split.value,
        "source_sha256": str(passive.candidate.sha256),
        "extraction_sha256": str(extraction),
        "audit_semantics_sha256": str(audit_semantics),
        "adapter_id": binding.adapter_id,
        "adapter_semantic_version": None if adapter_match is None else adapter_match.group(1),
        "evidence_projector_version": evidence_projector_version,
        "selectors": selectors,
    }
    instance = object.__new__(VerificationArtifactIdentity)
    object.__setattr__(
        instance,
        "verification_evidence_id",
        "verification-evidence-"
        + str(
            _digest(
                "claimci.verification-artifact.identity.v1",
                identity_material,
            )
        ),
    )
    object.__setattr__(instance, "path", binding.path)
    object.__setattr__(instance, "kind", binding.kind)
    object.__setattr__(instance, "role", binding.role)
    object.__setattr__(instance, "dataset_split", binding.dataset_split)
    object.__setattr__(instance, "source_sha256", passive.candidate.sha256)
    object.__setattr__(instance, "extraction_sha256", extraction)
    object.__setattr__(instance, "audit_semantics_sha256", audit_semantics)
    object.__setattr__(instance, "adapter_id", binding.adapter_id)
    object.__setattr__(
        instance,
        "adapter_semantic_version",
        None if adapter_match is None else adapter_match.group(1),
    )
    object.__setattr__(instance, "evidence_projector_version", evidence_projector_version)
    object.__setattr__(instance, "selectors", selectors)
    return instance


@dataclass(frozen=True, slots=True, init=False)
class MeasurementProtocolReference:
    semantic_protocol_id: str
    source_snapshot_id: str

    def __init__(self) -> None:
        raise TypeError("MeasurementProtocolReference must be created from a protocol identity")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementProtocolReference is final")

    @classmethod
    def from_identity(cls, identity: MeasurementProtocolIdentity) -> "MeasurementProtocolReference":
        if type(identity) is not MeasurementProtocolIdentity:
            raise TypeError("measurement reference requires MeasurementProtocolIdentity")
        instance = object.__new__(MeasurementProtocolReference)
        object.__setattr__(instance, "semantic_protocol_id", identity.semantic_protocol_id)
        object.__setattr__(instance, "source_snapshot_id", identity.source_snapshot_id)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class AuditSemanticsCompatibility:
    audit_policy_semantics: str
    rule_set_semantics: str
    claim_compiler_semantics: str
    evidence_projector_versions: tuple[str, ...]
    adapter_semantic_versions: tuple[str, ...]
    evidence_profile_semantics: str
    benchmark_compiler_semantics: str | None
    audit_semantics_sha256: Sha256Digest
    version: str

    def __init__(self) -> None:
        raise TypeError("AuditSemanticsCompatibility must be created by fixed policy")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AuditSemanticsCompatibility is final")

    @classmethod
    def _from_fixed_versions(
        cls,
        *,
        audit_policy_semantics: str,
        rule_set_semantics: str,
        claim_compiler_semantics: str,
        evidence_projector_versions: tuple[str, ...],
        adapter_semantic_versions: tuple[str, ...],
        evidence_profile_semantics: str,
        benchmark_compiler_semantics: str | None = None,
    ) -> "AuditSemanticsCompatibility":
        text_values = {
            "audit policy semantics": audit_policy_semantics,
            "rule-set semantics": rule_set_semantics,
            "claim compiler semantics": claim_compiler_semantics,
            "evidence profile semantics": evidence_profile_semantics,
        }
        for label, value in text_values.items():
            _identifier(value, label)
        if benchmark_compiler_semantics is not None:
            _identifier(benchmark_compiler_semantics, "benchmark compiler semantics")
        for label, values in (
            ("evidence projector versions", evidence_projector_versions),
            ("adapter semantic versions", adapter_semantic_versions),
        ):
            if not isinstance(values, tuple) or not values or len(values) > 32:
                raise VerificationContractError(f"{label} exceed their bound")
            for value in values:
                _identifier(value, label)
        projectors = tuple(sorted(set(evidence_projector_versions)))
        adapters = tuple(sorted(set(adapter_semantic_versions)))
        material = {
            "version": AUDIT_SEMANTICS_COMPATIBILITY_VERSION,
            "audit_policy_semantics": audit_policy_semantics,
            "rule_set_semantics": rule_set_semantics,
            "claim_compiler_semantics": claim_compiler_semantics,
            "evidence_projector_versions": projectors,
            "adapter_semantic_versions": adapters,
            "evidence_profile_semantics": evidence_profile_semantics,
            "benchmark_compiler_semantics": benchmark_compiler_semantics,
        }
        instance = object.__new__(AuditSemanticsCompatibility)
        for name, value in (
            ("audit_policy_semantics", audit_policy_semantics),
            ("rule_set_semantics", rule_set_semantics),
            ("claim_compiler_semantics", claim_compiler_semantics),
            ("evidence_projector_versions", projectors),
            ("adapter_semantic_versions", adapters),
            ("evidence_profile_semantics", evidence_profile_semantics),
            ("benchmark_compiler_semantics", benchmark_compiler_semantics),
            (
                "audit_semantics_sha256",
                _digest("claimci.audit-semantics-compatibility.v1", material),
            ),
            ("version", AUDIT_SEMANTICS_COMPATIBILITY_VERSION),
        ):
            object.__setattr__(instance, name, value)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class VerificationClaimIdentity:
    claim_id: str
    canonical_claim_sha256: Sha256Digest
    audit_spec_sha256: Sha256Digest

    def __init__(self) -> None:
        raise TypeError("VerificationClaimIdentity must be created from an executable plan")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("VerificationClaimIdentity is final")


@dataclass(frozen=True, slots=True, init=False)
class VerificationMappingIdentity:
    source_mapping_id: str
    trust: MappingTrust
    binding_evidence_ids: tuple[str, ...]
    bindings_sha256: Sha256Digest
    approval_sha256: Sha256Digest | None

    def __init__(self) -> None:
        raise TypeError("VerificationMappingIdentity must be created from selected bindings")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("VerificationMappingIdentity is final")


@dataclass(frozen=True, slots=True, init=False)
class EvidenceProfileIdentity:
    selection_id: str
    profile_id: EvidenceProfileId
    profile_version: int
    benchmark_variant: str | None
    result_form: str | None
    activation_policy_id: str
    complete: bool
    profile_semantics_sha256: Sha256Digest

    def __init__(self) -> None:
        raise TypeError("EvidenceProfileIdentity must be created from fixed profile policy")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EvidenceProfileIdentity is final")


@dataclass(frozen=True, slots=True, init=False)
class EvidenceObligationIdentity:
    policy_id: str
    bundle_sha256: Sha256Digest
    obligation_semantics_sha256: Sha256Digest
    obligation_count: int

    def __init__(self) -> None:
        raise TypeError("EvidenceObligationIdentity must be created from engine obligations")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EvidenceObligationIdentity is final")


def _claim_identity(plan: EphemeralAuditPlan) -> VerificationClaimIdentity:
    if type(plan.audit_claim) is not AuditClaimSpec:
        raise AnalysisContractError("verification snapshot requires AuditClaimSpec")
    claim_projection: object = (
        {
            "claim_id": plan.claim.claim_id,
            "text": plan.claim.text,
        }
        if plan.scientific_claim is None
        else claim_semantic_projection(plan.scientific_claim)
    )
    instance = object.__new__(VerificationClaimIdentity)
    object.__setattr__(instance, "claim_id", plan.claim.claim_id)
    object.__setattr__(
        instance,
        "canonical_claim_sha256",
        _digest("claimci.verification-claim.canonical.v1", claim_projection),
    )
    object.__setattr__(
        instance,
        "audit_spec_sha256",
        _digest(
            "claimci.verification-claim.audit-spec.v1",
            audit_relevant_claim_projection(plan.audit_claim),
        ),
    )
    return instance


def _profile_identity(plan: EphemeralAuditPlan) -> EvidenceProfileIdentity:
    selection: EvidenceProfileSelection | None = (
        None if plan.profiled_policy is None else plan.profiled_policy.profile_selection
    )
    if selection is None:
        values = {
            "selection_id": "legacy-training-experiment-v0",
            "profile_id": EvidenceProfileId.TRAINING_EXPERIMENT_V0.value,
            "profile_version": 0,
            "benchmark_variant": None,
            "result_form": None,
            "activation_policy_id": "claimci.profile.training-experiment.v0",
            "complete": True,
        }
    else:
        if type(selection) is not EvidenceProfileSelection:
            raise TypeError("verification profile requires EvidenceProfileSelection")
        values = {
            "selection_id": selection.selection_id,
            "profile_id": selection.profile_id.value,
            "profile_version": selection.profile_version,
            "benchmark_variant": (
                None if selection.benchmark_variant is None else selection.benchmark_variant.value
            ),
            "result_form": None if selection.result_form is None else selection.result_form.value,
            "activation_policy_id": selection.activation_policy_id,
            "complete": selection.complete,
        }
    instance = object.__new__(EvidenceProfileIdentity)
    object.__setattr__(instance, "selection_id", values["selection_id"])
    object.__setattr__(instance, "profile_id", EvidenceProfileId(values["profile_id"]))
    object.__setattr__(instance, "profile_version", values["profile_version"])
    object.__setattr__(instance, "benchmark_variant", values["benchmark_variant"])
    object.__setattr__(instance, "result_form", values["result_form"])
    object.__setattr__(instance, "activation_policy_id", values["activation_policy_id"])
    object.__setattr__(instance, "complete", values["complete"])
    semantics = {
        key: value
        for key, value in values.items()
        if key != "selection_id"
    }
    object.__setattr__(
        instance,
        "profile_semantics_sha256",
        _digest("claimci.verification-profile.semantics.v1", semantics),
    )
    return instance


def _obligation_semantic_projection(bundle: EvidenceObligationBundle) -> object:
    return {
        "version": bundle.version,
        "policy_id": bundle.policy_id,
        "compiler_state": bundle.compiler_state.value,
        "compiler_reason": bundle.compiler_reason.value,
        "compiler_effect": bundle.compiler_effect.value,
        "decision": bundle.decision.value,
        "obligations": tuple(
            {
                "target": item.target,
                "state": item.state.value,
                "reason": item.reason.value,
                "effect": item.effect.value,
                "dependencies": item.dependency_ids,
            }
            for item in bundle.obligations
        ),
    }


def _obligation_identity(plan: EphemeralAuditPlan) -> EvidenceObligationIdentity:
    bundle = plan.evidence_obligations
    if bundle is None:
        full: object = {
            "policy_id": "claimci.legacy-training-obligations.v1",
            "state": "legacy_executable_plan",
        }
        semantics = full
        count = 0
        policy_id = "claimci.legacy-training-obligations.v1"
    else:
        if type(bundle) is not EvidenceObligationBundle:
            raise TypeError("verification obligations require EvidenceObligationBundle")
        full = bundle
        semantics = _obligation_semantic_projection(bundle)
        count = len(bundle.obligations)
        policy_id = bundle.policy_id
    instance = object.__new__(EvidenceObligationIdentity)
    object.__setattr__(instance, "policy_id", policy_id)
    object.__setattr__(
        instance,
        "bundle_sha256",
        _digest("claimci.verification-obligations.bundle.v1", full),
    )
    object.__setattr__(
        instance,
        "obligation_semantics_sha256",
        _digest("claimci.verification-obligations.semantics.v1", semantics),
    )
    object.__setattr__(instance, "obligation_count", count)
    return instance


def _normalized_extraction_projection(evidence: NormalizedEvidence) -> object:
    return {
        "adapter_id": evidence.adapter_match.adapter_id,
        "selectors": tuple(
            field_mapping_identity(item)
            for item in sorted(evidence.adapter_match.mappings, key=field_mapping_identity)
        ),
        "observations": tuple(
            {
                "metric_name": item.metric_name,
                "metric_value": item.metric_value,
                "run_id": item.run_id,
                "seed": item.seed,
                "experiment_role": item.experiment_role.value,
                "config_values": tuple(
                    (value.key, value.value)
                    for value in sorted(item.config_values, key=lambda value: value.key)
                ),
                "dataset_references": tuple(
                    (str(value.path), value.split)
                    for value in sorted(item.dataset_references, key=lambda value: str(value.path))
                ),
                "compute_evidence": tuple(
                    (value.name, value.value, value.unit)
                    for value in sorted(item.compute_evidence, key=lambda value: value.name)
                ),
            }
            for item in evidence.observations
        ),
    }


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON value {token}")


def _dataset_record_hashes(content: bytes) -> tuple[str, ...] | None:
    try:
        text = content.decode("utf-8")
        hashes: list[str] = []
        for line in text.split("\n"):
            if not line.strip():
                continue
            value = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=unique_json_object,
            )
            validate_json_graph(value, label="verification dataset")
            hashes.append(hash_sample(value))
        if not hashes:
            raise ValueError("dataset has no records")
        return tuple(hashes)
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        return None


def _dataset_hash_projection(content: bytes) -> object:
    hashes = _dataset_record_hashes(content)
    if hashes is None:
        return {
            "state": "invalid",
            "source_sha256": hashlib.sha256(content).hexdigest(),
        }
    counts = Counter(hashes)
    return {
        "state": "valid",
        "record_count": len(hashes),
        "record_multiset": tuple(
            {"sha256": digest, "count": counts[digest]}
            for digest in sorted(counts)
        ),
    }


def _artifact_semantic_projections(
    evidence: NormalizedEvidence,
    binding: ArtifactBinding,
    passive: PassiveArtifact,
    *,
    metric: str,
    benchmark_profile: bool,
    benchmark_result_form: BenchmarkResultForm | None = None,
) -> tuple[object, object, str]:
    if benchmark_profile:
        if type(benchmark_result_form) is not BenchmarkResultForm:
            raise TypeError("Benchmark semantics require a fixed result form")
    elif benchmark_result_form is not None:
        raise TypeError("Training semantics cannot carry a Benchmark result form")
    extraction = _normalized_extraction_projection(evidence)
    if binding.kind is ArtifactKind.DATASET:
        semantics = _dataset_hash_projection(passive.content)
        return extraction, semantics, "claimci.projector.dataset-jsonl.v1"
    if benchmark_profile:
        benchmark_configs: dict[str, object] = {}
        benchmark_config_encodings: dict[str, bytes] = {}
        for observation in evidence.observations:
            for value in observation.config_values:
                if value.key not in _BENCHMARK_PROFILE_FIELDS:
                    continue
                recovered = _canonical_profile_scalar(value.key, value.value)
                encoded = _canonical_json_bytes(recovered)
                previous = benchmark_config_encodings.get(value.key)
                if previous is not None and previous != encoded:
                    raise AnalysisContractError(
                        "verification Benchmark config values conflict"
                    )
                benchmark_configs[value.key] = recovered
                benchmark_config_encodings[value.key] = encoded
        configs = tuple(sorted(benchmark_configs.items()))
    else:
        configs = tuple(
            sorted(
                (value.key, value.value)
                for observation in evidence.observations
                for value in observation.config_values
            )
        )
    metric_values: list[dict[str, object]] = []
    represented_metric_mismatches: set[str] = set()
    for observation in evidence.observations:
        if observation.metric_name is None:
            continue
        if observation.metric_name != metric:
            if benchmark_profile:
                represented_metric_mismatches.add(observation.metric_name)
            continue
        represented: dict[str, object] = {
            "metric_name": observation.metric_name,
            "metric_value": observation.metric_value,
        }
        if (
            not benchmark_profile
            or benchmark_result_form is BenchmarkResultForm.RAW_RUN_SERIES
        ):
            represented["seed"] = (
                observation.seed
                if isinstance(observation.seed, int)
                and not isinstance(observation.seed, bool)
                else None
            )
        metric_values.append(represented)
    metrics = tuple(sorted(metric_values, key=_canonical_json_bytes))
    if binding.kind in {ArtifactKind.RESULTS, ArtifactKind.BENCHMARK}:
        semantics = {"metric_observations": metrics}
        if benchmark_profile:
            semantics["profile_config"] = configs
            semantics["represented_metric_mismatches"] = tuple(
                sorted(represented_metric_mismatches)
            )
        version = (
            "claimci.projector.benchmark-evidence.v1"
            if benchmark_profile
            else "claimci.projector.training-results.v1"
        )
        return extraction, semantics, version
    if binding.kind in {ArtifactKind.CONFIG, ArtifactKind.DOCUMENT}:
        semantics = {"config_values": configs}
        version = (
            "claimci.projector.benchmark-config.v1"
            if benchmark_profile
            else "claimci.projector.training-config.v1"
        )
        return extraction, semantics, version
    raise AnalysisContractError("verification artifact kind is not consumed by Audit")


def _mapping_identity(
    plan: EphemeralAuditPlan,
    artifacts: tuple[VerificationArtifactIdentity, ...],
) -> VerificationMappingIdentity:
    selected = plan.selected_mapping
    if type(selected) is MappingCandidate:
        source_mapping_id = selected.mapping_id
        trust = selected.trust
        approval = None
    elif type(selected) is RepoMapping:
        source_mapping_id = selected.source_mapping_id
        trust = selected.trust
        approval = _digest(
            "claimci.verification-mapping.approval.v1",
            {
                "approved_by": selected.approved_by,
                "source_mapping_id": selected.source_mapping_id,
                "source_trust": selected.source_trust.value,
                "approval_source_id": selected.approval_provenance.source_id,
            },
        )
    else:
        raise TypeError("verification mapping requires a selected trusted mapping")
    binding_ids = tuple(item.verification_evidence_id for item in artifacts)
    instance = object.__new__(VerificationMappingIdentity)
    object.__setattr__(instance, "source_mapping_id", source_mapping_id)
    object.__setattr__(instance, "trust", trust)
    object.__setattr__(instance, "binding_evidence_ids", binding_ids)
    object.__setattr__(
        instance,
        "bindings_sha256",
        _digest("claimci.verification-mapping.bindings.v1", binding_ids),
    )
    object.__setattr__(instance, "approval_sha256", approval)
    return instance


@dataclass(frozen=True, slots=True, init=False)
class VerificationInputSnapshot:
    repository: RepositoryIdentity
    pr_number: int | None
    head_sha: GitCommitSha
    base_sha: GitCommitSha | None
    claim_identity: VerificationClaimIdentity | None
    mapping_identity: VerificationMappingIdentity | None
    artifacts: tuple[VerificationArtifactIdentity, ...]
    artifact_count: int
    artifacts_sha256: Sha256Digest | None
    profile_identity: EvidenceProfileIdentity | None
    obligation_identity: EvidenceObligationIdentity | None
    baseline_measurement: MeasurementProtocolReference | None
    candidate_measurement: MeasurementProtocolReference | None
    audit_compatibility: AuditSemanticsCompatibility | None
    captured_audit_input_sha256: Sha256Digest | None
    comparison_frame_sha256: Sha256Digest | None
    input_snapshot_sha256: Sha256Digest
    capability: VerificationSnapshotCapability
    reason_code: str | None
    version: str

    def __init__(self) -> None:
        raise TypeError("VerificationInputSnapshot must be created by exact-head materialization")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("VerificationInputSnapshot is final")

    @classmethod
    def unavailable(
        cls,
        *,
        repository: RepositoryIdentity,
        head_sha: GitCommitSha,
        pr_number: int | None = None,
        base_sha: GitCommitSha | None = None,
        reason_code: str = "snapshot_unavailable",
    ) -> "VerificationInputSnapshot":
        if type(repository) is not RepositoryIdentity:
            raise TypeError("unavailable snapshot requires RepositoryIdentity")
        if type(head_sha) is not GitCommitSha:
            raise TypeError("unavailable snapshot requires GitCommitSha")
        if pr_number is not None and (
            isinstance(pr_number, bool)
            or not isinstance(pr_number, int)
            or pr_number < 1
        ):
            raise VerificationContractError(
                "unavailable snapshot pr_number must be a positive integer"
            )
        if base_sha is not None and type(base_sha) is not GitCommitSha:
            raise TypeError("unavailable snapshot base_sha must be GitCommitSha")
        _identifier(reason_code, "snapshot unavailable reason")
        material = {
            "version": VERIFICATION_INPUT_SNAPSHOT_VERSION,
            "repository": repository.full_name,
            "pr_number": pr_number,
            "head_sha": str(head_sha),
            "base_sha": None if base_sha is None else str(base_sha),
            "capability": VerificationSnapshotCapability.UNAVAILABLE.value,
            "reason_code": reason_code,
        }
        return _snapshot_instance(
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
            base_sha=base_sha,
            claim_identity=None,
            mapping_identity=None,
            artifacts=(),
            artifact_count=0,
            artifacts_sha256=None,
            profile_identity=None,
            obligation_identity=None,
            baseline_measurement=None,
            candidate_measurement=None,
            audit_compatibility=None,
            captured_audit_input_sha256=None,
            comparison_frame_sha256=None,
            input_snapshot_sha256=_digest("claimci.verification-input-snapshot.v1", material),
            capability=VerificationSnapshotCapability.UNAVAILABLE,
            reason_code=reason_code,
        )


def _snapshot_instance(**values: object) -> VerificationInputSnapshot:
    instance = object.__new__(VerificationInputSnapshot)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    object.__setattr__(instance, "version", VERIFICATION_INPUT_SNAPSHOT_VERSION)
    return instance


def build_verification_input_snapshot(
    plan: EphemeralAuditPlan,
    runtime: object,
    bound: tuple[tuple[NormalizedEvidence, ArtifactBinding], ...],
    captured: Mapping[str, PassiveArtifact],
    measurement_pair: MeasurementProtocolPair,
    *,
    max_complete_bytes: int = VERIFICATION_INPUT_COMPLETE_MAX_BYTES,
) -> VerificationInputSnapshot:
    """Construct the trusted pre-Audit snapshot from freshly revalidated inputs."""

    from .materialize import RuntimeExecutionContext

    if type(plan) is not EphemeralAuditPlan:
        raise TypeError("verification snapshot requires EphemeralAuditPlan")
    if type(runtime) is not RuntimeExecutionContext:
        raise TypeError("verification snapshot requires RuntimeExecutionContext")
    if type(measurement_pair) is not MeasurementProtocolPair:
        raise TypeError("verification snapshot requires MeasurementProtocolPair")
    if (
        not isinstance(bound, tuple)
        or not bound
        or len(bound) > VERIFICATION_INPUT_MAX_ARTIFACTS
        or not all(
            type(evidence) is NormalizedEvidence and type(binding) is ArtifactBinding
            for evidence, binding in bound
        )
    ):
        raise VerificationContractError("verification bound evidence exceeds its limit")
    if not isinstance(captured, Mapping):
        raise TypeError("verification snapshot requires captured artifacts")
    if (
        isinstance(max_complete_bytes, bool)
        or not isinstance(max_complete_bytes, int)
        or max_complete_bytes < 1
        or max_complete_bytes > VERIFICATION_INPUT_COMPLETE_MAX_BYTES
    ):
        raise VerificationContractError("verification complete byte bound is invalid")
    if plan.repository != runtime.repository or plan.head_sha != runtime.head_sha:
        raise AnalysisContractError("verification runtime identity does not match the plan")
    if type(plan.audit_claim) is not AuditClaimSpec:
        raise AnalysisContractError("verification snapshot requires an executable Audit spec")

    benchmark = (
        plan.profiled_policy is not None
        and plan.profiled_policy.profile_id is EvidenceProfileId.BENCHMARK_MEASUREMENT_V0
    )
    benchmark_result_form = (
        None
        if not benchmark
        else plan.profiled_policy.profile_selection.result_form
    )
    artifacts: list[VerificationArtifactIdentity] = []
    frame_by_id: dict[str, Sha256Digest] = {}
    for evidence, binding in sorted(
        bound,
        key=lambda item: (
            str(item[1].path),
            item[1].role.value,
            "" if item[1].dataset_split is None else item[1].dataset_split.value,
            tuple(field_mapping_identity(value) for value in item[1].mappings),
        ),
    ):
        passive = captured.get(str(binding.path))
        if type(passive) is not PassiveArtifact:
            raise AnalysisContractError("verification snapshot is missing captured bytes")
        extraction, audit_semantics, projector = _artifact_semantic_projections(
            evidence,
            binding,
            passive,
            metric=plan.audit_claim.metric,
            benchmark_profile=benchmark,
            benchmark_result_form=benchmark_result_form,
        )
        artifact = _artifact_identity_from_materialized(
            evidence=evidence,
            binding=binding,
            passive=passive,
            extraction_projection=extraction,
            audit_semantics_projection=audit_semantics,
            evidence_projector_version=projector,
        )
        artifacts.append(artifact)
        frame_projection = (
            {"profile_config": audit_semantics.get("profile_config", ())}
            if isinstance(audit_semantics, Mapping)
            and binding.kind in {ArtifactKind.RESULTS, ArtifactKind.BENCHMARK}
            else audit_semantics
        )
        frame_by_id[artifact.verification_evidence_id] = _digest(
            "claimci.verification-artifact.comparison-frame.v1",
            frame_projection,
        )
    artifact_values = tuple(artifacts)
    claim = _claim_identity(plan)
    profile = _profile_identity(plan)
    obligation = _obligation_identity(plan)
    mapping = _mapping_identity(plan, artifact_values)
    baseline_measurement = MeasurementProtocolReference.from_identity(measurement_pair.baseline)
    candidate_measurement = MeasurementProtocolReference.from_identity(measurement_pair.candidate)
    adapter_versions = tuple(
        sorted(
            {
                f"{item.adapter_id}:{item.adapter_semantic_version or 'unversioned'}"
                for item in artifact_values
            }
        )
    )
    projector_versions = tuple(
        sorted({item.evidence_projector_version for item in artifact_values})
    )
    compatibility = AuditSemanticsCompatibility._from_fixed_versions(
        audit_policy_semantics=(
            "claimci.audit.benchmark.v0" if benchmark else "claimci.audit.training.v1"
        ),
        rule_set_semantics="claimci.audit-rules.v1",
        claim_compiler_semantics="claimci.claim-compiler.v0",
        evidence_projector_versions=projector_versions,
        adapter_semantic_versions=adapter_versions,
        evidence_profile_semantics=str(profile.profile_semantics_sha256),
        benchmark_compiler_semantics=(
            "claimci.benchmark-compiler.v0" if benchmark else None
        ),
    )
    binding_semantics = tuple(
        sorted(
            (
                {
                    "kind": item.kind.value,
                    "role": item.role.value,
                    "dataset_split": None
                    if item.dataset_split is None
                    else item.dataset_split.value,
                    "selectors": item.selectors,
                    "audit_semantics_sha256": str(item.audit_semantics_sha256),
                }
                for item in artifact_values
            ),
            key=_canonical_json_bytes,
        )
    )
    captured_digest = _digest(
        "claimci.verification-input.captured-audit.v1",
        {
            "audit_spec_sha256": str(claim.audit_spec_sha256),
            "profile_semantics_sha256": str(profile.profile_semantics_sha256),
            "obligation_semantics_sha256": str(obligation.obligation_semantics_sha256),
            "bindings": binding_semantics,
            "measurement_semantic_protocol_ids": {
                "baseline": baseline_measurement.semantic_protocol_id,
                "candidate": candidate_measurement.semantic_protocol_id,
            },
            "audit_compatibility": str(compatibility.audit_semantics_sha256),
        },
    )
    comparison_evidence = tuple(
        sorted(
            (
                {
                    "kind": item.kind.value,
                    "role": item.role.value,
                    "dataset_split": None
                    if item.dataset_split is None
                    else item.dataset_split.value,
                    "semantics_sha256": str(
                        item.audit_semantics_sha256
                        if item.role
                        in {ExperimentRole.BASELINE, ExperimentRole.REFERENCE}
                        else frame_by_id[item.verification_evidence_id]
                    ),
                }
                for item in artifact_values
                if item.role
                in {ExperimentRole.BASELINE, ExperimentRole.REFERENCE}
                or item.kind not in {ArtifactKind.RESULTS, ArtifactKind.BENCHMARK}
                or benchmark
            ),
            key=_canonical_json_bytes,
        )
    )
    comparison_digest = _digest(
        "claimci.verification-input.comparison-frame.v1",
        {
            "audit_spec_sha256": str(claim.audit_spec_sha256),
            "evidence": comparison_evidence,
            "profile_semantics_sha256": str(profile.profile_semantics_sha256),
            "measurement_semantic_protocol_ids": {
                "baseline": baseline_measurement.semantic_protocol_id,
                "candidate": candidate_measurement.semantic_protocol_id,
            },
            "audit_compatibility": str(compatibility.audit_semantics_sha256),
        },
    )
    artifact_root = _digest("claimci.verification-input.artifacts.v1", artifact_values)
    full_projection = {
        "version": VERIFICATION_INPUT_SNAPSHOT_VERSION,
        "repository": plan.repository.full_name,
        "pr_number": plan.pr_number,
        "head_sha": str(plan.head_sha),
        "base_sha": None if runtime.base_sha is None else str(runtime.base_sha),
        "claim_identity": claim,
        "mapping_identity": mapping,
        "artifacts": artifact_values,
        "profile_identity": profile,
        "obligation_identity": obligation,
        "baseline_measurement": baseline_measurement,
        "candidate_measurement": candidate_measurement,
        "audit_compatibility": compatibility,
        "captured_audit_input_sha256": str(captured_digest),
        "comparison_frame_sha256": str(comparison_digest),
    }
    input_digest = _digest("claimci.verification-input-snapshot.v1", full_projection)
    complete_bytes = len(_canonical_json_bytes(full_projection))
    complete = complete_bytes <= max_complete_bytes
    return _snapshot_instance(
        repository=plan.repository,
        pr_number=plan.pr_number,
        head_sha=plan.head_sha,
        base_sha=runtime.base_sha,
        claim_identity=claim,
        mapping_identity=mapping,
        artifacts=artifact_values if complete else (),
        artifact_count=len(artifact_values),
        artifacts_sha256=artifact_root,
        profile_identity=profile,
        obligation_identity=obligation,
        baseline_measurement=baseline_measurement,
        candidate_measurement=candidate_measurement,
        audit_compatibility=compatibility,
        captured_audit_input_sha256=captured_digest,
        comparison_frame_sha256=comparison_digest,
        input_snapshot_sha256=input_digest,
        capability=(
            VerificationSnapshotCapability.COMPLETE
            if complete
            else VerificationSnapshotCapability.IDENTITY_ONLY
        ),
        reason_code=None if complete else "snapshot_size_limit",
    )


def verification_snapshot_json_bytes(snapshot: VerificationInputSnapshot) -> bytes:
    """Serialize one snapshot without assigning a Hosted envelope budget."""

    if type(snapshot) is not VerificationInputSnapshot:
        raise TypeError("verification serialization requires VerificationInputSnapshot")
    return _canonical_json_bytes(snapshot)


__all__ = [
    "AUDIT_SEMANTICS_COMPATIBILITY_VERSION",
    "AuditSemanticsCompatibility",
    "EvidenceObligationIdentity",
    "EvidenceProfileIdentity",
    "MeasurementProtocolReference",
    "VERIFICATION_INPUT_COMPLETE_MAX_BYTES",
    "VERIFICATION_INPUT_SNAPSHOT_VERSION",
    "VerificationArtifactIdentity",
    "VerificationClaimIdentity",
    "VerificationContractError",
    "VerificationInputSnapshot",
    "VerificationMappingIdentity",
    "VerificationSelectorIdentity",
    "VerificationSnapshotCapability",
    "VerificationTablePredicateIdentity",
    "verification_snapshot_json_bytes",
]
