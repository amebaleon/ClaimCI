"""Fixed evidence profiles for deterministic ClaimCI planning."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .claim_types import (
    CanonicalScientificClaim,
    ClaimEvidencePolicy,
    MetricImprovementClaim,
    claim_evidence_policy,
)
from .contracts import (
    AnalysisContractError,
    ArtifactBinding,
    ArtifactKind,
    DatasetSplit,
    ExperimentRole,
    MappingCandidate,
    NormalizedEvidence,
    ProvenanceKind,
    RepoMapping,
)
from .evidence_identity import evidence_for_binding, field_mapping_projection


class EvidenceProfileId(str, Enum):
    TRAINING_EXPERIMENT_V0 = "TRAINING_EXPERIMENT_V0"
    BENCHMARK_MEASUREMENT_V0 = "BENCHMARK_MEASUREMENT_V0"


class BenchmarkVariant(str, Enum):
    LATENCY = "LATENCY"
    THROUGHPUT_RESOURCE = "THROUGHPUT_RESOURCE"
    KERNEL = "KERNEL"
    MODEL_QUALITY = "MODEL_QUALITY"
    COST = "COST"


class BenchmarkResultForm(str, Enum):
    RAW_RUN_SERIES = "RAW_RUN_SERIES"
    REPORTED_AGGREGATE = "REPORTED_AGGREGATE"
    DETERMINISTIC_DERIVATION = "DETERMINISTIC_DERIVATION"


class ProfileRoleMode(str, Enum):
    PAIRED_BASELINE_CANDIDATE = "PAIRED_BASELINE_CANDIDATE"
    SHARED_REFERENCE = "SHARED_REFERENCE"
    SHARED_REFERENCE_OR_PAIRED = "SHARED_REFERENCE_OR_PAIRED"


class ProfileSupportKind(str, Enum):
    ARTIFACT_BINDING = "ARTIFACT_BINDING"
    NORMALIZED_METRIC = "NORMALIZED_METRIC"
    STRUCTURED_COMPONENT = "STRUCTURED_COMPONENT"
    DETERMINISTIC_DERIVATION = "DETERMINISTIC_DERIVATION"


class ProfileSelectionState(str, Enum):
    SELECTED = "selected"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


BENCHMARK_PROFILE_SLOTS = (
    "benchmark.subject",
    "benchmark.results",
    "benchmark.workload",
    "benchmark.measurement_config",
    "measurement.metric_identity",
    "benchmark.run_protocol",
    "benchmark.environment",
    "benchmark.evaluator",
)

_MAX_SELECTION_FACTS = 192
_MAX_EXECUTABLE_BINDINGS = 16


_BENCHMARK_PROFILE_FIELDS = frozenset(
    {
        "benchmark.subject.id",
        "benchmark.subject.implementation",
        "benchmark.workload.id",
        "benchmark.workload.model",
        "benchmark.workload.input_shape",
        "benchmark.workload.corpus",
        "benchmark.workload.sequence_length",
        "benchmark.workload.batch_size",
        "benchmark.workload.concurrency",
        "benchmark.workload.tensor_shape",
        "benchmark.workload.dtype",
        "benchmark.workload.layout",
        "benchmark.workload.tensor_generation",
        "benchmark.workload.workload_volume",
        "benchmark.workload.token_volume",
        "benchmark.measurement_config.timing_boundary",
        "benchmark.measurement_config.measurement_window",
        "benchmark.measurement_config.memory_collection",
        "benchmark.measurement_config.network_boundary",
        "benchmark.run_protocol.result_form",
        "benchmark.run_protocol.sample_count",
        "benchmark.run_protocol.step_count",
        "benchmark.run_protocol.warmup",
        "benchmark.run_protocol.retry_policy",
        "benchmark.run_protocol.aggregation",
        "benchmark.run_protocol.statistic",
        "benchmark.run_protocol.timing_method",
        "benchmark.run_protocol.generation_seed",
        "benchmark.environment.hardware",
        "benchmark.environment.software",
        "benchmark.environment.runtime",
        "benchmark.environment.compiler",
        "benchmark.environment.region",
        "benchmark.environment.provider",
        "benchmark.environment.sku",
        "benchmark.environment.pricing_snapshot",
        "benchmark.environment.pricing_version",
        "benchmark.evaluator.id",
        "benchmark.evaluator.version",
        "benchmark.evaluator.scoring",
        "benchmark.evaluator.metric_definition",
        "benchmark.evaluator.metric_config",
        "benchmark.evaluator.normalization",
        "benchmark.evaluator.postprocessing",
        "benchmark.evaluator.formula_family",
        "benchmark.evaluator.unit_value",
        "benchmark.evaluator.quantity",
        "benchmark.evaluator.included_costs",
        "benchmark.evaluator.excluded_costs",
        "measurement.metric_identity.definition",
        "measurement.metric_identity.configuration",
        "measurement.metric_identity.version",
    }
)


def _canonical_id(value: object, label: str, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AnalysisContractError(f"{label} must be a canonical identifier")
    return value


def _digest(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True, slots=True, init=False)
class EvidenceProfileDefinition:
    profile_id: EvidenceProfileId
    version: int
    slot_ids: tuple[str, ...]
    benchmark_variants: tuple[BenchmarkVariant, ...]

    def __init__(self) -> None:
        raise TypeError("EvidenceProfileDefinition must come from the fixed registry")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EvidenceProfileDefinition is final")


def _definition(
    profile_id: EvidenceProfileId,
    *,
    slot_ids: tuple[str, ...],
    variants: tuple[BenchmarkVariant, ...] = (),
) -> EvidenceProfileDefinition:
    instance = object.__new__(EvidenceProfileDefinition)
    object.__setattr__(instance, "profile_id", profile_id)
    object.__setattr__(instance, "version", 0)
    object.__setattr__(instance, "slot_ids", slot_ids)
    object.__setattr__(instance, "benchmark_variants", variants)
    return instance


_PROFILE_REGISTRY = MappingProxyType(
    {
        EvidenceProfileId.TRAINING_EXPERIMENT_V0: _definition(
            EvidenceProfileId.TRAINING_EXPERIMENT_V0,
            slot_ids=(
                "artifact.baseline.results.metric",
                "artifact.candidate.results.metric",
                "artifact.baseline.config",
                "artifact.candidate.config",
                "artifact.baseline.dataset.train",
                "artifact.baseline.dataset.eval",
                "artifact.candidate.dataset.train",
                "artifact.candidate.dataset.eval",
            ),
        ),
        EvidenceProfileId.BENCHMARK_MEASUREMENT_V0: _definition(
            EvidenceProfileId.BENCHMARK_MEASUREMENT_V0,
            slot_ids=BENCHMARK_PROFILE_SLOTS,
            variants=tuple(BenchmarkVariant),
        ),
    }
)


def evidence_profile_registry() -> Mapping[EvidenceProfileId, EvidenceProfileDefinition]:
    return _PROFILE_REGISTRY


@dataclass(frozen=True, slots=True, init=False)
class ProfileSelectionFact:
    fact_id: str
    name: str
    value: str
    source_binding_ids: tuple[str, ...]
    provenance_kind: ProvenanceKind

    def __init__(self) -> None:
        raise TypeError("ProfileSelectionFact must be recovered by fixed policy")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ProfileSelectionFact is final")


def _selection_fact(
    name: str,
    value: object,
    source_binding_ids: tuple[str, ...],
) -> ProfileSelectionFact:
    _canonical_id(name, "profile fact name")
    if not source_binding_ids or len(source_binding_ids) > 16:
        raise AnalysisContractError("profile fact source bindings exceed their bound")
    try:
        canonical_value = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise AnalysisContractError("profile fact value is not bounded JSON") from error
    if len(canonical_value) > 1_024:
        raise AnalysisContractError("profile fact value exceeds its bound")
    material = {
        "name": name,
        "value": canonical_value,
        "sources": sorted(source_binding_ids),
    }
    instance = object.__new__(ProfileSelectionFact)
    object.__setattr__(instance, "fact_id", _digest("profile-fact-", material))
    object.__setattr__(instance, "name", name)
    object.__setattr__(instance, "value", canonical_value)
    object.__setattr__(instance, "source_binding_ids", tuple(sorted(source_binding_ids)))
    object.__setattr__(instance, "provenance_kind", ProvenanceKind.ADAPTER_EXTRACTION)
    return instance


@dataclass(frozen=True, slots=True, init=False)
class EvidenceProfileSelection:
    selection_id: str
    profile_id: EvidenceProfileId
    profile_version: int
    benchmark_variant: BenchmarkVariant | None
    result_form: BenchmarkResultForm | None
    activation_policy_id: str
    complete: bool
    facts: tuple[ProfileSelectionFact, ...]

    def __init__(self) -> None:
        raise TypeError("EvidenceProfileSelection must be created by deterministic policy")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EvidenceProfileSelection is final")


def _bounded_facts(
    facts: tuple[ProfileSelectionFact, ...],
) -> tuple[ProfileSelectionFact, ...]:
    if not isinstance(facts, tuple) or not all(
        type(item) is ProfileSelectionFact for item in facts
    ):
        raise AnalysisContractError("profile facts must be exact recovered records")
    ordered = tuple(
        sorted(
            {item.fact_id: item for item in facts}.values(),
            key=lambda item: item.fact_id,
        )
    )
    if len(ordered) > _MAX_SELECTION_FACTS:
        raise AnalysisContractError("profile facts exceed their bound")
    return ordered


def _selection(
    profile_id: EvidenceProfileId,
    *,
    variant: BenchmarkVariant | None,
    result_form: BenchmarkResultForm | None,
    complete: bool,
    facts: tuple[ProfileSelectionFact, ...],
) -> EvidenceProfileSelection:
    ordered = _bounded_facts(facts)
    if profile_id is EvidenceProfileId.TRAINING_EXPERIMENT_V0:
        if variant is not None or result_form is not None:
            raise AnalysisContractError("Training profile cannot carry Benchmark fields")
        activation = "claimci.profile.training-experiment.v0"
    else:
        if variant is None:
            raise AnalysisContractError("Benchmark profile requires a variant")
        activation = f"claimci.profile.benchmark-measurement.{variant.value.casefold()}.v0"
    material = {
        "profile_id": profile_id.value,
        "version": 0,
        "variant": None if variant is None else variant.value,
        "result_form": None if result_form is None else result_form.value,
        "activation": activation,
        "complete": complete,
        "facts": [item.fact_id for item in ordered],
    }
    instance = object.__new__(EvidenceProfileSelection)
    object.__setattr__(instance, "selection_id", _digest("profile-selection-", material))
    object.__setattr__(instance, "profile_id", profile_id)
    object.__setattr__(instance, "profile_version", 0)
    object.__setattr__(instance, "benchmark_variant", variant)
    object.__setattr__(instance, "result_form", result_form)
    object.__setattr__(instance, "activation_policy_id", activation)
    object.__setattr__(instance, "complete", complete)
    object.__setattr__(instance, "facts", ordered)
    return instance


@dataclass(frozen=True, slots=True, init=False)
class ProfileSelectionOutcome:
    state: ProfileSelectionState
    selection: EvidenceProfileSelection | None
    reason: str | None
    facts: tuple[ProfileSelectionFact, ...]

    def __init__(self) -> None:
        raise TypeError("ProfileSelectionOutcome must be created by deterministic policy")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ProfileSelectionOutcome is final")


def _outcome(
    state: ProfileSelectionState,
    *,
    selection: EvidenceProfileSelection | None,
    reason: str | None,
    facts: tuple[ProfileSelectionFact, ...],
) -> ProfileSelectionOutcome:
    ordered = _bounded_facts(facts)
    if (state is ProfileSelectionState.SELECTED) != (selection is not None):
        raise AnalysisContractError("profile outcome state is inconsistent")
    if (state is ProfileSelectionState.SELECTED) == (reason is not None):
        raise AnalysisContractError("profile outcome reason is inconsistent")
    instance = object.__new__(ProfileSelectionOutcome)
    object.__setattr__(instance, "state", state)
    object.__setattr__(instance, "selection", selection)
    object.__setattr__(instance, "reason", reason)
    object.__setattr__(instance, "facts", ordered)
    return instance


@dataclass(frozen=True, slots=True, init=False)
class ProfiledEvidencePolicy:
    policy_id: str
    profile_id: EvidenceProfileId
    claim_policy: ClaimEvidencePolicy
    profile_selection: EvidenceProfileSelection
    claim_template_ids: tuple[str, ...]
    profile_template_ids: tuple[str, ...]
    comparison_template_id: str

    def __init__(self) -> None:
        raise TypeError("ProfiledEvidencePolicy must be created by fixed policy")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ProfiledEvidencePolicy is final")


def profiled_evidence_policy(
    claim: CanonicalScientificClaim,
    selection: EvidenceProfileSelection,
) -> ProfiledEvidencePolicy:
    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("profiled policy requires CanonicalScientificClaim")
    if type(selection) is not EvidenceProfileSelection:
        raise TypeError("profiled policy requires EvidenceProfileSelection")
    policy = claim_evidence_policy(claim)
    claim_templates = tuple(
        item for item in policy.obligation_template_ids if item.startswith("claim.")
    )
    if selection.profile_id is EvidenceProfileId.TRAINING_EXPERIMENT_V0:
        profile_templates = tuple(
            item
            for item in policy.obligation_template_ids
            if item.startswith("artifact.") or item.startswith("measurement.")
        )
        policy_id = policy.policy_id
    else:
        active = _active_benchmark_slots(selection)
        profile_templates = active
        policy_id = (
            f"{policy.policy_id}+{selection.profile_id.value.casefold()}+"
            f"{selection.activation_policy_id}"
        )
    instance = object.__new__(ProfiledEvidencePolicy)
    object.__setattr__(instance, "policy_id", policy_id)
    object.__setattr__(instance, "profile_id", selection.profile_id)
    object.__setattr__(instance, "claim_policy", policy)
    object.__setattr__(instance, "profile_selection", selection)
    object.__setattr__(instance, "claim_template_ids", claim_templates)
    object.__setattr__(instance, "profile_template_ids", profile_templates)
    object.__setattr__(instance, "comparison_template_id", "comparison.readiness")
    return instance


def _active_benchmark_slots(selection: EvidenceProfileSelection) -> tuple[str, ...]:
    variant = selection.benchmark_variant
    assert variant is not None
    slots = list(BENCHMARK_PROFILE_SLOTS[:6])
    if variant in {
        BenchmarkVariant.LATENCY,
        BenchmarkVariant.THROUGHPUT_RESOURCE,
        BenchmarkVariant.KERNEL,
        BenchmarkVariant.COST,
    }:
        slots.append("benchmark.environment")
    if variant in {BenchmarkVariant.MODEL_QUALITY, BenchmarkVariant.COST}:
        slots.append("benchmark.evaluator")
    return tuple(slots)


_LATENCY = frozenset({"latency", "runtime", "wall clock", "wall-clock", "cold start", "cold-start"})
_THROUGHPUT = frozenset(
    {
        "throughput",
        "tokens per second",
        "tokens/s",
        "samples per second",
        "samples/s",
        "peak memory",
        "peak vram",
        "vram",
        "memory",
        "resource",
    }
)
_KERNEL = frozenset({"kernel time", "kernel latency", "bandwidth"})
_COST = frozenset({"cost", "cost per request", "cost/request", "cost per token", "cost/token"})
_QUALITY = frozenset({"accuracy", "f1", "bleu", "perplexity", "loss"})

_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "benchmark.workload.sequence_length",
        "benchmark.workload.batch_size",
        "benchmark.workload.concurrency",
        "benchmark.workload.workload_volume",
        "benchmark.workload.token_volume",
        "benchmark.run_protocol.sample_count",
        "benchmark.run_protocol.step_count",
    }
)
_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "benchmark.run_protocol.warmup",
        "benchmark.run_protocol.generation_seed",
    }
)
_DECIMAL_FIELDS = frozenset(
    {
        "benchmark.evaluator.unit_value",
        "benchmark.evaluator.quantity",
    }
)


def _canonical_profile_scalar(key: str, value: object) -> object:
    """Recover fixed scalar semantics without guessing arbitrary table types."""

    if key in _POSITIVE_INTEGER_FIELDS | _NONNEGATIVE_INTEGER_FIELDS:
        if isinstance(value, str):
            if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
                raise AnalysisContractError(
                    f"Benchmark profile field {key!r} must be a canonical integer"
                )
            try:
                value = int(value)
            except ValueError as error:
                raise AnalysisContractError(
                    f"Benchmark profile field {key!r} exceeds its integer bound"
                ) from error
        minimum = 1 if key in _POSITIVE_INTEGER_FIELDS else 0
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > 2**63 - 1
        ):
            raise AnalysisContractError(
                f"Benchmark profile field {key!r} is outside its integer bound"
            )
        return value
    if key in _DECIMAL_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise AnalysisContractError(
                f"Benchmark profile field {key!r} must be a finite decimal"
            )
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise AnalysisContractError(
                f"Benchmark profile field {key!r} must be a finite decimal"
            ) from error
        if not number.is_finite():
            raise AnalysisContractError(
                f"Benchmark profile field {key!r} must be a finite decimal"
            )
        canonical = number.normalize()
        return "0" if canonical == 0 else str(canonical)
    return value


def _normalized_metric(metric: str) -> str:
    value = re.sub(r"[_-]+", " ", metric.casefold())
    return " ".join(value.split())


def _metric_variant(metric: str, source_text: str = "") -> BenchmarkVariant | None:
    value = _normalized_metric(metric)
    source = _normalized_metric(source_text)
    if re.search(r"\bkernel (?:time|latency|bandwidth)\b", source):
        return BenchmarkVariant.KERNEL
    if re.search(r"\bcost(?: per|/)(?: request| token)\b", source):
        return BenchmarkVariant.COST
    if value in _LATENCY:
        return BenchmarkVariant.LATENCY
    if value in _THROUGHPUT:
        return BenchmarkVariant.THROUGHPUT_RESOURCE
    if value in _KERNEL or ("kernel" in value and "time" in value):
        return BenchmarkVariant.KERNEL
    if value in _COST or value.startswith("cost "):
        return BenchmarkVariant.COST
    return BenchmarkVariant.MODEL_QUALITY if value in _QUALITY else None


def _binding_id(binding: ArtifactBinding, evidence: NormalizedEvidence) -> str:
    return _digest(
        "profile-binding-",
        {
            "path": str(binding.path),
            "kind": binding.kind.value,
            "role": binding.role.value,
            "split": None if binding.dataset_split is None else binding.dataset_split.value,
            "sha256": str(evidence.artifact.sha256),
            "adapter": binding.adapter_id,
            "mappings": field_mapping_projection(binding.mappings),
        },
    )


def _mapping_evidence(
    mapping: MappingCandidate | RepoMapping,
    normalized_evidence: tuple[NormalizedEvidence, ...],
) -> tuple[tuple[ArtifactBinding, NormalizedEvidence], ...]:
    values: list[tuple[ArtifactBinding, NormalizedEvidence]] = []
    for binding in mapping.bindings:
        evidence = evidence_for_binding(binding, normalized_evidence)
        if evidence is not None and all(
            item.kind is ProvenanceKind.ADAPTER_EXTRACTION
            for item in evidence.adapter_match.match_evidence
        ):
            values.append((binding, evidence))
    return tuple(values)


def _mapping_is_profile_trusted(
    mapping: MappingCandidate | RepoMapping,
) -> bool:
    if type(mapping) is RepoMapping:
        return True
    provenance = (
        mapping.provenance,
        *(binding.provenance for binding in mapping.bindings),
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
    return not any(
        item.kind is ProvenanceKind.PROVIDER_PROPOSAL for item in provenance
    )


def _training_complete(
    mapping: MappingCandidate | RepoMapping,
    normalized_evidence: tuple[NormalizedEvidence, ...],
) -> bool:
    for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
        if not any(item.role is role and item.kind is ArtifactKind.RESULTS for item in mapping.bindings):
            return False
        if not any(item.role is role and item.kind is ArtifactKind.CONFIG for item in mapping.bindings):
            return False
        for split in (DatasetSplit.TRAIN, DatasetSplit.EVAL):
            if sum(
                item.role is role
                and item.kind is ArtifactKind.DATASET
                and item.dataset_split is split
                for item in mapping.bindings
            ) != 1:
                return False
    for binding in mapping.bindings:
        evidence = evidence_for_binding(binding, normalized_evidence)
        if evidence is None or any(
            item.kind is not ProvenanceKind.ADAPTER_EXTRACTION
            for item in evidence.adapter_match.match_evidence
        ):
            return False
        represented_roles = {
            observation.experiment_role
            for observation in evidence.observations
            if observation.experiment_role is not ExperimentRole.UNSPECIFIED
        }
        if represented_roles and represented_roles != {binding.role}:
            return False
    return True


def _training_shape_present(mapping: MappingCandidate | RepoMapping) -> bool:
    return all(
        any(
            binding.role is role and binding.kind is kind
            for binding in mapping.bindings
        )
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
        for kind in (ArtifactKind.RESULTS, ArtifactKind.CONFIG)
    )


def _config_values(
    pairs: tuple[tuple[ArtifactBinding, NormalizedEvidence], ...],
) -> dict[ExperimentRole, dict[str, tuple[object, str]]]:
    result: dict[ExperimentRole, dict[str, tuple[object, str]]] = {}
    for binding, evidence in pairs:
        if binding.role not in {
            ExperimentRole.BASELINE,
            ExperimentRole.CANDIDATE,
            ExperimentRole.REFERENCE,
        }:
            continue
        binding_id = _binding_id(binding, evidence)
        values = result.setdefault(binding.role, {})
        selected_keys = {
            value
            for mapping in binding.mappings
            for value in (mapping.target_field, mapping.selector.expression)
        }
        for observation in evidence.observations:
            if (
                observation.provenance.kind
                is not ProvenanceKind.ADAPTER_EXTRACTION
            ):
                continue
            for item in observation.config_values:
                if item.provenance.kind is not ProvenanceKind.ADAPTER_EXTRACTION:
                    continue
                if item.key not in _BENCHMARK_PROFILE_FIELDS:
                    continue
                if item.key not in selected_keys:
                    continue
                recovered = _canonical_profile_scalar(item.key, item.value)
                existing = values.get(item.key)
                if existing is not None and json.dumps(
                    existing[0],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ) != json.dumps(
                    recovered,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ):
                    raise AnalysisContractError(
                        f"Benchmark profile field {item.key!r} conflicts within one role"
                    )
                values.setdefault(item.key, (recovered, binding_id))
    return result


def _metrics(
    pairs: tuple[tuple[ArtifactBinding, NormalizedEvidence], ...],
) -> dict[ExperimentRole, tuple[tuple[str, float, str], ...]]:
    result: dict[ExperimentRole, list[tuple[str, float, str]]] = {}
    for binding, evidence in pairs:
        if binding.role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}:
            continue
        binding_id = _binding_id(binding, evidence)
        if not any(
            mapping.target_field == "metric_value"
            for mapping in binding.mappings
        ):
            continue
        for observation in evidence.observations:
            if (
                observation.provenance.kind
                is not ProvenanceKind.ADAPTER_EXTRACTION
            ):
                continue
            if observation.metric_name is None or observation.metric_value is None:
                continue
            result.setdefault(binding.role, []).append(
                (observation.metric_name, observation.metric_value, binding_id)
            )
    return {role: tuple(values) for role, values in result.items()}


def _component_present(
    values: dict[ExperimentRole, dict[str, tuple[object, str]]],
    prefix: str,
) -> bool:
    reference = any(key.startswith(prefix + ".") for key in values.get(ExperimentRole.REFERENCE, {}))
    paired = all(
        any(key.startswith(prefix + ".") for key in values.get(role, {}))
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
    )
    return reference or paired


def _profile_key_present(
    values: dict[ExperimentRole, dict[str, tuple[object, str]]],
    key: str,
) -> bool:
    if key in values.get(ExperimentRole.REFERENCE, {}):
        return True
    return all(
        key in values.get(role, {})
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
    )


def _one_profile_key_present(
    values: dict[ExperimentRole, dict[str, tuple[object, str]]],
    keys: tuple[str, ...],
) -> bool:
    return any(_profile_key_present(values, key) for key in keys)


def _variant_components_complete(
    values: dict[ExperimentRole, dict[str, tuple[object, str]]],
    variant: BenchmarkVariant,
    result_form: BenchmarkResultForm | None,
) -> bool:
    required: tuple[str, ...]
    alternatives: tuple[tuple[str, ...], ...] = ()
    if variant is BenchmarkVariant.LATENCY:
        required = (
            "benchmark.workload.id",
            "benchmark.measurement_config.timing_boundary",
            "benchmark.run_protocol.warmup",
            "benchmark.run_protocol.sample_count",
            "benchmark.run_protocol.aggregation",
            "benchmark.environment.hardware",
        )
    elif variant is BenchmarkVariant.THROUGHPUT_RESOURCE:
        required = (
            "benchmark.workload.id",
            "benchmark.run_protocol.aggregation",
            "benchmark.environment.hardware",
            "benchmark.environment.runtime",
        )
        alternatives = (
            (
                "benchmark.workload.batch_size",
                "benchmark.workload.concurrency",
                "benchmark.workload.sequence_length",
            ),
            (
                "benchmark.run_protocol.sample_count",
                "benchmark.run_protocol.step_count",
            ),
            (
                "benchmark.measurement_config.timing_boundary",
                "benchmark.measurement_config.memory_collection",
            ),
        )
    elif variant is BenchmarkVariant.KERNEL:
        required = (
            "benchmark.workload.tensor_shape",
            "benchmark.workload.dtype",
            "benchmark.workload.layout",
            "benchmark.workload.tensor_generation",
            "benchmark.run_protocol.generation_seed",
            "benchmark.run_protocol.warmup",
            "benchmark.run_protocol.timing_method",
            "benchmark.environment.compiler",
            "benchmark.environment.runtime",
            "benchmark.environment.hardware",
        )
    elif variant is BenchmarkVariant.MODEL_QUALITY:
        required = (
            "benchmark.evaluator.id",
            "benchmark.evaluator.scoring",
        )
        alternatives = (
            ("benchmark.workload.corpus", "benchmark.workload.id"),
            (
                "measurement.metric_identity.definition",
                "measurement.metric_identity.configuration",
                "benchmark.evaluator.metric_definition",
                "benchmark.evaluator.metric_config",
            ),
        )
    else:
        required = (
            "benchmark.environment.provider",
            "benchmark.environment.sku",
            "benchmark.measurement_config.measurement_window",
            "benchmark.evaluator.included_costs",
            "benchmark.evaluator.excluded_costs",
        )
        alternatives = (
            (
                "benchmark.workload.workload_volume",
                "benchmark.workload.token_volume",
            ),
            (
                "benchmark.environment.pricing_snapshot",
                "benchmark.environment.pricing_version",
            ),
        )
        if result_form is BenchmarkResultForm.DETERMINISTIC_DERIVATION:
            required = (*required, "benchmark.evaluator.formula_family")
        else:
            required = (*required, "benchmark.run_protocol.aggregation")
    return all(_profile_key_present(values, key) for key in required) and all(
        _one_profile_key_present(values, choices) for choices in alternatives
    )


def _single_result_form(
    values: dict[ExperimentRole, dict[str, tuple[object, str]]],
) -> BenchmarkResultForm | None:
    represented = {
        value
        for role_values in values.values()
        for key, (value, _binding_id_value) in role_values.items()
        if key == "benchmark.run_protocol.result_form" and isinstance(value, str)
    }
    if not represented:
        return BenchmarkResultForm.RAW_RUN_SERIES
    if len(represented) != 1:
        return None
    try:
        return BenchmarkResultForm(next(iter(represented)))
    except ValueError:
        return None


def _benchmark_facts(
    mapping: MappingCandidate | RepoMapping,
    normalized_evidence: tuple[NormalizedEvidence, ...],
    *,
    metric: str,
    variant: BenchmarkVariant,
) -> tuple[tuple[ProfileSelectionFact, ...], BenchmarkResultForm | None, bool, bool]:
    pairs = _mapping_evidence(mapping, normalized_evidence)
    configs = _config_values(pairs)
    metrics = _metrics(pairs)
    facts: list[ProfileSelectionFact] = []
    for role, role_values in sorted(configs.items(), key=lambda item: item[0].value):
        for key, (value, binding_id) in sorted(role_values.items()):
            if key.startswith("benchmark.") or key.startswith("measurement.metric_identity."):
                facts.append(_selection_fact(f"{role.value}.{key}", value, (binding_id,)))
    for role, observations in sorted(metrics.items(), key=lambda item: item[0].value):
        identities = tuple(
            sorted(
                {
                    (metric_name, binding_id)
                    for metric_name, _value, binding_id in observations
                }
            )
        )
        for metric_name, binding_id in identities:
            facts.append(
                _selection_fact(
                    f"{role.value}.measurement.metric_identity.name",
                    metric_name,
                    (binding_id,),
                )
            )

    positive = any(
        fact.name.split(".", 1)[1].startswith(
            ("benchmark.workload.", "benchmark.measurement_config.", "benchmark.run_protocol.", "benchmark.environment.", "benchmark.evaluator.")
        )
        for fact in facts
    )
    result_form = _single_result_form(configs)
    metric_recovered = all(
        role in metrics
        and metrics[role]
        and len({_normalized_metric(item[0]) for item in metrics[role]}) == 1
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
    )
    common = all(
        _component_present(configs, prefix)
        for prefix in (
            "benchmark.subject",
            "benchmark.workload",
            "benchmark.measurement_config",
            "benchmark.run_protocol",
        )
    ) and metric_recovered
    needs_environment = variant in {
        BenchmarkVariant.LATENCY,
        BenchmarkVariant.THROUGHPUT_RESOURCE,
        BenchmarkVariant.KERNEL,
        BenchmarkVariant.COST,
    }
    needs_evaluator = variant in {
        BenchmarkVariant.MODEL_QUALITY,
        BenchmarkVariant.COST,
    }
    complete = (
        common
        and result_form is not None
        and (not needs_environment or _component_present(configs, "benchmark.environment"))
        and (not needs_evaluator or _component_present(configs, "benchmark.evaluator"))
        and _variant_components_complete(configs, variant, result_form)
    )
    def paired_value(key: str) -> tuple[object, object] | None:
        baseline = configs.get(ExperimentRole.BASELINE, {}).get(key)
        candidate = configs.get(ExperimentRole.CANDIDATE, {}).get(key)
        if baseline is None or candidate is None:
            return None
        return baseline[0], candidate[0]

    sample_counts = paired_value("benchmark.run_protocol.sample_count")
    aggregations = paired_value("benchmark.run_protocol.aggregation")
    if result_form is BenchmarkResultForm.RAW_RUN_SERIES:
        if sample_counts is None or aggregations is None:
            complete = False
        else:
            counts: list[int] = []
            for item in sample_counts:
                try:
                    count = int(item) if isinstance(item, str) else item
                except (TypeError, ValueError):
                    complete = False
                    break
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 1
                    or count > 2**63 - 1
                ):
                    complete = False
                    break
                counts.append(count)
            if len(counts) == 2 and any(
                counts[index] != len(metrics.get(role, ()))
                for index, role in enumerate(
                    (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
                )
            ):
                complete = False
            if any(item != "arithmetic_mean_v1" for item in aggregations):
                complete = False
    if result_form is BenchmarkResultForm.REPORTED_AGGREGATE:
        complete = complete and all(
            key in configs.get(role, {})
            for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
            for key in (
                "benchmark.run_protocol.sample_count",
                "benchmark.run_protocol.aggregation",
                "benchmark.run_protocol.statistic",
            )
        )
        complete = complete and all(
            len(metrics.get(role, ())) == 1
            for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
        )
    if result_form is BenchmarkResultForm.DETERMINISTIC_DERIVATION:
        complete = complete and all(
            key in configs.get(role, {})
            for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE)
            for key in (
                "benchmark.evaluator.formula_family",
                "benchmark.evaluator.unit_value",
                "benchmark.evaluator.quantity",
            )
        )
        formula = paired_value("benchmark.evaluator.formula_family")
        complete = complete and formula == ("multiply_v1", "multiply_v1")
        for role in (ExperimentRole.BASELINE, ExperimentRole.CANDIDATE):
            for key in (
                "benchmark.evaluator.unit_value",
                "benchmark.evaluator.quantity",
            ):
                represented = configs.get(role, {}).get(key)
                if represented is None or isinstance(represented[0], bool):
                    complete = False
                    continue
                try:
                    number = float(represented[0])
                except (TypeError, ValueError, OverflowError):
                    complete = False
                else:
                    if not (number == number and abs(number) != float("inf")):
                        complete = False
    return tuple(facts), result_form, positive, complete


def select_evidence_profile(
    claim: CanonicalScientificClaim,
    *,
    normalized_evidence: tuple[NormalizedEvidence, ...],
    mapping_candidates: tuple[MappingCandidate, ...],
    approved_mapping: RepoMapping | None = None,
) -> ProfileSelectionOutcome:
    """Select a profile from fixed claim semantics and recovered passive facts."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("profile selection requires CanonicalScientificClaim")
    if type(claim.primary) is not MetricImprovementClaim:
        raise AnalysisContractError("profile selection requires MetricImprovementClaim")
    if not isinstance(normalized_evidence, tuple) or not all(
        type(item) is NormalizedEvidence for item in normalized_evidence
    ):
        raise TypeError("profile selection evidence is invalid")
    if not isinstance(mapping_candidates, tuple) or not all(
        type(item) is MappingCandidate for item in mapping_candidates
    ):
        raise TypeError("profile selection mappings are invalid")
    if approved_mapping is not None and type(approved_mapping) is not RepoMapping:
        raise TypeError("profile selection approval is invalid")

    variant = _metric_variant(claim.primary.metric, claim.reference.text)
    mappings: tuple[MappingCandidate | RepoMapping, ...] = (
        (approved_mapping,) if approved_mapping is not None else mapping_candidates
    )
    records: list[
        tuple[
            tuple[
                tuple[ProfileSelectionFact, ...],
                BenchmarkResultForm | None,
                bool,
                bool,
            ],
            bool,
            bool,
            bool,
        ]
    ] = []
    for mapping in mappings:
        training = _training_complete(
            mapping,
            normalized_evidence,
        )
        trusted = _mapping_is_profile_trusted(mapping)
        recovered: tuple[
            tuple[ProfileSelectionFact, ...],
            BenchmarkResultForm | None,
            bool,
            bool,
        ] = ((), None, False, False)
        if variant is not None:
            try:
                recovered = _benchmark_facts(
                    mapping,
                    normalized_evidence,
                    metric=claim.primary.metric,
                    variant=variant,
                )
            except AnalysisContractError:
                recovered = ((), None, False, False)
        records.append(
            (recovered, training, _training_shape_present(mapping), trusted)
        )
    has_trusted_complete = any(
        trusted and (training or recovered[3])
        for recovered, training, _training_positive, trusted in records
    )
    effective = tuple(
        item
        for item in records
        if not has_trusted_complete or item[3]
    )
    training_complete = any(
        training for _recovered, training, _training_positive, _trusted in effective
    )
    training_positive = any(
        positive for _recovered, _training, positive, _trusted in effective
    )
    benchmark = tuple(
        recovered
        for recovered, _training, _training_positive, _trusted in effective
    )
    benchmark_positive = any(item[2] for item in benchmark)
    complete_benchmark = tuple(item for item in benchmark if item[3])
    benchmark_complete = bool(complete_benchmark)
    complete_result_forms = {
        item[1] for item in complete_benchmark if item[1] is not None
    }
    representative = next(
        (item for item in benchmark if item[3]),
        next((item for item in benchmark if item[2]), None),
    )
    facts = () if representative is None else representative[0]

    quality_or_unknown = variant in {None, BenchmarkVariant.MODEL_QUALITY}
    if quality_or_unknown and training_complete and benchmark_complete:
        return _outcome(
            ProfileSelectionState.AMBIGUOUS,
            selection=None,
            reason="evidence_profile_ambiguous",
            facts=facts,
        )
    if quality_or_unknown and training_complete and not benchmark_complete:
        selection = _selection(
            EvidenceProfileId.TRAINING_EXPERIMENT_V0,
            variant=None,
            result_form=None,
            complete=True,
            facts=(),
        )
        return _outcome(
            ProfileSelectionState.SELECTED,
            selection=selection,
            reason=None,
            facts=(),
        )
    if variant is not None and (not quality_or_unknown or benchmark_positive):
        if len(complete_result_forms) > 1:
            selection = _selection(
                EvidenceProfileId.BENCHMARK_MEASUREMENT_V0,
                variant=variant,
                result_form=None,
                complete=False,
                facts=(),
            )
            return _outcome(
                ProfileSelectionState.SELECTED,
                selection=selection,
                reason=None,
                facts=(),
            )
        selected = next((item for item in benchmark if item[3]), None)
        if selected is None:
            selected = next((item for item in benchmark if item[2]), None)
        result_form = None if selected is None else selected[1]
        selected_facts = facts if selected is None else selected[0]
        selection = _selection(
            EvidenceProfileId.BENCHMARK_MEASUREMENT_V0,
            variant=variant,
            result_form=result_form or BenchmarkResultForm.RAW_RUN_SERIES,
            complete=benchmark_complete,
            facts=selected_facts,
        )
        return _outcome(
            ProfileSelectionState.SELECTED,
            selection=selection,
            reason=None,
            facts=selected_facts,
        )
    if training_positive:
        selection = _selection(
            EvidenceProfileId.TRAINING_EXPERIMENT_V0,
            variant=None,
            result_form=None,
            complete=False,
            facts=(),
        )
        return _outcome(
            ProfileSelectionState.SELECTED,
            selection=selection,
            reason=None,
            facts=(),
        )
    return _outcome(
        ProfileSelectionState.UNRESOLVED,
        selection=None,
        reason="required_profile_evidence_not_recovered",
        facts=(),
    )


def mapping_matches_evidence_profile(
    claim: CanonicalScientificClaim,
    selection: EvidenceProfileSelection,
    mapping: MappingCandidate | RepoMapping,
    normalized_evidence: tuple[NormalizedEvidence, ...],
) -> bool:
    """Return whether one exact mapping satisfies one engine-selected profile."""

    if type(claim) is not CanonicalScientificClaim:
        raise TypeError("profile mapping check requires CanonicalScientificClaim")
    if type(selection) is not EvidenceProfileSelection:
        raise TypeError("profile mapping check requires EvidenceProfileSelection")
    if type(mapping) not in {MappingCandidate, RepoMapping}:
        raise TypeError("profile mapping check requires a validated mapping")
    if selection.profile_id is EvidenceProfileId.TRAINING_EXPERIMENT_V0:
        return _training_complete(mapping, normalized_evidence)
    variant = selection.benchmark_variant
    if variant is None or type(claim.primary) is not MetricImprovementClaim:
        return False
    if len(mapping.bindings) > _MAX_EXECUTABLE_BINDINGS:
        return False
    try:
        _facts, result_form, _positive, complete = _benchmark_facts(
            mapping,
            normalized_evidence,
            metric=claim.primary.metric,
            variant=variant,
        )
    except AnalysisContractError:
        return False
    return complete and (
        selection.result_form is None or result_form is selection.result_form
    )


__all__ = [
    "BENCHMARK_PROFILE_SLOTS",
    "BenchmarkResultForm",
    "BenchmarkVariant",
    "EvidenceProfileDefinition",
    "EvidenceProfileId",
    "EvidenceProfileSelection",
    "ProfileRoleMode",
    "ProfileSelectionFact",
    "ProfileSelectionOutcome",
    "ProfileSelectionState",
    "ProfileSupportKind",
    "ProfiledEvidencePolicy",
    "evidence_profile_registry",
    "mapping_matches_evidence_profile",
    "profiled_evidence_policy",
    "select_evidence_profile",
]
