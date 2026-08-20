"""Bounded deterministic measurement-protocol identities and drift reports."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Mapping, TypeAlias


class MeasurementContractError(ValueError):
    """A measurement contract or canonical identity is invalid."""


class MeasurementComponentKind(str, Enum):
    EVALUATOR_IMPLEMENTATION = "evaluator_implementation"
    METRIC = "metric"
    EVALUATION_DATASET = "evaluation_dataset"
    EVALUATION_CONFIG = "evaluation_config"
    PROMPT_TEMPLATE = "prompt_template"
    DECODING_PARAMETERS = "decoding_parameters"
    TEST_SELECTION = "test_selection"
    RETRY_AGGREGATION = "retry_aggregation"
    NORMALIZATION_POSTPROCESSING = "normalization_postprocessing"
    BENCHMARK_VERSION = "benchmark_version"


class MeasurementComponentScope(str, Enum):
    MEASUREMENT_PROCEDURE = "measurement_procedure"
    SYSTEM_UNDER_TEST = "system_under_test"
    UNKNOWN = "unknown"


class MeasurementCoverage(str, Enum):
    RECOVERED = "recovered"
    BYTE_ONLY = "byte_only"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


class MeasurementDriftState(str, Enum):
    VERIFIED = "verified"
    WARNING = "warning"
    INSUFFICIENT = "insufficient"
    INVALIDATES = "invalidates"
    NOT_ASSESSED = "not_assessed"


class ClaimCIVerificationReduction(str, Enum):
    """ClaimCI's reduction, not a claim about an upstream experiment."""

    ARITHMETIC_MEAN_V1 = "arithmetic_mean_v1"


class UpstreamAggregationProcedure(str, Enum):
    ARITHMETIC_MEAN_V1 = "arithmetic_mean_v1"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"
    BEST_OF_N = "best_of_n"
    RETRY_FILTERING = "retry_filtering"
    ADJUDICATION = "adjudication"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


_CANONICAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"(?!0+\Z)[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_SOURCE_KINDS = frozenset(
    {
        "results",
        "config",
        "dataset",
        "benchmark",
        "document",
        "source",
        "test",
        "manifest",
    }
)
_SOURCE_ROLES = frozenset({"baseline", "candidate", "reference", "unspecified"})
_SOURCE_SPLITS = frozenset({"train", "eval"})
_WINDOWS_FORBIDDEN_PATH_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
        *(f"COM{number}" for number in "¹²³"),
        *(f"LPT{number}" for number in "¹²³"),
    }
)
_COMPONENT_SCHEMA_IDS = {
    MeasurementComponentKind.EVALUATOR_IMPLEMENTATION: (
        "claimci.measurement.evaluator.v1"
    ),
    MeasurementComponentKind.METRIC: "claimci.measurement.metric.v1",
    MeasurementComponentKind.EVALUATION_DATASET: (
        "claimci.measurement.evaluation-dataset.v1"
    ),
    MeasurementComponentKind.EVALUATION_CONFIG: (
        "claimci.measurement.evaluation-config.v1"
    ),
    MeasurementComponentKind.PROMPT_TEMPLATE: (
        "claimci.measurement.prompt-template.v1"
    ),
    MeasurementComponentKind.DECODING_PARAMETERS: (
        "claimci.measurement.decoding.v1"
    ),
    MeasurementComponentKind.TEST_SELECTION: (
        "claimci.measurement.test-selection.v1"
    ),
    MeasurementComponentKind.RETRY_AGGREGATION: (
        "claimci.measurement.retry-aggregation.v1"
    ),
    MeasurementComponentKind.NORMALIZATION_POSTPROCESSING: (
        "claimci.measurement.normalization.v1"
    ),
    MeasurementComponentKind.BENCHMARK_VERSION: (
        "claimci.measurement.benchmark.v1"
    ),
}


def _bounded_text(
    value: object,
    label: str,
    *,
    maximum: int,
    canonical: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or (canonical and value != value.strip())
    ):
        raise MeasurementContractError(f"{label} must be bounded canonical text")
    return value


def _canonical_id(value: object, label: str) -> str:
    text = _bounded_text(value, label, maximum=256, canonical=True)
    if _CANONICAL_ID.fullmatch(text) is None:
        raise MeasurementContractError(f"{label} must be a canonical identifier")
    return text


def _repository_path(value: object) -> str:
    text = _bounded_text(value, "measurement source path", maximum=1_024, canonical=True)
    parts = text.split("/")
    if (
        "\\" in text
        or PurePosixPath(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or bool(PureWindowsPath(text).drive)
        or any(part in {"", ".", ".."} for part in parts)
        or any(character in _WINDOWS_FORBIDDEN_PATH_CHARACTERS for character in text)
        or any(part != part.strip() or part.endswith(".") for part in parts)
        or any(
            part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_BASENAMES
            for part in parts
        )
        or PurePosixPath(text).as_posix() != text
    ):
        raise MeasurementContractError(
            "measurement source path must be portable and repository-relative"
        )
    return text


def _json_scalar(value: object, label: str) -> str | bool | int | float:
    if isinstance(value, str):
        return _bounded_text(value, label, maximum=4_096)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        try:
            json.dumps(value, allow_nan=False)
        except ValueError as error:
            raise MeasurementContractError(f"{label} integer is not JSON-safe") from error
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise MeasurementContractError(f"{label} must be a finite JSON scalar")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(prefix: str, value: object) -> str:
    return f"{prefix}-" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class MeasurementSemanticField:
    key: str
    value: str | bool | int | float

    def __post_init__(self) -> None:
        _canonical_id(self.key, "measurement semantic field key")
        _json_scalar(self.value, "measurement semantic field value")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementSemanticField is final")


@dataclass(frozen=True, slots=True)
class MeasurementSourceSelector:
    target_field: str
    kind: str
    expression: str

    def __post_init__(self) -> None:
        _canonical_id(self.target_field, "measurement selector target")
        _canonical_id(self.kind, "measurement selector kind")
        _bounded_text(
            self.expression,
            "measurement selector expression",
            maximum=1_024,
            canonical=True,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementSourceSelector is final")


@dataclass(frozen=True, slots=True)
class MeasurementSourceBinding:
    path: str
    kind: str
    sha256: str
    size: int
    adapter_id: str
    adapter_version: str | None
    selectors: tuple[MeasurementSourceSelector, ...]
    role: str
    split: str | None = None
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        _repository_path(self.path)
        if self.kind not in _SOURCE_KINDS:
            raise MeasurementContractError("measurement source kind is unsupported")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise MeasurementContractError("measurement source SHA-256 is invalid")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
            or self.size > 2**63 - 1
        ):
            raise MeasurementContractError("measurement source size is invalid")
        _canonical_id(self.adapter_id, "measurement adapter_id")
        if self.adapter_version is not None:
            _canonical_id(self.adapter_version, "measurement adapter version")
        if (
            not isinstance(self.selectors, tuple)
            or len(self.selectors) > 32
            or not all(type(item) is MeasurementSourceSelector for item in self.selectors)
        ):
            raise MeasurementContractError("measurement selectors exceed their bound")
        ordered = tuple(
            sorted(
                self.selectors,
                key=lambda item: (item.target_field, item.kind, item.expression),
            )
        )
        if len(set(ordered)) != len(ordered):
            raise MeasurementContractError("measurement selectors must be unique")
        object.__setattr__(self, "selectors", ordered)
        if self.role not in _SOURCE_ROLES:
            raise MeasurementContractError("measurement source role is invalid")
        if self.split is not None and self.split not in _SOURCE_SPLITS:
            raise MeasurementContractError("measurement source split is invalid")
        if (self.kind == "dataset") != (self.split is not None):
            raise MeasurementContractError(
                "measurement dataset bindings require an exact split"
            )
        material = {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size": self.size,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "selectors": [
                {
                    "target_field": item.target_field,
                    "kind": item.kind,
                    "expression": item.expression,
                }
                for item in ordered
            ],
            "role": self.role,
            "split": self.split,
        }
        object.__setattr__(self, "binding_id", _digest("measurement-binding", material))

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementSourceBinding is final")


@dataclass(frozen=True, slots=True, init=False)
class MeasurementSourceSnapshot:
    source_snapshot_id: str
    repository_id: str
    head_sha: str
    bindings: tuple[MeasurementSourceBinding, ...]

    def __init__(self) -> None:
        raise TypeError("MeasurementSourceSnapshot must be created through its factory")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementSourceSnapshot is final")

    @classmethod
    def from_bindings(
        cls,
        *,
        repository_id: str,
        head_sha: str,
        bindings: tuple[MeasurementSourceBinding, ...],
    ) -> "MeasurementSourceSnapshot":
        _bounded_text(repository_id, "measurement repository identity", maximum=512, canonical=True)
        if not isinstance(head_sha, str) or _GIT_SHA.fullmatch(head_sha) is None:
            raise MeasurementContractError("measurement head SHA is invalid")
        if (
            not isinstance(bindings, tuple)
            or not bindings
            or len(bindings) > 16
            or not all(type(item) is MeasurementSourceBinding for item in bindings)
        ):
            raise MeasurementContractError("measurement source bindings exceed their bound")
        ordered = tuple(sorted(bindings, key=lambda item: item.binding_id))
        if len({item.binding_id for item in ordered}) != len(ordered):
            raise MeasurementContractError("measurement source bindings must be unique")
        material = {
            "repository_id": repository_id,
            "head_sha": head_sha,
            "bindings": [item.binding_id for item in ordered],
        }
        instance = object.__new__(MeasurementSourceSnapshot)
        object.__setattr__(
            instance,
            "source_snapshot_id",
            _digest("measurement-source", material),
        )
        object.__setattr__(instance, "repository_id", repository_id)
        object.__setattr__(instance, "head_sha", head_sha)
        object.__setattr__(instance, "bindings", ordered)
        return instance


@dataclass(frozen=True, slots=True)
class MeasurementProtocolComponent:
    kind: MeasurementComponentKind
    scope: MeasurementComponentScope
    coverage: MeasurementCoverage
    schema_id: str
    semantic_fields: tuple[MeasurementSemanticField, ...]
    source_binding_ids: tuple[str, ...] = ()
    semantic_component_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.kind) is not MeasurementComponentKind:
            raise TypeError("measurement component kind is invalid")
        if type(self.scope) is not MeasurementComponentScope:
            raise TypeError("measurement component scope is invalid")
        if type(self.coverage) is not MeasurementCoverage:
            raise TypeError("measurement component coverage is invalid")
        _canonical_id(self.schema_id, "measurement component schema_id")
        if self.scope is not MeasurementComponentScope.MEASUREMENT_PROCEDURE:
            raise MeasurementContractError(
                "v1 measurement component scope must come from the fixed policy"
            )
        if self.schema_id != _COMPONENT_SCHEMA_IDS[self.kind]:
            raise MeasurementContractError(
                "measurement component schema must match the fixed v1 policy"
            )
        if (
            not isinstance(self.semantic_fields, tuple)
            or len(self.semantic_fields) > 32
            or not all(type(item) is MeasurementSemanticField for item in self.semantic_fields)
        ):
            raise MeasurementContractError("measurement semantic fields exceed their bound")
        ordered_fields = tuple(sorted(self.semantic_fields, key=lambda item: item.key))
        if len({item.key for item in ordered_fields}) != len(ordered_fields):
            raise MeasurementContractError("measurement semantic field keys must be unique")
        object.__setattr__(self, "semantic_fields", ordered_fields)
        if (
            not isinstance(self.source_binding_ids, tuple)
            or len(self.source_binding_ids) > 16
        ):
            raise MeasurementContractError("measurement source references exceed their bound")
        for binding_id in self.source_binding_ids:
            _canonical_id(binding_id, "measurement source binding_id")
        ordered_bindings = tuple(sorted(self.source_binding_ids))
        if len(set(ordered_bindings)) != len(ordered_bindings):
            raise MeasurementContractError("measurement source references must be unique")
        object.__setattr__(self, "source_binding_ids", ordered_bindings)
        if self.coverage is MeasurementCoverage.RECOVERED and not ordered_fields:
            raise MeasurementContractError("recovered measurement component needs semantic fields")
        if self.coverage is not MeasurementCoverage.RECOVERED and ordered_fields:
            raise MeasurementContractError(
                "unrecovered measurement component cannot assert semantic fields"
            )
        if self.coverage is MeasurementCoverage.BYTE_ONLY and not ordered_bindings:
            raise MeasurementContractError("byte-only component requires exact source identity")
        material = {
            "kind": self.kind.value,
            "scope": self.scope.value,
            "coverage": self.coverage.value,
            "schema_id": self.schema_id,
            "semantic_fields": [
                {"key": item.key, "value": item.value} for item in ordered_fields
            ],
        }
        object.__setattr__(
            self,
            "semantic_component_id",
            _digest("measurement-component", material),
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementProtocolComponent is final")


@dataclass(frozen=True, slots=True, init=False)
class MeasurementProtocolIdentity:
    semantic_protocol_id: str
    source_snapshot_id: str
    policy_id: str
    components: tuple[MeasurementProtocolComponent, ...]

    def __init__(self) -> None:
        raise TypeError("MeasurementProtocolIdentity must be created through its factory")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementProtocolIdentity is final")

    @classmethod
    def from_components(
        cls,
        *,
        policy_id: str,
        source_snapshot: MeasurementSourceSnapshot,
        components: tuple[MeasurementProtocolComponent, ...],
    ) -> "MeasurementProtocolIdentity":
        _canonical_id(policy_id, "measurement policy_id")
        if type(source_snapshot) is not MeasurementSourceSnapshot:
            raise TypeError("measurement identity requires a source snapshot")
        if (
            not isinstance(components, tuple)
            or not components
            or len(components) > len(MeasurementComponentKind)
            or not all(type(item) is MeasurementProtocolComponent for item in components)
        ):
            raise MeasurementContractError("measurement components exceed their bound")
        ordered = tuple(sorted(components, key=lambda item: item.kind.value))
        if len({item.kind for item in ordered}) != len(ordered):
            raise MeasurementContractError("measurement component kinds must be unique")
        issued_binding_ids = {item.binding_id for item in source_snapshot.bindings}
        if any(
            binding_id not in issued_binding_ids
            for component in ordered
            for binding_id in component.source_binding_ids
        ):
            raise MeasurementContractError(
                "measurement component references an unissued source binding"
            )
        material = {
            "domain": "claimci.measurement-protocol",
            "schema_version": "v1",
            "policy_id": policy_id,
            "components": [
                {
                    "kind": item.kind.value,
                    "semantic_component_id": item.semantic_component_id,
                }
                for item in ordered
            ],
        }
        instance = object.__new__(MeasurementProtocolIdentity)
        object.__setattr__(
            instance,
            "semantic_protocol_id",
            _digest("measurement-semantic", material),
        )
        object.__setattr__(
            instance,
            "source_snapshot_id",
            source_snapshot.source_snapshot_id,
        )
        object.__setattr__(instance, "policy_id", policy_id)
        object.__setattr__(instance, "components", ordered)
        return instance


@dataclass(frozen=True, slots=True)
class MeasurementProtocolPair:
    baseline: MeasurementProtocolIdentity
    candidate: MeasurementProtocolIdentity

    def __post_init__(self) -> None:
        if type(self.baseline) is not MeasurementProtocolIdentity or type(
            self.candidate
        ) is not MeasurementProtocolIdentity:
            raise TypeError("measurement protocol pair requires exact identities")
        if self.baseline.policy_id != self.candidate.policy_id:
            raise MeasurementContractError("measurement protocol policies must match")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementProtocolPair is final")


@dataclass(frozen=True, slots=True)
class MeasurementAuditContext:
    """Trusted exact-head source snapshots supplied by confined materialization."""

    baseline_source: MeasurementSourceSnapshot
    candidate_source: MeasurementSourceSnapshot
    policy_id: str = "claimci.measurement.pilot.v1"

    def __post_init__(self) -> None:
        if type(self.baseline_source) is not MeasurementSourceSnapshot or type(
            self.candidate_source
        ) is not MeasurementSourceSnapshot:
            raise TypeError("measurement Audit context requires exact source snapshots")
        _canonical_id(self.policy_id, "measurement Audit policy_id")
        if (
            self.baseline_source.repository_id
            != self.candidate_source.repository_id
            or self.baseline_source.head_sha != self.candidate_source.head_sha
        ):
            raise MeasurementContractError(
                "measurement Audit source snapshots must share repository and head"
            )
        if any(
            item.role != "baseline" for item in self.baseline_source.bindings
        ) or any(item.role != "candidate" for item in self.candidate_source.bindings):
            raise MeasurementContractError(
                "measurement Audit source snapshots must preserve exact experiment roles"
            )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementAuditContext is final")


def _flatten_evaluation(config: Mapping[str, object] | None) -> dict[str, object] | None:
    if config is None or not isinstance(config, MappingABC):
        return None
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, MappingABC):
        return None
    flattened: dict[str, object] = {}
    stack: list[tuple[str, object, int]] = [
        (str(key), value, 1) for key, value in evaluation.items()
    ]
    nodes = 0
    while stack:
        key, value, depth = stack.pop()
        nodes += 1
        if nodes > 256 or depth > 8:
            return None
        if not key or len(key) > 256:
            return None
        if isinstance(value, MappingABC):
            for child_key, child in value.items():
                if not isinstance(child_key, str) or not child_key:
                    return None
                stack.append((f"{key}.{child_key}", child, depth + 1))
            continue
        try:
            flattened[key] = _json_scalar(value, "recovered measurement value")
        except MeasurementContractError:
            return None
    return dict(sorted(flattened.items()))


def _fields(values: Mapping[str, object]) -> tuple[MeasurementSemanticField, ...] | None:
    if len(values) > 32:
        return None
    try:
        return tuple(
            MeasurementSemanticField(key, value)  # type: ignore[arg-type]
            for key, value in sorted(values.items())
        )
    except (MeasurementContractError, TypeError):
        return None


def _binding_ids(
    snapshot: MeasurementSourceSnapshot,
    *,
    kinds: frozenset[str],
    split: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        item.binding_id
        for item in snapshot.bindings
        if item.kind in kinds and (split is None or item.split == split)
    )


def _component_from_values(
    *,
    kind: MeasurementComponentKind,
    schema_id: str,
    values: Mapping[str, object] | None,
    source_binding_ids: tuple[str, ...],
    required: bool,
) -> MeasurementProtocolComponent:
    fields = None if values is None else _fields(values)
    if fields:
        coverage = MeasurementCoverage.RECOVERED
    elif required:
        coverage = MeasurementCoverage.MISSING
    else:
        coverage = MeasurementCoverage.NOT_APPLICABLE
    return MeasurementProtocolComponent(
        kind=kind,
        scope=MeasurementComponentScope.MEASUREMENT_PROCEDURE,
        coverage=coverage,
        schema_id=schema_id,
        semantic_fields=fields or (),
        source_binding_ids=source_binding_ids if coverage is not MeasurementCoverage.NOT_APPLICABLE else (),
    )


def _dataset_values(
    evaluation: Mapping[str, object] | None,
    hashes: tuple[str, ...] | None,
) -> dict[str, object] | None:
    if evaluation is None or hashes is None or not hashes:
        return None
    required = {
        "dataset_identifier": evaluation.get("dataset_identifier"),
        "dataset_version": evaluation.get("dataset_version"),
        "split": evaluation.get("split"),
    }
    if any(
        not isinstance(value, (str, int, float))
        or isinstance(value, bool)
        or (isinstance(value, str) and not value.strip())
        for value in required.values()
    ):
        return None
    if not all(isinstance(item, str) and _SHA256.fullmatch(item) for item in hashes):
        return None
    counts = Counter(hashes)
    commitment = hashlib.sha256(
        _canonical_bytes(
            [{"sha256": digest, "count": counts[digest]} for digest in sorted(counts)]
        )
    ).hexdigest()
    return {
        "declared_identifier": required["dataset_identifier"],
        "declared_version": required["dataset_version"],
        "declared_split": required["split"],
        "record_count": len(hashes),
        "record_multiset_sha256": commitment,
    }


def _optional_values(
    evaluation: Mapping[str, object] | None,
    *tokens: str,
) -> dict[str, object] | None:
    if evaluation is None:
        return None
    selected = {
        f"evaluation.{key}": value
        for key, value in evaluation.items()
        if any(
            key == token or key.startswith(f"{token}.")
            for token in tokens
        )
    }
    return selected or None


def _recover_protocol(
    *,
    policy_id: str,
    source: MeasurementSourceSnapshot,
    metric: str,
    config: Mapping[str, object] | None,
    evaluation_hashes: tuple[str, ...] | None,
) -> MeasurementProtocolIdentity:
    _bounded_text(metric, "measurement metric", maximum=256, canonical=True)
    evaluation = _flatten_evaluation(config)
    config_ids = _binding_ids(source, kinds=frozenset({"config"}))
    result_ids = _binding_ids(source, kinds=frozenset({"results"}))
    eval_dataset_ids = _binding_ids(
        source,
        kinds=frozenset({"dataset"}),
        split="eval",
    )
    metric_values: dict[str, object] = {"name": metric}
    if evaluation is not None:
        for key, value in evaluation.items():
            if key in {"metric", "metric_name", "metric_definition", "metric_version"} or key.startswith(
                "metric_config."
            ):
                metric_values[f"evaluation.{key}"] = value
    components = [
        _component_from_values(
            kind=MeasurementComponentKind.METRIC,
            schema_id="claimci.measurement.metric.v1",
            values=metric_values,
            source_binding_ids=tuple(sorted((*result_ids, *config_ids))),
            required=True,
        ),
        _component_from_values(
            kind=MeasurementComponentKind.EVALUATION_DATASET,
            schema_id="claimci.measurement.evaluation-dataset.v1",
            values=_dataset_values(evaluation, evaluation_hashes),
            source_binding_ids=tuple(sorted((*config_ids, *eval_dataset_ids))),
            required=True,
        ),
        _component_from_values(
            kind=MeasurementComponentKind.EVALUATION_CONFIG,
            schema_id="claimci.measurement.evaluation-config.v1",
            values=(
                None
                if evaluation is None
                else {f"evaluation.{key}": value for key, value in evaluation.items()}
            ),
            source_binding_ids=config_ids,
            required=True,
        ),
    ]
    optional = (
        (
            MeasurementComponentKind.EVALUATOR_IMPLEMENTATION,
            "claimci.measurement.evaluator.v1",
            ("evaluator", "scorer"),
        ),
        (
            MeasurementComponentKind.PROMPT_TEMPLATE,
            "claimci.measurement.prompt-template.v1",
            ("prompt", "prompt_template", "template"),
        ),
        (
            MeasurementComponentKind.DECODING_PARAMETERS,
            "claimci.measurement.decoding.v1",
            ("decoding", "sampling"),
        ),
        (
            MeasurementComponentKind.TEST_SELECTION,
            "claimci.measurement.test-selection.v1",
            ("test_selection", "selection", "subset", "filter"),
        ),
        (
            MeasurementComponentKind.RETRY_AGGREGATION,
            "claimci.measurement.retry-aggregation.v1",
            ("aggregation", "aggregation_procedure", "retry", "adjudication"),
        ),
        (
            MeasurementComponentKind.NORMALIZATION_POSTPROCESSING,
            "claimci.measurement.normalization.v1",
            ("normalization", "postprocessing", "postprocess"),
        ),
        (
            MeasurementComponentKind.BENCHMARK_VERSION,
            "claimci.measurement.benchmark.v1",
            ("benchmark",),
        ),
    )
    for kind, schema_id, tokens in optional:
        components.append(
            _component_from_values(
                kind=kind,
                schema_id=schema_id,
                values=_optional_values(evaluation, *tokens),
                source_binding_ids=config_ids,
                required=False,
            )
        )
    return MeasurementProtocolIdentity.from_components(
        policy_id=policy_id,
        source_snapshot=source,
        components=tuple(components),
    )


def recover_measurement_protocol_pair(
    context: MeasurementAuditContext,
    *,
    metric: str,
    baseline_config: Mapping[str, object] | None,
    candidate_config: Mapping[str, object] | None,
    baseline_evaluation_hashes: tuple[str, ...] | None,
    candidate_evaluation_hashes: tuple[str, ...] | None,
) -> MeasurementProtocolPair:
    """Recover semantic identities only from confined native Audit inputs."""

    if type(context) is not MeasurementAuditContext:
        raise TypeError("measurement recovery requires a trusted Audit context")
    return MeasurementProtocolPair(
        baseline=_recover_protocol(
            policy_id=context.policy_id,
            source=context.baseline_source,
            metric=metric,
            config=baseline_config,
            evaluation_hashes=baseline_evaluation_hashes,
        ),
        candidate=_recover_protocol(
            policy_id=context.policy_id,
            source=context.candidate_source,
            metric=metric,
            config=candidate_config,
            evaluation_hashes=candidate_evaluation_hashes,
        ),
    )


@dataclass(frozen=True, slots=True, init=False)
class MeasurementDriftFinding:
    component_kind: MeasurementComponentKind
    state: MeasurementDriftState
    reason_code: str
    baseline_component_id: str | None
    candidate_component_id: str | None
    native_rule_ids: tuple[str, ...] = ()

    def __init__(self) -> None:
        raise TypeError(
            "MeasurementDriftFinding must be created by the deterministic comparator"
        )

    def __post_init__(self) -> None:
        if type(self.component_kind) is not MeasurementComponentKind:
            raise TypeError("measurement drift component kind is invalid")
        if type(self.state) is not MeasurementDriftState:
            raise TypeError("measurement drift state is invalid")
        _canonical_id(self.reason_code, "measurement drift reason_code")
        for value in (self.baseline_component_id, self.candidate_component_id):
            if value is not None:
                _canonical_id(value, "measurement component identity")
        if (
            not isinstance(self.native_rule_ids, tuple)
            or len(self.native_rule_ids) > 8
        ):
            raise MeasurementContractError("measurement native rule references exceed bound")
        for rule_id in self.native_rule_ids:
            _canonical_id(rule_id, "measurement native rule_id")
        if len(set(self.native_rule_ids)) != len(self.native_rule_ids):
            raise MeasurementContractError("measurement native rule IDs must be unique")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementDriftFinding is final")

    @classmethod
    def _from_comparison(
        cls,
        component_kind: MeasurementComponentKind,
        state: MeasurementDriftState,
        reason_code: str,
        baseline_component_id: str | None,
        candidate_component_id: str | None,
        native_rule_ids: tuple[str, ...] = (),
    ) -> "MeasurementDriftFinding":
        instance = object.__new__(MeasurementDriftFinding)
        object.__setattr__(instance, "component_kind", component_kind)
        object.__setattr__(instance, "state", state)
        object.__setattr__(instance, "reason_code", reason_code)
        object.__setattr__(
            instance,
            "baseline_component_id",
            baseline_component_id,
        )
        object.__setattr__(
            instance,
            "candidate_component_id",
            candidate_component_id,
        )
        object.__setattr__(instance, "native_rule_ids", native_rule_ids)
        instance.__post_init__()
        return instance


@dataclass(frozen=True, slots=True, init=False)
class MeasurementDriftReport:
    state: MeasurementDriftState
    baseline_semantic_protocol_id: str
    candidate_semantic_protocol_id: str
    baseline_source_snapshot_id: str
    candidate_source_snapshot_id: str
    findings: tuple[MeasurementDriftFinding, ...]
    claimci_verification_reduction: ClaimCIVerificationReduction = (
        ClaimCIVerificationReduction.ARITHMETIC_MEAN_V1
    )

    def __init__(self) -> None:
        raise TypeError(
            "MeasurementDriftReport must be created by the deterministic comparator"
        )

    def __post_init__(self) -> None:
        if type(self.state) is not MeasurementDriftState:
            raise TypeError("measurement drift report state is invalid")
        for value in (
            self.baseline_semantic_protocol_id,
            self.candidate_semantic_protocol_id,
            self.baseline_source_snapshot_id,
            self.candidate_source_snapshot_id,
        ):
            _canonical_id(value, "measurement report identity")
        if (
            not isinstance(self.findings, tuple)
            or len(self.findings) > len(MeasurementComponentKind)
            or not all(type(item) is MeasurementDriftFinding for item in self.findings)
        ):
            raise MeasurementContractError("measurement drift findings exceed bound")
        if type(self.claimci_verification_reduction) is not ClaimCIVerificationReduction:
            raise TypeError("ClaimCI verification reduction is invalid")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MeasurementDriftReport is final")

    @classmethod
    def _from_comparison(
        cls,
        *,
        state: MeasurementDriftState,
        baseline_semantic_protocol_id: str,
        candidate_semantic_protocol_id: str,
        baseline_source_snapshot_id: str,
        candidate_source_snapshot_id: str,
        findings: tuple[MeasurementDriftFinding, ...],
    ) -> "MeasurementDriftReport":
        instance = object.__new__(MeasurementDriftReport)
        object.__setattr__(instance, "state", state)
        object.__setattr__(
            instance,
            "baseline_semantic_protocol_id",
            baseline_semantic_protocol_id,
        )
        object.__setattr__(
            instance,
            "candidate_semantic_protocol_id",
            candidate_semantic_protocol_id,
        )
        object.__setattr__(
            instance,
            "baseline_source_snapshot_id",
            baseline_source_snapshot_id,
        )
        object.__setattr__(
            instance,
            "candidate_source_snapshot_id",
            candidate_source_snapshot_id,
        )
        object.__setattr__(instance, "findings", findings)
        object.__setattr__(
            instance,
            "claimci_verification_reduction",
            ClaimCIVerificationReduction.ARITHMETIC_MEAN_V1,
        )
        instance.__post_init__()
        return instance


def _component_state(
    kind: MeasurementComponentKind,
    baseline: MeasurementProtocolComponent | None,
    candidate: MeasurementProtocolComponent | None,
    native_rule_ids: frozenset[str],
) -> MeasurementDriftFinding:
    baseline_id = None if baseline is None else baseline.semantic_component_id
    candidate_id = None if candidate is None else candidate.semantic_component_id
    if baseline is None or candidate is None:
        return MeasurementDriftFinding._from_comparison(
            kind,
            MeasurementDriftState.INSUFFICIENT,
            "component_identity_missing",
            baseline_id,
            candidate_id,
        )
    if (
        baseline.coverage is MeasurementCoverage.NOT_APPLICABLE
        and candidate.coverage is MeasurementCoverage.NOT_APPLICABLE
    ):
        return MeasurementDriftFinding._from_comparison(
            kind,
            MeasurementDriftState.NOT_ASSESSED,
            "optional_component_not_represented",
            baseline_id,
            candidate_id,
        )
    if MeasurementCoverage.MISSING in {baseline.coverage, candidate.coverage}:
        relevant = tuple(
            sorted(
                rule_id
                for rule_id in native_rule_ids
                if rule_id in {"CONFIG.MISSING_FIELDS"}
                or rule_id.startswith("DATASET.MISSING")
                or rule_id.startswith("DATASET.INVALID")
            )
        )
        return MeasurementDriftFinding._from_comparison(
            kind,
            MeasurementDriftState.INSUFFICIENT,
            "required_component_not_recovered",
            baseline_id,
            candidate_id,
            relevant,
        )
    if MeasurementCoverage.BYTE_ONLY in {baseline.coverage, candidate.coverage}:
        changed = baseline.source_binding_ids != candidate.source_binding_ids
        return MeasurementDriftFinding._from_comparison(
            kind,
            MeasurementDriftState.WARNING if changed else MeasurementDriftState.NOT_ASSESSED,
            "source_snapshot_changed_without_semantic_recovery"
            if changed
            else "byte_identity_only",
            baseline_id,
            candidate_id,
        )
    if baseline.semantic_component_id == candidate.semantic_component_id:
        return MeasurementDriftFinding._from_comparison(
            kind,
            MeasurementDriftState.VERIFIED,
            "recovered_semantics_match",
            baseline_id,
            candidate_id,
        )
    candidates = (
        {"DATASET.EVALUATION_MISMATCH", "CONFIG.EVALUATION_MISMATCH"}
        if kind is MeasurementComponentKind.EVALUATION_DATASET
        else {"CONFIG.EVALUATION_MISMATCH"}
    )
    invalidating = tuple(sorted(native_rule_ids & candidates))
    if invalidating:
        return MeasurementDriftFinding._from_comparison(
            kind,
            MeasurementDriftState.INVALIDATES,
            "recovered_measurement_semantics_differ",
            baseline_id,
            candidate_id,
            invalidating,
        )
    insufficient = tuple(
        sorted(
            rule_id
            for rule_id in native_rule_ids
            if rule_id == "CONFIG.MISSING_FIELDS"
            or rule_id.startswith("DATASET.MISSING")
            or rule_id.startswith("DATASET.INVALID")
        )
    )
    return MeasurementDriftFinding._from_comparison(
        kind,
        MeasurementDriftState.INSUFFICIENT,
        "semantic_difference_has_no_native_comparison",
        baseline_id,
        candidate_id,
        insufficient,
    )


def compare_measurement_protocols(
    pair: MeasurementProtocolPair,
    *,
    native_rule_ids: tuple[str, ...],
) -> MeasurementDriftReport:
    """Compare trusted identities while leaving verdict authority to native Audit."""

    if type(pair) is not MeasurementProtocolPair:
        raise TypeError("measurement comparison requires a protocol pair")
    if not isinstance(native_rule_ids, tuple) or len(native_rule_ids) > 128:
        raise MeasurementContractError("measurement native rule set exceeds bound")
    for rule_id in native_rule_ids:
        _canonical_id(rule_id, "measurement native rule_id")
    rule_ids = frozenset(native_rule_ids)
    baseline = {item.kind: item for item in pair.baseline.components}
    candidate = {item.kind: item for item in pair.candidate.components}
    kinds = tuple(sorted(set(baseline) | set(candidate), key=lambda item: item.value))
    findings = tuple(
        _component_state(kind, baseline.get(kind), candidate.get(kind), rule_ids)
        for kind in kinds
    )
    states = {item.state for item in findings}
    if MeasurementDriftState.INVALIDATES in states:
        state = MeasurementDriftState.INVALIDATES
    elif MeasurementDriftState.INSUFFICIENT in states:
        state = MeasurementDriftState.INSUFFICIENT
    elif MeasurementDriftState.WARNING in states:
        state = MeasurementDriftState.WARNING
    elif MeasurementDriftState.VERIFIED in states:
        state = MeasurementDriftState.VERIFIED
    else:
        state = MeasurementDriftState.NOT_ASSESSED
    return MeasurementDriftReport._from_comparison(
        state=state,
        baseline_semantic_protocol_id=pair.baseline.semantic_protocol_id,
        candidate_semantic_protocol_id=pair.candidate.semantic_protocol_id,
        baseline_source_snapshot_id=pair.baseline.source_snapshot_id,
        candidate_source_snapshot_id=pair.candidate.source_snapshot_id,
        findings=findings,
    )


def measurement_report_to_jsonable(report: MeasurementDriftReport) -> dict[str, object]:
    """Return the bounded public companion without copying semantic source values."""

    if type(report) is not MeasurementDriftReport:
        raise TypeError("measurement serialization requires a drift report")
    return {
        "state": report.state.value,
        "semantic_protocol_ids": {
            "baseline": report.baseline_semantic_protocol_id,
            "candidate": report.candidate_semantic_protocol_id,
        },
        "source_snapshot_ids": {
            "baseline": report.baseline_source_snapshot_id,
            "candidate": report.candidate_source_snapshot_id,
        },
        "claimci_verification_reduction": report.claimci_verification_reduction.value,
        "findings": [
            {
                "component_kind": item.component_kind.value,
                "state": item.state.value,
                "reason_code": item.reason_code,
                "baseline_component_id": item.baseline_component_id,
                "candidate_component_id": item.candidate_component_id,
                "native_rule_ids": list(item.native_rule_ids),
            }
            for item in report.findings
        ],
    }


__all__ = [
    "ClaimCIVerificationReduction",
    "MeasurementAuditContext",
    "MeasurementComponentKind",
    "MeasurementComponentScope",
    "MeasurementContractError",
    "MeasurementCoverage",
    "MeasurementDriftFinding",
    "MeasurementDriftReport",
    "MeasurementDriftState",
    "MeasurementProtocolComponent",
    "MeasurementProtocolIdentity",
    "MeasurementProtocolPair",
    "MeasurementSemanticField",
    "MeasurementSourceBinding",
    "MeasurementSourceSelector",
    "MeasurementSourceSnapshot",
    "UpstreamAggregationProcedure",
    "compare_measurement_protocols",
    "measurement_report_to_jsonable",
    "recover_measurement_protocol_pair",
]
