"""Bounded source trace contracts for exact deterministic analysis evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from claimci.models import AuditResult, Verdict
from claimci.report import render_json

from .contracts import (
    AdvisoryResearchInterpretation,
    AnalysisAuthority,
    ArtifactKind,
    DatasetSplit,
    ExperimentRole,
    GitCommitSha,
    ProvenanceKind,
    RepositoryPath,
    SelectorKind,
    Sha256Digest,
    to_jsonable,
)


EVIDENCE_TRACE_VERSION = "claimci.evidence-trace.v1"
EVIDENCE_TRACE_MAX_BYTES = 16_384
EVIDENCE_TRACE_MAX_ENTRIES = 32
EVIDENCE_TRACE_MAX_SELECTORS = 32
EVIDENCE_TRACE_MAX_RULE_IDS = 32
EVIDENCE_TRACE_MAX_SOURCE_IDS = 32
EVIDENCE_TRACE_MAX_DIRECT_VALUES = 8
EVIDENCE_TRACE_MAX_TEXT = 512
_MAX_COMMITMENT_NODES = 100_000
_MAX_COMMITMENT_BYTES = 16 * 1024 * 1024

_ADAPTER_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_RULE_ID = re.compile(r"[A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)+")
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}")


class TraceContractError(ValueError):
    """A trace value violates a bounded provenance or authority contract."""


class TraceLimitError(TraceContractError):
    """A trace exceeds its independent structural or byte budget."""


class TraceRecordKind(str, Enum):
    NATURAL_LANGUAGE_CLAIM = "natural_language_claim"
    LLM_SEMANTIC_PROPOSAL = "llm_semantic_proposal"
    PASSIVE_SOURCE_EVIDENCE = "passive_source_evidence"
    DERIVED_DETERMINISTIC_EVIDENCE = "derived_deterministic_evidence"
    APPROVED_MAPPING = "approved_mapping"


class TraceAuthorityClass(str, Enum):
    NON_AUTHORITATIVE_INPUT = "non_authoritative_input"
    PASSIVE_SOURCE = "passive_source"
    DETERMINISTIC_DERIVATION = "deterministic_derivation"
    EXPLICIT_MAPPING_APPROVAL = "explicit_mapping_approval"


class TraceCompleteness(str, Enum):
    COMPLETE = "complete"
    BOUNDED = "bounded"
    UNAVAILABLE = "unavailable"


class TraceValueType(str, Enum):
    NULL = "null"
    BOOLEAN = "boolean"
    NUMBER = "number"
    STRING = "string"
    SEQUENCE = "sequence"
    MAPPING = "mapping"
    BYTES = "bytes"


TraceDirectScalar = None | bool | int | float


def _bounded_text(value: object, label: str, *, maximum: int = EVIDENCE_TRACE_MAX_TEXT) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise TraceContractError(f"{label} must be bounded non-empty text")
    if value != value.strip() or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TraceContractError(f"{label} must be canonical bounded text")
    return value


def _identifier(value: object, label: str) -> str:
    return _bounded_text(value, label, maximum=128)


def _canonical_value(value: object, *, state: list[int]) -> object:
    state[0] += 1
    if state[0] > _MAX_COMMITMENT_NODES:
        raise TraceLimitError("trace source representation has too many values")
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise TraceContractError("trace source numbers must be finite")
        return {"type": "number", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, Mapping):
        pairs: list[list[object]] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TraceContractError("trace source mapping keys must be text")
            pairs.append([key, _canonical_value(value[key], state=state)])
        return {"type": "mapping", "value": pairs}
    if isinstance(value, (tuple, list)):
        return {
            "type": "sequence",
            "value": [_canonical_value(item, state=state) for item in value],
        }
    raise TraceContractError("trace source values must be finite JSON-compatible data")


def _direct_preview(value: object, output: list[TraceDirectScalar]) -> None:
    if len(output) >= EVIDENCE_TRACE_MAX_DIRECT_VALUES:
        return
    if value is None or type(value) in {bool, int}:
        output.append(value)
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise TraceContractError("trace source numbers must be finite")
        output.append(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _direct_preview(item, output)
            if len(output) >= EVIDENCE_TRACE_MAX_DIRECT_VALUES:
                return
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _direct_preview(item, output)
            if len(output) >= EVIDENCE_TRACE_MAX_DIRECT_VALUES:
                return


def _leaf_count(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(_leaf_count(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_leaf_count(item) for item in value)
    return 1


def _value_type(value: object) -> TraceValueType:
    if value is None:
        return TraceValueType.NULL
    if type(value) is bool:
        return TraceValueType.BOOLEAN
    if type(value) in {int, float}:
        return TraceValueType.NUMBER
    if isinstance(value, str):
        return TraceValueType.STRING
    if isinstance(value, Mapping):
        return TraceValueType.MAPPING
    if isinstance(value, (tuple, list)):
        return TraceValueType.SEQUENCE
    raise TraceContractError("trace source value type is unsupported")


@dataclass(frozen=True, slots=True)
class BoundedValueRepresentation:
    """A commitment to source values with only safe scalar previews."""

    value_type: TraceValueType
    count: int
    canonical_sha256: Sha256Digest
    direct_values: tuple[TraceDirectScalar, ...] = ()
    byte_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.value_type, TraceValueType):
            raise TypeError("trace value_type must be TraceValueType")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise TraceContractError("trace value count must be a non-negative integer")
        if not isinstance(self.canonical_sha256, Sha256Digest):
            object.__setattr__(
                self,
                "canonical_sha256",
                Sha256Digest(self.canonical_sha256),
            )
        if (
            not isinstance(self.direct_values, tuple)
            or len(self.direct_values) > EVIDENCE_TRACE_MAX_DIRECT_VALUES
        ):
            raise TraceLimitError("trace direct values exceed the preview bound")
        for value in self.direct_values:
            if value is not None and type(value) not in {bool, int, float}:
                raise TraceContractError(
                    "trace direct values may contain only numeric, boolean, or null scalars"
                )
            if type(value) is float and not math.isfinite(value):
                raise TraceContractError("trace direct values must be finite")
        if self.byte_count is not None and (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
        ):
            raise TraceContractError("trace byte_count must be a non-negative integer")

    @classmethod
    def from_value(cls, value: object) -> "BoundedValueRepresentation":
        canonical = _canonical_value(value, state=[0])
        try:
            encoded = json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            raise TraceContractError("trace source representation is not canonical JSON") from error
        if len(encoded) > _MAX_COMMITMENT_BYTES:
            raise TraceLimitError("trace source representation exceeds its input byte bound")
        preview: list[TraceDirectScalar] = []
        _direct_preview(value, preview)
        return cls(
            value_type=_value_type(value),
            count=_leaf_count(value),
            canonical_sha256=Sha256Digest(hashlib.sha256(encoded).hexdigest()),
            direct_values=tuple(preview),
            byte_count=len(encoded),
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "BoundedValueRepresentation":
        if not isinstance(value, bytes):
            raise TypeError("trace byte representation requires immutable bytes")
        if len(value) > _MAX_COMMITMENT_BYTES:
            raise TraceLimitError("trace byte representation exceeds its input bound")
        digest = hashlib.sha256(b"claimci-evidence-trace-bytes-v1\0" + value).hexdigest()
        return cls(
            value_type=TraceValueType.BYTES,
            count=len(value),
            canonical_sha256=Sha256Digest(digest),
            byte_count=len(value),
        )


@dataclass(frozen=True, slots=True)
class TraceSelector:
    target_field: str
    kind: SelectorKind
    expression: str

    def __post_init__(self) -> None:
        _identifier(self.target_field, "trace selector target_field")
        if not isinstance(self.kind, SelectorKind):
            raise TypeError("trace selector kind must be SelectorKind")
        _bounded_text(self.expression, "trace selector expression", maximum=1_024)


_AUTHORITY_BY_KIND = {
    TraceRecordKind.NATURAL_LANGUAGE_CLAIM: TraceAuthorityClass.NON_AUTHORITATIVE_INPUT,
    TraceRecordKind.LLM_SEMANTIC_PROPOSAL: TraceAuthorityClass.NON_AUTHORITATIVE_INPUT,
    TraceRecordKind.PASSIVE_SOURCE_EVIDENCE: TraceAuthorityClass.PASSIVE_SOURCE,
    TraceRecordKind.DERIVED_DETERMINISTIC_EVIDENCE: TraceAuthorityClass.DETERMINISTIC_DERIVATION,
    TraceRecordKind.APPROVED_MAPPING: TraceAuthorityClass.EXPLICIT_MAPPING_APPROVAL,
}


@dataclass(frozen=True, slots=True)
class EvidenceTraceEntry:
    trace_id: str
    record_kind: TraceRecordKind
    authority: TraceAuthorityClass
    head_sha: GitCommitSha
    provenance_kind: ProvenanceKind | None
    detail_code: str
    source_path: RepositoryPath | None = None
    source_id: str | None = None
    artifact_path: RepositoryPath | None = None
    artifact_sha256: Sha256Digest | None = None
    artifact_size: int | None = None
    artifact_kind: ArtifactKind | None = None
    adapter_id: str | None = None
    adapter_version: str | None = None
    selectors: tuple[TraceSelector, ...] = ()
    role: ExperimentRole = ExperimentRole.UNSPECIFIED
    dataset_split: DatasetSplit | None = None
    source_value: BoundedValueRepresentation | None = None
    normalized_value: BoundedValueRepresentation | None = None
    consumer_rule_ids: tuple[str, ...] = ()
    source_trace_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.trace_id, "trace_id")
        if not isinstance(self.record_kind, TraceRecordKind):
            raise TypeError("trace record_kind must be TraceRecordKind")
        if not isinstance(self.authority, TraceAuthorityClass):
            raise TypeError("trace authority must be TraceAuthorityClass")
        if self.authority is not _AUTHORITY_BY_KIND[self.record_kind]:
            raise TraceContractError("trace record kind has an invalid authority class")
        if not isinstance(self.head_sha, GitCommitSha):
            object.__setattr__(self, "head_sha", GitCommitSha(self.head_sha))
        if self.provenance_kind is not None and not isinstance(
            self.provenance_kind, ProvenanceKind
        ):
            raise TypeError("trace provenance_kind must be ProvenanceKind or null")
        _bounded_text(self.detail_code, "trace detail_code")
        if self.source_path is not None and not isinstance(self.source_path, RepositoryPath):
            object.__setattr__(self, "source_path", RepositoryPath(self.source_path))
        if self.source_id is not None:
            _identifier(self.source_id, "trace source_id")

        artifact_values = (
            self.artifact_path,
            self.artifact_sha256,
            self.artifact_size,
            self.artifact_kind,
        )
        if any(item is not None for item in artifact_values) and not all(
            item is not None for item in artifact_values
        ):
            raise TraceContractError("trace artifact identity fields must appear together")
        if self.artifact_path is not None and not isinstance(
            self.artifact_path, RepositoryPath
        ):
            object.__setattr__(self, "artifact_path", RepositoryPath(self.artifact_path))
        if self.artifact_sha256 is not None and not isinstance(
            self.artifact_sha256, Sha256Digest
        ):
            object.__setattr__(
                self,
                "artifact_sha256",
                Sha256Digest(self.artifact_sha256),
            )
        if self.artifact_size is not None and (
            isinstance(self.artifact_size, bool)
            or not isinstance(self.artifact_size, int)
            or self.artifact_size < 0
        ):
            raise TraceContractError("trace artifact_size must be non-negative")
        if self.artifact_kind is not None and not isinstance(self.artifact_kind, ArtifactKind):
            raise TypeError("trace artifact_kind must be ArtifactKind or null")
        if self.adapter_id is not None and (
            not isinstance(self.adapter_id, str) or not _ADAPTER_ID.fullmatch(self.adapter_id)
        ):
            raise TraceContractError("trace adapter_id must be a trusted registry identifier")
        if self.adapter_version is not None:
            _bounded_text(self.adapter_version, "trace adapter_version", maximum=64)
        if (
            not isinstance(self.selectors, tuple)
            or len(self.selectors) > EVIDENCE_TRACE_MAX_SELECTORS
            or not all(type(item) is TraceSelector for item in self.selectors)
        ):
            raise TraceLimitError("trace selectors must be a bounded TraceSelector tuple")
        selector_keys = tuple(
            (item.target_field, item.kind, item.expression) for item in self.selectors
        )
        if len(set(selector_keys)) != len(selector_keys):
            raise TraceContractError("trace selectors must be unique")
        if not isinstance(self.role, ExperimentRole):
            raise TypeError("trace role must be ExperimentRole")
        if self.dataset_split is not None and not isinstance(self.dataset_split, DatasetSplit):
            raise TypeError("trace dataset_split must be DatasetSplit or null")
        for label, value in (
            ("source_value", self.source_value),
            ("normalized_value", self.normalized_value),
        ):
            if value is not None and type(value) is not BoundedValueRepresentation:
                raise TypeError(f"trace {label} must be BoundedValueRepresentation or null")
        _validate_text_tuple(
            self.consumer_rule_ids,
            "consumer_rule_ids",
            maximum=EVIDENCE_TRACE_MAX_RULE_IDS,
            validator=_RULE_ID,
        )
        _validate_text_tuple(
            self.source_trace_ids,
            "source_trace_ids",
            maximum=EVIDENCE_TRACE_MAX_SOURCE_IDS,
        )
        if self.trace_id in self.source_trace_ids:
            raise TraceContractError("trace entry cannot cite itself as a source")

        if (
            self.provenance_kind is ProvenanceKind.PROVIDER_PROPOSAL
            and self.authority is not TraceAuthorityClass.NON_AUTHORITATIVE_INPUT
        ):
            raise TraceContractError(
                "provider provenance cannot become passive or deterministic evidence"
            )
        if self.record_kind is TraceRecordKind.PASSIVE_SOURCE_EVIDENCE:
            if self.provenance_kind is not ProvenanceKind.ADAPTER_EXTRACTION:
                raise TraceContractError(
                    "passive source evidence requires adapter extraction provenance"
                )
            if self.artifact_path is None or self.adapter_id is None:
                raise TraceContractError("passive source evidence requires artifact identity")
            if self.source_value is None or self.normalized_value is None:
                raise TraceContractError("passive source evidence requires value representations")
            if not self.consumer_rule_ids:
                raise TraceContractError("passive source evidence requires consumer rule IDs")
            if self.source_trace_ids:
                raise TraceContractError("passive source evidence cannot cite proposed sources")
        elif self.record_kind is TraceRecordKind.DERIVED_DETERMINISTIC_EVIDENCE:
            if self.provenance_kind is not None:
                raise TraceContractError("deterministic derivation cannot carry source provenance")
            if self.normalized_value is None or not self.source_trace_ids:
                raise TraceContractError("deterministic derivation requires values and sources")
            if not self.consumer_rule_ids:
                raise TraceContractError("deterministic derivation requires consumer rule IDs")
        elif self.record_kind is TraceRecordKind.LLM_SEMANTIC_PROPOSAL:
            if self.provenance_kind is not ProvenanceKind.PROVIDER_PROPOSAL:
                raise TraceContractError("LLM proposal requires provider provenance")
            if self.consumer_rule_ids:
                raise TraceContractError("LLM proposal cannot consume deterministic rules")
        elif self.record_kind is TraceRecordKind.APPROVED_MAPPING:
            if self.provenance_kind is not ProvenanceKind.USER_APPROVED:
                raise TraceContractError("approved mapping requires user approval provenance")
            if not self.source_trace_ids:
                raise TraceContractError("approved mapping requires exact source bindings")
            if self.consumer_rule_ids:
                raise TraceContractError("mapping approval is not deterministic rule authority")
        elif self.record_kind is TraceRecordKind.NATURAL_LANGUAGE_CLAIM:
            if self.consumer_rule_ids:
                raise TraceContractError("natural-language claim is not deterministic evidence")
            if self.source_value is None:
                raise TraceContractError("natural-language claim requires a source commitment")

        if self.record_kind is not TraceRecordKind.PASSIVE_SOURCE_EVIDENCE and any(
            item is not None for item in artifact_values
        ):
            raise TraceContractError("only passive evidence may carry artifact identity")
        if self.dataset_split is not None and self.artifact_kind is not ArtifactKind.DATASET:
            raise TraceContractError("only dataset evidence may carry a split")


def _validate_text_tuple(
    values: object,
    label: str,
    *,
    maximum: int,
    validator: re.Pattern[str] | None = None,
) -> None:
    if not isinstance(values, tuple) or len(values) > maximum:
        raise TraceLimitError(f"trace {label} exceeds its bound")
    if len(set(values)) != len(values):
        raise TraceContractError(f"trace {label} must be unique")
    for value in values:
        _identifier(value, f"trace {label} item")
        if validator is not None and not validator.fullmatch(value):
            raise TraceContractError(f"trace {label} item is invalid")


@dataclass(frozen=True, slots=True, init=False)
class DeterministicAuditTrace:
    verdict: Verdict
    audit_sha256: Sha256Digest
    rule_ids: tuple[str, ...]
    authority: AnalysisAuthority

    def __init__(self) -> None:
        raise TypeError("DeterministicAuditTrace must be created through from_audit_result")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("DeterministicAuditTrace is final")

    @classmethod
    def from_audit_result(cls, result: AuditResult) -> "DeterministicAuditTrace":
        if type(result) is not AuditResult:
            raise TypeError("deterministic trace authority requires an actual AuditResult")
        if type(result.verdict) is not Verdict:
            raise TypeError("deterministic trace authority requires an actual Verdict")
        rendered = render_json(result).encode("utf-8")
        rules = tuple(item.rule_id for item in result.findings)
        _validate_text_tuple(
            rules,
            "deterministic rule_ids",
            maximum=EVIDENCE_TRACE_MAX_RULE_IDS,
            validator=_RULE_ID,
        )
        instance = object.__new__(DeterministicAuditTrace)
        object.__setattr__(instance, "verdict", result.verdict)
        object.__setattr__(
            instance,
            "audit_sha256",
            Sha256Digest(hashlib.sha256(rendered).hexdigest()),
        )
        object.__setattr__(instance, "rule_ids", rules)
        object.__setattr__(instance, "authority", AnalysisAuthority.DETERMINISTIC)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class AdvisoryResearchTrace:
    summary_sha256: Sha256Digest
    interpretation_count: int
    missing_evidence_count: int
    authority: AnalysisAuthority

    def __init__(self) -> None:
        raise TypeError(
            "AdvisoryResearchTrace must be created through from_interpretation"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AdvisoryResearchTrace is final")

    @classmethod
    def from_interpretation(
        cls,
        interpretation: AdvisoryResearchInterpretation,
    ) -> "AdvisoryResearchTrace":
        if type(interpretation) is not AdvisoryResearchInterpretation:
            raise TypeError("advisory trace requires AdvisoryResearchInterpretation")
        material = json.dumps(
            {
                "summary": interpretation.summary,
                "interpretations": interpretation.interpretations,
                "missing_evidence": interpretation.missing_evidence,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        instance = object.__new__(AdvisoryResearchTrace)
        object.__setattr__(
            instance,
            "summary_sha256",
            Sha256Digest(hashlib.sha256(material).hexdigest()),
        )
        object.__setattr__(
            instance,
            "interpretation_count",
            len(interpretation.interpretations),
        )
        object.__setattr__(
            instance,
            "missing_evidence_count",
            len(interpretation.missing_evidence),
        )
        object.__setattr__(instance, "authority", AnalysisAuthority.ADVISORY)
        return instance


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EvidenceTraceBundle:
    head_sha: GitCommitSha
    completeness: TraceCompleteness
    entries: tuple[EvidenceTraceEntry, ...]
    deterministic_authority: DeterministicAuditTrace
    advisory_interpretation: AdvisoryResearchTrace | None = None
    omitted_entry_count: int = 0
    omitted_entries_sha256: Sha256Digest | None = None
    reason_code: str | None = None
    version: str = field(default=EVIDENCE_TRACE_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.head_sha, GitCommitSha):
            object.__setattr__(self, "head_sha", GitCommitSha(self.head_sha))
        if not isinstance(self.completeness, TraceCompleteness):
            raise TypeError("trace completeness must be TraceCompleteness")
        if (
            not isinstance(self.entries, tuple)
            or len(self.entries) > EVIDENCE_TRACE_MAX_ENTRIES
            or not all(type(item) is EvidenceTraceEntry for item in self.entries)
        ):
            raise TraceLimitError("trace entries exceed their structural bound")
        if type(self.deterministic_authority) is not DeterministicAuditTrace:
            raise TypeError("trace deterministic_authority must be DeterministicAuditTrace")
        if self.advisory_interpretation is not None and type(
            self.advisory_interpretation
        ) is not AdvisoryResearchTrace:
            raise TypeError("trace advisory_interpretation must be AdvisoryResearchTrace")
        if isinstance(self.omitted_entry_count, bool) or not isinstance(
            self.omitted_entry_count, int
        ) or self.omitted_entry_count < 0:
            raise TraceContractError("trace omitted_entry_count must be non-negative")
        if self.omitted_entries_sha256 is not None and not isinstance(
            self.omitted_entries_sha256, Sha256Digest
        ):
            object.__setattr__(
                self,
                "omitted_entries_sha256",
                Sha256Digest(self.omitted_entries_sha256),
            )
        if self.reason_code is not None and (
            not isinstance(self.reason_code, str) or not _REASON_CODE.fullmatch(self.reason_code)
        ):
            raise TraceContractError("trace reason_code is invalid")

        ids = tuple(item.trace_id for item in self.entries)
        if len(set(ids)) != len(ids):
            raise TraceContractError("trace entry identifiers must be unique")
        if any(item.head_sha != self.head_sha for item in self.entries):
            raise TraceContractError("trace entries must use the bundle head SHA")
        by_id = {item.trace_id: item for item in self.entries}
        for entry in self.entries:
            for source_id in entry.source_trace_ids:
                source = by_id.get(source_id)
                if source is None:
                    raise TraceContractError("trace entry cites an unknown source trace")
                if entry.record_kind is TraceRecordKind.DERIVED_DETERMINISTIC_EVIDENCE and source.authority not in {
                    TraceAuthorityClass.PASSIVE_SOURCE,
                    TraceAuthorityClass.DETERMINISTIC_DERIVATION,
                }:
                    raise TraceContractError(
                        "deterministic derivation may cite only passive or derived evidence"
                    )

        if self.completeness is TraceCompleteness.COMPLETE:
            if not self.entries or self.omitted_entry_count or self.omitted_entries_sha256 is not None or self.reason_code is not None:
                raise TraceContractError("complete trace cannot carry omissions or a reason")
        elif self.completeness is TraceCompleteness.BOUNDED:
            if self.omitted_entry_count < 1 or self.omitted_entries_sha256 is None or self.reason_code is None:
                raise TraceContractError("bounded trace requires committed omissions and a reason")
        elif self.completeness is TraceCompleteness.UNAVAILABLE:
            if self.entries or self.omitted_entry_count or self.omitted_entries_sha256 is not None or self.reason_code is None:
                raise TraceContractError("unavailable trace must carry only a fixed reason")

        if len(_canonical_json_bytes(self)) > EVIDENCE_TRACE_MAX_BYTES:
            raise TraceLimitError("evidence trace exceeds the 16384-byte limit")

    def with_advisory(
        self,
        interpretation: AdvisoryResearchInterpretation,
    ) -> "EvidenceTraceBundle":
        return dataclasses.replace(
            self,
            advisory_interpretation=AdvisoryResearchTrace.from_interpretation(
                interpretation
            ),
        )

    @classmethod
    def bounded_from(
        cls,
        bundle: "EvidenceTraceBundle",
        *,
        reason_code: str = "trace_size_limit",
    ) -> "EvidenceTraceBundle":
        if type(bundle) is not EvidenceTraceBundle:
            raise TypeError("bounded trace projection requires EvidenceTraceBundle")
        digest = hashlib.sha256(_canonical_json_bytes(bundle.entries)).hexdigest()
        return cls(
            head_sha=bundle.head_sha,
            completeness=TraceCompleteness.BOUNDED,
            entries=(),
            deterministic_authority=bundle.deterministic_authority,
            advisory_interpretation=bundle.advisory_interpretation,
            omitted_entry_count=len(bundle.entries),
            omitted_entries_sha256=Sha256Digest(digest),
            reason_code=reason_code,
        )

    @classmethod
    def unavailable(
        cls,
        *,
        head_sha: GitCommitSha,
        result: AuditResult,
        reason_code: str = "trace_construction_unavailable",
    ) -> "EvidenceTraceBundle":
        return cls(
            head_sha=head_sha,
            completeness=TraceCompleteness.UNAVAILABLE,
            entries=(),
            deterministic_authority=DeterministicAuditTrace.from_audit_result(result),
            reason_code=reason_code,
        )


@dataclass(frozen=True, slots=True)
class EphemeralAuditExecution:
    audit_result: AuditResult
    trace: EvidenceTraceBundle

    def __post_init__(self) -> None:
        if type(self.audit_result) is not AuditResult:
            raise TypeError("ephemeral audit execution requires an actual AuditResult")
        if type(self.trace) is not EvidenceTraceBundle:
            raise TypeError("ephemeral audit execution requires EvidenceTraceBundle")
        expected = DeterministicAuditTrace.from_audit_result(self.audit_result)
        if self.trace.deterministic_authority != expected:
            raise TraceContractError(
                "ephemeral trace authority does not match its AuditResult"
            )


def trace_json_bytes(bundle: EvidenceTraceBundle) -> bytes:
    if type(bundle) is not EvidenceTraceBundle:
        raise TypeError("trace serialization requires EvidenceTraceBundle")
    encoded = _canonical_json_bytes(bundle)
    if len(encoded) > EVIDENCE_TRACE_MAX_BYTES:
        raise TraceLimitError("evidence trace exceeds the independent byte limit")
    return encoded


__all__ = [
    "AdvisoryResearchTrace",
    "BoundedValueRepresentation",
    "DeterministicAuditTrace",
    "EVIDENCE_TRACE_MAX_BYTES",
    "EVIDENCE_TRACE_VERSION",
    "EphemeralAuditExecution",
    "EvidenceTraceBundle",
    "EvidenceTraceEntry",
    "TraceAuthorityClass",
    "TraceCompleteness",
    "TraceContractError",
    "TraceLimitError",
    "TraceRecordKind",
    "TraceSelector",
    "TraceValueType",
    "trace_json_bytes",
]
