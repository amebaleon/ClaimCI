"""Immutable common contracts for ClaimCI analysis and later integrations.

Repository and pull-request content reaches adapters only as validated metadata
and passive bytes.  This module contains no discovery, execution, provider, or
workflow behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Protocol

from claimci.models import AuditResult, Direction, Verdict
from claimci.report import render_json

from .confidence import Confidence


class AnalysisContractError(ValueError):
    """A controlled violation of a public analysis contract."""


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_ADAPTER_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_TARGET_FIELD = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z")
_DOTTED_SEGMENT = re.compile(r"[A-Za-z0-9_-]+\Z")
_COLUMN_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_. -]{0,255}\Z")
_INVALID_JSON_POINTER_ESCAPE = re.compile(r"~(?:[^01]|$)")
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


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisContractError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise AnalysisContractError(f"{label} exceeds {maximum} characters")
    return value


def _bounded_id(value: object, label: str, *, maximum: int = 128) -> str:
    text = _bounded_text(value, label, maximum=maximum)
    if text != text.strip() or _has_control(text):
        raise AnalysisContractError(f"{label} must be a canonical text identifier")
    return text


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


class RepositoryPath(str):
    """One canonical, portable, repository-relative POSIX file path."""

    def __new__(cls, value: str) -> RepositoryPath:
        if not isinstance(value, str):
            raise TypeError("repository path must be text")
        if not value or value != value.strip():
            raise AnalysisContractError("repository path must be non-empty and canonical")
        if "\x00" in value:
            raise AnalysisContractError("repository path contains NUL")
        if "\\" in value:
            raise AnalysisContractError("repository path must use forward slashes")
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        parts = value.split("/")
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or any(part in {"", ".", ".."} for part in parts)
            or _has_control(value)
            or any(
                character in _WINDOWS_FORBIDDEN_PATH_CHARACTERS
                for character in value
            )
            or any(part != part.strip() or part.endswith(".") for part in parts)
            or any(
                part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_BASENAMES
                for part in parts
            )
            or posix.as_posix() != value
        ):
            raise AnalysisContractError(
                "repository path must be a canonical confined relative file path"
            )
        return str.__new__(cls, value)


class Sha256Digest(str):
    """A lowercase hexadecimal SHA-256 digest."""

    def __new__(cls, value: str) -> Sha256Digest:
        if not isinstance(value, str):
            raise TypeError("sha256 must be text")
        if not _SHA256.fullmatch(value):
            raise AnalysisContractError(
                "sha256 must be a lowercase 64-character hexadecimal digest"
            )
        return str.__new__(cls, value)


class GitCommitSha(str):
    """A non-zero lowercase Git commit object identifier."""

    def __new__(cls, value: str) -> GitCommitSha:
        if not isinstance(value, str):
            raise TypeError("commit SHA must be text")
        if not _GIT_COMMIT.fullmatch(value) or set(value) == {"0"}:
            raise AnalysisContractError(
                "commit SHA must be a non-zero lowercase 40- or 64-character digest"
            )
        return str.__new__(cls, value)


class ArtifactKind(str, Enum):
    RESULTS = "results"
    CONFIG = "config"
    DATASET = "dataset"
    BENCHMARK = "benchmark"
    DOCUMENT = "document"
    SOURCE = "source"
    TEST = "test"
    MANIFEST = "manifest"


class ExperimentRole(str, Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    REFERENCE = "reference"
    UNSPECIFIED = "unspecified"


class DatasetSplit(str, Enum):
    """A repository mapping identity for one native Audit dataset slot."""

    TRAIN = "train"
    EVAL = "eval"


class ProvenanceKind(str, Enum):
    DETERMINISTIC_DISCOVERY = "deterministic_discovery"
    ADAPTER_EXTRACTION = "adapter_extraction"
    MANIFEST_HINT = "manifest_hint"
    PROVIDER_PROPOSAL = "provider_proposal"
    USER_APPROVED = "user_approved"


class MappingTrust(str, Enum):
    INFERRED = "inferred"
    MANIFEST_HINT = "manifest_hint"
    USER_APPROVED = "user_approved"


class SelectorKind(str, Enum):
    JSON_POINTER = "json_pointer"
    DOTTED_PATH = "dotted_path"
    COLUMN = "column"


class AnalysisState(str, Enum):
    COMPLETE = "complete"
    MAPPING_NEEDED = "mapping_needed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class AnalysisAuthority(str, Enum):
    DETERMINISTIC = "deterministic"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class FieldProvenance:
    kind: ProvenanceKind
    detail: str
    source_path: RepositoryPath | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProvenanceKind):
            raise TypeError("provenance kind must be ProvenanceKind")
        _bounded_text(self.detail, "provenance detail", maximum=4_096)
        if self.source_path is not None and not isinstance(
            self.source_path, RepositoryPath
        ):
            object.__setattr__(self, "source_path", RepositoryPath(self.source_path))
        if self.source_id is not None:
            _bounded_id(self.source_id, "provenance source_id", maximum=256)


@dataclass(frozen=True, slots=True)
class EvidenceSelector:
    kind: SelectorKind
    expression: str
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SelectorKind):
            raise TypeError("selector kind must be SelectorKind")
        expression = _bounded_text(
            self.expression,
            "selector expression",
            maximum=1_024,
        )
        if _has_control(expression):
            raise AnalysisContractError("selector expression contains control characters")
        if self.kind is SelectorKind.JSON_POINTER and (
            not expression.startswith("/")
            or _INVALID_JSON_POINTER_ESCAPE.search(expression) is not None
        ):
            raise AnalysisContractError(
                "JSON pointer selector must start with '/' and use valid escapes"
            )
        if self.kind is SelectorKind.DOTTED_PATH and (
            expression.startswith(".")
            or expression.endswith(".")
            or any(not _DOTTED_SEGMENT.fullmatch(part) for part in expression.split("."))
        ):
            raise AnalysisContractError("dotted selector path is invalid")
        if self.kind is SelectorKind.COLUMN and not _COLUMN_NAME.fullmatch(expression):
            raise AnalysisContractError("column selector name is invalid")
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("selector provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class FieldMapping:
    target_field: str
    selector: EvidenceSelector
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_field, str)
            or len(self.target_field) > 128
            or not _TARGET_FIELD.fullmatch(self.target_field)
        ):
            raise AnalysisContractError("mapping target_field is invalid")
        if not isinstance(self.selector, EvidenceSelector):
            raise TypeError("mapping selector must be EvidenceSelector")
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("mapping provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class ArtifactCandidate:
    path: RepositoryPath
    kind: ArtifactKind
    sha256: Sha256Digest
    size: int
    confidence: Confidence
    discovery_reason: str
    relevant_claim_ids: tuple[str, ...]
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.path, RepositoryPath):
            object.__setattr__(self, "path", RepositoryPath(self.path))
        if not isinstance(self.kind, ArtifactKind):
            raise TypeError("artifact kind must be ArtifactKind")
        if not isinstance(self.sha256, Sha256Digest):
            object.__setattr__(self, "sha256", Sha256Digest(self.sha256))
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise AnalysisContractError("artifact size must be a non-negative integer")
        if not isinstance(self.confidence, Confidence):
            raise TypeError("artifact confidence must be Confidence")
        _bounded_text(self.discovery_reason, "discovery reason", maximum=4_096)
        if not isinstance(self.relevant_claim_ids, tuple):
            raise AnalysisContractError(
                "relevant_claim_ids must be a tuple of bounded identifiers"
            )
        for claim_id in self.relevant_claim_ids:
            _bounded_id(claim_id, "relevant claim ID", maximum=128)
        if len(set(self.relevant_claim_ids)) != len(self.relevant_claim_ids):
            raise AnalysisContractError("relevant_claim_ids must be unique")
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("artifact provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class PassiveArtifact:
    candidate: ArtifactCandidate
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ArtifactCandidate):
            raise TypeError("passive artifact candidate must be ArtifactCandidate")
        if not isinstance(self.content, bytes):
            raise TypeError("passive artifact content must be immutable bytes")
        if len(self.content) != self.candidate.size:
            raise AnalysisContractError("passive artifact size does not match candidate")
        digest = hashlib.sha256(self.content).hexdigest()
        if digest != self.candidate.sha256:
            raise AnalysisContractError(
                "passive artifact sha256 digest does not match candidate"
            )


@dataclass(frozen=True, slots=True)
class AdapterMatch:
    adapter_id: str
    path: RepositoryPath
    confidence: Confidence
    mappings: tuple[FieldMapping, ...]
    match_evidence: tuple[FieldProvenance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or not _ADAPTER_ID.fullmatch(
            self.adapter_id
        ):
            raise AnalysisContractError("adapter_id must be a trusted registry identifier")
        if not isinstance(self.path, RepositoryPath):
            object.__setattr__(self, "path", RepositoryPath(self.path))
        if not isinstance(self.confidence, Confidence):
            raise TypeError("adapter confidence must be Confidence")
        if not isinstance(self.mappings, tuple) or not all(
            isinstance(mapping, FieldMapping) for mapping in self.mappings
        ):
            raise TypeError("adapter mappings must be a tuple of FieldMapping values")
        targets = tuple(mapping.target_field for mapping in self.mappings)
        if len(set(targets)) != len(targets):
            raise AnalysisContractError("adapter mappings must have unique target fields")
        if not isinstance(self.match_evidence, tuple) or not self.match_evidence or not all(
            isinstance(item, FieldProvenance) for item in self.match_evidence
        ):
            raise TypeError(
                "adapter match_evidence must be a non-empty tuple of FieldProvenance values"
            )


class Adapter(Protocol):
    """A deterministic parser of validated passive artifact bytes."""

    def probe(self, artifact: PassiveArtifact) -> AdapterMatch | None:
        ...

    def extract(
        self,
        artifact: PassiveArtifact,
        match: AdapterMatch,
    ) -> NormalizedEvidence:
        ...


JsonScalar = str | int | float | bool | None


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AnalysisContractError(f"{label} must be finite")
    return normalized


def _json_scalar(value: object, label: str) -> JsonScalar:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise AnalysisContractError(f"{label} must be a finite JSON scalar")


@dataclass(frozen=True, slots=True)
class ConfigValue:
    key: str
    value: JsonScalar
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _bounded_id(self.key, "config key", maximum=256)
        _json_scalar(self.value, "config value")
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("config value provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class DatasetReference:
    path: RepositoryPath
    split: str | None
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.path, RepositoryPath):
            object.__setattr__(self, "path", RepositoryPath(self.path))
        if self.split is not None:
            _bounded_id(self.split, "dataset split", maximum=64)
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("dataset reference provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class ComputeEvidence:
    name: str
    value: float
    unit: str | None
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _bounded_id(self.name, "compute evidence name", maximum=256)
        object.__setattr__(
            self,
            "value",
            _finite_number(self.value, "compute evidence value"),
        )
        if self.unit is not None:
            _bounded_id(self.unit, "compute evidence unit", maximum=64)
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("compute evidence provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class NormalizedObservation:
    provenance: FieldProvenance
    metric_name: str | None = None
    metric_value: float | None = None
    run_id: str | None = None
    seed: str | int | None = None
    experiment_role: ExperimentRole = ExperimentRole.UNSPECIFIED
    config_values: tuple[ConfigValue, ...] = ()
    dataset_references: tuple[DatasetReference, ...] = ()
    compute_evidence: tuple[ComputeEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("observation provenance must be FieldProvenance")
        if (self.metric_name is None) != (self.metric_value is None):
            raise AnalysisContractError(
                "observation metric_name and metric_value must appear together"
            )
        if self.metric_name is not None:
            _bounded_text(self.metric_name, "metric name", maximum=256)
            object.__setattr__(
                self,
                "metric_value",
                _finite_number(self.metric_value, "metric value"),
            )
        if self.run_id is not None:
            _bounded_id(self.run_id, "run_id", maximum=256)
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, (str, int)):
                raise TypeError("seed must be text, an integer, or null")
            if isinstance(self.seed, str):
                _bounded_id(self.seed, "seed", maximum=128)
        if not isinstance(self.experiment_role, ExperimentRole):
            raise TypeError("experiment_role must be ExperimentRole")
        if not isinstance(self.config_values, tuple) or not all(
            isinstance(item, ConfigValue) for item in self.config_values
        ):
            raise TypeError("config_values must be a tuple of ConfigValue values")
        if not isinstance(self.dataset_references, tuple) or not all(
            isinstance(item, DatasetReference) for item in self.dataset_references
        ):
            raise TypeError(
                "dataset_references must be a tuple of DatasetReference values"
            )
        if not isinstance(self.compute_evidence, tuple) or not all(
            isinstance(item, ComputeEvidence) for item in self.compute_evidence
        ):
            raise TypeError(
                "compute_evidence must be a tuple of ComputeEvidence values"
            )
        if (
            self.metric_name is None
            and not self.config_values
            and not self.dataset_references
            and not self.compute_evidence
        ):
            raise AnalysisContractError(
                "normalized observation must contain substantive evidence"
            )
        config_keys = tuple(item.key for item in self.config_values)
        if len(set(config_keys)) != len(config_keys):
            raise AnalysisContractError("config_values keys must be unique")
        dataset_paths = tuple(item.path for item in self.dataset_references)
        if len(set(dataset_paths)) != len(dataset_paths):
            raise AnalysisContractError("dataset reference paths must be unique")
        compute_names = tuple(item.name for item in self.compute_evidence)
        if len(set(compute_names)) != len(compute_names):
            raise AnalysisContractError("compute evidence names must be unique")


@dataclass(frozen=True, slots=True)
class NormalizedEvidence:
    evidence_id: str
    artifact: ArtifactCandidate
    adapter_match: AdapterMatch
    observations: tuple[NormalizedObservation, ...]

    def __post_init__(self) -> None:
        _bounded_id(self.evidence_id, "evidence_id", maximum=128)
        if not isinstance(self.artifact, ArtifactCandidate):
            raise TypeError("normalized evidence artifact must be ArtifactCandidate")
        if not isinstance(self.adapter_match, AdapterMatch):
            raise TypeError("normalized evidence adapter_match must be AdapterMatch")
        if self.artifact.path != self.adapter_match.path:
            raise AnalysisContractError(
                "normalized evidence artifact and adapter paths must match"
            )
        if not isinstance(self.observations, tuple) or not self.observations or not all(
            isinstance(item, NormalizedObservation) for item in self.observations
        ):
            raise TypeError(
                "normalized evidence observations must be a non-empty tuple"
            )


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: RepositoryPath
    kind: ArtifactKind
    role: ExperimentRole
    adapter_id: str | None
    mappings: tuple[FieldMapping, ...]
    provenance: FieldProvenance
    dataset_split: DatasetSplit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, RepositoryPath):
            object.__setattr__(self, "path", RepositoryPath(self.path))
        if not isinstance(self.kind, ArtifactKind):
            raise TypeError("binding kind must be ArtifactKind")
        if not isinstance(self.role, ExperimentRole):
            raise TypeError("binding role must be ExperimentRole")
        if self.adapter_id is not None and (
            not isinstance(self.adapter_id, str)
            or not _ADAPTER_ID.fullmatch(self.adapter_id)
        ):
            raise AnalysisContractError(
                "binding adapter_id must be a trusted registry identifier"
            )
        if not isinstance(self.mappings, tuple) or not all(
            isinstance(item, FieldMapping) for item in self.mappings
        ):
            raise TypeError("binding mappings must be a tuple of FieldMapping values")
        targets = tuple(mapping.target_field for mapping in self.mappings)
        if len(set(targets)) != len(targets):
            raise AnalysisContractError("binding mappings must have unique targets")
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("binding provenance must be FieldProvenance")
        if self.dataset_split is not None and not isinstance(
            self.dataset_split,
            DatasetSplit,
        ):
            raise TypeError("binding dataset split must be DatasetSplit or null")
        if self.kind is not ArtifactKind.DATASET and self.dataset_split is not None:
            raise AnalysisContractError(
                "only a dataset binding may carry a dataset split"
            )


def _validate_bindings(bindings: object, label: str) -> tuple[ArtifactBinding, ...]:
    if not isinstance(bindings, tuple) or not bindings or not all(
        isinstance(item, ArtifactBinding) for item in bindings
    ):
        raise TypeError(f"{label} must be a non-empty tuple of ArtifactBinding values")
    keys = tuple(
        (item.path, item.kind, item.role, item.dataset_split)
        for item in bindings
    )
    if len(set(keys)) != len(keys):
        raise AnalysisContractError(f"{label} must not contain duplicate bindings")
    roles_by_path: dict[RepositoryPath, set[ExperimentRole]] = {}
    for binding in bindings:
        roles_by_path.setdefault(binding.path, set()).add(binding.role)
    if any(
        {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}.issubset(roles)
        for roles in roles_by_path.values()
    ):
        raise AnalysisContractError(
            f"{label} cannot assign one path to conflicting baseline and candidate roles"
        )
    return bindings


def _field_mapping_projection(
    mappings: tuple[FieldMapping, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                item.target_field,
                item.selector.kind.value,
                item.selector.expression,
            )
            for item in mappings
        )
    )


@dataclass(frozen=True, slots=True)
class MappingCandidate:
    mapping_id: str
    bindings: tuple[ArtifactBinding, ...]
    confidence: Confidence
    trust: MappingTrust
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _bounded_id(self.mapping_id, "mapping_id", maximum=128)
        _validate_bindings(self.bindings, "mapping bindings")
        if not isinstance(self.confidence, Confidence):
            raise TypeError("mapping confidence must be Confidence")
        if not isinstance(self.trust, MappingTrust):
            raise TypeError("mapping trust must be MappingTrust")
        if self.trust not in {MappingTrust.INFERRED, MappingTrust.MANIFEST_HINT}:
            raise AnalysisContractError(
                "mapping candidates may only be inferred or manifest hints"
            )
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("mapping provenance must be FieldProvenance")
        if (
            self.trust is MappingTrust.MANIFEST_HINT
            and self.provenance.kind is not ProvenanceKind.MANIFEST_HINT
        ):
            raise AnalysisContractError(
                "manifest-hint mapping requires manifest-hint provenance"
            )
        if (
            self.trust is MappingTrust.INFERRED
            and self.provenance.kind in {
                ProvenanceKind.MANIFEST_HINT,
                ProvenanceKind.USER_APPROVED,
            }
        ):
            raise AnalysisContractError(
                "inferred mapping provenance cannot claim manifest or approved trust"
            )


@dataclass(frozen=True, slots=True)
class MappingChoice:
    choice_id: str
    label: str
    bindings: tuple[ArtifactBinding, ...]

    def __post_init__(self) -> None:
        _bounded_id(self.choice_id, "mapping choice_id", maximum=128)
        _bounded_text(self.label, "mapping choice label", maximum=512)
        _validate_bindings(self.bindings, "mapping choice bindings")


@dataclass(frozen=True, slots=True)
class MappingQuestion:
    question_id: str
    prompt: str
    choices: tuple[MappingChoice, ...]
    relevant_claim_id: str | None = None

    def __post_init__(self) -> None:
        _bounded_id(self.question_id, "mapping question_id", maximum=128)
        _bounded_text(self.prompt, "mapping question prompt", maximum=1_024)
        if not isinstance(self.choices, tuple):
            raise TypeError("mapping question choices must be a tuple")
        if not 2 <= len(self.choices) <= 8 or not all(
            isinstance(item, MappingChoice) for item in self.choices
        ):
            raise AnalysisContractError(
                "mapping question must contain two through eight choices"
            )
        choice_ids = tuple(choice.choice_id for choice in self.choices)
        if len(set(choice_ids)) != len(choice_ids):
            raise AnalysisContractError("mapping question choice IDs must be unique")
        if self.relevant_claim_id is not None:
            _bounded_id(
                self.relevant_claim_id,
                "mapping question relevant_claim_id",
                maximum=128,
            )


_REPOSITORY_COMPONENT = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    owner: str
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.owner, str) or not _REPOSITORY_COMPONENT.fullmatch(
            self.owner
        ):
            raise AnalysisContractError("repository owner is invalid")
        if not isinstance(self.name, str) or not _REPOSITORY_COMPONENT.fullmatch(
            self.name
        ):
            raise AnalysisContractError("repository name is invalid")

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass(frozen=True, slots=True, init=False)
class RepoMapping:
    repository: RepositoryIdentity
    bindings: tuple[ArtifactBinding, ...]
    approved_by: str
    source_mapping_id: str
    source_trust: MappingTrust
    approval_provenance: FieldProvenance
    trust: MappingTrust

    def __init__(self) -> None:
        raise TypeError("RepoMapping must be created through RepoMapping.approve")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("RepoMapping is final; use the explicit approval factory")

    @classmethod
    def approve(
        cls,
        repository: RepositoryIdentity,
        candidate: MappingCandidate,
        *,
        approved_by: str,
    ) -> RepoMapping:
        if type(repository) is not RepositoryIdentity:
            raise TypeError("approved mapping repository must be RepositoryIdentity")
        if type(candidate) is not MappingCandidate:
            raise TypeError("approved mapping candidate must be MappingCandidate")
        approver = _bounded_id(approved_by, "approved_by", maximum=256)
        instance = object.__new__(RepoMapping)
        object.__setattr__(instance, "repository", repository)
        object.__setattr__(instance, "bindings", candidate.bindings)
        object.__setattr__(instance, "approved_by", approver)
        object.__setattr__(instance, "source_mapping_id", candidate.mapping_id)
        object.__setattr__(instance, "source_trust", candidate.trust)
        object.__setattr__(
            instance,
            "approval_provenance",
            FieldProvenance(
                kind=ProvenanceKind.USER_APPROVED,
                detail=f"repository mapping explicitly approved by {approver}",
                source_id=approver,
            ),
        )
        object.__setattr__(instance, "trust", MappingTrust.USER_APPROVED)
        return instance

    def scope_to_runtime_bindings(
        self,
        bindings: tuple[ArtifactBinding, ...],
    ) -> RepoMapping:
        """Narrow approved slots and attach only validated runtime adapter data."""

        validated = _validate_bindings(bindings, "runtime-approved bindings")
        approved_by_key = {
            (item.path, item.kind, item.role, item.dataset_split): item
            for item in self.bindings
        }
        for binding in validated:
            key = (
                binding.path,
                binding.kind,
                binding.role,
                binding.dataset_split,
            )
            approved = approved_by_key.get(key)
            if approved is None:
                raise AnalysisContractError(
                    "runtime binding is not an approved repository mapping slot"
                )
            if binding.adapter_id is None:
                raise AnalysisContractError(
                    "runtime-approved binding requires a validated adapter"
                )
            if approved.adapter_id is not None and (
                binding.adapter_id != approved.adapter_id
            ):
                raise AnalysisContractError(
                    "runtime adapter conflicts with the approved binding"
                )
            if approved.mappings and _field_mapping_projection(
                binding.mappings
            ) != _field_mapping_projection(approved.mappings):
                raise AnalysisContractError(
                    "runtime selectors conflict with the approved binding"
                )
            if binding.provenance != approved.provenance:
                raise AnalysisContractError(
                    "runtime binding provenance must preserve the approved slot"
                )

        if validated == self.bindings:
            return self

        instance = object.__new__(RepoMapping)
        object.__setattr__(instance, "repository", self.repository)
        object.__setattr__(instance, "bindings", validated)
        object.__setattr__(instance, "approved_by", self.approved_by)
        object.__setattr__(instance, "source_mapping_id", self.source_mapping_id)
        object.__setattr__(instance, "source_trust", self.source_trust)
        object.__setattr__(
            instance,
            "approval_provenance",
            self.approval_provenance,
        )
        object.__setattr__(instance, "trust", MappingTrust.USER_APPROVED)
        return instance


@dataclass(frozen=True, slots=True)
class ClaimReference:
    claim_id: str
    text: str
    source_path: RepositoryPath | None
    confidence: Confidence
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _bounded_id(self.claim_id, "claim_id", maximum=128)
        _bounded_text(self.text, "claim text", maximum=16_000)
        if self.source_path is not None and not isinstance(
            self.source_path, RepositoryPath
        ):
            object.__setattr__(self, "source_path", RepositoryPath(self.source_path))
        if not isinstance(self.confidence, Confidence):
            raise TypeError("claim confidence must be Confidence")
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("claim provenance must be FieldProvenance")


@dataclass(frozen=True, slots=True)
class ClaimedMetricValue:
    """A descriptive value quoted by a claim, never deterministic evidence."""

    role: ExperimentRole
    value: float
    unit: str | None
    provenance: FieldProvenance
    raw: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {ExperimentRole.BASELINE, ExperimentRole.CANDIDATE}:
            raise AnalysisContractError(
                "claimed metric value role must be baseline or candidate"
            )
        object.__setattr__(
            self,
            "value",
            _finite_number(self.value, "claimed metric value"),
        )
        if self.unit is not None:
            unit = _bounded_text(self.unit, "claimed metric value unit", maximum=64)
            if unit != unit.strip() or _has_control(unit):
                raise AnalysisContractError(
                    "claimed metric value unit must be canonical text"
                )
        if not isinstance(self.provenance, FieldProvenance):
            raise TypeError("claimed metric value provenance must be FieldProvenance")
        if self.raw is not None:
            raw = _bounded_text(self.raw, "claimed metric raw token", maximum=512)
            if raw != raw.strip() or _has_control(raw):
                raise AnalysisContractError(
                    "claimed metric raw token must be canonical text"
                )


@dataclass(frozen=True, slots=True)
class AuditClaimSpec:
    """A validated claim policy to test, without any verdict authority."""

    claim_id: str
    metric: str
    direction: Direction
    minimum_absolute_improvement: float | None
    metric_provenance: FieldProvenance
    direction_provenance: FieldProvenance
    threshold_provenance: FieldProvenance | None
    claimed_values: tuple[ClaimedMetricValue, ...] = ()

    def __post_init__(self) -> None:
        _bounded_id(self.claim_id, "audit claim_id", maximum=128)
        metric = _bounded_text(self.metric, "audit metric", maximum=256)
        if metric != metric.strip() or _has_control(metric):
            raise AnalysisContractError("audit metric must be canonical text")
        if not isinstance(self.direction, Direction):
            raise TypeError("audit direction must be Direction")
        if (self.minimum_absolute_improvement is None) != (
            self.threshold_provenance is None
        ):
            raise AnalysisContractError(
                "audit threshold and threshold provenance must appear together"
            )
        if self.minimum_absolute_improvement is not None:
            threshold = _finite_number(
                self.minimum_absolute_improvement,
                "minimum absolute improvement",
            )
            if threshold < 0:
                raise AnalysisContractError(
                    "minimum absolute improvement must be non-negative"
                )
            object.__setattr__(self, "minimum_absolute_improvement", threshold)
        for label, provenance in (
            ("metric", self.metric_provenance),
            ("direction", self.direction_provenance),
        ):
            if not isinstance(provenance, FieldProvenance):
                raise TypeError(f"audit {label} provenance must be FieldProvenance")
        if self.threshold_provenance is not None and not isinstance(
            self.threshold_provenance,
            FieldProvenance,
        ):
            raise TypeError(
                "audit threshold provenance must be FieldProvenance or null"
            )
        if not isinstance(self.claimed_values, tuple) or not all(
            isinstance(item, ClaimedMetricValue) for item in self.claimed_values
        ):
            raise TypeError(
                "audit claimed_values must be a tuple of ClaimedMetricValue values"
            )
        roles = tuple(item.role for item in self.claimed_values)
        if len(set(roles)) != len(roles):
            raise AnalysisContractError("audit claimed value roles must be unique")


@dataclass(frozen=True, slots=True)
class MissingEvidence:
    kind: ArtifactKind
    role: ExperimentRole
    description: str
    claim_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            raise TypeError("missing evidence kind must be ArtifactKind")
        if not isinstance(self.role, ExperimentRole):
            raise TypeError("missing evidence role must be ExperimentRole")
        _bounded_text(self.description, "missing evidence description", maximum=4_096)
        if self.claim_id is not None:
            _bounded_id(self.claim_id, "missing evidence claim_id", maximum=128)


def _evidence_role_matches(
    evidence: NormalizedEvidence,
    expected: ExperimentRole,
) -> bool:
    return all(
        observation.experiment_role in {expected, ExperimentRole.UNSPECIFIED}
        for observation in evidence.observations
    )


@dataclass(frozen=True, slots=True)
class EphemeralAuditPlan:
    plan_id: str
    repository: RepositoryIdentity
    pr_number: int | None
    head_sha: GitCommitSha
    claim: ClaimReference
    baseline_evidence: tuple[NormalizedEvidence, ...]
    candidate_evidence: tuple[NormalizedEvidence, ...]
    mapping_provenance: tuple[FieldProvenance, ...]
    missing_evidence: tuple[MissingEvidence, ...]
    confidence: Confidence
    audit_claim: AuditClaimSpec | None = None
    selected_mapping: MappingCandidate | RepoMapping | None = None
    semantic_proposal_provenance: FieldProvenance | None = None
    scientific_claim: "CanonicalScientificClaim | None" = None
    claim_policy: "ClaimEvidencePolicy | None" = None
    ephemeral: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        _bounded_id(self.plan_id, "plan_id", maximum=128)
        if not isinstance(self.repository, RepositoryIdentity):
            raise TypeError("plan repository must be RepositoryIdentity")
        if self.pr_number is not None and (
            isinstance(self.pr_number, bool)
            or not isinstance(self.pr_number, int)
            or self.pr_number < 1
        ):
            raise AnalysisContractError("plan pr_number must be a positive integer")
        if not isinstance(self.head_sha, GitCommitSha):
            object.__setattr__(self, "head_sha", GitCommitSha(self.head_sha))
        if not isinstance(self.claim, ClaimReference):
            raise TypeError("plan claim must be ClaimReference")
        for label, values, role in (
            ("baseline", self.baseline_evidence, ExperimentRole.BASELINE),
            ("candidate", self.candidate_evidence, ExperimentRole.CANDIDATE),
        ):
            if not isinstance(values, tuple) or not all(
                isinstance(item, NormalizedEvidence) for item in values
            ):
                raise TypeError(f"plan {label}_evidence must be a tuple")
            if not all(_evidence_role_matches(item, role) for item in values):
                raise AnalysisContractError(
                    f"plan {label} evidence has a conflicting experiment role"
                )
        evidence_ids = tuple(
            item.evidence_id
            for item in (*self.baseline_evidence, *self.candidate_evidence)
        )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise AnalysisContractError("plan evidence IDs must be unique")
        if not isinstance(self.mapping_provenance, tuple) or not self.mapping_provenance or not all(
            isinstance(item, FieldProvenance) for item in self.mapping_provenance
        ):
            raise TypeError(
                "plan mapping_provenance must be a non-empty tuple of FieldProvenance"
            )
        if not isinstance(self.missing_evidence, tuple) or not all(
            isinstance(item, MissingEvidence) for item in self.missing_evidence
        ):
            raise TypeError("plan missing_evidence must be a tuple of MissingEvidence")
        if not isinstance(self.confidence, Confidence):
            raise TypeError("plan confidence must be Confidence")
        if self.audit_claim is not None and type(self.audit_claim) is not AuditClaimSpec:
            raise TypeError("plan audit_claim must be AuditClaimSpec or null")
        if self.selected_mapping is not None and type(self.selected_mapping) not in {
            MappingCandidate,
            RepoMapping,
        }:
            raise TypeError(
                "plan selected_mapping must be MappingCandidate, RepoMapping, or null"
            )
        if (self.audit_claim is None) != (self.selected_mapping is None):
            raise AnalysisContractError(
                "executable plan audit claim and selected mapping must appear together"
            )
        if (
            self.audit_claim is not None
            and self.audit_claim.claim_id != self.claim.claim_id
        ):
            raise AnalysisContractError(
                "plan audit claim ID must match its claim reference"
            )
        if self.semantic_proposal_provenance is not None:
            if type(self.semantic_proposal_provenance) is not FieldProvenance:
                raise TypeError(
                    "plan semantic proposal provenance must be FieldProvenance or null"
                )
            if (
                self.semantic_proposal_provenance.kind
                is not ProvenanceKind.PROVIDER_PROPOSAL
            ):
                raise AnalysisContractError(
                    "plan semantic proposal must remain provider provenance"
                )
        if (self.scientific_claim is None) != (self.claim_policy is None):
            raise AnalysisContractError(
                "plan scientific claim and claim policy must appear together"
            )
        if self.scientific_claim is not None:
            from .claim_types import (
                CanonicalScientificClaim,
                ClaimEvidencePolicy,
                MetricImprovementClaim,
                claim_evidence_policy,
                compile_audit_claim,
            )

            if type(self.scientific_claim) is not CanonicalScientificClaim:
                raise TypeError(
                    "plan scientific_claim must be CanonicalScientificClaim or null"
                )
            if type(self.claim_policy) is not ClaimEvidencePolicy:
                raise TypeError("plan claim_policy must be ClaimEvidencePolicy or null")
            if self.scientific_claim.reference != self.claim:
                raise AnalysisContractError(
                    "plan scientific claim reference must match its claim"
                )
            if self.claim_policy != claim_evidence_policy(self.scientific_claim):
                raise AnalysisContractError(
                    "plan claim policy must match canonical claim semantics"
                )
            if type(self.scientific_claim.primary) is not MetricImprovementClaim:
                raise AnalysisContractError(
                    "executable plan requires a metric improvement primary claim"
                )
            if self.audit_claim != compile_audit_claim(self.scientific_claim):
                raise AnalysisContractError(
                    "plan audit claim must match the deterministic compiler"
                )


def _deep_freeze(value: object) -> object:
    """Copy finite JSON values into recursively immutable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisContractError(
                "deterministic snapshot values must be finite"
            )
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("deterministic snapshot mapping keys must be text")
            frozen[key] = _deep_freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    raise TypeError(
        "deterministic snapshot values must be finite JSON-compatible values"
    )


@dataclass(frozen=True, slots=True, init=False)
class DeterministicAuditOutcome:
    """An immutable authority snapshot created only from the existing audit."""

    verdict: Verdict
    payload: Mapping[str, object]
    authority: AnalysisAuthority

    def __init__(self) -> None:
        raise TypeError(
            "DeterministicAuditOutcome must be created through from_audit_result"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "DeterministicAuditOutcome is final; use the AuditResult factory"
        )

    @classmethod
    def from_audit_result(
        cls,
        result: AuditResult,
    ) -> DeterministicAuditOutcome:
        """Snapshot one actual deterministic result without reinterpreting it."""

        if type(result) is not AuditResult:
            raise TypeError("deterministic authority requires an actual AuditResult")
        if type(result.verdict) is not Verdict:
            raise TypeError("deterministic authority requires an actual Verdict")
        payload = json.loads(render_json(result))
        if not isinstance(payload, dict):
            raise AnalysisContractError(
                "deterministic audit rendering must produce a JSON object"
            )
        instance = object.__new__(DeterministicAuditOutcome)
        object.__setattr__(instance, "verdict", result.verdict)
        object.__setattr__(instance, "payload", _deep_freeze(payload))
        object.__setattr__(instance, "authority", AnalysisAuthority.DETERMINISTIC)
        return instance


def _validate_text_tuple(
    values: object,
    label: str,
    *,
    item_maximum: int,
) -> None:
    if not isinstance(values, tuple) or not all(
        isinstance(item, str)
        and bool(item.strip())
        and len(item) <= item_maximum
        for item in values
    ):
        raise AnalysisContractError(
            f"{label} must be a tuple of non-empty bounded strings"
        )


@dataclass(frozen=True, slots=True)
class AdvisoryResearchInterpretation:
    """Non-blocking research interpretation with no verdict-bearing fields."""

    summary: str
    interpretations: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    confidence: Confidence
    authority: AnalysisAuthority = field(
        default=AnalysisAuthority.ADVISORY,
        init=False,
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AdvisoryResearchInterpretation is a final advisory type")

    def __post_init__(self) -> None:
        _bounded_text(self.summary, "advisory summary", maximum=16_000)
        _validate_text_tuple(
            self.interpretations,
            "advisory interpretations",
            item_maximum=16_000,
        )
        _validate_text_tuple(
            self.missing_evidence,
            "advisory missing_evidence",
            item_maximum=4_096,
        )
        if not isinstance(self.confidence, Confidence):
            raise TypeError("advisory confidence must be Confidence")


@dataclass(frozen=True, slots=True)
class UnifiedAnalysisResult:
    """One result envelope with deterministic and advisory lanes kept separate."""

    state: AnalysisState
    deterministic: DeterministicAuditOutcome | None = None
    research_interpretation: AdvisoryResearchInterpretation | None = None
    mapping_question: MappingQuestion | None = None
    unavailable_reason: str | None = None
    missing_evidence: tuple[MissingEvidence, ...] = ()
    evidence_trace: "EvidenceTraceBundle | None" = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("UnifiedAnalysisResult is final to preserve authority")

    def __post_init__(self) -> None:
        if not isinstance(self.state, AnalysisState):
            raise TypeError("analysis state must be AnalysisState")
        if self.deterministic is not None and type(
            self.deterministic
        ) is not DeterministicAuditOutcome:
            raise TypeError(
                "deterministic result must be DeterministicAuditOutcome"
            )
        if self.research_interpretation is not None and type(
            self.research_interpretation
        ) is not AdvisoryResearchInterpretation:
            raise TypeError(
                "research interpretation must be AdvisoryResearchInterpretation"
            )
        if self.mapping_question is not None and not isinstance(
            self.mapping_question,
            MappingQuestion,
        ):
            raise TypeError("mapping question must be MappingQuestion")
        if self.unavailable_reason is not None:
            _bounded_text(
                self.unavailable_reason,
                "unavailable reason",
                maximum=4_096,
            )
        if not isinstance(self.missing_evidence, tuple) or not all(
            isinstance(item, MissingEvidence) for item in self.missing_evidence
        ):
            raise TypeError(
                "unified missing_evidence must be a tuple of MissingEvidence values"
            )
        if self.evidence_trace is not None:
            from .trace import EvidenceTraceBundle

            if type(self.evidence_trace) is not EvidenceTraceBundle:
                raise TypeError("unified evidence_trace must be EvidenceTraceBundle")
            if self.deterministic is None:
                raise AnalysisContractError(
                    "evidence trace requires a deterministic outcome"
                )
            if (
                self.evidence_trace.deterministic_authority.verdict
                is not self.deterministic.verdict
            ):
                raise AnalysisContractError(
                    "evidence trace verdict does not match deterministic authority"
                )
            if (
                self.evidence_trace.advisory_interpretation is None
            ) != (self.research_interpretation is None):
                raise AnalysisContractError(
                    "evidence trace advisory state does not match Research Review"
                )

        if self.state is AnalysisState.COMPLETE:
            if self.deterministic is None:
                raise AnalysisContractError(
                    "complete analysis requires a deterministic outcome"
                )
            if self.research_interpretation is None:
                raise AnalysisContractError(
                    "complete analysis requires a research interpretation"
                )
            if self.mapping_question is not None or self.unavailable_reason is not None:
                raise AnalysisContractError(
                    "complete analysis cannot carry mapping or unavailable state"
                )
        elif self.state is AnalysisState.MAPPING_NEEDED:
            if self.mapping_question is None:
                raise AnalysisContractError(
                    "mapping-needed analysis requires a mapping question"
                )
            if self.deterministic is not None:
                raise AnalysisContractError(
                    "mapping-needed analysis cannot carry a deterministic outcome"
                )
            if self.unavailable_reason is not None:
                raise AnalysisContractError(
                    "mapping-needed analysis cannot carry an unavailable reason"
                )
            if self.evidence_trace is not None:
                raise AnalysisContractError(
                    "mapping-needed analysis cannot carry deterministic evidence trace"
                )
        elif self.state is AnalysisState.UNAVAILABLE:
            if self.unavailable_reason is None:
                raise AnalysisContractError(
                    "unavailable analysis requires a reason"
                )
            if any(
                value is not None
                for value in (
                    self.deterministic,
                    self.research_interpretation,
                    self.mapping_question,
                )
            ):
                raise AnalysisContractError(
                    "unavailable analysis cannot carry an available result"
                )
            if self.evidence_trace is not None:
                raise AnalysisContractError(
                    "unavailable analysis cannot carry deterministic evidence trace"
                )
        elif not any(
            value is not None
            for value in (
                self.deterministic,
                self.research_interpretation,
                self.mapping_question,
                self.unavailable_reason,
            )
        ) and not self.missing_evidence:
            raise AnalysisContractError(
                "partial analysis requires an available result, question, or reason"
            )

    @property
    def authoritative_verdict(self) -> Verdict | None:
        """Return only the verdict produced by deterministic Audit authority."""

        return None if self.deterministic is None else self.deterministic.verdict


def to_jsonable(value: object) -> object:
    """Return a detached JSON-compatible view of approved analysis values."""

    if type(value).__module__ == "claimci.analysis.trace":
        from .trace import trace_to_jsonable

        return trace_to_jsonable(value)
    if isinstance(value, Confidence):
        return value.value
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, (RepositoryPath, Sha256Digest, GitCommitSha)):
        return str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        try:
            json.dumps(value, allow_nan=False)
        except ValueError as error:
            raise AnalysisContractError(
                "serialized integer must fit the standard JSON encoder"
            ) from error
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisContractError("serialized numbers must be finite")
        return value
    if isinstance(value, Mapping):
        serialized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("serialized mapping keys must be text")
            serialized[key] = to_jsonable(item)
        return serialized
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if is_dataclass(value) and type(value).__module__.startswith("claimci.analysis"):
        return {
            item.name: to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    raise TypeError("value is not an approved JSON-serializable analysis contract")


__all__ = [
    "Adapter",
    "AdapterMatch",
    "AdvisoryResearchInterpretation",
    "AnalysisAuthority",
    "AnalysisContractError",
    "AnalysisState",
    "ArtifactCandidate",
    "ArtifactBinding",
    "ArtifactKind",
    "AuditClaimSpec",
    "ClaimReference",
    "ClaimedMetricValue",
    "ComputeEvidence",
    "ConfigValue",
    "DatasetReference",
    "DeterministicAuditOutcome",
    "EphemeralAuditPlan",
    "EvidenceSelector",
    "ExperimentRole",
    "FieldMapping",
    "FieldProvenance",
    "GitCommitSha",
    "MappingCandidate",
    "MappingChoice",
    "MappingQuestion",
    "MappingTrust",
    "MissingEvidence",
    "NormalizedEvidence",
    "NormalizedObservation",
    "PassiveArtifact",
    "ProvenanceKind",
    "RepoMapping",
    "RepositoryIdentity",
    "RepositoryPath",
    "SelectorKind",
    "Sha256Digest",
    "UnifiedAnalysisResult",
    "to_jsonable",
]
