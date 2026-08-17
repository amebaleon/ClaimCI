"""Immutable common contracts for ClaimCI analysis and later integrations.

Repository and pull-request content reaches adapters only as validated metadata
and passive bytes.  This module contains no discovery, execution, provider, or
workflow behavior.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Protocol

from .confidence import Confidence


class AnalysisContractError(ValueError):
    """A controlled violation of a public analysis contract."""


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_ADAPTER_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_TARGET_FIELD = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z")
_DOTTED_SEGMENT = re.compile(r"[A-Za-z0-9_-]+\Z")


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisContractError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise AnalysisContractError(f"{label} exceeds {maximum} characters")
    return value


def _bounded_id(value: object, label: str, *, maximum: int = 128) -> str:
    text = _bounded_text(value, label, maximum=maximum)
    if text != text.strip() or any(ord(character) < 32 for character in text):
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
        if self.kind is SelectorKind.JSON_POINTER and not expression.startswith("/"):
            raise AnalysisContractError("JSON pointer selector must start with '/'")
        if self.kind is SelectorKind.DOTTED_PATH and (
            expression.startswith(".")
            or expression.endswith(".")
            or any(not _DOTTED_SEGMENT.fullmatch(part) for part in expression.split("."))
        ):
            raise AnalysisContractError("dotted selector path is invalid")
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
        if not isinstance(self.relevant_claim_ids, tuple) or not all(
            isinstance(claim_id, str)
            and claim_id.strip()
            and claim_id == claim_id.strip()
            and len(claim_id) <= 128
            for claim_id in self.relevant_claim_ids
        ):
            raise AnalysisContractError(
                "relevant_claim_ids must be a tuple of bounded identifiers"
            )
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


__all__ = [
    "Adapter",
    "AdapterMatch",
    "AnalysisAuthority",
    "AnalysisContractError",
    "AnalysisState",
    "ArtifactCandidate",
    "ArtifactKind",
    "EvidenceSelector",
    "ExperimentRole",
    "FieldMapping",
    "FieldProvenance",
    "GitCommitSha",
    "MappingTrust",
    "PassiveArtifact",
    "ProvenanceKind",
    "RepositoryPath",
    "SelectorKind",
    "Sha256Digest",
]
