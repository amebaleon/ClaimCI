"""LLM-safe value objects for ClaimCI research review.

This module intentionally does not import the deterministic ClaimCI model
layer.  Provider-originated data can describe claims and interpretations, but
it has no type path for constructing blocking scientific authority.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType


class ReviewError(ValueError):
    """A controlled review configuration, source, or provider error."""


class ClaimType(str, Enum):
    METRIC_IMPROVEMENT = "metric_improvement"
    COMPUTE_EQUIVALENCE = "compute_equivalence"
    HELD_OUT_EVALUATION = "held_out_evaluation"
    RESOURCE_REDUCTION = "resource_reduction"
    COMPONENT_CAUSALITY = "component_causality"
    NO_EXTERNAL_REWARD = "no_external_reward"
    IMPLEMENTATION_CLAIM = "implementation_claim"
    OTHER_SCIENTIFIC = "other_scientific"


class ClaimDirection(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"
    NOT_APPLICABLE = "not_applicable"


class MagnitudeKind(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    UNSPECIFIED = "unspecified"


class SourceKind(str, Enum):
    PULL_REQUEST_TITLE = "pull_request_title"
    PULL_REQUEST_DESCRIPTION = "pull_request_description"
    REPOSITORY_FILE = "repository_file"


class ReviewMaterialKind(str, Enum):
    """Passive material categories used only to plan evidence routes."""

    DOCUMENT = "document"
    SOURCE = "source"
    TEST = "test"
    CONFIG = "config"
    RESULT = "result"
    MANIFEST = "manifest"
    BENCHMARK = "benchmark"
    SUBMISSION_CONFIG = "submission_config"
    OTHER = "other"


class ReviewStatus(str, Enum):
    DISABLED = "DISABLED"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class GateDisposition(str, Enum):
    PASS_COMPLETE = "pass_complete"
    PASS_PARTIAL = "pass_partial"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


class ComparisonBasis(str, Enum):
    DIRECT_BASE = "direct_base"
    MERGE_BASE = "merge_base"


class ChangeStatus(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


class ChangeInventorySource(str, Enum):
    TRUSTED_GIT_OBJECT_GRAPH = "trusted_git_object_graph"
    LEGACY_PAIRWISE = "legacy_pairwise"


class SnapshotRole(str, Enum):
    REQUESTED_BASE = "requested_base"
    COMPARISON_BASE = "comparison_base"
    HEAD = "head"


_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FULL_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
MAX_RECORDED_TOKEN_COUNT = 1_000_000_000_000
MAX_CHANGE_INVENTORY_ENTRIES = 8_192


def _nonempty_text(value: object, label: str, *, max_chars: int = 16_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError(f"{label} must be a non-empty string")
    if len(value) > max_chars:
        raise ReviewError(f"{label} exceeds {max_chars} characters")
    return value


def _relative_path(value: object, label: str) -> str:
    text = _nonempty_text(value, label, max_chars=4_096)
    if "\x00" in text:
        raise ReviewError(f"{label} contains NUL")
    if "\\" in text:
        raise ReviewError(f"{label} must use unambiguous forward slashes")
    portable = text
    if (
        PurePosixPath(portable).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or ".." in portable.split("/")
    ):
        raise ReviewError(f"{label} must be a confined relative path")
    normalized = PurePosixPath(portable).as_posix()
    if normalized in {"", "."}:
        raise ReviewError(f"{label} must name a file")
    return normalized


def _full_git_object_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _FULL_GIT_OBJECT_ID.fullmatch(value):
        raise ReviewError(f"{label} must be a full lowercase Git object ID")
    return value


def _portable_inventory_path(value: object) -> str:
    """Validate an already-canonical portable repository path.

    Inventory metadata is an identity boundary, so aliases are rejected rather
    than normalized into another spelling.
    """

    text = _nonempty_text(value, "change path", max_chars=4_096)
    windows = PureWindowsPath(text)
    components = text.split("/")
    if (
        "\\" in text
        or windows.drive
        or PurePosixPath(text).is_absolute()
        or any(component in {"", ".", ".."} for component in components)
        or any(unicodedata.category(character) == "Cc" for character in text)
    ):
        raise ReviewError("change path must be a canonical portable relative path")
    if PurePosixPath(text).as_posix() != text:
        raise ReviewError("change path must be a canonical portable relative path")
    return text


@dataclass(frozen=True)
class SnapshotIdentity:
    role: SnapshotRole
    root: Path
    sha: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, SnapshotRole):
            raise ReviewError("snapshot role is invalid")
        if not isinstance(self.root, Path):
            raise ReviewError("snapshot root must be a Path")
        try:
            if not self.root.is_absolute() or self.root.is_symlink():
                raise ReviewError("snapshot root must be a resolved regular directory")
            resolved = self.root.resolve(strict=True)
        except ReviewError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise ReviewError("snapshot root must be a resolved regular directory") from exc
        if resolved != self.root or not resolved.is_dir():
            raise ReviewError("snapshot root must be a resolved regular directory")
        _full_git_object_id(self.sha, "snapshot sha")


@dataclass(frozen=True)
class DeclaredReviewCoordinates:
    """Raw all-or-none coordinates retained even when a root is invalid."""

    requested_base_sha: str
    comparison_base_sha: str
    head_sha: str
    comparison_basis: ComparisonBasis
    invalid_root_roles: tuple[SnapshotRole, ...] = ()

    def __post_init__(self) -> None:
        for label in (
            "requested_base_sha",
            "comparison_base_sha",
            "head_sha",
        ):
            if not isinstance(getattr(self, label), str):
                raise ReviewError("declared review coordinates must be strings")
        if not isinstance(self.comparison_basis, ComparisonBasis):
            raise ReviewError("declared review comparison basis is invalid")
        role_order = {
            SnapshotRole.REQUESTED_BASE: 0,
            SnapshotRole.COMPARISON_BASE: 1,
            SnapshotRole.HEAD: 2,
        }
        if (
            not isinstance(self.invalid_root_roles, tuple)
            or any(role not in role_order for role in self.invalid_root_roles)
            or tuple(sorted(set(self.invalid_root_roles), key=role_order.__getitem__))
            != self.invalid_root_roles
        ):
            raise ReviewError("declared review invalid-root roles are invalid")


@dataclass(frozen=True)
class ChangeEntry:
    path: str
    status: ChangeStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _portable_inventory_path(self.path))
        if not isinstance(self.status, ChangeStatus):
            raise ReviewError("change status is invalid")


@dataclass(frozen=True)
class ChangeInventory:
    schema_version: int
    requested_base_sha: str
    comparison_base_sha: str
    head_sha: str
    comparison_basis: ComparisonBasis
    source: ChangeInventorySource
    declared_entry_count: int
    complete: bool
    entries: tuple[ChangeEntry, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ReviewError("change inventory schema_version must be 1")
        _full_git_object_id(self.requested_base_sha, "requested base sha")
        _full_git_object_id(self.comparison_base_sha, "comparison base sha")
        _full_git_object_id(self.head_sha, "head sha")
        if not isinstance(self.comparison_basis, ComparisonBasis):
            raise ReviewError("comparison basis is invalid")
        if not isinstance(self.source, ChangeInventorySource):
            raise ReviewError("change inventory source is invalid")
        if (
            isinstance(self.declared_entry_count, bool)
            or not isinstance(self.declared_entry_count, int)
            or not 0 <= self.declared_entry_count <= MAX_CHANGE_INVENTORY_ENTRIES
        ):
            raise ReviewError("declared change entry count is invalid")
        if not isinstance(self.complete, bool):
            raise ReviewError("change inventory completeness must be a boolean")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, ChangeEntry) for entry in self.entries
        ):
            raise ReviewError("change inventory entries must be ChangeEntry values")
        if self.declared_entry_count != len(self.entries):
            raise ReviewError("declared change entry count is inconsistent")
        if self.source is ChangeInventorySource.TRUSTED_GIT_OBJECT_GRAPH and not self.complete:
            raise ReviewError("trusted Git change inventory must be complete")
        if (
            self.comparison_basis is ComparisonBasis.DIRECT_BASE
            and self.comparison_base_sha != self.requested_base_sha
        ):
            raise ReviewError("direct comparison base must equal requested base")
        expected = tuple(sorted(self.entries, key=lambda entry: (entry.path, entry.status.value)))
        if expected != self.entries:
            raise ReviewError("change inventory entries must be in canonical POSIX order")
        folded_paths = tuple(entry.path.casefold() for entry in self.entries)
        if len(set(folded_paths)) != len(folded_paths):
            raise ReviewError("change inventory paths must be unique")


def _positive_int(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewError(f"{label} must be an integer")
    if not 1 <= value <= maximum:
        raise ReviewError(f"{label} must be between 1 and {maximum}")
    return value


@dataclass(frozen=True)
class ReviewLimits:
    max_calls: int = 2
    max_context_chars: int = 60_000
    max_output_chars: int = 24_000
    max_files: int = 24
    max_file_chars: int = 16_000
    max_claims: int = 16
    extraction_max_output_tokens: int = 5_000
    synthesis_max_output_tokens: int = 4_000
    timeout_seconds: float = 30.0
    retries: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.max_calls, "max_calls", 2)
        _positive_int(self.max_context_chars, "max_context_chars", 60_000)
        _positive_int(self.max_output_chars, "max_output_chars", 24_000)
        _positive_int(self.max_files, "max_files", 24)
        _positive_int(self.max_file_chars, "max_file_chars", 16_000)
        _positive_int(self.max_claims, "max_claims", 16)
        _positive_int(
            self.extraction_max_output_tokens,
            "extraction_max_output_tokens",
            5_000,
        )
        _positive_int(
            self.synthesis_max_output_tokens,
            "synthesis_max_output_tokens",
            4_000,
        )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise ReviewError("timeout_seconds must be a finite number from 0 through 120")
        try:
            timeout = float(self.timeout_seconds)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ReviewError(
                "timeout_seconds must be a finite number from 0 through 120"
            ) from exc
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ReviewError("timeout_seconds must be a finite number from 0 through 120")
        if isinstance(self.retries, bool) or self.retries != 0:
            raise ReviewError("provider retries must be zero")


@dataclass(frozen=True)
class ReviewConfig:
    schema_version: int = 1
    enabled: bool = False
    policy: str = "advisory"
    provider: str = "openai"
    model: str = "gpt-5.6-terra"
    limits: ReviewLimits = field(default_factory=ReviewLimits)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ReviewError("review schema_version must be 1")
        if not isinstance(self.enabled, bool):
            raise ReviewError("review enabled must be a boolean")
        if self.policy != "advisory":
            raise ReviewError("Day 3 review policy must be advisory")
        _nonempty_text(self.provider, "review provider", max_chars=64)
        _nonempty_text(self.model, "review model", max_chars=128)
        if not isinstance(self.limits, ReviewLimits):
            raise ReviewError("review limits must be ReviewLimits")


@dataclass(frozen=True)
class SourceLocation:
    source_id: str
    kind: SourceKind
    path: str | None
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        _nonempty_text(self.source_id, "source_id", max_chars=128)
        if not isinstance(self.kind, SourceKind):
            raise ReviewError("source kind is invalid")
        if self.kind is SourceKind.REPOSITORY_FILE:
            normalized = _relative_path(self.path, "source path")
            object.__setattr__(self, "path", normalized)
        elif self.path is not None:
            raise ReviewError("pull-request metadata sources cannot declare a path")
        if isinstance(self.start_line, bool) or not isinstance(self.start_line, int):
            raise ReviewError("source start_line must be an integer")
        if isinstance(self.end_line, bool) or not isinstance(self.end_line, int):
            raise ReviewError("source end_line must be an integer")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ReviewError("source line span is invalid")


@dataclass(frozen=True)
class ClaimMagnitude:
    raw: str
    value: float | None = None
    unit: str | None = None
    kind: MagnitudeKind = MagnitudeKind.UNSPECIFIED

    def __post_init__(self) -> None:
        _nonempty_text(self.raw, "claimed magnitude", max_chars=512)
        if not isinstance(self.kind, MagnitudeKind):
            raise ReviewError("claimed magnitude kind is invalid")
        if self.value is not None:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(float(self.value))
            ):
                raise ReviewError("claimed magnitude value must be finite")
        if self.unit is not None:
            _nonempty_text(self.unit, "claimed magnitude unit", max_chars=128)


@dataclass(frozen=True)
class ScientificClaim:
    claim_id: str
    source_text: str
    claim_type: ClaimType
    subject: str
    source: SourceLocation
    metric: str | None = None
    direction: ClaimDirection = ClaimDirection.NOT_APPLICABLE
    claimed_magnitude: ClaimMagnitude | None = None
    qualifiers: tuple[str, ...] = ()
    confidence: float = 0.0
    evidence_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_text(self.claim_id, "claim_id", max_chars=128)
        _nonempty_text(self.source_text, "claim source_text")
        if not isinstance(self.claim_type, ClaimType):
            raise ReviewError("claim type is invalid")
        _nonempty_text(self.subject, "claim subject", max_chars=1_024)
        if not isinstance(self.source, SourceLocation):
            raise ReviewError("claim source must be a SourceLocation")
        if self.metric is not None:
            _nonempty_text(self.metric, "claim metric", max_chars=256)
        if not isinstance(self.direction, ClaimDirection):
            raise ReviewError("claim direction is invalid")
        if self.claimed_magnitude is not None and not isinstance(
            self.claimed_magnitude, ClaimMagnitude
        ):
            raise ReviewError("claimed_magnitude is invalid")
        if not isinstance(self.qualifiers, tuple) or not all(
            isinstance(item, str) and item.strip() and len(item) <= 1_024
            for item in self.qualifiers
        ):
            raise ReviewError("claim qualifiers must be bounded non-empty strings")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0 <= float(self.confidence) <= 1
        ):
            raise ReviewError("claim confidence must be finite from 0 through 1")
        if not isinstance(self.evidence_hints, tuple) or not all(
            isinstance(item, str) and item.strip() and len(item) <= 4_096
            for item in self.evidence_hints
        ):
            raise ReviewError("evidence hints must be bounded non-empty strings")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    kind: SourceKind
    path: str | None
    text: str
    sha256: str

    def __post_init__(self) -> None:
        _nonempty_text(self.source_id, "source_id", max_chars=128)
        if not isinstance(self.kind, SourceKind):
            raise ReviewError("source kind is invalid")
        if self.kind is SourceKind.REPOSITORY_FILE:
            object.__setattr__(self, "path", _relative_path(self.path, "source path"))
        elif self.path is not None:
            raise ReviewError("pull-request metadata sources cannot declare a path")
        if not isinstance(self.text, str):
            raise ReviewError("source text must be a string")
        if not isinstance(self.sha256, str) or not _HEX_DIGEST.fullmatch(self.sha256):
            raise ReviewError("source sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SourceBundle:
    sources: tuple[SourceRecord, ...] = ()
    repository_paths: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    total_chars: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple) or not all(
            isinstance(item, SourceRecord) for item in self.sources
        ):
            raise ReviewError("sources must be SourceRecord values")
        if not isinstance(self.repository_paths, tuple):
            raise ReviewError("repository_paths must be a tuple")
        normalized = tuple(_relative_path(path, "repository path") for path in self.repository_paths)
        if normalized != self.repository_paths:
            object.__setattr__(self, "repository_paths", normalized)
        if not isinstance(self.changed_paths, tuple):
            raise ReviewError("changed_paths must be a tuple")
        normalized_changed = tuple(
            _relative_path(path, "changed repository path")
            for path in self.changed_paths
        )
        if not set(normalized_changed).issubset(set(normalized)):
            raise ReviewError("changed_paths must be issued repository paths")
        if normalized_changed != self.changed_paths:
            object.__setattr__(self, "changed_paths", normalized_changed)
        if (
            isinstance(self.total_chars, bool)
            or not isinstance(self.total_chars, int)
            or self.total_chars < 0
            or self.total_chars != sum(len(source.text) for source in self.sources)
        ):
            raise ReviewError("source total_chars is inconsistent")


@dataclass(frozen=True)
class MaterialClaimSeed:
    """Deterministic routing metadata, never a scientific claim or evidence."""

    origin_source_id: str
    category: str
    route_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.origin_source_id, "material seed origin_source_id", max_chars=128)
        category = _nonempty_text(self.category, "material seed category", max_chars=64)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", category):
            raise ReviewError("material seed category must be a lowercase identifier")
        if not isinstance(self.route_terms, tuple) or not self.route_terms:
            raise ReviewError("material seed route_terms must be a non-empty tuple")
        if not all(
            isinstance(term, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9_.+-]*", term)
            and len(term) <= 128
            for term in self.route_terms
        ):
            raise ReviewError("material seed route_terms are invalid")
        if tuple(sorted(set(self.route_terms))) != self.route_terms:
            raise ReviewError("material seed route_terms must be sorted and unique")


@dataclass(frozen=True)
class ScopeIssue:
    """Stable, content-free explanation of one bounded-scope omission."""

    code: str
    path: str | None = None
    observed: int | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        code = _nonempty_text(self.code, "scope issue code", max_chars=128)
        if not (
            re.fullmatch(r"PREFLIGHT_G[123]_[A-Z0-9_]+", code)
            or code == "PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"
        ):
            raise ReviewError("scope issue code is invalid")
        if self.path is not None:
            object.__setattr__(self, "path", _portable_inventory_path(self.path))
        for label in ("observed", "limit"):
            value = getattr(self, label)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ReviewError(
                    f"scope issue {label} must be a non-negative integer or null"
                )

    def as_dict(self) -> dict[str, object]:
        """Return the stable sparse issue representation used by reports."""

        value: dict[str, object] = {"code": self.code}
        if self.path is not None:
            value["path"] = self.path
        if self.observed is not None:
            value["observed"] = self.observed
        if self.limit is not None:
            value["limit"] = self.limit
        return value


@dataclass(frozen=True)
class ReviewInventoryFailure:
    """Content-free typed failure carried from inventory construction."""

    code: str

    def __post_init__(self) -> None:
        issue = ScopeIssue(code=self.code)
        if not issue.code.startswith("PREFLIGHT_G1_"):
            raise ReviewError("review inventory failure must be a Gate 1 code")


@dataclass(frozen=True)
class ReviewScope:
    """Complete inventory plus its bounded, deterministic materialization plan."""

    mode: str
    inventory: ChangeInventory
    issued_paths: tuple[str, ...]
    issued_changed_paths: tuple[str, ...]
    selected_paths: tuple[str, ...]
    sources: tuple[SourceRecord, ...]
    seeds: tuple[MaterialClaimSeed, ...]
    complete: bool
    issues: tuple[ScopeIssue, ...]
    materialized_chars: int = 0
    materialized_path_chars: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"declared_changed_v1", "legacy_pairwise_v1"}:
            raise ReviewError("review scope mode is invalid")
        if not isinstance(self.inventory, ChangeInventory):
            raise ReviewError("review scope inventory is invalid")
        for label in ("issued_paths", "issued_changed_paths", "selected_paths"):
            values = getattr(self, label)
            if not isinstance(values, tuple):
                raise ReviewError(f"review scope {label} must be a tuple")
            normalized = tuple(_portable_inventory_path(path) for path in values)
            if normalized != values or len(set(values)) != len(values):
                raise ReviewError(
                    f"review scope {label} must contain unique canonical paths"
                )
        issued = set(self.issued_paths)
        selected = set(self.selected_paths)
        if not set(self.issued_changed_paths).issubset(issued):
            raise ReviewError("issued changed paths must be issued paths")
        if not selected.issubset(issued):
            raise ReviewError("selected paths must be issued paths")
        inventory_head_paths = {
            entry.path
            for entry in self.inventory.entries
            if entry.status is not ChangeStatus.DELETED
        }
        if not set(self.issued_changed_paths).issubset(inventory_head_paths):
            raise ReviewError("issued changed paths must be non-deleted inventory paths")
        if not isinstance(self.sources, tuple) or not all(
            isinstance(source, SourceRecord) for source in self.sources
        ):
            raise ReviewError("review scope sources must be SourceRecord values")
        if any(
            source.path is not None and source.path not in selected
            for source in self.sources
        ):
            raise ReviewError("repository sources must be selected paths")
        if not isinstance(self.seeds, tuple) or not all(
            isinstance(seed, MaterialClaimSeed) for seed in self.seeds
        ):
            raise ReviewError("review scope seeds must be MaterialClaimSeed values")
        if not isinstance(self.complete, bool):
            raise ReviewError("review scope completeness must be a boolean")
        if not isinstance(self.issues, tuple) or not all(
            isinstance(issue, ScopeIssue) for issue in self.issues
        ):
            raise ReviewError("review scope issues must be ScopeIssue values")
        if (
            isinstance(self.materialized_chars, bool)
            or not isinstance(self.materialized_chars, int)
            or self.materialized_chars < 0
        ):
            raise ReviewError("review scope materialized_chars must be non-negative")
        if (
            not isinstance(self.materialized_path_chars, tuple)
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or item[0] not in selected
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] < 0
                for item in self.materialized_path_chars
            )
            or tuple(sorted(self.materialized_path_chars))
            != self.materialized_path_chars
            or len({path for path, _chars in self.materialized_path_chars})
            != len(self.materialized_path_chars)
        ):
            raise ReviewError("review scope materialized_path_chars is invalid")


@dataclass(frozen=True)
class PreflightGateResult:
    gate: int
    disposition: GateDisposition
    reasons: tuple[ScopeIssue, ...]
    metrics: Mapping[str, int | str | bool]

    def __post_init__(self) -> None:
        if isinstance(self.gate, bool) or self.gate not in {1, 2, 3}:
            raise ReviewError("preflight gate must be 1, 2, or 3")
        if not isinstance(self.disposition, GateDisposition):
            raise ReviewError("preflight gate disposition is invalid")
        if not isinstance(self.reasons, tuple) or not all(
            isinstance(reason, ScopeIssue) for reason in self.reasons
        ):
            raise ReviewError("preflight gate reasons must be ScopeIssue values")
        upstream = "PREFLIGHT_NOT_EVALUATED_UPSTREAM_FAILURE"
        if self.disposition is GateDisposition.PASS_COMPLETE and self.reasons:
            raise ReviewError("a complete preflight gate cannot contain reasons")
        if self.disposition in {
            GateDisposition.PASS_PARTIAL,
            GateDisposition.FAIL,
        } and not self.reasons:
            raise ReviewError("a partial or failed preflight gate requires a reason")
        if self.disposition is GateDisposition.NOT_EVALUATED:
            if tuple(reason.code for reason in self.reasons) != (upstream,):
                raise ReviewError("a skipped preflight gate requires the upstream reason")
        elif any(reason.code == upstream for reason in self.reasons):
            raise ReviewError("the upstream reason is only valid for a skipped gate")
        if not isinstance(self.metrics, Mapping):
            raise ReviewError("preflight gate metrics must be a mapping")
        normalized: dict[str, int | str | bool] = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str) or not key:
                raise ReviewError("preflight metric names must be non-empty strings")
            if not isinstance(value, (int, str, bool)):
                raise ReviewError("preflight metric values must be scalar")
            normalized[key] = value
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(sorted(normalized.items()))),
        )


@dataclass(frozen=True)
class ReviewPreflight:
    schema_version: int
    requested_base_sha: str
    comparison_base_sha: str
    head_sha: str
    gates: tuple[PreflightGateResult, ...]
    ready_for_provider: bool
    review_status_ceiling: ReviewStatus
    scope: ReviewScope | None
    comparison_basis: ComparisonBasis | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ReviewError("review preflight schema_version must be 1")
        _full_git_object_id(self.requested_base_sha, "preflight requested base sha")
        _full_git_object_id(self.comparison_base_sha, "preflight comparison base sha")
        _full_git_object_id(self.head_sha, "preflight head sha")
        if (
            not isinstance(self.gates, tuple)
            or tuple(gate.gate for gate in self.gates) != (1, 2, 3)
        ):
            raise ReviewError("review preflight must contain gates 1, 2, and 3 in order")
        dispositions = tuple(gate.disposition for gate in self.gates)
        failure_indexes = tuple(
            index
            for index, disposition in enumerate(dispositions)
            if disposition is GateDisposition.FAIL
        )
        not_evaluated_indexes = tuple(
            index
            for index, disposition in enumerate(dispositions)
            if disposition is GateDisposition.NOT_EVALUATED
        )
        if failure_indexes:
            first_failure = failure_indexes[0]
            pass_dispositions = {
                GateDisposition.PASS_COMPLETE,
                GateDisposition.PASS_PARTIAL,
            }
            if (
                failure_indexes != (first_failure,)
                or any(
                    disposition not in pass_dispositions
                    for disposition in dispositions[:first_failure]
                )
                or dispositions[first_failure + 1 :]
                != (GateDisposition.NOT_EVALUATED,) * (2 - first_failure)
            ):
                raise ReviewError("review preflight gate sequence is impossible")
        elif not_evaluated_indexes:
            raise ReviewError("review preflight gate sequence is impossible")
        failed = GateDisposition.FAIL in dispositions
        expected_ready = all(
            disposition
            in {GateDisposition.PASS_COMPLETE, GateDisposition.PASS_PARTIAL}
            for disposition in dispositions
        )
        expected_ceiling = (
            ReviewStatus.UNAVAILABLE
            if failed
            else (
                ReviewStatus.PARTIAL
                if GateDisposition.PASS_PARTIAL in dispositions
                else ReviewStatus.COMPLETE
            )
        )
        if not isinstance(self.ready_for_provider, bool) or (
            self.ready_for_provider != expected_ready
        ):
            raise ReviewError("review preflight provider readiness is inconsistent")
        if self.review_status_ceiling is not expected_ceiling:
            raise ReviewError("review preflight status ceiling is inconsistent")
        if self.scope is not None and not isinstance(self.scope, ReviewScope):
            raise ReviewError("review preflight scope is invalid")
        if self.comparison_basis is not None and not isinstance(
            self.comparison_basis, ComparisonBasis
        ):
            raise ReviewError("review preflight comparison basis is invalid")
        if (
            self.scope is not None
            and self.comparison_basis is not None
            and self.comparison_basis is not self.scope.inventory.comparison_basis
        ):
            raise ReviewError("review preflight comparison basis is inconsistent")


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None

    def __post_init__(self) -> None:
        for label in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, label)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ReviewError(f"{label} must be a non-negative integer or null")
            if value is not None and value > MAX_RECORDED_TOKEN_COUNT:
                # Provider usage is observability only. Implausibly large
                # counters are represented as unavailable instead of making
                # JSON/Markdown integer conversion fail.
                object.__setattr__(self, label, None)
        if self.estimated_cost_usd is not None:
            try:
                cost = Decimal(str(self.estimated_cost_usd))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ReviewError("estimated_cost_usd must be a decimal") from exc
            if not cost.is_finite() or cost < 0:
                raise ReviewError("estimated_cost_usd must be finite and non-negative")
            object.__setattr__(self, "estimated_cost_usd", cost)


@dataclass(frozen=True)
class ProviderCallRecord:
    task: str
    provider: str
    model: str
    request_id: str | None
    usage: ProviderUsage
    input_chars: int
    output_chars: int

    def __post_init__(self) -> None:
        _nonempty_text(self.task, "provider task", max_chars=64)
        _nonempty_text(self.provider, "provider name", max_chars=64)
        _nonempty_text(self.model, "provider model", max_chars=128)
        if self.request_id is not None:
            _nonempty_text(self.request_id, "provider request_id", max_chars=256)
        if not isinstance(self.usage, ProviderUsage):
            raise ReviewError("provider call usage must be ProviderUsage")
        for label in ("input_chars", "output_chars"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReviewError(f"{label} must be a non-negative integer")
