"""Bounded passive repository access for auto discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from claimci.analysis.contracts import RepositoryPath, Sha256Digest
from claimci.passive_files import (
    PassiveFileError,
    capture_confined_regular_file,
    inspect_confined_regular_file,
)
from claimci.review.models import ReviewError, ReviewLimits, SourceBundle
from claimci.review.sources import collect_review_sources

from .models import ArtifactIssue, DiscoveryError, DiscoveryLimits


@dataclass(frozen=True, slots=True)
class RepositoryContext:
    """One trusted root plus the exact bounded path index issued by Review."""

    head_root: Path
    base_root: Path | None
    source_bundle: SourceBundle
    repository_paths: tuple[RepositoryPath, ...]
    changed_paths: tuple[RepositoryPath, ...]


@dataclass(frozen=True, slots=True)
class InspectedArtifact:
    """Metadata observed from one confined passive head artifact."""

    path: RepositoryPath
    sha256: Sha256Digest
    size: int


@dataclass(slots=True)
class ChangeComparisonBudget:
    remaining_files: int
    remaining_bytes: int

    @classmethod
    def from_limits(cls, limits: DiscoveryLimits) -> ChangeComparisonBudget:
        return cls(
            remaining_files=limits.max_change_comparison_files,
            remaining_bytes=limits.max_change_comparison_bytes,
        )


@dataclass(slots=True)
class ArtifactInspectionBudget:
    """Aggregate integrity-I/O allowance for streamed discovery artifacts."""

    remaining_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.remaining_bytes, bool) or not isinstance(
            self.remaining_bytes, int
        ):
            raise TypeError("artifact inspection budget must be an integer")
        if self.remaining_bytes < 0:
            raise DiscoveryError("artifact inspection budget cannot be negative")

    @classmethod
    def from_limits(cls, limits: DiscoveryLimits) -> ArtifactInspectionBudget:
        return cls(remaining_bytes=limits.max_stream_total_bytes)


def _resolved_root(raw: Path, label: str) -> Path:
    try:
        root = Path(raw).resolve()
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise DiscoveryError(f"{label} is invalid") from error
    if not root.is_dir():
        raise DiscoveryError(f"{label} must be an existing directory")
    return root


def collect_repository_context(
    head_root: Path,
    *,
    base_root: Path | None = None,
    pr_title: str = "",
    pr_description: str = "",
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> RepositoryContext:
    """Reuse Review's bounded index and changed-source collection unchanged."""

    if not isinstance(limits, DiscoveryLimits):
        raise TypeError("discovery limits must be DiscoveryLimits")
    head = _resolved_root(head_root, "head repository root")
    base = None if base_root is None else _resolved_root(base_root, "base repository root")
    try:
        bundle = collect_review_sources(
            head,
            base_root=base,
            pr_title=pr_title,
            pr_description=pr_description,
            limits=ReviewLimits(),
        )
    except ReviewError as error:
        raise DiscoveryError(
            "repository could not be indexed within discovery bounds"
        ) from error
    try:
        repository_paths = tuple(RepositoryPath(path) for path in bundle.repository_paths)
        changed_paths = tuple(RepositoryPath(path) for path in bundle.changed_paths)
    except (TypeError, ValueError) as error:
        raise DiscoveryError("repository index issued a non-portable path") from error
    return RepositoryContext(
        head_root=head,
        base_root=base,
        source_bundle=bundle,
        repository_paths=repository_paths,
        changed_paths=changed_paths,
    )


def _confined_regular_file(root: Path, path: RepositoryPath) -> Path | None:
    cursor = root
    try:
        for part in PurePosixPath(path).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None
        resolved = cursor.resolve()
        resolved.relative_to(root)
        if not resolved.is_file() or not os.path.isfile(resolved):
            return None
        return resolved
    except (OSError, ValueError, RuntimeError):
        return None


def inspect_artifact(
    context: RepositoryContext,
    path: RepositoryPath,
    *,
    limits: DiscoveryLimits,
    budget: ArtifactInspectionBudget | None = None,
) -> InspectedArtifact | ArtifactIssue:
    """Inspect one issued passive artifact without interpreting its content."""

    if not isinstance(context, RepositoryContext):
        raise TypeError("repository context must be RepositoryContext")
    if not isinstance(limits, DiscoveryLimits):
        raise TypeError("discovery limits must be DiscoveryLimits")
    if budget is not None and not isinstance(budget, ArtifactInspectionBudget):
        raise TypeError("budget must be ArtifactInspectionBudget or null")
    if not isinstance(path, RepositoryPath):
        try:
            path = RepositoryPath(path)
        except (TypeError, ValueError) as error:
            raise DiscoveryError("artifact path is unsafe") from error
    if path not in set(context.repository_paths):
        raise DiscoveryError("artifact path is not an indexed repository path")
    resolved = _confined_regular_file(context.head_root, path)
    if resolved is None:
        return ArtifactIssue(path, "artifact is not a confined regular file")
    streamable = PurePosixPath(str(path)).suffix.casefold() in {
        ".jsonl",
        ".csv",
        ".tsv",
    }
    discovery_bound = limits.max_artifact_bytes
    try:
        observed_size = resolved.stat().st_size
        if observed_size > limits.max_artifact_bytes and streamable:
            if budget is not None and observed_size > budget.remaining_bytes:
                return ArtifactIssue(
                    path,
                    "artifact exceeds the streaming discovery work budget",
                )
            capture = inspect_confined_regular_file(
                context.head_root,
                str(path),
                max_bytes=limits.max_stream_artifact_bytes,
            )
            discovery_bound = limits.max_stream_artifact_bytes
            if budget is not None:
                budget.remaining_bytes -= capture.size
        else:
            capture = capture_confined_regular_file(
                context.head_root,
                str(path),
                max_bytes=limits.max_artifact_bytes,
            )
            discovery_bound = limits.max_artifact_bytes
    except PassiveFileError as error:
        if error.code == "too_large":
            return ArtifactIssue(
                path,
                f"artifact exceeds the {discovery_bound}-byte discovery limit",
            )
        return ArtifactIssue(
            path,
            "artifact changed or is not a confined regular file",
        )
    except OSError:
        return ArtifactIssue(
            path,
            "artifact changed or is not a confined regular file",
        )
    return InspectedArtifact(
        path=path,
        sha256=Sha256Digest(capture.sha256),
        size=capture.size,
    )


def read_artifact_text(
    context: RepositoryContext,
    path: RepositoryPath,
    *,
    limits: DiscoveryLimits,
) -> tuple[InspectedArtifact, str] | ArtifactIssue:
    """Read one already-issued bounded artifact as passive UTF-8 text."""

    if not isinstance(context, RepositoryContext):
        raise TypeError("repository context must be RepositoryContext")
    if not isinstance(limits, DiscoveryLimits):
        raise TypeError("discovery limits must be DiscoveryLimits")
    if not isinstance(path, RepositoryPath):
        try:
            path = RepositoryPath(path)
        except (TypeError, ValueError) as error:
            raise DiscoveryError("artifact path is unsafe") from error
    if path not in set(context.repository_paths):
        raise DiscoveryError("artifact path is not an indexed repository path")
    try:
        capture = capture_confined_regular_file(
            context.head_root,
            str(path),
            max_bytes=limits.max_artifact_bytes,
        )
    except PassiveFileError as error:
        if error.code == "too_large":
            return ArtifactIssue(
                path,
                f"artifact exceeds the {limits.max_artifact_bytes}-byte discovery limit",
            )
        return ArtifactIssue(path, "artifact changed or is not a confined regular file")
    inspection = InspectedArtifact(
        path=path,
        sha256=Sha256Digest(capture.sha256),
        size=capture.size,
    )
    try:
        text = capture.content.decode("utf-8")
    except UnicodeError:
        return ArtifactIssue(inspection.path, "artifact is not valid UTF-8 text")
    return inspection, text.replace("\r\n", "\n").replace("\r", "\n")


def artifact_changed(
    context: RepositoryContext,
    inspection: InspectedArtifact,
    *,
    budget: ChangeComparisonBudget,
) -> bool | None:
    """Return changed state for one shortlisted artifact under a bounded budget."""

    if context.base_root is None:
        return True
    base_path = _confined_regular_file(context.base_root, inspection.path)
    if base_path is None:
        return True
    try:
        base_size = base_path.stat().st_size
    except OSError:
        return None
    if base_size < 0:
        return None
    if base_size != inspection.size:
        return True
    required_bytes = base_size + inspection.size
    if budget.remaining_files < 1 or required_bytes > budget.remaining_bytes:
        return None
    budget.remaining_files -= 1
    budget.remaining_bytes -= required_bytes
    try:
        capture = capture_confined_regular_file(
            context.base_root,
            str(inspection.path),
            max_bytes=inspection.size,
        )
        return capture.sha256 != inspection.sha256
    except (OSError, PassiveFileError):
        return None


__all__ = [
    "ArtifactInspectionBudget",
    "ChangeComparisonBudget",
    "InspectedArtifact",
    "RepositoryContext",
    "artifact_changed",
    "collect_repository_context",
    "inspect_artifact",
    "read_artifact_text",
]
